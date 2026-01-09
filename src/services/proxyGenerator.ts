// GPU-accelerated proxy frame generator using WebCodecs
// Uses VideoDecoder for hardware decoding and workers for parallel WebP encoding

import * as MP4BoxModule from 'mp4box';
const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

// Configuration
const PROXY_FPS = 30;
const PROXY_QUALITY = 0.92;
const PROXY_MAX_WIDTH = 1920;
const WORKER_POOL_SIZE = navigator.hardwareConcurrency || 4;
const BATCH_SIZE = 10;

// MP4Box types
interface MP4ArrayBuffer extends ArrayBuffer {
  fileStart: number;
}

interface Sample {
  number: number;
  track_id: number;
  data: ArrayBuffer;
  size: number;
  cts: number;
  dts: number;
  duration: number;
  is_sync: boolean;
  timescale: number;
}

interface MP4VideoTrack {
  id: number;
  codec: string;
  duration: number;
  timescale: number;
  nb_samples: number;
  video: { width: number; height: number };
}

interface MP4File {
  onReady: (info: { videoTracks: MP4VideoTrack[] }) => void;
  onSamples: (trackId: number, ref: any, samples: Sample[]) => void;
  onError: (error: string) => void;
  appendBuffer: (buffer: MP4ArrayBuffer) => number;
  start: () => void;
  flush: () => void;
  setExtractionOptions: (trackId: number, user: any, options: { nbSamples: number }) => void;
  getTrackById: (id: number) => any;
}

// ProxyFrame documents the data format used by encoding workers:
// { frameIndex: number, blob: Blob }

interface GeneratorResult {
  frameCount: number;
  fps: number;
}

// Worker code as a string (will be converted to blob URL)
const workerCode = `
  self.onmessage = async function(e) {
    const { frameIndex, imageData, width, height, quality } = e.data;

    // Create OffscreenCanvas and draw ImageData
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);

    // Encode to WebP
    const blob = await canvas.convertToBlob({ type: 'image/webp', quality });

    // Send back the result
    self.postMessage({ frameIndex, blob });
  };
`;

class ProxyGeneratorGPU {
  private mp4File: MP4File | null = null;
  private decoder: VideoDecoder | null = null;
  private workers: Worker[] = [];
  private workerBusy: boolean[] = [];
  private pendingFrames: Map<number, { resolve: (blob: Blob) => void; reject: (err: Error) => void }> = new Map();
  private frameQueue: { frameIndex: number; imageData: ImageData }[] = [];

  private videoTrack: MP4VideoTrack | null = null;
  private samples: Sample[] = [];
  private codecConfig: VideoDecoderConfig | null = null;

  private canvas: OffscreenCanvas | null = null;
  private ctx: OffscreenCanvasRenderingContext2D | null = null;

  private outputWidth = 0;
  private outputHeight = 0;
  private duration = 0;
  private totalFrames = 0;
  private processedFrames = 0;
  private decodedFrames: Map<number, VideoFrame> = new Map();
  private frameTimestamps: number[] = [];

  private onProgress: ((progress: number) => void) | null = null;
  private checkCancelled: (() => boolean) | null = null;
  private isCancelled = false;

  private resolveGeneration: ((result: GeneratorResult | null) => void) | null = null;

  constructor() {
    this.initWorkers();
  }

  private initWorkers() {
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);

    for (let i = 0; i < WORKER_POOL_SIZE; i++) {
      const worker = new Worker(workerUrl);
      worker.onmessage = (e) => this.handleWorkerMessage(i, e.data);
      worker.onerror = (e) => console.error('[ProxyGen] Worker error:', e);
      this.workers.push(worker);
      this.workerBusy.push(false);
    }

    console.log(`[ProxyGen] Initialized ${WORKER_POOL_SIZE} encoding workers`);
  }

  private handleWorkerMessage(workerIndex: number, data: { frameIndex: number; blob: Blob }) {
    this.workerBusy[workerIndex] = false;

    const pending = this.pendingFrames.get(data.frameIndex);
    if (pending) {
      pending.resolve(data.blob);
      this.pendingFrames.delete(data.frameIndex);
    }

    // Process next frame in queue
    this.processFrameQueue();
  }

  private processFrameQueue() {
    if (this.frameQueue.length === 0) return;

    // Find an available worker
    const workerIndex = this.workerBusy.findIndex(busy => !busy);
    if (workerIndex === -1) return;

    const frame = this.frameQueue.shift()!;
    this.workerBusy[workerIndex] = true;

    this.workers[workerIndex].postMessage({
      frameIndex: frame.frameIndex,
      imageData: frame.imageData,
      width: this.outputWidth,
      height: this.outputHeight,
      quality: PROXY_QUALITY,
    });
  }

  private async encodeFrameAsync(frameIndex: number, imageData: ImageData): Promise<Blob> {
    return new Promise((resolve, reject) => {
      this.pendingFrames.set(frameIndex, { resolve, reject });
      this.frameQueue.push({ frameIndex, imageData });
      this.processFrameQueue();
    });
  }

  async generate(
    file: File,
    mediaFileId: string,
    onProgress: (progress: number) => void,
    checkCancelled: () => boolean,
    saveFrame: (frame: { id: string; mediaFileId: string; frameIndex: number; blob: Blob }) => Promise<void>
  ): Promise<GeneratorResult | null> {
    this.onProgress = onProgress;
    this.checkCancelled = checkCancelled;
    this.isCancelled = false;
    this.processedFrames = 0;
    this.samples = [];
    this.decodedFrames.clear();
    this.frameTimestamps = [];

    return new Promise(async (resolve, _reject) => {
      this.resolveGeneration = resolve;

      try {
        // Check for WebCodecs support
        if (!('VideoDecoder' in window)) {
          console.warn('[ProxyGen] WebCodecs not supported, falling back to legacy method');
          resolve(null); // Signal to use fallback
          return;
        }

        // Load file with MP4Box
        const loaded = await this.loadWithMP4Box(file);
        if (!loaded) {
          console.warn('[ProxyGen] Failed to parse video, falling back to legacy method');
          resolve(null);
          return;
        }

        console.log(`[ProxyGen] Video loaded: ${this.outputWidth}x${this.outputHeight}, ${this.totalFrames} frames at ${PROXY_FPS}fps`);

        // Create canvas for frame extraction
        this.canvas = new OffscreenCanvas(this.outputWidth, this.outputHeight);
        this.ctx = this.canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;

        // Calculate target frame timestamps (at PROXY_FPS)
        for (let i = 0; i < this.totalFrames; i++) {
          this.frameTimestamps.push(i / PROXY_FPS);
        }

        // Initialize decoder
        this.initDecoder();

        // Process samples
        await this.processSamples(mediaFileId, saveFrame);

      } catch (error) {
        console.error('[ProxyGen] Generation failed:', error);
        _reject(error);
      }
    });
  }

  private async loadWithMP4Box(file: File): Promise<boolean> {
    return new Promise(async (resolve) => {
      this.mp4File = MP4Box.createFile();
      const mp4File = this.mp4File!;

      mp4File.onReady = (info: { videoTracks: MP4VideoTrack[] }) => {
        if (info.videoTracks.length === 0) {
          resolve(false);
          return;
        }

        this.videoTrack = info.videoTracks[0];
        const track = this.videoTrack;

        // Calculate output dimensions
        let width = track.video.width;
        let height = track.video.height;
        if (width > PROXY_MAX_WIDTH) {
          height = Math.round((PROXY_MAX_WIDTH / width) * height);
          width = PROXY_MAX_WIDTH;
        }
        this.outputWidth = width;
        this.outputHeight = height;

        this.duration = track.duration / track.timescale;
        this.totalFrames = Math.ceil(this.duration * PROXY_FPS);

        // Get codec config
        const trak = this.mp4File!.getTrackById(track.id);
        const codecString = this.getCodecString(track.codec, trak);

        this.codecConfig = {
          codec: codecString,
          codedWidth: track.video.width,
          codedHeight: track.video.height,
          hardwareAcceleration: 'prefer-hardware',
        };

        // Check if codec is supported
        VideoDecoder.isConfigSupported(this.codecConfig).then(support => {
          if (!support.supported) {
            console.warn('[ProxyGen] Codec not supported:', codecString);
            resolve(false);
            return;
          }

          // Extract all samples
          mp4File.setExtractionOptions(track.id, null, { nbSamples: Infinity });
          mp4File.start();
        });
      };

      mp4File.onSamples = (_trackId: number, _ref: any, samples: Sample[]) => {
        this.samples.push(...samples);
      };

      mp4File.onError = (error: string) => {
        console.error('[ProxyGen] MP4Box error:', error);
        resolve(false);
      };

      // Read file in chunks
      const reader = file.stream().getReader();
      let offset = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const buffer = value.buffer as MP4ArrayBuffer;
          buffer.fileStart = offset;
          offset += value.byteLength;
          mp4File.appendBuffer(buffer);
        }
        mp4File.flush();

        // Wait a bit for onReady to be called
        await new Promise(r => setTimeout(r, 100));

        if (this.videoTrack) {
          resolve(true);
        } else {
          resolve(false);
        }
      } catch (e) {
        console.error('[ProxyGen] File read error:', e);
        resolve(false);
      }
    });
  }

  private getCodecString(codec: string, trak: any): string {
    // Handle different codec types
    if (codec.startsWith('avc1')) {
      // H.264/AVC
      const avcC = trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]?.avcC;
      if (avcC) {
        const profile = avcC.AVCProfileIndication.toString(16).padStart(2, '0');
        const compat = avcC.profile_compatibility.toString(16).padStart(2, '0');
        const level = avcC.AVCLevelIndication.toString(16).padStart(2, '0');
        return `avc1.${profile}${compat}${level}`;
      }
      return 'avc1.640028'; // Fallback to high profile
    }

    if (codec.startsWith('hvc1') || codec.startsWith('hev1')) {
      // H.265/HEVC
      return codec;
    }

    if (codec.startsWith('vp09')) {
      return codec;
    }

    if (codec.startsWith('av01')) {
      return codec;
    }

    return codec;
  }

  private initDecoder() {
    if (!this.codecConfig) return;

    this.decoder = new VideoDecoder({
      output: (frame) => this.handleDecodedFrame(frame),
      error: (error) => console.error('[ProxyGen] Decoder error:', error),
    });

    this.decoder.configure(this.codecConfig);
    console.log('[ProxyGen] Decoder configured:', this.codecConfig.codec);
  }

  private handleDecodedFrame(frame: VideoFrame) {
    // Find the closest target frame index for this timestamp
    const timestamp = frame.timestamp / 1_000_000; // Convert to seconds
    const frameIndex = Math.round(timestamp * PROXY_FPS);

    if (frameIndex >= 0 && frameIndex < this.totalFrames) {
      // Store frame for processing
      this.decodedFrames.set(frameIndex, frame);
    } else {
      frame.close();
    }
  }

  private async processSamples(
    mediaFileId: string,
    saveFrame: (frame: { id: string; mediaFileId: string; frameIndex: number; blob: Blob }) => Promise<void>
  ): Promise<void> {
    if (!this.decoder || !this.ctx || !this.canvas) return;

    const targetFrameSet = new Set<number>();
    for (let i = 0; i < this.totalFrames; i++) {
      targetFrameSet.add(i);
    }

    // Decode all samples
    console.log(`[ProxyGen] Decoding ${this.samples.length} samples...`);

    for (const sample of this.samples) {
      if (this.checkCancelled?.()) {
        this.isCancelled = true;
        break;
      }

      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts / sample.timescale) * 1_000_000,
        duration: (sample.duration / sample.timescale) * 1_000_000,
        data: sample.data,
      });

      this.decoder.decode(chunk);
    }

    // Wait for decoder to finish
    await this.decoder.flush();
    console.log(`[ProxyGen] Decoded ${this.decodedFrames.size} frames`);

    // Process decoded frames
    const batch: { id: string; mediaFileId: string; frameIndex: number; blob: Blob }[] = [];

    // Sort frame indices
    const sortedIndices = Array.from(this.decodedFrames.keys()).sort((a, b) => a - b);

    for (const frameIndex of sortedIndices) {
      if (this.checkCancelled?.()) {
        this.isCancelled = true;
        break;
      }

      const frame = this.decodedFrames.get(frameIndex);
      if (!frame) continue;

      try {
        // Draw frame to canvas
        this.ctx.drawImage(frame, 0, 0, this.outputWidth, this.outputHeight);
        frame.close();
        this.decodedFrames.delete(frameIndex);

        // Get image data for worker
        const imageData = this.ctx.getImageData(0, 0, this.outputWidth, this.outputHeight);

        // Encode in parallel using workers
        const blob = await this.encodeFrameAsync(frameIndex, imageData);

        // Add to batch
        const frameId = `${mediaFileId}_${frameIndex.toString().padStart(6, '0')}`;
        batch.push({ id: frameId, mediaFileId, frameIndex, blob });

        this.processedFrames++;
        this.onProgress?.(Math.round((this.processedFrames / this.totalFrames) * 100));

        // Save batch
        if (batch.length >= BATCH_SIZE) {
          for (const f of batch) {
            await saveFrame(f);
          }
          batch.length = 0;
        }
      } catch (e) {
        console.error('[ProxyGen] Frame processing error:', e);
        frame.close();
      }
    }

    // Save remaining batch
    for (const f of batch) {
      await saveFrame(f);
    }

    // Clean up remaining frames
    for (const frame of this.decodedFrames.values()) {
      frame.close();
    }
    this.decodedFrames.clear();

    if (this.isCancelled) {
      this.resolveGeneration?.(null);
    } else {
      this.resolveGeneration?.({
        frameCount: this.processedFrames,
        fps: PROXY_FPS,
      });
    }
  }

  destroy() {
    this.decoder?.close();
    this.workers.forEach(w => w.terminate());
    for (const frame of this.decodedFrames.values()) {
      frame.close();
    }
    this.decodedFrames.clear();
  }
}

// Singleton instance
let generatorInstance: ProxyGeneratorGPU | null = null;

export function getProxyGenerator(): ProxyGeneratorGPU {
  if (!generatorInstance) {
    generatorInstance = new ProxyGeneratorGPU();
  }
  return generatorInstance;
}

export { ProxyGeneratorGPU, PROXY_FPS, PROXY_QUALITY, PROXY_MAX_WIDTH };
