// GPU-accelerated proxy frame generator using WebCodecs + WebGPU
// Uses VideoDecoder for hardware decoding and GPU for batch resize
// Workers handle parallel WebP encoding

import * as MP4BoxModule from 'mp4box';
import { engine } from '../engine/WebGPUEngine';
import { ProxyResizePipeline } from '../engine/proxy/ProxyResizePipeline';

const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

// Configuration
const PROXY_FPS = 30;
const PROXY_QUALITY = 0.92;
const PROXY_MAX_WIDTH = 1280; // Changed from 1920 for faster generation
const WORKER_POOL_SIZE = navigator.hardwareConcurrency || 4;
const BATCH_SIZE = ProxyResizePipeline.getBatchSize(); // 16 frames per GPU batch
const DB_BATCH_SIZE = 10; // IndexedDB write batch size

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

interface GeneratorResult {
  frameCount: number;
  fps: number;
}

// Worker code that accepts raw RGBA pixels
const workerCode = `
  self.onmessage = async function(e) {
    const { frameIndex, pixels, width, height, quality } = e.data;

    // Create ImageData from raw pixels
    const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);

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
  private frameQueue: { frameIndex: number; pixels: Uint8Array }[] = [];

  private videoTrack: MP4VideoTrack | null = null;
  private samples: Sample[] = [];
  private codecConfig: VideoDecoderConfig | null = null;

  // GPU resize pipeline
  private resizePipeline: ProxyResizePipeline | null = null;

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
      pixels: frame.pixels,
      width: this.outputWidth,
      height: this.outputHeight,
      quality: PROXY_QUALITY,
    }, [frame.pixels.buffer]); // Transfer buffer for performance
  }

  private encodeFrameAsync(frameIndex: number, pixels: Uint8Array): Promise<Blob> {
    return new Promise((resolve, reject) => {
      this.pendingFrames.set(frameIndex, { resolve, reject });
      this.frameQueue.push({ frameIndex, pixels });
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

    return new Promise(async (resolve, reject) => {
      this.resolveGeneration = resolve;

      try {
        // Check for WebCodecs support
        if (!('VideoDecoder' in window)) {
          reject(new Error('WebCodecs VideoDecoder not available in this browser'));
          return;
        }

        // Check for WebGPU support
        const device = engine.getDevice();
        if (!device) {
          reject(new Error('WebGPU device not available'));
          return;
        }

        // Log GPU info for verification
        const adapterInfo = (device as any).adapterInfo;
        console.log('%c[ProxyGen] 🎮 GPU ACCELERATION ACTIVE', 'color: #00ff00; font-weight: bold; font-size: 14px');
        console.log('[ProxyGen] GPU Device:', {
          vendor: adapterInfo?.vendor || 'unknown',
          architecture: adapterInfo?.architecture || 'unknown',
          device: adapterInfo?.device || 'unknown',
          description: adapterInfo?.description || 'WebGPU Device',
        });

        // Load file with MP4Box
        const loaded = await this.loadWithMP4Box(file);
        if (!loaded) {
          reject(new Error('Failed to parse video file or no supported codec found'));
          return;
        }

        console.log(`[ProxyGen] Video loaded: ${this.videoTrack!.video.width}x${this.videoTrack!.video.height}, targeting ${this.outputWidth}x${this.outputHeight} at ${PROXY_FPS}fps`);

        // Initialize GPU resize pipeline
        this.resizePipeline = new ProxyResizePipeline(device);
        this.resizePipeline.initializeAtlas(
          this.videoTrack!.video.width,
          this.videoTrack!.video.height,
          PROXY_MAX_WIDTH
        );

        // Update output dimensions from pipeline
        const dims = this.resizePipeline.getFrameDimensions();
        this.outputWidth = dims.width;
        this.outputHeight = dims.height;

        // Calculate target frame timestamps (at PROXY_FPS)
        for (let i = 0; i < this.totalFrames; i++) {
          this.frameTimestamps.push(i / PROXY_FPS);
        }

        // Initialize decoder
        this.initDecoder();

        // Process samples in batches
        await this.processSamplesGPU(mediaFileId, saveFrame);

      } catch (error) {
        console.error('[ProxyGen] Generation failed:', error);
        reject(error);
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

        // Calculate output dimensions (will be refined by GPU pipeline)
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

        // Get codec config with fallback support
        const trak = this.mp4File!.getTrackById(track.id);
        const codecString = this.getCodecString(track.codec, trak);

        console.log(`[ProxyGen] Detected codec: ${codecString}, trying to find supported configuration...`);

        // Try to find a supported codec configuration
        this.findSupportedCodec(codecString, track.video.width, track.video.height).then(config => {
          if (!config) {
            console.warn('[ProxyGen] No supported codec configuration found');
            resolve(false);
            return;
          }

          this.codecConfig = config;

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

  /**
   * Try multiple codec configurations until one is supported
   */
  private async findSupportedCodec(
    baseCodec: string,
    width: number,
    height: number
  ): Promise<VideoDecoderConfig | null> {
    // Common H.264 codec strings to try
    const h264Fallbacks = [
      baseCodec,
      'avc1.42001e', // Baseline L3.0
      'avc1.4d001e', // Main L3.0
      'avc1.64001e', // High L3.0
      'avc1.640028', // High L4.0
      'avc1.4d0028', // Main L4.0
      'avc1.42E01E', // Constrained Baseline
      'avc1.4D401E', // Main
      'avc1.640029', // High L4.1
    ];

    const codecsToTry = baseCodec.startsWith('avc1') ? h264Fallbacks : [baseCodec];

    console.log(`[ProxyGen] Testing ${codecsToTry.length} codec configurations for ${width}x${height}...`);

    // Try without specifying hardwareAcceleration (let browser decide)
    for (const codec of codecsToTry) {
      const config: VideoDecoderConfig = {
        codec,
        codedWidth: width,
        codedHeight: height,
      };

      try {
        const support = await VideoDecoder.isConfigSupported(config);
        console.log(`[ProxyGen] Codec ${codec}: ${support.supported ? '✓ supported' : '✗ not supported'}`);
        if (support.supported) {
          return config;
        }
      } catch (e) {
        console.log(`[ProxyGen] Codec ${codec}: ✗ error - ${e}`);
      }
    }

    console.warn(`[ProxyGen] No supported codec found for ${baseCodec}`);
    return null;
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
      // Store frame for batch processing
      this.decodedFrames.set(frameIndex, frame);
    } else {
      frame.close();
    }
  }

  private async processSamplesGPU(
    mediaFileId: string,
    saveFrame: (frame: { id: string; mediaFileId: string; frameIndex: number; blob: Blob }) => Promise<void>
  ): Promise<void> {
    if (!this.decoder || !this.resizePipeline) return;

    console.log(`[ProxyGen] Decoding ${this.samples.length} samples with GPU batch processing...`);

    // Decode all samples first
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
    console.log(`[ProxyGen] Decoded ${this.decodedFrames.size} frames, starting GPU batch processing...`);

    // Sort frame indices
    const sortedIndices = Array.from(this.decodedFrames.keys()).sort((a, b) => a - b);

    // Performance tracking
    const startTime = performance.now();
    let totalGpuTime = 0;
    let totalEncodeTime = 0;
    let batchCount = 0;

    // Process frames in batches
    let batchStart = 0;
    const dbBatch: { id: string; mediaFileId: string; frameIndex: number; blob: Blob }[] = [];

    while (batchStart < sortedIndices.length) {
      if (this.checkCancelled?.()) {
        this.isCancelled = true;
        break;
      }

      // Collect frames for this GPU batch
      const batchIndices = sortedIndices.slice(batchStart, batchStart + BATCH_SIZE);
      const batchFrames: VideoFrame[] = [];
      const batchFrameIndices: number[] = [];

      for (const frameIndex of batchIndices) {
        const frame = this.decodedFrames.get(frameIndex);
        if (frame) {
          batchFrames.push(frame);
          batchFrameIndices.push(frameIndex);
        }
      }

      if (batchFrames.length > 0) {
        try {
          batchCount++;

          // GPU batch resize with timing
          const gpuStart = performance.now();
          const pixelArrays = await this.resizePipeline.processBatch(batchFrames);
          const gpuEnd = performance.now();
          const gpuTime = gpuEnd - gpuStart;
          totalGpuTime += gpuTime;

          // Close video frames after GPU processing
          for (const frame of batchFrames) {
            frame.close();
          }

          // Encode all frames in parallel using workers
          const encodeStart = performance.now();
          const encodePromises = pixelArrays.map((pixels, i) =>
            this.encodeFrameAsync(batchFrameIndices[i], pixels)
          );

          const blobs = await Promise.all(encodePromises);
          const encodeEnd = performance.now();
          const encodeTime = encodeEnd - encodeStart;
          totalEncodeTime += encodeTime;

          // Log batch performance (every 5 batches to avoid spam)
          if (batchCount % 5 === 0 || batchCount === 1) {
            console.log(`[ProxyGen] Batch ${batchCount}: GPU=${gpuTime.toFixed(1)}ms, Encode=${encodeTime.toFixed(1)}ms, Frames=${batchFrames.length}`);
          }

          // Add to DB batch
          for (let i = 0; i < blobs.length; i++) {
            const frameIndex = batchFrameIndices[i];
            const frameId = `${mediaFileId}_${frameIndex.toString().padStart(6, '0')}`;
            dbBatch.push({ id: frameId, mediaFileId, frameIndex, blob: blobs[i] });
            this.decodedFrames.delete(frameIndex);
          }

          this.processedFrames += batchFrames.length;
          this.onProgress?.(Math.round((this.processedFrames / this.totalFrames) * 100));

          // Save DB batch periodically
          if (dbBatch.length >= DB_BATCH_SIZE) {
            for (const f of dbBatch) {
              await saveFrame(f);
            }
            dbBatch.length = 0;
          }

        } catch (e) {
          console.error('[ProxyGen] GPU batch processing error:', e);
          // Close frames on error
          for (const frame of batchFrames) {
            frame.close();
          }
        }
      }

      batchStart += BATCH_SIZE;
    }

    // Save remaining DB batch
    for (const f of dbBatch) {
      await saveFrame(f);
    }

    // Clean up remaining frames
    for (const frame of this.decodedFrames.values()) {
      frame.close();
    }
    this.decodedFrames.clear();

    // Clean up GPU pipeline
    this.resizePipeline.destroy();
    this.resizePipeline = null;

    if (this.isCancelled) {
      this.resolveGeneration?.(null);
    } else {
      const totalTime = performance.now() - startTime;
      const fps = this.processedFrames / (totalTime / 1000);

      console.log('%c[ProxyGen] ✅ GPU Processing Complete', 'color: #00ff00; font-weight: bold');
      console.log('[ProxyGen] Performance Summary:', {
        totalFrames: this.processedFrames,
        totalBatches: batchCount,
        totalTime: `${(totalTime / 1000).toFixed(1)}s`,
        framesPerSecond: fps.toFixed(1),
        gpuTime: `${(totalGpuTime / 1000).toFixed(2)}s (${((totalGpuTime / totalTime) * 100).toFixed(1)}%)`,
        encodeTime: `${(totalEncodeTime / 1000).toFixed(2)}s (${((totalEncodeTime / totalTime) * 100).toFixed(1)}%)`,
        avgGpuPerBatch: `${(totalGpuTime / batchCount).toFixed(1)}ms`,
        avgGpuPerFrame: `${(totalGpuTime / this.processedFrames).toFixed(2)}ms`,
      });

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
    this.resizePipeline?.destroy();
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
