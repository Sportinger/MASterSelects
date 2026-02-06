// Proxy frame generator using WebCodecs VideoDecoder + parallel OffscreenCanvas → JPEG
// Decodes source video with hardware VideoDecoder, resizes on a pool of OffscreenCanvases,
// then saves individual JPEG frames via convertToBlob for instant scrubbing.

import { Logger } from './logger';
import * as MP4BoxModule from 'mp4box';

const log = Logger.create('ProxyGenerator');

const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

// Configuration
const PROXY_FPS = 30;
const PROXY_MAX_WIDTH = 1280;
const JPEG_QUALITY = 0.82;
const CANVAS_POOL_SIZE = 8;       // Parallel encoding canvases
const DECODE_BATCH_SIZE = 30;     // Feed 30 samples at a time before yielding

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

interface CanvasSlot {
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
}

class ProxyGeneratorWebCodecs {
  private mp4File: MP4File | null = null;
  private decoder: VideoDecoder | null = null;

  private videoTrack: MP4VideoTrack | null = null;
  private samples: Sample[] = [];
  private codecConfig: VideoDecoderConfig | null = null;

  private outputWidth = 0;
  private outputHeight = 0;
  private duration = 0;
  private totalFrames = 0;
  private processedFrames = 0;
  private savedFrameIndices = new Set<number>();
  private decodedFrames: Map<number, VideoFrame> = new Map();

  // Pool of canvases for parallel encoding
  private canvasPool: CanvasSlot[] = [];

  private onProgress: ((progress: number) => void) | null = null;
  private checkCancelled: (() => boolean) | null = null;
  private saveFrame: ((frame: { frameIndex: number; blob: Blob }) => Promise<void>) | null = null;
  private isCancelled = false;

  async generate(
    file: File,
    _mediaFileId: string,
    onProgress: (progress: number) => void,
    checkCancelled: () => boolean,
    saveFrame: (frame: { frameIndex: number; blob: Blob }) => Promise<void>,
  ): Promise<{ frameCount: number; fps: number } | null> {
    this.onProgress = onProgress;
    this.checkCancelled = checkCancelled;
    this.saveFrame = saveFrame;
    this.isCancelled = false;
    this.processedFrames = 0;
    this.savedFrameIndices.clear();
    this.samples = [];
    this.decodedFrames.clear();
    this.canvasPool = [];

    try {
      if (!('VideoDecoder' in window)) {
        throw new Error('WebCodecs VideoDecoder not available');
      }

      // Load file with MP4Box
      const loaded = await this.loadWithMP4Box(file);
      if (!loaded) {
        throw new Error('Failed to parse video file or no supported codec found');
      }

      log.info(`Source: ${this.videoTrack!.video.width}x${this.videoTrack!.video.height} → Proxy: ${this.outputWidth}x${this.outputHeight} @ ${PROXY_FPS}fps`);

      // Initialize canvas pool for parallel encoding
      for (let i = 0; i < CANVAS_POOL_SIZE; i++) {
        const canvas = new OffscreenCanvas(this.outputWidth, this.outputHeight);
        const ctx = canvas.getContext('2d')!;
        this.canvasPool.push({ canvas, ctx });
      }

      // Initialize decoder
      this.initDecoder();

      // Process all samples
      try {
        await this.processSamples();
      } catch (firstError) {
        log.warn('First decode attempt failed, trying without description...');
        this.closeDecodedFrames();
        this.processedFrames = 0;
        this.savedFrameIndices.clear();

        if (this.codecConfig?.description) {
          const configWithoutDesc: VideoDecoderConfig = {
            codec: this.codecConfig.codec,
            codedWidth: this.codecConfig.codedWidth,
            codedHeight: this.codecConfig.codedHeight,
          };
          const support = await VideoDecoder.isConfigSupported(configWithoutDesc);
          if (support.supported) {
            log.info('Retrying without description...');
            this.codecConfig = configWithoutDesc;
            this.decoder?.close();
            this.initDecoder();
            await this.processSamples();
          } else {
            throw firstError;
          }
        } else {
          throw firstError;
        }
      }

      // Finalize
      if (this.isCancelled || this.processedFrames === 0) {
        this.cleanup();
        if (this.isCancelled) {
          log.info('Generation cancelled');
          return null;
        }
        log.error('No frames were processed!');
        return null;
      }

      log.info(`Proxy complete: ${this.processedFrames} frames saved as JPEG`);
      this.cleanup();

      return {
        frameCount: this.processedFrames,
        fps: PROXY_FPS,
      };
    } catch (error) {
      log.error('Generation failed', error);
      this.cleanup();
      throw error;
    }
  }

  private async loadWithMP4Box(file: File): Promise<boolean> {
    // eslint-disable-next-line no-async-promise-executor
    return new Promise(async (resolve) => {
      this.mp4File = MP4Box.createFile();
      const mp4File = this.mp4File!;
      let expectedSamples = 0;
      let samplesReady = false;
      let codecReady = false;

      const checkComplete = () => {
        if (codecReady && samplesReady) {
          log.info(`Extracted ${this.samples.length} samples from video`);
          resolve(true);
        }
      };

      mp4File.onReady = async (info: { videoTracks: MP4VideoTrack[] }) => {
        if (info.videoTracks.length === 0) {
          resolve(false);
          return;
        }

        this.videoTrack = info.videoTracks[0];
        const track = this.videoTrack;
        expectedSamples = track.nb_samples;

        // Calculate output dimensions
        let width = track.video.width;
        let height = track.video.height;
        if (width > PROXY_MAX_WIDTH) {
          height = Math.round((PROXY_MAX_WIDTH / width) * height);
          width = PROXY_MAX_WIDTH;
        }
        // Ensure even dimensions
        this.outputWidth = width & ~1;
        this.outputHeight = height & ~1;

        this.duration = track.duration / track.timescale;
        this.totalFrames = Math.ceil(this.duration * PROXY_FPS);

        log.info(`Duration: ${this.duration.toFixed(3)}s, totalFrames: ${this.totalFrames}, samples: ${expectedSamples}`);

        // Get codec config
        const trak = this.mp4File!.getTrackById(track.id);
        const codecString = this.getCodecString(track.codec, trak);
        log.debug(`Detected codec: ${codecString}`);

        // Get AVC description
        let description: Uint8Array | undefined;
        if (codecString.startsWith('avc1')) {
          const avcC = trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]?.avcC;
          if (avcC) {
            const stream = new (MP4Box as any).DataStream(undefined, 0, (MP4Box as any).DataStream.BIG_ENDIAN);
            avcC.write(stream);
            const totalWritten = stream.position || stream.buffer.byteLength;
            if (totalWritten > 8) {
              description = new Uint8Array(stream.buffer.slice(8, totalWritten));
              log.debug(`Got AVC description: ${description.length} bytes`);
            }
          }
        }

        const config = await this.findSupportedCodec(codecString, track.video.width, track.video.height, description);
        if (!config) {
          resolve(false);
          return;
        }

        this.codecConfig = config;
        codecReady = true;

        mp4File.setExtractionOptions(track.id, null, { nbSamples: Infinity });
        mp4File.start();
        mp4File.flush();
        checkComplete();
      };

      mp4File.onSamples = (_trackId: number, _ref: any, samples: Sample[]) => {
        this.samples.push(...samples);
        if (this.samples.length >= expectedSamples) {
          samplesReady = true;
          checkComplete();
        }
      };

      mp4File.onError = (error: string) => {
        log.error('MP4Box error', error);
        resolve(false);
      };

      const fileData = await file.arrayBuffer();

      try {
        const buffer1 = fileData.slice(0) as MP4ArrayBuffer;
        buffer1.fileStart = 0;
        mp4File.appendBuffer(buffer1);
        mp4File.flush();

        // Poll for codec readiness
        const maxCodecWait = 3000;
        const pollStart = performance.now();
        while (!codecReady && performance.now() - pollStart < maxCodecWait) {
          await new Promise(r => setTimeout(r, 20));
        }

        if (!codecReady) {
          log.warn('Codec not ready after polling');
          resolve(false);
          return;
        }

        if (this.samples.length === 0) {
          const buffer2 = fileData.slice(0) as MP4ArrayBuffer;
          buffer2.fileStart = 0;
          mp4File.appendBuffer(buffer2);
          mp4File.flush();
        }

        // Poll for samples
        const maxSampleWait = 3000;
        const samplePollStart = performance.now();
        while (!samplesReady && performance.now() - samplePollStart < maxSampleWait) {
          if (this.samples.length > 0 && this.samples.length >= expectedSamples) {
            samplesReady = true;
            break;
          }
          await new Promise(r => setTimeout(r, 20));
        }

        if (!samplesReady && this.samples.length > 0) {
          samplesReady = true;
        }

        if (samplesReady) {
          checkComplete();
        } else {
          log.error('No samples extracted');
          resolve(false);
        }
      } catch (e) {
        log.error('File read error', e);
        resolve(false);
      }
    });
  }

  private getCodecString(codec: string, trak: any): string {
    if (codec.startsWith('avc1')) {
      const avcC = trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]?.avcC;
      if (avcC) {
        const profile = avcC.AVCProfileIndication.toString(16).padStart(2, '0');
        const compat = avcC.profile_compatibility.toString(16).padStart(2, '0');
        const level = avcC.AVCLevelIndication.toString(16).padStart(2, '0');
        return `avc1.${profile}${compat}${level}`;
      }
      return 'avc1.640028';
    }
    return codec;
  }

  private async findSupportedCodec(
    baseCodec: string,
    width: number,
    height: number,
    description?: Uint8Array
  ): Promise<VideoDecoderConfig | null> {
    const h264Fallbacks = [
      baseCodec,
      'avc1.42001e', 'avc1.4d001e', 'avc1.64001e',
      'avc1.640028', 'avc1.4d0028', 'avc1.42E01E',
      'avc1.4D401E', 'avc1.640029',
    ];

    const codecsToTry = baseCodec.startsWith('avc1') ? h264Fallbacks : [baseCodec];

    for (const codec of codecsToTry) {
      const config: VideoDecoderConfig = {
        codec,
        codedWidth: width,
        codedHeight: height,
        hardwareAcceleration: 'prefer-hardware',
        ...(description && { description }),
      };

      try {
        const support = await VideoDecoder.isConfigSupported(config);
        if (support.supported) {
          log.debug(`Decoder codec ${codec}: supported`);
          return config;
        }
      } catch {
        // Try next
      }
    }

    log.warn(`No supported decoder codec found for ${baseCodec}`);
    return null;
  }

  private initDecoder() {
    if (!this.codecConfig) return;

    let errorCount = 0;

    this.decoder = new VideoDecoder({
      output: (frame) => {
        this.handleDecodedFrame(frame);
      },
      error: (error) => {
        errorCount++;
        if (errorCount <= 5) {
          log.error('Decoder error', error.message || error);
        }
      },
    });

    this.decoder.configure(this.codecConfig);
    log.debug('Decoder configured', {
      codec: this.codecConfig.codec,
      size: `${this.codecConfig.codedWidth}x${this.codecConfig.codedHeight}`,
    });
  }

  private handleDecodedFrame(frame: VideoFrame) {
    const timestamp = frame.timestamp / 1_000_000;
    const frameIndex = Math.round(timestamp * PROXY_FPS);

    if (frameIndex >= 0 && frameIndex < this.totalFrames && !this.savedFrameIndices.has(frameIndex)) {
      const existing = this.decodedFrames.get(frameIndex);
      if (existing) existing.close();
      this.decodedFrames.set(frameIndex, frame);
    } else {
      frame.close();
    }
  }

  /**
   * Encode + save a batch of decoded frames in parallel using the canvas pool.
   * Each canvas independently does drawImage → convertToBlob → saveFrame.
   */
  private async processAccumulatedFrames(): Promise<void> {
    if (this.decodedFrames.size === 0) return;

    const sortedIndices = Array.from(this.decodedFrames.keys()).sort((a, b) => a - b);

    // Process in chunks of CANVAS_POOL_SIZE
    for (let offset = 0; offset < sortedIndices.length; offset += CANVAS_POOL_SIZE) {
      const batch = sortedIndices.slice(offset, offset + CANVAS_POOL_SIZE);

      await Promise.all(batch.map(async (frameIdx, slotIdx) => {
        const frame = this.decodedFrames.get(frameIdx);
        if (!frame) return;
        this.decodedFrames.delete(frameIdx);

        const slot = this.canvasPool[slotIdx];

        // Resize onto this slot's canvas
        slot.ctx.drawImage(frame, 0, 0, this.outputWidth, this.outputHeight);
        frame.close();

        // Encode to JPEG (3-5x faster than WebP)
        const blob = await slot.canvas.convertToBlob({
          type: 'image/jpeg',
          quality: JPEG_QUALITY,
        });

        // Save via callback
        await this.saveFrame!({ frameIndex: frameIdx, blob });

        this.savedFrameIndices.add(frameIdx);
        this.processedFrames++;
      }));

      this.onProgress?.(Math.min(100, Math.round((this.processedFrames / this.totalFrames) * 100)));
    }
  }

  private async processSamples(): Promise<void> {
    if (!this.decoder) return;

    const sortedSamples = [...this.samples].sort((a, b) => a.dts - b.dts);

    const keyframeCount = sortedSamples.filter(s => s.is_sync).length;
    log.info(`Decoding ${sortedSamples.length} samples (${keyframeCount} keyframes)...`);

    const firstKeyframeIdx = sortedSamples.findIndex(s => s.is_sync);
    if (firstKeyframeIdx === -1) throw new Error('No keyframes found');

    const startTime = performance.now();
    let decodeErrors = 0;

    // Feed decoder in batches, then yield + process accumulated frames
    for (let batchStart = firstKeyframeIdx; batchStart < sortedSamples.length; batchStart += DECODE_BATCH_SIZE) {
      if (this.checkCancelled?.()) {
        this.isCancelled = true;
        break;
      }

      if (this.decoder.state === 'closed') {
        log.error('Decoder closed unexpectedly');
        break;
      }

      const batchEnd = Math.min(batchStart + DECODE_BATCH_SIZE, sortedSamples.length);

      // Feed a batch of samples to the decoder
      for (let i = batchStart; i < batchEnd; i++) {
        const sample = sortedSamples[i];

        const chunk = new EncodedVideoChunk({
          type: sample.is_sync ? 'key' : 'delta',
          timestamp: (sample.cts / sample.timescale) * 1_000_000,
          duration: (sample.duration / sample.timescale) * 1_000_000,
          data: sample.data,
        });

        try {
          this.decoder.decode(chunk);
        } catch {
          decodeErrors++;
          if (decodeErrors > 50) {
            log.error('Too many decode errors, stopping');
            return;
          }
        }
      }

      // Yield to let decoder output callbacks fire
      await new Promise(resolve => setTimeout(resolve, 0));

      // Process accumulated decoded frames in parallel
      if (this.decodedFrames.size >= CANVAS_POOL_SIZE) {
        await this.processAccumulatedFrames();
      }
    }

    // Flush decoder
    try {
      if (this.decoder.state !== 'closed') {
        await Promise.race([
          this.decoder.flush(),
          new Promise<void>((_, reject) => setTimeout(() => reject(new Error('Flush timeout')), 5000))
        ]);
      }
    } catch { /* ignore */ }

    try {
      if (this.decoder.state !== 'closed') this.decoder.close();
    } catch { /* ignore */ }

    // Yield to collect last decoded frames
    await new Promise(resolve => setTimeout(resolve, 10));

    // Process all remaining decoded frames
    if (this.decodedFrames.size > 0) {
      await this.processAccumulatedFrames();
    }

    const totalTime = performance.now() - startTime;
    const fps = this.processedFrames / (totalTime / 1000);
    log.info(`Complete: ${this.processedFrames}/${this.totalFrames} frames in ${(totalTime / 1000).toFixed(1)}s (${fps.toFixed(1)} fps encode)`);
  }

  private closeDecodedFrames() {
    for (const frame of this.decodedFrames.values()) frame.close();
    this.decodedFrames.clear();
  }

  private cleanup() {
    try { this.decoder?.close(); } catch { /* ignore */ }
    this.closeDecodedFrames();
    this.canvasPool = [];
    this.decoder = null;
  }
}

// Singleton instance
let generatorInstance: ProxyGeneratorWebCodecs | null = null;

export function getProxyGenerator(): ProxyGeneratorWebCodecs {
  if (!generatorInstance) {
    generatorInstance = new ProxyGeneratorWebCodecs();
  }
  return generatorInstance;
}

export { ProxyGeneratorWebCodecs, PROXY_FPS, PROXY_MAX_WIDTH };
