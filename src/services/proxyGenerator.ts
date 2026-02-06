// Proxy video generator using WebCodecs VideoEncoder + Mp4Muxer
// Decodes source → resizes on OffscreenCanvas → encodes all-keyframe H.264 → single MP4
// All keyframes = every frame independently decodable = instant random access for scrubbing

import { Logger } from './logger';
import * as MP4BoxModule from 'mp4box';
import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4Target } from 'mp4-muxer';

const log = Logger.create('ProxyGenerator');

const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

// Configuration
const PROXY_FPS = 30;
const PROXY_MAX_WIDTH = 1280;
const PROXY_BITRATE = 5_000_000; // 5 Mbps
const ENCODER_CODEC = 'avc1.4d0028'; // Main Profile L4.0

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

export interface GeneratorResult {
  blob: Blob;
  frameCount: number;
  fps: number;
}

class ProxyGeneratorMP4 {
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
  private decodedFrames: Map<number, VideoFrame> = new Map();

  private resizeCanvas: OffscreenCanvas | null = null;
  private resizeCtx: OffscreenCanvasRenderingContext2D | null = null;

  private encoder: VideoEncoder | null = null;
  private muxer: Mp4Muxer<Mp4Target> | null = null;

  private onProgress: ((progress: number) => void) | null = null;
  private checkCancelled: (() => boolean) | null = null;
  private isCancelled = false;

  async generate(
    file: File,
    onProgress: (progress: number) => void,
    checkCancelled: () => boolean,
  ): Promise<GeneratorResult | null> {
    this.onProgress = onProgress;
    this.checkCancelled = checkCancelled;
    this.isCancelled = false;
    this.processedFrames = 0;
    this.samples = [];
    this.decodedFrames.clear();

    try {
      if (!('VideoDecoder' in window) || !('VideoEncoder' in window)) {
        throw new Error('WebCodecs not available');
      }

      // Load file with MP4Box
      const loaded = await this.loadWithMP4Box(file);
      if (!loaded) {
        throw new Error('Failed to parse video file or no supported codec found');
      }

      log.info(`Source: ${this.videoTrack!.video.width}x${this.videoTrack!.video.height} → Proxy: ${this.outputWidth}x${this.outputHeight} @ ${PROXY_FPS}fps`);

      // Check encoder support
      const encoderSupport = await VideoEncoder.isConfigSupported({
        codec: ENCODER_CODEC,
        width: this.outputWidth,
        height: this.outputHeight,
        bitrate: PROXY_BITRATE,
        framerate: PROXY_FPS,
      });
      if (!encoderSupport.supported) {
        throw new Error(`Encoder codec ${ENCODER_CODEC} not supported`);
      }

      // Initialize resize canvas
      this.resizeCanvas = new OffscreenCanvas(this.outputWidth, this.outputHeight);
      this.resizeCtx = this.resizeCanvas.getContext('2d')!;

      // Initialize encoder + muxer
      this.initEncoderAndMuxer();

      // Initialize decoder
      this.initDecoder();

      // Process all samples
      try {
        await this.processSamples();
      } catch (firstError) {
        log.warn('First decode attempt failed, trying without description...');
        this.decodedFrames.clear();
        this.processedFrames = 0;

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
            // Re-init encoder/muxer for clean state
            this.encoder?.close();
            this.initEncoderAndMuxer();
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

      // Flush encoder and finalize muxer
      await this.encoder!.flush();
      this.encoder!.close();
      this.muxer!.finalize();

      const { buffer } = this.muxer!.target;
      const blob = new Blob([buffer], { type: 'video/mp4' });

      log.info(`Proxy complete: ${this.processedFrames} frames, ${(blob.size / 1024 / 1024).toFixed(2)}MB`);

      this.cleanup();

      return {
        blob,
        frameCount: this.processedFrames,
        fps: PROXY_FPS,
      };
    } catch (error) {
      log.error('Generation failed', error);
      this.cleanup();
      throw error;
    }
  }

  private initEncoderAndMuxer() {
    // Create muxer
    this.muxer = new Mp4Muxer({
      target: new Mp4Target(),
      video: {
        codec: 'avc',
        width: this.outputWidth,
        height: this.outputHeight,
      },
      fastStart: 'in-memory',
    });

    // Create encoder
    this.encoder = new VideoEncoder({
      output: (chunk, meta) => {
        this.muxer!.addVideoChunk(chunk, meta);
      },
      error: (e) => {
        log.error('Encoder error:', e);
      },
    });

    this.encoder.configure({
      codec: ENCODER_CODEC,
      width: this.outputWidth,
      height: this.outputHeight,
      bitrate: PROXY_BITRATE,
      framerate: PROXY_FPS,
      latencyMode: 'quality',
    });

    log.debug(`Encoder configured: ${this.outputWidth}x${this.outputHeight} @ ${PROXY_BITRATE / 1_000_000}Mbps (all keyframes)`);
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
        // Ensure even dimensions for H.264
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

    if (frameIndex >= 0 && frameIndex < this.totalFrames) {
      const existing = this.decodedFrames.get(frameIndex);
      if (existing) existing.close();
      this.decodedFrames.set(frameIndex, frame);
    } else {
      frame.close();
    }
  }

  private encodeFrame(frame: VideoFrame, frameIndex: number) {
    if (!this.encoder || !this.resizeCanvas || !this.resizeCtx) return;

    // Draw source frame onto resize canvas (bilinear filtering)
    this.resizeCtx.drawImage(frame, 0, 0, this.outputWidth, this.outputHeight);
    frame.close();

    // Create new VideoFrame from resized canvas
    const timestampMicros = Math.round(frameIndex * (1_000_000 / PROXY_FPS));
    const durationMicros = Math.round(1_000_000 / PROXY_FPS);

    const resizedFrame = new VideoFrame(this.resizeCanvas, {
      timestamp: timestampMicros,
      duration: durationMicros,
    });

    // Encode as keyframe (every frame is a keyframe for instant random access)
    this.encoder.encode(resizedFrame, { keyFrame: true });
    resizedFrame.close();

    this.processedFrames++;
    this.onProgress?.(Math.min(100, Math.round((this.processedFrames / this.totalFrames) * 100)));
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
    let samplesDecoded = 0;

    // Process frames inline as they're decoded (prevents DPB overflow)
    const processAccumulatedFrames = () => {
      if (this.decodedFrames.size < 4) return;

      const sortedIndices = Array.from(this.decodedFrames.keys()).sort((a, b) => a - b);
      const batch = sortedIndices.slice(0, Math.min(16, sortedIndices.length));

      for (const idx of batch) {
        const frame = this.decodedFrames.get(idx);
        if (frame) {
          this.decodedFrames.delete(idx);
          this.encodeFrame(frame, idx);
        }
      }
    };

    // Decode loop
    for (let i = firstKeyframeIdx; i < sortedSamples.length; i++) {
      const sample = sortedSamples[i];

      if (this.checkCancelled?.()) {
        this.isCancelled = true;
        break;
      }

      if (this.decoder.state === 'closed') {
        log.error('Decoder closed unexpectedly');
        break;
      }

      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts / sample.timescale) * 1_000_000,
        duration: (sample.duration / sample.timescale) * 1_000_000,
        data: sample.data,
      });

      try {
        this.decoder.decode(chunk);
        samplesDecoded++;
      } catch {
        decodeErrors++;
        if (decodeErrors > 50) {
          log.error('Too many decode errors, stopping');
          break;
        }
      }

      // Yield to let decoder output callback fire
      if (i % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }

      // Process decoded frames to free DPB
      if (this.decodedFrames.size >= 4) {
        processAccumulatedFrames();
      }
    }

    // Wait for decoder to finish
    const maxWaitTime = 120000;
    const waitStart = performance.now();
    let lastCount = this.decodedFrames.size + this.processedFrames;
    let stallCount = 0;

    while (performance.now() - waitStart < maxWaitTime) {
      if (this.decodedFrames.size >= 4) {
        processAccumulatedFrames();
      }

      if (this.processedFrames >= this.totalFrames * 0.95) break;

      const currentCount = this.decodedFrames.size + this.processedFrames;
      if (currentCount === lastCount) {
        stallCount++;
        if (stallCount > 100) break; // 5s stall
      } else {
        stallCount = 0;
        lastCount = currentCount;
      }

      await new Promise(resolve => setTimeout(resolve, 50));
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

    // Process remaining decoded frames
    while (this.decodedFrames.size > 0) {
      const sortedIndices = Array.from(this.decodedFrames.keys()).sort((a, b) => a - b);
      for (const idx of sortedIndices) {
        const frame = this.decodedFrames.get(idx);
        if (frame) {
          this.decodedFrames.delete(idx);
          this.encodeFrame(frame, idx);
        }
      }
    }

    const totalTime = performance.now() - startTime;
    const fps = this.processedFrames / (totalTime / 1000);
    log.info(`Encode complete: ${this.processedFrames}/${this.totalFrames} frames in ${(totalTime / 1000).toFixed(1)}s (${fps.toFixed(1)} fps)`);
  }

  private cleanup() {
    try { this.decoder?.close(); } catch { /* ignore */ }
    try { this.encoder?.close(); } catch { /* ignore */ }
    for (const frame of this.decodedFrames.values()) frame.close();
    this.decodedFrames.clear();
    this.resizeCanvas = null;
    this.resizeCtx = null;
    this.decoder = null;
    this.encoder = null;
    this.muxer = null;
  }
}

// Singleton instance
let generatorInstance: ProxyGeneratorMP4 | null = null;

export function getProxyGenerator(): ProxyGeneratorMP4 {
  if (!generatorInstance) {
    generatorInstance = new ProxyGeneratorMP4();
  }
  return generatorInstance;
}

export { ProxyGeneratorMP4, PROXY_FPS, PROXY_MAX_WIDTH };
