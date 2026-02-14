// Video encoder wrapper using WebCodecs and mp4/webm muxers

import { Logger } from '../../services/logger';
import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4Target } from 'mp4-muxer';

const log = Logger.create('VideoEncoder');
import { Muxer as WebmMuxer, ArrayBufferTarget as WebmTarget } from 'webm-muxer';
import { AudioEncoderWrapper, type AudioCodec, type EncodedAudioResult } from '../audio';
import type { ExportSettings, VideoCodec, ContainerFormat } from './types';
import { getCodecString, getMp4MuxerCodec, getWebmMuxerCodec, isCodecSupportedInContainer, getFallbackCodec } from './codecHelpers';

type MuxerType = Mp4Muxer<Mp4Target> | WebmMuxer<WebmTarget>;

export class VideoEncoderWrapper {
  private encoder: VideoEncoder | null = null;
  private muxer: MuxerType | null = null;
  private settings: ExportSettings;
  private encodedFrameCount = 0;
  private isClosed = false;
  private hasAudio = false;
  private audioCodec: AudioCodec = 'aac';
  private containerFormat: ContainerFormat = 'mp4';
  private effectiveVideoCodec: VideoCodec = 'h264';

  constructor(settings: ExportSettings) {
    this.settings = settings;
    this.hasAudio = settings.includeAudio ?? false;
    this.containerFormat = settings.container ?? 'mp4';
  }

  async init(): Promise<boolean> {
    if (!('VideoEncoder' in window)) {
      log.error('WebCodecs not supported');
      return false;
    }

    // Determine audio codec based on container
    await this.initializeAudioCodec();

    // Determine effective video codec based on container compatibility
    this.effectiveVideoCodec = this.settings.codec;
    if (!isCodecSupportedInContainer(this.settings.codec, this.containerFormat)) {
      log.warn(`${this.settings.codec} not supported in ${this.containerFormat}, using fallback`);
      this.effectiveVideoCodec = getFallbackCodec(this.containerFormat);
    }

    // Check codec support
    const codecString = getCodecString(this.effectiveVideoCodec);
    try {
      const support = await VideoEncoder.isConfigSupported({
        codec: codecString,
        width: this.settings.width,
        height: this.settings.height,
        bitrate: this.settings.bitrate,
        framerate: this.settings.fps,
      });

      if (!support.supported) {
        log.error(`Codec not supported: ${codecString}`);
        return false;
      }
    } catch (e) {
      log.error('Codec support check failed:', e);
      return false;
    }

    // Create muxer
    this.createMuxer();

    // Create encoder
    this.encoder = new VideoEncoder({
      output: (chunk, meta) => {
        if (this.muxer) {
          this.muxer.addVideoChunk(chunk, meta);
        }
        this.encodedFrameCount++;
      },
      error: (e) => {
        log.error('Encode error:', e);
      },
    });

    await this.encoder.configure({
      codec: codecString,
      width: this.settings.width,
      height: this.settings.height,
      bitrate: this.settings.bitrate,
      framerate: this.settings.fps,
      latencyMode: 'quality',
      bitrateMode: 'variable',
    });

    log.info(`Initialized: ${this.settings.width}x${this.settings.height} @ ${this.settings.fps}fps (${this.effectiveVideoCodec.toUpperCase()})`);
    return true;
  }

  private async initializeAudioCodec(): Promise<void> {
    if (!this.hasAudio) return;

    if (this.containerFormat === 'webm') {
      const opusSupported = await AudioEncoderWrapper.isOpusSupported();
      if (opusSupported) {
        this.audioCodec = 'opus';
        log.info('Using Opus audio for WebM');
      } else {
        log.warn('Opus not supported, disabling audio for WebM');
        this.hasAudio = false;
      }
    } else {
      const aacSupported = await AudioEncoderWrapper.isAACSupported();
      if (aacSupported) {
        this.audioCodec = 'aac';
        log.info('Using AAC audio for MP4');
      } else {
        const opusSupported = await AudioEncoderWrapper.isOpusSupported();
        if (opusSupported) {
          this.audioCodec = 'opus';
          log.info('AAC not supported, using Opus audio for MP4 (fallback)');
        } else {
          log.warn('No audio codec supported, disabling audio');
          this.hasAudio = false;
        }
      }
    }
  }

  private createMuxer(): void {
    const webmVideoCodec = getWebmMuxerCodec(this.effectiveVideoCodec);
    const mp4VideoCodec = getMp4MuxerCodec(this.effectiveVideoCodec);
    const sampleRate = this.settings.audioSampleRate ?? 48000;

    if (this.containerFormat === 'webm') {
      this.muxer = this.hasAudio
        ? new WebmMuxer({
            target: new WebmTarget(),
            video: { codec: webmVideoCodec, width: this.settings.width, height: this.settings.height },
            audio: { codec: 'A_OPUS', sampleRate, numberOfChannels: 2 },
          })
        : new WebmMuxer({
            target: new WebmTarget(),
            video: { codec: webmVideoCodec, width: this.settings.width, height: this.settings.height },
          });
      log.info(`Using WebM/${this.effectiveVideoCodec.toUpperCase()} with ${this.hasAudio ? 'Opus' : 'no'} audio`);
    } else {
      this.muxer = this.hasAudio
        ? new Mp4Muxer({
            target: new Mp4Target(),
            video: { codec: mp4VideoCodec, width: this.settings.width, height: this.settings.height },
            audio: { codec: this.audioCodec, sampleRate, numberOfChannels: 2 },
            fastStart: 'in-memory',
          })
        : new Mp4Muxer({
            target: new Mp4Target(),
            video: { codec: mp4VideoCodec, width: this.settings.width, height: this.settings.height },
            fastStart: 'in-memory',
          });
      log.info(`Using MP4/${this.effectiveVideoCodec.toUpperCase()} with ${this.hasAudio ? this.audioCodec.toUpperCase() : 'no'} audio`);
    }
  }

  getContainerFormat(): ContainerFormat {
    return this.containerFormat;
  }

  getAudioCodec(): AudioCodec {
    return this.audioCodec;
  }

  async encodeFrame(pixels: Uint8ClampedArray, frameIndex: number, keyframeInterval?: number): Promise<void> {
    if (!this.encoder || this.isClosed) {
      throw new Error('Encoder not initialized or already closed');
    }

    const timestampMicros = Math.round(frameIndex * (1_000_000 / this.settings.fps));
    const durationMicros = Math.round(1_000_000 / this.settings.fps);

    const frame = new VideoFrame(pixels.buffer, {
      format: 'RGBA',
      codedWidth: this.settings.width,
      codedHeight: this.settings.height,
      timestamp: timestampMicros,
      duration: durationMicros,
    });

    // FPS-based keyframe interval (default: 1 keyframe per second)
    const interval = keyframeInterval ?? this.settings.fps;
    const keyFrame = frameIndex % interval === 0;

    this.encoder.encode(frame, { keyFrame });
    frame.close();

    // Yield to event loop periodically - use queueMicrotask for lower latency
    if (frameIndex % 30 === 0) {
      await new Promise<void>(resolve => queueMicrotask(() => resolve()));
    }
  }

  /**
   * Encode a VideoFrame directly (zero-copy path from OffscreenCanvas).
   * The caller is responsible for closing the frame after this returns.
   */
  async encodeVideoFrame(frame: VideoFrame, frameIndex: number, keyframeInterval?: number): Promise<void> {
    if (!this.encoder || this.isClosed) {
      throw new Error('Encoder not initialized or already closed');
    }

    // FPS-based keyframe interval (default: 1 keyframe per second)
    const interval = keyframeInterval ?? this.settings.fps;
    const keyFrame = frameIndex % interval === 0;
    this.encoder.encode(frame, { keyFrame });

    // Yield to event loop periodically
    if (frameIndex % 30 === 0) {
      await new Promise<void>(resolve => queueMicrotask(() => resolve()));
    }
  }

  addAudioChunks(audioResult: EncodedAudioResult): void {
    if (!this.muxer || !this.hasAudio) {
      log.warn('Cannot add audio: muxer not ready or audio not enabled');
      return;
    }

    log.debug(`Adding ${audioResult.chunks.length} audio chunks`);

    for (let i = 0; i < audioResult.chunks.length; i++) {
      const chunk = audioResult.chunks[i];
      const meta = audioResult.metadata[i];
      this.muxer.addAudioChunk(chunk, meta);
    }

    log.debug('Audio chunks added successfully');
  }

  async finish(): Promise<Blob> {
    if (!this.encoder || !this.muxer) {
      throw new Error('Encoder not initialized');
    }

    this.isClosed = true;
    await this.encoder.flush();
    this.encoder.close();
    this.muxer.finalize();

    const { buffer } = this.muxer.target;
    const mimeType = this.containerFormat === 'webm' ? 'video/webm' : 'video/mp4';

    log.info(`Finished: ${this.encodedFrameCount} frames, ${(buffer.byteLength / 1024 / 1024).toFixed(2)}MB (${this.containerFormat.toUpperCase()})`);
    return new Blob([buffer], { type: mimeType });
  }

  cancel(): void {
    if (this.encoder && !this.isClosed) {
      this.isClosed = true;
      try {
        this.encoder.close();
      } catch {}
    }
  }
}
