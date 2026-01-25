/**
 * AudioEncoder - Encode AudioBuffer to AAC or Opus using WebCodecs
 *
 * Features:
 * - AAC-LC encoding via WebCodecs AudioEncoder (MP4 container)
 * - Opus encoding as fallback for Linux (WebM container)
 * - Chunked encoding for memory efficiency
 * - Progress callbacks
 * - Automatic codec selection based on browser support
 */

import { Logger } from '../../services/logger';

const log = Logger.create('AudioEncoder');

export type AudioCodec = 'aac' | 'opus';

export interface AudioEncoderSettings {
  sampleRate: number;      // 44100 or 48000
  numberOfChannels: number; // 1 or 2
  bitrate: number;         // 128000 - 320000
  codec?: AudioCodec;      // 'aac' or 'opus' (auto-detected if not specified)
}

export interface EncodedAudioResult {
  chunks: EncodedAudioChunk[];
  metadata: EncodedAudioChunkMetadata[];
  duration: number;
  settings: AudioEncoderSettings;
  codec: AudioCodec;
  codecString: string; // e.g. 'mp4a.40.2' or 'opus'
}

export type AudioEncoderProgressCallback = (progress: {
  encodedSamples: number;
  totalSamples: number;
  percent: number;
}) => void;

export class AudioEncoderWrapper {
  private encoder: AudioEncoder | null = null;
  private settings: AudioEncoderSettings;
  private chunks: EncodedAudioChunk[] = [];
  private metadata: EncodedAudioChunkMetadata[] = [];
  private encodedSamples = 0;
  private isClosed = false;
  private onProgress: AudioEncoderProgressCallback | null = null;
  private totalSamples = 0;
  private activeCodec: AudioCodec = 'aac';
  private activeCodecString: string = 'mp4a.40.2';

  constructor(settings: AudioEncoderSettings) {
    this.settings = {
      sampleRate: settings.sampleRate || 48000,
      numberOfChannels: settings.numberOfChannels || 2,
      bitrate: settings.bitrate || 256000,
      codec: settings.codec, // Can be undefined for auto-detect
    };
  }

  /**
   * Check if any audio encoding is supported (AAC or Opus)
   */
  static async isSupported(): Promise<boolean> {
    const result = await AudioEncoderWrapper.detectSupportedCodec();
    return result !== null;
  }

  /**
   * Check if AAC encoding is supported
   */
  static async isAACSupported(): Promise<boolean> {
    if (!('AudioEncoder' in window)) {
      log.debug('AudioEncoder not in window');
      return false;
    }

    try {
      const config = {
        codec: 'mp4a.40.2', // AAC-LC
        sampleRate: 48000,
        numberOfChannels: 2,
        bitrate: 256000,
      };
      log.debug('Checking AAC support with config:', config);
      const support = await AudioEncoder.isConfigSupported(config);
      log.debug('AAC support result:', support);
      return support.supported === true;
    } catch (e) {
      log.error('AAC support check error:', e);
      return false;
    }
  }

  /**
   * Check if Opus encoding is supported
   */
  static async isOpusSupported(): Promise<boolean> {
    if (!('AudioEncoder' in window)) {
      return false;
    }

    try {
      const support = await AudioEncoder.isConfigSupported({
        codec: 'opus',
        sampleRate: 48000,
        numberOfChannels: 2,
        bitrate: 128000,
      });
      return support.supported === true;
    } catch {
      return false;
    }
  }

  /**
   * Detect which codec is supported (prefers AAC, falls back to Opus)
   */
  static async detectSupportedCodec(): Promise<{ codec: AudioCodec; codecString: string } | null> {
    if (await AudioEncoderWrapper.isAACSupported()) {
      return { codec: 'aac', codecString: 'mp4a.40.2' };
    }
    if (await AudioEncoderWrapper.isOpusSupported()) {
      return { codec: 'opus', codecString: 'opus' };
    }
    return null;
  }

  /**
   * Get the active codec being used
   */
  getActiveCodec(): { codec: AudioCodec; codecString: string } {
    return { codec: this.activeCodec, codecString: this.activeCodecString };
  }

  /**
   * Initialize the encoder
   */
  async init(): Promise<boolean> {
    if (!('AudioEncoder' in window)) {
      console.error('[AudioEncoder] WebCodecs AudioEncoder not available');
      return false;
    }

    // Determine which codec to use
    let codecToUse: { codec: AudioCodec; codecString: string } | null = null;

    if (this.settings.codec === 'aac') {
      if (await AudioEncoderWrapper.isAACSupported()) {
        codecToUse = { codec: 'aac', codecString: 'mp4a.40.2' };
      }
    } else if (this.settings.codec === 'opus') {
      if (await AudioEncoderWrapper.isOpusSupported()) {
        codecToUse = { codec: 'opus', codecString: 'opus' };
      }
    } else {
      // Auto-detect: prefer AAC, fall back to Opus
      codecToUse = await AudioEncoderWrapper.detectSupportedCodec();
    }

    if (!codecToUse) {
      console.error('[AudioEncoder] No supported audio codec found (tried AAC and Opus)');
      return false;
    }

    this.activeCodec = codecToUse.codec;
    this.activeCodecString = codecToUse.codecString;

    // Adjust bitrate for Opus (it uses lower bitrates than AAC for same quality)
    const bitrate = this.activeCodec === 'opus'
      ? Math.min(this.settings.bitrate, 192000) // Opus max recommended is 192kbps
      : this.settings.bitrate;

    const config: AudioEncoderConfig = {
      codec: this.activeCodecString,
      sampleRate: this.settings.sampleRate,
      numberOfChannels: this.settings.numberOfChannels,
      bitrate: bitrate,
    };

    try {
      const support = await AudioEncoder.isConfigSupported(config);
      if (!support.supported) {
        console.error(`[AudioEncoder] ${this.activeCodec.toUpperCase()} configuration not supported`);
        return false;
      }
    } catch (e) {
      console.error('[AudioEncoder] Config support check failed:', e);
      return false;
    }

    this.encoder = new AudioEncoder({
      output: (chunk, meta) => this.handleChunk(chunk, meta),
      error: (e) => this.handleError(e),
    });

    try {
      this.encoder.configure(config);
      console.log(`[AudioEncoder] Initialized with ${this.activeCodec.toUpperCase()}: ${this.settings.sampleRate}Hz, ${this.settings.numberOfChannels}ch, ${bitrate / 1000}kbps`);
      return true;
    } catch (e) {
      console.error('[AudioEncoder] Configure failed:', e);
      return false;
    }
  }

  /**
   * Handle encoded chunk output
   */
  private handleChunk(chunk: EncodedAudioChunk, meta?: EncodedAudioChunkMetadata): void {
    this.chunks.push(chunk);
    if (meta) {
      this.metadata.push(meta);
    }
  }

  /**
   * Handle encoder errors
   */
  private handleError(error: DOMException): void {
    console.error('[AudioEncoder] Encode error:', error);
  }

  /**
   * Encode an AudioBuffer to AAC chunks
   * @param buffer - Mixed AudioBuffer to encode
   * @param onProgress - Optional progress callback
   */
  async encode(
    buffer: AudioBuffer,
    onProgress?: AudioEncoderProgressCallback
  ): Promise<void> {
    if (!this.encoder || this.isClosed) {
      throw new Error('[AudioEncoder] Encoder not initialized or already closed');
    }

    this.onProgress = onProgress || null;
    this.totalSamples = buffer.length;
    this.encodedSamples = 0;
    this.chunks = [];
    this.metadata = [];

    // AAC frame size is typically 1024 samples
    const frameSize = 1024;
    const totalFrames = Math.ceil(buffer.length / frameSize);

    console.log(`[AudioEncoder] Encoding ${buffer.duration.toFixed(2)}s audio (${totalFrames} frames)`);

    for (let frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
      const startSample = frameIndex * frameSize;
      const endSample = Math.min(startSample + frameSize, buffer.length);
      const numSamples = endSample - startSample;

      // Extract interleaved samples for this frame
      const frameData = this.extractFrameData(buffer, startSample, numSamples);

      // Create AudioData
      const timestamp = Math.round((startSample / buffer.sampleRate) * 1_000_000); // microseconds

      const audioData = new AudioData({
        format: 'f32-planar',
        sampleRate: buffer.sampleRate,
        numberOfFrames: numSamples,
        numberOfChannels: buffer.numberOfChannels,
        timestamp,
        data: frameData.buffer as ArrayBuffer,
      });

      // Encode
      this.encoder.encode(audioData);
      audioData.close();

      // Update progress
      this.encodedSamples = endSample;
      if (this.onProgress) {
        this.onProgress({
          encodedSamples: this.encodedSamples,
          totalSamples: this.totalSamples,
          percent: Math.round((this.encodedSamples / this.totalSamples) * 100),
        });
      }

      // Yield to allow UI updates every 100 frames
      if (frameIndex % 100 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    console.log(`[AudioEncoder] Encoding complete, ${this.chunks.length} chunks`);
  }

  /**
   * Extract frame data from AudioBuffer as Float32Array
   * Format: planar (all samples of channel 0, then all samples of channel 1, etc.)
   */
  private extractFrameData(
    buffer: AudioBuffer,
    startSample: number,
    numSamples: number
  ): Float32Array {
    const channels = buffer.numberOfChannels;
    const frameData = new Float32Array(numSamples * channels);

    for (let ch = 0; ch < channels; ch++) {
      const channelData = buffer.getChannelData(ch);
      const offset = ch * numSamples;

      for (let i = 0; i < numSamples; i++) {
        const sampleIndex = startSample + i;
        frameData[offset + i] = sampleIndex < channelData.length
          ? channelData[sampleIndex]
          : 0; // Pad with silence if needed
      }
    }

    return frameData;
  }

  /**
   * Flush remaining data and finalize
   */
  async finalize(): Promise<EncodedAudioResult> {
    if (!this.encoder) {
      throw new Error('[AudioEncoder] Encoder not initialized');
    }

    if (!this.isClosed) {
      await this.encoder.flush();
      this.encoder.close();
      this.isClosed = true;
    }

    const duration = this.totalSamples / this.settings.sampleRate;

    console.log(`[AudioEncoder] Finalized: ${this.chunks.length} chunks, ${duration.toFixed(2)}s (${this.activeCodec.toUpperCase()})`);

    return {
      chunks: this.chunks,
      metadata: this.metadata,
      duration,
      settings: this.settings,
      codec: this.activeCodec,
      codecString: this.activeCodecString,
    };
  }

  /**
   * Get encoded chunks (for muxing)
   */
  getChunks(): EncodedAudioChunk[] {
    return this.chunks;
  }

  /**
   * Get metadata for chunks
   */
  getMetadata(): EncodedAudioChunkMetadata[] {
    return this.metadata;
  }

  /**
   * Reset encoder for reuse
   */
  async reset(): Promise<boolean> {
    if (this.encoder && !this.isClosed) {
      this.encoder.close();
    }

    this.encoder = null;
    this.chunks = [];
    this.metadata = [];
    this.encodedSamples = 0;
    this.totalSamples = 0;
    this.isClosed = false;

    return this.init();
  }

  /**
   * Check if encoder is ready
   */
  isReady(): boolean {
    return this.encoder !== null && !this.isClosed;
  }

  /**
   * Get current settings
   */
  getSettings(): AudioEncoderSettings {
    return { ...this.settings };
  }
}

/**
 * Helper to get recommended audio bitrate
 */
export function getRecommendedAudioBitrate(quality: 'low' | 'medium' | 'high' | 'lossless'): number {
  switch (quality) {
    case 'low': return 128000;
    case 'medium': return 192000;
    case 'high': return 256000;
    case 'lossless': return 320000;
    default: return 256000;
  }
}

/**
 * Audio codec info for UI
 */
export const AUDIO_CODEC_INFO = {
  aac: {
    name: 'AAC-LC',
    codec: 'mp4a.40.2',
    container: 'mp4',
    description: 'Advanced Audio Coding - universal compatibility',
    bitrateRange: { min: 64000, max: 320000 },
    sampleRates: [44100, 48000],
  },
  opus: {
    name: 'Opus',
    codec: 'opus',
    container: 'webm',
    description: 'Open source codec - great quality at low bitrates',
    bitrateRange: { min: 32000, max: 192000 },
    sampleRates: [48000],
  },
} as const;
