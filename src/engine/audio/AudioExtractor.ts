/**
 * AudioExtractor - Decode audio from video/audio files into AudioBuffer
 *
 * Features:
 * - Decode any audio/video file to AudioBuffer
 * - Trim to specific time range (inPoint/outPoint)
 * - Cache decoded buffers for efficiency
 * - Handle files without audio gracefully
 */

import { Logger } from '../../services/logger';

const log = Logger.create('AudioExtractor');

export interface ExtractedAudio {
  buffer: AudioBuffer;
  duration: number;
  sampleRate: number;
  numberOfChannels: number;
}

export class AudioExtractor {
  private audioContext: AudioContext | null = null;
  private cache: Map<string, AudioBuffer> = new Map();
  private maxCacheSize = 5; // Max number of cached buffers

  /**
   * Initialize the AudioContext (lazy initialization)
   */
  private getContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
    }
    return this.audioContext;
  }

  /**
   * Extract audio from a media file
   * @param file - Video or audio file
   * @param cacheKey - Optional key for caching (e.g., mediaFileId)
   * @returns Decoded AudioBuffer
   */
  async extractAudio(file: File, cacheKey?: string): Promise<AudioBuffer> {
    // Check cache first
    if (cacheKey && this.cache.has(cacheKey)) {
      log.debug(`Cache hit for ${cacheKey}`);
      return this.cache.get(cacheKey)!;
    }

    log.info(`Extracting audio from ${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`);

    try {
      // Read file as ArrayBuffer
      const arrayBuffer = await file.arrayBuffer();

      // Decode audio data
      const context = this.getContext();
      const audioBuffer = await context.decodeAudioData(arrayBuffer);

      log.debug(`Decoded: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.numberOfChannels}ch, ${audioBuffer.sampleRate}Hz`);

      // Cache the result
      if (cacheKey) {
        this.addToCache(cacheKey, audioBuffer);
      }

      return audioBuffer;
    } catch (error) {
      // Check if this might be a video without audio
      if (error instanceof DOMException && error.name === 'EncodingError') {
        log.warn(`No audio track in ${file.name}, creating silent buffer`);
        return this.createSilentBuffer(1, 48000); // 1 second silent buffer
      }
      throw new AudioExtractionError(
        `Failed to decode audio from ${file.name}: ${error}`,
        file.name
      );
    }
  }

  /**
   * Extract audio from an HTMLVideoElement or HTMLAudioElement
   * Useful when we already have a loaded media element
   */
  async extractFromElement(
    element: HTMLVideoElement | HTMLAudioElement,
    cacheKey?: string
  ): Promise<AudioBuffer> {
    // Check cache first
    if (cacheKey && this.cache.has(cacheKey)) {
      log.debug(`Cache hit for ${cacheKey}`);
      return this.cache.get(cacheKey)!;
    }

    // Get the source URL
    const src = element.src || element.currentSrc;
    if (!src) {
      throw new AudioExtractionError('No source URL for media element', 'unknown');
    }

    try {
      // Fetch the source
      const response = await fetch(src);
      const arrayBuffer = await response.arrayBuffer();

      // Decode
      const context = this.getContext();
      const audioBuffer = await context.decodeAudioData(arrayBuffer);

      // Cache
      if (cacheKey) {
        this.addToCache(cacheKey, audioBuffer);
      }

      return audioBuffer;
    } catch (error) {
      throw new AudioExtractionError(
        `Failed to extract audio from element: ${error}`,
        src
      );
    }
  }

  /**
   * Trim an AudioBuffer to a specific time range
   * @param buffer - Source AudioBuffer
   * @param startTime - Start time in seconds
   * @param endTime - End time in seconds
   * @returns New trimmed AudioBuffer
   */
  trimBuffer(buffer: AudioBuffer, startTime: number, endTime: number): AudioBuffer {
    const sampleRate = buffer.sampleRate;
    const startSample = Math.floor(startTime * sampleRate);
    const endSample = Math.min(Math.ceil(endTime * sampleRate), buffer.length);
    const newLength = endSample - startSample;

    if (newLength <= 0) {
      log.warn('Trim resulted in empty buffer');
      return this.createSilentBuffer(0.001, sampleRate);
    }

    const context = this.getContext();
    const trimmedBuffer = context.createBuffer(
      buffer.numberOfChannels,
      newLength,
      sampleRate
    );

    // Copy each channel
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const sourceData = buffer.getChannelData(channel);
      const destData = trimmedBuffer.getChannelData(channel);

      for (let i = 0; i < newLength; i++) {
        destData[i] = sourceData[startSample + i];
      }
    }

    return trimmedBuffer;
  }

  /**
   * Resample an AudioBuffer to a target sample rate
   * @param buffer - Source AudioBuffer
   * @param targetSampleRate - Target sample rate (e.g., 48000)
   * @returns Resampled AudioBuffer
   */
  async resampleBuffer(buffer: AudioBuffer, targetSampleRate: number): Promise<AudioBuffer> {
    if (buffer.sampleRate === targetSampleRate) {
      return buffer;
    }

    const offlineContext = new OfflineAudioContext(
      buffer.numberOfChannels,
      Math.ceil(buffer.duration * targetSampleRate),
      targetSampleRate
    );

    const source = offlineContext.createBufferSource();
    source.buffer = buffer;
    source.connect(offlineContext.destination);
    source.start(0);

    return await offlineContext.startRendering();
  }

  /**
   * Create a silent AudioBuffer
   * @param duration - Duration in seconds
   * @param sampleRate - Sample rate (default 48000)
   * @returns Silent AudioBuffer
   */
  createSilentBuffer(duration: number, sampleRate: number = 48000): AudioBuffer {
    const context = this.getContext();
    const length = Math.max(1, Math.ceil(duration * sampleRate));
    return context.createBuffer(2, length, sampleRate); // Stereo silent buffer
  }

  /**
   * Convert mono buffer to stereo
   */
  convertToStereo(buffer: AudioBuffer): AudioBuffer {
    if (buffer.numberOfChannels >= 2) {
      return buffer;
    }

    const context = this.getContext();
    const stereoBuffer = context.createBuffer(
      2,
      buffer.length,
      buffer.sampleRate
    );

    const monoData = buffer.getChannelData(0);
    const leftData = stereoBuffer.getChannelData(0);
    const rightData = stereoBuffer.getChannelData(1);

    // Copy mono to both channels
    for (let i = 0; i < buffer.length; i++) {
      leftData[i] = monoData[i];
      rightData[i] = monoData[i];
    }

    return stereoBuffer;
  }

  /**
   * Get buffer info without full extraction (for UI display)
   */
  getBufferInfo(buffer: AudioBuffer): ExtractedAudio {
    return {
      buffer,
      duration: buffer.duration,
      sampleRate: buffer.sampleRate,
      numberOfChannels: buffer.numberOfChannels,
    };
  }

  // ============ Cache Management ============

  /**
   * Add buffer to cache with LRU eviction
   */
  private addToCache(key: string, buffer: AudioBuffer): void {
    // Evict oldest if at capacity
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) {
        log.debug(`Evicting cached buffer: ${firstKey}`);
        this.cache.delete(firstKey);
      }
    }

    this.cache.set(key, buffer);
    log.debug(`Cached buffer: ${key} (${this.cache.size}/${this.maxCacheSize})`);
  }

  /**
   * Get cached buffer
   */
  getCached(key: string): AudioBuffer | null {
    return this.cache.get(key) || null;
  }

  /**
   * Check if buffer is cached
   */
  hasCached(key: string): boolean {
    return this.cache.has(key);
  }

  /**
   * Clear all cached buffers
   */
  clearCache(): void {
    const count = this.cache.size;
    this.cache.clear();
    log.debug(`Cleared ${count} cached buffers`);
  }

  /**
   * Set maximum cache size
   */
  setMaxCacheSize(size: number): void {
    this.maxCacheSize = Math.max(1, size);

    // Evict if over new limit
    while (this.cache.size > this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) {
        this.cache.delete(firstKey);
      }
    }
  }

  /**
   * Get current cache stats
   */
  getCacheStats(): { size: number; maxSize: number; keys: string[] } {
    return {
      size: this.cache.size,
      maxSize: this.maxCacheSize,
      keys: Array.from(this.cache.keys()),
    };
  }

  // ============ Cleanup ============

  /**
   * Destroy the extractor and release resources
   */
  destroy(): void {
    this.clearCache();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    log.info('Destroyed');
  }
}

/**
 * Custom error for audio extraction failures
 */
export class AudioExtractionError extends Error {
  readonly fileName: string;
  readonly recoverable: boolean;

  constructor(
    message: string,
    fileName: string,
    recoverable: boolean = false
  ) {
    super(message);
    this.name = 'AudioExtractionError';
    this.fileName = fileName;
    this.recoverable = recoverable;
  }
}

// Singleton instance for convenience
export const audioExtractor = new AudioExtractor();
