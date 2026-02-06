// Proxy audio cache - manages audio proxy loading and varispeed scrub audio

import { Logger } from './logger';
import { projectFileService } from './projectFileService';
import { fileSystemService } from './fileSystemService';
import { useMediaStore } from '../stores/mediaStore';

const log = Logger.create('ProxyFrameCache');

class ProxyFrameCache {
  // Audio proxy cache
  private audioCache: Map<string, HTMLAudioElement> = new Map();
  private audioLoadingPromises: Map<string, Promise<HTMLAudioElement | null>> = new Map();

  // Audio buffer cache for instant scrubbing (Web Audio API)
  private audioBufferCache: Map<string, AudioBuffer> = new Map();
  private audioBufferFailed: Set<string> = new Set();
  private audioBufferLoading = new Set<string>();
  private audioContext: AudioContext | null = null;
  private scrubGain: GainNode | null = null;

  // Varispeed scrubbing state
  private scrubSource: AudioBufferSourceNode | null = null;
  private scrubSourceGain: GainNode | null = null;
  private scrubStartTime = 0;
  private scrubStartPosition = 0;
  private scrubCurrentMediaId: string | null = null;
  private scrubLastPosition = 0;
  private scrubLastTime = 0;
  private scrubIsActive = false;

  /**
   * Get cached audio proxy element, or load it if not cached
   */
  async getAudioProxy(mediaFileId: string): Promise<HTMLAudioElement | null> {
    const cached = this.audioCache.get(mediaFileId);
    if (cached) return cached;

    const existingPromise = this.audioLoadingPromises.get(mediaFileId);
    if (existingPromise) return existingPromise;

    const loadPromise = this.loadAudioProxy(mediaFileId);
    this.audioLoadingPromises.set(mediaFileId, loadPromise);

    try {
      const audio = await loadPromise;
      if (audio) {
        this.audioCache.set(mediaFileId, audio);
      }
      return audio;
    } finally {
      this.audioLoadingPromises.delete(mediaFileId);
    }
  }

  /**
   * Get cached audio proxy synchronously (returns null if not yet loaded)
   */
  getCachedAudioProxy(mediaFileId: string): HTMLAudioElement | null {
    return this.audioCache.get(mediaFileId) || null;
  }

  /**
   * Preload audio proxy for a media file
   */
  async preloadAudioProxy(mediaFileId: string): Promise<void> {
    await this.getAudioProxy(mediaFileId);
  }

  /**
   * Load audio proxy from project folder
   */
  private async loadAudioProxy(mediaFileId: string): Promise<HTMLAudioElement | null> {
    try {
      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      const audioFile = await projectFileService.getProxyAudio(storageKey);
      if (!audioFile) return null;

      const audio = new Audio();
      audio.src = URL.createObjectURL(audioFile);
      audio.preload = 'auto';

      await new Promise<void>((resolve, reject) => {
        const onCanPlay = () => {
          audio.removeEventListener('canplaythrough', onCanPlay);
          audio.removeEventListener('error', onError);
          resolve();
        };
        const onError = () => {
          audio.removeEventListener('canplaythrough', onCanPlay);
          audio.removeEventListener('error', onError);
          reject(new Error('Failed to load audio proxy'));
        };
        audio.addEventListener('canplaythrough', onCanPlay);
        audio.addEventListener('error', onError);
        audio.load();
      });

      log.info(`Audio proxy loaded for ${mediaFileId}`);
      return audio;
    } catch (e) {
      log.warn(`Failed to load audio proxy for ${mediaFileId}`, e);
      return null;
    }
  }

  /**
   * Get or create AudioContext for scrubbing
   */
  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
      this.scrubGain = this.audioContext.createGain();
      this.scrubGain.connect(this.audioContext.destination);
      this.scrubGain.gain.value = 0.85;
    }
    return this.audioContext;
  }

  /**
   * Get AudioBuffer for a media file (decode on first request)
   * Works with BOTH proxy audio AND original video files
   */
  async getAudioBuffer(mediaFileId: string): Promise<AudioBuffer | null> {
    const cached = this.audioBufferCache.get(mediaFileId);
    if (cached) return cached;

    if (this.audioBufferFailed.has(mediaFileId)) return null;

    if (this.audioBufferLoading.has(mediaFileId)) return null;
    this.audioBufferLoading.add(mediaFileId);

    try {
      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      let arrayBuffer: ArrayBuffer | null = null;

      // Try 1: Proxy audio file (fastest, smallest)
      const audioFile = await projectFileService.getProxyAudio(storageKey);
      if (audioFile) {
        log.debug(`Loading from proxy audio: ${mediaFileId}`);
        arrayBuffer = await audioFile.arrayBuffer();
      }

      // Try 2: Original video file URL (extract audio from video)
      if (!arrayBuffer && mediaFile?.url) {
        log.debug(`Loading from video URL: ${mediaFileId}`);
        try {
          const response = await fetch(mediaFile.url);
          arrayBuffer = await response.arrayBuffer();
        } catch (e) {
          log.warn('Failed to fetch video URL', e);
        }
      }

      // Try 3: File handle (if available)
      if (!arrayBuffer) {
        const fileHandle = fileSystemService.getFileHandle(mediaFileId);
        if (fileHandle) {
          log.debug(`Loading from file handle: ${mediaFileId}`);
          try {
            const file = await fileHandle.getFile();
            arrayBuffer = await file.arrayBuffer();
          } catch (e) {
            log.warn('Failed to read file handle', e);
          }
        }
      }

      if (!arrayBuffer) {
        log.warn(`No audio source found for ${mediaFileId}`);
        this.audioBufferLoading.delete(mediaFileId);
        return null;
      }

      const audioContext = this.getAudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));

      this.audioBufferCache.set(mediaFileId, audioBuffer);
      this.audioBufferLoading.delete(mediaFileId);
      log.debug(`Decoded ${mediaFileId}: ${audioBuffer.duration.toFixed(1)}s, ${audioBuffer.numberOfChannels}ch`);

      return audioBuffer;
    } catch (e) {
      this.audioBufferFailed.add(mediaFileId);
      this.audioBufferLoading.delete(mediaFileId);
      log.debug(`No audio in ${mediaFileId} (or decode failed)`);
      return null;
    }
  }

  /**
   * VARISPEED SCRUBBING - Call this continuously while scrubbing
   * Audio plays continuously and follows the scrub position/speed
   */
  playScrubAudio(mediaFileId: string, targetTime: number, _duration: number = 0.15): void {
    const buffer = this.audioBufferCache.get(mediaFileId);
    if (!buffer) {
      log.debug(`No AudioBuffer for ${mediaFileId} - loading...`);
      this.getAudioBuffer(mediaFileId);
      return;
    }

    if (!this.scrubIsActive) {
      log.debug(`VARISPEED starting at ${targetTime.toFixed(2)}s`);
    }

    const ctx = this.getAudioContext();
    if (ctx.state === 'suspended') {
      ctx.resume();
    }

    const now = performance.now();
    const clampedTarget = Math.max(0, Math.min(targetTime, buffer.duration - 0.1));

    const timeDelta = (now - this.scrubLastTime) / 1000;
    const posDelta = clampedTarget - this.scrubLastPosition;
    this.scrubLastPosition = clampedTarget;
    this.scrubLastTime = now;

    const needNewSource =
      !this.scrubIsActive ||
      this.scrubCurrentMediaId !== mediaFileId ||
      !this.scrubSource;

    if (needNewSource) {
      this.stopScrubAudio();

      this.scrubSourceGain = ctx.createGain();
      this.scrubSourceGain.connect(this.scrubGain!);
      this.scrubSourceGain.gain.value = 0.9;

      this.scrubSource = ctx.createBufferSource();
      this.scrubSource.buffer = buffer;
      this.scrubSource.connect(this.scrubSourceGain);
      this.scrubSource.playbackRate.value = 1.0;

      this.scrubSource.start(0, clampedTarget);
      this.scrubStartTime = ctx.currentTime;
      this.scrubStartPosition = clampedTarget;
      this.scrubCurrentMediaId = mediaFileId;
      this.scrubIsActive = true;

      this.scrubSource.onended = () => {
        this.scrubIsActive = false;
        this.scrubSource = null;
      };
    } else if (this.scrubSource && timeDelta > 0.001) {
      const elapsedAudioTime = (ctx.currentTime - this.scrubStartTime) * this.scrubSource.playbackRate.value;
      const currentAudioPos = this.scrubStartPosition + elapsedAudioTime;
      const drift = clampedTarget - currentAudioPos;

      if (Math.abs(drift) > 0.3) {
        this.stopScrubAudio();
        return;
      }

      const scrubSpeed = timeDelta > 0.01 ? Math.abs(posDelta) / timeDelta : 1;
      const driftCorrection = drift * 2;
      let targetRate = Math.max(0.25, Math.min(4.0, scrubSpeed + driftCorrection));

      if (posDelta < -0.001) {
        targetRate = 0.25;
      }

      const currentRate = this.scrubSource.playbackRate.value;
      const smoothedRate = currentRate + (targetRate - currentRate) * 0.3;
      this.scrubSource.playbackRate.value = Math.max(0.25, Math.min(4.0, smoothedRate));
    }
  }

  /**
   * Stop scrub audio - call when scrubbing ends
   */
  stopScrubAudio(): void {
    if (this.scrubSource) {
      try {
        this.scrubSource.stop();
        this.scrubSource.disconnect();
      } catch { /* ignore */ }
      this.scrubSource = null;
    }
    if (this.scrubSourceGain) {
      try {
        this.scrubSourceGain.disconnect();
      } catch { /* ignore */ }
      this.scrubSourceGain = null;
    }
    this.scrubIsActive = false;
    this.scrubCurrentMediaId = null;
  }

  /**
   * Check if audio buffer is ready for instant scrubbing
   */
  hasAudioBuffer(mediaFileId: string): boolean {
    return this.audioBufferCache.has(mediaFileId);
  }

  /**
   * Clear audio caches for a specific media file
   */
  clearForMedia(mediaFileId: string) {
    const audio = this.audioCache.get(mediaFileId);
    if (audio) {
      audio.pause();
      URL.revokeObjectURL(audio.src);
      this.audioCache.delete(mediaFileId);
    }
    this.audioBufferCache.delete(mediaFileId);
  }

  /**
   * Clear all audio caches
   */
  clearAll() {
    for (const [, audio] of this.audioCache) {
      audio.pause();
      URL.revokeObjectURL(audio.src);
    }
    this.audioCache.clear();
    this.audioBufferCache.clear();
  }
}

// Singleton instance
export const proxyFrameCache = new ProxyFrameCache();
