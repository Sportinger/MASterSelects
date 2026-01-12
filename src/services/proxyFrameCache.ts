// Proxy frame cache - loads and caches WebP frames for fast playback

import { projectFileService } from './projectFileService';
import { useMediaStore } from '../stores/mediaStore';

// Cache settings
const MAX_CACHE_SIZE = 300; // Maximum frames to keep in memory (10s at 30fps)
const PRELOAD_AHEAD_FRAMES = 30; // Preload this many frames ahead for smooth playback (1s at 30fps)
const PARALLEL_LOAD_COUNT = 8; // Load this many frames in parallel for faster preload

// Frame cache entry
interface CachedFrame {
  mediaFileId: string;
  frameIndex: number;
  image: HTMLImageElement;
  timestamp: number; // For LRU eviction
}

class ProxyFrameCache {
  private cache: Map<string, CachedFrame> = new Map();
  private loadingPromises: Map<string, Promise<HTMLImageElement | null>> = new Map();
  private preloadQueue: string[] = [];
  private isPreloading = false;

  // Audio proxy cache
  private audioCache: Map<string, HTMLAudioElement> = new Map();
  private audioLoadingPromises: Map<string, Promise<HTMLAudioElement | null>> = new Map();

  // Audio buffer cache for instant scrubbing (Web Audio API)
  private audioBufferCache: Map<string, AudioBuffer> = new Map();
  private audioContext: AudioContext | null = null;
  private scrubGain: GainNode | null = null;

  // Get cache key
  private getKey(mediaFileId: string, frameIndex: number): string {
    return `${mediaFileId}_${frameIndex}`;
  }

  // Synchronously get a frame if it's already in memory cache
  // Also triggers preloading of upcoming frames (even if current frame not cached)
  getCachedFrame(mediaFileId: string, frameIndex: number, fps: number = 30): HTMLImageElement | null {
    const key = this.getKey(mediaFileId, frameIndex);
    const cached = this.cache.get(key);

    // ALWAYS trigger preloading, even if current frame isn't cached
    // This ensures nested composition frames get preloaded when playhead enters them
    this.schedulePreload(mediaFileId, frameIndex, fps);

    if (cached) {
      cached.timestamp = Date.now(); // Update for LRU
      return cached.image;
    }
    return null;
  }

  // Get nearest cached frame for scrubbing fallback
  // Returns the closest frame within maxDistance frames
  getNearestCachedFrame(mediaFileId: string, frameIndex: number, maxDistance: number = 15): HTMLImageElement | null {
    // Check exact frame first
    const exactKey = this.getKey(mediaFileId, frameIndex);
    const exact = this.cache.get(exactKey);
    if (exact) {
      exact.timestamp = Date.now();
      return exact.image;
    }

    // Search nearby frames (prefer recent frames, then earlier)
    for (let d = 1; d <= maxDistance; d++) {
      // Check frame ahead first (more recent content)
      const aheadKey = this.getKey(mediaFileId, frameIndex + d);
      const ahead = this.cache.get(aheadKey);
      if (ahead) {
        ahead.timestamp = Date.now();
        return ahead.image;
      }

      // Then check frame behind
      if (frameIndex - d >= 0) {
        const behindKey = this.getKey(mediaFileId, frameIndex - d);
        const behind = this.cache.get(behindKey);
        if (behind) {
          behind.timestamp = Date.now();
          return behind.image;
        }
      }
    }

    return null;
  }

  // Get a frame from cache or load it
  async getFrame(mediaFileId: string, time: number, fps: number = 30): Promise<HTMLImageElement | null> {
    const frameIndex = Math.floor(time * fps);
    const key = this.getKey(mediaFileId, frameIndex);

    // Check cache first
    const cached = this.cache.get(key);
    if (cached) {
      cached.timestamp = Date.now(); // Update for LRU
      return cached.image;
    }

    // Check if already loading
    const loadingPromise = this.loadingPromises.get(key);
    if (loadingPromise) {
      return loadingPromise;
    }

    // Load from IndexedDB
    const promise = this.loadFrame(mediaFileId, frameIndex);
    this.loadingPromises.set(key, promise);

    try {
      const image = await promise;
      if (image) {
        this.addToCache(mediaFileId, frameIndex, image);
        // Trigger preload of upcoming frames
        this.schedulePreload(mediaFileId, frameIndex, fps);
      }
      return image;
    } finally {
      this.loadingPromises.delete(key);
    }
  }

  // Load a single frame - ONLY from project folder (no browser cache)
  private async loadFrame(mediaFileId: string, frameIndex: number): Promise<HTMLImageElement | null> {
    try {
      let blob: Blob | null = null;

      // Get the media file to find its fileHash (used for proxy folder naming)
      const mediaFile = useMediaStore.getState().files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      // Debug logging
      if (frameIndex === 0) {
        console.log('[ProxyCache] Loading frame 0 for:', mediaFile?.name);
        console.log('[ProxyCache] storageKey:', storageKey);
        console.log('[ProxyCache] projectOpen:', projectFileService.isProjectOpen());
        console.log('[ProxyCache] proxyStatus:', mediaFile?.proxyStatus);
      }

      // Load from project folder ONLY (no IndexedDB fallback)
      if (projectFileService.isProjectOpen()) {
        blob = await projectFileService.getProxyFrame(storageKey, frameIndex);
        if (frameIndex === 0) {
          console.log('[ProxyCache] Frame 0 blob:', blob ? `${blob.size} bytes` : 'null');
        }
      }

      if (!blob) return null;

      // Create image from blob
      const url = URL.createObjectURL(blob);
      const image = new Image();

      return new Promise((resolve) => {
        image.onload = () => {
          URL.revokeObjectURL(url);
          resolve(image);
        };
        image.onerror = () => {
          URL.revokeObjectURL(url);
          resolve(null);
        };
        image.src = url;
      });
    } catch (e) {
      console.warn('[ProxyCache] Failed to load frame:', e);
      return null;
    }
  }

  // Add frame to cache
  private addToCache(mediaFileId: string, frameIndex: number, image: HTMLImageElement) {
    const key = this.getKey(mediaFileId, frameIndex);

    // Evict old frames if cache is full
    if (this.cache.size >= MAX_CACHE_SIZE) {
      this.evictOldest();
    }

    this.cache.set(key, {
      mediaFileId,
      frameIndex,
      image,
      timestamp: Date.now(),
    });
  }

  // Evict oldest frame from cache (LRU)
  private evictOldest() {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache) {
      if (entry.timestamp < oldestTime) {
        oldestTime = entry.timestamp;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  // Schedule preloading of current and upcoming frames
  private schedulePreload(mediaFileId: string, currentFrameIndex: number, _fps: number) {
    // Add current frame first (highest priority for nested comp entry)
    // Then add upcoming frames to preload queue
    for (let i = 0; i <= PRELOAD_AHEAD_FRAMES; i++) {
      const frameIndex = currentFrameIndex + i;
      const key = this.getKey(mediaFileId, frameIndex);

      // Skip if already cached or in queue
      if (!this.cache.has(key) && !this.preloadQueue.includes(key)) {
        // Insert current frame at front of queue for priority loading
        if (i === 0) {
          this.preloadQueue.unshift(key);
        } else {
          this.preloadQueue.push(key);
        }
      }
    }

    // Start preloading if not already
    if (!this.isPreloading) {
      this.processPreloadQueue();
    }
  }

  // Process preload queue with parallel loading for speed
  private async processPreloadQueue() {
    this.isPreloading = true;

    while (this.preloadQueue.length > 0) {
      // Load multiple frames in parallel for faster preloading
      const batch: string[] = [];
      while (batch.length < PARALLEL_LOAD_COUNT && this.preloadQueue.length > 0) {
        const key = this.preloadQueue.shift();
        if (key && !this.cache.has(key)) {
          batch.push(key);
        }
      }

      if (batch.length === 0) continue;

      // Load batch in parallel
      const loadPromises = batch.map(async (key) => {
        const [mediaFileId, frameIndexStr] = key.split('_');
        const frameIndex = parseInt(frameIndexStr, 10);

        const image = await this.loadFrame(mediaFileId, frameIndex);
        if (image) {
          this.addToCache(mediaFileId, frameIndex, image);
        }
        return { key, success: !!image };
      });

      await Promise.all(loadPromises);

      // Brief yield to main thread between batches
      await new Promise((r) => setTimeout(r, 0));
    }

    this.isPreloading = false;
  }

  // ============================================
  // AUDIO PROXY METHODS
  // ============================================

  /**
   * Get cached audio proxy element, or load it if not cached
   * Returns null if no audio proxy exists
   */
  async getAudioProxy(mediaFileId: string): Promise<HTMLAudioElement | null> {
    // Check cache first
    const cached = this.audioCache.get(mediaFileId);
    if (cached) {
      return cached;
    }

    // Check if already loading
    const existingPromise = this.audioLoadingPromises.get(mediaFileId);
    if (existingPromise) {
      return existingPromise;
    }

    // Start loading
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
    // Just call getAudioProxy which handles caching
    await this.getAudioProxy(mediaFileId);
  }

  /**
   * Load audio proxy from project folder
   */
  private async loadAudioProxy(mediaFileId: string): Promise<HTMLAudioElement | null> {
    try {
      // Get storage key (prefer fileHash for deduplication)
      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      // Load audio file from project folder
      const audioFile = await projectFileService.getProxyAudio(storageKey);
      if (!audioFile) {
        return null;
      }

      // Create audio element with object URL
      const audio = new Audio();
      audio.src = URL.createObjectURL(audioFile);
      audio.preload = 'auto';

      // Wait for audio to be ready
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
        // Start loading
        audio.load();
      });

      console.log(`[ProxyFrameCache] Audio proxy loaded for ${mediaFileId}`);
      return audio;
    } catch (e) {
      console.warn(`[ProxyFrameCache] Failed to load audio proxy for ${mediaFileId}:`, e);
      return null;
    }
  }

  // ============================================
  // INSTANT AUDIO SCRUBBING (Web Audio API)
  // ============================================

  /**
   * Get or create AudioContext for scrubbing
   */
  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
      this.scrubGain = this.audioContext.createGain();
      this.scrubGain.connect(this.audioContext.destination);
      this.scrubGain.gain.value = 1.0; // Full volume for scrubbing
    }
    return this.audioContext;
  }

  /**
   * Get AudioBuffer for a media file (decode on first request)
   */
  async getAudioBuffer(mediaFileId: string): Promise<AudioBuffer | null> {
    // Check cache
    const cached = this.audioBufferCache.get(mediaFileId);
    if (cached) return cached;

    try {
      // Get storage key
      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      // Load audio file from project folder
      const audioFile = await projectFileService.getProxyAudio(storageKey);
      if (!audioFile) return null;

      // Decode to AudioBuffer
      const arrayBuffer = await audioFile.arrayBuffer();
      const audioContext = this.getAudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // Cache it
      this.audioBufferCache.set(mediaFileId, audioBuffer);
      console.log(`[ProxyFrameCache] Audio buffer decoded for ${mediaFileId}: ${audioBuffer.duration.toFixed(1)}s`);

      return audioBuffer;
    } catch (e) {
      console.warn(`[ProxyFrameCache] Failed to decode audio buffer:`, e);
      return null;
    }
  }

  // Track active scrub sources for overlapping playback
  private activeScrubSources: AudioBufferSourceNode[] = [];

  /**
   * Play instant scrub audio at a specific time
   * Uses AudioBuffer for zero-latency seeking
   * Allows overlapping snippets for continuous sound during fast scrubbing
   */
  playScrubAudio(mediaFileId: string, time: number, duration: number = 0.15): void {
    const buffer = this.audioBufferCache.get(mediaFileId);
    if (!buffer) {
      // Start loading buffer for next time
      this.getAudioBuffer(mediaFileId);
      return;
    }

    try {
      const ctx = this.getAudioContext();

      // Resume context if suspended (browser autoplay policy)
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      // Clean up finished sources (keep max 3 overlapping)
      // Note: sources auto-remove via onended callback, this is just a safety limit
      if (this.activeScrubSources.length > 5) {
        this.activeScrubSources = this.activeScrubSources.slice(-3);
      }

      // If too many sources, stop oldest
      while (this.activeScrubSources.length > 3) {
        const oldest = this.activeScrubSources.shift();
        if (oldest) {
          try {
            oldest.stop();
            oldest.disconnect();
          } catch { /* ignore */ }
        }
      }

      // Create new source with its own gain for fade
      const sourceGain = ctx.createGain();
      sourceGain.connect(this.scrubGain!);
      sourceGain.gain.value = 1.0;

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(sourceGain);

      // Calculate valid start time
      const startTime = Math.max(0, Math.min(time, buffer.duration - duration));

      // Auto-cleanup when done
      source.onended = () => {
        const idx = this.activeScrubSources.indexOf(source);
        if (idx >= 0) this.activeScrubSources.splice(idx, 1);
        sourceGain.disconnect();
      };

      // Play snippet
      source.start(0, startTime, duration);
      this.activeScrubSources.push(source);
    } catch {
      // Ignore scrub errors
    }
  }

  /**
   * Check if audio buffer is ready for instant scrubbing
   */
  hasAudioBuffer(mediaFileId: string): boolean {
    return this.audioBufferCache.has(mediaFileId);
  }

  // Clear cache for a specific media file
  clearForMedia(mediaFileId: string) {
    for (const [key] of this.cache) {
      if (key.startsWith(mediaFileId + '_')) {
        this.cache.delete(key);
      }
    }
    this.preloadQueue = this.preloadQueue.filter((k) => !k.startsWith(mediaFileId + '_'));

    // Also clear audio cache
    const audio = this.audioCache.get(mediaFileId);
    if (audio) {
      audio.pause();
      URL.revokeObjectURL(audio.src);
      this.audioCache.delete(mediaFileId);
    }

    // Clear audio buffer cache
    this.audioBufferCache.delete(mediaFileId);
  }

  // Clear entire cache
  clearAll() {
    this.cache.clear();
    this.preloadQueue = [];

    // Clear audio cache
    for (const [, audio] of this.audioCache) {
      audio.pause();
      URL.revokeObjectURL(audio.src);
    }
    this.audioCache.clear();

    // Clear audio buffer cache
    this.audioBufferCache.clear();
  }

  // Get cache stats
  getStats() {
    return {
      cachedFrames: this.cache.size,
      preloadQueueSize: this.preloadQueue.length,
      isPreloading: this.isPreloading,
    };
  }
}

// Singleton instance
export const proxyFrameCache = new ProxyFrameCache();
