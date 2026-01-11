// Proxy frame cache - loads and caches WebP frames for fast playback

import { projectDB } from './projectDB';
import { projectFileService } from './projectFileService';

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

  // Load a single frame - check project folder first, then IndexedDB
  private async loadFrame(mediaFileId: string, frameIndex: number): Promise<HTMLImageElement | null> {
    try {
      let blob: Blob | null = null;

      // Try project folder first if a project is open
      if (projectFileService.isProjectOpen()) {
        blob = await projectFileService.getProxyFrame(mediaFileId, frameIndex);
      }

      // Fall back to IndexedDB if not found in project folder
      if (!blob) {
        const frame = await projectDB.getProxyFrame(mediaFileId, frameIndex);
        if (frame) {
          blob = frame.blob;
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

  // Clear cache for a specific media file
  clearForMedia(mediaFileId: string) {
    for (const [key] of this.cache) {
      if (key.startsWith(mediaFileId + '_')) {
        this.cache.delete(key);
      }
    }
    this.preloadQueue = this.preloadQueue.filter((k) => !k.startsWith(mediaFileId + '_'));
  }

  // Clear entire cache
  clearAll() {
    this.cache.clear();
    this.preloadQueue = [];
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
