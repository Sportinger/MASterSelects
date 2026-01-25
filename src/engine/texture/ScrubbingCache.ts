// Scrubbing frame cache for instant access during timeline scrubbing
// Also includes RAM preview composite cache for instant playback

import { Logger } from '../../services/logger';
import type { GpuFrameCacheEntry } from '../core/types';

const log = Logger.create('ScrubbingCache');

export class ScrubbingCache {
  private device: GPUDevice;

  // Scrubbing frame cache - pre-decoded frames for instant access
  // Key: "videoSrc:frameTime" -> { texture, view }
  // Uses Map insertion order for O(1) LRU operations
  private scrubbingCache: Map<string, { texture: GPUTexture; view: GPUTextureView }> = new Map();
  private maxScrubbingCacheFrames = 300; // ~10 seconds at 30fps, ~2.4GB VRAM at 1080p

  // Last valid frame cache - keeps last frame visible during seeks
  private lastFrameTextures: Map<HTMLVideoElement, GPUTexture> = new Map();
  private lastFrameViews: Map<HTMLVideoElement, GPUTextureView> = new Map();
  private lastFrameSizes: Map<HTMLVideoElement, { width: number; height: number }> = new Map();
  private lastCaptureTime: Map<HTMLVideoElement, number> = new Map();

  // RAM Preview cache - fully composited frames for instant playback
  // Key: time (quantized to frame) -> ImageData (CPU-side for memory efficiency)
  // Uses Map insertion order for O(1) LRU operations
  private compositeCache: Map<number, ImageData> = new Map();
  private maxCompositeCacheFrames = 900; // 30 seconds at 30fps

  // GPU texture cache for instant RAM Preview playback (no CPU->GPU upload needed)
  // Limited size to conserve VRAM (~500MB at 1080p for 60 frames)
  // Uses Map insertion order for O(1) LRU operations
  private gpuFrameCache: Map<number, GpuFrameCacheEntry> = new Map();
  private maxGpuCacheFrames = 60; // ~500MB at 1080p

  constructor(device: GPUDevice) {
    this.device = device;
  }

  // === SCRUBBING FRAME CACHE ===

  // Cache a frame at a specific time for instant scrubbing access
  cacheFrameAtTime(video: HTMLVideoElement, time: number): void {
    if (video.videoWidth === 0 || video.readyState < 2) return;

    const key = `${video.src}:${time.toFixed(3)}`;
    if (this.scrubbingCache.has(key)) return; // Already cached

    const width = video.videoWidth;
    const height = video.videoHeight;

    // Create texture for this frame
    const texture = this.device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    try {
      this.device.queue.copyExternalImageToTexture(
        { source: video },
        { texture },
        [width, height]
      );

      // Add to cache (Map maintains insertion order)
      this.scrubbingCache.set(key, { texture, view: texture.createView() });

      // LRU eviction - evict oldest (first) entries
      while (this.scrubbingCache.size > this.maxScrubbingCacheFrames) {
        const oldestKey = this.scrubbingCache.keys().next().value;
        if (oldestKey) {
          const oldEntry = this.scrubbingCache.get(oldestKey);
          oldEntry?.texture.destroy();
          this.scrubbingCache.delete(oldestKey);
        }
      }
    } catch {
      texture.destroy();
    }
  }

  // Get cached frame for scrubbing
  getCachedFrame(videoSrc: string, time: number): GPUTextureView | null {
    const key = `${videoSrc}:${time.toFixed(3)}`;
    const entry = this.scrubbingCache.get(key);
    if (entry) {
      // Move to end of Map (delete + re-add) for O(1) LRU update
      this.scrubbingCache.delete(key);
      this.scrubbingCache.set(key, entry);
      return entry.view;
    }
    return null;
  }

  // Get scrubbing cache stats
  getScrubbingCacheStats(): { count: number; maxCount: number } {
    return {
      count: this.scrubbingCache.size,
      maxCount: this.maxScrubbingCacheFrames,
    };
  }

  // Clear scrubbing cache for a specific video
  clearScrubbingCache(videoSrc?: string): void {
    if (videoSrc) {
      // Clear only frames from this video
      for (const [key, entry] of this.scrubbingCache) {
        if (key.startsWith(videoSrc)) {
          entry.texture.destroy();
          this.scrubbingCache.delete(key);
        }
      }
    } else {
      // Clear all
      for (const entry of this.scrubbingCache.values()) {
        entry.texture.destroy();
      }
      this.scrubbingCache.clear();
    }
  }

  // === LAST FRAME CACHE ===

  // Capture current video frame to a persistent GPU texture (for last-frame cache)
  captureVideoFrame(video: HTMLVideoElement): void {
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    const width = video.videoWidth;
    const height = video.videoHeight;

    // Get or create texture for this video
    let texture = this.lastFrameTextures.get(video);
    const existingSize = this.lastFrameSizes.get(video);

    // Recreate if size changed
    if (!texture || !existingSize || existingSize.width !== width || existingSize.height !== height) {
      texture?.destroy();
      texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.lastFrameTextures.set(video, texture);
      this.lastFrameSizes.set(video, { width, height });
      this.lastFrameViews.set(video, texture.createView());
    }

    // Copy current frame to texture
    try {
      this.device.queue.copyExternalImageToTexture(
        { source: video },
        { texture },
        [width, height]
      );
    } catch {
      // Video might not be ready - ignore
    }
  }

  // Get last cached frame for a video (used during seeks)
  getLastFrame(video: HTMLVideoElement): { view: GPUTextureView; width: number; height: number } | null {
    const view = this.lastFrameViews.get(video);
    const size = this.lastFrameSizes.get(video);
    if (view && size) {
      return { view, width: size.width, height: size.height };
    }
    return null;
  }

  // Get/set last capture time
  getLastCaptureTime(video: HTMLVideoElement): number {
    return this.lastCaptureTime.get(video) || 0;
  }

  setLastCaptureTime(video: HTMLVideoElement, time: number): void {
    this.lastCaptureTime.set(video, time);
  }

  // Cleanup resources for a video that's no longer used
  cleanupVideo(video: HTMLVideoElement): void {
    const texture = this.lastFrameTextures.get(video);
    if (texture) {
      texture.destroy();
      this.lastFrameTextures.delete(video);
    }
    this.lastFrameViews.delete(video);
    this.lastFrameSizes.delete(video);
    this.lastCaptureTime.delete(video);
  }

  // === RAM PREVIEW COMPOSITE CACHE ===

  // Quantize time to frame number at 30fps for cache key
  quantizeTime(time: number): number {
    return Math.round(time * 30) / 30;
  }

  // Cache composite frame data
  cacheCompositeFrame(time: number, imageData: ImageData): void {
    const key = this.quantizeTime(time);
    if (this.compositeCache.has(key)) return;

    this.compositeCache.set(key, imageData);

    // Evict oldest frames if over limit
    while (this.compositeCache.size > this.maxCompositeCacheFrames) {
      const oldestKey = this.compositeCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.compositeCache.delete(oldestKey);
      }
    }
  }

  // Get cached composite frame if available
  getCachedCompositeFrame(time: number): ImageData | null {
    const key = this.quantizeTime(time);
    const imageData = this.compositeCache.get(key);

    if (imageData) {
      // Move to end of Map for O(1) LRU update
      this.compositeCache.delete(key);
      this.compositeCache.set(key, imageData);
      return imageData;
    }
    return null;
  }

  // Check if a frame is cached
  hasCompositeCacheFrame(time: number): boolean {
    return this.compositeCache.has(this.quantizeTime(time));
  }

  // Get composite cache stats
  getCompositeCacheStats(outputWidth: number, outputHeight: number): { count: number; maxFrames: number; memoryMB: number } {
    const count = this.compositeCache.size;
    const bytesPerFrame = outputWidth * outputHeight * 4;
    const memoryMB = (count * bytesPerFrame) / (1024 * 1024);
    return { count, maxFrames: this.maxCompositeCacheFrames, memoryMB };
  }

  // === GPU FRAME CACHE ===

  // Get cached GPU frame
  getGpuCachedFrame(time: number): GpuFrameCacheEntry | null {
    const key = this.quantizeTime(time);
    const entry = this.gpuFrameCache.get(key);
    if (entry) {
      // Move to end of Map for O(1) LRU update
      this.gpuFrameCache.delete(key);
      this.gpuFrameCache.set(key, entry);
      return entry;
    }
    return null;
  }

  // Add to GPU cache
  addToGpuCache(time: number, entry: GpuFrameCacheEntry): void {
    const key = this.quantizeTime(time);
    this.gpuFrameCache.set(key, entry);

    // Evict oldest GPU cached frames if over limit
    while (this.gpuFrameCache.size > this.maxGpuCacheFrames) {
      const oldestKey = this.gpuFrameCache.keys().next().value;
      if (oldestKey !== undefined) {
        const oldEntry = this.gpuFrameCache.get(oldestKey);
        oldEntry?.texture.destroy();
        this.gpuFrameCache.delete(oldestKey);
      }
    }
  }

  // Clear composite cache
  clearCompositeCache(): void {
    this.compositeCache.clear();

    // Clear GPU frame cache
    for (const entry of this.gpuFrameCache.values()) {
      entry.texture.destroy();
    }
    this.gpuFrameCache.clear();

    log.debug('Composite cache cleared');
  }

  // Clear all caches
  clearAll(): void {
    this.clearScrubbingCache();
    this.clearCompositeCache();

    // Clear last frame caches
    for (const texture of this.lastFrameTextures.values()) {
      texture.destroy();
    }
    this.lastFrameTextures.clear();
    this.lastFrameViews.clear();
    this.lastFrameSizes.clear();
    this.lastCaptureTime.clear();

    log.debug('All caches cleared');
  }

  destroy(): void {
    this.clearAll();
  }
}
