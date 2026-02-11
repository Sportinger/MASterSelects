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
  private maxCompositeCacheBytes = 512 * 1024 * 1024; // 512MB memory limit
  private compositeCacheBytes = 0; // Track actual memory usage

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
      // Don't destroy textures - let GC handle to avoid GPU conflicts
      while (this.scrubbingCache.size > this.maxScrubbingCacheFrames) {
        const oldestKey = this.scrubbingCache.keys().next().value;
        if (oldestKey) {
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
    // Don't destroy textures - let GC handle to avoid GPU conflicts
    if (videoSrc) {
      // Clear only frames from this video
      for (const key of this.scrubbingCache.keys()) {
        if (key.startsWith(videoSrc)) {
          this.scrubbingCache.delete(key);
        }
      }
    } else {
      // Clear all
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

    // Recreate if size changed (don't destroy old - let GC handle to avoid GPU conflicts)
    if (!texture || !existingSize || existingSize.width !== width || existingSize.height !== height) {
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

  // Capture video frame via canvas intermediate (CPU decode path)
  // Works even when the GPU decoder hasn't presented a frame yet (e.g. after page reload)
  // canvas.drawImage uses the browser's CPU-side decoded frame which is always available at readyState >= 2
  captureVideoFrameViaCanvas(video: HTMLVideoElement): void {
    if (video.videoWidth === 0 || video.videoHeight === 0 || video.readyState < 2) return;

    const width = video.videoWidth;
    const height = video.videoHeight;

    // Draw video to canvas (forces CPU decode)
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, width, height);

    // Get or create texture
    let texture = this.lastFrameTextures.get(video);
    const existingSize = this.lastFrameSizes.get(video);

    if (!texture || !existingSize || existingSize.width !== width || existingSize.height !== height) {
      texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.lastFrameTextures.set(video, texture);
      this.lastFrameSizes.set(video, { width, height });
      this.lastFrameViews.set(video, texture.createView());
    }

    try {
      this.device.queue.copyExternalImageToTexture(
        { source: canvas },
        { texture },
        [width, height]
      );
    } catch {
      // Canvas copy failed - nothing we can do
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
    // Don't destroy textures - let GC handle to avoid GPU conflicts
    this.lastFrameTextures.delete(video);
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

    const frameBytes = imageData.data.byteLength;
    this.compositeCache.set(key, imageData);
    this.compositeCacheBytes += frameBytes;

    // Evict oldest frames if over frame count OR memory limit
    while (
      this.compositeCache.size > this.maxCompositeCacheFrames ||
      this.compositeCacheBytes > this.maxCompositeCacheBytes
    ) {
      const oldestKey = this.compositeCache.keys().next().value;
      if (oldestKey !== undefined) {
        const evicted = this.compositeCache.get(oldestKey);
        if (evicted) this.compositeCacheBytes -= evicted.data.byteLength;
        this.compositeCache.delete(oldestKey);
      } else {
        break;
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
  getCompositeCacheStats(_outputWidth: number, _outputHeight: number): { count: number; maxFrames: number; memoryMB: number } {
    const count = this.compositeCache.size;
    const memoryMB = this.compositeCacheBytes / (1024 * 1024);
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
    // Don't destroy textures - let GC handle to avoid GPU conflicts
    while (this.gpuFrameCache.size > this.maxGpuCacheFrames) {
      const oldestKey = this.gpuFrameCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.gpuFrameCache.delete(oldestKey);
      }
    }
  }

  // Clear composite cache
  clearCompositeCache(): void {
    this.compositeCache.clear();
    this.compositeCacheBytes = 0;

    // Clear GPU frame cache (don't destroy - let GC handle to avoid GPU conflicts)
    this.gpuFrameCache.clear();

    log.debug('Composite cache cleared');
  }

  // Clear all caches
  clearAll(): void {
    this.clearScrubbingCache();
    this.clearCompositeCache();

    // Clear last frame caches (don't destroy - let GC handle to avoid GPU conflicts)
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
