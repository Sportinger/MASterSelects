// Scrubbing frame cache for instant access during timeline scrubbing
// Also includes RAM preview composite cache for instant playback

import { Logger } from '../../services/logger';
import { vfPipelineMonitor } from '../../services/vfPipelineMonitor';
import type { GpuFrameCacheEntry } from '../core/types';

const log = Logger.create('ScrubbingCache');

export class ScrubbingCache {
  private device: GPUDevice;

  // Scrubbing frame cache - pre-decoded frames for instant access
  // Key: "videoSrc:quantizedFrameTime" -> { texture, view }
  // Time is quantized to frame boundaries (1/30s) for better cache hit rate
  // Uses Map insertion order for O(1) LRU operations
  private scrubbingCache: Map<string, { texture: GPUTexture; view: GPUTextureView }> = new Map();
  private maxScrubbingCacheFrames = 300; // ~10 seconds at 30fps, ~2.4GB VRAM at 1080p
  private readonly SCRUB_CACHE_FPS = 30; // Quantization granularity for scrubbing cache keys

  // Last valid frame cache - keeps last frame visible during seeks
  private lastFrameTextures: Map<HTMLVideoElement, GPUTexture> = new Map();
  private lastFrameViews: Map<HTMLVideoElement, GPUTextureView> = new Map();
  private lastFrameSizes: Map<HTMLVideoElement, { width: number; height: number }> = new Map();
  private lastFrameMediaTimes: Map<HTMLVideoElement, number> = new Map();
  private lastFrameOwners = new WeakMap<HTMLVideoElement, string>();
  private lastCaptureTime: Map<HTMLVideoElement, number> = new Map();
  private lastPresentedFrameTimes = new WeakMap<HTMLVideoElement, number>();
  private lastPresentedFrameOwners = new WeakMap<HTMLVideoElement, string>();
  private lastOwnerMissSignature = new WeakMap<HTMLVideoElement, string>();
  private lastOwnerMissAt = new WeakMap<HTMLVideoElement, number>();

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

  // Quantize time to nearest frame boundary for consistent cache keys.
  // Two scrub positions within the same frame (e.g. 1.5001s and 1.5009s at 30fps)
  // map to the same key, dramatically improving cache hit rate.
  private quantizeToFrame(time: number): string {
    return (Math.round(time * this.SCRUB_CACHE_FPS) / this.SCRUB_CACHE_FPS).toFixed(3);
  }

  private frameIndexForTime(time: number): number {
    return Math.round(time * this.SCRUB_CACHE_FPS);
  }

  private getScrubbingKey(videoSrc: string, time: number): string {
    return `${videoSrc}:${this.quantizeToFrame(time)}`;
  }

  private getScrubbingKeyTime(key: string): number {
    const index = key.lastIndexOf(':');
    if (index === -1) {
      return 0;
    }
    const parsed = Number(key.slice(index + 1));
    return Number.isFinite(parsed) ? parsed : 0;
  }

  private touchScrubbingEntry(
    key: string,
    entry: { texture: GPUTexture; view: GPUTextureView }
  ): { view: GPUTextureView; mediaTime: number } {
    this.scrubbingCache.delete(key);
    this.scrubbingCache.set(key, entry);
    return {
      view: entry.view,
      mediaTime: this.getScrubbingKeyTime(key),
    };
  }

  // Cache a frame at a specific time for instant scrubbing access
  cacheFrameAtTime(video: HTMLVideoElement, time: number): void {
    if (video.videoWidth === 0 || video.readyState < 2) return;

    const key = this.getScrubbingKey(video.src, time);
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

  // Get cached frame for scrubbing (uses quantized time for better hit rate)
  getCachedFrame(videoSrc: string, time: number): GPUTextureView | null {
    return this.getCachedFrameEntry(videoSrc, time)?.view ?? null;
  }

  getCachedFrameEntry(
    videoSrc: string,
    time: number
  ): { view: GPUTextureView; mediaTime: number } | null {
    const key = this.getScrubbingKey(videoSrc, time);
    const entry = this.scrubbingCache.get(key);
    if (entry) {
      return this.touchScrubbingEntry(key, entry);
    }
    return null;
  }

  // Return the nearest cached frame around the requested media time.
  // This is used during scrubbing to avoid flashing black while the exact
  // seek target is still decoding.
  getNearestCachedFrame(
    videoSrc: string,
    time: number,
    maxDistanceFrames: number = 6
  ): GPUTextureView | null {
    return this.getNearestCachedFrameEntry(videoSrc, time, maxDistanceFrames)?.view ?? null;
  }

  getNearestCachedFrameEntry(
    videoSrc: string,
    time: number,
    maxDistanceFrames: number = 6
  ): { view: GPUTextureView; mediaTime: number } | null {
    const exact = this.getCachedFrameEntry(videoSrc, time);
    if (exact) {
      return exact;
    }

    const baseFrame = this.frameIndexForTime(time);
    for (let distance = 1; distance <= maxDistanceFrames; distance++) {
      const previous = this.scrubbingCache.get(
        this.getScrubbingKey(videoSrc, (baseFrame - distance) / this.SCRUB_CACHE_FPS)
      );
      if (previous) {
        return this.touchScrubbingEntry(
          this.getScrubbingKey(videoSrc, (baseFrame - distance) / this.SCRUB_CACHE_FPS),
          previous
        );
      }

      const next = this.scrubbingCache.get(
        this.getScrubbingKey(videoSrc, (baseFrame + distance) / this.SCRUB_CACHE_FPS)
      );
      if (next) {
        return this.touchScrubbingEntry(
          this.getScrubbingKey(videoSrc, (baseFrame + distance) / this.SCRUB_CACHE_FPS),
          next
        );
      }
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
  captureVideoFrame(video: HTMLVideoElement, ownerId?: string): boolean {
    if (video.videoWidth === 0 || video.videoHeight === 0) return false;

    const width = video.videoWidth;
    const height = video.videoHeight;

    // Reuse the existing texture when dimensions match so we never replace
    // a known-good frame with an uninitialized texture if the copy fails.
    const existingTexture = this.lastFrameTextures.get(video);
    const existingSize = this.lastFrameSizes.get(video);
    const canReuseExisting =
      !!existingTexture &&
      !!existingSize &&
      existingSize.width === width &&
      existingSize.height === height;

    let texture = existingTexture;
    let view = this.lastFrameViews.get(video);
    let createdFreshTexture = false;

    if (!canReuseExisting) {
      texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      view = texture.createView();
      createdFreshTexture = true;
    }

    // Copy current frame to texture
    try {
      this.device.queue.copyExternalImageToTexture(
        { source: video },
        { texture: texture! },
        [width, height]
      );
      if (createdFreshTexture) {
        this.lastFrameTextures.set(video, texture!);
        this.lastFrameViews.set(video, view!);
        this.lastFrameSizes.set(video, { width, height });
      }
      this.lastFrameMediaTimes.set(video, video.currentTime);
      if (ownerId) {
        this.lastFrameOwners.set(video, ownerId);
      }
      return true;
    } catch {
      if (createdFreshTexture) {
        texture?.destroy();
      }
      return false;
    }
  }

  // Capture video frame via createImageBitmap (async forced decode)
  // This is the ONLY API that forces Chrome to actually decode a video frame.
  // After page reload, all sync APIs (canvas.drawImage, importExternalTexture,
  // new VideoFrame, copyExternalImageToTexture) return black/empty data because
  // Chrome defers frame decoding. createImageBitmap forces async decode.
  async captureVideoFrameViaImageBitmap(video: HTMLVideoElement, ownerId?: string): Promise<boolean> {
    if (video.videoWidth === 0 || video.videoHeight === 0 || video.readyState < 2) {
      return false;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;

    try {
      // createImageBitmap is the ONLY browser API that forces actual frame decode
      const bitmap = await createImageBitmap(video);

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

      this.device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture },
        [width, height]
      );
      bitmap.close();
      this.lastFrameMediaTimes.set(video, video.currentTime);
      if (ownerId) {
        this.lastFrameOwners.set(video, ownerId);
      }
      log.debug('Pre-cached video frame via createImageBitmap', { width, height });
      return true;
    } catch (e) {
      log.warn('captureVideoFrameViaImageBitmap failed', e);
      return false;
    }
  }

  captureVideoFrameAtTime(
    video: HTMLVideoElement,
    mediaTime: number,
    ownerId?: string
  ): boolean {
    const captured = this.captureVideoFrame(video, ownerId);
    if (captured && Number.isFinite(mediaTime)) {
      this.lastFrameMediaTimes.set(video, mediaTime);
    }
    return captured;
  }

  captureVideoFrameIfCloser(
    video: HTMLVideoElement,
    targetTime: number,
    candidateTime: number,
    ownerId?: string,
    minIntervalMs: number = 50
  ): boolean {
    if (!Number.isFinite(candidateTime) || !Number.isFinite(targetTime)) {
      return false;
    }

    const existing = this.getLastFrame(video, ownerId);
    const existingDrift =
      typeof existing?.mediaTime === 'number' && Number.isFinite(existing.mediaTime)
        ? Math.abs(existing.mediaTime - targetTime)
        : Number.POSITIVE_INFINITY;
    const candidateDrift = Math.abs(candidateTime - targetTime);

    if (candidateDrift + 0.001 >= existingDrift) {
      return false;
    }

    const now = performance.now();
    const lastCapture = this.getLastCaptureTime(video);
    if (now - lastCapture < minIntervalMs) {
      return false;
    }

    const captured = this.captureVideoFrameAtTime(video, candidateTime, ownerId);
    if (captured) {
      this.setLastCaptureTime(video, now);
    }
    return captured;
  }

  // Get last cached frame for a video (used during seeks)
  getLastFrame(
    video: HTMLVideoElement,
    ownerId?: string
  ): { view: GPUTextureView; width: number; height: number; mediaTime?: number } | null {
    const view = this.lastFrameViews.get(video);
    const size = this.lastFrameSizes.get(video);
    const owner = this.lastFrameOwners.get(video);
    if (ownerId && owner !== ownerId) {
      const now = performance.now();
      const signature = `${ownerId}:${owner ?? 'none'}`;
      const lastSignature = this.lastOwnerMissSignature.get(video);
      const lastAt = this.lastOwnerMissAt.get(video) ?? 0;
      if (signature !== lastSignature || now - lastAt > 120) {
        this.lastOwnerMissSignature.set(video, signature);
        this.lastOwnerMissAt.set(video, now);
        vfPipelineMonitor.record('vf_scrub_owner_miss', {
          requestedClipId: ownerId,
          cachedClipId: owner ?? 'none',
          cachedTimeMs: Math.round((this.lastFrameMediaTimes.get(video) ?? -1) * 1000),
          currentTimeMs: Math.round(video.currentTime * 1000),
        });
      }
      return null;
    }
    if (view && size) {
      return {
        view,
        width: size.width,
        height: size.height,
        mediaTime: this.lastFrameMediaTimes.get(video),
      };
    }
    return null;
  }

  // Only reuse a copied last-frame fallback when it was captured close to the
  // requested media time. This avoids flashing unrelated frames during seeks.
  getLastFrameNearTime(
    video: HTMLVideoElement,
    targetTime: number,
    maxDeltaSeconds: number,
    ownerId?: string
  ): { view: GPUTextureView; width: number; height: number; mediaTime?: number } | null {
    const frame = this.getLastFrame(video, ownerId);
    if (!frame) {
      return null;
    }

    const frameTime = frame.mediaTime;
    if (typeof frameTime !== 'number' || !Number.isFinite(frameTime)) {
      return null;
    }

    return Math.abs(frameTime - targetTime) <= maxDeltaSeconds
      ? frame
      : null;
  }

  // Get/set last capture time
  getLastCaptureTime(video: HTMLVideoElement): number {
    return this.lastCaptureTime.get(video) || 0;
  }

  setLastCaptureTime(video: HTMLVideoElement, time: number): void {
    this.lastCaptureTime.set(video, time);
  }

  getLastPresentedTime(video: HTMLVideoElement): number | undefined {
    return this.lastPresentedFrameTimes.get(video);
  }

  getLastPresentedOwner(video: HTMLVideoElement): string | undefined {
    return this.lastPresentedFrameOwners.get(video);
  }

  markFramePresented(
    video: HTMLVideoElement,
    time: number = video.currentTime,
    ownerId?: string
  ): void {
    if (Number.isFinite(time)) {
      this.lastPresentedFrameTimes.set(video, time);
    }
    if (ownerId) {
      this.lastPresentedFrameOwners.set(video, ownerId);
    }
  }

  // Cleanup resources for a video that's no longer used
  cleanupVideo(video: HTMLVideoElement): void {
    // Don't destroy textures - let GC handle to avoid GPU conflicts
    this.lastFrameTextures.delete(video);
    this.lastFrameViews.delete(video);
    this.lastFrameSizes.delete(video);
    this.lastFrameMediaTimes.delete(video);
    this.lastFrameOwners.delete(video);
    this.lastCaptureTime.delete(video);
    this.lastPresentedFrameTimes.delete(video);
    this.lastPresentedFrameOwners.delete(video);
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
    this.lastFrameMediaTimes.clear();
    this.lastFrameOwners = new WeakMap();
    this.lastCaptureTime.clear();
    this.lastPresentedFrameTimes = new WeakMap();

    log.debug('All caches cleared');
  }

  destroy(): void {
    this.clearAll();
  }
}
