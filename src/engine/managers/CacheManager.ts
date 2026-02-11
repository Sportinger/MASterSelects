// CacheManager - Extracted from WebGPUEngine
// Owns ScrubbingCache lifecycle, video time tracking, and RAM preview canvas state

import { ScrubbingCache } from '../texture/ScrubbingCache';
import { Logger } from '../../services/logger';

const log = Logger.create('CacheManager');

export class CacheManager {
  private scrubbingCache: ScrubbingCache | null = null;
  private lastVideoTime: Map<string, number> = new Map();
  private ramPlaybackCanvas: HTMLCanvasElement | null = null;
  private ramPlaybackCtx: CanvasRenderingContext2D | null = null;

  // --- Lifecycle ---

  initialize(device: GPUDevice): void {
    this.scrubbingCache = new ScrubbingCache(device);
  }

  handleDeviceLost(): void {
    this.scrubbingCache = null;
    this.lastVideoTime.clear();
  }

  destroy(): void {
    this.scrubbingCache?.destroy();
    this.scrubbingCache = null;
    this.lastVideoTime.clear();
    this.ramPlaybackCanvas = null;
    this.ramPlaybackCtx = null;
  }

  // --- Scrubbing Cache ---

  cacheFrameAtTime(video: HTMLVideoElement, time: number): void {
    this.scrubbingCache?.cacheFrameAtTime(video, time);
  }

  getCachedFrame(videoSrc: string, time: number): GPUTextureView | null {
    return this.scrubbingCache?.getCachedFrame(videoSrc, time) ?? null;
  }

  getScrubbingCacheStats(): { count: number; maxCount: number } {
    return this.scrubbingCache?.getScrubbingCacheStats() ?? { count: 0, maxCount: 0 };
  }

  clearScrubbingCache(videoSrc?: string): void {
    this.scrubbingCache?.clearScrubbingCache(videoSrc);
  }

  ensureVideoFrameCached(video: HTMLVideoElement): void {
    if (this.scrubbingCache && !this.scrubbingCache.getLastFrame(video)) {
      this.scrubbingCache.captureVideoFrame(video);
    }
  }

  async preCacheVideoFrame(video: HTMLVideoElement): Promise<boolean> {
    if (!this.scrubbingCache) return false;
    return this.scrubbingCache.captureVideoFrameViaImageBitmap(video);
  }

  cleanupVideoCache(video: HTMLVideoElement): void {
    this.scrubbingCache?.cleanupVideo(video);
  }

  // --- RAM Preview / Composite Cache ---

  async cacheCompositeFrame(
    time: number,
    readPixels: () => Promise<Uint8ClampedArray | null>,
    getResolution: () => { width: number; height: number }
  ): Promise<void> {
    if (!this.scrubbingCache) return;
    if (this.scrubbingCache.hasCompositeCacheFrame(time)) return;

    const pixels = await readPixels();
    if (!pixels) return;

    const { width, height } = getResolution();

    if (this.scrubbingCache.getCompositeCacheStats(width, height).count === 0) {
      let nonZero = 0;
      for (let i = 0; i < Math.min(1000, pixels.length); i++) {
        if (pixels[i] !== 0) nonZero++;
      }
      log.debug('RAM Preview first frame', { nonZero, width, height });
    }

    const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
    this.scrubbingCache.cacheCompositeFrame(time, imageData);
  }

  getCachedCompositeFrame(time: number): ImageData | null {
    return this.scrubbingCache?.getCachedCompositeFrame(time) ?? null;
  }

  hasCompositeCacheFrame(time: number): boolean {
    return this.scrubbingCache?.hasCompositeCacheFrame(time) ?? false;
  }

  clearCompositeCache(): void {
    this.scrubbingCache?.clearCompositeCache();
    this.ramPlaybackCanvas = null;
    this.ramPlaybackCtx = null;
    log.debug('Composite cache cleared');
  }

  getCompositeCacheStats(getResolution: () => { width: number; height: number }): { count: number; maxFrames: number; memoryMB: number } {
    const { width, height } = getResolution();
    return this.scrubbingCache?.getCompositeCacheStats(width, height) ?? { count: 0, maxFrames: 0, memoryMB: 0 };
  }

  // --- General Cache ---

  clearAll(): void {
    this.scrubbingCache?.clearAll();
    log.debug('Cleared all caches');
  }

  clearVideoTimeTracking(): void {
    this.lastVideoTime.clear();
    log.debug('Cleared video texture cache');
  }

  // --- Video Time Tracking ---

  getLastVideoTime(clipId: string): number | undefined {
    return this.lastVideoTime.get(clipId);
  }

  setLastVideoTime(clipId: string, time: number): void {
    this.lastVideoTime.set(clipId, time);
  }

  // --- RAM Playback Canvas ---

  getRamPlaybackCanvas(): HTMLCanvasElement | null {
    return this.ramPlaybackCanvas;
  }

  getRamPlaybackCtx(): CanvasRenderingContext2D | null {
    return this.ramPlaybackCtx;
  }

  setRamPlaybackCanvas(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): void {
    this.ramPlaybackCanvas = canvas;
    this.ramPlaybackCtx = ctx;
  }

  // --- Direct ScrubbingCache Access (for renderCachedFrame and other complex methods) ---

  getScrubbingCache(): ScrubbingCache | null {
    return this.scrubbingCache;
  }
}
