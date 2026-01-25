// LayerCache - Layer caching to avoid rebuilding every frame
// Uses reference equality and frame quantization for change detection

import { Logger } from '../logger';
import type { Layer, TimelineClip, TimelineTrack } from '../../types';
import type { FrameContext } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';

const log = Logger.create('LayerCache');

/**
 * Cache check result
 */
export interface CacheCheckResult {
  useCache: boolean;
  layers: Layer[];
}

/**
 * LayerCache - Manages layer caching for performance
 */
export class LayerCache {
  private cachedLayers: Layer[] = [];
  private cacheValid = false;

  // Change detection state
  private lastPlayheadFrame = -1;
  private lastClipsRef: TimelineClip[] | null = null;
  private lastTracksRef: TimelineTrack[] | null = null;
  private lastActiveCompId: string | null = null;
  private lastIsPlaying = false;
  private lastProxyEnabled = false;

  // Stats for debugging
  private cacheHits = 0;
  private cacheMisses = 0;
  private lastStatsLog = 0;

  /**
   * Invalidate the cache - call when external changes occur
   */
  invalidate(): void {
    this.cacheValid = false;
  }

  /**
   * Check if cached layers can be used
   * Returns the cache check result with either cached layers or empty array
   */
  checkCache(ctx: FrameContext): CacheCheckResult {
    // Check if we can use cached layers
    const clipsChanged = ctx.clips !== this.lastClipsRef;
    const tracksChanged = ctx.tracks !== this.lastTracksRef;
    const frameChanged = ctx.frameNumber !== this.lastPlayheadFrame;
    const compChanged = ctx.activeCompId !== this.lastActiveCompId;
    const playingChanged = ctx.isPlaying !== this.lastIsPlaying;
    const proxyChanged = ctx.proxyEnabled !== this.lastProxyEnabled;

    // Check if any clip has keyframes (need per-frame updates for animation)
    const hasKeyframedClips = ctx.clips.some(c => ctx.hasKeyframes(c.id));

    const needsRebuild = !this.cacheValid ||
      clipsChanged ||
      tracksChanged ||
      compChanged ||
      playingChanged ||
      proxyChanged ||
      (frameChanged && (ctx.isPlaying || ctx.isDraggingPlayhead || hasKeyframedClips));

    // Log cache stats periodically
    this.logStats(ctx.now);

    // Return cached layers if nothing important changed
    if (!needsRebuild && this.cachedLayers.length > 0) {
      this.cacheHits++;
      return { useCache: true, layers: this.cachedLayers };
    }

    this.cacheMisses++;

    // Update change detection state
    this.lastPlayheadFrame = ctx.frameNumber;
    this.lastClipsRef = ctx.clips;
    this.lastTracksRef = ctx.tracks;
    this.lastActiveCompId = ctx.activeCompId;
    this.lastIsPlaying = ctx.isPlaying;
    this.lastProxyEnabled = ctx.proxyEnabled;

    return { useCache: false, layers: [] };
  }

  /**
   * Store layers in cache
   */
  setCachedLayers(layers: Layer[]): void {
    this.cachedLayers = layers;
    this.cacheValid = true;
  }

  /**
   * Get current cache stats
   */
  getStats(): { hits: number; misses: number; hitRate: number } {
    const total = this.cacheHits + this.cacheMisses;
    return {
      hits: this.cacheHits,
      misses: this.cacheMisses,
      hitRate: total > 0 ? this.cacheHits / total : 0,
    };
  }

  /**
   * Log cache stats periodically
   */
  private logStats(now: number): void {
    if (now - this.lastStatsLog > 5000) {
      const total = this.cacheHits + this.cacheMisses;
      if (total > 0) {
        const hitRate = ((this.cacheHits / total) * 100).toFixed(1);
        log.debug(`Hit rate: ${hitRate}% (${this.cacheHits}/${total})`);
      }
      this.cacheHits = 0;
      this.cacheMisses = 0;
      this.lastStatsLog = now;
    }
  }
}
