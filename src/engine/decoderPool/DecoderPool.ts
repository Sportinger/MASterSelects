// DecoderPool — shared video decoder management.
// Multiple clips using the same source file share a single decoder when possible.
// Idle decoders are reused via seek rather than destroyed.

import type {
  DecoderHandle,
  DecoderRequest,
  DecoderPoolConfig,
  DecoderPoolStats,
} from './types.ts';
import { Logger } from '../../services/logger.ts';

const log = Logger.create('DecoderPool');

const DEFAULT_CONFIG: DecoderPoolConfig = {
  maxDecoders: 8,
  shareToleranceFrames: 1,
  idleTimeoutMs: 30_000,
};

export class DecoderPool {
  private decoders = new Map<string, DecoderHandle[]>();
  private config: DecoderPoolConfig;
  private frameCounter = 0;
  private stats: DecoderPoolStats = {
    activeDecoders: 0,
    idleDecoders: 0,
    sharedDecoders: 0,
    totalCreated: 0,
    totalEvicted: 0,
    totalShares: 0,
  };

  constructor(config?: Partial<DecoderPoolConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Acquire a decoder for a media file at a specific time.
   * Returns an existing shared decoder if one is nearby, otherwise creates or reuses.
   */
  acquire(request: DecoderRequest): DecoderHandle | null {
    const { mediaFileId, sourceTime, priority, clipId } = request;
    const tolerance = (request.shareTolerance ?? this.config.shareToleranceFrames) / 30; // frames → seconds

    const handles = this.decoders.get(mediaFileId) ?? [];

    // 1. Check for shareable decoder at same position (± tolerance)
    for (const handle of handles) {
      if (Math.abs(handle.currentTime - sourceTime) <= tolerance) {
        handle.refCount++;
        handle.lastAccessFrame = this.frameCounter;
        handle.priority = this.higherPriority(handle.priority, priority);
        this.stats.totalShares++;
        log.debug('Sharing decoder', { mediaFileId, clipId, sourceTime, handleTime: handle.currentTime });
        return handle;
      }
    }

    // 2. Check for idle decoder on same mediaFileId → seek + reuse
    for (const handle of handles) {
      if (handle.refCount === 0) {
        handle.refCount = 1;
        handle.currentTime = sourceTime;
        handle.lastAccessFrame = this.frameCounter;
        handle.priority = priority;
        log.debug('Reusing idle decoder', { mediaFileId, clipId, sourceTime });
        return handle;
      }
    }

    // 3. Check total count
    const totalActive = this.getTotalDecoderCount();
    if (totalActive >= this.config.maxDecoders) {
      // Try to evict an idle decoder from another mediaFileId
      const evicted = this.evictLRU();
      if (!evicted) {
        log.warn('Cannot acquire decoder: pool exhausted', { mediaFileId, totalActive });
        return null;
      }
    }

    // 4. Create new handle (DOM element creation is external — the handle is a tracking wrapper)
    const handle: DecoderHandle = {
      id: `dec_${mediaFileId}_${Date.now()}`,
      mediaFileId,
      decoderType: 'HTMLVideoElement',
      priority,
      refCount: 1,
      currentTime: sourceTime,
      lastAccessFrame: this.frameCounter,
      isShared: false,
    };

    if (!this.decoders.has(mediaFileId)) {
      this.decoders.set(mediaFileId, []);
    }
    this.decoders.get(mediaFileId)!.push(handle);
    this.stats.totalCreated++;

    log.debug('Created new decoder', { mediaFileId, clipId, sourceTime });
    return handle;
  }

  /**
   * Release a decoder handle (decrements refCount).
   * The decoder stays alive for potential reuse.
   */
  release(handle: DecoderHandle): void {
    if (handle.refCount > 0) {
      handle.refCount--;
    }
  }

  /**
   * Call once per frame to age decoders and update stats.
   */
  tick(): void {
    this.frameCounter++;
    this.updateStats();
  }

  /**
   * Get current pool statistics.
   */
  getStats(): DecoderPoolStats {
    return { ...this.stats };
  }

  /**
   * Destroy all decoders (cleanup on engine teardown).
   */
  destroy(): void {
    this.decoders.clear();
    this.stats = {
      activeDecoders: 0,
      idleDecoders: 0,
      sharedDecoders: 0,
      totalCreated: 0,
      totalEvicted: 0,
      totalShares: 0,
    };
  }

  // === Private ===

  private getTotalDecoderCount(): number {
    let count = 0;
    for (const handles of this.decoders.values()) {
      count += handles.length;
    }
    return count;
  }

  private evictLRU(): boolean {
    let oldestHandle: DecoderHandle | null = null;
    let oldestKey: string | null = null;
    let oldestFrame = Infinity;

    for (const [key, handles] of this.decoders) {
      for (const handle of handles) {
        if (handle.refCount === 0 && handle.lastAccessFrame < oldestFrame) {
          oldestHandle = handle;
          oldestKey = key;
          oldestFrame = handle.lastAccessFrame;
        }
      }
    }

    if (!oldestHandle || !oldestKey) return false;

    const handles = this.decoders.get(oldestKey)!;
    const idx = handles.indexOf(oldestHandle);
    if (idx >= 0) handles.splice(idx, 1);
    if (handles.length === 0) this.decoders.delete(oldestKey);

    this.stats.totalEvicted++;
    log.debug('Evicted idle decoder', { mediaFileId: oldestKey, age: this.frameCounter - oldestFrame });
    return true;
  }

  private updateStats(): void {
    let active = 0;
    let idle = 0;
    let shared = 0;
    for (const handles of this.decoders.values()) {
      for (const h of handles) {
        if (h.refCount > 0) {
          active++;
          if (h.refCount > 1) shared++;
        } else {
          idle++;
        }
      }
    }
    this.stats.activeDecoders = active;
    this.stats.idleDecoders = idle;
    this.stats.sharedDecoders = shared;
  }

  private higherPriority(
    a: DecoderHandle['priority'],
    b: DecoderHandle['priority']
  ): DecoderHandle['priority'] {
    const order = { playback: 0, scrub: 1, preload: 2 };
    return order[a] <= order[b] ? a : b;
  }
}
