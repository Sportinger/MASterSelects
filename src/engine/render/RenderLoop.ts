// Animation loop with idle detection and frame rate limiting

import { Logger } from '../../services/logger';
import type { PerformanceStats } from '../stats/PerformanceStats';

const log = Logger.create('RenderLoop');

export interface RenderLoopCallbacks {
  isRecovering: () => boolean;
  isExporting: () => boolean;
  onRender: () => void;
}

export class RenderLoop {
  private performanceStats: PerformanceStats;
  private callbacks: RenderLoopCallbacks;
  private animationId: number | null = null;
  private isRunning = false;

  // Idle mode
  private lastActivityTime = 0;
  private isIdle = false;
  private renderRequested = false;
  private lastRenderedPlayhead = -1;

  // Frame rate limiting
  private hasActiveVideo = false;
  private lastRenderTime = 0;

  private readonly IDLE_TIMEOUT = 1000; // 1s before idle
  private readonly VIDEO_FRAME_TIME = 16.67; // ~60fps target

  private lastFpsReset = 0;

  constructor(
    performanceStats: PerformanceStats,
    callbacks: RenderLoopCallbacks
  ) {
    this.performanceStats = performanceStats;
    this.callbacks = callbacks;
  }

  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.lastActivityTime = performance.now();
    this.isIdle = false;
    log.info('Starting');

    let lastTimestamp = 0;

    const loop = (timestamp: number) => {
      if (!this.isRunning) return;

      const rafGap = lastTimestamp > 0 ? timestamp - lastTimestamp : 0;
      lastTimestamp = timestamp;

      // Idle detection
      const timeSinceActivity = timestamp - this.lastActivityTime;
      if (!this.isIdle && !this.renderRequested && timeSinceActivity > this.IDLE_TIMEOUT) {
        this.isIdle = true;
        log.debug('Entering idle mode');
      }

      if (this.isIdle && this.renderRequested) {
        this.isIdle = false;
        log.debug('Waking from idle');
      }

      this.renderRequested = false;

      // Skip during device recovery
      if (this.callbacks.isRecovering()) {
        this.animationId = requestAnimationFrame(loop);
        return;
      }

      // Call render callback (unless exporting)
      if (!this.callbacks.isExporting()) {
        try {
          this.callbacks.onRender();
        } catch (e) {
          log.error('Error in render callback', e);
          // Continue loop despite error to prevent freeze
        }
      }

      // Skip stats when idle
      if (this.isIdle) {
        this.animationId = requestAnimationFrame(loop);
        return;
      }

      // Frame rate limiting for video
      if (this.hasActiveVideo) {
        const timeSinceLastRender = timestamp - this.lastRenderTime;
        if (timeSinceLastRender < this.VIDEO_FRAME_TIME) {
          this.animationId = requestAnimationFrame(loop);
          return;
        }
        this.lastRenderTime = timestamp;
      }

      // Record RAF gap for stats
      if (lastTimestamp > 0) {
        this.performanceStats.recordRafGap(rafGap);
      }

      // Reset per-second counters
      if (timestamp - this.lastFpsReset >= 1000) {
        this.performanceStats.resetPerSecondCounters();
        this.lastFpsReset = timestamp;
      }

      this.animationId = requestAnimationFrame(loop);
    };

    this.animationId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.isRunning = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  requestRender(): void {
    this.lastActivityTime = performance.now();
    this.renderRequested = true;
    if (this.isIdle) {
      this.isIdle = false;
    }
  }

  getIsIdle(): boolean {
    return this.isIdle;
  }

  updatePlayheadTracking(playhead: number): boolean {
    const changed = Math.abs(playhead - this.lastRenderedPlayhead) > 0.0001;
    if (changed) {
      this.lastRenderedPlayhead = playhead;
      this.requestRender();
    }
    return changed;
  }

  setHasActiveVideo(hasVideo: boolean): void {
    this.hasActiveVideo = hasVideo;
  }
}
