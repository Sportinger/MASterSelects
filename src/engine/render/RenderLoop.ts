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
  private isPlaying = false;
  private isScrubbing = false;
  private lastRenderTime = 0;

  // Health monitoring - detect frozen render loop
  private lastSuccessfulRender = 0;
  private watchdogTimer: ReturnType<typeof setInterval> | null = null;
  private renderCount = 0;

  private readonly IDLE_TIMEOUT = 1000; // 1s before idle
  private readonly VIDEO_FRAME_TIME = 16.67; // ~60fps target
  private readonly SCRUB_FRAME_TIME = 33; // ~30fps during scrubbing (avoids wasted renders while video seeks)
  private readonly WATCHDOG_INTERVAL = 2000; // Check every 2s
  private readonly WATCHDOG_STALL_THRESHOLD = 3000; // 3s without render = stalled

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
    this.lastSuccessfulRender = performance.now();
    this.isIdle = false;
    this.renderCount = 0;
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

      // Skip stats when idle (but still allow occasional renders for UI updates)
      if (this.isIdle) {
        this.animationId = requestAnimationFrame(loop);
        return;
      }

      // Frame rate limiting for video
      if (this.hasActiveVideo) {
        const timeSinceLastRender = timestamp - this.lastRenderTime;
        if (this.isPlaying) {
          // Playback: ~60fps target
          if (timeSinceLastRender < this.VIDEO_FRAME_TIME) {
            this.animationId = requestAnimationFrame(loop);
            return;
          }
        } else if (this.isScrubbing) {
          // Scrubbing: ~30fps - avoids 4-5 wasted renders per seek cycle
          // Video seeks take ~50-80ms, so rendering at 60fps just re-composites
          // the same cached frame. 30fps catches the seeked frame promptly
          // while halving GPU work.
          if (timeSinceLastRender < this.SCRUB_FRAME_TIME) {
            this.animationId = requestAnimationFrame(loop);
            return;
          }
        }
        this.lastRenderTime = timestamp;
      }

      // Call render callback (unless exporting)
      if (!this.callbacks.isExporting()) {
        try {
          this.callbacks.onRender();
          this.lastSuccessfulRender = timestamp;
          this.renderCount++;
        } catch (e) {
          log.error('Error in render callback', e);
          // Continue loop despite error to prevent freeze
        }
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

    // Start watchdog timer to detect stalled render loops
    this.startWatchdog();
  }

  stop(): void {
    this.isRunning = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    this.stopWatchdog();
  }

  private startWatchdog(): void {
    this.stopWatchdog();
    this.watchdogTimer = setInterval(() => {
      this.checkHealth();
    }, this.WATCHDOG_INTERVAL);
  }

  private stopWatchdog(): void {
    if (this.watchdogTimer !== null) {
      clearInterval(this.watchdogTimer);
      this.watchdogTimer = null;
    }
  }

  /**
   * Check render loop health - detect and recover from stalls.
   * When playing, the render loop should be rendering every frame.
   * If it hasn't rendered for WATCHDOG_STALL_THRESHOLD ms, force a wake-up.
   */
  private checkHealth(): void {
    if (!this.isRunning) return;

    const now = performance.now();
    const timeSinceRender = now - this.lastSuccessfulRender;

    // During recovery, don't interfere
    if (this.callbacks.isRecovering()) return;

    // Check if we're stalled (no render for too long while we should be rendering)
    if (timeSinceRender > this.WATCHDOG_STALL_THRESHOLD) {
      // If we're idle and not playing, this is expected - not a stall
      if (this.isIdle && !this.isPlaying) return;

      log.warn(`Render stall detected: ${timeSinceRender.toFixed(0)}ms since last render (idle=${this.isIdle}, playing=${this.isPlaying})`);

      // Force wake from idle
      this.isIdle = false;
      this.renderRequested = true;
      this.lastActivityTime = now;

      // If the RAF loop itself has died (animationId is null but isRunning is true),
      // restart it
      if (this.animationId === null && this.isRunning) {
        log.warn('RAF loop died - restarting');
        this.stop();
        this.start();
      }
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

  getLastSuccessfulRenderTime(): number {
    return this.lastSuccessfulRender;
  }

  getRenderCount(): number {
    return this.renderCount;
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

  setIsPlaying(playing: boolean): void {
    this.isPlaying = playing;
  }

  setIsScrubbing(scrubbing: boolean): void {
    this.isScrubbing = scrubbing;
    if (scrubbing) {
      // Reset render time so first scrub frame renders immediately
      this.lastRenderTime = 0;
    }
  }
}
