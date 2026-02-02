// Stats tracking for WebGPU engine - FPS, timing, frame drops

import type { EngineStats, DetailedStats, ProfileData } from '../core/types';
import { audioStatusTracker } from '../../services/audioManager';

export class PerformanceStats {
  // FPS tracking
  private frameCount = 0;
  private fps = 0;
  private fpsUpdateTime = 0;

  // Detailed stats
  private detailedStats: DetailedStats = {
    rafGap: 0,
    importTexture: 0,
    renderPass: 0,
    submit: 0,
    total: 0,
    dropsTotal: 0,
    dropsLastSecond: 0,
    dropsThisSecond: 0,
    lastDropReason: 'none',
    lastRafTime: 0,
    decoder: 'none',
  };

  // Last recorded timing (for debugging via getLastTiming())
  private lastTiming: ProfileData | null = null;

  // Frame time buffer (ring buffer for O(1) operations)
  private frameTimeBuffer = new Float32Array(60);
  private frameTimeIndex = 0;
  private frameTimeCount = 0;
  private lastFrameStart = 0;
  private statsCounter = 0;

  // Layer count for stats display
  private lastLayerCount = 0;

  private readonly TARGET_FRAME_TIME = 16.67; // 60fps target

  setDecoder(decoder: DetailedStats['decoder']): void {
    this.detailedStats.decoder = decoder;
  }

  setLayerCount(count: number): void {
    this.lastLayerCount = count;
  }

  recordRafGap(gap: number): void {
    this.detailedStats.rafGap = this.detailedStats.rafGap * 0.9 + gap * 0.1;

    // Detect frame drops - gap > 2x target means missed frames
    if (gap > this.TARGET_FRAME_TIME * 2) {
      const missedFrames = Math.max(1, Math.round(gap / this.TARGET_FRAME_TIME) - 1);
      this.detailedStats.dropsTotal += missedFrames;
      this.detailedStats.dropsThisSecond += missedFrames;
      this.detailedStats.lastDropReason = 'slow_raf';
    }
  }

  recordRenderTiming(timing: ProfileData): void {
    this.lastTiming = timing;

    // Update smoothed stats
    this.detailedStats.importTexture = this.detailedStats.importTexture * 0.9 + timing.importTexture * 0.1;
    this.detailedStats.renderPass = this.detailedStats.renderPass * 0.9 + timing.renderPass * 0.1;
    this.detailedStats.submit = this.detailedStats.submit * 0.9 + timing.submit * 0.1;
    this.detailedStats.total = this.detailedStats.total * 0.9 + timing.total * 0.1;

    // Detect slow render drops
    if (timing.total > this.TARGET_FRAME_TIME) {
      if (timing.importTexture > this.TARGET_FRAME_TIME * 0.5) {
        this.detailedStats.lastDropReason = 'slow_import';
      } else {
        this.detailedStats.lastDropReason = 'slow_render';
      }
    }
  }

  resetPerSecondCounters(): void {
    this.detailedStats.dropsLastSecond = this.detailedStats.dropsThisSecond;
    this.detailedStats.dropsThisSecond = 0;
  }

  updateStats(): void {
    this.frameCount++;
    this.statsCounter++;

    if (this.statsCounter >= 10) {
      this.statsCounter = 0;
      const now = performance.now();

      if (this.lastFrameStart > 0) {
        const frameTime = (now - this.lastFrameStart) / 10;
        this.frameTimeBuffer[this.frameTimeIndex] = frameTime;
        this.frameTimeIndex = (this.frameTimeIndex + 1) % 60;
        if (this.frameTimeCount < 60) this.frameTimeCount++;
      }
      this.lastFrameStart = now;

      // Update FPS every 250ms (4x per second) for more responsive display
      const elapsed = now - this.fpsUpdateTime;
      if (elapsed >= 250) {
        // Scale to frames per second
        this.fps = Math.round((this.frameCount / elapsed) * 1000);
        this.frameCount = 0;
        this.fpsUpdateTime = now;
      }
    }
  }

  getStats(isIdle: boolean): EngineStats {
    let sum = 0;
    for (let i = 0; i < this.frameTimeCount; i++) {
      sum += this.frameTimeBuffer[i];
    }
    const avgFrameTime = this.frameTimeCount > 0 ? sum / this.frameTimeCount : 0;

    return {
      fps: this.fps,
      frameTime: avgFrameTime,
      gpuMemory: 0,
      timing: {
        rafGap: this.detailedStats.rafGap,
        importTexture: this.detailedStats.importTexture,
        renderPass: this.detailedStats.renderPass,
        submit: this.detailedStats.submit,
        total: this.detailedStats.total,
      },
      drops: {
        count: this.detailedStats.dropsTotal,
        lastSecond: this.detailedStats.dropsLastSecond,
        reason: this.detailedStats.lastDropReason,
      },
      layerCount: this.lastLayerCount,
      targetFps: 60,
      decoder: this.detailedStats.decoder,
      audio: audioStatusTracker.getStatus(),
      isIdle,
    };
  }

  /** Get last timing for debugging */
  getLastTiming(): ProfileData | null {
    return this.lastTiming;
  }
}
