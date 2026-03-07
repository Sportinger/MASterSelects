// PlaybackHealthMonitor - Detects playback anomalies, logs diagnostics, auto-recovers
//
// Anomaly types:
//   FRAME_STALL      - video.currentTime unchanged for ~1.5s during playback
//   WARMUP_STUCK     - video in warmingUpVideos for > 3s
//   RVFC_ORPHANED    - RVFC handle for clip not in current timeline
//   SEEK_STUCK       - video.seeking === true for > 2s
//   READYSTATE_DROP  - video.readyState < 2 during playback (not seeking)
//   GPU_SURFACE_COLD - playing video not in videoGpuReady
//   RENDER_STALL     - no render for > 3s while playing
//   HIGH_DROP_RATE   - > 10 drops/second from engine stats

import { Logger } from './logger';
import { engine } from '../engine/WebGPUEngine';
import { layerBuilder } from './layerBuilder';
import { useTimelineStore } from '../stores/timeline';

const log = Logger.create('PlaybackHealth');

// --- Types ---

type AnomalyType =
  | 'FRAME_STALL'
  | 'WARMUP_STUCK'
  | 'RVFC_ORPHANED'
  | 'SEEK_STUCK'
  | 'READYSTATE_DROP'
  | 'GPU_SURFACE_COLD'
  | 'RENDER_STALL'
  | 'HIGH_DROP_RATE';

interface AnomalyEvent {
  type: AnomalyType;
  timestamp: number;
  clipId?: string;
  detail?: string;
  recovered: boolean;
}

interface VideoTimeTracker {
  lastTime: number;
  staleCount: number;
}

// --- Constants ---

const POLL_INTERVAL = 500;
const MAX_ANOMALY_LOG = 200;
const COOLDOWN_MS = 5000;
const FRAME_STALL_POLLS = 3;        // 3 polls × 500ms = 1.5s
const WARMUP_STUCK_MS = 3000;
const SEEK_STUCK_MS = 2000;
const RENDER_STALL_MS = 3000;
const HIGH_DROP_THRESHOLD = 10;

// --- Service ---

export class PlaybackHealthMonitor {
  private intervalId: number | null = null;
  private startTime = 0;

  // Per-video tracking
  private videoTimeTracker = new Map<string, VideoTimeTracker>();
  private warmupStartTimes = new WeakMap<HTMLVideoElement, number>();
  private seekStartTimes = new Map<string, number>();

  // Anomaly log (ring buffer)
  private anomalyLog: AnomalyEvent[] = [];
  private anomalyCounts: Record<AnomalyType, number> = {
    FRAME_STALL: 0,
    WARMUP_STUCK: 0,
    RVFC_ORPHANED: 0,
    SEEK_STUCK: 0,
    READYSTATE_DROP: 0,
    GPU_SURFACE_COLD: 0,
    RENDER_STALL: 0,
    HIGH_DROP_RATE: 0,
  };
  private lastAnomalyTime: Partial<Record<AnomalyType, number>> = {};

  start(): void {
    if (this.intervalId !== null) return;
    this.startTime = performance.now();
    this.intervalId = -1; // sentinel to indicate "started"
    this.scheduleNextCheck();
    this.exposeConsoleAPI();
    log.info('Health monitor started');
  }

  stop(): void {
    if (this.intervalId !== null && this.intervalId !== -1) {
      if (typeof cancelIdleCallback !== 'undefined') {
        cancelIdleCallback(this.intervalId);
      } else {
        clearTimeout(this.intervalId);
      }
    }
    this.intervalId = null;
    log.info('Health monitor stopped');
  }

  private scheduleNextCheck(): void {
    if (typeof requestIdleCallback !== 'undefined') {
      this.intervalId = requestIdleCallback(() => {
        this.checkHealth();
        if (this.intervalId !== null) {
          this.scheduleNextCheck();
        }
      }, { timeout: POLL_INTERVAL }) as unknown as number;
    } else {
      this.intervalId = setTimeout(() => {
        this.checkHealth();
        if (this.intervalId !== null) {
          this.scheduleNextCheck();
        }
      }, POLL_INTERVAL) as unknown as number;
    }
  }

  // --- Main check loop ---

  private checkHealth(): void {
    const { isPlaying, clips, playheadPosition } = useTimelineStore.getState();
    const now = performance.now();

    // Gather active video clips at playhead
    const clipsAtTime = clips.filter(
      (c) => playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
    );
    const videoClips = clipsAtTime.filter((c) => c.source?.videoElement);

    const vsm = layerBuilder.getVideoSyncManager();

    // 1. FRAME_STALL
    if (isPlaying) {
      for (const clip of videoClips) {
        const video = clip.source!.videoElement!;
        const tracker = this.videoTimeTracker.get(clip.id);
        if (tracker) {
          if (Math.abs(video.currentTime - tracker.lastTime) < 0.001) {
            tracker.staleCount++;
            if (tracker.staleCount >= FRAME_STALL_POLLS) {
              if (this.recordAnomaly('FRAME_STALL', clip.id, `currentTime stuck at ${video.currentTime.toFixed(3)}`)) {
                this.recoverFrameStall(video);
              }
              tracker.staleCount = 0;
            }
          } else {
            tracker.lastTime = video.currentTime;
            tracker.staleCount = 0;
          }
        } else {
          this.videoTimeTracker.set(clip.id, { lastTime: video.currentTime, staleCount: 0 });
        }
      }
    }

    // 2. WARMUP_STUCK
    for (const clip of videoClips) {
      const video = clip.source!.videoElement!;
      if (vsm.isVideoWarmingUp(video)) {
        const warmupStart = this.warmupStartTimes.get(video);
        if (warmupStart) {
          if (now - warmupStart > WARMUP_STUCK_MS) {
            if (this.recordAnomaly('WARMUP_STUCK', clip.id, `warmup for ${((now - warmupStart) / 1000).toFixed(1)}s`)) {
              vsm.clearWarmupState(video);
            }
            this.warmupStartTimes.delete(video);
          }
        } else {
          this.warmupStartTimes.set(video, now);
        }
      } else {
        this.warmupStartTimes.delete(video);
      }
    }

    // 3. RVFC_ORPHANED
    const activeRvfcClipIds = vsm.getActiveRvfcClipIds();
    const currentClipIds = new Set(clips.map((c) => c.id));
    for (const clipId of activeRvfcClipIds) {
      if (!currentClipIds.has(clipId)) {
        if (this.recordAnomaly('RVFC_ORPHANED', clipId, 'RVFC handle for clip not in timeline')) {
          vsm.cancelRvfcHandle(clipId);
        }
      }
    }

    // 4. SEEK_STUCK
    for (const clip of videoClips) {
      const video = clip.source!.videoElement!;
      if (video.seeking) {
        const seekStart = this.seekStartTimes.get(clip.id);
        if (seekStart) {
          if (now - seekStart > SEEK_STUCK_MS) {
            if (this.recordAnomaly('SEEK_STUCK', clip.id, `seeking for ${((now - seekStart) / 1000).toFixed(1)}s`)) {
              this.recoverSeekStuck(video);
            }
            this.seekStartTimes.delete(clip.id);
          }
        } else {
          this.seekStartTimes.set(clip.id, now);
        }
      } else {
        this.seekStartTimes.delete(clip.id);
      }
    }

    // 5. READYSTATE_DROP
    if (isPlaying) {
      for (const clip of videoClips) {
        const video = clip.source!.videoElement!;
        if (video.readyState < 2 && !video.seeking) {
          this.recordAnomaly('READYSTATE_DROP', clip.id, `readyState=${video.readyState}`);
        }
      }
    }

    // 6. GPU_SURFACE_COLD
    if (isPlaying) {
      const lc = engine.getLayerCollector();
      if (lc) {
        for (const clip of videoClips) {
          const video = clip.source!.videoElement!;
          // Skip if video is currently warming up — warmup will handle GPU readiness
          if (vsm.isVideoWarmingUp(video)) continue;
          if (!video.paused && !lc.isVideoGpuReady(video)) {
            if (this.recordAnomaly('GPU_SURFACE_COLD', clip.id, 'playing video not GPU-ready')) {
              lc.resetVideoGpuReady(video);
            }
          }
        }
      }
    }

    // 7. RENDER_STALL
    if (isPlaying) {
      const rl = engine.getRenderLoop();
      if (rl) {
        const lastRender = rl.getLastSuccessfulRenderTime();
        if (lastRender > 0 && now - lastRender > RENDER_STALL_MS) {
          if (this.recordAnomaly('RENDER_STALL', undefined, `no render for ${((now - lastRender) / 1000).toFixed(1)}s`)) {
            engine.requestRender();
          }
        }
      }
    }

    // 8. HIGH_DROP_RATE
    const stats = engine.getStats();
    if (stats.drops && stats.drops.lastSecond > HIGH_DROP_THRESHOLD) {
      this.recordAnomaly('HIGH_DROP_RATE', undefined, `${stats.drops.lastSecond} drops/sec`);
    }

    // Cleanup stale tracker entries for clips no longer in timeline
    const currentClipIdSet = new Set(clips.map((c) => c.id));
    for (const id of this.videoTimeTracker.keys()) {
      if (!currentClipIdSet.has(id)) this.videoTimeTracker.delete(id);
    }
    for (const id of this.seekStartTimes.keys()) {
      if (!currentClipIdSet.has(id)) this.seekStartTimes.delete(id);
    }
  }

  // --- Anomaly recording with cooldown ---

  private recordAnomaly(type: AnomalyType, clipId?: string, detail?: string): boolean {
    const now = performance.now();
    const lastTime = this.lastAnomalyTime[type] ?? 0;
    if (now - lastTime < COOLDOWN_MS) return false;

    this.lastAnomalyTime[type] = now;
    this.anomalyCounts[type]++;

    const event: AnomalyEvent = {
      type,
      timestamp: now,
      clipId,
      detail,
      recovered: type !== 'HIGH_DROP_RATE' && type !== 'READYSTATE_DROP',
    };

    this.anomalyLog.push(event);
    if (this.anomalyLog.length > MAX_ANOMALY_LOG) {
      this.anomalyLog.shift();
    }

    log.warn(`[${type}]${clipId ? ` clip=${clipId}` : ''} ${detail || ''}`);
    return true;
  }

  // --- Recovery methods ---

  private recoverFrameStall(video: HTMLVideoElement): void {
    const time = video.currentTime;
    const dur = video.duration;
    // EOF stall: seeking past end is futile — clamp back
    if (isFinite(dur) && time >= dur - 0.002) {
      video.currentTime = dur - 0.001;
      engine.requestRender();
      return;
    }

    const { isPlaying } = useTimelineStore.getState();
    if (isPlaying) {
      // During playback: just nudge time to unstick decoder.
      // Do NOT pause — that races with AudioSyncHandler and causes
      // "play() interrupted by pause()" errors.
      video.currentTime = time + 0.001;
      engine.requestRender();
    } else {
      // When paused: play/pause cycle to force GPU decode
      video.play().then(() => {
        video.pause();
        video.currentTime = time;
        engine.requestRender();
      }).catch(() => {
        video.currentTime = time + 0.001;
        engine.requestRender();
      });
    }
  }

  private recoverSeekStuck(video: HTMLVideoElement): void {
    const time = video.currentTime;
    video.currentTime = time;
    engine.requestRender();
  }

  softReset(): void {
    const { clips, playheadPosition } = useTimelineStore.getState();
    const vsm = layerBuilder.getVideoSyncManager();
    const lc = engine.getLayerCollector();

    const videoClips = clips.filter(
      (c) =>
        c.source?.videoElement &&
        playheadPosition >= c.startTime &&
        playheadPosition < c.startTime + c.duration
    );

    // Force decode all
    for (const clip of videoClips) {
      const video = clip.source!.videoElement!;
      vsm.clearWarmupState(video);
      if (lc) lc.resetVideoGpuReady(video);
    }

    // Clear orphaned RVFC handles
    const rvfcIds = vsm.getActiveRvfcClipIds();
    const currentIds = new Set(clips.map((c) => c.id));
    for (const id of rvfcIds) {
      if (!currentIds.has(id)) vsm.cancelRvfcHandle(id);
    }

    engine.requestRender();
    log.info('Soft reset completed');
  }

  forceDecodeAll(): void {
    const { clips, playheadPosition } = useTimelineStore.getState();
    const lc = engine.getLayerCollector();

    const videoClips = clips.filter(
      (c) =>
        c.source?.videoElement &&
        playheadPosition >= c.startTime &&
        playheadPosition < c.startTime + c.duration
    );

    for (const clip of videoClips) {
      const video = clip.source!.videoElement!;
      if (lc) lc.resetVideoGpuReady(video);
    }

    engine.requestRender();
    log.info('Force decode all completed');
  }

  clearWarmups(): void {
    const { clips, playheadPosition } = useTimelineStore.getState();
    const vsm = layerBuilder.getVideoSyncManager();

    const videoClips = clips.filter(
      (c) =>
        c.source?.videoElement &&
        playheadPosition >= c.startTime &&
        playheadPosition < c.startTime + c.duration
    );

    for (const clip of videoClips) {
      vsm.clearWarmupState(clip.source!.videoElement!);
    }

    // WeakMap doesn't support clear() — entries GC naturally when video elements are removed
    this.warmupStartTimes = new WeakMap();
    log.info('Warmups cleared');
  }

  clearOrphans(): void {
    const { clips } = useTimelineStore.getState();
    const vsm = layerBuilder.getVideoSyncManager();
    const currentIds = new Set(clips.map((c) => c.id));

    for (const id of vsm.getActiveRvfcClipIds()) {
      if (!currentIds.has(id)) vsm.cancelRvfcHandle(id);
    }

    log.info('Orphaned handles cleared');
  }

  reset(): void {
    this.videoTimeTracker.clear();
    this.warmupStartTimes = new WeakMap();
    this.seekStartTimes.clear();
    this.anomalyLog.length = 0;
    for (const key of Object.keys(this.anomalyCounts) as AnomalyType[]) {
      this.anomalyCounts[key] = 0;
    }
    this.lastAnomalyTime = {};
    log.info('Health monitor reset');
  }

  // --- Console API ---

  snapshot(): {
    status: string;
    uptime: number;
    anomalyCounts: Record<AnomalyType, number>;
    videoStates: Array<{ clipId: string; currentTime: number; readyState: number; seeking: boolean; paused: boolean }>;
  } {
    const { clips, playheadPosition, isPlaying } = useTimelineStore.getState();
    const videoClips = clips.filter(
      (c) =>
        c.source?.videoElement &&
        playheadPosition >= c.startTime &&
        playheadPosition < c.startTime + c.duration
    );

    const totalAnomalies = Object.values(this.anomalyCounts).reduce((a, b) => a + b, 0);
    const status = totalAnomalies === 0 ? 'healthy' : isPlaying ? 'degraded' : 'idle-with-issues';

    return {
      status,
      uptime: Math.round((performance.now() - this.startTime) / 1000),
      anomalyCounts: { ...this.anomalyCounts },
      videoStates: videoClips.map((c) => {
        const v = c.source!.videoElement!;
        return {
          clipId: c.id,
          currentTime: v.currentTime,
          readyState: v.readyState,
          seeking: v.seeking,
          paused: v.paused,
        };
      }),
    };
  }

  anomalies(filterType?: AnomalyType): AnomalyEvent[] {
    if (filterType) return this.anomalyLog.filter((e) => e.type === filterType);
    return [...this.anomalyLog];
  }

  videos(): Array<{
    clipId: string;
    src: string;
    currentTime: number;
    readyState: number;
    seeking: boolean;
    paused: boolean;
    played: number;
    warmingUp: boolean;
    gpuReady: boolean;
  }> {
    const { clips, playheadPosition } = useTimelineStore.getState();
    const vsm = layerBuilder.getVideoSyncManager();
    const lc = engine.getLayerCollector();

    return clips
      .filter(
        (c) =>
          c.source?.videoElement &&
          playheadPosition >= c.startTime &&
          playheadPosition < c.startTime + c.duration
      )
      .map((c) => {
        const v = c.source!.videoElement!;
        return {
          clipId: c.id,
          src: v.src?.split('/').pop() || v.currentSrc?.split('/').pop() || '(blob)',
          currentTime: v.currentTime,
          readyState: v.readyState,
          seeking: v.seeking,
          paused: v.paused,
          played: v.played.length,
          warmingUp: vsm.isVideoWarmingUp(v),
          gpuReady: lc?.isVideoGpuReady(v) ?? false,
        };
      });
  }

  private exposeConsoleAPI(): void {
    const monitor = this;
    (window as any).__PLAYBACK_HEALTH__ = {
      snapshot: () => monitor.snapshot(),
      anomalies: (type?: AnomalyType) => monitor.anomalies(type),
      videos: () => monitor.videos(),
      recover: {
        softReset: () => monitor.softReset(),
        forceDecodeAll: () => monitor.forceDecodeAll(),
        clearWarmups: () => monitor.clearWarmups(),
        clearOrphans: () => monitor.clearOrphans(),
      },
      reset: () => monitor.reset(),
    };
  }
}

// --- HMR Singleton ---

const hot = typeof import.meta !== 'undefined'
  ? (import.meta as { hot?: { data: Record<string, unknown> } }).hot
  : undefined;

let instance: PlaybackHealthMonitor;
if (hot?.data?.healthMonitor) {
  instance = hot.data.healthMonitor as PlaybackHealthMonitor;
} else {
  instance = new PlaybackHealthMonitor();
  if (hot) hot.data.healthMonitor = instance;
}

export const playbackHealthMonitor = instance;
