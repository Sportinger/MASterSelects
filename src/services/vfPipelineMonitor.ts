// VF (HTMLVideo + VideoFrame) Pipeline Monitor
// Dedicated ring-buffer monitor for debugging VF-mode playback and audio.
// Exposed as window.__VF_PIPELINE__ for console inspection.

export type VFPipelineEventType =
  // Frame delivery
  | 'vf_capture'         // VideoFrame created from HTMLVideoElement
  | 'vf_read'            // Frame consumed by LayerCollector
  | 'vf_drop'            // VideoFrame creation failed
  | 'vf_gpu_cold'        // GPU surface not ready (warmup needed)
  | 'vf_gpu_ready'       // GPU surface activated
  // Sync
  | 'vf_play'            // video.play() in VF path
  | 'vf_pause'           // video.pause() in VF path
  | 'vf_seek_fast'       // fastSeek() during drag scrub
  | 'vf_seek_precise'    // currentTime = X (exact seek)
  | 'vf_seek_done'       // seeked event / RVFC callback
  | 'vf_settle_seek'     // precise seek after scrub-stop (HTMLVideo)
  | 'vf_wc_settle_seek'  // precise seek after scrub-stop (WebCodecs)
  | 'vf_drift'           // drift correction during playback
  // Audio
  | 'audio_drift'        // audio element drift from expected time
  | 'audio_drift_correct'// audio re-synced (drift > 300ms)
  | 'audio_status'       // status transition (sync/drift/silent/error)
  | 'audio_master_change'// master audio element changed
  | 'audio_rate_change'  // playbackRate changed
  // Health
  | 'vf_stall'           // frame delivery gap > threshold during play
  | 'vf_readystate_drop';// video.readyState < 2 during playback

export interface VFPipelineEvent {
  type: VFPipelineEventType;
  t: number; // performance.now()
  detail?: Record<string, number | string>;
}

const MAX_EVENTS = 5000;

class VfPipelineMonitor {
  private buffer: VFPipelineEvent[] = [];
  private head = 0;
  private count = 0;

  // Stall detection
  private lastCaptureTime = 0;
  private playing = false;

  // Frame drop detection (frame replaced before read)
  private frameReadSinceLastCapture = true;

  // Throttle vf_read to 1-in-10
  private frameReadCounter = 0;

  // Audio status transition tracking
  private lastAudioStatus = '';

  record(type: VFPipelineEventType, detail?: Record<string, number | string>): void {
    // Throttle vf_read: only record every 10th
    if (type === 'vf_read') {
      this.frameReadSinceLastCapture = true;
      this.frameReadCounter++;
      if (this.frameReadCounter % 10 !== 0) return;
    }

    const event: VFPipelineEvent = { type, t: performance.now(), detail };

    if (this.count < MAX_EVENTS) {
      this.buffer.push(event);
      this.count++;
    } else {
      this.buffer[this.head] = event;
    }
    this.head = (this.head + 1) % MAX_EVENTS;

    // Track play/pause for stall detection
    if (type === 'vf_play') this.playing = true;
    if (type === 'vf_pause') this.playing = false;

    // Stall detection: if playing and capture gap > 100ms (3+ missed frames at 30fps)
    if (type === 'vf_capture') {
      const now = event.t;
      if (this.playing && this.lastCaptureTime > 0) {
        const gap = now - this.lastCaptureTime;
        if (gap > 100) {
          this.record('vf_stall', { gapMs: Math.round(gap) });
        }
      }
      this.lastCaptureTime = now;

      // Frame drop: previous capture was never read
      if (!this.frameReadSinceLastCapture) {
        this.record('vf_drop', { reason: 'not_read' });
      }
      this.frameReadSinceLastCapture = false;
    }

    // Audio status transition detection
    if (type === 'audio_status' && detail?.status) {
      const newStatus = String(detail.status);
      if (this.lastAudioStatus && this.lastAudioStatus !== newStatus) {
        detail.from = this.lastAudioStatus;
      }
      this.lastAudioStatus = newStatus;
    }
  }

  /** Get ordered events (oldest first) */
  private ordered(): VFPipelineEvent[] {
    if (this.count < MAX_EVENTS) return this.buffer.slice();
    return [
      ...this.buffer.slice(this.head),
      ...this.buffer.slice(0, this.head),
    ];
  }

  /** Last N events (default 50) */
  events(n = 50): VFPipelineEvent[] {
    const all = this.ordered();
    return all.slice(-n);
  }

  /** Only stall events */
  stalls(): VFPipelineEvent[] {
    return this.ordered().filter(e => e.type === 'vf_stall');
  }

  /** Only seek events */
  seeks(): VFPipelineEvent[] {
    return this.ordered().filter(e =>
      e.type === 'vf_seek_fast' || e.type === 'vf_seek_precise' || e.type === 'vf_seek_done'
    );
  }

  /** Events within the last N ms (default 5000) */
  timeline(ms = 5000): VFPipelineEvent[] {
    const cutoff = performance.now() - ms;
    return this.ordered().filter(e => e.t >= cutoff);
  }

  /** Only audio events */
  audioEvents(n = 50): VFPipelineEvent[] {
    const all = this.ordered().filter(e => e.type.startsWith('audio_'));
    return all.slice(-n);
  }

  /** Only audio drift events */
  audioDrifts(): VFPipelineEvent[] {
    return this.ordered().filter(e => e.type === 'audio_drift');
  }

  /** Audio events within the last N ms */
  audioTimeline(ms = 5000): VFPipelineEvent[] {
    const cutoff = performance.now() - ms;
    return this.ordered().filter(e => e.type.startsWith('audio_') && e.t >= cutoff);
  }

  /** Show events surrounding each stall (500ms before, 200ms after) */
  stallContext(): { stallAt: number; gapMs: number; before: VFPipelineEvent[]; after: VFPipelineEvent[] }[] {
    const all = this.ordered();
    const stallEvents = all.filter(e => e.type === 'vf_stall');
    return stallEvents.map(stall => {
      const before = all.filter(e => e.t >= stall.t - 500 && e.t < stall.t && e.type !== 'vf_read');
      const after = all.filter(e => e.t > stall.t && e.t <= stall.t + 200 && e.type !== 'vf_read');
      return {
        stallAt: Math.round(stall.t),
        gapMs: Number(stall.detail?.gapMs ?? 0),
        before,
        after,
      };
    });
  }

  /** Aggregate stats */
  stats(): Record<string, number> {
    const all = this.ordered();
    const counts: Record<string, number> = {
      totalEvents: all.length,
      captures: 0,
      reads: 0,
      drops: 0,
      gpuCold: 0,
      gpuReady: 0,
      plays: 0,
      pauses: 0,
      seeksFast: 0,
      seeksPrecise: 0,
      seeksDone: 0,
      driftCorrections: 0,
      stalls: 0,
      readyStateDrops: 0,
      audioDriftEvents: 0,
      audioStatusChanges: 0,
      audioMasterChanges: 0,
      audioRateChanges: 0,
    };

    let stallTotalMs = 0;
    const audioDrifts: number[] = [];
    const seekDurations: number[] = [];

    // Track seek pairs for duration calculation
    let lastSeekStartTime = 0;

    for (const e of all) {
      switch (e.type) {
        case 'vf_capture': counts.captures++; break;
        case 'vf_read': counts.reads++; break;
        case 'vf_drop': counts.drops++; break;
        case 'vf_gpu_cold': counts.gpuCold++; break;
        case 'vf_gpu_ready': counts.gpuReady++; break;
        case 'vf_play': counts.plays++; break;
        case 'vf_pause': counts.pauses++; break;
        case 'vf_seek_fast':
          counts.seeksFast++;
          lastSeekStartTime = e.t;
          break;
        case 'vf_seek_precise':
          counts.seeksPrecise++;
          lastSeekStartTime = e.t;
          break;
        case 'vf_seek_done':
          counts.seeksDone++;
          if (lastSeekStartTime > 0) {
            seekDurations.push(e.t - lastSeekStartTime);
            lastSeekStartTime = 0;
          }
          break;
        case 'vf_drift': counts.driftCorrections++; break;
        case 'vf_stall':
          counts.stalls++;
          if (e.detail?.gapMs !== undefined) stallTotalMs += Number(e.detail.gapMs);
          break;
        case 'vf_readystate_drop': counts.readyStateDrops++; break;
        case 'audio_drift':
          counts.audioDriftEvents++;
          if (e.detail?.driftMs !== undefined) audioDrifts.push(Math.abs(Number(e.detail.driftMs)));
          break;
        case 'audio_status':
          if (e.detail?.from) counts.audioStatusChanges++;
          break;
        case 'audio_master_change': counts.audioMasterChanges++; break;
        case 'audio_rate_change': counts.audioRateChanges++; break;
      }
    }

    const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const max = (arr: number[]) => arr.length ? Math.max(...arr) : 0;

    return {
      ...counts,
      stallTotalMs: Math.round(stallTotalMs),
      avgSeekDurationMs: Math.round(avg(seekDurations) * 100) / 100,
      maxSeekDurationMs: Math.round(max(seekDurations) * 100) / 100,
      avgAudioDriftMs: Math.round(avg(audioDrifts) * 100) / 100,
      maxAudioDriftMs: Math.round(max(audioDrifts) * 100) / 100,
    };
  }

  /** Reset all data */
  reset(): void {
    this.buffer = [];
    this.head = 0;
    this.count = 0;
    this.lastCaptureTime = 0;
    this.playing = false;
    this.frameReadSinceLastCapture = true;
    this.lastAudioStatus = '';
  }
}

// Singleton
export const vfPipelineMonitor = new VfPipelineMonitor();

// Expose on window for console access
if (typeof window !== 'undefined') {
  (window as any).__VF_PIPELINE__ = {
    events: (n?: number) => vfPipelineMonitor.events(n),
    stalls: () => vfPipelineMonitor.stalls(),
    stallContext: () => vfPipelineMonitor.stallContext(),
    seeks: () => vfPipelineMonitor.seeks(),
    stats: () => vfPipelineMonitor.stats(),
    reset: () => vfPipelineMonitor.reset(),
    timeline: (ms?: number) => vfPipelineMonitor.timeline(ms),
    // Audio-specific
    audioEvents: (n?: number) => vfPipelineMonitor.audioEvents(n),
    audioDrifts: () => vfPipelineMonitor.audioDrifts(),
    audioTimeline: (ms?: number) => vfPipelineMonitor.audioTimeline(ms),
  };
}
