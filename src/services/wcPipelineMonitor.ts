// WebCodecs Pipeline Monitor
// Lightweight ring-buffer monitor for all WebCodecs decode pipeline events.
// Exposed as window.__WC_PIPELINE__ for console inspection.

export type PipelineEventType =
  | 'decode_feed'
  | 'decode_output'
  | 'frame_read'
  | 'frame_drop'
  | 'seek_start'
  | 'seek_end'
  | 'drift_correct'
  | 'queue_pressure'
  | 'stall'
  | 'rAF_gap'
  | 'play'
  | 'pause';

export interface PipelineEvent {
  type: PipelineEventType;
  t: number; // performance.now()
  detail?: Record<string, number | string>;
}

const MAX_EVENTS = 5000;

class WcPipelineMonitor {
  private buffer: PipelineEvent[] = [];
  private head = 0;
  private count = 0;

  // Stall detection state
  private lastOutputTime = 0;
  private playing = false;

  // Frame drop detection
  private frameReadSinceLastOutput = true;

  // Throttle frame_read to 1-in-10 to save buffer space
  private frameReadCounter = 0;

  record(type: PipelineEventType, detail?: Record<string, number | string>): void {
    // Throttle frame_read: only record every 10th to save buffer for important events
    if (type === 'frame_read') {
      this.frameReadSinceLastOutput = true;
      this.frameReadCounter++;
      if (this.frameReadCounter % 10 !== 0) return;
    }

    const event: PipelineEvent = { type, t: performance.now(), detail };

    if (this.count < MAX_EVENTS) {
      this.buffer.push(event);
      this.count++;
    } else {
      this.buffer[this.head] = event;
    }
    this.head = (this.head + 1) % MAX_EVENTS;

    // Track play/pause for stall detection
    if (type === 'play') this.playing = true;
    if (type === 'pause') this.playing = false;

    // Stall detection: if playing and output gap > 100ms
    // (30fps = ~33ms per frame, so 100ms = 3+ missed frames = real freeze)
    if (type === 'decode_output') {
      const now = event.t;
      if (this.playing && this.lastOutputTime > 0) {
        const gap = now - this.lastOutputTime;
        if (gap > 100) {
          // Record stall inline (won't recurse because type !== decode_output)
          this.record('stall', { gapMs: Math.round(gap) });
        }
      }
      this.lastOutputTime = now;

      // Frame drop: previous frame was never read
      if (!this.frameReadSinceLastOutput) {
        this.record('frame_drop');
      }
      this.frameReadSinceLastOutput = false;
    }
  }

  /** Get ordered events (oldest first) */
  private ordered(): PipelineEvent[] {
    if (this.count < MAX_EVENTS) return this.buffer.slice();
    // Ring buffer is full — reorder from oldest to newest
    return [
      ...this.buffer.slice(this.head),
      ...this.buffer.slice(0, this.head),
    ];
  }

  /** Last N events (default 50) */
  events(n = 50): PipelineEvent[] {
    const all = this.ordered();
    return all.slice(-n);
  }

  /** Only stall events */
  stalls(): PipelineEvent[] {
    return this.ordered().filter(e => e.type === 'stall');
  }

  /** Only seek events */
  seeks(): PipelineEvent[] {
    return this.ordered().filter(e => e.type === 'seek_start' || e.type === 'seek_end');
  }

  /** Events within the last N ms (default 5000) */
  timeline(ms = 5000): PipelineEvent[] {
    const cutoff = performance.now() - ms;
    return this.ordered().filter(e => e.t >= cutoff);
  }

  /** Aggregate stats */
  stats(): Record<string, number> {
    const all = this.ordered();
    const counts: Record<string, number> = {
      totalEvents: all.length,
      decodeFed: 0,
      decodeOutput: 0,
      frameReads: 0,
      frameDrops: 0,
      seeks: 0,
      stalls: 0,
      driftCorrections: 0,
      queuePressure: 0,
    };

    let decodeLats: number[] = [];
    let seekDurations: number[] = [];
    let queueDepths: number[] = [];
    let stallTotalMs = 0;

    for (const e of all) {
      switch (e.type) {
        case 'decode_feed': counts.decodeFed++; break;
        case 'decode_output':
          counts.decodeOutput++;
          if (e.detail?.queueSize !== undefined) {
            queueDepths.push(Number(e.detail.queueSize));
          }
          break;
        case 'frame_read': counts.frameReads++; break;
        case 'frame_drop': counts.frameDrops++; break;
        case 'seek_start': counts.seeks++; break;
        case 'seek_end':
          if (e.detail?.durationMs !== undefined) {
            seekDurations.push(Number(e.detail.durationMs));
          }
          break;
        case 'stall':
          counts.stalls++;
          if (e.detail?.gapMs !== undefined) {
            stallTotalMs += Number(e.detail.gapMs);
          }
          break;
        case 'drift_correct': counts.driftCorrections++; break;
        case 'queue_pressure':
          counts.queuePressure++;
          if (e.detail?.queueSize !== undefined) {
            queueDepths.push(Number(e.detail.queueSize));
          }
          break;
      }
    }

    // Compute decode latency from consecutive feed→output pairs
    let lastFeedTime = 0;
    for (const e of all) {
      if (e.type === 'decode_feed') lastFeedTime = e.t;
      if (e.type === 'decode_output' && lastFeedTime > 0) {
        decodeLats.push(e.t - lastFeedTime);
      }
    }

    const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const max = (arr: number[]) => arr.length ? Math.max(...arr) : 0;

    return {
      ...counts,
      avgDecodeLat: Math.round(avg(decodeLats) * 100) / 100,
      maxDecodeLat: Math.round(max(decodeLats) * 100) / 100,
      avgSeekDuration: Math.round(avg(seekDurations) * 100) / 100,
      maxSeekDuration: Math.round(max(seekDurations) * 100) / 100,
      avgQueueDepth: Math.round(avg(queueDepths) * 100) / 100,
      maxQueueDepth: max(queueDepths),
      stallCount: counts.stalls,
      stallTotalMs: Math.round(stallTotalMs),
    };
  }

  /** Show events surrounding each stall (500ms before, 200ms after) */
  stallContext(): { stallAt: number; gapMs: number; before: PipelineEvent[]; after: PipelineEvent[] }[] {
    const all = this.ordered();
    const stalls = all.filter(e => e.type === 'stall');
    return stalls.map(stall => {
      const before = all.filter(e => e.t >= stall.t - 500 && e.t < stall.t && e.type !== 'frame_read');
      const after = all.filter(e => e.t > stall.t && e.t <= stall.t + 200 && e.type !== 'frame_read');
      return {
        stallAt: Math.round(stall.t),
        gapMs: Number(stall.detail?.gapMs ?? 0),
        before,
        after,
      };
    });
  }

  /** Reset all data */
  reset(): void {
    this.buffer = [];
    this.head = 0;
    this.count = 0;
    this.lastOutputTime = 0;
    this.playing = false;
    this.frameReadSinceLastOutput = true;
  }
}

// Singleton
export const wcPipelineMonitor = new WcPipelineMonitor();

// Expose on window for console access
if (typeof window !== 'undefined') {
  (window as any).__WC_PIPELINE__ = {
    events: (n?: number) => wcPipelineMonitor.events(n),
    stalls: () => wcPipelineMonitor.stalls(),
    stallContext: () => wcPipelineMonitor.stallContext(),
    seeks: () => wcPipelineMonitor.seeks(),
    stats: () => wcPipelineMonitor.stats(),
    reset: () => wcPipelineMonitor.reset(),
    timeline: (ms?: number) => wcPipelineMonitor.timeline(ms),
  };
}
