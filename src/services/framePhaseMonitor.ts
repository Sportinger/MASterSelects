export interface FramePhaseSample {
  t: number;
  mode: 'live' | 'cached' | 'skipped';
  statsMs: number;
  buildMs: number;
  renderMs: number;
  syncVideoMs: number;
  syncAudioMs: number;
  cacheMs: number;
  totalMs: number;
}

export interface FramePhaseSummary {
  windowMs: number;
  samples: number;
  liveSamples: number;
  cachedSamples: number;
  skippedSamples: number;
  avgTotalMs: number;
  p95TotalMs: number;
  maxTotalMs: number;
  avgStatsMs: number;
  avgBuildMs: number;
  avgRenderMs: number;
  avgSyncVideoMs: number;
  avgSyncAudioMs: number;
  avgCacheMs: number;
  maxBuildMs: number;
  maxRenderMs: number;
  maxSyncVideoMs: number;
  maxSyncAudioMs: number;
  maxCacheMs: number;
}

const MAX_SAMPLES = 2400;
const DEFAULT_WINDOW_MS = 5000;

function round(value: number, precision = 2): number {
  const factor = 10 ** precision;
  return Math.round(value * factor) / factor;
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function max(values: number[]): number {
  if (values.length === 0) return 0;
  return Math.max(...values);
}

/** Like percentile() but expects an already-sorted array (avoids copy+sort). */
function percentileFromSorted(sorted: number[], ratio: number): number {
  if (sorted.length === 0) return 0;

  const index = (sorted.length - 1) * ratio;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sorted[lower];
  }

  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

class FramePhaseMonitor {
  private buffer: FramePhaseSample[] = [];
  private head = 0;
  private count = 0;

  // Cache: ordered() result invalidated on record()
  private _orderedCache: FramePhaseSample[] | null = null;

  // Cache: summary() result invalidated when new samples arrive
  private _summaryCache: FramePhaseSummary | null = null;
  private _summaryCacheKey = '';  // "count:windowMs" to detect staleness

  record(sample: Omit<FramePhaseSample, 't'> & { t?: number }): void {
    const entry: FramePhaseSample = {
      ...sample,
      t: sample.t ?? performance.now(),
    };

    if (this.count < MAX_SAMPLES) {
      this.buffer.push(entry);
      this.count++;
    } else {
      this.buffer[this.head] = entry;
    }

    this.head = (this.head + 1) % MAX_SAMPLES;

    // Invalidate caches on new data
    this._orderedCache = null;
    this._summaryCache = null;
    this._summaryCacheKey = '';
  }

  private ordered(): FramePhaseSample[] {
    if (this._orderedCache) return this._orderedCache;

    let result: FramePhaseSample[];
    if (this.count < MAX_SAMPLES) {
      result = this.buffer.slice();
    } else {
      result = [
        ...this.buffer.slice(this.head),
        ...this.buffer.slice(0, this.head),
      ];
    }

    this._orderedCache = result;
    return result;
  }

  timeline(ms = DEFAULT_WINDOW_MS): FramePhaseSample[] {
    const cutoff = performance.now() - ms;
    return this.ordered().filter((entry) => entry.t >= cutoff);
  }

  summary(ms = DEFAULT_WINDOW_MS): FramePhaseSummary {
    // Return cached summary if no new samples have been added since last call
    const cacheKey = `${this.count}:${ms}`;
    if (this._summaryCache && this._summaryCacheKey === cacheKey) {
      return this._summaryCache;
    }

    const entries = this.timeline(ms);

    // Extract all metric arrays in a single pass instead of 7 separate .map() calls
    const len = entries.length;
    const totals = new Array<number>(len);
    const statsTimes = new Array<number>(len);
    const buildTimes = new Array<number>(len);
    const renderTimes = new Array<number>(len);
    const syncVideoTimes = new Array<number>(len);
    const syncAudioTimes = new Array<number>(len);
    const cacheTimes = new Array<number>(len);
    let liveSamples = 0;
    let cachedSamples = 0;
    let skippedSamples = 0;

    for (let i = 0; i < len; i++) {
      const entry = entries[i];
      totals[i] = entry.totalMs;
      statsTimes[i] = entry.statsMs;
      buildTimes[i] = entry.buildMs;
      renderTimes[i] = entry.renderMs;
      syncVideoTimes[i] = entry.syncVideoMs;
      syncAudioTimes[i] = entry.syncAudioMs;
      cacheTimes[i] = entry.cacheMs;
      if (entry.mode === 'live') liveSamples++;
      else if (entry.mode === 'cached') cachedSamples++;
      else skippedSamples++;
    }

    // Sort totals once and reuse for avg, p95, max (avoids 3 separate sort+copy)
    const sortedTotals = [...totals].sort((a, b) => a - b);

    const result: FramePhaseSummary = {
      windowMs: ms,
      samples: len,
      liveSamples,
      cachedSamples,
      skippedSamples,
      avgTotalMs: round(average(totals)),
      p95TotalMs: round(percentileFromSorted(sortedTotals, 0.95)),
      maxTotalMs: len > 0 ? round(sortedTotals[len - 1]) : 0,
      avgStatsMs: round(average(statsTimes)),
      avgBuildMs: round(average(buildTimes)),
      avgRenderMs: round(average(renderTimes)),
      avgSyncVideoMs: round(average(syncVideoTimes)),
      avgSyncAudioMs: round(average(syncAudioTimes)),
      avgCacheMs: round(average(cacheTimes)),
      maxBuildMs: round(max(buildTimes)),
      maxRenderMs: round(max(renderTimes)),
      maxSyncVideoMs: round(max(syncVideoTimes)),
      maxSyncAudioMs: round(max(syncAudioTimes)),
      maxCacheMs: round(max(cacheTimes)),
    };

    this._summaryCache = result;
    this._summaryCacheKey = cacheKey;
    return result;
  }

  reset(): void {
    this.buffer = [];
    this.head = 0;
    this.count = 0;
    this._orderedCache = null;
    this._summaryCache = null;
    this._summaryCacheKey = '';
  }
}

export const framePhaseMonitor = new FramePhaseMonitor();

if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__FRAME_PHASES__ = {
    timeline: (ms?: number) => framePhaseMonitor.timeline(ms),
    summary: (ms?: number) => framePhaseMonitor.summary(ms),
    reset: () => framePhaseMonitor.reset(),
  };
}
