import type { EngineStats } from '../types';
import type { PipelineEvent } from './wcPipelineMonitor';
import type { VFPipelineEvent } from './vfPipelineMonitor';

export type PlaybackDebugStats = NonNullable<EngineStats['playback']>;
export type PlaybackPipeline = PlaybackDebugStats['pipeline'];

export interface PlaybackHealthVideoState {
  clipId: string;
  src: string;
  currentTime: number;
  readyState: number;
  seeking: boolean;
  paused: boolean;
  played: number;
  warmingUp: boolean;
  gpuReady: boolean;
}

export interface PlaybackHealthAnomaly {
  type: string;
  timestamp: number;
  clipId?: string;
  detail?: string;
  recovered: boolean;
}

interface FrameCadenceSummary {
  frameEvents: number;
  cadenceFps: number;
  avgFrameGapMs: number;
  p95FrameGapMs: number;
  maxFrameGapMs: number;
}

interface WcTimelineSummary {
  cadence: FrameCadenceSummary;
  stalls: number;
  seeks: number;
  advanceSeeks: number;
  driftCorrections: number;
  queuePressureEvents: number;
  avgDecodeLatencyMs: number;
  avgSeekLatencyMs: number;
  avgQueueDepth: number;
  maxQueueDepth: number;
}

interface VfTimelineSummary {
  cadence: FrameCadenceSummary;
  stalls: number;
  seeks: number;
  advanceSeeks: number;
  driftCorrections: number;
  readyStateDrops: number;
  avgSeekLatencyMs: number;
  avgAudioDriftMs: number;
}

const DEFAULT_WINDOW_MS = 5000;

function round(value: number, precision = 1): number {
  const factor = 10 ** precision;
  return Math.round(value * factor) / factor;
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function max(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return Math.max(...values);
}

function percentile(values: number[], ratio: number): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const index = (sorted.length - 1) * ratio;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sorted[lower];
  }

  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function getNumericDetail(
  detail: Record<string, number | string> | undefined,
  key: string
): number | undefined {
  const value = detail?.[key];
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

export function summarizeFrameCadence(timestamps: number[]): FrameCadenceSummary {
  if (timestamps.length === 0) {
    return {
      frameEvents: 0,
      cadenceFps: 0,
      avgFrameGapMs: 0,
      p95FrameGapMs: 0,
      maxFrameGapMs: 0,
    };
  }

  const gaps: number[] = [];
  for (let index = 1; index < timestamps.length; index++) {
    const gap = timestamps[index] - timestamps[index - 1];
    if (gap > 0) {
      gaps.push(gap);
    }
  }

  const avgFrameGapMs = average(gaps);

  return {
    frameEvents: timestamps.length,
    cadenceFps: avgFrameGapMs > 0 ? round(1000 / avgFrameGapMs, 1) : 0,
    avgFrameGapMs: round(avgFrameGapMs, 1),
    p95FrameGapMs: round(percentile(gaps, 0.95), 1),
    maxFrameGapMs: round(max(gaps), 1),
  };
}

function summarizeWcTimeline(events: PipelineEvent[]): WcTimelineSummary {
  const frameTimes = events
    .filter((event) => event.type === 'decode_output')
    .map((event) => event.t);

  const decodeLatencies: number[] = [];
  const seekDurations: number[] = [];
  const queueDepths: number[] = [];
  let lastFeedTime: number | null = null;

  for (const event of events) {
    if (event.type === 'decode_feed') {
      lastFeedTime = event.t;
      continue;
    }

    if (event.type === 'decode_output') {
      if (lastFeedTime !== null) {
        decodeLatencies.push(event.t - lastFeedTime);
      }
      const queueSize = getNumericDetail(event.detail, 'queueSize');
      if (queueSize !== undefined) {
        queueDepths.push(queueSize);
      }
      continue;
    }

    if (event.type === 'queue_pressure') {
      const queueSize = getNumericDetail(event.detail, 'queueSize');
      if (queueSize !== undefined) {
        queueDepths.push(queueSize);
      }
      continue;
    }

    if (event.type === 'seek_end') {
      const durationMs = getNumericDetail(event.detail, 'durationMs');
      if (durationMs !== undefined) {
        seekDurations.push(durationMs);
      }
    }
  }

  return {
    cadence: summarizeFrameCadence(frameTimes),
    stalls: events.filter((event) => event.type === 'stall').length,
    seeks: events.filter(
      (event) => event.type === 'seek_start' || event.type === 'advance_seek'
    ).length,
    advanceSeeks: events.filter((event) => event.type === 'advance_seek').length,
    driftCorrections: events.filter((event) => event.type === 'drift_correct').length,
    queuePressureEvents: events.filter((event) => event.type === 'queue_pressure').length,
    avgDecodeLatencyMs: round(average(decodeLatencies), 1),
    avgSeekLatencyMs: round(average(seekDurations), 1),
    avgQueueDepth: round(average(queueDepths), 1),
    maxQueueDepth: round(max(queueDepths), 1),
  };
}

function summarizeVfTimeline(events: VFPipelineEvent[]): VfTimelineSummary {
  const frameTimes = events
    .filter((event) => event.type === 'vf_capture')
    .map((event) => event.t);

  const seekDurations: number[] = [];
  const audioDrifts: number[] = [];
  let lastSeekStartTime: number | null = null;

  for (const event of events) {
    if (event.type === 'vf_seek_fast' || event.type === 'vf_seek_precise') {
      lastSeekStartTime = event.t;
      continue;
    }

    if (event.type === 'vf_seek_done') {
      if (lastSeekStartTime !== null) {
        seekDurations.push(event.t - lastSeekStartTime);
        lastSeekStartTime = null;
      }
      continue;
    }

    if (event.type === 'audio_drift') {
      const driftMs = getNumericDetail(event.detail, 'driftMs');
      if (driftMs !== undefined) {
        audioDrifts.push(Math.abs(driftMs));
      }
    }
  }

  return {
    cadence: summarizeFrameCadence(frameTimes),
    stalls: events.filter((event) => event.type === 'vf_stall').length,
    seeks: events.filter(
      (event) => event.type === 'vf_seek_fast' || event.type === 'vf_seek_precise'
    ).length,
    advanceSeeks: 0,
    driftCorrections: events.filter((event) => event.type === 'vf_drift').length,
    readyStateDrops: events.filter((event) => event.type === 'vf_readystate_drop').length,
    avgSeekLatencyMs: round(average(seekDurations), 1),
    avgAudioDriftMs: round(average(audioDrifts), 1),
  };
}

export function mapDecoderToPlaybackPipeline(
  decoder: EngineStats['decoder']
): PlaybackPipeline {
  if (decoder === 'WebCodecs') {
    return 'webcodecs';
  }
  if (decoder === 'HTMLVideo(VF)') {
    return 'vf';
  }
  if (decoder === 'NativeHelper') {
    return 'native';
  }
  if (decoder === 'ParallelDecode') {
    return 'parallel';
  }
  if (decoder.startsWith('HTMLVideo')) {
    return 'html';
  }
  return 'none';
}

function derivePlaybackStatus(stats: Omit<PlaybackDebugStats, 'status'>): PlaybackDebugStats['status'] {
  const severeCadence = stats.p95FrameGapMs >= 85 || stats.maxFrameGapMs >= 140;
  const degradedCadence = stats.p95FrameGapMs >= 50 || stats.avgFrameGapMs >= 40;
  const noReadyFrames = stats.activeVideos > 0 && stats.worstReadyState > 0 && stats.worstReadyState < 2;

  if (
    stats.stalls > 0 ||
    severeCadence ||
    stats.healthAnomalies > 0 ||
    stats.readyStateDrops > 0 ||
    stats.coldVideos > 0 ||
    noReadyFrames
  ) {
    return 'bad';
  }

  if (
    degradedCadence ||
    stats.queuePressureEvents > 30 ||
    stats.seeks >= 3 ||
    stats.driftCorrections > 0 ||
    stats.seekingVideos > 0 ||
    stats.warmingUpVideos > 0
  ) {
    return 'warn';
  }

  return 'ok';
}

export function buildPlaybackDebugStats(params: {
  decoder: EngineStats['decoder'];
  now?: number;
  windowMs?: number;
  wcTimeline?: PipelineEvent[];
  vfTimeline?: VFPipelineEvent[];
  healthVideos?: PlaybackHealthVideoState[];
  healthAnomalies?: PlaybackHealthAnomaly[];
}): PlaybackDebugStats {
  const now = params.now ?? performance.now();
  const windowMs = params.windowMs ?? DEFAULT_WINDOW_MS;
  const pipeline = mapDecoderToPlaybackPipeline(params.decoder);
  const healthVideos = params.healthVideos ?? [];
  const recentHealthAnomalies = (params.healthAnomalies ?? []).filter(
    (anomaly) => anomaly.timestamp >= now - windowMs
  );

  const wcSummary = summarizeWcTimeline(params.wcTimeline ?? []);
  const vfSummary = summarizeVfTimeline(params.vfTimeline ?? []);
  const activeVideos = healthVideos.length;
  const worstReadyState = activeVideos > 0
    ? Math.min(...healthVideos.map((video) => video.readyState))
    : 0;

  const base: Omit<PlaybackDebugStats, 'status'> = {
    windowMs,
    pipeline,
    frameEvents: 0,
    cadenceFps: 0,
    avgFrameGapMs: 0,
    p95FrameGapMs: 0,
    maxFrameGapMs: 0,
    stalls: 0,
    seeks: 0,
    advanceSeeks: 0,
    driftCorrections: 0,
    readyStateDrops: 0,
    queuePressureEvents: 0,
    healthAnomalies: recentHealthAnomalies.length,
    activeVideos,
    seekingVideos: healthVideos.filter((video) => video.seeking).length,
    warmingUpVideos: healthVideos.filter((video) => video.warmingUp).length,
    coldVideos: healthVideos.filter((video) => !video.gpuReady).length,
    worstReadyState,
    lastAnomalyType: recentHealthAnomalies.at(-1)?.type,
  };

  if (pipeline === 'webcodecs') {
    Object.assign(base, wcSummary.cadence, {
      stalls: wcSummary.stalls,
      seeks: wcSummary.seeks,
      advanceSeeks: wcSummary.advanceSeeks,
      driftCorrections: wcSummary.driftCorrections,
      queuePressureEvents: wcSummary.queuePressureEvents,
      avgDecodeLatencyMs: wcSummary.avgDecodeLatencyMs,
      avgSeekLatencyMs: wcSummary.avgSeekLatencyMs,
      avgQueueDepth: wcSummary.avgQueueDepth,
      maxQueueDepth: wcSummary.maxQueueDepth,
    });
  } else if (pipeline === 'vf') {
    Object.assign(base, vfSummary.cadence, {
      stalls: vfSummary.stalls,
      seeks: vfSummary.seeks,
      advanceSeeks: vfSummary.advanceSeeks,
      driftCorrections: vfSummary.driftCorrections,
      readyStateDrops: vfSummary.readyStateDrops,
      avgSeekLatencyMs: vfSummary.avgSeekLatencyMs,
      avgAudioDriftMs: vfSummary.avgAudioDriftMs,
    });
  }

  return {
    ...base,
    status: derivePlaybackStatus(base),
  };
}
