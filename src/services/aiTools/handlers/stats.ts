import { useEngineStore } from '../../../stores/engineStore';
import { engine } from '../../../engine/WebGPUEngine';
import { Logger } from '../../logger';
import { getPlaybackDebugStats } from '../../playbackDebugSnapshot';
import { buildPlaybackDebugStats } from '../../playbackDebugStats';
import { playbackHealthMonitor } from '../../playbackHealthMonitor';
import { vfPipelineMonitor } from '../../vfPipelineMonitor';
import { wcPipelineMonitor } from '../../wcPipelineMonitor';
import type { ToolResult } from '../types';

const DEFAULT_PLAYBACK_WINDOW_MS = 5000;
const MAX_TRACE_WINDOW_MS = 120000;
const MAX_TRACE_EVENTS = 2000;

function roundOptional(v: number | undefined): number | undefined {
  return typeof v === 'number' ? round(v) : undefined;
}

function cloneCounts(counts: Record<string, number> | undefined): Record<string, number> | undefined {
  return counts ? { ...counts } : undefined;
}

function serializePlayback(playback: ReturnType<typeof getPlaybackDebugStats>): Record<string, unknown> {
  return {
    status: playback.status,
    windowMs: playback.windowMs,
    pipeline: playback.pipeline,
    frameEvents: playback.frameEvents,
    cadenceFps: round(playback.cadenceFps),
    avgFrameGapMs: round(playback.avgFrameGapMs),
    p95FrameGapMs: round(playback.p95FrameGapMs),
    maxFrameGapMs: round(playback.maxFrameGapMs),
    previewFrames: playback.previewFrames,
    previewUpdates: playback.previewUpdates,
    previewRenderFps: round(playback.previewRenderFps),
    previewUpdateFps: round(playback.previewUpdateFps),
    avgPreviewRenderGapMs: round(playback.avgPreviewRenderGapMs),
    p95PreviewRenderGapMs: round(playback.p95PreviewRenderGapMs),
    maxPreviewRenderGapMs: round(playback.maxPreviewRenderGapMs),
    avgPreviewUpdateGapMs: round(playback.avgPreviewUpdateGapMs),
    p95PreviewUpdateGapMs: round(playback.p95PreviewUpdateGapMs),
    maxPreviewUpdateGapMs: round(playback.maxPreviewUpdateGapMs),
    stalePreviewFrames: playback.stalePreviewFrames,
    stalePreviewWhileTargetMoved: playback.stalePreviewWhileTargetMoved,
    previewFreezeEvents: playback.previewFreezeEvents,
    previewFreezeFrames: playback.previewFreezeFrames,
    longestPreviewFreezeFrames: playback.longestPreviewFreezeFrames,
    longestPreviewFreezeMs: round(playback.longestPreviewFreezeMs),
    lastPreviewFreezePath: playback.lastPreviewFreezePath,
    lastPreviewFreezeClipId: playback.lastPreviewFreezeClipId,
    lastPreviewFreezeDurationMs: roundOptional(playback.lastPreviewFreezeDurationMs),
    previewPathCounts: cloneCounts(playback.previewPathCounts),
    scrubPathCounts: cloneCounts(playback.scrubPathCounts),
    avgPreviewDriftMs: round(playback.avgPreviewDriftMs),
    maxPreviewDriftMs: round(playback.maxPreviewDriftMs),
    stalls: playback.stalls,
    seeks: playback.seeks,
    advanceSeeks: playback.advanceSeeks,
    driftCorrections: playback.driftCorrections,
    readyStateDrops: playback.readyStateDrops,
    queuePressureEvents: playback.queuePressureEvents,
    healthAnomalies: playback.healthAnomalies,
    activeVideos: playback.activeVideos,
    playingVideos: playback.playingVideos,
    seekingVideos: playback.seekingVideos,
    warmingUpVideos: playback.warmingUpVideos,
    coldVideos: playback.coldVideos,
    worstReadyState: playback.worstReadyState,
    avgDecodeLatencyMs: roundOptional(playback.avgDecodeLatencyMs),
    avgSeekLatencyMs: roundOptional(playback.avgSeekLatencyMs),
    avgQueueDepth: roundOptional(playback.avgQueueDepth),
    maxQueueDepth: roundOptional(playback.maxQueueDepth),
    avgAudioDriftMs: roundOptional(playback.avgAudioDriftMs),
    decoderResets: playback.decoderResets,
    pendingSeekResolves: playback.pendingSeekResolves,
    avgPendingSeekMs: roundOptional(playback.avgPendingSeekMs),
    maxPendingSeekMs: roundOptional(playback.maxPendingSeekMs),
    collectorHolds: playback.collectorHolds,
    collectorDrops: playback.collectorDrops,
    lastAnomalyType: playback.lastAnomalyType,
  };
}

function collectCacheSnapshot(): Record<string, unknown> {
  return {
    scrubbing: engine.getScrubbingCacheStats(),
    composite: engine.getCompositeCacheStats(),
  };
}

function collectSnapshot(playbackWindowMs = DEFAULT_PLAYBACK_WINDOW_MS) {
  const { engineStats, gpuInfo, isEngineReady } = useEngineStore.getState();
  const s = engineStats;
  const playback = getPlaybackDebugStats(s.decoder, playbackWindowMs);

  const snapshot: Record<string, unknown> = {
    timestamp: Date.now(),
    engineReady: isEngineReady,
    fps: s.fps,
    targetFps: s.targetFps,
    isIdle: s.isIdle,
    timing: {
      rafGap: round(s.timing.rafGap),
      importTexture: round(s.timing.importTexture),
      renderPass: round(s.timing.renderPass),
      submit: round(s.timing.submit),
      total: round(s.timing.total),
    },
    drops: s.drops,
    decoder: s.decoder,
    layerCount: s.layerCount,
    audio: s.audio,
    health: playbackHealthMonitor.snapshot(),
    cache: collectCacheSnapshot(),
    pipelineStats: {
      wc: wcPipelineMonitor.stats(),
      vf: vfPipelineMonitor.stats(),
    },
  };

  snapshot.playback = serializePlayback(playback);

  if (s.webCodecsInfo) {
    snapshot.webCodecs = s.webCodecsInfo;
  }

  if (gpuInfo) {
    snapshot.gpu = gpuInfo;
  }

  return snapshot;
}

function round(v: number): number {
  return Math.round(v * 100) / 100;
}

export async function handleGetStats(): Promise<ToolResult> {
  return {
    success: true,
    data: collectSnapshot(),
  };
}

export async function handleGetStatsHistory(
  args: Record<string, unknown>
): Promise<ToolResult> {
  const samples = Math.min(Math.max((args.samples as number) || 5, 1), 30);
  const intervalMs = Math.max((args.intervalMs as number) || 200, 100);

  const history: Record<string, unknown>[] = [];

  // Collect first sample immediately
  history.push(collectSnapshot());

  // Collect remaining samples
  for (let i = 1; i < samples; i++) {
    await new Promise(resolve => setTimeout(resolve, intervalMs));
    history.push(collectSnapshot());
  }

  // Compute summary
  const fpsList = history.map(s => s.fps as number);
  const totalList = history.map(s => (s.timing as { total: number }).total);

  return {
    success: true,
    data: {
      samples: history.length,
      intervalMs,
      durationMs: (samples - 1) * intervalMs,
      summary: {
        fpsMin: Math.min(...fpsList),
        fpsMax: Math.max(...fpsList),
        fpsAvg: round(fpsList.reduce((a, b) => a + b, 0) / fpsList.length),
        renderTimeMin: Math.min(...totalList),
        renderTimeMax: Math.max(...totalList),
        renderTimeAvg: round(totalList.reduce((a, b) => a + b, 0) / totalList.length),
      },
      snapshots: history,
    },
  };
}

export async function handleGetLogs(
  args: Record<string, unknown>
): Promise<ToolResult> {
  const limit = Math.min(Math.max(Number(args.limit) || 100, 1), 500);
  const moduleFilter = typeof args.module === 'string' ? args.module.trim().toLowerCase() : '';
  const search = typeof args.search === 'string' ? args.search.trim().toLowerCase() : '';
  const level = typeof args.level === 'string' ? args.level.toUpperCase() : '';

  let logs = Logger.getBuffer(
    level === 'DEBUG' || level === 'INFO' || level === 'WARN' || level === 'ERROR'
      ? level
      : undefined
  );

  if (moduleFilter) {
    logs = logs.filter((entry) => entry.module.toLowerCase().includes(moduleFilter));
  }

  if (search) {
    logs = logs.filter((entry) =>
      entry.message.toLowerCase().includes(search) ||
      JSON.stringify(entry.data ?? '').toLowerCase().includes(search)
    );
  }

  const recentLogs = logs.slice(-limit);

  return {
    success: true,
    data: {
      count: recentLogs.length,
      totalMatched: logs.length,
      logs: recentLogs,
    },
  };
}

export async function handleGetPlaybackTrace(
  args: Record<string, unknown>
): Promise<ToolResult> {
  const windowMs = Math.min(
    Math.max(Number(args.windowMs) || DEFAULT_PLAYBACK_WINDOW_MS, 100),
    MAX_TRACE_WINDOW_MS
  );
  const limit = Math.min(Math.max(Number(args.limit) || 200, 1), MAX_TRACE_EVENTS);
  const { engineStats, isEngineReady, gpuInfo } = useEngineStore.getState();

  const wcTimeline = wcPipelineMonitor.timeline(windowMs);
  const vfTimeline = vfPipelineMonitor.timeline(windowMs);
  const healthVideos = playbackHealthMonitor.videos();
  const now = performance.now();
  const healthAnomalies = playbackHealthMonitor
    .anomalies()
    .filter((anomaly) => anomaly.timestamp >= now - windowMs);
  const playback = buildPlaybackDebugStats({
    decoder: engineStats.decoder,
    now,
    windowMs,
    wcTimeline,
    vfTimeline,
    healthVideos,
    healthAnomalies,
  });

  return {
    success: true,
    data: {
      timestamp: Date.now(),
      decoder: engineStats.decoder,
      engineReady: isEngineReady,
      windowMs,
      limit,
      playback: serializePlayback(playback),
      health: playbackHealthMonitor.snapshot(),
      cache: collectCacheSnapshot(),
      gpu: gpuInfo,
      wcStats: wcPipelineMonitor.stats(),
      vfStats: vfPipelineMonitor.stats(),
      wcEvents: wcTimeline.slice(-limit),
      vfEvents: vfTimeline.slice(-limit),
      healthVideos,
      healthAnomalies,
    },
  };
}
