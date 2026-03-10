import { useEngineStore } from '../../../stores/engineStore';
import { Logger } from '../../logger';
import { playbackHealthMonitor } from '../../playbackHealthMonitor';
import { vfPipelineMonitor } from '../../vfPipelineMonitor';
import { wcPipelineMonitor } from '../../wcPipelineMonitor';
import type { ToolResult } from '../types';

function collectSnapshot() {
  const { engineStats, gpuInfo, isEngineReady } = useEngineStore.getState();
  const s = engineStats;

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
  };

  if (s.playback) {
    snapshot.playback = {
      status: s.playback.status,
      pipeline: s.playback.pipeline,
      frameEvents: s.playback.frameEvents,
      cadenceFps: round(s.playback.cadenceFps),
      avgFrameGapMs: round(s.playback.avgFrameGapMs),
      p95FrameGapMs: round(s.playback.p95FrameGapMs),
      maxFrameGapMs: round(s.playback.maxFrameGapMs),
      previewFrames: s.playback.previewFrames,
      previewUpdates: s.playback.previewUpdates,
      previewRenderFps: round(s.playback.previewRenderFps),
      previewUpdateFps: round(s.playback.previewUpdateFps),
      avgPreviewRenderGapMs: round(s.playback.avgPreviewRenderGapMs),
      p95PreviewRenderGapMs: round(s.playback.p95PreviewRenderGapMs),
      maxPreviewRenderGapMs: round(s.playback.maxPreviewRenderGapMs),
      avgPreviewUpdateGapMs: round(s.playback.avgPreviewUpdateGapMs),
      p95PreviewUpdateGapMs: round(s.playback.p95PreviewUpdateGapMs),
      maxPreviewUpdateGapMs: round(s.playback.maxPreviewUpdateGapMs),
      stalePreviewFrames: s.playback.stalePreviewFrames,
      stalePreviewWhileTargetMoved: s.playback.stalePreviewWhileTargetMoved,
      avgPreviewDriftMs: round(s.playback.avgPreviewDriftMs),
      maxPreviewDriftMs: round(s.playback.maxPreviewDriftMs),
      stalls: s.playback.stalls,
      seeks: s.playback.seeks,
      advanceSeeks: s.playback.advanceSeeks,
      driftCorrections: s.playback.driftCorrections,
      readyStateDrops: s.playback.readyStateDrops,
      queuePressureEvents: s.playback.queuePressureEvents,
      healthAnomalies: s.playback.healthAnomalies,
      activeVideos: s.playback.activeVideos,
      playingVideos: s.playback.playingVideos,
      seekingVideos: s.playback.seekingVideos,
      warmingUpVideos: s.playback.warmingUpVideos,
      coldVideos: s.playback.coldVideos,
      worstReadyState: s.playback.worstReadyState,
      avgDecodeLatencyMs: s.playback.avgDecodeLatencyMs ? round(s.playback.avgDecodeLatencyMs) : undefined,
      avgSeekLatencyMs: s.playback.avgSeekLatencyMs ? round(s.playback.avgSeekLatencyMs) : undefined,
      avgQueueDepth: s.playback.avgQueueDepth ? round(s.playback.avgQueueDepth) : undefined,
      maxQueueDepth: s.playback.maxQueueDepth ? round(s.playback.maxQueueDepth) : undefined,
      avgAudioDriftMs: s.playback.avgAudioDriftMs ? round(s.playback.avgAudioDriftMs) : undefined,
      decoderResets: s.playback.decoderResets,
      pendingSeekResolves: s.playback.pendingSeekResolves,
      avgPendingSeekMs: s.playback.avgPendingSeekMs ? round(s.playback.avgPendingSeekMs) : undefined,
      maxPendingSeekMs: s.playback.maxPendingSeekMs ? round(s.playback.maxPendingSeekMs) : undefined,
      collectorHolds: s.playback.collectorHolds,
      collectorDrops: s.playback.collectorDrops,
      lastAnomalyType: s.playback.lastAnomalyType,
    };
  }

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
  const windowMs = Math.min(Math.max(Number(args.windowMs) || 5000, 100), 30000);
  const limit = Math.min(Math.max(Number(args.limit) || 120, 1), 500);
  const { engineStats } = useEngineStore.getState();

  const wcTimeline = wcPipelineMonitor.timeline(windowMs);
  const vfTimeline = vfPipelineMonitor.timeline(windowMs);
  const healthVideos = playbackHealthMonitor.videos();
  const healthAnomalies = playbackHealthMonitor
    .anomalies()
    .filter((anomaly) => anomaly.timestamp >= Date.now() - windowMs);

  return {
    success: true,
    data: {
      decoder: engineStats.decoder,
      windowMs,
      wcStats: wcPipelineMonitor.stats(),
      vfStats: vfPipelineMonitor.stats(),
      wcEvents: wcTimeline.slice(-limit),
      vfEvents: vfTimeline.slice(-limit),
      healthVideos,
      healthAnomalies,
    },
  };
}
