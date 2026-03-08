import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { engine } from '../../src/engine/WebGPUEngine';
import { layerBuilder } from '../../src/services/layerBuilder';
import { PlaybackHealthMonitor } from '../../src/services/playbackHealthMonitor';

const hoisted = vi.hoisted(() => ({
  timelineState: {
    isPlaying: false,
    playheadPosition: 0,
    clips: [] as any[],
  },
  logInfo: vi.fn(),
  logWarn: vi.fn(),
}));

vi.mock('../../src/stores/timeline', () => ({
  useTimelineStore: {
    getState: vi.fn(() => hoisted.timelineState),
  },
}));

vi.mock('../../src/services/logger', () => ({
  Logger: {
    create: vi.fn(() => ({
      info: hoisted.logInfo,
      warn: hoisted.logWarn,
      debug: vi.fn(),
    })),
  },
}));

function createVideo(overrides: Partial<HTMLVideoElement> = {}): HTMLVideoElement {
  return {
    currentTime: 0,
    duration: 60,
    readyState: 4,
    seeking: false,
    paused: false,
    src: 'file:///demo.mp4',
    currentSrc: 'file:///demo.mp4',
    played: { length: 1 } as TimeRanges,
    play: vi.fn().mockResolvedValue(undefined),
    pause: vi.fn(),
    ...overrides,
  } as unknown as HTMLVideoElement;
}

function createClip(video: HTMLVideoElement, webCodecsPlayer?: { isFullMode?: () => boolean }) {
  return {
    id: 'clip-1',
    startTime: 0,
    duration: 10,
    source: {
      type: 'video',
      videoElement: video,
      webCodecsPlayer,
    },
  } as any;
}

describe('PlaybackHealthMonitor', () => {
  const videoSyncManager = {
    isVideoWarmingUp: vi.fn(() => false),
    clearWarmupState: vi.fn(),
    getActiveRvfcClipIds: vi.fn(() => []),
    cancelRvfcHandle: vi.fn(),
  };

  const layerCollector = {
    isVideoGpuReady: vi.fn(() => true),
    resetVideoGpuReady: vi.fn(),
  };

  const renderLoop = {
    getLastSuccessfulRenderTime: vi.fn(() => 0),
  };

  let now = 1000;

  beforeEach(() => {
    now = 1000;
    hoisted.timelineState.isPlaying = false;
    hoisted.timelineState.playheadPosition = 0;
    hoisted.timelineState.clips = [];

    hoisted.logInfo.mockReset();
    hoisted.logWarn.mockReset();

    videoSyncManager.isVideoWarmingUp.mockReset();
    videoSyncManager.isVideoWarmingUp.mockReturnValue(false);
    videoSyncManager.clearWarmupState.mockReset();
    videoSyncManager.getActiveRvfcClipIds.mockReset();
    videoSyncManager.getActiveRvfcClipIds.mockReturnValue([]);
    videoSyncManager.cancelRvfcHandle.mockReset();

    layerCollector.isVideoGpuReady.mockReset();
    layerCollector.isVideoGpuReady.mockReturnValue(true);
    layerCollector.resetVideoGpuReady.mockReset();

    renderLoop.getLastSuccessfulRenderTime.mockReset();
    renderLoop.getLastSuccessfulRenderTime.mockReturnValue(0);

    (layerBuilder as any).getVideoSyncManager = vi.fn(() => videoSyncManager);

    (engine as any).getLayerCollector = vi.fn(() => layerCollector);
    (engine as any).getRenderLoop = vi.fn(() => renderLoop);
    (engine as any).getStats = vi.fn(() => ({
      drops: {
        lastSecond: 0,
      },
    }));
    (engine as any).requestRender.mockReset();

    vi.spyOn(performance, 'now').mockImplementation(() => now);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('records GPU_SURFACE_COLD for HTML-video playback when the element is not GPU-ready', () => {
    const video = createVideo({ paused: false });
    hoisted.timelineState.isPlaying = true;
    hoisted.timelineState.playheadPosition = 1;
    hoisted.timelineState.clips = [createClip(video)];
    layerCollector.isVideoGpuReady.mockReturnValue(false);

    const monitor = new PlaybackHealthMonitor();
    (monitor as any).checkHealth();

    expect(monitor.anomalies('GPU_SURFACE_COLD')).toHaveLength(1);
    expect(layerCollector.resetVideoGpuReady).toHaveBeenCalledWith(video);
  });

  it('skips HTML-only health anomalies for full WebCodecs clips', () => {
    const video = createVideo({
      currentTime: 21.713,
      readyState: 0,
      paused: false,
    });
    const fullModeProvider = {
      isFullMode: () => true,
    };

    hoisted.timelineState.isPlaying = true;
    hoisted.timelineState.playheadPosition = 1;
    hoisted.timelineState.clips = [createClip(video, fullModeProvider)];
    layerCollector.isVideoGpuReady.mockReturnValue(false);

    const monitor = new PlaybackHealthMonitor();
    (monitor as any).checkHealth();
    now += 500;
    (monitor as any).checkHealth();
    now += 500;
    (monitor as any).checkHealth();
    now += 500;
    (monitor as any).checkHealth();

    expect(monitor.anomalies('GPU_SURFACE_COLD')).toHaveLength(0);
    expect(monitor.anomalies('READYSTATE_DROP')).toHaveLength(0);
    expect(monitor.anomalies('FRAME_STALL')).toHaveLength(0);
    expect(layerCollector.resetVideoGpuReady).not.toHaveBeenCalled();
    expect((engine as any).requestRender).not.toHaveBeenCalled();
  });

  it('records high drop rate with the anomaly cooldown applied', () => {
    (engine as any).getStats = vi.fn(() => ({
      drops: {
        lastSecond: 25,
      },
    }));

    const monitor = new PlaybackHealthMonitor();

    (monitor as any).checkHealth();
    now += 1000;
    (monitor as any).checkHealth();
    now += 5001;
    (monitor as any).checkHealth();

    expect(monitor.anomalies('HIGH_DROP_RATE')).toHaveLength(2);
    expect(hoisted.logWarn).toHaveBeenCalledTimes(2);
  });
});
