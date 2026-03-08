import { beforeEach, describe, expect, it, vi } from 'vitest';

const hoisted = vi.hoisted(() => ({
  createFrameContext: vi.fn(),
  syncBackground: vi.fn(),
}));

vi.mock('../../src/services/layerBuilder/FrameContext', () => ({
  createFrameContext: () => hoisted.createFrameContext(),
  getClipTimeInfo: vi.fn(),
  getMediaFileForClip: vi.fn(),
}));

vi.mock('../../src/services/layerPlaybackManager', () => ({
  layerPlaybackManager: {
    syncVideoElements: (...args: unknown[]) => hoisted.syncBackground(...args),
  },
}));

vi.mock('../../src/services/mediaRuntime/runtimePlayback', () => ({
  canUseSharedPreviewRuntimeSession: vi.fn(() => true),
  ensureRuntimeFrameProvider: vi.fn(),
  getPreviewRuntimeSource: vi.fn((source: unknown) => source),
  getRuntimeFrameProvider: vi.fn(() => null),
  getScrubRuntimeSource: vi.fn((source: unknown) => source),
  updateRuntimePlaybackTime: vi.fn(),
}));

vi.mock('../../src/services/vfPipelineMonitor', () => ({
  vfPipelineMonitor: {
    record: vi.fn(),
  },
}));

vi.mock('../../src/services/logger', () => ({
  Logger: {
    create: vi.fn(() => ({
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    })),
  },
}));

import { VideoSyncManager } from '../../src/services/layerBuilder/VideoSyncManager';

describe('VideoSyncManager same-frame sync gate', () => {
  beforeEach(() => {
    hoisted.createFrameContext.mockReset();
    hoisted.syncBackground.mockReset();
  });

  it('does not skip a same-frame playback sync when clip references changed asynchronously', () => {
    const manager = new VideoSyncManager() as any;
    const syncClipVideo = vi.spyOn(manager, 'syncClipVideo').mockImplementation(() => {});
    vi.spyOn(manager, 'warmupUpcomingClips').mockImplementation(() => {});
    vi.spyOn(manager, 'preBufferUpcomingVideoAudio').mockImplementation(() => {});
    vi.spyOn(manager, 'updateLastTrackState').mockImplementation(() => {});

    const video = {
      paused: true,
      played: { length: 1 },
      currentTime: 1,
    } as unknown as HTMLVideoElement;

    const clipA = {
      id: 'clip-1',
      trackId: 'track-v1',
      inPoint: 0,
      outPoint: 10,
      source: {
        type: 'video',
        videoElement: video,
      },
    };

    const clipB = {
      ...clipA,
      source: {
        ...clipA.source,
        webCodecsPlayer: {
          isFullMode: () => true,
        },
      },
    };

    hoisted.createFrameContext
      .mockReturnValueOnce({
        isPlaying: true,
        isDraggingPlayhead: false,
        frameNumber: 10,
        playheadPosition: 1,
        clips: [clipA],
        clipsAtTime: [clipA],
        clipsByTrackId: new Map([['track-v1', clipA]]),
      })
      .mockReturnValueOnce({
        isPlaying: true,
        isDraggingPlayhead: false,
        frameNumber: 10,
        playheadPosition: 1,
        clips: [clipB],
        clipsAtTime: [clipB],
        clipsByTrackId: new Map([['track-v1', clipB]]),
      });

    manager.syncVideoElements();
    manager.syncVideoElements();

    expect(syncClipVideo).toHaveBeenCalledTimes(2);
  });
});
