import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flags } from '../../src/engine/featureFlags';
import { VideoSyncManager } from '../../src/services/layerBuilder/VideoSyncManager';

describe('VideoSyncManager paused WebCodecs provider selection', () => {
  beforeEach(() => {
    vi.useRealTimers();
    flags.useFullWebCodecsPlayback = true;
  });

  it('keeps driving the clip player while the scrub runtime is still cold', () => {
    const manager = new VideoSyncManager() as any;
    const clipPlayer = {
      isFullMode: () => true,
      hasFrame: () => false,
      getCurrentFrame: () => null,
      currentTime: 1,
    };
    const scrubProvider = {
      isFullMode: () => true,
      hasFrame: () => false,
      getCurrentFrame: () => null,
      currentTime: 1.02,
      getPendingSeekTime: () => 1.02,
    };

    const provider = manager.getPausedWebCodecsProvider(
      { webCodecsPlayer: clipPlayer },
      scrubProvider,
      1.01
    );

    expect(provider).toBe(clipPlayer);
  });

  it('switches to the scrub runtime once it has a frame near the target', () => {
    const manager = new VideoSyncManager() as any;
    const clipPlayer = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 900_000 }),
      currentTime: 0.9,
    };
    const scrubProvider = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      currentTime: 1.01,
      getPendingSeekTime: () => 1.01,
    };

    const provider = manager.getPausedWebCodecsProvider(
      { webCodecsPlayer: clipPlayer },
      scrubProvider,
      1.01
    );

    expect(provider).toBe(scrubProvider);
  });

  it('forces a paused seek when the provider is already at the target time but still has no frame', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      hasFrame: () => false,
      getCurrentFrame: () => null,
      getPendingSeekTime: () => null,
      isDecodePending: () => false,
    };

    expect(manager.shouldSeekPausedWebCodecsProvider(provider, 1)).toBe(true);
  });

  it('does not force a paused seek when the provider already has a frame at the target time', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      getPendingSeekTime: () => null,
      isDecodePending: () => false,
    };

    expect(manager.shouldSeekPausedWebCodecsProvider(provider, 1)).toBe(false);
  });

  it('does not re-seek while the same paused seek target is already pending', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      hasFrame: () => false,
      getCurrentFrame: () => null,
      getPendingSeekTime: () => 1,
      isDecodePending: () => false,
    };

    expect(manager.shouldSeekPausedWebCodecsProvider(provider, 1)).toBe(false);
  });

  it('blocks audio start until the playback provider has a frame at the target', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      hasFrame: () => false,
      getCurrentFrame: () => null,
      getPendingSeekTime: () => 1,
    };

    expect(manager.isPlaybackProviderReadyForAudioStart(provider, 1)).toBe(false);
  });

  it('allows audio start once the playback provider has a frame near the target', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1.01,
      hasFrame: () => true,
      hasBufferedFutureFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_010_000 }),
      getPendingSeekTime: () => 1.01,
    };

    expect(manager.isPlaybackProviderReadyForAudioStart(provider, 1.01)).toBe(true);
  });

  it('blocks audio start until a future playback frame is buffered', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1.01,
      hasFrame: () => true,
      hasBufferedFutureFrame: () => false,
      getCurrentFrame: () => ({ timestamp: 1_010_000 }),
      getPendingSeekTime: () => 1.01,
    };

    expect(manager.isPlaybackProviderReadyForAudioStart(provider, 1.01)).toBe(false);
  });

  it('allows a new fast seek when a busy scrub provider is stale and the target moved', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      getPendingSeekTime: () => null,
      isDecodePending: () => true,
    };

    manager.lastWcFastSeekTarget['clip:scrub'] = 1;
    manager.lastWcFastSeekAt['clip:scrub'] = performance.now() - 120;

    expect(manager.shouldFastSeekPausedWebCodecsProvider(provider, 'clip:scrub', 1.4)).toBe(true);
  });

  it('keeps fast seek blocked while the current busy decode is still fresh', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      getPendingSeekTime: () => null,
      isDecodePending: () => true,
    };

    manager.lastWcFastSeekTarget['clip:scrub'] = 1;
    manager.lastWcFastSeekAt['clip:scrub'] = performance.now() - 20;

    expect(manager.shouldFastSeekPausedWebCodecsProvider(provider, 'clip:scrub', 1.4)).toBe(false);
  });

  it('debounces precise scrub seeks while keeping the latest target', async () => {
    vi.useFakeTimers();

    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      seek: vi.fn(),
    };

    manager.schedulePreciseWcSeek('clip:scrub', provider, 1.2);
    await vi.advanceTimersByTimeAsync(60);
    manager.schedulePreciseWcSeek('clip:scrub', provider, 1.6);

    await vi.advanceTimersByTimeAsync(70);

    expect(provider.seek).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(60);

    expect(provider.seek).toHaveBeenCalledTimes(1);
    expect(provider.seek).toHaveBeenCalledWith(1.6);
  });

  it('uses a direct precise seek during drag for nearby forward targets', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      seek: vi.fn(),
      fastSeek: vi.fn(),
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      getPendingSeekTime: () => null,
      isDecodePending: () => false,
    };

    manager.syncPausedWebCodecsProvider(provider, 'clip:scrub', 1.18, true, true);

    expect(provider.seek).toHaveBeenCalledWith(1.18);
    expect(provider.fastSeek).not.toHaveBeenCalled();
  });

  it('keeps the fallback provider on fast seek only while a dedicated scrub provider warms up', () => {
    const manager = new VideoSyncManager() as any;
    const provider = {
      currentTime: 1,
      seek: vi.fn(),
      fastSeek: vi.fn(),
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      getPendingSeekTime: () => null,
      isDecodePending: () => false,
    };

    manager.syncPausedWebCodecsProvider(provider, 'clip:fallback', 1.18, true, false, false);

    expect(provider.seek).not.toHaveBeenCalled();
    expect(provider.fastSeek).toHaveBeenCalledWith(1.18);
  });

  it('routes full WebCodecs clips through HTML seek logic while dragging', () => {
    const manager = new VideoSyncManager() as any;
    const syncFullWebCodecs = vi.spyOn(manager, 'syncFullWebCodecs');
    const throttledSeek = vi.spyOn(manager, 'throttledSeek').mockImplementation(() => {});

    const video = {
      currentTime: 0,
      paused: true,
      seeking: false,
      readyState: 4,
      played: { length: 1 },
      pause: vi.fn(),
    } as any;

    manager.syncClipVideo({
      id: 'clip-1',
      trackId: 'track-v1',
      startTime: 0,
      inPoint: 0,
      outPoint: 10,
      duration: 10,
      reversed: false,
      source: {
        videoElement: video,
        webCodecsPlayer: {
          isFullMode: () => true,
        },
      },
    }, {
      isPlaying: false,
      isDraggingPlayhead: true,
      playbackSpeed: 1,
      now: 1000,
      playheadPosition: 1.5,
      hasKeyframes: () => false,
      getInterpolatedSpeed: () => 1,
      getSourceTimeForClip: () => 1.5,
    } as any);

    expect(syncFullWebCodecs).not.toHaveBeenCalled();
    expect(throttledSeek).toHaveBeenCalled();
  });

  it('routes full WebCodecs clips through HTML sync logic when preview WebCodecs is disabled', () => {
    flags.useFullWebCodecsPlayback = false;

    const manager = new VideoSyncManager() as any;
    const syncFullWebCodecs = vi.spyOn(manager, 'syncFullWebCodecs');
    const throttledSeek = vi.spyOn(manager, 'throttledSeek').mockImplementation(() => {});

    const video = {
      currentTime: 0,
      paused: true,
      seeking: false,
      readyState: 4,
      played: { length: 1 },
      pause: vi.fn(),
      playbackRate: 1,
    } as any;

    manager.syncClipVideo({
      id: 'clip-2',
      trackId: 'track-v1',
      startTime: 0,
      inPoint: 0,
      outPoint: 10,
      duration: 10,
      reversed: false,
      source: {
        videoElement: video,
        webCodecsPlayer: {
          isFullMode: () => true,
        },
      },
    }, {
      isPlaying: false,
      isDraggingPlayhead: false,
      playbackSpeed: 1,
      now: 1000,
      playheadPosition: 1.5,
      hasKeyframes: () => false,
      getInterpolatedSpeed: () => 1,
      getSourceTimeForClip: () => 1.5,
    } as any);

    expect(syncFullWebCodecs).not.toHaveBeenCalled();
    expect(throttledSeek).toHaveBeenCalled();
  });

  it('does not reuse the previous HTML video element across same-source reordered cuts when preview WebCodecs is disabled', () => {
    flags.useFullWebCodecsPlayback = false;

    const manager = new VideoSyncManager() as any;
    const previousVideo = {
      currentTime: 6.35,
    } as HTMLVideoElement;
    const nextVideo = {
      currentTime: 0,
    } as HTMLVideoElement;
    const file = new File(['video'], 'reordered.mp4', { type: 'video/mp4' });

    manager.lastTrackState.set('track-v1', {
      clipId: 'clip-prev',
      fileId: 'media-1',
      file,
      videoElement: previousVideo,
      outPoint: 1.4,
    });

    manager.computeHandoffs({
      isPlaying: true,
      isDraggingPlayhead: false,
      clipsAtTime: [{
        id: 'clip-next',
        trackId: 'track-v1',
        file,
        inPoint: 6.8,
        outPoint: 8.4,
        source: {
          mediaFileId: 'media-1',
          videoElement: nextVideo,
        },
      }],
    } as any);

    expect(manager.getHandoffVideoElement('clip-next')).toBeNull();
  });

  it('does not reuse the previous HTML video element across same-source reordered cuts when the source-time jump is too large', () => {
    flags.useFullWebCodecsPlayback = false;

    const manager = new VideoSyncManager() as any;
    const previousVideo = {
      currentTime: 4.2,
    } as HTMLVideoElement;
    const nextVideo = {
      currentTime: 0,
    } as HTMLVideoElement;
    const file = new File(['video'], 'reordered-large-jump.mp4', { type: 'video/mp4' });

    manager.lastTrackState.set('track-v1', {
      clipId: 'clip-prev',
      fileId: 'media-1',
      file,
      videoElement: previousVideo,
      outPoint: 1.4,
    });

    manager.computeHandoffs({
      isPlaying: true,
      isDraggingPlayhead: false,
      clipsAtTime: [{
        id: 'clip-next',
        trackId: 'track-v1',
        file,
        inPoint: 6.8,
        outPoint: 8.4,
        source: {
          mediaFileId: 'media-1',
          videoElement: nextVideo,
        },
      }],
    } as any);

    expect(manager.getHandoffVideoElement('clip-next')).toBeNull();
  });
});
