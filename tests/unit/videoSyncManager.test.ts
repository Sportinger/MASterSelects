import { beforeEach, describe, expect, it, vi } from 'vitest';
import { engine } from '../../src/engine/WebGPUEngine';
import { flags } from '../../src/engine/featureFlags';
import { VideoSyncManager } from '../../src/services/layerBuilder/VideoSyncManager';

describe('VideoSyncManager paused WebCodecs provider selection', () => {
  beforeEach(() => {
    vi.useRealTimers();
    flags.useFullWebCodecsPlayback = true;
    (engine as any).ensureVideoFrameCached = vi.fn();
    (engine as any).getLastPresentedVideoTime = vi.fn(() => undefined);
    (engine as any).requestNewFrameRender = vi.fn();
    (engine as any).cacheFrameAtTime = vi.fn();
    (engine as any).markVideoFramePresented = vi.fn();
    (engine as any).captureVideoFrameAtTime = vi.fn(() => false);
    (engine as any).markVideoGpuReady = vi.fn();
    (engine as any).requestRender = vi.fn();
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

  it('prefers the shared runtime when its frame is closer to the paused target than the clip player', () => {
    const manager = new VideoSyncManager() as any;
    const clipPlayer = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 22_589_233 }),
      currentTime: 22.589233,
    };
    const sharedRuntimeProvider = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 8_700_000 }),
      currentTime: 8.7,
      getPendingSeekTime: () => 8.7,
    };

    const provider = manager.getPausedWebCodecsProvider(
      { webCodecsPlayer: clipPlayer },
      sharedRuntimeProvider,
      8.68
    );

    expect(provider).toBe(sharedRuntimeProvider);
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

  it('routes full WebCodecs clips through dedicated WebCodecs sync while dragging', () => {
    const manager = new VideoSyncManager() as any;
    const syncFullWebCodecs = vi.spyOn(manager, 'syncFullWebCodecs').mockImplementation(() => {});
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

    expect(syncFullWebCodecs).toHaveBeenCalledTimes(1);
    expect(throttledSeek).not.toHaveBeenCalled();
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

  it('pre-captures paused HTML frames with the active clip id as owner', () => {
    flags.useFullWebCodecsPlayback = false;

    const manager = new VideoSyncManager() as any;
    const ensureVideoFrameCached = (engine as any).ensureVideoFrameCached;

    const video = {
      currentTime: 1.5,
      paused: true,
      seeking: false,
      readyState: 4,
      played: { length: 1 },
      pause: vi.fn(),
      playbackRate: 1,
    } as any;

    manager.syncClipVideo({
      id: 'clip-owner',
      trackId: 'track-v1',
      startTime: 0,
      inPoint: 0,
      outPoint: 10,
      duration: 10,
      reversed: false,
      source: {
        videoElement: video,
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

    expect(ensureVideoFrameCached).toHaveBeenCalledWith(video, 'clip-owner');
  });

  it('rate-limits drag precise seeks when fastSeek is unavailable', () => {
    const manager = new VideoSyncManager() as any;

    let currentTime = 0;
    const assignedSeekTimes: number[] = [];
    const video = {
      duration: 10,
      seeking: false,
      addEventListener: vi.fn(),
    } as any;

    Object.defineProperty(video, 'currentTime', {
      configurable: true,
      get: () => currentTime,
      set: (value: number) => {
        assignedSeekTimes.push(value);
        currentTime = value;
      },
    });

    manager.throttledSeek('clip-html', video, 1.5, {
      isDraggingPlayhead: true,
      now: 1000,
    } as any);

    manager.throttledSeek('clip-html', video, 2.0, {
      isDraggingPlayhead: true,
      now: 1040,
    } as any);

    expect(assignedSeekTimes).toEqual([1.5]);
  });

  it('preloads the paused jump neighborhood after a large paused seek', () => {
    flags.useFullWebCodecsPlayback = false;

    const manager = new VideoSyncManager() as any;
    const startTargetedWarmup = vi
      .spyOn(manager, 'startTargetedWarmup')
      .mockImplementation(() => {});

    const video = {
      currentTime: 0,
      readyState: 4,
      seeking: false,
      preload: 'metadata',
      src: 'file:///jump.mp4',
      currentSrc: 'file:///jump.mp4',
    } as any;

    manager.preloadPausedJumpNeighborhood({
      isPlaying: false,
      isDraggingPlayhead: false,
      playheadPosition: 5,
      clips: [{
        id: 'clip-jump',
        trackId: 'track-v1',
        startTime: 0,
        inPoint: 0,
        outPoint: 10,
        duration: 10,
        source: {
          videoElement: video,
        },
      }],
      clipsAtTime: [{
        id: 'clip-jump',
        trackId: 'track-v1',
        startTime: 0,
        inPoint: 0,
        outPoint: 10,
        duration: 10,
        source: {
          videoElement: video,
        },
      }],
      getInterpolatedSpeed: () => 1,
      getSourceTimeForClip: () => 5,
    } as any);

    expect(startTargetedWarmup).toHaveBeenCalledWith('clip-jump', video, 5, {
      proactive: true,
      requestRender: true,
    });
  });

  it('does not spam paused jump preload for the same paused target', () => {
    flags.useFullWebCodecsPlayback = false;

    const manager = new VideoSyncManager() as any;
    const startTargetedWarmup = vi
      .spyOn(manager, 'startTargetedWarmup')
      .mockImplementation(() => {});

    const video = {
      currentTime: 0,
      readyState: 4,
      seeking: false,
      preload: 'metadata',
      src: 'file:///jump.mp4',
      currentSrc: 'file:///jump.mp4',
    } as any;

    const ctx = {
      isPlaying: false,
      isDraggingPlayhead: false,
      playheadPosition: 5,
      clips: [{
        id: 'clip-jump',
        trackId: 'track-v1',
        startTime: 0,
        inPoint: 0,
        outPoint: 10,
        duration: 10,
        source: {
          videoElement: video,
        },
      }],
      clipsAtTime: [{
        id: 'clip-jump',
        trackId: 'track-v1',
        startTime: 0,
        inPoint: 0,
        outPoint: 10,
        duration: 10,
        source: {
          videoElement: video,
        },
      }],
      getInterpolatedSpeed: () => 1,
      getSourceTimeForClip: () => 5,
    } as any;

    manager.preloadPausedJumpNeighborhood(ctx);
    manager.preloadPausedJumpNeighborhood(ctx);

    expect(startTargetedWarmup).toHaveBeenCalledTimes(1);
  });

  it('aborts a stuck targeted warmup when no frame arrives', async () => {
    vi.useFakeTimers();

    const manager = new VideoSyncManager() as any;
    const video = {
      currentTime: 1,
      readyState: 1,
      preload: 'metadata',
      muted: false,
      play: vi.fn().mockResolvedValue(undefined),
      pause: vi.fn(),
      requestVideoFrameCallback: vi.fn(() => 1),
    } as any;

    manager.startTargetedWarmup('clip-warm', video, 1);
    await vi.runAllTicks();

    expect(manager.isVideoWarmingUp(video)).toBe(true);

    await vi.advanceTimersByTimeAsync(950);

    expect(manager.isVideoWarmingUp(video)).toBe(false);
    expect((engine as any).markVideoGpuReady).not.toHaveBeenCalled();
    expect(video.pause).toHaveBeenCalled();
  });

  it('falls back to finishing warmup when the target frame is already ready', async () => {
    vi.useFakeTimers();

    const manager = new VideoSyncManager() as any;
    const video = {
      currentTime: 1,
      readyState: 4,
      preload: 'metadata',
      muted: false,
      play: vi.fn().mockResolvedValue(undefined),
      pause: vi.fn(),
      requestVideoFrameCallback: vi.fn(() => 1),
    } as any;

    manager.startTargetedWarmup('clip-warm', video, 1);
    await vi.runAllTicks();
    await vi.advanceTimersByTimeAsync(950);

    expect(manager.isVideoWarmingUp(video)).toBe(false);
    expect((engine as any).markVideoGpuReady).toHaveBeenCalledWith(video);
    expect((engine as any).cacheFrameAtTime).toHaveBeenCalledWith(video, 1);
  });

  it('clears in-flight HTML seek state before starting a targeted warmup', async () => {
    vi.useFakeTimers();

    const manager = new VideoSyncManager() as any;
    const video = {
      currentTime: 1,
      readyState: 1,
      preload: 'metadata',
      muted: false,
      play: vi.fn().mockResolvedValue(undefined),
      pause: vi.fn(),
      requestVideoFrameCallback: vi.fn(() => 3),
      cancelVideoFrameCallback: vi.fn(),
    } as any;

    manager.rvfcHandles['clip-warm'] = 7;
    manager.pendingSeekTargets['clip-warm'] = 2.4;
    manager.pendingSeekStartedAt['clip-warm'] = 123;
    manager.queuedSeekTargets['clip-warm'] = 2.8;
    manager.latestSeekTargets['clip-warm'] = 3.1;
    manager.seekedFlushArmed.add('clip-warm');
    manager.preciseSeekTimers['clip-warm'] = setTimeout(() => {}, 5000);

    manager.startTargetedWarmup('clip-warm', video, 1.5);
    await vi.runAllTicks();

    expect(video.cancelVideoFrameCallback).toHaveBeenCalledWith(7);
    expect(manager.pendingSeekTargets['clip-warm']).toBeUndefined();
    expect(manager.pendingSeekStartedAt['clip-warm']).toBeUndefined();
    expect(manager.queuedSeekTargets['clip-warm']).toBeUndefined();
    expect(manager.latestSeekTargets['clip-warm']).toBeUndefined();
    expect(manager.preciseSeekTimers['clip-warm']).toBeUndefined();
    expect(manager.seekedFlushArmed.has('clip-warm')).toBe(false);
  });

  it('retargets an active warmup when the paused scrub target jumps far away', () => {
    const manager = new VideoSyncManager() as any;
    const video = {
      pause: vi.fn(),
    } as any;

    manager.warmingUpVideos.add(video);
    manager.warmupClipIds.set(video, 'clip-warm');
    manager.warmupTargetTimes.set(video, 1);

    const clearWarmupState = vi
      .spyOn(manager, 'clearWarmupState')
      .mockImplementation(() => {});
    const startTargetedWarmup = vi
      .spyOn(manager, 'startTargetedWarmup')
      .mockImplementation(() => {});

    manager.maybeRetargetActiveWarmup('clip-warm', video, 2.25, 1000, {
      isPlaying: false,
      isDragging: true,
      requestRender: true,
    });

    expect(clearWarmupState).toHaveBeenCalledWith(video);
    expect(startTargetedWarmup).toHaveBeenCalledWith('clip-warm', video, 2.25, {
      proactive: false,
      requestRender: true,
      resumeAfterWarmup: false,
    });
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
