import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AudioSyncHandler } from '../../src/services/layerBuilder/AudioSyncHandler';
import { AudioTrackSyncManager } from '../../src/services/layerBuilder/AudioTrackSyncManager';
import { audioRoutingManager } from '../../src/services/audioRoutingManager';
import { proxyFrameCache } from '../../src/services/proxyFrameCache';

function makeClip(overrides: Record<string, unknown> = {}) {
  return {
    id: 'clip-1',
    trackId: 'track-1',
    name: 'clip',
    startTime: 0,
    duration: 10,
    inPoint: 0,
    outPoint: 10,
    effects: [],
    preservesPitch: true,
    ...overrides,
  } as any;
}

function makeFrameContext(overrides: Record<string, unknown> = {}) {
  const clips = (overrides.clips as any[]) ?? [];
  const clipsAtTime = (overrides.clipsAtTime as any[]) ?? clips;
  const mediaFiles = (overrides.mediaFiles as any[]) ?? [];

  return {
    clips,
    clipsAtTime,
    tracks: [],
    videoTracks: (overrides.videoTracks as any[]) ?? [],
    audioTracks: (overrides.audioTracks as any[]) ?? [],
    visibleVideoTrackIds: (overrides.visibleVideoTrackIds as Set<string>) ?? new Set(),
    unmutedAudioTrackIds: (overrides.unmutedAudioTrackIds as Set<string>) ?? new Set(),
    clipsByTrackId: new Map(clipsAtTime.map((clip: any) => [clip.trackId, clip])),
    mediaFiles,
    mediaFileById: new Map(mediaFiles.map((file: any) => [file.id, file])),
    mediaFileByName: new Map(),
    compositionById: new Map(),
    isPlaying: false,
    isDraggingPlayhead: true,
    playheadPosition: 1,
    playbackSpeed: 1,
    proxyEnabled: false,
    activeCompId: 'default',
    frameNumber: 30,
    now: 100,
    getInterpolatedTransform: () => ({}),
    getInterpolatedEffects: () => [],
    getInterpolatedSpeed: () => 1,
    getSourceTimeForClip: (_clipId: string, clipLocalTime: number) => clipLocalTime,
    hasKeyframes: () => false,
    ...overrides,
  } as any;
}

function stubProxyFrameCache(overrides: { hasAudioBuffer?: boolean } = {}) {
  const originalPlayScrubAudio = (proxyFrameCache as any).playScrubAudio;
  const originalHasAudioBuffer = (proxyFrameCache as any).hasAudioBuffer;
  const originalGetCachedAudioProxy = (proxyFrameCache as any).getCachedAudioProxy;
  const originalPreloadAudioProxy = (proxyFrameCache as any).preloadAudioProxy;
  const originalGetAudioBuffer = (proxyFrameCache as any).getAudioBuffer;
  const originalStopScrubAudio = (proxyFrameCache as any).stopScrubAudio;
  const playScrubAudio = vi.fn();
  const hasAudioBuffer = vi.fn(() => overrides.hasAudioBuffer ?? true);
  const getCachedAudioProxy = vi.fn(() => null);
  const preloadAudioProxy = vi.fn();
  const getAudioBuffer = vi.fn();
  const stopScrubAudio = vi.fn();

  (proxyFrameCache as any).playScrubAudio = playScrubAudio;
  (proxyFrameCache as any).hasAudioBuffer = hasAudioBuffer;
  (proxyFrameCache as any).getCachedAudioProxy = getCachedAudioProxy;
  (proxyFrameCache as any).preloadAudioProxy = preloadAudioProxy;
  (proxyFrameCache as any).getAudioBuffer = getAudioBuffer;
  (proxyFrameCache as any).stopScrubAudio = stopScrubAudio;

  return {
    playScrubAudio,
    hasAudioBuffer,
    getCachedAudioProxy,
    preloadAudioProxy,
    getAudioBuffer,
    stopScrubAudio,
    restore: () => {
      (proxyFrameCache as any).playScrubAudio = originalPlayScrubAudio;
      (proxyFrameCache as any).hasAudioBuffer = originalHasAudioBuffer;
      (proxyFrameCache as any).getCachedAudioProxy = originalGetCachedAudioProxy;
      (proxyFrameCache as any).preloadAudioProxy = originalPreloadAudioProxy;
      (proxyFrameCache as any).getAudioBuffer = originalGetAudioBuffer;
      (proxyFrameCache as any).stopScrubAudio = originalStopScrubAudio;
    },
  };
}

describe('scrub audio sync', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('applies clip volume during fallback scrub playback instead of a fixed default', () => {
    const handler = new AudioSyncHandler();
    const play = vi.fn().mockResolvedValue(undefined);
    const element = {
      muted: false,
      volume: 1,
      playbackRate: 1,
      currentTime: 0,
      paused: true,
      play,
      pause: vi.fn(),
    } as any;

    handler.syncAudioElement(
      {
        element,
        clip: makeClip(),
        clipTime: 1.2,
        absSpeed: 1,
        isMuted: false,
        canBeMaster: false,
        type: 'audioTrack',
        volume: 0.2,
      },
      makeFrameContext(),
      { audioPlayingCount: 0, maxAudioDrift: 0, hasAudioError: false, masterSet: false }
    );

    expect(element.volume).toBe(0.2);
    expect(element.currentTime).toBe(1.2);
    expect(play).toHaveBeenCalledOnce();
  });

  it('routes scrub fallback through Web Audio when EQ is present', () => {
    const handler = new AudioSyncHandler();
    const applyEffects = vi.spyOn(audioRoutingManager, 'applyEffects').mockResolvedValue(true);
    const element = {
      muted: false,
      volume: 1,
      playbackRate: 1,
      currentTime: 0,
      paused: true,
      play: vi.fn().mockResolvedValue(undefined),
      pause: vi.fn(),
    } as any;

    handler.syncAudioElement(
      {
        element,
        clip: makeClip(),
        clipTime: 0.5,
        absSpeed: 1,
        isMuted: false,
        canBeMaster: false,
        type: 'audioTrack',
        volume: 0.75,
        eqGains: [0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
      },
      makeFrameContext(),
      { audioPlayingCount: 0, maxAudioDrift: 0, hasAudioError: false, masterSet: false }
    );

    expect(applyEffects).toHaveBeenCalledWith(element, 0.75, [0, 0, 0, 0, 0, 6, 0, 0, 0, 0]);
  });

  it('uses linked audio clip settings for varispeed scrub audio and skips proxy fallback duplication', () => {
    const manager = new AudioTrackSyncManager() as any;
    const syncAudioElement = vi.fn();
    manager.audioSyncHandler = { syncAudioElement, stopScrubAudio: vi.fn() };

    const cacheStub = stubProxyFrameCache();

    const videoElement = {
      muted: false,
      currentSrc: 'blob:video-src',
      src: 'blob:video-src',
    } as any;

    const videoClip = makeClip({
      id: 'video-1',
      trackId: 'video-track',
      linkedClipId: 'audio-1',
      mediaFileId: 'media-1',
      source: { type: 'video', videoElement },
    });

    const linkedAudioClip = makeClip({
      id: 'audio-1',
      trackId: 'audio-track',
      linkedClipId: 'video-1',
      source: { type: 'audio', audioElement: { paused: true, src: 'blob:audio-src', readyState: 4 } },
    });

    const ctx = makeFrameContext({
      clips: [videoClip, linkedAudioClip],
      clipsAtTime: [videoClip, linkedAudioClip],
      videoTracks: [{ id: 'video-track', type: 'video', visible: true }],
      audioTracks: [{ id: 'audio-track', type: 'audio', muted: false }],
      visibleVideoTrackIds: new Set(['video-track']),
      unmutedAudioTrackIds: new Set(['audio-track']),
      proxyEnabled: true,
      mediaFiles: [{ id: 'media-1', name: 'clip.mp4', hasProxyAudio: true, proxyStatus: 'ready' }],
      getInterpolatedEffects: (clipId: string) => clipId === 'audio-1'
        ? [
            { id: 'vol-1', type: 'audio-volume', params: { volume: 0.25 } },
            { id: 'eq-1', type: 'audio-eq', params: { band1k: 6 } },
          ]
        : [],
    });

    manager.syncVideoClipAudio(
      ctx,
      { audioPlayingCount: 0, maxAudioDrift: 0, hasAudioError: false, masterSet: false }
    );

    expect(cacheStub.playScrubAudio).toHaveBeenCalledWith(
      'media-1',
      1,
      undefined,
      'blob:video-src',
      expect.objectContaining({
        volume: 0.25,
        eqGains: expect.arrayContaining([6]),
      })
    );
    expect(syncAudioElement).not.toHaveBeenCalled();
    expect(videoElement.muted).toBe(true);

    cacheStub.restore();
  });

  it('suppresses linked audio clip scrub fallback once varispeed scrub audio is ready', () => {
    const manager = new AudioTrackSyncManager() as any;
    const syncAudioElement = vi.fn();
    manager.audioSyncHandler = { syncAudioElement, stopScrubAudio: vi.fn() };

    const cacheStub = stubProxyFrameCache();

    const videoClip = makeClip({
      id: 'video-1',
      trackId: 'video-track',
      linkedClipId: 'audio-1',
      mediaFileId: 'media-1',
      source: { type: 'video', videoElement: { muted: false } },
    });

    const audioElement = {
      paused: true,
      pause: vi.fn(),
      src: 'blob:audio-src',
      readyState: 4,
    } as any;

    const linkedAudioClip = makeClip({
      id: 'audio-1',
      trackId: 'audio-track',
      linkedClipId: 'video-1',
      source: { type: 'audio', audioElement },
    });

    const ctx = makeFrameContext({
      clips: [videoClip, linkedAudioClip],
      clipsAtTime: [videoClip, linkedAudioClip],
      audioTracks: [{ id: 'audio-track', type: 'audio', muted: false }],
      videoTracks: [{ id: 'video-track', type: 'video', visible: true }],
      unmutedAudioTrackIds: new Set(['audio-track']),
      visibleVideoTrackIds: new Set(['video-track']),
      mediaFiles: [{ id: 'media-1', name: 'clip.mp4' }],
    });

    manager.syncAudioTrackClips(
      ctx,
      { audioPlayingCount: 0, maxAudioDrift: 0, hasAudioError: false, masterSet: false }
    );

    expect(syncAudioElement).not.toHaveBeenCalled();
    cacheStub.restore();
  });
});
