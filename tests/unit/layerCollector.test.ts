import { beforeEach, describe, expect, it, vi } from 'vitest';

const hoisted = vi.hoisted(() => ({
  getRuntimeFrameProvider: vi.fn(),
  readRuntimeFrameForSource: vi.fn(),
  wcRecord: vi.fn(),
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

vi.mock('../../src/services/mediaRuntime/runtimePlayback', () => ({
  getRuntimeFrameProvider: (...args: unknown[]) => hoisted.getRuntimeFrameProvider(...args),
  readRuntimeFrameForSource: (...args: unknown[]) => hoisted.readRuntimeFrameForSource(...args),
}));

vi.mock('../../src/services/wcPipelineMonitor', () => ({
  wcPipelineMonitor: {
    record: (...args: unknown[]) => hoisted.wcRecord(...args),
  },
}));

import { LayerCollector } from '../../src/engine/render/LayerCollector';
import { flags } from '../../src/engine/featureFlags';
import { useTimelineStore } from '../../src/stores/timeline';

const defaultUserAgent = navigator.userAgent;

describe('LayerCollector', () => {
  beforeEach(() => {
    hoisted.getRuntimeFrameProvider.mockReset();
    hoisted.readRuntimeFrameForSource.mockReset();
    hoisted.wcRecord.mockReset();
    flags.useFullWebCodecsPlayback = true;
    useTimelineStore.setState({ isDraggingPlayhead: false });
    Object.defineProperty(globalThis.navigator, 'userAgent', {
      configurable: true,
      value: defaultUserAgent,
    });
  });

  it('uses the clip WebCodecs frame while a separate scrub runtime session is still cold', () => {
    const clipFrame = {
      timestamp: 2_000_000,
      displayWidth: 1920,
      displayHeight: 1080,
    };

    const clipProvider = {
      currentTime: 2,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: vi.fn(() => clipFrame),
      getPendingSeekTime: vi.fn(() => null),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };

    const scrubRuntimeProvider = {
      currentTime: 2,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: vi.fn(() => null),
      getPendingSeekTime: vi.fn(() => 3),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };

    hoisted.getRuntimeFrameProvider.mockReturnValue(scrubRuntimeProvider);
    hoisted.readRuntimeFrameForSource.mockReturnValue(null);

    const extTex = { label: 'video-texture' };
    const textureManager = {
      importVideoTexture: vi.fn(() => extTex),
    };

    const layer = {
      id: 'layer-1',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: clipProvider,
        runtimeSourceId: 'media:test',
        runtimeSessionKey: 'interactive-scrub:track-1:media:test',
      },
    } as any;

    const collector = new LayerCollector();
    const result = collector.collect([layer], {
      textureManager: textureManager as any,
      scrubbingCache: null,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: false,
    });

    expect(result).toHaveLength(1);
    expect(textureManager.importVideoTexture).toHaveBeenCalledWith(clipFrame);
    expect(clipProvider.getCurrentFrame).toHaveBeenCalledTimes(1);
    expect(hoisted.readRuntimeFrameForSource).not.toHaveBeenCalled();
    expect(collector.getDecoder()).toBe('WebCodecs');
    expect(collector.hasActiveVideo()).toBe(true);
  });

  it('holds the last successful frame for the same provider while a pending target is still settling', () => {
    const stableFrame = {
      timestamp: 2_000_000,
      displayWidth: 1920,
      displayHeight: 1080,
    };

    const provider = {
      currentTime: 2,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: vi.fn(() => stableFrame),
      getPendingSeekTime: vi.fn(() => null),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };

    hoisted.getRuntimeFrameProvider.mockReturnValue(null);
    hoisted.readRuntimeFrameForSource.mockReturnValue(null);

    const extTex = { label: 'video-texture' };
    const textureManager = {
      importVideoTexture: vi.fn(() => extTex),
    };

    const layer = {
      id: 'layer-1',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: provider,
      },
    } as any;

    const collector = new LayerCollector();
    const deps = {
      textureManager: textureManager as any,
      scrubbingCache: null,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: true,
    };

    const initial = collector.collect([layer], deps);
    expect(initial).toHaveLength(1);

    provider.getPendingSeekTime.mockReturnValue(2.4);

    const pending = collector.collect([layer], deps);
    expect(pending).toHaveLength(1);
    expect(provider.getCurrentFrame).toHaveBeenCalledTimes(2);
  });

  it('does not reuse an unstable frame across a provider change', () => {
    const oldProvider = {
      currentTime: 2,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: vi.fn(() => ({
        timestamp: 2_000_000,
        displayWidth: 1920,
        displayHeight: 1080,
      })),
      getPendingSeekTime: vi.fn(() => null),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };
    const newProvider = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: vi.fn(() => ({
        timestamp: 0,
        displayWidth: 1920,
        displayHeight: 1080,
      })),
      getPendingSeekTime: vi.fn(() => 2.4),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };

    hoisted.getRuntimeFrameProvider.mockReturnValue(null);
    hoisted.readRuntimeFrameForSource.mockReturnValue(null);

    const textureManager = {
      importVideoTexture: vi.fn(() => ({ label: 'video-texture' })),
    };

    const collector = new LayerCollector();
    const deps = {
      textureManager: textureManager as any,
      scrubbingCache: null,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: true,
    };

    collector.collect([{
      id: 'layer-1',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: oldProvider,
      },
    } as any], deps);

    const result = collector.collect([{
      id: 'layer-1',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: newProvider,
      },
    } as any], deps);

    expect(result).toHaveLength(0);
    expect(newProvider.getCurrentFrame).not.toHaveBeenCalled();
  });

  it('does not reuse a pending shared-session frame after the active clip changes on the same layer', () => {
    const sharedProvider = {
      currentTime: 2,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      hasFrame: () => true,
      getCurrentFrame: vi.fn(() => ({
        timestamp: 2_000_000,
        displayWidth: 1920,
        displayHeight: 1080,
      })),
      getPendingSeekTime: vi.fn(() => null),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };

    hoisted.getRuntimeFrameProvider.mockReturnValue(sharedProvider);
    hoisted.readRuntimeFrameForSource.mockReturnValue(null);

    const textureManager = {
      importVideoTexture: vi.fn(() => ({ label: 'video-texture' })),
    };

    const collector = new LayerCollector();
    const deps = {
      textureManager: textureManager as any,
      scrubbingCache: null,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: true,
    };

    collector.collect([{
      id: 'layer-1',
      sourceClipId: 'clip-a',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: sharedProvider,
        runtimeSourceId: 'media:test',
        runtimeSessionKey: 'interactive-track:track-1:media:test',
      },
    } as any], deps);

    sharedProvider.getPendingSeekTime.mockReturnValue(2.4);

    const result = collector.collect([{
      id: 'layer-1',
      sourceClipId: 'clip-b',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: sharedProvider,
        runtimeSourceId: 'media:test',
        runtimeSessionKey: 'interactive-track:track-1:media:test',
      },
    } as any], deps);

    expect(result).toHaveLength(0);
    expect(sharedProvider.getCurrentFrame).toHaveBeenCalledTimes(1);
  });

  it('promotes a stable runtime frame once the scrub session has one, even if the cached layer still points at the clip player', () => {
    const clipProvider = {
      currentTime: 0.9,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      hasFrame: () => false,
      getCurrentFrame: vi.fn(() => null),
      getPendingSeekTime: vi.fn(() => null),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };
    const runtimeFrame = {
      timestamp: 1_000_000,
      displayWidth: 1920,
      displayHeight: 1080,
    };
    const runtimeProvider = {
      currentTime: 1,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      hasFrame: () => true,
      getCurrentFrame: vi.fn(() => runtimeFrame),
      getPendingSeekTime: vi.fn(() => null),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };

    hoisted.getRuntimeFrameProvider.mockReturnValue(runtimeProvider);
    hoisted.readRuntimeFrameForSource.mockReturnValue(null);

    const textureManager = {
      importVideoTexture: vi.fn(() => ({ label: 'video-texture' })),
    };

    const collector = new LayerCollector();
    const result = collector.collect([{
      id: 'layer-1',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: clipProvider,
        runtimeSourceId: 'media:test',
        runtimeSessionKey: 'interactive-scrub:track-1:media:test',
      },
    } as any], {
      textureManager: textureManager as any,
      scrubbingCache: null,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: false,
    });

    expect(result).toHaveLength(1);
    expect(runtimeProvider.getCurrentFrame).toHaveBeenCalledTimes(1);
    expect(clipProvider.getCurrentFrame).not.toHaveBeenCalled();
  });

  it('renders an available pending WebCodecs frame during drag scrubbing instead of dropping to black', () => {
    useTimelineStore.setState({ isDraggingPlayhead: true });

    const frame = {
      timestamp: 2_000_000,
      displayWidth: 1920,
      displayHeight: 1080,
    };

    const provider = {
      currentTime: 2,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      hasFrame: () => true,
      getCurrentFrame: vi.fn(() => frame),
      getPendingSeekTime: vi.fn(() => 2.6),
      getDebugInfo: vi.fn(() => null),
      pause: vi.fn(),
      seek: vi.fn(),
    };

    hoisted.getRuntimeFrameProvider.mockReturnValue(null);
    hoisted.readRuntimeFrameForSource.mockReturnValue(null);

    const textureManager = {
      importVideoTexture: vi.fn(() => ({ label: 'video-texture' })),
    };

    const collector = new LayerCollector();
    const result = collector.collect([{
      id: 'layer-1',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        webCodecsPlayer: provider,
      },
    } as any], {
      textureManager: textureManager as any,
      scrubbingCache: null,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: false,
    });

    expect(result).toHaveLength(1);
    expect(provider.getCurrentFrame).toHaveBeenCalledTimes(1);
  });

  it('prefers HTML video preview when full WebCodecs playback is disabled', () => {
    flags.useFullWebCodecsPlayback = false;

    const video = {
      src: 'blob:test-video',
      currentTime: 1.25,
      readyState: 4,
      seeking: false,
      paused: false,
      videoWidth: 1920,
      videoHeight: 1080,
    } as any;

    const webCodecsPlayer = {
      isFullMode: () => true,
      getCurrentFrame: vi.fn(() => ({
        timestamp: 1_250_000,
        displayWidth: 1920,
        displayHeight: 1080,
      })),
    };

    hoisted.getRuntimeFrameProvider.mockReturnValue(null);
    hoisted.readRuntimeFrameForSource.mockReturnValue(null);

    const textureManager = {
      importVideoTexture: vi.fn(() => ({ label: 'html-video-texture' })),
    };

    const collector = new LayerCollector();
    const result = collector.collect([{
      id: 'layer-html',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        videoElement: video,
        webCodecsPlayer,
      },
    } as any], {
      textureManager: textureManager as any,
      scrubbingCache: null,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: true,
    });

    expect(result).toHaveLength(1);
    expect(textureManager.importVideoTexture).toHaveBeenCalledWith(video);
    expect(webCodecsPlayer.getCurrentFrame).not.toHaveBeenCalled();
    expect(collector.getDecoder()).toBe('HTMLVideo');
  });

  it('tracks paused HTML preview time per video element instead of per shared src', () => {
    flags.useFullWebCodecsPlayback = false;

    const sharedSrc = 'blob:shared-video';
    const firstVideo = {
      src: sharedSrc,
      currentTime: 1.25,
      readyState: 4,
      seeking: false,
      paused: false,
      videoWidth: 1920,
      videoHeight: 1080,
    } as any;
    const secondVideo = {
      src: sharedSrc,
      currentTime: 1.25,
      readyState: 4,
      seeking: false,
      paused: false,
      videoWidth: 1920,
      videoHeight: 1080,
    } as any;

    const textureManager = {
      importVideoTexture: vi.fn((video: HTMLVideoElement) => ({ label: `tex-${video === firstVideo ? 'first' : 'second'}` })),
    };
    const staleSecondFrame = { view: { label: 'stale-second-frame' }, width: 1920, height: 1080 };
    const scrubbingCache = {
      getLastFrame: vi.fn((video: HTMLVideoElement) => video === secondVideo ? staleSecondFrame : null),
      getLastFrameNearTime: vi.fn(() => null),
      getLastPresentedTime: vi.fn((video: HTMLVideoElement) => video.currentTime),
      getLastPresentedOwner: vi.fn(() => undefined),
      getCachedFrame: vi.fn(() => null),
      getLastCaptureTime: vi.fn(() => 0),
      captureVideoFrame: vi.fn(),
      setLastCaptureTime: vi.fn(),
      cacheFrameAtTime: vi.fn(),
    };
    const lastVideoTimes = new Map<string, number>();
    const deps = {
      textureManager: textureManager as any,
      scrubbingCache: scrubbingCache as any,
      getLastVideoTime: (key: string) => lastVideoTimes.get(key),
      setLastVideoTime: (key: string, time: number) => {
        lastVideoTimes.set(key, time);
      },
      isExporting: false,
      isPlaying: false,
    };

    const collector = new LayerCollector();

    collector.collect([{
      id: 'layer-first',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        videoElement: firstVideo,
      },
    } as any], deps);

    const result = collector.collect([{
      id: 'layer-second',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        videoElement: secondVideo,
      },
    } as any], deps);

    expect(result).toHaveLength(1);
    expect(textureManager.importVideoTexture).toHaveBeenCalledTimes(2);
    expect(textureManager.importVideoTexture).toHaveBeenLastCalledWith(secondVideo);
    expect(result[0]?.externalTexture).toEqual({ label: 'tex-second' });
  });

  it('reports live HTML playback time instead of a stale last-presented timestamp', () => {
    flags.useFullWebCodecsPlayback = false;

    const video = {
      src: 'blob:test-video',
      currentTime: 4.5,
      readyState: 4,
      seeking: false,
      paused: false,
      videoWidth: 1920,
      videoHeight: 1080,
    } as any;

    const textureManager = {
      importVideoTexture: vi.fn(() => ({ label: 'html-video-texture' })),
    };
    const scrubbingCache = {
      getLastPresentedTime: vi.fn(() => 1.25),
      getLastPresentedOwner: vi.fn(() => undefined),
      getLastFrame: vi.fn(() => null),
      getLastFrameNearTime: vi.fn(() => null),
      getCachedFrameEntry: vi.fn(() => null),
      getNearestCachedFrameEntry: vi.fn(() => null),
      getLastCaptureTime: vi.fn(() => 0),
      captureVideoFrame: vi.fn(),
      setLastCaptureTime: vi.fn(),
      cacheFrameAtTime: vi.fn(),
      captureVideoFrameIfCloser: vi.fn(),
    };

    const collector = new LayerCollector();
    const result = collector.collect([{
      id: 'layer-live-html',
      sourceClipId: 'clip-live-html',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        videoElement: video,
      },
    } as any], {
      textureManager: textureManager as any,
      scrubbingCache: scrubbingCache as any,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: true,
    });

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      isVideo: true,
      externalTexture: { label: 'html-video-texture' },
      displayedMediaTime: 4.5,
      targetMediaTime: 4.5,
      previewPath: 'live-import',
    });
  });

  it('uses copied HTML video frames on Firefox instead of external textures', () => {
    flags.useFullWebCodecsPlayback = false;
    Object.defineProperty(globalThis.navigator, 'userAgent', {
      configurable: true,
      value: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:145.0) Gecko/20100101 Firefox/145.0',
    });

    const video = {
      src: 'blob:test-video',
      currentTime: 1.25,
      readyState: 4,
      seeking: false,
      paused: false,
      videoWidth: 1920,
      videoHeight: 1080,
    } as any;

    const copiedFrame = { view: { label: 'copied-frame' }, width: 1920, height: 1080 };
    const textureManager = {
      importVideoTexture: vi.fn(() => ({ label: 'html-video-texture' })),
    };
    const scrubbingCache = {
      captureVideoFrame: vi.fn(() => true),
      getLastFrame: vi.fn(() => copiedFrame),
      getLastFrameNearTime: vi.fn(() => copiedFrame),
      getLastPresentedTime: vi.fn(() => 1.25),
      getLastPresentedOwner: vi.fn(() => undefined),
      getCachedFrame: vi.fn(() => null),
      getLastCaptureTime: vi.fn(() => 0),
      setLastCaptureTime: vi.fn(),
      cacheFrameAtTime: vi.fn(),
    };

    const collector = new LayerCollector();
    const result = collector.collect([{
      id: 'layer-firefox',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        videoElement: video,
      },
    } as any], {
      textureManager: textureManager as any,
      scrubbingCache: scrubbingCache as any,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: true,
    });

    expect(result).toHaveLength(1);
    expect(scrubbingCache.captureVideoFrame).toHaveBeenCalledWith(video, undefined);
    expect(textureManager.importVideoTexture).not.toHaveBeenCalled();
    expect(result[0]).toMatchObject({
      isVideo: false,
      externalTexture: null,
      textureView: copiedFrame.view,
      sourceWidth: copiedFrame.width,
      sourceHeight: copiedFrame.height,
    });
    expect(collector.getDecoder()).toBe('HTMLVideo');
    expect(collector.hasActiveVideo()).toBe(true);
  });

  it('keeps the last same-clip frame during hard scrubs instead of dropping to black', () => {
    flags.useFullWebCodecsPlayback = false;
    useTimelineStore.setState({ isDraggingPlayhead: true });

    const video = {
      src: 'blob:test-video',
      currentTime: 18,
      readyState: 1,
      seeking: true,
      paused: true,
      videoWidth: 1920,
      videoHeight: 1080,
    } as any;

    const heldFrame = {
      view: { label: 'held-same-clip-frame' },
      width: 1920,
      height: 1080,
      mediaTime: 2.5,
    };
    const textureManager = {
      importVideoTexture: vi.fn(() => null),
    };
    const scrubbingCache = {
      getLastPresentedTime: vi.fn(() => undefined),
      getLastPresentedOwner: vi.fn(() => undefined),
      getLastFrame: vi.fn(() => heldFrame),
      getLastFrameNearTime: vi.fn(() => null),
      getCachedFrameEntry: vi.fn(() => null),
      getNearestCachedFrameEntry: vi.fn(() => null),
      getLastCaptureTime: vi.fn(() => 0),
      captureVideoFrame: vi.fn(),
      setLastCaptureTime: vi.fn(),
      cacheFrameAtTime: vi.fn(),
      captureVideoFrameIfCloser: vi.fn(),
    };

    const collector = new LayerCollector();
    const result = collector.collect([{
      id: 'layer-hard-scrub',
      sourceClipId: 'clip-hard-scrub',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        mediaTime: 18,
        videoElement: video,
      },
    } as any], {
      textureManager: textureManager as any,
      scrubbingCache: scrubbingCache as any,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: false,
    });

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      isVideo: false,
      externalTexture: null,
      textureView: heldFrame.view,
      sourceWidth: heldFrame.width,
      sourceHeight: heldFrame.height,
      displayedMediaTime: heldFrame.mediaTime,
      targetMediaTime: 18,
      previewPath: 'same-clip-hold',
    });
  });

  it('uses a same-clip hold as the last fallback when a seeked HTML frame cannot import', () => {
    flags.useFullWebCodecsPlayback = false;
    useTimelineStore.setState({ isDraggingPlayhead: true });

    const video = {
      src: 'blob:test-video',
      currentTime: 25,
      readyState: 4,
      seeking: true,
      paused: true,
      videoWidth: 1920,
      videoHeight: 1080,
    } as any;

    const heldFrame = {
      view: { label: 'held-seeking-frame' },
      width: 1920,
      height: 1080,
      mediaTime: 7.25,
    };
    const textureManager = {
      importVideoTexture: vi.fn(() => null),
    };
    const scrubbingCache = {
      getLastPresentedTime: vi.fn(() => undefined),
      getLastPresentedOwner: vi.fn(() => undefined),
      getLastFrame: vi.fn(() => heldFrame),
      getLastFrameNearTime: vi.fn(() => null),
      getCachedFrameEntry: vi.fn(() => null),
      getNearestCachedFrameEntry: vi.fn(() => null),
      getLastCaptureTime: vi.fn(() => 0),
      captureVideoFrame: vi.fn(),
      setLastCaptureTime: vi.fn(),
      cacheFrameAtTime: vi.fn(),
      captureVideoFrameIfCloser: vi.fn(),
    };

    const collector = new LayerCollector();
    const result = collector.collect([{
      id: 'layer-seek-fallback',
      sourceClipId: 'clip-seek-fallback',
      name: 'Video',
      visible: true,
      opacity: 1,
      blendMode: 'normal',
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      source: {
        type: 'video',
        mediaTime: 25,
        videoElement: video,
      },
    } as any], {
      textureManager: textureManager as any,
      scrubbingCache: scrubbingCache as any,
      getLastVideoTime: () => undefined,
      setLastVideoTime: () => {},
      isExporting: false,
      isPlaying: false,
    });

    expect(result).toHaveLength(1);
    expect(textureManager.importVideoTexture).not.toHaveBeenCalled();
    expect(result[0]).toMatchObject({
      isVideo: false,
      externalTexture: null,
      textureView: heldFrame.view,
      sourceWidth: heldFrame.width,
      sourceHeight: heldFrame.height,
      displayedMediaTime: heldFrame.mediaTime,
      targetMediaTime: 25,
      previewPath: 'same-clip-hold',
    });
  });
});
