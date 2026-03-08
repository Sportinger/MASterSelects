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
import { useTimelineStore } from '../../src/stores/timeline';

describe('LayerCollector', () => {
  beforeEach(() => {
    hoisted.getRuntimeFrameProvider.mockReset();
    hoisted.readRuntimeFrameForSource.mockReset();
    hoisted.wcRecord.mockReset();
    useTimelineStore.setState({ isDraggingPlayhead: false });
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
});
