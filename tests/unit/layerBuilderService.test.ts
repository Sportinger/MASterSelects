import { beforeEach, describe, expect, it } from 'vitest';
import { LayerBuilderService } from '../../src/services/layerBuilder/LayerBuilderService';
import { useTimelineStore } from '../../src/stores/timeline';
import { useMediaStore } from '../../src/stores/mediaStore';
import { DEFAULT_TRANSFORM } from '../../src/stores/timeline/constants';

const initialTimelineState = useTimelineStore.getState();
const initialMediaState = useMediaStore.getState();

describe('LayerBuilderService paused visual provider selection', () => {
  beforeEach(() => {
    useTimelineStore.setState(initialTimelineState);
    useMediaStore.setState(initialMediaState);
  });

  it('treats a full WebCodecs source as renderable even before the video element is attached', () => {
    const service = new LayerBuilderService() as any;

    expect(
      service.hasRenderableVideoSource({
        webCodecsPlayer: {
          isFullMode: () => true,
        },
      })
    ).toBe(true);
  });

  it('does not treat a source without video element or full WebCodecs player as renderable video', () => {
    const service = new LayerBuilderService() as any;

    expect(
      service.hasRenderableVideoSource({
        webCodecsPlayer: {
          isFullMode: () => false,
        },
      })
    ).toBe(false);
  });

  it('keeps the clip player when the scrub runtime is near the target but has no frame', () => {
    const service = new LayerBuilderService() as any;
    const clipPlayer = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      currentTime: 1,
    };
    const runtimeProvider = {
      isFullMode: () => true,
      hasFrame: () => false,
      getCurrentFrame: () => null,
      currentTime: 1.02,
      getPendingSeekTime: () => 1.02,
    };

    const provider = service.getPausedVisualProvider(
      { webCodecsPlayer: clipPlayer } as any,
      runtimeProvider as any,
      1.01
    );

    expect(provider).toBe(clipPlayer);
  });

  it('uses the scrub runtime once it has a frame near the target', () => {
    const service = new LayerBuilderService() as any;
    const clipPlayer = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 900_000 }),
      currentTime: 0.9,
    };
    const runtimeProvider = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      currentTime: 1.01,
      getPendingSeekTime: () => 1.01,
    };

    const provider = service.getPausedVisualProvider(
      { webCodecsPlayer: clipPlayer } as any,
      runtimeProvider as any,
      1.01
    );

    expect(provider).toBe(runtimeProvider);
  });

  it('prefers the provider whose frame is closer to the paused target', () => {
    const service = new LayerBuilderService() as any;
    const clipPlayer = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 22_589_233 }),
      currentTime: 22.589233,
    };
    const runtimeProvider = {
      isFullMode: () => true,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 8_700_000 }),
      currentTime: 8.7,
      getPendingSeekTime: () => 8.7,
    };

    const provider = service.getPausedVisualProvider(
      { webCodecsPlayer: clipPlayer } as any,
      runtimeProvider as any,
      8.68
    );

    expect(provider).toBe(runtimeProvider);
  });

  it('builds primary layers from timeline clips even when no active composition is selected', () => {
    const service = new LayerBuilderService();
    const clipPlayer = {
      isFullMode: () => true,
      isSimpleMode: () => false,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_000_000 }),
      getPendingSeekTime: () => null,
      getDebugInfo: () => null,
      currentTime: 1,
      isPlaying: false,
      pause: () => {},
      seek: () => {},
    };

    useMediaStore.setState({
      activeCompositionId: null,
      activeLayerSlots: {},
      layerOpacities: {},
      files: [],
      compositions: [],
      proxyEnabled: false,
    } as any);

    useTimelineStore.setState({
      tracks: [
        {
          id: 'track-v1',
          name: 'Video 1',
          type: 'video',
          visible: true,
          muted: false,
          solo: false,
        },
      ],
      clips: [
        {
          id: 'clip-1',
          trackId: 'track-v1',
          name: 'clip.mp4',
          startTime: 0,
          duration: 10,
          inPoint: 0,
          outPoint: 10,
          effects: [],
          transform: { ...DEFAULT_TRANSFORM },
          source: {
            type: 'video',
            naturalDuration: 10,
            webCodecsPlayer: clipPlayer,
          },
          isLoading: false,
        },
      ],
      playheadPosition: 1,
      isPlaying: false,
      isDraggingPlayhead: false,
      playbackSpeed: 1,
    } as any);

    const layers = service.buildLayersFromStore();

    expect(layers).toHaveLength(1);
    expect(layers[0]?.source?.webCodecsPlayer).toBe(clipPlayer);
  });

  it('uses HTML video as the visual source while actively dragging the playhead', () => {
    const service = new LayerBuilderService();
    const videoElement = { currentTime: 1.25 } as HTMLVideoElement;
    const clipPlayer = {
      isFullMode: () => true,
      isSimpleMode: () => false,
      hasFrame: () => true,
      getCurrentFrame: () => ({ timestamp: 1_250_000 }),
      getPendingSeekTime: () => null,
      getDebugInfo: () => null,
      currentTime: 1.25,
      isPlaying: false,
      pause: () => {},
      seek: () => {},
    };

    useMediaStore.setState({
      activeCompositionId: null,
      activeLayerSlots: {},
      layerOpacities: {},
      files: [],
      compositions: [],
      proxyEnabled: false,
    } as any);

    useTimelineStore.setState({
      tracks: [
        {
          id: 'track-v1',
          name: 'Video 1',
          type: 'video',
          visible: true,
          muted: false,
          solo: false,
        },
      ],
      clips: [
        {
          id: 'clip-1',
          trackId: 'track-v1',
          name: 'clip.mp4',
          startTime: 0,
          duration: 10,
          inPoint: 0,
          outPoint: 10,
          effects: [],
          transform: { ...DEFAULT_TRANSFORM },
          source: {
            type: 'video',
            naturalDuration: 10,
            videoElement,
            webCodecsPlayer: clipPlayer,
          },
          isLoading: false,
        },
      ],
      playheadPosition: 1.25,
      isPlaying: false,
      isDraggingPlayhead: true,
      playbackSpeed: 1,
    } as any);

    const layers = service.buildLayersFromStore();

    expect(layers).toHaveLength(1);
    expect(layers[0]?.source?.videoElement).toBe(videoElement);
    expect(layers[0]?.source?.webCodecsPlayer).toBeUndefined();
    expect(layers[0]?.source?.runtimeSessionKey).toBeUndefined();
  });
});
