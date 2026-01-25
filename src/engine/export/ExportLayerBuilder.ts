// Layer building for export rendering

import { Logger } from '../../services/logger';
import type { Layer, NestedCompositionData, BlendMode } from '../../types';

const log = Logger.create('ExportLayerBuilder');
import type { TimelineClip, TimelineTrack } from '../../stores/timeline/types';
import type { ExportClipState, BaseLayerProps, FrameContext } from './types';
import { useMediaStore } from '../../stores/mediaStore';
import { ParallelDecodeManager } from '../ParallelDecodeManager';

// Cache video tracks and solo state at export start (don't change during export)
let cachedVideoTracks: TimelineTrack[] | null = null;
let cachedAnyVideoSolo = false;

export function initializeLayerBuilder(tracks: TimelineTrack[]): void {
  cachedVideoTracks = tracks.filter(t => t.type === 'video');
  cachedAnyVideoSolo = cachedVideoTracks.some(t => t.solo);
}

export function cleanupLayerBuilder(): void {
  cachedVideoTracks = null;
  cachedAnyVideoSolo = false;
}

/**
 * Build layers for rendering at a specific time.
 * Uses FrameContext for O(1) lookups - no getState() calls per frame.
 */
export function buildLayersAtTime(
  ctx: FrameContext,
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Layer[] {
  const { time, clipsByTrack } = ctx;
  const layers: Layer[] = [];

  if (!cachedVideoTracks) {
    log.error('Not initialized - call initializeLayerBuilder first');
    return [];
  }

  const isTrackVisible = (track: TimelineTrack) => {
    if (!track.visible) return false;
    if (cachedAnyVideoSolo) return track.solo;
    return true;
  };

  // Build layers in track order (bottom to top)
  for (let trackIndex = 0; trackIndex < cachedVideoTracks.length; trackIndex++) {
    const track = cachedVideoTracks[trackIndex];
    if (!isTrackVisible(track)) continue;

    // O(1) lookup instead of O(n) find
    const clip = clipsByTrack.get(track.id);
    if (!clip) continue;

    const clipLocalTime = time - clip.startTime;
    const baseLayerProps = buildBaseLayerProps(clip, clipLocalTime, trackIndex, ctx);

    // Handle nested compositions
    if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
      const nestedLayers = buildNestedLayersForExport(clip, clipLocalTime + (clip.inPoint || 0), time, parallelDecoder, useParallelDecode);

      if (nestedLayers.length > 0) {
        const composition = useMediaStore.getState().compositions.find(c => c.id === clip.compositionId);
        const compWidth = composition?.width || 1920;
        const compHeight = composition?.height || 1080;

        const nestedCompData: NestedCompositionData = {
          compositionId: clip.compositionId || clip.id,
          layers: nestedLayers,
          width: compWidth,
          height: compHeight,
        };

        layers.push({
          ...baseLayerProps,
          source: {
            type: 'image', // Nested comps are pre-rendered to texture
            nestedComposition: nestedCompData,
          },
        });
      }
      continue;
    }

    // Handle video clips
    if (clip.source?.type === 'video' && clip.source.videoElement) {
      const layer = buildVideoLayer(clip, baseLayerProps, time, clipStates, parallelDecoder, useParallelDecode);
      if (layer) layers.push(layer);
    }
    // Handle image clips
    else if (clip.source?.type === 'image' && clip.source.imageElement) {
      layers.push({
        ...baseLayerProps,
        source: { type: 'image', imageElement: clip.source.imageElement },
      });
    }
    // Handle text clips
    else if (clip.source?.type === 'text' && clip.source.textCanvas) {
      layers.push({
        ...baseLayerProps,
        source: { type: 'text', textCanvas: clip.source.textCanvas },
      });
    }
  }

  return layers;
}

/**
 * Build base layer properties from clip transform.
 * Uses FrameContext methods for transform/effects interpolation.
 */
function buildBaseLayerProps(
  clip: TimelineClip,
  clipLocalTime: number,
  trackIndex: number,
  ctx: FrameContext
): BaseLayerProps {
  const { getInterpolatedTransform, getInterpolatedEffects } = ctx;

  // Get transform safely with defaults
  let transform;
  try {
    transform = getInterpolatedTransform(clip.id, clipLocalTime);
  } catch (e) {
    log.warn(`Transform interpolation failed for clip ${clip.id}`, e);
    transform = {
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: { x: 0, y: 0, z: 0 },
      opacity: 1,
      blendMode: 'normal' as BlendMode,
    };
  }

  // Get effects safely
  let effects: any[] = [];
  try {
    effects = getInterpolatedEffects(clip.id, clipLocalTime);
  } catch (e) {
    log.warn(`Effects interpolation failed for clip ${clip.id}`, e);
  }

  return {
    id: `export_layer_${trackIndex}`,
    name: clip.name,
    visible: true,
    opacity: transform.opacity ?? 1,
    blendMode: (transform.blendMode || 'normal') as BlendMode,
    effects,
    position: {
      x: transform.position?.x ?? 0,
      y: transform.position?.y ?? 0,
      z: transform.position?.z ?? 0,
    },
    scale: {
      x: transform.scale?.x ?? 1,
      y: transform.scale?.y ?? 1,
    },
    rotation: {
      x: ((transform.rotation?.x ?? 0) * Math.PI) / 180,
      y: ((transform.rotation?.y ?? 0) * Math.PI) / 180,
      z: ((transform.rotation?.z ?? 0) * Math.PI) / 180,
    },
  };
}

/**
 * Build video layer with appropriate source (parallel > webcodecs > HTMLVideoElement).
 */
function buildVideoLayer(
  clip: TimelineClip,
  baseLayerProps: BaseLayerProps,
  time: number,
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Layer | null {
  const video = clip.source!.videoElement!;
  const clipState = clipStates.get(clip.id);

  // Try parallel decoder first
  if (useParallelDecode && parallelDecoder && parallelDecoder.hasClip(clip.id)) {
    const videoFrame = parallelDecoder.getFrameForClip(clip.id, time);
    if (videoFrame) {
      return {
        ...baseLayerProps,
        source: {
          type: 'video',
          videoElement: video,
          videoFrame: videoFrame,
        },
      };
    }
    log.warn(`Parallel decode missing frame for clip "${clip.name}" at time ${time.toFixed(3)}s`);
  }

  // Try sequential WebCodecs VideoFrame
  if (clipState?.isSequential && clipState.webCodecsPlayer) {
    const videoFrame = clipState.webCodecsPlayer.getCurrentFrame();
    if (videoFrame) {
      return {
        ...baseLayerProps,
        source: {
          type: 'video',
          videoElement: video,
          webCodecsPlayer: clipState.webCodecsPlayer,
        },
      };
    }
  }

  // Fall back to HTMLVideoElement
  const videoReady = video.readyState >= 2 && !video.seeking;
  if (videoReady) {
    if (useParallelDecode && parallelDecoder?.hasClip(clip.id)) {
      log.warn(`Falling back to HTMLVideoElement for "${clip.name}" - frame may be incorrect`);
    }
    return {
      ...baseLayerProps,
      source: {
        type: 'video',
        videoElement: video,
      },
    };
  }

  log.warn(`Video not ready for clip ${clip.id}, readyState: ${video.readyState}, seeking: ${video.seeking}`);
  return null;
}

/**
 * Build layers for a nested composition at export time.
 */
function buildNestedLayersForExport(
  clip: TimelineClip,
  nestedTime: number,
  mainTimelineTime: number,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Layer[] {
  if (!clip.nestedClips || !clip.nestedTracks) return [];

  const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
  const layers: Layer[] = [];

  for (let i = 0; i < nestedVideoTracks.length; i++) {
    const nestedTrack = nestedVideoTracks[i];
    const nestedClip = clip.nestedClips.find(
      nc =>
        nc.trackId === nestedTrack.id &&
        nestedTime >= nc.startTime &&
        nestedTime < nc.startTime + nc.duration
    );

    if (!nestedClip) continue;

    const baseLayer = buildNestedBaseLayer(nestedClip);

    // Try parallel decoder first, then video element
    if (nestedClip.source?.videoElement) {
      if (useParallelDecode && parallelDecoder && parallelDecoder.hasClip(nestedClip.id)) {
        const videoFrame = parallelDecoder.getFrameForClip(nestedClip.id, mainTimelineTime);
        if (videoFrame) {
          layers.push({
            ...baseLayer,
            source: {
              type: 'video',
              videoElement: nestedClip.source.videoElement,
              videoFrame: videoFrame,
            },
          } as Layer);
          continue;
        }
        log.warn(`Parallel decode missing frame for nested clip "${nestedClip.name}"`);
      }

      // Fall back to video element
      if (useParallelDecode && parallelDecoder?.hasClip(nestedClip.id)) {
        log.warn(`Falling back to HTMLVideoElement for nested clip "${nestedClip.name}"`);
      }
      layers.push({
        ...baseLayer,
        source: {
          type: 'video',
          videoElement: nestedClip.source.videoElement,
          webCodecsPlayer: nestedClip.source.webCodecsPlayer,
        },
      } as Layer);
    } else if (nestedClip.source?.imageElement) {
      layers.push({
        ...baseLayer,
        source: { type: 'image', imageElement: nestedClip.source.imageElement },
      } as Layer);
    } else if (nestedClip.source?.textCanvas) {
      layers.push({
        ...baseLayer,
        source: { type: 'text', textCanvas: nestedClip.source.textCanvas },
      } as Layer);
    }
  }

  return layers;
}

/**
 * Build base layer for nested clip.
 */
function buildNestedBaseLayer(nestedClip: TimelineClip): BaseLayerProps {
  const transform = nestedClip.transform || {
    position: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1 },
    rotation: { x: 0, y: 0, z: 0 },
    opacity: 1,
    blendMode: 'normal' as BlendMode,
  };

  return {
    id: `nested-export-${nestedClip.id}`,
    name: nestedClip.name,
    visible: true,
    opacity: transform.opacity ?? 1,
    blendMode: (transform.blendMode || 'normal') as BlendMode,
    effects: nestedClip.effects || [],
    position: {
      x: transform.position?.x || 0,
      y: transform.position?.y || 0,
      z: transform.position?.z || 0,
    },
    scale: {
      x: transform.scale?.x ?? 1,
      y: transform.scale?.y ?? 1,
    },
    rotation: {
      x: ((transform.rotation?.x || 0) * Math.PI) / 180,
      y: ((transform.rotation?.y || 0) * Math.PI) / 180,
      z: ((transform.rotation?.z || 0) * Math.PI) / 180,
    },
  };
}
