// Layer building for export rendering

import { Logger } from '../../services/logger';
import type { Layer, NestedCompositionData, BlendMode, ClipTransform } from '../../types';

const log = Logger.create('ExportLayerBuilder');
import type { TimelineClip, TimelineTrack } from '../../stores/timeline/types';
import type { ExportClipState, BaseLayerProps, FrameContext } from './types';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { ParallelDecodeManager } from '../ParallelDecodeManager';
import { getInterpolatedClipTransform } from '../../utils/keyframeInterpolation';
import { DEFAULT_TRANSFORM } from '../../stores/timeline/constants';

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
    // Handle text and solid clips
    else if ((clip.source?.type === 'text' || clip.source?.type === 'solid') && clip.source.textCanvas) {
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
 * Falls back to HTMLVideoElement if other methods fail.
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

  // PARALLEL DECODE MODE - try parallel decode first
  if (useParallelDecode && parallelDecoder) {
    if (parallelDecoder.hasClip(clip.id)) {
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
      // Frame not available - log and fall through to HTMLVideoElement fallback
      log.warn(`Parallel decode frame not available for clip "${clip.name}" at ${time.toFixed(3)}s, using HTMLVideoElement fallback`);
    } else {
      log.warn(`Clip "${clip.name}" not in parallel decoder, using HTMLVideoElement fallback`);
    }
  }

  // SEQUENTIAL MODE (single clip) - use WebCodecs player
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
    log.warn(`Sequential decode frame not available for clip "${clip.name}" at ${time.toFixed(3)}s, using HTMLVideoElement fallback`);
  }

  // FALLBACK: Use HTMLVideoElement directly (less accurate but doesn't fail)
  if (video.readyState >= 2) {
    log.debug(`Using HTMLVideoElement fallback for clip "${clip.name}" at ${time.toFixed(3)}s`);
    return {
      ...baseLayerProps,
      source: {
        type: 'video',
        videoElement: video,
      },
    };
  }

  // Video not ready at all - skip this frame
  log.warn(`Video not ready for clip "${clip.name}" at ${time.toFixed(3)}s (readyState: ${video.readyState}), skipping frame`);
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

  // Filter for video tracks that are visible (default to visible if not explicitly set to false)
  const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible !== false);
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

    // Calculate the clip-local time for keyframe interpolation
    const nestedClipLocalTime = nestedTime - nestedClip.startTime;
    const baseLayer = buildNestedBaseLayer(nestedClip, nestedClipLocalTime);

    // Video clips - try parallel decode first, fallback to HTMLVideoElement
    if (nestedClip.source?.videoElement) {
      if (useParallelDecode && parallelDecoder) {
        // Try parallel decode if clip is registered
        if (parallelDecoder.hasClip(nestedClip.id)) {
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
          // Frame not available - log and fall through to HTMLVideoElement fallback
          log.warn(`Parallel decode frame not available for nested clip "${nestedClip.name}" at ${mainTimelineTime.toFixed(3)}s, using HTMLVideoElement fallback`);
        } else {
          log.warn(`Nested clip "${nestedClip.name}" not in parallel decoder, using HTMLVideoElement fallback`);
        }
      }

      // Fallback: use HTMLVideoElement (less accurate but doesn't fail export)
      const video = nestedClip.source.videoElement;
      if (video.readyState >= 2) {
        layers.push({
          ...baseLayer,
          source: {
            type: 'video',
            videoElement: video,
          },
        } as Layer);
      } else {
        log.warn(`Nested clip "${nestedClip.name}" video not ready (readyState=${video.readyState}), skipping frame`);
      }
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
 * Build base layer for nested clip with keyframe interpolation.
 */
function buildNestedBaseLayer(nestedClip: TimelineClip, nestedClipLocalTime: number): BaseLayerProps {
  // Get keyframes directly from the store (same approach as playback)
  const { clipKeyframes } = useTimelineStore.getState();
  const keyframes = clipKeyframes.get(nestedClip.id) || [];

  // Build base transform from the nested clip's static transform
  const baseTransform: ClipTransform = {
    opacity: nestedClip.transform?.opacity ?? DEFAULT_TRANSFORM.opacity,
    blendMode: nestedClip.transform?.blendMode ?? DEFAULT_TRANSFORM.blendMode,
    position: {
      x: nestedClip.transform?.position?.x ?? DEFAULT_TRANSFORM.position.x,
      y: nestedClip.transform?.position?.y ?? DEFAULT_TRANSFORM.position.y,
      z: nestedClip.transform?.position?.z ?? DEFAULT_TRANSFORM.position.z,
    },
    scale: {
      x: nestedClip.transform?.scale?.x ?? DEFAULT_TRANSFORM.scale.x,
      y: nestedClip.transform?.scale?.y ?? DEFAULT_TRANSFORM.scale.y,
    },
    rotation: {
      x: nestedClip.transform?.rotation?.x ?? DEFAULT_TRANSFORM.rotation.x,
      y: nestedClip.transform?.rotation?.y ?? DEFAULT_TRANSFORM.rotation.y,
      z: nestedClip.transform?.rotation?.z ?? DEFAULT_TRANSFORM.rotation.z,
    },
  };

  // Interpolate transform using keyframes (supports opacity fades, position animations, etc.)
  const transform = keyframes.length > 0
    ? getInterpolatedClipTransform(keyframes, nestedClipLocalTime, baseTransform)
    : baseTransform;

  // Interpolate effect parameters if there are effect keyframes
  const effectKeyframes = keyframes.filter(k => k.property.startsWith('effect.'));
  let effects = nestedClip.effects || [];
  if (effectKeyframes.length > 0 && effects.length > 0) {
    effects = effects.map(effect => {
      const newParams = { ...effect.params };
      Object.keys(effect.params).forEach(paramName => {
        if (typeof effect.params[paramName] !== 'number') return;
        const propertyKey = `effect.${effect.id}.${paramName}`;
        const paramKeyframes = effectKeyframes.filter(k => k.property === propertyKey);
        if (paramKeyframes.length > 0) {
          const sorted = [...paramKeyframes].sort((a, b) => a.time - b.time);
          if (nestedClipLocalTime <= sorted[0].time) {
            newParams[paramName] = sorted[0].value;
          } else if (nestedClipLocalTime >= sorted[sorted.length - 1].time) {
            newParams[paramName] = sorted[sorted.length - 1].value;
          } else {
            for (let i = 0; i < sorted.length - 1; i++) {
              if (nestedClipLocalTime >= sorted[i].time && nestedClipLocalTime <= sorted[i + 1].time) {
                const t = (nestedClipLocalTime - sorted[i].time) / (sorted[i + 1].time - sorted[i].time);
                newParams[paramName] = sorted[i].value + t * (sorted[i + 1].value - sorted[i].value);
                break;
              }
            }
          }
        }
      });
      return { ...effect, params: newParams };
    });
  }

  return {
    id: `nested-export-${nestedClip.id}`,
    name: nestedClip.name,
    visible: true,
    opacity: transform.opacity ?? 1,
    blendMode: (transform.blendMode || 'normal') as BlendMode,
    effects,
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
