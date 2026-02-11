// LayerBuilderService - Main orchestrator for layer building
// Uses modular components for caching, transforms, and audio sync

import type { TimelineClip, Layer, NestedCompositionData, BlendMode, ClipTransform } from '../../types';
import type { FrameContext, NativeDecoderState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { playheadState } from './PlayheadState';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip, isVideoTrackVisible, getClipForTrack } from './FrameContext';
import { LayerCache } from './LayerCache';
import { TransformCache } from './TransformCache';
import { AudioSyncHandler, createAudioSyncState, finalizeAudioSync, resumeAudioContextIfNeeded } from './AudioSyncHandler';
import { proxyFrameCache } from '../proxyFrameCache';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { Logger } from '../logger';
import { getInterpolatedClipTransform } from '../../utils/keyframeInterpolation';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { DEFAULT_TRANSFORM } from '../../stores/timeline/constants';
import { engine } from '../../engine/WebGPUEngine';

const log = Logger.create('LayerBuilder');

/**
 * Get interpolated volume for a clip from audio-volume effect
 */
function getClipVolume(ctx: FrameContext, clip: TimelineClip, clipLocalTime: number): number {
  const effects = ctx.getInterpolatedEffects(clip.id, clipLocalTime);
  const volumeEffect = effects.find(e => e.type === 'audio-volume');
  return (volumeEffect?.params?.volume as number) ?? 1;
}

// EQ band parameter names (matching audio-eq effect)
const EQ_BAND_PARAMS = [
  'band31', 'band62', 'band125', 'band250', 'band500',
  'band1k', 'band2k', 'band4k', 'band8k', 'band16k'
];

/**
 * Get interpolated EQ gains for a clip from audio-eq effect
 * Returns array of 10 gain values in dB, or undefined if no EQ effect
 */
function getClipEQGains(ctx: FrameContext, clip: TimelineClip, clipLocalTime: number): number[] | undefined {
  const effects = ctx.getInterpolatedEffects(clip.id, clipLocalTime);
  const eqEffect = effects.find(e => e.type === 'audio-eq');
  if (!eqEffect) return undefined;

  return EQ_BAND_PARAMS.map(param => (eqEffect.params?.[param] as number) ?? 0);
}


/**
 * LayerBuilderService - Builds render layers from timeline state
 * Optimized with caching, memoization, and object reuse
 */
export class LayerBuilderService {
  // Sub-modules
  private layerCache = new LayerCache();
  private transformCache = new TransformCache();
  private audioSyncHandler = new AudioSyncHandler();

  // Native decoder state
  private nativeDecoderState = new Map<string, NativeDecoderState>();

  // Video sync throttling
  private lastVideoSyncFrame = -1;
  private lastVideoSyncPlaying = false;
  private lastSeekRef: Record<string, number> = {};

  // Audio sync throttling
  private lastAudioSyncTime = 0;
  private playbackStartFrames = 0;

  // Proxy frame refs for fallback
  private proxyFramesRef = new Map<string, { frameIndex: number; image: HTMLImageElement }>();

  // Lookahead preloading
  private lastLookaheadTime = 0;

  // Active audio proxies tracking
  private activeAudioProxies = new Map<string, HTMLAudioElement>();

  // Videos currently being warmed up (brief play to activate GPU surface)
  // After page reload, video GPU surfaces are empty — all sync rendering APIs
  // (importExternalTexture, canvas.drawImage, copyExternalImageToTexture) return black.
  // The ONLY way to populate the GPU surface is video.play().
  // We do this lazily on first scrub attempt, not during restore, because
  // the render loop's syncClipVideo would immediately pause the warmup video.
  private warmingUpVideos = new WeakSet<HTMLVideoElement>();

  /**
   * Invalidate all caches (layer cache and transform cache)
   */
  invalidateCache(): void {
    this.layerCache.invalidate();
    this.transformCache.clear();
  }

  /**
   * Build layers for the current frame
   * Main entry point - called from render loop
   */
  buildLayersFromStore(): Layer[] {
    // Create frame context (single store read)
    const ctx = createFrameContext();

    // Check cache (only for primary layers — background layers are cheap to rebuild)
    const cacheResult = this.layerCache.checkCache(ctx);
    let primaryLayers: Layer[];
    if (cacheResult.useCache) {
      primaryLayers = cacheResult.layers;
    } else {
      primaryLayers = this.buildLayers(ctx);
      // Preload upcoming nested comp frames during playback
      if (ctx.isPlaying) {
        this.preloadUpcomingNestedCompFrames(ctx);
      }
    }

    // Merge background layers from active layer slots
    const mergedLayers = this.mergeBackgroundLayers(primaryLayers, ctx.playheadPosition);

    // Cache merged result
    this.layerCache.setCachedLayers(mergedLayers);
    return mergedLayers;
  }

  /**
   * Merge primary (editor) layers with background composition layers.
   * Render order: D (bottom) → C → B → A (top)
   * The primary composition's layers go at the position of its layer slot.
   */
  private mergeBackgroundLayers(primaryLayers: Layer[], playheadPosition: number): Layer[] {
    const { activeLayerSlots, activeCompositionId } = useMediaStore.getState();
    const slotEntries = Object.entries(activeLayerSlots);

    // No active layer slots → return primary layers as-is (backwards compatible)
    if (slotEntries.length === 0) {
      return primaryLayers;
    }

    // Find which layer the primary (editor) composition is on
    let primaryLayerIndex = -1;
    for (const [key, compId] of slotEntries) {
      if (compId === activeCompositionId) {
        primaryLayerIndex = Number(key);
        break;
      }
    }

    // Collect all layer indices, sorted A=0 (top) → D=3 (bottom)
    // layers[0] is rendered last (on top) by the compositor's reverse iteration
    const layerIndices = slotEntries
      .map(([key]) => Number(key))
      .sort((a, b) => a - b); // Ascending: A=0 first (top) → D=3 last (bottom)

    const merged: Layer[] = [];

    for (const layerIndex of layerIndices) {
      if (layerIndex === primaryLayerIndex) {
        // Insert primary layers at this position
        merged.push(...primaryLayers);
      } else {
        // Build background layer from LayerPlaybackManager
        const bgLayer = layerPlaybackManager.buildLayersForLayer(layerIndex, playheadPosition);
        if (bgLayer) {
          merged.push(bgLayer);
        }
      }
    }

    // If primary comp is not in any slot, add its layers on top
    if (primaryLayerIndex === -1 && primaryLayers.length > 0) {
      merged.push(...primaryLayers);
    }

    return merged;
  }

  /**
   * Build layers from frame context
   * Handles transitions by rendering both clips with crossfade opacity
   */
  private buildLayers(ctx: FrameContext): Layer[] {
    const layers: Layer[] = [];

    ctx.videoTracks.forEach((track, layerIndex) => {
      if (!isVideoTrackVisible(ctx, track.id)) {
        return;
      }

      // Get all clips on this track at the current time
      const trackClips = ctx.clipsAtTime.filter(c => c.trackId === track.id);

      if (trackClips.length === 0) return;

      // Check if we're in a transition (two clips overlapping with transition data)
      if (trackClips.length >= 2) {
        // Sort by start time to get outgoing (earlier) and incoming (later) clips
        trackClips.sort((a, b) => a.startTime - b.startTime);
        const outgoingClip = trackClips[0];
        const incomingClip = trackClips[1];

        // Check if they have transition data linking them
        if (outgoingClip.transitionOut && outgoingClip.transitionOut.linkedClipId === incomingClip.id) {
          // We're in a transition! Build both layers with adjusted opacity
          const transitionDuration = outgoingClip.transitionOut.duration;
          const transitionStart = incomingClip.startTime;

          // Calculate transition progress (0 = start, 1 = end)
          const progress = Math.max(0, Math.min(1,
            (ctx.playheadPosition - transitionStart) / transitionDuration
          ));

          // Outgoing clip: opacity fades from 1 to 0
          const outgoingOpacity = 1 - progress;
          // Incoming clip: opacity fades from 0 to 1
          const incomingOpacity = progress;

          // Build outgoing clip layer (rendered first, behind)
          const outgoingLayer = this.buildLayerForClip(outgoingClip, layerIndex, ctx, outgoingOpacity);
          if (outgoingLayer) {
            layers.push(outgoingLayer);
          }

          // Build incoming clip layer (rendered second, on top)
          const incomingLayer = this.buildLayerForClip(incomingClip, layerIndex, ctx, incomingOpacity);
          if (incomingLayer) {
            layers.push(incomingLayer);
          }

          return; // Skip normal single-clip handling
        }
      }

      // Normal case: single clip or no transition
      const clip = trackClips[0];
      const layer = this.buildLayerForClip(clip, layerIndex, ctx);
      if (layer) {
        layers[layerIndex] = layer;
      }
    });

    return layers;
  }

  /**
   * Build a layer for a clip based on its type
   * @param opacityOverride - Optional opacity override for transitions (0-1)
   */
  private buildLayerForClip(clip: TimelineClip, layerIndex: number, ctx: FrameContext, opacityOverride?: number): Layer | null {
    // Nested composition
    if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
      return this.buildNestedCompLayer(clip, layerIndex, ctx, opacityOverride);
    }

    // Native decoder (ProRes/DNxHD turbo mode)
    if (clip.source?.nativeDecoder) {
      return this.buildNativeDecoderLayer(clip, layerIndex, ctx, opacityOverride);
    }

    // Video clip
    if (clip.source?.videoElement) {
      return this.buildVideoLayer(clip, layerIndex, ctx, opacityOverride);
    }

    // Image clip
    if (clip.source?.imageElement) {
      return this.buildImageLayer(clip, layerIndex, ctx, opacityOverride);
    }

    // Text clip
    if (clip.source?.textCanvas) {
      return this.buildTextLayer(clip, layerIndex, ctx, opacityOverride);
    }

    return null;
  }

  /**
   * Build nested composition layer
   */
  private buildNestedCompLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext, opacityOverride?: number): Layer | null {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const nestedLayers = this.buildNestedLayers(clip, timeInfo.clipTime, ctx);

    if (nestedLayers.length === 0) return null;

    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipTime);

    const composition = ctx.compositionById.get(clip.compositionId || '');
    const compWidth = composition?.width || 1920;
    const compHeight = composition?.height || 1080;

    const nestedCompData: NestedCompositionData = {
      compositionId: clip.compositionId || clip.id,
      layers: nestedLayers,
      width: compWidth,
      height: compHeight,
      currentTime: ctx.playheadPosition,
    };

    // Apply transition opacity override if provided
    const finalOpacity = opacityOverride !== undefined
      ? transform.opacity * opacityOverride
      : transform.opacity;

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}_${clip.id}`,
      name: clip.name,
      visible: true,
      opacity: finalOpacity,
      blendMode: transform.blendMode as BlendMode,
      source: { type: 'image', nestedComposition: nestedCompData },
      effects,
      position: transform.position,
      scale: transform.scale,
      rotation: transform.rotation,
    };

    this.addMaskProperties(layer, clip);
    return layer;
  }

  /**
   * Build native decoder layer
   */
  private buildNativeDecoderLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext, opacityOverride?: number): Layer {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    // Apply transition opacity override if provided
    const finalOpacity = opacityOverride !== undefined
      ? transform.opacity * opacityOverride
      : transform.opacity;

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}_${clip.id}`,
      name: clip.name,
      visible: true,
      opacity: finalOpacity,
      blendMode: transform.blendMode as BlendMode,
      source: { type: 'video', nativeDecoder: clip.source!.nativeDecoder },
      effects,
      position: transform.position,
      scale: transform.scale,
      rotation: transform.rotation,
    };

    this.addMaskProperties(layer, clip);
    return layer;
  }

  /**
   * Build video layer (with proxy support)
   */
  private buildVideoLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext, opacityOverride?: number): Layer | null {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const mediaFile = getMediaFileForClip(ctx, clip);

    // Check for proxy usage
    if (ctx.proxyEnabled && mediaFile?.proxyFps) {
      const proxyLayer = this.tryBuildProxyLayer(clip, layerIndex, timeInfo.clipTime, mediaFile, ctx, opacityOverride);
      if (proxyLayer) return proxyLayer;
    }

    // Direct video layer
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    // Apply transition opacity override if provided
    const finalOpacity = opacityOverride !== undefined
      ? transform.opacity * opacityOverride
      : transform.opacity;

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: finalOpacity,
      blendMode: transform.blendMode as BlendMode,
      source: {
        type: 'video',
        videoElement: clip.source!.videoElement,
        webCodecsPlayer: clip.source?.webCodecsPlayer,
      },
      effects,
      position: transform.position,
      scale: transform.scale,
      rotation: transform.rotation,
    };

    this.addMaskProperties(layer, clip);
    return layer;
  }

  /**
   * Try to build proxy layer, returns null if not available
   */
  private tryBuildProxyLayer(
    clip: TimelineClip,
    layerIndex: number,
    clipTime: number,
    mediaFile: any,
    ctx: FrameContext,
    opacityOverride?: number
  ): Layer | null {
    const proxyFps = mediaFile.proxyFps || 30;
    const frameIndex = Math.floor(clipTime * proxyFps);

    // Check proxy availability
    let useProxy = false;
    if (mediaFile.proxyStatus === 'ready') {
      useProxy = true;
    } else if (mediaFile.proxyStatus === 'generating' && (mediaFile.proxyProgress || 0) > 0) {
      const totalFrames = Math.ceil((mediaFile.duration || 10) * proxyFps);
      const maxGeneratedFrame = Math.floor(totalFrames * ((mediaFile.proxyProgress || 0) / 100));
      useProxy = frameIndex < maxGeneratedFrame;
    }

    if (!useProxy) return null;

    // Try to get cached frame
    const cacheKey = `${mediaFile.id}_${clip.id}`;
    const cachedFrame = proxyFrameCache.getCachedFrame(mediaFile.id, frameIndex, proxyFps);

    if (cachedFrame) {
      this.proxyFramesRef.set(cacheKey, { frameIndex, image: cachedFrame });
      return this.buildImageLayerFromElement(clip, layerIndex, cachedFrame, clipTime, ctx, opacityOverride);
    }

    // Try to get nearest cached frame for smooth scrubbing
    const nearestFrame = proxyFrameCache.getNearestCachedFrame(mediaFile.id, frameIndex, 30);
    if (nearestFrame) {
      return this.buildImageLayerFromElement(clip, layerIndex, nearestFrame, clipTime, ctx, opacityOverride);
    }

    // Use previous cached frame as fallback
    const cached = this.proxyFramesRef.get(cacheKey);
    if (cached?.image) {
      return this.buildImageLayerFromElement(clip, layerIndex, cached.image, clipTime, ctx, opacityOverride);
    }

    // No proxy frame available - return null to fall back to video
    return null;
  }

  /**
   * Build image layer
   */
  private buildImageLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext, opacityOverride?: number): Layer {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    // Apply transition opacity override if provided
    const finalOpacity = opacityOverride !== undefined
      ? transform.opacity * opacityOverride
      : transform.opacity;

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: finalOpacity,
      blendMode: transform.blendMode as BlendMode,
      source: { type: 'image', imageElement: clip.source!.imageElement },
      effects,
      position: transform.position,
      scale: transform.scale,
      rotation: transform.rotation,
    };

    this.addMaskProperties(layer, clip);
    return layer;
  }

  /**
   * Build image layer from an image element (for proxy frames)
   */
  private buildImageLayerFromElement(
    clip: TimelineClip,
    layerIndex: number,
    imageElement: HTMLImageElement,
    localTime: number,
    ctx: FrameContext,
    opacityOverride?: number
  ): Layer {
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, localTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, localTime);

    // Apply transition opacity override if provided
    const finalOpacity = opacityOverride !== undefined
      ? transform.opacity * opacityOverride
      : transform.opacity;

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: finalOpacity,
      blendMode: transform.blendMode as BlendMode,
      source: { type: 'image', imageElement },
      effects,
      position: transform.position,
      scale: transform.scale,
      rotation: transform.rotation,
    };

    this.addMaskProperties(layer, clip);
    return layer;
  }

  /**
   * Build text layer
   */
  private buildTextLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext, opacityOverride?: number): Layer {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    // Apply transition opacity override if provided
    const finalOpacity = opacityOverride !== undefined
      ? transform.opacity * opacityOverride
      : transform.opacity;

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: finalOpacity,
      blendMode: transform.blendMode as BlendMode,
      source: { type: 'text', textCanvas: clip.source!.textCanvas },
      effects,
      position: transform.position,
      scale: transform.scale,
      rotation: transform.rotation,
    };

    this.addMaskProperties(layer, clip);
    return layer;
  }

  /**
   * Add mask properties to layer if clip has masks
   */
  private addMaskProperties(layer: Layer, clip: TimelineClip): void {
    if (clip.masks && clip.masks.length > 0) {
      layer.maskClipId = clip.id;
      layer.maskInvert = clip.masks.some(m => m.inverted);
    }
  }

  /**
   * Build nested layers (simplified - delegates to separate method for full implementation)
   */
  private buildNestedLayers(clip: TimelineClip, clipTime: number, ctx: FrameContext): Layer[] {
    if (!clip.nestedClips || !clip.nestedTracks) return [];

    // Filter for video tracks that are visible (default to visible if not explicitly set)
    const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible !== false);
    const layers: Layer[] = [];

    // Debug: log nested clip info once per second
    if (Math.floor(ctx.now / 1000) !== Math.floor((ctx.now - 16) / 1000)) {
      log.info('buildNestedLayers', {
        compClipId: clip.id,
        clipTime,
        nestedTrackCount: clip.nestedTracks.length,
        nestedVideoTrackCount: nestedVideoTracks.length,
        nestedTracks: clip.nestedTracks.map(t => ({ id: t.id, type: t.type, visible: t.visible })),
        nestedClipCount: clip.nestedClips.length,
        nestedClips: clip.nestedClips.map(nc => ({
          id: nc.id,
          name: nc.name,
          trackId: nc.trackId,
          startTime: nc.startTime,
          duration: nc.duration,
          isLoading: nc.isLoading,
          hasVideoElement: !!nc.source?.videoElement,
        })),
      });
    }

    // Iterate forwards to maintain correct layer order (track 0 = bottom, track N = top)
    for (let i = 0; i < nestedVideoTracks.length; i++) {
      const nestedTrack = nestedVideoTracks[i];
      const nestedClip = clip.nestedClips.find(
        nc =>
          nc.trackId === nestedTrack.id &&
          clipTime >= nc.startTime &&
          clipTime < nc.startTime + nc.duration
      );

      if (!nestedClip) {
        // Log why no clip was found for this track
        const clipsOnTrack = clip.nestedClips.filter(nc => nc.trackId === nestedTrack.id);
        if (clipsOnTrack.length > 0) {
          log.debug('No active clip on track at time', {
            trackId: nestedTrack.id,
            clipTime,
            clipsOnTrack: clipsOnTrack.map(nc => ({
              name: nc.name,
              startTime: nc.startTime,
              endTime: nc.startTime + nc.duration,
            })),
          });
        }
        continue;
      }

      // nestedLocalTime is the time within the clip (0 to duration) - used for keyframe interpolation
      const nestedLocalTime = clipTime - nestedClip.startTime;

      // Build layer based on source type (pass nestedLocalTime for keyframe interpolation)
      const nestedLayer = this.buildNestedClipLayer(nestedClip, nestedLocalTime, ctx);
      if (nestedLayer) {
        layers.push(nestedLayer);
      } else {
        log.debug('Failed to build nested layer', {
          clipId: nestedClip.id,
          name: nestedClip.name,
          isLoading: nestedClip.isLoading,
          hasVideoElement: !!nestedClip.source?.videoElement,
          hasImageElement: !!nestedClip.source?.imageElement,
          videoReadyState: nestedClip.source?.videoElement?.readyState,
        });
      }
    }

    return layers;
  }

  /**
   * Build layer for a nested clip
   */
  private buildNestedClipLayer(nestedClip: TimelineClip, nestedClipLocalTime: number, _ctx: FrameContext): Layer | null {
    // Get keyframes directly from the store (nested clips aren't in ctx.clips, so we can't use ctx.getInterpolatedTransform)
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
            // Simple linear interpolation for effect params
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

    const baseLayer = {
      id: `nested-layer-${nestedClip.id}`,
      name: nestedClip.name,
      visible: true,
      opacity: transform.opacity ?? 1,
      blendMode: transform.blendMode || 'normal',
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

    // Add mask properties
    if (nestedClip.masks && nestedClip.masks.length > 0) {
      (baseLayer as any).maskClipId = nestedClip.id;
      (baseLayer as any).maskInvert = nestedClip.masks.some(m => m.inverted);
    }

    // Skip clips that are still loading
    if (nestedClip.isLoading) {
      return null;
    }

    if (nestedClip.source?.videoElement) {
      return {
        ...baseLayer,
        source: {
          type: 'video',
          videoElement: nestedClip.source.videoElement,
          webCodecsPlayer: nestedClip.source.webCodecsPlayer,
        },
      } as Layer;
    } else if (nestedClip.source?.imageElement) {
      return {
        ...baseLayer,
        source: { type: 'image', imageElement: nestedClip.source.imageElement },
      } as Layer;
    }

    return null;
  }

  /**
   * Preload proxy frames for upcoming nested compositions
   */
  private preloadUpcomingNestedCompFrames(ctx: FrameContext): void {
    if (ctx.now - this.lastLookaheadTime < LAYER_BUILDER_CONSTANTS.LOOKAHEAD_INTERVAL) {
      return;
    }
    this.lastLookaheadTime = ctx.now;

    const lookaheadEnd = ctx.playheadPosition + LAYER_BUILDER_CONSTANTS.LOOKAHEAD_SECONDS;

    // Find upcoming nested comps
    const upcomingNestedComps = ctx.clips.filter(clip =>
      clip.isComposition &&
      clip.nestedClips &&
      clip.nestedClips.length > 0 &&
      clip.startTime > ctx.playheadPosition &&
      clip.startTime < lookaheadEnd
    );

    for (const nestedCompClip of upcomingNestedComps) {
      this.preloadNestedCompFrames(nestedCompClip, ctx);
    }
  }

  /**
   * Preload frames for a specific nested composition
   */
  private preloadNestedCompFrames(nestedCompClip: TimelineClip, ctx: FrameContext): void {
    if (!nestedCompClip.nestedClips) return;

    const nestedStartTime = nestedCompClip.inPoint || 0;

    for (const nestedClip of nestedCompClip.nestedClips) {
      if (!nestedClip.source?.videoElement) continue;

      // Check if active at start
      if (nestedStartTime < nestedClip.startTime ||
          nestedStartTime >= nestedClip.startTime + nestedClip.duration) {
        continue;
      }

      const mediaFile = getMediaFileForClip(ctx, nestedClip);
      if (!mediaFile?.proxyFps) continue;
      if (mediaFile.proxyStatus !== 'ready' && mediaFile.proxyStatus !== 'generating') continue;

      // Calculate frame to preload
      const nestedLocalTime = nestedStartTime - nestedClip.startTime;
      const nestedClipTime = nestedClip.reversed
        ? nestedClip.outPoint - nestedLocalTime
        : nestedLocalTime + nestedClip.inPoint;

      const proxyFps = mediaFile.proxyFps;
      const frameIndex = Math.floor(nestedClipTime * proxyFps);

      // Preload 60 frames
      const framesToPreload = Math.min(60, Math.ceil(proxyFps * 2));
      for (let i = 0; i < framesToPreload; i++) {
        proxyFrameCache.getCachedFrame(mediaFile.id, frameIndex + i, proxyFps);
      }
    }
  }

  // ==================== VIDEO SYNC ====================

  /**
   * Sync video elements to current playhead
   */
  syncVideoElements(): void {
    const ctx = createFrameContext();

    // Skip if same frame during playback
    if (ctx.isPlaying && !ctx.isDraggingPlayhead &&
        ctx.frameNumber === this.lastVideoSyncFrame &&
        ctx.isPlaying === this.lastVideoSyncPlaying) {
      return;
    }
    this.lastVideoSyncFrame = ctx.frameNumber;
    this.lastVideoSyncPlaying = ctx.isPlaying;

    // Sync each clip at playhead
    for (const clip of ctx.clipsAtTime) {
      this.syncClipVideo(clip, ctx);

      // Sync nested composition videos
      if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        this.syncNestedCompVideos(clip, ctx);
      }
    }

    // Pause videos not at playhead
    for (const clip of ctx.clips) {
      if (clip.source?.videoElement) {
        const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
          ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;
        if (!isAtPlayhead && !clip.source.videoElement.paused) {
          clip.source.videoElement.pause();
        }
      }

      // Pause nested comp videos not at playhead
      if (clip.isComposition && clip.nestedClips) {
        const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
          ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;
        if (!isAtPlayhead) {
          for (const nestedClip of clip.nestedClips) {
            if (nestedClip.source?.videoElement && !nestedClip.source.videoElement.paused) {
              nestedClip.source.videoElement.pause();
            }
          }
        }
      }
    }

    // Sync background layer video elements
    layerPlaybackManager.syncVideoElements(ctx.playheadPosition, ctx.isPlaying);
  }

  /**
   * Sync nested composition video elements
   * Uses same logic as regular clips: play during playback, seek when paused
   * Also ensures videos have decoded frames (readyState >= 2) for rendering
   */
  private syncNestedCompVideos(compClip: TimelineClip, ctx: FrameContext): void {
    if (!compClip.nestedClips || !compClip.nestedTracks) return;

    // Calculate time within the composition
    const compLocalTime = ctx.playheadPosition - compClip.startTime;
    const compTime = compLocalTime + compClip.inPoint;

    for (const nestedClip of compClip.nestedClips) {
      if (!nestedClip.source?.videoElement) continue;

      // Check if nested clip is active at current comp time
      const isActive = compTime >= nestedClip.startTime && compTime < nestedClip.startTime + nestedClip.duration;

      if (!isActive) {
        // Pause if not active
        if (!nestedClip.source.videoElement.paused) {
          nestedClip.source.videoElement.pause();
        }
        continue;
      }

      // Calculate time within the nested clip
      const nestedLocalTime = compTime - nestedClip.startTime;
      const nestedClipTime = nestedClip.reversed
        ? nestedClip.outPoint - nestedLocalTime
        : nestedLocalTime + nestedClip.inPoint;

      const video = nestedClip.source.videoElement;
      const webCodecsPlayer = nestedClip.source.webCodecsPlayer;
      const timeDiff = Math.abs(video.currentTime - nestedClipTime);

      // Pre-capture: ensure scrubbing cache has a frame before seeking
      if (!video.seeking && video.readyState >= 2) {
        engine.ensureVideoFrameCached(video);
      }

      // During playback: let video play naturally (like regular clips)
      if (ctx.isPlaying) {
        if (video.paused) {
          video.play().catch(() => {});
        }
        // Only seek if significantly out of sync (>0.5s)
        if (timeDiff > 0.5) {
          video.currentTime = nestedClipTime;
        }
      } else {
        // When paused: pause video and seek to exact time
        if (!video.paused) video.pause();

        // Force first-frame decode for videos that haven't played yet (e.g. after reload)
        if (video.played.length === 0 && !video.seeking && !this.forceDecodeInProgress.has(nestedClip.id)) {
          this.forceVideoFrameDecode(nestedClip.id, video);
        }

        const seekThreshold = ctx.isDraggingPlayhead ? 0.1 : 0.05;
        if (timeDiff > seekThreshold) {
          this.throttledSeek(nestedClip.id, video, nestedClipTime, ctx);
          video.addEventListener('seeked', () => engine.requestRender(), { once: true });
        }

        // If video readyState < 2 (no frame data), force decode via play/pause
        // This can happen after seeking to unbuffered regions
        if (video.readyState < 2 && !video.seeking) {
          this.forceVideoFrameDecode(nestedClip.id, video);
        }
      }

      // Sync WebCodecsPlayer only when not playing (it handles its own playback)
      if (webCodecsPlayer && !ctx.isPlaying) {
        const wcTimeDiff = Math.abs(webCodecsPlayer.currentTime - nestedClipTime);
        if (wcTimeDiff > 0.05) {
          webCodecsPlayer.seek(nestedClipTime);
        }
      }
    }
  }

  // Track which videos are being force-decoded to avoid duplicate calls
  private forceDecodeInProgress = new Set<string>();

  /**
   * Force video to decode current frame by briefly playing
   * Used when video has never played (after reload) or readyState drops below 2
   */
  private forceVideoFrameDecode(clipId: string, video: HTMLVideoElement): void {
    if (this.forceDecodeInProgress.has(clipId)) return;
    this.forceDecodeInProgress.add(clipId);

    const currentTime = video.currentTime;
    video.muted = true; // Prevent autoplay restrictions
    video.play()
      .then(() => {
        video.pause();
        video.currentTime = currentTime;
        this.forceDecodeInProgress.delete(clipId);
        engine.requestRender();
      })
      .catch(() => {
        // Fallback: tiny seek to trigger decode
        video.currentTime = currentTime + 0.001;
        this.forceDecodeInProgress.delete(clipId);
        engine.requestRender();
      });
  }

  /**
   * Sync a single clip's video element
   */
  private syncClipVideo(clip: TimelineClip, ctx: FrameContext): void {
    // Handle native decoder
    if (clip.source?.nativeDecoder) {
      this.syncNativeDecoder(clip, ctx);
      return;
    }

    if (!clip.source?.videoElement) return;

    const video = clip.source.videoElement;
    const timeInfo = getClipTimeInfo(ctx, clip);
    const mediaFile = getMediaFileForClip(ctx, clip);

    // Check proxy mode
    const useProxy = ctx.proxyEnabled && mediaFile?.proxyFps &&
      (mediaFile.proxyStatus === 'ready' || mediaFile.proxyStatus === 'generating');

    if (useProxy) {
      // In proxy mode: pause video
      if (!video.paused) video.pause();
      if (!video.muted) video.muted = true;
      return;
    }

    // Skip sync during GPU surface warmup — the video is playing briefly
    // to activate Chrome's GPU decoder. Don't pause or seek it.
    if (this.warmingUpVideos.has(video)) return;

    // Warmup: after page reload, video GPU surfaces are empty.
    // importExternalTexture, canvas.drawImage, etc. all return black.
    // The ONLY fix is video.play() to activate the GPU compositor.
    // We do this here (not during restore) because restore-time warmup
    // gets immediately killed by this very function's "pause if not playing" logic.
    if (!ctx.isPlaying && video.readyState >= 2 && !video.seeking &&
        video.played.length === 0 && !this.warmingUpVideos.has(video)) {
      this.warmingUpVideos.add(video);
      const targetTime = timeInfo.clipTime;
      video.play().then(() => {
        // Wait for actual frame presentation via requestVideoFrameCallback
        const rvfc = (video as any).requestVideoFrameCallback;
        if (typeof rvfc === 'function') {
          rvfc.call(video, () => {
            // Frame is now presented to GPU — capture it
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = targetTime;
            this.warmingUpVideos.delete(video);
            engine.requestRender();
          });
        } else {
          // Fallback: wait 100ms for frame presentation
          setTimeout(() => {
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = targetTime;
            this.warmingUpVideos.delete(video);
            engine.requestRender();
          }, 100);
        }
      }).catch(() => {
        this.warmingUpVideos.delete(video);
      });
      return; // Skip normal sync — warmup is handling video state
    }

    // Normal video sync
    const timeDiff = Math.abs(video.currentTime - timeInfo.clipTime);

    // Pre-capture: ensure scrubbing cache has a frame BEFORE seeking
    if (!video.seeking && video.readyState >= 2) {
      engine.ensureVideoFrameCached(video);
    }

    // Reverse playback: either clip is reversed OR timeline playbackSpeed is negative
    // H.264 can't play backwards, so we seek frame-by-frame
    const isReversePlayback = clip.reversed || ctx.playbackSpeed < 0;

    if (isReversePlayback) {
      // For reverse: pause video and seek to each frame
      if (!video.paused) video.pause();
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.02;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
    } else if (ctx.playbackSpeed !== 1) {
      // Non-standard forward speed (2x, 4x, etc.): seek frame-by-frame for accuracy
      if (!video.paused) video.pause();
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.03;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
    } else {
      // Normal 1x forward playback: let video play naturally
      if (ctx.isPlaying && video.paused) {
        video.play().catch(() => {});
      } else if (!ctx.isPlaying && !video.paused) {
        video.pause();
      }

      if (!ctx.isPlaying) {
        // 0.04s ≈ slightly more than 1 frame at 30fps.
        // Previous 0.1s threshold skipped up to 3 frames during slow scrubbing.
        const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.04;
        if (timeDiff > seekThreshold) {
          this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
        }

        // Force decode if readyState dropped after seek
        if (video.readyState < 2 && !video.seeking) {
          this.forceVideoFrameDecode(clip.id, video);
        }
      }
    }
  }

  /**
   * Hybrid seeking strategy for smooth scrubbing on all codec types:
   *
   * During drag (fast scrubbing):
   *   Phase 1: fastSeek → instant keyframe feedback (<10ms, shows nearest I-frame)
   *   Phase 2: deferred precise seek → exact frame when scrubbing pauses (debounced 120ms)
   *
   * This solves the long-GOP problem: YouTube/phone videos with 5-7s keyframe distance
   * previously showed a stale cached frame for 100-300ms per seek (currentTime decodes
   * from keyframe to target). Now the user sees the nearest keyframe immediately, then
   * the exact frame fills in when they pause.
   *
   * When not dragging (single click / arrow keys): precise seek via currentTime.
   *
   * RVFC (requestVideoFrameCallback) triggers re-render when the decoded frame is
   * actually presented to the compositor — more accurate than the 'seeked' event.
   */
  private rvfcHandles: Record<string, number> = {};
  private preciseSeekTimers: Record<string, ReturnType<typeof setTimeout>> = {};
  private latestSeekTargets: Record<string, number> = {};

  private throttledSeek(clipId: string, video: HTMLVideoElement, time: number, ctx: FrameContext): void {
    const lastSeek = this.lastSeekRef[clipId] || 0;
    const threshold = ctx.isDraggingPlayhead ? 50 : 33;
    if (ctx.now - lastSeek > threshold) {
      if (ctx.isDraggingPlayhead && 'fastSeek' in video) {
        // Phase 1: Instant keyframe feedback via fastSeek.
        // For all-intra codecs this IS the exact frame. For long-GOP codecs
        // this shows the nearest keyframe — better than a stale cached frame.
        video.fastSeek(time);

        // Phase 2: Schedule deferred precise seek for exact frame.
        // Debounced: resets on each new scrub position, only fires when
        // the user pauses or slows their scrubbing.
        this.latestSeekTargets[clipId] = time;
        clearTimeout(this.preciseSeekTimers[clipId]);
        this.preciseSeekTimers[clipId] = setTimeout(() => {
          const target = this.latestSeekTargets[clipId];
          // Only do precise seek if the fastSeek landed far from the target
          // (i.e., this is a long-GOP video where fastSeek shows a different frame)
          if (target !== undefined && Math.abs(video.currentTime - target) > 0.01) {
            video.currentTime = target;
            // Register RVFC for when the precise frame arrives
            this.registerRVFC(clipId, video);
          }
        }, 120);
      } else {
        // Not dragging: precise seek immediately (click, arrow keys, etc.)
        video.currentTime = time;
        clearTimeout(this.preciseSeekTimers[clipId]);
      }
      this.lastSeekRef[clipId] = ctx.now;

      // Register RVFC to trigger re-render when the decoded frame is presented.
      this.registerRVFC(clipId, video);
    }
  }

  private registerRVFC(clipId: string, video: HTMLVideoElement): void {
    const rvfc = (video as any).requestVideoFrameCallback;
    if (typeof rvfc === 'function') {
      const prevHandle = this.rvfcHandles[clipId];
      if (prevHandle !== undefined) {
        (video as any).cancelVideoFrameCallback(prevHandle);
      }
      this.rvfcHandles[clipId] = rvfc.call(video, () => {
        delete this.rvfcHandles[clipId];
        // Bypass the scrub rate limiter — a fresh decoded frame should be displayed immediately
        engine.requestNewFrameRender();
      });
    }
  }

  /**
   * Sync native decoder
   */
  private syncNativeDecoder(clip: TimelineClip, ctx: FrameContext): void {
    const nativeDecoder = clip.source!.nativeDecoder!;
    const timeInfo = getClipTimeInfo(ctx, clip);

    const fps = nativeDecoder.fps || 25;
    const targetFrame = Math.round(timeInfo.clipTime * fps);

    // Get or create state
    let state = this.nativeDecoderState.get(clip.id);
    if (!state) {
      state = { lastSeekTime: 0, lastSeekFrame: -1, isPending: false };
      this.nativeDecoderState.set(clip.id, state);
    }

    const timeSinceLastSeek = ctx.now - state.lastSeekTime;
    const shouldSeek = !state.isPending &&
      (targetFrame !== state.lastSeekFrame || timeSinceLastSeek > 100);

    if (shouldSeek && timeSinceLastSeek >= LAYER_BUILDER_CONSTANTS.NATIVE_SEEK_THROTTLE_MS) {
      state.lastSeekTime = ctx.now;
      state.lastSeekFrame = targetFrame;
      state.isPending = true;

      nativeDecoder.seekToFrame(targetFrame, ctx.isDraggingPlayhead)
        .then(() => { state!.isPending = false; })
        .catch((err: unknown) => { state!.isPending = false; console.warn('[NH] seek failed frame', targetFrame, err); });
    }
  }

  // ==================== AUDIO SYNC ====================

  /**
   * Sync audio elements to current playhead
   */
  syncAudioElements(): void {
    const ctx = createFrameContext();

    // At non-standard playback speeds (reverse or fast-forward), mute all audio
    // Audio can't play backwards and fast-forward sounds bad
    if (ctx.playbackSpeed !== 1 && ctx.isPlaying) {
      this.muteAllAudio(ctx);
      return;
    }

    // Handle playback start
    const isStartup = playheadState.playbackJustStarted;
    if (isStartup) {
      this.playbackStartFrames++;
      if (this.playbackStartFrames > 10) {
        playheadState.playbackJustStarted = false;
        this.playbackStartFrames = 0;
      }
    } else {
      // Throttle audio sync
      if (ctx.now - this.lastAudioSyncTime < LAYER_BUILDER_CONSTANTS.AUDIO_SYNC_INTERVAL) {
        return;
      }
    }
    this.lastAudioSyncTime = ctx.now;

    // Resume audio context if needed
    resumeAudioContextIfNeeded(ctx.isPlaying, ctx.isDraggingPlayhead);

    // Create sync state
    const state = createAudioSyncState();

    // Sync audio track clips
    this.syncAudioTrackClips(ctx, state);

    // Sync video clip audio (proxies and elements)
    this.syncVideoClipAudio(ctx, state);

    // Sync nested comp mixdown
    this.syncNestedCompMixdown(ctx, state);

    // Pause inactive audio
    this.pauseInactiveAudio(ctx);

    // Sync background layer audio elements
    layerPlaybackManager.syncAudioElements(ctx.playheadPosition, ctx.isPlaying);

    // Finalize
    finalizeAudioSync(state, ctx.isPlaying);
  }

  /**
   * Sync audio track clips
   */
  private syncAudioTrackClips(ctx: FrameContext, state: import('./types').AudioSyncState): void {
    for (const track of ctx.audioTracks) {
      const clip = getClipForTrack(ctx, track.id);
      if (!clip?.source?.audioElement) continue;

      // Skip audio elements without a valid source (e.g., empty audio from nested comps without audio)
      const audio = clip.source.audioElement;
      if (!audio.src && audio.readyState === 0) continue;

      const timeInfo = getClipTimeInfo(ctx, clip);
      const isMuted = !ctx.unmutedAudioTrackIds.has(track.id);

      this.audioSyncHandler.syncAudioElement({
        element: clip.source.audioElement,
        clip,
        clipTime: timeInfo.clipTime,
        absSpeed: timeInfo.absSpeed,
        isMuted,
        canBeMaster: true,
        type: 'audioTrack',
        volume: getClipVolume(ctx, clip, timeInfo.clipLocalTime),
        eqGains: getClipEQGains(ctx, clip, timeInfo.clipLocalTime),
      }, ctx, state);
    }
  }

  /**
   * Sync video clip audio (proxies and varispeed scrubbing)
   */
  private syncVideoClipAudio(ctx: FrameContext, state: import('./types').AudioSyncState): void {
    const activeVideoClipIds = new Set<string>();

    for (const track of ctx.videoTracks) {
      const clip = getClipForTrack(ctx, track.id);
      if (!clip?.source?.videoElement || clip.isComposition) continue;

      const mediaFile = getMediaFileForClip(ctx, clip);
      const timeInfo = getClipTimeInfo(ctx, clip);
      const isMuted = !isVideoTrackVisible(ctx, track.id);
      const mediaFileId = mediaFile?.id || clip.mediaFileId || clip.id;

      // Varispeed scrubbing for all clips
      if (ctx.isDraggingPlayhead && !isMuted) {
        const video = clip.source.videoElement;
        if (!video.muted) video.muted = true;
        proxyFrameCache.playScrubAudio(mediaFileId, timeInfo.clipTime);
      } else if (!ctx.isDraggingPlayhead) {
        proxyFrameCache.stopScrubAudio();
      }

      // Audio proxy handling
      const shouldUseAudioProxy = ctx.proxyEnabled &&
        mediaFile?.hasProxyAudio &&
        (mediaFile.proxyStatus === 'ready' || mediaFile.proxyStatus === 'generating');

      if (shouldUseAudioProxy && mediaFile) {
        activeVideoClipIds.add(clip.id);

        const video = clip.source.videoElement;
        if (!video.muted) video.muted = true;

        const audioProxy = proxyFrameCache.getCachedAudioProxy(mediaFile.id);
        if (audioProxy) {
          this.activeAudioProxies.set(clip.id, audioProxy);

          this.audioSyncHandler.syncAudioElement({
            element: audioProxy,
            clip,
            clipTime: timeInfo.clipTime,
            absSpeed: timeInfo.absSpeed,
            isMuted,
            canBeMaster: !state.masterSet,
            type: 'audioProxy',
            volume: getClipVolume(ctx, clip, timeInfo.clipLocalTime),
            eqGains: getClipEQGains(ctx, clip, timeInfo.clipLocalTime),
          }, ctx, state);
        } else {
          // Trigger preload
          proxyFrameCache.preloadAudioProxy(mediaFile.id);
          proxyFrameCache.getAudioBuffer(mediaFile.id);
        }
      }
    }

    // Pause inactive audio proxies
    for (const [clipId, audioProxy] of this.activeAudioProxies) {
      if (!activeVideoClipIds.has(clipId) && !audioProxy.paused) {
        audioProxy.pause();
        this.activeAudioProxies.delete(clipId);
      }
    }
  }

  /**
   * Sync nested composition mixdown audio
   */
  private syncNestedCompMixdown(ctx: FrameContext, state: import('./types').AudioSyncState): void {
    for (const clip of ctx.clipsAtTime) {
      if (!clip.isComposition || !clip.mixdownAudio || !clip.hasMixdownAudio) continue;

      const timeInfo = getClipTimeInfo(ctx, clip);
      const track = ctx.videoTracks.find(t => t.id === clip.trackId);
      const isMuted = track ? !isVideoTrackVisible(ctx, track.id) : false;

      this.audioSyncHandler.syncAudioElement({
        element: clip.mixdownAudio,
        clip,
        clipTime: timeInfo.clipTime,
        absSpeed: timeInfo.absSpeed,
        isMuted,
        canBeMaster: !state.masterSet,
        type: 'mixdown',
        volume: getClipVolume(ctx, clip, timeInfo.clipLocalTime),
        eqGains: getClipEQGains(ctx, clip, timeInfo.clipLocalTime),
      }, ctx, state);
    }
  }

  /**
   * Pause audio not at playhead
   */
  private pauseInactiveAudio(ctx: FrameContext): void {
    for (const clip of ctx.clips) {
      const isAtPlayhead = ctx.clipsAtTime.some(c => c.id === clip.id);

      if (clip.source?.audioElement && !isAtPlayhead && !clip.source.audioElement.paused) {
        clip.source.audioElement.pause();
      }

      if (clip.mixdownAudio && !isAtPlayhead && !clip.mixdownAudio.paused) {
        clip.mixdownAudio.pause();
      }
    }
  }

  /**
   * Mute all audio during non-standard playback (reverse or fast-forward)
   * Audio can't play backwards and fast-forward audio sounds bad
   */
  private muteAllAudio(ctx: FrameContext): void {
    // Clear master audio since we're not using audio sync
    playheadState.hasMasterAudio = false;
    playheadState.masterAudioElement = null;

    // Pause all audio elements
    for (const clip of ctx.clips) {
      if (clip.source?.audioElement && !clip.source.audioElement.paused) {
        clip.source.audioElement.pause();
      }
      if (clip.source?.videoElement && !clip.source.videoElement.muted) {
        clip.source.videoElement.muted = true;
      }
      if (clip.mixdownAudio && !clip.mixdownAudio.paused) {
        clip.mixdownAudio.pause();
      }
    }
  }
}
