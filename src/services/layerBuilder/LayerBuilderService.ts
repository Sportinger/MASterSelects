// LayerBuilderService - Main orchestrator for layer building
// Delegates video sync to VideoSyncManager and audio sync to AudioTrackSyncManager

import type { TimelineClip, Layer, NestedCompositionData, BlendMode, ClipTransform } from '../../types';
import type { FrameContext } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip, isVideoTrackVisible } from './FrameContext';
import { LayerCache } from './LayerCache';
import { TransformCache } from './TransformCache';
import { VideoSyncManager } from './VideoSyncManager';
import { AudioTrackSyncManager } from './AudioTrackSyncManager';
import { proxyFrameCache } from '../proxyFrameCache';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { Logger } from '../logger';
import { getInterpolatedClipTransform } from '../../utils/keyframeInterpolation';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { DEFAULT_TRANSFORM } from '../../stores/timeline/constants';

const log = Logger.create('LayerBuilder');

/**
 * LayerBuilderService - Builds render layers from timeline state
 * Optimized with caching, memoization, and object reuse
 */
export class LayerBuilderService {
  // Sub-modules
  private layerCache = new LayerCache();
  private transformCache = new TransformCache();
  private videoSyncManager = new VideoSyncManager();
  private audioTrackSyncManager = new AudioTrackSyncManager();

  // Proxy frame refs for fallback
  private proxyFramesRef = new Map<string, { frameIndex: number; image: HTMLImageElement }>();

  // Lookahead preloading
  private lastLookaheadTime = 0;

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

    // No active editor composition → no primary layers to build
    // (all active comps are background layers managed by layerPlaybackManager)
    const hasActiveComp = useMediaStore.getState().activeCompositionId != null;
    if (!hasActiveComp) {
      this.layerCache.invalidate();
      return this.mergeBackgroundLayers([], ctx.playheadPosition);
    }

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

    // Cache primary layers only — background layers are rebuilt fresh each frame
    // (caching merged layers would cause double-rendering of background layers on cache hit)
    this.layerCache.setCachedLayers(primaryLayers);

    // Merge background layers from active layer slots
    return this.mergeBackgroundLayers(primaryLayers, ctx.playheadPosition);
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

    const { layerOpacities } = useMediaStore.getState();

    for (const layerIndex of layerIndices) {
      if (layerIndex === primaryLayerIndex) {
        // Insert primary layers at this position, applying layer opacity
        const layerOpacity = layerOpacities[layerIndex] ?? 1;
        // Filter out undefined entries from sparse arrays (buildLayers uses layers[trackIndex]=...)
        const actualPrimaryLayers = primaryLayers.filter((l): l is Layer => l != null);
        if (layerOpacity < 1 && actualPrimaryLayers.length > 0) {
          // Apply per-clip opacity multiplication — simpler and works with all decoder types
          // (nativeDecoder, WebCodecs, HTMLVideo, etc.) without needing NestedCompRenderer
          for (const layer of actualPrimaryLayers) {
            merged.push({ ...layer, opacity: layer.opacity * layerOpacity });
          }
        } else {
          merged.push(...actualPrimaryLayers);
        }
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
      merged.push(...primaryLayers.filter((l): l is Layer => l != null));
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

  // ==================== VIDEO & AUDIO SYNC (delegated) ====================

  /**
   * Sync video elements to current playhead
   */
  syncVideoElements(): void {
    this.videoSyncManager.syncVideoElements();
  }

  /**
   * Sync audio elements to current playhead
   */
  syncAudioElements(): void {
    this.audioTrackSyncManager.syncAudioElements();
  }
}
