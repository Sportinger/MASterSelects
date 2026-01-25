// LayerBuilderService - Main orchestrator for layer building
// Uses modular components for caching, transforms, and audio sync

import type { TimelineClip, Layer, NestedCompositionData, BlendMode } from '../../types';
import type { FrameContext, NativeDecoderState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { playheadState } from './PlayheadState';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip, isVideoTrackVisible, getClipForTrack } from './FrameContext';
import { LayerCache } from './LayerCache';
import { TransformCache } from './TransformCache';
import { AudioSyncHandler, createAudioSyncState, finalizeAudioSync, resumeAudioContextIfNeeded } from './AudioSyncHandler';
import { proxyFrameCache } from '../proxyFrameCache';
import { Logger } from '../logger';

const log = Logger.create('LayerBuilder');

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

    // Check cache
    const cacheResult = this.layerCache.checkCache(ctx);
    if (cacheResult.useCache) {
      return cacheResult.layers;
    }

    // Build layers
    const layers = this.buildLayers(ctx);

    // Preload upcoming nested comp frames during playback
    if (ctx.isPlaying) {
      this.preloadUpcomingNestedCompFrames(ctx);
    }

    // Cache and return
    this.layerCache.setCachedLayers(layers);
    return layers;
  }

  /**
   * Build layers from frame context
   */
  private buildLayers(ctx: FrameContext): Layer[] {
    const layers: Layer[] = [];

    ctx.videoTracks.forEach((track, layerIndex) => {
      if (!isVideoTrackVisible(ctx, track.id)) {
        return;
      }

      const clip = getClipForTrack(ctx, track.id);
      if (!clip) return;

      const layer = this.buildLayerForClip(clip, layerIndex, ctx);
      if (layer) {
        layers[layerIndex] = layer;
      }
    });

    return layers;
  }

  /**
   * Build a layer for a clip based on its type
   */
  private buildLayerForClip(clip: TimelineClip, layerIndex: number, ctx: FrameContext): Layer | null {
    // Nested composition
    if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
      return this.buildNestedCompLayer(clip, layerIndex, ctx);
    }

    // Native decoder (ProRes/DNxHD turbo mode)
    if (clip.source?.nativeDecoder) {
      return this.buildNativeDecoderLayer(clip, layerIndex, ctx);
    }

    // Video clip
    if (clip.source?.videoElement) {
      return this.buildVideoLayer(clip, layerIndex, ctx);
    }

    // Image clip
    if (clip.source?.imageElement) {
      return this.buildImageLayer(clip, layerIndex, ctx);
    }

    // Text clip
    if (clip.source?.textCanvas) {
      return this.buildTextLayer(clip, layerIndex, ctx);
    }

    return null;
  }

  /**
   * Build nested composition layer
   */
  private buildNestedCompLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext): Layer | null {
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

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity,
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
  private buildNativeDecoderLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext): Layer {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity,
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
  private buildVideoLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext): Layer | null {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const mediaFile = getMediaFileForClip(ctx, clip);

    // Check for proxy usage
    if (ctx.proxyEnabled && mediaFile?.proxyFps) {
      const proxyLayer = this.tryBuildProxyLayer(clip, layerIndex, timeInfo.clipTime, mediaFile, ctx);
      if (proxyLayer) return proxyLayer;
    }

    // Direct video layer
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity,
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
    ctx: FrameContext
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
      return this.buildImageLayerFromElement(clip, layerIndex, cachedFrame, clipTime, ctx);
    }

    // Use previous cached frame as fallback
    const cached = this.proxyFramesRef.get(cacheKey);
    if (cached?.image) {
      return this.buildImageLayerFromElement(clip, layerIndex, cached.image, clipTime, ctx);
    }

    return null;
  }

  /**
   * Build image layer
   */
  private buildImageLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext): Layer {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity,
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
    ctx: FrameContext
  ): Layer {
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, localTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, localTime);

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity,
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
  private buildTextLayer(clip: TimelineClip, layerIndex: number, ctx: FrameContext): Layer {
    const timeInfo = getClipTimeInfo(ctx, clip);
    const transform = this.transformCache.getTransform(
      `${ctx.activeCompId}_${layerIndex}`,
      ctx.getInterpolatedTransform(clip.id, timeInfo.clipLocalTime)
    );
    const effects = ctx.getInterpolatedEffects(clip.id, timeInfo.clipLocalTime);

    const layer: Layer = {
      id: `${ctx.activeCompId}_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity,
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

    const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
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

    for (let i = nestedVideoTracks.length - 1; i >= 0; i--) {
      const nestedTrack = nestedVideoTracks[i];
      const nestedClip = clip.nestedClips.find(
        nc =>
          nc.trackId === nestedTrack.id &&
          clipTime >= nc.startTime &&
          clipTime < nc.startTime + nc.duration
      );

      if (!nestedClip) continue;

      const nestedLocalTime = clipTime - nestedClip.startTime;
      const nestedClipTime = nestedClip.reversed
        ? nestedClip.outPoint - nestedLocalTime
        : nestedLocalTime + nestedClip.inPoint;

      // Build layer based on source type
      const nestedLayer = this.buildNestedClipLayer(nestedClip, nestedClipTime, ctx);
      if (nestedLayer) {
        layers.push(nestedLayer);
      }
    }

    return layers;
  }

  /**
   * Build layer for a nested clip
   */
  private buildNestedClipLayer(nestedClip: TimelineClip, _nestedClipTime: number, _ctx: FrameContext): Layer | null {
    const transform = nestedClip.transform || {
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: { x: 0, y: 0, z: 0 },
      opacity: 1,
      blendMode: 'normal' as const,
    };

    const baseLayer = {
      id: `nested-layer-${nestedClip.id}`,
      name: nestedClip.name,
      visible: true,
      opacity: transform.opacity ?? 1,
      blendMode: transform.blendMode || 'normal',
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
  }

  /**
   * Sync nested composition video elements
   * Uses same logic as regular clips: play during playback, seek when paused
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

        const seekThreshold = ctx.isDraggingPlayhead ? 0.1 : 0.05;
        if (timeDiff > seekThreshold) {
          this.throttledSeek(nestedClip.id, video, nestedClipTime, ctx);
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

    // Normal video sync
    const timeDiff = Math.abs(video.currentTime - timeInfo.clipTime);

    if (clip.reversed) {
      if (!video.paused) video.pause();
      const seekThreshold = ctx.isDraggingPlayhead ? 0.1 : 0.03;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
    } else {
      if (ctx.isPlaying && video.paused) {
        video.play().catch(() => {});
      } else if (!ctx.isPlaying && !video.paused) {
        video.pause();
      }

      if (!ctx.isPlaying) {
        const seekThreshold = ctx.isDraggingPlayhead ? 0.1 : 0.05;
        if (timeDiff > seekThreshold) {
          this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
        }
      }
    }
  }

  /**
   * Throttled video seek
   */
  private throttledSeek(clipId: string, video: HTMLVideoElement, time: number, ctx: FrameContext): void {
    const lastSeek = this.lastSeekRef[clipId] || 0;
    const threshold = ctx.isDraggingPlayhead ? 80 : 33;
    if (ctx.now - lastSeek > threshold) {
      if (ctx.isDraggingPlayhead && 'fastSeek' in video) {
        video.fastSeek(time);
      } else {
        video.currentTime = time;
      }
      this.lastSeekRef[clipId] = ctx.now;
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
        .catch(() => { state!.isPending = false; });
    }
  }

  // ==================== AUDIO SYNC ====================

  /**
   * Sync audio elements to current playhead
   */
  syncAudioElements(): void {
    const ctx = createFrameContext();

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
}
