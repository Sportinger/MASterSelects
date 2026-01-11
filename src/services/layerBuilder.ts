// LayerBuilder - Calculates render layers on-demand without React state overhead
// Called directly from the render loop for maximum performance

import type { TimelineClip, TimelineTrack, Layer, Effect, NestedCompositionData, AnimatableProperty } from '../types';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { proxyFrameCache } from './proxyFrameCache';

// Helper: Check if effects have changed
function effectsChanged(
  layerEffects: Effect[] | undefined,
  clipEffects: Effect[] | undefined
): boolean {
  const le = layerEffects || [];
  const ce = clipEffects || [];
  if (le.length !== ce.length) return true;
  for (let i = 0; i < le.length; i++) {
    if (le[i].id !== ce[i].id || le[i].enabled !== ce[i].enabled) return true;
    const lp = le[i].params;
    const cp = ce[i].params;
    const lKeys = Object.keys(lp);
    const cKeys = Object.keys(cp);
    if (lKeys.length !== cKeys.length) return true;
    for (const key of lKeys) {
      if (lp[key] !== cp[key]) return true;
    }
  }
  return false;
}

interface LayerBuilderContext {
  playheadPosition: number;
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  isPlaying: boolean;
  isDraggingPlayhead: boolean;
  clipKeyframes: Map<string, Array<{ id: string; clipId: string; time: number; property: AnimatableProperty; value: number; easing: string }>>;
  getInterpolatedTransform: (clipId: string, localTime: number) => {
    position: { x: number; y: number; z: number };
    scale: { x: number; y: number };
    rotation: { x: number; y: number; z: number };
    opacity: number;
    blendMode: string;
  };
  getInterpolatedEffects: (clipId: string, localTime: number) => Effect[];
  getInterpolatedSpeed: (clipId: string, localTime: number) => number;
  getSourceTimeForClip: (clipId: string, localTime: number) => number;
}

class LayerBuilderService {
  private lastSeekRef: { [clipId: string]: number } = {};
  private proxyFramesRef: Map<string, { frameIndex: number; image: HTMLImageElement }> = new Map();
  private proxyLoadingRef: Set<string> = new Set();

  /**
   * Build layers for the current frame - called directly from render loop
   * This avoids React state updates and re-render overhead
   */
  buildLayers(ctx: LayerBuilderContext): Layer[] {
    const {
      playheadPosition,
      clips,
      tracks,
      isPlaying,
      isDraggingPlayhead,
      getInterpolatedTransform,
      getInterpolatedEffects,
      getInterpolatedSpeed,
      getSourceTimeForClip,
    } = ctx;

    const videoTracks = tracks.filter(t => t.type === 'video' && t.visible !== false);

    // Get clips at current playhead
    const clipsAtTime = clips.filter(
      c => playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
    );

    const layers: Layer[] = [];

    videoTracks.forEach((track, layerIndex) => {
      const clip = clipsAtTime.find(c => c.trackId === track.id);

      if (clip?.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        // Handle nested composition
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const nestedLayers = this.buildNestedLayers(clip, clipTime, isPlaying);
        const interpolatedTransform = getInterpolatedTransform(clip.id, clipTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, clipTime);

        // Get composition dimensions
        const mediaStore = useMediaStore.getState();
        const composition = mediaStore.compositions.find(c => c.id === clip.compositionId);
        const compWidth = composition?.width || 1920;
        const compHeight = composition?.height || 1080;

        if (nestedLayers.length > 0) {
          const nestedCompData: NestedCompositionData = {
            compositionId: clip.compositionId || clip.id,
            layers: nestedLayers,
            width: compWidth,
            height: compHeight,
          };

          layers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: true,
            opacity: interpolatedTransform.opacity,
            blendMode: interpolatedTransform.blendMode,
            source: {
              type: 'video',
              nestedComposition: nestedCompData,
            },
            effects: interpolatedEffects,
            position: { x: interpolatedTransform.position.x, y: interpolatedTransform.position.y, z: interpolatedTransform.position.z },
            scale: { x: interpolatedTransform.scale.x, y: interpolatedTransform.scale.y },
            rotation: {
              x: (interpolatedTransform.rotation.x * Math.PI) / 180,
              y: (interpolatedTransform.rotation.y * Math.PI) / 180,
              z: (interpolatedTransform.rotation.z * Math.PI) / 180,
            },
          };
        }
      } else if (clip?.source?.videoElement) {
        // Handle video clip
        const layer = this.buildVideoLayer(
          clip,
          track,
          layerIndex,
          playheadPosition,
          isPlaying,
          isDraggingPlayhead,
          getInterpolatedTransform,
          getInterpolatedEffects,
          getInterpolatedSpeed,
          getSourceTimeForClip
        );
        if (layer) {
          layers[layerIndex] = layer;
        }
      } else if (clip?.source?.imageElement) {
        // Handle image clip
        const imageClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, imageClipLocalTime);
        const imageInterpolatedEffects = getInterpolatedEffects(clip.id, imageClipLocalTime);

        layers[layerIndex] = {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'image',
            imageElement: clip.source.imageElement,
          },
          effects: imageInterpolatedEffects,
          position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
          scale: { x: transform.scale.x, y: transform.scale.y },
          rotation: {
            x: (transform.rotation.x * Math.PI) / 180,
            y: (transform.rotation.y * Math.PI) / 180,
            z: (transform.rotation.z * Math.PI) / 180,
          },
        };
      } else if (clip?.source?.textCanvas) {
        // Handle text clip
        const textClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, textClipLocalTime);
        const textInterpolatedEffects = getInterpolatedEffects(clip.id, textClipLocalTime);

        layers[layerIndex] = {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'text',
            textCanvas: clip.source.textCanvas,
          },
          effects: textInterpolatedEffects,
          position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
          scale: { x: transform.scale.x, y: transform.scale.y },
          rotation: {
            x: (transform.rotation.x * Math.PI) / 180,
            y: (transform.rotation.y * Math.PI) / 180,
            z: (transform.rotation.z * Math.PI) / 180,
          },
        };
      }
    });

    return layers;
  }

  /**
   * Build video layer with video seeking and proxy handling
   */
  private buildVideoLayer(
    clip: TimelineClip,
    track: TimelineTrack,
    layerIndex: number,
    playheadPosition: number,
    isPlaying: boolean,
    isDraggingPlayhead: boolean,
    getInterpolatedTransform: LayerBuilderContext['getInterpolatedTransform'],
    getInterpolatedEffects: LayerBuilderContext['getInterpolatedEffects'],
    getInterpolatedSpeed: LayerBuilderContext['getInterpolatedSpeed'],
    getSourceTimeForClip: LayerBuilderContext['getSourceTimeForClip']
  ): Layer | null {
    const clipLocalTime = playheadPosition - clip.startTime;
    const keyframeLocalTime = clipLocalTime;
    const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
    const initialSpeed = getInterpolatedSpeed(clip.id, 0);
    const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
    const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
    const video = clip.source!.videoElement!;
    const webCodecsPlayer = clip.source?.webCodecsPlayer;
    const timeDiff = Math.abs(video.currentTime - clipTime);

    // Check for proxy usage
    const mediaStore = useMediaStore.getState();
    const mediaFile = mediaStore.files.find(
      f => f.name === clip.name || clip.source?.mediaFileId === f.id
    );
    const proxyFps = mediaFile?.proxyFps || 30;
    const frameIndex = Math.floor(clipTime * proxyFps);
    let useProxy = false;

    if (mediaStore.proxyEnabled && mediaFile?.proxyFps) {
      if (mediaFile.proxyStatus === 'ready') {
        useProxy = true;
      } else if (mediaFile.proxyStatus === 'generating' && (mediaFile.proxyProgress || 0) > 0) {
        const totalFrames = Math.ceil((mediaFile.duration || 10) * proxyFps);
        const maxGeneratedFrame = Math.floor(totalFrames * ((mediaFile.proxyProgress || 0) / 100));
        useProxy = frameIndex < maxGeneratedFrame;
      }
    }

    if (useProxy && mediaFile) {
      // Proxy playback - use cached proxy frames
      const cacheKey = `${mediaFile.id}_${clip.id}`;
      const cachedInService = proxyFrameCache.getCachedFrame(mediaFile.id, frameIndex, proxyFps);

      // Keep video playing for audio but muted
      if (!video.muted) video.muted = true;
      if (isPlaying && video.paused) video.play().catch(() => {});
      else if (!isPlaying && !video.paused) video.pause();

      if (cachedInService) {
        this.proxyFramesRef.set(cacheKey, { frameIndex, image: cachedInService });
        const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

        return {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'image',
            imageElement: cachedInService,
          },
          effects: interpolatedEffects,
          position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
          scale: { x: transform.scale.x, y: transform.scale.y },
          rotation: {
            x: (transform.rotation.x * Math.PI) / 180,
            y: (transform.rotation.y * Math.PI) / 180,
            z: (transform.rotation.z * Math.PI) / 180,
          },
        };
      }

      // Use cached proxy frame if available while loading new one
      const cached = this.proxyFramesRef.get(cacheKey);
      if (cached?.image) {
        const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

        return {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'image',
            imageElement: cached.image,
          },
          effects: interpolatedEffects,
          position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
          scale: { x: transform.scale.x, y: transform.scale.y },
          rotation: {
            x: (transform.rotation.x * Math.PI) / 180,
            y: (transform.rotation.y * Math.PI) / 180,
            z: (transform.rotation.z * Math.PI) / 180,
          },
        };
      }
    }

    // Direct video playback
    if (webCodecsPlayer) {
      const wcTimeDiff = Math.abs(webCodecsPlayer.currentTime - clipTime);
      if (wcTimeDiff > 0.05) {
        webCodecsPlayer.seek(clipTime);
      }
    }

    if (clip.reversed) {
      if (!video.paused) video.pause();
      const seekThreshold = isDraggingPlayhead ? 0.1 : 0.03;
      if (timeDiff > seekThreshold) {
        const now = performance.now();
        const lastSeek = this.lastSeekRef[clip.id] || 0;
        if (now - lastSeek > 33) {
          video.currentTime = clipTime;
          this.lastSeekRef[clip.id] = now;
        }
      }
    } else {
      if (isPlaying && video.paused) video.play().catch(() => {});
      else if (!isPlaying && !video.paused) video.pause();

      if (!isPlaying) {
        const seekThreshold = isDraggingPlayhead ? 0.1 : 0.05;
        if (timeDiff > seekThreshold) {
          const now = performance.now();
          const lastSeek = this.lastSeekRef[clip.id] || 0;
          if (now - lastSeek > (isDraggingPlayhead ? 80 : 33)) {
            if (isDraggingPlayhead && 'fastSeek' in video) {
              video.fastSeek(clipTime);
            } else {
              video.currentTime = clipTime;
            }
            this.lastSeekRef[clip.id] = now;
          }
        }
      }
    }

    const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
    const videoInterpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

    return {
      id: `timeline_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity,
      blendMode: transform.blendMode,
      source: {
        type: 'video',
        videoElement: video,
        webCodecsPlayer: webCodecsPlayer,
      },
      effects: videoInterpolatedEffects,
      position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
      scale: { x: transform.scale.x, y: transform.scale.y },
      rotation: {
        x: (transform.rotation.x * Math.PI) / 180,
        y: (transform.rotation.y * Math.PI) / 180,
        z: (transform.rotation.z * Math.PI) / 180,
      },
    };
  }

  /**
   * Build nested composition layers
   */
  private buildNestedLayers(clip: TimelineClip, clipTime: number, isPlaying: boolean): Layer[] {
    if (!clip.nestedClips || !clip.nestedTracks) return [];

    const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
    const layers: Layer[] = [];

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

      // Update video currentTime
      if (nestedClip.source?.videoElement) {
        const video = nestedClip.source.videoElement;
        const timeDiff = Math.abs(video.currentTime - nestedClipTime);
        if (timeDiff > 0.05) {
          video.currentTime = nestedClipTime;
        }
        if (isPlaying && video.paused) {
          video.play().catch(() => {});
        } else if (!isPlaying && !video.paused) {
          video.pause();
        }
      }

      const transform = nestedClip.transform || {
        position: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1 },
        rotation: { x: 0, y: 0, z: 0 },
        anchor: { x: 0.5, y: 0.5 },
        opacity: 1,
        blendMode: 'normal' as const,
      };

      if (nestedClip.source?.videoElement) {
        layers.push({
          id: `nested-layer-${nestedClip.id}`,
          name: nestedClip.name,
          visible: true,
          opacity: transform.opacity ?? 1,
          blendMode: transform.blendMode || 'normal',
          source: {
            type: 'video',
            videoElement: nestedClip.source.videoElement,
            webCodecsPlayer: nestedClip.source.webCodecsPlayer,
          },
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
        });
      } else if (nestedClip.source?.imageElement) {
        layers.push({
          id: `nested-layer-${nestedClip.id}`,
          name: nestedClip.name,
          visible: true,
          opacity: transform.opacity ?? 1,
          blendMode: transform.blendMode || 'normal',
          source: {
            type: 'image',
            imageElement: nestedClip.source.imageElement,
          },
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
        });
      }
    }

    return layers;
  }
}

// Singleton instance
export const layerBuilder = new LayerBuilderService();
