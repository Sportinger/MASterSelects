// LayerPlaybackManager - Manages background composition playback for Resolume-style multi-layer mode
// Each slot grid layer (A-D) can have an active composition; this service loads their media elements
// and provides layers for rendering. The primary (editor) composition is handled by the timeline store.

import type { TimelineClip, TimelineTrack, Layer, NestedCompositionData } from '../types';
import { engine } from '../engine/WebGPUEngine';
import type { Composition } from '../stores/mediaStore/types';
import { useMediaStore } from '../stores/mediaStore';
import { DEFAULT_TRANSFORM } from '../stores/timeline/constants';
import { Logger } from './logger';

const log = Logger.create('LayerPlayback');

interface LayerCompState {
  compositionId: string;
  composition: Composition;
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  duration: number;
  // Independent time tracking (wall-clock based, not tied to global playhead)
  activatedAt: number;   // performance.now() when this layer was activated
  pausedAt: number | null; // if paused, the elapsed time at pause; null = running
}

class LayerPlaybackManager {
  // Layer index (0=A, 1=B, 2=C, 3=D) → loaded composition state
  private layerStates = new Map<number, LayerCompState>();

  /**
   * Activate a composition on a layer — loads its timelineData and creates media elements
   */
  activateLayer(layerIndex: number, compositionId: string, initialElapsed?: number): void {
    // Deactivate current layer first
    this.deactivateLayer(layerIndex);

    const { compositions, files } = useMediaStore.getState();
    const comp = compositions.find(c => c.id === compositionId);
    if (!comp) {
      log.warn(`Composition ${compositionId} not found`);
      return;
    }

    const timelineData = comp.timelineData;
    if (!timelineData || !timelineData.clips || timelineData.clips.length === 0) {
      log.info(`Composition ${comp.name} has no timeline data, activating with empty state`);
      this.layerStates.set(layerIndex, {
        compositionId,
        composition: comp,
        clips: [],
        tracks: timelineData?.tracks || [],
        duration: comp.duration,
        activatedAt: performance.now() - (initialElapsed ?? 0) * 1000,
        pausedAt: null,
      });
      return;
    }

    // Hydrate serialized clips into live TimelineClips with media elements
    const hydratedClips: TimelineClip[] = [];

    for (const serializedClip of timelineData.clips) {
      const clip: TimelineClip = {
        id: serializedClip.id,
        trackId: serializedClip.trackId,
        name: serializedClip.name,
        file: null as any,
        startTime: serializedClip.startTime,
        duration: serializedClip.duration,
        inPoint: serializedClip.inPoint,
        outPoint: serializedClip.outPoint,
        source: null,
        transform: serializedClip.transform || { ...DEFAULT_TRANSFORM },
        effects: serializedClip.effects || [],
        mediaFileId: serializedClip.mediaFileId,
        reversed: serializedClip.reversed,
        isComposition: serializedClip.isComposition,
        compositionId: serializedClip.compositionId,
        masks: serializedClip.masks,
      };

      // Find media file
      const mediaFile = files.find(f => f.id === serializedClip.mediaFileId);
      if (!mediaFile && !serializedClip.isComposition) {
        // Can't load without media file (unless it's a nested comp)
        clip.isLoading = false;
        hydratedClips.push(clip);
        continue;
      }

      const sourceType = serializedClip.sourceType;
      const fileUrl = mediaFile?.url;

      if (sourceType === 'video' && fileUrl) {
        clip.isLoading = true;
        this.loadVideoForClip(clip, layerIndex, fileUrl, serializedClip.mediaFileId);
      } else if (sourceType === 'audio' && fileUrl) {
        clip.isLoading = true;
        this.loadAudioForClip(clip, layerIndex, fileUrl, serializedClip.mediaFileId);
      } else if (sourceType === 'image' && fileUrl) {
        clip.isLoading = true;
        this.loadImageForClip(clip, layerIndex, fileUrl);
      } else {
        clip.isLoading = false;
      }

      hydratedClips.push(clip);
    }

    this.layerStates.set(layerIndex, {
      compositionId,
      composition: comp,
      clips: hydratedClips,
      tracks: timelineData.tracks,
      duration: comp.duration,
      activatedAt: performance.now(),
      pausedAt: null,
    });

    log.info(`Activated layer ${layerIndex} with composition "${comp.name}" (${hydratedClips.length} clips)`);
  }

  /**
   * Deactivate a layer — pause and clean up media elements
   */
  deactivateLayer(layerIndex: number): void {
    const state = this.layerStates.get(layerIndex);
    if (!state) return;

    for (const clip of state.clips) {
      if (clip.source?.videoElement) {
        clip.source.videoElement.pause();
        clip.source.videoElement.src = '';
        clip.source.videoElement.load();
      }
      if (clip.source?.audioElement) {
        clip.source.audioElement.pause();
        clip.source.audioElement.src = '';
        clip.source.audioElement.load();
      }
    }

    this.layerStates.delete(layerIndex);
    log.info(`Deactivated layer ${layerIndex}`);
  }

  /**
   * Deactivate all layers
   */
  deactivateAll(): void {
    for (const layerIndex of Array.from(this.layerStates.keys())) {
      this.deactivateLayer(layerIndex);
    }
  }

  /**
   * Get the state for a layer (or undefined if not active)
   */
  getLayerState(layerIndex: number): LayerCompState | undefined {
    return this.layerStates.get(layerIndex);
  }

  /**
   * Check if a layer has an active composition
   */
  isLayerActive(layerIndex: number): boolean {
    return this.layerStates.has(layerIndex);
  }

  /**
   * Get the independent elapsed time for a layer (wall-clock based, not global playhead).
   * Loops back to 0 when reaching composition duration.
   */
  getLayerTime(state: LayerCompState): number {
    const elapsed = (performance.now() - state.activatedAt) / 1000;
    // Loop within composition duration
    return state.duration > 0 ? elapsed % state.duration : 0;
  }

  /**
   * Build render layers for a background composition.
   * Uses independent wall-clock time per layer, NOT the global playhead.
   */
  buildLayersForLayer(layerIndex: number, _playheadPosition: number): Layer | null {
    const state = this.layerStates.get(layerIndex);
    if (!state) return null;

    const layerTime = this.getLayerTime(state);
    const innerLayers = this.buildInnerLayers(state, layerTime);
    if (innerLayers.length === 0) return null;

    const nestedCompData: NestedCompositionData = {
      compositionId: state.compositionId,
      layers: innerLayers,
      width: state.composition.width,
      height: state.composition.height,
      currentTime: layerTime,
    };

    // Read per-layer opacity from store
    const layerOpacity = useMediaStore.getState().layerOpacities[layerIndex] ?? 1;

    // Wrap as a single full-screen layer
    const layer: Layer = {
      id: `bg-layer-${layerIndex}-${state.compositionId}`,
      name: `Layer ${String.fromCharCode(65 + layerIndex)}: ${state.composition.name}`,
      visible: true,
      opacity: layerOpacity,
      blendMode: 'normal',
      source: { type: 'image', nestedComposition: nestedCompData },
      effects: [],
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: { x: 0, y: 0, z: 0 },
    };

    return layer;
  }

  /**
   * Build inner layers for a background composition (same logic as buildNestedLayers)
   */
  private buildInnerLayers(state: LayerCompState, playheadPosition: number): Layer[] {
    const layers: Layer[] = [];
    const videoTracks = state.tracks.filter(t => t.type === 'video' && t.visible !== false);

    // Clamp playhead to composition duration
    const time = Math.max(0, Math.min(playheadPosition, state.duration));

    for (const track of videoTracks) {
      // Find clip on this track at current time
      const clip = state.clips.find(
        c => c.trackId === track.id && time >= c.startTime && time < c.startTime + c.duration
      );
      if (!clip || clip.isLoading) continue;

      const clipLocalTime = time - clip.startTime;

      // Build transform
      const transform = clip.transform || DEFAULT_TRANSFORM;
      const layer = this.buildClipLayer(clip, clipLocalTime, transform);
      if (layer) {
        layers.push(layer);
      }
    }

    return layers;
  }

  /**
   * Build a single layer from a clip
   */
  private buildClipLayer(clip: TimelineClip, _clipLocalTime: number, transform: typeof DEFAULT_TRANSFORM): Layer | null {
    const baseLayer = {
      id: `bg-clip-${clip.id}`,
      name: clip.name,
      visible: true,
      opacity: transform.opacity ?? 1,
      blendMode: (transform.blendMode || 'normal') as any,
      effects: clip.effects || [],
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

    if (clip.source?.videoElement) {
      return {
        ...baseLayer,
        source: {
          type: 'video',
          videoElement: clip.source.videoElement,
          webCodecsPlayer: clip.source.webCodecsPlayer,
        },
      } as Layer;
    }

    if (clip.source?.imageElement) {
      return {
        ...baseLayer,
        source: { type: 'image', imageElement: clip.source.imageElement },
      } as Layer;
    }

    if (clip.source?.textCanvas) {
      return {
        ...baseLayer,
        source: { type: 'text', textCanvas: clip.source.textCanvas },
      } as Layer;
    }

    return null;
  }

  /**
   * Sync video elements for all background layers (each uses its own independent time)
   */
  syncVideoElements(_playheadPosition: number, _isPlaying: boolean): void {
    for (const [, state] of this.layerStates) {
      const time = this.getLayerTime(state);

      for (const clip of state.clips) {
        if (!clip.source?.videoElement) continue;

        const video = clip.source.videoElement;
        const isActive = time >= clip.startTime && time < clip.startTime + clip.duration;

        if (!isActive) {
          if (!video.paused) video.pause();
          continue;
        }

        const clipLocalTime = time - clip.startTime;
        const clipTime = clip.reversed
          ? clip.outPoint - clipLocalTime
          : clipLocalTime + clip.inPoint;

        const timeDiff = Math.abs(video.currentTime - clipTime);

        // Background layers always play (independent of global playback state)
        if (video.paused) video.play().catch(() => {});
        if (timeDiff > 0.5) {
          video.currentTime = clipTime;
        }
      }
    }
  }

  /**
   * Sync audio elements for all background layers (each uses its own independent time)
   */
  syncAudioElements(_playheadPosition: number, _isPlaying: boolean): void {
    for (const [, state] of this.layerStates) {
      const time = this.getLayerTime(state);

      for (const clip of state.clips) {
        if (!clip.source?.audioElement) continue;

        const audio = clip.source.audioElement;
        const isActive = time >= clip.startTime && time < clip.startTime + clip.duration;

        if (!isActive) {
          if (!audio.paused) audio.pause();
          continue;
        }

        const clipLocalTime = time - clip.startTime;
        const clipTime = clip.reversed
          ? clip.outPoint - clipLocalTime
          : clipLocalTime + clip.inPoint;

        const timeDiff = Math.abs(audio.currentTime - clipTime);

        // Background layers always play (independent of global playback state)
        if (audio.paused) audio.play().catch(() => {});
        if (timeDiff > 0.3) {
          audio.currentTime = clipTime;
        }
      }
    }
  }

  /**
   * Check if any background layers are active
   */
  hasActiveLayers(): boolean {
    return this.layerStates.size > 0;
  }

  /**
   * Get all active layer indices
   */
  getActiveLayerIndices(): number[] {
    return Array.from(this.layerStates.keys());
  }

  // === Private media loading methods ===

  private loadVideoForClip(clip: TimelineClip, layerIndex: number, url: string, mediaFileId: string): void {
    const video = document.createElement('video');
    video.src = url;
    video.muted = true; // Background layers muted by default
    video.playsInline = true;
    video.preload = 'auto';
    video.crossOrigin = 'anonymous';

    video.addEventListener('canplaythrough', () => {
      clip.source = {
        type: 'video',
        videoElement: video,
        naturalDuration: video.duration,
        mediaFileId,
      };
      clip.isLoading = false;
      log.debug(`Video loaded for background clip ${clip.name} on layer ${layerIndex}`);
      // Pre-cache frame via createImageBitmap for immediate scrubbing without play()
      engine.preCacheVideoFrame(video);
    }, { once: true });

    video.addEventListener('error', () => {
      clip.isLoading = false;
      log.warn(`Failed to load video for background clip ${clip.name}`);
    }, { once: true });
  }

  private loadAudioForClip(clip: TimelineClip, layerIndex: number, url: string, mediaFileId: string): void {
    const audio = document.createElement('audio');
    audio.src = url;
    audio.preload = 'auto';

    audio.addEventListener('canplaythrough', () => {
      clip.source = {
        type: 'audio',
        audioElement: audio,
        naturalDuration: audio.duration,
        mediaFileId,
      };
      clip.isLoading = false;
      log.debug(`Audio loaded for background clip ${clip.name} on layer ${layerIndex}`);
    }, { once: true });

    audio.addEventListener('error', () => {
      clip.isLoading = false;
      log.warn(`Failed to load audio for background clip ${clip.name}`);
    }, { once: true });
  }

  private loadImageForClip(clip: TimelineClip, layerIndex: number, url: string): void {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = url;

    img.addEventListener('load', () => {
      clip.source = {
        type: 'image',
        imageElement: img,
      };
      clip.isLoading = false;
      log.debug(`Image loaded for background clip ${clip.name} on layer ${layerIndex}`);
    }, { once: true });

    img.addEventListener('error', () => {
      clip.isLoading = false;
      log.warn(`Failed to load image for background clip ${clip.name}`);
    }, { once: true });
  }
}

// Singleton
export const layerPlaybackManager = new LayerPlaybackManager();
