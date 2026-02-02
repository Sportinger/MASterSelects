// CompositionRenderer - Evaluates any composition at a given time and returns renderable layers
// This enables multiple previews showing different compositions simultaneously

import { Logger } from './logger';
import type { Layer, SerializableClip, TimelineTrack, TimelineClip, NestedCompositionData } from '../types';

const log = Logger.create('CompositionRenderer');
import { useMediaStore } from '../stores/mediaStore';
import { useTimelineStore } from '../stores/timeline';
import { calculateSourceTime } from '../utils/speedIntegration';
import { textRenderer } from './textRenderer';
import { proxyFrameCache } from './proxyFrameCache';

// Source cache entry for a composition
interface CompositionSources {
  compositionId: string;
  clipSources: Map<string, {
    clipId: string;
    type: 'video' | 'image' | 'audio' | 'text';
    videoElement?: HTMLVideoElement;
    webCodecsPlayer?: import('../engine/WebCodecsPlayer').WebCodecsPlayer;
    imageElement?: HTMLImageElement;
    textCanvas?: HTMLCanvasElement;
    file?: File;
    naturalDuration: number;
  }>;
  isReady: boolean;
  lastAccessTime: number;
}

// Evaluated layer result
export interface EvaluatedLayer extends Omit<Layer, 'id'> {
  id: string;
  clipId: string;
}

class CompositionRendererService {
  // Cache of prepared sources per composition
  private compositionSources: Map<string, CompositionSources> = new Map();

  // Callbacks for when a composition is ready
  private readyCallbacks: Map<string, (() => void)[]> = new Map();

  /**
   * Prepare a composition for rendering - loads all video/image sources
   */
  async prepareComposition(compositionId: string): Promise<boolean> {
    log.info(`prepareComposition called for ${compositionId}`);

    // Already prepared?
    const existing = this.compositionSources.get(compositionId);
    if (existing?.isReady) {
      log.debug(`prepareComposition: already ready, returning cached`);
      existing.lastAccessTime = Date.now();
      return true;
    }
    log.debug(`prepareComposition: not ready, preparing...`);

    const { activeCompositionId } = useMediaStore.getState();
    const composition = useMediaStore.getState().compositions.find(c => c.id === compositionId);

    if (!composition) {
      log.warn(`Composition ${compositionId} not found`);
      return false;
    }

    // Check if this is the active composition - use timeline store data
    const isActiveComp = compositionId === activeCompositionId;

    let clips: (SerializableClip | TimelineClip)[];

    if (isActiveComp) {
      // Active composition - use live data from timeline store
      clips = useTimelineStore.getState().clips;
      log.info(`Preparing ACTIVE composition: ${composition.name} (${clips.length} clips from timeline store)`);
    } else if (composition.timelineData) {
      // Non-active composition - use serialized data
      clips = composition.timelineData.clips || [];
      log.info(`Preparing composition: ${composition.name} (${clips.length} clips from timelineData)`);
    } else {
      log.warn(`Composition ${compositionId} has no timeline data`);
      return false;
    }

    const sources: CompositionSources = {
      compositionId,
      clipSources: new Map(),
      isReady: false,
      lastAccessTime: Date.now(),
    };

    this.compositionSources.set(compositionId, sources);

    const mediaFiles = useMediaStore.getState().files;

    // Load sources for all video/image clips
    const loadPromises: Promise<void>[] = [];

    for (const clip of clips) {
      // Handle both TimelineClip (active) and SerializableClip (stored)
      const timelineClip = clip as TimelineClip;
      const serializableClip = clip as SerializableClip;

      // Get source type - TimelineClip has source.type, SerializableClip has sourceType
      const sourceType = timelineClip.source?.type || serializableClip.sourceType;

      // Get media file ID
      const mediaFileId = timelineClip.source?.mediaFileId || serializableClip.mediaFileId;

      log.debug(`Processing clip ${clip.id}: sourceType=${sourceType}, mediaFileId=${mediaFileId || 'NONE'}, isActive=${isActiveComp}`);

      if (!mediaFileId) {
        // For active composition, the video/image/text elements are already loaded
        if (isActiveComp && timelineClip.source) {
          if (sourceType === 'video' && timelineClip.source.videoElement) {
            sources.clipSources.set(clip.id, {
              clipId: clip.id,
              type: 'video',
              videoElement: timelineClip.source.videoElement,
              webCodecsPlayer: timelineClip.source.webCodecsPlayer, // Pass through WebCodecsPlayer for hardware decoding
              file: timelineClip.file,
              naturalDuration: timelineClip.source.naturalDuration || timelineClip.source.videoElement.duration || 0,
            });
          } else if (sourceType === 'image' && timelineClip.source.imageElement) {
            sources.clipSources.set(clip.id, {
              clipId: clip.id,
              type: 'image',
              imageElement: timelineClip.source.imageElement,
              file: timelineClip.file,
              naturalDuration: timelineClip.source.naturalDuration || 5,
            });
          } else if (sourceType === 'text' && timelineClip.source.textCanvas) {
            sources.clipSources.set(clip.id, {
              clipId: clip.id,
              type: 'text',
              textCanvas: timelineClip.source.textCanvas,
              naturalDuration: clip.duration,
            });
          }
        }

        // Handle text clips from serialized data (non-active composition)
        if (sourceType === 'text' && serializableClip.textProperties) {
          const textCanvas = textRenderer.render(serializableClip.textProperties);
          if (textCanvas) {
            sources.clipSources.set(clip.id, {
              clipId: clip.id,
              type: 'text',
              textCanvas,
              naturalDuration: clip.duration,
            });
          }
        }

        continue;
      }

      // Find the media file
      const mediaFile = mediaFiles.find(f => f.id === mediaFileId);

      if (!mediaFile?.file) {
        log.warn(`Media file not found for clip ${clip.id}`, {
          mediaFileId,
          availableFileIds: mediaFiles.map(f => f.id).slice(0, 5), // First 5 for brevity
          totalFiles: mediaFiles.length,
        });
        continue;
      }

      log.debug(`Found media file for clip ${clip.id}: ${mediaFile.name}`);

      if (sourceType === 'video') {
        loadPromises.push(this.loadVideoSource(sources, serializableClip, mediaFile.file));
      } else if (sourceType === 'image') {
        loadPromises.push(this.loadImageSource(sources, serializableClip, mediaFile.file));
      }
    }

    // Wait for all sources to load
    log.info(`prepareComposition: waiting for ${loadPromises.length} sources to load`);
    await Promise.all(loadPromises);

    sources.isReady = true;
    log.info(`Composition ready: ${composition.name}, ${sources.clipSources.size} sources loaded`);

    if (sources.clipSources.size === 0 && clips.length > 0) {
      log.warn(`prepareComposition: No sources loaded for ${clips.length} clips! Check mediaFileId values.`);
      for (const clip of clips) {
        const sc = clip as SerializableClip;
        log.warn(`  Clip ${clip.id}: sourceType=${sc.sourceType}, mediaFileId=${sc.mediaFileId || 'MISSING'}`);
      }
    }

    // Notify any waiting callbacks
    const callbacks = this.readyCallbacks.get(compositionId) || [];
    callbacks.forEach(cb => cb());
    this.readyCallbacks.delete(compositionId);

    return true;
  }

  private loadVideoSource(sources: CompositionSources, clip: SerializableClip, file: File): Promise<void> {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.loop = false; // We control playback manually
      video.muted = true;
      video.playsInline = true;
      video.crossOrigin = 'anonymous';
      video.preload = 'auto';

      video.addEventListener('canplaythrough', () => {
        sources.clipSources.set(clip.id, {
          clipId: clip.id,
          type: 'video',
          videoElement: video,
          file,
          naturalDuration: video.duration || clip.naturalDuration || 0,
        });
        log.debug(`Video loaded: ${file.name}`);
        resolve();
      }, { once: true });

      video.addEventListener('error', () => {
        log.error(`Failed to load video: ${file.name}`);
        resolve(); // Don't block on errors
      }, { once: true });

      video.load();
    });
  }

  private loadImageSource(sources: CompositionSources, clip: SerializableClip, file: File): Promise<void> {
    return new Promise((resolve) => {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.crossOrigin = 'anonymous';

      img.onload = () => {
        sources.clipSources.set(clip.id, {
          clipId: clip.id,
          type: 'image',
          imageElement: img,
          file,
          naturalDuration: clip.naturalDuration || 5,
        });
        log.debug(`Image loaded: ${file.name}`);
        resolve();
      };

      img.onerror = () => {
        log.error(`Failed to load image: ${file.name}`);
        resolve();
      };
    });
  }

  /**
   * Evaluate a composition at a specific time - returns layers ready for rendering
   */
  evaluateAtTime(compositionId: string, time: number): EvaluatedLayer[] {
    const sources = this.compositionSources.get(compositionId);
    if (!sources?.isReady) {
      log.warn(`evaluateAtTime: sources not ready for ${compositionId}`);
      return [];
    }

    sources.lastAccessTime = Date.now();

    const { activeCompositionId } = useMediaStore.getState();
    const composition = useMediaStore.getState().compositions.find(c => c.id === compositionId);
    if (!composition) {
      log.warn(`evaluateAtTime: composition not found ${compositionId}`);
      return [];
    }

    // Check if this is the active composition
    const isActiveComp = compositionId === activeCompositionId;
    log.debug(`evaluateAtTime: ${composition.name}, isActive=${isActiveComp}, time=${time.toFixed(2)}`);

    let clips: (SerializableClip | TimelineClip)[];
    let tracks: TimelineTrack[];

    if (isActiveComp) {
      // Active composition - use live data from timeline store
      const timelineState = useTimelineStore.getState();
      clips = timelineState.clips;
      tracks = timelineState.tracks;
    } else if (composition.timelineData) {
      // Non-active composition - use serialized data
      clips = composition.timelineData.clips || [];
      tracks = composition.timelineData.tracks || [];
      log.debug(`evaluateAtTime: using timelineData, ${clips.length} clips, ${tracks.length} tracks`);

      // Log clip details for debugging
      for (const clip of clips) {
        const sc = clip as SerializableClip;
        log.debug(`evaluateAtTime clip: ${sc.id}, type=${sc.sourceType}, mediaFileId=${sc.mediaFileId || 'NONE'}`);
      }
    } else {
      log.warn(`evaluateAtTime: comp ${composition.name} has NO timelineData!`);
      return [];
    }

    // Find video tracks (in order for layering)
    const videoTracks = tracks.filter((t: TimelineTrack) => t.type === 'video');
    log.debug(`evaluateAtTime: ${videoTracks.length} video tracks, clipSources: ${sources.clipSources.size}`);

    // Build layers from bottom to top (reverse track order)
    const layers: EvaluatedLayer[] = [];

    for (let trackIndex = videoTracks.length - 1; trackIndex >= 0; trackIndex--) {
      const track = videoTracks[trackIndex];

      // Find clip at current time on this track
      const clipAtTime = clips.find((c) =>
        c.trackId === track.id &&
        time >= c.startTime &&
        time < c.startTime + c.duration
      );

      if (!clipAtTime) continue;
      if (!track.visible) continue;

      // Handle nested compositions
      const timelineClip = clipAtTime as TimelineClip;
      if (timelineClip.isComposition && timelineClip.compositionId) {
        const nestedLayer = this.evaluateNestedComposition(
          timelineClip,
          time,
          compositionId
        );
        if (nestedLayer) {
          layers.push(nestedLayer);
        }
        continue;
      }

      const source = sources.clipSources.get(clipAtTime.id);
      if (!source) continue;

      // Calculate clip-local time (on timeline, relative to clip start)
      const timelineLocalTime = time - clipAtTime.startTime;
      // Calculate source time using speed (nested comps don't have keyframes, use default speed)
      const defaultSpeed = clipAtTime.speed ?? (clipAtTime.reversed ? -1 : 1);
      const sourceTime = calculateSourceTime([], timelineLocalTime, defaultSpeed);
      // Determine start point based on playback direction
      const startPoint = defaultSpeed >= 0 ? (clipAtTime.inPoint || 0) : (clipAtTime.outPoint || source.naturalDuration);
      const clipTime = Math.max(0, Math.min(source.naturalDuration, startPoint + sourceTime));

      // Seek video to correct time
      if (source.videoElement) {
        // Only seek if significantly different (avoid micro-seeks)
        if (Math.abs(source.videoElement.currentTime - clipTime) > 0.05) {
          source.videoElement.currentTime = clipTime;
        }
      }

      // Build layer object
      const transform = clipAtTime.transform || {
        position: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1 },
        rotation: { x: 0, y: 0, z: 0 },
        anchor: { x: 0.5, y: 0.5 },
        opacity: 1,
      };

      // Build layer source based on type
      let layerSource: EvaluatedLayer['source'] = null;
      if (source.videoElement) {
        layerSource = {
          type: 'video',
          file: source.file,
          videoElement: source.videoElement,
          webCodecsPlayer: source.webCodecsPlayer,
        };
      } else if (source.imageElement) {
        layerSource = {
          type: 'image',
          file: source.file,
          imageElement: source.imageElement,
        };
      } else if (source.textCanvas) {
        layerSource = {
          type: 'text',
          textCanvas: source.textCanvas,
        };
      }

      const layer: EvaluatedLayer = {
        id: `${compositionId}-${clipAtTime.id}`,
        clipId: clipAtTime.id,
        name: clipAtTime.name,
        visible: true,
        opacity: transform.opacity ?? 1,
        blendMode: 'normal',
        source: layerSource,
        effects: clipAtTime.effects || [],
        position: transform.position || { x: 0, y: 0, z: 0 },
        scale: transform.scale || { x: 1, y: 1 },
        rotation: typeof transform.rotation === 'number'
          ? transform.rotation
          : transform.rotation?.z || 0,
      };

      layers.push(layer);
    }

    return layers;
  }

  /**
   * Evaluate a nested composition clip and return a layer with nested composition data
   */
  private evaluateNestedComposition(
    clip: TimelineClip,
    parentTime: number,
    parentCompId: string
  ): EvaluatedLayer | null {
    if (!clip.nestedClips || !clip.nestedTracks) {
      return null;
    }

    // Calculate time within the nested composition
    const clipLocalTime = parentTime - clip.startTime;
    const nestedTime = clipLocalTime + (clip.inPoint || 0);

    // Get composition dimensions
    const mediaStore = useMediaStore.getState();
    const nestedComp = mediaStore.compositions.find(c => c.id === clip.compositionId);
    const compWidth = nestedComp?.width || 1920;
    const compHeight = nestedComp?.height || 1080;

    // Build layers for the nested composition
    const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
    const nestedLayers: Layer[] = [];

    for (let i = nestedVideoTracks.length - 1; i >= 0; i--) {
      const nestedTrack = nestedVideoTracks[i];
      const nestedClip = clip.nestedClips.find(
        nc =>
          nc.trackId === nestedTrack.id &&
          nestedTime >= nc.startTime &&
          nestedTime < nc.startTime + nc.duration
      );

      if (!nestedClip) continue;

      const nestedLocalTime = nestedTime - nestedClip.startTime;
      const nestedClipTime = nestedClip.reversed
        ? nestedClip.outPoint - nestedLocalTime
        : nestedLocalTime + nestedClip.inPoint;

      // Build transform
      const transform = nestedClip.transform || {
        position: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1 },
        rotation: { x: 0, y: 0, z: 0 },
        anchor: { x: 0.5, y: 0.5 },
        opacity: 1,
        blendMode: 'normal' as const,
      };

      const baseLayer = {
        id: `${parentCompId}-nested-${nestedClip.id}`,
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

      // Handle video clips with proxy support
      if (nestedClip.source?.videoElement) {
        const nestedMediaFile = mediaStore.files.find(f =>
          f.id === nestedClip.source?.mediaFileId ||
          f.name === nestedClip.file?.name ||
          f.name === nestedClip.name
        );

        const shouldUseProxy = mediaStore.proxyEnabled &&
          nestedMediaFile?.proxyFps &&
          (nestedMediaFile.proxyStatus === 'ready' || nestedMediaFile.proxyStatus === 'generating');

        if (shouldUseProxy && nestedMediaFile) {
          const proxyFps = nestedMediaFile.proxyFps || 30;
          const frameIndex = Math.floor(nestedClipTime * proxyFps);
          const cachedFrame = proxyFrameCache.getCachedFrame(nestedMediaFile.id, frameIndex, proxyFps);

          if (cachedFrame) {
            nestedLayers.push({
              ...baseLayer,
              source: {
                type: 'image',
                imageElement: cachedFrame,
              },
            } as Layer);
            continue;
          }
        }

        // Sync video time
        const video = nestedClip.source.videoElement;
        if (Math.abs(video.currentTime - nestedClipTime) > 0.05) {
          video.currentTime = nestedClipTime;
        }

        nestedLayers.push({
          ...baseLayer,
          source: {
            type: 'video',
            videoElement: video,
            webCodecsPlayer: nestedClip.source.webCodecsPlayer,
          },
        } as Layer);
      } else if (nestedClip.source?.imageElement) {
        nestedLayers.push({
          ...baseLayer,
          source: {
            type: 'image',
            imageElement: nestedClip.source.imageElement,
          },
        } as Layer);
      } else if (nestedClip.source?.textCanvas) {
        nestedLayers.push({
          ...baseLayer,
          source: {
            type: 'text',
            textCanvas: nestedClip.source.textCanvas,
          },
        } as Layer);
      }
    }

    if (nestedLayers.length === 0) {
      return null;
    }

    // Build the nested composition data
    const nestedCompData: NestedCompositionData = {
      compositionId: clip.compositionId || clip.id,
      layers: nestedLayers,
      width: compWidth,
      height: compHeight,
    };

    // Get clip transform
    const clipTransform = clip.transform || {
      position: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1 },
      rotation: { x: 0, y: 0, z: 0 },
      opacity: 1,
      blendMode: 'normal' as const,
    };

    return {
      id: `${parentCompId}-${clip.id}`,
      clipId: clip.id,
      name: clip.name,
      visible: true,
      opacity: clipTransform.opacity ?? 1,
      blendMode: clipTransform.blendMode || 'normal',
      source: {
        type: 'video',
        nestedComposition: nestedCompData,
      },
      effects: clip.effects || [],
      position: clipTransform.position || { x: 0, y: 0, z: 0 },
      scale: clipTransform.scale || { x: 1, y: 1 },
      rotation: typeof clipTransform.rotation === 'number'
        ? clipTransform.rotation
        : (clipTransform.rotation?.z || 0) * Math.PI / 180,
    };
  }

  /**
   * Check if a composition is prepared and ready
   */
  isReady(compositionId: string): boolean {
    return this.compositionSources.get(compositionId)?.isReady ?? false;
  }

  /**
   * Wait for a composition to be ready
   */
  onReady(compositionId: string, callback: () => void): void {
    if (this.isReady(compositionId)) {
      callback();
      return;
    }

    const callbacks = this.readyCallbacks.get(compositionId) || [];
    callbacks.push(callback);
    this.readyCallbacks.set(compositionId, callbacks);
  }

  /**
   * Dispose of a composition's sources
   */
  disposeComposition(compositionId: string): void {
    const sources = this.compositionSources.get(compositionId);
    if (!sources) return;

    for (const source of sources.clipSources.values()) {
      if (source.videoElement) {
        source.videoElement.pause();
        URL.revokeObjectURL(source.videoElement.src);
      }
      if (source.imageElement) {
        URL.revokeObjectURL(source.imageElement.src);
      }
    }

    this.compositionSources.delete(compositionId);
    log.debug(`Disposed composition: ${compositionId}`);
  }

  /**
   * Get list of prepared compositions
   */
  getPreparedCompositions(): string[] {
    return Array.from(this.compositionSources.keys()).filter(id =>
      this.compositionSources.get(id)?.isReady
    );
  }

  /**
   * Cleanup unused compositions (those not accessed recently)
   */
  cleanup(maxAgeMs: number = 60000): void {
    const now = Date.now();
    for (const [id, sources] of this.compositionSources.entries()) {
      if (now - sources.lastAccessTime > maxAgeMs) {
        this.disposeComposition(id);
      }
    }
  }

  /**
   * Invalidate a composition's cache so it gets re-prepared on next use
   * Call this when a composition's timelineData changes
   */
  invalidateComposition(compositionId: string): void {
    const sources = this.compositionSources.get(compositionId);
    if (sources) {
      log.debug(`Invalidating composition: ${compositionId}`);
      // Mark as not ready - will be re-prepared on next access
      sources.isReady = false;
      // Clear cached clip sources (they may be stale)
      sources.clipSources.clear();
    }
  }

  /**
   * Invalidate all non-active compositions
   * Call this when switching active compositions (timelineData may have changed)
   */
  invalidateAllExceptActive(): void {
    const { activeCompositionId } = useMediaStore.getState();
    for (const [id, sources] of this.compositionSources.entries()) {
      if (id !== activeCompositionId) {
        sources.isReady = false;
        sources.clipSources.clear();
      }
    }
    log.debug('Invalidated all non-active compositions');
  }

  /**
   * Invalidate a composition AND all parent compositions that contain it as nested
   * Call this when a composition's content changes (clips added/removed/modified)
   */
  invalidateCompositionAndParents(compositionId: string): void {
    // First invalidate the composition itself
    this.invalidateComposition(compositionId);

    // Find all parent compositions that contain this as a nested comp
    const { compositions } = useMediaStore.getState();

    for (const comp of compositions) {
      if (comp.id === compositionId) continue;

      // Check if this composition contains the changed one as a nested clip
      const clips = comp.timelineData?.clips || [];
      const hasNested = clips.some(clip =>
        clip.isComposition && clip.compositionId === compositionId
      );

      if (hasNested) {
        log.debug(`Invalidating parent composition: ${comp.name} (contains ${compositionId})`);
        this.invalidateComposition(comp.id);
        // Recursively invalidate grandparents
        this.invalidateCompositionAndParents(comp.id);
      }
    }
  }

  /**
   * Invalidate ALL cached compositions - use when major changes occur
   */
  invalidateAll(): void {
    for (const [, sources] of this.compositionSources.entries()) {
      sources.isReady = false;
      sources.clipSources.clear();
    }
    log.debug('Invalidated ALL compositions');
  }
}

// Singleton instance
export const compositionRenderer = new CompositionRendererService();
