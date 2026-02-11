// RAM Preview Engine - generates cached frames for RAM preview playback
// Extracted from playbackSlice.startRamPreview to separate concerns:
// - This service: frame generation (seeking, layer building, rendering, caching)
// - playbackSlice: orchestration (start/stop, state updates, progress)

import type { Layer, NestedCompositionData, TimelineClip, TimelineTrack } from '../types';
import { RAM_PREVIEW_FPS, FRAME_TOLERANCE } from '../stores/timeline/constants';

/** Quantize time to frame grid (same as stores/timeline/utils.quantizeTime) */
function quantizeTime(time: number): number {
  return Math.round(time * 30) / 30;
}

/** Convert clip transform (degrees) to Layer rotation (radians) */
function degreesToRadians(deg: { x: number; y: number; z: number }): { x: number; y: number; z: number } {
  const f = Math.PI / 180;
  return { x: deg.x * f, y: deg.y * f, z: deg.z * f };
}

// Minimal engine interface — avoids importing WebGPUEngine class directly
export interface RamPreviewRenderEngine {
  render: (layers: Layer[]) => void;
  cacheCompositeFrame: (time: number) => Promise<void>;
}

export interface RamPreviewOptions {
  start: number;
  end: number;
  centerTime: number;
  clips: TimelineClip[];
  tracks: TimelineTrack[];
}

export interface RamPreviewDeps {
  isCancelled: () => boolean;
  isFrameCached: (quantizedTime: number) => boolean;
  getSourceTimeForClip: (clipId: string, localTime: number) => number;
  getInterpolatedSpeed: (clipId: string, time: number) => number;
  getCompositionDimensions: (compositionId: string) => { width: number; height: number };
  onFrameCached: (time: number) => void;
  onProgress: (percent: number) => void;
}

export interface RamPreviewResult {
  completed: boolean;
  frameCount: number;
}

export class RamPreviewEngine {
  private engine: RamPreviewRenderEngine;

  constructor(engine: RamPreviewRenderEngine) {
    this.engine = engine;
  }

  /**
   * Generate RAM preview frames spreading outward from centerTime.
   * Seeks videos, builds layers, renders, and caches each frame.
   */
  async generate(options: RamPreviewOptions, deps: RamPreviewDeps): Promise<RamPreviewResult> {
    const { start, end, clips, tracks } = options;
    const fps = RAM_PREVIEW_FPS;
    const frameInterval = 1 / fps;

    // Generate frame times spreading outward from playhead
    const frameTimes = this.buildFrameTimes(start, end, options.centerTime, frameInterval, clips);

    if (frameTimes.length === 0) {
      return { completed: true, frameCount: 0 };
    }

    const totalFrames = frameTimes.length;
    const videoTracks = tracks.filter(t => t.type === 'video');
    let completed = true;

    for (let frame = 0; frame < totalFrames; frame++) {
      if (deps.isCancelled()) {
        completed = false;
        break;
      }

      const time = frameTimes[frame];

      // Skip already-cached frames
      if (deps.isFrameCached(quantizeTime(time))) {
        deps.onProgress(((frame + 1) / totalFrames) * 100);
        continue;
      }

      // Get clips at this time
      const clipsAtTime = clips.filter(c =>
        time >= c.startTime && time < c.startTime + c.duration
      );

      // Build layers (seek videos + construct Layer objects)
      const layers = await this.buildLayersForFrame(
        time, clipsAtTime, videoTracks, deps
      );

      if (deps.isCancelled()) {
        completed = false;
        break;
      }

      // Verify video positions haven't drifted
      if (!this.verifyVideoPositions(time, clipsAtTime, deps)) {
        deps.onProgress(((frame + 1) / totalFrames) * 100);
        continue;
      }

      // Render and cache
      if (layers.length > 0) {
        this.engine.render(layers);
      }
      await this.engine.cacheCompositeFrame(time);
      deps.onFrameCached(time);

      // Update progress
      deps.onProgress(((frame + 1) / totalFrames) * 100);

      // Yield to allow UI updates every 3 frames
      if (frame % 3 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    return { completed, frameCount: totalFrames };
  }

  // === Private helpers ===

  /**
   * Generate frame times spreading outward from center, only at times
   * where video/image/composition clips exist.
   */
  private buildFrameTimes(
    start: number, end: number, centerTime: number,
    frameInterval: number, clips: TimelineClip[]
  ): number[] {
    const center = Math.max(start, Math.min(end, centerTime));
    const frameTimes: number[] = [];

    const hasContentAt = (time: number) =>
      clips.some(c =>
        time >= c.startTime &&
        time < c.startTime + c.duration &&
        (c.source?.type === 'video' || c.source?.type === 'image' || c.isComposition)
      );

    if (hasContentAt(center)) {
      frameTimes.push(center);
    }

    let offset = frameInterval;
    while (offset <= (end - start)) {
      const rightTime = center + offset;
      const leftTime = center - offset;

      if (rightTime <= end && hasContentAt(rightTime)) {
        frameTimes.push(rightTime);
      }
      if (leftTime >= start && hasContentAt(leftTime)) {
        frameTimes.push(leftTime);
      }

      offset += frameInterval;
    }

    return frameTimes;
  }

  /**
   * Build Layer[] for a single frame time by seeking all videos and
   * constructing layer objects from clips.
   */
  private async buildLayersForFrame(
    time: number,
    clipsAtTime: TimelineClip[],
    videoTracks: TimelineTrack[],
    deps: RamPreviewDeps
  ): Promise<Layer[]> {
    const layers: Layer[] = [];

    for (const clip of clipsAtTime) {
      const track = videoTracks.find(t => t.id === clip.trackId);
      if (!track?.visible) continue;

      if (clip.source?.type === 'video' && clip.source.videoElement) {
        const layer = await this.buildVideoLayer(clip, time, deps);
        if (layer) layers.push(layer);
      } else if (clip.source?.type === 'image' && clip.source.imageElement) {
        layers.push(this.buildImageLayer(clip));
      } else if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        const layer = await this.buildNestedCompLayer(clip, time, deps);
        if (layer) layers.push(layer);
      }
    }

    // Sort layers by track order
    const trackOrder = new Map(videoTracks.map((t, i) => [t.id, i]));
    layers.sort((a, b) => {
      const clipA = clipsAtTime.find(c => c.id === a.id);
      const clipB = clipsAtTime.find(c => c.id === b.id);
      const orderA = clipA ? (trackOrder.get(clipA.trackId) ?? 0) : 0;
      const orderB = clipB ? (trackOrder.get(clipB.trackId) ?? 0) : 0;
      return orderA - orderB;
    });

    return layers;
  }

  /** Seek and build a Layer for a video clip */
  private async buildVideoLayer(
    clip: TimelineClip, time: number, deps: RamPreviewDeps
  ): Promise<Layer | null> {
    const video = clip.source!.videoElement!;
    const webCodecsPlayer = clip.source!.webCodecsPlayer;

    // Calculate source time using speed integration (handles keyframes)
    const clipLocalTime = time - clip.startTime;
    const sourceTime = deps.getSourceTimeForClip(clip.id, clipLocalTime);
    const initialSpeed = deps.getInterpolatedSpeed(clip.id, 0);
    const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
    const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

    // Seek video
    if (webCodecsPlayer) {
      if (deps.isCancelled()) return null;
      try {
        await webCodecsPlayer.seekAsync(clipTime);
      } catch {
        video.currentTime = clipTime;
        await new Promise(r => setTimeout(r, 50));
      }
    } else {
      if (deps.isCancelled()) return null;
      await this.seekHTMLVideo(video, clipTime, 200);
    }

    if (deps.isCancelled()) return null;

    return this.clipToLayer(clip, {
      type: 'video', videoElement: video, webCodecsPlayer,
    });
  }

  /** Build a Layer for an image clip (no async needed) */
  private buildImageLayer(clip: TimelineClip): Layer {
    return this.clipToLayer(clip, {
      type: 'image', imageElement: clip.source!.imageElement!,
    });
  }

  /** Build a Layer for a nested composition clip */
  private async buildNestedCompLayer(
    clip: TimelineClip, time: number, deps: RamPreviewDeps
  ): Promise<Layer | null> {
    const clipLocalTime = time - clip.startTime;
    const clipTime = clipLocalTime + clip.inPoint;

    const nestedVideoTracks = clip.nestedTracks?.filter(t => t.type === 'video' && t.visible) || [];
    const nestedLayers: Layer[] = [];

    for (const nestedTrack of nestedVideoTracks) {
      const nestedClip = clip.nestedClips!.find(nc =>
        nc.trackId === nestedTrack.id &&
        clipTime >= nc.startTime &&
        clipTime < nc.startTime + nc.duration
      );
      if (!nestedClip) continue;

      const nestedLocalTime = clipTime - nestedClip.startTime;
      const nestedClipTime = nestedClip.reversed
        ? nestedClip.outPoint - nestedLocalTime
        : nestedLocalTime + nestedClip.inPoint;

      if (nestedClip.source?.videoElement) {
        const nestedVideo = nestedClip.source.videoElement;
        const nestedWebCodecs = nestedClip.source.webCodecsPlayer;

        if (nestedWebCodecs) {
          try {
            await nestedWebCodecs.seekAsync(nestedClipTime);
          } catch {
            nestedVideo.currentTime = nestedClipTime;
            await new Promise(r => setTimeout(r, 50));
          }
        } else {
          await this.seekHTMLVideo(nestedVideo, nestedClipTime, 150);
        }

        nestedLayers.push(this.clipToLayer(nestedClip, {
          type: 'video', videoElement: nestedVideo, webCodecsPlayer: nestedWebCodecs,
        }, `nested-${nestedClip.id}`));
      } else if (nestedClip.source?.imageElement) {
        nestedLayers.push(this.clipToLayer(nestedClip, {
          type: 'image', imageElement: nestedClip.source.imageElement,
        }, `nested-${nestedClip.id}`));
      }
    }

    if (nestedLayers.length === 0) return null;

    const { width: compWidth, height: compHeight } = deps.getCompositionDimensions(
      clip.compositionId || clip.id
    );

    const nestedCompData: NestedCompositionData = {
      compositionId: clip.compositionId || clip.id,
      layers: nestedLayers,
      width: compWidth,
      height: compHeight,
    };

    return this.clipToLayer(clip, {
      type: 'image', nestedComposition: nestedCompData,
    });
  }

  /** Convert a TimelineClip + source info into a Layer object */
  private clipToLayer(
    clip: TimelineClip,
    source: Layer['source'],
    idOverride?: string
  ): Layer {
    const pos = clip.transform?.position ?? { x: 0, y: 0, z: 0 };
    const scl = clip.transform?.scale ?? { x: 1, y: 1 };
    const rot = clip.transform?.rotation ?? { x: 0, y: 0, z: 0 };

    return {
      id: idOverride ?? clip.id,
      name: clip.name,
      visible: true,
      opacity: clip.transform?.opacity ?? 1,
      blendMode: clip.transform?.blendMode ?? 'normal',
      source,
      effects: clip.effects || [],
      position: { x: pos.x, y: pos.y, z: pos.z },
      scale: { x: scl.x, y: scl.y },
      rotation: degreesToRadians(rot),
    };
  }

  /** Seek an HTMLVideoElement with timeout */
  private seekHTMLVideo(video: HTMLVideoElement, targetTime: number, timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        video.removeEventListener('seeked', onSeeked);
        resolve();
      }, timeoutMs);

      const onSeeked = () => {
        clearTimeout(timeout);
        video.removeEventListener('seeked', onSeeked);
        resolve();
      };

      video.addEventListener('seeked', onSeeked);
      video.currentTime = targetTime;
    });
  }

  /**
   * Verify all videos at this frame are still at the expected position.
   * Returns false if any position drifted (e.g. user scrubbed during generation).
   */
  private verifyVideoPositions(
    time: number, clipsAtTime: TimelineClip[], deps: RamPreviewDeps
  ): boolean {
    for (const clip of clipsAtTime) {
      if (clip.source?.type !== 'video' || !clip.source.videoElement) continue;

      const video = clip.source.videoElement;
      const localTime = time - clip.startTime;
      const sourceTime = deps.getSourceTimeForClip(clip.id, localTime);
      const initialSpeed = deps.getInterpolatedSpeed(clip.id, 0);
      const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
      const expectedTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

      if (Math.abs(video.currentTime - expectedTime) > FRAME_TOLERANCE) {
        return false;
      }
    }
    return true;
  }
}
