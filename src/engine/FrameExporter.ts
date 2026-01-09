// Frame-by-frame exporter for precise video rendering
// Combines VideoEncoder and FrameExporter in one file to avoid import issues

import { Muxer, ArrayBufferTarget } from 'mp4-muxer';
import { engine } from './WebGPUEngine';
import { useTimelineStore } from '../stores/timeline';
import type { Layer } from '../types';

// ============ TYPES ============

export interface ExportSettings {
  width: number;
  height: number;
  fps: number;
  codec: 'h264' | 'vp9';
  bitrate: number;
  startTime: number;
  endTime: number;
}

export interface ExportProgress {
  currentFrame: number;
  totalFrames: number;
  percent: number;
  estimatedTimeRemaining: number;
  currentTime: number;
}

export interface FullExportSettings extends ExportSettings {
  filename?: string;
}

// ============ VIDEO ENCODER ============

class VideoEncoderWrapper {
  private encoder: VideoEncoder | null = null;
  private muxer: Muxer<ArrayBufferTarget> | null = null;
  private settings: ExportSettings;
  private encodedFrameCount = 0;
  private isClosed = false;

  constructor(settings: ExportSettings) {
    this.settings = settings;
  }

  async init(): Promise<boolean> {
    if (!('VideoEncoder' in window)) {
      console.error('[VideoEncoder] WebCodecs not supported');
      return false;
    }

    const codecString = this.settings.codec === 'h264'
      ? 'avc1.640028'
      : 'vp09.00.10.08';

    try {
      const support = await VideoEncoder.isConfigSupported({
        codec: codecString,
        width: this.settings.width,
        height: this.settings.height,
        bitrate: this.settings.bitrate,
        framerate: this.settings.fps,
      });

      if (!support.supported) {
        console.error('[VideoEncoder] Codec not supported:', codecString);
        return false;
      }
    } catch (e) {
      console.error('[VideoEncoder] Codec support check failed:', e);
      return false;
    }

    this.muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: this.settings.codec === 'h264' ? 'avc' : 'vp9',
        width: this.settings.width,
        height: this.settings.height,
      },
      fastStart: 'in-memory',
    });

    this.encoder = new VideoEncoder({
      output: (chunk, meta) => {
        if (this.muxer) {
          this.muxer.addVideoChunk(chunk, meta);
        }
        this.encodedFrameCount++;
      },
      error: (e) => {
        console.error('[VideoEncoder] Encode error:', e);
      },
    });

    await this.encoder.configure({
      codec: codecString,
      width: this.settings.width,
      height: this.settings.height,
      bitrate: this.settings.bitrate,
      framerate: this.settings.fps,
      latencyMode: 'quality',
      bitrateMode: 'variable',
    });

    console.log(`[VideoEncoder] Initialized: ${this.settings.width}x${this.settings.height} @ ${this.settings.fps}fps`);
    return true;
  }

  async encodeFrame(pixels: Uint8ClampedArray, frameIndex: number): Promise<void> {
    if (!this.encoder || this.isClosed) {
      throw new Error('Encoder not initialized or already closed');
    }

    const timestampMicros = Math.round(frameIndex * (1_000_000 / this.settings.fps));
    const durationMicros = Math.round(1_000_000 / this.settings.fps);

    const frame = new VideoFrame(pixels.buffer, {
      format: 'RGBA',
      codedWidth: this.settings.width,
      codedHeight: this.settings.height,
      timestamp: timestampMicros,
      duration: durationMicros,
    });

    const keyFrame = frameIndex % 30 === 0;
    this.encoder.encode(frame, { keyFrame });
    frame.close();

    if (frameIndex % 10 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  async finish(): Promise<Blob> {
    if (!this.encoder || !this.muxer) {
      throw new Error('Encoder not initialized');
    }

    this.isClosed = true;
    await this.encoder.flush();
    this.encoder.close();
    this.muxer.finalize();

    const { buffer } = this.muxer.target;
    const mimeType = this.settings.codec === 'h264' ? 'video/mp4' : 'video/webm';

    console.log(`[VideoEncoder] Finished: ${this.encodedFrameCount} frames, ${(buffer.byteLength / 1024 / 1024).toFixed(2)}MB`);
    return new Blob([buffer], { type: mimeType });
  }

  cancel(): void {
    if (this.encoder && !this.isClosed) {
      this.isClosed = true;
      try {
        this.encoder.close();
      } catch {}
    }
  }
}

// ============ FRAME EXPORTER ============

export class FrameExporter {
  private settings: FullExportSettings;
  private encoder: VideoEncoderWrapper | null = null;
  private isCancelled = false;
  private frameTimes: number[] = [];

  constructor(settings: FullExportSettings) {
    this.settings = settings;
  }

  async export(onProgress: (progress: ExportProgress) => void): Promise<Blob | null> {
    const { fps, startTime, endTime, width, height } = this.settings;
    const frameDuration = 1 / fps;
    const totalFrames = Math.ceil((endTime - startTime) * fps);

    console.log(`[FrameExporter] Starting export: ${width}x${height} @ ${fps}fps, ${totalFrames} frames`);

    this.encoder = new VideoEncoderWrapper(this.settings);
    const initialized = await this.encoder.init();
    if (!initialized) {
      console.error('[FrameExporter] Failed to initialize encoder');
      return null;
    }

    const originalDimensions = engine.getOutputDimensions();
    engine.setResolution(width, height);

    try {
      for (let frame = 0; frame < totalFrames; frame++) {
        if (this.isCancelled) {
          console.log('[FrameExporter] Export cancelled');
          this.encoder.cancel();
          engine.setResolution(originalDimensions.width, originalDimensions.height);
          return null;
        }

        const frameStart = performance.now();
        const time = startTime + frame * frameDuration;

        await this.seekAllClipsToTime(time);
        const layers = this.buildLayersAtTime(time);

        engine.render(layers);

        const pixels = await engine.readPixels();
        if (!pixels) {
          console.error('[FrameExporter] Failed to read pixels at frame', frame);
          continue;
        }

        await this.encoder.encodeFrame(pixels, frame);

        const frameTime = performance.now() - frameStart;
        this.frameTimes.push(frameTime);
        if (this.frameTimes.length > 30) this.frameTimes.shift();

        const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        const remainingFrames = totalFrames - frame - 1;
        const estimatedTimeRemaining = (remainingFrames * avgFrameTime) / 1000;

        onProgress({
          currentFrame: frame + 1,
          totalFrames,
          percent: ((frame + 1) / totalFrames) * 100,
          estimatedTimeRemaining,
          currentTime: time,
        });
      }

      const blob = await this.encoder.finish();
      console.log(`[FrameExporter] Export complete: ${(blob.size / 1024 / 1024).toFixed(2)}MB`);
      engine.setResolution(originalDimensions.width, originalDimensions.height);
      return blob;
    } catch (error) {
      console.error('[FrameExporter] Export error:', error);
      engine.setResolution(originalDimensions.width, originalDimensions.height);
      return null;
    }
  }

  cancel(): void {
    this.isCancelled = true;
  }

  private async seekAllClipsToTime(time: number): Promise<void> {
    const clips = useTimelineStore.getState().getClipsAtTime(time);
    const tracks = useTimelineStore.getState().tracks;
    const seekPromises: Promise<void>[] = [];

    for (const clip of clips) {
      const track = tracks.find(t => t.id === clip.trackId);
      if (!track?.visible) continue;

      if (clip.source?.type === 'video' && clip.source.videoElement) {
        const video = clip.source.videoElement;
        const clipLocalTime = time - clip.startTime;
        // Handle reversed clips
        const clipTime = clip.reversed
          ? clip.outPoint - clipLocalTime
          : clipLocalTime + clip.inPoint;
        seekPromises.push(this.seekVideo(video, clipTime));
      }
    }

    await Promise.all(seekPromises);
  }

  private seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
    return new Promise((resolve) => {
      if (Math.abs(video.currentTime - time) < 0.001 && !video.seeking && video.readyState >= 2) {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        console.warn('[FrameExporter] Seek timeout at', time);
        resolve();
      }, 1000);

      const onSeeked = () => {
        clearTimeout(timeout);
        video.removeEventListener('seeked', onSeeked);

        // Wait for video to be fully ready (not seeking, has data)
        const waitForReady = () => {
          if (!video.seeking && video.readyState >= 2) {
            // Give browser one more frame to finalize
            requestAnimationFrame(() => resolve());
          } else {
            requestAnimationFrame(waitForReady);
          }
        };
        waitForReady();
      };

      video.addEventListener('seeked', onSeeked);
      video.currentTime = Math.max(0, Math.min(time, video.duration || 0));
    });
  }

  private buildLayersAtTime(time: number): Layer[] {
    const clips = useTimelineStore.getState().getClipsAtTime(time);
    const tracks = useTimelineStore.getState().tracks;
    const layers: Layer[] = [];

    const videoTracks = tracks.filter(t => t.type === 'video');
    const trackOrder = new Map(videoTracks.map((t, i) => [t.id, i]));

    // Check if any video track has solo enabled
    const anyVideoSolo = videoTracks.some(t => t.solo);

    // Helper to determine effective visibility for a video track (respecting solo)
    const isTrackVisible = (track: typeof videoTracks[0]) => {
      if (!track.visible) return false;
      if (anyVideoSolo) return track.solo;
      return true;
    };

    const sortedClips = clips
      .filter(clip => {
        const track = tracks.find(t => t.id === clip.trackId);
        return track?.type === 'video' && isTrackVisible(track);
      })
      .sort((a, b) => {
        const orderA = trackOrder.get(a.trackId) ?? 0;
        const orderB = trackOrder.get(b.trackId) ?? 0;
        return orderA - orderB;
      });

    for (const clip of sortedClips) {
      if (!clip.source) continue;

      const layer: Layer = {
        id: clip.id,
        name: clip.name,
        visible: true,
        opacity: clip.transform.opacity,
        blendMode: clip.transform.blendMode,
        source: null,
        effects: [],
        position: { x: clip.transform.position.x, y: clip.transform.position.y },
        scale: { x: clip.transform.scale.x, y: clip.transform.scale.y },
        rotation: { x: clip.transform.rotation.x * (Math.PI / 180), y: clip.transform.rotation.y * (Math.PI / 180), z: clip.transform.rotation.z * (Math.PI / 180) },
      };

      if (clip.source.type === 'video' && clip.source.videoElement) {
        layer.source = { type: 'video', videoElement: clip.source.videoElement };
      } else if (clip.source.type === 'image' && clip.source.imageElement) {
        layer.source = { type: 'image', imageElement: clip.source.imageElement };
      }

      if (layer.source) layers.push(layer);
    }

    return layers;
  }

  static isSupported(): boolean {
    return 'VideoEncoder' in window && 'VideoFrame' in window;
  }

  static getPresetResolutions() {
    return [
      { label: '4K (3840x2160)', width: 3840, height: 2160 },
      { label: '1080p (1920x1080)', width: 1920, height: 1080 },
      { label: '720p (1280x720)', width: 1280, height: 720 },
      { label: '480p (854x480)', width: 854, height: 480 },
    ];
  }

  static getPresetFrameRates() {
    return [
      { label: '60 fps', fps: 60 },
      { label: '30 fps', fps: 30 },
      { label: '25 fps (PAL)', fps: 25 },
      { label: '24 fps (Film)', fps: 24 },
    ];
  }

  static getRecommendedBitrate(width: number, _height: number, _fps: number): number {
    if (width >= 3840) return 35_000_000;
    if (width >= 1920) return 15_000_000;
    if (width >= 1280) return 8_000_000;
    return 5_000_000;
  }
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
