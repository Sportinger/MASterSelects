// Frame-by-frame exporter for precise video rendering
// Main orchestrator - delegates to specialized modules

import { engine } from '../WebGPUEngine';
import { AudioExportPipeline, type EncodedAudioResult } from '../audio';
import { ParallelDecodeManager } from '../ParallelDecodeManager';
import { useTimelineStore } from '../../stores/timeline';
import type { FullExportSettings, ExportProgress, ExportMode, ExportClipState, FrameContext } from './types';
import { getFrameTolerance, getKeyframeInterval } from './types';
import { VideoEncoderWrapper } from './VideoEncoderWrapper';
import { prepareClipsForExport, cleanupExportMode } from './ClipPreparation';
import { seekAllClipsToTime, waitForAllVideosReady } from './VideoSeeker';
import { buildLayersAtTime, initializeLayerBuilder, cleanupLayerBuilder } from './ExportLayerBuilder';
import {
  RESOLUTION_PRESETS,
  FRAME_RATE_PRESETS,
  CONTAINER_FORMATS,
  getVideoCodecsForContainer,
  getRecommendedBitrate,
  BITRATE_RANGE,
  formatBitrate,
  checkCodecSupport,
} from './codecHelpers';

export class FrameExporter {
  private settings: FullExportSettings;
  private encoder: VideoEncoderWrapper | null = null;
  private audioPipeline: AudioExportPipeline | null = null;
  private isCancelled = false;
  private frameTimes: number[] = [];
  private clipStates: Map<string, ExportClipState> = new Map();
  private exportMode: ExportMode;
  private parallelDecoder: ParallelDecodeManager | null = null;
  private useParallelDecode = false;

  constructor(settings: FullExportSettings) {
    this.settings = settings;
    this.exportMode = settings.exportMode ?? 'fast';
  }

  async export(onProgress: (progress: ExportProgress) => void): Promise<Blob | null> {
    const { fps, startTime, endTime, width, height, includeAudio } = this.settings;
    const frameDuration = 1 / fps;
    const totalFrames = Math.ceil((endTime - startTime) * fps);

    console.log(`[FrameExporter] Starting export: ${width}x${height} @ ${fps}fps, ${totalFrames} frames, audio: ${includeAudio ? 'yes' : 'no'}`);

    // Initialize encoder
    this.encoder = new VideoEncoderWrapper(this.settings);
    const initialized = await this.encoder.init();
    if (!initialized) {
      console.error('[FrameExporter] Failed to initialize encoder');
      return null;
    }

    // Initialize audio pipeline
    if (includeAudio) {
      this.audioPipeline = new AudioExportPipeline({
        sampleRate: this.settings.audioSampleRate ?? 48000,
        bitrate: this.settings.audioBitrate ?? 256000,
        normalize: this.settings.normalizeAudio ?? false,
      });
    }

    const originalDimensions = engine.getOutputDimensions();
    engine.setResolution(width, height);
    engine.setExporting(true);

    // Initialize export canvas for zero-copy VideoFrame creation
    const useZeroCopy = engine.initExportCanvas(width, height);
    if (useZeroCopy) {
      console.log('[FrameExporter] Using zero-copy export path (OffscreenCanvas → VideoFrame)');
    } else {
      console.log('[FrameExporter] Falling back to readPixels export path');
    }

    try {
      // Prepare clips for export
      const preparation = await prepareClipsForExport(this.settings, this.exportMode);
      this.clipStates = preparation.clipStates;
      this.parallelDecoder = preparation.parallelDecoder;
      this.useParallelDecode = preparation.useParallelDecode;
      this.exportMode = preparation.exportMode;

      // Initialize layer builder cache (tracks don't change during export)
      const { tracks } = useTimelineStore.getState();
      initializeLayerBuilder(tracks);

      // Pre-calculate frame tolerance
      const frameTolerance = getFrameTolerance(fps);
      const keyframeInterval = getKeyframeInterval(fps);

      // Phase 1: Encode video frames
      for (let frame = 0; frame < totalFrames; frame++) {
        if (this.isCancelled) {
          console.log('[FrameExporter] Export cancelled');
          this.encoder.cancel();
          this.audioPipeline?.cancel();
          this.cleanup(originalDimensions);
          return null;
        }

        const frameStart = performance.now();
        const time = startTime + frame * frameDuration;

        // Create FrameContext once per frame - avoids repeated getState() calls
        const ctx = this.createFrameContext(time, fps, frameTolerance);

        if (frame % 30 === 0 || frame < 5) {
          console.log(`[FrameExporter] Processing frame ${frame}/${totalFrames} at time ${time.toFixed(3)}s`);
        }

        await seekAllClipsToTime(ctx, this.clipStates, this.parallelDecoder, this.useParallelDecode);
        await waitForAllVideosReady(ctx, this.clipStates, this.parallelDecoder, this.useParallelDecode);

        const layers = buildLayersAtTime(ctx, this.clipStates, this.parallelDecoder, this.useParallelDecode);

        if (layers.length === 0 && frame === 0) {
          console.warn('[FrameExporter] No layers at time', time);
        }

        // Check GPU device validity
        if (!engine.isDeviceValid()) {
          throw new Error('WebGPU device lost during export. Try keeping the browser tab in focus.');
        }

        engine.render(layers);

        // Calculate timestamp and duration in microseconds
        const timestampMicros = Math.round(frame * (1_000_000 / fps));
        const durationMicros = Math.round(1_000_000 / fps);

        if (useZeroCopy) {
          // Zero-copy path: create VideoFrame directly from OffscreenCanvas
          const videoFrame = engine.createVideoFrameFromExport(timestampMicros, durationMicros);
          if (!videoFrame) {
            if (!engine.isDeviceValid()) {
              throw new Error('WebGPU device lost during export. Try keeping the browser tab in focus.');
            }
            console.error('[FrameExporter] Failed to create VideoFrame at frame', frame);
            continue;
          }
          await this.encoder.encodeVideoFrame(videoFrame, frame, keyframeInterval);
          videoFrame.close();
        } else {
          // Fallback: read pixels from GPU (slower)
          const pixels = await engine.readPixels();
          if (!pixels) {
            if (!engine.isDeviceValid()) {
              throw new Error('WebGPU device lost during export. Try keeping the browser tab in focus.');
            }
            console.error('[FrameExporter] Failed to read pixels at frame', frame);
            continue;
          }
          await this.encoder.encodeFrame(pixels, frame, keyframeInterval);
        }

        // Early cancellation check after expensive encode
        if (this.isCancelled) {
          this.cleanup(originalDimensions);
          return null;
        }

        // Update progress
        const frameTime = performance.now() - frameStart;
        this.frameTimes.push(frameTime);
        if (this.frameTimes.length > 30) this.frameTimes.shift();

        const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        const remainingFrames = totalFrames - frame - 1;
        const videoWeight = includeAudio ? 0.95 : 1.0;
        const videoPercent = ((frame + 1) / totalFrames) * 100 * videoWeight;

        onProgress({
          phase: 'video',
          currentFrame: frame + 1,
          totalFrames,
          percent: videoPercent,
          estimatedTimeRemaining: (remainingFrames * avgFrameTime) / 1000,
          currentTime: time,
        });
      }

      // Phase 2: Export audio
      let audioResult: EncodedAudioResult | null = null;
      if (includeAudio && this.audioPipeline) {
        if (this.isCancelled) {
          this.cleanup(originalDimensions);
          return null;
        }

        console.log('[FrameExporter] Starting audio export...');

        audioResult = await this.audioPipeline.exportAudio(startTime, endTime, (audioProgress) => {
          if (this.isCancelled) return;

          onProgress({
            phase: 'audio',
            currentFrame: totalFrames,
            totalFrames,
            percent: 95 + (audioProgress.percent * 0.05),
            estimatedTimeRemaining: 0,
            currentTime: endTime,
            audioPhase: audioProgress.phase,
            audioPercent: audioProgress.percent,
          });
        });

        if (audioResult && audioResult.chunks.length > 0) {
          this.encoder.addAudioChunks(audioResult);
        } else {
          console.log('[FrameExporter] No audio to add');
        }
      }

      const blob = await this.encoder.finish();
      console.log(`[FrameExporter] Export complete: ${(blob.size / 1024 / 1024).toFixed(2)}MB`);
      this.cleanup(originalDimensions);
      return blob;
    } catch (error) {
      console.error('[FrameExporter] Export error:', error);
      this.cleanup(originalDimensions);
      return null;
    }
  }

  cancel(): void {
    this.isCancelled = true;
    this.audioPipeline?.cancel();
    cleanupExportMode(this.clipStates, this.parallelDecoder);
  }

  private cleanup(originalDimensions: { width: number; height: number }): void {
    cleanupExportMode(this.clipStates, this.parallelDecoder);
    cleanupLayerBuilder();
    this.parallelDecoder = null;
    this.useParallelDecode = false;
    engine.cleanupExportCanvas();
    engine.setExporting(false);
    engine.setResolution(originalDimensions.width, originalDimensions.height);
  }

  /**
   * Create FrameContext for a single frame - caches all state lookups.
   * This is the key optimization: one getState() call per frame instead of 5+.
   */
  private createFrameContext(time: number, fps: number, frameTolerance: number): FrameContext {
    const state = useTimelineStore.getState();
    const clipsAtTime = state.getClipsAtTime(time);

    // Build O(1) lookup maps
    const trackMap = new Map(state.tracks.map(t => [t.id, t]));
    const clipsByTrack = new Map(clipsAtTime.map(c => [c.trackId, c]));

    return {
      time,
      fps,
      frameTolerance,
      clipsAtTime,
      trackMap,
      clipsByTrack,
      getInterpolatedTransform: state.getInterpolatedTransform,
      getInterpolatedEffects: state.getInterpolatedEffects,
      getSourceTimeForClip: state.getSourceTimeForClip,
      getInterpolatedSpeed: state.getInterpolatedSpeed,
    };
  }

  // Static helper methods - delegate to codecHelpers
  static isSupported(): boolean {
    return 'VideoEncoder' in window && 'VideoFrame' in window;
  }

  static getPresetResolutions() {
    return RESOLUTION_PRESETS;
  }

  static getPresetFrameRates() {
    return FRAME_RATE_PRESETS;
  }

  static getRecommendedBitrate(width: number, _height: number, _fps: number): number {
    return getRecommendedBitrate(width);
  }

  static getContainerFormats() {
    return CONTAINER_FORMATS;
  }

  static getVideoCodecs(container: 'mp4' | 'webm') {
    return getVideoCodecsForContainer(container);
  }

  static async checkCodecSupport(codec: 'h264' | 'h265' | 'vp9' | 'av1', width: number, height: number): Promise<boolean> {
    return checkCodecSupport(codec, width, height);
  }

  static getBitrateRange() {
    return BITRATE_RANGE;
  }

  static formatBitrate(bitrate: number): string {
    return formatBitrate(bitrate);
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
