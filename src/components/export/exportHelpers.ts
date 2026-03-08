// Runtime-backed helpers for the FFmpeg export panel.

import { Logger } from '../../services/logger';
import { useTimelineStore } from '../../stores/timeline';
import type { Layer } from '../../types';
import { prepareClipsForExport, cleanupExportMode } from '../../engine/export/ClipPreparation';
import {
  buildLayersAtTime as buildExportLayersAtTime,
  cleanupLayerBuilder,
  initializeLayerBuilder,
} from '../../engine/export/ExportLayerBuilder';
import { seekAllClipsToTime, waitForAllVideosReady } from '../../engine/export/VideoSeeker';
import { getFrameTolerance } from '../../engine/export/types';
import type { ExportClipState, ExportSettings, FrameContext } from '../../engine/export/types';
import type { ParallelDecodeManager } from '../../engine/ParallelDecodeManager';

const log = Logger.create('ExportHelpers');

interface FFmpegFrameRendererOptions {
  width: number;
  height: number;
  fps: number;
  startTime: number;
  endTime: number;
}

const PRECISE_FFMPEG_EXPORT_DEFAULTS: Pick<ExportSettings, 'codec' | 'container' | 'bitrate'> = {
  codec: 'h264',
  container: 'mp4',
  bitrate: 10_000_000,
};

export class FFmpegFrameRenderer {
  private readonly options: FFmpegFrameRendererOptions;
  private clipStates = new Map<string, ExportClipState>();
  private parallelDecoder: ParallelDecodeManager | null = null;
  private useParallelDecode = false;
  private readonly frameTolerance: number;
  private initialized = false;
  private cancelled = false;
  private cleanedUp = false;

  constructor(options: FFmpegFrameRendererOptions) {
    this.options = options;
    this.frameTolerance = getFrameTolerance(options.fps);
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    const preparation = await prepareClipsForExport(
      {
        ...PRECISE_FFMPEG_EXPORT_DEFAULTS,
        width: this.options.width,
        height: this.options.height,
        fps: this.options.fps,
        startTime: this.options.startTime,
        endTime: this.options.endTime,
      },
      'precise'
    );

    this.clipStates = preparation.clipStates;
    this.parallelDecoder = preparation.parallelDecoder;
    this.useParallelDecode = preparation.useParallelDecode;

    const { tracks } = useTimelineStore.getState();
    initializeLayerBuilder(tracks);

    this.initialized = true;
    log.info('Initialized FFmpeg frame renderer with runtime-backed export sessions');
  }

  cancel(): void {
    this.cancelled = true;
  }

  isCancelled(): boolean {
    return this.cancelled;
  }

  async buildLayersAtTime(time: number): Promise<Layer[]> {
    this.ensureReady();
    this.throwIfCancelled();

    const ctx = this.createFrameContext(time);
    await seekAllClipsToTime(
      ctx,
      this.clipStates,
      this.parallelDecoder,
      this.useParallelDecode
    );

    this.throwIfCancelled();

    await waitForAllVideosReady(
      ctx,
      this.clipStates,
      this.parallelDecoder,
      this.useParallelDecode
    );

    this.throwIfCancelled();

    return buildExportLayersAtTime(
      ctx,
      this.clipStates,
      this.parallelDecoder,
      this.useParallelDecode
    );
  }

  cleanup(): void {
    if (this.cleanedUp) {
      return;
    }

    cleanupExportMode(this.clipStates, this.parallelDecoder);
    cleanupLayerBuilder();

    this.parallelDecoder = null;
    this.useParallelDecode = false;
    this.initialized = false;
    this.cleanedUp = true;

    log.info('Cleaned up FFmpeg frame renderer');
  }

  private createFrameContext(time: number): FrameContext {
    const state = useTimelineStore.getState();
    const clipsAtTime = state.getClipsAtTime(time);

    return {
      time,
      fps: this.options.fps,
      frameTolerance: this.frameTolerance,
      clipsAtTime,
      trackMap: new Map(state.tracks.map(track => [track.id, track])),
      clipsByTrack: new Map(clipsAtTime.map(clip => [clip.trackId, clip])),
      getInterpolatedTransform: state.getInterpolatedTransform,
      getInterpolatedEffects: state.getInterpolatedEffects,
      getSourceTimeForClip: state.getSourceTimeForClip,
      getInterpolatedSpeed: state.getInterpolatedSpeed,
    };
  }

  private ensureReady(): void {
    if (!this.initialized || this.cleanedUp) {
      throw new Error('FFmpeg frame renderer is not initialized');
    }
  }

  private throwIfCancelled(): void {
    if (this.cancelled) {
      throw new Error('FFmpeg frame rendering cancelled');
    }
  }
}
