// Frame-by-frame exporter for precise video rendering
// Combines VideoEncoder and FrameExporter in one file to avoid import issues

import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4Target } from 'mp4-muxer';
import { Muxer as WebmMuxer, ArrayBufferTarget as WebmTarget } from 'webm-muxer';
import { engine } from './WebGPUEngine';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import type { Layer, TimelineClip, NestedCompositionData } from '../types';
import { AudioExportPipeline, type AudioExportProgress, type EncodedAudioResult, AudioEncoderWrapper, type AudioCodec } from './audio';

// ============ TYPES ============

export type VideoCodec = 'h264' | 'h265' | 'vp9' | 'av1';
export type ContainerFormat = 'mp4' | 'webm';

export interface ExportSettings {
  width: number;
  height: number;
  fps: number;
  codec: VideoCodec;
  container: ContainerFormat;
  bitrate: number;
  startTime: number;
  endTime: number;
  // Audio settings
  includeAudio?: boolean;
  audioSampleRate?: 44100 | 48000;
  audioBitrate?: number;  // 128000 - 320000
  normalizeAudio?: boolean;
}

export interface ExportProgress {
  phase: 'video' | 'audio' | 'muxing';
  currentFrame: number;
  totalFrames: number;
  percent: number;
  estimatedTimeRemaining: number;
  currentTime: number;
  audioPhase?: AudioExportProgress['phase'];
  audioPercent?: number;
}

export interface FullExportSettings extends ExportSettings {
  filename?: string;
}

// ============ VIDEO ENCODER ============

type MuxerType = Mp4Muxer<Mp4Target> | WebmMuxer<WebmTarget>;

class VideoEncoderWrapper {
  private encoder: VideoEncoder | null = null;
  private muxer: MuxerType | null = null;
  private settings: ExportSettings;
  private encodedFrameCount = 0;
  private isClosed = false;
  private hasAudio = false;
  private audioCodec: AudioCodec = 'aac';
  private containerFormat: ContainerFormat = 'mp4';
  private effectiveVideoCodec: VideoCodec = 'h264';

  constructor(settings: ExportSettings) {
    this.settings = settings;
    this.hasAudio = settings.includeAudio ?? false;
    this.containerFormat = settings.container ?? 'mp4';
  }

  async init(): Promise<boolean> {
    if (!('VideoEncoder' in window)) {
      console.error('[VideoEncoder] WebCodecs not supported');
      return false;
    }

    // Determine audio codec based on container
    if (this.hasAudio) {
      if (this.containerFormat === 'webm') {
        // WebM uses Opus
        const opusSupported = await AudioEncoderWrapper.detectSupportedCodec();
        if (opusSupported?.codec === 'opus') {
          this.audioCodec = 'opus';
        } else {
          console.warn('[VideoEncoder] Opus not supported, disabling audio for WebM');
          this.hasAudio = false;
        }
      } else {
        // MP4 uses AAC
        const aacSupported = await AudioEncoderWrapper.detectSupportedCodec();
        if (aacSupported?.codec === 'aac') {
          this.audioCodec = 'aac';
        } else {
          console.warn('[VideoEncoder] AAC not supported, disabling audio for MP4');
          this.hasAudio = false;
        }
      }
    }

    // Determine effective video codec based on container compatibility
    this.effectiveVideoCodec = this.settings.codec;

    // WebM only supports VP9 and AV1
    if (this.containerFormat === 'webm' && (this.settings.codec === 'h264' || this.settings.codec === 'h265')) {
      console.warn(`[VideoEncoder] ${this.settings.codec} not supported in WebM, using VP9`);
      this.effectiveVideoCodec = 'vp9';
    }

    // MP4 supports H264, H265, VP9, AV1
    const codecString = this.getCodecString(this.effectiveVideoCodec);

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

    // Create appropriate muxer based on container format
    const webmVideoCodec = this.effectiveVideoCodec === 'av1' ? 'V_AV1' : 'V_VP9';
    const mp4VideoCodec = this.getMp4MuxerCodec(this.effectiveVideoCodec);

    if (this.containerFormat === 'webm') {
      // WebM muxer (VP9 or AV1 video, Opus audio)
      if (this.hasAudio) {
        this.muxer = new WebmMuxer({
          target: new WebmTarget(),
          video: {
            codec: webmVideoCodec,
            width: this.settings.width,
            height: this.settings.height,
          },
          audio: {
            codec: 'A_OPUS',
            sampleRate: this.settings.audioSampleRate ?? 48000,
            numberOfChannels: 2,
          },
        });
      } else {
        this.muxer = new WebmMuxer({
          target: new WebmTarget(),
          video: {
            codec: webmVideoCodec,
            width: this.settings.width,
            height: this.settings.height,
          },
        });
      }
      console.log(`[VideoEncoder] Using WebM/${this.effectiveVideoCodec.toUpperCase()} with ${this.hasAudio ? 'Opus' : 'no'} audio`);
    } else {
      // MP4 muxer (H264, H265, VP9, AV1 video, AAC audio)
      if (this.hasAudio) {
        this.muxer = new Mp4Muxer({
          target: new Mp4Target(),
          video: {
            codec: mp4VideoCodec,
            width: this.settings.width,
            height: this.settings.height,
          },
          audio: {
            codec: 'aac',
            sampleRate: this.settings.audioSampleRate ?? 48000,
            numberOfChannels: 2,
          },
          fastStart: 'in-memory',
        });
      } else {
        this.muxer = new Mp4Muxer({
          target: new Mp4Target(),
          video: {
            codec: mp4VideoCodec,
            width: this.settings.width,
            height: this.settings.height,
          },
          fastStart: 'in-memory',
        });
      }
      console.log(`[VideoEncoder] Using MP4/${this.effectiveVideoCodec.toUpperCase()} with ${this.hasAudio ? 'AAC' : 'no'} audio`);
    }

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

    console.log(`[VideoEncoder] Initialized: ${this.settings.width}x${this.settings.height} @ ${this.settings.fps}fps (${this.effectiveVideoCodec.toUpperCase()})`);
    return true;
  }

  getContainerFormat(): 'mp4' | 'webm' {
    return this.containerFormat;
  }

  getAudioCodec(): AudioCodec {
    return this.audioCodec;
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

  /**
   * Add encoded audio chunks to the muxer
   */
  addAudioChunks(audioResult: EncodedAudioResult): void {
    if (!this.muxer || !this.hasAudio) {
      console.warn('[VideoEncoder] Cannot add audio: muxer not ready or audio not enabled');
      return;
    }

    console.log(`[VideoEncoder] Adding ${audioResult.chunks.length} audio chunks`);

    for (let i = 0; i < audioResult.chunks.length; i++) {
      const chunk = audioResult.chunks[i];
      const meta = audioResult.metadata[i];
      this.muxer.addAudioChunk(chunk, meta);
    }

    console.log(`[VideoEncoder] Audio chunks added successfully`);
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
    const mimeType = this.containerFormat === 'webm' ? 'video/webm' : 'video/mp4';

    console.log(`[VideoEncoder] Finished: ${this.encodedFrameCount} frames, ${(buffer.byteLength / 1024 / 1024).toFixed(2)}MB (${this.containerFormat.toUpperCase()})`);
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

  // Get WebCodecs codec string for VideoEncoder
  private getCodecString(codec: VideoCodec): string {
    switch (codec) {
      case 'h264':
        return 'avc1.640028'; // High Profile, Level 4.0
      case 'h265':
        return 'hvc1.1.6.L93.B0'; // Main Profile, Level 3.1
      case 'vp9':
        return 'vp09.00.10.08'; // Profile 0, Level 1.0, 8-bit
      case 'av1':
        return 'av01.0.04M.08'; // Main Profile, Level 3.0, 8-bit
      default:
        return 'avc1.640028';
    }
  }

  // Get mp4-muxer codec identifier
  private getMp4MuxerCodec(codec: VideoCodec): 'avc' | 'hevc' | 'vp9' | 'av1' {
    switch (codec) {
      case 'h264':
        return 'avc';
      case 'h265':
        return 'hevc';
      case 'vp9':
        return 'vp9';
      case 'av1':
        return 'av1';
      default:
        return 'avc';
    }
  }
}

// ============ FRAME EXPORTER ============

export class FrameExporter {
  private settings: FullExportSettings;
  private encoder: VideoEncoderWrapper | null = null;
  private audioPipeline: AudioExportPipeline | null = null;
  private isCancelled = false;
  private frameTimes: number[] = [];

  // Sequential export tracking - maps clip ID to last exported source time
  private clipExportState: Map<string, { lastSourceTime: number; isSequential: boolean }> = new Map();
  private exportPrepared = false;

  constructor(settings: FullExportSettings) {
    this.settings = settings;
  }

  async export(onProgress: (progress: ExportProgress) => void): Promise<Blob | null> {
    const { fps, startTime, endTime, width, height, includeAudio } = this.settings;
    const frameDuration = 1 / fps;
    const totalFrames = Math.ceil((endTime - startTime) * fps);

    console.log(`[FrameExporter] Starting export: ${width}x${height} @ ${fps}fps, ${totalFrames} frames, audio: ${includeAudio ? 'yes' : 'no'}`);

    this.encoder = new VideoEncoderWrapper(this.settings);
    const initialized = await this.encoder.init();
    if (!initialized) {
      console.error('[FrameExporter] Failed to initialize encoder');
      return null;
    }

    // Initialize audio pipeline if audio is enabled
    if (includeAudio) {
      this.audioPipeline = new AudioExportPipeline({
        sampleRate: this.settings.audioSampleRate ?? 48000,
        bitrate: this.settings.audioBitrate ?? 256000,
        normalize: this.settings.normalizeAudio ?? false,
      });
    }

    const originalDimensions = engine.getOutputDimensions();
    engine.setResolution(width, height);

    // Enable export mode - pauses all preview rendering to prevent conflicts
    engine.setExporting(true);

    try {
      // Prepare all clips for sequential export (avoids decoder reset on each frame)
      await this.prepareClipsForExport(startTime);

      // Phase 1: Encode video frames
      for (let frame = 0; frame < totalFrames; frame++) {
        if (this.isCancelled) {
          console.log('[FrameExporter] Export cancelled');
          this.encoder.cancel();
          this.audioPipeline?.cancel();
          engine.setExporting(false);
          engine.setResolution(originalDimensions.width, originalDimensions.height);
          return null;
        }

        const frameStart = performance.now();
        const time = startTime + frame * frameDuration;

        if (frame % 10 === 0) {
          console.log(`[FrameExporter] Processing frame ${frame}/${totalFrames} at time ${time.toFixed(3)}s`);
        }

        console.log(`[FrameExporter] Frame ${frame}: seeking...`);
        await this.seekAllClipsToTime(time);
        console.log(`[FrameExporter] Frame ${frame}: seek done`);

        // Wait for all video elements to be ready after seeking
        await this.waitForAllVideosReady(time);

        console.log(`[FrameExporter] Frame ${frame}: building layers...`);
        const layers = this.buildLayersAtTime(time);
        console.log(`[FrameExporter] Frame ${frame}: ${layers.length} layers`);

        if (layers.length === 0) {
          console.warn('[FrameExporter] No layers at time', time);
        }

        console.log(`[FrameExporter] Frame ${frame}: rendering...`);
        engine.render(layers);

        console.log(`[FrameExporter] Frame ${frame}: reading pixels...`);
        const pixels = await engine.readPixels();
        if (!pixels) {
          console.error('[FrameExporter] Failed to read pixels at frame', frame);
          continue;
        }
        console.log(`[FrameExporter] Frame ${frame}: got ${pixels.length} bytes`);

        console.log(`[FrameExporter] Frame ${frame}: encoding...`);
        await this.encoder.encodeFrame(pixels, frame);
        console.log(`[FrameExporter] Frame ${frame}: done`);

        const frameTime = performance.now() - frameStart;
        this.frameTimes.push(frameTime);
        if (this.frameTimes.length > 30) this.frameTimes.shift();

        const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        const remainingFrames = totalFrames - frame - 1;
        // If audio is included, video is ~95% of total work (audio encoding is fast)
        const videoWeight = includeAudio ? 0.95 : 1.0;
        const videoPercent = ((frame + 1) / totalFrames) * 100 * videoWeight;
        const estimatedTimeRemaining = (remainingFrames * avgFrameTime) / 1000;

        onProgress({
          phase: 'video',
          currentFrame: frame + 1,
          totalFrames,
          percent: videoPercent,
          estimatedTimeRemaining,
          currentTime: time,
        });
      }

      // Phase 2: Export audio if enabled
      let audioResult: EncodedAudioResult | null = null;
      if (includeAudio && this.audioPipeline) {
        if (this.isCancelled) {
          engine.setResolution(originalDimensions.width, originalDimensions.height);
          return null;
        }

        console.log('[FrameExporter] Starting audio export...');

        audioResult = await this.audioPipeline.exportAudio(
          startTime,
          endTime,
          (audioProgress) => {
            if (this.isCancelled) return;

            // Audio is ~5% of total work (95-100%)
            const audioPercent = 95 + (audioProgress.percent * 0.05);

            onProgress({
              phase: 'audio',
              currentFrame: totalFrames,
              totalFrames,
              percent: audioPercent,
              estimatedTimeRemaining: 0, // Hard to estimate for audio
              currentTime: endTime,
              audioPhase: audioProgress.phase,
              audioPercent: audioProgress.percent,
            });
          }
        );

        // Add audio chunks to muxer if we have audio
        if (audioResult && audioResult.chunks.length > 0) {
          this.encoder.addAudioChunks(audioResult);
        } else {
          console.log('[FrameExporter] No audio to add (no audio clips in range or export failed)');
        }
      }

      const blob = await this.encoder.finish();
      console.log(`[FrameExporter] Export complete: ${(blob.size / 1024 / 1024).toFixed(2)}MB`);
      this.cleanupExportMode();
      engine.setExporting(false);
      engine.setResolution(originalDimensions.width, originalDimensions.height);
      return blob;
    } catch (error) {
      console.error('[FrameExporter] Export error:', error);
      this.cleanupExportMode();
      engine.setExporting(false);
      engine.setResolution(originalDimensions.width, originalDimensions.height);
      return null;
    }
  }

  cancel(): void {
    this.isCancelled = true;
    this.audioPipeline?.cancel();
    this.cleanupExportMode();
  }

  /**
   * Prepare all video clips for sequential export.
   * This initializes WebCodecsPlayers in export mode so they can decode
   * frames sequentially without resetting the decoder each time.
   */
  private async prepareClipsForExport(startTime: number): Promise<void> {
    const clips = useTimelineStore.getState().clips;
    const tracks = useTimelineStore.getState().tracks;

    console.log('[FrameExporter] Preparing clips for sequential export...');

    for (const clip of clips) {
      const track = tracks.find(t => t.id === clip.trackId);
      if (!track?.visible) continue;

      // Handle nested compositions
      if (clip.isComposition && clip.nestedClips) {
        for (const nestedClip of clip.nestedClips) {
          if (nestedClip.source?.webCodecsPlayer) {
            // Calculate start time for nested clip
            const clipLocalTime = Math.max(0, startTime - clip.startTime);
            const nestedTime = clipLocalTime + (clip.inPoint || 0);
            if (nestedTime >= nestedClip.startTime && nestedTime < nestedClip.startTime + nestedClip.duration) {
              const nestedLocalTime = nestedTime - nestedClip.startTime;
              const nestedClipTime = nestedClip.reversed
                ? nestedClip.outPoint - nestedLocalTime
                : nestedLocalTime + nestedClip.inPoint;

              await nestedClip.source.webCodecsPlayer.prepareForSequentialExport(nestedClipTime);
              this.clipExportState.set(nestedClip.id, {
                lastSourceTime: nestedClipTime,
                isSequential: !nestedClip.reversed,
              });
            }
          }
        }
        continue;
      }

      // Handle regular video clips
      if (clip.source?.type === 'video' && clip.source.webCodecsPlayer) {
        // Calculate start time for this clip
        const clipLocalTime = Math.max(0, startTime - clip.startTime);

        // Check if clip uses speed keyframes or is reversed
        let clipTime: number;
        let isSequential = !clip.reversed;

        try {
          const sourceTime = useTimelineStore.getState().getSourceTimeForClip(clip.id, clipLocalTime);
          const initialSpeed = useTimelineStore.getState().getInterpolatedSpeed(clip.id, 0);
          const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
          clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

          // Check if clip has non-standard speed (speed keyframes make it non-sequential)
          const hasSpeedKeyframes = useTimelineStore.getState().hasKeyframes(clip.id, 'speed');
          if (hasSpeedKeyframes || initialSpeed !== 1) {
            isSequential = false;
          }
        } catch {
          clipTime = clip.reversed
            ? clip.outPoint - clipLocalTime
            : clipLocalTime + clip.inPoint;
          clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, clipTime));
        }

        await clip.source.webCodecsPlayer.prepareForSequentialExport(clipTime);
        this.clipExportState.set(clip.id, {
          lastSourceTime: clipTime,
          isSequential,
        });

        console.log(`[FrameExporter] Prepared clip "${clip.name}" for export (sequential: ${isSequential})`);
      }
    }

    this.exportPrepared = true;
    console.log(`[FrameExporter] ${this.clipExportState.size} clips prepared for sequential export`);
  }

  /**
   * Cleanup export mode for all clips
   */
  private cleanupExportMode(): void {
    if (!this.exportPrepared) return;

    const clips = useTimelineStore.getState().clips;

    for (const clip of clips) {
      if (clip.isComposition && clip.nestedClips) {
        for (const nestedClip of clip.nestedClips) {
          if (nestedClip.source?.webCodecsPlayer?.isExportMode()) {
            nestedClip.source.webCodecsPlayer.endSequentialExport();
          }
        }
        continue;
      }

      if (clip.source?.webCodecsPlayer?.isExportMode()) {
        clip.source.webCodecsPlayer.endSequentialExport();
      }
    }

    this.clipExportState.clear();
    this.exportPrepared = false;
    console.log('[FrameExporter] Export mode cleanup complete');
  }

  private async seekAllClipsToTime(time: number): Promise<void> {
    const clips = useTimelineStore.getState().getClipsAtTime(time);
    const tracks = useTimelineStore.getState().tracks;
    const seekPromises: Promise<void>[] = [];
    const seekDescriptions: string[] = [];

    // Debug: log clips found at this time (only first frame)
    if (time < 0.1) {
      console.log(`[FrameExporter] Clips at time ${time.toFixed(3)}:`, clips.map(c => ({
        name: c.name,
        isComposition: c.isComposition,
        hasVideoElement: !!c.source?.videoElement,
        sourceType: c.source?.type,
        trackId: c.trackId,
      })));
    }

    for (const clip of clips) {
      const track = tracks.find(t => t.id === clip.trackId);
      if (!track?.visible) continue;

      // Handle nested composition clips
      if (clip.isComposition && clip.nestedClips && clip.nestedTracks) {
        const clipLocalTime = time - clip.startTime;
        const nestedTime = clipLocalTime + (clip.inPoint || 0);

        for (const nestedClip of clip.nestedClips) {
          // Check if nested clip is active at this time
          if (nestedTime >= nestedClip.startTime && nestedTime < nestedClip.startTime + nestedClip.duration) {
            if (nestedClip.source?.videoElement) {
              const nestedLocalTime = nestedTime - nestedClip.startTime;
              const nestedClipTime = nestedClip.reversed
                ? nestedClip.outPoint - nestedLocalTime
                : nestedLocalTime + nestedClip.inPoint;

              if (nestedClip.source.webCodecsPlayer) {
                // Use sequential export if prepared
                const exportState = this.clipExportState.get(nestedClip.id);
                if (exportState && nestedClip.source.webCodecsPlayer.isExportMode()) {
                  seekPromises.push(this.seekClipSequential(
                    nestedClip.source.webCodecsPlayer,
                    nestedClipTime,
                    exportState,
                    nestedClip.id
                  ));
                } else {
                  seekPromises.push(nestedClip.source.webCodecsPlayer.seekAsync(nestedClipTime));
                }
              } else {
                seekPromises.push(this.seekVideo(nestedClip.source.videoElement, nestedClipTime));
              }
            }
          }
        }
        continue;
      }

      // Handle regular video clips
      if (clip.source?.type === 'video' && clip.source.videoElement) {
        const clipLocalTime = time - clip.startTime;

        // Simple calculation matching nested clip approach
        // For clips with speed keyframes, use the store's calculation
        let clipTime: number;
        try {
          const sourceTime = useTimelineStore.getState().getSourceTimeForClip(clip.id, clipLocalTime);
          const initialSpeed = useTimelineStore.getState().getInterpolatedSpeed(clip.id, 0);
          const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
          clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
        } catch (e) {
          // Fallback to simple calculation if speed functions fail
          clipTime = clip.reversed
            ? clip.outPoint - clipLocalTime
            : clipLocalTime + clip.inPoint;
          clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, clipTime));
        }

        if (clip.source.webCodecsPlayer) {
          // Use sequential export if prepared
          const exportState = this.clipExportState.get(clip.id);
          if (exportState && clip.source.webCodecsPlayer.isExportMode()) {
            seekDescriptions.push(`Sequential:${clip.name}`);
            seekPromises.push(this.seekClipSequential(
              clip.source.webCodecsPlayer,
              clipTime,
              exportState,
              clip.id
            ));
          } else {
            seekDescriptions.push(`WebCodecs:${clip.name}`);
            seekPromises.push(clip.source.webCodecsPlayer.seekAsync(clipTime));
          }
        } else {
          seekDescriptions.push(`HTMLVideo:${clip.name}`);
          seekPromises.push(this.seekVideo(clip.source.videoElement, clipTime));
        }
      }
    }

    if (seekPromises.length > 0) {
      console.log(`[FrameExporter] Seeking ${seekPromises.length} clips:`, seekDescriptions);
      await Promise.all(seekPromises);
      console.log(`[FrameExporter] All seeks completed`);
    }
  }

  /**
   * Seek a clip using sequential export optimization.
   * If the clip can be decoded sequentially (next frame), use decodeNextFrameForExport().
   * Otherwise, fall back to seekDuringExport() which still avoids full reset when possible.
   */
  private async seekClipSequential(
    player: import('./WebCodecsPlayer').WebCodecsPlayer,
    targetTime: number,
    exportState: { lastSourceTime: number; isSequential: boolean },
    clipId: string
  ): Promise<void> {
    // Calculate frame duration (assume ~30fps for tolerance calculation)
    const frameDuration = 1 / 30;
    const timeDelta = targetTime - exportState.lastSourceTime;

    // Check if we can use sequential decode:
    // - Clip must be marked as sequential (not reversed, no speed keyframes)
    // - Time must be moving forward by approximately one frame
    const isNextFrame = exportState.isSequential &&
      timeDelta > 0 &&
      timeDelta < frameDuration * 2; // Allow some tolerance

    if (isNextFrame) {
      // Sequential decode - just get the next frame
      await player.decodeNextFrameForExport();
    } else {
      // Need to jump - use seekDuringExport which is still optimized
      await player.seekDuringExport(targetTime);
    }

    // Update last source time
    exportState.lastSourceTime = targetTime;
    this.clipExportState.set(clipId, exportState);
  }

  private seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
    return new Promise((resolve) => {
      const targetTime = Math.max(0, Math.min(time, video.duration || 0));

      const timeout = setTimeout(() => {
        console.warn('[FrameExporter] Seek timeout at', targetTime);
        resolve();
      }, 2000);

      // Use requestVideoFrameCallback if available - guarantees frame is decoded
      const waitForFrame = () => {
        if ('requestVideoFrameCallback' in video) {
          (video as any).requestVideoFrameCallback(() => {
            clearTimeout(timeout);
            resolve();
          });
        } else {
          // Fallback: wait for readyState and extra frames
          let retries = 0;
          const maxRetries = 30;
          const vid = video as HTMLVideoElement; // Type assertion for closure

          const waitForReady = () => {
            retries++;
            if (!vid.seeking && vid.readyState >= 3) {
              clearTimeout(timeout);
              requestAnimationFrame(() => {
                requestAnimationFrame(() => resolve());
              });
            } else if (retries < maxRetries) {
              requestAnimationFrame(waitForReady);
            } else {
              clearTimeout(timeout);
              resolve();
            }
          };
          waitForReady();
        }
      };

      // If already at correct time with frame ready, still wait for one frame callback
      // to ensure the GPU has the latest frame
      if (Math.abs(video.currentTime - targetTime) < 0.01 && !video.seeking && video.readyState >= 3) {
        waitForFrame();
        return;
      }

      const onSeeked = () => {
        video.removeEventListener('seeked', onSeeked);
        waitForFrame();
      };

      video.addEventListener('seeked', onSeeked);
      video.currentTime = targetTime;
    });
  }

  /**
   * Wait for all video clips at a given time to have their frames ready
   */
  private async waitForAllVideosReady(time: number): Promise<void> {
    const clips = useTimelineStore.getState().getClipsAtTime(time);
    const tracks = useTimelineStore.getState().tracks;

    const videoClips = clips.filter(clip => {
      const track = tracks.find(t => t.id === clip.trackId);
      return track?.visible && clip.source?.type === 'video' && clip.source.videoElement;
    });

    if (videoClips.length === 0) return;

    // Wait for all videos to be ready (with timeout)
    const maxWaitTime = 2000;
    const startTime = performance.now();

    while (performance.now() - startTime < maxWaitTime) {
      let allReady = true;

      for (const clip of videoClips) {
        const video = clip.source!.videoElement!;
        const webCodecsPlayer = clip.source!.webCodecsPlayer;

        const videoReady = video.readyState >= 2 && !video.seeking;
        const hasWebCodecsFrame = webCodecsPlayer?.hasFrame() ?? false;

        if (!videoReady && !hasWebCodecsFrame) {
          allReady = false;
          break;
        }
      }

      if (allReady) {
        // Give one more frame for GPU texture upload
        await new Promise(r => requestAnimationFrame(r));
        return;
      }

      // Wait one frame and check again
      await new Promise(r => requestAnimationFrame(r));
    }

    console.warn('[FrameExporter] Timeout waiting for videos to be ready at time', time);
  }

  private buildLayersAtTime(time: number): Layer[] {
    const timelineState = useTimelineStore.getState();
    const { clips, tracks, getInterpolatedTransform, getInterpolatedEffects } = timelineState;
    const layers: Layer[] = [];

    const videoTracks = tracks.filter(t => t.type === 'video');
    const anyVideoSolo = videoTracks.some(t => t.solo);

    const isTrackVisible = (track: typeof videoTracks[0]) => {
      if (!track.visible) return false;
      if (anyVideoSolo) return track.solo;
      return true;
    };

    // Get clips at current time
    const clipsAtTime = clips.filter(
      c => time >= c.startTime && time < c.startTime + c.duration
    );

    // Build layers in track order (bottom to top)
    for (let trackIndex = 0; trackIndex < videoTracks.length; trackIndex++) {
      const track = videoTracks[trackIndex];
      if (!isTrackVisible(track)) continue;

      const clip = clipsAtTime.find(c => c.trackId === track.id);
      if (!clip) continue;

      const clipLocalTime = time - clip.startTime;

      // Get transform safely with defaults
      let transform;
      try {
        transform = getInterpolatedTransform(clip.id, clipLocalTime);
      } catch (e) {
        console.warn('[FrameExporter] Transform interpolation failed for clip', clip.id, e);
        transform = {
          position: { x: 0, y: 0, z: 0 },
          scale: { x: 1, y: 1 },
          rotation: { x: 0, y: 0, z: 0 },
          opacity: 1,
          blendMode: 'normal' as const,
        };
      }

      // Get effects safely
      let effects: typeof layers[0]['effects'] = [];
      try {
        effects = getInterpolatedEffects(clip.id, clipLocalTime);
      } catch (e) {
        console.warn('[FrameExporter] Effects interpolation failed for clip', clip.id, e);
      }

      const baseLayerProps = {
        id: `export_layer_${trackIndex}`,
        name: clip.name,
        visible: true,
        opacity: transform.opacity ?? 1,
        blendMode: transform.blendMode || 'normal',
        effects,
        position: {
          x: transform.position?.x ?? 0,
          y: transform.position?.y ?? 0,
          z: transform.position?.z ?? 0,
        },
        scale: {
          x: transform.scale?.x ?? 1,
          y: transform.scale?.y ?? 1,
        },
        rotation: {
          x: ((transform.rotation?.x ?? 0) * Math.PI) / 180,
          y: ((transform.rotation?.y ?? 0) * Math.PI) / 180,
          z: ((transform.rotation?.z ?? 0) * Math.PI) / 180,
        },
      };

      // Handle nested compositions
      if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        const nestedLayers = this.buildNestedLayersForExport(clip, clipLocalTime + (clip.inPoint || 0));

        if (nestedLayers.length > 0) {
          const composition = useMediaStore.getState().compositions.find(c => c.id === clip.compositionId);
          const compWidth = composition?.width || 1920;
          const compHeight = composition?.height || 1080;

          const nestedCompData: NestedCompositionData = {
            compositionId: clip.compositionId || clip.id,
            layers: nestedLayers,
            width: compWidth,
            height: compHeight,
          };

          layers.push({
            ...baseLayerProps,
            source: {
              type: 'video',
              nestedComposition: nestedCompData,
            },
          });
        }
        continue;
      }

      // Handle video clips
      if (clip.source?.type === 'video' && clip.source.videoElement) {
        const video = clip.source.videoElement;
        const webCodecsPlayer = clip.source.webCodecsPlayer;

        // Check if video is ready:
        // - HTMLVideoElement has readyState >= 2 (HAVE_CURRENT_DATA), OR
        // - WebCodecsPlayer has a decoded frame available
        const videoReady = video.readyState >= 2;
        const hasWebCodecsFrame = webCodecsPlayer?.hasFrame() ?? false;

        if (videoReady || hasWebCodecsFrame) {
          layers.push({
            ...baseLayerProps,
            source: {
              type: 'video',
              videoElement: video,
              webCodecsPlayer: webCodecsPlayer,
            },
          });
        } else {
          console.warn('[FrameExporter] Video not ready for clip', clip.id, 'readyState:', video.readyState, 'hasWebCodecsFrame:', hasWebCodecsFrame);
        }
      }
      // Handle image clips
      else if (clip.source?.type === 'image' && clip.source.imageElement) {
        layers.push({
          ...baseLayerProps,
          source: { type: 'image', imageElement: clip.source.imageElement },
        });
      }
      // Handle text clips
      else if (clip.source?.type === 'text' && clip.source.textCanvas) {
        layers.push({
          ...baseLayerProps,
          source: { type: 'text', textCanvas: clip.source.textCanvas },
        });
      }
    }

    return layers;
  }

  /**
   * Build layers for a nested composition at export time
   * Note: Video seeking is handled by seekAllClipsToTime, not here
   */
  private buildNestedLayersForExport(clip: TimelineClip, nestedTime: number): Layer[] {
    if (!clip.nestedClips || !clip.nestedTracks) return [];

    const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
    const layers: Layer[] = [];

    // Process tracks in same order as main timeline (0 to N, bottom to top)
    for (let i = 0; i < nestedVideoTracks.length; i++) {
      const nestedTrack = nestedVideoTracks[i];
      const nestedClip = clip.nestedClips.find(
        nc =>
          nc.trackId === nestedTrack.id &&
          nestedTime >= nc.startTime &&
          nestedTime < nc.startTime + nc.duration
      );

      if (!nestedClip) continue;

      const transform = nestedClip.transform || {
        position: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1 },
        rotation: { x: 0, y: 0, z: 0 },
        opacity: 1,
        blendMode: 'normal' as const,
      };

      const baseLayer = {
        id: `nested-export-${nestedClip.id}`,
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

      // Don't seek here - seekAllClipsToTime handles all seeking
      // Just build the layer with the video element
      if (nestedClip.source?.videoElement) {
        layers.push({
          ...baseLayer,
          source: {
            type: 'video',
            videoElement: nestedClip.source.videoElement,
            webCodecsPlayer: nestedClip.source.webCodecsPlayer,
          },
        } as Layer);
      } else if (nestedClip.source?.imageElement) {
        layers.push({
          ...baseLayer,
          source: {
            type: 'image',
            imageElement: nestedClip.source.imageElement,
          },
        } as Layer);
      } else if (nestedClip.source?.textCanvas) {
        layers.push({
          ...baseLayer,
          source: {
            type: 'text',
            textCanvas: nestedClip.source.textCanvas,
          },
        } as Layer);
      }
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

  static getContainerFormats(): Array<{ id: ContainerFormat; label: string; extension: string }> {
    return [
      { id: 'mp4', label: 'MP4', extension: '.mp4' },
      { id: 'webm', label: 'WebM', extension: '.webm' },
    ];
  }

  static getVideoCodecs(container: ContainerFormat): Array<{ id: VideoCodec; label: string; description: string }> {
    if (container === 'webm') {
      return [
        { id: 'vp9', label: 'VP9', description: 'Good quality, widely supported' },
        { id: 'av1', label: 'AV1', description: 'Best quality, slow encoding' },
      ];
    }
    // MP4 container
    return [
      { id: 'h264', label: 'H.264 (AVC)', description: 'Most compatible, fast encoding' },
      { id: 'h265', label: 'H.265 (HEVC)', description: 'Better compression, limited support' },
      { id: 'vp9', label: 'VP9', description: 'Good quality, open codec' },
      { id: 'av1', label: 'AV1', description: 'Best quality, slow encoding' },
    ];
  }

  static async checkCodecSupport(codec: VideoCodec, width: number, height: number): Promise<boolean> {
    if (!('VideoEncoder' in window)) return false;

    const codecStrings: Record<VideoCodec, string> = {
      h264: 'avc1.640028',
      h265: 'hvc1.1.6.L93.B0',
      vp9: 'vp09.00.10.08',
      av1: 'av01.0.04M.08',
    };

    try {
      const support = await VideoEncoder.isConfigSupported({
        codec: codecStrings[codec],
        width,
        height,
        bitrate: 10_000_000,
        framerate: 30,
      });
      return support.supported ?? false;
    } catch {
      return false;
    }
  }

  static getBitrateRange(): { min: number; max: number; step: number } {
    return { min: 1_000_000, max: 100_000_000, step: 500_000 };
  }

  static formatBitrate(bitrate: number): string {
    if (bitrate >= 1_000_000) {
      return `${(bitrate / 1_000_000).toFixed(1)} Mbps`;
    }
    return `${(bitrate / 1_000).toFixed(0)} Kbps`;
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
