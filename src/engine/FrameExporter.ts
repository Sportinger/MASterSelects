// Frame-by-frame exporter for precise video rendering
// Combines VideoEncoder and FrameExporter in one file to avoid import issues

import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4Target } from 'mp4-muxer';
import { Muxer as WebmMuxer, ArrayBufferTarget as WebmTarget } from 'webm-muxer';
import { engine } from './WebGPUEngine';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import type { Layer, TimelineClip, NestedCompositionData } from '../types';
import { AudioExportPipeline, type AudioExportProgress, type EncodedAudioResult, AudioEncoderWrapper, type AudioCodec } from './audio';
import { fileSystemService } from '../services/fileSystemService';
import { ParallelDecodeManager } from './ParallelDecodeManager';

// ============ TYPES ============

export type VideoCodec = 'h264' | 'h265' | 'vp9' | 'av1';
export type ContainerFormat = 'mp4' | 'webm';

export type ExportMode = 'fast' | 'precise';

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
  // Export mode
  exportMode?: ExportMode;  // 'fast' = WebCodecs sequential, 'precise' = HTMLVideoElement (default: fast)
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
        const opusSupported = await AudioEncoderWrapper.isOpusSupported();
        if (opusSupported) {
          this.audioCodec = 'opus';
          console.log('[VideoEncoder] Using Opus audio for WebM');
        } else {
          console.warn('[VideoEncoder] Opus not supported, disabling audio for WebM');
          this.hasAudio = false;
        }
      } else {
        // MP4 prefers AAC, but can use Opus as fallback (some browsers support Opus in MP4)
        const aacSupported = await AudioEncoderWrapper.isAACSupported();
        if (aacSupported) {
          this.audioCodec = 'aac';
          console.log('[VideoEncoder] Using AAC audio for MP4');
        } else {
          // Try Opus as fallback for MP4 (not ideal but better than no audio)
          const opusSupported = await AudioEncoderWrapper.isOpusSupported();
          if (opusSupported) {
            this.audioCodec = 'opus';
            console.log('[VideoEncoder] AAC not supported, using Opus audio for MP4 (fallback)');
          } else {
            console.warn('[VideoEncoder] No audio codec supported, disabling audio');
            this.hasAudio = false;
          }
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
      // MP4 muxer (H264, H265, VP9, AV1 video, AAC/Opus audio)
      if (this.hasAudio) {
        this.muxer = new Mp4Muxer({
          target: new Mp4Target(),
          video: {
            codec: mp4VideoCodec,
            width: this.settings.width,
            height: this.settings.height,
          },
          audio: {
            codec: this.audioCodec, // Use determined codec (aac or opus)
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
      console.log(`[VideoEncoder] Using MP4/${this.effectiveVideoCodec.toUpperCase()} with ${this.hasAudio ? this.audioCodec.toUpperCase() : 'no'} audio`);
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
        return 'avc1.4d0028'; // Main Profile, Level 4.0 (better VLC compatibility)
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

// Track which clips are using WebCodecs sequential decoding
interface ExportClipState {
  clipId: string;
  webCodecsPlayer: any; // WebCodecsPlayer
  lastSampleIndex: number;
  isSequential: boolean; // true if using sequential decoding
}

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
    this.exportMode = settings.exportMode ?? 'fast'; // Default to fast mode
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

        if (frame % 30 === 0) {
          console.log(`[FrameExporter] Processing frame ${frame}/${totalFrames} at time ${time.toFixed(3)}s`);
        }

        await this.seekAllClipsToTime(time);
        await this.waitForAllVideosReady(time);

        const layers = this.buildLayersAtTime(time);

        if (layers.length === 0 && frame === 0) {
          console.warn('[FrameExporter] No layers at time', time);
        }

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
   * Prepare all video clips for export based on export mode.
   * FAST mode: WebCodecs with MP4Box parsing - sequential decoding, very fast
   * PRECISE mode: HTMLVideoElement seeking - frame-accurate but slower
   * NO FALLBACKS - if the chosen mode fails, export fails with error
   */
  private async prepareClipsForExport(startTime: number): Promise<void> {
    const { clips, tracks } = useTimelineStore.getState();
    const mediaFiles = useMediaStore.getState().files;
    const endTime = this.settings.endTime;

    // Find all video clips that will be in the export range
    const videoClips = clips.filter(clip => {
      const track = tracks.find(t => t.id === clip.trackId);
      if (!track?.visible || track.type !== 'video') return false;
      const clipEnd = clip.startTime + clip.duration;
      return clip.startTime < endTime && clipEnd > startTime;
    });

    console.log(`[FrameExporter] Preparing ${videoClips.length} video clips for ${this.exportMode.toUpperCase()} export...`);

    if (this.exportMode === 'precise') {
      // PRECISE MODE: Use HTMLVideoElement seeking (original method)
      for (const clip of videoClips) {
        if (clip.source?.type !== 'video') continue;
        this.clipStates.set(clip.id, {
          clipId: clip.id,
          webCodecsPlayer: null,
          lastSampleIndex: 0,
          isSequential: false,
        });
        console.log(`[FrameExporter] Clip ${clip.name}: PRECISE mode (HTMLVideoElement seeking)`);
      }
      console.log(`[FrameExporter] All ${videoClips.length} clips using PRECISE HTMLVideoElement seeking`);
      return;
    }

    // FAST MODE: Try WebCodecs with MP4Box parsing, fallback to PRECISE on codec errors
    try {
      await this.initializeFastMode(videoClips, mediaFiles, startTime);
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      // Check if this is a codec/parsing error that can be handled by PRECISE mode
      if (error.includes('not supported') || error.includes('FAST export failed')) {
        console.warn(`[FrameExporter] FAST mode failed, auto-fallback to PRECISE: ${error}`);
        this.exportMode = 'precise';
        // Clean up any partially initialized states
        this.clipStates.clear();
        // Run PRECISE mode for all video clips
        for (const clip of videoClips) {
          if (clip.source?.type !== 'video') continue;
          this.clipStates.set(clip.id, {
            clipId: clip.id,
            webCodecsPlayer: null,
            lastSampleIndex: 0,
            isSequential: false,
          });
          console.log(`[FrameExporter] Clip ${clip.name}: PRECISE mode (HTMLVideoElement) [auto-fallback]`);
        }
        console.log(`[FrameExporter] All ${videoClips.length} clips using PRECISE mode (auto-fallback from FAST)`);
        return;
      }
      // Re-throw other errors
      throw e;
    }
  }

  /**
   * Initialize FAST mode with WebCodecs - extracted for try-catch wrapper
   */
  private async initializeFastMode(
    videoClips: TimelineClip[],
    mediaFiles: any[],
    startTime: number
  ): Promise<void> {
    const { WebCodecsPlayer } = await import('./WebCodecsPlayer');

    // Separate composition clips from regular video clips
    const regularVideoClips: typeof videoClips = [];
    const compositionClips: typeof videoClips = [];
    const nestedVideoClips: Array<{
      clip: TimelineClip;
      parentClip: TimelineClip;
    }> = [];

    for (const clip of videoClips) {
      if (clip.source?.type !== 'video') continue;

      if (clip.isComposition) {
        compositionClips.push(clip);
        this.clipStates.set(clip.id, {
          clipId: clip.id,
          webCodecsPlayer: null,
          lastSampleIndex: 0,
          isSequential: false,
        });
        console.log(`[FrameExporter] Clip ${clip.name}: Composition with nested clips`);

        // Collect nested video clips from this composition
        if (clip.nestedClips) {
          for (const nestedClip of clip.nestedClips) {
            if (nestedClip.source?.type === 'video' && nestedClip.source.videoElement) {
              nestedVideoClips.push({ clip: nestedClip, parentClip: clip });
            }
          }
        }
      } else {
        regularVideoClips.push(clip);
      }
    }

    // Combine regular clips and nested clips for parallel decoding count
    const totalVideoClips = regularVideoClips.length + nestedVideoClips.length;

    // Use parallel decoding if we have 2+ total video clips (including nested)
    if (totalVideoClips >= 2) {
      console.log(`[FrameExporter] Using PARALLEL decoding for ${regularVideoClips.length} regular + ${nestedVideoClips.length} nested = ${totalVideoClips} video clips`);
      await this.initializeParallelDecoding(regularVideoClips, mediaFiles, startTime, nestedVideoClips);
      return;
    }

    // Single clip: use original sequential approach
    for (const clip of regularVideoClips) {
      const mediaFileId = clip.source!.mediaFileId;
      const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null;

      // Get the file data - try multiple sources
      const fileData = await this.loadClipFileData(clip, mediaFile);

      if (!fileData) {
        throw new Error(`FAST export failed: Could not load file data for clip "${clip.name}". Try PRECISE mode instead.`);
      }

      // Detect file format from magic bytes
      const header = new Uint8Array(fileData.slice(0, 12));
      const isMOV = header[4] === 0x66 && header[5] === 0x74 && header[6] === 0x79 && header[7] === 0x70 &&
                    (header[8] === 0x71 && header[9] === 0x74); // 'ftyp' + 'qt' = MOV
      const fileType = isMOV ? 'MOV' : 'MP4';

      console.log(`[FrameExporter] Loaded ${clip.name} (${(fileData.byteLength / 1024 / 1024).toFixed(1)}MB, ${fileType})`);

      // Create dedicated WebCodecs player for export
      const exportPlayer = new WebCodecsPlayer({
        useSimpleMode: false,
        loop: false,
      });

      // Load and parse with MP4Box
      try {
        await exportPlayer.loadArrayBuffer(fileData);
      } catch (e) {
        const hint = isMOV ? ' MOV containers may have unsupported audio codecs.' : '';
        throw new Error(`FAST export failed: WebCodecs/MP4Box parsing failed for clip "${clip.name}": ${e}.${hint} Try PRECISE mode instead.`);
      }

      // Calculate the clip's start time within the export
      const clipStartInExport = Math.max(0, startTime - clip.startTime);
      const clipTime = clip.reversed
        ? clip.outPoint - clipStartInExport
        : clipStartInExport + clip.inPoint;

      // Initialize sequential decoding
      await exportPlayer.prepareForSequentialExport(clipTime);

      this.clipStates.set(clip.id, {
        clipId: clip.id,
        webCodecsPlayer: exportPlayer,
        lastSampleIndex: exportPlayer.getCurrentSampleIndex(),
        isSequential: true,
      });

      console.log(`[FrameExporter] Clip ${clip.name}: FAST mode enabled (${exportPlayer.width}x${exportPlayer.height})`);
    }

    console.log(`[FrameExporter] All ${videoClips.length} clips using FAST WebCodecs sequential decoding`);
  }

  /**
   * Cleanup export mode - destroy dedicated export players
   */
  private cleanupExportMode(): void {
    // Cleanup parallel decoder if used
    if (this.parallelDecoder) {
      this.parallelDecoder.cleanup();
      this.parallelDecoder = null;
      this.useParallelDecode = false;
    }

    // Destroy all dedicated export WebCodecs players
    for (const state of this.clipStates.values()) {
      if (state.webCodecsPlayer && state.isSequential) {
        try {
          state.webCodecsPlayer.endSequentialExport();
          state.webCodecsPlayer.destroy(); // Clean up the dedicated export player
        } catch (e) {
          // Ignore cleanup errors
        }
      }
    }
    this.clipStates.clear();
    console.log('[FrameExporter] Export cleanup complete');
  }

  /**
   * Load file data for a clip from various sources
   */
  private async loadClipFileData(clip: TimelineClip, mediaFile: any): Promise<ArrayBuffer | null> {
    let fileData: ArrayBuffer | null = null;

    // 1. Try media file's file handle via fileSystemService
    const storedHandle = mediaFile?.hasFileHandle ? fileSystemService.getFileHandle(clip.mediaFileId || '') : null;
    if (!fileData && storedHandle) {
      try {
        const file = await storedHandle.getFile();
        fileData = await file.arrayBuffer();
      } catch (e) {
        console.warn(`[FrameExporter] Media file handle failed for ${clip.name}:`, e);
      }
    }

    // 2. Try clip's file property directly
    if (!fileData && clip.file) {
      try {
        fileData = await clip.file.arrayBuffer();
      } catch (e) {
        console.warn(`[FrameExporter] Clip file access failed for ${clip.name}:`, e);
      }
    }

    // 3. Try media file's blob URL
    if (!fileData && mediaFile?.url) {
      try {
        const response = await fetch(mediaFile.url);
        fileData = await response.arrayBuffer();
      } catch (e) {
        console.warn(`[FrameExporter] Media blob URL fetch failed for ${clip.name}:`, e);
      }
    }

    // 4. Try video element's src (blob URL)
    if (!fileData && clip.source?.videoElement?.src) {
      try {
        const response = await fetch(clip.source.videoElement.src);
        fileData = await response.arrayBuffer();
      } catch (e) {
        console.warn(`[FrameExporter] Video src fetch failed for ${clip.name}:`, e);
      }
    }

    return fileData;
  }

  /**
   * Initialize parallel decoding for multiple video clips (including nested)
   */
  private async initializeParallelDecoding(
    clips: TimelineClip[],
    mediaFiles: any[],
    _startTime: number,  // Reserved for future seek optimization
    nestedClips: Array<{ clip: TimelineClip; parentClip: TimelineClip }> = []
  ): Promise<void> {
    this.parallelDecoder = new ParallelDecodeManager();

    // Collect clip info for parallel decoder
    const clipInfos: Array<{
      clipId: string;
      clipName: string;
      fileData: ArrayBuffer;
      startTime: number;
      duration: number;
      inPoint: number;
      outPoint: number;
      reversed: boolean;
      isNested?: boolean;
      parentClipId?: string;
      parentStartTime?: number;
      parentInPoint?: number;
    }> = [];

    // Load regular clips file data in parallel
    const loadPromises = clips.map(async (clip) => {
      const mediaFileId = clip.source!.mediaFileId;
      const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null;
      const fileData = await this.loadClipFileData(clip, mediaFile);

      if (!fileData) {
        throw new Error(`FAST export failed: Could not load file data for clip "${clip.name}". Try PRECISE mode instead.`);
      }

      return {
        clipId: clip.id,
        clipName: clip.name,
        fileData,
        startTime: clip.startTime,
        duration: clip.duration,
        inPoint: clip.inPoint,
        outPoint: clip.outPoint,
        reversed: clip.reversed || false,
      };
    });

    // Load nested clips file data in parallel
    const nestedLoadPromises = nestedClips.map(async ({ clip, parentClip }) => {
      const mediaFileId = clip.source!.mediaFileId;
      const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null;
      const fileData = await this.loadClipFileData(clip, mediaFile);

      if (!fileData) {
        console.warn(`[FrameExporter] Could not load nested clip "${clip.name}", will use HTMLVideoElement`);
        return null;
      }

      // For nested clips, we need to track parent timing for proper timeline mapping
      return {
        clipId: clip.id,
        clipName: `${parentClip.name}/${clip.name}`,
        fileData,
        startTime: clip.startTime,  // Time within the composition
        duration: clip.duration,
        inPoint: clip.inPoint,
        outPoint: clip.outPoint,
        reversed: clip.reversed || false,
        isNested: true,
        parentClipId: parentClip.id,
        parentStartTime: parentClip.startTime,
        parentInPoint: parentClip.inPoint || 0,
      };
    });

    const loadedClips = await Promise.all(loadPromises);
    const loadedNestedClips = (await Promise.all(nestedLoadPromises)).filter(c => c !== null);

    clipInfos.push(...loadedClips);
    clipInfos.push(...loadedNestedClips as any[]);

    console.log(`[FrameExporter] Loaded ${loadedClips.length} regular + ${loadedNestedClips.length} nested clips for parallel decoding`);

    // Initialize parallel decoder
    await this.parallelDecoder.initialize(clipInfos, this.settings.fps);

    // Mark regular clips as using parallel decoding
    for (const clip of clips) {
      this.clipStates.set(clip.id, {
        clipId: clip.id,
        webCodecsPlayer: null,
        lastSampleIndex: 0,
        isSequential: false,
      });
    }

    // Mark nested clips as using parallel decoding
    for (const { clip } of nestedClips) {
      this.clipStates.set(clip.id, {
        clipId: clip.id,
        webCodecsPlayer: null,
        lastSampleIndex: 0,
        isSequential: false,
      });
    }

    this.useParallelDecode = true;
    console.log(`[FrameExporter] Parallel decoding initialized for ${clipInfos.length} total clips`);
  }

  private async seekAllClipsToTime(time: number): Promise<void> {
    // PARALLEL DECODE MODE: Use parallel decoder for pre-fetched frames
    if (this.useParallelDecode && this.parallelDecoder) {
      // Prefetch frames for this time (decodes in parallel across all clips)
      await this.parallelDecoder.prefetchFramesForTime(time);

      // Also handle composition clips that aren't in parallel decode
      const clips = useTimelineStore.getState().getClipsAtTime(time);
      const tracks = useTimelineStore.getState().tracks;
      const seekPromises: Promise<void>[] = [];

      for (const clip of clips) {
        const track = tracks.find(t => t.id === clip.trackId);
        if (!track?.visible) continue;

        // Handle nested composition clips
        // Only seek with HTMLVideoElement if the clip is NOT managed by parallel decoder
        if (clip.isComposition && clip.nestedClips && clip.nestedTracks) {
          const clipLocalTime = time - clip.startTime;
          const nestedTime = clipLocalTime + (clip.inPoint || 0);

          for (const nestedClip of clip.nestedClips) {
            if (nestedTime >= nestedClip.startTime && nestedTime < nestedClip.startTime + nestedClip.duration) {
              if (nestedClip.source?.videoElement) {
                // Skip HTMLVideoElement seek if parallel decoder handles this clip
                if (this.parallelDecoder.hasClip(nestedClip.id)) {
                  continue;  // Parallel decoder already prefetched this frame
                }
                const nestedLocalTime = nestedTime - nestedClip.startTime;
                const nestedClipTime = nestedClip.reversed
                  ? nestedClip.outPoint - nestedLocalTime
                  : nestedLocalTime + nestedClip.inPoint;
                seekPromises.push(this.seekVideo(nestedClip.source.videoElement, nestedClipTime));
              }
            }
          }
        }
      }

      if (seekPromises.length > 0) {
        await Promise.all(seekPromises);
      }

      // Advance buffer position for next frame
      this.parallelDecoder.advanceToTime(time);
      return;
    }

    // SEQUENTIAL MODE: Original approach for single video or precise mode
    const clips = useTimelineStore.getState().getClipsAtTime(time);
    const tracks = useTimelineStore.getState().tracks;
    const seekPromises: Promise<void>[] = [];

    for (const clip of clips) {
      const track = tracks.find(t => t.id === clip.trackId);
      if (!track?.visible) continue;

      // Handle nested composition clips (always use HTMLVideoElement for nested)
      if (clip.isComposition && clip.nestedClips && clip.nestedTracks) {
        const clipLocalTime = time - clip.startTime;
        const nestedTime = clipLocalTime + (clip.inPoint || 0);

        for (const nestedClip of clip.nestedClips) {
          if (nestedTime >= nestedClip.startTime && nestedTime < nestedClip.startTime + nestedClip.duration) {
            if (nestedClip.source?.videoElement) {
              const nestedLocalTime = nestedTime - nestedClip.startTime;
              const nestedClipTime = nestedClip.reversed
                ? nestedClip.outPoint - nestedLocalTime
                : nestedLocalTime + nestedClip.inPoint;
              seekPromises.push(this.seekVideo(nestedClip.source.videoElement, nestedClipTime));
            }
          }
        }
        continue;
      }

      // Handle regular video clips
      if (clip.source?.type === 'video' && clip.source.videoElement) {
        const clipLocalTime = time - clip.startTime;

        // Calculate clip time (handles speed keyframes and reversed clips)
        let clipTime: number;
        try {
          const sourceTime = useTimelineStore.getState().getSourceTimeForClip(clip.id, clipLocalTime);
          const initialSpeed = useTimelineStore.getState().getInterpolatedSpeed(clip.id, 0);
          const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
          clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
        } catch {
          clipTime = clip.reversed
            ? clip.outPoint - clipLocalTime
            : clipLocalTime + clip.inPoint;
          clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, clipTime));
        }

        // Get export state for this clip
        const clipState = this.clipStates.get(clip.id);

        if (clipState?.isSequential && clipState.webCodecsPlayer) {
          // FAST MODE: WebCodecs sequential decoding
          seekPromises.push(clipState.webCodecsPlayer.seekDuringExport(clipTime));
        } else {
          // PRECISE MODE: HTMLVideoElement seeking (frame-accurate but slower)
          seekPromises.push(this.seekVideo(clip.source.videoElement, clipTime));
        }
      }
    }

    if (seekPromises.length > 0) {
      await Promise.all(seekPromises);
    }
  }

  private seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
    return new Promise((resolve) => {
      const targetTime = Math.max(0, Math.min(time, video.duration || 0));

      const timeout = setTimeout(() => {
        console.warn('[FrameExporter] Seek timeout at', targetTime);
        resolve();
      }, 500); // 500ms for AV1 and other slow-decoding codecs

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

    // Filter out clips using WebCodecs (they're already ready)
    const htmlVideoClips = videoClips.filter(clip => {
      const clipState = this.clipStates.get(clip.id);
      return !clipState?.isSequential; // Only wait for HTMLVideoElement clips
    });

    if (htmlVideoClips.length === 0) {
      // All clips using WebCodecs - no waiting needed
      return;
    }

    // Wait for HTMLVideoElement clips to be ready (with timeout)
    const maxWaitTime = 100;
    const startWait = performance.now();

    while (performance.now() - startWait < maxWaitTime) {
      let allReady = true;

      for (const clip of htmlVideoClips) {
        const video = clip.source!.videoElement!;
        const videoReady = video.readyState >= 2 && !video.seeking;

        if (!videoReady) {
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
        const nestedLayers = this.buildNestedLayersForExport(clip, clipLocalTime + (clip.inPoint || 0), time);

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

      // Handle video clips - prefer parallel decode, then WebCodecs, then HTMLVideoElement
      if (clip.source?.type === 'video' && clip.source.videoElement) {
        const video = clip.source.videoElement;
        const clipState = this.clipStates.get(clip.id);

        // Try parallel decoder first (fastest for multi-clip exports)
        if (this.useParallelDecode && this.parallelDecoder && this.parallelDecoder.hasClip(clip.id)) {
          const videoFrame = this.parallelDecoder.getFrameForClip(clip.id, time);
          if (videoFrame) {
            layers.push({
              ...baseLayerProps,
              source: {
                type: 'video',
                videoElement: video, // Keep as fallback
                videoFrame: videoFrame, // Direct VideoFrame from parallel decoder
              },
            });
            continue;
          }
        }

        // Try sequential WebCodecs VideoFrame (fast for single clip)
        if (clipState?.isSequential && clipState.webCodecsPlayer) {
          const videoFrame = clipState.webCodecsPlayer.getCurrentFrame();
          if (videoFrame) {
            layers.push({
              ...baseLayerProps,
              source: {
                type: 'video',
                videoElement: video, // Keep as fallback
                webCodecsPlayer: clipState.webCodecsPlayer, // Engine will use VideoFrame from this
              },
            });
            continue;
          }
        }

        // Fall back to HTMLVideoElement
        const videoReady = video.readyState >= 2 && !video.seeking;
        if (videoReady) {
          layers.push({
            ...baseLayerProps,
            source: {
              type: 'video',
              videoElement: video,
            },
          });
        } else {
          console.warn('[FrameExporter] Video not ready for clip', clip.id, 'readyState:', video.readyState, 'seeking:', video.seeking);
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
   * @param clip - The composition clip
   * @param nestedTime - Time within the composition
   * @param mainTimelineTime - Time on the main timeline (for parallel decoder)
   */
  private buildNestedLayersForExport(clip: TimelineClip, nestedTime: number, mainTimelineTime: number): Layer[] {
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

      // Try parallel decoder first (for nested clips), then fall back to video element
      if (nestedClip.source?.videoElement) {
        // Check if this nested clip has a parallel decoded frame
        if (this.useParallelDecode && this.parallelDecoder && this.parallelDecoder.hasClip(nestedClip.id)) {
          const videoFrame = this.parallelDecoder.getFrameForClip(nestedClip.id, mainTimelineTime);
          if (videoFrame) {
            layers.push({
              ...baseLayer,
              source: {
                type: 'video',
                videoElement: nestedClip.source.videoElement,
                videoFrame: videoFrame,  // Use parallel decoded frame
              },
            } as Layer);
            continue;
          }
        }

        // Fall back to video element
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
      h264: 'avc1.4d0028',
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
