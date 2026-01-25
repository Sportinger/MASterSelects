# FrameExporter Refactoring Plan

**Target**: Reduce `src/engine/FrameExporter.ts` from 1510 LOC to ~350 LOC main file with helpers below ~300 LOC each.

## Current Issues

### Giant Functions
| Function | Lines | Location |
|----------|-------|----------|
| `buildLayersAtTime` | 177 | 1123-1300 |
| `export` | 153 | 369-522 |
| `seekAllClipsToTime` | 110 | 894-1004 |
| `initializeParallelDecoding` | 108 | 785-892 |
| `initializeFastMode` | 105 | 600-705 |
| `buildNestedLayersForExport` | 102 | 1309-1411 |

### Duplicated Code (~150 LOC)
1. **Transform building** - Repeated in `buildLayersAtTime` and `buildNestedLayersForExport`
2. **Video source handling** - Parallel → WebCodecs → HTMLVideoElement fallback pattern duplicated
3. **Codec string maps** - In `getCodecString` (lines 311-324) and `checkCodecSupport` (lines 1468-1473)
4. **Base layer construction** - Repeated property mapping pattern

### Structure Issues
1. VideoEncoderWrapper (287 LOC) is self-contained but embedded in same file
2. Static preset methods (85 LOC) could be constants
3. ExportClipState interface defined mid-file (line 346-352)

---

## New Structure

```
src/engine/export/
├── index.ts                    # Re-exports (~30 LOC)
├── types.ts                    # Export types & interfaces (~80 LOC)
├── VideoEncoderWrapper.ts      # Video encoding class (~290 LOC)
├── FrameExporter.ts            # Main orchestrator (~350 LOC)
├── ClipPreparation.ts          # Clip setup & initialization (~250 LOC)
├── VideoSeeker.ts              # Seeking & ready-state logic (~180 LOC)
├── ExportLayerBuilder.ts       # Layer building for render (~250 LOC)
└── codecHelpers.ts             # Codec strings & presets (~80 LOC)
```

---

## Phase 1: Create types.ts and codecHelpers.ts

### Step 1.1: Create types.ts

**File**: `src/engine/export/types.ts`

```typescript
// Export-related types and interfaces

import type { Layer } from '../../types';

// ============ VIDEO CODECS ============

export type VideoCodec = 'h264' | 'h265' | 'vp9' | 'av1';
export type ContainerFormat = 'mp4' | 'webm';
export type ExportMode = 'fast' | 'precise';

// ============ EXPORT SETTINGS ============

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
  exportMode?: ExportMode;  // 'fast' = WebCodecs sequential, 'precise' = HTMLVideoElement
}

export interface FullExportSettings extends ExportSettings {
  filename?: string;
}

// ============ PROGRESS ============

export interface ExportProgress {
  phase: 'video' | 'audio' | 'muxing';
  currentFrame: number;
  totalFrames: number;
  percent: number;
  estimatedTimeRemaining: number;
  currentTime: number;
  audioPhase?: 'mixing' | 'encoding' | 'complete';
  audioPercent?: number;
}

// ============ INTERNAL STATE ============

export interface ExportClipState {
  clipId: string;
  webCodecsPlayer: any; // WebCodecsPlayer
  lastSampleIndex: number;
  isSequential: boolean; // true if using sequential decoding
}

// ============ PRESETS ============

export interface ResolutionPreset {
  label: string;
  width: number;
  height: number;
}

export interface FrameRatePreset {
  label: string;
  fps: number;
}

export interface ContainerFormatOption {
  id: ContainerFormat;
  label: string;
  extension: string;
}

export interface VideoCodecOption {
  id: VideoCodec;
  label: string;
  description: string;
}

// ============ LAYER BUILDING ============

export interface LayerTransformData {
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number };
  opacity: number;
  blendMode: string;
}

export interface BaseLayerProps {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  blendMode: string;
  effects: any[];
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number };
}
```

### Step 1.2: Create codecHelpers.ts

**File**: `src/engine/export/codecHelpers.ts`

```typescript
// Codec configuration and preset helpers

import type {
  VideoCodec,
  ContainerFormat,
  ResolutionPreset,
  FrameRatePreset,
  ContainerFormatOption,
  VideoCodecOption,
} from './types';

// ============ CODEC STRINGS ============

/**
 * Get WebCodecs codec string for VideoEncoder configuration.
 */
export function getCodecString(codec: VideoCodec): string {
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

/**
 * Get mp4-muxer codec identifier.
 */
export function getMp4MuxerCodec(codec: VideoCodec): 'avc' | 'hevc' | 'vp9' | 'av1' {
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

/**
 * Get WebM muxer video codec identifier.
 */
export function getWebmMuxerCodec(codec: VideoCodec): 'V_VP9' | 'V_AV1' {
  return codec === 'av1' ? 'V_AV1' : 'V_VP9';
}

/**
 * Check if codec is supported in container.
 */
export function isCodecSupportedInContainer(codec: VideoCodec, container: ContainerFormat): boolean {
  if (container === 'webm') {
    // WebM only supports VP9 and AV1
    return codec === 'vp9' || codec === 'av1';
  }
  // MP4 supports all codecs
  return true;
}

/**
 * Get fallback codec for container.
 */
export function getFallbackCodec(container: ContainerFormat): VideoCodec {
  return container === 'webm' ? 'vp9' : 'h264';
}

// ============ PRESETS ============

export const RESOLUTION_PRESETS: ResolutionPreset[] = [
  { label: '4K (3840x2160)', width: 3840, height: 2160 },
  { label: '1080p (1920x1080)', width: 1920, height: 1080 },
  { label: '720p (1280x720)', width: 1280, height: 720 },
  { label: '480p (854x480)', width: 854, height: 480 },
];

export const FRAME_RATE_PRESETS: FrameRatePreset[] = [
  { label: '60 fps', fps: 60 },
  { label: '30 fps', fps: 30 },
  { label: '25 fps (PAL)', fps: 25 },
  { label: '24 fps (Film)', fps: 24 },
];

export const CONTAINER_FORMATS: ContainerFormatOption[] = [
  { id: 'mp4', label: 'MP4', extension: '.mp4' },
  { id: 'webm', label: 'WebM', extension: '.webm' },
];

export function getVideoCodecsForContainer(container: ContainerFormat): VideoCodecOption[] {
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

// ============ BITRATE ============

export function getRecommendedBitrate(width: number): number {
  if (width >= 3840) return 35_000_000;
  if (width >= 1920) return 15_000_000;
  if (width >= 1280) return 8_000_000;
  return 5_000_000;
}

export const BITRATE_RANGE = {
  min: 1_000_000,
  max: 100_000_000,
  step: 500_000,
};

export function formatBitrate(bitrate: number): string {
  if (bitrate >= 1_000_000) {
    return `${(bitrate / 1_000_000).toFixed(1)} Mbps`;
  }
  return `${(bitrate / 1_000).toFixed(0)} Kbps`;
}

// ============ CODEC SUPPORT CHECK ============

export async function checkCodecSupport(
  codec: VideoCodec,
  width: number,
  height: number
): Promise<boolean> {
  if (!('VideoEncoder' in window)) return false;

  try {
    const support = await VideoEncoder.isConfigSupported({
      codec: getCodecString(codec),
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
```

---

## Phase 2: Extract VideoEncoderWrapper

### Step 2.1: Create VideoEncoderWrapper.ts

**File**: `src/engine/export/VideoEncoderWrapper.ts`

```typescript
// Video encoder wrapper using WebCodecs and mp4/webm muxers

import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4Target } from 'mp4-muxer';
import { Muxer as WebmMuxer, ArrayBufferTarget as WebmTarget } from 'webm-muxer';
import { AudioEncoderWrapper, type AudioCodec, type EncodedAudioResult } from '../audio';
import type { ExportSettings, VideoCodec, ContainerFormat } from './types';
import { getCodecString, getMp4MuxerCodec, getWebmMuxerCodec, isCodecSupportedInContainer, getFallbackCodec } from './codecHelpers';

type MuxerType = Mp4Muxer<Mp4Target> | WebmMuxer<WebmTarget>;

export class VideoEncoderWrapper {
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
    await this.initializeAudioCodec();

    // Determine effective video codec based on container compatibility
    this.effectiveVideoCodec = this.settings.codec;
    if (!isCodecSupportedInContainer(this.settings.codec, this.containerFormat)) {
      console.warn(`[VideoEncoder] ${this.settings.codec} not supported in ${this.containerFormat}, using fallback`);
      this.effectiveVideoCodec = getFallbackCodec(this.containerFormat);
    }

    // Check codec support
    const codecString = getCodecString(this.effectiveVideoCodec);
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

    // Create muxer
    this.createMuxer();

    // Create encoder
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

  private async initializeAudioCodec(): Promise<void> {
    if (!this.hasAudio) return;

    if (this.containerFormat === 'webm') {
      const opusSupported = await AudioEncoderWrapper.isOpusSupported();
      if (opusSupported) {
        this.audioCodec = 'opus';
        console.log('[VideoEncoder] Using Opus audio for WebM');
      } else {
        console.warn('[VideoEncoder] Opus not supported, disabling audio for WebM');
        this.hasAudio = false;
      }
    } else {
      const aacSupported = await AudioEncoderWrapper.isAACSupported();
      if (aacSupported) {
        this.audioCodec = 'aac';
        console.log('[VideoEncoder] Using AAC audio for MP4');
      } else {
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

  private createMuxer(): void {
    const webmVideoCodec = getWebmMuxerCodec(this.effectiveVideoCodec);
    const mp4VideoCodec = getMp4MuxerCodec(this.effectiveVideoCodec);
    const sampleRate = this.settings.audioSampleRate ?? 48000;

    if (this.containerFormat === 'webm') {
      this.muxer = this.hasAudio
        ? new WebmMuxer({
            target: new WebmTarget(),
            video: { codec: webmVideoCodec, width: this.settings.width, height: this.settings.height },
            audio: { codec: 'A_OPUS', sampleRate, numberOfChannels: 2 },
          })
        : new WebmMuxer({
            target: new WebmTarget(),
            video: { codec: webmVideoCodec, width: this.settings.width, height: this.settings.height },
          });
      console.log(`[VideoEncoder] Using WebM/${this.effectiveVideoCodec.toUpperCase()} with ${this.hasAudio ? 'Opus' : 'no'} audio`);
    } else {
      this.muxer = this.hasAudio
        ? new Mp4Muxer({
            target: new Mp4Target(),
            video: { codec: mp4VideoCodec, width: this.settings.width, height: this.settings.height },
            audio: { codec: this.audioCodec, sampleRate, numberOfChannels: 2 },
            fastStart: 'in-memory',
          })
        : new Mp4Muxer({
            target: new Mp4Target(),
            video: { codec: mp4VideoCodec, width: this.settings.width, height: this.settings.height },
            fastStart: 'in-memory',
          });
      console.log(`[VideoEncoder] Using MP4/${this.effectiveVideoCodec.toUpperCase()} with ${this.hasAudio ? this.audioCodec.toUpperCase() : 'no'} audio`);
    }
  }

  getContainerFormat(): ContainerFormat {
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

    // Yield to event loop periodically
    if (frameIndex % 10 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

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
}
```

---

## Phase 3: Extract ClipPreparation

### Step 3.1: Create ClipPreparation.ts

**File**: `src/engine/export/ClipPreparation.ts`

```typescript
// Clip preparation and initialization for export

import type { TimelineClip } from '../../types';
import type { ExportSettings, ExportClipState, ExportMode } from './types';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { fileSystemService } from '../../services/fileSystemService';
import { ParallelDecodeManager } from '../ParallelDecodeManager';

export interface ClipPreparationResult {
  clipStates: Map<string, ExportClipState>;
  parallelDecoder: ParallelDecodeManager | null;
  useParallelDecode: boolean;
  exportMode: ExportMode;
}

/**
 * Prepare all video clips for export based on export mode.
 * FAST mode: WebCodecs with MP4Box parsing - sequential decoding, very fast
 * PRECISE mode: HTMLVideoElement seeking - frame-accurate but slower
 */
export async function prepareClipsForExport(
  settings: ExportSettings,
  exportMode: ExportMode
): Promise<ClipPreparationResult> {
  const { clips, tracks } = useTimelineStore.getState();
  const mediaFiles = useMediaStore.getState().files;
  const startTime = settings.startTime;
  const endTime = settings.endTime;

  const clipStates = new Map<string, ExportClipState>();

  // Find all video clips that will be in the export range
  const videoClips = clips.filter(clip => {
    const track = tracks.find(t => t.id === clip.trackId);
    if (!track?.visible || track.type !== 'video') return false;
    const clipEnd = clip.startTime + clip.duration;
    return clip.startTime < endTime && clipEnd > startTime;
  });

  console.log(`[FrameExporter] Preparing ${videoClips.length} video clips for ${exportMode.toUpperCase()} export...`);

  if (exportMode === 'precise') {
    return initializePreciseMode(videoClips, clipStates);
  }

  // FAST MODE: Try WebCodecs with MP4Box parsing
  try {
    return await initializeFastMode(videoClips, mediaFiles, startTime, clipStates, settings.fps);
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e);
    // Check if this is a codec/parsing error that can be handled by PRECISE mode
    if (error.includes('not supported') || error.includes('FAST export failed')) {
      console.warn(`[FrameExporter] FAST mode failed, auto-fallback to PRECISE: ${error}`);
      clipStates.clear();
      return initializePreciseMode(videoClips, clipStates);
    }
    throw e;
  }
}

function initializePreciseMode(
  videoClips: TimelineClip[],
  clipStates: Map<string, ExportClipState>
): ClipPreparationResult {
  for (const clip of videoClips) {
    if (clip.source?.type !== 'video') continue;
    clipStates.set(clip.id, {
      clipId: clip.id,
      webCodecsPlayer: null,
      lastSampleIndex: 0,
      isSequential: false,
    });
    console.log(`[FrameExporter] Clip ${clip.name}: PRECISE mode (HTMLVideoElement seeking)`);
  }
  console.log(`[FrameExporter] All ${videoClips.length} clips using PRECISE HTMLVideoElement seeking`);

  return {
    clipStates,
    parallelDecoder: null,
    useParallelDecode: false,
    exportMode: 'precise',
  };
}

async function initializeFastMode(
  videoClips: TimelineClip[],
  mediaFiles: any[],
  startTime: number,
  clipStates: Map<string, ExportClipState>,
  fps: number
): Promise<ClipPreparationResult> {
  const { WebCodecsPlayer } = await import('../WebCodecsPlayer');

  // Separate composition clips from regular video clips
  const regularVideoClips: TimelineClip[] = [];
  const nestedVideoClips: Array<{ clip: TimelineClip; parentClip: TimelineClip }> = [];

  for (const clip of videoClips) {
    if (clip.source?.type !== 'video') continue;

    if (clip.isComposition) {
      clipStates.set(clip.id, {
        clipId: clip.id,
        webCodecsPlayer: null,
        lastSampleIndex: 0,
        isSequential: false,
      });
      console.log(`[FrameExporter] Clip ${clip.name}: Composition with nested clips`);

      // Collect nested video clips
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

  // Use parallel decoding if we have 2+ total video clips
  const totalVideoClips = regularVideoClips.length + nestedVideoClips.length;
  if (totalVideoClips >= 2) {
    console.log(`[FrameExporter] Using PARALLEL decoding for ${regularVideoClips.length} regular + ${nestedVideoClips.length} nested = ${totalVideoClips} video clips`);
    return initializeParallelDecoding(regularVideoClips, mediaFiles, startTime, nestedVideoClips, clipStates, fps);
  }

  // Single clip: use sequential approach
  for (const clip of regularVideoClips) {
    const mediaFileId = clip.source!.mediaFileId;
    const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null;
    const fileData = await loadClipFileData(clip, mediaFile);

    if (!fileData) {
      throw new Error(`FAST export failed: Could not load file data for clip "${clip.name}". Try PRECISE mode instead.`);
    }

    // Detect file format from magic bytes
    const header = new Uint8Array(fileData.slice(0, 12));
    const isMOV = header[4] === 0x66 && header[5] === 0x74 && header[6] === 0x79 && header[7] === 0x70 &&
                  (header[8] === 0x71 && header[9] === 0x74);
    const fileType = isMOV ? 'MOV' : 'MP4';

    console.log(`[FrameExporter] Loaded ${clip.name} (${(fileData.byteLength / 1024 / 1024).toFixed(1)}MB, ${fileType})`);

    // Create dedicated WebCodecs player for export
    const exportPlayer = new WebCodecsPlayer({ useSimpleMode: false, loop: false });

    try {
      await exportPlayer.loadArrayBuffer(fileData);
    } catch (e) {
      const hint = isMOV ? ' MOV containers may have unsupported audio codecs.' : '';
      throw new Error(`FAST export failed: WebCodecs/MP4Box parsing failed for clip "${clip.name}": ${e}.${hint} Try PRECISE mode instead.`);
    }

    // Calculate clip start time
    const clipStartInExport = Math.max(0, startTime - clip.startTime);
    const clipTime = clip.reversed
      ? clip.outPoint - clipStartInExport
      : clipStartInExport + clip.inPoint;

    await exportPlayer.prepareForSequentialExport(clipTime);

    clipStates.set(clip.id, {
      clipId: clip.id,
      webCodecsPlayer: exportPlayer,
      lastSampleIndex: exportPlayer.getCurrentSampleIndex(),
      isSequential: true,
    });

    console.log(`[FrameExporter] Clip ${clip.name}: FAST mode enabled (${exportPlayer.width}x${exportPlayer.height})`);
  }

  console.log(`[FrameExporter] All ${videoClips.length} clips using FAST WebCodecs sequential decoding`);

  return {
    clipStates,
    parallelDecoder: null,
    useParallelDecode: false,
    exportMode: 'fast',
  };
}

async function initializeParallelDecoding(
  clips: TimelineClip[],
  mediaFiles: any[],
  _startTime: number,
  nestedClips: Array<{ clip: TimelineClip; parentClip: TimelineClip }>,
  clipStates: Map<string, ExportClipState>,
  fps: number
): Promise<ClipPreparationResult> {
  const parallelDecoder = new ParallelDecodeManager();

  // Load all clip file data in parallel
  const loadPromises = clips.map(async (clip) => {
    const mediaFileId = clip.source!.mediaFileId;
    const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null;
    const fileData = await loadClipFileData(clip, mediaFile);

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

  // Load nested clips
  const nestedLoadPromises = nestedClips.map(async ({ clip, parentClip }) => {
    const mediaFileId = clip.source!.mediaFileId;
    const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null;
    const fileData = await loadClipFileData(clip, mediaFile);

    if (!fileData) {
      console.warn(`[FrameExporter] Could not load nested clip "${clip.name}", will use HTMLVideoElement`);
      return null;
    }

    return {
      clipId: clip.id,
      clipName: `${parentClip.name}/${clip.name}`,
      fileData,
      startTime: clip.startTime,
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

  const clipInfos = [...loadedClips, ...loadedNestedClips as any[]];

  console.log(`[FrameExporter] Loaded ${loadedClips.length} regular + ${loadedNestedClips.length} nested clips for parallel decoding`);

  await parallelDecoder.initialize(clipInfos, fps);

  // Mark clips as using parallel decoding
  for (const clip of clips) {
    clipStates.set(clip.id, {
      clipId: clip.id,
      webCodecsPlayer: null,
      lastSampleIndex: 0,
      isSequential: false,
    });
  }

  for (const { clip } of nestedClips) {
    clipStates.set(clip.id, {
      clipId: clip.id,
      webCodecsPlayer: null,
      lastSampleIndex: 0,
      isSequential: false,
    });
  }

  console.log(`[FrameExporter] Parallel decoding initialized for ${clipInfos.length} total clips`);

  return {
    clipStates,
    parallelDecoder,
    useParallelDecode: true,
    exportMode: 'fast',
  };
}

/**
 * Load file data for a clip from various sources.
 */
export async function loadClipFileData(clip: TimelineClip, mediaFile: any): Promise<ArrayBuffer | null> {
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
 * Cleanup export mode - destroy dedicated export players.
 */
export function cleanupExportMode(
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null
): void {
  // Cleanup parallel decoder
  if (parallelDecoder) {
    parallelDecoder.cleanup();
  }

  // Destroy all dedicated export WebCodecs players
  for (const state of clipStates.values()) {
    if (state.webCodecsPlayer && state.isSequential) {
      try {
        state.webCodecsPlayer.endSequentialExport();
        state.webCodecsPlayer.destroy();
      } catch (e) {
        // Ignore cleanup errors
      }
    }
  }

  clipStates.clear();
  console.log('[FrameExporter] Export cleanup complete');
}
```

---

## Phase 4: Extract VideoSeeker

### Step 4.1: Create VideoSeeker.ts

**File**: `src/engine/export/VideoSeeker.ts`

```typescript
// Video seeking and ready-state management for export

import type { TimelineClip, TimelineTrack } from '../../types';
import type { ExportClipState } from './types';
import { useTimelineStore } from '../../stores/timeline';
import { ParallelDecodeManager } from '../ParallelDecodeManager';

/**
 * Seek all clips to the specified time for frame export.
 */
export async function seekAllClipsToTime(
  time: number,
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Promise<void> {
  // PARALLEL DECODE MODE
  if (useParallelDecode && parallelDecoder) {
    await parallelDecoder.prefetchFramesForTime(time);

    // Handle composition clips not in parallel decode
    const clips = useTimelineStore.getState().getClipsAtTime(time);
    const tracks = useTimelineStore.getState().tracks;
    const seekPromises: Promise<void>[] = [];

    for (const clip of clips) {
      const track = tracks.find(t => t.id === clip.trackId);
      if (!track?.visible) continue;

      // Handle nested composition clips
      if (clip.isComposition && clip.nestedClips && clip.nestedTracks) {
        const clipLocalTime = time - clip.startTime;
        const nestedTime = clipLocalTime + (clip.inPoint || 0);

        for (const nestedClip of clip.nestedClips) {
          if (nestedTime >= nestedClip.startTime && nestedTime < nestedClip.startTime + nestedClip.duration) {
            if (nestedClip.source?.videoElement) {
              // Skip if parallel decoder handles this
              if (parallelDecoder.hasClip(nestedClip.id)) continue;

              const nestedLocalTime = nestedTime - nestedClip.startTime;
              const nestedClipTime = nestedClip.reversed
                ? nestedClip.outPoint - nestedLocalTime
                : nestedLocalTime + nestedClip.inPoint;
              seekPromises.push(seekVideo(nestedClip.source.videoElement, nestedClipTime));
            }
          }
        }
      }
    }

    if (seekPromises.length > 0) {
      await Promise.all(seekPromises);
    }

    parallelDecoder.advanceToTime(time);
    return;
  }

  // SEQUENTIAL MODE
  await seekSequentialMode(time, clipStates);
}

async function seekSequentialMode(
  time: number,
  clipStates: Map<string, ExportClipState>
): Promise<void> {
  const clips = useTimelineStore.getState().getClipsAtTime(time);
  const tracks = useTimelineStore.getState().tracks;
  const seekPromises: Promise<void>[] = [];

  for (const clip of clips) {
    const track = tracks.find(t => t.id === clip.trackId);
    if (!track?.visible) continue;

    // Handle nested composition clips
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
            seekPromises.push(seekVideo(nestedClip.source.videoElement, nestedClipTime));
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

      const clipState = clipStates.get(clip.id);

      if (clipState?.isSequential && clipState.webCodecsPlayer) {
        // FAST MODE: WebCodecs sequential decoding
        seekPromises.push(clipState.webCodecsPlayer.seekDuringExport(clipTime));
      } else {
        // PRECISE MODE: HTMLVideoElement seeking
        seekPromises.push(seekVideo(clip.source.videoElement, clipTime));
      }
    }
  }

  if (seekPromises.length > 0) {
    await Promise.all(seekPromises);
  }
}

/**
 * Seek a video element to a specific time with frame-accurate waiting.
 */
export function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const targetTime = Math.max(0, Math.min(time, video.duration || 0));

    const timeout = setTimeout(() => {
      console.warn('[FrameExporter] Seek timeout at', targetTime);
      resolve();
    }, 500); // 500ms for AV1 and other slow-decoding codecs

    const waitForFrame = () => {
      if ('requestVideoFrameCallback' in video) {
        (video as any).requestVideoFrameCallback(() => {
          clearTimeout(timeout);
          resolve();
        });
      } else {
        // Fallback: wait for readyState
        let retries = 0;
        const maxRetries = 30;
        const vid = video;

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

    // If already at correct time, still wait for frame callback
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
 * Wait for all video clips at a given time to have their frames ready.
 */
export async function waitForAllVideosReady(
  time: number,
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Promise<void> {
  const clips = useTimelineStore.getState().getClipsAtTime(time);
  const tracks = useTimelineStore.getState().tracks;

  const videoClips = clips.filter(clip => {
    const track = tracks.find(t => t.id === clip.trackId);
    return track?.visible && clip.source?.type === 'video' && clip.source.videoElement;
  });

  if (videoClips.length === 0) return;

  // Filter out clips using WebCodecs or parallel decode
  const htmlVideoClips = videoClips.filter(clip => {
    const clipState = clipStates.get(clip.id);
    if (clipState?.isSequential) return false;
    if (useParallelDecode && parallelDecoder?.hasClip(clip.id)) return false;
    return true;
  });

  if (htmlVideoClips.length === 0) return;

  // Wait for HTMLVideoElement clips
  const maxWaitTime = 100;
  const startWait = performance.now();

  while (performance.now() - startWait < maxWaitTime) {
    let allReady = true;

    for (const clip of htmlVideoClips) {
      const video = clip.source!.videoElement!;
      if (video.readyState < 2 || video.seeking) {
        allReady = false;
        break;
      }
    }

    if (allReady) {
      await new Promise(r => requestAnimationFrame(r));
      return;
    }

    await new Promise(r => requestAnimationFrame(r));
  }

  console.warn('[FrameExporter] Timeout waiting for videos to be ready at time', time);
}
```

---

## Phase 5: Extract ExportLayerBuilder

### Step 5.1: Create ExportLayerBuilder.ts

**File**: `src/engine/export/ExportLayerBuilder.ts`

```typescript
// Layer building for export rendering

import type { Layer, TimelineClip, NestedCompositionData } from '../../types';
import type { ExportClipState, BaseLayerProps } from './types';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { ParallelDecodeManager } from '../ParallelDecodeManager';

/**
 * Build layers for rendering at a specific time.
 */
export function buildLayersAtTime(
  time: number,
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Layer[] {
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
    const baseLayerProps = buildBaseLayerProps(clip, clipLocalTime, trackIndex, getInterpolatedTransform, getInterpolatedEffects);

    // Handle nested compositions
    if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
      const nestedLayers = buildNestedLayersForExport(clip, clipLocalTime + (clip.inPoint || 0), time, parallelDecoder, useParallelDecode);

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
      const layer = buildVideoLayer(clip, baseLayerProps, time, clipStates, parallelDecoder, useParallelDecode);
      if (layer) layers.push(layer);
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
 * Build base layer properties from clip transform.
 */
function buildBaseLayerProps(
  clip: TimelineClip,
  clipLocalTime: number,
  trackIndex: number,
  getInterpolatedTransform: Function,
  getInterpolatedEffects: Function
): BaseLayerProps {
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
      blendMode: 'normal',
    };
  }

  // Get effects safely
  let effects: any[] = [];
  try {
    effects = getInterpolatedEffects(clip.id, clipLocalTime);
  } catch (e) {
    console.warn('[FrameExporter] Effects interpolation failed for clip', clip.id, e);
  }

  return {
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
}

/**
 * Build video layer with appropriate source (parallel > webcodecs > HTMLVideoElement).
 */
function buildVideoLayer(
  clip: TimelineClip,
  baseLayerProps: BaseLayerProps,
  time: number,
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Layer | null {
  const video = clip.source!.videoElement!;
  const clipState = clipStates.get(clip.id);

  // Try parallel decoder first
  if (useParallelDecode && parallelDecoder && parallelDecoder.hasClip(clip.id)) {
    const videoFrame = parallelDecoder.getFrameForClip(clip.id, time);
    if (videoFrame) {
      return {
        ...baseLayerProps,
        source: {
          type: 'video',
          videoElement: video,
          videoFrame: videoFrame,
        },
      };
    }
    console.warn(`[FrameExporter] Parallel decode missing frame for clip "${clip.name}" at time ${time.toFixed(3)}s`);
  }

  // Try sequential WebCodecs VideoFrame
  if (clipState?.isSequential && clipState.webCodecsPlayer) {
    const videoFrame = clipState.webCodecsPlayer.getCurrentFrame();
    if (videoFrame) {
      return {
        ...baseLayerProps,
        source: {
          type: 'video',
          videoElement: video,
          webCodecsPlayer: clipState.webCodecsPlayer,
        },
      };
    }
  }

  // Fall back to HTMLVideoElement
  const videoReady = video.readyState >= 2 && !video.seeking;
  if (videoReady) {
    if (useParallelDecode && parallelDecoder?.hasClip(clip.id)) {
      console.warn(`[FrameExporter] Falling back to HTMLVideoElement for "${clip.name}" - frame may be incorrect`);
    }
    return {
      ...baseLayerProps,
      source: {
        type: 'video',
        videoElement: video,
      },
    };
  }

  console.warn('[FrameExporter] Video not ready for clip', clip.id, 'readyState:', video.readyState, 'seeking:', video.seeking);
  return null;
}

/**
 * Build layers for a nested composition at export time.
 */
function buildNestedLayersForExport(
  clip: TimelineClip,
  nestedTime: number,
  mainTimelineTime: number,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Layer[] {
  if (!clip.nestedClips || !clip.nestedTracks) return [];

  const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
  const layers: Layer[] = [];

  for (let i = 0; i < nestedVideoTracks.length; i++) {
    const nestedTrack = nestedVideoTracks[i];
    const nestedClip = clip.nestedClips.find(
      nc =>
        nc.trackId === nestedTrack.id &&
        nestedTime >= nc.startTime &&
        nestedTime < nc.startTime + nc.duration
    );

    if (!nestedClip) continue;

    const baseLayer = buildNestedBaseLayer(nestedClip);

    // Try parallel decoder first, then video element
    if (nestedClip.source?.videoElement) {
      if (useParallelDecode && parallelDecoder && parallelDecoder.hasClip(nestedClip.id)) {
        const videoFrame = parallelDecoder.getFrameForClip(nestedClip.id, mainTimelineTime);
        if (videoFrame) {
          layers.push({
            ...baseLayer,
            source: {
              type: 'video',
              videoElement: nestedClip.source.videoElement,
              videoFrame: videoFrame,
            },
          } as Layer);
          continue;
        }
        console.warn(`[FrameExporter] Parallel decode missing frame for nested clip "${nestedClip.name}"`);
      }

      // Fall back to video element
      if (useParallelDecode && parallelDecoder?.hasClip(nestedClip.id)) {
        console.warn(`[FrameExporter] Falling back to HTMLVideoElement for nested clip "${nestedClip.name}"`);
      }
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
        source: { type: 'image', imageElement: nestedClip.source.imageElement },
      } as Layer);
    } else if (nestedClip.source?.textCanvas) {
      layers.push({
        ...baseLayer,
        source: { type: 'text', textCanvas: nestedClip.source.textCanvas },
      } as Layer);
    }
  }

  return layers;
}

/**
 * Build base layer for nested clip.
 */
function buildNestedBaseLayer(nestedClip: TimelineClip): BaseLayerProps {
  const transform = nestedClip.transform || {
    position: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1 },
    rotation: { x: 0, y: 0, z: 0 },
    opacity: 1,
    blendMode: 'normal',
  };

  return {
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
}
```

---

## Phase 6: Update Main FrameExporter.ts

### Step 6.1: Create index.ts

**File**: `src/engine/export/index.ts`

```typescript
// Export module - re-exports all public types and classes

export type {
  VideoCodec,
  ContainerFormat,
  ExportMode,
  ExportSettings,
  FullExportSettings,
  ExportProgress,
  ResolutionPreset,
  FrameRatePreset,
  ContainerFormatOption,
  VideoCodecOption,
} from './types';

export { VideoEncoderWrapper } from './VideoEncoderWrapper';
export { FrameExporter, downloadBlob } from './FrameExporter';

// Codec helpers for UI
export {
  RESOLUTION_PRESETS,
  FRAME_RATE_PRESETS,
  CONTAINER_FORMATS,
  getVideoCodecsForContainer,
  getRecommendedBitrate,
  BITRATE_RANGE,
  formatBitrate,
  checkCodecSupport,
} from './codecHelpers';
```

### Step 6.2: Update FrameExporter.ts

**File**: `src/engine/export/FrameExporter.ts`

```typescript
// Frame-by-frame exporter for precise video rendering
// Main orchestrator - delegates to specialized modules

import { engine } from '../WebGPUEngine';
import { AudioExportPipeline, type AudioExportProgress, type EncodedAudioResult } from '../audio';
import { ParallelDecodeManager } from '../ParallelDecodeManager';
import type { FullExportSettings, ExportProgress, ExportMode, ExportClipState } from './types';
import { VideoEncoderWrapper } from './VideoEncoderWrapper';
import { prepareClipsForExport, cleanupExportMode } from './ClipPreparation';
import { seekAllClipsToTime, waitForAllVideosReady } from './VideoSeeker';
import { buildLayersAtTime } from './ExportLayerBuilder';
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

    try {
      // Prepare clips for export
      const preparation = await prepareClipsForExport(this.settings, this.exportMode);
      this.clipStates = preparation.clipStates;
      this.parallelDecoder = preparation.parallelDecoder;
      this.useParallelDecode = preparation.useParallelDecode;
      this.exportMode = preparation.exportMode;

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

        if (frame % 30 === 0 || frame < 5) {
          console.log(`[FrameExporter] Processing frame ${frame}/${totalFrames} at time ${time.toFixed(3)}s`);
        }

        await seekAllClipsToTime(time, this.clipStates, this.parallelDecoder, this.useParallelDecode);
        await waitForAllVideosReady(time, this.clipStates, this.parallelDecoder, this.useParallelDecode);

        const layers = buildLayersAtTime(time, this.clipStates, this.parallelDecoder, this.useParallelDecode);

        if (layers.length === 0 && frame === 0) {
          console.warn('[FrameExporter] No layers at time', time);
        }

        // Check GPU device validity
        if (!engine.isDeviceValid()) {
          throw new Error('WebGPU device lost during export. Try keeping the browser tab in focus.');
        }

        engine.render(layers);

        const pixels = await engine.readPixels();
        if (!pixels) {
          if (!engine.isDeviceValid()) {
            throw new Error('WebGPU device lost during export. Try keeping the browser tab in focus.');
          }
          console.error('[FrameExporter] Failed to read pixels at frame', frame);
          continue;
        }

        await this.encoder.encodeFrame(pixels, frame);

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
    this.parallelDecoder = null;
    this.useParallelDecode = false;
    engine.setExporting(false);
    engine.setResolution(originalDimensions.width, originalDimensions.height);
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
```

---

## Phase 7: Update Imports

### Step 7.1: Update files that import from FrameExporter.ts

Search for files importing from `../engine/FrameExporter` or `./FrameExporter` and update to:

```typescript
// Old:
import { FrameExporter, downloadBlob } from '../engine/FrameExporter';
import type { ExportSettings, ExportProgress } from '../engine/FrameExporter';

// New:
import { FrameExporter, downloadBlob } from '../engine/export';
import type { ExportSettings, ExportProgress } from '../engine/export';
```

Files to update:
- `src/components/export/ExportDialog.tsx`
- `src/components/export/ExportPanel.tsx`
- `src/services/aiTools.ts` (if using export types)

---

## Validation Commands

After each phase, run:

```bash
# TypeScript check
npx tsc --noEmit

# Build test
npm run build

# Lint check
npm run lint
```

---

## Rollback

If issues occur, revert:
```bash
git checkout src/engine/FrameExporter.ts
rm -rf src/engine/export
```

---

## Notes for AI Agent

1. **Order matters**: Create types.ts and codecHelpers.ts first, then VideoEncoderWrapper, then the other modules

2. **Imports**: The new modules import from relative paths within `src/engine/export/`. External imports (ParallelDecodeManager, stores, etc.) use `../` or `../../` paths

3. **Type exports**: Ensure all types used by external files are exported from index.ts

4. **Static methods**: The FrameExporter class keeps static methods as wrappers for backwards compatibility, but delegates to codecHelpers

5. **Test focus**: Test the export functionality thoroughly after extraction - seek timing and parallel decode are critical

6. **Keep downloadBlob**: This utility function stays in FrameExporter.ts and is re-exported from index.ts

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| FrameExporter.ts LOC | 1510 | ~350 |
| Largest file | FrameExporter.ts (1510) | ClipPreparation.ts (~250) |
| Duplicate code | ~150 LOC | ~0 LOC |
| Files | 1 | 8 |
