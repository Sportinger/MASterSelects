// Clip preparation and initialization for export

import { Logger } from '../../services/logger';
import type { TimelineClip } from '../../stores/timeline/types';

const log = Logger.create('ClipPreparation');
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
  const endPrepare = log.time('prepareClipsForExport TOTAL');
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

  log.info(`Preparing ${videoClips.length} video clips for ${exportMode.toUpperCase()} export...`);

  if (exportMode === 'precise') {
    endPrepare();
    return initializePreciseMode(videoClips, clipStates);
  }

  // FAST MODE: WebCodecs with MP4Box parsing
  // NOTE: Auto-fallback to PRECISE mode is DISABLED for debugging
  // All errors will be thrown to surface issues with FAST mode
  return await initializeFastMode(videoClips, mediaFiles, startTime, clipStates, settings.fps, endPrepare);
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
    log.debug(`Clip ${clip.name}: PRECISE mode (HTMLVideoElement seeking)`);
  }
  log.info(`All ${videoClips.length} clips using PRECISE HTMLVideoElement seeking`);

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
  fps: number,
  endPrepare: () => void
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
      log.debug(`Clip ${clip.name}: Composition with nested clips`);

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
    log.info(`Using PARALLEL decoding for ${regularVideoClips.length} regular + ${nestedVideoClips.length} nested = ${totalVideoClips} video clips`);
    return initializeParallelDecoding(regularVideoClips, mediaFiles, startTime, nestedVideoClips, clipStates, fps, endPrepare);
  }

  // Single clip: use sequential approach
  for (const clip of regularVideoClips) {
    const mediaFileId = clip.source!.mediaFileId;
    const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null;

    const endLoad = log.time(`loadClipFileData "${clip.name}"`);
    const fileData = await loadClipFileData(clip, mediaFile);
    endLoad();

    if (!fileData) {
      throw new Error(`FAST export failed: Could not load file data for clip "${clip.name}". Try PRECISE mode instead.`);
    }

    // Detect file format from magic bytes
    const header = new Uint8Array(fileData.slice(0, 12));
    const isMOV = header[4] === 0x66 && header[5] === 0x74 && header[6] === 0x79 && header[7] === 0x70 &&
                  (header[8] === 0x71 && header[9] === 0x74);
    const fileType = isMOV ? 'MOV' : 'MP4';

    log.debug(`Loaded ${clip.name} (${(fileData.byteLength / 1024 / 1024).toFixed(1)}MB, ${fileType})`);

    // Create dedicated WebCodecs player for export
    const exportPlayer = new WebCodecsPlayer({ useSimpleMode: false, loop: false });

    const endParse = log.time(`loadArrayBuffer "${clip.name}"`);
    try {
      await exportPlayer.loadArrayBuffer(fileData);
      endParse();
    } catch (e) {
      endParse();
      const hint = isMOV ? ' MOV containers may have unsupported audio codecs.' : '';
      throw new Error(`FAST export failed: WebCodecs/MP4Box parsing failed for clip "${clip.name}": ${e}.${hint} Try PRECISE mode instead.`);
    }

    // Calculate clip start time
    const clipStartInExport = Math.max(0, startTime - clip.startTime);
    const clipTime = clip.reversed
      ? clip.outPoint - clipStartInExport
      : clipStartInExport + clip.inPoint;

    const endSeqPrep = log.time(`prepareForSequentialExport "${clip.name}"`);
    await exportPlayer.prepareForSequentialExport(clipTime);
    endSeqPrep();

    clipStates.set(clip.id, {
      clipId: clip.id,
      webCodecsPlayer: exportPlayer,
      lastSampleIndex: exportPlayer.getCurrentSampleIndex(),
      isSequential: true,
    });

    log.debug(`Clip ${clip.name}: FAST mode enabled (${exportPlayer.width}x${exportPlayer.height})`);
  }

  log.info(`All ${videoClips.length} clips using FAST WebCodecs sequential decoding`);
  endPrepare();

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
  fps: number,
  endPrepare: () => void
): Promise<ClipPreparationResult> {
  const parallelDecoder = new ParallelDecodeManager();

  // Load all clip file data in parallel
  const endLoadAll = log.time('loadAllClipFileData');
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
      log.warn(`Could not load nested clip "${clip.name}", will use HTMLVideoElement`);
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
  endLoadAll();

  const clipInfos = [...loadedClips, ...loadedNestedClips as any[]];

  log.info(`Loaded ${loadedClips.length} regular + ${loadedNestedClips.length} nested clips for parallel decoding`);

  const endParallelInit = log.time('parallelDecoder.initialize');
  await parallelDecoder.initialize(clipInfos, fps);
  endParallelInit();

  // Pre-decode first frame to ensure it's ready when export starts
  // This is critical because the parallel decoder initializes lazily
  const endPrefetch = log.time('parallelDecoder.prefetchFirstFrame');
  await parallelDecoder.prefetchFramesForTime(_startTime);

  // Verify first frame is actually decoded for clips active at start time
  const MAX_RETRIES = 5;
  for (const clipInfo of clipInfos) {
    // Check if clip is active at start time
    let clipActiveAtStart: boolean;
    if (clipInfo.isNested && clipInfo.parentStartTime !== undefined) {
      // Nested clip: check if parent comp is active and clip is active within it
      const compTime = _startTime - clipInfo.parentStartTime - (clipInfo.parentInPoint || 0);
      clipActiveAtStart = compTime >= clipInfo.startTime && compTime < clipInfo.startTime + clipInfo.duration;
    } else {
      // Regular clip
      clipActiveAtStart = _startTime >= clipInfo.startTime && _startTime < clipInfo.startTime + clipInfo.duration;
    }

    console.log(`[ClipPrep] Clip "${clipInfo.clipName}": startTime=${clipInfo.startTime}, exportStart=${_startTime}, active=${clipActiveAtStart}`);
    if (!clipActiveAtStart) {
      console.log(`[ClipPrep] SKIPPING "${clipInfo.clipName}" - not active at export start`);
      continue;
    }
    console.log(`[ClipPrep] VERIFYING "${clipInfo.clipName}" first frame`);


    let frame = parallelDecoder.getFrameForClip(clipInfo.clipId, _startTime);

    if (!frame) {
      // Retry with delays
      for (let retry = 0; retry < MAX_RETRIES && !frame; retry++) {
        log.warn(`First frame not ready for "${clipInfo.clipName}" (attempt ${retry + 1}/${MAX_RETRIES}), retrying...`);
        await new Promise(r => setTimeout(r, 200)); // Give decoder time
        await parallelDecoder.prefetchFramesForTime(_startTime);
        frame = parallelDecoder.getFrameForClip(clipInfo.clipId, _startTime);
      }
    }

    if (!frame) {
      throw new Error(`Failed to decode first frame for clip "${clipInfo.clipName}" after ${MAX_RETRIES} attempts. The video file may be corrupted or use an unsupported codec.`);
    }
  }
  endPrefetch();

  // Also seek all HTMLVideoElements to their correct start positions as fallback
  // This ensures correct frame if parallel decoder fails and falls back to video element
  const seekPromises: Promise<void>[] = [];
  for (const clip of clips) {
    if (clip.source?.videoElement) {
      const video = clip.source.videoElement;
      const clipLocalTime = Math.max(0, _startTime - clip.startTime);
      const sourceTime = clip.reversed
        ? clip.outPoint - clipLocalTime
        : clip.inPoint + clipLocalTime;

      seekPromises.push(new Promise<void>((resolve) => {
        const targetTime = Math.max(0, Math.min(sourceTime, video.duration || 0));
        if (Math.abs(video.currentTime - targetTime) < 0.01) {
          resolve();
          return;
        }
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
        video.currentTime = targetTime;
        // Timeout fallback
        setTimeout(() => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        }, 500);
      }));
    }
  }
  if (seekPromises.length > 0) {
    await Promise.all(seekPromises);
    // Wait for frames to be ready after seek
    await new Promise(r => setTimeout(r, 50));
    log.info(`Seeked ${seekPromises.length} video elements to start positions as fallback`);
  }

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

  log.info(`Parallel decoding initialized for ${clipInfos.length} total clips`);
  endPrepare();

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
      log.warn(`Media file handle failed for ${clip.name}:`, e);
    }
  }

  // 2. Try clip's file property directly
  if (!fileData && clip.file) {
    try {
      fileData = await clip.file.arrayBuffer();
    } catch (e) {
      log.warn(`Clip file access failed for ${clip.name}:`, e);
    }
  }

  // 3. Try media file's blob URL
  if (!fileData && mediaFile?.url) {
    try {
      const response = await fetch(mediaFile.url);
      fileData = await response.arrayBuffer();
    } catch (e) {
      log.warn(`Media blob URL fetch failed for ${clip.name}:`, e);
    }
  }

  // 4. Try video element's src (blob URL)
  if (!fileData && clip.source?.videoElement?.src) {
    try {
      const response = await fetch(clip.source.videoElement.src);
      fileData = await response.arrayBuffer();
    } catch (e) {
      log.warn(`Video src fetch failed for ${clip.name}:`, e);
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
  log.info('Export cleanup complete');
}
