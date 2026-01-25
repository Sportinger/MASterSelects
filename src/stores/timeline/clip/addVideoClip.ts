// Video clip addition - extracted from addClip
// Handles video file loading, WebCodecs initialization, thumbnails, and linked audio

import type { TimelineClip, TimelineTrack } from '../../../types';
import { DEFAULT_TRANSFORM } from '../constants';
import { generateThumbnails } from '../utils';
import { useMediaStore } from '../../mediaStore';
import { useSettingsStore } from '../../settingsStore';
import { NativeDecoder } from '../../../services/nativeHelper';
import { isProfessionalCodecFile } from '../helpers/mediaTypeHelpers';
import {
  initWebCodecsPlayer,
  warmUpVideoDecoder,
  createVideoElement,
  createAudioElement,
  waitForVideoMetadata,
} from '../helpers/webCodecsHelpers';
import { shouldSkipWaveform, generateWaveformForFile } from '../helpers/waveformHelpers';
import { generateLinkedClipIds } from '../helpers/idGenerator';
import { blobUrlManager } from '../helpers/blobUrlManager';
import { updateClipById, createUpdateBatch } from '../helpers/clipStateHelpers';

export interface AddVideoClipParams {
  trackId: string;
  file: File;
  startTime: number;
  estimatedDuration: number;
  mediaFileId?: string;
  tracks: TimelineTrack[];
  findAvailableAudioTrack: (startTime: number, duration: number) => string | null;
}

export interface AddVideoClipResult {
  videoClip: TimelineClip;
  audioClip: TimelineClip | null;
  audioClipId: string | undefined;
}

/**
 * Create placeholder clips for video (and linked audio) immediately.
 * Returns clips ready to be added to state while media loads in background.
 */
export function createVideoClipPlaceholders(params: AddVideoClipParams): AddVideoClipResult {
  const { trackId, file, startTime, estimatedDuration, mediaFileId, findAvailableAudioTrack } = params;

  const { videoId: clipId, audioId } = generateLinkedClipIds();
  const audioTrackId = findAvailableAudioTrack(startTime, estimatedDuration);
  const audioClipId = audioTrackId ? audioId : undefined;

  const videoClip: TimelineClip = {
    id: clipId,
    trackId,
    name: file.name,
    file,
    startTime,
    duration: estimatedDuration,
    inPoint: 0,
    outPoint: estimatedDuration,
    source: { type: 'video', naturalDuration: estimatedDuration, mediaFileId },
    linkedClipId: audioClipId,
    transform: { ...DEFAULT_TRANSFORM },
    effects: [],
    isLoading: true,
  };

  let audioClip: TimelineClip | null = null;
  if (audioTrackId && audioClipId) {
    audioClip = {
      id: audioClipId,
      trackId: audioTrackId,
      name: `${file.name} (Audio)`,
      file,
      startTime,
      duration: estimatedDuration,
      inPoint: 0,
      outPoint: estimatedDuration,
      source: { type: 'audio', naturalDuration: estimatedDuration, mediaFileId },
      linkedClipId: clipId,
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      isLoading: true,
    };
  }

  return { videoClip, audioClip, audioClipId };
}

export interface LoadVideoMediaParams {
  clipId: string;
  audioClipId?: string;
  file: File;
  mediaFileId?: string;
  thumbnailsEnabled: boolean;
  waveformsEnabled: boolean;
  updateClip: (id: string, updates: Partial<TimelineClip>) => void;
  setClips: (updater: (clips: TimelineClip[]) => TimelineClip[]) => void;
}

/**
 * Load video media in background - handles Native Helper, WebCodecs, thumbnails, and audio.
 */
export async function loadVideoMedia(params: LoadVideoMediaParams): Promise<void> {
  const {
    clipId,
    audioClipId,
    file,
    mediaFileId,
    thumbnailsEnabled,
    waveformsEnabled,
    updateClip,
    setClips,
  } = params;

  // Check if this is a professional codec file that needs Native Helper
  const isProfessional = isProfessionalCodecFile(file);
  const { turboModeEnabled, nativeHelperConnected } = useSettingsStore.getState();
  const useNativeDecoder = isProfessional && turboModeEnabled && nativeHelperConnected;

  let nativeDecoder: NativeDecoder | null = null;
  let video: HTMLVideoElement | null = null;
  let naturalDuration = 5; // default estimate

  // Try Native Helper for professional codecs (ProRes, DNxHD)
  if (useNativeDecoder) {
    try {
      const mediaFile = mediaFileId
        ? useMediaStore.getState().files.find(f => f.id === mediaFileId)
        : null;
      let filePath = mediaFile?.absolutePath || (file as any).path;

      // If no absolute path, try common locations
      if (!filePath || !filePath.startsWith('/')) {
        filePath = `/home/admin/Desktop/${file.name}`;
        console.log(`[Video] No absolute path found, trying:`, filePath);
      }

      console.log(`[Video] Opening ${file.name} with Native Helper`);
      nativeDecoder = await NativeDecoder.open(filePath);
      naturalDuration = nativeDecoder.duration;

      console.log(`[Video] Native Helper ready: ${nativeDecoder.width}x${nativeDecoder.height} @ ${nativeDecoder.fps}fps`);

      // Decode initial frame so preview isn't black
      await nativeDecoder.seekToFrame(0);

      updateClip(clipId, {
        duration: naturalDuration,
        outPoint: naturalDuration,
        source: {
          type: 'video',
          naturalDuration,
          mediaFileId,
          nativeDecoder,
          filePath,
        },
        isLoading: false,
      });

      if (audioClipId) {
        updateClip(audioClipId, { duration: naturalDuration, outPoint: naturalDuration });
      }
    } catch (err) {
      console.warn(`[Video] Native Helper failed, falling back to browser:`, err);
      nativeDecoder = null;
    }
  }

  // Fallback to HTMLVideoElement if not using native decoder
  if (!nativeDecoder) {
    video = createVideoElement(file);
    // Track the blob URL for cleanup
    blobUrlManager.create(clipId, file, 'video');
    await waitForVideoMetadata(video);

    naturalDuration = video.duration || 5;

    // Update clip with actual duration
    updateClip(clipId, {
      duration: naturalDuration,
      outPoint: naturalDuration,
      source: { type: 'video', videoElement: video, naturalDuration, mediaFileId },
      isLoading: false,
    });

    if (audioClipId) {
      updateClip(audioClipId, { duration: naturalDuration, outPoint: naturalDuration });
    }

    // Warm up video decoder in background (non-blocking)
    warmUpVideoDecoder(video).then(() => {
      console.log(`[Video] Decoder warmed up for ${file.name}`);
    });

    // Initialize WebCodecsPlayer for hardware-accelerated decoding
    const webCodecsPlayer = await initWebCodecsPlayer(video, file.name);
    if (webCodecsPlayer) {
      setClips(clips => clips.map(c => {
        if (c.id !== clipId || !c.source) return c;
        return {
          ...c,
          source: { ...c.source, webCodecsPlayer },
        };
      }));
    }
  }

  // Generate thumbnails in background (non-blocking) - only if enabled and not large file
  const isLargeFile = shouldSkipWaveform(file);
  if (thumbnailsEnabled && !isLargeFile && video) {
    generateThumbnailsAsync(video, naturalDuration, clipId, file.name, setClips);
  } else if (nativeDecoder) {
    console.log(`[Thumbnails] Skipping for NativeDecoder file: ${file.name}`);
  }

  // Load audio for linked clip (skip for NativeDecoder - browser can't decode ProRes/DNxHD audio)
  if (audioClipId && !nativeDecoder) {
    await loadLinkedAudio(file, audioClipId, naturalDuration, mediaFileId, waveformsEnabled, updateClip, setClips);
  } else if (audioClipId && nativeDecoder) {
    console.log(`[Audio] Skipping for NativeDecoder file: ${file.name}`);
    updateClip(audioClipId, {
      source: { type: 'audio', naturalDuration, mediaFileId },
      isLoading: false,
    });
  }

  // Sync to media store
  const mediaStore = useMediaStore.getState();
  if (!mediaStore.getFileByName(file.name)) {
    mediaStore.importFile(file);
  }
}

/**
 * Generate thumbnails asynchronously without blocking.
 */
async function generateThumbnailsAsync(
  video: HTMLVideoElement,
  duration: number,
  clipId: string,
  fileName: string,
  setClips: (updater: (clips: TimelineClip[]) => TimelineClip[]) => void
): Promise<void> {
  try {
    // Wait for video to be ready for thumbnail generation
    await new Promise<void>((resolve) => {
      if (video.readyState >= 2) {
        resolve();
      } else {
        video.oncanplay = () => resolve();
        setTimeout(resolve, 2000); // Timeout fallback
      }
    });

    console.log(`[Thumbnails] Starting generation for ${fileName}...`);
    const thumbnails = await generateThumbnails(video, duration);
    console.log(`[Thumbnails] Complete: ${thumbnails.length} thumbnails for ${fileName}`);

    setClips(clips => clips.map(c => c.id === clipId ? { ...c, thumbnails } : c));

    // Seek back to start
    video.currentTime = 0;
  } catch (e) {
    console.warn('[Thumbnails] Failed:', e);
  }
}

/**
 * Load audio element and generate waveform for linked audio clip.
 */
async function loadLinkedAudio(
  file: File,
  audioClipId: string,
  naturalDuration: number,
  mediaFileId: string | undefined,
  waveformsEnabled: boolean,
  updateClip: (id: string, updates: Partial<TimelineClip>) => void,
  setClips: (updater: (clips: TimelineClip[]) => TimelineClip[]) => void
): Promise<void> {
  const audio = createAudioElement(file);
  // Track the blob URL for cleanup
  blobUrlManager.create(audioClipId, file, 'audio');

  // Mark audio clip as ready immediately
  updateClip(audioClipId, {
    source: { type: 'audio', audioElement: audio, naturalDuration, mediaFileId },
    isLoading: false,
  });

  // Generate waveform in background (non-blocking) - only if enabled and not large file
  const isLargeFile = shouldSkipWaveform(file);
  if (waveformsEnabled && !isLargeFile) {
    // Mark waveform generation starting
    setClips(clips => updateClipById(clips, audioClipId, { waveformGenerating: true, waveformProgress: 0 }));

    try {
      const waveform = await generateWaveformForFile(file);
      setClips(clips => updateClipById(clips, audioClipId, { waveform, waveformGenerating: false, waveformProgress: 100 }));
    } catch (e) {
      console.warn('[Waveform] Failed:', e);
      setClips(clips => updateClipById(clips, audioClipId, { waveformGenerating: false }));
    }
  }
}
