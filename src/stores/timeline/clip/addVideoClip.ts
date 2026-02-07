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
import { updateClipById } from '../helpers/clipStateHelpers';
import { detectVideoAudio } from '../helpers/audioDetection';
import { getMP4MetadataFast, estimateDurationFromFileSize } from '../helpers/mp4MetadataHelper';
import { Logger } from '../../../services/logger';

const log = Logger.create('AddVideoClip');

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
        log.debug('No absolute path found, trying', { filePath });
      }

      log.debug('Opening with Native Helper', { file: file.name });
      nativeDecoder = await NativeDecoder.open(filePath);
      naturalDuration = nativeDecoder.duration;

      log.debug('Native Helper ready', { width: nativeDecoder.width, height: nativeDecoder.height, fps: nativeDecoder.fps });

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
      log.warn('Native Helper failed, falling back to browser', err);
      nativeDecoder = null;
    }
  }

  // Fallback to HTMLVideoElement if not using native decoder
  if (!nativeDecoder) {
    video = createVideoElement(file);
    // Track the blob URL for cleanup
    blobUrlManager.create(clipId, file, 'video');

    // Race: MP4Box container parsing vs HTMLVideoElement metadata
    // MP4Box reads from both start+end of file to handle camera MOV files
    // where the moov atom is at the end (not web-optimized)
    const [mp4Meta, _] = await Promise.all([
      getMP4MetadataFast(file, 6000),
      waitForVideoMetadata(video, 8000),
    ]);

    // Prefer MP4Box duration (works with any codec, reads moov from end)
    // Fall back to video element, then file size estimate
    if (mp4Meta?.duration && mp4Meta.duration > 0) {
      naturalDuration = mp4Meta.duration;
      log.debug('Using MP4Box duration', { file: file.name, duration: naturalDuration.toFixed(2) });
    } else if (video.duration && isFinite(video.duration)) {
      naturalDuration = video.duration;
      log.debug('Using video element duration', { file: file.name, duration: naturalDuration.toFixed(2) });
    } else {
      // Last resort: estimate from file size
      naturalDuration = estimateDurationFromFileSize(file);
      log.warn('Duration unknown, estimated from file size', { file: file.name, duration: naturalDuration.toFixed(2), size: file.size });
    }

    // Set isLoading: false immediately so clip becomes interactive
    updateClip(clipId, {
      duration: naturalDuration,
      outPoint: naturalDuration,
      source: { type: 'video', videoElement: video, naturalDuration, mediaFileId },
      isLoading: false,
    });

    if (audioClipId) {
      updateClip(audioClipId, { duration: naturalDuration, outPoint: naturalDuration });
    }

    // Audio detection in background (non-blocking)
    // Use MP4Box result if available, otherwise detect separately
    if (mp4Meta) {
      if (!mp4Meta.hasAudio && audioClipId) {
        log.debug('MP4Box: no audio tracks, removing audio clip', { file: file.name });
        setClips(clips => clips.filter(c => c.id !== audioClipId));
        blobUrlManager.revokeAll(audioClipId);
      }
    } else {
      detectVideoAudio(file).then(videoHasAudio => {
        if (!videoHasAudio) {
          log.debug('Video has no audio tracks', { file: file.name });
          if (audioClipId) {
            log.debug('Removing audio clip for video without audio', { file: file.name });
            setClips(clips => clips.filter(c => c.id !== audioClipId));
            blobUrlManager.revokeAll(audioClipId);
          }
        }
      });
    }

    // If video element eventually loads real metadata, update duration
    // (covers the file-size-estimate fallback case)
    if (!video.duration || !isFinite(video.duration)) {
      video.addEventListener('loadedmetadata', () => {
        if (video!.duration && isFinite(video!.duration) && video!.duration !== naturalDuration) {
          const realDuration = video!.duration;
          log.debug('Late metadata arrived, updating duration', { file: file.name, from: naturalDuration.toFixed(2), to: realDuration.toFixed(2) });
          setClips(clips => clips.map(c => {
            if (c.id !== clipId) return c;
            return {
              ...c,
              duration: realDuration,
              outPoint: realDuration,
              source: c.source ? { ...c.source, naturalDuration: realDuration } : c.source,
            };
          }));
          // Also update audio clip duration
          if (audioClipId) {
            setClips(clips => clips.map(c => {
              if (c.id !== audioClipId) return c;
              return { ...c, duration: realDuration, outPoint: realDuration };
            }));
          }
        }
      }, { once: true });
    }

    // Warm up video decoder in background (non-blocking)
    warmUpVideoDecoder(video).then(() => {
      log.debug('Decoder warmed up', { file: file.name });
    });

    // Initialize WebCodecsPlayer for hardware-accelerated decoding (non-blocking)
    initWebCodecsPlayer(video, file.name).then(webCodecsPlayer => {
      if (webCodecsPlayer) {
        setClips(clips => clips.map(c => {
          if (c.id !== clipId || !c.source) return c;
          return {
            ...c,
            source: { ...c.source, webCodecsPlayer },
          };
        }));
      }
    });
  }

  // Generate thumbnails in background (non-blocking) - only if enabled and not large file
  const isLargeFile = shouldSkipWaveform(file);
  if (thumbnailsEnabled && !isLargeFile && video) {
    generateThumbnailsAsync(video, naturalDuration, clipId, file.name, setClips);
  } else if (nativeDecoder) {
    log.debug('Skipping thumbnails for NativeDecoder file', { file: file.name });
  }

  // Load audio for linked clip (skip for NativeDecoder - browser can't decode ProRes/DNxHD audio)
  // For browser path, audio clip is already created and will be removed by background detectVideoAudio if no audio
  if (audioClipId && !nativeDecoder) {
    loadLinkedAudio(file, audioClipId, naturalDuration, mediaFileId, waveformsEnabled, updateClip, setClips);
  } else if (audioClipId && nativeDecoder) {
    log.debug('Skipping audio decoding for NativeDecoder file (audio clip kept)', { file: file.name });
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

    log.debug('Starting thumbnail generation', { file: fileName });
    const thumbnails = await generateThumbnails(video, duration);
    log.debug('Thumbnails complete', { count: thumbnails.length, file: fileName });

    setClips(clips => clips.map(c => c.id === clipId ? { ...c, thumbnails } : c));

    // Seek back to start
    video.currentTime = 0;
  } catch (e) {
    log.warn('Thumbnail generation failed', e);
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
      log.warn('Waveform generation failed', e);
      setClips(clips => updateClipById(clips, audioClipId, { waveformGenerating: false }));
    }
  }
}

