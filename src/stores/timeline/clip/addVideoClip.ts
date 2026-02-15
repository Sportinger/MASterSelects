// Video clip addition - extracted from addClip
// Handles video file loading, WebCodecs initialization, thumbnails, and linked audio

import type { TimelineClip, TimelineTrack } from '../../../types';
import { DEFAULT_TRANSFORM, calculateNativeScale } from '../constants';
import { useMediaStore } from '../../mediaStore';
import { useSettingsStore } from '../../settingsStore';
import { NativeDecoder } from '../../../services/nativeHelper';
import { NativeHelperClient } from '../../../services/nativeHelper/NativeHelperClient';
import {
  initWebCodecsFullMode,
  createAudioElement,
} from '../helpers/webCodecsHelpers';
import { shouldSkipWaveform, generateWaveformForFile } from '../helpers/waveformHelpers';
import { generateLinkedClipIds } from '../helpers/idGenerator';
import { blobUrlManager } from '../helpers/blobUrlManager';
import { updateClipById } from '../helpers/clipStateHelpers';
import { Logger } from '../../../services/logger';
import { thumbnailCache } from '../../../services/thumbnailCache';

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

  // Use native decoder when Turbo Mode is on and helper is connected
  // FFmpeg can decode all formats (H.264, ProRes, DNxHD, etc.) with HW acceleration
  const { nativeDecodeEnabled, nativeHelperConnected } = useSettingsStore.getState();
  const useNativeDecoder = nativeDecodeEnabled && nativeHelperConnected;

  let nativeDecoder: NativeDecoder | null = null;
  let naturalDuration = 5; // default estimate

  // Try Native Helper for professional codecs (ProRes, DNxHD)
  if (useNativeDecoder) {
    try {
      const mediaFile = mediaFileId
        ? useMediaStore.getState().files.find(f => f.id === mediaFileId)
        : null;
      let filePath = mediaFile?.absolutePath || (file as any).path;

      // Check if we have a valid absolute path (Unix: /... , Windows: C:\...)
      const isAbsolute = filePath && (filePath.startsWith('/') || /^[A-Za-z]:[/\\]/.test(filePath));

      // If no absolute path, ask the native helper to locate the file
      if (!isAbsolute) {
        log.debug('No absolute path found, asking native helper to locate', { filename: file.name });
        const located = await NativeHelperClient.locateFile(file.name);
        if (located) {
          filePath = located;
          log.debug('Native helper located file', { filePath });
        } else {
          throw new Error(`Could not locate file "${file.name}" on disk. Try importing via File > Open.`);
        }
      }

      log.debug('Opening with Native Helper', { file: file.name });
      nativeDecoder = await NativeDecoder.open(filePath);
      naturalDuration = nativeDecoder.duration;

      log.debug('Native Helper ready', { width: nativeDecoder.width, height: nativeDecoder.height, fps: nativeDecoder.fps });

      // Decode initial frame so preview isn't black
      await nativeDecoder.seekToFrame(0);

      // Calculate native pixel scale so content appears at actual size
      const nativeScale = calculateNativeScale(nativeDecoder.width, nativeDecoder.height);

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
        transform: { ...DEFAULT_TRANSFORM, scale: nativeScale },
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

  // Fallback to WebCodecs Full Mode if not using native decoder
  if (!nativeDecoder) {
    try {
      const { player, audioPlayer } = await initWebCodecsFullMode(file);
      naturalDuration = player.duration;

      // Calculate native pixel scale so content appears at actual size
      const nativeScale = (player.width && player.height)
        ? calculateNativeScale(player.width, player.height)
        : { x: 1, y: 1 };

      log.debug('WebCodecs Full Mode ready', {
        file: file.name,
        width: player.width,
        height: player.height,
        duration: naturalDuration.toFixed(2),
        hasAudio: player.hasAudioTrack(),
      });

      // Set isLoading: false immediately so clip becomes interactive
      updateClip(clipId, {
        duration: naturalDuration,
        outPoint: naturalDuration,
        source: {
          type: 'video',
          webCodecsPlayer: player,
          audioPlayer: audioPlayer ?? undefined,
          naturalDuration,
          mediaFileId,
        },
        transform: { ...DEFAULT_TRANSFORM, scale: nativeScale },
        isLoading: false,
      });

      if (audioClipId) {
        updateClip(audioClipId, { duration: naturalDuration, outPoint: naturalDuration });
      }

      // Audio detection: use WebCodecsPlayer's MP4Box info
      if (!player.hasAudioTrack() && audioClipId) {
        log.debug('WebCodecs: no audio tracks, removing audio clip', { file: file.name });
        setClips(clips => clips.filter(c => c.id !== audioClipId));
      }
    } catch (err) {
      log.error('WebCodecs Full Mode failed', err);
      // Don't fall back to HTMLVideoElement — this is a fatal error for the clip
      updateClip(clipId, { isLoading: false });
      return;
    }
  }

  // Preload thumbnails into on-demand cache (non-blocking)
  const isLargeFile = shouldSkipWaveform(file);
  if (thumbnailsEnabled && !isLargeFile && mediaFileId) {
    thumbnailCache.preloadClip(mediaFileId, naturalDuration, file);
  }

  // Load audio for linked clip (skip for NativeDecoder - browser can't decode ProRes/DNxHD audio)
  // For WebCodecs path, audio is already loaded via initWebCodecsFullMode
  if (audioClipId && !nativeDecoder) {
    // Get the audioPlayer we already created during initWebCodecsFullMode
    // Also need to set up the audio clip source and generate waveform
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

