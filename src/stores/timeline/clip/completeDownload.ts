// YouTube download completion - extracted from completeDownload
// Handles converting pending download clips to actual video clips

import type { TimelineClip } from '../../../types';
import { DEFAULT_TRANSFORM } from '../constants';
import { useMediaStore } from '../../mediaStore';
import { initWebCodecsPlayer, createAudioElement } from '../helpers/webCodecsHelpers';
import { generateDownloadThumbnails } from '../helpers/thumbnailHelpers';
import { generateWaveformForFile } from '../helpers/waveformHelpers';
import { generateClipId } from '../helpers/idGenerator';
import { blobUrlManager } from '../helpers/blobUrlManager';
import { updateClipById } from '../helpers/clipStateHelpers';

export interface CompleteDownloadParams {
  clipId: string;
  file: File;
  clips: TimelineClip[];
  waveformsEnabled: boolean;
  findAvailableAudioTrack: (startTime: number, duration: number) => string | null;
  updateDuration: () => void;
  invalidateCache: () => void;
  set: (state: any) => void;
  get: () => any;
}

/**
 * Complete a pending YouTube download - convert to actual video clip.
 */
export async function completeDownload(params: CompleteDownloadParams): Promise<void> {
  const {
    clipId,
    file,
    clips,
    waveformsEnabled,
    findAvailableAudioTrack,
    updateDuration,
    invalidateCache,
    set,
    get,
  } = params;

  const clip = clips.find(c => c.id === clipId);
  if (!clip?.isPendingDownload) {
    console.warn('[Download] Clip not found or not pending:', clipId);
    return;
  }

  console.log(`[Download] Completing download for: ${clipId}`);

  // Create and load video element - track URL for cleanup
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  const url = blobUrlManager.create(clipId, file, 'video');
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
    video.load();
  });

  const naturalDuration = video.duration || 30;
  const initialThumbnails = clip.youtubeThumbnail ? [clip.youtubeThumbnail] : [];
  video.currentTime = 0;

  // Import to media store
  const mediaStore = useMediaStore.getState();
  const mediaFile = await mediaStore.importFile(file);

  // Find/create audio track
  const audioTrackId = findAvailableAudioTrack(clip.startTime, naturalDuration);
  const audioClipId = audioTrackId ? generateClipId('clip-audio-yt') : undefined;

  // Update video clip
  const updatedClips = clips.map(c => {
    if (c.id !== clipId) return c;
    return {
      ...c,
      file,
      duration: naturalDuration,
      outPoint: naturalDuration,
      source: {
        type: 'video' as const,
        videoElement: video,
        naturalDuration,
        mediaFileId: mediaFile.id,
      },
      mediaFileId: mediaFile.id,
      linkedClipId: audioClipId,
      thumbnails: initialThumbnails,
      isPendingDownload: false,
      downloadProgress: undefined,
      youtubeVideoId: undefined,
      youtubeThumbnail: undefined,
    };
  });

  // Create linked audio clip
  if (audioTrackId && audioClipId) {
    const audioClip: TimelineClip = {
      id: audioClipId,
      trackId: audioTrackId,
      name: `${clip.name} (Audio)`,
      file,
      startTime: clip.startTime,
      duration: naturalDuration,
      inPoint: 0,
      outPoint: naturalDuration,
      source: { type: 'audio', naturalDuration, mediaFileId: mediaFile.id },
      mediaFileId: mediaFile.id,
      linkedClipId: clipId,
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      isLoading: false,
    };
    updatedClips.push(audioClip);
    console.log(`[Download] Created linked audio clip: ${audioClipId}`);
  }

  set({ clips: updatedClips });
  updateDuration();
  invalidateCache();

  console.log(`[Download] Complete: ${clipId}, duration: ${naturalDuration}s`);

  // Initialize WebCodecsPlayer
  const webCodecsPlayer = await initWebCodecsPlayer(video, 'YouTube download');
  if (webCodecsPlayer) {
    const currentClips = get().clips;
    const targetClip = currentClips.find((c: TimelineClip) => c.id === clipId);
    if (targetClip?.source) {
      set({
        clips: updateClipById(currentClips, clipId, {
          source: { ...targetClip.source, webCodecsPlayer }
        }),
      });
    }
  }

  // Load audio element for linked clip
  if (audioTrackId && audioClipId) {
    const audio = createAudioElement(file);
    // Share the same blob URL reference for the audio clip
    blobUrlManager.share(clipId, audioClipId, 'video');
    audio.src = url;

    set({
      clips: updateClipById(get().clips, audioClipId, {
        source: { type: 'audio' as const, audioElement: audio, naturalDuration, mediaFileId: mediaFile.id }
      }),
    });

    // Generate waveform in background
    if (waveformsEnabled) {
      generateWaveformAsync(audioClipId, file, get, set);
    }
  }

  // Generate real thumbnails in background
  generateThumbnailsAsync(clipId, video, naturalDuration, get, set);
}

/**
 * Generate waveform asynchronously.
 */
async function generateWaveformAsync(
  audioClipId: string,
  file: File,
  get: () => any,
  set: (state: any) => void
): Promise<void> {
  set({ clips: updateClipById(get().clips, audioClipId, { waveformGenerating: true, waveformProgress: 0 }) });

  try {
    const waveform = await generateWaveformForFile(file);
    set({ clips: updateClipById(get().clips, audioClipId, { waveform, waveformGenerating: false }) });
    console.log(`[Download] Waveform generated for audio clip`);
  } catch (e) {
    console.warn('[Download] Waveform generation failed:', e);
    set({ clips: updateClipById(get().clips, audioClipId, { waveformGenerating: false }) });
  }
}

/**
 * Generate thumbnails asynchronously.
 */
async function generateThumbnailsAsync(
  clipId: string,
  video: HTMLVideoElement,
  duration: number,
  get: () => any,
  set: (state: any) => void
): Promise<void> {
  // Wait for video to be ready instead of arbitrary delay
  await new Promise<void>(resolve => {
    if (video.readyState >= 2) {
      resolve();
    } else {
      video.addEventListener('canplay', () => resolve(), { once: true });
      setTimeout(resolve, 1000); // Fallback timeout
    }
  });

  try {
    const thumbnails = await generateDownloadThumbnails(video, duration);
    video.currentTime = 0;
    set({ clips: updateClipById(get().clips, clipId, { thumbnails }) });
    console.log(`[Download] Generated ${thumbnails.length} thumbnails`);
  } catch (e) {
    console.warn('[Download] Thumbnail generation failed:', e);
  }
}
