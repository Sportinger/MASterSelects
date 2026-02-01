// Composition clip addition - extracted from addCompClip
// Handles nested composition loading, audio mixdown, and linked audio creation

import type { TimelineClip, TimelineTrack } from '../../../types';
import type { Composition } from '../types';
import { DEFAULT_TRANSFORM } from '../constants';
import { useMediaStore } from '../../mediaStore';
import { initWebCodecsPlayer } from '../helpers/webCodecsHelpers';
import { findOrCreateAudioTrack, createCompositionAudioClip } from '../helpers/audioTrackHelpers';
import { generateSilentWaveform } from '../helpers/waveformHelpers';
import { generateCompClipId, generateClipId, generateNestedClipId } from '../helpers/idGenerator';
import { blobUrlManager } from '../helpers/blobUrlManager';
import { updateClipById } from '../helpers/clipStateHelpers';
import { Logger } from '../../../services/logger';
import { thumbnailRenderer } from '../../../services/thumbnailRenderer';
import { compositionRenderer } from '../../../services/compositionRenderer';

const log = Logger.create('AddCompClip');

export interface AddCompClipParams {
  trackId: string;
  composition: Composition;
  startTime: number;
  findNonOverlappingPosition: (clipId: string, startTime: number, trackId: string, duration: number) => number;
}

/**
 * Create a content hash for nested composition change detection.
 */
export function createNestedContentHash(timelineData: any): string {
  if (!timelineData) return '';
  const clipData = timelineData.clips?.map((c: any) => ({
    id: c.id,
    inPoint: c.inPoint,
    outPoint: c.outPoint,
    startTime: c.startTime,
    effectCount: c.effects?.length ?? 0,
  })) ?? [];
  return JSON.stringify({
    clipCount: timelineData.clips?.length ?? 0,
    duration: timelineData.duration,
    clips: clipData,
  });
}

/**
 * Create placeholder composition clip immediately.
 */
export function createCompClipPlaceholder(params: AddCompClipParams): TimelineClip {
  const { trackId, composition, startTime, findNonOverlappingPosition } = params;

  const clipId = generateCompClipId();
  const compDuration = composition.timelineData?.duration ?? composition.duration;
  const finalStartTime = findNonOverlappingPosition(clipId, startTime, trackId, compDuration);

  // Create content hash for change detection
  const nestedContentHash = createNestedContentHash(composition.timelineData);

  return {
    id: clipId,
    trackId,
    name: composition.name,
    file: new File([], composition.name),
    startTime: finalStartTime,
    duration: compDuration,
    inPoint: 0,
    outPoint: compDuration,
    source: { type: 'video', naturalDuration: compDuration },
    transform: { ...DEFAULT_TRANSFORM },
    effects: [],
    isLoading: true,
    isComposition: true,
    compositionId: composition.id,
    nestedClips: [],
    nestedTracks: [],
    nestedContentHash,
  };
}

export interface LoadNestedClipsParams {
  compClipId: string;
  composition: Composition;
  get: () => any;
  set: (state: any) => void;
}

/**
 * Helper to update a nested clip inside a comp clip (immutable update).
 * This ensures React/Zustand detects the change and triggers re-renders.
 */
function updateNestedClipInCompClip(
  clips: TimelineClip[],
  compClipId: string,
  nestedClipId: string,
  updates: Partial<TimelineClip>
): TimelineClip[] {
  return clips.map(clip => {
    if (clip.id !== compClipId || !clip.nestedClips) return clip;

    // Check if the nested clip exists
    const nestedClipExists = clip.nestedClips.some(nc => nc.id === nestedClipId);
    if (!nestedClipExists) return clip;

    // Create new nestedClips array with updated nested clip
    const updatedNestedClips = clip.nestedClips.map(nc =>
      nc.id === nestedClipId ? { ...nc, ...updates } : nc
    );

    // Return new comp clip object to trigger re-render
    return { ...clip, nestedClips: updatedNestedClips };
  });
}

/**
 * Load nested clips from composition's timeline data.
 */
export async function loadNestedClips(params: LoadNestedClipsParams): Promise<TimelineClip[]> {
  const { compClipId, composition, get, set } = params;

  if (!composition.timelineData) return [];

  const mediaStore = useMediaStore.getState();
  const nestedClips: TimelineClip[] = [];

  log.info('loadNestedClips', {
    compClipId,
    compositionId: composition.id,
    compositionName: composition.name,
    serializedClipCount: composition.timelineData.clips.length,
    serializedClips: composition.timelineData.clips.map((c: any) => ({
      id: c.id,
      name: c.name,
      trackId: c.trackId,
      mediaFileId: c.mediaFileId,
      sourceType: c.sourceType,
    })),
    availableMediaFiles: mediaStore.files.map(f => ({ id: f.id, name: f.name })),
  });

  for (const serializedClip of composition.timelineData.clips) {
    const mediaFile = mediaStore.files.find(f => f.id === serializedClip.mediaFileId);
    if (!mediaFile?.file) {
      log.warn('Could not find media file for nested clip', {
        clip: serializedClip.name,
        mediaFileId: serializedClip.mediaFileId,
        sourceType: serializedClip.sourceType,
      });
      continue;
    }

    const nestedClip: TimelineClip = {
      id: generateNestedClipId(compClipId, serializedClip.id),
      trackId: serializedClip.trackId,
      name: serializedClip.name,
      file: mediaFile.file,
      startTime: serializedClip.startTime,
      duration: serializedClip.duration,
      inPoint: serializedClip.inPoint,
      outPoint: serializedClip.outPoint,
      source: null,
      thumbnails: serializedClip.thumbnails,
      linkedClipId: serializedClip.linkedClipId,
      waveform: serializedClip.waveform,
      transform: serializedClip.transform,
      effects: serializedClip.effects || [],
      masks: serializedClip.masks || [],
      isLoading: true,
    };

    nestedClips.push(nestedClip);

    // Load media element async - track URL for cleanup
    const type = serializedClip.sourceType;
    const urlType = type === 'video' ? 'video' : type === 'audio' ? 'audio' : 'image';
    const fileUrl = blobUrlManager.create(nestedClip.id, mediaFile.file, urlType as 'video' | 'audio' | 'image');

    if (type === 'video') {
      loadVideoNestedClip(compClipId, nestedClip.id, fileUrl, mediaFile.file.name, get, set);
    } else if (type === 'audio') {
      loadAudioNestedClip(compClipId, nestedClip.id, fileUrl, get, set);
    } else if (type === 'image') {
      loadImageNestedClip(compClipId, nestedClip.id, fileUrl, get, set);
    }
  }

  return nestedClips;
}

function loadVideoNestedClip(
  compClipId: string,
  nestedClipId: string,
  fileUrl: string,
  fileName: string,
  get: () => any,
  set: (state: any) => void
): void {
  const video = document.createElement('video');
  video.src = fileUrl;
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.crossOrigin = 'anonymous';

  // Force browser to start loading
  video.load();

  video.addEventListener('loadedmetadata', async () => {
    // Force browser to decode actual video frames by playing briefly
    // This ensures readyState reaches HAVE_CURRENT_DATA (2) or higher
    try {
      await video.play();
      video.pause();
      video.currentTime = 0;

      // Wait for the seek to complete and frame to be decoded
      await new Promise<void>((resolve) => {
        const checkReady = () => {
          if (video.readyState >= 2) {
            resolve();
          } else {
            // Keep checking until ready
            requestAnimationFrame(checkReady);
          }
        };
        video.addEventListener('seeked', () => {
          checkReady();
        }, { once: true });
        // Fallback: also check immediately in case already ready
        checkReady();
      });
    } catch (e) {
      // play() might fail due to autoplay policy, try alternative approach
      log.debug('Play failed, trying seek approach', { nestedClipId, error: e });
      video.currentTime = 0.001; // Seek slightly to trigger frame decode
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
        // Timeout fallback
        setTimeout(resolve, 500);
      });
    }

    const source: TimelineClip['source'] = {
      type: 'video',
      videoElement: video,
      naturalDuration: video.duration,
    };

    // Initialize WebCodecsPlayer
    const webCodecsPlayer = await initWebCodecsPlayer(video, fileName);
    if (webCodecsPlayer) {
      source.webCodecsPlayer = webCodecsPlayer;
    }

    // Immutably update the nested clip inside the comp clip
    const updatedClips = updateNestedClipInCompClip(get().clips, compClipId, nestedClipId, {
      source,
      isLoading: false,
    });

    set({ clips: updatedClips });

    // Invalidate cache to ensure re-render with new video
    const { invalidateCache } = get();
    if (invalidateCache) invalidateCache();

    log.debug('Nested video loaded', {
      compClipId,
      nestedClipId,
      fileName,
      readyState: video.readyState
    });
  }, { once: true });

  video.addEventListener('error', (e) => {
    log.error('Nested video load error', { compClipId, nestedClipId, fileName, error: e });
  });
}

function loadAudioNestedClip(
  compClipId: string,
  nestedClipId: string,
  fileUrl: string,
  get: () => any,
  set: (state: any) => void
): void {
  const audio = document.createElement('audio');
  audio.src = fileUrl;
  audio.preload = 'auto';

  audio.addEventListener('canplaythrough', () => {
    // Immutably update the nested clip inside the comp clip
    set({
      clips: updateNestedClipInCompClip(get().clips, compClipId, nestedClipId, {
        source: {
          type: 'audio',
          audioElement: audio,
          naturalDuration: audio.duration,
        },
        isLoading: false,
      }),
    });

    // Invalidate cache
    const { invalidateCache } = get();
    if (invalidateCache) invalidateCache();

    log.debug('Nested audio loaded', { compClipId, nestedClipId });
  }, { once: true });
}

function loadImageNestedClip(
  compClipId: string,
  nestedClipId: string,
  fileUrl: string,
  get: () => any,
  set: (state: any) => void
): void {
  const img = new Image();
  img.src = fileUrl;

  img.addEventListener('load', () => {
    // Immutably update the nested clip inside the comp clip
    set({
      clips: updateNestedClipInCompClip(get().clips, compClipId, nestedClipId, {
        source: { type: 'image', imageElement: img },
        isLoading: false,
      }),
    });

    // Invalidate cache to ensure re-render
    const { invalidateCache } = get();
    if (invalidateCache) invalidateCache();

    log.debug('Nested image loaded', { compClipId, nestedClipId });
  }, { once: true });
}

export interface GenerateCompThumbnailsParams {
  clipId: string;
  nestedClips: TimelineClip[];
  compDuration: number;
  thumbnailsEnabled: boolean;
  get: () => any;
  set: (state: any) => void;
}

/**
 * Generate thumbnails for nested composition using WebGPU rendering.
 * Shows all layers with effects, not just the first video.
 * Falls back to polling first video if WebGPU fails.
 */
export async function generateCompThumbnails(params: GenerateCompThumbnailsParams): Promise<void> {
  const { clipId, compDuration, thumbnailsEnabled, get, set } = params;

  if (!thumbnailsEnabled) return;

  // Get the composition ID from the clip
  const compClip = get().clips.find((c: TimelineClip) => c.id === clipId);
  if (!compClip?.compositionId) {
    log.warn('No composition ID for comp clip', { clipId });
    return;
  }

  // Invalidate cached composition data to ensure fresh sources
  // This is critical when nested comp content has changed
  compositionRenderer.invalidateComposition(compClip.compositionId);

  // Try WebGPU-based rendering first (shows all layers with effects)
  try {
    log.info('Generating WebGPU thumbnails for nested comp', { clipId, compositionId: compClip.compositionId, compDuration });

    const thumbnails = await thumbnailRenderer.generateCompositionThumbnails(
      compClip.compositionId,
      compDuration,
      { count: 10, width: 160, height: 90 }
    );

    log.info('WebGPU thumbnail result', { clipId, count: thumbnails.length, hasData: thumbnails.length > 0 });

    if (thumbnails.length > 0) {
      set({ clips: updateClipById(get().clips, clipId, { thumbnails }) });
      log.info('Set thumbnails for nested comp', { clipId, count: thumbnails.length });
      return;
    } else {
      log.warn('WebGPU returned empty thumbnails', { clipId, compositionId: compClip.compositionId });
    }
  } catch (e) {
    log.warn('WebGPU thumbnail generation failed, falling back to video-based', e);
  }

  // Fallback: Use the first video's frames (original method)
  await generateCompThumbnailsFallback(params);
}

/**
 * Fallback: Generate thumbnails from first video in nested composition.
 * Used when WebGPU rendering is not available or fails.
 */
async function generateCompThumbnailsFallback(params: GenerateCompThumbnailsParams): Promise<void> {
  const { clipId, nestedClips, compDuration, get, set } = params;

  // Find the first video clip by file type or source type
  const firstVideoClip = nestedClips.find(c =>
    c.file?.type?.startsWith('video/') ||
    c.source?.type === 'video' ||
    // Fallback: check file extension
    /\.(mp4|mov|webm|avi|mkv|m4v)$/i.test(c.file?.name || c.name || '')
  );
  const firstVideoClipId = firstVideoClip?.id;
  if (!firstVideoClipId) return;

  // Import generateThumbnails lazily for fallback
  const { generateThumbnails } = await import('../utils');

  let attempts = 0;
  const maxAttempts = 50; // 5 seconds max (50 * 100ms)

  const checkAndGenerate = async () => {
    if (!get().thumbnailsEnabled) return;

    // Get fresh state - the nestedClips might have been updated since initial call
    const compClip = get().clips.find((c: TimelineClip) => c.id === clipId);
    const currentNestedClip = compClip?.nestedClips?.find((nc: TimelineClip) => nc.id === firstVideoClipId);
    const video = currentNestedClip?.source?.videoElement;

    if (video && video.readyState >= 2) {
      try {
        const thumbnails = await generateThumbnails(video, compDuration);
        set({ clips: updateClipById(get().clips, clipId, { thumbnails }) });
        log.debug('Generated fallback thumbnails for nested comp', { clipId, count: thumbnails.length });
      } catch (e) {
        log.warn('Failed to generate fallback thumbnails for nested comp', e);
      }
    } else if (attempts < maxAttempts) {
      // Video not ready yet, try again
      attempts++;
      setTimeout(checkAndGenerate, 100);
    } else {
      log.warn('Timeout waiting for nested video to load for thumbnails', { clipId });
    }
  };

  // Start checking after a short delay to allow initial load
  setTimeout(checkAndGenerate, 100);
}

export interface CreateCompLinkedAudioParams {
  compClipId: string;
  composition: Composition;
  compClipStartTime: number;
  compDuration: number;
  tracks: TimelineTrack[];
  set: (state: any) => void;
  get: () => any;
}

/**
 * Create linked audio clip for composition (with or without actual audio).
 * MERGED from 3 duplicate branches in original code.
 */
export async function createCompLinkedAudioClip(params: CreateCompLinkedAudioParams): Promise<void> {
  const { compClipId, composition, compClipStartTime, compDuration, tracks, set, get } = params;

  // Mark as generating
  set({ clips: updateClipById(get().clips, compClipId, { mixdownGenerating: true }) });

  let hasAudio = false;
  let mixdownAudio: HTMLAudioElement | undefined;
  let waveform: number[] = [];
  let mixdownBuffer: AudioBuffer | undefined;

  // Only try to generate mixdown if we have timeline data
  if (composition.timelineData) {
    try {
      const { compositionAudioMixer } = await import('../../../services/compositionAudioMixer');
      log.debug('Generating audio mixdown for nested comp', { composition: composition.name });
      const mixdownResult = await compositionAudioMixer.mixdownComposition(composition.id);

      if (mixdownResult?.hasAudio) {
        hasAudio = true;
        mixdownAudio = compositionAudioMixer.createAudioElement(mixdownResult.buffer);
        mixdownAudio.preload = 'auto';
        waveform = mixdownResult.waveform;
        mixdownBuffer = mixdownResult.buffer;
      }
    } catch (e) {
      log.error('Failed to generate audio mixdown for nested comp', e);
    }
  }

  // Find or create audio track
  const { trackId: audioTrackId, newTrack } = findOrCreateAudioTrack(tracks);
  if (newTrack) {
    set({ tracks: [...get().tracks, newTrack] });
    log.debug('Created new audio track for nested comp', { composition: composition.name });
  }

  // Create audio clip
  const audioClipId = generateClipId('clip-comp-audio');
  const audioClip = createCompositionAudioClip({
    clipId: audioClipId,
    trackId: audioTrackId,
    compositionName: composition.name,
    compositionId: composition.id,
    startTime: compClipStartTime,
    duration: compDuration,
    audioElement: mixdownAudio || document.createElement('audio'),
    waveform: waveform.length > 0 ? waveform : generateSilentWaveform(compDuration),
    mixdownBuffer,
    linkedClipId: compClipId,
  });

  // Update comp clip and add audio clip
  const clipsAfter = get().clips;
  set({
    clips: [
      ...clipsAfter.map((c: TimelineClip) =>
        c.id === compClipId
          ? { ...c, linkedClipId: audioClipId, mixdownGenerating: false, hasMixdownAudio: hasAudio }
          : c
      ),
      audioClip,
    ],
  });

  log.debug('Created linked audio clip for nested comp', { composition: composition.name, hasAudio });
}
