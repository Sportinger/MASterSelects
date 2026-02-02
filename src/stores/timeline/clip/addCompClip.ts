// Composition clip addition - extracted from addCompClip
// Handles nested composition loading, audio mixdown, and linked audio creation

import type { TimelineClip, TimelineTrack, CompositionTimelineData, SerializableClip, Keyframe } from '../../../types';
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
// Note: compositionRenderer is used elsewhere for cache invalidation

const log = Logger.create('AddCompClip');

// Store interaction types for composition clip operations
interface CompClipStoreState {
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  thumbnailsEnabled: boolean;
  clipKeyframes: Map<string, Keyframe[]>;
  invalidateCache?: () => void;
}

type CompClipStoreGet = () => CompClipStoreState;
type CompClipStoreSet = (state: Partial<CompClipStoreState>) => void;

export interface AddCompClipParams {
  trackId: string;
  composition: Composition;
  startTime: number;
  findNonOverlappingPosition: (clipId: string, startTime: number, trackId: string, duration: number) => number;
}

/**
 * Calculate normalized boundary positions (0-1) for all clips in a nested composition.
 * These are used to render visual markers showing where clips start/end.
 */
export function calculateNestedClipBoundaries(
  timelineData: CompositionTimelineData | undefined,
  compDuration: number
): number[] {
  if (!timelineData?.clips || !timelineData?.tracks || compDuration <= 0) {
    return [];
  }

  // Get visible video track IDs
  const videoTrackIds = new Set(
    timelineData.tracks
      .filter((t: { type: string; visible?: boolean }) => t.type === 'video' && t.visible !== false)
      .map((t: { id: string }) => t.id)
  );

  // Collect all clip boundaries on video tracks
  const boundaries = new Set<number>();

  for (const clip of timelineData.clips) {
    // Only include clips on visible video tracks
    if (!videoTrackIds.has(clip.trackId)) continue;

    const startNorm = clip.startTime / compDuration;
    const endNorm = (clip.startTime + clip.duration) / compDuration;

    // Only add if within valid range (0-1)
    if (startNorm >= 0 && startNorm <= 1) {
      boundaries.add(startNorm);
    }
    if (endNorm >= 0 && endNorm <= 1) {
      boundaries.add(endNorm);
    }
  }

  // Convert to sorted array, excluding 0 and 1 (the comp's own boundaries)
  return Array.from(boundaries)
    .filter(b => b > 0.001 && b < 0.999) // Exclude very start/end
    .sort((a, b) => a - b);
}

/**
 * Build clip segments with thumbnails for nested composition display.
 * Each segment represents one clip with its own thumbnails.
 */
export interface ClipSegmentData {
  clipId: string;
  clipName: string;
  startNorm: number;
  endNorm: number;
  thumbnails: string[];
}

export async function buildClipSegments(
  timelineData: CompositionTimelineData | undefined,
  compDuration: number,
  nestedClips: TimelineClip[]
): Promise<ClipSegmentData[]> {
  if (!timelineData?.clips || !timelineData?.tracks || compDuration <= 0) {
    return [];
  }

  const { generateThumbnails } = await import('../utils');

  // Get visible video track IDs
  const videoTrackIds = new Set(
    timelineData.tracks
      .filter((t: { type: string; visible?: boolean }) => t.type === 'video' && t.visible !== false)
      .map((t: { id: string }) => t.id)
  );

  const segments: ClipSegmentData[] = [];

  // Process each serialized clip
  for (const serializedClip of timelineData.clips) {
    // Only include clips on visible video tracks
    if (!videoTrackIds.has(serializedClip.trackId)) continue;

    // Skip audio-only clips
    if (serializedClip.sourceType === 'audio') continue;

    const startNorm = serializedClip.startTime / compDuration;
    const endNorm = (serializedClip.startTime + serializedClip.duration) / compDuration;

    // Find the corresponding loaded nested clip
    const nestedClip = nestedClips.find(nc =>
      nc.id.includes(serializedClip.id) || nc.name === serializedClip.name
    );

    let thumbnails: string[] = [];

    // Generate thumbnails from the nested clip's source
    if (nestedClip?.source?.videoElement) {
      const video = nestedClip.source.videoElement;
      if (video.readyState >= 2) {
        try {
          // Generate thumbnails for this clip's duration
          const clipDuration = serializedClip.outPoint - serializedClip.inPoint;
          const inPoint = serializedClip.inPoint || 0;
          // Calculate thumb count based on segment width, minimum 1
          const segmentWidth = endNorm - startNorm;
          const thumbCount = Math.max(1, Math.ceil(segmentWidth * 10)); // ~10 thumbs for full width
          thumbnails = await generateThumbnails(video, clipDuration, inPoint, thumbCount);
        } catch (e) {
          log.warn('Failed to generate segment thumbnails', { clipId: serializedClip.id, error: e });
        }
      }
    } else if (nestedClip?.source?.imageElement) {
      // For images, create a single thumbnail from the image
      const img = nestedClip.source.imageElement;
      try {
        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 90;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          thumbnails = [canvas.toDataURL('image/jpeg', 0.7)];
        }
      } catch (e) {
        log.warn('Failed to generate image segment thumbnail', { clipId: serializedClip.id });
      }
    }

    segments.push({
      clipId: serializedClip.id,
      clipName: serializedClip.name,
      startNorm,
      endNorm,
      thumbnails,
    });
  }

  // Sort by start position
  segments.sort((a, b) => a.startNorm - b.startNorm);

  log.info('Built clip segments', {
    segmentCount: segments.length,
    segments: segments.map(s => ({
      name: s.clipName,
      range: `${(s.startNorm * 100).toFixed(1)}%-${(s.endNorm * 100).toFixed(1)}%`,
      thumbCount: s.thumbnails.length,
    })),
  });

  return segments;
}

/**
 * Create a content hash for nested composition change detection.
 */
export function createNestedContentHash(timelineData: CompositionTimelineData | undefined): string {
  if (!timelineData) return '';
  const clipData = timelineData.clips?.map((c) => ({
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
  get: CompClipStoreGet;
  set: CompClipStoreSet;
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

  // Collect keyframes for nested clips (will be added to store)
  const nestedKeyframes = new Map<string, Keyframe[]>();

  log.info('loadNestedClips', {
    compClipId,
    compositionId: composition.id,
    compositionName: composition.name,
    serializedClipCount: composition.timelineData.clips.length,
    serializedClips: composition.timelineData.clips.map((c: SerializableClip) => ({
      id: c.id,
      name: c.name,
      trackId: c.trackId,
      mediaFileId: c.mediaFileId,
      sourceType: c.sourceType,
      hasKeyframes: !!(c.keyframes && c.keyframes.length > 0),
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

    const nestedClipId = generateNestedClipId(compClipId, serializedClip.id);

    const nestedClip: TimelineClip = {
      id: nestedClipId,
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

    // Load keyframes for this nested clip (with updated clip ID)
    if (serializedClip.keyframes && serializedClip.keyframes.length > 0) {
      const updatedKeyframes = serializedClip.keyframes.map((kf: Keyframe) => ({
        ...kf,
        clipId: nestedClipId, // Update clip ID to match nested clip ID
      }));
      nestedKeyframes.set(nestedClipId, updatedKeyframes);
      log.info('Loaded keyframes for nested clip', {
        nestedClipId,
        originalClipId: serializedClip.id,
        keyframeCount: updatedKeyframes.length,
        properties: [...new Set(updatedKeyframes.map((k: Keyframe) => k.property))],
      });
    }

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

  // Merge nested clip keyframes into the store's clipKeyframes Map
  if (nestedKeyframes.size > 0) {
    const currentKeyframes = get().clipKeyframes;
    const mergedKeyframes = new Map(currentKeyframes);
    nestedKeyframes.forEach((keyframes, clipId) => {
      mergedKeyframes.set(clipId, keyframes);
    });
    set({ clipKeyframes: mergedKeyframes });
    log.info('Merged nested clip keyframes into store', {
      compClipId,
      nestedKeyframeClipCount: nestedKeyframes.size,
      totalKeyframeClipCount: mergedKeyframes.size,
    });
  }

  return nestedClips;
}

function loadVideoNestedClip(
  compClipId: string,
  nestedClipId: string,
  fileUrl: string,
  fileName: string,
  get: CompClipStoreGet,
  set: CompClipStoreSet
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
  get: CompClipStoreGet,
  set: CompClipStoreSet
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
  get: CompClipStoreGet,
  set: CompClipStoreSet
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
  /** Pre-calculated boundary positions (normalized 0-1) for segment-based thumbnails */
  boundaries?: number[];
  get: CompClipStoreGet;
  set: CompClipStoreSet;
}

/**
 * Generate thumbnails for nested composition using WebGPU rendering.
 * Shows all layers with effects, not just the first video.
 * Uses segment boundaries to ensure each clip section gets a representative thumbnail.
 * Falls back to polling first video if WebGPU fails.
 */
export async function generateCompThumbnails(params: GenerateCompThumbnailsParams): Promise<void> {
  const { clipId, compDuration, thumbnailsEnabled, boundaries, get, set } = params;

  if (!thumbnailsEnabled) return;

  // Get the composition ID from the clip
  const compClip = get().clips.find((c: TimelineClip) => c.id === clipId);
  if (!compClip?.compositionId) {
    log.warn('No composition ID for comp clip', { clipId });
    return;
  }

  // NOTE: We don't invalidate here anymore - invalidation is already done by
  // compositionRenderer.invalidateCompositionAndParents() when switching compositions.
  // Invalidating here caused race conditions when multiple comp clips reference the same composition.

  // Try WebGPU-based rendering first (shows all layers with effects)
  try {
    log.info('Generating WebGPU thumbnails for nested comp', {
      clipId,
      compositionId: compClip.compositionId,
      compDuration,
      boundaryCount: boundaries?.length ?? 0,
      boundaries: boundaries?.map(b => (b * 100).toFixed(1) + '%'),
    });

    const thumbnails = await thumbnailRenderer.generateCompositionThumbnails(
      compClip.compositionId,
      compDuration,
      { count: 10, width: 160, height: 90, boundaries }
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
    log.error('WebGPU thumbnail generation failed, falling back to video-based', e);
  }

  // Fallback: Use the first video's frames (original method)
  log.warn('Using FALLBACK thumbnail generation (first video only)');
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
  set: CompClipStoreSet;
  get: CompClipStoreGet;
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
