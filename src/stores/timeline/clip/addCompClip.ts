// Composition clip addition - extracted from addCompClip
// Handles nested composition loading, audio mixdown, and linked audio creation

import type { TimelineClip, TimelineTrack } from '../../../types';
import type { Composition } from '../types';
import { DEFAULT_TRANSFORM } from '../constants';
import { generateThumbnails } from '../utils';
import { useMediaStore } from '../../mediaStore';
import { initWebCodecsPlayer } from '../helpers/webCodecsHelpers';
import { findOrCreateAudioTrack, createCompositionAudioClip } from '../helpers/audioTrackHelpers';
import { generateSilentWaveform } from '../helpers/waveformHelpers';
import { generateCompClipId, generateClipId, generateNestedClipId } from '../helpers/idGenerator';
import { blobUrlManager } from '../helpers/blobUrlManager';
import { updateClipById } from '../helpers/clipStateHelpers';

export interface AddCompClipParams {
  trackId: string;
  composition: Composition;
  startTime: number;
  findNonOverlappingPosition: (clipId: string, startTime: number, trackId: string, duration: number) => number;
}

/**
 * Create placeholder composition clip immediately.
 */
export function createCompClipPlaceholder(params: AddCompClipParams): TimelineClip {
  const { trackId, composition, startTime, findNonOverlappingPosition } = params;

  const clipId = generateCompClipId();
  const compDuration = composition.timelineData?.duration ?? composition.duration;
  const finalStartTime = findNonOverlappingPosition(clipId, startTime, trackId, compDuration);

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
  };
}

export interface LoadNestedClipsParams {
  compClipId: string;
  composition: Composition;
  get: () => any;
  set: (state: any) => void;
}

/**
 * Load nested clips from composition's timeline data.
 */
export async function loadNestedClips(params: LoadNestedClipsParams): Promise<TimelineClip[]> {
  const { compClipId, composition, get, set } = params;

  if (!composition.timelineData) return [];

  const mediaStore = useMediaStore.getState();
  const nestedClips: TimelineClip[] = [];

  for (const serializedClip of composition.timelineData.clips) {
    const mediaFile = mediaStore.files.find(f => f.id === serializedClip.mediaFileId);
    if (!mediaFile?.file) {
      console.warn('[Nested Comp] Could not find media file:', serializedClip.name);
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
      loadVideoNestedClip(nestedClip, fileUrl, mediaFile.file.name, get, set);
    } else if (type === 'audio') {
      loadAudioNestedClip(nestedClip, fileUrl);
    } else if (type === 'image') {
      loadImageNestedClip(nestedClip, fileUrl, get, set);
    }
  }

  return nestedClips;
}

function loadVideoNestedClip(
  nestedClip: TimelineClip,
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

  video.addEventListener('canplaythrough', async () => {
    nestedClip.source = {
      type: 'video',
      videoElement: video,
      naturalDuration: video.duration,
    };
    nestedClip.isLoading = false;

    // Initialize WebCodecsPlayer
    const webCodecsPlayer = await initWebCodecsPlayer(video, fileName);
    if (webCodecsPlayer) {
      nestedClip.source = { ...nestedClip.source, webCodecsPlayer };
    }

    // Trigger re-render
    set({ clips: [...get().clips] });
  }, { once: true });
}

function loadAudioNestedClip(nestedClip: TimelineClip, fileUrl: string): void {
  const audio = document.createElement('audio');
  audio.src = fileUrl;
  audio.preload = 'auto';

  audio.addEventListener('canplaythrough', () => {
    nestedClip.source = {
      type: 'audio',
      audioElement: audio,
      naturalDuration: audio.duration,
    };
    nestedClip.isLoading = false;
  }, { once: true });
}

function loadImageNestedClip(
  nestedClip: TimelineClip,
  fileUrl: string,
  get: () => any,
  set: (state: any) => void
): void {
  const img = new Image();
  img.src = fileUrl;

  img.addEventListener('load', () => {
    nestedClip.source = { type: 'image', imageElement: img };
    nestedClip.isLoading = false;
    set({ clips: [...get().clips] });
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
 * Generate thumbnails from first video in nested composition.
 */
export function generateCompThumbnails(params: GenerateCompThumbnailsParams): void {
  const { clipId, nestedClips, compDuration, thumbnailsEnabled, get, set } = params;

  if (!thumbnailsEnabled) return;

  const firstVideoClip = nestedClips.find(c => c.file.type.startsWith('video/'));
  if (!firstVideoClip) return;

  // Wait for video to load using event listener instead of arbitrary timeout
  const checkAndGenerate = async () => {
    if (!get().thumbnailsEnabled) return;
    const video = firstVideoClip.source?.videoElement;
    if (video && video.readyState >= 2) {
      try {
        const thumbnails = await generateThumbnails(video, compDuration);
        set({ clips: updateClipById(get().clips, clipId, { thumbnails }) });
      } catch (e) {
        console.warn('[Nested Comp] Failed to generate thumbnails:', e);
      }
    }
  };

  // Try immediately if ready, otherwise wait for canplay event
  const video = firstVideoClip.source?.videoElement;
  if (video) {
    if (video.readyState >= 2) {
      checkAndGenerate();
    } else {
      video.addEventListener('canplay', checkAndGenerate, { once: true });
      // Fallback timeout in case event never fires
      setTimeout(checkAndGenerate, 2000);
    }
  }
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
      console.log(`[Nested Comp] Generating audio mixdown for ${composition.name}...`);
      const mixdownResult = await compositionAudioMixer.mixdownComposition(composition.id);

      if (mixdownResult?.hasAudio) {
        hasAudio = true;
        mixdownAudio = compositionAudioMixer.createAudioElement(mixdownResult.buffer);
        mixdownAudio.preload = 'auto';
        waveform = mixdownResult.waveform;
        mixdownBuffer = mixdownResult.buffer;
      }
    } catch (e) {
      console.error('[Nested Comp] Failed to generate audio mixdown:', e);
    }
  }

  // Find or create audio track
  const { trackId: audioTrackId, newTrack } = findOrCreateAudioTrack(tracks);
  if (newTrack) {
    set({ tracks: [...get().tracks, newTrack] });
    console.log(`[Nested Comp] Created new audio track for ${composition.name}`);
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

  console.log(`[Nested Comp] Created linked audio clip for ${composition.name} (hasAudio: ${hasAudio})`);
}
