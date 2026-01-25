// Audio clip addition - extracted from addClip
// Handles audio file loading and waveform generation

import type { TimelineClip } from '../../../types';
import { DEFAULT_TRANSFORM } from '../constants';
import { useMediaStore } from '../../mediaStore';
import { createAudioElement } from '../helpers/webCodecsHelpers';
import { generateWaveformForFile, AUDIO_WAVEFORM_THRESHOLD } from '../helpers/waveformHelpers';
import { generateClipId } from '../helpers/idGenerator';
import { blobUrlManager } from '../helpers/blobUrlManager';

export interface AddAudioClipParams {
  trackId: string;
  file: File;
  startTime: number;
  estimatedDuration: number;
  mediaFileId?: string;
}

/**
 * Create placeholder audio clip immediately.
 * Returns clip ready to be added to state while media loads in background.
 */
export function createAudioClipPlaceholder(params: AddAudioClipParams): TimelineClip {
  const { trackId, file, startTime, estimatedDuration, mediaFileId } = params;
  const clipId = generateClipId('clip-audio');

  return {
    id: clipId,
    trackId,
    name: file.name,
    file,
    startTime,
    duration: estimatedDuration,
    inPoint: 0,
    outPoint: estimatedDuration,
    source: { type: 'audio', naturalDuration: estimatedDuration, mediaFileId },
    transform: { ...DEFAULT_TRANSFORM },
    effects: [],
    isLoading: true,
  };
}

export interface LoadAudioMediaParams {
  clip: TimelineClip;
  file: File;
  mediaFileId?: string;
  waveformsEnabled: boolean;
  updateClip: (id: string, updates: Partial<TimelineClip>) => void;
}

/**
 * Load audio media in background - handles metadata and waveform generation.
 */
export async function loadAudioMedia(params: LoadAudioMediaParams): Promise<void> {
  const { clip, file, mediaFileId, waveformsEnabled, updateClip } = params;

  // Create and load audio element
  const audio = createAudioElement(file);
  // Track the blob URL for cleanup
  blobUrlManager.create(clip.id, file, 'audio');

  // Wait for metadata
  await new Promise<void>((resolve) => {
    audio.onloadedmetadata = () => resolve();
    audio.onerror = () => resolve();
  });

  const naturalDuration = audio.duration || clip.duration;

  // Check if this is a large file (audio-only has higher threshold)
  const isLargeFile = file.size > AUDIO_WAVEFORM_THRESHOLD;

  // Mark clip as ready first (waveform will load in background)
  updateClip(clip.id, {
    duration: naturalDuration,
    outPoint: naturalDuration,
    source: { type: 'audio', audioElement: audio, naturalDuration, mediaFileId },
    isLoading: false,
    waveformGenerating: waveformsEnabled && !isLargeFile,
    waveformProgress: 0,
  });

  // Generate waveform in background - only if enabled and not very large
  if (isLargeFile) {
    console.log(`[Waveform] Skipping for very large file (${(file.size / 1024 / 1024).toFixed(0)}MB): ${file.name}`);
  }

  if (waveformsEnabled && !isLargeFile) {
    // Run waveform generation async (don't await)
    generateWaveformAsync(clip.id, file, updateClip);
  }

  // Sync to media store
  const mediaStore = useMediaStore.getState();
  if (!mediaStore.getFileByName(file.name)) {
    mediaStore.importFile(file);
  }
}

/**
 * Generate waveform asynchronously without blocking.
 */
async function generateWaveformAsync(
  clipId: string,
  file: File,
  updateClip: (id: string, updates: Partial<TimelineClip>) => void
): Promise<void> {
  try {
    console.log(`[Waveform] Starting generation for ${file.name}...`);
    const waveform = await generateWaveformForFile(file);
    console.log(`[Waveform] Complete: ${waveform.length} samples for ${file.name}`);

    updateClip(clipId, {
      waveform,
      waveformGenerating: false,
      waveformProgress: 100,
    });
  } catch (e) {
    console.warn('[Waveform] Failed:', e);
    updateClip(clipId, { waveformGenerating: false });
  }
}
