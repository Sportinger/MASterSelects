// File type detection utilities for timeline drag & drop

import { DURATION_CHECK_TIMEOUT } from '../constants';

const VIDEO_EXTENSIONS = [
  'mov', 'mp4', 'm4v', 'mxf', 'avi', 'mkv', 'webm',  // Common
  'ts', 'mts', 'm2ts',                               // Transport streams
  'wmv', 'asf', 'flv', 'f4v',                        // Windows/Flash
  '3gp', '3g2', 'ogv', 'vob', 'mpg', 'mpeg',         // Other
];

const AUDIO_EXTENSIONS = [
  'mp3', 'wav', 'ogg', 'aac', 'm4a', 'flac', 'wma', 'aiff', 'alac', 'opus',
];

const IMAGE_EXTENSIONS = [
  'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff', 'tif', 'heic', 'heif',
];

/**
 * Check if file is a video by MIME type or extension
 */
export function isVideoFile(file: File): boolean {
  if (file.type.startsWith('video/')) return true;
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  return VIDEO_EXTENSIONS.includes(ext);
}

/**
 * Check if file is an audio file by MIME type or extension
 */
export function isAudioFile(file: File): boolean {
  if (file.type.startsWith('audio/')) return true;
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  return AUDIO_EXTENSIONS.includes(ext);
}

/**
 * Check if file is any media type (video/audio/image)
 */
export function isMediaFile(file: File): boolean {
  if (
    file.type.startsWith('video/') ||
    file.type.startsWith('audio/') ||
    file.type.startsWith('image/')
  ) {
    return true;
  }
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  return (
    VIDEO_EXTENSIONS.includes(ext) ||
    AUDIO_EXTENSIONS.includes(ext) ||
    IMAGE_EXTENSIONS.includes(ext)
  );
}

/**
 * Video metadata result including duration and audio presence
 */
export interface VideoMetadata {
  duration: number | null;
  hasAudio: boolean;
}

/**
 * Quick metadata check for dragged video files
 * Returns duration and whether the video has audio tracks
 */
export async function getVideoMetadataQuick(
  file: File,
  timeoutMs = DURATION_CHECK_TIMEOUT
): Promise<VideoMetadata | null> {
  if (!isVideoFile(file)) return null;

  // Get duration from video element
  const durationResult = await new Promise<{ duration: number | null; blobUrl: string } | null>((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    const blobUrl = URL.createObjectURL(file);

    const cleanup = () => {
      video.remove();
    };

    const timeoutId = setTimeout(() => {
      cleanup();
      URL.revokeObjectURL(blobUrl);
      resolve(null);
    }, timeoutMs);

    video.onloadedmetadata = () => {
      clearTimeout(timeoutId);
      const dur = video.duration;
      cleanup();
      resolve({
        duration: isFinite(dur) ? dur : null,
        blobUrl, // Keep blob URL for audio check
      });
    };

    video.onerror = () => {
      clearTimeout(timeoutId);
      cleanup();
      URL.revokeObjectURL(blobUrl);
      resolve(null);
    };

    video.src = blobUrl;
  });

  if (!durationResult) return null;

  // Check for audio using Web Audio API (quick check with small sample)
  const hasAudio = await checkHasAudioQuick(file);

  URL.revokeObjectURL(durationResult.blobUrl);

  return {
    duration: durationResult.duration,
    hasAudio,
  };
}

/**
 * Quick check if file has audio using Web Audio API
 * Uses a small sample to keep it fast for drag preview
 */
async function checkHasAudioQuick(file: File): Promise<boolean> {
  try {
    const audioContext = new AudioContext();

    // Read first 512KB - enough to detect audio presence
    const maxBytes = 512 * 1024;
    const blob = file.slice(0, Math.min(file.size, maxBytes));
    const arrayBuffer = await blob.arrayBuffer();

    try {
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const hasAudio = audioBuffer.numberOfChannels > 0 && audioBuffer.length > 0;
      await audioContext.close();
      return hasAudio;
    } catch {
      // decodeAudioData throws if there's no audio
      await audioContext.close();
      return false;
    }
  } catch {
    // Default to true on error (will be checked again during import)
    return true;
  }
}

/**
 * Quick duration check for dragged video files
 * Returns null if not a video or duration cannot be determined
 */
export async function getVideoDurationQuick(
  file: File,
  timeoutMs = DURATION_CHECK_TIMEOUT
): Promise<number | null> {
  const metadata = await getVideoMetadataQuick(file, timeoutMs);
  return metadata?.duration ?? null;
}
