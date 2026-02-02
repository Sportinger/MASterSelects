// Thumbnail generation helper - eliminates 3x duplication in clip loading
// Handles video and image thumbnail generation

import { seekVideo } from '../utils';
import { Logger } from '../../../services/logger';

const log = Logger.create('ThumbnailHelpers');

export interface ThumbnailOptions {
  maxCount?: number;
  width?: number;
  height?: number;
  quality?: number;
  intervalSeconds?: number;
}

/**
 * Generate thumbnails from a video element.
 * Used for video clips and composition clips.
 * Now covers 0% to 100% of duration.
 */
export async function generateVideoThumbnails(
  video: HTMLVideoElement,
  duration: number,
  options: ThumbnailOptions = {}
): Promise<string[]> {
  const {
    maxCount = 10,
    width,
    height = 40,
    quality = 0.6,
    intervalSeconds = 30,
  } = options;

  const thumbnails: string[] = [];
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    log.warn('Could not get canvas context');
    return thumbnails;
  }

  // Calculate dimensions preserving aspect ratio
  const thumbHeight = height;
  const thumbWidth = width || Math.round((video.videoWidth / video.videoHeight) * thumbHeight);
  canvas.width = thumbWidth;
  canvas.height = thumbHeight;

  // Calculate number of thumbnails (more for longer videos, up to maxCount)
  const thumbCount = Math.max(1, Math.min(maxCount, Math.ceil(duration / intervalSeconds)));

  // Generate from 0% to 100% of duration
  for (let i = 0; i < thumbCount; i++) {
    // Use (thumbCount - 1) as divisor so last thumbnail is at 100%
    const time = thumbCount > 1 ? (i / (thumbCount - 1)) * duration : 0;
    // Clamp to slightly before end to avoid seek issues
    const clampedTime = Math.min(time, duration - 0.01);
    try {
      await seekVideo(video, clampedTime);
      ctx.drawImage(video, 0, 0, thumbWidth, thumbHeight);
      thumbnails.push(canvas.toDataURL('image/jpeg', quality));
    } catch (e) {
      log.warn('Thumbnail failed at time', { time: clampedTime, error: e });
    }
  }

  return thumbnails;
}

/**
 * Generate a single thumbnail from an image element.
 */
export function generateImageThumbnail(
  img: HTMLImageElement,
  options: { height?: number; quality?: number } = {}
): string | null {
  const { height = 40, quality = 0.6 } = options;

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    return null;
  }

  const thumbWidth = Math.round((img.width / img.height) * height);
  canvas.width = thumbWidth;
  canvas.height = height;
  ctx.drawImage(img, 0, 0, thumbWidth, height);

  return canvas.toDataURL('image/jpeg', quality);
}

/**
 * Generate thumbnails for YouTube/downloaded video with specific sizing.
 * Now covers 0% to 100% of duration.
 */
export async function generateDownloadThumbnails(
  video: HTMLVideoElement,
  duration: number
): Promise<string[]> {
  const thumbCount = Math.max(1, Math.min(10, Math.ceil(duration / 30)));
  const thumbnails: string[] = [];
  const canvas = document.createElement('canvas');
  canvas.width = 160;
  canvas.height = 90;
  const ctx = canvas.getContext('2d');

  if (!ctx) return thumbnails;

  for (let i = 0; i < thumbCount; i++) {
    // Use (thumbCount - 1) as divisor so last thumbnail is at 100%
    const time = thumbCount > 1 ? (i / (thumbCount - 1)) * duration : 0;
    // Clamp to slightly before end to avoid seek issues
    video.currentTime = Math.min(time, duration - 0.01);
    await new Promise<void>(resolve => {
      video.onseeked = () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        thumbnails.push(canvas.toDataURL('image/jpeg', 0.6));
        resolve();
      };
    });
  }

  return thumbnails;
}
