// Thumbnail creation and deduplication

import { THUMBNAIL_TIMEOUT } from '../constants';
import { projectFileService } from '../../../services/projectFileService';
import { Logger } from '../../../services/logger';

const log = Logger.create('Thumbnail');

/**
 * Create thumbnail for video or image.
 */
export async function createThumbnail(
  file: File,
  type: 'video' | 'image'
): Promise<string | undefined> {
  return new Promise((resolve) => {
    if (type === 'image') {
      resolve(URL.createObjectURL(file));
      return;
    }

    if (type === 'video') {
      const video = document.createElement('video');
      const url = URL.createObjectURL(file);
      video.src = url;
      video.muted = true;
      video.playsInline = true;

      const timeout = setTimeout(() => {
        log.warn('Timeout:', file.name);
        URL.revokeObjectURL(url);
        resolve(undefined);
      }, THUMBNAIL_TIMEOUT);

      const cleanup = () => {
        clearTimeout(timeout);
        URL.revokeObjectURL(url);
      };

      video.onloadedmetadata = () => {
        video.currentTime = Math.min(1, video.duration * 0.1);
      };

      video.onseeked = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 90;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          resolve(canvas.toDataURL('image/jpeg', 0.7));
        } else {
          resolve(undefined);
        }
        cleanup();
      };

      video.onerror = () => {
        cleanup();
        resolve(undefined);
      };

      video.load();
    } else {
      resolve(undefined);
    }
  });
}

/**
 * Handle thumbnail deduplication - check for existing, save new.
 * UNIFIED: Replaces 3 duplicate blocks in original code.
 */
export async function handleThumbnailDedup(
  fileHash: string | undefined,
  thumbnailUrl: string | undefined
): Promise<string | undefined> {
  if (!fileHash || !projectFileService.isProjectOpen()) {
    return thumbnailUrl;
  }

  try {
    // Check for existing thumbnail
    const existingBlob = await projectFileService.getThumbnail(fileHash);
    if (existingBlob && existingBlob.size > 0) {
      log.debug('Reusing existing for hash:', fileHash.slice(0, 8));
      return URL.createObjectURL(existingBlob);
    }

    // Save new thumbnail
    if (thumbnailUrl) {
      const blob = await fetchThumbnailBlob(thumbnailUrl);
      if (blob && blob.size > 0) {
        await projectFileService.saveThumbnail(fileHash, blob);
        log.debug('Saved to project folder:', fileHash.slice(0, 8));
      }
    }
  } catch (e) {
    log.warn('Dedup error:', e);
  }

  return thumbnailUrl;
}

/**
 * Fetch thumbnail blob from data URL or blob URL.
 */
async function fetchThumbnailBlob(url: string): Promise<Blob | null> {
  if (url.startsWith('data:') || url.startsWith('blob:')) {
    const response = await fetch(url);
    return response.blob();
  }
  return null;
}
