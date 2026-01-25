// Media info extraction helpers

import { CONTAINER_MAP, MEDIA_INFO_TIMEOUT } from '../constants';
import { Logger } from '../../../services/logger';

const log = Logger.create('MediaInfo');

export interface MediaInfo {
  width?: number;
  height?: number;
  duration?: number;
  fps?: number;
  codec?: string;
  container?: string;
  fileSize?: number;
}

/**
 * Get container format from file extension.
 */
export function getContainerFormat(fileName: string): string {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  return CONTAINER_MAP[ext] || ext.toUpperCase();
}

/**
 * Parse FPS from filename (patterns like "25fps", "_30p", etc.).
 */
export function parseFpsFromFilename(fileName: string): number | undefined {
  const patterns = [
    /[_\-\s(](\d{2}(?:\.\d+)?)\s*fps/i,
    /[_\-\s(](\d{2}(?:\.\d+)?)\s*p[_\-\s).]/i,
    /(\d{2}(?:\.\d+)?)fps/i,
  ];

  for (const pattern of patterns) {
    const match = fileName.match(pattern);
    if (match) {
      const fps = parseFloat(match[1]);
      if (fps >= 10 && fps <= 240) return fps;
    }
  }
  return undefined;
}

/**
 * Get codec info from file (best effort).
 */
export async function getCodecInfo(file: File): Promise<string | undefined> {
  try {
    const ext = file.name.split('.').pop()?.toLowerCase();

    // Video codecs
    if (ext === 'mp4' || ext === 'm4v' || ext === 'mov') return 'H.264';
    if (ext === 'webm') return 'VP9';
    if (ext === 'mkv') return 'H.264';

    // Audio codecs
    if (ext === 'mp3') return 'MP3';
    if (ext === 'aac' || ext === 'm4a') return 'AAC';
    if (ext === 'wav') return 'PCM';
    if (ext === 'ogg') return 'Vorbis';
    if (ext === 'flac') return 'FLAC';
  } catch {
    // Ignore
  }
  return undefined;
}

/**
 * Get media dimensions, duration, and metadata.
 */
export async function getMediaInfo(
  file: File,
  type: 'video' | 'audio' | 'image'
): Promise<MediaInfo> {
  const container = getContainerFormat(file.name);
  const fileSize = file.size;
  const codec = await getCodecInfo(file);

  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      console.warn('[MediaInfo] Timeout:', file.name);
      resolve({ container, fileSize, codec });
    }, MEDIA_INFO_TIMEOUT);

    const cleanup = (url?: string) => {
      clearTimeout(timeout);
      if (url) URL.revokeObjectURL(url);
    };

    if (type === 'image') {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.src = url;
      img.onload = () => {
        resolve({ width: img.width, height: img.height, container, fileSize, codec });
        cleanup(url);
      };
      img.onerror = () => {
        resolve({ container, fileSize });
        cleanup(url);
      };
    } else if (type === 'video') {
      const video = document.createElement('video');
      const url = URL.createObjectURL(file);
      video.src = url;
      video.muted = true;
      video.playsInline = true;

      video.onloadedmetadata = () => {
        resolve({
          width: video.videoWidth,
          height: video.videoHeight,
          duration: video.duration,
          fps: parseFpsFromFilename(file.name),
          codec,
          container,
          fileSize,
        });
        cleanup(url);
      };
      video.onerror = () => {
        resolve({ container, fileSize, codec });
        cleanup(url);
      };
      video.load();
    } else if (type === 'audio') {
      const audio = document.createElement('audio');
      const url = URL.createObjectURL(file);
      audio.src = url;
      audio.onloadedmetadata = () => {
        resolve({ duration: audio.duration, codec, container, fileSize });
        cleanup(url);
      };
      audio.onerror = () => {
        resolve({ container, fileSize });
        cleanup(url);
      };
    } else {
      cleanup();
      resolve({ container, fileSize });
    }
  });
}
