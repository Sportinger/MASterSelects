// File loading utilities for WebVJ Mixer

import { Logger } from '../services/logger';

const log = Logger.create('FileLoader');

export interface FilePickerOptions {
  accept?: string[];
  multiple?: boolean;
}

// Check if File System Access API is supported
export function isFileSystemAccessSupported(): boolean {
  return 'showOpenFilePicker' in window;
}

// Open file picker using File System Access API (Chrome/Edge)
export async function openFilePicker(
  options: FilePickerOptions = {}
): Promise<File[]> {
  const { accept = ['video/*', 'image/*'], multiple = false } = options;

  if (isFileSystemAccessSupported()) {
    try {
      const fileTypes = accept.map((type) => {
        if (type === 'video/*') {
          return {
            description: 'Video files',
            accept: {
              'video/*': ['.mp4', '.webm', '.mov', '.avi', '.mkv'],
            },
          };
        }
        if (type === 'image/*') {
          return {
            description: 'Image files',
            accept: {
              'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
            },
          };
        }
        return {
          description: 'Files',
          accept: { [type]: [] },
        };
      });

      const handles = await (window as unknown as { showOpenFilePicker: (opts: object) => Promise<FileSystemFileHandle[]> }).showOpenFilePicker({
        multiple,
        types: fileTypes,
      });

      const files: File[] = [];
      for (const handle of handles) {
        const file = await handle.getFile();
        files.push(file);
      }
      return files;
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        return [];
      }
      throw error;
    }
  }

  // Fallback to input element for Firefox/Safari
  return openFilePickerFallback(options);
}

// Fallback file picker using input element
function openFilePickerFallback(options: FilePickerOptions): Promise<File[]> {
  const { accept = ['video/*', 'image/*'], multiple = false } = options;

  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = accept.join(',');
    input.multiple = multiple;

    input.onchange = () => {
      const files = input.files ? Array.from(input.files) : [];
      resolve(files);
    };

    input.oncancel = () => {
      resolve([]);
    };

    input.click();
  });
}

// Open directory picker (for media libraries)
export async function openDirectoryPicker(): Promise<FileSystemDirectoryHandle | null> {
  if (!('showDirectoryPicker' in window)) {
    log.warn('Directory picker not supported');
    return null;
  }

  try {
    return await (window as unknown as { showDirectoryPicker: () => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker();
  } catch (error) {
    if ((error as Error).name === 'AbortError') {
      return null;
    }
    throw error;
  }
}

// Get all media files from a directory
export async function getMediaFilesFromDirectory(
  dirHandle: FileSystemDirectoryHandle
): Promise<File[]> {
  const files: File[] = [];
  const mediaExtensions = ['.mp4', '.webm', '.mov', '.png', '.jpg', '.jpeg', '.gif', '.webp'];

  async function scanDirectory(handle: FileSystemDirectoryHandle) {
    // Use entries() which is more widely typed
    const entries = (handle as unknown as AsyncIterable<FileSystemHandle>);
    for await (const entry of entries) {
      if (entry.kind === 'file') {
        const fileHandle = entry as FileSystemFileHandle;
        const ext = entry.name.toLowerCase().substring(entry.name.lastIndexOf('.'));
        if (mediaExtensions.includes(ext)) {
          const file = await fileHandle.getFile();
          files.push(file);
        }
      } else if (entry.kind === 'directory') {
        await scanDirectory(entry as FileSystemDirectoryHandle);
      }
    }
  }

  await scanDirectory(dirHandle);
  return files;
}

// Create video element from file
export function createVideoFromFile(file: File): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.src = URL.createObjectURL(file);
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';

    video.onloadeddata = () => resolve(video);
    video.onerror = () => reject(new Error(`Failed to load video: ${file.name}`));
  });
}

// Create image element from file
export function createImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.crossOrigin = 'anonymous';

    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
  });
}

// Generate video thumbnail
export async function generateVideoThumbnail(
  video: HTMLVideoElement,
  time = 0
): Promise<string> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 90;

    video.currentTime = time;

    video.onseeked = () => {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      resolve(canvas.toDataURL('image/jpeg', 0.7));
    };
  });
}
