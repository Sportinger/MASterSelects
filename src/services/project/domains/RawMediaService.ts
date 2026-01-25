// Raw folder operations and media import service

import { Logger } from '../../logger';
import { PROJECT_FOLDERS } from '../core/constants';

const log = Logger.create('RawMedia');
import { FileStorageService } from '../core/FileStorageService';
import type { ProjectMediaFile } from '../types';

export class RawMediaService {
  private fileStorage: FileStorageService;

  constructor(fileStorage: FileStorageService) {
    this.fileStorage = fileStorage;
  }

  // ============================================
  // RAW FOLDER OPERATIONS
  // ============================================

  /**
   * Copy a file to the Raw/ folder in the project
   * Returns the file handle and relative path if successful
   * If file with same name and size already exists, returns existing file instead of copying
   */
  async copyToRawFolder(
    projectHandle: FileSystemDirectoryHandle,
    file: File,
    fileName?: string
  ): Promise<{ handle: FileSystemFileHandle; relativePath: string; alreadyExisted: boolean } | null> {
    try {
      // Get or create Raw folder
      const rawFolder = await projectHandle.getDirectoryHandle(PROJECT_FOLDERS.RAW, { create: true });

      // Use provided fileName or original file name
      const targetName = fileName || file.name;

      // Check if file already exists with same name and size
      try {
        const existingHandle = await rawFolder.getFileHandle(targetName, { create: false });
        const existingFile = await existingHandle.getFile();

        if (existingFile.size === file.size) {
          // File with same name and size already exists - reuse it
          const relativePath = `${PROJECT_FOLDERS.RAW}/${targetName}`;
          console.log(`[RawMedia] File already exists in Raw folder with same size: ${relativePath}`);
          return { handle: existingHandle, relativePath, alreadyExisted: true };
        }
      } catch {
        // File doesn't exist, will create new one
      }

      // Check if file already exists - if so, add suffix
      let finalName = targetName;
      let counter = 1;
      while (true) {
        try {
          await rawFolder.getFileHandle(finalName, { create: false });
          // File exists (but different size), try with suffix
          const ext = targetName.lastIndexOf('.');
          if (ext > 0) {
            finalName = `${targetName.slice(0, ext)}_${counter}${targetName.slice(ext)}`;
          } else {
            finalName = `${targetName}_${counter}`;
          }
          counter++;
        } catch {
          // File doesn't exist, we can use this name
          break;
        }
      }

      // Create and write the file
      const fileHandle = await rawFolder.getFileHandle(finalName, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(file);
      await writable.close();

      const relativePath = `${PROJECT_FOLDERS.RAW}/${finalName}`;
      console.log(`[RawMedia] Copied ${file.name} to ${relativePath} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);

      return { handle: fileHandle, relativePath, alreadyExisted: false };
    } catch (e) {
      console.error('[RawMedia] Failed to copy file to Raw folder:', e);
      return null;
    }
  }

  /**
   * Get a file from the Raw/ folder by relative path
   */
  async getFileFromRaw(
    projectHandle: FileSystemDirectoryHandle,
    relativePath: string
  ): Promise<{ file: File; handle: FileSystemFileHandle } | null> {
    try {
      // Parse the relative path (e.g., "Raw/video.mp4")
      const parts = relativePath.split('/');
      if (parts[0] !== PROJECT_FOLDERS.RAW || parts.length !== 2) {
        return null;
      }

      const rawFolder = await projectHandle.getDirectoryHandle(PROJECT_FOLDERS.RAW);
      const fileHandle = await rawFolder.getFileHandle(parts[1]);
      const file = await fileHandle.getFile();

      return { file, handle: fileHandle };
    } catch (e) {
      return null;
    }
  }

  /**
   * Check if a file exists in the Raw/ folder by name
   */
  async hasFileInRaw(
    projectHandle: FileSystemDirectoryHandle,
    fileName: string
  ): Promise<boolean> {
    try {
      const rawFolder = await projectHandle.getDirectoryHandle(PROJECT_FOLDERS.RAW);
      await rawFolder.getFileHandle(fileName, { create: false });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Scan the Raw/ folder for files matching missing file names
   * Returns a map of lowercase filename -> file handle
   */
  async scanRawFolder(
    projectHandle: FileSystemDirectoryHandle
  ): Promise<Map<string, FileSystemFileHandle>> {
    const foundFiles = new Map<string, FileSystemFileHandle>();

    try {
      const rawFolder = await projectHandle.getDirectoryHandle(PROJECT_FOLDERS.RAW);

      for await (const entry of (rawFolder as any).values()) {
        if (entry.kind === 'file') {
          foundFiles.set(entry.name.toLowerCase(), entry);
        }
      }
    } catch {
      // Raw folder doesn't exist or can't be read
    }

    return foundFiles;
  }

  // ============================================
  // MEDIA IMPORT
  // ============================================

  /**
   * Import media file (creates reference, doesn't copy)
   */
  async importMediaFile(
    file: File,
    fileHandle?: FileSystemFileHandle
  ): Promise<ProjectMediaFile | null> {
    const id = `media-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Determine file type
    let type: 'video' | 'audio' | 'image' = 'video';
    if (file.type.startsWith('audio/')) type = 'audio';
    else if (file.type.startsWith('image/')) type = 'image';

    // Get source path (if available from File System Access API)
    let sourcePath = file.name;
    if (fileHandle) {
      sourcePath = fileHandle.name;
    }

    const mediaFile: ProjectMediaFile = {
      id,
      name: file.name,
      type,
      sourcePath,
      hasProxy: false,
      folderId: null,
      importedAt: new Date().toISOString(),
    };

    // Get metadata (will be filled in async)
    if (type === 'video' || type === 'audio') {
      // Create temp URL to get duration
      const url = URL.createObjectURL(file);
      const media = type === 'video' ? document.createElement('video') : document.createElement('audio');

      await new Promise<void>((resolve) => {
        media.onloadedmetadata = () => {
          mediaFile.duration = media.duration;
          if (type === 'video' && media instanceof HTMLVideoElement) {
            mediaFile.width = media.videoWidth;
            mediaFile.height = media.videoHeight;
          }
          URL.revokeObjectURL(url);
          resolve();
        };
        media.onerror = () => {
          URL.revokeObjectURL(url);
          resolve();
        };
        media.src = url;
      });
    } else if (type === 'image') {
      const url = URL.createObjectURL(file);
      const img = new Image();

      await new Promise<void>((resolve) => {
        img.onload = () => {
          mediaFile.width = img.naturalWidth;
          mediaFile.height = img.naturalHeight;
          URL.revokeObjectURL(url);
          resolve();
        };
        img.onerror = () => {
          URL.revokeObjectURL(url);
          resolve();
        };
        img.src = url;
      });
    }

    return mediaFile;
  }

  // ============================================
  // YOUTUBE DOWNLOADS
  // ============================================

  /**
   * Save a YouTube download to the project's YT folder
   * Returns the File object with correct name for timeline use
   */
  async saveYouTubeDownload(
    projectHandle: FileSystemDirectoryHandle,
    blob: Blob,
    title: string
  ): Promise<File | null> {
    try {
      // Sanitize filename
      const sanitizedTitle = title.replace(/[^a-zA-Z0-9\s\-_]/g, '').substring(0, 100).trim();
      const fileName = `${sanitizedTitle}.mp4`;

      // Write to YT folder
      const success = await this.fileStorage.writeFile(projectHandle, 'YT', fileName, blob);
      if (!success) {
        console.error('[RawMedia] Failed to write YouTube file');
        return null;
      }

      // Return as File object
      const file = new File([blob], fileName, { type: 'video/mp4' });
      console.log(`[RawMedia] Saved YouTube download: YT/${fileName}`);
      return file;
    } catch (e) {
      console.error('[RawMedia] Failed to save YouTube download:', e);
      return null;
    }
  }
}
