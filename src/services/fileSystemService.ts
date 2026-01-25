// File System Access API Service
// Provides access to actual file paths and persistent storage locations

import { Logger } from './logger';
import { projectDB } from './projectDB';

const log = Logger.create('FileSystemService');

// Types for File System Access API
interface FileSystemHandlePermissionDescriptor {
  mode?: 'read' | 'readwrite';
}

// Extend the FileSystemHandle interface
declare global {
  interface FileSystemHandle {
    queryPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>;
    requestPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>;
  }
}

// Storage keys for IndexedDB
const STORAGE_KEYS = {
  PROXY_FOLDER: 'proxyFolderHandle',
  RAW_FOLDER: 'rawFolderHandle',
  FILE_HANDLES: 'fileHandles',
} as const;

// Check if File System Access API is supported
export function isFileSystemAccessSupported(): boolean {
  return 'showOpenFilePicker' in window && 'showDirectoryPicker' in window;
}

// Store for file handles (maps file ID to handle)
const fileHandleCache = new Map<string, FileSystemFileHandle>();

// Directory handles
let proxyFolderHandle: FileSystemDirectoryHandle | null = null;

// Initialize from IndexedDB
export async function initFileSystemService(): Promise<void> {
  if (!isFileSystemAccessSupported()) {
    log.info('File System Access API not supported');
    return;
  }

  try {
    // Try to restore directory handles from IndexedDB
    const storedProxyHandle = await projectDB.getStoredHandle(STORAGE_KEYS.PROXY_FOLDER);
    const storedRawHandle = await projectDB.getStoredHandle(STORAGE_KEYS.RAW_FOLDER);

    if (storedProxyHandle) {
      // Verify we still have permission
      const permission = await storedProxyHandle.queryPermission({ mode: 'readwrite' });
      if (permission === 'granted') {
        proxyFolderHandle = storedProxyHandle as FileSystemDirectoryHandle;
        log.info('Restored proxy folder handle');
      }
    }

    if (storedRawHandle) {
      const permission = await storedRawHandle.queryPermission({ mode: 'read' });
      if (permission === 'granted') {
        // Reserved for future raw file folder support
        log.info('Restored raw folder handle');
      }
    }
  } catch (e) {
    log.warn('Failed to restore handles', e);
  }
}

// Pick files using File System Access API
export async function pickFiles(options?: {
  multiple?: boolean;
  types?: Array<{
    description: string;
    accept: Record<string, string[]>;
  }>;
}): Promise<Array<{ file: File; handle: FileSystemFileHandle }> | null> {
  if (!isFileSystemAccessSupported()) {
    return null;
  }

  try {
    const handles = await (window as any).showOpenFilePicker({
      multiple: options?.multiple ?? true,
      types: options?.types ?? [
        {
          description: 'Media Files',
          accept: {
            'video/*': ['.mp4', '.webm', '.mov', '.avi', '.mkv'],
            'audio/*': ['.mp3', '.wav', '.ogg', '.aac', '.m4a'],
            'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
          },
        },
      ],
    });

    const results: Array<{ file: File; handle: FileSystemFileHandle }> = [];
    for (const handle of handles) {
      const file = await handle.getFile();
      results.push({ file, handle });
    }

    return results;
  } catch (e: any) {
    if (e.name === 'AbortError') {
      // User cancelled
      return null;
    }
    log.error('Failed to pick files', e);
    return null;
  }
}

// Store a file handle for later access
export function storeFileHandle(fileId: string, handle: FileSystemFileHandle): void {
  fileHandleCache.set(fileId, handle);
}

// Get stored file handle
export function getFileHandle(fileId: string): FileSystemFileHandle | undefined {
  return fileHandleCache.get(fileId);
}

// Pick a directory for proxy storage
export async function pickProxyFolder(): Promise<FileSystemDirectoryHandle | null> {
  if (!isFileSystemAccessSupported()) {
    return null;
  }

  try {
    const handle = await (window as any).showDirectoryPicker({
      mode: 'readwrite',
      startIn: 'documents',
    });

    proxyFolderHandle = handle;

    // Store in IndexedDB for persistence
    await projectDB.storeHandle(STORAGE_KEYS.PROXY_FOLDER, handle);
    log.info('Proxy folder set:', handle.name);

    return handle;
  } catch (e: any) {
    if (e.name === 'AbortError') {
      return null;
    }
    log.error('Failed to pick proxy folder', e);
    return null;
  }
}

// Get the proxy folder handle
export function getProxyFolderHandle(): FileSystemDirectoryHandle | null {
  return proxyFolderHandle;
}

// Check if proxy folder is set
export function hasProxyFolder(): boolean {
  return proxyFolderHandle !== null;
}

// Request permission for a stored handle (needed after page reload)
export async function requestHandlePermission(
  handle: FileSystemHandle,
  mode: 'read' | 'readwrite' = 'read'
): Promise<boolean> {
  try {
    const permission = await handle.queryPermission({ mode });
    if (permission === 'granted') {
      return true;
    }

    const result = await handle.requestPermission({ mode });
    return result === 'granted';
  } catch (e) {
    console.error('[FileSystem] Failed to request permission:', e);
    return false;
  }
}

// Save a proxy frame to the proxy folder
export async function saveProxyFrame(
  mediaFileId: string,
  frameIndex: number,
  blob: Blob
): Promise<boolean> {
  if (!proxyFolderHandle) {
    return false;
  }

  try {
    // Create a subdirectory for this media file
    let mediaDir: FileSystemDirectoryHandle;
    try {
      mediaDir = await proxyFolderHandle.getDirectoryHandle(mediaFileId, { create: true });
    } catch {
      // If we can't create subdirs, save to root
      mediaDir = proxyFolderHandle;
    }

    // Create the frame file
    const fileName = `frame_${frameIndex.toString().padStart(6, '0')}.webp`;
    const fileHandle = await mediaDir.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();

    return true;
  } catch (e) {
    console.error('[FileSystem] Failed to save proxy frame:', e);
    return false;
  }
}

// Get the path to the proxy folder (for display purposes)
export function getProxyFolderName(): string | null {
  return proxyFolderHandle?.name ?? null;
}

// Get info about file location
export interface FileLocationInfo {
  name: string;
  isAccessible: boolean;
  handle?: FileSystemFileHandle;
}

export async function getFileLocation(fileId: string): Promise<FileLocationInfo | null> {
  const handle = fileHandleCache.get(fileId);
  if (!handle) {
    return null;
  }

  try {
    // Check if we still have access
    const permission = await handle.queryPermission({ mode: 'read' });
    return {
      name: handle.name,
      isAccessible: permission === 'granted',
      handle,
    };
  } catch {
    return {
      name: handle.name,
      isAccessible: false,
      handle,
    };
  }
}

// Open containing folder (limited support - shows path info)
export async function showInExplorer(
  type: 'raw' | 'proxy',
  fileId?: string
): Promise<{ success: boolean; path?: string; message: string }> {
  if (type === 'proxy') {
    if (!proxyFolderHandle) {
      return {
        success: false,
        message: 'Proxy folder not set. Please select a proxy folder first.',
      };
    }

    // We can't directly open the folder, but we can tell the user where it is
    const folderName = proxyFolderHandle.name;
    return {
      success: true,
      path: folderName,
      message: `Proxy files are stored in: ${folderName}${fileId ? `/${fileId}/` : ''}`,
    };
  }

  if (type === 'raw' && fileId) {
    const handle = fileHandleCache.get(fileId);
    if (!handle) {
      return {
        success: false,
        message: 'File handle not available. The file was imported using the legacy method.',
      };
    }

    return {
      success: true,
      path: handle.name,
      message: `File: ${handle.name}`,
    };
  }

  return {
    success: false,
    message: 'Invalid request',
  };
}

// Create the file system service singleton
export const fileSystemService = {
  isSupported: isFileSystemAccessSupported,
  init: initFileSystemService,
  pickFiles,
  storeFileHandle,
  getFileHandle,
  pickProxyFolder,
  getProxyFolderHandle,
  hasProxyFolder,
  requestHandlePermission,
  saveProxyFrame,
  getProxyFolderName,
  getFileLocation,
  showInExplorer,
};
