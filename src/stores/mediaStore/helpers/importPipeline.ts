// Unified import pipeline - eliminates 3x duplicate import logic

import type { MediaFile, ProxyStatus } from '../types';
import { PROXY_FPS } from '../constants';
import { detectMediaType } from '../../timeline/helpers/mediaTypeHelpers';
import { calculateFileHash } from './fileHashHelpers';
import { getMediaInfo } from './mediaInfoHelpers';
import { createThumbnail, handleThumbnailDedup } from './thumbnailHelpers';
import { projectFileService } from '../../../services/projectFileService';
import { fileSystemService } from '../../../services/fileSystemService';
import { projectDB } from '../../../services/projectDB';
import { useSettingsStore } from '../../settingsStore';
import { Logger } from '../../../services/logger';

const log = Logger.create('Import');

export interface ImportParams {
  file: File;
  id: string;
  handle?: FileSystemFileHandle;
  absolutePath?: string;
}

export interface ImportResult {
  mediaFile: MediaFile;
  projectFileHandle?: FileSystemFileHandle;
}

/**
 * Generate unique ID.
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Unified import pipeline for all import methods.
 * Replaces duplicate logic in importFile, importFilesWithPicker, importFilesWithHandles.
 */
export async function processImport(params: ImportParams): Promise<ImportResult> {
  const { file, id, handle, absolutePath } = params;

  // Store handle if provided (for original file location)
  if (handle) {
    fileSystemService.storeFileHandle(id, handle);
    await projectDB.storeHandle(`media_${id}`, handle);
  }

  // Detect type using shared helper from clipSlice
  const type = detectMediaType(file) as 'video' | 'audio' | 'image';
  const url = URL.createObjectURL(file);

  // Get info and thumbnail in parallel
  const [info, rawThumbnail] = await Promise.all([
    getMediaInfo(file, type),
    type !== 'audio' ? createThumbnail(file, type as 'video' | 'image') : Promise.resolve(undefined),
  ]);

  // Calculate hash for deduplication
  const fileHash = await calculateFileHash(file);

  // Handle thumbnail deduplication (unified - was 3x duplicate)
  const thumbnailUrl = await handleThumbnailDedup(fileHash, rawThumbnail);

  // Check for existing proxy (unified - was 3x duplicate)
  const proxyInfo = await checkExistingProxy(fileHash, type);

  // Copy to Raw folder if enabled (unified - was 3x duplicate)
  const copyResult = await copyToRawIfEnabled(file, id);

  // Build MediaFile
  const mediaFile: MediaFile = {
    id,
    name: file.name,
    type,
    parentId: null,
    createdAt: Date.now(),
    file,
    url,
    thumbnailUrl,
    fileHash,
    hasFileHandle: !!handle || !!copyResult?.handle,
    filePath: handle?.name || file.name,
    absolutePath,
    projectPath: copyResult?.relativePath,
    ...info,
    ...proxyInfo,
  };

  return {
    mediaFile,
    projectFileHandle: copyResult?.handle,
  };
}

/**
 * Check for existing proxy by hash.
 * UNIFIED: Replaces 3 duplicate blocks.
 */
async function checkExistingProxy(
  fileHash: string | undefined,
  type: 'video' | 'audio' | 'image'
): Promise<{
  proxyStatus: ProxyStatus;
  proxyFrameCount?: number;
  proxyFps?: number;
  proxyProgress?: number;
}> {
  if (!fileHash || type !== 'video' || !projectFileService.isProjectOpen()) {
    return { proxyStatus: 'none' };
  }

  const frameCount = await projectFileService.getProxyFrameCount(fileHash);
  if (frameCount > 0) {
    log.debug('Found existing proxy:', fileHash.slice(0, 8), 'frames:', frameCount);
    return {
      proxyStatus: 'ready',
      proxyFrameCount: frameCount,
      proxyFps: PROXY_FPS,
      proxyProgress: 100,
    };
  }

  return { proxyStatus: 'none' };
}

/**
 * Copy file to Raw folder if setting enabled.
 * UNIFIED: Replaces 3 duplicate blocks.
 */
async function copyToRawIfEnabled(
  file: File,
  mediaId: string
): Promise<{ relativePath: string; handle: FileSystemFileHandle } | null> {
  const { copyMediaToProject } = useSettingsStore.getState();

  if (!copyMediaToProject || !projectFileService.isProjectOpen()) {
    return null;
  }

  const result = await projectFileService.copyToRawFolder(file);
  if (result) {
    // Store the project file handle for the RAW copy
    fileSystemService.storeFileHandle(`${mediaId}_project`, result.handle);
    await projectDB.storeHandle(`media_${mediaId}_project`, result.handle);
    log.debug('Copied to Raw folder:', result.relativePath);
    return { relativePath: result.relativePath, handle: result.handle };
  }

  return null;
}

/**
 * Process multiple files in parallel batches.
 */
export async function batchImport<T>(
  items: T[],
  batchSize: number,
  processor: (item: T) => Promise<unknown>
): Promise<void> {
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    await Promise.all(batch.map(processor));
  }
}
