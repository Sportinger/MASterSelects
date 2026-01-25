// File management actions - remove, rename, reload
// SIMPLIFIED: Uses RAW folder for easy relinking

import type { MediaSliceCreator } from '../types';
import { projectFileService } from '../../../services/projectFileService';
import { fileSystemService } from '../../../services/fileSystemService';
import { projectDB } from '../../../services/projectDB';
import { useTimelineStore } from '../../timeline';
import { Logger } from '../../../services/logger';

const log = Logger.create('Reload');

export interface FileManageActions {
  removeFile: (id: string) => void;
  renameFile: (id: string, name: string) => void;
  reloadFile: (id: string) => Promise<boolean>;
  reloadAllFiles: () => Promise<number>;
}

export const createFileManageSlice: MediaSliceCreator<FileManageActions> = (set, get) => ({
  removeFile: (id: string) => {
    const file = get().files.find((f) => f.id === id);
    if (file?.url) URL.revokeObjectURL(file.url);
    if (file?.thumbnailUrl?.startsWith('blob:')) URL.revokeObjectURL(file.thumbnailUrl);

    set((state) => ({
      files: state.files.filter((f) => f.id !== id),
      selectedIds: state.selectedIds.filter((sid) => sid !== id),
    }));
  },

  renameFile: (id: string, name: string) => {
    set((state) => ({
      files: state.files.map((f) => (f.id === id ? { ...f, name } : f)),
    }));
  },

  /**
   * Reload a single file - tries RAW folder first, then falls back to file handle.
   */
  reloadFile: async (id: string) => {
    const mediaFile = get().files.find(f => f.id === id);
    if (!mediaFile) return false;

    let file: File | undefined;
    let handle: FileSystemFileHandle | undefined;

    // Try 1: Get from project RAW folder (we already have folder permission!)
    if (mediaFile.projectPath && projectFileService.isProjectOpen()) {
      const result = await projectFileService.getFileFromRaw(mediaFile.projectPath);
      if (result) {
        file = result.file;
        handle = result.handle;
        log.debug('Got file from RAW folder:', mediaFile.projectPath);
      }
    }

    // Try 2: Fallback to stored file handle
    if (!file) {
      const storedHandle = await projectDB.getStoredHandle(`media_${id}`);
      if (storedHandle && 'getFile' in storedHandle) {
        try {
          const permission = await (storedHandle as FileSystemFileHandle).queryPermission({ mode: 'read' });
          if (permission === 'granted') {
            file = await (storedHandle as FileSystemFileHandle).getFile();
            handle = storedHandle as FileSystemFileHandle;
            log.debug('Got file from stored handle:', mediaFile.name);
          } else {
            const newPermission = await (storedHandle as FileSystemFileHandle).requestPermission({ mode: 'read' });
            if (newPermission === 'granted') {
              file = await (storedHandle as FileSystemFileHandle).getFile();
              handle = storedHandle as FileSystemFileHandle;
              log.debug('Got file from stored handle (after permission):', mediaFile.name);
            }
          }
        } catch (e) {
          log.warn('Failed to get file from stored handle:', e);
        }
      }
    }

    if (!file) {
      log.warn('Could not reload file:', mediaFile.name);
      return false;
    }

    // Store handle if we got one
    if (handle) {
      fileSystemService.storeFileHandle(id, handle);
    }

    // Revoke old URL
    if (mediaFile.url) URL.revokeObjectURL(mediaFile.url);

    // Create new URL
    const url = URL.createObjectURL(file);

    // Update store
    set((state) => ({
      files: state.files.map((f) =>
        f.id === id ? { ...f, file, url, hasFileHandle: true } : f
      ),
    }));

    // Update timeline clips
    await updateTimelineClips(id, file);

    log.info('Success:', mediaFile.name);
    return true;
  },

  /**
   * Reload all files that need reloading.
   * SIMPLIFIED: Batch reload from RAW folder - no user prompts needed!
   */
  reloadAllFiles: async () => {
    const filesToReload = get().files.filter(f => !f.file);
    if (filesToReload.length === 0) {
      log.debug('No files need reloading');
      return 0;
    }

    log.info('Reloading', filesToReload.length, 'files...');
    let totalReloaded = 0;

    for (const mediaFileToReload of filesToReload) {
      // Inline reload logic to avoid calling get().reloadFile()
      let file: File | undefined;
      let handle: FileSystemFileHandle | undefined;

      // Try 1: Get from project RAW folder
      if (mediaFileToReload.projectPath && projectFileService.isProjectOpen()) {
        const result = await projectFileService.getFileFromRaw(mediaFileToReload.projectPath);
        if (result) {
          file = result.file;
          handle = result.handle;
        }
      }

      // Try 2: Fallback to stored file handle
      if (!file) {
        const storedHandle = await projectDB.getStoredHandle(`media_${mediaFileToReload.id}`);
        if (storedHandle && 'getFile' in storedHandle) {
          try {
            const permission = await (storedHandle as FileSystemFileHandle).queryPermission({ mode: 'read' });
            if (permission === 'granted') {
              file = await (storedHandle as FileSystemFileHandle).getFile();
              handle = storedHandle as FileSystemFileHandle;
            }
          } catch {
            // Ignore
          }
        }
      }

      if (!file) continue;

      if (handle) {
        fileSystemService.storeFileHandle(mediaFileToReload.id, handle);
      }

      if (mediaFileToReload.url) URL.revokeObjectURL(mediaFileToReload.url);
      const url = URL.createObjectURL(file);

      set((state) => ({
        files: state.files.map((f) =>
          f.id === mediaFileToReload.id ? { ...f, file, url, hasFileHandle: true } : f
        ),
      }));

      await updateTimelineClips(mediaFileToReload.id, file);
      totalReloaded++;
    }

    log.info('Complete:', totalReloaded, 'files reloaded');
    return totalReloaded;
  },
});

/**
 * Update timeline clips with reloaded file.
 */
async function updateTimelineClips(mediaFileId: string, file: File): Promise<void> {
  const timelineStore = useTimelineStore.getState();
  const clips = timelineStore.clips.filter(
    c => c.source?.mediaFileId === mediaFileId && c.needsReload
  );

  for (const clip of clips) {
    timelineStore.updateClip(clip.id, {
      file,
      needsReload: false,
      isLoading: true,
    });
  }

  if (clips.length > 0) {
    log.debug('Updated', clips.length, 'timeline clips');
  }
}
