// File import actions - unified import logic

import type { MediaFile, MediaSliceCreator } from '../types';
import { generateId, processImport } from '../helpers/importPipeline';
import { detectMediaType } from '../../timeline/helpers/mediaTypeHelpers';
import { fileSystemService } from '../../../services/fileSystemService';
import { projectDB } from '../../../services/projectDB';
import { Logger } from '../../../services/logger';

const log = Logger.create('Import');

export interface FileImportActions {
  importFile: (file: File, parentId?: string | null) => Promise<MediaFile>;
  importFiles: (files: FileList | File[], parentId?: string | null) => Promise<MediaFile[]>;
  importFilesWithPicker: () => Promise<MediaFile[]>;
  importFilesWithHandles: (filesWithHandles: Array<{
    file: File;
    handle: FileSystemFileHandle;
    absolutePath?: string;
  }>) => Promise<MediaFile[]>;
}

/**
 * Create a placeholder MediaFile that appears instantly in the media panel.
 * Shows as grey/loading while the full import runs in the background.
 */
function createPlaceholder(file: File, id: string, parentId?: string | null): MediaFile {
  const type = detectMediaType(file) as 'video' | 'audio' | 'image';
  return {
    id,
    name: file.name,
    type,
    parentId: parentId ?? null,
    createdAt: Date.now(),
    file,
    url: '',
    fileSize: file.size,
    isImporting: true,
  };
}

/**
 * Merge import result into placeholder, preserving any state changes
 * that may have happened during import (e.g. folder moves).
 */
function finalizePlaceholder(state: { files: MediaFile[] }, id: string, result: MediaFile): { files: MediaFile[] } {
  return {
    files: state.files.map(f => {
      if (f.id !== id) return f;
      // Merge: keep parentId/labelColor from current state (user may have moved it),
      // but take everything else from import result
      return {
        ...result,
        parentId: f.parentId,
        labelColor: f.labelColor,
        isImporting: false,
      };
    }),
  };
}

export const createFileImportSlice: MediaSliceCreator<FileImportActions> = (set, _get) => ({
  importFile: async (file: File, parentId?: string | null) => {
    const id = generateId();
    log.info(`Starting: ${file.name} type: ${file.type} size: ${file.size}`);

    // Phase 1: Add placeholder instantly
    const placeholder = createPlaceholder(file, id, parentId);
    set((state) => ({
      files: [...state.files, placeholder],
    }));

    // Phase 2: Full import in background
    try {
      const result = await processImport({ file, id, parentId });
      set((state) => finalizePlaceholder(state, id, result.mediaFile));
      log.info('Complete:', result.mediaFile.name);
      return result.mediaFile;
    } catch (err) {
      log.error(`Import failed: ${file.name}`, err);
      // Remove placeholder on failure
      set((state) => ({
        files: state.files.filter(f => f.id !== id),
      }));
      throw err;
    }
  },

  importFiles: async (files: FileList | File[], parentId?: string | null) => {
    const fileArray = Array.from(files);
    const imported: MediaFile[] = [];

    // Phase 1: Add all placeholders instantly
    const entries = fileArray.map(file => ({
      file,
      id: generateId(),
    }));

    set((state) => ({
      files: [
        ...state.files,
        ...entries.map(e => createPlaceholder(e.file, e.id, parentId)),
      ],
    }));

    // Phase 2: Process in parallel batches of 3
    const batchSize = 3;
    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);
      const results = await Promise.all(
        batch.map(async ({ file, id }) => {
          try {
            const result = await processImport({ file, id, parentId });
            set((state) => finalizePlaceholder(state, id, result.mediaFile));
            return result.mediaFile;
          } catch (err) {
            log.error(`Import failed: ${file.name}`, err);
            set((state) => ({
              files: state.files.filter(f => f.id !== id),
            }));
            return null;
          }
        })
      );
      imported.push(...results.filter((r): r is MediaFile => r !== null));
    }

    return imported;
  },

  importFilesWithPicker: async () => {
    const result = await fileSystemService.pickFiles();
    if (!result || result.length === 0) return [];

    const imported: MediaFile[] = [];

    // Phase 1: Add all placeholders instantly
    const entries = result.map(({ file, handle }) => ({
      file,
      handle,
      id: generateId(),
    }));

    set((state) => ({
      files: [
        ...state.files,
        ...entries.map(e => createPlaceholder(e.file, e.id)),
      ],
    }));

    // Phase 2: Process each file
    for (const { file, handle, id } of entries) {
      // Store original handle
      fileSystemService.storeFileHandle(id, handle);
      await projectDB.storeHandle(`media_${id}`, handle);
      log.debug('Stored file handle for ID:', id);

      try {
        const importResult = await processImport({ file, id, handle });
        set((state) => finalizePlaceholder(state, id, importResult.mediaFile));
        imported.push(importResult.mediaFile);
      } catch (err) {
        log.error(`Import failed: ${file.name}`, err);
        set((state) => ({
          files: state.files.filter(f => f.id !== id),
        }));
      }
    }

    return imported;
  },

  importFilesWithHandles: async (filesWithHandles) => {
    const imported: MediaFile[] = [];

    // Phase 1: Add all placeholders instantly
    const entries = filesWithHandles.map(({ file, handle, absolutePath }) => ({
      file,
      handle,
      absolutePath,
      id: generateId(),
    }));

    set((state) => ({
      files: [
        ...state.files,
        ...entries.map(e => createPlaceholder(e.file, e.id)),
      ],
    }));

    // Phase 2: Process each file
    for (const { file, handle, absolutePath, id } of entries) {
      // Store original handle
      fileSystemService.storeFileHandle(id, handle);
      await projectDB.storeHandle(`media_${id}`, handle);
      log.debug('Stored file handle for ID:', id);

      try {
        const importResult = await processImport({ file, id, handle, absolutePath });
        set((state) => finalizePlaceholder(state, id, importResult.mediaFile));
        imported.push(importResult.mediaFile);
      } catch (err) {
        log.error(`Import failed: ${file.name}`, err);
        set((state) => ({
          files: state.files.filter(f => f.id !== id),
        }));
      }
    }

    return imported;
  },
});
