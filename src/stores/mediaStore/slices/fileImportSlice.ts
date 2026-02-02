// File import actions - unified import logic

import type { MediaFile, MediaSliceCreator } from '../types';
import { generateId, processImport } from '../helpers/importPipeline';
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

export const createFileImportSlice: MediaSliceCreator<FileImportActions> = (set, _get) => ({
  importFile: async (file: File, parentId?: string | null) => {
    log.info(`Starting: ${file.name} type: ${file.type} size: ${file.size}`);

    const result = await processImport({
      file,
      id: generateId(),
      parentId,
    });

    set((state) => ({
      files: [...state.files, result.mediaFile],
    }));

    log.info('Complete:', result.mediaFile.name);
    return result.mediaFile;
  },

  importFiles: async (files: FileList | File[], parentId?: string | null) => {
    const fileArray = Array.from(files);
    const imported: MediaFile[] = [];

    // Process in parallel batches of 3
    const batchSize = 3;
    for (let i = 0; i < fileArray.length; i += batchSize) {
      const batch = fileArray.slice(i, i + batchSize);
      const results = await Promise.all(
        batch.map(async (file) => {
          const result = await processImport({
            file,
            id: generateId(),
            parentId,
          });
          set((state) => ({
            files: [...state.files, result.mediaFile],
          }));
          return result.mediaFile;
        })
      );
      imported.push(...results);
    }

    return imported;
  },

  importFilesWithPicker: async () => {
    const result = await fileSystemService.pickFiles();
    if (!result || result.length === 0) return [];

    const imported: MediaFile[] = [];

    for (const { file, handle } of result) {
      const id = generateId();

      // Store original handle (for reference, but RAW folder is primary)
      fileSystemService.storeFileHandle(id, handle);
      await projectDB.storeHandle(`media_${id}`, handle);
      log.debug('Stored file handle for ID:', id);

      const importResult = await processImport({ file, id, handle });

      set((state) => ({
        files: [...state.files, importResult.mediaFile],
      }));

      imported.push(importResult.mediaFile);
    }

    return imported;
  },

  importFilesWithHandles: async (filesWithHandles) => {
    const imported: MediaFile[] = [];

    for (const { file, handle, absolutePath } of filesWithHandles) {
      const id = generateId();

      // Store original handle (for reference, but RAW folder is primary)
      fileSystemService.storeFileHandle(id, handle);
      await projectDB.storeHandle(`media_${id}`, handle);
      log.debug('Stored file handle for ID:', id);

      const importResult = await processImport({ file, id, handle, absolutePath });

      set((state) => ({
        files: [...state.files, importResult.mediaFile],
      }));

      imported.push(importResult.mediaFile);
    }

    return imported;
  },
});
