// Low-level file I/O operations for project storage
// Provides reusable primitives for all domain services

import { Logger } from '../../logger';
import { PROJECT_FOLDERS, PROJECT_FOLDER_PATHS, type ProjectFolderKey } from './constants';

const log = Logger.create('FileStorage');

export class FileStorageService {
  /**
   * Navigate to a subfolder within a base directory, optionally creating it
   */
  async navigateToFolder(
    baseHandle: FileSystemDirectoryHandle,
    folderPath: string,
    create = false
  ): Promise<FileSystemDirectoryHandle | null> {
    try {
      let folder = baseHandle;
      for (const part of folderPath.split('/')) {
        folder = await folder.getDirectoryHandle(part, { create });
      }
      return folder;
    } catch (e) {
      if (!create) return null;
      throw e;
    }
  }

  /**
   * Get a file handle from a project subfolder
   */
  async getFileHandle(
    projectHandle: FileSystemDirectoryHandle,
    subFolder: ProjectFolderKey,
    fileName: string,
    create = false
  ): Promise<FileSystemFileHandle | null> {
    try {
      const folderPath = PROJECT_FOLDERS[subFolder];
      const folder = await this.navigateToFolder(projectHandle, folderPath, create);
      if (!folder) return null;
      return await folder.getFileHandle(fileName, { create });
    } catch (e) {
      if (!create) return null;
      throw e;
    }
  }

  /**
   * Write a file to a project subfolder
   */
  async writeFile(
    projectHandle: FileSystemDirectoryHandle,
    subFolder: ProjectFolderKey,
    fileName: string,
    content: Blob | string
  ): Promise<boolean> {
    try {
      const handle = await this.getFileHandle(projectHandle, subFolder, fileName, true);
      if (!handle) return false;

      const writable = await handle.createWritable();
      await writable.write(content);
      await writable.close();
      return true;
    } catch (e) {
      log.error(`Failed to write ${subFolder}/${fileName}:`, e);
      return false;
    }
  }

  /**
   * Read a file from a project subfolder
   */
  async readFile(
    projectHandle: FileSystemDirectoryHandle,
    subFolder: ProjectFolderKey,
    fileName: string
  ): Promise<File | null> {
    try {
      const handle = await this.getFileHandle(projectHandle, subFolder, fileName);
      if (!handle) return null;
      return await handle.getFile();
    } catch (e) {
      return null;
    }
  }

  /**
   * Check if a file exists in a project subfolder
   */
  async fileExists(
    projectHandle: FileSystemDirectoryHandle,
    subFolder: ProjectFolderKey,
    fileName: string
  ): Promise<boolean> {
    const handle = await this.getFileHandle(projectHandle, subFolder, fileName);
    return handle !== null;
  }

  /**
   * Delete a file from a project subfolder
   */
  async deleteFile(
    projectHandle: FileSystemDirectoryHandle,
    subFolder: ProjectFolderKey,
    fileName: string
  ): Promise<boolean> {
    try {
      const folderPath = PROJECT_FOLDERS[subFolder];
      const folder = await this.navigateToFolder(projectHandle, folderPath, false);
      if (!folder) return false;

      await folder.removeEntry(fileName);
      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * List files in a project subfolder
   */
  async listFiles(
    projectHandle: FileSystemDirectoryHandle,
    subFolder: ProjectFolderKey
  ): Promise<string[]> {
    try {
      const folderPath = PROJECT_FOLDERS[subFolder];
      const folder = await this.navigateToFolder(projectHandle, folderPath, false);
      if (!folder) return [];

      const files: string[] = [];
      for await (const entry of (folder as any).values()) {
        if (entry.kind === 'file') {
          files.push(entry.name);
        }
      }
      return files;
    } catch (e) {
      return [];
    }
  }

  /**
   * Create all project subfolders
   */
  async createProjectFolders(handle: FileSystemDirectoryHandle): Promise<void> {
    for (const folderPath of PROJECT_FOLDER_PATHS) {
      await this.navigateToFolder(handle, folderPath, true);
    }
  }
}

// Singleton instance
export const fileStorageService = new FileStorageService();
