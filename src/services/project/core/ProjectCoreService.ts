// Project lifecycle management service
// Handles create, open, save, close, rename, backup operations

import { Logger } from '../../logger';
import { projectDB } from '../../projectDB';
import { apiKeyManager } from '../../apiKeyManager';

const log = Logger.create('ProjectCore');
import { FileStorageService } from './FileStorageService';
import { MAX_BACKUPS, PROJECT_FOLDERS } from './constants';
import type { ProjectFile, ProjectMediaFile, ProjectComposition, ProjectFolder } from '../types';

const KEYS_FILE_NAME = '.keys.enc';

export class ProjectCoreService {
  private projectHandle: FileSystemDirectoryHandle | null = null;
  private projectData: ProjectFile | null = null;
  private isDirty = false;
  private autoSaveInterval: number | null = null;
  private pendingHandle: FileSystemDirectoryHandle | null = null;
  private permissionNeeded = false;
  private fileStorage: FileStorageService;

  constructor(fileStorage: FileStorageService) {
    this.fileStorage = fileStorage;
  }

  // ============================================
  // GETTERS & STATE CHECKS
  // ============================================

  isSupported(): boolean {
    return 'showDirectoryPicker' in window && 'showSaveFilePicker' in window;
  }

  getProjectHandle(): FileSystemDirectoryHandle | null {
    return this.projectHandle;
  }

  getProjectData(): ProjectFile | null {
    return this.projectData;
  }

  isProjectOpen(): boolean {
    return this.projectHandle !== null && this.projectData !== null;
  }

  hasUnsavedChanges(): boolean {
    return this.isDirty;
  }

  markDirty(): void {
    this.isDirty = true;
  }

  needsPermission(): boolean {
    return this.permissionNeeded && this.pendingHandle !== null;
  }

  getPendingProjectName(): string | null {
    return this.pendingHandle?.name || null;
  }

  // ============================================
  // PERMISSION HANDLING
  // ============================================

  async requestPendingPermission(): Promise<boolean> {
    if (!this.pendingHandle) return false;

    try {
      const result = await this.pendingHandle.requestPermission({ mode: 'readwrite' });
      if (result === 'granted') {
        const success = await this.loadProject(this.pendingHandle);
        if (success) {
          this.pendingHandle = null;
          this.permissionNeeded = false;
          return true;
        }
      }
    } catch (e) {
      log.warn('Failed to request permission:', e);
    }
    return false;
  }

  // ============================================
  // PROJECT OPERATIONS
  // ============================================

  async createProject(name: string): Promise<boolean> {
    if (!this.isSupported()) {
      log.error('File System Access API not supported');
      return false;
    }

    try {
      const handle = await (window as any).showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'documents',
      });

      await projectDB.storeHandle('projectsFolder', handle);
      const projectFolder = await handle.getDirectoryHandle(name, { create: true });
      return await this.initializeProject(projectFolder, name);
    } catch (e: any) {
      if (e.name === 'AbortError') return false;
      log.error('Failed to create project:', e);
      return false;
    }
  }

  async createProjectInFolder(handle: FileSystemDirectoryHandle, name: string): Promise<boolean> {
    if (!this.isSupported()) {
      log.error('File System Access API not supported');
      return false;
    }

    try {
      await projectDB.storeHandle('projectsFolder', handle);
      const projectFolder = await handle.getDirectoryHandle(name, { create: true });
      return await this.initializeProject(projectFolder, name);
    } catch (e: any) {
      log.error('Failed to create project in folder:', e);
      return false;
    }
  }

  private async initializeProject(projectFolder: FileSystemDirectoryHandle, name: string): Promise<boolean> {
    try {
      await this.fileStorage.createProjectFolders(projectFolder);

      const mainCompId = `comp-${Date.now()}`;

      const initialProject: ProjectFile = {
        version: 1,
        name,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        settings: {
          width: 1920,
          height: 1080,
          frameRate: 30,
          sampleRate: 48000,
        },
        media: [],
        compositions: [{
          id: mainCompId,
          name: 'Main Comp',
          width: 1920,
          height: 1080,
          frameRate: 30,
          duration: 60,
          backgroundColor: '#000000',
          folderId: null,
          tracks: [
            { id: 'track-v1', name: 'Video 1', type: 'video', height: 60, locked: false, visible: true, muted: false, solo: false },
            { id: 'track-a1', name: 'Audio 1', type: 'audio', height: 40, locked: false, visible: true, muted: false, solo: false },
          ],
          clips: [],
          markers: [],
        }],
        folders: [],
        activeCompositionId: mainCompId,
        openCompositionIds: [mainCompId],
        expandedFolderIds: [],
      };

      await this.writeProjectJson(projectFolder, initialProject);

      this.projectHandle = projectFolder;
      this.projectData = initialProject;
      this.isDirty = false;

      await this.storeLastProject(projectFolder);
      this.startAutoSave();

      // Save any existing API keys to the new project
      await this.saveKeysFile();

      log.info(`Created project: ${name}`);
      return true;
    } catch (e) {
      log.error('Failed to initialize project:', e);
      return false;
    }
  }

  async openProject(): Promise<boolean> {
    if (!this.isSupported()) {
      log.error('File System Access API not supported');
      return false;
    }

    try {
      const handle = await (window as any).showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'documents',
      });

      return await this.loadProject(handle);
    } catch (e: any) {
      if (e.name === 'AbortError') return false;
      log.error('Failed to open project:', e);
      return false;
    }
  }

  async loadProject(handle: FileSystemDirectoryHandle): Promise<boolean> {
    try {
      const projectFile = await handle.getFileHandle('project.json');
      const file = await projectFile.getFile();
      const content = await file.text();
      const projectData = JSON.parse(content) as ProjectFile;

      if (projectData.version !== 1) {
        log.error('Unsupported project version:', projectData.version);
        return false;
      }

      await this.fileStorage.createProjectFolders(handle);

      this.projectHandle = handle;
      this.projectData = projectData;
      this.isDirty = false;

      await this.storeLastProject(handle);
      this.startAutoSave();

      // Try to restore API keys from file if IndexedDB keys are empty
      try {
        const existingKeys = await apiKeyManager.getAllKeys();
        const hasAnyKey = Object.values(existingKeys).some((v: string) => v !== '');
        if (!hasAnyKey) {
          await this.loadKeysFile();
        }
      } catch (e) {
        log.warn('Failed to check/restore API keys:', e);
      }

      log.info(`Opened project: ${projectData.name}`);
      return true;
    } catch (e) {
      log.error('Failed to load project:', e);
      return false;
    }
  }

  async saveProject(): Promise<boolean> {
    if (!this.projectHandle || !this.projectData) {
      log.error('No project open');
      return false;
    }

    try {
      this.projectData.updatedAt = new Date().toISOString();
      await this.writeProjectJson(this.projectHandle, this.projectData);
      this.isDirty = false;

      // Also update the keys file
      await this.saveKeysFile();

      log.debug('Project saved');
      return true;
    } catch (e) {
      log.error('Failed to save project:', e);
      return false;
    }
  }

  closeProject(): void {
    this.stopAutoSave();
    this.projectHandle = null;
    this.projectData = null;
    this.isDirty = false;
    log.info('Project closed');
  }

  // ============================================
  // BACKUP OPERATIONS
  // ============================================

  async createBackup(): Promise<boolean> {
    if (!this.projectHandle || !this.projectData) {
      return false;
    }

    try {
      const projectFile = await this.projectHandle.getFileHandle('project.json');
      const file = await projectFile.getFile();
      const content = await file.text();

      const now = new Date();
      const timestamp = now.toISOString()
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .slice(0, 19);
      const backupFileName = `project_${timestamp}.json`;

      const backupsFolder = await this.projectHandle.getDirectoryHandle(PROJECT_FOLDERS.BACKUPS, { create: true });

      const backupHandle = await backupsFolder.getFileHandle(backupFileName, { create: true });
      const writable = await backupHandle.createWritable();
      await writable.write(content);
      await writable.close();

      log.debug(`Created backup: ${backupFileName}`);

      await this.cleanupOldBackups(backupsFolder);

      return true;
    } catch (e) {
      log.error('Failed to create backup:', e);
      return false;
    }
  }

  private async cleanupOldBackups(backupsFolder: FileSystemDirectoryHandle): Promise<void> {
    try {
      const backups: { name: string; file: File }[] = [];

      for await (const entry of (backupsFolder as any).values()) {
        if (entry.kind === 'file' && entry.name.startsWith('project_') && entry.name.endsWith('.json')) {
          const file = await entry.getFile();
          backups.push({ name: entry.name, file });
        }
      }

      backups.sort((a, b) => b.file.lastModified - a.file.lastModified);

      if (backups.length > MAX_BACKUPS) {
        const toRemove = backups.slice(MAX_BACKUPS);
        for (const backup of toRemove) {
          await backupsFolder.removeEntry(backup.name);
          log.debug(`Removed old backup: ${backup.name}`);
        }
      }
    } catch (e) {
      log.warn('Failed to cleanup old backups:', e);
    }
  }

  // ============================================
  // RENAME OPERATIONS
  // ============================================

  async renameProject(newName: string): Promise<boolean> {
    if (!this.projectHandle || !this.projectData) {
      log.error('No project open');
      return false;
    }

    const trimmedName = newName.trim();
    if (!trimmedName || trimmedName === this.projectData.name) {
      return false;
    }

    const invalidChars = /[<>:"/\\|?*]/;
    if (invalidChars.test(trimmedName)) {
      log.error('Invalid characters in project name');
      return false;
    }

    try {
      const parentHandle = await projectDB.getStoredHandle('projectsFolder');
      if (!parentHandle || parentHandle.kind !== 'directory') {
        // No parent folder stored - just update the display name in project.json
        log.info(`No parent folder handle, updating display name only to "${trimmedName}"`);
        this.projectData.name = trimmedName;
        this.projectData.updatedAt = new Date().toISOString();
        await this.writeProjectJson(this.projectHandle, this.projectData);
        this.isDirty = false;
        return true;
      }

      const parentDir = parentHandle as FileSystemDirectoryHandle;

      // Verify we have write permission on the parent
      const permission = await parentDir.queryPermission({ mode: 'readwrite' });
      if (permission !== 'granted') {
        // No permission on parent - just update display name in project.json
        log.info(`No write permission on parent folder, updating display name only to "${trimmedName}"`);
        this.projectData.name = trimmedName;
        this.projectData.updatedAt = new Date().toISOString();
        await this.writeProjectJson(this.projectHandle, this.projectData);
        this.isDirty = false;
        return true;
      }

      const oldName = this.projectHandle.name;

      // If the folder name already matches the new name, just update project data
      if (trimmedName === oldName) {
        this.projectData.name = trimmedName;
        this.projectData.updatedAt = new Date().toISOString();
        await this.writeProjectJson(this.projectHandle, this.projectData);
        this.isDirty = false;
        log.info(`Project display name updated to "${trimmedName}"`);
        return true;
      }

      // Check if a different folder with that name already exists
      let existingFolder: FileSystemDirectoryHandle | null = null;
      try {
        existingFolder = await parentDir.getDirectoryHandle(trimmedName, { create: false });
      } catch {
        // Good - folder doesn't exist
      }

      if (existingFolder) {
        // Check if the existing folder is a leftover (no project.json = not a real project)
        let hasProjectJson = false;
        try {
          await existingFolder.getFileHandle('project.json', { create: false });
          hasProjectJson = true;
        } catch {
          // No project.json
        }

        if (hasProjectJson) {
          log.error(`Folder "${trimmedName}" already contains a project`);
          return false;
        }

        // Leftover folder without project.json - remove it
        log.debug(`Removing leftover folder: ${trimmedName}`);
        try {
          await parentDir.removeEntry(trimmedName, { recursive: true });
        } catch (e) {
          log.error('Failed to remove leftover folder:', e);
          return false;
        }
      }

      const newFolder = await parentDir.getDirectoryHandle(trimmedName, { create: true });
      await this.copyDirectoryContents(this.projectHandle, newFolder);

      this.projectData.name = trimmedName;
      this.projectData.updatedAt = new Date().toISOString();
      await this.writeProjectJson(newFolder, this.projectData);

      this.projectHandle = newFolder;

      await projectDB.storeHandle('lastProject', newFolder);

      try {
        await parentDir.removeEntry(oldName, { recursive: true });
        log.debug(`Deleted old folder: ${oldName}`);
      } catch (e) {
        log.warn('Failed to delete old folder:', e);
      }

      this.isDirty = false;
      log.info(`Project renamed from "${oldName}" to "${trimmedName}"`);
      return true;
    } catch (e) {
      log.error('Failed to rename project:', e);
      return false;
    }
  }

  private async copyDirectoryContents(
    source: FileSystemDirectoryHandle,
    target: FileSystemDirectoryHandle
  ): Promise<void> {
    for await (const entry of (source as any).values()) {
      if (entry.kind === 'file') {
        const sourceFile = await entry.getFile();
        const targetFile = await target.getFileHandle(entry.name, { create: true });
        const writable = await targetFile.createWritable();
        await writable.write(sourceFile);
        await writable.close();
      } else if (entry.kind === 'directory') {
        const subDir = await target.getDirectoryHandle(entry.name, { create: true });
        await this.copyDirectoryContents(entry, subDir);
      }
    }
  }

  // ============================================
  // RESTORE OPERATIONS
  // ============================================

  async restoreLastProject(): Promise<boolean> {
    try {
      const handle = await projectDB.getStoredHandle('lastProject');
      if (!handle || handle.kind !== 'directory') return false;

      const permission = await handle.queryPermission({ mode: 'readwrite' });
      if (permission === 'granted') {
        const loaded = await this.loadProject(handle as FileSystemDirectoryHandle);

        if (!loaded) {
          log.info('Project not found, trying to recreate...');
          return await this.recreateProjectFromParent();
        }

        return loaded;
      } else {
        this.pendingHandle = handle as FileSystemDirectoryHandle;
        this.permissionNeeded = true;
        log.info('Permission needed for:', handle.name);
        return false;
      }
    } catch (e) {
      log.warn('Failed to restore last project:', e);
      return await this.recreateProjectFromParent();
    }
  }

  private async recreateProjectFromParent(): Promise<boolean> {
    try {
      const parentHandle = await projectDB.getStoredHandle('projectsFolder');
      if (!parentHandle || parentHandle.kind !== 'directory') {
        log.info('No parent folder stored, cannot recreate');
        await this.clearStoredHandles();
        return false;
      }

      const permission = await parentHandle.queryPermission({ mode: 'readwrite' });
      if (permission !== 'granted') {
        this.pendingHandle = parentHandle as FileSystemDirectoryHandle;
        this.permissionNeeded = true;
        log.info('Permission needed for parent folder');
        return false;
      }

      log.info('Recreating Untitled project...');
      const success = await this.createProjectInFolder(parentHandle as FileSystemDirectoryHandle, 'Untitled');
      if (success) {
        log.info('Successfully recreated Untitled project');
      }
      return success;
    } catch (e) {
      log.warn('Failed to recreate project from parent:', e);
      await this.clearStoredHandles();
      return false;
    }
  }

  private async clearStoredHandles(): Promise<void> {
    try {
      await projectDB.deleteHandle('lastProject');
      await projectDB.deleteHandle('projectsFolder');
      log.debug('Cleared stored handles');
    } catch (e) {
      log.warn('Failed to clear stored handles:', e);
    }
  }

  // ============================================
  // UPDATE OPERATIONS
  // ============================================

  updateProjectData(updates: Partial<ProjectFile>): void {
    if (!this.projectData) return;
    Object.assign(this.projectData, updates);
    this.markDirty();
  }

  updateMedia(media: ProjectMediaFile[]): void {
    if (!this.projectData) return;
    this.projectData.media = media;
    this.markDirty();
  }

  updateCompositions(compositions: ProjectComposition[]): void {
    if (!this.projectData) return;
    this.projectData.compositions = compositions;
    this.markDirty();
  }

  updateFolders(folders: ProjectFolder[]): void {
    if (!this.projectData) return;
    this.projectData.folders = folders;
    this.markDirty();
  }

  // ============================================
  // AUTO-SAVE
  // ============================================

  startAutoSave(): void {
    this.stopAutoSave();
    this.autoSaveInterval = window.setInterval(() => {
      if (this.isDirty) {
        this.saveProject();
      }
    }, 30000);
  }

  stopAutoSave(): void {
    if (this.autoSaveInterval !== null) {
      clearInterval(this.autoSaveInterval);
      this.autoSaveInterval = null;
    }
  }

  // ============================================
  // API KEYS FILE (.keys.enc)
  // ============================================

  async saveKeysFile(): Promise<void> {
    if (!this.projectHandle) return;

    try {
      const content = await apiKeyManager.exportKeysForFile();
      if (!content) {
        log.debug('No API keys to save to file');
        return;
      }

      const fileHandle = await this.projectHandle.getFileHandle(KEYS_FILE_NAME, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(content);
      await writable.close();

      log.debug('API keys saved to project file');
    } catch (e) {
      log.warn('Failed to save keys file:', e);
    }
  }

  async loadKeysFile(): Promise<boolean> {
    if (!this.projectHandle) return false;

    try {
      const fileHandle = await this.projectHandle.getFileHandle(KEYS_FILE_NAME);
      const file = await fileHandle.getFile();
      const content = await file.text();

      if (!content) return false;

      const restored = await apiKeyManager.importKeysFromFile(content);
      if (restored) {
        log.info('API keys restored from project file');
      }
      return restored;
    } catch {
      // File doesn't exist — that's fine
      return false;
    }
  }

  // ============================================
  // HELPERS
  // ============================================

  private async writeProjectJson(handle: FileSystemDirectoryHandle, data: ProjectFile): Promise<void> {
    const fileHandle = await handle.getFileHandle('project.json', { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(JSON.stringify(data, null, 2));
    await writable.close();
  }

  private async storeLastProject(handle: FileSystemDirectoryHandle): Promise<void> {
    try {
      await projectDB.storeHandle('lastProject', handle);
    } catch (e) {
      log.warn('Failed to store last project:', e);
    }
  }
}
