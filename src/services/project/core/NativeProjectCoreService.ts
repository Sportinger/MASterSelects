// Project lifecycle management via Native Helper
// Mirrors ProjectCoreService but uses string paths + NativeHelperClient
// instead of FileSystemDirectoryHandle. Enables project persistence in Firefox.

import { Logger } from '../../logger';
import { apiKeyManager } from '../../apiKeyManager';
import { NativeHelperClient } from '../../nativeHelper/NativeHelperClient';
import { PROJECT_FOLDER_PATHS, MAX_BACKUPS } from './constants';
import type { ProjectFile, ProjectMediaFile, ProjectComposition, ProjectFolder } from '../types';

const log = Logger.create('NativeProjectCore');

const KEYS_FILE_NAME = '.keys.enc';
const LAST_PROJECT_KEY = 'ms-native-last-project-path';

export class NativeProjectCoreService {
  private projectPath: string | null = null;
  private projectData: ProjectFile | null = null;
  private isDirty = false;
  private autoSaveInterval: number | null = null;
  private client = NativeHelperClient;

  // ============================================
  // GETTERS & STATE CHECKS
  // ============================================

  isSupported(): boolean {
    return this.client.isConnected();
  }

  getProjectPath(): string | null {
    return this.projectPath;
  }

  /** Compatibility: returns null since we don't use FSA handles */
  getProjectHandle(): null {
    return null;
  }

  getProjectData(): ProjectFile | null {
    return this.projectData;
  }

  isProjectOpen(): boolean {
    return this.projectPath !== null && this.projectData !== null;
  }

  hasUnsavedChanges(): boolean {
    return this.isDirty;
  }

  markDirty(): void {
    this.isDirty = true;
  }

  /** Not needed for native mode — no permission prompts */
  needsPermission(): boolean {
    return false;
  }

  getPendingProjectName(): string | null {
    return null;
  }

  async requestPendingPermission(): Promise<boolean> {
    return false;
  }

  // ============================================
  // PATH HELPERS
  // ============================================

  private joinPath(...parts: string[]): string {
    // Normalize to forward slashes, then join
    return parts
      .map(p => p.replace(/\\/g, '/').replace(/\/+$/, ''))
      .join('/');
  }

  // ============================================
  // PROJECT OPERATIONS
  // ============================================

  async createProject(name: string): Promise<boolean> {
    if (!this.client.isConnected()) {
      log.error('Native Helper not connected');
      return false;
    }

    try {
      const projectRoot = await this.client.getProjectRoot();
      if (!projectRoot) {
        log.error('Cannot determine project root');
        return false;
      }

      const projectPath = this.joinPath(projectRoot, name);
      return await this.initializeProject(projectPath, name);
    } catch (e) {
      log.error('Failed to create project:', e);
      return false;
    }
  }

  async createProjectAtPath(basePath: string, name: string): Promise<boolean> {
    if (!this.client.isConnected()) {
      log.error('Native Helper not connected');
      return false;
    }

    try {
      const projectPath = this.joinPath(basePath, name);
      return await this.initializeProject(projectPath, name);
    } catch (e) {
      log.error('Failed to create project at path:', e);
      return false;
    }
  }

  private async initializeProject(projectPath: string, name: string): Promise<boolean> {
    try {
      // Create project folder
      if (!await this.client.createDir(projectPath)) {
        log.error('Failed to create project directory');
        return false;
      }

      // Create subfolder structure
      for (const folderPath of PROJECT_FOLDER_PATHS) {
        await this.client.createDir(this.joinPath(projectPath, folderPath));
      }

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

      // Write project.json
      const jsonPath = this.joinPath(projectPath, 'project.json');
      if (!await this.client.writeFile(jsonPath, JSON.stringify(initialProject, null, 2))) {
        log.error('Failed to write project.json');
        return false;
      }

      this.projectPath = projectPath;
      this.projectData = initialProject;
      this.isDirty = false;

      this.storeLastProject(projectPath);
      this.startAutoSave();

      // Save any existing API keys
      await this.saveKeysFile();

      log.info(`Created project: ${name} at ${projectPath}`);
      return true;
    } catch (e) {
      log.error('Failed to initialize project:', e);
      return false;
    }
  }

  async loadProject(projectPath: string): Promise<boolean> {
    try {
      const jsonPath = this.joinPath(projectPath, 'project.json');
      const content = await this.client.readFileText(jsonPath);

      if (!content) {
        log.error('Cannot read project.json at', projectPath);
        return false;
      }

      const projectData = JSON.parse(content) as ProjectFile;

      if (projectData.version !== 1) {
        log.error('Unsupported project version:', projectData.version);
        return false;
      }

      // Ensure folder structure exists
      for (const folderPath of PROJECT_FOLDER_PATHS) {
        await this.client.createDir(this.joinPath(projectPath, folderPath));
      }

      this.projectPath = projectPath;
      this.projectData = projectData;
      this.isDirty = false;

      this.storeLastProject(projectPath);
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
    if (!this.projectPath || !this.projectData) {
      log.error('No project open');
      return false;
    }

    try {
      this.projectData.updatedAt = new Date().toISOString();
      const jsonPath = this.joinPath(this.projectPath, 'project.json');

      if (!await this.client.writeFile(jsonPath, JSON.stringify(this.projectData, null, 2))) {
        log.error('Failed to write project.json');
        return false;
      }

      this.isDirty = false;
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
    this.projectPath = null;
    this.projectData = null;
    this.isDirty = false;
    log.info('Project closed');
  }

  // ============================================
  // BACKUP OPERATIONS
  // ============================================

  async createBackup(): Promise<boolean> {
    if (!this.projectPath || !this.projectData) {
      return false;
    }

    try {
      const jsonPath = this.joinPath(this.projectPath, 'project.json');
      const content = await this.client.readFileText(jsonPath);
      if (!content) return false;

      const now = new Date();
      const timestamp = now.toISOString()
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .slice(0, 19);
      const backupFileName = `project_${timestamp}.json`;
      const backupPath = this.joinPath(this.projectPath, 'Backups', backupFileName);

      if (!await this.client.writeFile(backupPath, content)) {
        return false;
      }

      log.debug(`Created backup: ${backupFileName}`);
      await this.cleanupOldBackups();

      return true;
    } catch (e) {
      log.error('Failed to create backup:', e);
      return false;
    }
  }

  private async cleanupOldBackups(): Promise<void> {
    if (!this.projectPath) return;

    try {
      const backupsDir = this.joinPath(this.projectPath, 'Backups');
      const entries = await this.client.listDir(backupsDir);

      const backups = entries
        .filter(e => e.kind === 'file' && e.name.startsWith('project_') && e.name.endsWith('.json'))
        .sort((a, b) => b.modified - a.modified);

      if (backups.length > MAX_BACKUPS) {
        const toRemove = backups.slice(MAX_BACKUPS);
        for (const backup of toRemove) {
          await this.client.deleteFile(this.joinPath(backupsDir, backup.name));
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
    if (!this.projectPath || !this.projectData) {
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
      // Get parent directory
      const parts = this.projectPath.replace(/\\/g, '/').split('/');
      parts.pop(); // Remove current folder name
      const parentPath = parts.join('/');
      const newPath = this.joinPath(parentPath, trimmedName);

      // Check if destination already exists
      const { exists } = await this.client.exists(newPath);
      if (exists) {
        // Check if it has a project.json
        const { exists: hasProject } = await this.client.exists(this.joinPath(newPath, 'project.json'));
        if (hasProject) {
          log.error(`Folder "${trimmedName}" already contains a project`);
          return false;
        }
        // Leftover folder without project.json — remove it
        await this.client.deleteFile(newPath, true);
      }

      // Rename the folder
      if (!await this.client.rename(this.projectPath, newPath)) {
        // Rename failed — just update display name in project.json
        log.info(`Cannot rename folder, updating display name only to "${trimmedName}"`);
        this.projectData.name = trimmedName;
        this.projectData.updatedAt = new Date().toISOString();
        await this.saveProject();
        return true;
      }

      this.projectPath = newPath;
      this.projectData.name = trimmedName;
      this.projectData.updatedAt = new Date().toISOString();

      // Write updated project.json to new location
      const jsonPath = this.joinPath(newPath, 'project.json');
      await this.client.writeFile(jsonPath, JSON.stringify(this.projectData, null, 2));

      this.storeLastProject(newPath);
      this.isDirty = false;

      log.info(`Project renamed to "${trimmedName}"`);
      return true;
    } catch (e) {
      log.error('Failed to rename project:', e);
      return false;
    }
  }

  // ============================================
  // RESTORE OPERATIONS
  // ============================================

  async restoreLastProject(): Promise<boolean> {
    const lastPath = localStorage.getItem(LAST_PROJECT_KEY);
    if (!lastPath) return false;

    if (!this.client.isConnected()) {
      log.debug('Native Helper not connected, cannot restore project');
      return false;
    }

    try {
      const { exists, kind } = await this.client.exists(lastPath);
      if (!exists || kind !== 'directory') {
        log.info('Last project folder no longer exists');
        localStorage.removeItem(LAST_PROJECT_KEY);
        return false;
      }

      return await this.loadProject(lastPath);
    } catch (e) {
      log.warn('Failed to restore last project:', e);
      return false;
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
    if (!this.projectPath) return;

    try {
      const content = await apiKeyManager.exportKeysForFile();
      if (!content) {
        log.debug('No API keys to save to file');
        return;
      }

      const keysPath = this.joinPath(this.projectPath, KEYS_FILE_NAME);
      await this.client.writeFile(keysPath, content);

      log.debug('API keys saved to project file');
    } catch (e) {
      log.warn('Failed to save keys file:', e);
    }
  }

  async loadKeysFile(): Promise<boolean> {
    if (!this.projectPath) return false;

    try {
      const keysPath = this.joinPath(this.projectPath, KEYS_FILE_NAME);
      const content = await this.client.readFileText(keysPath);
      if (!content) return false;

      const restored = await apiKeyManager.importKeysFromFile(content);
      if (restored) {
        log.info('API keys restored from project file');
      }
      return restored;
    } catch {
      return false;
    }
  }

  // ============================================
  // HELPERS
  // ============================================

  private storeLastProject(path: string): void {
    try {
      localStorage.setItem(LAST_PROJECT_KEY, path);
    } catch (e) {
      log.warn('Failed to store last project path:', e);
    }
  }

  // ============================================
  // PROJECT LISTING (for project picker UI)
  // ============================================

  /**
   * List all projects in the default project root
   */
  async listProjects(): Promise<Array<{ name: string; path: string; modified: number }>> {
    try {
      const projectRoot = await this.client.getProjectRoot();
      if (!projectRoot) return [];

      // Ensure root exists
      await this.client.createDir(projectRoot);

      const entries = await this.client.listDir(projectRoot);
      const projects: Array<{ name: string; path: string; modified: number }> = [];

      for (const entry of entries) {
        if (entry.kind !== 'directory') continue;

        // Check if this directory has a project.json
        const projectJsonPath = this.joinPath(projectRoot, entry.name, 'project.json');
        const { exists } = await this.client.exists(projectJsonPath);

        if (exists) {
          projects.push({
            name: entry.name,
            path: this.joinPath(projectRoot, entry.name),
            modified: entry.modified,
          });
        }
      }

      // Sort by most recently modified first
      projects.sort((a, b) => b.modified - a.modified);
      return projects;
    } catch (e) {
      log.error('Failed to list projects:', e);
      return [];
    }
  }
}
