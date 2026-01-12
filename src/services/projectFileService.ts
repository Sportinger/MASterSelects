// Project File Service
// Handles all project file/folder operations on the local filesystem
// Uses File System Access API for full local storage

import { projectDB } from './projectDB';

// ============================================
// PROJECT STRUCTURE TYPES
// ============================================

/**
 * Analysis cache file structure (stored in Analysis/{mediaId}.json)
 */
interface StoredAnalysisFile {
  mediaFileId: string;
  analyses: {
    [rangeKey: string]: {
      frames: unknown[];
      sampleInterval: number;
      createdAt: number;
    };
  };
}

export interface ProjectFile {
  version: 1;
  name: string;
  createdAt: string;
  updatedAt: string;

  // Project settings
  settings: {
    width: number;
    height: number;
    frameRate: number;
    sampleRate: number;
  };

  // Media references (paths relative to project folder or absolute)
  media: ProjectMediaFile[];

  // Compositions (timelines)
  compositions: ProjectComposition[];

  // Folders for organization
  folders: ProjectFolder[];

  // Active state
  activeCompositionId: string | null;
  openCompositionIds: string[];
  expandedFolderIds: string[];
}

export interface ProjectMediaFile {
  id: string;
  name: string;
  type: 'video' | 'audio' | 'image';

  // Path to original file (absolute or relative to Raw/)
  sourcePath: string;

  // Metadata
  duration?: number;
  width?: number;
  height?: number;
  frameRate?: number;

  // Proxy status
  hasProxy: boolean;

  // Folder organization
  folderId: string | null;

  // Timestamps
  importedAt: string;
}

export interface ProjectComposition {
  id: string;
  name: string;
  width: number;
  height: number;
  frameRate: number;
  duration: number;
  backgroundColor: string;
  folderId: string | null;

  // Tracks and clips
  tracks: ProjectTrack[];
  clips: ProjectClip[];

  // Markers
  markers: ProjectMarker[];
}

export interface ProjectTrack {
  id: string;
  name: string;
  type: 'video' | 'audio';
  height: number;
  locked: boolean;
  visible: boolean;
  muted: boolean;
  solo: boolean;
}

export interface ProjectClip {
  id: string;
  trackId: string;
  mediaId: string; // Reference to ProjectMediaFile.id

  // Timeline position
  startTime: number;
  duration: number;

  // Source trimming
  inPoint: number;
  outPoint: number;

  // Transform
  transform: {
    x: number;
    y: number;
    z: number;
    scaleX: number;
    scaleY: number;
    rotation: number;
    rotationX: number;
    rotationY: number;
    anchorX: number;
    anchorY: number;
    opacity: number;
    blendMode: string;
  };

  // Effects
  effects: ProjectEffect[];

  // Masks
  masks: ProjectMask[];

  // Keyframes
  keyframes: ProjectKeyframe[];

  // Audio
  volume: number;
  audioEnabled: boolean;

  // Flags
  reversed: boolean;
  disabled: boolean;
}

export interface ProjectEffect {
  id: string;
  type: string;
  name: string;
  enabled: boolean;
  params: Record<string, number | boolean | string>;
}

export interface ProjectMask {
  id: string;
  name: string;
  mode: 'add' | 'subtract' | 'intersect';
  inverted: boolean;
  opacity: number;
  feather: number;
  featherQuality: number;
  visible: boolean;
  closed: boolean;
  vertices: Array<{
    x: number;
    y: number;
    inTangent: { x: number; y: number };
    outTangent: { x: number; y: number };
  }>;
  position: { x: number; y: number };
}

export interface ProjectKeyframe {
  id: string;
  property: string;
  time: number;
  value: number;
  easing: string;
  bezierHandles?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

export interface ProjectMarker {
  id: string;
  time: number;
  name: string;
  color: string;
  duration: number;
}

export interface ProjectFolder {
  id: string;
  name: string;
  parentId: string | null;
  color?: string;
}

// ============================================
// PROJECT FOLDER STRUCTURE
// ============================================

const PROJECT_FOLDERS = {
  RAW: 'Raw',
  PROXY: 'Proxy',
  ANALYSIS: 'Analysis',
  TRANSCRIPTS: 'Transcripts',
  CACHE: 'Cache',
  CACHE_THUMBNAILS: 'Cache/thumbnails',
  CACHE_WAVEFORMS: 'Cache/waveforms',
  RENDERS: 'Renders',
  BACKUPS: 'Backups',
} as const;

const MAX_BACKUPS = 20;

// ============================================
// PROJECT FILE SERVICE
// ============================================

class ProjectFileService {
  private projectHandle: FileSystemDirectoryHandle | null = null;
  private projectPath: string | null = null;
  private projectData: ProjectFile | null = null;
  private isDirty = false;
  private autoSaveInterval: number | null = null;
  private pendingHandle: FileSystemDirectoryHandle | null = null; // Handle waiting for permission
  private permissionNeeded = false;

  // Check if File System Access API is supported
  isSupported(): boolean {
    return 'showDirectoryPicker' in window && 'showSaveFilePicker' in window;
  }

  // Get current project handle
  getProjectHandle(): FileSystemDirectoryHandle | null {
    return this.projectHandle;
  }

  // Get current project data
  getProjectData(): ProjectFile | null {
    return this.projectData;
  }

  // Check if project is open
  isProjectOpen(): boolean {
    return this.projectHandle !== null && this.projectData !== null;
  }

  // Check if project has unsaved changes
  hasUnsavedChanges(): boolean {
    return this.isDirty;
  }

  // Mark project as dirty (has changes)
  markDirty(): void {
    this.isDirty = true;
  }

  // Check if permission is needed to restore last project
  needsPermission(): boolean {
    return this.permissionNeeded && this.pendingHandle !== null;
  }

  // Get pending project name (for UI)
  getPendingProjectName(): string | null {
    return this.pendingHandle?.name || null;
  }

  // Request permission for pending handle (must be called from user gesture)
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
      console.warn('[ProjectFile] Failed to request permission:', e);
    }
    return false;
  }

  // ============================================
  // PROJECT OPERATIONS
  // ============================================

  /**
   * Create a new project (asks user to pick folder)
   */
  async createProject(name: string): Promise<boolean> {
    if (!this.isSupported()) {
      console.error('[ProjectFile] File System Access API not supported');
      return false;
    }

    try {
      // Let user pick where to save the project
      const handle = await (window as any).showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'documents',
      });

      // Create project subfolder
      const projectFolder = await handle.getDirectoryHandle(name, { create: true });

      return await this.initializeProject(projectFolder, name);
    } catch (e: any) {
      if (e.name === 'AbortError') {
        return false; // User cancelled
      }
      console.error('[ProjectFile] Failed to create project:', e);
      return false;
    }
  }

  /**
   * Create a new project directly in the given folder handle
   * Used when user already selected a folder (e.g., from WelcomeOverlay)
   */
  async createProjectInFolder(handle: FileSystemDirectoryHandle, name: string): Promise<boolean> {
    if (!this.isSupported()) {
      console.error('[ProjectFile] File System Access API not supported');
      return false;
    }

    try {
      // Store the parent folder so we can recreate project if deleted
      await projectDB.storeHandle('projectsFolder', handle);

      // Create project subfolder inside the selected folder
      const projectFolder = await handle.getDirectoryHandle(name, { create: true });
      return await this.initializeProject(projectFolder, name);
    } catch (e: any) {
      console.error('[ProjectFile] Failed to create project in folder:', e);
      return false;
    }
  }

  /**
   * Initialize a project in the given folder (creates structure and project.json)
   */
  private async initializeProject(projectFolder: FileSystemDirectoryHandle, name: string): Promise<boolean> {
    try {
      // Create all subfolders
      await this.createProjectFolders(projectFolder);

      // Create Main Comp with unique ID
      const mainCompId = `comp-${Date.now()}`;

      // Create initial project.json with Main Comp open by default
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

      // Save project.json
      await this.writeProjectJson(projectFolder, initialProject);

      // Set as current project
      this.projectHandle = projectFolder;
      this.projectPath = name;
      this.projectData = initialProject;
      this.isDirty = false;

      // Store as last opened project
      await this.storeLastProject(projectFolder);

      // Start auto-save
      this.startAutoSave();

      console.log(`[ProjectFile] Created project: ${name}`);
      return true;
    } catch (e) {
      console.error('[ProjectFile] Failed to initialize project:', e);
      return false;
    }
  }

  /**
   * Open an existing project (by selecting folder)
   */
  async openProject(): Promise<boolean> {
    if (!this.isSupported()) {
      console.error('[ProjectFile] File System Access API not supported');
      return false;
    }

    try {
      // Let user pick the project folder
      const handle = await (window as any).showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'documents',
      });

      return await this.loadProject(handle);
    } catch (e: any) {
      if (e.name === 'AbortError') {
        return false; // User cancelled
      }
      console.error('[ProjectFile] Failed to open project:', e);
      return false;
    }
  }

  /**
   * Load project from a directory handle
   */
  async loadProject(handle: FileSystemDirectoryHandle): Promise<boolean> {
    try {
      // Read project.json
      const projectFile = await handle.getFileHandle('project.json');
      const file = await projectFile.getFile();
      const content = await file.text();
      const projectData = JSON.parse(content) as ProjectFile;

      // Validate version
      if (projectData.version !== 1) {
        console.error('[ProjectFile] Unsupported project version:', projectData.version);
        return false;
      }

      // Ensure all folders exist
      await this.createProjectFolders(handle);

      // Set as current project
      this.projectHandle = handle;
      this.projectPath = handle.name;
      this.projectData = projectData;
      this.isDirty = false;

      // Store as last opened project
      await this.storeLastProject(handle);

      // Start auto-save
      this.startAutoSave();

      console.log(`[ProjectFile] Opened project: ${projectData.name}`);
      return true;
    } catch (e) {
      console.error('[ProjectFile] Failed to load project:', e);
      return false;
    }
  }

  /**
   * Save the current project
   */
  async saveProject(): Promise<boolean> {
    if (!this.projectHandle || !this.projectData) {
      console.error('[ProjectFile] No project open');
      return false;
    }

    try {
      this.projectData.updatedAt = new Date().toISOString();
      await this.writeProjectJson(this.projectHandle, this.projectData);
      this.isDirty = false;
      console.log('[ProjectFile] Project saved');
      return true;
    } catch (e) {
      console.error('[ProjectFile] Failed to save project:', e);
      return false;
    }
  }

  /**
   * Close the current project
   */
  closeProject(): void {
    this.stopAutoSave();
    this.projectHandle = null;
    this.projectPath = null;
    this.projectData = null;
    this.isDirty = false;
    console.log('[ProjectFile] Project closed');
  }

  /**
   * Create a backup of the current project before saving
   * Keeps only the last MAX_BACKUPS backups
   */
  async createBackup(): Promise<boolean> {
    if (!this.projectHandle || !this.projectData) {
      return false;
    }

    try {
      // Read current project.json
      const projectFile = await this.projectHandle.getFileHandle('project.json');
      const file = await projectFile.getFile();
      const content = await file.text();

      // Create timestamp for backup filename
      const now = new Date();
      const timestamp = now.toISOString()
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .slice(0, 19); // Format: 2026-01-11_14-30-00
      const backupFileName = `project_${timestamp}.json`;

      // Get or create Backups folder
      const backupsFolder = await this.projectHandle.getDirectoryHandle(PROJECT_FOLDERS.BACKUPS, { create: true });

      // Write backup file
      const backupHandle = await backupsFolder.getFileHandle(backupFileName, { create: true });
      const writable = await backupHandle.createWritable();
      await writable.write(content);
      await writable.close();

      console.log(`[ProjectFile] Created backup: ${backupFileName}`);

      // Cleanup old backups
      await this.cleanupOldBackups(backupsFolder);

      return true;
    } catch (e) {
      console.error('[ProjectFile] Failed to create backup:', e);
      return false;
    }
  }

  /**
   * Remove old backups, keeping only the last MAX_BACKUPS
   */
  private async cleanupOldBackups(backupsFolder: FileSystemDirectoryHandle): Promise<void> {
    try {
      // List all backup files
      const backups: { name: string; file: File }[] = [];

      for await (const entry of (backupsFolder as any).values()) {
        if (entry.kind === 'file' && entry.name.startsWith('project_') && entry.name.endsWith('.json')) {
          const file = await entry.getFile();
          backups.push({ name: entry.name, file });
        }
      }

      // Sort by modification time (newest first)
      backups.sort((a, b) => b.file.lastModified - a.file.lastModified);

      // Remove old backups
      if (backups.length > MAX_BACKUPS) {
        const toRemove = backups.slice(MAX_BACKUPS);
        for (const backup of toRemove) {
          await backupsFolder.removeEntry(backup.name);
          console.log(`[ProjectFile] Removed old backup: ${backup.name}`);
        }
      }
    } catch (e) {
      console.warn('[ProjectFile] Failed to cleanup old backups:', e);
    }
  }

  // ============================================
  // FILE OPERATIONS
  // ============================================

  /**
   * Get a file handle from a project subfolder
   */
  async getFileHandle(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string,
    create = false
  ): Promise<FileSystemFileHandle | null> {
    if (!this.projectHandle) return null;

    try {
      const folderPath = PROJECT_FOLDERS[subFolder];
      let folder = this.projectHandle;

      // Navigate to subfolder
      for (const part of folderPath.split('/')) {
        folder = await folder.getDirectoryHandle(part, { create });
      }

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
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string,
    content: Blob | string
  ): Promise<boolean> {
    try {
      const handle = await this.getFileHandle(subFolder, fileName, true);
      if (!handle) return false;

      const writable = await handle.createWritable();
      await writable.write(content);
      await writable.close();
      return true;
    } catch (e) {
      console.error(`[ProjectFile] Failed to write ${subFolder}/${fileName}:`, e);
      return false;
    }
  }

  /**
   * Read a file from a project subfolder
   */
  async readFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<File | null> {
    try {
      const handle = await this.getFileHandle(subFolder, fileName);
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
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<boolean> {
    const handle = await this.getFileHandle(subFolder, fileName);
    return handle !== null;
  }

  /**
   * Delete a file from a project subfolder
   */
  async deleteFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<boolean> {
    if (!this.projectHandle) return false;

    try {
      const folderPath = PROJECT_FOLDERS[subFolder];
      let folder = this.projectHandle;

      for (const part of folderPath.split('/')) {
        folder = await folder.getDirectoryHandle(part);
      }

      await folder.removeEntry(fileName);
      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * List files in a project subfolder
   */
  async listFiles(subFolder: keyof typeof PROJECT_FOLDERS): Promise<string[]> {
    if (!this.projectHandle) return [];

    try {
      const folderPath = PROJECT_FOLDERS[subFolder];
      let folder = this.projectHandle;

      for (const part of folderPath.split('/')) {
        folder = await folder.getDirectoryHandle(part);
      }

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

  // ============================================
  // MEDIA OPERATIONS
  // ============================================

  /**
   * Import media file (creates reference, doesn't copy)
   */
  async importMediaFile(file: File, fileHandle?: FileSystemFileHandle): Promise<ProjectMediaFile | null> {
    if (!this.projectData) return null;

    const id = `media-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Determine file type
    let type: 'video' | 'audio' | 'image' = 'video';
    if (file.type.startsWith('audio/')) type = 'audio';
    else if (file.type.startsWith('image/')) type = 'image';

    // Get source path (if available from File System Access API)
    let sourcePath = file.name;
    if (fileHandle) {
      // Store the handle for later access
      // Note: We can't get the full path, but we can store the handle
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

    // Add to project
    this.projectData.media.push(mediaFile);
    this.markDirty();

    return mediaFile;
  }

  // ============================================
  // THUMBNAIL & WAVEFORM OPERATIONS
  // ============================================

  /**
   * Save thumbnail by file hash (for deduplication)
   */
  async saveThumbnail(fileHash: string, blob: Blob): Promise<boolean> {
    return this.writeFile('CACHE_THUMBNAILS', `${fileHash}.jpg`, blob);
  }

  /**
   * Get thumbnail by file hash
   */
  async getThumbnail(fileHash: string): Promise<Blob | null> {
    const file = await this.readFile('CACHE_THUMBNAILS', `${fileHash}.jpg`);
    return file;
  }

  /**
   * Check if thumbnail exists by file hash
   */
  async hasThumbnail(fileHash: string): Promise<boolean> {
    const thumb = await this.getThumbnail(fileHash);
    return thumb !== null && thumb.size > 0;
  }

  /**
   * Save waveform data for a media file
   */
  async saveWaveform(mediaId: string, waveformData: Float32Array): Promise<boolean> {
    const blob = new Blob([waveformData.buffer], { type: 'application/octet-stream' });
    return this.writeFile('CACHE_WAVEFORMS', `${mediaId}.waveform`, blob);
  }

  /**
   * Get waveform data for a media file
   */
  async getWaveform(mediaId: string): Promise<Float32Array | null> {
    const file = await this.readFile('CACHE_WAVEFORMS', `${mediaId}.waveform`);
    if (!file) return null;

    const buffer = await file.arrayBuffer();
    return new Float32Array(buffer);
  }

  // ============================================
  // PROXY OPERATIONS
  // ============================================

  /**
   * Save proxy frame
   */
  async saveProxyFrame(mediaId: string, frameIndex: number, blob: Blob): Promise<boolean> {
    if (!this.projectHandle) {
      console.error('[ProjectFile] No project handle for proxy save!');
      return false;
    }

    try {
      // Get or create media subfolder in Proxy/
      const proxyFolder = await this.projectHandle.getDirectoryHandle(PROJECT_FOLDERS.PROXY, { create: true });
      const mediaFolder = await proxyFolder.getDirectoryHandle(mediaId, { create: true });

      const fileName = `frame_${frameIndex.toString().padStart(6, '0')}.webp`;
      const fileHandle = await mediaFolder.getFileHandle(fileName, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(blob);
      await writable.close();

      if (frameIndex === 0 || frameIndex === 5) {
        console.log(`[ProjectFile] Saved proxy frame ${frameIndex} to ${this.projectHandle.name}/${PROJECT_FOLDERS.PROXY}/${mediaId}/${fileName} (${blob.size} bytes)`);
      }
      return true;
    } catch (e) {
      console.error('[ProjectFile] Failed to save proxy frame:', e);
      return false;
    }
  }

  /**
   * Get proxy frame
   */
  async getProxyFrame(mediaId: string, frameIndex: number): Promise<Blob | null> {
    if (!this.projectHandle) return null;

    try {
      const proxyFolder = await this.projectHandle.getDirectoryHandle(PROJECT_FOLDERS.PROXY);
      const mediaFolder = await proxyFolder.getDirectoryHandle(mediaId);
      const fileName = `frame_${frameIndex.toString().padStart(6, '0')}.webp`;
      const fileHandle = await mediaFolder.getFileHandle(fileName);
      return await fileHandle.getFile();
    } catch (e) {
      return null;
    }
  }

  /**
   * Check if proxy exists for media
   */
  async hasProxy(mediaId: string): Promise<boolean> {
    if (!this.projectHandle) return false;

    try {
      const proxyFolder = await this.projectHandle.getDirectoryHandle(PROJECT_FOLDERS.PROXY);
      await proxyFolder.getDirectoryHandle(mediaId);
      return true;
    } catch (e) {
      return false;
    }
  }

  // ============================================
  // ANALYSIS OPERATIONS (Range-based caching)
  // ============================================

  /**
   * Get range key for analysis caching (matches format: "inPoint-outPoint")
   */
  private getAnalysisRangeKey(inPoint: number, outPoint: number): string {
    return `${inPoint.toFixed(3)}-${outPoint.toFixed(3)}`;
  }

  /**
   * Analysis cache structure stored in file
   */
  private async getAnalysisRecord(mediaId: string): Promise<StoredAnalysisFile | null> {
    const file = await this.readFile('ANALYSIS', `${mediaId}.json`);
    if (!file) return null;

    try {
      const text = await file.text();
      return JSON.parse(text) as StoredAnalysisFile;
    } catch (e) {
      return null;
    }
  }

  /**
   * Save analysis data for a media file with range-based caching
   */
  async saveAnalysis(
    mediaId: string,
    inPoint: number,
    outPoint: number,
    frames: unknown[],
    sampleInterval: number
  ): Promise<boolean> {
    const rangeKey = this.getAnalysisRangeKey(inPoint, outPoint);

    // Get existing record or create new
    const existing = await this.getAnalysisRecord(mediaId);
    const record: StoredAnalysisFile = existing || {
      mediaFileId: mediaId,
      analyses: {},
    };

    // Add/update this range
    record.analyses[rangeKey] = {
      frames,
      sampleInterval,
      createdAt: Date.now(),
    };

    const json = JSON.stringify(record, null, 2);
    return this.writeFile('ANALYSIS', `${mediaId}.json`, json);
  }

  /**
   * Get analysis data for a specific time range
   */
  async getAnalysis(
    mediaId: string,
    inPoint: number,
    outPoint: number
  ): Promise<{ frames: unknown[]; sampleInterval: number } | null> {
    const record = await this.getAnalysisRecord(mediaId);
    if (!record) return null;

    const rangeKey = this.getAnalysisRangeKey(inPoint, outPoint);
    const analysis = record.analyses[rangeKey];

    if (!analysis) return null;
    return { frames: analysis.frames, sampleInterval: analysis.sampleInterval };
  }

  /**
   * Check if analysis exists for a specific time range
   */
  async hasAnalysis(mediaId: string, inPoint: number, outPoint: number): Promise<boolean> {
    const analysis = await this.getAnalysis(mediaId, inPoint, outPoint);
    return analysis !== null;
  }

  /**
   * Get all cached analysis ranges for a media file
   */
  async getAnalysisRanges(mediaId: string): Promise<string[]> {
    const record = await this.getAnalysisRecord(mediaId);
    if (!record) return [];
    return Object.keys(record.analyses);
  }

  /**
   * Delete all analysis for a media file
   */
  async deleteAnalysis(mediaId: string): Promise<boolean> {
    return this.deleteFile('ANALYSIS', `${mediaId}.json`);
  }

  // ============================================
  // TRANSCRIPT OPERATIONS
  // ============================================

  /**
   * Save transcript for a media file
   */
  async saveTranscript(mediaId: string, transcript: unknown): Promise<boolean> {
    const json = JSON.stringify(transcript, null, 2);
    return this.writeFile('TRANSCRIPTS', `${mediaId}.json`, json);
  }

  /**
   * Get transcript for a media file
   */
  async getTranscript(mediaId: string): Promise<unknown | null> {
    const file = await this.readFile('TRANSCRIPTS', `${mediaId}.json`);
    if (!file) return null;

    try {
      const text = await file.text();
      return JSON.parse(text);
    } catch (e) {
      return null;
    }
  }

  // ============================================
  // HELPER METHODS
  // ============================================

  /**
   * Create all project subfolders
   */
  private async createProjectFolders(handle: FileSystemDirectoryHandle): Promise<void> {
    const folders = [
      PROJECT_FOLDERS.RAW,
      PROJECT_FOLDERS.PROXY,
      PROJECT_FOLDERS.ANALYSIS,
      PROJECT_FOLDERS.TRANSCRIPTS,
      PROJECT_FOLDERS.CACHE,
      PROJECT_FOLDERS.CACHE_THUMBNAILS,
      PROJECT_FOLDERS.CACHE_WAVEFORMS,
      PROJECT_FOLDERS.RENDERS,
      PROJECT_FOLDERS.BACKUPS,
    ];

    for (const folderPath of folders) {
      let folder = handle;
      for (const part of folderPath.split('/')) {
        folder = await folder.getDirectoryHandle(part, { create: true });
      }
    }
  }

  /**
   * Write project.json to disk
   */
  private async writeProjectJson(handle: FileSystemDirectoryHandle, data: ProjectFile): Promise<void> {
    const fileHandle = await handle.getFileHandle('project.json', { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(JSON.stringify(data, null, 2));
    await writable.close();
  }

  /**
   * Store last opened project in IndexedDB
   */
  private async storeLastProject(handle: FileSystemDirectoryHandle): Promise<void> {
    try {
      await projectDB.storeHandle('lastProject', handle);
    } catch (e) {
      console.warn('[ProjectFile] Failed to store last project:', e);
    }
  }

  /**
   * Try to restore last opened project
   * Returns true if restored, false if not restored but permission might be needed
   */
  async restoreLastProject(): Promise<boolean> {
    try {
      const handle = await projectDB.getStoredHandle('lastProject');
      if (!handle || handle.kind !== 'directory') return false;

      // Check permission silently (no popup)
      const permission = await handle.queryPermission({ mode: 'readwrite' });
      if (permission === 'granted') {
        // Permission already granted, try to load project
        const loaded = await this.loadProject(handle as FileSystemDirectoryHandle);

        if (!loaded) {
          // Project doesn't exist (was deleted) - try to recreate from parent folder
          console.log('[ProjectFile] Project not found, trying to recreate...');
          return await this.recreateProjectFromParent();
        }

        return loaded;
      } else {
        // Permission not granted - store handle for later, show UI prompt
        this.pendingHandle = handle as FileSystemDirectoryHandle;
        this.permissionNeeded = true;
        console.log('[ProjectFile] Permission needed for:', handle.name);
        return false;
      }
    } catch (e) {
      console.warn('[ProjectFile] Failed to restore last project:', e);
      // Try to recreate from parent folder
      return await this.recreateProjectFromParent();
    }
  }

  /**
   * Recreate "Untitled" project from stored parent folder
   * Used when project was deleted but we still have the parent folder handle
   */
  private async recreateProjectFromParent(): Promise<boolean> {
    try {
      const parentHandle = await projectDB.getStoredHandle('projectsFolder');
      if (!parentHandle || parentHandle.kind !== 'directory') {
        console.log('[ProjectFile] No parent folder stored, cannot recreate');
        // Clear invalid stored handles so WelcomeOverlay shows
        await this.clearStoredHandles();
        return false;
      }

      // Check permission for parent folder
      const permission = await parentHandle.queryPermission({ mode: 'readwrite' });
      if (permission !== 'granted') {
        // Need permission - store for later
        this.pendingHandle = parentHandle as FileSystemDirectoryHandle;
        this.permissionNeeded = true;
        console.log('[ProjectFile] Permission needed for parent folder');
        return false;
      }

      // Recreate "Untitled" project
      console.log('[ProjectFile] Recreating Untitled project...');
      const success = await this.createProjectInFolder(parentHandle as FileSystemDirectoryHandle, 'Untitled');
      if (success) {
        console.log('[ProjectFile] Successfully recreated Untitled project');
      }
      return success;
    } catch (e) {
      console.warn('[ProjectFile] Failed to recreate project from parent:', e);
      // Clear invalid stored handles so WelcomeOverlay shows
      await this.clearStoredHandles();
      return false;
    }
  }

  /**
   * Clear stored handles (used when project can't be restored)
   * This will cause WelcomeOverlay to show on next check
   */
  private async clearStoredHandles(): Promise<void> {
    try {
      await projectDB.deleteHandle('lastProject');
      await projectDB.deleteHandle('projectsFolder');
      console.log('[ProjectFile] Cleared stored handles');
    } catch (e) {
      console.warn('[ProjectFile] Failed to clear stored handles:', e);
    }
  }

  /**
   * Start auto-save interval
   */
  private startAutoSave(): void {
    this.stopAutoSave();
    this.autoSaveInterval = window.setInterval(() => {
      if (this.isDirty) {
        this.saveProject();
      }
    }, 30000); // Auto-save every 30 seconds
  }

  /**
   * Stop auto-save interval
   */
  private stopAutoSave(): void {
    if (this.autoSaveInterval !== null) {
      clearInterval(this.autoSaveInterval);
      this.autoSaveInterval = null;
    }
  }

  // ============================================
  // RENAME PROJECT (INCLUDING FOLDER)
  // ============================================

  /**
   * Rename the project and its folder on disk
   * @param newName The new project name
   * @returns true if successful, false otherwise
   */
  async renameProject(newName: string): Promise<boolean> {
    if (!this.projectHandle || !this.projectData) {
      console.error('[ProjectFile] No project open');
      return false;
    }

    // Validate new name
    const trimmedName = newName.trim();
    if (!trimmedName || trimmedName === this.projectData.name) {
      return false;
    }

    // Check for invalid characters in folder name
    const invalidChars = /[<>:"/\\|?*]/;
    if (invalidChars.test(trimmedName)) {
      console.error('[ProjectFile] Invalid characters in project name');
      return false;
    }

    try {
      // Get the parent folder handle
      const parentHandle = await projectDB.getStoredHandle('projectsFolder');
      if (!parentHandle || parentHandle.kind !== 'directory') {
        // Try to get parent by other means - this shouldn't happen normally
        console.error('[ProjectFile] Cannot rename: parent folder handle not available');
        // Fall back to just renaming in project.json
        this.projectData.name = trimmedName;
        this.markDirty();
        await this.saveProject();
        return true;
      }

      const parentDir = parentHandle as FileSystemDirectoryHandle;

      // Check if new folder name already exists
      try {
        await parentDir.getDirectoryHandle(trimmedName, { create: false });
        console.error('[ProjectFile] Folder with that name already exists');
        return false;
      } catch {
        // Good - folder doesn't exist
      }

      const oldName = this.projectHandle.name;

      // Create new folder
      const newFolder = await parentDir.getDirectoryHandle(trimmedName, { create: true });

      // Copy all contents recursively
      await this.copyDirectoryContents(this.projectHandle, newFolder);

      // Update project.json in the new folder with new name
      this.projectData.name = trimmedName;
      this.projectData.updatedAt = new Date().toISOString();
      await this.writeProjectJson(newFolder, this.projectData);

      // Update our handle to point to new folder
      const oldHandle = this.projectHandle;
      this.projectHandle = newFolder;
      this.projectPath = trimmedName;

      // Update stored handles
      await projectDB.storeHandle('lastProject', newFolder);

      // Delete old folder (after everything is copied)
      try {
        await parentDir.removeEntry(oldName, { recursive: true });
        console.log(`[ProjectFile] Deleted old folder: ${oldName}`);
      } catch (e) {
        console.warn('[ProjectFile] Failed to delete old folder:', e);
        // Not critical - new folder is already working
      }

      this.isDirty = false;
      console.log(`[ProjectFile] Project renamed from "${oldName}" to "${trimmedName}"`);
      return true;
    } catch (e) {
      console.error('[ProjectFile] Failed to rename project:', e);
      return false;
    }
  }

  /**
   * Copy all contents from one directory to another (recursive)
   */
  private async copyDirectoryContents(
    source: FileSystemDirectoryHandle,
    target: FileSystemDirectoryHandle
  ): Promise<void> {
    for await (const entry of (source as any).values()) {
      if (entry.kind === 'file') {
        // Copy file
        const sourceFile = await entry.getFile();
        const targetFile = await target.getFileHandle(entry.name, { create: true });
        const writable = await targetFile.createWritable();
        await writable.write(sourceFile);
        await writable.close();
      } else if (entry.kind === 'directory') {
        // Create subdirectory and copy contents recursively
        const subDir = await target.getDirectoryHandle(entry.name, { create: true });
        await this.copyDirectoryContents(entry, subDir);
      }
    }
  }

  // ============================================
  // UPDATE PROJECT DATA
  // ============================================

  /**
   * Update project data (from stores)
   */
  updateProjectData(updates: Partial<ProjectFile>): void {
    if (!this.projectData) return;
    Object.assign(this.projectData, updates);
    this.markDirty();
  }

  /**
   * Update media list
   */
  updateMedia(media: ProjectMediaFile[]): void {
    if (!this.projectData) return;
    this.projectData.media = media;
    this.markDirty();
  }

  /**
   * Update compositions
   */
  updateCompositions(compositions: ProjectComposition[]): void {
    if (!this.projectData) return;
    this.projectData.compositions = compositions;
    this.markDirty();
  }

  /**
   * Update folders
   */
  updateFolders(folders: ProjectFolder[]): void {
    if (!this.projectData) return;
    this.projectData.folders = folders;
    this.markDirty();
  }
}

// Singleton instance
export const projectFileService = new ProjectFileService();
