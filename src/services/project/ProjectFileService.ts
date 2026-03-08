// Project File Service Facade
// Delegates to domain services while maintaining the original API for backward compatibility
// Supports two backends: FSA (Chrome) and Native Helper (Firefox)

import { Logger } from '../logger';
import { FileStorageService, fileStorageService } from './core/FileStorageService';
import { NativeFileStorageService, nativeFileStorageService } from './core/NativeFileStorageService';
import { NativeProjectCoreService } from './core/NativeProjectCoreService';

const log = Logger.create('ProjectFileService');
import { ProjectCoreService } from './core/ProjectCoreService';
import { AnalysisService } from './domains/AnalysisService';
import { TranscriptService } from './domains/TranscriptService';
import { CacheService } from './domains/CacheService';
import { ProxyStorageService } from './domains/ProxyStorageService';
import { RawMediaService } from './domains/RawMediaService';
import { PROJECT_FOLDERS, type ProjectFolderKey } from './core/constants';
import type { ProjectFile, ProjectMediaFile, ProjectComposition, ProjectFolder } from './types';

export type ProjectBackend = 'fsa' | 'native';

class ProjectFileService {
  // Domain services
  private readonly coreService: ProjectCoreService;
  private readonly fileStorage: FileStorageService;
  private readonly analysisService: AnalysisService;
  private readonly transcriptService: TranscriptService;
  private readonly cacheService: CacheService;
  private readonly proxyStorageService: ProxyStorageService;
  private readonly rawMediaService: RawMediaService;

  // Native Helper backend (lazy-initialized)
  private nativeCoreService: NativeProjectCoreService | null = null;
  private nativeFileStorage: NativeFileStorageService | null = null;
  private _activeBackend: ProjectBackend = 'fsa';

  constructor() {
    this.fileStorage = fileStorageService;
    this.coreService = new ProjectCoreService(this.fileStorage);
    this.analysisService = new AnalysisService(this.fileStorage);
    this.transcriptService = new TranscriptService(this.fileStorage);
    this.cacheService = new CacheService(this.fileStorage);
    this.proxyStorageService = new ProxyStorageService();
    this.rawMediaService = new RawMediaService(this.fileStorage);
  }

  // ============================================
  // BACKEND SELECTION
  // ============================================

  /** Get the currently active backend */
  get activeBackend(): ProjectBackend {
    return this._activeBackend;
  }

  /** Check if FSA (File System Access API) is available */
  get isFsaAvailable(): boolean {
    return 'showDirectoryPicker' in window && 'showSaveFilePicker' in window;
  }

  /** Switch to native helper backend (for Firefox) */
  activateNativeBackend(): void {
    if (!this.nativeCoreService) {
      this.nativeCoreService = new NativeProjectCoreService();
      this.nativeFileStorage = nativeFileStorageService;
    }
    this._activeBackend = 'native';
    log.info('Switched to Native Helper backend');
  }

  /** Switch back to FSA backend (for Chrome) */
  activateFsaBackend(): void {
    this._activeBackend = 'fsa';
    log.info('Switched to FSA backend');
  }

  /** Get native core service (for native-specific operations like listProjects) */
  getNativeCoreService(): NativeProjectCoreService | null {
    return this.nativeCoreService;
  }

  /** Get native file storage (for native-specific operations like getFileUrl) */
  getNativeFileStorage(): NativeFileStorageService | null {
    return this.nativeFileStorage;
  }

  // ============================================
  // CORE SERVICE DELEGATION (routes to FSA or Native)
  // ============================================

  /** Helper to get the active core service */
  private get core(): ProjectCoreService | NativeProjectCoreService {
    if (this._activeBackend === 'native' && this.nativeCoreService) {
      return this.nativeCoreService;
    }
    return this.coreService;
  }

  isSupported(): boolean {
    if (this._activeBackend === 'native') {
      return this.nativeCoreService?.isSupported() ?? false;
    }
    return this.coreService.isSupported();
  }

  getProjectHandle(): FileSystemDirectoryHandle | null {
    // Only FSA backend has a handle
    if (this._activeBackend === 'fsa') {
      return this.coreService.getProjectHandle();
    }
    return null;
  }

  /** Get project path (native backend) or null */
  getProjectPath(): string | null {
    if (this._activeBackend === 'native' && this.nativeCoreService) {
      return this.nativeCoreService.getProjectPath();
    }
    return null;
  }

  getProjectData(): ProjectFile | null {
    return this.core.getProjectData();
  }

  isProjectOpen(): boolean {
    return this.core.isProjectOpen();
  }

  hasUnsavedChanges(): boolean {
    return this.core.hasUnsavedChanges();
  }

  markDirty(): void {
    this.core.markDirty();
  }

  needsPermission(): boolean {
    return this.core.needsPermission();
  }

  getPendingProjectName(): string | null {
    return this.core.getPendingProjectName();
  }

  async requestPendingPermission(): Promise<boolean> {
    return this.core.requestPendingPermission();
  }

  async createProject(name: string): Promise<boolean> {
    return this.core.createProject(name);
  }

  async createProjectInFolder(handle: FileSystemDirectoryHandle, name: string): Promise<boolean> {
    // Only FSA supports this
    return this.coreService.createProjectInFolder(handle, name);
  }

  async openProject(): Promise<boolean> {
    if (this._activeBackend === 'fsa') {
      return this.coreService.openProject();
    }
    // Native backend doesn't have a directory picker — use loadProject with a path
    log.warn('openProject() called on native backend — use loadProject(path) instead');
    return false;
  }

  async loadProject(handleOrPath: FileSystemDirectoryHandle | string): Promise<boolean> {
    if (typeof handleOrPath === 'string') {
      // Native path
      if (this.nativeCoreService) {
        return this.nativeCoreService.loadProject(handleOrPath);
      }
      return false;
    }
    // FSA handle
    return this.coreService.loadProject(handleOrPath);
  }

  async saveProject(): Promise<boolean> {
    return this.core.saveProject();
  }

  closeProject(): void {
    this.core.closeProject();
  }

  async createBackup(): Promise<boolean> {
    return this.core.createBackup();
  }

  async renameProject(newName: string): Promise<boolean> {
    return this.core.renameProject(newName);
  }

  async restoreLastProject(): Promise<boolean> {
    return this.core.restoreLastProject();
  }

  async saveKeysFile(): Promise<void> {
    return this.core.saveKeysFile();
  }

  async loadKeysFile(): Promise<boolean> {
    return this.core.loadKeysFile();
  }

  updateProjectData(updates: Partial<ProjectFile>): void {
    this.core.updateProjectData(updates);
  }

  updateMedia(media: ProjectMediaFile[]): void {
    this.core.updateMedia(media);
  }

  updateCompositions(compositions: ProjectComposition[]): void {
    this.core.updateCompositions(compositions);
  }

  updateFolders(folders: ProjectFolder[]): void {
    this.core.updateFolders(folders);
  }

  // ============================================
  // FILE STORAGE DELEGATION (routes to FSA or Native)
  // ============================================

  async getFileHandle(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string,
    create = false
  ): Promise<FileSystemFileHandle | null> {
    // Only FSA backend returns file handles
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.fileStorage.getFileHandle(handle, subFolder as ProjectFolderKey, fileName, create);
  }

  async writeFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string,
    content: Blob | string
  ): Promise<boolean> {
    if (this._activeBackend === 'native' && this.nativeFileStorage && this.nativeCoreService) {
      const path = this.nativeCoreService.getProjectPath();
      if (!path) return false;
      return this.nativeFileStorage.writeFile(path, subFolder as ProjectFolderKey, fileName, content);
    }
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.fileStorage.writeFile(handle, subFolder as ProjectFolderKey, fileName, content);
  }

  async readFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<File | null> {
    if (this._activeBackend === 'native' && this.nativeFileStorage && this.nativeCoreService) {
      // Native backend: read via HTTP and wrap in File object
      const path = this.nativeCoreService.getProjectPath();
      if (!path) return null;
      const buffer = await this.nativeFileStorage.readFileBinary(path, subFolder as ProjectFolderKey, fileName);
      if (!buffer) return null;
      return new File([buffer], fileName);
    }
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.fileStorage.readFile(handle, subFolder as ProjectFolderKey, fileName);
  }

  async fileExists(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<boolean> {
    if (this._activeBackend === 'native' && this.nativeFileStorage && this.nativeCoreService) {
      const path = this.nativeCoreService.getProjectPath();
      if (!path) return false;
      return this.nativeFileStorage.fileExists(path, subFolder as ProjectFolderKey, fileName);
    }
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.fileStorage.fileExists(handle, subFolder as ProjectFolderKey, fileName);
  }

  async deleteFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<boolean> {
    if (this._activeBackend === 'native' && this.nativeFileStorage && this.nativeCoreService) {
      const path = this.nativeCoreService.getProjectPath();
      if (!path) return false;
      return this.nativeFileStorage.deleteFile(path, subFolder as ProjectFolderKey, fileName);
    }
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.fileStorage.deleteFile(handle, subFolder as ProjectFolderKey, fileName);
  }

  async listFiles(subFolder: keyof typeof PROJECT_FOLDERS): Promise<string[]> {
    if (this._activeBackend === 'native' && this.nativeFileStorage && this.nativeCoreService) {
      const path = this.nativeCoreService.getProjectPath();
      if (!path) return [];
      return this.nativeFileStorage.listFiles(path, subFolder as ProjectFolderKey);
    }
    const handle = this.coreService.getProjectHandle();
    if (!handle) return [];
    return this.fileStorage.listFiles(handle, subFolder as ProjectFolderKey);
  }

  // ============================================
  // RAW MEDIA SERVICE DELEGATION
  // ============================================

  async copyToRawFolder(file: File, fileName?: string): Promise<{ handle: FileSystemFileHandle; relativePath: string; alreadyExisted: boolean } | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) {
      log.warn('No project open, cannot copy to Raw folder');
      return null;
    }
    return this.rawMediaService.copyToRawFolder(handle, file, fileName);
  }

  async getFileFromRaw(relativePath: string): Promise<{ file: File; handle: FileSystemFileHandle } | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.rawMediaService.getFileFromRaw(handle, relativePath);
  }

  async hasFileInRaw(fileName: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.rawMediaService.hasFileInRaw(handle, fileName);
  }

  async scanRawFolder(): Promise<Map<string, FileSystemFileHandle>> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return new Map();
    return this.rawMediaService.scanRawFolder(handle);
  }

  async importMediaFile(file: File, fileHandle?: FileSystemFileHandle): Promise<ProjectMediaFile | null> {
    const projectData = this.coreService.getProjectData();
    if (!projectData) return null;

    const mediaFile = await this.rawMediaService.importMediaFile(file, fileHandle);
    if (!mediaFile) return null;

    // Add to project
    projectData.media.push(mediaFile);
    this.coreService.markDirty();

    return mediaFile;
  }

  async saveDownload(blob: Blob, title: string, platform: string): Promise<File | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) {
      log.warn('No project open, cannot save download to project');
      return null;
    }
    return this.rawMediaService.saveDownload(handle, blob, title, platform);
  }

  async checkDownloadExists(title: string, platform: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.rawMediaService.checkDownloadExists(handle, title, platform);
  }

  async getDownloadFile(title: string, platform: string): Promise<File | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.rawMediaService.getDownloadFile(handle, title, platform);
  }

  // ============================================
  // CACHE SERVICE DELEGATION
  // ============================================

  async saveThumbnail(fileHash: string, blob: Blob): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.cacheService.saveThumbnail(handle, fileHash, blob);
  }

  async getThumbnail(fileHash: string): Promise<Blob | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.cacheService.getThumbnail(handle, fileHash);
  }

  async hasThumbnail(fileHash: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.cacheService.hasThumbnail(handle, fileHash);
  }

  async saveWaveform(mediaId: string, waveformData: Float32Array): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.cacheService.saveWaveform(handle, mediaId, waveformData);
  }

  async getWaveform(mediaId: string): Promise<Float32Array | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.cacheService.getWaveform(handle, mediaId);
  }

  // ============================================
  // PROXY STORAGE SERVICE DELEGATION
  // ============================================

  async saveProxyFrame(mediaId: string, frameIndex: number, blob: Blob): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) {
      log.error('No project handle for proxy save!');
      return false;
    }
    return this.proxyStorageService.saveProxyFrame(handle, mediaId, frameIndex, blob);
  }

  async getProxyFrame(mediaId: string, frameIndex: number): Promise<Blob | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.proxyStorageService.getProxyFrame(handle, mediaId, frameIndex);
  }

  async hasProxy(mediaId: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.proxyStorageService.hasProxy(handle, mediaId);
  }

  async getProxyFrameCount(mediaId: string): Promise<number> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return 0;
    return this.proxyStorageService.getProxyFrameCount(handle, mediaId);
  }

  async getProxyFrameIndices(mediaId: string): Promise<Set<number>> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return new Set();
    return this.proxyStorageService.getProxyFrameIndices(handle, mediaId);
  }

  async saveProxyVideo(mediaId: string, blob: Blob): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) {
      log.error('No project handle for proxy video save!');
      return false;
    }
    return this.proxyStorageService.saveProxyVideo(handle, mediaId, blob);
  }

  async getProxyVideo(mediaId: string): Promise<File | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.proxyStorageService.getProxyVideo(handle, mediaId);
  }

  async hasProxyVideo(mediaId: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.proxyStorageService.hasProxyVideo(handle, mediaId);
  }

  async saveProxyAudio(mediaId: string, blob: Blob): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) {
      log.error('No project handle for audio proxy save!');
      return false;
    }
    return this.proxyStorageService.saveProxyAudio(handle, mediaId, blob);
  }

  async getProxyAudio(mediaId: string): Promise<File | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.proxyStorageService.getProxyAudio(handle, mediaId);
  }

  async hasProxyAudio(mediaId: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.proxyStorageService.hasProxyAudio(handle, mediaId);
  }

  // ============================================
  // ANALYSIS SERVICE DELEGATION
  // ============================================

  async saveAnalysis(
    mediaId: string,
    inPoint: number,
    outPoint: number,
    frames: unknown[],
    sampleInterval: number
  ): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.analysisService.saveAnalysis(handle, mediaId, inPoint, outPoint, frames, sampleInterval);
  }

  async getAnalysis(
    mediaId: string,
    inPoint: number,
    outPoint: number
  ): Promise<{ frames: unknown[]; sampleInterval: number } | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.analysisService.getAnalysis(handle, mediaId, inPoint, outPoint);
  }

  async hasAnalysis(mediaId: string, inPoint: number, outPoint: number): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.analysisService.hasAnalysis(handle, mediaId, inPoint, outPoint);
  }

  async getAnalysisRanges(mediaId: string): Promise<string[]> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return [];
    return this.analysisService.getAnalysisRanges(handle, mediaId);
  }

  async getAllAnalysisMerged(mediaId: string): Promise<{ frames: unknown[]; sampleInterval: number } | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.analysisService.getAllAnalysisMerged(handle, mediaId);
  }

  async deleteAnalysis(mediaId: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.analysisService.deleteAnalysis(handle, mediaId);
  }

  // ============================================
  // TRANSCRIPT SERVICE DELEGATION
  // ============================================

  async saveTranscript(mediaId: string, transcript: unknown, transcribedRanges?: [number, number][]): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.transcriptService.saveTranscript(handle, mediaId, transcript, transcribedRanges);
  }

  async getTranscript(mediaId: string): Promise<{ words: unknown[]; transcribedRanges?: [number, number][] } | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.transcriptService.getTranscript(handle, mediaId);
  }

  async getTranscribedRanges(mediaId: string): Promise<[number, number][]> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return [];
    return this.transcriptService.getTranscribedRanges(handle, mediaId);
  }
}

// Singleton instance
export const projectFileService = new ProjectFileService();
