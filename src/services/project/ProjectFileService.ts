// Project File Service Facade
// Delegates to domain services while maintaining the original API for backward compatibility

import { Logger } from '../logger';
import { FileStorageService, fileStorageService } from './core/FileStorageService';

const log = Logger.create('ProjectFileService');
import { ProjectCoreService } from './core/ProjectCoreService';
import { AnalysisService } from './domains/AnalysisService';
import { TranscriptService } from './domains/TranscriptService';
import { CacheService } from './domains/CacheService';
import { ProxyStorageService } from './domains/ProxyStorageService';
import { RawMediaService } from './domains/RawMediaService';
import { PROJECT_FOLDERS, type ProjectFolderKey } from './core/constants';
import type { ProjectFile, ProjectMediaFile, ProjectComposition, ProjectFolder } from './types';

class ProjectFileService {
  // Domain services
  private readonly coreService: ProjectCoreService;
  private readonly fileStorage: FileStorageService;
  private readonly analysisService: AnalysisService;
  private readonly transcriptService: TranscriptService;
  private readonly cacheService: CacheService;
  private readonly proxyStorageService: ProxyStorageService;
  private readonly rawMediaService: RawMediaService;

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
  // CORE SERVICE DELEGATION
  // ============================================

  isSupported(): boolean {
    return this.coreService.isSupported();
  }

  getProjectHandle(): FileSystemDirectoryHandle | null {
    return this.coreService.getProjectHandle();
  }

  getProjectData(): ProjectFile | null {
    return this.coreService.getProjectData();
  }

  isProjectOpen(): boolean {
    return this.coreService.isProjectOpen();
  }

  hasUnsavedChanges(): boolean {
    return this.coreService.hasUnsavedChanges();
  }

  markDirty(): void {
    this.coreService.markDirty();
  }

  needsPermission(): boolean {
    return this.coreService.needsPermission();
  }

  getPendingProjectName(): string | null {
    return this.coreService.getPendingProjectName();
  }

  async requestPendingPermission(): Promise<boolean> {
    return this.coreService.requestPendingPermission();
  }

  async createProject(name: string): Promise<boolean> {
    return this.coreService.createProject(name);
  }

  async createProjectInFolder(handle: FileSystemDirectoryHandle, name: string): Promise<boolean> {
    return this.coreService.createProjectInFolder(handle, name);
  }

  async openProject(): Promise<boolean> {
    return this.coreService.openProject();
  }

  async loadProject(handle: FileSystemDirectoryHandle): Promise<boolean> {
    return this.coreService.loadProject(handle);
  }

  async saveProject(): Promise<boolean> {
    return this.coreService.saveProject();
  }

  closeProject(): void {
    this.coreService.closeProject();
  }

  async createBackup(): Promise<boolean> {
    return this.coreService.createBackup();
  }

  async renameProject(newName: string): Promise<boolean> {
    return this.coreService.renameProject(newName);
  }

  async restoreLastProject(): Promise<boolean> {
    return this.coreService.restoreLastProject();
  }

  updateProjectData(updates: Partial<ProjectFile>): void {
    this.coreService.updateProjectData(updates);
  }

  updateMedia(media: ProjectMediaFile[]): void {
    this.coreService.updateMedia(media);
  }

  updateCompositions(compositions: ProjectComposition[]): void {
    this.coreService.updateCompositions(compositions);
  }

  updateFolders(folders: ProjectFolder[]): void {
    this.coreService.updateFolders(folders);
  }

  // ============================================
  // FILE STORAGE DELEGATION
  // ============================================

  async getFileHandle(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string,
    create = false
  ): Promise<FileSystemFileHandle | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.fileStorage.getFileHandle(handle, subFolder as ProjectFolderKey, fileName, create);
  }

  async writeFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string,
    content: Blob | string
  ): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.fileStorage.writeFile(handle, subFolder as ProjectFolderKey, fileName, content);
  }

  async readFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<File | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.fileStorage.readFile(handle, subFolder as ProjectFolderKey, fileName);
  }

  async fileExists(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.fileStorage.fileExists(handle, subFolder as ProjectFolderKey, fileName);
  }

  async deleteFile(
    subFolder: keyof typeof PROJECT_FOLDERS,
    fileName: string
  ): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.fileStorage.deleteFile(handle, subFolder as ProjectFolderKey, fileName);
  }

  async listFiles(subFolder: keyof typeof PROJECT_FOLDERS): Promise<string[]> {
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

  async saveYouTubeDownload(blob: Blob, title: string): Promise<File | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) {
      log.warn('No project open, cannot save YouTube download to project');
      return null;
    }
    return this.rawMediaService.saveYouTubeDownload(handle, blob, title);
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

  async deleteAnalysis(mediaId: string): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.analysisService.deleteAnalysis(handle, mediaId);
  }

  // ============================================
  // TRANSCRIPT SERVICE DELEGATION
  // ============================================

  async saveTranscript(mediaId: string, transcript: unknown): Promise<boolean> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return false;
    return this.transcriptService.saveTranscript(handle, mediaId, transcript);
  }

  async getTranscript(mediaId: string): Promise<unknown | null> {
    const handle = this.coreService.getProjectHandle();
    if (!handle) return null;
    return this.transcriptService.getTranscript(handle, mediaId);
  }
}

// Singleton instance
export const projectFileService = new ProjectFileService();
