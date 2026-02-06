// Project persistence slice - save, load, init

import type { Composition, MediaFile, MediaFolder, TextItem, SolidItem, MediaSliceCreator, ProxyStatus } from '../types';
import { PROXY_FPS, DEFAULT_COMPOSITION } from '../constants';
import { generateId } from '../helpers/importPipeline';
import { projectDB, type StoredProject } from '../../../services/projectDB';
import { projectFileService } from '../../../services/projectFileService';
import { fileSystemService } from '../../../services/fileSystemService';
import { useTimelineStore } from '../../timeline';
import { engine } from '../../../engine/WebGPUEngine';
import { Logger } from '../../../services/logger';

const log = Logger.create('Project');

export interface ProjectActions {
  initFromDB: () => Promise<void>;
  saveProject: (name?: string) => Promise<string>;
  loadProject: (projectId: string) => Promise<void>;
  newProject: () => void;
  getProjectList: () => Promise<StoredProject[]>;
  deleteProject: (projectId: string) => Promise<void>;
  setProjectName: (name: string) => void;
  pickProxyFolder: () => Promise<FileSystemDirectoryHandle | null>;
  showInExplorer: (type: 'raw' | 'proxy', mediaFileId?: string) => Promise<{ success: boolean; message: string }>;
}

export const createProjectSlice: MediaSliceCreator<ProjectActions> = (set, get) => ({
  setProjectName: (name: string) => {
    set({ currentProjectName: name });
  },

  pickProxyFolder: async () => {
    const result = await fileSystemService.pickProxyFolder();
    if (result) {
      set({ proxyFolderName: fileSystemService.getProxyFolderName() });
    }
    return result;
  },

  showInExplorer: async (type: 'raw' | 'proxy', mediaFileId?: string) => {
    return fileSystemService.showInExplorer(type, mediaFileId);
  },

  initFromDB: async () => {
    set({ isLoading: true });
    try {
      const storedFiles = await projectDB.getAllMediaFiles();
      const { files } = get();

      // Restore files from file handles
      const updatedFiles = await Promise.all(
        files.map(async (mediaFile) => {
          const stored = storedFiles.find((sf) => sf.id === mediaFile.id);
          if (!stored) return mediaFile;

          let file: File | undefined;
          let url = mediaFile.url;
          let thumbnailUrl = mediaFile.thumbnailUrl;

          // Try to get file handle from IndexedDB
          const handle = await projectDB.getStoredHandle(`media_${mediaFile.id}`);
          if (handle && 'getFile' in handle) {
            try {
              const permission = await (handle as FileSystemFileHandle).queryPermission({ mode: 'read' });
              if (permission === 'granted') {
                file = await (handle as FileSystemFileHandle).getFile();
                url = URL.createObjectURL(file);
                fileSystemService.storeFileHandle(mediaFile.id, handle as FileSystemFileHandle);
                log.debug('Restored file from handle:', stored.name);
              } else {
                const newPermission = await (handle as FileSystemFileHandle).requestPermission({ mode: 'read' });
                if (newPermission === 'granted') {
                  file = await (handle as FileSystemFileHandle).getFile();
                  url = URL.createObjectURL(file);
                  fileSystemService.storeFileHandle(mediaFile.id, handle as FileSystemFileHandle);
                  log.debug(`Restored file from handle (after permission): ${stored.name}`);
                }
              }
            } catch (e) {
              log.warn(`Failed to restore file from handle: ${stored.name}`, e);
            }
          }

          // Restore thumbnail from project folder
          if (stored.fileHash && projectFileService.isProjectOpen()) {
            const thumbBlob = await projectFileService.getThumbnail(stored.fileHash);
            if (thumbBlob) {
              thumbnailUrl = URL.createObjectURL(thumbBlob);
            }
          }

          // Check for existing proxy by hash (fallback to mediaId for older projects)
          let proxyStatus: ProxyStatus = 'none';
          let proxyFrameCount: number | undefined;
          if (stored.type === 'video' && projectFileService.isProjectOpen()) {
            // Try fileHash first, then fall back to mediaId (for backwards compatibility)
            const storageKey = stored.fileHash || mediaFile.id;
            const frameCount = await projectFileService.getProxyFrameCount(storageKey);
            if (frameCount > 0) {
              proxyStatus = 'ready';
              proxyFrameCount = frameCount;
            }
          }

          return {
            ...mediaFile,
            file,
            url,
            thumbnailUrl,
            fileHash: stored.fileHash,
            hasFileHandle: !!file,
            proxyStatus,
            proxyFrameCount,
            proxyFps: proxyFrameCount ? PROXY_FPS : undefined,
            proxyProgress: proxyFrameCount ? 100 : 0,
            duration: stored.duration ?? mediaFile.duration,
            width: stored.width ?? mediaFile.width,
            height: stored.height ?? mediaFile.height,
            fps: stored.fps ?? mediaFile.fps,
            codec: stored.codec ?? mediaFile.codec,
            container: stored.container ?? mediaFile.container,
            fileSize: stored.fileSize ?? mediaFile.fileSize,
          };
        })
      );

      set({ files: updatedFiles, isLoading: false });
      log.info(`Restored ${storedFiles.length} files from IndexedDB`);
    } catch (e) {
      log.error('Failed to init from IndexedDB:', e);
      set({ isLoading: false });
    }
  },

  saveProject: async (name?: string) => {
    const state = get();
    const projectName = name || state.currentProjectName;
    const projectId = state.currentProjectId || generateId();

    // Save current timeline to active composition first
    if (state.activeCompositionId) {
      const timelineStore = useTimelineStore.getState();
      const timelineData = timelineStore.getSerializableState();
      set((s) => ({
        compositions: s.compositions.map((c) =>
          c.id === state.activeCompositionId ? { ...c, timelineData } : c
        ),
      }));
    }

    const project: StoredProject = {
      id: projectId,
      name: projectName,
      createdAt: state.currentProjectId ? Date.now() : Date.now(),
      updatedAt: Date.now(),
      data: {
        compositions: get().compositions,
        folders: state.folders,
        activeCompositionId: state.activeCompositionId,
        openCompositionIds: state.openCompositionIds,
        expandedFolderIds: state.expandedFolderIds,
        mediaFileIds: state.files.map((f) => f.id),
        textItems: state.textItems,
        solidItems: state.solidItems,
      },
    };

    await projectDB.saveProject(project);
    set({ currentProjectId: projectId, currentProjectName: projectName });
    log.info('Saved:', projectName);
    return projectId;
  },

  loadProject: async (projectId: string) => {
    set({ isLoading: true });
    try {
      const project = await projectDB.getProject(projectId);
      if (!project) {
        throw new Error('Project not found');
      }

      // Load media files from IndexedDB
      const storedFiles = await projectDB.getAllMediaFiles();
      const mediaFileMap = new Map(storedFiles.map((f) => [f.id, f]));

      // Restore files from metadata (legacy fallback)
      const files: MediaFile[] = [];
      for (const fileId of project.data.mediaFileIds) {
        const stored = mediaFileMap.get(fileId);
        if (stored) {
          const storedWithBlob = stored as typeof stored & { blob?: Blob; thumbnailBlob?: Blob };
          if (storedWithBlob.blob) {
            const file = new File([storedWithBlob.blob], stored.name, { type: storedWithBlob.blob.type });
            const url = URL.createObjectURL(file);
            let thumbnailUrl: string | undefined;
            if (storedWithBlob.thumbnailBlob) {
              thumbnailUrl = URL.createObjectURL(storedWithBlob.thumbnailBlob);
            }
            files.push({
              id: stored.id,
              name: stored.name,
              type: stored.type,
              parentId: null,
              createdAt: stored.createdAt,
              file,
              url,
              thumbnailUrl,
              duration: stored.duration,
              width: stored.width,
              height: stored.height,
            });
          }
        }
      }

      // Clear timeline first
      const timelineStore = useTimelineStore.getState();
      timelineStore.clearTimeline();

      // Clear the render frame
      engine.clearFrame();

      // Restore state
      set({
        files,
        compositions: project.data.compositions as Composition[],
        folders: project.data.folders as MediaFolder[],
        textItems: (project.data.textItems as TextItem[]) || [],
        solidItems: (project.data.solidItems as SolidItem[]) || [],
        activeCompositionId: null,
        openCompositionIds: (project.data.openCompositionIds as string[]) || [],
        expandedFolderIds: project.data.expandedFolderIds,
        currentProjectId: projectId,
        currentProjectName: project.name,
        isLoading: false,
      });

      // Load active composition's timeline
      if (project.data.activeCompositionId) {
        const comp = (project.data.compositions as Composition[]).find(
          (c) => c.id === project.data.activeCompositionId
        );
        if (comp) {
          await timelineStore.loadState(comp.timelineData);
          set({
            activeCompositionId: project.data.activeCompositionId,
            openCompositionIds: get().openCompositionIds.includes(project.data.activeCompositionId as string)
              ? get().openCompositionIds
              : [...get().openCompositionIds, project.data.activeCompositionId as string]
          });
        }
      }

      log.info('Loaded:', project.name);
    } catch (e) {
      log.error('Failed to load:', e);
      set({ isLoading: false });
      throw e;
    }
  },

  newProject: () => {
    // Clear timeline first
    const timelineStore = useTimelineStore.getState();
    timelineStore.clearTimeline();

    // Clear the render frame
    engine.clearFrame();

    // Create new default composition
    const newCompId = `comp-${Date.now()}`;
    const newComposition: Composition = {
      ...DEFAULT_COMPOSITION,
      id: newCompId,
      createdAt: Date.now(),
    };

    // Reset all state
    set({
      files: [],
      compositions: [newComposition],
      folders: [],
      activeCompositionId: newCompId,
      openCompositionIds: [newCompId],
      selectedIds: [],
      expandedFolderIds: [],
      currentProjectId: null,
      currentProjectName: 'Untitled Project',
      proxyEnabled: false,
      proxyGenerationQueue: [],
      currentlyGeneratingProxyId: null,
    });

    // Load empty timeline
    timelineStore.loadState(undefined);

    log.info('New project created');
  },

  getProjectList: async () => {
    return projectDB.getAllProjects();
  },

  deleteProject: async (projectId: string) => {
    await projectDB.deleteProject(projectId);
    log.info('Deleted:', projectId);
  },
});
