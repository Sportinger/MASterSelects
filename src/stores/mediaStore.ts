// Zustand store for media/project management (like After Effects Project panel)

import { create } from 'zustand';
import { subscribeWithSelector, persist } from 'zustand/middleware';
import { useTimelineStore } from './timelineStore';
import { projectDB, type StoredMediaFile, type StoredProject } from '../services/projectDB';

// Media item types
export type MediaType = 'video' | 'audio' | 'image' | 'composition';

// Base media item
export interface MediaItem {
  id: string;
  name: string;
  type: MediaType;
  parentId: string | null; // Folder ID or null for root
  createdAt: number;
}

// Proxy status for video files
export type ProxyStatus = 'none' | 'generating' | 'ready' | 'error';

// Imported file
export interface MediaFile extends MediaItem {
  type: 'video' | 'audio' | 'image';
  file?: File; // Original file reference
  url: string; // Object URL or path
  duration?: number; // For video/audio
  width?: number; // For video/image
  height?: number; // For video/image
  thumbnailUrl?: string;
  // Proxy support (for video files)
  proxyStatus?: ProxyStatus;
  proxyProgress?: number; // 0-100
  proxyFrameCount?: number; // Total frames in proxy
  proxyFps?: number; // Frame rate of proxy (e.g., 30)
}

// Composition (like After Effects comp)
export interface Composition extends MediaItem {
  type: 'composition';
  width: number;
  height: number;
  frameRate: number;
  duration: number; // In seconds
  backgroundColor: string;
  timelineData?: import('../types').CompositionTimelineData; // Stored timeline state
}

// Folder for organization
export interface MediaFolder {
  id: string;
  name: string;
  parentId: string | null;
  isExpanded: boolean;
  createdAt: number;
}

// Union type for all items
export type ProjectItem = MediaFile | Composition | MediaFolder;

interface MediaState {
  // Items
  files: MediaFile[];
  compositions: Composition[];
  folders: MediaFolder[];

  // Active composition (the one being edited in timeline)
  activeCompositionId: string | null;

  // Selection
  selectedIds: string[];
  expandedFolderIds: string[];

  // Actions - Files
  importFile: (file: File) => Promise<MediaFile>;
  importFiles: (files: FileList | File[]) => Promise<MediaFile[]>;
  removeFile: (id: string) => void;
  renameFile: (id: string, name: string) => void;

  // Actions - Compositions
  createComposition: (name: string, settings?: Partial<Composition>) => Composition;
  duplicateComposition: (id: string) => Composition | null;
  removeComposition: (id: string) => void;
  updateComposition: (id: string, updates: Partial<Composition>) => void;

  // Actions - Folders
  createFolder: (name: string, parentId?: string | null) => MediaFolder;
  removeFolder: (id: string) => void;
  renameFolder: (id: string, name: string) => void;
  toggleFolderExpanded: (id: string) => void;

  // Actions - Organization
  moveToFolder: (itemIds: string[], folderId: string | null) => void;
  setSelection: (ids: string[]) => void;
  addToSelection: (id: string) => void;
  removeFromSelection: (id: string) => void;
  clearSelection: () => void;

  // Getters
  getItemsByFolder: (folderId: string | null) => ProjectItem[];
  getItemById: (id: string) => ProjectItem | undefined;
  getFileByName: (name: string) => MediaFile | undefined;

  // Composition management
  setActiveComposition: (id: string | null) => void;
  getActiveComposition: () => Composition | undefined;

  // Project persistence (IndexedDB)
  initFromDB: () => Promise<void>;
  saveProject: (name?: string) => Promise<string>;
  loadProject: (projectId: string) => Promise<void>;
  getProjectList: () => Promise<StoredProject[]>;
  deleteProject: (projectId: string) => Promise<void>;
  currentProjectId: string | null;
  currentProjectName: string;
  setProjectName: (name: string) => void;

  // Proxy system
  proxyEnabled: boolean;
  setProxyEnabled: (enabled: boolean) => void;
  generateProxy: (mediaFileId: string) => Promise<void>;
  cancelProxyGeneration: (mediaFileId: string) => void;
  updateProxyProgress: (mediaFileId: string, progress: number) => void;
  setProxyStatus: (mediaFileId: string, status: ProxyStatus) => void;
  getNextFileNeedingProxy: () => MediaFile | undefined;
  proxyGenerationQueue: string[]; // Queue of media file IDs to generate proxies for
  currentlyGeneratingProxyId: string | null;
  isLoading: boolean;
}

// Generate unique ID
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Detect media type from file
function getMediaType(file: File): 'video' | 'audio' | 'image' {
  if (file.type.startsWith('video/')) return 'video';
  if (file.type.startsWith('audio/')) return 'audio';
  if (file.type.startsWith('image/')) return 'image';
  // Fallback based on extension
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (['mp4', 'webm', 'mov', 'avi', 'mkv'].includes(ext || '')) return 'video';
  if (['mp3', 'wav', 'ogg', 'aac', 'm4a'].includes(ext || '')) return 'audio';
  return 'image';
}

// Create thumbnail for video/image
async function createThumbnail(file: File, type: 'video' | 'image'): Promise<string | undefined> {
  return new Promise((resolve) => {
    if (type === 'image') {
      const url = URL.createObjectURL(file);
      resolve(url);
    } else if (type === 'video') {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.currentTime = 1; // Seek to 1 second
      video.onloadeddata = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 90;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          resolve(canvas.toDataURL('image/jpeg', 0.7));
        } else {
          resolve(undefined);
        }
        URL.revokeObjectURL(video.src);
      };
      video.onerror = () => resolve(undefined);
    } else {
      resolve(undefined);
    }
  });
}

// Get media dimensions/duration
async function getMediaInfo(file: File, type: 'video' | 'audio' | 'image'): Promise<{
  width?: number;
  height?: number;
  duration?: number;
}> {
  return new Promise((resolve) => {
    if (type === 'image') {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = () => {
        resolve({ width: img.width, height: img.height });
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => resolve({});
    } else if (type === 'video') {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.onloadedmetadata = () => {
        resolve({
          width: video.videoWidth,
          height: video.videoHeight,
          duration: video.duration,
        });
        URL.revokeObjectURL(video.src);
      };
      video.onerror = () => resolve({});
    } else if (type === 'audio') {
      const audio = document.createElement('audio');
      audio.src = URL.createObjectURL(file);
      audio.onloadedmetadata = () => {
        resolve({ duration: audio.duration });
        URL.revokeObjectURL(audio.src);
      };
      audio.onerror = () => resolve({});
    } else {
      resolve({});
    }
  });
}

// Proxy generation settings
const PROXY_FPS = 30; // Generate 30 frames per second
const PROXY_QUALITY = 0.92; // WebP quality (0-1), high for color accuracy
const PROXY_MAX_WIDTH = 1920; // Max width, keep aspect ratio

// Track active proxy generation for cancellation
const activeProxyGenerations = new Map<string, { cancelled: boolean }>();

// Generate proxy frames from a video file
async function generateProxyFrames(
  mediaFile: MediaFile,
  onProgress: (progress: number) => void,
  checkCancelled: () => boolean
): Promise<{ frameCount: number; fps: number } | null> {
  if (!mediaFile.file || mediaFile.type !== 'video') {
    return null;
  }

  const video = document.createElement('video');
  video.src = URL.createObjectURL(mediaFile.file);
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';

  // Wait for video to be ready
  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error('Failed to load video'));
    setTimeout(() => reject(new Error('Video load timeout')), 30000);
  });

  // Wait for video to be fully ready for seeking
  await new Promise<void>((resolve) => {
    if (video.readyState >= 3) {
      resolve();
    } else {
      video.oncanplaythrough = () => resolve();
      setTimeout(resolve, 5000); // Timeout fallback
    }
  });

  const duration = video.duration;
  const totalFrames = Math.ceil(duration * PROXY_FPS);

  // Calculate dimensions (maintain aspect ratio, max 1920 width)
  let width = video.videoWidth;
  let height = video.videoHeight;
  if (width > PROXY_MAX_WIDTH) {
    height = Math.round((PROXY_MAX_WIDTH / width) * height);
    width = PROXY_MAX_WIDTH;
  }

  // Create canvas for frame extraction
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) {
    URL.revokeObjectURL(video.src);
    return null;
  }

  console.log(`[Proxy] Generating ${totalFrames} frames at ${width}x${height} for ${mediaFile.name}`);

  // Batch frames for efficient DB writes
  const BATCH_SIZE = 10;
  let batch: import('../services/projectDB').StoredProxyFrame[] = [];
  let generatedCount = 0;

  for (let frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
    if (checkCancelled()) {
      console.log('[Proxy] Generation cancelled');
      URL.revokeObjectURL(video.src);
      return null;
    }

    const time = frameIndex / PROXY_FPS;

    // Seek to frame time
    await new Promise<void>((resolve) => {
      const onSeeked = () => {
        video.removeEventListener('seeked', onSeeked);
        resolve();
      };
      video.addEventListener('seeked', onSeeked);
      video.currentTime = time;
      // Timeout fallback
      setTimeout(() => {
        video.removeEventListener('seeked', onSeeked);
        resolve();
      }, 1000);
    });

    // Wait for frame to be ready
    await new Promise<void>((resolve) => {
      if (video.readyState >= 2) {
        resolve();
      } else {
        const checkReady = () => {
          if (video.readyState >= 2) resolve();
          else requestAnimationFrame(checkReady);
        };
        checkReady();
        setTimeout(resolve, 500);
      }
    });

    // Draw frame to canvas
    ctx.drawImage(video, 0, 0, width, height);

    // Convert to WebP blob
    const blob = await new Promise<Blob>((resolve) => {
      canvas.toBlob(
        (b) => resolve(b || new Blob()),
        'image/webp',
        PROXY_QUALITY
      );
    });

    // Add to batch
    const frameId = `${mediaFile.id}_${frameIndex.toString().padStart(6, '0')}`;
    batch.push({
      id: frameId,
      mediaFileId: mediaFile.id,
      frameIndex,
      blob,
    });

    generatedCount++;

    // Save batch when full
    if (batch.length >= BATCH_SIZE) {
      await projectDB.saveProxyFramesBatch(batch);
      batch = [];
    }

    // Update progress
    const progress = Math.round((generatedCount / totalFrames) * 100);
    onProgress(progress);

    // Yield to UI every 5 frames
    if (frameIndex % 5 === 0) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  // Save remaining batch
  if (batch.length > 0) {
    await projectDB.saveProxyFramesBatch(batch);
  }

  URL.revokeObjectURL(video.src);
  console.log(`[Proxy] Completed ${generatedCount} frames for ${mediaFile.name}`);

  return { frameCount: generatedCount, fps: PROXY_FPS };
}

// Default composition created on first load
const DEFAULT_COMPOSITION: Composition = {
  id: 'comp-1',
  name: 'Comp 1',
  type: 'composition',
  parentId: null,
  createdAt: Date.now(),
  width: 1920,
  height: 1080,
  frameRate: 30,
  duration: 60,
  backgroundColor: '#000000',
};

export const useMediaStore = create<MediaState>()(
  subscribeWithSelector(
    persist(
      (set, get) => ({
        files: [],
        compositions: [DEFAULT_COMPOSITION],
        folders: [],
        activeCompositionId: 'comp-1',
        selectedIds: [],
        expandedFolderIds: [],
        currentProjectId: null,
        currentProjectName: 'Untitled Project',
        isLoading: false,
        // Proxy system state
        proxyEnabled: true, // Use proxies for preview by default
        proxyGenerationQueue: [],
        currentlyGeneratingProxyId: null,

        importFile: async (file: File) => {
          const type = getMediaType(file);
          const url = URL.createObjectURL(file);
          const [info, thumbnailUrl] = await Promise.all([
            getMediaInfo(file, type),
            createThumbnail(file, type as 'video' | 'image'),
          ]);

          const mediaFile: MediaFile = {
            id: generateId(),
            name: file.name,
            type,
            parentId: null,
            createdAt: Date.now(),
            file,
            url,
            thumbnailUrl,
            ...info,
          };

          set((state) => ({
            files: [...state.files, mediaFile],
          }));

          // Save file blob to IndexedDB for persistence
          try {
            const storedFile: StoredMediaFile = {
              id: mediaFile.id,
              name: file.name,
              type,
              blob: file,
              duration: info.duration,
              width: info.width,
              height: info.height,
              createdAt: mediaFile.createdAt,
            };
            // Store thumbnail as blob if it's a data URL
            if (thumbnailUrl && thumbnailUrl.startsWith('data:')) {
              const response = await fetch(thumbnailUrl);
              storedFile.thumbnailBlob = await response.blob();
            }
            await projectDB.saveMediaFile(storedFile);
            console.log('[MediaStore] Saved file to IndexedDB:', file.name);
          } catch (e) {
            console.warn('[MediaStore] Failed to save file to IndexedDB:', e);
          }

          return mediaFile;
        },

        importFiles: async (files: FileList | File[]) => {
          const fileArray = Array.from(files);
          const imported: MediaFile[] = [];
          for (const file of fileArray) {
            const mediaFile = await get().importFile(file);
            imported.push(mediaFile);
          }
          return imported;
        },

        removeFile: (id: string) => {
          const file = get().files.find((f) => f.id === id);
          if (file?.url) {
            URL.revokeObjectURL(file.url);
          }
          if (file?.thumbnailUrl && file.thumbnailUrl.startsWith('blob:')) {
            URL.revokeObjectURL(file.thumbnailUrl);
          }
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

        createComposition: (name: string, settings?: Partial<Composition>) => {
          const comp: Composition = {
            id: generateId(),
            name,
            type: 'composition',
            parentId: null,
            createdAt: Date.now(),
            width: settings?.width ?? 1920,
            height: settings?.height ?? 1080,
            frameRate: settings?.frameRate ?? 30,
            duration: settings?.duration ?? 60,
            backgroundColor: settings?.backgroundColor ?? '#000000',
          };

          set((state) => ({
            compositions: [...state.compositions, comp],
          }));

          return comp;
        },

        duplicateComposition: (id: string) => {
          const original = get().compositions.find((c) => c.id === id);
          if (!original) return null;

          const duplicate: Composition = {
            ...original,
            id: generateId(),
            name: `${original.name} Copy`,
            createdAt: Date.now(),
          };

          set((state) => ({
            compositions: [...state.compositions, duplicate],
          }));

          return duplicate;
        },

        removeComposition: (id: string) => {
          set((state) => ({
            compositions: state.compositions.filter((c) => c.id !== id),
            selectedIds: state.selectedIds.filter((sid) => sid !== id),
            // Clear active composition if we're deleting it
            activeCompositionId: state.activeCompositionId === id ? null : state.activeCompositionId,
          }));
        },

        updateComposition: (id: string, updates: Partial<Composition>) => {
          set((state) => ({
            compositions: state.compositions.map((c) =>
              c.id === id ? { ...c, ...updates } : c
            ),
          }));
        },

        createFolder: (name: string, parentId: string | null = null) => {
          const folder: MediaFolder = {
            id: generateId(),
            name,
            parentId,
            isExpanded: true,
            createdAt: Date.now(),
          };

          set((state) => ({
            folders: [...state.folders, folder],
            expandedFolderIds: [...state.expandedFolderIds, folder.id],
          }));

          return folder;
        },

        removeFolder: (id: string) => {
          // Also move children to parent
          const folder = get().folders.find((f) => f.id === id);
          const parentId = folder?.parentId ?? null;

          set((state) => ({
            folders: state.folders.filter((f) => f.id !== id),
            files: state.files.map((f) =>
              f.parentId === id ? { ...f, parentId } : f
            ),
            compositions: state.compositions.map((c) =>
              c.parentId === id ? { ...c, parentId } : c
            ),
            selectedIds: state.selectedIds.filter((sid) => sid !== id),
            expandedFolderIds: state.expandedFolderIds.filter((eid) => eid !== id),
          }));
        },

        renameFolder: (id: string, name: string) => {
          set((state) => ({
            folders: state.folders.map((f) => (f.id === id ? { ...f, name } : f)),
          }));
        },

        toggleFolderExpanded: (id: string) => {
          set((state) => ({
            expandedFolderIds: state.expandedFolderIds.includes(id)
              ? state.expandedFolderIds.filter((eid) => eid !== id)
              : [...state.expandedFolderIds, id],
          }));
        },

        moveToFolder: (itemIds: string[], folderId: string | null) => {
          set((state) => ({
            files: state.files.map((f) =>
              itemIds.includes(f.id) ? { ...f, parentId: folderId } : f
            ),
            compositions: state.compositions.map((c) =>
              itemIds.includes(c.id) ? { ...c, parentId: folderId } : c
            ),
            folders: state.folders.map((f) =>
              itemIds.includes(f.id) ? { ...f, parentId: folderId } : f
            ),
          }));
        },

        setSelection: (ids: string[]) => {
          set({ selectedIds: ids });
        },

        addToSelection: (id: string) => {
          set((state) => ({
            selectedIds: state.selectedIds.includes(id)
              ? state.selectedIds
              : [...state.selectedIds, id],
          }));
        },

        removeFromSelection: (id: string) => {
          set((state) => ({
            selectedIds: state.selectedIds.filter((sid) => sid !== id),
          }));
        },

        clearSelection: () => {
          set({ selectedIds: [] });
        },

        getItemsByFolder: (folderId: string | null) => {
          const { files, compositions, folders } = get();
          const items: ProjectItem[] = [
            ...folders.filter((f) => f.parentId === folderId),
            ...compositions.filter((c) => c.parentId === folderId),
            ...files.filter((f) => f.parentId === folderId),
          ];
          return items;
        },

        getItemById: (id: string) => {
          const { files, compositions, folders } = get();
          return (
            files.find((f) => f.id === id) ||
            compositions.find((c) => c.id === id) ||
            folders.find((f) => f.id === id)
          );
        },

        getFileByName: (name: string) => {
          const { files } = get();
          return files.find((f) => f.name === name);
        },

        setActiveComposition: (id: string | null) => {
          const { activeCompositionId, compositions } = get();
          const timelineStore = useTimelineStore.getState();

          // Save current timeline state to the current composition (if any)
          if (activeCompositionId) {
            const timelineData = timelineStore.getSerializableState();
            set((state) => ({
              compositions: state.compositions.map((c) =>
                c.id === activeCompositionId ? { ...c, timelineData } : c
              ),
            }));
          }

          // Update active composition
          set({ activeCompositionId: id });

          // Load timeline state from the new composition
          if (id) {
            const newComp = compositions.find((c) => c.id === id);
            timelineStore.loadState(newComp?.timelineData);
          } else {
            // No composition selected - clear timeline
            timelineStore.clearTimeline();
          }
        },

        getActiveComposition: () => {
          const { compositions, activeCompositionId } = get();
          return compositions.find((c) => c.id === activeCompositionId);
        },

        setProjectName: (name: string) => {
          set({ currentProjectName: name });
        },

        // ============ Proxy System ============

        setProxyEnabled: (enabled: boolean) => {
          set({ proxyEnabled: enabled });
        },

        updateProxyProgress: (mediaFileId: string, progress: number) => {
          const { files } = get();
          set({
            files: files.map((f) =>
              f.id === mediaFileId ? { ...f, proxyProgress: progress } : f
            ),
          });
        },

        setProxyStatus: (mediaFileId: string, status: ProxyStatus) => {
          const { files } = get();
          set({
            files: files.map((f) =>
              f.id === mediaFileId ? { ...f, proxyStatus: status } : f
            ),
          });
        },

        getNextFileNeedingProxy: () => {
          const { files, currentlyGeneratingProxyId } = get();
          // Find video files that don't have proxies yet and aren't currently generating
          return files.find(
            (f) =>
              f.type === 'video' &&
              f.file &&
              f.proxyStatus !== 'ready' &&
              f.proxyStatus !== 'generating' &&
              f.id !== currentlyGeneratingProxyId
          );
        },

        generateProxy: async (mediaFileId: string) => {
          const { files, currentlyGeneratingProxyId, updateProxyProgress, setProxyStatus } = get();

          // Don't start if already generating something
          if (currentlyGeneratingProxyId) {
            console.log('[Proxy] Already generating, queuing:', mediaFileId);
            return;
          }

          const mediaFile = files.find((f) => f.id === mediaFileId);
          if (!mediaFile || mediaFile.type !== 'video' || !mediaFile.file) {
            console.warn('[Proxy] Invalid media file:', mediaFileId);
            return;
          }

          // Check if proxy already exists
          const hasExisting = await projectDB.hasProxy(mediaFileId);
          if (hasExisting) {
            console.log('[Proxy] Proxy already exists for:', mediaFile.name);
            set({
              files: get().files.map((f) =>
                f.id === mediaFileId
                  ? { ...f, proxyStatus: 'ready' as ProxyStatus, proxyProgress: 100 }
                  : f
              ),
            });
            return;
          }

          // Set up cancellation tracking
          const controller = { cancelled: false };
          activeProxyGenerations.set(mediaFileId, controller);

          set({ currentlyGeneratingProxyId: mediaFileId });
          setProxyStatus(mediaFileId, 'generating');
          updateProxyProgress(mediaFileId, 0);

          try {
            const result = await generateProxyFrames(
              mediaFile,
              (progress) => updateProxyProgress(mediaFileId, progress),
              () => controller.cancelled
            );

            if (result) {
              // Update media file with proxy info
              set({
                files: get().files.map((f) =>
                  f.id === mediaFileId
                    ? {
                        ...f,
                        proxyStatus: 'ready' as ProxyStatus,
                        proxyProgress: 100,
                        proxyFrameCount: result.frameCount,
                        proxyFps: result.fps,
                      }
                    : f
                ),
              });
              console.log('[Proxy] Completed for:', mediaFile.name);
            } else if (!controller.cancelled) {
              setProxyStatus(mediaFileId, 'error');
            }
          } catch (e) {
            console.error('[Proxy] Generation failed:', e);
            setProxyStatus(mediaFileId, 'error');
          } finally {
            activeProxyGenerations.delete(mediaFileId);
            set({ currentlyGeneratingProxyId: null });
          }
        },

        cancelProxyGeneration: (mediaFileId: string) => {
          const controller = activeProxyGenerations.get(mediaFileId);
          if (controller) {
            controller.cancelled = true;
            console.log('[Proxy] Cancelled generation for:', mediaFileId);
          }
          // Reset status
          const { files, currentlyGeneratingProxyId } = get();
          if (currentlyGeneratingProxyId === mediaFileId) {
            set({
              currentlyGeneratingProxyId: null,
              files: files.map((f) =>
                f.id === mediaFileId
                  ? { ...f, proxyStatus: 'none' as ProxyStatus, proxyProgress: 0 }
                  : f
              ),
            });
          }
        },

        // Initialize from IndexedDB - restore file blobs and check proxy status
        initFromDB: async () => {
          set({ isLoading: true });
          try {
            const storedFiles = await projectDB.getAllMediaFiles();
            const { files } = get();

            // Match stored blobs with existing file metadata
            const updatedFiles = await Promise.all(
              files.map(async (mediaFile) => {
                const stored = storedFiles.find((sf) => sf.id === mediaFile.id);
                if (stored) {
                  // Restore file blob and URL
                  const file = new File([stored.blob], stored.name, { type: stored.blob.type });
                  const url = URL.createObjectURL(file);
                  let thumbnailUrl = mediaFile.thumbnailUrl;
                  if (stored.thumbnailBlob) {
                    thumbnailUrl = URL.createObjectURL(stored.thumbnailBlob);
                  }

                  // Check if proxy exists for video files
                  let proxyStatus: ProxyStatus = 'none';
                  let proxyFrameCount: number | undefined;
                  if (stored.type === 'video') {
                    const frameCount = await projectDB.getProxyFrameCount(mediaFile.id);
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
                    proxyStatus,
                    proxyFrameCount,
                    proxyFps: proxyFrameCount ? PROXY_FPS : undefined,
                    proxyProgress: proxyFrameCount ? 100 : 0,
                  };
                }
                return mediaFile;
              })
            );

            set({ files: updatedFiles, isLoading: false });
            console.log('[MediaStore] Restored', storedFiles.length, 'files from IndexedDB');
          } catch (e) {
            console.error('[MediaStore] Failed to init from IndexedDB:', e);
            set({ isLoading: false });
          }
        },

        // Save current project
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
              expandedFolderIds: state.expandedFolderIds,
              mediaFileIds: state.files.map((f) => f.id),
            },
          };

          await projectDB.saveProject(project);
          set({ currentProjectId: projectId, currentProjectName: projectName });
          console.log('[MediaStore] Project saved:', projectName);
          return projectId;
        },

        // Load a project
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

            // Restore files with blobs
            const files: MediaFile[] = [];
            for (const fileId of project.data.mediaFileIds) {
              const stored = mediaFileMap.get(fileId);
              if (stored) {
                const file = new File([stored.blob], stored.name, { type: stored.blob.type });
                const url = URL.createObjectURL(file);
                let thumbnailUrl: string | undefined;
                if (stored.thumbnailBlob) {
                  thumbnailUrl = URL.createObjectURL(stored.thumbnailBlob);
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

            // Clear timeline first
            const timelineStore = useTimelineStore.getState();
            timelineStore.clearTimeline();

            // Restore state
            set({
              files,
              compositions: project.data.compositions as Composition[],
              folders: project.data.folders as MediaFolder[],
              activeCompositionId: null, // Will be set below
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
                set({ activeCompositionId: project.data.activeCompositionId });
              }
            }

            console.log('[MediaStore] Project loaded:', project.name);
          } catch (e) {
            console.error('[MediaStore] Failed to load project:', e);
            set({ isLoading: false });
            throw e;
          }
        },

        // Get list of all projects
        getProjectList: async () => {
          return projectDB.getAllProjects();
        },

        // Delete a project
        deleteProject: async (projectId: string) => {
          await projectDB.deleteProject(projectId);
          console.log('[MediaStore] Project deleted:', projectId);
        },
      }),
      {
        name: 'webvj-media',
        partialize: (state) => ({
          // Don't persist file blobs, only metadata
          files: state.files.map(({ file, ...rest }) => rest),
          compositions: state.compositions,
          folders: state.folders,
          activeCompositionId: state.activeCompositionId,
          expandedFolderIds: state.expandedFolderIds,
          currentProjectId: state.currentProjectId,
          currentProjectName: state.currentProjectName,
        }),
      }
    )
  )
);

// Auto-initialize from IndexedDB on app load
if (typeof window !== 'undefined') {
  // Delay init slightly to ensure store is ready
  setTimeout(() => {
    useMediaStore.getState().initFromDB();
  }, 100);
}
