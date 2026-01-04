// Zustand store for media/project management (like After Effects Project panel)

import { create } from 'zustand';
import { subscribeWithSelector, persist } from 'zustand/middleware';

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

// Imported file
export interface MediaFile extends MediaItem {
  type: 'video' | 'audio' | 'image';
  file?: File; // Original file reference
  url: string; // Object URL or path
  duration?: number; // For video/audio
  width?: number; // For video/image
  height?: number; // For video/image
  thumbnailUrl?: string;
}

// Composition (like After Effects comp)
export interface Composition extends MediaItem {
  type: 'composition';
  width: number;
  height: number;
  frameRate: number;
  duration: number; // In seconds
  backgroundColor: string;
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

export const useMediaStore = create<MediaState>()(
  subscribeWithSelector(
    persist(
      (set, get) => ({
        files: [],
        compositions: [],
        folders: [],
        selectedIds: [],
        expandedFolderIds: [],

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
      }),
      {
        name: 'webvj-media',
        partialize: (state) => ({
          // Don't persist file blobs, only metadata
          files: state.files.map(({ file, ...rest }) => rest),
          compositions: state.compositions,
          folders: state.folders,
          expandedFolderIds: state.expandedFolderIds,
        }),
      }
    )
  )
);
