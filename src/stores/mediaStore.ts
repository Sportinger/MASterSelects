// Zustand store for media/project management (like After Effects Project panel)

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { useTimelineStore } from './timeline';
import { projectDB, type StoredMediaFile, type StoredProject } from '../services/projectDB';
import { fileSystemService } from '../services/fileSystemService';
import { projectFileService } from '../services/projectFileService';
import { engine } from '../engine/WebGPUEngine';
import { compositionRenderer } from '../services/compositionRenderer';

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
  fps?: number; // Frame rate for video
  codec?: string; // Video/audio codec (e.g., H.264, VP9)
  container?: string; // Container format (e.g., MP4, MKV, WebM)
  fileSize?: number; // File size in bytes
  fileHash?: string; // Hash of file content (for proxy deduplication)
  thumbnailUrl?: string;
  // Proxy support (for video files)
  proxyStatus?: ProxyStatus;
  proxyProgress?: number; // 0-100
  proxyFrameCount?: number; // Total frames in proxy
  proxyFps?: number; // Frame rate of proxy (e.g., 30)
  // File System Access API support
  hasFileHandle?: boolean; // True if imported via File System Access API
  filePath?: string; // Display path (folder name / file name)
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
  // Open composition tabs (like browser tabs)
  openCompositionIds: string[];

  // Selection
  selectedIds: string[];
  expandedFolderIds: string[];

  // Actions - Files
  importFile: (file: File) => Promise<MediaFile>;
  importFiles: (files: FileList | File[]) => Promise<MediaFile[]>;
  removeFile: (id: string) => void;
  renameFile: (id: string, name: string) => void;
  reloadFile: (id: string) => Promise<boolean>; // Re-request file permission after refresh
  reloadAllFiles: () => Promise<number>; // Reload all files that need permission, returns count

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
  openCompositionTab: (id: string) => void;
  closeCompositionTab: (id: string) => void;
  getOpenCompositions: () => Composition[];
  reorderCompositionTabs: (fromIndex: number, toIndex: number) => void;

  // Project persistence (IndexedDB)
  initFromDB: () => Promise<void>;
  saveProject: (name?: string) => Promise<string>;
  loadProject: (projectId: string) => Promise<void>;
  newProject: () => void;
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

  // File System Access API
  fileSystemSupported: boolean;
  proxyFolderName: string | null;
  importFilesWithPicker: () => Promise<MediaFile[]>;
  importFilesWithHandles: (filesWithHandles: Array<{ file: File; handle: FileSystemFileHandle }>) => Promise<MediaFile[]>;
  pickProxyFolder: () => Promise<boolean>;
  showInExplorer: (type: 'raw' | 'proxy', mediaFileId?: string) => Promise<{ success: boolean; message: string }>;
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

// Calculate file hash for proxy deduplication (first 2MB + file size for speed)
async function calculateFileHash(file: File): Promise<string> {
  try {
    // Read first 2MB of file (enough to uniquely identify most videos)
    const HASH_SIZE = 2 * 1024 * 1024;
    const slice = file.slice(0, Math.min(file.size, HASH_SIZE));
    const buffer = await slice.arrayBuffer();

    // Include file size in hash to differentiate truncated files
    const sizeBuffer = new ArrayBuffer(8);
    const sizeView = new DataView(sizeBuffer);
    sizeView.setBigUint64(0, BigInt(file.size), true);

    // Combine buffers
    const combined = new Uint8Array(buffer.byteLength + 8);
    combined.set(new Uint8Array(buffer), 0);
    combined.set(new Uint8Array(sizeBuffer), buffer.byteLength);

    // Calculate SHA-256 hash
    const hashBuffer = await crypto.subtle.digest('SHA-256', combined);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  } catch (e) {
    console.warn('[MediaStore] Failed to calculate file hash:', e);
    return '';
  }
}

// Get container format from file extension
function getContainerFormat(fileName: string): string {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  const containerMap: Record<string, string> = {
    mp4: 'MP4',
    m4v: 'MP4',
    mov: 'MOV',
    mkv: 'MKV',
    webm: 'WebM',
    avi: 'AVI',
    wmv: 'WMV',
    flv: 'FLV',
    ogv: 'OGV',
    '3gp': '3GP',
    mp3: 'MP3',
    wav: 'WAV',
    ogg: 'OGG',
    flac: 'FLAC',
    aac: 'AAC',
    m4a: 'M4A',
    jpg: 'JPEG',
    jpeg: 'JPEG',
    png: 'PNG',
    gif: 'GIF',
    webp: 'WebP',
    bmp: 'BMP',
    svg: 'SVG',
  };
  return containerMap[ext] || ext.toUpperCase();
}

// Try to parse FPS from filename (common patterns like "25fps", "_30p", etc.)
function parseFpsFromFilename(fileName: string): number | undefined {
  // Match patterns like "25fps", "30fps", "24p", "60p", "29.97fps"
  // More flexible: look for number followed by fps/p anywhere in filename
  const patterns = [
    /[_\-\s\(](\d{2}(?:\.\d+)?)\s*fps/i,  // _25fps, -30fps, (24fps
    /[_\-\s\(](\d{2}(?:\.\d+)?)\s*p[_\-\s\)\.]/i,  // _24p_, -30p.
    /(\d{2}(?:\.\d+)?)fps/i,  // 25fps anywhere
  ];

  for (const pattern of patterns) {
    const match = fileName.match(pattern);
    if (match) {
      const fps = parseFloat(match[1]);
      // Valid FPS range: 10-240 (excludes resolution like 1080p)
      if (fps >= 10 && fps <= 240) return fps;
    }
  }
  return undefined;
}

// Get codec info from file (best effort - browsers have limited support)
async function getCodecInfo(file: File): Promise<string | undefined> {
  // Try to read first few bytes for codec detection
  try {
    const buffer = await file.slice(0, 32).arrayBuffer();
    const view = new DataView(buffer);

    // Check for common signatures
    const ext = file.name.split('.').pop()?.toLowerCase();

    // For MP4/MOV, codec is usually H.264 or H.265
    if (ext === 'mp4' || ext === 'm4v' || ext === 'mov') {
      // We could parse moov/trak boxes but that's complex
      // Return common default
      return 'H.264';
    }

    // For WebM, usually VP8 or VP9
    if (ext === 'webm') {
      // Check for WebM signature (1A 45 DF A3)
      if (view.getUint32(0) === 0x1A45DFA3) {
        return 'VP9'; // Modern WebM usually VP9
      }
    }

    // For MKV
    if (ext === 'mkv') {
      return 'H.264'; // Most common
    }

    // Audio codecs
    if (ext === 'mp3') return 'MP3';
    if (ext === 'aac' || ext === 'm4a') return 'AAC';
    if (ext === 'wav') return 'PCM';
    if (ext === 'ogg') return 'Vorbis';
    if (ext === 'flac') return 'FLAC';

  } catch {
    // Ignore errors
  }
  return undefined;
}

// Get media dimensions/duration
async function getMediaInfo(file: File, type: 'video' | 'audio' | 'image'): Promise<{
  width?: number;
  height?: number;
  duration?: number;
  fps?: number;
  codec?: string;
  container?: string;
  fileSize?: number;
}> {
  const container = getContainerFormat(file.name);
  const fileSize = file.size;
  const codec = await getCodecInfo(file);

  return new Promise((resolve) => {
    if (type === 'image') {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = () => {
        resolve({ width: img.width, height: img.height, container, fileSize, codec });
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => resolve({ container, fileSize });
    } else if (type === 'video') {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.onloadedmetadata = () => {
        // Try to get FPS from filename
        const fps = parseFpsFromFilename(file.name);

        resolve({
          width: video.videoWidth,
          height: video.videoHeight,
          duration: video.duration,
          fps,
          codec,
          container,
          fileSize,
        });
        URL.revokeObjectURL(video.src);
      };
      video.onerror = () => resolve({ container, fileSize });
    } else if (type === 'audio') {
      const audio = document.createElement('audio');
      audio.src = URL.createObjectURL(file);
      audio.onloadedmetadata = () => {
        resolve({ duration: audio.duration, codec, container, fileSize });
        URL.revokeObjectURL(audio.src);
      };
      audio.onerror = () => resolve({ container, fileSize });
    } else {
      resolve({ container, fileSize });
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

    // Use fileHash for deduplication, fallback to mediaFile.id
    const storageKey = mediaFile.fileHash || mediaFile.id;
    const frameId = `${storageKey}_${frameIndex.toString().padStart(6, '0')}`;
    batch.push({
      id: frameId,
      mediaFileId: mediaFile.id, // Keep for backwards compatibility
      fileHash: mediaFile.fileHash, // Add hash for deduplication
      frameIndex,
      blob,
    });

    // Also save to file system if folder is selected
    if (fileSystemService.hasProxyFolder()) {
      await fileSystemService.saveProxyFrame(storageKey, frameIndex, blob);
    }

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
      (set, get) => ({
        files: [],
        compositions: [DEFAULT_COMPOSITION],
        folders: [],
        activeCompositionId: 'comp-1',
        openCompositionIds: ['comp-1'],
        selectedIds: [],
        expandedFolderIds: [],
        currentProjectId: null,
        currentProjectName: 'Untitled Project',
        isLoading: false,
        // Proxy system state
        proxyEnabled: false, // Proxies disabled by default
        proxyGenerationQueue: [],
        currentlyGeneratingProxyId: null,

        // File System Access API state
        fileSystemSupported: fileSystemService.isSupported(),
        proxyFolderName: fileSystemService.getProxyFolderName(),

        importFile: async (file: File) => {
          const type = getMediaType(file);
          const url = URL.createObjectURL(file);
          const [info, thumbnailUrl] = await Promise.all([
            getMediaInfo(file, type),
            createThumbnail(file, type as 'video' | 'image'),
          ]);

          // Calculate file hash for deduplication
          const fileHash = await calculateFileHash(file);

          // Check for existing thumbnail by hash
          let finalThumbnailUrl = thumbnailUrl;
          if (fileHash) {
            try {
              // Try local file system first (if project is open), then IndexedDB
              let existingThumbBlob: Blob | null = null;
              if (projectFileService.isProjectOpen()) {
                existingThumbBlob = await projectFileService.getThumbnail(fileHash);
              }
              if (!existingThumbBlob) {
                const existingThumb = await projectDB.getThumbnail(fileHash);
                if (existingThumb?.blob?.size > 0) {
                  existingThumbBlob = existingThumb.blob;
                }
              }

              if (existingThumbBlob && existingThumbBlob.size > 0) {
                // Reuse existing thumbnail
                finalThumbnailUrl = URL.createObjectURL(existingThumbBlob);
                console.log('[MediaStore] Reusing existing thumbnail for hash:', fileHash.slice(0, 8));
              } else if (thumbnailUrl) {
                // Save new thumbnail by hash
                let thumbBlob: Blob | null = null;
                if (thumbnailUrl.startsWith('data:')) {
                  const response = await fetch(thumbnailUrl);
                  thumbBlob = await response.blob();
                } else if (thumbnailUrl.startsWith('blob:')) {
                  const response = await fetch(thumbnailUrl);
                  thumbBlob = await response.blob();
                }
                if (thumbBlob && thumbBlob.size > 0) {
                  // Save to local file system if project is open, otherwise to IndexedDB
                  if (projectFileService.isProjectOpen()) {
                    await projectFileService.saveThumbnail(fileHash, thumbBlob);
                    console.log('[MediaStore] Saved thumbnail to project folder:', fileHash.slice(0, 8));
                  } else {
                    await projectDB.saveThumbnail({
                      fileHash,
                      blob: thumbBlob,
                      createdAt: Date.now(),
                    });
                  }
                }
              }
            } catch (e) {
              console.warn('[MediaStore] Thumbnail dedup error, using original:', e);
              // Keep original thumbnailUrl
            }
          }

          // Check for existing proxy by hash
          let proxyStatus: ProxyStatus = 'none';
          let proxyFrameCount: number | undefined;
          if (fileHash && type === 'video') {
            const existingProxyCount = await projectDB.getProxyFrameCountByHash(fileHash);
            if (existingProxyCount > 0) {
              proxyStatus = 'ready';
              proxyFrameCount = existingProxyCount;
              console.log('[MediaStore] Found existing proxy for hash:', fileHash.slice(0, 8), 'frames:', existingProxyCount);
            }
          }

          const mediaFile: MediaFile = {
            id: generateId(),
            name: file.name,
            type,
            parentId: null,
            createdAt: Date.now(),
            file,
            url,
            thumbnailUrl: finalThumbnailUrl,
            fileHash,
            proxyStatus,
            proxyFrameCount,
            proxyFps: proxyFrameCount ? PROXY_FPS : undefined,
            ...info,
          };

          set((state) => ({
            files: [...state.files, mediaFile],
          }));

          // Save metadata only (no blob) to IndexedDB
          try {
            const storedFile: StoredMediaFile = {
              id: mediaFile.id,
              name: file.name,
              type,
              fileHash,
              duration: info.duration,
              width: info.width,
              height: info.height,
              fps: info.fps,
              codec: info.codec,
              container: info.container,
              fileSize: info.fileSize,
              createdAt: mediaFile.createdAt,
            };
            await projectDB.saveMediaFile(storedFile);
            console.log('[MediaStore] Saved file metadata to IndexedDB:', file.name);
          } catch (e) {
            console.warn('[MediaStore] Failed to save file metadata:', e);
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

        reloadFile: async (id: string) => {
          const mediaFile = get().files.find(f => f.id === id);
          if (!mediaFile) return false;

          // Try to get file handle from in-memory storage first
          let handle = fileSystemService.getFileHandle(id);

          // If not in memory, try to get from IndexedDB
          if (!handle) {
            console.log('[MediaStore] No handle in memory for:', mediaFile.name, 'ID:', id, '- checking IndexedDB with key:', `media_${id}`);
            try {
              const storedHandle = await projectDB.getStoredHandle(`media_${id}`);
              console.log('[MediaStore] IndexedDB result for', `media_${id}`, ':', storedHandle ? 'found' : 'not found');
              if (storedHandle && storedHandle.kind === 'file') {
                handle = storedHandle as FileSystemFileHandle;
                // Cache it in memory for future use
                fileSystemService.storeFileHandle(id, handle);
                console.log('[MediaStore] Retrieved handle from IndexedDB for:', mediaFile.name);
              }
            } catch (e) {
              console.warn('[MediaStore] Failed to get handle from IndexedDB:', e);
            }
          }

          if (!handle) {
            // No handle available - file was likely imported via drag-and-drop or old file input
            // Prompt user to re-select the file
            console.log('[MediaStore] No handle for:', mediaFile.name, '- prompting user to re-select');

            try {
              // Open file picker for user to re-select the file
              const [newHandle] = await window.showOpenFilePicker({
                multiple: false,
                types: [{
                  description: 'Media Files',
                  accept: {
                    'video/*': [],
                    'audio/*': [],
                    'image/*': [],
                  },
                }],
              });

              if (newHandle) {
                handle = newHandle;
                // Store the handle for future use
                fileSystemService.storeFileHandle(id, handle);
                await projectDB.storeHandle(`media_${id}`, handle);
                console.log('[MediaStore] Stored new handle for:', mediaFile.name);
              }
            } catch (e) {
              // User cancelled or error
              console.warn('[MediaStore] File picker cancelled or failed:', e);
              return false;
            }
          }

          if (!handle) {
            return false;
          }

          try {
            // Request permission
            const permission = await handle.requestPermission({ mode: 'read' });
            if (permission !== 'granted') {
              console.warn('[MediaStore] Permission denied for:', mediaFile.name);
              return false;
            }

            // Get the file
            const file = await handle.getFile();
            const url = URL.createObjectURL(file);

            // Revoke old URL if exists
            if (mediaFile.url) {
              URL.revokeObjectURL(mediaFile.url);
            }

            // Update media store
            set((state) => ({
              files: state.files.map((f) =>
                f.id === id ? { ...f, file, url } : f
              ),
            }));

            console.log('[MediaStore] Reloaded file:', mediaFile.name);

            // Update timeline clips that use this media file
            const { useTimelineStore } = await import('./timeline');
            const timelineStore = useTimelineStore.getState();
            const clips = timelineStore.clips.filter(
              c => c.source?.mediaFileId === id && c.needsReload
            );

            if (clips.length > 0) {
              // Update clips with new file
              for (const clip of clips) {
                timelineStore.updateClip(clip.id, {
                  file,
                  needsReload: false,
                  isLoading: true, // Trigger video element reload
                });
              }
              console.log('[MediaStore] Updated', clips.length, 'clips with reloaded file');
            }

            return true;
          } catch (e) {
            console.error('[MediaStore] Failed to reload file:', mediaFile.name, e);
            return false;
          }
        },

        reloadAllFiles: async () => {
          const filesToReload = get().files.filter(f => !f.file);
          if (filesToReload.length === 0) {
            console.log('[MediaStore] No files need reloading');
            return 0;
          }

          // Debug: List all stored handles
          try {
            const allHandles = await projectDB.getAllHandles();
            console.log('[MediaStore] All stored handles in IndexedDB:', allHandles.map(h => h.key));
          } catch (e) {
            console.warn('[MediaStore] Could not list handles:', e);
          }

          console.log('[MediaStore] Files to reload:', filesToReload.map(f => ({ id: f.id, name: f.name })));
          console.log('[MediaStore] Reloading', filesToReload.length, 'files...');
          let reloadedCount = 0;

          for (const mediaFile of filesToReload) {
            const success = await get().reloadFile(mediaFile.id);
            if (success) {
              reloadedCount++;
            }
          }

          console.log('[MediaStore] Reloaded', reloadedCount, '/', filesToReload.length, 'files');
          return reloadedCount;
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
            // Remove from open tabs
            openCompositionIds: state.openCompositionIds.filter((cid) => cid !== id),
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
          const currentPlayhead = timelineStore.playheadPosition;
          const currentClips = timelineStore.clips;

          // Calculate synced playhead for nested composition switch
          let syncedPlayhead: number | null = null;

          if (activeCompositionId && id) {
            // Check if new comp (id) is nested in current comp (activeCompositionId)
            const nestedClip = currentClips.find(
              (c) => c.isComposition && c.compositionId === id
            );
            if (nestedClip) {
              const clipStart = nestedClip.startTime;
              const clipEnd = clipStart + nestedClip.duration;
              const inPoint = nestedClip.inPoint || 0;

              // If current playhead is within the nested clip's range
              if (currentPlayhead >= clipStart && currentPlayhead < clipEnd) {
                // Calculate the child composition's playhead
                syncedPlayhead = (currentPlayhead - clipStart) + inPoint;
              }
            }

            // Check reverse: if current comp is nested in the new comp
            const newComp = compositions.find((c) => c.id === id);
            if (newComp?.timelineData?.clips) {
              const parentClip = newComp.timelineData.clips.find(
                (c: { isComposition?: boolean; compositionId?: string }) =>
                  c.isComposition && c.compositionId === activeCompositionId
              );
              if (parentClip) {
                const clipStart = parentClip.startTime;
                const inPoint = parentClip.inPoint || 0;
                // Calculate parent's playhead from child's current playhead
                syncedPlayhead = clipStart + (currentPlayhead - inPoint);
              }
            }
          }

          // Save current timeline state to the current composition (if any)
          if (activeCompositionId) {
            const timelineData = timelineStore.getSerializableState();
            set((state) => ({
              compositions: state.compositions.map((c) =>
                c.id === activeCompositionId ? { ...c, timelineData } : c
              ),
            }));
            // Invalidate the composition's cached sources since timelineData changed
            compositionRenderer.invalidateComposition(activeCompositionId);
          }

          // Update active composition
          set({ activeCompositionId: id });

          // Load timeline state from the new composition
          // IMPORTANT: Get fresh compositions AFTER saving to include any updates
          if (id) {
            const freshCompositions = get().compositions;
            const newComp = freshCompositions.find((c) => c.id === id);
            timelineStore.loadState(newComp?.timelineData);

            // Apply synced playhead if we calculated one
            if (syncedPlayhead !== null && syncedPlayhead >= 0) {
              timelineStore.setPlayheadPosition(syncedPlayhead);
            }

            // Reset zoom to fully zoomed out and scroll to start
            timelineStore.setZoom(0.1); // MIN_ZOOM
            timelineStore.setScrollX(0);
          } else {
            // No composition selected - clear timeline
            timelineStore.clearTimeline();
          }
        },

        getActiveComposition: () => {
          const { compositions, activeCompositionId } = get();
          return compositions.find((c) => c.id === activeCompositionId);
        },

        openCompositionTab: (id: string) => {
          const { openCompositionIds, setActiveComposition } = get();
          // Add to open tabs if not already open
          if (!openCompositionIds.includes(id)) {
            set({ openCompositionIds: [...openCompositionIds, id] });
          }
          // Switch to the composition
          setActiveComposition(id);
        },

        closeCompositionTab: (id: string) => {
          const { openCompositionIds, activeCompositionId, setActiveComposition } = get();
          const newOpenIds = openCompositionIds.filter((cid) => cid !== id);
          set({ openCompositionIds: newOpenIds });

          // If we closed the active tab, switch to another one
          if (activeCompositionId === id && newOpenIds.length > 0) {
            // Switch to the previous tab or the first available
            const closedIndex = openCompositionIds.indexOf(id);
            const newActiveIndex = Math.min(closedIndex, newOpenIds.length - 1);
            setActiveComposition(newOpenIds[newActiveIndex]);
          } else if (newOpenIds.length === 0) {
            setActiveComposition(null);
          }
        },

        getOpenCompositions: () => {
          const { compositions, openCompositionIds } = get();
          return openCompositionIds
            .map((id) => compositions.find((c) => c.id === id))
            .filter((c): c is Composition => c !== undefined);
        },

        reorderCompositionTabs: (fromIndex: number, toIndex: number) => {
          const { openCompositionIds } = get();
          if (fromIndex < 0 || fromIndex >= openCompositionIds.length) return;
          if (toIndex < 0 || toIndex >= openCompositionIds.length) return;
          if (fromIndex === toIndex) return;

          const newOrder = [...openCompositionIds];
          const [moved] = newOrder.splice(fromIndex, 1);
          newOrder.splice(toIndex, 0, moved);
          set({ openCompositionIds: newOrder });
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

          console.log(`[Proxy] Starting generation for ${mediaFile.name}...`);

          // Check where to store proxies:
          // 1. If project is open → use project's Proxy/ folder (no prompt needed)
          // 2. If no project → proxies go to IndexedDB (browser storage)
          if (projectFileService.isProjectOpen()) {
            const projectName = projectFileService.getProjectData()?.name;
            console.log(`[Proxy] Using project folder: ${projectName}/Proxy/`);
          } else {
            console.log('[Proxy] No project open, using browser storage for proxies');
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

          // Set proxyFps immediately so partial proxy can be used during generation
          set({
            files: get().files.map((f) =>
              f.id === mediaFileId ? { ...f, proxyFps: PROXY_FPS } : f
            ),
          });

          // Helper to save frames to storage
          // Use fileHash for folder naming (for deduplication), fallback to mediaFileId
          const storageKey = mediaFile.fileHash || mediaFileId;
          const saveFrame = async (frame: { id: string; mediaFileId: string; frameIndex: number; blob: Blob; fileHash?: string }) => {
            // Save to project folder if a project is open (primary storage)
            if (projectFileService.isProjectOpen()) {
              await projectFileService.saveProxyFrame(storageKey, frame.frameIndex, frame.blob);
            }
            // Also save to IndexedDB for redundancy and faster access
            await projectDB.saveProxyFrame({
              ...frame,
              id: `${storageKey}_${frame.frameIndex.toString().padStart(6, '0')}`,
              fileHash: mediaFile.fileHash,
            });
          };

          try {
            let result: { frameCount: number; fps: number } | null = null;

            // GPU-accelerated generation (WebCodecs + WebGPU)
            const { getProxyGenerator } = await import('../services/proxyGenerator');
            const generator = getProxyGenerator();

            console.log('[Proxy] Starting GPU-accelerated generation...');
            result = await generator.generate(
              mediaFile.file!,
              mediaFileId,
              (progress) => updateProxyProgress(mediaFileId, progress),
              () => controller.cancelled,
              saveFrame
            );

            if (result) {
              console.log(`[Proxy] GPU generation completed: ${result.frameCount} frames`);
            } else if (!controller.cancelled) {
              console.error('[Proxy] GPU generation failed - no result returned');
            }

            if (result) {
              // Update media file with proxy info
              set({
                files: get().files.map((f) =>
                  f.id === mediaFileId
                    ? {
                        ...f,
                        proxyStatus: 'ready' as ProxyStatus,
                        proxyProgress: 100,
                        proxyFrameCount: result!.frameCount,
                        proxyFps: result!.fps,
                      }
                    : f
                ),
              });
              console.log(`[Proxy] Complete: ${result.frameCount} frames for ${mediaFile.name}`);
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

        // ============ File System Access API ============

        // Import files using the File System Access API (with file handles)
        importFilesWithPicker: async () => {
          const result = await fileSystemService.pickFiles();
          if (!result || result.length === 0) {
            return [];
          }

          const imported: MediaFile[] = [];
          for (const { file, handle } of result) {
            const id = generateId();

            // Store the file handle in memory and IndexedDB for persistence
            fileSystemService.storeFileHandle(id, handle);
            await projectDB.storeHandle(`media_${id}`, handle);
            console.log('[MediaStore] Stored file handle for ID:', id, 'key:', `media_${id}`);

            const type = getMediaType(file);
            const url = URL.createObjectURL(file);
            const [info, thumbnailUrl] = await Promise.all([
              getMediaInfo(file, type),
              createThumbnail(file, type as 'video' | 'image'),
            ]);

            // Calculate file hash for deduplication
            const fileHash = await calculateFileHash(file);

            // Check for existing thumbnail by hash
            let finalThumbnailUrl = thumbnailUrl;
            if (fileHash) {
              try {
                // Try local file system first (if project is open), then IndexedDB
                let existingThumbBlob: Blob | null = null;
                if (projectFileService.isProjectOpen()) {
                  existingThumbBlob = await projectFileService.getThumbnail(fileHash);
                }
                if (!existingThumbBlob) {
                  const existingThumb = await projectDB.getThumbnail(fileHash);
                  if (existingThumb?.blob?.size > 0) {
                    existingThumbBlob = existingThumb.blob;
                  }
                }

                if (existingThumbBlob && existingThumbBlob.size > 0) {
                  finalThumbnailUrl = URL.createObjectURL(existingThumbBlob);
                  console.log('[MediaStore] Reusing existing thumbnail for hash:', fileHash.slice(0, 8));
                } else if (thumbnailUrl) {
                  let thumbBlob: Blob | null = null;
                  if (thumbnailUrl.startsWith('data:')) {
                    const response = await fetch(thumbnailUrl);
                    thumbBlob = await response.blob();
                  } else if (thumbnailUrl.startsWith('blob:')) {
                    const response = await fetch(thumbnailUrl);
                    thumbBlob = await response.blob();
                  }
                  if (thumbBlob && thumbBlob.size > 0) {
                    // Save to local file system if project is open, otherwise to IndexedDB
                    if (projectFileService.isProjectOpen()) {
                      await projectFileService.saveThumbnail(fileHash, thumbBlob);
                      console.log('[MediaStore] Saved thumbnail to project folder:', fileHash.slice(0, 8));
                    } else {
                      await projectDB.saveThumbnail({
                        fileHash,
                        blob: thumbBlob,
                        createdAt: Date.now(),
                      });
                    }
                  }
                }
              } catch (e) {
                console.warn('[MediaStore] Thumbnail dedup error, using original:', e);
              }
            }

            // Check for existing proxy by hash
            let proxyStatus: ProxyStatus = 'none';
            let proxyFrameCount: number | undefined;
            if (fileHash && type === 'video') {
              const existingProxyCount = await projectDB.getProxyFrameCountByHash(fileHash);
              if (existingProxyCount > 0) {
                proxyStatus = 'ready';
                proxyFrameCount = existingProxyCount;
                console.log('[MediaStore] Found existing proxy for hash:', fileHash.slice(0, 8), 'frames:', existingProxyCount);
              }
            }

            const mediaFile: MediaFile = {
              id,
              name: file.name,
              type,
              parentId: null,
              createdAt: Date.now(),
              file,
              url,
              thumbnailUrl: finalThumbnailUrl,
              hasFileHandle: true,
              filePath: handle.name,
              fileHash,
              proxyStatus,
              proxyFrameCount,
              proxyFps: proxyFrameCount ? PROXY_FPS : undefined,
              ...info,
            };

            set((state) => ({
              files: [...state.files, mediaFile],
            }));

            // Save metadata only (no blob) to IndexedDB
            try {
              const storedFile: StoredMediaFile = {
                id: mediaFile.id,
                name: file.name,
                type,
                fileHash,
                duration: info.duration,
                width: info.width,
                height: info.height,
                fps: info.fps,
                codec: info.codec,
                container: info.container,
                fileSize: info.fileSize,
                createdAt: mediaFile.createdAt,
              };
              await projectDB.saveMediaFile(storedFile);
              console.log('[MediaStore] Saved file with handle:', file.name);
            } catch (e) {
              console.warn('[MediaStore] Failed to save file metadata:', e);
            }

            imported.push(mediaFile);
          }

          return imported;
        },

        // Import files with existing handles (from drag-and-drop)
        importFilesWithHandles: async (filesWithHandles) => {
          const imported: MediaFile[] = [];

          for (const { file, handle } of filesWithHandles) {
            const id = generateId();

            // Store the file handle in memory and IndexedDB for persistence
            fileSystemService.storeFileHandle(id, handle);
            await projectDB.storeHandle(`media_${id}`, handle);
            console.log('[MediaStore] Stored file handle from drop for ID:', id);

            const type = getMediaType(file);
            const url = URL.createObjectURL(file);
            const [info, thumbnailUrl] = await Promise.all([
              getMediaInfo(file, type),
              createThumbnail(file, type as 'video' | 'image'),
            ]);

            // Calculate file hash for deduplication
            const fileHash = await calculateFileHash(file);

            // Check for existing thumbnail by hash
            let finalThumbnailUrl = thumbnailUrl;
            if (fileHash) {
              try {
                // Try local file system first (if project is open), then IndexedDB
                let existingThumbBlob: Blob | null = null;
                if (projectFileService.isProjectOpen()) {
                  existingThumbBlob = await projectFileService.getThumbnail(fileHash);
                }
                if (!existingThumbBlob) {
                  const existingThumb = await projectDB.getThumbnail(fileHash);
                  if (existingThumb?.blob?.size > 0) {
                    existingThumbBlob = existingThumb.blob;
                  }
                }

                if (existingThumbBlob && existingThumbBlob.size > 0) {
                  finalThumbnailUrl = URL.createObjectURL(existingThumbBlob);
                  console.log('[MediaStore] Reusing existing thumbnail for hash:', fileHash.slice(0, 8));
                } else if (thumbnailUrl) {
                  let thumbBlob: Blob | null = null;
                  if (thumbnailUrl.startsWith('data:')) {
                    const response = await fetch(thumbnailUrl);
                    thumbBlob = await response.blob();
                  } else if (thumbnailUrl.startsWith('blob:')) {
                    const response = await fetch(thumbnailUrl);
                    thumbBlob = await response.blob();
                  }
                  if (thumbBlob && thumbBlob.size > 0) {
                    // Save to local file system if project is open, otherwise to IndexedDB
                    if (projectFileService.isProjectOpen()) {
                      await projectFileService.saveThumbnail(fileHash, thumbBlob);
                      console.log('[MediaStore] Saved thumbnail to project folder:', fileHash.slice(0, 8));
                    } else {
                      await projectDB.saveThumbnail({
                        fileHash,
                        blob: thumbBlob,
                        createdAt: Date.now(),
                      });
                    }
                  }
                }
              } catch (e) {
                console.warn('[MediaStore] Thumbnail dedup error:', e);
              }
            }

            // Check for existing proxy by hash
            let proxyStatus: ProxyStatus = 'none';
            let proxyFrameCount: number | undefined;
            if (fileHash && type === 'video') {
              const existingProxyCount = await projectDB.getProxyFrameCountByHash(fileHash);
              if (existingProxyCount > 0) {
                proxyStatus = 'ready';
                proxyFrameCount = existingProxyCount;
                console.log('[MediaStore] Reusing existing proxy for hash:', fileHash.slice(0, 8));
              }
            }

            const mediaFile: MediaFile = {
              id,
              name: file.name,
              type,
              parentId: null,
              createdAt: Date.now(),
              file,
              url,
              thumbnailUrl: finalThumbnailUrl,
              duration: info.duration,
              width: info.width,
              height: info.height,
              fps: info.fps,
              codec: info.codec,
              container: info.container,
              fileSize: info.fileSize,
              fileHash,
              hasFileHandle: true,
              filePath: file.name,
              proxyStatus,
              proxyFrameCount,
              proxyFps: proxyFrameCount ? PROXY_FPS : undefined,
              proxyProgress: proxyFrameCount ? 100 : 0,
            };

            set((state) => ({
              files: [...state.files, mediaFile],
            }));

            // Save to IndexedDB
            try {
              const storedFile: StoredMediaFile = {
                id,
                name: file.name,
                type: mediaFile.type,
                fileHash,
                duration: info.duration,
                width: info.width,
                height: info.height,
                fps: info.fps,
                codec: info.codec,
                container: info.container,
                fileSize: info.fileSize,
                createdAt: mediaFile.createdAt,
              };
              await projectDB.saveMediaFile(storedFile);
              console.log('[MediaStore] Saved file with handle from drop:', file.name);
            } catch (e) {
              console.warn('[MediaStore] Failed to save file metadata:', e);
            }

            imported.push(mediaFile);
          }

          return imported;
        },

        // Pick a folder for proxy storage
        pickProxyFolder: async () => {
          const handle = await fileSystemService.pickProxyFolder();
          if (handle) {
            set({ proxyFolderName: handle.name });
            console.log('[MediaStore] Proxy folder set to:', handle.name);
            return true;
          }
          return false;
        },

        // Show file in explorer
        showInExplorer: async (type, mediaFileId) => {
          const result = await fileSystemService.showInExplorer(type, mediaFileId);
          return result;
        },

        // Initialize from IndexedDB - restore files from handles and check proxy status
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

                // Try to restore file from handle
                let file: File | undefined;
                let url = mediaFile.url;
                let thumbnailUrl = mediaFile.thumbnailUrl;

                // Try to get file handle from IndexedDB
                const handle = await projectDB.getStoredHandle(`media_${mediaFile.id}`);
                if (handle && 'getFile' in handle) {
                  try {
                    // Request permission if needed
                    const permission = await (handle as FileSystemFileHandle).queryPermission({ mode: 'read' });
                    if (permission === 'granted') {
                      file = await (handle as FileSystemFileHandle).getFile();
                      url = URL.createObjectURL(file);
                      fileSystemService.storeFileHandle(mediaFile.id, handle as FileSystemFileHandle);
                      console.log('[MediaStore] Restored file from handle:', stored.name);
                    } else {
                      // Try to request permission
                      const newPermission = await (handle as FileSystemFileHandle).requestPermission({ mode: 'read' });
                      if (newPermission === 'granted') {
                        file = await (handle as FileSystemFileHandle).getFile();
                        url = URL.createObjectURL(file);
                        fileSystemService.storeFileHandle(mediaFile.id, handle as FileSystemFileHandle);
                        console.log('[MediaStore] Restored file from handle (after permission):', stored.name);
                      }
                    }
                  } catch (e) {
                    console.warn('[MediaStore] Failed to restore file from handle:', stored.name, e);
                  }
                }

                // Restore thumbnail from hash - try local file system first, then IndexedDB
                if (stored.fileHash) {
                  let thumbBlob: Blob | null = null;
                  if (projectFileService.isProjectOpen()) {
                    thumbBlob = await projectFileService.getThumbnail(stored.fileHash);
                  }
                  if (!thumbBlob) {
                    const thumbData = await projectDB.getThumbnail(stored.fileHash);
                    if (thumbData?.blob) {
                      thumbBlob = thumbData.blob;
                    }
                  }
                  if (thumbBlob) {
                    thumbnailUrl = URL.createObjectURL(thumbBlob);
                  }
                }

                // Check for existing proxy by hash
                let proxyStatus: ProxyStatus = 'none';
                let proxyFrameCount: number | undefined;
                if (stored.type === 'video' && stored.fileHash) {
                  const frameCount = await projectDB.getProxyFrameCountByHash(stored.fileHash);
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
                  // Restore metadata
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
              openCompositionIds: state.openCompositionIds,
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

            // Clear the render frame (removes old project's last frame)
            engine.clearFrame();

            // Restore state
            set({
              files,
              compositions: project.data.compositions as Composition[],
              folders: project.data.folders as MediaFolder[],
              activeCompositionId: null, // Will be set below
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
                  // Ensure the active composition is in the open tabs
                  openCompositionIds: get().openCompositionIds.includes(project.data.activeCompositionId as string)
                    ? get().openCompositionIds
                    : [...get().openCompositionIds, project.data.activeCompositionId as string]
                });
              }
            }

            console.log('[MediaStore] Project loaded:', project.name);
          } catch (e) {
            console.error('[MediaStore] Failed to load project:', e);
            set({ isLoading: false });
            throw e;
          }
        },

        // Create a new empty project
        newProject: () => {
          // Clear timeline first
          const timelineStore = useTimelineStore.getState();
          timelineStore.clearTimeline();

          // Clear the render frame (removes old project's last frame)
          engine.clearFrame();

          // Create new default composition
          const newCompId = `comp-${Date.now()}`;
          const newComposition: Composition = {
            id: newCompId,
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

          // Load the new composition's empty timeline
          timelineStore.loadState(undefined);

          console.log('[MediaStore] New project created');
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
      })
  )
);

// Save current timeline to active composition (for persistence)
function saveTimelineToActiveComposition() {
  const { activeCompositionId } = useMediaStore.getState();
  if (activeCompositionId) {
    const timelineStore = useTimelineStore.getState();
    const timelineData = timelineStore.getSerializableState();
    useMediaStore.setState((state) => ({
      compositions: state.compositions.map((c) =>
        c.id === activeCompositionId ? { ...c, timelineData } : c
      ),
    }));
  }
}

// Export for external use (e.g., after transcription completes)
export function triggerTimelineSave() {
  saveTimelineToActiveComposition();
  console.log('[MediaStore] Timeline saved to composition');
}

// Auto-initialize from IndexedDB on app load
if (typeof window !== 'undefined') {
  // Delay init slightly to ensure store is ready
  setTimeout(async () => {
    // Initialize file system service (restore handles from IndexedDB)
    await fileSystemService.init();

    // Update store with proxy folder name if restored
    const proxyFolderName = fileSystemService.getProxyFolderName();
    if (proxyFolderName) {
      useMediaStore.setState({ proxyFolderName });
    }

    // Initialize media from IndexedDB
    await useMediaStore.getState().initFromDB();

    // Restore active composition's timeline after media files are loaded
    const { activeCompositionId, compositions } = useMediaStore.getState();
    if (activeCompositionId) {
      const activeComp = compositions.find((c) => c.id === activeCompositionId);
      if (activeComp?.timelineData) {
        console.log('[MediaStore] Restoring timeline for:', activeComp.name);
        const timelineStore = useTimelineStore.getState();
        await timelineStore.loadState(activeComp.timelineData);
      }
    }
  }, 100);

  // Save timeline before page unload (for refresh/close)
  window.addEventListener('beforeunload', () => {
    // Don't save if we're clearing cache
    if ((window as any).__CLEARING_CACHE__) return;
    saveTimelineToActiveComposition();
  });

  // Also save timeline periodically (every 30 seconds) as backup
  setInterval(() => {
    // Don't save if we're clearing cache
    if ((window as any).__CLEARING_CACHE__) return;
    saveTimelineToActiveComposition();
  }, 30000);
}
