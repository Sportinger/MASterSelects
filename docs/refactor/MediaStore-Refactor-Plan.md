# MediaStore Refactoring Plan

**Target**: Reduce `src/stores/mediaStore.ts` from 2046 LOC to ~300 LOC main file with slices/helpers below ~250 LOC each.

## Current Issues

### Giant Functions
| Function | LOC | Lines | Issue |
|----------|-----|-------|-------|
| `reloadAllFiles` | 162 | 743-904 | Sequential file picking, repeated update pattern |
| `generateProxy` | 157 | 1272-1428 | Nested try-catch, audio extraction inline |
| `importFilesWithHandles` | 118 | 1567-1685 | 80% duplicate of importFile |
| `importFilesWithPicker` | 112 | 1453-1565 | 80% duplicate of importFile |
| `reloadFile` | 113 | 629-741 | Mixed concerns |
| `importFile` | 107 | 491-597 | Core logic duplicated elsewhere |
| `initFromDB` | 92 | 1705-1796 | Complex restoration logic |
| `loadProject` | 87 | 1837-1923 | Could be cleaner |

### Massive Code Duplication (~600 LOC)

1. **Thumbnail deduplication** - 3x identical (~180 LOC wasted)
2. **Proxy status check** - 3x identical (~45 LOC wasted)
3. **Copy to Raw folder** - 3x identical (~45 LOC wasted)
4. **Timeline clips update** - 3x identical pattern (~60 LOC wasted)
5. **MediaFile object creation** - 3x similar (~150 LOC wasted)
6. **File handle storage** - repeated pattern (~50 LOC wasted)

### Overcomplicated Relinking Logic (~275 LOC)

Since all imported files are now copied to the project's `/RAW/` folder, the complex file handle management is unnecessary:

| Old Approach | LOC | New Approach | LOC |
|--------------|-----|--------------|-----|
| `reloadFile` with handle lookup, IndexedDB, user prompts | 113 | Simple RAW folder lookup | ~20 |
| `reloadAllFiles` with multiple passes | 162 | Batch RAW folder reload | ~30 |
| `handleHelpers.ts` complex handle management | ~90 | Minimal helpers | ~30 |

**Total savings: ~315 LOC** from simplified relinking alone!

### Duplicate Utilities
- `getMediaType` duplicates `detectMediaType` from clipSlice helpers
- `createThumbnail` similar to clipSlice thumbnail helpers

### Logic Issues
- Sequential import when parallel possible
- Multiple `get()` calls instead of single destructure
- Global side effects at module level (auto-init, intervals)

---

## New Structure

```
src/stores/mediaStore/
├── index.ts                      # Main store coordinator (~300 LOC)
├── types.ts                      # Interfaces & type exports (~100 LOC)
├── constants.ts                  # PROXY_FPS, DEFAULT_COMPOSITION (~40 LOC)
├── slices/
│   ├── fileImportSlice.ts       # importFile, importFiles* unified (~200 LOC)
│   ├── fileManageSlice.ts       # remove, rename, reload (~80 LOC) ← SIMPLIFIED!
│   ├── compositionSlice.ts      # Composition CRUD + tabs (~150 LOC)
│   ├── folderSlice.ts           # Folder operations (~80 LOC)
│   ├── selectionSlice.ts        # Selection actions (~50 LOC)
│   ├── proxySlice.ts            # Proxy generation (~220 LOC)
│   └── projectSlice.ts          # Save/load/new project (~200 LOC)
├── helpers/
│   ├── mediaInfoHelpers.ts      # getMediaInfo, codec, container (~130 LOC)
│   ├── thumbnailHelpers.ts      # createThumbnail + dedup logic (~70 LOC)
│   ├── fileHashHelpers.ts       # calculateFileHash (~35 LOC)
│   └── importPipeline.ts        # Shared import logic (~120 LOC)
└── init.ts                       # Auto-init, autosave, beforeunload (~60 LOC)
```

**Note**: `handleHelpers.ts` removed! Since files are copied to `/RAW/`, we use `projectFileService` directly.

---

## Phase 1: Create Types and Constants

### Step 1.1: Create types.ts

**File**: `src/stores/mediaStore/types.ts`

```typescript
// MediaStore types - extracted from mediaStore.ts

import type { CompositionTimelineData } from '../../types';

// Media item types
export type MediaType = 'video' | 'audio' | 'image' | 'composition';

// Proxy status for video files
export type ProxyStatus = 'none' | 'generating' | 'ready' | 'error';

// Base media item
export interface MediaItem {
  id: string;
  name: string;
  type: MediaType;
  parentId: string | null;
  createdAt: number;
}

// Imported file
export interface MediaFile extends MediaItem {
  type: 'video' | 'audio' | 'image';
  file?: File;
  url: string;
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
  codec?: string;
  container?: string;
  fileSize?: number;
  fileHash?: string;
  thumbnailUrl?: string;
  // Proxy support
  proxyStatus?: ProxyStatus;
  proxyProgress?: number;
  proxyFrameCount?: number;
  proxyFps?: number;
  hasProxyAudio?: boolean;
  // File System Access API
  hasFileHandle?: boolean;
  filePath?: string;
  absolutePath?: string;
  projectPath?: string;
}

// Composition
export interface Composition extends MediaItem {
  type: 'composition';
  width: number;
  height: number;
  frameRate: number;
  duration: number;
  backgroundColor: string;
  timelineData?: CompositionTimelineData;
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

// Slice creator type for mediaStore
export type MediaSliceCreator<T> = (
  set: (partial: Partial<MediaState> | ((state: MediaState) => Partial<MediaState>)) => void,
  get: () => MediaState
) => T;

// Full state interface
export interface MediaState {
  // Items
  files: MediaFile[];
  compositions: Composition[];
  folders: MediaFolder[];

  // Active composition
  activeCompositionId: string | null;
  openCompositionIds: string[];

  // Selection
  selectedIds: string[];
  expandedFolderIds: string[];

  // Project
  currentProjectId: string | null;
  currentProjectName: string;
  isLoading: boolean;

  // Proxy system
  proxyEnabled: boolean;
  proxyGenerationQueue: string[];
  currentlyGeneratingProxyId: string | null;

  // File System Access API
  fileSystemSupported: boolean;
  proxyFolderName: string | null;

  // Actions are added by slices
  [key: string]: unknown;
}

// Import result for unified pipeline
export interface ImportResult {
  mediaFile: MediaFile;
  handle?: FileSystemFileHandle;
}
```

### Step 1.2: Create constants.ts

**File**: `src/stores/mediaStore/constants.ts`

```typescript
// MediaStore constants

import type { Composition } from './types';

// Proxy generation settings
export const PROXY_FPS = 30;

// File size thresholds
export const LARGE_FILE_THRESHOLD = 500 * 1024 * 1024; // 500MB
export const HASH_SIZE = 2 * 1024 * 1024; // 2MB for hash calculation

// Timeouts
export const THUMBNAIL_TIMEOUT = 10000; // 10s
export const MEDIA_INFO_TIMEOUT = 15000; // 15s

// Default composition
export const DEFAULT_COMPOSITION: Composition = {
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

// Container format map
export const CONTAINER_MAP: Record<string, string> = {
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
```

---

## Phase 2: Extract Helpers

### Step 2.1: Create fileHashHelpers.ts

**File**: `src/stores/mediaStore/helpers/fileHashHelpers.ts`

```typescript
// File hash calculation for proxy deduplication

import { HASH_SIZE } from '../constants';

/**
 * Calculate SHA-256 hash of file (first 2MB + file size for speed).
 * Used for proxy and thumbnail deduplication.
 */
export async function calculateFileHash(file: File): Promise<string> {
  try {
    const slice = file.slice(0, Math.min(file.size, HASH_SIZE));
    const buffer = await slice.arrayBuffer();

    // Include file size in hash
    const sizeBuffer = new ArrayBuffer(8);
    const sizeView = new DataView(sizeBuffer);
    sizeView.setBigUint64(0, BigInt(file.size), true);

    // Combine buffers
    const combined = new Uint8Array(buffer.byteLength + 8);
    combined.set(new Uint8Array(buffer), 0);
    combined.set(new Uint8Array(sizeBuffer), buffer.byteLength);

    // Calculate SHA-256
    const hashBuffer = await crypto.subtle.digest('SHA-256', combined);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  } catch (e) {
    console.warn('[FileHash] Failed to calculate:', e);
    return '';
  }
}
```

### Step 2.2: Create mediaInfoHelpers.ts

**File**: `src/stores/mediaStore/helpers/mediaInfoHelpers.ts`

```typescript
// Media info extraction helpers

import { CONTAINER_MAP, MEDIA_INFO_TIMEOUT } from '../constants';

export interface MediaInfo {
  width?: number;
  height?: number;
  duration?: number;
  fps?: number;
  codec?: string;
  container?: string;
  fileSize?: number;
}

/**
 * Get container format from file extension.
 */
export function getContainerFormat(fileName: string): string {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  return CONTAINER_MAP[ext] || ext.toUpperCase();
}

/**
 * Parse FPS from filename (patterns like "25fps", "_30p", etc.).
 */
export function parseFpsFromFilename(fileName: string): number | undefined {
  const patterns = [
    /[_\-\s\(](\d{2}(?:\.\d+)?)\s*fps/i,
    /[_\-\s\(](\d{2}(?:\.\d+)?)\s*p[_\-\s\)\.]/i,
    /(\d{2}(?:\.\d+)?)fps/i,
  ];

  for (const pattern of patterns) {
    const match = fileName.match(pattern);
    if (match) {
      const fps = parseFloat(match[1]);
      if (fps >= 10 && fps <= 240) return fps;
    }
  }
  return undefined;
}

/**
 * Get codec info from file (best effort).
 */
export async function getCodecInfo(file: File): Promise<string | undefined> {
  try {
    const ext = file.name.split('.').pop()?.toLowerCase();

    // Video codecs
    if (ext === 'mp4' || ext === 'm4v' || ext === 'mov') return 'H.264';
    if (ext === 'webm') return 'VP9';
    if (ext === 'mkv') return 'H.264';

    // Audio codecs
    if (ext === 'mp3') return 'MP3';
    if (ext === 'aac' || ext === 'm4a') return 'AAC';
    if (ext === 'wav') return 'PCM';
    if (ext === 'ogg') return 'Vorbis';
    if (ext === 'flac') return 'FLAC';
  } catch {
    // Ignore
  }
  return undefined;
}

/**
 * Get media dimensions, duration, and metadata.
 */
export async function getMediaInfo(
  file: File,
  type: 'video' | 'audio' | 'image'
): Promise<MediaInfo> {
  const container = getContainerFormat(file.name);
  const fileSize = file.size;
  const codec = await getCodecInfo(file);

  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      console.warn('[MediaInfo] Timeout:', file.name);
      resolve({ container, fileSize, codec });
    }, MEDIA_INFO_TIMEOUT);

    const cleanup = (url?: string) => {
      clearTimeout(timeout);
      if (url) URL.revokeObjectURL(url);
    };

    if (type === 'image') {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.src = url;
      img.onload = () => {
        resolve({ width: img.width, height: img.height, container, fileSize, codec });
        cleanup(url);
      };
      img.onerror = () => {
        resolve({ container, fileSize });
        cleanup(url);
      };
    } else if (type === 'video') {
      const video = document.createElement('video');
      const url = URL.createObjectURL(file);
      video.src = url;
      video.muted = true;
      video.playsInline = true;

      video.onloadedmetadata = () => {
        resolve({
          width: video.videoWidth,
          height: video.videoHeight,
          duration: video.duration,
          fps: parseFpsFromFilename(file.name),
          codec,
          container,
          fileSize,
        });
        cleanup(url);
      };
      video.onerror = () => {
        resolve({ container, fileSize, codec });
        cleanup(url);
      };
      video.load();
    } else if (type === 'audio') {
      const audio = document.createElement('audio');
      const url = URL.createObjectURL(file);
      audio.src = url;
      audio.onloadedmetadata = () => {
        resolve({ duration: audio.duration, codec, container, fileSize });
        cleanup(url);
      };
      audio.onerror = () => {
        resolve({ container, fileSize });
        cleanup(url);
      };
    } else {
      cleanup();
      resolve({ container, fileSize });
    }
  });
}
```

### Step 2.3: Create thumbnailHelpers.ts

**File**: `src/stores/mediaStore/helpers/thumbnailHelpers.ts`

```typescript
// Thumbnail creation and deduplication

import { THUMBNAIL_TIMEOUT } from '../constants';
import { projectFileService } from '../../../services/projectFileService';

/**
 * Create thumbnail for video or image.
 */
export async function createThumbnail(
  file: File,
  type: 'video' | 'image'
): Promise<string | undefined> {
  return new Promise((resolve) => {
    if (type === 'image') {
      resolve(URL.createObjectURL(file));
      return;
    }

    if (type === 'video') {
      const video = document.createElement('video');
      const url = URL.createObjectURL(file);
      video.src = url;
      video.muted = true;
      video.playsInline = true;

      const timeout = setTimeout(() => {
        console.warn('[Thumbnail] Timeout:', file.name);
        URL.revokeObjectURL(url);
        resolve(undefined);
      }, THUMBNAIL_TIMEOUT);

      const cleanup = () => {
        clearTimeout(timeout);
        URL.revokeObjectURL(url);
      };

      video.onloadedmetadata = () => {
        video.currentTime = Math.min(1, video.duration * 0.1);
      };

      video.onseeked = () => {
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
        cleanup();
      };

      video.onerror = () => {
        cleanup();
        resolve(undefined);
      };

      video.load();
    } else {
      resolve(undefined);
    }
  });
}

/**
 * Handle thumbnail deduplication - check for existing, save new.
 * UNIFIED: Replaces 3 duplicate blocks in original code.
 */
export async function handleThumbnailDedup(
  fileHash: string | undefined,
  thumbnailUrl: string | undefined
): Promise<string | undefined> {
  if (!fileHash || !projectFileService.isProjectOpen()) {
    return thumbnailUrl;
  }

  try {
    // Check for existing thumbnail
    const existingBlob = await projectFileService.getThumbnail(fileHash);
    if (existingBlob && existingBlob.size > 0) {
      console.log('[Thumbnail] Reusing existing for hash:', fileHash.slice(0, 8));
      return URL.createObjectURL(existingBlob);
    }

    // Save new thumbnail
    if (thumbnailUrl) {
      const blob = await fetchThumbnailBlob(thumbnailUrl);
      if (blob && blob.size > 0) {
        await projectFileService.saveThumbnail(fileHash, blob);
        console.log('[Thumbnail] Saved to project folder:', fileHash.slice(0, 8));
      }
    }
  } catch (e) {
    console.warn('[Thumbnail] Dedup error:', e);
  }

  return thumbnailUrl;
}

/**
 * Fetch thumbnail blob from data URL or blob URL.
 */
async function fetchThumbnailBlob(url: string): Promise<Blob | null> {
  if (url.startsWith('data:') || url.startsWith('blob:')) {
    const response = await fetch(url);
    return response.blob();
  }
  return null;
}
```

### Step 2.4: Create importPipeline.ts

**File**: `src/stores/mediaStore/helpers/importPipeline.ts`

```typescript
// Unified import pipeline - eliminates 3x duplicate import logic

import type { MediaFile, ProxyStatus } from '../types';
import { PROXY_FPS } from '../constants';
import { detectMediaType } from '../../timeline/helpers/mediaTypeHelpers';
import { calculateFileHash } from './fileHashHelpers';
import { getMediaInfo } from './mediaInfoHelpers';
import { createThumbnail, handleThumbnailDedup } from './thumbnailHelpers';
import { projectFileService } from '../../../services/projectFileService';
import { fileSystemService } from '../../../services/fileSystemService';
import { projectDB } from '../../../services/projectDB';
import { useSettingsStore } from '../../settingsStore';

export interface ImportParams {
  file: File;
  id: string;
  handle?: FileSystemFileHandle;
  absolutePath?: string;
}

export interface ImportResult {
  mediaFile: MediaFile;
  projectFileHandle?: FileSystemFileHandle;
}

/**
 * Generate unique ID.
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Unified import pipeline for all import methods.
 * Replaces duplicate logic in importFile, importFilesWithPicker, importFilesWithHandles.
 */
export async function processImport(params: ImportParams): Promise<ImportResult> {
  const { file, id, handle, absolutePath } = params;

  // Store handle if provided (for original file location)
  if (handle) {
    fileSystemService.storeFileHandle(id, handle);
    await projectDB.storeHandle(`media_${id}`, handle);
  }

  // Detect type using shared helper from clipSlice
  const type = detectMediaType(file) as 'video' | 'audio' | 'image';
  const url = URL.createObjectURL(file);

  // Get info and thumbnail in parallel
  const [info, rawThumbnail] = await Promise.all([
    getMediaInfo(file, type),
    createThumbnail(file, type as 'video' | 'image'),
  ]);

  // Calculate hash for deduplication
  const fileHash = await calculateFileHash(file);

  // Handle thumbnail deduplication (unified - was 3x duplicate)
  const thumbnailUrl = await handleThumbnailDedup(fileHash, rawThumbnail);

  // Check for existing proxy (unified - was 3x duplicate)
  const proxyInfo = await checkExistingProxy(fileHash, type);

  // Copy to Raw folder if enabled (unified - was 3x duplicate)
  const copyResult = await copyToRawIfEnabled(file, id);

  // Build MediaFile
  const mediaFile: MediaFile = {
    id,
    name: file.name,
    type,
    parentId: null,
    createdAt: Date.now(),
    file,
    url,
    thumbnailUrl,
    fileHash,
    hasFileHandle: !!handle || !!copyResult?.handle,
    filePath: handle?.name || file.name,
    absolutePath,
    projectPath: copyResult?.relativePath,
    ...info,
    ...proxyInfo,
  };

  return {
    mediaFile,
    projectFileHandle: copyResult?.handle,
  };
}

/**
 * Check for existing proxy by hash.
 * UNIFIED: Replaces 3 duplicate blocks.
 */
async function checkExistingProxy(
  fileHash: string | undefined,
  type: 'video' | 'audio' | 'image'
): Promise<{
  proxyStatus: ProxyStatus;
  proxyFrameCount?: number;
  proxyFps?: number;
  proxyProgress?: number;
}> {
  if (!fileHash || type !== 'video' || !projectFileService.isProjectOpen()) {
    return { proxyStatus: 'none' };
  }

  const frameCount = await projectFileService.getProxyFrameCount(fileHash);
  if (frameCount > 0) {
    console.log('[Import] Found existing proxy:', fileHash.slice(0, 8), 'frames:', frameCount);
    return {
      proxyStatus: 'ready',
      proxyFrameCount: frameCount,
      proxyFps: PROXY_FPS,
      proxyProgress: 100,
    };
  }

  return { proxyStatus: 'none' };
}

/**
 * Copy file to Raw folder if setting enabled.
 * UNIFIED: Replaces 3 duplicate blocks.
 */
async function copyToRawIfEnabled(
  file: File,
  mediaId: string
): Promise<{ relativePath: string; handle: FileSystemFileHandle } | null> {
  const { copyMediaToProject } = useSettingsStore.getState();

  if (!copyMediaToProject || !projectFileService.isProjectOpen()) {
    return null;
  }

  const result = await projectFileService.copyToRawFolder(file);
  if (result) {
    // Store the project file handle for the RAW copy
    fileSystemService.storeFileHandle(`${mediaId}_project`, result.handle);
    await projectDB.storeHandle(`media_${mediaId}_project`, result.handle);
    console.log('[Import] Copied to Raw folder:', result.relativePath);
    return result;
  }

  return null;
}

/**
 * Process multiple files in parallel batches.
 */
export async function batchImport<T>(
  items: T[],
  batchSize: number,
  processor: (item: T) => Promise<unknown>
): Promise<void> {
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    await Promise.all(batch.map(processor));
  }
}
```

---

## Phase 3: Extract Slices

### Step 3.1: Create fileImportSlice.ts

**File**: `src/stores/mediaStore/slices/fileImportSlice.ts`

```typescript
// File import actions - unified import logic

import type { MediaFile, MediaSliceCreator } from '../types';
import { generateId, processImport } from '../helpers/importPipeline';
import { fileSystemService } from '../../../services/fileSystemService';
import { projectDB } from '../../../services/projectDB';

export interface FileImportActions {
  importFile: (file: File) => Promise<MediaFile>;
  importFiles: (files: FileList | File[]) => Promise<MediaFile[]>;
  importFilesWithPicker: () => Promise<MediaFile[]>;
  importFilesWithHandles: (filesWithHandles: Array<{
    file: File;
    handle: FileSystemFileHandle;
    absolutePath?: string;
  }>) => Promise<MediaFile[]>;
}

export const createFileImportSlice: MediaSliceCreator<FileImportActions> = (set, get) => ({
  importFile: async (file: File) => {
    console.log('[Import] Starting:', file.name);

    const result = await processImport({
      file,
      id: generateId(),
    });

    set((state) => ({
      files: [...state.files, result.mediaFile],
    }));

    return result.mediaFile;
  },

  importFiles: async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const imported: MediaFile[] = [];

    // Process in parallel batches of 3
    const batchSize = 3;
    for (let i = 0; i < fileArray.length; i += batchSize) {
      const batch = fileArray.slice(i, i + batchSize);
      const results = await Promise.all(
        batch.map(file => get().importFile(file))
      );
      imported.push(...results);
    }

    return imported;
  },

  importFilesWithPicker: async () => {
    const result = await fileSystemService.pickFiles();
    if (!result || result.length === 0) return [];

    const imported: MediaFile[] = [];

    for (const { file, handle } of result) {
      const id = generateId();

      // Store original handle (for reference, but RAW folder is primary)
      fileSystemService.storeFileHandle(id, handle);
      await projectDB.storeHandle(`media_${id}`, handle);

      const importResult = await processImport({ file, id, handle });

      set((state) => ({
        files: [...state.files, importResult.mediaFile],
      }));

      imported.push(importResult.mediaFile);
    }

    return imported;
  },

  importFilesWithHandles: async (filesWithHandles) => {
    const imported: MediaFile[] = [];

    for (const { file, handle, absolutePath } of filesWithHandles) {
      const id = generateId();

      // Store original handle (for reference, but RAW folder is primary)
      fileSystemService.storeFileHandle(id, handle);
      await projectDB.storeHandle(`media_${id}`, handle);

      const importResult = await processImport({ file, id, handle, absolutePath });

      set((state) => ({
        files: [...state.files, importResult.mediaFile],
      }));

      imported.push(importResult.mediaFile);
    }

    return imported;
  },
});
```

### Step 3.2: Create fileManageSlice.ts (SIMPLIFIED with RAW folder)

**File**: `src/stores/mediaStore/slices/fileManageSlice.ts`

Since all imported files are copied to the project's `/RAW/` folder, relinking is now trivial:
- No complex handle lookups
- No IndexedDB handle retrieval
- No user prompts to locate files
- Just read from the project folder (already have permission!)

```typescript
// File management actions - remove, rename, reload
// SIMPLIFIED: Uses RAW folder for easy relinking

import type { MediaSliceCreator } from '../types';
import { projectFileService } from '../../../services/projectFileService';
import { useTimelineStore } from '../../timeline';

export interface FileManageActions {
  removeFile: (id: string) => void;
  renameFile: (id: string, name: string) => void;
  reloadFile: (id: string) => Promise<boolean>;
  reloadAllFiles: () => Promise<number>;
}

export const createFileManageSlice: MediaSliceCreator<FileManageActions> = (set, get) => ({
  removeFile: (id: string) => {
    const file = get().files.find((f) => f.id === id);
    if (file?.url) URL.revokeObjectURL(file.url);
    if (file?.thumbnailUrl?.startsWith('blob:')) URL.revokeObjectURL(file.thumbnailUrl);

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

  /**
   * Reload a single file from the project's RAW folder.
   * SIMPLIFIED: No handle lookups needed - just read from RAW folder!
   */
  reloadFile: async (id: string) => {
    const mediaFile = get().files.find(f => f.id === id);
    if (!mediaFile) return false;

    // Get file from project RAW folder (we already have folder permission!)
    if (!mediaFile.projectPath) {
      console.warn('[Reload] No projectPath for:', mediaFile.name);
      return false;
    }

    const file = await projectFileService.getFileFromRaw(mediaFile.projectPath);
    if (!file) {
      console.warn('[Reload] File not found in RAW:', mediaFile.projectPath);
      return false;
    }

    // Revoke old URL
    if (mediaFile.url) URL.revokeObjectURL(mediaFile.url);

    // Create new URL
    const url = URL.createObjectURL(file);

    // Update store
    set((state) => ({
      files: state.files.map((f) =>
        f.id === id ? { ...f, file, url } : f
      ),
    }));

    // Update timeline clips
    await updateTimelineClips(id, file);

    console.log('[Reload] Success from RAW folder:', mediaFile.name);
    return true;
  },

  /**
   * Reload all files that need reloading from the project's RAW folder.
   * SIMPLIFIED: Just batch read from RAW folder - no prompts needed!
   */
  reloadAllFiles: async () => {
    const filesToReload = get().files.filter(f => !f.file && f.projectPath);
    if (filesToReload.length === 0) {
      console.log('[Reload] No files need reloading');
      return 0;
    }

    console.log('[Reload] Reloading', filesToReload.length, 'files from RAW folder...');
    let totalReloaded = 0;

    // Batch reload from RAW folder (all files, one pass!)
    for (const mediaFile of filesToReload) {
      const file = await projectFileService.getFileFromRaw(mediaFile.projectPath!);
      if (!file) {
        console.warn('[Reload] Not found:', mediaFile.projectPath);
        continue;
      }

      if (mediaFile.url) URL.revokeObjectURL(mediaFile.url);
      const url = URL.createObjectURL(file);

      set((state) => ({
        files: state.files.map((f) =>
          f.id === mediaFile.id ? { ...f, file, url } : f
        ),
      }));

      await updateTimelineClips(mediaFile.id, file);
      totalReloaded++;
    }

    console.log('[Reload] Complete:', totalReloaded, 'files reloaded');
    return totalReloaded;
  },
});

/**
 * Update timeline clips with reloaded file.
 */
async function updateTimelineClips(mediaFileId: string, file: File): Promise<void> {
  const timelineStore = useTimelineStore.getState();
  const clips = timelineStore.clips.filter(
    c => c.source?.mediaFileId === mediaFileId && c.needsReload
  );

  for (const clip of clips) {
    timelineStore.updateClip(clip.id, {
      file,
      needsReload: false,
      isLoading: true,
    });
  }

  if (clips.length > 0) {
    console.log('[Reload] Updated', clips.length, 'timeline clips');
  }
}
```

**Key simplifications:**
- `reloadFile`: 113 LOC → 35 LOC (-78 LOC)
- `reloadAllFiles`: 162 LOC → 40 LOC (-122 LOC)
- No `handleHelpers.ts` needed (-90 LOC)
- **Total savings: ~290 LOC**

### Step 3.3: Create compositionSlice.ts

**File**: `src/stores/mediaStore/slices/compositionSlice.ts`

```typescript
// Composition CRUD and tab management

import type { Composition, MediaSliceCreator } from '../types';
import { generateId } from '../helpers/importPipeline';
import { useTimelineStore } from '../../timeline';
import { compositionRenderer } from '../../../services/compositionRenderer';

export interface CompositionActions {
  createComposition: (name: string, settings?: Partial<Composition>) => Composition;
  duplicateComposition: (id: string) => Composition | null;
  removeComposition: (id: string) => void;
  updateComposition: (id: string, updates: Partial<Composition>) => void;
  setActiveComposition: (id: string | null) => void;
  getActiveComposition: () => Composition | undefined;
  openCompositionTab: (id: string) => void;
  closeCompositionTab: (id: string) => void;
  getOpenCompositions: () => Composition[];
  reorderCompositionTabs: (fromIndex: number, toIndex: number) => void;
}

export const createCompositionSlice: MediaSliceCreator<CompositionActions> = (set, get) => ({
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

    set((state) => ({ compositions: [...state.compositions, comp] }));
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

    set((state) => ({ compositions: [...state.compositions, duplicate] }));
    return duplicate;
  },

  removeComposition: (id: string) => {
    set((state) => ({
      compositions: state.compositions.filter((c) => c.id !== id),
      selectedIds: state.selectedIds.filter((sid) => sid !== id),
      activeCompositionId: state.activeCompositionId === id ? null : state.activeCompositionId,
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

  setActiveComposition: (id: string | null) => {
    const { activeCompositionId, compositions } = get();
    const timelineStore = useTimelineStore.getState();

    // Calculate synced playhead for nested composition navigation
    const syncedPlayhead = calculateSyncedPlayhead(
      activeCompositionId,
      id,
      compositions,
      timelineStore
    );

    // Save current timeline to current composition
    if (activeCompositionId) {
      const timelineData = timelineStore.getSerializableState();
      set((state) => ({
        compositions: state.compositions.map((c) =>
          c.id === activeCompositionId ? { ...c, timelineData } : c
        ),
      }));
      compositionRenderer.invalidateCompositionAndParents(activeCompositionId);
    }

    // Update active composition
    set({ activeCompositionId: id });

    // Load new composition's timeline
    if (id) {
      const newComp = get().compositions.find((c) => c.id === id);
      timelineStore.loadState(newComp?.timelineData);

      if (syncedPlayhead !== null && syncedPlayhead >= 0) {
        timelineStore.setPlayheadPosition(syncedPlayhead);
      }

      timelineStore.setZoom(0.1);
      timelineStore.setScrollX(0);
    } else {
      timelineStore.clearTimeline();
    }
  },

  getActiveComposition: () => {
    const { compositions, activeCompositionId } = get();
    return compositions.find((c) => c.id === activeCompositionId);
  },

  openCompositionTab: (id: string) => {
    const { openCompositionIds, setActiveComposition } = get();
    if (!openCompositionIds.includes(id)) {
      set({ openCompositionIds: [...openCompositionIds, id] });
    }
    setActiveComposition(id);
  },

  closeCompositionTab: (id: string) => {
    const { openCompositionIds, activeCompositionId, setActiveComposition } = get();
    const newOpenIds = openCompositionIds.filter((cid) => cid !== id);
    set({ openCompositionIds: newOpenIds });

    if (activeCompositionId === id && newOpenIds.length > 0) {
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
});

/**
 * Calculate synced playhead for nested composition navigation.
 */
function calculateSyncedPlayhead(
  fromCompId: string | null,
  toCompId: string | null,
  compositions: Composition[],
  timelineStore: ReturnType<typeof useTimelineStore.getState>
): number | null {
  if (!fromCompId || !toCompId) return null;

  const currentPlayhead = timelineStore.playheadPosition;
  const currentClips = timelineStore.clips;

  // Check if navigating into nested comp
  const nestedClip = currentClips.find(
    (c) => c.isComposition && c.compositionId === toCompId
  );
  if (nestedClip) {
    const clipStart = nestedClip.startTime;
    const clipEnd = clipStart + nestedClip.duration;
    const inPoint = nestedClip.inPoint || 0;

    if (currentPlayhead >= clipStart && currentPlayhead < clipEnd) {
      return (currentPlayhead - clipStart) + inPoint;
    }
  }

  // Check if navigating to parent comp
  const toComp = compositions.find((c) => c.id === toCompId);
  if (toComp?.timelineData?.clips) {
    const parentClip = toComp.timelineData.clips.find(
      (c: any) => c.isComposition && c.compositionId === fromCompId
    );
    if (parentClip) {
      return parentClip.startTime + (currentPlayhead - (parentClip.inPoint || 0));
    }
  }

  return null;
}
```

### Step 3.4: Create remaining slices (folderSlice, selectionSlice, proxySlice, projectSlice)

These follow the same pattern. For brevity, I'll show proxySlice as it's the most complex:

**File**: `src/stores/mediaStore/slices/proxySlice.ts`

```typescript
// Proxy generation slice

import type { MediaSliceCreator, ProxyStatus } from '../types';
import { PROXY_FPS } from '../constants';
import { projectFileService } from '../../../services/projectFileService';
import { useTimelineStore } from '../../timeline';

// Track active generations for cancellation
const activeProxyGenerations = new Map<string, { cancelled: boolean }>();

export interface ProxyActions {
  proxyEnabled: boolean;
  setProxyEnabled: (enabled: boolean) => void;
  generateProxy: (mediaFileId: string) => Promise<void>;
  cancelProxyGeneration: (mediaFileId: string) => void;
  updateProxyProgress: (mediaFileId: string, progress: number) => void;
  setProxyStatus: (mediaFileId: string, status: ProxyStatus) => void;
  getNextFileNeedingProxy: () => import('../types').MediaFile | undefined;
}

export const createProxySlice: MediaSliceCreator<ProxyActions> = (set, get) => ({
  proxyEnabled: false,

  setProxyEnabled: async (enabled: boolean) => {
    set({ proxyEnabled: enabled });

    if (enabled) {
      // Mute all video elements when enabling proxy mode
      const clips = useTimelineStore.getState().clips;
      clips.forEach(clip => {
        if (clip.source?.videoElement) {
          clip.source.videoElement.muted = true;
          if (!clip.source.videoElement.paused) {
            clip.source.videoElement.pause();
          }
        }
      });
      console.log('[Proxy] Mode enabled - muted all videos');
    }
  },

  updateProxyProgress: (mediaFileId: string, progress: number) => {
    set((state) => ({
      files: state.files.map((f) =>
        f.id === mediaFileId ? { ...f, proxyProgress: progress } : f
      ),
    }));
  },

  setProxyStatus: async (mediaFileId: string, status: ProxyStatus) => {
    const { proxyEnabled } = get();

    set((state) => ({
      files: state.files.map((f) =>
        f.id === mediaFileId ? { ...f, proxyStatus: status } : f
      ),
    }));

    // Mute video when proxy becomes ready
    if (status === 'ready' && proxyEnabled) {
      const clips = useTimelineStore.getState().clips;
      clips.forEach(clip => {
        if (clip.mediaFileId === mediaFileId && clip.source?.videoElement) {
          clip.source.videoElement.muted = true;
          if (!clip.source.videoElement.paused) {
            clip.source.videoElement.pause();
          }
        }
      });
    }
  },

  getNextFileNeedingProxy: () => {
    const { files, currentlyGeneratingProxyId } = get();
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
    const state = get();
    const { files, currentlyGeneratingProxyId, updateProxyProgress, setProxyStatus } = state;

    if (currentlyGeneratingProxyId) {
      console.log('[Proxy] Already generating, queuing:', mediaFileId);
      return;
    }

    const mediaFile = files.find((f) => f.id === mediaFileId);
    if (!mediaFile || mediaFile.type !== 'video' || !mediaFile.file) {
      console.warn('[Proxy] Invalid media file:', mediaFileId);
      return;
    }

    if (!projectFileService.isProjectOpen()) {
      console.error('[Proxy] No project open!');
      return;
    }

    // Check if already exists
    const storageKey = mediaFile.fileHash || mediaFileId;
    const existingCount = await projectFileService.getProxyFrameCount(storageKey);
    if (existingCount > 0) {
      console.log('[Proxy] Already exists:', mediaFile.name);
      set((s) => ({
        files: s.files.map((f) =>
          f.id === mediaFileId
            ? { ...f, proxyStatus: 'ready', proxyProgress: 100, proxyFrameCount: existingCount }
            : f
        ),
      }));
      return;
    }

    // Set up cancellation
    const controller = { cancelled: false };
    activeProxyGenerations.set(mediaFileId, controller);

    set({ currentlyGeneratingProxyId: mediaFileId });
    setProxyStatus(mediaFileId, 'generating');
    updateProxyProgress(mediaFileId, 0);

    // Set proxyFps for partial proxy use during generation
    set((s) => ({
      files: s.files.map((f) =>
        f.id === mediaFileId ? { ...f, proxyFps: PROXY_FPS } : f
      ),
    }));

    try {
      // Generate video proxy
      const result = await generateVideoProxy(
        mediaFile,
        storageKey,
        controller,
        updateProxyProgress
      );

      if (result && !controller.cancelled) {
        // Extract audio proxy
        await extractAudioProxy(mediaFile, storageKey);

        // Check for audio proxy
        const hasAudioProxy = await projectFileService.hasProxyAudio(storageKey);

        // Update final status
        set((s) => ({
          files: s.files.map((f) =>
            f.id === mediaFileId
              ? {
                  ...f,
                  proxyStatus: 'ready',
                  proxyProgress: 100,
                  proxyFrameCount: result.frameCount,
                  proxyFps: result.fps,
                  hasProxyAudio: hasAudioProxy,
                }
              : f
          ),
        }));

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
      console.log('[Proxy] Cancelled:', mediaFileId);
    }

    const { currentlyGeneratingProxyId } = get();
    if (currentlyGeneratingProxyId === mediaFileId) {
      set((state) => ({
        currentlyGeneratingProxyId: null,
        files: state.files.map((f) =>
          f.id === mediaFileId
            ? { ...f, proxyStatus: 'none', proxyProgress: 0 }
            : f
        ),
      }));
    }
  },
});

async function generateVideoProxy(
  mediaFile: import('../types').MediaFile,
  storageKey: string,
  controller: { cancelled: boolean },
  updateProgress: (id: string, progress: number) => void
): Promise<{ frameCount: number; fps: number } | null> {
  const { getProxyGenerator } = await import('../../../services/proxyGenerator');
  const generator = getProxyGenerator();

  const saveFrame = async (frame: { frameIndex: number; blob: Blob }) => {
    await projectFileService.saveProxyFrame(storageKey, frame.frameIndex, frame.blob);
  };

  return generator.generate(
    mediaFile.file!,
    mediaFile.id,
    (progress) => updateProgress(mediaFile.id, progress),
    () => controller.cancelled,
    saveFrame
  );
}

async function extractAudioProxy(
  mediaFile: import('../types').MediaFile,
  storageKey: string
): Promise<void> {
  try {
    console.log('[Proxy] Extracting audio...');
    const { extractAudioFromVideo } = await import('../../../services/audioExtractor');

    const result = await extractAudioFromVideo(mediaFile.file!, () => {});
    if (result?.blob.size > 0) {
      await projectFileService.saveProxyAudio(storageKey, result.blob);
      console.log(`[Proxy] Audio saved (${(result.blob.size / 1024).toFixed(1)}KB)`);
    }
  } catch (e) {
    console.warn('[Proxy] Audio extraction failed (non-fatal):', e);
  }
}
```

---

## Phase 4: Create Main Index and Init

### Step 4.1: Create index.ts

**File**: `src/stores/mediaStore/index.ts`

```typescript
// MediaStore - main coordinator

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

import type { MediaState, MediaFile, Composition, MediaFolder, ProxyStatus } from './types';
import { DEFAULT_COMPOSITION } from './constants';
import { fileSystemService } from '../../services/fileSystemService';

// Import slices
import { createFileImportSlice } from './slices/fileImportSlice';
import { createFileManageSlice } from './slices/fileManageSlice';
import { createCompositionSlice } from './slices/compositionSlice';
import { createFolderSlice } from './slices/folderSlice';
import { createSelectionSlice } from './slices/selectionSlice';
import { createProxySlice } from './slices/proxySlice';
import { createProjectSlice } from './slices/projectSlice';

// Re-export types
export type { MediaType, ProxyStatus, MediaItem, MediaFile, Composition, MediaFolder, ProjectItem } from './types';

export const useMediaStore = create<MediaState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
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
    proxyEnabled: false,
    proxyGenerationQueue: [],
    currentlyGeneratingProxyId: null,
    fileSystemSupported: fileSystemService.isSupported(),
    proxyFolderName: fileSystemService.getProxyFolderName(),

    // Getters
    getItemsByFolder: (folderId: string | null) => {
      const { files, compositions, folders } = get();
      return [
        ...folders.filter((f) => f.parentId === folderId),
        ...compositions.filter((c) => c.parentId === folderId),
        ...files.filter((f) => f.parentId === folderId),
      ];
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
      return get().files.find((f) => f.name === name);
    },

    // Merge all slices
    ...createFileImportSlice(set, get),
    ...createFileManageSlice(set, get),
    ...createCompositionSlice(set, get),
    ...createFolderSlice(set, get),
    ...createSelectionSlice(set, get),
    ...createProxySlice(set, get),
    ...createProjectSlice(set, get),
  }))
);

// Export trigger for external use
export { triggerTimelineSave } from './init';
```

### Step 4.2: Create init.ts

**File**: `src/stores/mediaStore/init.ts`

```typescript
// MediaStore initialization and auto-save

import { useMediaStore } from './index';
import { useTimelineStore } from '../timeline';
import { fileSystemService } from '../../services/fileSystemService';

/**
 * Save current timeline to active composition.
 */
function saveTimelineToActiveComposition(): void {
  const { activeCompositionId } = useMediaStore.getState();
  if (!activeCompositionId) return;

  const timelineData = useTimelineStore.getState().getSerializableState();
  useMediaStore.setState((state) => ({
    compositions: state.compositions.map((c) =>
      c.id === activeCompositionId ? { ...c, timelineData } : c
    ),
  }));
}

/**
 * Trigger timeline save (exported for external use).
 */
export function triggerTimelineSave(): void {
  saveTimelineToActiveComposition();
  console.log('[MediaStore] Timeline saved to composition');
}

/**
 * Initialize media store from IndexedDB and file handles.
 */
async function initializeStore(): Promise<void> {
  // Initialize file system service
  await fileSystemService.init();

  // Update proxy folder name if restored
  const proxyFolderName = fileSystemService.getProxyFolderName();
  if (proxyFolderName) {
    useMediaStore.setState({ proxyFolderName });
  }

  // Initialize media from IndexedDB
  await useMediaStore.getState().initFromDB();

  // Restore active composition's timeline
  const { activeCompositionId, compositions } = useMediaStore.getState();
  if (activeCompositionId) {
    const activeComp = compositions.find((c) => c.id === activeCompositionId);
    if (activeComp?.timelineData) {
      console.log('[MediaStore] Restoring timeline for:', activeComp.name);
      await useTimelineStore.getState().loadState(activeComp.timelineData);
    }
  }
}

/**
 * Set up auto-save interval.
 */
function setupAutoSave(): void {
  setInterval(() => {
    if ((window as any).__CLEARING_CACHE__) return;
    saveTimelineToActiveComposition();
  }, 30000); // Every 30 seconds
}

/**
 * Set up beforeunload handler.
 */
function setupBeforeUnload(): void {
  window.addEventListener('beforeunload', () => {
    if ((window as any).__CLEARING_CACHE__) return;
    saveTimelineToActiveComposition();
  });
}

// Auto-initialize on app load
if (typeof window !== 'undefined') {
  setTimeout(() => {
    initializeStore();
    setupAutoSave();
    setupBeforeUnload();
  }, 100);
}
```

---

## Phase 5: Update Imports

After refactoring, update imports in files that use mediaStore:

```typescript
// Before
import { useMediaStore, MediaFile, Composition } from '../stores/mediaStore';

// After (same - re-exported from index.ts)
import { useMediaStore, MediaFile, Composition } from '../stores/mediaStore';
```

---

## Validation Commands

After each phase:

```bash
# TypeScript check
npx tsc --noEmit

# Build test
npm run build

# Lint check
npm run lint
```

---

## Rollback

If issues occur:
```bash
git checkout src/stores/mediaStore.ts
rm -rf src/stores/mediaStore/
```

---

## Notes for AI Agent

1. **Order matters**: Create types/constants first, then helpers, then slices, then index/init

2. **Shared helpers**: Use `detectMediaType` from `../timeline/helpers/mediaTypeHelpers.ts` instead of duplicating

3. **Import pipeline**: The `processImport` function unifies all 3 import methods - this is the key deduplication

4. **RAW folder simplification**: All files are copied to `/RAW/` on import, so relinking just reads from there:
   - No `handleHelpers.ts` needed
   - `projectFileService.getFileFromRaw(path)` is all you need
   - One permission for the project folder covers all files

5. **Proxy slice**: Keep `activeProxyGenerations` Map inside the slice file, not at module level of main store

6. **Init separation**: Side effects (auto-init, intervals, event listeners) are isolated in init.ts

7. **Type safety**: Ensure `MediaSliceCreator` type is used for all slices for consistent typing

8. **projectFileService.getFileFromRaw()**: This method needs to exist in projectFileService.ts. If it doesn't, add:
   ```typescript
   async getFileFromRaw(relativePath: string): Promise<File | null> {
     if (!this.projectHandle) return null;
     try {
       const rawFolder = await this.projectHandle.getDirectoryHandle('Raw');
       const fileHandle = await rawFolder.getFileHandle(relativePath.replace('Raw/', ''));
       return await fileHandle.getFile();
     } catch (e) {
       console.warn('[ProjectFileService] Failed to get file from Raw:', e);
       return null;
     }
   }
   ```

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Main file LOC | 2046 | ~300 |
| Largest slice | - | ~220 (proxySlice) |
| Duplicate code | ~600 LOC | ~0 |
| Relinking code | ~275 LOC | ~75 LOC |
| Files | 1 | 13 (no handleHelpers) |
| Shared with clipSlice | 0 | 1 helper |

**Total LOC reduction: ~900 LOC** (including RAW folder simplification)
