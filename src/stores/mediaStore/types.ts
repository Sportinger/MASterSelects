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
  audioCodec?: string;
  container?: string;
  fileSize?: number;
  bitrate?: number;      // bits per second
  hasAudio?: boolean;    // Does video have audio tracks?
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
