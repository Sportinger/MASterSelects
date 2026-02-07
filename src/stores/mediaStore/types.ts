// MediaStore types - extracted from mediaStore.ts

import type { CompositionTimelineData } from '../../types';

// Media item types
export type MediaType = 'video' | 'audio' | 'image' | 'composition' | 'text' | 'solid';

// Proxy status for video files
export type ProxyStatus = 'none' | 'generating' | 'ready' | 'error';

// Label colors (AE-style)
export type LabelColor = 'none' | 'red' | 'yellow' | 'blue' | 'green' | 'purple' | 'orange' | 'pink' | 'cyan' | 'brown' | 'lavender' | 'peach' | 'seafoam' | 'fuchsia' | 'tan' | 'aqua';

// Base media item
export interface MediaItem {
  id: string;
  name: string;
  type: MediaType;
  parentId: string | null;
  createdAt: number;
  labelColor?: LabelColor;
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
  proxyVideoUrl?: string;
  // File System Access API
  hasFileHandle?: boolean;
  filePath?: string;
  absolutePath?: string;
  projectPath?: string;
  // Import loading state
  isImporting?: boolean;
}

// Text item (for Media Panel - can be dragged to timeline)
export interface TextItem extends MediaItem {
  type: 'text';
  text: string;
  fontFamily: string;
  fontSize: number;
  color: string;
  duration: number; // Default duration when added to timeline
}

// Solid color item (for Media Panel - can be dragged to timeline)
export interface SolidItem extends MediaItem {
  type: 'solid';
  color: string;
  width: number;
  height: number;
  duration: number; // Default duration when added to timeline
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
  labelColor?: LabelColor;
}

// Union type for all items
export type ProjectItem = MediaFile | Composition | MediaFolder | TextItem | SolidItem;

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
  textItems: TextItem[];
  solidItems: SolidItem[];

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
