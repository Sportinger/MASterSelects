// Structural Sharing types — efficient undo/redo snapshots

import type { TimelineClip, TimelineTrack, Keyframe } from '../../types/index.ts';
import type { TimelineMarker } from '../../stores/timeline/types.ts';

/**
 * A serialized clip — TimelineClip without DOM references.
 * DOM refs are stored in the DomRefRegistry and re-linked on restore.
 */
export interface SerializedClipState {
  id: string;
  trackId: string;
  name: string;
  startTime: number;
  duration: number;
  inPoint: number;
  outPoint: number;
  sourceType: 'video' | 'audio' | 'image' | 'text' | 'solid';
  mediaFileId?: string;
  transform: TimelineClip['transform'];
  effects: TimelineClip['effects'];
  speed?: number;
  reversed?: boolean;
  isComposition?: boolean;
  compositionId?: string;
  masks?: TimelineClip['masks'];
  textProperties?: TimelineClip['textProperties'];
  solidColor?: string;
  transitionIn?: TimelineClip['transitionIn'];
  transitionOut?: TimelineClip['transitionOut'];
}

/**
 * V2 snapshot — uses structural sharing.
 * Unchanged clips are shared references to the previous snapshot's array entries.
 */
export interface HistorySnapshotV2 {
  timestamp: number;
  label: string;

  /** Clip states — shared refs for unchanged clips, new objects for changed ones */
  clips: SerializedClipState[];
  tracks: TimelineTrack[];
  clipKeyframes: Record<string, Keyframe[]>;
  markers: TimelineMarker[];

  /** IDs of clips that were actually cloned (rest are shared references) */
  changedClipIds: string[];
}

/**
 * Interface for the DOM reference registry.
 * Centralizes ownership of HTML media elements.
 * DOM elements live here, not on serialized clips.
 */
export interface DomRefRegistryInterface {
  getVideoElement(mediaFileId: string): HTMLVideoElement | undefined;
  getAudioElement(mediaFileId: string): HTMLAudioElement | undefined;
  getImageElement(mediaFileId: string): HTMLImageElement | undefined;
  getTextCanvas(clipId: string): HTMLCanvasElement | undefined;

  registerVideoElement(mediaFileId: string, element: HTMLVideoElement): void;
  registerAudioElement(mediaFileId: string, element: HTMLAudioElement): void;
  registerImageElement(mediaFileId: string, element: HTMLImageElement): void;
  registerTextCanvas(clipId: string, canvas: HTMLCanvasElement): void;

  unregister(mediaFileId: string): void;
  unregisterTextCanvas(clipId: string): void;
}
