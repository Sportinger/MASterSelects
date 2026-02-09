// Composition-related types

import type { ProjectKeyframe, ProjectMarker, ProjectEffect, ProjectMask, ProjectTransform } from './timeline.types';

export interface ProjectTrack {
  id: string;
  name: string;
  type: 'video' | 'audio';
  height: number;
  locked: boolean;
  visible: boolean;
  muted: boolean;
  solo: boolean;
}

export interface ProjectClip {
  id: string;
  trackId: string;
  name?: string;
  mediaId: string; // Reference to ProjectMediaFile.id (empty for composition clips)

  // Timeline position
  startTime: number;
  duration: number;

  // Source trimming
  inPoint: number;
  outPoint: number;

  // Transform
  transform: ProjectTransform;

  // Effects
  effects: ProjectEffect[];

  // Masks
  masks: ProjectMask[];

  // Keyframes
  keyframes: ProjectKeyframe[];

  // Audio
  volume: number;
  audioEnabled: boolean;

  // Flags
  reversed: boolean;
  disabled: boolean;

  // Nested composition support
  isComposition?: boolean;
  compositionId?: string;

  // Additional clip metadata (for restoration)
  sourceType?: 'video' | 'audio' | 'image' | 'text' | 'solid';
  naturalDuration?: number;
  linkedClipId?: string;
  linkedGroupId?: string;
  thumbnails?: string[];
  waveform?: number[];

  // Text clip support
  textProperties?: any;

  // Solid clip support
  solidColor?: string;

  // Transcript data
  transcript?: any[];
  transcriptStatus?: string;

  // Analysis data
  analysis?: any;
  analysisStatus?: string;
}

export interface ProjectComposition {
  id: string;
  name: string;
  width: number;
  height: number;
  frameRate: number;
  duration: number;
  backgroundColor: string;
  folderId: string | null;

  // Tracks and clips
  tracks: ProjectTrack[];
  clips: ProjectClip[];

  // Markers
  markers: ProjectMarker[];
}
