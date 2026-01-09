// Timeline-specific types for component props

import type { TimelineClip, TimelineTrack, AnimatableProperty } from '../../types';

// Clip drag state (Premiere-style)
export interface ClipDragState {
  clipId: string;
  originalStartTime: number;
  originalTrackId: string;
  grabOffsetX: number;      // Where on the clip we grabbed (in pixels)
  currentX: number;         // Current mouse X position
  currentTrackId: string;
  snappedTime: number | null;  // Snapped position (if snapping)
  isSnapping: boolean;         // Whether currently snapping
  altKeyPressed: boolean;      // If true, skip linked group movement (independent drag)
  forcingOverlap: boolean;     // If true, user has pushed through resistance and is forcing overlap
}

// Clip trim state
export interface ClipTrimState {
  clipId: string;
  edge: 'left' | 'right';
  originalStartTime: number;
  originalDuration: number;
  originalInPoint: number;
  originalOutPoint: number;
  startX: number;
  currentX: number;
  altKey: boolean;  // If true, don't trim linked clip
}

// In/Out marker drag state
export interface MarkerDragState {
  type: 'in' | 'out';
  startX: number;
  originalTime: number;
}

// External file drag preview state
export interface ExternalDragState {
  trackId: string;
  startTime: number;
  x: number;
  y: number;
  audioTrackId?: string;  // Preview for linked audio clip
  isVideo?: boolean;      // Is the dragged file a video?
  isAudio?: boolean;      // Is the dragged file audio-only?
  duration?: number;      // Actual duration of dragged file
  newTrackType?: 'video' | 'audio' | null;  // If hovering over "new track" drop zone
}

// Context menu state for clip right-click
export interface ContextMenuState {
  x: number;
  y: number;
  clipId: string;
}

// Marquee selection state for rectangle selection
export interface MarqueeState {
  startX: number;      // Start X position relative to track-lanes
  startY: number;      // Start Y position relative to track-lanes
  currentX: number;    // Current X position
  currentY: number;    // Current Y position
  startScrollX: number; // ScrollX at the time of starting selection
  initialSelection: Set<string>; // Clips that were selected before marquee started (for shift+drag)
}

// Props for TimelineRuler component
export interface TimelineRulerProps {
  duration: number;
  zoom: number;
  scrollX: number;
  onRulerMouseDown: (e: React.MouseEvent) => void;
  formatTime: (seconds: number) => string;
}

// Props for TimelineControls component
export interface TimelineControlsProps {
  isPlaying: boolean;
  loopPlayback: boolean;
  playheadPosition: number;
  duration: number;
  zoom: number;
  inPoint: number | null;
  outPoint: number | null;
  ramPreviewEnabled: boolean;
  proxyEnabled: boolean;
  currentlyGeneratingProxyId: string | null;
  mediaFilesWithProxy: number;
  showTranscriptMarkers: boolean;
  thumbnailsEnabled: boolean;
  waveformsEnabled: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onToggleLoop: () => void;
  onSetZoom: (zoom: number) => void;
  onSetInPoint: () => void;
  onSetOutPoint: () => void;
  onClearInOut: () => void;
  onToggleRamPreview: () => void;
  onToggleProxy: () => void;
  onToggleTranscriptMarkers: () => void;
  onToggleThumbnails: () => void;
  onToggleWaveforms: () => void;
  onAddVideoTrack: () => void;
  onAddAudioTrack: () => void;
  onSetDuration: (duration: number) => void;
  onFitToWindow: () => void;
  formatTime: (seconds: number) => string;
  parseTime: (timeStr: string) => number | null;
}

// Props for TimelineHeader component
export interface TimelineHeaderProps {
  track: TimelineTrack;
  isDimmed: boolean;
  isExpanded: boolean;
  dynamicHeight: number;
  hasKeyframes: boolean;
  selectedClipIds: Set<string>;
  clips: TimelineClip[];
  onToggleExpand: () => void;
  onToggleSolo: () => void;
  onToggleMuted: () => void;
  onToggleVisible: () => void;
  onWheel: (e: React.WheelEvent) => void;
  // For property labels
  isTrackPropertyGroupExpanded: (trackId: string, group: string) => boolean;
  toggleTrackPropertyGroupExpanded: (trackId: string, group: string) => void;
  getClipKeyframes: (clipId: string) => Array<{
    id: string;
    clipId: string;
    time: number;
    property: AnimatableProperty;
    value: number;
    easing: string;
  }>;
}

// Props for TimelineTrack component
export interface TimelineTrackProps {
  track: TimelineTrack;
  clips: TimelineClip[];
  isDimmed: boolean;
  isExpanded: boolean;
  dynamicHeight: number;
  isDragTarget: boolean;
  isExternalDragTarget: boolean;
  selectedClipIds: Set<string>;
  clipDrag: ClipDragState | null;
  clipTrim: ClipTrimState | null;
  externalDrag: ExternalDragState | null;
  zoom: number;
  scrollX: number;
  timelineRef: React.RefObject<HTMLDivElement | null>;
  onClipMouseDown: (e: React.MouseEvent, clipId: string) => void;
  onClipContextMenu: (e: React.MouseEvent, clipId: string) => void;
  onTrimStart: (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => void;
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragEnter: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  renderClip: (clip: TimelineClip, trackId: string) => React.ReactNode;
  // For keyframe tracks
  isTrackPropertyGroupExpanded: (trackId: string, group: string) => boolean;
  getClipKeyframes: (clipId: string) => Array<{
    id: string;
    clipId: string;
    time: number;
    property: AnimatableProperty;
    value: number;
    easing: string;
  }>;
  renderKeyframeDiamonds: (trackId: string, property: AnimatableProperty) => React.ReactNode;
  timeToPixel: (time: number) => number;
  pixelToTime: (pixel: number) => number;
}

// Props for TimelineClip component
export interface TimelineClipProps {
  clip: TimelineClip;
  trackId: string;
  track: TimelineTrack;
  tracks: TimelineTrack[];
  clips: TimelineClip[];
  isSelected: boolean;
  isInLinkedGroup: boolean;  // True if clip has linkedGroupId (multicam)
  isDragging: boolean;
  isTrimming: boolean;
  isLinkedToDragging: boolean;
  isLinkedToTrimming: boolean;
  clipDrag: ClipDragState | null;
  clipTrim: ClipTrimState | null;
  zoom: number;
  scrollX: number;
  timelineRef: React.RefObject<HTMLDivElement | null>;
  proxyEnabled: boolean;
  proxyStatus: 'none' | 'generating' | 'ready' | 'error' | undefined;
  proxyProgress: number;
  showTranscriptMarkers: boolean;
  onMouseDown: (e: React.MouseEvent) => void;
  onContextMenu: (e: React.MouseEvent) => void;
  onTrimStart: (e: React.MouseEvent, edge: 'left' | 'right') => void;
  hasKeyframes: (clipId: string, property?: AnimatableProperty) => boolean;
  timeToPixel: (time: number) => number;
  pixelToTime: (pixel: number) => number;
  formatTime: (seconds: number) => string;
}

// Props for TimelineKeyframes component
export interface TimelineKeyframesProps {
  trackId: string;
  property: AnimatableProperty;
  clips: TimelineClip[];
  selectedKeyframeIds: Set<string>;
  getClipKeyframes: (clipId: string) => Array<{
    id: string;
    clipId: string;
    time: number;
    property: AnimatableProperty;
    value: number;
    easing: string;
  }>;
  onSelectKeyframe: (keyframeId: string, addToSelection: boolean) => void;
  timeToPixel: (time: number) => number;
}

// Waveform props
export interface WaveformProps {
  waveform: number[];
  width: number;
  height: number;
}
