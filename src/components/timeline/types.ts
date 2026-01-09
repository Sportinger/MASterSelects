// Timeline-specific types for component props

import type { TimelineClip, TimelineTrack, AnimatableProperty, ClipTransform, BezierHandle, EasingType } from '../../types';

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
  dragStartTime: number;       // Timestamp when drag started (for track-change delay)
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
  initialKeyframeSelection: Set<string>; // Keyframes that were selected before marquee started
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
  playheadPosition: number;
  onToggleExpand: () => void;
  onToggleSolo: () => void;
  onToggleMuted: () => void;
  onToggleVisible: () => void;
  onRenameTrack: (name: string) => void;
  onWheel: (e: React.WheelEvent) => void;
  // For property labels - clipKeyframes map triggers re-render when keyframes change
  clipKeyframes: Map<string, Array<{ id: string; clipId: string; time: number; property: AnimatableProperty; value: number; easing: string }>>;
  getClipKeyframes: (clipId: string) => Array<{
    id: string;
    clipId: string;
    time: number;
    property: AnimatableProperty;
    value: number;
    easing: string;
  }>;
  // Keyframe controls
  getInterpolatedTransform: (clipId: string, clipLocalTime: number) => ClipTransform;
  addKeyframe: (clipId: string, property: AnimatableProperty, value: number) => void;
  setPlayheadPosition: (time: number) => void;
  setPropertyValue: (clipId: string, property: AnimatableProperty, value: number) => void;
  // Curve editor
  expandedCurveProperties: Map<string, Set<AnimatableProperty>>;
  onToggleCurveExpanded: (trackId: string, property: AnimatableProperty) => void;
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
  selectedKeyframeIds: Set<string>;
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
  // For keyframe tracks - clipKeyframes map triggers re-render when keyframes change
  clipKeyframes: Map<string, Array<{ id: string; clipId: string; time: number; property: AnimatableProperty; value: number; easing: string }>>;
  renderKeyframeDiamonds: (trackId: string, property: AnimatableProperty) => React.ReactNode;
  timeToPixel: (time: number) => number;
  pixelToTime: (pixel: number) => number;
  // Curve editor
  expandedCurveProperties: Map<string, Set<AnimatableProperty>>;
  onSelectKeyframe: (keyframeId: string, addToSelection: boolean) => void;
  onMoveKeyframe: (keyframeId: string, newTime: number) => void;
  onUpdateBezierHandle: (keyframeId: string, handle: 'in' | 'out', position: BezierHandle) => void;
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
  clipKeyframes: Map<string, Array<{
    id: string;
    clipId: string;
    time: number;
    property: AnimatableProperty;
    value: number;
    easing: string;
  }>>;
  clipDrag: ClipDragState | null;
  scrollX: number;
  timelineRef: React.RefObject<HTMLDivElement | null>;
  onSelectKeyframe: (keyframeId: string, addToSelection: boolean) => void;
  onMoveKeyframe: (keyframeId: string, newTime: number) => void;
  onUpdateKeyframe: (keyframeId: string, updates: { easing?: EasingType }) => void;
  timeToPixel: (time: number) => number;
  pixelToTime: (pixel: number) => number;
}

// Waveform props
export interface WaveformProps {
  waveform: number[];
  width: number;
  height: number;
}
