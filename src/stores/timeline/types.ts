// Timeline store types and interfaces

import type {
  TimelineClip,
  TimelineTrack,
  ClipTransform,
  CompositionTimelineData,
  Keyframe,
  AnimatableProperty,
  EasingType,
  BezierHandle,
  ClipMask,
  MaskVertex,
  Effect,
  TextClipProperties
} from '../../types';
import type { Composition } from '../mediaStore';

// Re-export imported types for convenience
export type {
  TimelineClip,
  TimelineTrack,
  ClipTransform,
  CompositionTimelineData,
  Keyframe,
  AnimatableProperty,
  EasingType,
  BezierHandle,
  ClipMask,
  MaskVertex,
  Effect,
  Composition,
  TextClipProperties,
};

// Mask edit mode types
export type MaskEditMode = 'none' | 'drawing' | 'editing' | 'drawingRect' | 'drawingEllipse' | 'drawingPen';

// Timeline tool mode types
export type TimelineToolMode = 'select' | 'cut';

// Timeline state interface
export interface TimelineState {
  // Core state
  tracks: TimelineTrack[];
  clips: TimelineClip[];
  playheadPosition: number;
  duration: number;
  zoom: number;
  scrollX: number;
  snappingEnabled: boolean;
  isPlaying: boolean;
  isDraggingPlayhead: boolean;
  selectedClipIds: Set<string>;

  // In/Out markers
  inPoint: number | null;
  outPoint: number | null;
  loopPlayback: boolean;

  // Duration lock (when true, duration won't auto-update based on clips)
  durationLocked: boolean;

  // RAM Preview state
  ramPreviewEnabled: boolean;
  ramPreviewProgress: number | null;
  ramPreviewRange: { start: number; end: number } | null;
  isRamPreviewing: boolean;
  cachedFrameTimes: Set<number>;

  // Export progress state
  isExporting: boolean;
  exportProgress: number | null;  // 0-100 percentage
  exportCurrentTime: number | null;  // Current time being rendered
  exportRange: { start: number; end: number } | null;

  // Performance toggles
  thumbnailsEnabled: boolean;
  waveformsEnabled: boolean;

  // Keyframe animation state
  clipKeyframes: Map<string, Keyframe[]>;
  keyframeRecordingEnabled: Set<string>;
  expandedTracks: Set<string>;
  expandedTrackPropertyGroups: Map<string, Set<string>>;
  selectedKeyframeIds: Set<string>;
  expandedCurveProperties: Map<string, Set<AnimatableProperty>>;  // trackId -> expanded curve editors

  // Mask state
  maskEditMode: MaskEditMode;
  activeMaskId: string | null;
  selectedVertexIds: Set<string>;
  maskDrawStart: { x: number; y: number } | null;
  maskDragging: boolean; // True during vertex/mask drag - skips texture regeneration

  // Tool mode
  toolMode: TimelineToolMode;
}

// Track actions interface
export interface TrackActions {
  addTrack: (type: 'video' | 'audio') => string;
  removeTrack: (id: string) => void;
  renameTrack: (id: string, name: string) => void;
  setTrackMuted: (id: string, muted: boolean) => void;
  setTrackVisible: (id: string, visible: boolean) => void;
  setTrackSolo: (id: string, solo: boolean) => void;
  setTrackHeight: (id: string, height: number) => void;
  scaleTracksOfType: (type: 'video' | 'audio', delta: number) => void;
  // Track parenting (layer linking)
  setTrackParent: (trackId: string, parentTrackId: string | null) => void;
  getTrackChildren: (trackId: string) => TimelineTrack[];
}

// Clip actions interface
export interface ClipActions {
  addClip: (trackId: string, file: File, startTime: number, estimatedDuration?: number, mediaFileId?: string) => Promise<void>;
  addCompClip: (trackId: string, composition: Composition, startTime: number) => void;
  updateClip: (id: string, updates: Partial<TimelineClip>) => void;
  removeClip: (id: string) => void;
  moveClip: (id: string, newStartTime: number, newTrackId?: string, skipLinked?: boolean, skipGroup?: boolean) => void;
  trimClip: (id: string, inPoint: number, outPoint: number) => void;
  splitClip: (clipId: string, splitTime: number) => void;
  splitClipAtPlayhead: () => void;
  updateClipTransform: (id: string, transform: Partial<ClipTransform>) => void;
  toggleClipReverse: (id: string) => void;
  addClipEffect: (clipId: string, effectType: string) => void;
  removeClipEffect: (clipId: string, effectId: string) => void;
  updateClipEffect: (clipId: string, effectId: string, params: Partial<Effect['params']>) => void;
  setClipEffectEnabled: (clipId: string, effectId: string, enabled: boolean) => void;
  // Multicam group linking
  createLinkedGroup: (clipIds: string[], offsets: Map<string, number>) => void;
  unlinkGroup: (clipId: string) => void;
  // Waveform generation
  generateWaveformForClip: (clipId: string) => Promise<void>;
  // Parenting (pick whip)
  setClipParent: (clipId: string, parentClipId: string | null) => void;
  getClipChildren: (clipId: string) => TimelineClip[];
  // Audio pitch preservation
  setClipPreservesPitch: (clipId: string, preservesPitch: boolean) => void;
  // Text clip actions
  addTextClip: (trackId: string, startTime: number, duration?: number) => Promise<string | null>;
  updateTextProperties: (clipId: string, props: Partial<TextClipProperties>) => void;
  // YouTube pending download clips
  addPendingDownloadClip: (trackId: string, startTime: number, videoId: string, title: string, thumbnail: string, estimatedDuration?: number) => string;
  updateDownloadProgress: (clipId: string, progress: number) => void;
  completeDownload: (clipId: string, file: File) => Promise<void>;
  setDownloadError: (clipId: string, error: string) => void;
}

// Playback actions interface
export interface PlaybackActions {
  setPlayheadPosition: (position: number) => void;
  setDraggingPlayhead: (dragging: boolean) => void;
  play: () => Promise<void>;
  pause: () => void;
  stop: () => void;
  setZoom: (zoom: number) => void;
  toggleSnapping: () => void;
  setScrollX: (scrollX: number) => void;
  setInPoint: (time: number | null) => void;
  setOutPoint: (time: number | null) => void;
  clearInOut: () => void;
  setInPointAtPlayhead: () => void;
  setOutPointAtPlayhead: () => void;
  setLoopPlayback: (loop: boolean) => void;
  toggleLoopPlayback: () => void;
  setDuration: (duration: number) => void;
  // Tool mode
  setToolMode: (mode: TimelineToolMode) => void;
  toggleCutTool: () => void;
}

// RAM Preview actions interface
export interface RamPreviewActions {
  toggleRamPreviewEnabled: () => void;
  startRamPreview: () => Promise<void>;
  cancelRamPreview: () => void;
  clearRamPreview: () => void;
  addCachedFrame: (time: number) => void;
  getCachedRanges: () => Array<{ start: number; end: number }>;
  invalidateCache: () => void;
  // Performance toggles
  toggleThumbnailsEnabled: () => void;
  toggleWaveformsEnabled: () => void;
}

// Export progress actions interface
export interface ExportActions {
  setExportProgress: (progress: number | null, currentTime: number | null) => void;
  startExport: (start: number, end: number) => void;
  endExport: () => void;
}

// Selection actions interface
export interface SelectionActions {
  // Clip selection (multi-select support)
  selectClip: (id: string | null, addToSelection?: boolean) => void;
  selectClips: (ids: string[]) => void;
  addClipToSelection: (id: string) => void;
  removeClipFromSelection: (id: string) => void;
  clearClipSelection: () => void;
  // Keyframe selection
  selectKeyframe: (keyframeId: string, addToSelection?: boolean) => void;
  deselectAllKeyframes: () => void;
  deleteSelectedKeyframes: () => void;
}

// Keyframe actions interface
export interface KeyframeActions {
  addKeyframe: (clipId: string, property: AnimatableProperty, value: number, time?: number, easing?: EasingType) => void;
  removeKeyframe: (keyframeId: string) => void;
  updateKeyframe: (keyframeId: string, updates: Partial<Omit<Keyframe, 'id' | 'clipId'>>) => void;
  moveKeyframe: (keyframeId: string, newTime: number) => void;
  getClipKeyframes: (clipId: string) => Keyframe[];
  getInterpolatedTransform: (clipId: string, clipLocalTime: number) => ClipTransform;
  getInterpolatedEffects: (clipId: string, clipLocalTime: number) => Effect[];
  getInterpolatedSpeed: (clipId: string, clipLocalTime: number) => number;
  getSourceTimeForClip: (clipId: string, clipLocalTime: number) => number;
  hasKeyframes: (clipId: string, property?: AnimatableProperty) => boolean;
  toggleKeyframeRecording: (clipId: string, property: AnimatableProperty) => void;
  isRecording: (clipId: string, property: AnimatableProperty) => boolean;
  setPropertyValue: (clipId: string, property: AnimatableProperty, value: number) => void;
  toggleTrackExpanded: (trackId: string) => void;
  isTrackExpanded: (trackId: string) => boolean;
  toggleTrackPropertyGroupExpanded: (trackId: string, groupName: string) => void;
  isTrackPropertyGroupExpanded: (trackId: string, groupName: string) => boolean;
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number;
  trackHasKeyframes: (trackId: string) => boolean;
  // Curve editor expansion
  toggleCurveExpanded: (trackId: string, property: AnimatableProperty) => void;
  isCurveExpanded: (trackId: string, property: AnimatableProperty) => boolean;
  // Bezier handle manipulation
  updateBezierHandle: (keyframeId: string, handle: 'in' | 'out', position: BezierHandle) => void;
}

// Mask actions interface
export interface MaskActions {
  setMaskEditMode: (mode: MaskEditMode) => void;
  setMaskDragging: (dragging: boolean) => void;
  setMaskDrawStart: (point: { x: number; y: number } | null) => void;
  setActiveMask: (clipId: string | null, maskId: string | null) => void;
  selectVertex: (vertexId: string, addToSelection?: boolean) => void;
  deselectAllVertices: () => void;
  addMask: (clipId: string, mask?: Partial<ClipMask>) => string;
  removeMask: (clipId: string, maskId: string) => void;
  updateMask: (clipId: string, maskId: string, updates: Partial<ClipMask>) => void;
  reorderMasks: (clipId: string, fromIndex: number, toIndex: number) => void;
  getClipMasks: (clipId: string) => ClipMask[];
  addVertex: (clipId: string, maskId: string, vertex: Omit<MaskVertex, 'id'>, index?: number) => string;
  removeVertex: (clipId: string, maskId: string, vertexId: string) => void;
  updateVertex: (clipId: string, maskId: string, vertexId: string, updates: Partial<MaskVertex>, skipCacheInvalidation?: boolean) => void;
  closeMask: (clipId: string, maskId: string) => void;
  addRectangleMask: (clipId: string) => string;
  addEllipseMask: (clipId: string) => string;
}

// Utils interface
export interface TimelineUtils {
  getClipsAtTime: (time: number) => TimelineClip[];
  updateDuration: () => void;
  findAvailableAudioTrack: (startTime: number, duration: number) => string;
  getSnappedPosition: (clipId: string, desiredStartTime: number, trackId: string) => { startTime: number; snapped: boolean };
  findNonOverlappingPosition: (clipId: string, desiredStartTime: number, trackId: string, duration: number) => number;
  // Get position with magnetic resistance at clip edges - returns adjusted position and whether user has "broken through"
  // Uses pixel-based resistance (zoom converts time distance to pixels)
  getPositionWithResistance: (clipId: string, desiredStartTime: number, trackId: string, duration: number, zoom?: number) => { startTime: number; forcingOverlap: boolean };
  // Trim any clips that the placed clip overlaps with
  trimOverlappingClips: (clipId: string, startTime: number, trackId: string, duration: number) => void;
  getSerializableState: () => CompositionTimelineData;
  loadState: (data: CompositionTimelineData | undefined) => Promise<void>;
  clearTimeline: () => void;
}

// Combined store interface
export interface TimelineStore extends
  TimelineState,
  TrackActions,
  ClipActions,
  PlaybackActions,
  RamPreviewActions,
  ExportActions,
  SelectionActions,
  KeyframeActions,
  MaskActions,
  TimelineUtils {}

// Slice creator type
export type SliceCreator<T> = (
  set: (partial: Partial<TimelineStore> | ((state: TimelineStore) => Partial<TimelineStore>)) => void,
  get: () => TimelineStore
) => T;
