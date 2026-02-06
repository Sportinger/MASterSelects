// Timeline store selectors - Optimized for minimal re-renders
// Using individual selectors allows Zustand to only trigger re-renders
// when the specific selected value changes, not when ANY store value changes.

import type { TimelineStore } from './types';

// ===========================================
// CORE DATA SELECTORS (frequently changing)
// ===========================================

export const selectTracks = (state: TimelineStore) => state.tracks;
export const selectClips = (state: TimelineStore) => state.clips;
export const selectPlayheadPosition = (state: TimelineStore) => state.playheadPosition;
export const selectDuration = (state: TimelineStore) => state.duration;
export const selectZoom = (state: TimelineStore) => state.zoom;
export const selectScrollX = (state: TimelineStore) => state.scrollX;
export const selectIsPlaying = (state: TimelineStore) => state.isPlaying;
export const selectSelectedClipIds = (state: TimelineStore) => state.selectedClipIds;
export const selectMarkers = (state: TimelineStore) => state.markers;

// ===========================================
// UI STATE SELECTORS (less frequent changes)
// ===========================================

export const selectSnappingEnabled = (state: TimelineStore) => state.snappingEnabled;
export const selectInPoint = (state: TimelineStore) => state.inPoint;
export const selectOutPoint = (state: TimelineStore) => state.outPoint;
export const selectLoopPlayback = (state: TimelineStore) => state.loopPlayback;
export const selectToolMode = (state: TimelineStore) => state.toolMode;
export const selectThumbnailsEnabled = (state: TimelineStore) => state.thumbnailsEnabled;
export const selectWaveformsEnabled = (state: TimelineStore) => state.waveformsEnabled;
export const selectIsDraggingPlayhead = (state: TimelineStore) => state.isDraggingPlayhead;

// ===========================================
// PREVIEW/EXPORT STATE SELECTORS
// ===========================================

export const selectRamPreviewEnabled = (state: TimelineStore) => state.ramPreviewEnabled;
export const selectRamPreviewProgress = (state: TimelineStore) => state.ramPreviewProgress;
export const selectRamPreviewRange = (state: TimelineStore) => state.ramPreviewRange;
export const selectIsRamPreviewing = (state: TimelineStore) => state.isRamPreviewing;
export const selectIsExporting = (state: TimelineStore) => state.isExporting;
export const selectExportProgress = (state: TimelineStore) => state.exportProgress;
export const selectExportRange = (state: TimelineStore) => state.exportRange;
export const selectIsProxyCaching = (state: TimelineStore) => state.isProxyCaching;
export const selectProxyCacheProgress = (state: TimelineStore) => state.proxyCacheProgress;

// ===========================================
// KEYFRAME STATE SELECTORS
// ===========================================

export const selectSelectedKeyframeIds = (state: TimelineStore) => state.selectedKeyframeIds;
export const selectClipKeyframes = (state: TimelineStore) => state.clipKeyframes;
export const selectExpandedCurveProperties = (state: TimelineStore) => state.expandedCurveProperties;

// ===========================================
// GROUPED STATE SELECTORS (for useShallow)
// Reduces 29 individual subscriptions to 6 grouped ones.
// Use with useShallow() from 'zustand/react/shallow'.
// ===========================================

// Core timeline structure (changes on edits)
export const selectCoreData = (state: TimelineStore) => ({
  tracks: state.tracks,
  clips: state.clips,
  duration: state.duration,
  selectedClipIds: state.selectedClipIds,
  markers: state.markers,
});

// Playback state (changes every frame during playback)
export const selectPlaybackState = (state: TimelineStore) => ({
  playheadPosition: state.playheadPosition,
  isPlaying: state.isPlaying,
  isDraggingPlayhead: state.isDraggingPlayhead,
});

// View state (changes on zoom/scroll)
export const selectViewState = (state: TimelineStore) => ({
  zoom: state.zoom,
  scrollX: state.scrollX,
});

// UI settings (rarely changes)
export const selectUISettings = (state: TimelineStore) => ({
  snappingEnabled: state.snappingEnabled,
  inPoint: state.inPoint,
  outPoint: state.outPoint,
  loopPlayback: state.loopPlayback,
  toolMode: state.toolMode,
  thumbnailsEnabled: state.thumbnailsEnabled,
  waveformsEnabled: state.waveformsEnabled,
});

// Preview/export state (changes during preview/export operations)
export const selectPreviewExportState = (state: TimelineStore) => ({
  ramPreviewEnabled: state.ramPreviewEnabled,
  ramPreviewProgress: state.ramPreviewProgress,
  ramPreviewRange: state.ramPreviewRange,
  isRamPreviewing: state.isRamPreviewing,
  isExporting: state.isExporting,
  exportProgress: state.exportProgress,
  exportRange: state.exportRange,
  isProxyCaching: state.isProxyCaching,
  proxyCacheProgress: state.proxyCacheProgress,
});

// Keyframe state (changes during keyframe edits)
export const selectKeyframeState = (state: TimelineStore) => ({
  selectedKeyframeIds: state.selectedKeyframeIds,
  clipKeyframes: state.clipKeyframes,
  expandedCurveProperties: state.expandedCurveProperties,
});

// ===========================================
// DERIVED SELECTORS (computed from state)
// ===========================================

export const selectVideoTracks = (state: TimelineStore) =>
  state.tracks.filter(t => t.type === 'video');

export const selectAudioTracks = (state: TimelineStore) =>
  state.tracks.filter(t => t.type === 'audio');

export const selectHasAnyVideoSolo = (state: TimelineStore) =>
  state.tracks.some(t => t.type === 'video' && t.solo);

export const selectHasAnyAudioSolo = (state: TimelineStore) =>
  state.tracks.some(t => t.type === 'audio' && t.solo);

// ===========================================
// STABLE ACTION SELECTORS
// Actions are stable references and don't cause re-renders.
// We group them for convenience while maintaining type safety.
// ===========================================

// Playback actions
export const selectPlaybackActions = (state: TimelineStore) => ({
  play: state.play,
  pause: state.pause,
  stop: state.stop,
  playForward: state.playForward,
  playReverse: state.playReverse,
  setPlayheadPosition: state.setPlayheadPosition,
  setDraggingPlayhead: state.setDraggingPlayhead,
});

// Track actions
export const selectTrackActions = (state: TimelineStore) => ({
  addTrack: state.addTrack,
  isTrackExpanded: state.isTrackExpanded,
  toggleTrackExpanded: state.toggleTrackExpanded,
  getExpandedTrackHeight: state.getExpandedTrackHeight,
  trackHasKeyframes: state.trackHasKeyframes,
  setTrackParent: state.setTrackParent,
});

// Clip actions
export const selectClipActions = (state: TimelineStore) => ({
  addClip: state.addClip,
  addCompClip: state.addCompClip,
  addTextClip: state.addTextClip,
  moveClip: state.moveClip,
  trimClip: state.trimClip,
  removeClip: state.removeClip,
  selectClip: state.selectClip,
  unlinkGroup: state.unlinkGroup,
  splitClip: state.splitClip,
  splitClipAtPlayhead: state.splitClipAtPlayhead,
  toggleClipReverse: state.toggleClipReverse,
  updateClipTransform: state.updateClipTransform,
  setClipParent: state.setClipParent,
  generateWaveformForClip: state.generateWaveformForClip,
});

// Transform/interpolation getters (stable functions)
export const selectTransformGetters = (state: TimelineStore) => ({
  getInterpolatedTransform: state.getInterpolatedTransform,
  getInterpolatedEffects: state.getInterpolatedEffects,
  getInterpolatedSpeed: state.getInterpolatedSpeed,
  getSourceTimeForClip: state.getSourceTimeForClip,
  getSnappedPosition: state.getSnappedPosition,
  getPositionWithResistance: state.getPositionWithResistance,
});

// Keyframe actions
export const selectKeyframeActions = (state: TimelineStore) => ({
  getClipKeyframes: state.getClipKeyframes,
  selectKeyframe: state.selectKeyframe,
  deselectAllKeyframes: state.deselectAllKeyframes,
  hasKeyframes: state.hasKeyframes,
  addKeyframe: state.addKeyframe,
  moveKeyframe: state.moveKeyframe,
  updateKeyframe: state.updateKeyframe,
  removeKeyframe: state.removeKeyframe,
  setPropertyValue: state.setPropertyValue,
  toggleCurveExpanded: state.toggleCurveExpanded,
  updateBezierHandle: state.updateBezierHandle,
});

// In/out point actions
export const selectInOutActions = (state: TimelineStore) => ({
  setInPoint: state.setInPoint,
  setOutPoint: state.setOutPoint,
  setInPointAtPlayhead: state.setInPointAtPlayhead,
  setOutPointAtPlayhead: state.setOutPointAtPlayhead,
  clearInOut: state.clearInOut,
});

// Zoom/scroll actions
export const selectViewActions = (state: TimelineStore) => ({
  setZoom: state.setZoom,
  setScrollX: state.setScrollX,
  setDuration: state.setDuration,
  toggleSnapping: state.toggleSnapping,
});

// Preview actions
export const selectPreviewActions = (state: TimelineStore) => ({
  toggleLoopPlayback: state.toggleLoopPlayback,
  toggleRamPreviewEnabled: state.toggleRamPreviewEnabled,
  startRamPreview: state.startRamPreview,
  cancelRamPreview: state.cancelRamPreview,
  getCachedRanges: state.getCachedRanges,
  getProxyCachedRanges: state.getProxyCachedRanges,
  startProxyCachePreload: state.startProxyCachePreload,
  cancelProxyCachePreload: state.cancelProxyCachePreload,
});

// Tool actions
export const selectToolActions = (state: TimelineStore) => ({
  setToolMode: state.setToolMode,
  toggleCutTool: state.toggleCutTool,
  toggleThumbnailsEnabled: state.toggleThumbnailsEnabled,
  toggleWaveformsEnabled: state.toggleWaveformsEnabled,
});

// Marker actions
export const selectMarkerActions = (state: TimelineStore) => ({
  addMarker: state.addMarker,
  moveMarker: state.moveMarker,
  removeMarker: state.removeMarker,
});

// Clipboard actions
export const selectClipboardActions = (state: TimelineStore) => ({
  copyClips: state.copyClips,
  pasteClips: state.pasteClips,
});
