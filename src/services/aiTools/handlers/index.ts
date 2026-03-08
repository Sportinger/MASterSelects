// AI Tool Handlers - Main dispatcher

import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import type { ToolResult } from '../types';

// Import handlers by category
import {
  handleGetTimelineState,
  handleSetPlayhead,
  handleSetInOutPoints,
} from './timeline';

import {
  handleGetClipDetails,
  handleGetClipsInTimeRange,
  handleSplitClip,
  handleDeleteClip,
  handleDeleteClips,
  handleCutRangesFromClip,
  handleMoveClip,
  handleTrimClip,
  handleSplitClipEvenly,
  handleSplitClipAtTimes,
  handleReorderClips,
  handleSelectClips,
  handleClearSelection,
  handleAddClipSegment,
} from './clips';

import {
  handleCreateTrack,
  handleDeleteTrack,
  handleSetTrackVisibility,
  handleSetTrackMuted,
} from './tracks';

import {
  handleGetClipAnalysis,
  handleGetClipTranscript,
  handleFindSilentSections,
  handleFindLowQualitySections,
  handleStartClipAnalysis,
  handleStartClipTranscription,
} from './analysis';

import {
  handleCaptureFrame,
  handleGetCutPreviewQuad,
  handleGetFramesAtTimes,
} from './preview';

import {
  handleGetMediaItems,
  handleCreateMediaFolder,
  handleRenameMediaItem,
  handleDeleteMediaItem,
  handleMoveMediaItems,
  handleCreateComposition,
  handleSelectMediaItems,
  handleImportLocalFiles,
  handleListLocalFiles,
} from './media';

import {
  handleSearchYouTube,
  handleListVideoFormats,
  handleDownloadAndImportVideo,
  handleGetYouTubeVideos,
} from './youtube';

import { handleSetTransform } from './transform';

import {
  handleListEffects,
  handleAddEffect,
  handleRemoveEffect,
  handleUpdateEffect,
} from './effects';

import {
  handleGetKeyframes,
  handleAddKeyframe,
  handleRemoveKeyframe,
} from './keyframes';

import {
  handlePlay,
  handlePause,
  handleSetClipSpeed,
  handleUndo,
  handleRedo,
  handleAddMarker,
  handleGetMarkers,
  handleRemoveMarker,
} from './playback';

import {
  handleAddTransition,
  handleRemoveTransition,
} from './transitions';

import {
  handleGetMasks,
  handleAddRectangleMask,
  handleAddEllipseMask,
  handleAddMask,
  handleRemoveMask,
  handleUpdateMask,
  handleAddVertex,
  handleRemoveVertex,
  handleUpdateVertex,
} from './masks';

import {
  handleGetStats,
  handleGetLogs,
  handleGetPlaybackTrace,
  handleGetStatsHistory,
} from './stats';

// Handler registry - maps tool names to handler functions
const timelineHandlers: Record<string, (args: Record<string, unknown>, store: ReturnType<typeof useTimelineStore.getState>) => Promise<ToolResult>> = {
  getTimelineState: handleGetTimelineState,
  setPlayhead: handleSetPlayhead,
  setInOutPoints: handleSetInOutPoints,
  getClipDetails: handleGetClipDetails,
  getClipsInTimeRange: handleGetClipsInTimeRange,
  splitClip: handleSplitClip,
  deleteClip: handleDeleteClip,
  deleteClips: handleDeleteClips,
  cutRangesFromClip: handleCutRangesFromClip,
  moveClip: handleMoveClip,
  trimClip: handleTrimClip,
  splitClipEvenly: handleSplitClipEvenly,
  splitClipAtTimes: handleSplitClipAtTimes,
  reorderClips: handleReorderClips,
  selectClips: handleSelectClips,
  clearSelection: handleClearSelection,
  createTrack: handleCreateTrack,
  deleteTrack: handleDeleteTrack,
  setTrackVisibility: handleSetTrackVisibility,
  setTrackMuted: handleSetTrackMuted,
  getClipAnalysis: handleGetClipAnalysis,
  getClipTranscript: handleGetClipTranscript,
  findSilentSections: handleFindSilentSections,
  findLowQualitySections: handleFindLowQualitySections,
  startClipAnalysis: handleStartClipAnalysis,
  startClipTranscription: handleStartClipTranscription,
  captureFrame: handleCaptureFrame,
  getCutPreviewQuad: handleGetCutPreviewQuad,
  getFramesAtTimes: handleGetFramesAtTimes,
  // Transform
  setTransform: handleSetTransform,
  // Effects
  addEffect: handleAddEffect,
  removeEffect: handleRemoveEffect,
  updateEffect: handleUpdateEffect,
  // Keyframes
  getKeyframes: handleGetKeyframes,
  addKeyframe: handleAddKeyframe,
  // Playback & Control
  play: handlePlay,
  pause: handlePause,
  setClipSpeed: handleSetClipSpeed,
  // Markers
  addMarker: handleAddMarker,
  getMarkers: handleGetMarkers,
  removeMarker: handleRemoveMarker,
  // Transitions
  addTransition: handleAddTransition,
  removeTransition: handleRemoveTransition,
  // Masks
  getMasks: handleGetMasks,
  addRectangleMask: handleAddRectangleMask,
  addEllipseMask: handleAddEllipseMask,
  addMask: handleAddMask,
  removeMask: handleRemoveMask,
  updateMask: handleUpdateMask,
  addVertex: handleAddVertex,
  removeVertex: handleRemoveVertex,
  updateVertex: handleUpdateVertex,
};

const mediaHandlers: Record<string, (args: Record<string, unknown>, store: ReturnType<typeof useMediaStore.getState>) => Promise<ToolResult>> = {
  getMediaItems: handleGetMediaItems,
  createMediaFolder: handleCreateMediaFolder,
  renameMediaItem: handleRenameMediaItem,
  deleteMediaItem: handleDeleteMediaItem,
  moveMediaItems: handleMoveMediaItems,
  createComposition: handleCreateComposition,
  selectMediaItems: handleSelectMediaItems,
  importLocalFiles: handleImportLocalFiles,
};

// Self-contained handlers (no store dependency, or fetch own stores)
const selfContainedHandlers: Record<string, (args: Record<string, unknown>) => Promise<ToolResult>> = {
  listLocalFiles: handleListLocalFiles,
  addClipSegment: handleAddClipSegment,
  listEffects: handleListEffects,
  removeKeyframe: handleRemoveKeyframe,
  undo: handleUndo,
  redo: handleRedo,
  // Stats
  getStats: handleGetStats,
  getStatsHistory: handleGetStatsHistory,
  getLogs: handleGetLogs,
  getPlaybackTrace: handleGetPlaybackTrace,
};

// YouTube handlers - self-contained, fetch their own stores
const youtubeHandlers: Record<string, (args: Record<string, unknown>) => Promise<ToolResult>> = {
  searchYouTube: handleSearchYouTube,
  listVideoFormats: handleListVideoFormats,
  downloadAndImportVideo: handleDownloadAndImportVideo,
  getYouTubeVideos: handleGetYouTubeVideos,
};

/**
 * Execute a tool by name
 * Dispatches to the appropriate handler based on tool name
 */
export async function executeToolInternal(
  toolName: string,
  args: Record<string, unknown>,
  timelineStore: ReturnType<typeof useTimelineStore.getState>,
  mediaStore: ReturnType<typeof useMediaStore.getState>
): Promise<ToolResult> {
  // Check timeline handlers first
  if (toolName in timelineHandlers) {
    return timelineHandlers[toolName](args, timelineStore);
  }

  // Check media handlers
  if (toolName in mediaHandlers) {
    return mediaHandlers[toolName](args, mediaStore);
  }

  // Check self-contained handlers (no store dependency)
  if (toolName in selfContainedHandlers) {
    return selfContainedHandlers[toolName](args);
  }

  // Check YouTube handlers
  if (toolName in youtubeHandlers) {
    return youtubeHandlers[toolName](args);
  }

  // Unknown tool
  return { success: false, error: `Unknown tool: ${toolName}` };
}

// Re-export individual handlers for direct use if needed
export {
  // Timeline
  handleGetTimelineState,
  handleSetPlayhead,
  handleSetInOutPoints,
  // Clips
  handleGetClipDetails,
  handleGetClipsInTimeRange,
  handleSplitClip,
  handleDeleteClip,
  handleDeleteClips,
  handleCutRangesFromClip,
  handleMoveClip,
  handleTrimClip,
  handleSplitClipEvenly,
  handleSplitClipAtTimes,
  handleReorderClips,
  handleSelectClips,
  handleClearSelection,
  handleAddClipSegment,
  // Tracks
  handleCreateTrack,
  handleDeleteTrack,
  handleSetTrackVisibility,
  handleSetTrackMuted,
  // Analysis
  handleGetClipAnalysis,
  handleGetClipTranscript,
  handleFindSilentSections,
  handleFindLowQualitySections,
  handleStartClipAnalysis,
  handleStartClipTranscription,
  // Preview
  handleCaptureFrame,
  handleGetCutPreviewQuad,
  handleGetFramesAtTimes,
  // Media
  handleGetMediaItems,
  handleCreateMediaFolder,
  handleRenameMediaItem,
  handleDeleteMediaItem,
  handleMoveMediaItems,
  handleCreateComposition,
  handleSelectMediaItems,
  handleImportLocalFiles,
  handleListLocalFiles,
  // YouTube
  handleSearchYouTube,
  handleListVideoFormats,
  handleDownloadAndImportVideo,
  handleGetYouTubeVideos,
  // Transform
  handleSetTransform,
  // Effects
  handleListEffects,
  handleAddEffect,
  handleRemoveEffect,
  handleUpdateEffect,
  // Keyframes
  handleGetKeyframes,
  handleAddKeyframe,
  handleRemoveKeyframe,
  // Playback & Control
  handlePlay,
  handlePause,
  handleSetClipSpeed,
  handleUndo,
  handleRedo,
  // Markers
  handleAddMarker,
  handleGetMarkers,
  handleRemoveMarker,
  // Transitions
  handleAddTransition,
  handleRemoveTransition,
  // Masks
  handleGetMasks,
  handleAddRectangleMask,
  handleAddEllipseMask,
  handleAddMask,
  handleRemoveMask,
  handleUpdateMask,
  handleAddVertex,
  handleRemoveVertex,
  handleUpdateVertex,
  // Stats
  handleGetStats,
  handleGetLogs,
  handleGetPlaybackTrace,
  handleGetStatsHistory,
};
