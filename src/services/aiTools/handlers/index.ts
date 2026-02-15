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
} from './media';

import {
  handleSearchYouTube,
  handleListVideoFormats,
  handleDownloadAndImportVideo,
  handleGetYouTubeVideos,
} from './youtube';

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
};

const mediaHandlers: Record<string, (args: Record<string, unknown>, store: ReturnType<typeof useMediaStore.getState>) => Promise<ToolResult>> = {
  getMediaItems: handleGetMediaItems,
  createMediaFolder: handleCreateMediaFolder,
  renameMediaItem: handleRenameMediaItem,
  deleteMediaItem: handleDeleteMediaItem,
  moveMediaItems: handleMoveMediaItems,
  createComposition: handleCreateComposition,
  selectMediaItems: handleSelectMediaItems,
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
  // YouTube
  handleSearchYouTube,
  handleListVideoFormats,
  handleDownloadAndImportVideo,
  handleGetYouTubeVideos,
};
