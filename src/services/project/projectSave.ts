// Project Save — sync stores to project file format

import { Logger } from '../logger';
import { useMediaStore, type MediaFile, type Composition, type MediaFolder } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { useYouTubeStore } from '../../stores/youtubeStore';
import { useDockStore } from '../../stores/dockStore';
import {
  projectFileService,
  type ProjectMediaFile,
  type ProjectComposition,
  type ProjectTrack,
  type ProjectClip,
  type ProjectMarker,
  type ProjectFolder,
} from '../projectFileService';

const log = Logger.create('ProjectSync');

// ============================================
// CONVERTER HELPERS (store → project format)
// ============================================

/**
 * Convert mediaStore files to ProjectMediaFile format
 */
function convertMediaFiles(files: MediaFile[]): ProjectMediaFile[] {
  return files.map((file) => ({
    id: file.id,
    name: file.name,
    type: file.type as 'video' | 'audio' | 'image',
    sourcePath: file.filePath || file.name,
    duration: file.duration,
    width: file.width,
    height: file.height,
    frameRate: file.fps,
    codec: file.codec,
    audioCodec: file.audioCodec,
    container: file.container,
    bitrate: file.bitrate,
    fileSize: file.fileSize,
    hasAudio: file.hasAudio,
    hasProxy: file.proxyStatus === 'ready',
    folderId: file.parentId,
    importedAt: new Date(file.createdAt).toISOString(),
  }));
}

/**
 * Convert mediaStore folders to ProjectFolder format
 */
function convertFolders(folders: MediaFolder[]): ProjectFolder[] {
  return folders.map((folder) => ({
    id: folder.id,
    name: folder.name,
    parentId: folder.parentId,
  }));
}

/**
 * Convert compositions to ProjectComposition format
 */
function convertCompositions(compositions: Composition[]): ProjectComposition[] {
  return compositions.map((comp) => {
    const timelineData = comp.timelineData;

    // Convert tracks
    const tracks: ProjectTrack[] = (timelineData?.tracks || []).map((t: any) => ({
      id: t.id,
      name: t.name,
      type: t.type,
      height: t.height || 60,
      locked: t.locked || false,
      visible: t.visible !== false,
      muted: t.muted || false,
      solo: t.solo || false,
    }));

    // Convert clips
    const clips: ProjectClip[] = (timelineData?.clips || []).map((c: any) => ({
      id: c.id,
      trackId: c.trackId,
      name: c.name || '',
      mediaId: c.source?.mediaFileId || c.mediaFileId || c.mediaId || '',
      sourceType: c.source?.type || c.sourceType || 'video',
      naturalDuration: c.source?.naturalDuration || c.naturalDuration,
      linkedClipId: c.linkedClipId,
      linkedGroupId: c.linkedGroupId,
      waveform: c.waveform,
      startTime: c.startTime,
      duration: c.duration,
      inPoint: c.inPoint || 0,
      outPoint: c.outPoint || c.duration,
      transform: {
        // Store uses nested format (ClipTransform): position.x, scale.x, rotation.z
        // Project file uses flat format (ProjectTransform): x, scaleX, rotation
        // Support both formats for backwards compatibility
        x: c.transform?.position?.x ?? c.transform?.x ?? 0,
        y: c.transform?.position?.y ?? c.transform?.y ?? 0,
        z: c.transform?.position?.z ?? c.transform?.z ?? 0,
        scaleX: c.transform?.scale?.x ?? c.transform?.scaleX ?? 1,
        scaleY: c.transform?.scale?.y ?? c.transform?.scaleY ?? 1,
        rotation: c.transform?.rotation?.z ?? (typeof c.transform?.rotation === 'number' ? c.transform.rotation : 0),
        rotationX: c.transform?.rotation?.x ?? c.transform?.rotationX ?? 0,
        rotationY: c.transform?.rotation?.y ?? c.transform?.rotationY ?? 0,
        anchorX: c.transform?.anchorX ?? 0.5,
        anchorY: c.transform?.anchorY ?? 0.5,
        opacity: c.transform?.opacity ?? 1,
        blendMode: c.transform?.blendMode || 'normal',
      },
      effects: (c.effects || []).map((e: any) => ({
        id: e.id,
        type: e.type,
        name: e.name || e.type,
        enabled: e.enabled !== false,
        params: e.params || {},
      })),
      masks: (c.masks || []).map((m: any) => ({
        id: m.id,
        name: m.name || 'Mask',
        mode: m.mode || 'add',
        inverted: m.inverted || false,
        opacity: m.opacity ?? 1,
        feather: m.feather || 0,
        featherQuality: m.featherQuality || 8,
        visible: m.visible !== false,
        closed: m.closed !== false,
        vertices: m.vertices || [],
        position: m.position || { x: 0, y: 0 },
      })),
      keyframes: c.keyframes || [],
      volume: c.volume ?? 1,
      audioEnabled: c.audioEnabled !== false,
      reversed: c.reversed || false,
      disabled: c.disabled || false,
      // Nested composition support
      isComposition: c.isComposition || undefined,
      compositionId: c.compositionId || undefined,
      // Text clip support
      textProperties: c.textProperties || undefined,
      // Solid clip support
      solidColor: c.solidColor || undefined,
      // Transcript data
      transcript: c.transcript || undefined,
      transcriptStatus: c.transcriptStatus || undefined,
      // Analysis data
      analysis: c.analysis || undefined,
      analysisStatus: c.analysisStatus || undefined,
    }));

    // Note: markers not currently stored in CompositionTimelineData
    const markers: ProjectMarker[] = [];

    return {
      id: comp.id,
      name: comp.name,
      width: comp.width,
      height: comp.height,
      frameRate: comp.frameRate,
      duration: comp.duration,
      backgroundColor: comp.backgroundColor,
      folderId: comp.parentId,
      tracks,
      clips,
      markers,
    };
  });
}

// ============================================
// SYNC & SAVE
// ============================================

/**
 * Sync current store state to projectFileService
 */
export async function syncStoresToProject(): Promise<void> {
  const mediaState = useMediaStore.getState();
  const timelineStore = useTimelineStore.getState();

  // Save current timeline to active composition first
  if (mediaState.activeCompositionId) {
    const timelineData = timelineStore.getSerializableState();
    useMediaStore.setState((state) => ({
      compositions: state.compositions.map((c) =>
        c.id === mediaState.activeCompositionId ? { ...c, timelineData } : c
      ),
    }));
  }

  // Get fresh state after update
  const freshState = useMediaStore.getState();

  // Update project file data
  projectFileService.updateMedia(convertMediaFiles(freshState.files));
  projectFileService.updateCompositions(convertCompositions(freshState.compositions));
  projectFileService.updateFolders(convertFolders(freshState.folders));

  // Update active state
  const projectData = projectFileService.getProjectData();
  if (projectData) {
    projectData.activeCompositionId = freshState.activeCompositionId;
    projectData.openCompositionIds = freshState.openCompositionIds;
    projectData.expandedFolderIds = freshState.expandedFolderIds;
    projectData.slotAssignments = freshState.slotAssignments;

    // Save YouTube panel state
    const youtubeState = useYouTubeStore.getState().getState();
    projectData.youtube = youtubeState;

    // Save UI state (dock layout + composition view states)
    const dockLayout = useDockStore.getState().getLayoutForProject();

    // Build composition view state from all compositions
    const compositionViewState: Record<string, {
      playheadPosition?: number;
      zoom?: number;
      scrollX?: number;
      inPoint?: number | null;
      outPoint?: number | null;
    }> = {};

    // Get current timeline state for active composition
    const timelineState = useTimelineStore.getState();
    if (freshState.activeCompositionId) {
      compositionViewState[freshState.activeCompositionId] = {
        playheadPosition: timelineState.playheadPosition,
        zoom: timelineState.zoom,
        scrollX: timelineState.scrollX,
        inPoint: timelineState.inPoint,
        outPoint: timelineState.outPoint,
      };
    }

    // Also save view state from other compositions' timelineData
    for (const comp of freshState.compositions) {
      if (comp.id !== freshState.activeCompositionId && comp.timelineData) {
        compositionViewState[comp.id] = {
          playheadPosition: comp.timelineData.playheadPosition,
          zoom: comp.timelineData.zoom,
          scrollX: comp.timelineData.scrollX,
          inPoint: comp.timelineData.inPoint,
          outPoint: comp.timelineData.outPoint,
        };
      }
    }

    // Capture per-project UI settings from localStorage
    const mediaPanelColumns = localStorage.getItem('media-panel-column-order');
    const mediaPanelNameWidth = localStorage.getItem('media-panel-name-width');
    const transcriptLanguage = localStorage.getItem('transcriptLanguage');

    projectData.uiState = {
      dockLayout,
      compositionViewState,
      mediaPanelColumns: mediaPanelColumns ? JSON.parse(mediaPanelColumns) : undefined,
      mediaPanelNameWidth: mediaPanelNameWidth ? parseInt(mediaPanelNameWidth, 10) : undefined,
      transcriptLanguage: transcriptLanguage || undefined,
      thumbnailsEnabled: timelineState.thumbnailsEnabled,
      waveformsEnabled: timelineState.waveformsEnabled,
      proxyEnabled: useMediaStore.getState().proxyEnabled,
      showTranscriptMarkers: timelineState.showTranscriptMarkers,
    };

    // Save text and solid items
    (projectData as any).textItems = freshState.textItems;
    (projectData as any).solidItems = freshState.solidItems;
  }

  log.info(' Synced stores to project');
}

/**
 * Save current project
 */
export async function saveCurrentProject(): Promise<boolean> {
  if (!projectFileService.isProjectOpen()) {
    log.error(' No project open');
    return false;
  }

  await syncStoresToProject();
  return await projectFileService.saveProject();
}
