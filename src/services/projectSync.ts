// Project Sync Service
// Synchronizes mediaStore and timelineStore with projectFileService

import { useMediaStore, type MediaFile, type Composition, type MediaFolder } from '../stores/mediaStore';
import { useTimelineStore } from '../stores/timeline';
import {
  projectFileService,
  type ProjectMediaFile,
  type ProjectComposition,
  type ProjectTrack,
  type ProjectClip,
  type ProjectMarker,
  type ProjectFolder,
} from './projectFileService';
import { fileSystemService } from './fileSystemService';
import { projectDB } from './projectDB';

// ============================================
// EXPORT FROM STORES TO PROJECT FILE
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
    frameRate: undefined, // Not stored in current system
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
      thumbnails: c.thumbnails,
      linkedClipId: c.linkedClipId,
      linkedGroupId: c.linkedGroupId,
      waveform: c.waveform,
      startTime: c.startTime,
      duration: c.duration,
      inPoint: c.inPoint || 0,
      outPoint: c.outPoint || c.duration,
      transform: {
        x: c.transform?.x || 0,
        y: c.transform?.y || 0,
        z: c.transform?.z || 0,
        scaleX: c.transform?.scaleX || 1,
        scaleY: c.transform?.scaleY || 1,
        rotation: c.transform?.rotation || 0,
        rotationX: c.transform?.rotationX || 0,
        rotationY: c.transform?.rotationY || 0,
        anchorX: c.transform?.anchorX || 0.5,
        anchorY: c.transform?.anchorY || 0.5,
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
      keyframes: [], // Keyframes stored separately in timeline
      volume: c.volume ?? 1,
      audioEnabled: c.audioEnabled !== false,
      reversed: c.reversed || false,
      disabled: c.disabled || false,
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
  }

  console.log('[ProjectSync] Synced stores to project');
}

// ============================================
// IMPORT FROM PROJECT FILE TO STORES
// ============================================

/**
 * Convert ProjectMediaFile to MediaFile format
 */
async function convertProjectMediaToStore(projectMedia: ProjectMediaFile[]): Promise<MediaFile[]> {
  const files: MediaFile[] = [];

  for (const pm of projectMedia) {
    // Try to get file handle from in-memory storage first
    let handle = fileSystemService.getFileHandle(pm.id);
    let file: File | undefined;
    let url = '';
    let thumbnailUrl: string | undefined;

    // If not in memory, try to get from IndexedDB
    if (!handle) {
      try {
        const storedHandle = await projectDB.getStoredHandle(`media_${pm.id}`);
        if (storedHandle && storedHandle.kind === 'file') {
          handle = storedHandle as FileSystemFileHandle;
          // Cache in memory for future use
          fileSystemService.storeFileHandle(pm.id, handle);
          console.log('[ProjectSync] Retrieved handle from IndexedDB for:', pm.name);
        }
      } catch (e) {
        console.warn('[ProjectSync] Failed to get handle from IndexedDB:', pm.name, e);
      }
    }

    if (handle) {
      try {
        // Check permission first
        const permission = await handle.queryPermission({ mode: 'read' });
        if (permission === 'granted') {
          file = await handle.getFile();
          url = URL.createObjectURL(file);
          console.log('[ProjectSync] Restored file from handle:', pm.name);
        } else {
          // Permission needs to be requested - will be done by reloadFile
          console.log('[ProjectSync] File needs permission:', pm.name);
        }
      } catch (e) {
        console.warn(`[ProjectSync] Could not access file: ${pm.name}`, e);
      }
    }

    files.push({
      id: pm.id,
      name: pm.name,
      type: pm.type,
      parentId: pm.folderId,
      createdAt: new Date(pm.importedAt).getTime(),
      file,
      url,
      thumbnailUrl,
      duration: pm.duration,
      width: pm.width,
      height: pm.height,
      proxyStatus: pm.hasProxy ? 'ready' : 'none',
      hasFileHandle: !!handle,
      filePath: pm.sourcePath,
    });
  }

  return files;
}

/**
 * Convert ProjectComposition to Composition format
 */
function convertProjectCompositionToStore(projectComps: ProjectComposition[]): Composition[] {
  return projectComps.map((pc) => {
    // Convert back to timelineData format
    const timelineData = {
      tracks: pc.tracks.map((t) => ({
        id: t.id,
        name: t.name,
        type: t.type,
        height: t.height,
        locked: t.locked,
        visible: t.visible,
        muted: t.muted,
        solo: t.solo,
      })),
      clips: pc.clips.map((c) => ({
        id: c.id,
        trackId: c.trackId,
        name: (c as any).name || '',
        mediaFileId: c.mediaId,  // Map mediaId -> mediaFileId for loadState
        sourceType: (c as any).sourceType || 'video',
        naturalDuration: (c as any).naturalDuration,
        thumbnails: (c as any).thumbnails,
        linkedClipId: (c as any).linkedClipId,
        linkedGroupId: (c as any).linkedGroupId,
        waveform: (c as any).waveform,
        startTime: c.startTime,
        duration: c.duration,
        inPoint: c.inPoint,
        outPoint: c.outPoint,
        transform: c.transform,
        effects: c.effects,
        masks: c.masks,
        volume: c.volume,
        audioEnabled: c.audioEnabled,
        reversed: c.reversed,
        disabled: c.disabled,
      })),
      // Required CompositionTimelineData fields
      playheadPosition: 0,
      duration: pc.duration,
      zoom: 1,
      scrollX: 0,
      inPoint: null,
      outPoint: null,
      loopPlayback: false,
    };

    const comp: Composition = {
      id: pc.id,
      name: pc.name,
      type: 'composition',
      parentId: pc.folderId,
      createdAt: Date.now(),
      width: pc.width,
      height: pc.height,
      frameRate: pc.frameRate,
      duration: pc.duration,
      backgroundColor: pc.backgroundColor,
      timelineData: timelineData as any, // Type assertion for complex nested types
    };
    return comp;
  });
}

/**
 * Convert ProjectFolder to MediaFolder format
 */
function convertProjectFolderToStore(projectFolders: ProjectFolder[]): MediaFolder[] {
  return projectFolders.map((pf) => ({
    id: pf.id,
    name: pf.name,
    parentId: pf.parentId,
    isExpanded: true,
    createdAt: Date.now(),
  }));
}

/**
 * Load project data from projectFileService into stores
 */
export async function loadProjectToStores(): Promise<void> {
  const projectData = projectFileService.getProjectData();
  if (!projectData) {
    console.error('[ProjectSync] No project data to load');
    return;
  }

  // Convert and load data
  const files = await convertProjectMediaToStore(projectData.media);
  const compositions = convertProjectCompositionToStore(projectData.compositions);
  const folders = convertProjectFolderToStore(projectData.folders);

  // Clear timeline first
  const timelineStore = useTimelineStore.getState();
  timelineStore.clearTimeline();

  // Update media store
  useMediaStore.setState({
    files,
    compositions: compositions.length > 0 ? compositions : [{
      id: `comp-${Date.now()}`,
      name: 'Comp 1',
      type: 'composition',
      parentId: null,
      createdAt: Date.now(),
      width: projectData.settings.width,
      height: projectData.settings.height,
      frameRate: projectData.settings.frameRate,
      duration: 60,
      backgroundColor: '#000000',
    }],
    folders,
    activeCompositionId: projectData.activeCompositionId,
    openCompositionIds: projectData.openCompositionIds || [],
    expandedFolderIds: projectData.expandedFolderIds || [],
  });

  // Load active composition's timeline
  if (projectData.activeCompositionId) {
    const activeComp = compositions.find((c) => c.id === projectData.activeCompositionId);
    if (activeComp?.timelineData) {
      await timelineStore.loadState(activeComp.timelineData);
    }
  }

  console.log('[ProjectSync] Loaded project to stores:', projectData.name);
}

// ============================================
// PROJECT OPERATIONS
// ============================================

/**
 * Create a new project
 */
export async function createNewProject(name: string): Promise<boolean> {
  // Sync current state first (in case user wants to save)
  await syncStoresToProject();

  // Create project on filesystem
  const success = await projectFileService.createProject(name);
  if (!success) return false;

  // Reset stores for new project
  useMediaStore.getState().newProject();

  // Sync empty state to new project
  await syncStoresToProject();
  await projectFileService.saveProject();

  return true;
}

/**
 * Open an existing project
 */
export async function openExistingProject(): Promise<boolean> {
  const success = await projectFileService.openProject();
  if (!success) return false;

  // Load project data to stores
  await loadProjectToStores();

  return true;
}

/**
 * Save current project
 */
export async function saveCurrentProject(): Promise<boolean> {
  if (!projectFileService.isProjectOpen()) {
    console.error('[ProjectSync] No project open');
    return false;
  }

  await syncStoresToProject();
  return await projectFileService.saveProject();
}

/**
 * Close current project
 */
export function closeCurrentProject(): void {
  projectFileService.closeProject();
  useMediaStore.getState().newProject();
}

// ============================================
// AUTO-SYNC
// ============================================

/**
 * Mark project as dirty when stores change
 */
export function setupAutoSync(): void {
  // Subscribe to store changes and mark project dirty
  useMediaStore.subscribe(
    (state) => [state.files, state.compositions, state.folders],
    () => {
      if (projectFileService.isProjectOpen()) {
        projectFileService.markDirty();
      }
    }
  );

  useTimelineStore.subscribe(
    (state) => [state.clips, state.tracks],
    () => {
      if (projectFileService.isProjectOpen()) {
        projectFileService.markDirty();
      }
    }
  );

  console.log('[ProjectSync] Auto-sync setup complete');
}
