// Project Sync Service
// Synchronizes mediaStore and timelineStore with projectFileService

import { Logger } from './logger';
import { useMediaStore, type MediaFile, type Composition, type MediaFolder } from '../stores/mediaStore';
import { getMediaInfo } from '../stores/mediaStore/helpers/mediaInfoHelpers';
import { createThumbnail } from '../stores/mediaStore/helpers/thumbnailHelpers';
import { updateTimelineClips } from '../stores/mediaStore/slices/fileManageSlice';

const log = Logger.create('ProjectSync');
import { useTimelineStore } from '../stores/timeline';
import { useYouTubeStore } from '../stores/youtubeStore';
import { useDockStore } from '../stores/dockStore';
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
      keyframes: c.keyframes || [], // Preserve keyframes from clip
      volume: c.volume ?? 1,
      audioEnabled: c.audioEnabled !== false,
      reversed: c.reversed || false,
      disabled: c.disabled || false,
      // Nested composition support
      isComposition: c.isComposition || undefined,
      compositionId: c.compositionId || undefined,
      // Text clip support
      textProperties: c.textProperties || undefined,
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
    };
  }

  log.info(' Synced stores to project');
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
          log.info(`Retrieved handle from IndexedDB for: ${pm.name}`);
        }
      } catch (e) {
        log.warn(`Failed to get handle from IndexedDB: ${pm.name}`, e);
      }
    }

    if (handle) {
      try {
        // Check permission first
        const permission = await handle.queryPermission({ mode: 'read' });
        if (permission === 'granted') {
          file = await handle.getFile();
          url = URL.createObjectURL(file);
          log.info(' Restored file from handle:', pm.name);
        } else {
          // Permission needs to be requested - will be done by reloadFile
          log.info(' File needs permission:', pm.name);
        }
      } catch (e) {
        log.warn(`Could not access file: ${pm.name}`, e);
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
      fps: pm.frameRate,
      codec: pm.codec,
      audioCodec: pm.audioCodec,
      container: pm.container,
      bitrate: pm.bitrate,
      fileSize: pm.fileSize,
      hasAudio: pm.hasAudio,
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
function convertProjectCompositionToStore(
  projectComps: ProjectComposition[],
  compositionViewState?: Record<string, {
    playheadPosition?: number;
    zoom?: number;
    scrollX?: number;
    inPoint?: number | null;
    outPoint?: number | null;
  }>
): Composition[] {
  return projectComps.map((pc) => {
    // Get saved view state for this composition
    const viewState = compositionViewState?.[pc.id];

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
        name: c.name || '',
        mediaFileId: c.mediaId,  // Map mediaId -> mediaFileId for loadState
        sourceType: c.sourceType || 'video',
        naturalDuration: c.naturalDuration,
        thumbnails: c.thumbnails,
        linkedClipId: c.linkedClipId,
        linkedGroupId: c.linkedGroupId,
        waveform: c.waveform,
        startTime: c.startTime,
        duration: c.duration,
        inPoint: c.inPoint,
        outPoint: c.outPoint,
        transform: c.transform,
        effects: c.effects,
        masks: c.masks,
        keyframes: c.keyframes || [], // Restore keyframes
        volume: c.volume,
        audioEnabled: c.audioEnabled,
        reversed: c.reversed,
        disabled: c.disabled,
        // Nested composition support
        isComposition: c.isComposition,
        compositionId: c.compositionId,
        // Text clip support
        textProperties: c.textProperties,
        // Transcript data
        transcript: c.transcript,
        transcriptStatus: c.transcriptStatus,
        // Analysis data
        analysis: c.analysis,
        analysisStatus: c.analysisStatus,
      })),
      // Restore view state from saved uiState, or use defaults
      playheadPosition: viewState?.playheadPosition ?? 0,
      duration: pc.duration,
      zoom: viewState?.zoom ?? 1,
      scrollX: viewState?.scrollX ?? 0,
      inPoint: viewState?.inPoint ?? null,
      outPoint: viewState?.outPoint ?? null,
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
    log.error(' No project data to load');
    return;
  }

  // Convert and load data
  const files = await convertProjectMediaToStore(projectData.media);
  const compositions = convertProjectCompositionToStore(
    projectData.compositions,
    projectData.uiState?.compositionViewState
  );
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

  // Load YouTube panel state
  if (projectData.youtube) {
    useYouTubeStore.getState().loadState(projectData.youtube);
  } else {
    useYouTubeStore.getState().reset();
  }

  // Restore dock layout from project
  if (projectData.uiState?.dockLayout) {
    useDockStore.getState().setLayoutFromProject(projectData.uiState.dockLayout);
    log.info(' Restored dock layout from project');
  }

  // Restore per-project UI settings to localStorage
  if (projectData.uiState?.mediaPanelColumns) {
    localStorage.setItem('media-panel-column-order', JSON.stringify(projectData.uiState.mediaPanelColumns));
  }
  if (projectData.uiState?.mediaPanelNameWidth !== undefined) {
    localStorage.setItem('media-panel-name-width', String(projectData.uiState.mediaPanelNameWidth));
  }
  if (projectData.uiState?.transcriptLanguage) {
    localStorage.setItem('transcriptLanguage', projectData.uiState.transcriptLanguage);
  }

  log.info(' Loaded project to stores:', projectData.name);

  // Auto-relink missing files from Raw folder
  await autoRelinkFromRawFolder();

  // Restore thumbnails and refresh metadata in the background
  restoreMediaThumbnails();
  refreshMediaMetadata();
}

/**
 * Refresh media metadata (codec, bitrate, hasAudio) for all loaded files.
 * This runs in the background after project load to populate metadata fields.
 */
async function refreshMediaMetadata(): Promise<void> {
  const mediaState = useMediaStore.getState();
  // Refresh files that have a file object but are missing important metadata
  const filesToRefresh = mediaState.files.filter(f =>
    f.file && (
      f.codec === undefined ||
      f.container === undefined ||
      f.fileSize === undefined ||
      (f.type === 'video' && f.hasAudio === undefined)
    )
  );

  if (filesToRefresh.length === 0) {
    log.debug('No files need metadata refresh');
    return;
  }

  log.info(`Refreshing metadata for ${filesToRefresh.length} files...`);

  // Process files in parallel but with a limit to avoid overwhelming the browser
  const batchSize = 3;
  for (let i = 0; i < filesToRefresh.length; i += batchSize) {
    const batch = filesToRefresh.slice(i, i + batchSize);

    await Promise.all(batch.map(async (mediaFile) => {
      if (!mediaFile.file) return;

      try {
        const info = await getMediaInfo(mediaFile.file, mediaFile.type);

        // Update the file in the store
        useMediaStore.setState((state) => ({
          files: state.files.map((f) =>
            f.id === mediaFile.id
              ? {
                  ...f,
                  codec: info.codec || f.codec,
                  audioCodec: info.audioCodec,
                  container: info.container || f.container,
                  bitrate: info.bitrate || f.bitrate,
                  fileSize: info.fileSize || f.fileSize,
                  hasAudio: info.hasAudio ?? f.hasAudio,
                  fps: info.fps || f.fps,
                }
              : f
          ),
        }));

        log.debug(`Refreshed metadata for: ${mediaFile.name}`, {
          codec: info.codec,
          hasAudio: info.hasAudio,
          bitrate: info.bitrate,
        });
      } catch (e) {
        log.warn(`Failed to refresh metadata for: ${mediaFile.name}`, e);
      }
    }));
  }

  log.info('Media metadata refresh complete');
}

/**
 * Restore thumbnails for media files after project load.
 * Checks project folder first, then regenerates from file if needed.
 */
async function restoreMediaThumbnails(): Promise<void> {
  const mediaState = useMediaStore.getState();
  // Find files that need thumbnails (video/image files without thumbnailUrl)
  const filesToRestore = mediaState.files.filter(f =>
    f.file && !f.thumbnailUrl && (f.type === 'video' || f.type === 'image')
  );

  if (filesToRestore.length === 0) {
    log.debug('No thumbnails need restoration');
    return;
  }

  log.info(`Restoring thumbnails for ${filesToRestore.length} files...`);

  // Process in batches to avoid overwhelming browser
  const batchSize = 5;
  for (let i = 0; i < filesToRestore.length; i += batchSize) {
    const batch = filesToRestore.slice(i, i + batchSize);

    await Promise.all(batch.map(async (mediaFile) => {
      if (!mediaFile.file) return;

      try {
        let thumbnailUrl: string | undefined;

        // First try to get from project folder if we have a hash
        if (mediaFile.fileHash && projectFileService.isProjectOpen()) {
          const existingBlob = await projectFileService.getThumbnail(mediaFile.fileHash);
          if (existingBlob && existingBlob.size > 0) {
            thumbnailUrl = URL.createObjectURL(existingBlob);
            log.debug(`Restored thumbnail from project: ${mediaFile.name}`);
          }
        }

        // If not found in project, regenerate from file
        if (!thumbnailUrl) {
          thumbnailUrl = await createThumbnail(mediaFile.file, mediaFile.type as 'video' | 'image');
          log.debug(`Regenerated thumbnail: ${mediaFile.name}`);
        }

        if (thumbnailUrl) {
          useMediaStore.setState((state) => ({
            files: state.files.map((f) =>
              f.id === mediaFile.id ? { ...f, thumbnailUrl } : f
            ),
          }));
        }
      } catch (e) {
        log.warn(`Failed to restore thumbnail for: ${mediaFile.name}`, e);
      }
    }));
  }

  log.info('Thumbnail restoration complete');
}

/**
 * Automatically relink missing media files from the project's Raw folder
 * This runs silently after project load - no user interaction needed if all files are found
 */
async function autoRelinkFromRawFolder(): Promise<void> {
  if (!projectFileService.isProjectOpen()) return;

  const mediaState = useMediaStore.getState();
  const missingFiles = mediaState.files.filter(f => !f.file && !f.url);

  if (missingFiles.length === 0) {
    log.info(' No missing files to relink');
    return;
  }

  log.info(`Attempting auto-relink for ${missingFiles.length} missing files...`);

  // Scan the Raw folder
  const rawFiles = await projectFileService.scanRawFolder();
  if (rawFiles.size === 0) {
    log.info(' Raw folder is empty or not accessible');
    return;
  }

  log.debug(`Found ${rawFiles.size} files in Raw folder`);

  // Match and relink files
  let relinkedCount = 0;
  const updatedFiles = [...mediaState.files];

  for (let i = 0; i < updatedFiles.length; i++) {
    const file = updatedFiles[i];
    if (file.file || file.url) continue; // Already has file

    const searchName = file.name.toLowerCase();
    const handle = rawFiles.get(searchName);

    if (handle) {
      try {
        const fileObj = await handle.getFile();
        const url = URL.createObjectURL(fileObj);

        // Store handle for future access
        fileSystemService.storeFileHandle(file.id, handle);
        await projectDB.storeHandle(`media_${file.id}`, handle);

        // Update file entry
        updatedFiles[i] = {
          ...file,
          file: fileObj,
          url,
          hasFileHandle: true,
        };

        relinkedCount++;
        log.debug(`Auto-relinked: ${file.name}`);
      } catch (e) {
        log.warn(`Could not read file from Raw: ${file.name}`, e);
      }
    }
  }

  if (relinkedCount > 0) {
    // Update media store with relinked files
    useMediaStore.setState({ files: updatedFiles });
    log.info(`Auto-relinked ${relinkedCount}/${missingFiles.length} files from Raw folder`);

    // Update timeline clips with proper source elements (video/audio/image)
    for (const file of updatedFiles) {
      if (file.file) {
        await updateTimelineClips(file.id, file.file);
      }
    }

    // Reload nested composition clips that may need their content updated
    await reloadNestedCompositionClips();
  } else {
    log.info(' No files could be auto-relinked from Raw folder');
  }
}

/**
 * Reload nested clips for composition clips that are missing their content.
 * This is called after auto-relinking when media files become available.
 */
async function reloadNestedCompositionClips(): Promise<void> {
  const timelineStore = useTimelineStore.getState();
  const mediaStore = useMediaStore.getState();

  // Find composition clips that have no nested clips (need reload)
  const compClips = timelineStore.clips.filter(
    c => c.isComposition && c.compositionId && (!c.nestedClips || c.nestedClips.length === 0)
  );

  if (compClips.length === 0) return;

  log.info(`Reloading ${compClips.length} nested composition clips...`);

  for (const compClip of compClips) {
    const composition = mediaStore.compositions.find(c => c.id === compClip.compositionId);
    if (!composition?.timelineData) continue;

    const nestedClips: any[] = [];
    const nestedTracks = composition.timelineData.tracks;

    for (const nestedSerializedClip of composition.timelineData.clips) {
      const nestedMediaFile = mediaStore.files.find(f => f.id === nestedSerializedClip.mediaFileId);
      if (!nestedMediaFile?.file) continue;

      const nestedClip: any = {
        id: `nested-${compClip.id}-${nestedSerializedClip.id}`,
        trackId: nestedSerializedClip.trackId,
        name: nestedSerializedClip.name,
        file: nestedMediaFile.file,
        startTime: nestedSerializedClip.startTime,
        duration: nestedSerializedClip.duration,
        inPoint: nestedSerializedClip.inPoint,
        outPoint: nestedSerializedClip.outPoint,
        source: null,
        thumbnails: nestedSerializedClip.thumbnails,
        transform: nestedSerializedClip.transform,
        effects: nestedSerializedClip.effects || [],
        masks: nestedSerializedClip.masks || [],
        isLoading: true,
      };

      nestedClips.push(nestedClip);

      // Load the video/audio/image element
      const sourceType = nestedSerializedClip.sourceType;
      const fileUrl = URL.createObjectURL(nestedMediaFile.file);

      if (sourceType === 'video') {
        const video = document.createElement('video');
        video.src = fileUrl;
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';
        video.crossOrigin = 'anonymous';

        video.addEventListener('canplaythrough', () => {
          nestedClip.source = {
            type: 'video',
            videoElement: video,
            naturalDuration: video.duration,
          };
          nestedClip.isLoading = false;

          // Trigger state update
          const currentClips = timelineStore.clips;
          useTimelineStore.setState({ clips: [...currentClips] });
        }, { once: true });

        video.load();
      } else if (sourceType === 'audio') {
        const audio = document.createElement('audio');
        audio.src = fileUrl;
        audio.preload = 'auto';

        audio.addEventListener('canplaythrough', () => {
          nestedClip.source = {
            type: 'audio',
            audioElement: audio,
            naturalDuration: audio.duration,
          };
          nestedClip.isLoading = false;

          const currentClips = timelineStore.clips;
          useTimelineStore.setState({ clips: [...currentClips] });
        }, { once: true });

        audio.load();
      } else if (sourceType === 'image') {
        const img = new Image();
        img.src = fileUrl;
        img.crossOrigin = 'anonymous';

        img.addEventListener('load', () => {
          nestedClip.source = {
            type: 'image',
            imageElement: img,
          };
          nestedClip.isLoading = false;

          const currentClips = timelineStore.clips;
          useTimelineStore.setState({ clips: [...currentClips] });
        }, { once: true });
      }
    }

    // Update the composition clip with nested data
    if (nestedClips.length > 0) {
      timelineStore.updateClip(compClip.id, {
        nestedClips,
        nestedTracks,
        isLoading: false,
      });

      // Generate thumbnails if missing
      if (!compClip.thumbnails || compClip.thumbnails.length === 0) {
        const { generateCompThumbnails } = await import('../stores/timeline/clip/addCompClip');
        const compDuration = composition.timelineData?.duration ?? composition.duration;
        generateCompThumbnails({
          clipId: compClip.id,
          nestedClips,
          compDuration,
          thumbnailsEnabled: timelineStore.thumbnailsEnabled,
          get: useTimelineStore.getState,
          set: useTimelineStore.setState,
        });
      }
    }
  }

  log.info('Nested composition clips reloaded');
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
    log.error(' No project open');
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

  // Subscribe to YouTube store changes
  let prevYouTubeVideos = useYouTubeStore.getState().videos;
  useYouTubeStore.subscribe((state) => {
    if (state.videos !== prevYouTubeVideos) {
      prevYouTubeVideos = state.videos;
      if (projectFileService.isProjectOpen()) {
        projectFileService.markDirty();
      }
    }
  });

  // Subscribe to dock layout changes
  let prevDockLayout = useDockStore.getState().layout;
  useDockStore.subscribe((state) => {
    if (state.layout !== prevDockLayout) {
      prevDockLayout = state.layout;
      if (projectFileService.isProjectOpen()) {
        projectFileService.markDirty();
      }
    }
  });

  log.info(' Auto-sync setup complete');
}
