// Project Load — load project file data into stores + background restoration

import { Logger } from '../logger';
import { engine } from '../../engine/WebGPUEngine';
import { useMediaStore, type MediaFile, type Composition, type MediaFolder } from '../../stores/mediaStore';
import { getMediaInfo } from '../../stores/mediaStore/helpers/mediaInfoHelpers';
import { createThumbnail } from '../../stores/mediaStore/helpers/thumbnailHelpers';
import { updateTimelineClips } from '../../stores/mediaStore/slices/fileManageSlice';
import { useTimelineStore } from '../../stores/timeline';
import { useYouTubeStore } from '../../stores/youtubeStore';
import { useDockStore } from '../../stores/dockStore';
import { useSettingsStore } from '../../stores/settingsStore';
import {
  projectFileService,
  type ProjectMediaFile,
  type ProjectComposition,
  type ProjectFolder,
} from '../projectFileService';
import { fileSystemService } from '../fileSystemService';
import { projectDB } from '../projectDB';

const log = Logger.create('ProjectSync');

// ============================================
// REVERSE CONVERTERS (project format → store)
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
        linkedClipId: c.linkedClipId,
        linkedGroupId: c.linkedGroupId,
        waveform: c.waveform,
        startTime: c.startTime,
        duration: c.duration,
        inPoint: c.inPoint,
        outPoint: c.outPoint,
        // Convert flat ProjectTransform → nested ClipTransform format
        // Also supports already-nested format for forward compatibility
        transform: c.transform ? (() => {
          const t = c.transform as any;
          return {
            position: t.position ?? { x: t.x ?? 0, y: t.y ?? 0, z: t.z ?? 0 },
            scale: t.scale ?? { x: t.scaleX ?? 1, y: t.scaleY ?? 1 },
            rotation: t.rotation != null && typeof t.rotation === 'object'
              ? t.rotation
              : { x: t.rotationX ?? 0, y: t.rotationY ?? 0, z: t.rotation ?? 0 },
            opacity: t.opacity ?? 1,
            blendMode: t.blendMode ?? 'normal',
          };
        })() : undefined,
        effects: c.effects,
        masks: c.masks,
        keyframes: c.keyframes || [],
        volume: c.volume,
        audioEnabled: c.audioEnabled,
        reversed: c.reversed,
        disabled: c.disabled,
        // Nested composition support
        isComposition: c.isComposition,
        compositionId: c.compositionId,
        // Text clip support
        textProperties: c.textProperties,
        // Solid clip support
        solidColor: c.solidColor,
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

// ============================================
// LOAD PROJECT TO STORES
// ============================================

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

  // Restore text and solid items
  const textItems = (projectData as any).textItems || [];
  const solidItems = (projectData as any).solidItems || [];

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
    textItems,
    solidItems,
    activeCompositionId: projectData.activeCompositionId,
    openCompositionIds: projectData.openCompositionIds || [],
    expandedFolderIds: projectData.expandedFolderIds || [],
    slotAssignments: projectData.slotAssignments || {},
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

  // Restore view toggle states
  if (projectData.uiState) {
    const ui = projectData.uiState;
    const ts = useTimelineStore.getState();
    if (ui.thumbnailsEnabled !== undefined) ts.setThumbnailsEnabled(ui.thumbnailsEnabled);
    if (ui.waveformsEnabled !== undefined) ts.setWaveformsEnabled(ui.waveformsEnabled);
    if (ui.showTranscriptMarkers !== undefined) ts.setShowTranscriptMarkers(ui.showTranscriptMarkers);
    if (ui.proxyEnabled !== undefined) useMediaStore.getState().setProxyEnabled(ui.proxyEnabled);
  }

  // Reload API keys (may have been restored from .keys.enc during loadProject)
  await useSettingsStore.getState().loadApiKeys();

  log.info(' Loaded project to stores:', projectData.name);

  // Auto-relink missing files from Raw folder
  await autoRelinkFromRawFolder();

  // Restore thumbnails and refresh metadata in the background
  restoreMediaThumbnails();
  refreshMediaMetadata();
}

// ============================================
// BACKGROUND RESTORATION HELPERS
// ============================================

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

  // Scan the Raw folder - retry if empty (handle may not be ready yet)
  let rawFiles = await projectFileService.scanRawFolder();
  if (rawFiles.size === 0) {
    // Wait briefly and retry - the directory handle may need time on first load
    log.debug('Raw folder scan returned empty, retrying after delay...');
    await new Promise(resolve => setTimeout(resolve, 200));
    rawFiles = await projectFileService.scanRawFolder();
  }
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
      // Try with retries - file handle may need a moment to be ready
      let fileObj: File | undefined;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          fileObj = await handle.getFile();
          break; // Success
        } catch (e) {
          if (attempt < 2) {
            // Wait before retry
            await new Promise(resolve => setTimeout(resolve, 100 * (attempt + 1)));
          } else {
            log.warn(`Could not read file from Raw: ${file.name}`, e);
          }
        }
      }

      if (fileObj) {
        const url = URL.createObjectURL(fileObj);

        // Store handle for future access
        fileSystemService.storeFileHandle(file.id, handle);
        try {
          await projectDB.storeHandle(`media_${file.id}`, handle);
        } catch (e) {
          // IndexedDB may fail, but we can still use the file
          log.debug(`Could not store handle in IndexedDB: ${file.name}`);
        }

        // Update file entry
        updatedFiles[i] = {
          ...file,
          file: fileObj,
          url,
          hasFileHandle: true,
        };

        relinkedCount++;
        log.debug(`Auto-relinked from Raw: ${file.name}`);
      }
    } else {
      // Try to get from stored file handle in IndexedDB
      try {
        const storedHandle = await projectDB.getStoredHandle(`media_${file.id}`);
        if (storedHandle && storedHandle.kind === 'file') {
          const fileHandle = storedHandle as FileSystemFileHandle;
          const permission = await fileHandle.queryPermission({ mode: 'read' });

          if (permission === 'granted') {
            const fileObj = await fileHandle.getFile();
            const url = URL.createObjectURL(fileObj);

            fileSystemService.storeFileHandle(file.id, fileHandle);

            updatedFiles[i] = {
              ...file,
              file: fileObj,
              url,
              hasFileHandle: true,
            };

            relinkedCount++;
            log.debug(`Auto-relinked from IndexedDB handle: ${file.name}`);
          }
        }
      } catch (e) {
        // Silently ignore - will need manual reload
      }
    }
  }

  if (relinkedCount > 0) {
    // Update media store with relinked files
    useMediaStore.setState({ files: updatedFiles });
    log.info(`Auto-relinked ${relinkedCount}/${missingFiles.length} files from Raw folder`);

    // Small delay to allow state to settle before updating timeline clips
    await new Promise(resolve => setTimeout(resolve, 50));

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
        // Convert flat ProjectTransform → nested ClipTransform for nested clips too
        transform: nestedSerializedClip.transform ? (() => {
          const t = nestedSerializedClip.transform as any;
          return {
            position: t.position ?? { x: t.x ?? 0, y: t.y ?? 0, z: t.z ?? 0 },
            scale: t.scale ?? { x: t.scaleX ?? 1, y: t.scaleY ?? 1 },
            rotation: t.rotation != null && typeof t.rotation === 'object'
              ? t.rotation
              : { x: t.rotationX ?? 0, y: t.rotationY ?? 0, z: t.rotation ?? 0 },
            opacity: t.opacity ?? 1,
            blendMode: t.blendMode ?? 'normal',
          };
        })() : undefined,
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

          // Pre-cache frame via createImageBitmap for immediate scrubbing without play()
          engine.preCacheVideoFrame(video);
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

    }
  }

  log.info('Nested composition clips reloaded');
}
