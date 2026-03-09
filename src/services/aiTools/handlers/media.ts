// Media Panel Tool Handlers

import { useMediaStore } from '../../../stores/mediaStore';
import { useTimelineStore } from '../../../stores/timeline';
import type { ToolResult } from '../types';
import { Logger } from '../../logger';
import { activateDockPanel, flashPreviewCanvas } from '../aiFeedback';

const log = Logger.create('AITool:Media');

type MediaStore = ReturnType<typeof useMediaStore.getState>;

export async function handleGetMediaItems(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const folderId = (args.folderId as string | undefined) || null;
  const { files, compositions, folders } = mediaStore;

  // Filter by folder
  const folderFiles = files.filter(f => f.parentId === folderId);
  const folderComps = compositions.filter(c => c.parentId === folderId);
  const subFolders = folders.filter(f => f.parentId === folderId);

  return {
    success: true,
    data: {
      folderId: folderId || 'root',
      folders: subFolders.map(f => ({
        id: f.id,
        name: f.name,
        type: 'folder',
        isExpanded: f.isExpanded,
      })),
      files: folderFiles.map(f => ({
        id: f.id,
        name: f.name,
        type: f.type,
        duration: f.duration,
        width: f.width,
        height: f.height,
      })),
      compositions: folderComps.map(c => ({
        id: c.id,
        name: c.name,
        type: 'composition',
        width: c.width,
        height: c.height,
        duration: c.duration,
        frameRate: c.frameRate,
      })),
      totalItems: subFolders.length + folderFiles.length + folderComps.length,
      // Also include all folders for reference
      allFolders: folders.map(f => ({ id: f.id, name: f.name, parentId: f.parentId })),
    },
  };
}

export async function handleCreateMediaFolder(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const name = args.name as string;
  const parentFolderId = (args.parentFolderId as string | undefined) || null;

  const folder = mediaStore.createFolder(name, parentFolderId);

  return {
    success: true,
    data: {
      folderId: folder.id,
      folderName: folder.name,
      parentId: parentFolderId,
    },
  };
}

export async function handleRenameMediaItem(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const itemId = args.itemId as string;
  const newName = args.newName as string;

  // Try to find the item in files, compositions, or folders
  const file = mediaStore.files.find(f => f.id === itemId);
  const comp = mediaStore.compositions.find(c => c.id === itemId);
  const folder = mediaStore.folders.find(f => f.id === itemId);

  if (file) {
    mediaStore.renameFile(itemId, newName);
    return { success: true, data: { itemId, newName, type: 'file' } };
  } else if (comp) {
    mediaStore.updateComposition(itemId, { name: newName });
    return { success: true, data: { itemId, newName, type: 'composition' } };
  } else if (folder) {
    mediaStore.renameFolder(itemId, newName);
    return { success: true, data: { itemId, newName, type: 'folder' } };
  }

  return { success: false, error: `Item not found: ${itemId}` };
}

export async function handleDeleteMediaItem(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const itemId = args.itemId as string;

  const file = mediaStore.files.find(f => f.id === itemId);
  const comp = mediaStore.compositions.find(c => c.id === itemId);
  const folder = mediaStore.folders.find(f => f.id === itemId);

  if (file) {
    mediaStore.removeFile(itemId);
    return { success: true, data: { itemId, deletedName: file.name, type: 'file' } };
  } else if (comp) {
    mediaStore.removeComposition(itemId);
    return { success: true, data: { itemId, deletedName: comp.name, type: 'composition' } };
  } else if (folder) {
    mediaStore.removeFolder(itemId);
    return { success: true, data: { itemId, deletedName: folder.name, type: 'folder', note: 'All contents also deleted' } };
  }

  return { success: false, error: `Item not found: ${itemId}` };
}

export async function handleMoveMediaItems(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const itemIds = args.itemIds as string[];
  const targetFolderId = (args.targetFolderId as string | undefined) || null;

  // Verify target folder exists (if not root)
  if (targetFolderId !== null) {
    const targetFolder = mediaStore.folders.find(f => f.id === targetFolderId);
    if (!targetFolder) {
      return { success: false, error: `Target folder not found: ${targetFolderId}` };
    }
  }

  mediaStore.moveToFolder(itemIds, targetFolderId);

  return {
    success: true,
    data: {
      movedIds: itemIds,
      targetFolderId: targetFolderId || 'root',
      itemCount: itemIds.length,
    },
  };
}

export async function handleCreateComposition(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const name = args.name as string;
  const width = (args.width as number) || 1920;
  const height = (args.height as number) || 1080;
  const frameRate = (args.frameRate as number) || 30;
  const duration = (args.duration as number) || 60;
  const openAfterCreate = args.openAfterCreate !== false; // default true

  const comp = mediaStore.createComposition(name, {
    width,
    height,
    frameRate,
    duration,
  });

  // Auto-open so subsequent operations target this composition
  if (openAfterCreate) {
    mediaStore.openCompositionTab(comp.id);
  }

  return {
    success: true,
    data: {
      compositionId: comp.id,
      name: comp.name,
      width: comp.width,
      height: comp.height,
      frameRate: comp.frameRate,
      duration: comp.duration,
      opened: openAfterCreate,
    },
  };
}

export async function handleOpenComposition(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const compositionId = args.compositionId as string;

  const comp = mediaStore.compositions.find(c => c.id === compositionId);
  if (!comp) {
    return { success: false, error: `Composition not found: ${compositionId}` };
  }

  mediaStore.openCompositionTab(compositionId);

  return {
    success: true,
    data: {
      compositionId: comp.id,
      name: comp.name,
      width: comp.width,
      height: comp.height,
      frameRate: comp.frameRate,
      duration: comp.duration,
    },
  };
}

export async function handleSelectMediaItems(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const itemIds = args.itemIds as string[];
  mediaStore.setSelection(itemIds);
  return {
    success: true,
    data: { selectedIds: itemIds, count: itemIds.length },
  };
}

export async function handleImportLocalFiles(
  args: Record<string, unknown>,
  mediaStore: MediaStore
): Promise<ToolResult> {
  const paths = args.paths as string[];
  const addToTimeline = (args.addToTimeline as boolean) || false;

  // Visual feedback: activate media panel during import
  activateDockPanel('media');
  flashPreviewCanvas('import');

  const results: Array<{ id: string; name: string; type: string; duration?: number; path: string }> = [];
  const errors: Array<{ path: string; error: string }> = [];

  for (const filePath of paths) {
    try {
      const normalizedPath = filePath.replace(/\\/g, '/');
      const encodedPath = encodeURIComponent(normalizedPath);
      log.info(`Fetching: ${normalizedPath}`);

      const response = await fetch(`/api/local-file?path=${encodedPath}`);
      if (!response.ok) {
        errors.push({ path: filePath, error: `HTTP ${response.status}: ${response.statusText}` });
        continue;
      }

      const blob = await response.blob();
      const fileName = normalizedPath.split('/').pop() || 'unknown';
      const file = new File([blob], fileName, { type: blob.type });

      const mediaFile = await mediaStore.importFile(file);
      results.push({
        id: mediaFile.id,
        name: mediaFile.name,
        type: mediaFile.type,
        duration: mediaFile.duration,
        path: filePath,
      });
      log.info(`Imported: ${mediaFile.name} (${mediaFile.type})`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      log.error(`Failed to import: ${filePath}`, err);
      errors.push({ path: filePath, error: msg });
    }
  }

  // Optionally add to timeline
  if (addToTimeline && results.length > 0) {
    const timelineStore = useTimelineStore.getState();
    const requestedTrackId = args.trackId as string | undefined;
    const createTrack = (args.createTrack as boolean) || false;
    const trackType = (args.trackType as 'video' | 'audio') || 'video';
    const requestedStartTime = args.startTime as number | undefined;
    const sequential = args.sequential !== false; // default true

    let targetTrackId: string | null = null;

    // 1. Create new track if requested
    if (createTrack) {
      targetTrackId = timelineStore.addTrack(trackType);
      log.info(`Created new ${trackType} track: ${targetTrackId}`);
    }
    // 2. Use specified track
    else if (requestedTrackId) {
      const track = useTimelineStore.getState().tracks.find(t => t.id === requestedTrackId);
      if (track) {
        targetTrackId = requestedTrackId;
      } else {
        log.warn(`Track ${requestedTrackId} not found, falling back to first track`);
      }
    }

    // 3. Fallback: first matching track
    if (!targetTrackId) {
      const matchingTracks = useTimelineStore.getState().tracks.filter(t => t.type === trackType);
      targetTrackId = matchingTracks.length > 0 ? matchingTracks[0].id : null;
    }

    // 4. Last resort: create one
    if (!targetTrackId) {
      targetTrackId = useTimelineStore.getState().addTrack(trackType);
      log.info(`Auto-created ${trackType} track: ${targetTrackId}`);
    }

    // Determine start time
    let currentTime: number;
    if (requestedStartTime !== undefined) {
      currentTime = requestedStartTime;
    } else {
      // Append after last clip on this track
      const existingClips = useTimelineStore.getState().clips.filter(c => c.trackId === targetTrackId);
      currentTime = existingClips.length > 0
        ? Math.max(...existingClips.map(c => c.startTime + c.duration))
        : 0;
    }

    const placedClips: Array<{ name: string; trackId: string; startTime: number }> = [];
    for (const result of results) {
      const mediaFile = useMediaStore.getState().files.find(f => f.id === result.id);
      if (mediaFile && mediaFile.file) {
        await useTimelineStore.getState().addClip(targetTrackId!, mediaFile.file, currentTime, mediaFile.duration, mediaFile.id);
        placedClips.push({ name: result.name, trackId: targetTrackId!, startTime: currentTime });
        if (sequential) {
          currentTime += mediaFile.duration || 5;
        }
      }
    }
    return {
      success: errors.length === 0,
      data: {
        imported: results,
        errors: errors.length > 0 ? errors : undefined,
        totalImported: results.length,
        totalFailed: errors.length,
        placedClips,
        trackId: targetTrackId,
      },
    };
  }

  return {
    success: errors.length === 0,
    data: {
      imported: results,
      errors: errors.length > 0 ? errors : undefined,
      totalImported: results.length,
      totalFailed: errors.length,
    },
  };
}

export async function handleListLocalFiles(
  args: Record<string, unknown>,
): Promise<ToolResult> {
  const directory = args.directory as string;
  const extensions = args.extensions as string | undefined;

  try {
    const normalizedDir = directory.replace(/\\/g, '/');
    let url = `/api/local-files?dir=${encodeURIComponent(normalizedDir)}`;
    if (extensions) {
      url += `&ext=${encodeURIComponent(extensions)}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      const body = await response.json().catch(() => ({ error: response.statusText }));
      return { success: false, error: body.error || `HTTP ${response.status}` };
    }

    const data = await response.json();
    return {
      success: true,
      data: {
        directory: normalizedDir,
        files: data.files,
        totalFiles: data.files.length,
      },
    };
  } catch (err) {
    return { success: false, error: err instanceof Error ? err.message : 'Unknown error' };
  }
}
