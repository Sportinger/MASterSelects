// Media Panel Tool Handlers

import { useMediaStore } from '../../../stores/mediaStore';
import type { ToolResult } from '../types';

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

  const comp = mediaStore.createComposition(name, {
    width,
    height,
    frameRate,
    duration,
  });

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
