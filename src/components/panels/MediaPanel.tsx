// Media Panel - Project browser like After Effects

import React, { useCallback, useRef, useState, useEffect, memo } from 'react';
import { Logger } from '../../services/logger';

// Small file-type icons (AE style) - inline SVGs, 14px
const FileTypeIcon = memo(({ type }: { type?: string }) => {
  const size = 14;
  const style: React.CSSProperties = { width: size, height: size, flexShrink: 0, display: 'block' };

  switch (type) {
    case 'video':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="3" width="14" height="10" rx="1.5" fill="#4a6fa5" stroke="#6b9bd2" strokeWidth="0.7"/>
          <rect x="3" y="5" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="7" y="5" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="11" y="5" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="3" y="9" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="7" y="9" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
          <rect x="11" y="9" width="3" height="3" rx="0.5" fill="#2a4a75" opacity="0.7"/>
        </svg>
      );
    case 'audio':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#4a7a4a" stroke="#6aaa6a" strokeWidth="0.7"/>
          <path d="M4 6v4M6 5v6M8 4v8M10 5v6M12 6v4" stroke="#8fdf8f" strokeWidth="1.2" strokeLinecap="round"/>
        </svg>
      );
    case 'image':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#5a6a8a" stroke="#7a9aba" strokeWidth="0.7"/>
          <circle cx="5.5" cy="6" r="1.5" fill="#aaccee"/>
          <path d="M1.5 11l3.5-3 2.5 2 3-4 4 5v0.5c0 .55-.45 1-1 1h-12c-.55 0-1-.45-1-1z" fill="#7a9aba" opacity="0.8"/>
        </svg>
      );
    case 'composition':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#7a5a8a" stroke="#aa7abb" strokeWidth="0.7"/>
          <circle cx="8" cy="8" r="3.5" stroke="#cc99dd" strokeWidth="1" fill="none"/>
          <circle cx="8" cy="8" r="1" fill="#cc99dd"/>
        </svg>
      );
    case 'text':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#8a6a5a" stroke="#bb9a7a" strokeWidth="0.7"/>
          <text x="8" y="11.5" textAnchor="middle" fill="#eeddcc" fontSize="9" fontWeight="bold" fontFamily="sans-serif">T</text>
        </svg>
      );
    case 'solid':
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="12" rx="1.5" fill="#777" stroke="#999" strokeWidth="0.7"/>
          <rect x="4" y="5" width="8" height="6" rx="0.5" fill="#bbb"/>
        </svg>
      );
    default:
      return (
        <svg style={style} viewBox="0 0 16 16" fill="none">
          <path d="M4 1.5h5.5l4 4V14c0 .55-.45 1-1 1H4c-.55 0-1-.45-1-1V2.5c0-.55.45-1 1-1z" fill="#5a5a5a" stroke="#888" strokeWidth="0.7"/>
          <path d="M9.5 1.5v4h4" stroke="#888" strokeWidth="0.7" fill="#6a6a6a"/>
        </svg>
      );
  }
});

const log = Logger.create('MediaPanel');
import { useMediaStore } from '../../stores/mediaStore';
import type { MediaFile, Composition, ProjectItem } from '../../stores/mediaStore';
import type { LabelColor } from '../../stores/mediaStore/types';
import { useTimelineStore } from '../../stores/timeline';
import { useContextMenuPosition } from '../../hooks/useContextMenuPosition';
import { RelinkDialog } from '../common/RelinkDialog';

// AE label color palette
const LABEL_COLORS: { key: LabelColor; hex: string; name: string }[] = [
  { key: 'none', hex: 'transparent', name: 'None' },
  { key: 'red', hex: '#e2514c', name: 'Red' },
  { key: 'yellow', hex: '#dbb63b', name: 'Yellow' },
  { key: 'aqua', hex: '#4ec0c0', name: 'Aqua' },
  { key: 'pink', hex: '#d77bba', name: 'Pink' },
  { key: 'lavender', hex: '#a278c1', name: 'Lavender' },
  { key: 'peach', hex: '#e8a264', name: 'Peach' },
  { key: 'seafoam', hex: '#6bc488', name: 'Sea Foam' },
  { key: 'blue', hex: '#4a90e2', name: 'Blue' },
  { key: 'green', hex: '#6db849', name: 'Green' },
  { key: 'purple', hex: '#8b5fc7', name: 'Purple' },
  { key: 'orange', hex: '#e07934', name: 'Orange' },
  { key: 'brown', hex: '#a57249', name: 'Brown' },
  { key: 'fuchsia', hex: '#d14da1', name: 'Fuchsia' },
  { key: 'cyan', hex: '#49bce3', name: 'Cyan' },
  { key: 'tan', hex: '#c4a86c', name: 'Tan' },
];

function getLabelHex(color?: LabelColor): string {
  if (!color || color === 'none') return 'transparent';
  return LABEL_COLORS.find(c => c.key === color)?.hex || 'transparent';
}

// Column definitions
type ColumnId = 'label' | 'name' | 'duration' | 'resolution' | 'fps' | 'container' | 'codec' | 'audio' | 'bitrate' | 'size';

const COLUMN_LABELS_MAP: Record<ColumnId, string> = {
  label: '',
  name: 'Name',
  duration: 'Duration',
  resolution: 'Resolution',
  fps: 'FPS',
  container: 'Container',
  codec: 'Codec',
  audio: 'Audio',
  bitrate: 'Bitrate',
  size: 'Size',
};

const DEFAULT_COLUMN_ORDER: ColumnId[] = ['label', 'name', 'duration', 'resolution', 'fps', 'container', 'codec', 'audio', 'bitrate', 'size'];
const STORAGE_KEY = 'media-panel-column-order';

// Load column order from localStorage
function loadColumnOrder(): ColumnId[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as ColumnId[];
      // If all default columns are present and no extras, use saved order
      if (parsed.length === DEFAULT_COLUMN_ORDER.length &&
          DEFAULT_COLUMN_ORDER.every(col => parsed.includes(col))) {
        return parsed;
      }
      // If saved order is missing new columns, add them
      const missingColumns = DEFAULT_COLUMN_ORDER.filter(col => !parsed.includes(col));
      if (missingColumns.length > 0) {
        // Filter out any invalid columns and add missing ones
        const validColumns = parsed.filter(col => DEFAULT_COLUMN_ORDER.includes(col));
        // Ensure 'label' is always first
        const result = [...validColumns, ...missingColumns];
        const labelIdx = result.indexOf('label');
        if (labelIdx > 0) {
          result.splice(labelIdx, 1);
          result.unshift('label');
        }
        return result;
      }
    }
  } catch {
    // Ignore errors
  }
  return DEFAULT_COLUMN_ORDER;
}

export function MediaPanel() {
  // Reactive data - subscribe to specific values only
  const files = useMediaStore(state => state.files);
  const compositions = useMediaStore(state => state.compositions);
  const folders = useMediaStore(state => state.folders);
  const selectedIds = useMediaStore(state => state.selectedIds);
  const expandedFolderIds = useMediaStore(state => state.expandedFolderIds);
  const fileSystemSupported = useMediaStore(state => state.fileSystemSupported);
  const proxyFolderName = useMediaStore(state => state.proxyFolderName);
  const activeCompositionId = useMediaStore(state => state.activeCompositionId);

  // Actions from getState() - stable, no subscription needed
  const {
    importFiles,
    importFilesWithPicker,
    createComposition,
    createFolder,
    removeFile,
    removeComposition,
    removeFolder,
    renameFile,
    renameFolder,
    reloadFile,
    toggleFolderExpanded,
    setSelection,
    addToSelection,
    getItemsByFolder,
    openCompositionTab,
    updateComposition,
    generateProxy,
    cancelProxyGeneration,
    pickProxyFolder,
    showInExplorer,
    moveToFolder,
    createTextItem,
    getOrCreateTextFolder,
    createSolidItem,
    getOrCreateSolidFolder,
    setLabelColor,
  } = useMediaStore.getState();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; itemId?: string } | null>(null);
  const { menuRef: contextMenuRef, adjustedPosition: contextMenuPosition } = useContextMenuPosition(contextMenu);
  const [settingsDialog, setSettingsDialog] = useState<{ compositionId: string; width: number; height: number; frameRate: number; duration: number } | null>(null);
  const [addDropdownOpen, setAddDropdownOpen] = useState(false);
  const [dragOverFolderId, setDragOverFolderId] = useState<string | null>(null);
  const [internalDragId, setInternalDragId] = useState<string | null>(null);
  const [isExternalDragOver, setIsExternalDragOver] = useState(false);
  const [labelPickerItemId, setLabelPickerItemId] = useState<string | null>(null);
  const [labelPickerPos, setLabelPickerPos] = useState<{ x: number; y: number } | null>(null);

  // Column order state
  const [columnOrder, setColumnOrder] = useState<ColumnId[]>(loadColumnOrder);
  const [draggingColumn, setDraggingColumn] = useState<ColumnId | null>(null);
  const [dragOverColumn, setDragOverColumn] = useState<ColumnId | null>(null);

  // Save column order to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(columnOrder));
  }, [columnOrder]);

  // Column drag handlers
  const handleColumnDragStart = useCallback((e: React.DragEvent, columnId: ColumnId) => {
    e.stopPropagation();
    setDraggingColumn(columnId);
    e.dataTransfer.setData('application/x-column-id', columnId);
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleColumnDragOver = useCallback((e: React.DragEvent, columnId: ColumnId) => {
    e.preventDefault();
    e.stopPropagation();
    if (draggingColumn && draggingColumn !== columnId) {
      setDragOverColumn(columnId);
    }
  }, [draggingColumn]);

  const handleColumnDragLeave = useCallback(() => {
    setDragOverColumn(null);
  }, []);

  const handleColumnDrop = useCallback((e: React.DragEvent, targetColumnId: ColumnId) => {
    e.preventDefault();
    e.stopPropagation();
    const sourceColumnId = e.dataTransfer.getData('application/x-column-id') as ColumnId;
    if (sourceColumnId && sourceColumnId !== targetColumnId) {
      setColumnOrder(prev => {
        const newOrder = [...prev];
        const sourceIndex = newOrder.indexOf(sourceColumnId);
        const targetIndex = newOrder.indexOf(targetColumnId);
        newOrder.splice(sourceIndex, 1);
        newOrder.splice(targetIndex, 0, sourceColumnId);
        return newOrder;
      });
    }
    setDraggingColumn(null);
    setDragOverColumn(null);
  }, []);

  const handleColumnDragEnd = useCallback(() => {
    setDraggingColumn(null);
    setDragOverColumn(null);
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!addDropdownOpen) return;
    const handleClickOutside = () => setAddDropdownOpen(false);
    const timer = setTimeout(() => {
      document.addEventListener('click', handleClickOutside);
    }, 0);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('click', handleClickOutside);
    };
  }, [addDropdownOpen]);

  // Handle file import - prefer File System Access API for better file path access
  const handleImport = useCallback(async () => {
    if (fileSystemSupported) {
      // Use File System Access API - gives us file handles with path info
      await importFilesWithPicker();
    } else {
      // Fallback to traditional file input
      fileInputRef.current?.click();
    }
  }, [fileSystemSupported, importFilesWithPicker]);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await importFiles(e.target.files);
      e.target.value = ''; // Reset input
    }
  }, [importFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    // Check if this is an external file drag (from OS file explorer)
    const hasFiles = e.dataTransfer.types.includes('Files');
    const isInternalDrag = e.dataTransfer.types.includes('application/x-media-panel-item');

    log.debug('DragOver', { hasFiles, isInternalDrag, types: [...e.dataTransfer.types] });

    if (hasFiles && !isInternalDrag) {
      e.dataTransfer.dropEffect = 'copy';
      setIsExternalDragOver(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only reset if leaving the panel entirely (not just entering a child)
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
      setIsExternalDragOver(false);
    }
  }, []);

  // Handle item selection
  const handleItemClick = useCallback((id: string, e: React.MouseEvent) => {
    if (e.ctrlKey || e.metaKey) {
      addToSelection(id);
    } else if (e.shiftKey) {
      // TODO: Range selection
      addToSelection(id);
    } else {
      setSelection([id]);
    }
  }, [addToSelection, setSelection]);

  // Handle double-click (open/expand)
  const handleItemDoubleClick = useCallback(async (item: ProjectItem) => {
    if ('isExpanded' in item) {
      // It's a folder
      toggleFolderExpanded(item.id);
    } else if (item.type === 'composition') {
      // Open composition in timeline (as a tab)
      openCompositionTab(item.id);
    } else if ('file' in item && !item.file) {
      // Media file needs reload - request permission
      const success = await reloadFile(item.id);
      if (success) {
        log.info('File reloaded successfully');
      }
    }
  }, [toggleFolderExpanded, openCompositionTab, reloadFile]);

  // Context menu
  const handleContextMenu = useCallback((e: React.MouseEvent, itemId?: string) => {
    e.preventDefault();
    if (itemId && !selectedIds.includes(itemId)) {
      setSelection([itemId]);
    }
    setContextMenu({ x: e.clientX, y: e.clientY, itemId });
  }, [selectedIds, setSelection]);

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // Rename handling
  const startRename = useCallback((id: string, currentName: string) => {
    setRenamingId(id);
    setRenameValue(currentName);
    closeContextMenu();
  }, [closeContextMenu]);

  const finishRename = useCallback(() => {
    if (!renamingId || !renameValue.trim()) {
      setRenamingId(null);
      return;
    }

    const file = files.find(f => f.id === renamingId);
    const folder = folders.find(f => f.id === renamingId);
    const composition = compositions.find(c => c.id === renamingId);

    if (file) {
      renameFile(renamingId, renameValue.trim());
    } else if (folder) {
      renameFolder(renamingId, renameValue.trim());
    } else if (composition) {
      updateComposition(renamingId, { name: renameValue.trim() });
    }

    setRenamingId(null);
  }, [renamingId, renameValue, files, folders, compositions, renameFile, renameFolder, updateComposition]);

  // Handle click on item name to start rename
  const handleNameClick = useCallback((e: React.MouseEvent, id: string, currentName: string) => {
    e.stopPropagation();
    // Only start rename if item is already selected (double-click on name effect)
    if (selectedIds.includes(id)) {
      startRename(id, currentName);
    }
  }, [selectedIds, startRename]);

  // Delete selected items
  const handleDelete = useCallback(() => {
    selectedIds.forEach(id => {
      if (files.find(f => f.id === id)) removeFile(id);
      else if (compositions.find(c => c.id === id)) removeComposition(id);
      else if (folders.find(f => f.id === id)) removeFolder(id);
    });
    closeContextMenu();
  }, [selectedIds, files, compositions, folders, removeFile, removeComposition, removeFolder, closeContextMenu]);

  // New composition
  const handleNewComposition = useCallback(() => {
    createComposition(`Comp ${compositions.length + 1}`);
    closeContextMenu();
  }, [compositions.length, createComposition, closeContextMenu]);

  // New folder
  const handleNewFolder = useCallback(() => {
    createFolder(`New Folder`);
    closeContextMenu();
  }, [createFolder, closeContextMenu]);

  // New text item (in Media Panel, can be dragged to timeline)
  const handleNewText = useCallback(() => {
    const textFolderId = getOrCreateTextFolder();
    createTextItem(undefined, textFolderId);
    closeContextMenu();
  }, [createTextItem, getOrCreateTextFolder, closeContextMenu]);

  // New solid item (in Media Panel, can be dragged to timeline)
  const handleNewSolid = useCallback(() => {
    const solidFolderId = getOrCreateSolidFolder();
    createSolidItem(undefined, '#ffffff', solidFolderId);
    closeContextMenu();
  }, [createSolidItem, getOrCreateSolidFolder, closeContextMenu]);

  // Composition settings
  const openCompositionSettings = useCallback((comp: Composition) => {
    setSettingsDialog({
      compositionId: comp.id,
      width: comp.width,
      height: comp.height,
      frameRate: comp.frameRate,
      duration: comp.duration,
    });
    closeContextMenu();
  }, [closeContextMenu]);

  const saveCompositionSettings = useCallback(() => {
    if (!settingsDialog) return;
    updateComposition(settingsDialog.compositionId, {
      width: settingsDialog.width,
      height: settingsDialog.height,
      frameRate: settingsDialog.frameRate,
      duration: settingsDialog.duration,
    });
    // If this is the active composition, also update timeline duration
    if (settingsDialog.compositionId === activeCompositionId) {
      useTimelineStore.getState().setDuration(settingsDialog.duration);
    }
    setSettingsDialog(null);
  }, [settingsDialog, updateComposition, activeCompositionId]);

  // Handle drag start for media files and compositions (to drag to Timeline OR to folders)
  const handleDragStart = useCallback((e: React.DragEvent, item: ProjectItem) => {
    const isFolder = 'isExpanded' in item;

    // Mark as internal drag (for moving to folders)
    e.dataTransfer.setData('application/x-media-panel-item', item.id);
    setInternalDragId(item.id);

    // Don't set timeline data for folders
    if (isFolder) {
      e.dataTransfer.effectAllowed = 'move';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle composition drag
    if (item.type === 'composition') {
      const comp = item as Composition;
      // Don't allow dragging comp into itself (check active comp)
      if (comp.id === activeCompositionId) {
        e.preventDefault();
        return;
      }
      e.dataTransfer.setData('application/x-composition-id', comp.id);
      e.dataTransfer.effectAllowed = 'copyMove';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle text item drag
    if (item.type === 'text') {
      e.dataTransfer.setData('application/x-text-item-id', item.id);
      e.dataTransfer.effectAllowed = 'copyMove';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle solid item drag
    if (item.type === 'solid') {
      e.dataTransfer.setData('application/x-solid-item-id', item.id);
      e.dataTransfer.effectAllowed = 'copyMove';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle media file drag
    const mediaFile = item as MediaFile;
    if (!mediaFile.file) {
      // File not available - only allow internal move
      e.dataTransfer.effectAllowed = 'move';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Set the media file ID so Timeline can look it up
    e.dataTransfer.setData('application/x-media-file-id', mediaFile.id);
    e.dataTransfer.effectAllowed = 'copyMove';

    // Set drag image
    if (e.currentTarget instanceof HTMLElement) {
      e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
    }
  }, [activeCompositionId]);

  // Handle drag end (clear internal drag state)
  const handleDragEnd = useCallback(() => {
    setInternalDragId(null);
    setDragOverFolderId(null);
  }, []);

  // Handle drag over folder (for internal moves)
  const handleFolderDragOver = useCallback((e: React.DragEvent, folderId: string) => {
    // Only accept internal drags
    if (!e.dataTransfer.types.includes('application/x-media-panel-item')) {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'move';
    setDragOverFolderId(folderId);
  }, []);

  // Handle drag leave folder
  const handleFolderDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOverFolderId(null);
  }, []);

  // Handle drop on folder
  const handleFolderDrop = useCallback((e: React.DragEvent, folderId: string) => {
    e.preventDefault();
    e.stopPropagation();

    const itemId = e.dataTransfer.getData('application/x-media-panel-item');
    if (itemId && itemId !== folderId) {
      // Don't allow dropping a folder into itself or its children
      const draggedFolder = folders.find(f => f.id === itemId);
      if (draggedFolder) {
        // Check if target is a child of dragged folder (would create cycle)
        let parent = folders.find(f => f.id === folderId);
        while (parent) {
          if (parent.id === itemId) {
            // Would create cycle - abort
            setDragOverFolderId(null);
            setInternalDragId(null);
            return;
          }
          parent = folders.find(f => f.id === parent?.parentId);
        }
      }

      // Move item(s) to folder
      const itemsToMove = selectedIds.includes(itemId) ? selectedIds : [itemId];
      moveToFolder(itemsToMove, folderId);
    }

    setDragOverFolderId(null);
    setInternalDragId(null);
  }, [folders, selectedIds, moveToFolder]);

  // Handle drop on root (move out of folder or external file import)
  const handleRootDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsExternalDragOver(false);

    log.debug('Drop event', { types: [...e.dataTransfer.types], filesCount: e.dataTransfer.files.length });

    // Check if this is an external file drop
    if (!e.dataTransfer.types.includes('application/x-media-panel-item')) {
      // External file drop - try to get file handles for persistence
      const items = e.dataTransfer.items;
      if (items && items.length > 0) {
        const filesWithHandles: Array<{ file: File; handle: FileSystemFileHandle }> = [];
        const filesWithoutHandles: File[] = [];

        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          if (item.kind === 'file') {
            // Try to get file handle (File System Access API)
            if ('getAsFileSystemHandle' in item) {
              try {
                const handle = await (item as any).getAsFileSystemHandle();
                if (handle && handle.kind === 'file') {
                  const file = await handle.getFile();
                  filesWithHandles.push({ file, handle });
                  log.debug('Got file handle from drop', { name: file.name });
                }
              } catch {
                // Fallback to regular file
                const file = item.getAsFile();
                if (file) filesWithoutHandles.push(file);
              }
            } else {
              // Browser doesn't support getAsFileSystemHandle
              const file = item.getAsFile();
              if (file) filesWithoutHandles.push(file);
            }
          }
        }

        // Import files with handles using the store's method that saves handles
        if (filesWithHandles.length > 0) {
          log.info('Importing files WITH handles from drop', { count: filesWithHandles.length });
          const { importFilesWithHandles } = useMediaStore.getState();
          if (importFilesWithHandles) {
            await importFilesWithHandles(filesWithHandles);
          } else {
            // Fallback if method doesn't exist
            importFiles(filesWithHandles.map(f => f.file));
          }
        }

        // Import files without handles (old way)
        if (filesWithoutHandles.length > 0) {
          log.info('Importing files WITHOUT handles from drop', { count: filesWithoutHandles.length });
          importFiles(filesWithoutHandles);
        }
      }
      return;
    }

    // Internal drag - move to root
    const itemId = e.dataTransfer.getData('application/x-media-panel-item');
    if (itemId) {
      const itemsToMove = selectedIds.includes(itemId) ? selectedIds : [itemId];
      moveToFolder(itemsToMove, null); // null = root
    }

    setDragOverFolderId(null);
    setInternalDragId(null);
  }, [selectedIds, moveToFolder, importFiles]);

  // Format file size
  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return '–';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const formatBitrate = (bps?: number): string => {
    if (!bps) return '–';
    if (bps < 1000) return `${bps} bps`;
    if (bps < 1000 * 1000) return `${(bps / 1000).toFixed(0)} kbps`;
    return `${(bps / (1000 * 1000)).toFixed(1)} Mbps`;
  };

  // Name column width state (resizable)
  const [nameColumnWidth, setNameColumnWidth] = useState(() => {
    const stored = localStorage.getItem('media-panel-name-width');
    return stored ? parseInt(stored) : 250;
  });
  const resizeRef = useRef<{ startX: number; startWidth: number } | null>(null);

  // Save name column width
  useEffect(() => {
    localStorage.setItem('media-panel-name-width', String(nameColumnWidth));
  }, [nameColumnWidth]);

  // Resize handlers
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    resizeRef.current = { startX: e.clientX, startWidth: nameColumnWidth };

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (resizeRef.current) {
        const delta = moveEvent.clientX - resizeRef.current.startX;
        const newWidth = Math.max(120, Math.min(500, resizeRef.current.startWidth + delta));
        setNameColumnWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      resizeRef.current = null;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [nameColumnWidth]);

  // Render column content for an item
  const renderColumnContent = (
    colId: ColumnId,
    item: ProjectItem,
    depth: number,
    isFolder: boolean,
    isExpanded: boolean,
    isRenaming: boolean,
    isSelected: boolean,
    mediaFile: MediaFile | null
  ) => {
    switch (colId) {
      case 'label': {
        const labelColor = 'labelColor' in item ? (item as MediaFile).labelColor : undefined;
        const hex = getLabelHex(labelColor);
        return (
          <div
            className="media-col media-col-label"
            onClick={(e) => {
              e.stopPropagation();
              const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
              setLabelPickerItemId(item.id);
              setLabelPickerPos({ x: rect.left, y: rect.bottom + 2 });
            }}
          >
            <span
              className="media-label-dot"
              style={{
                background: hex === 'transparent' ? 'var(--border-color)' : hex,
                opacity: hex === 'transparent' ? 0.4 : 1,
              }}
            />
          </div>
        );
      }
      case 'name':
        return (
          <div
            className="media-col media-col-name"
            style={{ paddingLeft: `${4 + depth * 16}px`, width: nameColumnWidth, minWidth: nameColumnWidth, maxWidth: nameColumnWidth }}
          >
            {isFolder && (
              <span
                className={`media-folder-arrow ${isExpanded ? 'expanded' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleFolderExpanded(item.id);
                }}
              >
                ▶
              </span>
            )}
            <span className="media-item-icon">
              {isFolder
                ? <span className="media-folder-icon">&#128193;</span>
                : <FileTypeIcon type={'type' in item ? item.type : undefined} />
              }
            </span>
            {isRenaming ? (
              <input
                type="text"
                className="media-item-rename"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onBlur={finishRename}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') finishRename();
                  if (e.key === 'Escape') setRenamingId(null);
                }}
                autoFocus
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <span
                className={`media-item-name ${isSelected ? 'editable' : ''}`}
                onClick={(e) => handleNameClick(e, item.id, item.name)}
              >
                {item.name}
              </span>
            )}
            {'proxyStatus' in item && item.proxyStatus === 'ready' && (
              <span className="media-item-proxy-badge" title="Proxy generated">P</span>
            )}
            {'proxyStatus' in item && item.proxyStatus === 'generating' && (
              <span className="media-item-proxy-generating" title={`Generating proxy: ${(item as MediaFile).proxyProgress || 0}%`}>
                <span className="proxy-fill-badge">
                  <span className="proxy-fill-bg">P</span>
                  <span className="proxy-fill-progress" style={{ height: `${(item as MediaFile).proxyProgress || 0}%` }}>P</span>
                </span>
                <span className="proxy-percent">{(item as MediaFile).proxyProgress || 0}%</span>
              </span>
            )}
          </div>
        );
      case 'duration':
        return (
          <div className="media-col media-col-duration">
            {'duration' in item && item.duration ? formatDuration(item.duration) : '–'}
          </div>
        );
      case 'resolution':
        return (
          <div className="media-col media-col-resolution">
            {'width' in item && 'height' in item && item.width && item.height ? `${item.width}×${item.height}` : '–'}
          </div>
        );
      case 'fps':
        return (
          <div className="media-col media-col-fps">
            {mediaFile?.fps ? `${mediaFile.fps}` : ('type' in item && item.type === 'composition' ? (item as Composition).frameRate : '–')}
          </div>
        );
      case 'container':
        return <div className="media-col media-col-container">{mediaFile?.container || '–'}</div>;
      case 'codec':
        return <div className="media-col media-col-codec">{mediaFile?.codec || '–'}</div>;
      case 'audio':
        return <div className="media-col media-col-audio">
          {mediaFile?.type === 'audio' ? 'Yes' :
           mediaFile?.type === 'image' ? '–' :
           mediaFile?.hasAudio === true ? 'Yes' :
           mediaFile?.hasAudio === false ? 'No' : '–'}
        </div>;
      case 'bitrate':
        return <div className="media-col media-col-bitrate">{mediaFile?.bitrate ? formatBitrate(mediaFile.bitrate) : '–'}</div>;
      case 'size':
        return <div className="media-col media-col-size">{mediaFile ? formatFileSize(mediaFile.fileSize) : '–'}</div>;
      default:
        return null;
    }
  };

  // Render a single item
  const renderItem = (item: ProjectItem, depth: number = 0) => {
    const isFolder = 'isExpanded' in item;
    const isSelected = selectedIds.includes(item.id);
    const isRenaming = renamingId === item.id;
    const isExpanded = isFolder && expandedFolderIds.includes(item.id);
    const isMediaFile = !isFolder && 'type' in item && item.type !== 'composition' && item.type !== 'text' && item.type !== 'solid';
    const hasFile = isMediaFile && 'file' in item && !!(item as MediaFile).file;
    const canDrag = true;
    const isDragTarget = isFolder && dragOverFolderId === item.id;
    const isBeingDragged = internalDragId === item.id;
    const mediaFile = isMediaFile ? (item as MediaFile) : null;

    return (
      <div key={item.id}>
        <div
          className={`media-item ${isSelected ? 'selected' : ''} ${isFolder ? 'folder' : ''} ${isMediaFile && !hasFile ? 'no-file' : ''} ${isDragTarget ? 'drag-target' : ''} ${isBeingDragged ? 'dragging' : ''}`}
          draggable={canDrag}
          onDragStart={(e) => handleDragStart(e, item)}
          onDragEnd={handleDragEnd}
          onDragOver={isFolder ? (e) => handleFolderDragOver(e, item.id) : undefined}
          onDragLeave={isFolder ? handleFolderDragLeave : undefined}
          onDrop={isFolder ? (e) => handleFolderDrop(e, item.id) : undefined}
          onClick={(e) => handleItemClick(item.id, e)}
          onDoubleClick={() => handleItemDoubleClick(item)}
          onContextMenu={(e) => handleContextMenu(e, item.id)}
        >
          {columnOrder.map(colId => (
            <React.Fragment key={colId}>
              {renderColumnContent(colId, item, depth, isFolder, isExpanded, isRenaming, isSelected, mediaFile)}
            </React.Fragment>
          ))}
        </div>
        {isFolder && isExpanded && (
          <div className="media-folder-children">
            {getItemsByFolder(item.id).map(child => renderItem(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  // Get root items
  const rootItems = getItemsByFolder(null);
  const totalItems = files.length + compositions.length;

  // Check if any files need reload (lost permission after refresh)
  const filesNeedReload = files.some(f => !f.file);
  const filesNeedReloadCount = files.filter(f => !f.file).length;

  // Relink dialog state
  const [showRelinkDialog, setShowRelinkDialog] = useState(false);

  return (
    <div
      className={`media-panel ${isExternalDragOver ? 'drop-target' : ''}`}
      onDrop={handleRootDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => { if (contextMenu) closeContextMenu(); }}
    >
      {/* Header */}
      <div className="media-panel-header">
        <span className="media-panel-title">Project</span>
        <span className="media-panel-count">{totalItems} items</span>
        <div className="media-panel-actions">
          {filesNeedReload && (
            <button
              className="btn btn-sm btn-reload-all"
              onClick={() => setShowRelinkDialog(true)}
              title={`Restore access to ${filesNeedReloadCount} file${filesNeedReloadCount > 1 ? 's' : ''}`}
            >
              Relink ({filesNeedReloadCount})
            </button>
          )}
          <button className="btn btn-sm" onClick={handleImport} title="Import Media">
            Import
          </button>
          <div className="add-dropdown-container">
            <button
              className={`btn btn-sm add-dropdown-trigger ${addDropdownOpen ? 'active' : ''}`}
              onClick={() => setAddDropdownOpen(!addDropdownOpen)}
              title="Add New Item"
            >
              + Add ▾
            </button>
            {addDropdownOpen && (
              <div className="add-dropdown-menu">
                <div className="add-dropdown-item" onClick={() => { handleNewComposition(); setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon"><FileTypeIcon type="composition" /></span>
                  <span>Composition</span>
                </div>
                <div className="add-dropdown-item" onClick={() => { handleNewFolder(); setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon"><span className="media-folder-icon">&#128193;</span></span>
                  <span>Folder</span>
                </div>
                <div className="add-dropdown-separator" />
                <div className="add-dropdown-item" onClick={() => { handleNewText(); setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon"><FileTypeIcon type="text" /></span>
                  <span>Text</span>
                </div>
                <div className="add-dropdown-item" onClick={() => { handleNewSolid(); setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon"><FileTypeIcon type="solid" /></span>
                  <span>Solid</span>
                </div>
                <div className="add-dropdown-item" onClick={() => { /* TODO: Add adjustment layer */ setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon"><FileTypeIcon type="solid" /></span>
                  <span>Adjustment Layer</span>
                  <span className="add-dropdown-hint">Coming soon</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="video/*,audio/*,image/*"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* Item list with column headers */}
      <div className="media-panel-content">
        {rootItems.length === 0 ? (
          <div className="media-panel-empty">
            <div className="drop-icon">
              <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p>No media imported</p>
            <p className="hint">Drag & drop files here or click Import</p>
          </div>
        ) : (
          <div className="media-panel-table-wrapper">
            {/* Column headers */}
            <div className="media-column-headers">
              {columnOrder.map((colId) => (
                <div
                  key={colId}
                  className={`media-col media-col-${colId} ${draggingColumn === colId ? 'dragging' : ''} ${dragOverColumn === colId ? 'drag-over' : ''}`}
                  style={colId === 'name' ? { width: nameColumnWidth, minWidth: nameColumnWidth, maxWidth: nameColumnWidth } : undefined}
                  draggable={colId !== 'label'}
                  onDragStart={colId !== 'label' ? (e) => handleColumnDragStart(e, colId) : undefined}
                  onDragOver={colId !== 'label' ? (e) => handleColumnDragOver(e, colId) : undefined}
                  onDragLeave={colId !== 'label' ? handleColumnDragLeave : undefined}
                  onDrop={colId !== 'label' ? (e) => handleColumnDrop(e, colId) : undefined}
                  onDragEnd={colId !== 'label' ? handleColumnDragEnd : undefined}
                >
                  {COLUMN_LABELS_MAP[colId]}
                  {/* Resize handle after name column */}
                  {colId === 'name' && (
                    <div
                      className="media-col-resize-handle"
                      onMouseDown={handleResizeStart}
                      onClick={(e) => e.stopPropagation()}
                    />
                  )}
                </div>
              ))}
            </div>
            <div className="media-item-list">
              {rootItems.map(item => renderItem(item))}
            </div>
          </div>
        )}
      </div>

      {/* Drop overlay - shown when dragging files from outside */}
      {isExternalDragOver && (
        <div className="media-panel-drop-overlay">
          <div className="drop-overlay-content">
            <svg viewBox="0 0 24 24" width="64" height="64" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span>Drop files to import</span>
          </div>
        </div>
      )}

      {/* Context Menu */}
      {contextMenu && (() => {
        const selectedItem = contextMenu.itemId
          ? files.find(f => f.id === contextMenu.itemId) ||
            compositions.find(c => c.id === contextMenu.itemId) ||
            folders.find(f => f.id === contextMenu.itemId)
          : null;
        const isVideoFile = selectedItem && 'type' in selectedItem && selectedItem.type === 'video';
        const isComposition = selectedItem && 'type' in selectedItem && selectedItem.type === 'composition';
        const mediaFile = isVideoFile ? (selectedItem as MediaFile) : null;
        const composition = isComposition ? (selectedItem as Composition) : null;
        const isGenerating = mediaFile?.proxyStatus === 'generating';
        const hasProxy = mediaFile?.proxyStatus === 'ready';

        return (
          <div
            ref={contextMenuRef}
            className="media-context-menu"
            style={{
              position: 'fixed',
              left: contextMenuPosition?.x ?? contextMenu.x,
              top: contextMenuPosition?.y ?? contextMenu.y,
              zIndex: 10000,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="context-menu-item" onClick={handleImport}>
              Import Media...
            </div>
            <div className="context-menu-item" onClick={handleNewComposition}>
              New Composition
            </div>
            <div className="context-menu-item" onClick={handleNewFolder}>
              New Folder
            </div>
            {contextMenu.itemId && (
              <>
                <div className="context-menu-separator" />
                <div className="context-menu-item" onClick={() => {
                  if (selectedItem) startRename(selectedItem.id, selectedItem.name);
                }}>
                  Rename
                </div>

                {/* Composition Settings - only for compositions */}
                {isComposition && composition && (
                  <div className="context-menu-item" onClick={() => openCompositionSettings(composition)}>
                    Composition Settings...
                  </div>
                )}

                {/* Proxy Generation - only for video files */}
                {isVideoFile && mediaFile && (
                  <>
                    <div className="context-menu-separator" />
                    {isGenerating ? (
                      <div
                        className="context-menu-item"
                        onClick={() => {
                          cancelProxyGeneration(mediaFile.id);
                          closeContextMenu();
                        }}
                      >
                        Stop Proxy Generation ({mediaFile.proxyProgress || 0}%)
                      </div>
                    ) : hasProxy ? (
                      <div className="context-menu-item disabled">
                        Proxy Ready
                      </div>
                    ) : (
                      <div
                        className="context-menu-item"
                        onClick={() => {
                          generateProxy(mediaFile.id);
                          closeContextMenu();
                        }}
                      >
                        Generate Proxy
                      </div>
                    )}
                  </>
                )}

                {/* Show in Explorer submenu - only for video files with file data */}
                {isVideoFile && mediaFile?.file && (
                  <div className="context-menu-item has-submenu">
                    <span>Show in Explorer</span>
                    <span className="submenu-arrow">▶</span>
                    <div className="context-submenu">
                      <div
                        className="context-menu-item"
                        onClick={async () => {
                          const result = await showInExplorer('raw', mediaFile.id);
                          if (result.success) {
                            alert(result.message);
                          } else {
                            // Fallback: download the file
                            if (mediaFile.file) {
                              const url = URL.createObjectURL(mediaFile.file);
                              const a = document.createElement('a');
                              a.href = url;
                              a.download = mediaFile.name;
                              document.body.appendChild(a);
                              a.click();
                              document.body.removeChild(a);
                              URL.revokeObjectURL(url);
                            }
                          }
                          closeContextMenu();
                        }}
                      >
                        Raw {mediaFile.hasFileHandle && '(has path)'}
                      </div>
                      <div
                        className={`context-menu-item ${!hasProxy ? 'disabled' : ''}`}
                        onClick={async () => {
                          if (hasProxy) {
                            const result = await showInExplorer('proxy', mediaFile.id);
                            alert(result.message);
                          }
                          closeContextMenu();
                        }}
                      >
                        Proxy {!hasProxy ? '(not available)' : proxyFolderName ? `(${proxyFolderName})` : '(IndexedDB)'}
                      </div>
                    </div>
                  </div>
                )}

                {/* Set Proxy Folder - for video files */}
                {isVideoFile && (
                  <div
                    className="context-menu-item"
                    onClick={async () => {
                      await pickProxyFolder();
                      closeContextMenu();
                    }}
                  >
                    Set Proxy Folder... {proxyFolderName && `(${proxyFolderName})`}
                  </div>
                )}

                <div className="context-menu-separator" />
                <div className="context-menu-item danger" onClick={handleDelete}>
                  Delete
                </div>
              </>
            )}
          </div>
        );
      })()}

      {/* Composition Settings Dialog - Clean, no blur */}
      {settingsDialog && (
        <div
          className="comp-settings-overlay"
          onClick={() => setSettingsDialog(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'transparent',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10001,
          }}
        >
          <div
            className="comp-settings-dialog"
            onClick={(e) => e.stopPropagation()}
            style={{
              background: '#1e1e1e',
              border: '1px solid #3a3a3a',
              borderRadius: '6px',
              padding: '20px',
              minWidth: '340px',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
            }}
          >
            <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: 500, color: '#e0e0e0' }}>Composition Settings</h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              {/* Width */}
              <div>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '11px', color: '#888' }}>Width</label>
                <input
                  type="number"
                  value={settingsDialog.width}
                  onChange={(e) => setSettingsDialog({
                    ...settingsDialog,
                    width: Math.max(1, parseInt(e.target.value) || 1920),
                  })}
                  min="1"
                  max="7680"
                  style={{ width: '100%', padding: '6px 8px', background: '#2a2a2a', border: '1px solid #3a3a3a', borderRadius: '4px', color: '#fff', fontSize: '13px' }}
                />
              </div>

              {/* Height */}
              <div>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '11px', color: '#888' }}>Height</label>
                <input
                  type="number"
                  value={settingsDialog.height}
                  onChange={(e) => setSettingsDialog({
                    ...settingsDialog,
                    height: Math.max(1, parseInt(e.target.value) || 1080),
                  })}
                  min="1"
                  max="4320"
                  style={{ width: '100%', padding: '6px 8px', background: '#2a2a2a', border: '1px solid #3a3a3a', borderRadius: '4px', color: '#fff', fontSize: '13px' }}
                />
              </div>

              {/* Frame Rate */}
              <div>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '11px', color: '#888' }}>Frame Rate</label>
                <select
                  value={settingsDialog.frameRate}
                  onChange={(e) => setSettingsDialog({
                    ...settingsDialog,
                    frameRate: Number(e.target.value),
                  })}
                  style={{ width: '100%', padding: '6px 8px', background: '#2a2a2a', border: '1px solid #3a3a3a', borderRadius: '4px', color: '#fff', fontSize: '13px' }}
                >
                  <option value={23.976}>23.976 fps</option>
                  <option value={24}>24 fps</option>
                  <option value={25}>25 fps (PAL)</option>
                  <option value={29.97}>29.97 fps (NTSC)</option>
                  <option value={30}>30 fps</option>
                  <option value={50}>50 fps</option>
                  <option value={59.94}>59.94 fps</option>
                  <option value={60}>60 fps</option>
                </select>
              </div>

              {/* Duration */}
              <div>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '11px', color: '#888' }}>Duration (sec)</label>
                <input
                  type="number"
                  value={settingsDialog.duration}
                  onChange={(e) => setSettingsDialog({
                    ...settingsDialog,
                    duration: Math.max(1, parseFloat(e.target.value) || 60),
                  })}
                  min="1"
                  max="86400"
                  step="1"
                  style={{ width: '100%', padding: '6px 8px', background: '#2a2a2a', border: '1px solid #3a3a3a', borderRadius: '4px', color: '#fff', fontSize: '13px' }}
                />
              </div>
            </div>

            {/* Resolution Presets */}
            <div style={{ marginTop: '12px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '11px', color: '#888' }}>Presets</label>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                {[
                  { label: '1080p', w: 1920, h: 1080 },
                  { label: '4K', w: 3840, h: 2160 },
                  { label: '720p', w: 1280, h: 720 },
                  { label: '9:16', w: 1080, h: 1920 },
                  { label: '1:1', w: 1080, h: 1080 },
                ].map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => setSettingsDialog({ ...settingsDialog, width: preset.w, height: preset.h })}
                    style={{
                      padding: '4px 8px',
                      fontSize: '11px',
                      background: settingsDialog.width === preset.w && settingsDialog.height === preset.h ? '#4a90e2' : '#2a2a2a',
                      border: '1px solid #3a3a3a',
                      borderRadius: '3px',
                      color: '#fff',
                      cursor: 'pointer',
                    }}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Buttons */}
            <div style={{ display: 'flex', gap: '8px', marginTop: '16px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setSettingsDialog(null)}
                style={{ padding: '6px 16px', background: '#2a2a2a', border: '1px solid #3a3a3a', borderRadius: '4px', color: '#fff', cursor: 'pointer', fontSize: '12px' }}
              >
                Cancel
              </button>
              <button
                onClick={saveCompositionSettings}
                style={{ padding: '6px 16px', background: '#4a90e2', border: 'none', borderRadius: '4px', color: '#fff', cursor: 'pointer', fontSize: '12px' }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Label Color Picker */}
      {labelPickerItemId && labelPickerPos && (
        <>
          <div
            className="label-picker-backdrop"
            onClick={() => { setLabelPickerItemId(null); setLabelPickerPos(null); }}
          />
          <div
            className="label-picker-popup"
            style={{ position: 'fixed', left: labelPickerPos.x, top: labelPickerPos.y, zIndex: 10002 }}
          >
            {LABEL_COLORS.map(c => (
              <span
                key={c.key}
                className={`label-picker-swatch ${c.key === 'none' ? 'none' : ''}`}
                title={c.name}
                style={{ background: c.key === 'none' ? 'var(--bg-tertiary)' : c.hex }}
                onClick={() => {
                  const ids = selectedIds.includes(labelPickerItemId) ? selectedIds : [labelPickerItemId];
                  setLabelColor(ids, c.key);
                  setLabelPickerItemId(null);
                  setLabelPickerPos(null);
                }}
              >
                {c.key === 'none' && <span className="label-picker-x">&times;</span>}
              </span>
            ))}
          </div>
        </>
      )}

      {/* Relink Dialog */}
      {showRelinkDialog && (
        <RelinkDialog onClose={() => setShowRelinkDialog(false)} />
      )}
    </div>
  );
}

// Format duration as mm:ss
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
