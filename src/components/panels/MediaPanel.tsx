// Media Panel - Project browser like After Effects

import React, { useCallback, useRef, useState, useEffect } from 'react';
import { Logger } from '../../services/logger';
import { FileTypeIcon } from './media/FileTypeIcon';
import { LABEL_COLORS, getLabelHex } from './media/labelColors';
import { CompositionSettingsDialog } from './media/CompositionSettingsDialog';
import { SolidSettingsDialog } from './media/SolidSettingsDialog';
import { LabelColorPicker } from './media/LabelColorPicker';

const log = Logger.create('MediaPanel');
import { useMediaStore } from '../../stores/mediaStore';
import type { MediaFile, Composition, ProjectItem, SolidItem } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { useDockStore } from '../../stores/dockStore';
import { useContextMenuPosition } from '../../hooks/useContextMenuPosition';
import { RelinkDialog } from '../common/RelinkDialog';
import {
  clearExternalDragPayload,
  setExternalDragPayload,
} from '../timeline/utils/externalDragSession';

// Re-export for backward compatibility
export { LABEL_COLORS, getLabelHex } from './media/labelColors';

// Column definitions
type ColumnId = 'label' | 'name' | 'duration' | 'resolution' | 'fps' | 'container' | 'codec' | 'audio' | 'bitrate' | 'size';

const COLUMN_LABELS_MAP: Record<ColumnId, string> = {
  label: '●',
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

const DEFAULT_COLUMN_ORDER: ColumnId[] = ['name', 'label', 'duration', 'resolution', 'fps', 'container', 'codec', 'audio', 'bitrate', 'size'];
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
        return [...validColumns, ...missingColumns];
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
  const solidItems = useMediaStore(state => state.solidItems);
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
    updateSolidItem,
    setLabelColor,
  } = useMediaStore.getState();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const itemListRef = useRef<HTMLDivElement>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; itemId?: string } | null>(null);

  // Marquee selection state
  const [marquee, setMarquee] = useState<{ startX: number; startY: number; currentX: number; currentY: number } | null>(null);
  const marqueeRef = useRef<{ startX: number; startY: number; initialSelection: string[] } | null>(null);
  const { menuRef: contextMenuRef, adjustedPosition: contextMenuPosition } = useContextMenuPosition(contextMenu);
  const [settingsDialog, setSettingsDialog] = useState<{ compositionId: string; width: number; height: number; frameRate: number; duration: number } | null>(null);
  const [solidSettingsDialog, setSolidSettingsDialog] = useState<{ solidItemId: string; width: number; height: number; color: string } | null>(null);
  const [addDropdownOpen, setAddDropdownOpen] = useState(false);
  const [dragOverFolderId, setDragOverFolderId] = useState<string | null>(null);
  const [internalDragId, setInternalDragId] = useState<string | null>(null);
  const [isExternalDragOver, setIsExternalDragOver] = useState(false);
  const [labelPickerItemId, setLabelPickerItemId] = useState<string | null>(null);
  const [labelPickerPos, setLabelPickerPos] = useState<{ x: number; y: number } | null>(null);
  const [viewMode, setViewMode] = useState<'list' | 'grid'>(() => {
    return (localStorage.getItem('media-panel-view-mode') as 'list' | 'grid') || 'list';
  });
  // Grid view: current open folder (null = root)
  const [gridFolderId, setGridFolderId] = useState<string | null>(null);

  // Column order state
  const [columnOrder, setColumnOrder] = useState<ColumnId[]>(loadColumnOrder);
  const [draggingColumn, setDraggingColumn] = useState<ColumnId | null>(null);
  const [dragOverColumn, setDragOverColumn] = useState<ColumnId | null>(null);

  // Sort state
  const [sortColumn, setSortColumn] = useState<ColumnId | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

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

  // Sort handler - click on column header to sort
  const handleColumnSort = useCallback((colId: ColumnId) => {
    if (sortColumn === colId) {
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else {
        // Third click: remove sort
        setSortColumn(null);
        setSortDirection('asc');
      }
    } else {
      setSortColumn(colId);
      setSortDirection('asc');
    }
  }, [sortColumn, sortDirection]);

  // Sort items comparator
  const getSortValue = useCallback((item: ProjectItem, colId: ColumnId): string | number => {
    const mediaFile = ('type' in item && item.type !== 'composition' && item.type !== 'text' && item.type !== 'solid') ? item as MediaFile : null;
    switch (colId) {
      case 'name': return item.name.toLowerCase();
      case 'label': {
        const labelColor = 'labelColor' in item ? (item as MediaFile).labelColor : undefined;
        const idx = LABEL_COLORS.findIndex(c => c.key === (labelColor || 'none'));
        return idx >= 0 ? idx : 999;
      }
      case 'duration': return 'duration' in item && item.duration ? item.duration : 0;
      case 'resolution': return 'width' in item && 'height' in item && item.width && item.height ? item.width * item.height : 0;
      case 'fps': return mediaFile?.fps || ('type' in item && item.type === 'composition' ? (item as Composition).frameRate : 0);
      case 'container': return mediaFile?.container?.toLowerCase() || '';
      case 'codec': return mediaFile?.codec?.toLowerCase() || '';
      case 'audio': return mediaFile?.hasAudio ? 1 : 0;
      case 'bitrate': return mediaFile?.bitrate || 0;
      case 'size': return mediaFile?.fileSize || 0;
      default: return 0;
    }
  }, []);

  const sortItems = useCallback((items: ProjectItem[]): ProjectItem[] => {
    if (!sortColumn) return items;
    // Separate folders from other items - folders stay at top
    const folderItems = items.filter(i => 'isExpanded' in i);
    const nonFolderItems = items.filter(i => !('isExpanded' in i));

    const compare = (a: ProjectItem, b: ProjectItem): number => {
      const va = getSortValue(a, sortColumn);
      const vb = getSortValue(b, sortColumn);
      let result: number;
      if (typeof va === 'string' && typeof vb === 'string') {
        result = va.localeCompare(vb);
      } else {
        result = (va as number) - (vb as number);
      }
      return sortDirection === 'desc' ? -result : result;
    };

    folderItems.sort(compare);
    nonFolderItems.sort(compare);
    return [...folderItems, ...nonFolderItems];
  }, [sortColumn, sortDirection, getSortValue]);

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

  // Marquee selection handlers
  const handleMarqueeMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    // Ignore clicks on buttons, inputs, context menus
    if (target.closest('button, input, .context-menu')) return;

    // Don't start marquee when clicking on an item — let item drag handle it
    const clickedOnItem = !!target.closest('.media-item, .media-grid-item');
    if (clickedOnItem) return;

    const container = itemListRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const startX = e.clientX - rect.left + container.scrollLeft;
    const startY = e.clientY - rect.top + container.scrollTop;
    const clientStartX = e.clientX;
    const clientStartY = e.clientY;

    const initial = e.ctrlKey || e.metaKey ? [...selectedIds] : [];
    let isDragging = false;

    const handleMouseMove = (ev: MouseEvent) => {
      const dx = ev.clientX - clientStartX;
      const dy = ev.clientY - clientStartY;

      // Start marquee after 4px movement threshold
      if (!isDragging && (Math.abs(dx) > 4 || Math.abs(dy) > 4)) {
        isDragging = true;
        marqueeRef.current = { startX, startY, initialSelection: initial };
        if (!ev.ctrlKey && !ev.metaKey) {
          setSelection([]);
        }
      }

      if (!isDragging || !marqueeRef.current) return;

      const r = container.getBoundingClientRect();
      const cx = ev.clientX - r.left + container.scrollLeft;
      const cy = ev.clientY - r.top + container.scrollTop;
      setMarquee({ startX: marqueeRef.current.startX, startY: marqueeRef.current.startY, currentX: cx, currentY: cy });

      // Hit-test items
      const mLeft = Math.min(marqueeRef.current.startX, cx);
      const mRight = Math.max(marqueeRef.current.startX, cx);
      const mTop = Math.min(marqueeRef.current.startY, cy);
      const mBottom = Math.max(marqueeRef.current.startY, cy);

      const itemEls = container.querySelectorAll('.media-item, .media-grid-item');
      const hitIds: string[] = [];
      itemEls.forEach((el) => {
        const elRect = el.getBoundingClientRect();
        const elTop = elRect.top - r.top + container.scrollTop;
        const elBottom = elTop + elRect.height;
        const elLeft = elRect.left - r.left + container.scrollLeft;
        const elRight = elLeft + elRect.width;
        if (elRight > mLeft && elLeft < mRight && elBottom > mTop && elTop < mBottom) {
          const itemId = el.parentElement?.getAttribute('data-item-id');
          if (itemId) hitIds.push(itemId);
        }
      });

      const combined = [...new Set([...marqueeRef.current.initialSelection, ...hitIds])];
      setSelection(combined);
    };

    const handleMouseUp = () => {
      if (!isDragging) {
        // Clicked on empty space without dragging → deselect all
        if (!e.ctrlKey && !e.metaKey) {
          setSelection([]);
        }
      }
      isDragging = false;
      marqueeRef.current = null;
      setMarquee(null);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [selectedIds, setSelection]);

  // Handle item selection
  const handleItemClick = useCallback((id: string, e: React.MouseEvent) => {
    if (e.ctrlKey || e.metaKey) {
      // Toggle: add or remove
      if (selectedIds.includes(id)) {
        const { removeFromSelection } = useMediaStore.getState();
        removeFromSelection(id);
      } else {
        addToSelection(id);
      }
    } else if (e.shiftKey) {
      addToSelection(id);
    } else {
      setSelection([id]);
    }
  }, [addToSelection, setSelection, selectedIds]);

  // Handle double-click (open/expand)
  const handleItemDoubleClick = useCallback(async (item: ProjectItem) => {
    if ('isExpanded' in item) {
      // It's a folder — in grid view navigate into it, in list view toggle expand
      if (viewMode === 'grid') {
        setGridFolderId(item.id);
      } else {
        toggleFolderExpanded(item.id);
      }
    } else if (item.type === 'composition') {
      // Open composition in timeline (as a tab)
      openCompositionTab(item.id);
    } else if ((item.type === 'video' || item.type === 'image') && 'file' in item && (item as MediaFile).file) {
      // Open in source monitor
      useMediaStore.getState().setSourceMonitorFile(item.id);
    } else if ('file' in item && !item.file) {
      // Media file needs reload - request permission
      const success = await reloadFile(item.id);
      if (success) {
        log.info('File reloaded successfully');
      }
    }
  }, [toggleFolderExpanded, openCompositionTab, reloadFile, viewMode]);

  // Context menu
  const handleContextMenu = useCallback((e: React.MouseEvent, itemId?: string) => {
    e.preventDefault();
    if (itemId && !selectedIds.includes(itemId)) {
      // If right-clicking an unselected item, select only it (unless Ctrl held)
      if (e.ctrlKey || e.metaKey) {
        addToSelection(itemId);
      } else {
        setSelection([itemId]);
      }
    }
    setContextMenu({ x: e.clientX, y: e.clientY, itemId });
  }, [selectedIds, setSelection, addToSelection]);

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
    // Only start rename if item is already selected (double-click on name effect)
    if (selectedIds.includes(id)) {
      e.stopPropagation();
      startRename(id, currentName);
    }
  }, [selectedIds, startRename]);

  // Handle badge click — select clip using this media file, open properties panel with target tab
  const handleBadgeClick = useCallback((mediaFileId: string, tab: 'transcript' | 'analysis') => {
    const timelineState = useTimelineStore.getState();
    // Find a clip in the timeline that uses this media file
    const clip = timelineState.clips.find(c =>
      (c.source?.mediaFileId || c.mediaFileId) === mediaFileId
    );
    if (clip) {
      timelineState.selectClip(clip.id);
    }
    // Open clip-properties panel and dispatch tab switch after React re-renders
    useDockStore.getState().activatePanelType('clip-properties');
    requestAnimationFrame(() => {
      window.dispatchEvent(new CustomEvent('openPropertiesTab', { detail: { tab } }));
    });
  }, []);

  // Delete selected items
  const handleDelete = useCallback(() => {
    selectedIds.forEach(id => {
      if (files.find(f => f.id === id)) removeFile(id);
      else if (compositions.find(c => c.id === id)) removeComposition(id);
      else if (folders.find(f => f.id === id)) removeFolder(id);
    });
    closeContextMenu();
  }, [selectedIds, files, compositions, folders, removeFile, removeComposition, removeFolder, closeContextMenu]);

  // Get the active parent folder (grid view: current open folder, list view: selected folder or null)
  const getActiveParentId = useCallback((): string | null => {
    if (viewMode === 'grid' && gridFolderId) return gridFolderId;
    // In list view, if a single folder is selected, create inside it
    if (selectedIds.length === 1) {
      const sel = folders.find(f => f.id === selectedIds[0]);
      if (sel) return sel.id;
    }
    return null;
  }, [viewMode, gridFolderId, selectedIds, folders]);

  // New composition
  const handleNewComposition = useCallback(() => {
    createComposition(`Comp ${compositions.length + 1}`, { parentId: getActiveParentId() });
    closeContextMenu();
  }, [compositions.length, createComposition, getActiveParentId, closeContextMenu]);

  // New folder
  const handleNewFolder = useCallback(() => {
    createFolder('New Folder', getActiveParentId());
    closeContextMenu();
  }, [createFolder, getActiveParentId, closeContextMenu]);

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
    clearExternalDragPayload();

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
      setExternalDragPayload({
        kind: 'composition',
        id: comp.id,
        duration: comp.timelineData?.duration ?? comp.duration ?? 5,
        hasAudio: true,
        isAudio: false,
        isVideo: true,
      });
      e.dataTransfer.setData('application/x-composition-id', comp.id);
      e.dataTransfer.effectAllowed = 'copyMove';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle text item drag
    if (item.type === 'text') {
      setExternalDragPayload({
        kind: 'text',
        id: item.id,
        duration: item.duration,
        hasAudio: false,
        isAudio: false,
        isVideo: true,
      });
      e.dataTransfer.setData('application/x-text-item-id', item.id);
      e.dataTransfer.effectAllowed = 'copyMove';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle solid item drag
    if (item.type === 'solid') {
      setExternalDragPayload({
        kind: 'solid',
        id: item.id,
        duration: item.duration,
        hasAudio: false,
        isAudio: false,
        isVideo: true,
      });
      e.dataTransfer.setData('application/x-solid-item-id', item.id);
      e.dataTransfer.effectAllowed = 'copyMove';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle media file drag
    const mediaFile = item as MediaFile;
    if (!mediaFile.file || mediaFile.isImporting) {
      // File not available or still importing - only allow internal move
      e.dataTransfer.effectAllowed = 'move';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Set the media file ID so Timeline can look it up
    const isAudioOnly =
      mediaFile.file.type.startsWith('audio/') ||
      /\.(mp3|wav|ogg|aac|m4a|flac|wma|aiff|alac|opus)$/i.test(mediaFile.file.name);
    setExternalDragPayload({
      kind: 'media-file',
      id: mediaFile.id,
      duration: mediaFile.duration,
      hasAudio: mediaFile.type === 'image' ? false : isAudioOnly ? true : mediaFile.hasAudio,
      isAudio: isAudioOnly,
      isVideo: !isAudioOnly,
      file: mediaFile.file,
    });
    e.dataTransfer.setData('application/x-media-file-id', mediaFile.id);
    // Mark audio-only files so timeline can restrict drop targets to audio tracks
    if (isAudioOnly) {
      e.dataTransfer.setData('application/x-media-is-audio', 'true');
    }
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
    clearExternalDragPayload();
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
            {/* Transcript badge with coverage fill */}
            {'transcriptStatus' in item && (item as MediaFile).transcriptStatus === 'ready' && (() => {
              const cov = (item as MediaFile).transcriptCoverage ?? 0;
              const pct = Math.round(cov * 100);
              return pct >= 100 ? (
                <span
                  className="media-item-transcript-badge"
                  title="Fully transcribed — click to open"
                  onClick={(e) => { e.stopPropagation(); handleBadgeClick(item.id, 'transcript'); }}
                >T</span>
              ) : (
                <span
                  className="media-item-transcript-fill"
                  title={`${pct}% transcribed — click to open`}
                  onClick={(e) => { e.stopPropagation(); handleBadgeClick(item.id, 'transcript'); }}
                >
                  <span className="coverage-fill-badge transcript-fill">
                    <span className="coverage-fill-bg">T</span>
                    <span className="coverage-fill-progress" style={{ height: `${pct}%` }}>T</span>
                  </span>
                </span>
              );
            })()}
            {/* Analysis badge with coverage fill */}
            {'analysisStatus' in item && (item as MediaFile).analysisStatus === 'ready' && (() => {
              const cov = (item as MediaFile).analysisCoverage ?? 0;
              const pct = Math.round(cov * 100);
              return pct >= 100 ? (
                <span
                  className="media-item-analysis-badge"
                  title="Fully analyzed — click to open"
                  onClick={(e) => { e.stopPropagation(); handleBadgeClick(item.id, 'analysis'); }}
                >A</span>
              ) : (
                <span
                  className="media-item-analysis-fill"
                  title={`${pct}% analyzed — click to open`}
                  onClick={(e) => { e.stopPropagation(); handleBadgeClick(item.id, 'analysis'); }}
                >
                  <span className="coverage-fill-badge analysis-fill">
                    <span className="coverage-fill-bg">A</span>
                    <span className="coverage-fill-progress" style={{ height: `${pct}%` }}>A</span>
                  </span>
                </span>
              );
            })()}
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
    const isImporting = isMediaFile && !!(item as MediaFile).isImporting;
    const isDragTarget = isFolder && dragOverFolderId === item.id;
    const isBeingDragged = internalDragId === item.id;
    const mediaFile = isMediaFile ? (item as MediaFile) : null;

    return (
      <div key={item.id} data-item-id={item.id}>
        <div
          className={`media-item ${isSelected ? 'selected' : ''} ${isFolder ? 'folder' : ''} ${isMediaFile && !hasFile ? 'no-file' : ''} ${isImporting ? 'importing' : ''} ${isDragTarget ? 'drag-target' : ''} ${isBeingDragged ? 'dragging' : ''}`}
          draggable
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
            {sortItems(getItemsByFolder(item.id)).map(child => renderItem(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  // Build hover tooltip for grid items
  const buildGridTooltip = (item: ProjectItem, isFolder: boolean, isComp: boolean): string => {
    const parts: string[] = [item.name];

    if (isFolder) {
      const children = getItemsByFolder(item.id);
      parts.push(`${children.length} item${children.length !== 1 ? 's' : ''}`);
    } else if (isComp) {
      const comp = item as Composition;
      parts.push(`${comp.width}×${comp.height}`);
      parts.push(`${comp.frameRate} fps`);
      if (comp.duration) parts.push(formatDuration(comp.duration));
    } else if ('type' in item) {
      const mf = item as MediaFile;
      if (mf.width && mf.height) parts.push(`${mf.width}×${mf.height}`);
      if (mf.duration) parts.push(formatDuration(mf.duration));
      if (mf.codec) parts.push(mf.codec);
      if (mf.audioCodec) parts.push(mf.audioCodec);
      if (mf.fps) parts.push(`${mf.fps} fps`);
      if (mf.fileSize) parts.push(formatFileSize(mf.fileSize));
      if (mf.bitrate) parts.push(formatBitrate(mf.bitrate));
    }

    return parts.join('\n');
  };

  // Render a single grid item
  const renderGridItem = (item: ProjectItem) => {
    const isFolder = 'isExpanded' in item;
    const isSelected = selectedIds.includes(item.id);
    const isMediaFile = !isFolder && 'type' in item && item.type !== 'composition' && item.type !== 'text' && item.type !== 'solid';
    const mediaFile = isMediaFile ? (item as MediaFile) : null;
    const isComp = !isFolder && 'type' in item && item.type === 'composition';
    const comp = isComp ? (item as Composition) : null;
    const thumbUrl = mediaFile?.thumbnailUrl;
    const isDragTarget = isFolder && dragOverFolderId === item.id;

    // Duration badge: videos + compositions
    const duration = mediaFile?.duration || comp?.duration;

    // Folder item count
    const folderCount = isFolder ? getItemsByFolder(item.id).length : 0;

    return (
      <div key={item.id} data-item-id={item.id}>
        <div
          className={`media-grid-item ${isSelected ? 'selected' : ''} ${isFolder ? 'folder' : ''} ${isDragTarget ? 'drag-target' : ''}`}
          draggable
          onDragStart={(e) => handleDragStart(e, item)}
          onDragEnd={handleDragEnd}
          onDragOver={isFolder ? (e) => handleFolderDragOver(e, item.id) : undefined}
          onDragLeave={isFolder ? handleFolderDragLeave : undefined}
          onDrop={isFolder ? (e) => handleFolderDrop(e, item.id) : undefined}
          onClick={(e) => handleItemClick(item.id, e)}
          onDoubleClick={() => handleItemDoubleClick(item)}
          onContextMenu={(e) => handleContextMenu(e, item.id)}
          title={buildGridTooltip(item, isFolder, isComp)}
        >
          <div className="media-grid-thumb">
            {thumbUrl ? (
              <img src={thumbUrl} alt="" draggable={false} />
            ) : (
              <div className="media-grid-thumb-placeholder">
                <FileTypeIcon type={isFolder ? 'folder' : isComp ? 'composition' : (item as MediaFile).type} large />
              </div>
            )}
            {duration ? (
              <span className="media-grid-duration">{formatDuration(duration)}</span>
            ) : null}
            {isFolder && folderCount > 0 && (
              <span className="media-grid-badge">{folderCount}</span>
            )}
          </div>
          <div className="media-grid-name" title={item.name}>{item.name}</div>
        </div>
      </div>
    );
  };

  // Get root items (with sorting applied)
  const rootItems = sortItems(getItemsByFolder(null));
  const totalItems = files.length + compositions.length;

  // Grid view: items for current folder + breadcrumb path
  const gridItems = sortItems(getItemsByFolder(gridFolderId));
  const gridBreadcrumb: Array<{ id: string | null; name: string }> = [];
  if (gridFolderId) {
    // Build path from root to current folder
    const path: Array<{ id: string; name: string }> = [];
    let current = folders.find(f => f.id === gridFolderId);
    while (current) {
      path.unshift({ id: current.id, name: current.name });
      current = current.parentId ? folders.find(f => f.id === current!.parentId) : undefined;
    }
    gridBreadcrumb.push({ id: null, name: '/' });
    gridBreadcrumb.push(...path);
  }

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
          <button
            className={`btn btn-sm btn-icon media-view-toggle ${viewMode === 'list' ? 'active' : ''}`}
            onClick={() => { const m = 'list'; setViewMode(m); setGridFolderId(null); localStorage.setItem('media-panel-view-mode', m); }}
            title="List View"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><rect x="1" y="2" width="14" height="2" rx="0.5"/><rect x="1" y="7" width="14" height="2" rx="0.5"/><rect x="1" y="12" width="14" height="2" rx="0.5"/></svg>
          </button>
          <button
            className={`btn btn-sm btn-icon media-view-toggle ${viewMode === 'grid' ? 'active' : ''}`}
            onClick={() => { const m = 'grid'; setViewMode(m); localStorage.setItem('media-panel-view-mode', m); }}
            title="Grid View"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><rect x="1" y="1" width="6" height="6" rx="1"/><rect x="9" y="1" width="6" height="6" rx="1"/><rect x="1" y="9" width="6" height="6" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/></svg>
          </button>
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
        ) : viewMode === 'list' ? (
          <div className="media-panel-table-wrapper">
            {/* Column headers */}
            <div className="media-column-headers">
              {columnOrder.map((colId) => (
                <div
                  key={colId}
                  className={`media-col media-col-${colId} ${draggingColumn === colId ? 'dragging' : ''} ${dragOverColumn === colId ? 'drag-over' : ''} ${sortColumn === colId ? 'sorted' : ''}`}
                  style={colId === 'name' ? { width: nameColumnWidth, minWidth: nameColumnWidth, maxWidth: nameColumnWidth } : undefined}
                  draggable
                  onDragStart={(e) => handleColumnDragStart(e, colId)}
                  onDragOver={(e) => handleColumnDragOver(e, colId)}
                  onDragLeave={handleColumnDragLeave}
                  onDrop={(e) => handleColumnDrop(e, colId)}
                  onDragEnd={handleColumnDragEnd}
                  onClick={() => handleColumnSort(colId)}
                >
                  {COLUMN_LABELS_MAP[colId]}
                  {sortColumn === colId && (
                    <span className="media-sort-indicator">{sortDirection === 'asc' ? '▲' : '▼'}</span>
                  )}
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
            <div
              className="media-item-list"
              ref={itemListRef}
              onMouseDown={handleMarqueeMouseDown}
              style={{ position: 'relative' }}
            >
              {rootItems.map(item => renderItem(item))}
              {/* Marquee selection rectangle */}
              {marquee && (() => {
                const left = Math.min(marquee.startX, marquee.currentX);
                const top = Math.min(marquee.startY, marquee.currentY);
                const width = Math.abs(marquee.currentX - marquee.startX);
                const height = Math.abs(marquee.currentY - marquee.startY);
                if (width < 3 && height < 3) return null;
                return (
                  <div
                    className="media-marquee"
                    style={{ left, top, width, height }}
                  />
                );
              })()}
            </div>
          </div>
        ) : (
          /* Grid View */
          <div
            className="media-grid-wrapper"
            ref={itemListRef}
            onMouseDown={handleMarqueeMouseDown}
            style={{ position: 'relative' }}
          >
            {/* Breadcrumb for folder navigation */}
            {gridFolderId && (
              <div className="media-grid-breadcrumb">
                {gridBreadcrumb.map((crumb, i) => (
                  <React.Fragment key={crumb.id ?? 'root'}>
                    {i > 0 && <span className="media-grid-breadcrumb-sep">/</span>}
                    <button
                      className={`media-grid-breadcrumb-btn ${i === gridBreadcrumb.length - 1 ? 'active' : ''}`}
                      onClick={() => setGridFolderId(crumb.id)}
                    >
                      {crumb.name}
                    </button>
                  </React.Fragment>
                ))}
              </div>
            )}
            <div className="media-grid">
              {gridItems.map(item => renderGridItem(item))}
            </div>
            {/* Marquee selection rectangle */}
            {marquee && (() => {
              const left = Math.min(marquee.startX, marquee.currentX);
              const top = Math.min(marquee.startY, marquee.currentY);
              const width = Math.abs(marquee.currentX - marquee.startX);
              const height = Math.abs(marquee.currentY - marquee.startY);
              if (width < 3 && height < 3) return null;
              return (
                <div
                  className="media-marquee"
                  style={{ left, top, width, height }}
                />
              );
            })()}
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
        const multiSelect = selectedIds.length > 1;
        const selectedItem = contextMenu.itemId
          ? files.find(f => f.id === contextMenu.itemId) ||
            compositions.find(c => c.id === contextMenu.itemId) ||
            folders.find(f => f.id === contextMenu.itemId) ||
            solidItems.find(s => s.id === contextMenu.itemId)
          : null;
        const isVideoFile = selectedItem && 'type' in selectedItem && selectedItem.type === 'video';
        const isComposition = selectedItem && 'type' in selectedItem && selectedItem.type === 'composition';
        const isSolidItem = selectedItem && 'type' in selectedItem && selectedItem.type === 'solid';
        const mediaFile = isVideoFile ? (selectedItem as MediaFile) : null;
        const composition = isComposition ? (selectedItem as Composition) : null;
        const solidItem = isSolidItem ? (selectedItem as SolidItem) : null;
        const isGenerating = mediaFile?.proxyStatus === 'generating';
        const hasProxy = mediaFile?.proxyStatus === 'ready';
        // Available folders for "Move to Folder" submenu
        const availableFolders = folders.filter(f => !selectedIds.includes(f.id));

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
            {(contextMenu.itemId || multiSelect) && (
              <>
                <div className="context-menu-separator" />

                {/* Rename - only for single selection */}
                {!multiSelect && selectedItem && (
                  <div className="context-menu-item" onClick={() => {
                    startRename(selectedItem.id, selectedItem.name);
                  }}>
                    Rename
                  </div>
                )}

                {/* Move to Folder submenu */}
                {availableFolders.length > 0 && (
                  <div className="context-menu-item has-submenu">
                    <span>Move to Folder{multiSelect ? ` (${selectedIds.length})` : ''}</span>
                    <span className="submenu-arrow">▶</span>
                    <div className="context-submenu">
                      <div
                        className="context-menu-item"
                        onClick={() => {
                          moveToFolder(selectedIds, null);
                          closeContextMenu();
                        }}
                      >
                        Root (no folder)
                      </div>
                      <div className="context-menu-separator" />
                      {availableFolders.map(folder => (
                        <div
                          key={folder.id}
                          className="context-menu-item"
                          onClick={() => {
                            moveToFolder(selectedIds, folder.id);
                            closeContextMenu();
                          }}
                        >
                          {folder.name}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Composition Settings - only for single composition */}
                {!multiSelect && isComposition && composition && (
                  <div className="context-menu-item" onClick={() => openCompositionSettings(composition)}>
                    Composition Settings...
                  </div>
                )}

                {/* Solid Settings - only for single solid */}
                {!multiSelect && isSolidItem && solidItem && (
                  <div className="context-menu-item" onClick={() => {
                    setSolidSettingsDialog({
                      solidItemId: solidItem.id,
                      width: solidItem.width,
                      height: solidItem.height,
                      color: solidItem.color,
                    });
                    closeContextMenu();
                  }}>
                    Solid Settings...
                  </div>
                )}

                {/* Proxy Generation - only for single video */}
                {!multiSelect && isVideoFile && mediaFile && (
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

                {/* Show in Explorer submenu - only for single video with file */}
                {!multiSelect && isVideoFile && mediaFile?.file && (
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

                {/* Set Proxy Folder - for single video */}
                {!multiSelect && isVideoFile && (
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
                  Delete{multiSelect ? ` (${selectedIds.length} items)` : ''}
                </div>
              </>
            )}
          </div>
        );
      })()}

      {/* Composition Settings Dialog */}
      {settingsDialog && (
        <CompositionSettingsDialog
          settings={settingsDialog}
          onSettingsChange={setSettingsDialog}
          onSave={saveCompositionSettings}
          onCancel={() => setSettingsDialog(null)}
        />
      )}

      {/* Solid Settings Dialog */}
      {solidSettingsDialog && (
        <SolidSettingsDialog
          settings={solidSettingsDialog}
          onSettingsChange={setSolidSettingsDialog}
          onSave={() => {
            if (solidSettingsDialog) {
              updateSolidItem(solidSettingsDialog.solidItemId, {
                color: solidSettingsDialog.color,
                width: solidSettingsDialog.width,
                height: solidSettingsDialog.height,
              });
              setSolidSettingsDialog(null);
            }
          }}
          onCancel={() => setSolidSettingsDialog(null)}
        />
      )}

      {/* Label Color Picker */}
      {labelPickerItemId && labelPickerPos && (
        <LabelColorPicker
          position={labelPickerPos}
          selectedIds={selectedIds}
          labelPickerItemId={labelPickerItemId}
          onSelect={(ids, colorKey) => {
            setLabelColor(ids, colorKey as any);
            setLabelPickerItemId(null);
            setLabelPickerPos(null);
          }}
          onClose={() => { setLabelPickerItemId(null); setLabelPickerPos(null); }}
        />
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
