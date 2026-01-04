// Media Panel - Project browser like After Effects

import { useCallback, useRef, useState } from 'react';
import { useMediaStore } from '../stores/mediaStore';
import type { MediaFile, Composition, MediaFolder, ProjectItem } from '../stores/mediaStore';

export function MediaPanel() {
  const {
    files,
    compositions,
    folders,
    selectedIds,
    expandedFolderIds,
    importFiles,
    createComposition,
    createFolder,
    removeFile,
    removeComposition,
    removeFolder,
    renameFile,
    renameFolder,
    toggleFolderExpanded,
    setSelection,
    addToSelection,
    clearSelection,
    getItemsByFolder,
    setActiveComposition,
  } = useMediaStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; itemId?: string } | null>(null);

  // Handle file import
  const handleImport = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await importFiles(e.target.files);
      e.target.value = ''; // Reset input
    }
  }, [importFiles]);

  // Handle drag & drop import
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files.length > 0) {
      await importFiles(e.dataTransfer.files);
    }
  }, [importFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
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
  const handleItemDoubleClick = useCallback((item: ProjectItem) => {
    if ('isExpanded' in item) {
      // It's a folder
      toggleFolderExpanded(item.id);
    } else if (item.type === 'composition') {
      // Open composition in timeline
      setActiveComposition(item.id);
    }
  }, [toggleFolderExpanded, setActiveComposition]);

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

    if (file) {
      renameFile(renamingId, renameValue.trim());
    } else if (folder) {
      renameFolder(renamingId, renameValue.trim());
    }

    setRenamingId(null);
  }, [renamingId, renameValue, files, folders, renameFile, renameFolder]);

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

  // Handle drag start for media files (to drag to Timeline)
  const handleDragStart = useCallback((e: React.DragEvent, item: ProjectItem) => {
    // Only allow dragging media files (not folders or compositions)
    if ('isExpanded' in item || item.type === 'composition') {
      e.preventDefault();
      return;
    }

    // Get the file from the media item
    const mediaFile = item as MediaFile;
    if (!mediaFile.file) {
      // File not available (e.g., after page refresh)
      e.preventDefault();
      return;
    }

    // Set the media file ID so Timeline can look it up
    e.dataTransfer.setData('application/x-media-file-id', mediaFile.id);
    e.dataTransfer.effectAllowed = 'copy';

    // Set drag image
    if (e.currentTarget instanceof HTMLElement) {
      e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
    }
  }, []);

  // Render a single item
  const renderItem = (item: ProjectItem, depth: number = 0) => {
    const isFolder = 'isExpanded' in item;
    const isSelected = selectedIds.includes(item.id);
    const isRenaming = renamingId === item.id;
    const isExpanded = isFolder && expandedFolderIds.includes(item.id);
    const isMediaFile = !isFolder && item.type !== 'composition';
    const hasFile = isMediaFile && 'file' in item && !!(item as MediaFile).file;

    return (
      <div key={item.id}>
        <div
          className={`media-item ${isSelected ? 'selected' : ''} ${isFolder ? 'folder' : ''} ${isMediaFile && !hasFile ? 'no-file' : ''}`}
          style={{ paddingLeft: `${12 + depth * 16}px` }}
          draggable={isMediaFile && hasFile}
          onDragStart={(e) => handleDragStart(e, item)}
          onClick={(e) => handleItemClick(item.id, e)}
          onDoubleClick={() => handleItemDoubleClick(item)}
          onContextMenu={(e) => handleContextMenu(e, item.id)}
        >
          {/* Icon */}
          <span className="media-item-icon">
            {isFolder ? (isExpanded ? '📂' : '📁') :
             item.type === 'composition' ? '🎬' :
             item.type === 'video' ? '🎥' :
             item.type === 'audio' ? '🔊' :
             item.type === 'image' ? '🖼️' : '📄'}
          </span>

          {/* Thumbnail for media files */}
          {'thumbnailUrl' in item && item.thumbnailUrl && (
            <img
              src={item.thumbnailUrl}
              alt=""
              className="media-item-thumbnail"
              draggable={false}
            />
          )}

          {/* Name */}
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
            <span className="media-item-name">{item.name}</span>
          )}

          {/* Info */}
          {'duration' in item && item.duration && (
            <span className="media-item-info">
              {formatDuration(item.duration)}
            </span>
          )}
          {'width' in item && 'height' in item && item.width && item.height && (
            <span className="media-item-info">
              {item.width}×{item.height}
            </span>
          )}
        </div>

        {/* Render children if folder is expanded */}
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

  return (
    <div
      className="media-panel"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={() => contextMenu && closeContextMenu()}
    >
      {/* Header */}
      <div className="media-panel-header">
        <span className="media-panel-title">Project</span>
        <span className="media-panel-count">{totalItems} items</span>
        <div className="media-panel-actions">
          <button className="btn btn-sm" onClick={handleImport} title="Import Media">
            + Import
          </button>
          <button className="btn btn-sm" onClick={handleNewComposition} title="New Composition">
            + Comp
          </button>
          <button className="btn btn-sm" onClick={handleNewFolder} title="New Folder">
            + Folder
          </button>
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

      {/* Item list */}
      <div className="media-panel-content">
        {rootItems.length === 0 ? (
          <div className="media-panel-empty">
            <p>No media imported</p>
            <p className="hint">Drag & drop files here or click Import</p>
          </div>
        ) : (
          <div className="media-item-list">
            {rootItems.map(item => renderItem(item))}
          </div>
        )}
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <div
          className="media-context-menu"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button onClick={handleImport}>Import Media...</button>
          <button onClick={handleNewComposition}>New Composition</button>
          <button onClick={handleNewFolder}>New Folder</button>
          {contextMenu.itemId && (
            <>
              <div className="context-menu-divider" />
              <button onClick={() => {
                const item = files.find(f => f.id === contextMenu.itemId) ||
                             compositions.find(c => c.id === contextMenu.itemId) ||
                             folders.find(f => f.id === contextMenu.itemId);
                if (item) startRename(item.id, item.name);
              }}>
                Rename
              </button>
              <button onClick={handleDelete} className="danger">
                Delete
              </button>
            </>
          )}
        </div>
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
