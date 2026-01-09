// Media Panel - Project browser like After Effects

import { useCallback, useRef, useState, useEffect } from 'react';
import { useMediaStore } from '../../stores/mediaStore';
import type { MediaFile, Composition, ProjectItem } from '../../stores/mediaStore';

export function MediaPanel() {
  const {
    files,
    compositions,
    folders,
    selectedIds,
    expandedFolderIds,
    importFiles,
    importFilesWithPicker,
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
    getItemsByFolder,
    openCompositionTab,
    updateComposition,
    generateProxy,
    cancelProxyGeneration,
    fileSystemSupported,
    proxyFolderName,
    pickProxyFolder,
    showInExplorer,
    activeCompositionId,
    moveToFolder,
  } = useMediaStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; itemId?: string } | null>(null);
  const [settingsDialog, setSettingsDialog] = useState<{ compositionId: string; width: number; height: number; frameRate: number } | null>(null);
  const [addDropdownOpen, setAddDropdownOpen] = useState(false);
  const [dragOverFolderId, setDragOverFolderId] = useState<string | null>(null);
  const [internalDragId, setInternalDragId] = useState<string | null>(null);

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
      // Open composition in timeline (as a tab)
      openCompositionTab(item.id);
    }
  }, [toggleFolderExpanded, openCompositionTab]);

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

  // Composition settings
  const openCompositionSettings = useCallback((comp: Composition) => {
    setSettingsDialog({
      compositionId: comp.id,
      width: comp.width,
      height: comp.height,
      frameRate: comp.frameRate,
    });
    closeContextMenu();
  }, [closeContextMenu]);

  const saveCompositionSettings = useCallback(() => {
    if (!settingsDialog) return;
    updateComposition(settingsDialog.compositionId, {
      width: settingsDialog.width,
      height: settingsDialog.height,
      frameRate: settingsDialog.frameRate,
    });
    setSettingsDialog(null);
  }, [settingsDialog, updateComposition]);

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

  // Handle drop on root (move out of folder)
  const handleRootDrop = useCallback((e: React.DragEvent) => {
    // Only handle internal drags
    if (!e.dataTransfer.types.includes('application/x-media-panel-item')) {
      // External file drop - import
      if (e.dataTransfer.files.length > 0) {
        importFiles(e.dataTransfer.files);
      }
      return;
    }

    e.preventDefault();
    e.stopPropagation();

    const itemId = e.dataTransfer.getData('application/x-media-panel-item');
    if (itemId) {
      const itemsToMove = selectedIds.includes(itemId) ? selectedIds : [itemId];
      moveToFolder(itemsToMove, null); // null = root
    }

    setDragOverFolderId(null);
    setInternalDragId(null);
  }, [selectedIds, moveToFolder, importFiles]);

  // Render a single item
  const renderItem = (item: ProjectItem, depth: number = 0) => {
    const isFolder = 'isExpanded' in item;
    const isSelected = selectedIds.includes(item.id);
    const isRenaming = renamingId === item.id;
    const isExpanded = isFolder && expandedFolderIds.includes(item.id);
    const isMediaFile = !isFolder && 'type' in item && item.type !== 'composition';
    const hasFile = isMediaFile && 'file' in item && !!(item as MediaFile).file;
    // All items are draggable for internal moves
    const canDrag = true;
    const isDragTarget = isFolder && dragOverFolderId === item.id;
    const isBeingDragged = internalDragId === item.id;

    return (
      <div key={item.id}>
        <div
          className={`media-item ${isSelected ? 'selected' : ''} ${isFolder ? 'folder' : ''} ${isMediaFile && !hasFile ? 'no-file' : ''} ${isDragTarget ? 'drag-target' : ''} ${isBeingDragged ? 'dragging' : ''}`}
          style={{ paddingLeft: `${12 + depth * 16}px` }}
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
            <span
              className={`media-item-name ${isSelected ? 'editable' : ''}`}
              onClick={(e) => handleNameClick(e, item.id, item.name)}
            >
              {item.name}
            </span>
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
          {/* Proxy badge */}
          {'proxyStatus' in item && item.proxyStatus === 'ready' && (
            <span className="media-item-proxy-badge" title="Proxy generated">
              P
            </span>
          )}
          {'proxyStatus' in item && item.proxyStatus === 'generating' && (
            <span className="media-item-proxy-generating" title={`Generating proxy: ${(item as MediaFile).proxyProgress || 0}%`}>
              ⏳ {(item as MediaFile).proxyProgress || 0}%
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
      onDrop={handleRootDrop}
      onDragOver={handleDragOver}
      onClick={() => { contextMenu && closeContextMenu(); }}
    >
      {/* Header */}
      <div className="media-panel-header">
        <span className="media-panel-title">Project</span>
        <span className="media-panel-count">{totalItems} items</span>
        <div className="media-panel-actions">
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
                  <span className="add-dropdown-icon">🎬</span>
                  <span>Composition</span>
                </div>
                <div className="add-dropdown-item" onClick={() => { handleNewFolder(); setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon">📁</span>
                  <span>Folder</span>
                </div>
                <div className="add-dropdown-separator" />
                <div className="add-dropdown-item" onClick={() => { /* TODO: Add text layer */ setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon">T</span>
                  <span>Text</span>
                  <span className="add-dropdown-hint">Coming soon</span>
                </div>
                <div className="add-dropdown-item" onClick={() => { /* TODO: Add solid */ setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon">◼</span>
                  <span>Solid</span>
                  <span className="add-dropdown-hint">Coming soon</span>
                </div>
                <div className="add-dropdown-item" onClick={() => { /* TODO: Add adjustment layer */ setAddDropdownOpen(false); }}>
                  <span className="add-dropdown-icon">◐</span>
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
            className="media-context-menu"
            style={{
              position: 'fixed',
              left: contextMenu.x,
              top: contextMenu.y,
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

      {/* Composition Settings Dialog */}
      {settingsDialog && (
        <div
          className="modal-overlay"
          onClick={() => setSettingsDialog(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10001,
          }}
        >
          <div
            className="composition-settings-dialog"
            onClick={(e) => e.stopPropagation()}
            style={{
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: '8px',
              padding: '24px',
              minWidth: '320px',
              maxWidth: '400px',
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: '20px' }}>Composition Settings</h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* Width */}
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px' }}>Width</label>
                <input
                  type="number"
                  value={settingsDialog.width}
                  onChange={(e) => setSettingsDialog({
                    ...settingsDialog,
                    width: Math.max(1, parseInt(e.target.value) || 1920),
                  })}
                  min="1"
                  max="7680"
                  style={{ width: '100%', padding: '6px 8px' }}
                />
              </div>

              {/* Height */}
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px' }}>Height</label>
                <input
                  type="number"
                  value={settingsDialog.height}
                  onChange={(e) => setSettingsDialog({
                    ...settingsDialog,
                    height: Math.max(1, parseInt(e.target.value) || 1080),
                  })}
                  min="1"
                  max="4320"
                  style={{ width: '100%', padding: '6px 8px' }}
                />
              </div>

              {/* Frame Rate */}
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px' }}>Frame Rate</label>
                <select
                  value={settingsDialog.frameRate}
                  onChange={(e) => setSettingsDialog({
                    ...settingsDialog,
                    frameRate: Number(e.target.value),
                  })}
                  style={{ width: '100%', padding: '6px 8px' }}
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
            </div>

            {/* Buttons */}
            <div style={{ display: 'flex', gap: '8px', marginTop: '24px', justifyContent: 'flex-end' }}>
              <button
                className="btn"
                onClick={() => setSettingsDialog(null)}
                style={{ flex: 1 }}
              >
                Cancel
              </button>
              <button
                className="btn"
                onClick={saveCompositionSettings}
                style={{ flex: 1, background: '#4a90e2' }}
              >
                Save
              </button>
            </div>
          </div>
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
