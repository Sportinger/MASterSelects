// Media Panel - Project browser like After Effects

import { useCallback, useRef, useState } from 'react';
import { useMediaStore } from '../stores/mediaStore';
import type { MediaFile, Composition, ProjectItem } from '../stores/mediaStore';

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
    setActiveComposition,
    generateProxy,
    cancelProxyGeneration,
    fileSystemSupported,
    proxyFolderName,
    pickProxyFolder,
    showInExplorer,
  } = useMediaStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; itemId?: string } | null>(null);

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

  // Handle drag start for media files and compositions (to drag to Timeline)
  const handleDragStart = useCallback((e: React.DragEvent, item: ProjectItem) => {
    // Don't allow dragging folders
    if ('isExpanded' in item) {
      e.preventDefault();
      return;
    }

    // Handle composition drag
    if (item.type === 'composition') {
      const comp = item as Composition;
      // Don't allow dragging comp into itself (check active comp)
      const { activeCompositionId } = useMediaStore.getState();
      if (comp.id === activeCompositionId) {
        e.preventDefault();
        return;
      }
      e.dataTransfer.setData('application/x-composition-id', comp.id);
      e.dataTransfer.effectAllowed = 'copy';
      if (e.currentTarget instanceof HTMLElement) {
        e.dataTransfer.setDragImage(e.currentTarget, 10, 10);
      }
      return;
    }

    // Handle media file drag
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
    const isComposition = item.type === 'composition';
    const hasFile = isMediaFile && 'file' in item && !!(item as MediaFile).file;
    // Compositions are always draggable, media files only if they have the blob
    const canDrag = isComposition || hasFile;

    return (
      <div key={item.id}>
        <div
          className={`media-item ${isSelected ? 'selected' : ''} ${isFolder ? 'folder' : ''} ${isMediaFile && !hasFile ? 'no-file' : ''}`}
          style={{ paddingLeft: `${12 + depth * 16}px` }}
          draggable={canDrag}
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
      {contextMenu && (() => {
        const selectedItem = contextMenu.itemId
          ? files.find(f => f.id === contextMenu.itemId) ||
            compositions.find(c => c.id === contextMenu.itemId) ||
            folders.find(f => f.id === contextMenu.itemId)
          : null;
        const isVideoFile = selectedItem && 'type' in selectedItem && selectedItem.type === 'video';
        const mediaFile = isVideoFile ? (selectedItem as MediaFile) : null;
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
    </div>
  );
}

// Format duration as mm:ss
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
