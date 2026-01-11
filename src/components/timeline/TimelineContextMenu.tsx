// TimelineContextMenu - Right-click context menu for timeline clips
// Extracted from Timeline.tsx for better maintainability

import React, { useEffect, useCallback } from 'react';
import type { TimelineClip, MediaFile } from '../../types';
import type { ContextMenuState } from './types';
import { useContextMenuPosition } from '../../hooks/useContextMenuPosition';
import { useMediaStore } from '../../stores/mediaStore';

interface TimelineContextMenuProps {
  contextMenu: ContextMenuState | null;
  setContextMenu: (menu: ContextMenuState | null) => void;

  // Clip data
  clipMap: Map<string, TimelineClip>;
  selectedClipIds: Set<string>;

  // Actions
  selectClip: (clipId: string) => void;
  removeClip: (clipId: string) => void;
  splitClipAtPlayhead: () => void;
  toggleClipReverse: (clipId: string) => void;
  unlinkGroup: (clipId: string) => void;
  generateWaveformForClip: (clipId: string) => void;
  setMulticamDialogOpen: (open: boolean) => void;

  // Proxy
  proxyFolderName: string | null;
  showInExplorer: (type: 'raw' | 'proxy', fileId: string) => Promise<{ success: boolean; message: string }>;
  pickProxyFolder: () => Promise<void>;
}

export function TimelineContextMenu({
  contextMenu,
  setContextMenu,
  clipMap,
  selectedClipIds,
  selectClip,
  removeClip,
  splitClipAtPlayhead,
  toggleClipReverse,
  unlinkGroup,
  generateWaveformForClip,
  setMulticamDialogOpen,
  proxyFolderName,
  showInExplorer,
  pickProxyFolder,
}: TimelineContextMenuProps) {
  const { menuRef: contextMenuRef, adjustedPosition: contextMenuPosition } = useContextMenuPosition(contextMenu);

  // Get the media file for a clip
  const getMediaFileForClip = useCallback(
    (clipId: string): MediaFile | null => {
      const clip = clipMap.get(clipId);
      if (!clip) return null;

      const mediaStore = useMediaStore.getState();
      return mediaStore.files.find(
        (f) =>
          f.id === clip.source?.mediaFileId ||
          f.name === clip.name ||
          f.name === clip.name.replace(' (Audio)', '')
      ) || null;
    },
    [clipMap]
  );

  // Close context menu when clicking outside or pressing Escape
  useEffect(() => {
    if (!contextMenu) return;

    const handleClickOutside = () => {
      setContextMenu(null);
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setContextMenu(null);
      }
    };

    const timeoutId = setTimeout(() => {
      window.addEventListener('click', handleClickOutside);
    }, 0);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('click', handleClickOutside);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [contextMenu, setContextMenu]);

  // Handle "Show in Explorer" action
  const handleShowInExplorer = async (type: 'raw' | 'proxy') => {
    if (!contextMenu) return;

    const mediaFile = getMediaFileForClip(contextMenu.clipId);

    if (!mediaFile) {
      console.warn('[Timeline] Media file not found for clip');
      setContextMenu(null);
      return;
    }

    const result = await showInExplorer(type, mediaFile.id);

    if (result.success) {
      alert(result.message);
    } else {
      if (type === 'raw' && mediaFile.file) {
        const url = URL.createObjectURL(mediaFile.file);
        const a = document.createElement('a');
        a.href = url;
        a.download = mediaFile.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log('[Timeline] Downloaded raw file:', mediaFile.name);
      } else {
        alert(result.message);
      }
    }

    setContextMenu(null);
  };

  // Handle Set Proxy Folder
  const handleSetProxyFolder = async () => {
    await pickProxyFolder();
    setContextMenu(null);
  };

  // Handle Start/Stop Proxy Generation
  const handleProxyGeneration = (action: 'start' | 'stop') => {
    if (!contextMenu) return;

    const mediaFile = getMediaFileForClip(contextMenu.clipId);
    if (!mediaFile) {
      setContextMenu(null);
      return;
    }

    const mediaStore = useMediaStore.getState();

    if (action === 'start') {
      mediaStore.generateProxy(mediaFile.id);
      console.log('[Timeline] Starting proxy generation for:', mediaFile.name);
    } else {
      mediaStore.cancelProxyGeneration(mediaFile.id);
      console.log('[Timeline] Cancelled proxy generation for:', mediaFile.name);
    }

    setContextMenu(null);
  };

  // Handle right-click on clip (to be called from parent)
  const handleClipContextMenu = useCallback(
    (e: React.MouseEvent, clipId: string) => {
      e.preventDefault();
      e.stopPropagation();
      // If right-clicking on an unselected clip, select only that one
      // If right-clicking on a selected clip, keep the current multi-selection
      if (!selectedClipIds.has(clipId)) {
        selectClip(clipId);
      }
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        clipId,
      });
    },
    [selectClip, selectedClipIds, setContextMenu]
  );

  if (!contextMenu) return null;

  const mediaFile = getMediaFileForClip(contextMenu.clipId);
  const clip = clipMap.get(contextMenu.clipId);
  const isVideo = clip?.source?.type === 'video';
  const isGenerating = mediaFile?.proxyStatus === 'generating';
  const hasProxy = mediaFile?.proxyStatus === 'ready';

  return (
    <div
      ref={contextMenuRef}
      className="timeline-context-menu"
      style={{
        position: 'fixed',
        left: contextMenuPosition?.x ?? contextMenu.x,
        top: contextMenuPosition?.y ?? contextMenu.y,
        zIndex: 10000,
      }}
      onClick={(e) => e.stopPropagation()}
    >
      {isVideo && (
        <div className="context-menu-item has-submenu">
          <span>Show in Explorer</span>
          <span className="submenu-arrow">{'\u25B6'}</span>
          <div className="context-submenu">
            <div
              className="context-menu-item"
              onClick={() => handleShowInExplorer('raw')}
            >
              Raw {mediaFile?.hasFileHandle && '(has path)'}
            </div>
            <div
              className={`context-menu-item ${!hasProxy ? 'disabled' : ''}`}
              onClick={() => hasProxy && handleShowInExplorer('proxy')}
            >
              Proxy{' '}
              {!hasProxy
                ? '(not available)'
                : proxyFolderName
                ? `(${proxyFolderName})`
                : '(IndexedDB)'}
            </div>
          </div>
        </div>
      )}

      {isVideo && (
        <>
          <div className="context-menu-separator" />
          {isGenerating ? (
            <div
              className="context-menu-item"
              onClick={() => handleProxyGeneration('stop')}
            >
              Stop Proxy Generation ({mediaFile?.proxyProgress || 0}%)
            </div>
          ) : hasProxy ? (
            <div className="context-menu-item disabled">Proxy Ready</div>
          ) : (
            <div
              className="context-menu-item"
              onClick={() => handleProxyGeneration('start')}
            >
              Generate Proxy
            </div>
          )}

          <div className="context-menu-item" onClick={handleSetProxyFolder}>
            Set Proxy Folder... {proxyFolderName && `(${proxyFolderName})`}
          </div>
        </>
      )}

      <div className="context-menu-separator" />
      <div
        className="context-menu-item"
        onClick={() => {
          splitClipAtPlayhead();
          setContextMenu(null);
        }}
      >
        Split at Playhead (C)
      </div>

      {/* Multicam options */}
      {selectedClipIds.size > 1 && (
        <div
          className="context-menu-item"
          onClick={() => {
            setMulticamDialogOpen(true);
            setContextMenu(null);
          }}
        >
          Combine Multicam ({selectedClipIds.size} clips)
        </div>
      )}
      {clip?.linkedGroupId && (
        <div
          className="context-menu-item"
          onClick={() => {
            if (contextMenu.clipId) {
              unlinkGroup(contextMenu.clipId);
            }
            setContextMenu(null);
          }}
        >
          Unlink from Multicam
        </div>
      )}

      {isVideo && (
        <div
          className={`context-menu-item ${clip?.reversed ? 'checked' : ''}`}
          onClick={() => {
            if (contextMenu.clipId) {
              toggleClipReverse(contextMenu.clipId);
            }
            setContextMenu(null);
          }}
        >
          {clip?.reversed ? '\u2713 ' : ''}Reverse Playback
        </div>
      )}

      {/* Generate Waveform option for audio clips */}
      {clip?.source?.type === 'audio' && (
        <>
          <div className="context-menu-separator" />
          <div
            className={`context-menu-item ${clip?.waveformGenerating ? 'disabled' : ''}`}
            onClick={() => {
              if (contextMenu.clipId && !clip?.waveformGenerating) {
                generateWaveformForClip(contextMenu.clipId);
              }
              setContextMenu(null);
            }}
          >
            {clip?.waveformGenerating
              ? `Generating Waveform... ${clip?.waveformProgress || 0}%`
              : clip?.waveform && clip.waveform.length > 0
              ? 'Regenerate Waveform'
              : 'Generate Waveform'}
          </div>
        </>
      )}

      {(isVideo || clip?.source?.type === 'audio') && (
        <>
          <div className="context-menu-separator" />
          <div
            className={`context-menu-item ${clip?.transcriptStatus === 'transcribing' ? 'disabled' : ''}`}
            onClick={async () => {
              if (contextMenu.clipId && clip?.transcriptStatus !== 'transcribing') {
                const { transcribeClip } = await import('../../services/clipTranscriber');
                transcribeClip(contextMenu.clipId);
              }
              setContextMenu(null);
            }}
          >
            {clip?.transcriptStatus === 'transcribing'
              ? `Transcribing... ${clip?.transcriptProgress || 0}%`
              : clip?.transcriptStatus === 'ready'
              ? 'Re-transcribe'
              : 'Transcribe'}
          </div>
        </>
      )}

      <div className="context-menu-separator" />
      <div
        className="context-menu-item danger"
        onClick={() => {
          if (contextMenu.clipId) {
            removeClip(contextMenu.clipId);
          }
          setContextMenu(null);
        }}
      >
        Delete Clip
      </div>
    </div>
  );
}

// Export the handler creator for use in parent
export function useClipContextMenu(
  selectedClipIds: Set<string>,
  selectClip: (clipId: string) => void,
  setContextMenu: (menu: ContextMenuState | null) => void
) {
  return useCallback(
    (e: React.MouseEvent, clipId: string) => {
      e.preventDefault();
      e.stopPropagation();
      if (!selectedClipIds.has(clipId)) {
        selectClip(clipId);
      }
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        clipId,
      });
    },
    [selectClip, selectedClipIds, setContextMenu]
  );
}
