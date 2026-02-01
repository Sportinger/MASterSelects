// TimelineContextMenu - Right-click context menu for timeline clips
// Extracted from Timeline.tsx for better maintainability

import React, { useEffect, useCallback } from 'react';
import type { TimelineClip } from '../../types';
import type { MediaFile } from '../../stores/mediaStore';
import type { ContextMenuState } from './types';
import { useContextMenuPosition } from '../../hooks/useContextMenuPosition';
import { useMediaStore } from '../../stores/mediaStore';
import { useDockStore } from '../../stores/dockStore';
import { projectFileService } from '../../services/projectFileService';
import { Logger } from '../../services/logger';

const log = Logger.create('TimelineContextMenu');

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

  // File explorer
  showInExplorer: (type: 'raw' | 'proxy', fileId: string) => Promise<{ success: boolean; message: string }>;
}

export function TimelineContextMenu({
  contextMenu,
  setContextMenu,
  clipMap,
  selectedClipIds,
  selectClip: _selectClip,
  removeClip,
  splitClipAtPlayhead,
  toggleClipReverse,
  unlinkGroup,
  generateWaveformForClip,
  setMulticamDialogOpen,
  showInExplorer,
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
          f.id === clip.mediaFileId ||
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
      log.warn('Media file not found for clip');
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
        log.debug('Downloaded raw file:', mediaFile.name);
      } else {
        alert(result.message);
      }
    }

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
      log.debug('Starting proxy generation for:', mediaFile.name);
    } else {
      mediaStore.cancelProxyGeneration(mediaFile.id);
      log.debug('Cancelled proxy generation for:', mediaFile.name);
    }

    setContextMenu(null);
  };

  // Get clip info for composition detection
  const clip = contextMenu ? clipMap.get(contextMenu.clipId) : null;
  const isComposition = clip?.isComposition;

  // Open Media Panel when right-clicking on composition clips
  useEffect(() => {
    if (isComposition && contextMenu) {
      useDockStore.getState().activatePanelType('media');
    }
  }, [isComposition, contextMenu]);

  if (!contextMenu) return null;

  const mediaFile = getMediaFileForClip(contextMenu.clipId);
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
                : projectFileService.isProjectOpen()
                ? `(${projectFileService.getProjectData()?.name}/Proxy)`
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
