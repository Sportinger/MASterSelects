// External file drag & drop handling for timeline

import { useState, useCallback, useRef } from 'react';
import { useMediaStore } from '../../../stores/mediaStore';
import {
  isVideoFile,
  isAudioFile,
  isMediaFile,
  getVideoMetadataQuick,
} from '../utils/fileTypeHelpers';
import type { ExternalDragState } from '../types';
import type { TimelineTrack, TimelineClip } from '../../../types';
import type { Composition } from '../../../stores/mediaStore';
import { Logger } from '../../../services/logger';

const log = Logger.create('useExternalDrop');

interface UseExternalDropProps {
  timelineRef: React.RefObject<HTMLDivElement | null>;
  scrollX: number;
  tracks: TimelineTrack[];
  clips: TimelineClip[];
  pixelToTime: (pixel: number) => number;
  addTrack: (type: 'video' | 'audio') => string | undefined;
  addClip: (trackId: string, file: File, startTime: number, duration?: number, mediaFileId?: string) => void;
  addCompClip: (trackId: string, comp: Composition, startTime: number) => void;
  addTextClip: (trackId: string, startTime: number, duration?: number, skipMediaItem?: boolean) => Promise<string | null>;
  addSolidClip: (trackId: string, startTime: number, color?: string, duration?: number, skipMediaItem?: boolean) => string | null;
}

interface UseExternalDropReturn {
  externalDrag: ExternalDragState | null;
  setExternalDrag: React.Dispatch<React.SetStateAction<ExternalDragState | null>>;
  dragCounterRef: React.MutableRefObject<number>;
  handleTrackDragEnter: (e: React.DragEvent, trackId: string) => void;
  handleTrackDragOver: (e: React.DragEvent, trackId: string) => void;
  handleTrackDragLeave: (e: React.DragEvent) => void;
  handleTrackDrop: (e: React.DragEvent, trackId: string) => Promise<void>;
  handleNewTrackDragOver: (e: React.DragEvent, trackType: 'video' | 'audio') => void;
  handleNewTrackDrop: (e: React.DragEvent, trackType: 'video' | 'audio') => Promise<void>;
}

/**
 * Helper to extract file path from drag event
 */
function extractFilePath(e: React.DragEvent): string | undefined {
  // Try text/uri-list (Nautilus, Dolphin)
  const uriList = e.dataTransfer.getData('text/uri-list');
  if (uriList) {
    const uri = uriList.split('\n')[0]?.trim();
    if (uri?.startsWith('file://')) {
      return decodeURIComponent(uri.replace('file://', ''));
    }
  }

  // Try text/plain (some file managers)
  const plainText = e.dataTransfer.getData('text/plain');
  if (plainText?.startsWith('/') || plainText?.startsWith('file://')) {
    return plainText.startsWith('file://')
      ? decodeURIComponent(plainText.replace('file://', ''))
      : plainText;
  }

  // Try text/x-moz-url (Firefox)
  const mozUrl = e.dataTransfer.getData('text/x-moz-url');
  if (mozUrl?.startsWith('file://')) {
    return decodeURIComponent(mozUrl.split('\n')[0].replace('file://', ''));
  }

  return undefined;
}

export function useExternalDrop({
  timelineRef,
  scrollX,
  tracks,
  clips,
  pixelToTime,
  addTrack,
  addClip,
  addCompClip,
  addTextClip,
  addSolidClip,
}: UseExternalDropProps): UseExternalDropReturn {
  const [externalDrag, setExternalDrag] = useState<ExternalDragState | null>(null);
  const dragCounterRef = useRef(0);
  const dragMetadataCacheRef = useRef<{ url: string; duration: number; hasAudio: boolean } | null>(null);

  // Handle external file drag enter on track
  const handleTrackDragEnter = useCallback(
    (e: React.DragEvent, trackId: string) => {
      e.preventDefault();
      dragCounterRef.current++;

      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const startTime = pixelToTime(x);

      if (e.dataTransfer.types.includes('application/x-composition-id')) {
        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
        return;
      }

      if (e.dataTransfer.types.includes('application/x-media-file-id')) {
        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
        return;
      }

      if (e.dataTransfer.types.includes('application/x-text-item-id')) {
        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
        return;
      }

      if (e.dataTransfer.types.includes('application/x-solid-item-id')) {
        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
        return;
      }

      if (e.dataTransfer.types.includes('Files')) {
        let dur: number | undefined;
        let hasAudio: boolean | undefined;
        const items = e.dataTransfer.items;
        if (items && items.length > 0) {
          for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.kind === 'file') {
              const file = item.getAsFile();
              if (file && isVideoFile(file)) {
                const cacheKey = `${file.name}_${file.size}`;
                if (dragMetadataCacheRef.current?.url === cacheKey) {
                  dur = dragMetadataCacheRef.current.duration;
                  hasAudio = dragMetadataCacheRef.current.hasAudio;
                } else {
                  getVideoMetadataQuick(file).then((metadata) => {
                    if (metadata) {
                      dragMetadataCacheRef.current = {
                        url: cacheKey,
                        duration: metadata.duration ?? 5,
                        hasAudio: metadata.hasAudio,
                      };
                      setExternalDrag((prev) =>
                        prev ? { ...prev, duration: metadata.duration ?? 5, hasAudio: metadata.hasAudio } : null
                      );
                    }
                  });
                }
                break;
              }
            }
          }
        }

        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: dur, hasAudio });
      }
    },
    [scrollX, pixelToTime]
  );

  // Handle external file drag over track
  const handleTrackDragOver = useCallback(
    (e: React.DragEvent, trackId: string) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';

      const isCompDrag = e.dataTransfer.types.includes('application/x-composition-id');
      const isMediaPanelDrag = e.dataTransfer.types.includes('application/x-media-file-id');
      const isTextDrag = e.dataTransfer.types.includes('application/x-text-item-id');
      const isSolidDrag = e.dataTransfer.types.includes('application/x-solid-item-id');
      const isFileDrag = e.dataTransfer.types.includes('Files');

      if ((isCompDrag || isMediaPanelDrag || isTextDrag || isSolidDrag || isFileDrag) && timelineRef.current) {
        const rect = timelineRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);

        const targetTrack = tracks.find((t) => t.id === trackId);
        const isVideoTrack = targetTrack?.type === 'video';

        const previewDuration =
          externalDrag?.duration ?? dragMetadataCacheRef.current?.duration ?? 5;

        // Check if video has audio - default to true if not yet determined
        const videoHasAudio = externalDrag?.hasAudio ?? dragMetadataCacheRef.current?.hasAudio ?? true;

        let audioTrackId: string | undefined;
        // Only show audio preview if video has audio tracks
        if (isVideoTrack && videoHasAudio) {
          const audioTracks = tracks.filter((t) => t.type === 'audio');
          const endTime = startTime + previewDuration;

          for (const aTrack of audioTracks) {
            const trackClips = clips.filter((c) => c.trackId === aTrack.id);
            const hasOverlap = trackClips.some((clip) => {
              const clipEnd = clip.startTime + clip.duration;
              return !(endTime <= clip.startTime || startTime >= clipEnd);
            });
            if (!hasOverlap) {
              audioTrackId = aTrack.id;
              break;
            }
          }
          if (!audioTrackId) {
            audioTrackId = '__new_audio_track__';
          }
        }

        setExternalDrag((prev) => ({
          trackId,
          startTime,
          x: e.clientX,
          y: e.clientY,
          audioTrackId,
          isVideo: isVideoTrack,
          hasAudio: prev?.hasAudio ?? dragMetadataCacheRef.current?.hasAudio,
          duration: prev?.duration ?? dragMetadataCacheRef.current?.duration,
        }));
      }
    },
    [scrollX, pixelToTime, tracks, clips, externalDrag, timelineRef]
  );

  // Handle external file drag leave
  const handleTrackDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current--;

    if (dragCounterRef.current === 0) {
      setExternalDrag(null);
    }
  }, []);

  // Handle drag over "new track" drop zone
  const handleNewTrackDragOver = useCallback(
    (e: React.DragEvent, trackType: 'video' | 'audio') => {
      e.preventDefault();
      e.stopPropagation();
      e.dataTransfer.dropEffect = 'copy';

      if (timelineRef.current) {
        const rect = timelineRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);

        setExternalDrag((prev) => ({
          trackId: '__new_track__',
          startTime,
          x: e.clientX,
          y: e.clientY,
          duration: prev?.duration ?? dragMetadataCacheRef.current?.duration ?? 5,
          hasAudio: prev?.hasAudio ?? dragMetadataCacheRef.current?.hasAudio,
          newTrackType: trackType,
          isVideo: trackType === 'video',
          isAudio: trackType === 'audio',
        }));
      }
    },
    [scrollX, pixelToTime, timelineRef]
  );

  // Handle drop on "new track" zone - creates new track and adds clip
  const handleNewTrackDrop = useCallback(
    async (e: React.DragEvent, trackType: 'video' | 'audio') => {
      e.preventDefault();
      e.stopPropagation();

      const cachedDuration =
        externalDrag?.duration ?? dragMetadataCacheRef.current?.duration;

      dragCounterRef.current = 0;
      setExternalDrag(null);

      // Validate file type matches track type BEFORE creating track
      const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          const fileIsAudio = isAudioFile(mediaFile.file);
          if (fileIsAudio && trackType === 'video') {
            log.debug('Audio files can only be dropped on audio tracks');
            return;
          }
          if (!fileIsAudio && trackType === 'audio') {
            log.debug('Video/image files can only be dropped on video tracks');
            return;
          }
        }
      }

      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        const fileIsAudio = isAudioFile(file);
        if (fileIsAudio && trackType === 'video') {
          log.debug('Audio files can only be dropped on audio tracks');
          return;
        }
        if (!fileIsAudio && trackType === 'audio') {
          log.debug('Video/image files can only be dropped on video tracks');
          return;
        }
      }

      // Create a new track
      const newTrackId = addTrack(trackType);
      if (!newTrackId) return;

      const rect = timelineRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left + scrollX;
      const startTime = Math.max(0, pixelToTime(x));
      const filePath = extractFilePath(e);

      // Handle composition drag
      const compositionId = e.dataTransfer.getData('application/x-composition-id');
      if (compositionId) {
        const mediaStore = useMediaStore.getState();
        const comp = mediaStore.compositions.find((c) => c.id === compositionId);
        if (comp) {
          addCompClip(newTrackId, comp, startTime);
          return;
        }
      }

      // Handle text item drag (skipMediaItem=true since it already exists in media panel)
      const textItemId = e.dataTransfer.getData('application/x-text-item-id');
      if (textItemId) {
        const mediaStore = useMediaStore.getState();
        const textItem = mediaStore.textItems.find((t) => t.id === textItemId);
        if (textItem) {
          addTextClip(newTrackId, startTime, textItem.duration, true);
          return;
        }
      }

      // Handle solid item drag (skipMediaItem=true since it already exists in media panel)
      const solidItemId = e.dataTransfer.getData('application/x-solid-item-id');
      if (solidItemId) {
        const mediaStore = useMediaStore.getState();
        const solidItem = mediaStore.solidItems.find((s) => s.id === solidItemId);
        if (solidItem) {
          addSolidClip(newTrackId, startTime, solidItem.color, solidItem.duration, true);
          return;
        }
      }

      // Handle media panel drag
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          addClip(newTrackId, mediaFile.file, startTime, mediaFile.duration, mediaFileId);
          return;
        }
      }

      // Handle external file drop
      const items = e.dataTransfer.items;
      if (items && items.length > 0) {
        const item = items[0];
        if (item.kind === 'file') {
          const mediaStore = useMediaStore.getState();

          // Try to get file handle (File System Access API)
          if ('getAsFileSystemHandle' in item) {
            try {
              const handle = await (item as any).getAsFileSystemHandle();
              if (handle && handle.kind === 'file') {
                const file = await handle.getFile();
                if (filePath) (file as any).path = filePath;
                if (isMediaFile(file)) {
                  const imported = await mediaStore.importFilesWithHandles([
                    { file, handle, absolutePath: filePath },
                  ]);
                  if (imported.length > 0) {
                    addClip(newTrackId, file, startTime, cachedDuration, imported[0].id);
                    log.debug('Imported file with handle:', { name: file.name, absolutePath: filePath });
                  }
                  return;
                }
              }
            } catch (err) {
              log.warn('Could not get file handle, falling back:', err);
            }
          }

          // Fallback to regular file (no handle)
          const file = item.getAsFile();
          if (file && filePath) (file as any).path = filePath;
          if (file && isMediaFile(file)) {
            const importedFile = await mediaStore.importFile(file);
            addClip(newTrackId, file, startTime, cachedDuration, importedFile?.id);
          }
        }
      }
    },
    [scrollX, pixelToTime, addTrack, addCompClip, addClip, addTextClip, addSolidClip, externalDrag, timelineRef]
  );

  // Handle external file drop on track
  const handleTrackDrop = useCallback(
    async (e: React.DragEvent, trackId: string) => {
      e.preventDefault();

      const cachedDuration =
        externalDrag?.duration ?? dragMetadataCacheRef.current?.duration;

      dragCounterRef.current = 0;
      setExternalDrag(null);

      // Get track type for validation
      const targetTrack = tracks.find((t) => t.id === trackId);
      const isVideoTrack = targetTrack?.type === 'video';
      const isAudioTrack = targetTrack?.type === 'audio';

      const compositionId = e.dataTransfer.getData('application/x-composition-id');
      if (compositionId) {
        const mediaStore = useMediaStore.getState();
        const comp = mediaStore.compositions.find((c) => c.id === compositionId);
        if (comp) {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addCompClip(trackId, comp, Math.max(0, startTime));
          return;
        }
      }

      // Handle text item drag from media panel (skipMediaItem=true since it already exists)
      const textItemId = e.dataTransfer.getData('application/x-text-item-id');
      if (textItemId) {
        const mediaStore = useMediaStore.getState();
        const textItem = mediaStore.textItems.find((t) => t.id === textItemId);
        if (textItem && isVideoTrack) {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addTextClip(trackId, Math.max(0, startTime), textItem.duration, true);
          return;
        }
      }

      // Handle solid item drag from media panel (skipMediaItem=true since it already exists)
      const solidItemId = e.dataTransfer.getData('application/x-solid-item-id');
      if (solidItemId) {
        const mediaStore = useMediaStore.getState();
        const solidItem = mediaStore.solidItems.find((s) => s.id === solidItemId);
        if (solidItem && isVideoTrack) {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addSolidClip(trackId, Math.max(0, startTime), solidItem.color, solidItem.duration, true);
          return;
        }
      }

      const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          const fileIsAudio = isAudioFile(mediaFile.file);
          if (fileIsAudio && isVideoTrack) {
            log.debug('Audio files can only be dropped on audio tracks');
            return;
          }
          if (!fileIsAudio && isAudioTrack) {
            log.debug('Video/image files can only be dropped on video tracks');
            return;
          }

          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addClip(trackId, mediaFile.file, Math.max(0, startTime), mediaFile.duration, mediaFileId);
          return;
        }
      }

      // Handle external file drop
      const items = e.dataTransfer.items;
      const filePath = extractFilePath(e);

      log.debug('External drop', { items: items?.length, types: Array.from(e.dataTransfer.types) });
      log.debug('Final file path:', filePath || 'NOT AVAILABLE');

      if (items && items.length > 0) {
        const item = items[0];
        log.debug('Item details:', { kind: item.kind, type: item.type });
        if (item.kind === 'file') {
          // Capture rect before async operations (e.currentTarget becomes null after await)
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = Math.max(0, pixelToTime(x));
          const mediaStore = useMediaStore.getState();

          // Try to get file handle (File System Access API)
          if ('getAsFileSystemHandle' in item) {
            try {
              const handle = await (item as any).getAsFileSystemHandle();
              if (handle && handle.kind === 'file') {
                const file = await handle.getFile();
                // Attach file path if we got it from URI list
                if (filePath) {
                  (file as any).path = filePath;
                }
                log.debug('File from handle:', { name: file.name, type: file.type, size: file.size, path: filePath });
                if (isMediaFile(file)) {
                  // Validate track type
                  const fileIsAudio = isAudioFile(file);
                  if (fileIsAudio && isVideoTrack) {
                    log.debug('Audio files can only be dropped on audio tracks');
                    return;
                  }
                  if (!fileIsAudio && isAudioTrack) {
                    log.debug('Video/image files can only be dropped on video tracks');
                    return;
                  }

                  const imported = await mediaStore.importFilesWithHandles([{ file, handle, absolutePath: filePath }]);
                  if (imported.length > 0) {
                    addClip(trackId, file, startTime, cachedDuration, imported[0].id);
                    log.debug('Imported file with handle:', { name: file.name, absolutePath: filePath });
                  }
                  return;
                }
              }
            } catch (err) {
              log.warn('Could not get file handle, falling back:', err);
            }
          }

          // Fallback to regular file (no handle)
          const file = item.getAsFile();
          if (file && filePath) {
            (file as any).path = filePath;
          }
          log.debug('Fallback file:', { name: file?.name, type: file?.type, path: filePath });
          if (file && isMediaFile(file)) {
            const fileIsAudio = isAudioFile(file);
            if (fileIsAudio && isVideoTrack) {
              log.debug('Audio files can only be dropped on audio tracks');
              return;
            }
            if (!fileIsAudio && isAudioTrack) {
              log.debug('Video/image files can only be dropped on video tracks');
              return;
            }

            const importedFile = await mediaStore.importFile(file);
            addClip(trackId, file, startTime, cachedDuration, importedFile?.id);
          }
        }
      }
    },
    [scrollX, pixelToTime, addCompClip, addClip, addTextClip, addSolidClip, externalDrag, tracks, timelineRef]
  );

  return {
    externalDrag,
    setExternalDrag,
    dragCounterRef,
    handleTrackDragEnter,
    handleTrackDragOver,
    handleTrackDragLeave,
    handleTrackDrop,
    handleNewTrackDragOver,
    handleNewTrackDrop,
  };
}
