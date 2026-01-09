// Timeline component - Main orchestrator for video editing timeline
// Composes TimelineRuler, TimelineControls, TimelineHeader, TimelineTrack, TimelineClip, TimelineKeyframes

import { useRef, useState, useCallback, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { AnimatableProperty, Effect, TimelineClip as TimelineClipType } from '../../types';
import { useMediaStore } from '../../stores/mediaStore';
import { useMixerStore } from '../../stores/mixerStore';
import { engine } from '../../engine/WebGPUEngine';
import { proxyFrameCache } from '../../services/proxyFrameCache';

import { TimelineRuler } from './TimelineRuler';
import { TimelineControls } from './TimelineControls';
import { TimelineHeader } from './TimelineHeader';
import { TimelineTrack } from './TimelineTrack';
import { TimelineClip } from './TimelineClip';
import { TimelineKeyframes } from './TimelineKeyframes';
import {
  ALL_BLEND_MODES,
  RAM_PREVIEW_IDLE_DELAY,
  PROXY_IDLE_DELAY,
  DURATION_CHECK_TIMEOUT,
} from './constants';
import type {
  ClipDragState,
  ClipTrimState,
  MarkerDragState,
  ExternalDragState,
  ContextMenuState,
} from './types';

export function Timeline() {
  const {
    tracks,
    clips,
    playheadPosition,
    duration,
    zoom,
    scrollX,
    isPlaying,
    selectedClipId,
    inPoint,
    outPoint,
    addTrack,
    addClip,
    addCompClip,
    moveClip,
    trimClip,
    removeClip,
    selectClip,
    setPlayheadPosition,
    setZoom,
    setScrollX,
    play,
    pause,
    stop,
    setInPoint,
    setOutPoint,
    setInPointAtPlayhead,
    setOutPointAtPlayhead,
    clearInOut,
    getSnappedPosition,
    loopPlayback,
    toggleLoopPlayback,
    ramPreviewProgress,
    ramPreviewRange,
    isRamPreviewing,
    startRamPreview,
    cancelRamPreview,
    getCachedRanges,
    isDraggingPlayhead,
    setDraggingPlayhead,
    ramPreviewEnabled,
    toggleRamPreviewEnabled,
    splitClipAtPlayhead,
    toggleClipReverse,
    updateClipTransform,
    getInterpolatedTransform,
    getInterpolatedEffects,
    isTrackExpanded,
    toggleTrackExpanded,
    isTrackPropertyGroupExpanded,
    toggleTrackPropertyGroupExpanded,
    getExpandedTrackHeight,
    getClipKeyframes,
    selectKeyframe,
    selectedKeyframeIds,
    hasKeyframes,
    trackHasKeyframes,
    clipKeyframes,
  } = useTimelineStore();

  const {
    getActiveComposition,
    getOpenCompositions,
    proxyEnabled,
    setProxyEnabled,
    files: mediaFiles,
    currentlyGeneratingProxyId,
    proxyFolderName,
    pickProxyFolder,
    showInExplorer,
    getNextFileNeedingProxy,
    generateProxy,
  } = useMediaStore();
  const activeComposition = getActiveComposition();
  const openCompositions = getOpenCompositions();

  const timelineRef = useRef<HTMLDivElement>(null);
  const trackLanesRef = useRef<HTMLDivElement>(null);

  // Premiere-style clip dragging state
  const [clipDrag, setClipDrag] = useState<ClipDragState | null>(null);
  const clipDragRef = useRef(clipDrag);
  clipDragRef.current = clipDrag;

  // Clip trimming state
  const [clipTrim, setClipTrim] = useState<ClipTrimState | null>(null);
  const clipTrimRef = useRef(clipTrim);
  clipTrimRef.current = clipTrim;

  // In/Out marker drag state
  const [markerDrag, setMarkerDrag] = useState<MarkerDragState | null>(null);

  // External file drag preview state
  const [externalDrag, setExternalDrag] = useState<ExternalDragState | null>(null);

  // Context menu state for clip right-click
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const dragCounterRef = useRef(0);
  const dragDurationCacheRef = useRef<{ url: string; duration: number } | null>(null);

  // Time conversion helpers
  const timeToPixel = useCallback((time: number) => time * zoom, [zoom]);
  const pixelToTime = useCallback((pixel: number) => pixel / zoom, [zoom]);

  // Format time as MM:SS.ms
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  }, []);

  // Get clips at time helper
  const getClipsAtTime = useCallback(
    (time: number) => {
      return clips.filter((c) => time >= c.startTime && time < c.startTime + c.duration);
    },
    [clips]
  );

  // Keyboard shortcuts (global, works regardless of focus)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Space: toggle play/pause
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        if (isPlaying) {
          pause();
        } else {
          play();
        }
        return;
      }

      // I: set In point at playhead
      if (e.key === 'i' || e.key === 'I') {
        e.preventDefault();
        setInPointAtPlayhead();
        return;
      }

      // O: set Out point at playhead
      if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        setOutPointAtPlayhead();
        return;
      }

      // X: clear In/Out points
      if (e.key === 'x' || e.key === 'X') {
        e.preventDefault();
        clearInOut();
        return;
      }

      // L: toggle loop playback
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        toggleLoopPlayback();
        return;
      }

      // Delete/Backspace: remove selected clip from timeline
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        if (selectedClipId) {
          removeClip(selectedClipId);
        }
        return;
      }

      // C: Cut/split clip at playhead position
      if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        splitClipAtPlayhead();
        return;
      }

      // Shift + "+": Cycle through blend modes (forward)
      // Shift + "-": Cycle through blend modes (backward)
      if (
        e.shiftKey &&
        (e.key === '+' || e.key === '=' || e.key === '-' || e.key === '_')
      ) {
        e.preventDefault();
        if (selectedClipId) {
          const clip = clips.find((c) => c.id === selectedClipId);
          if (clip) {
            const currentMode = clip.transform.blendMode;
            const currentIndex = ALL_BLEND_MODES.indexOf(currentMode);
            const direction = e.key === '+' || e.key === '=' ? 1 : -1;
            const nextIndex =
              (currentIndex + direction + ALL_BLEND_MODES.length) %
              ALL_BLEND_MODES.length;
            const nextMode = ALL_BLEND_MODES[nextIndex];
            updateClipTransform(selectedClipId, { blendMode: nextMode });
          }
        }
        return;
      }

      // Arrow Left: Move playhead one frame backward
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (activeComposition) {
          const frameDuration = 1 / activeComposition.frameRate;
          const newPosition = Math.max(0, playheadPosition - frameDuration);
          setPlayheadPosition(newPosition);
        }
        return;
      }

      // Arrow Right: Move playhead one frame forward
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (activeComposition) {
          const frameDuration = 1 / activeComposition.frameRate;
          const newPosition = Math.min(duration, playheadPosition + frameDuration);
          setPlayheadPosition(newPosition);
        }
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    isPlaying,
    play,
    pause,
    setInPointAtPlayhead,
    setOutPointAtPlayhead,
    clearInOut,
    toggleLoopPlayback,
    selectedClipId,
    removeClip,
    splitClipAtPlayhead,
    clips,
    updateClipTransform,
    activeComposition,
    playheadPosition,
    duration,
    setPlayheadPosition,
  ]);

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
  }, [contextMenu]);

  // Handle right-click on clip
  const handleClipContextMenu = useCallback(
    (e: React.MouseEvent, clipId: string) => {
      e.preventDefault();
      e.stopPropagation();
      selectClip(clipId);
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        clipId,
      });
    },
    [selectClip]
  );

  // Get the media file for a clip (helper function)
  const getMediaFileForClip = useCallback(
    (clipId: string) => {
      const clip = clips.find((c) => c.id === clipId);
      if (!clip) return null;

      const mediaStore = useMediaStore.getState();
      return mediaStore.files.find(
        (f) =>
          f.id === clip.source?.mediaFileId ||
          f.name === clip.name ||
          f.name === clip.name.replace(' (Audio)', '')
      );
    },
    [clips]
  );

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

  // Auto-start RAM Preview after 2 seconds of idle (like After Effects)
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cancel RAM preview when user starts playing or scrubbing (keep cached frames)
  useEffect(() => {
    if ((isPlaying || isDraggingPlayhead) && isRamPreviewing) {
      cancelRamPreview();
    }
  }, [isPlaying, isDraggingPlayhead, isRamPreviewing, cancelRamPreview]);

  // Check for idle state and auto-start RAM preview
  useEffect(() => {
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }

    if (
      !ramPreviewEnabled ||
      isPlaying ||
      isRamPreviewing ||
      isDraggingPlayhead ||
      clips.length === 0
    ) {
      return;
    }

    const renderStart = inPoint ?? 0;
    const renderEnd =
      outPoint ?? Math.max(...clips.map((c) => c.startTime + c.duration));

    if (renderEnd - renderStart < 0.1) {
      return;
    }

    if (
      ramPreviewRange &&
      ramPreviewRange.start <= renderStart &&
      ramPreviewRange.end >= renderEnd
    ) {
      return;
    }

    idleTimerRef.current = setTimeout(() => {
      const state = useTimelineStore.getState();
      if (state.ramPreviewEnabled && !state.isPlaying && !state.isRamPreviewing) {
        startRamPreview();
      }
    }, RAM_PREVIEW_IDLE_DELAY);

    return () => {
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
        idleTimerRef.current = null;
      }
    };
  }, [
    ramPreviewEnabled,
    isPlaying,
    isRamPreviewing,
    isDraggingPlayhead,
    inPoint,
    outPoint,
    ramPreviewRange,
    clips,
    startRamPreview,
  ]);

  // Auto-generate proxies after 3 seconds of idle
  const proxyIdleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (proxyIdleTimerRef.current) {
      clearTimeout(proxyIdleTimerRef.current);
      proxyIdleTimerRef.current = null;
    }

    if (!proxyEnabled || isPlaying || currentlyGeneratingProxyId || isDraggingPlayhead) {
      return;
    }

    proxyIdleTimerRef.current = setTimeout(() => {
      const mediaStore = useMediaStore.getState();
      if (mediaStore.proxyEnabled && !mediaStore.currentlyGeneratingProxyId) {
        const nextFile = mediaStore.getNextFileNeedingProxy();
        if (nextFile) {
          console.log('[Proxy] Auto-starting proxy generation for:', nextFile.name);
          mediaStore.generateProxy(nextFile.id);
        }
      }
    }, PROXY_IDLE_DELAY);

    return () => {
      if (proxyIdleTimerRef.current) {
        clearTimeout(proxyIdleTimerRef.current);
        proxyIdleTimerRef.current = null;
      }
    };
  }, [
    proxyEnabled,
    isPlaying,
    currentlyGeneratingProxyId,
    isDraggingPlayhead,
    clips,
    getNextFileNeedingProxy,
    generateProxy,
  ]);

  // Track last seek time to throttle during scrubbing
  const lastSeekRef = useRef<{ [clipId: string]: number }>({});
  const pendingSeekRef = useRef<{ [clipId: string]: number }>({});

  // Apply pending seeks when scrubbing stops
  useEffect(() => {
    if (isDraggingPlayhead) return;

    Object.entries(pendingSeekRef.current).forEach(([clipId, seekTime]) => {
      const clip = clips.find((c) => c.id === clipId);
      if (clip?.source?.videoElement) {
        clip.source.videoElement.currentTime = seekTime;
      }
    });
    pendingSeekRef.current = {};
  }, [isDraggingPlayhead, clips]);

  // Track which clips are currently active (to detect clip changes)
  const activeClipIdsRef = useRef<string>('');

  // Track current proxy frames for each clip (for smooth proxy playback)
  const proxyFramesRef = useRef<
    Map<string, { frameIndex: number; image: HTMLImageElement }>
  >(new Map());
  const proxyLoadingRef = useRef<Set<string>>(new Set());

  // Sync timeline playback with Preview - update mixer layers based on clips at playhead
  useEffect(() => {
    if (isRamPreviewing) {
      return;
    }

    if (
      ramPreviewRange &&
      playheadPosition >= ramPreviewRange.start &&
      playheadPosition <= ramPreviewRange.end
    ) {
      if (engine.renderCachedFrame(playheadPosition)) {
        return;
      }
    }

    let clipsAtTime = getClipsAtTime(playheadPosition);

    if (clipDrag) {
      const draggedClipId = clipDrag.clipId;
      const rawPixelX = clipDrag.currentX
        ? clipDrag.currentX -
          (timelineRef.current?.getBoundingClientRect().left || 0) +
          scrollX -
          clipDrag.grabOffsetX
        : 0;
      const tempStartTime =
        clipDrag.snappedTime ??
        (clipDrag.currentX ? Math.max(0, rawPixelX / zoom) : null);

      if (tempStartTime !== null) {
        const modifiedClips = clips.map((c) => {
          if (c.id === draggedClipId) {
            return { ...c, startTime: tempStartTime, trackId: clipDrag.currentTrackId };
          }
          return c;
        });
        clipsAtTime = modifiedClips.filter(
          (c) =>
            playheadPosition >= c.startTime &&
            playheadPosition < c.startTime + c.duration
        );
      }
    }

    const currentActiveIds = clipsAtTime
      .filter(
        (c) =>
          c.source?.type === 'video' ||
          c.source?.type === 'image' ||
          c.isComposition
      )
      .map((c) => c.id)
      .sort()
      .join(',');

    const mediaStoreState = useMediaStore.getState();
    const hasActiveProxies =
      mediaStoreState.proxyEnabled &&
      clipsAtTime.some((clip) => {
        const mediaFile = mediaStoreState.files.find(
          (f) => f.id === clip.source?.mediaFileId || f.name === clip.name
        );
        return (
          mediaFile?.proxyStatus === 'ready' ||
          (mediaFile?.proxyStatus === 'generating' &&
            (mediaFile?.proxyProgress || 0) > 0)
        );
      });

    if (
      isPlaying &&
      !hasActiveProxies &&
      activeClipIdsRef.current.startsWith('playing:')
    ) {
      const prevActiveIds = activeClipIdsRef.current.slice(8);
      if (prevActiveIds === currentActiveIds) {
        clipsAtTime.forEach((clip) => {
          if (clip.source?.videoElement?.paused) {
            clip.source.videoElement.play().catch(() => {});
          }
        });
        return;
      }
    }

    if (isPlaying) {
      clipsAtTime.forEach((clip) => {
        if (clip.source?.videoElement?.paused) {
          clip.source.videoElement.play().catch(() => {});
        }
      });
      activeClipIdsRef.current = 'playing:' + currentActiveIds;
    } else {
      activeClipIdsRef.current = '';
    }

    const currentLayers = useMixerStore.getState().layers;
    const videoTracks = tracks.filter((t) => t.type === 'video');
    const anyVideoSolo = videoTracks.some((t) => t.solo);

    const isVideoTrackVisible = (track: (typeof videoTracks)[0]) => {
      if (!track.visible) return false;
      if (anyVideoSolo) return track.solo;
      return true;
    };

    const newLayers = [...currentLayers];
    let layersChanged = false;

    const effectsChanged = (
      layerEffects: Effect[] | undefined,
      clipEffects: Effect[] | undefined
    ): boolean => {
      const le = layerEffects || [];
      const ce = clipEffects || [];
      if (le.length !== ce.length) return true;
      for (let i = 0; i < le.length; i++) {
        if (le[i].id !== ce[i].id || le[i].enabled !== ce[i].enabled) return true;
        const lp = le[i].params;
        const cp = ce[i].params;
        const lKeys = Object.keys(lp);
        const cKeys = Object.keys(cp);
        if (lKeys.length !== cKeys.length) return true;
        for (const key of lKeys) {
          if (lp[key] !== cp[key]) return true;
        }
      }
      return false;
    };

    const getVideoFromClip = (
      clip: (typeof clips)[0],
      clipTime: number
    ): {
      video: HTMLVideoElement | null;
      transform: (typeof clip)['transform'];
      effects: Effect[];
    } => {
      const interpolatedTransform = getInterpolatedTransform(clip.id, clipTime);
      const interpolatedEffects = getInterpolatedEffects(clip.id, clipTime);

      if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        const nestedTime = clipTime;
        const nestedVideoTracks = (clip.nestedTracks || []).filter(
          (t) => t.type === 'video' && t.visible
        );

        for (const nestedTrack of nestedVideoTracks) {
          const nestedClip = clip.nestedClips.find(
            (nc) =>
              nc.trackId === nestedTrack.id &&
              nestedTime >= nc.startTime &&
              nestedTime < nc.startTime + nc.duration &&
              nc.source?.videoElement
          );

          if (nestedClip?.source?.videoElement) {
            const nestedLocalTime = nestedTime - nestedClip.startTime;
            const nestedClipTime = nestedClip.reversed
              ? nestedClip.outPoint - nestedLocalTime
              : nestedLocalTime + nestedClip.inPoint;
            const video = nestedClip.source.videoElement;

            const timeDiff = Math.abs(video.currentTime - nestedClipTime);
            if (timeDiff > 0.05) {
              video.currentTime = nestedClipTime;
            }

            if (isPlaying && video.paused) {
              video.play().catch(() => {});
            } else if (!isPlaying && !video.paused) {
              video.pause();
            }

            return {
              video,
              transform: interpolatedTransform,
              effects: interpolatedEffects,
            };
          }
        }
        return {
          video: null,
          transform: interpolatedTransform,
          effects: interpolatedEffects,
        };
      }

      return {
        video: clip.source?.videoElement || null,
        transform: interpolatedTransform,
        effects: interpolatedEffects,
      };
    };

    videoTracks.forEach((track, layerIndex) => {
      const clip = clipsAtTime.find((c) => c.trackId === track.id);
      const layer = currentLayers[layerIndex];

      if (clip?.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const { video, transform, effects } = getVideoFromClip(clip, clipTime);

        if (video) {
          const needsUpdate =
            !layer ||
            layer.source?.videoElement !== video ||
            layer.opacity !== transform.opacity ||
            layer.blendMode !== transform.blendMode ||
            layer.position.x !== transform.position.x ||
            layer.position.y !== transform.position.y ||
            layer.scale.x !== transform.scale.x ||
            layer.scale.y !== transform.scale.y ||
            (layer.rotation as { z?: number })?.z !==
              (transform.rotation.z * Math.PI) / 180 ||
            (layer.rotation as { x?: number })?.x !==
              (transform.rotation.x * Math.PI) / 180 ||
            (layer.rotation as { y?: number })?.y !==
              (transform.rotation.y * Math.PI) / 180 ||
            effectsChanged(layer.effects, effects);

          if (needsUpdate) {
            newLayers[layerIndex] = {
              id: `timeline_layer_${layerIndex}`,
              name: clip.name,
              visible: isVideoTrackVisible(track),
              opacity: transform.opacity,
              blendMode: transform.blendMode,
              source: {
                type: 'video',
                videoElement: video,
              },
              effects: effects,
              position: { x: transform.position.x, y: transform.position.y },
              scale: { x: transform.scale.x, y: transform.scale.y },
              rotation: {
                x: (transform.rotation.x * Math.PI) / 180,
                y: (transform.rotation.y * Math.PI) / 180,
                z: (transform.rotation.z * Math.PI) / 180,
              },
            };
            layersChanged = true;
          }
        } else {
          if (layer?.source) {
            newLayers[layerIndex] = undefined as unknown as (typeof newLayers)[0];
            layersChanged = true;
          }
        }
      } else if (clip?.source?.videoElement) {
        const clipLocalTime = playheadPosition - clip.startTime;
        const keyframeLocalTime = clip.reversed
          ? clip.duration - clipLocalTime
          : clipLocalTime;
        const clipTime = clip.reversed
          ? clip.outPoint - clipLocalTime
          : clipLocalTime + clip.inPoint;
        const video = clip.source.videoElement;
        const webCodecsPlayer = clip.source.webCodecsPlayer;
        const timeDiff = Math.abs(video.currentTime - clipTime);

        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find(
          (f) => f.name === clip.name || clip.source?.mediaFileId === f.id
        );
        const proxyFps = mediaFile?.proxyFps || 30;

        const frameIndex = Math.floor(clipTime * proxyFps);
        let useProxy = false;

        if (mediaStore.proxyEnabled && mediaFile?.proxyFps) {
          if (mediaFile.proxyStatus === 'ready') {
            useProxy = true;
          } else if (
            mediaFile.proxyStatus === 'generating' &&
            (mediaFile.proxyProgress || 0) > 0
          ) {
            const totalFrames = Math.ceil(
              (mediaFile.duration || 10) * proxyFps
            );
            const maxGeneratedFrame = Math.floor(
              totalFrames * ((mediaFile.proxyProgress || 0) / 100)
            );
            useProxy = frameIndex < maxGeneratedFrame;
          }
        }

        if (useProxy && mediaFile) {
          const cacheKey = `${mediaFile.id}_${clip.id}`;
          const cached = proxyFramesRef.current.get(cacheKey);

          if (!video.muted) {
            video.muted = true;
          }
          if (isPlaying && video.paused) {
            video.play().catch(() => {});
          } else if (!isPlaying && !video.paused) {
            video.pause();
          }

          if (!isPlaying) {
            const timeDiff2 = Math.abs(video.currentTime - clipTime);
            if (timeDiff2 > 0.1) {
              video.currentTime = clipTime;
            }
          }

          const loadKey = `${mediaFile.id}_${frameIndex}`;
          const cachedInService = proxyFrameCache.getCachedFrame(
            mediaFile.id,
            frameIndex,
            proxyFps
          );
          const interpolatedEffectsForProxy = getInterpolatedEffects(
            clip.id,
            keyframeLocalTime
          );

          if (cachedInService) {
            proxyFramesRef.current.set(cacheKey, {
              frameIndex,
              image: cachedInService,
            });

            const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
            newLayers[layerIndex] = {
              id: `timeline_layer_${layerIndex}`,
              name: clip.name,
              visible: isVideoTrackVisible(track),
              opacity: transform.opacity,
              blendMode: transform.blendMode,
              source: {
                type: 'image',
                imageElement: cachedInService,
              },
              effects: interpolatedEffectsForProxy,
              position: { x: transform.position.x, y: transform.position.y },
              scale: { x: transform.scale.x, y: transform.scale.y },
              rotation: {
                x: (transform.rotation.x * Math.PI) / 180,
                y: (transform.rotation.y * Math.PI) / 180,
                z: (transform.rotation.z * Math.PI) / 180,
              },
            };
            layersChanged = true;
          } else if (!cached || cached.frameIndex !== frameIndex) {
            if (!proxyLoadingRef.current.has(loadKey)) {
              proxyLoadingRef.current.add(loadKey);

              const capturedLayerIndex = layerIndex;
              const capturedTransform = getInterpolatedTransform(
                clip.id,
                keyframeLocalTime
              );
              const capturedTrackVisible = isVideoTrackVisible(track);
              const capturedClipName = clip.name;
              const capturedEffects = interpolatedEffectsForProxy;

              proxyFrameCache
                .getFrame(mediaFile.id, clipTime, proxyFps)
                .then((image) => {
                  proxyLoadingRef.current.delete(loadKey);
                  if (image) {
                    proxyFramesRef.current.set(cacheKey, { frameIndex, image });

                    const currentLayers2 = useMixerStore.getState().layers;
                    const updatedLayers = [...currentLayers2];
                    updatedLayers[capturedLayerIndex] = {
                      id: `timeline_layer_${capturedLayerIndex}`,
                      name: capturedClipName,
                      visible: capturedTrackVisible,
                      opacity: capturedTransform.opacity,
                      blendMode: capturedTransform.blendMode,
                      source: {
                        type: 'image',
                        imageElement: image,
                      },
                      effects: capturedEffects,
                      position: {
                        x: capturedTransform.position.x,
                        y: capturedTransform.position.y,
                      },
                      scale: {
                        x: capturedTransform.scale.x,
                        y: capturedTransform.scale.y,
                      },
                      rotation: {
                        x: (capturedTransform.rotation.x * Math.PI) / 180,
                        y: (capturedTransform.rotation.y * Math.PI) / 180,
                        z: (capturedTransform.rotation.z * Math.PI) / 180,
                      },
                    };
                    useMixerStore.setState({ layers: updatedLayers });
                  }
                });
            }

            if (cached?.image) {
              const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
              newLayers[layerIndex] = {
                id: `timeline_layer_${layerIndex}`,
                name: clip.name,
                visible: isVideoTrackVisible(track),
                opacity: transform.opacity,
                blendMode: transform.blendMode,
                source: {
                  type: 'image',
                  imageElement: cached.image,
                },
                effects: interpolatedEffectsForProxy,
                position: { x: transform.position.x, y: transform.position.y },
                scale: { x: transform.scale.x, y: transform.scale.y },
                rotation: {
                  x: (transform.rotation.x * Math.PI) / 180,
                  y: (transform.rotation.y * Math.PI) / 180,
                  z: (transform.rotation.z * Math.PI) / 180,
                },
              };
              layersChanged = true;
            }
          } else if (cached?.image) {
            const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
            const needsUpdate =
              !layer ||
              layer.source?.imageElement !== cached.image ||
              layer.source?.type !== 'image' ||
              effectsChanged(layer.effects, interpolatedEffectsForProxy);

            if (needsUpdate) {
              newLayers[layerIndex] = {
                id: `timeline_layer_${layerIndex}`,
                name: clip.name,
                visible: isVideoTrackVisible(track),
                opacity: transform.opacity,
                blendMode: transform.blendMode,
                source: {
                  type: 'image',
                  imageElement: cached.image,
                },
                effects: interpolatedEffectsForProxy,
                position: { x: transform.position.x, y: transform.position.y },
                scale: { x: transform.scale.x, y: transform.scale.y },
                rotation: {
                  x: (transform.rotation.x * Math.PI) / 180,
                  y: (transform.rotation.y * Math.PI) / 180,
                  z: (transform.rotation.z * Math.PI) / 180,
                },
              };
              layersChanged = true;
            }
          }
        } else {
          if (webCodecsPlayer) {
            const wcTimeDiff = Math.abs(webCodecsPlayer.currentTime - clipTime);
            if (wcTimeDiff > 0.05) {
              webCodecsPlayer.seek(clipTime);
            }
          }

          if (clip.reversed) {
            if (!video.paused) {
              video.pause();
            }
            const seekThreshold = isDraggingPlayhead ? 0.1 : 0.03;
            if (timeDiff > seekThreshold) {
              const now = performance.now();
              const lastSeek = lastSeekRef.current[clip.id] || 0;
              if (now - lastSeek > 33) {
                video.currentTime = clipTime;
                lastSeekRef.current[clip.id] = now;
              }
            }
          } else {
            if (isPlaying && video.paused) {
              video.play().catch(() => {});
            } else if (!isPlaying && !video.paused) {
              video.pause();
            }

            if (!isPlaying) {
              const seekThreshold = isDraggingPlayhead ? 0.1 : 0.05;

              if (timeDiff > seekThreshold) {
                const now = performance.now();
                const lastSeek = lastSeekRef.current[clip.id] || 0;

                if (isDraggingPlayhead && now - lastSeek < 80) {
                  pendingSeekRef.current[clip.id] = clipTime;
                } else {
                  if (isDraggingPlayhead && 'fastSeek' in video) {
                    video.fastSeek(clipTime);
                  } else {
                    video.currentTime = clipTime;
                  }
                  lastSeekRef.current[clip.id] = now;
                  delete pendingSeekRef.current[clip.id];
                }
              }
            }
          }

          const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
          const videoInterpolatedEffects = getInterpolatedEffects(
            clip.id,
            keyframeLocalTime
          );
          const needsUpdate =
            !layer ||
            layer.source?.videoElement !== video ||
            layer.source?.webCodecsPlayer !== webCodecsPlayer ||
            layer.opacity !== transform.opacity ||
            layer.blendMode !== transform.blendMode ||
            layer.position.x !== transform.position.x ||
            layer.position.y !== transform.position.y ||
            layer.scale.x !== transform.scale.x ||
            layer.scale.y !== transform.scale.y ||
            (layer.rotation as { z?: number })?.z !==
              (transform.rotation.z * Math.PI) / 180 ||
            (layer.rotation as { x?: number })?.x !==
              (transform.rotation.x * Math.PI) / 180 ||
            (layer.rotation as { y?: number })?.y !==
              (transform.rotation.y * Math.PI) / 180 ||
            effectsChanged(layer.effects, videoInterpolatedEffects);

          if (needsUpdate) {
            newLayers[layerIndex] = {
              id: `timeline_layer_${layerIndex}`,
              name: clip.name,
              visible: isVideoTrackVisible(track),
              opacity: transform.opacity,
              blendMode: transform.blendMode,
              source: {
                type: 'video',
                videoElement: video,
                webCodecsPlayer: webCodecsPlayer,
              },
              effects: videoInterpolatedEffects,
              position: { x: transform.position.x, y: transform.position.y },
              scale: { x: transform.scale.x, y: transform.scale.y },
              rotation: {
                x: (transform.rotation.x * Math.PI) / 180,
                y: (transform.rotation.y * Math.PI) / 180,
                z: (transform.rotation.z * Math.PI) / 180,
              },
            };
            layersChanged = true;
          }
        }
      } else if (clip?.source?.imageElement) {
        const img = clip.source.imageElement;
        const imageClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, imageClipLocalTime);
        const imageInterpolatedEffects = getInterpolatedEffects(
          clip.id,
          imageClipLocalTime
        );
        const needsUpdate =
          !layer ||
          layer.source?.imageElement !== img ||
          layer.opacity !== transform.opacity ||
          layer.blendMode !== transform.blendMode ||
          layer.position.x !== transform.position.x ||
          layer.position.y !== transform.position.y ||
          layer.scale.x !== transform.scale.x ||
          layer.scale.y !== transform.scale.y ||
          (layer.rotation as { z?: number })?.z !==
            (transform.rotation.z * Math.PI) / 180 ||
          (layer.rotation as { x?: number })?.x !==
            (transform.rotation.x * Math.PI) / 180 ||
          (layer.rotation as { y?: number })?.y !==
            (transform.rotation.y * Math.PI) / 180 ||
          effectsChanged(layer.effects, imageInterpolatedEffects);

        if (needsUpdate) {
          newLayers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: isVideoTrackVisible(track),
            opacity: transform.opacity,
            blendMode: transform.blendMode,
            source: {
              type: 'image',
              imageElement: img,
            },
            effects: imageInterpolatedEffects,
            position: { x: transform.position.x, y: transform.position.y },
            scale: { x: transform.scale.x, y: transform.scale.y },
            rotation: {
              x: (transform.rotation.x * Math.PI) / 180,
              y: (transform.rotation.y * Math.PI) / 180,
              z: (transform.rotation.z * Math.PI) / 180,
            },
          };
          layersChanged = true;
        }
      } else {
        if (layer?.source) {
          newLayers[layerIndex] = undefined as unknown as (typeof newLayers)[0];
          layersChanged = true;
        }
      }
    });

    if (layersChanged) {
      useMixerStore.setState({ layers: newLayers });
    }

    const audioTracks = tracks.filter((t) => t.type === 'audio');
    const anyAudioSolo = audioTracks.some((t) => t.solo);

    const isAudioTrackMuted = (track: (typeof audioTracks)[0]) => {
      if (track.muted) return true;
      if (anyAudioSolo) return !track.solo;
      return false;
    };

    audioTracks.forEach((track) => {
      const clip = clipsAtTime.find((c) => c.trackId === track.id);

      if (clip?.source?.audioElement) {
        const audio = clip.source.audioElement;
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const timeDiff = Math.abs(audio.currentTime - clipTime);

        if (timeDiff > 0.1 && !isDraggingPlayhead) {
          audio.currentTime = clipTime;
        }

        const effectivelyMuted = isAudioTrackMuted(track);
        audio.muted = effectivelyMuted;

        const shouldPlay = isPlaying && !effectivelyMuted && !isDraggingPlayhead;
        if (shouldPlay && audio.paused) {
          audio.play().catch(() => {});
        } else if (!shouldPlay && !audio.paused) {
          audio.pause();
        }
      }
    });

    clips.forEach((clip) => {
      if (clip.source?.audioElement) {
        const isAtPlayhead = clipsAtTime.some((c) => c.id === clip.id);
        if (!isAtPlayhead && !clip.source.audioElement.paused) {
          clip.source.audioElement.pause();
        }
      }
    });
  }, [
    playheadPosition,
    clips,
    tracks,
    isPlaying,
    isDraggingPlayhead,
    ramPreviewRange,
    isRamPreviewing,
    clipKeyframes,
    clipDrag,
    zoom,
    scrollX,
    getClipsAtTime,
    getInterpolatedTransform,
    getInterpolatedEffects,
  ]);

  // Playback loop
  useEffect(() => {
    if (!isPlaying) return;

    let intervalId: ReturnType<typeof setInterval>;

    const getActiveVideoClip = () => {
      const state = useTimelineStore.getState();
      const pos = state.playheadPosition;
      for (const clip of state.clips) {
        if (
          clip.source?.videoElement &&
          pos >= clip.startTime &&
          pos < clip.startTime + clip.duration
        ) {
          return clip;
        }
      }
      return null;
    };

    const updatePlayhead = () => {
      const state = useTimelineStore.getState();
      const { duration: dur, inPoint: ip, outPoint: op, loopPlayback: lp, pause: ps } = state;
      const effectiveEnd = op !== null ? op : dur;
      const effectiveStart = ip !== null ? ip : 0;

      const activeClip = getActiveVideoClip();
      let newPosition: number;

      if (activeClip?.source?.videoElement) {
        const video = activeClip.source.videoElement;
        newPosition = activeClip.startTime + video.currentTime - activeClip.inPoint;
      } else {
        newPosition = state.playheadPosition + 0.066;
      }

      if (newPosition >= effectiveEnd) {
        if (lp) {
          newPosition = effectiveStart;
          const clip = getActiveVideoClip();
          if (clip?.source?.videoElement) {
            clip.source.videoElement.currentTime = clip.reversed
              ? clip.outPoint
              : clip.inPoint;
          }
        } else {
          newPosition = effectiveEnd;
          ps();
          setPlayheadPosition(newPosition);
          return;
        }
      }

      setPlayheadPosition(newPosition);
    };

    intervalId = setInterval(updatePlayhead, 66);

    return () => clearInterval(intervalId);
  }, [isPlaying, setPlayheadPosition]);

  // Handle shift+mousewheel on track header to resize height
  const handleTrackHeaderWheel = useCallback(
    (e: React.WheelEvent, trackId: string) => {
      const track = tracks.find((t) => t.id === trackId);
      if (!track) return;

      if (e.altKey) {
        e.preventDefault();
        e.stopPropagation();
        const delta = e.deltaY > 0 ? -10 : 10;
        useTimelineStore.getState().scaleTracksOfType(track.type, delta);
      } else if (e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        const delta = e.deltaY > 0 ? -10 : 10;
        useTimelineStore.getState().setTrackHeight(trackId, track.height + delta);
      }
    },
    [tracks]
  );

  // Handle time ruler mousedown
  const handleRulerMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();

      if (isRamPreviewing) {
        cancelRamPreview();
      }

      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const time = pixelToTime(x);
      setPlayheadPosition(Math.max(0, Math.min(time, duration)));

      setDraggingPlayhead(true);
    },
    [
      isRamPreviewing,
      cancelRamPreview,
      scrollX,
      pixelToTime,
      duration,
      setPlayheadPosition,
      setDraggingPlayhead,
    ]
  );

  // Handle playhead drag
  const handlePlayheadMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (isRamPreviewing) {
        cancelRamPreview();
      }
      setDraggingPlayhead(true);
    },
    [isRamPreviewing, cancelRamPreview, setDraggingPlayhead]
  );

  // Handle In/Out marker drag
  const handleMarkerMouseDown = useCallback(
    (e: React.MouseEvent, type: 'in' | 'out') => {
      e.stopPropagation();
      e.preventDefault();
      const originalTime = type === 'in' ? inPoint : outPoint;
      if (originalTime === null) return;

      setMarkerDrag({
        type,
        startX: e.clientX,
        originalTime,
      });
    },
    [inPoint, outPoint]
  );

  // Handle marker dragging
  useEffect(() => {
    if (!markerDrag) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!timelineRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const time = Math.max(0, Math.min(pixelToTime(x), duration));

      if (markerDrag.type === 'in') {
        const maxTime = outPoint !== null ? outPoint : duration;
        setInPoint(Math.min(time, maxTime));
      } else {
        const minTime = inPoint !== null ? inPoint : 0;
        setOutPoint(Math.max(time, minTime));
      }
    };

    const handleMouseUp = () => {
      setMarkerDrag(null);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [markerDrag, scrollX, duration, inPoint, outPoint, setInPoint, setOutPoint, pixelToTime]);

  // Get all snap target times
  const getSnapTargetTimes = useCallback(() => {
    const snapTimes: number[] = [];
    clips.forEach((clip) => {
      snapTimes.push(clip.startTime);
      snapTimes.push(clip.startTime + clip.duration);

      const kfs = getClipKeyframes(clip.id);
      kfs.forEach((kf) => {
        const absTime = clip.startTime + kf.time;
        snapTimes.push(absTime);
      });
    });
    return snapTimes;
  }, [clips, getClipKeyframes]);

  // Playhead dragging
  useEffect(() => {
    if (!isDraggingPlayhead) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!timelineRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      let time = pixelToTime(x);

      if (e.shiftKey) {
        const snapTimes = getSnapTargetTimes();
        const snapThreshold = pixelToTime(10);

        let closestSnap: number | null = null;
        let closestDistance = Infinity;

        for (const snapTime of snapTimes) {
          const distance = Math.abs(time - snapTime);
          if (distance < closestDistance && distance < snapThreshold) {
            closestDistance = distance;
            closestSnap = snapTime;
          }
        }

        if (closestSnap !== null) {
          time = closestSnap;
        }
      }

      setPlayheadPosition(Math.max(0, Math.min(time, duration)));
    };

    const handleMouseUp = () => {
      setDraggingPlayhead(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [
    isDraggingPlayhead,
    scrollX,
    duration,
    setPlayheadPosition,
    setDraggingPlayhead,
    pixelToTime,
    getSnapTargetTimes,
  ]);

  // Premiere-style clip drag
  const handleClipMouseDown = useCallback(
    (e: React.MouseEvent, clipId: string) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();

      const clip = clips.find((c) => c.id === clipId);
      if (!clip) return;

      const clipElement = e.currentTarget as HTMLElement;
      const clipRect = clipElement.getBoundingClientRect();
      const grabOffsetX = e.clientX - clipRect.left;

      selectClip(clipId);

      const initialDrag: ClipDragState = {
        clipId,
        originalStartTime: clip.startTime,
        originalTrackId: clip.trackId,
        grabOffsetX,
        currentX: e.clientX,
        currentTrackId: clip.trackId,
        snappedTime: null,
        isSnapping: false,
      };
      setClipDrag(initialDrag);
      clipDragRef.current = initialDrag;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const drag = clipDragRef.current;
        if (!drag || !trackLanesRef.current || !timelineRef.current) return;

        const lanesRect = trackLanesRef.current.getBoundingClientRect();
        const mouseY = moveEvent.clientY - lanesRect.top;

        let currentY = 24;
        let newTrackId = drag.currentTrackId;
        for (const track of tracks) {
          if (mouseY >= currentY && mouseY < currentY + track.height) {
            newTrackId = track.id;
            break;
          }
          currentY += track.height;
        }

        const rect = timelineRef.current.getBoundingClientRect();
        const x = moveEvent.clientX - rect.left + scrollX - drag.grabOffsetX;
        const rawTime = Math.max(0, pixelToTime(x));
        const { startTime: snappedTime, snapped } = getSnappedPosition(
          drag.clipId,
          rawTime,
          newTrackId
        );

        const newDrag: ClipDragState = {
          ...drag,
          currentX: moveEvent.clientX,
          currentTrackId: newTrackId,
          snappedTime: snapped ? snappedTime : null,
          isSnapping: snapped,
        };
        setClipDrag(newDrag);
        clipDragRef.current = newDrag;
      };

      const handleMouseUp = (upEvent: MouseEvent) => {
        const drag = clipDragRef.current;
        if (drag && timelineRef.current) {
          const rect = timelineRef.current.getBoundingClientRect();
          const x = upEvent.clientX - rect.left + scrollX - drag.grabOffsetX;
          const newStartTime = Math.max(0, pixelToTime(x));
          moveClip(drag.clipId, newStartTime, drag.currentTrackId);
        }
        setClipDrag(null);
        clipDragRef.current = null;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [clips, tracks, scrollX, pixelToTime, selectClip, getSnappedPosition, moveClip]
  );

  // Handle trim start
  const handleTrimStart = useCallback(
    (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => {
      e.stopPropagation();
      e.preventDefault();

      const clip = clips.find((c) => c.id === clipId);
      if (!clip) return;

      selectClip(clipId);

      const initialTrim: ClipTrimState = {
        clipId,
        edge,
        originalStartTime: clip.startTime,
        originalDuration: clip.duration,
        originalInPoint: clip.inPoint,
        originalOutPoint: clip.outPoint,
        startX: e.clientX,
        currentX: e.clientX,
        altKey: e.altKey,
      };
      setClipTrim(initialTrim);
      clipTrimRef.current = initialTrim;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const newTrim = clipTrimRef.current;
        if (!newTrim) return;
        const updated = {
          ...newTrim,
          currentX: moveEvent.clientX,
          altKey: moveEvent.altKey,
        };
        setClipTrim(updated);
        clipTrimRef.current = updated;
      };

      const handleMouseUp = () => {
        const trim = clipTrimRef.current;
        if (!trim) {
          setClipTrim(null);
          clipTrimRef.current = null;
          document.removeEventListener('mousemove', handleMouseMove);
          document.removeEventListener('mouseup', handleMouseUp);
          return;
        }

        const clipToTrim = clips.find((c) => c.id === trim.clipId);
        if (!clipToTrim) {
          setClipTrim(null);
          clipTrimRef.current = null;
          document.removeEventListener('mousemove', handleMouseMove);
          document.removeEventListener('mouseup', handleMouseUp);
          return;
        }

        const deltaX = trim.currentX - trim.startX;
        const deltaTime = pixelToTime(deltaX);
        const maxDuration = clipToTrim.source?.naturalDuration || clipToTrim.duration;

        let newStartTime = trim.originalStartTime;
        let newInPoint = trim.originalInPoint;
        let newOutPoint = trim.originalOutPoint;

        if (trim.edge === 'left') {
          const maxTrim = trim.originalDuration - 0.1;
          const minTrim = -trim.originalInPoint;
          const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));
          newStartTime = trim.originalStartTime + clampedDelta;
          newInPoint = trim.originalInPoint + clampedDelta;
        } else {
          const maxExtend = maxDuration - trim.originalOutPoint;
          const minTrim = -(trim.originalDuration - 0.1);
          const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
          newOutPoint = trim.originalOutPoint + clampedDelta;
        }

        trimClip(clipToTrim.id, newInPoint, newOutPoint);
        if (trim.edge === 'left') {
          moveClip(clipToTrim.id, Math.max(0, newStartTime), clipToTrim.trackId, trim.altKey);
        }

        if (!trim.altKey && clipToTrim.linkedClipId) {
          const linkedClip = clips.find((c) => c.id === clipToTrim.linkedClipId);
          if (linkedClip) {
            const linkedMaxDuration =
              linkedClip.source?.naturalDuration || linkedClip.duration;
            if (trim.edge === 'left') {
              const linkedNewInPoint = Math.max(
                0,
                Math.min(linkedMaxDuration - 0.1, newInPoint)
              );
              trimClip(linkedClip.id, linkedNewInPoint, linkedClip.outPoint);
              moveClip(
                linkedClip.id,
                Math.max(0, newStartTime),
                linkedClip.trackId,
                true
              );
            } else {
              const linkedNewOutPoint = Math.max(
                0.1,
                Math.min(linkedMaxDuration, newOutPoint)
              );
              trimClip(linkedClip.id, linkedClip.inPoint, linkedNewOutPoint);
            }
          }
        }

        setClipTrim(null);
        clipTrimRef.current = null;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [clips, pixelToTime, selectClip, trimClip, moveClip]
  );

  // Quick duration check for dragged video files
  const getVideoDurationQuick = async (file: File): Promise<number | null> => {
    if (!file.type.startsWith('video/')) return null;

    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.preload = 'metadata';

      const cleanup = () => {
        URL.revokeObjectURL(video.src);
        video.remove();
      };

      video.onloadedmetadata = () => {
        const dur = video.duration;
        cleanup();
        resolve(isFinite(dur) ? dur : null);
      };

      video.onerror = () => {
        cleanup();
        resolve(null);
      };

      setTimeout(() => {
        cleanup();
        resolve(null);
      }, DURATION_CHECK_TIMEOUT);

      video.src = URL.createObjectURL(file);
    });
  };

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

      if (e.dataTransfer.types.includes('Files')) {
        let dur: number | undefined;
        const items = e.dataTransfer.items;
        if (items && items.length > 0) {
          for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.kind === 'file') {
              const file = item.getAsFile();
              if (file && file.type.startsWith('video/')) {
                const cacheKey = `${file.name}_${file.size}`;
                if (dragDurationCacheRef.current?.url === cacheKey) {
                  dur = dragDurationCacheRef.current.duration;
                } else {
                  getVideoDurationQuick(file).then((d) => {
                    if (d) {
                      dragDurationCacheRef.current = { url: cacheKey, duration: d };
                      setExternalDrag((prev) =>
                        prev ? { ...prev, duration: d } : null
                      );
                    }
                  });
                }
                break;
              }
            }
          }
        }

        setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: dur });
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
      const isFileDrag = e.dataTransfer.types.includes('Files');

      if ((isCompDrag || isMediaPanelDrag || isFileDrag) && timelineRef.current) {
        const rect = timelineRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);

        const targetTrack = tracks.find((t) => t.id === trackId);
        const isVideoTrack = targetTrack?.type === 'video';

        const previewDuration =
          externalDrag?.duration ?? dragDurationCacheRef.current?.duration ?? 5;

        let audioTrackId: string | undefined;
        if (isVideoTrack) {
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
          duration: prev?.duration ?? dragDurationCacheRef.current?.duration,
        }));
      }
    },
    [scrollX, pixelToTime, tracks, clips, externalDrag]
  );

  // Handle external file drag leave
  const handleTrackDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current--;

    if (dragCounterRef.current === 0) {
      setExternalDrag(null);
    }
  }, []);

  // Handle external file drop on track
  const handleTrackDrop = useCallback(
    (e: React.DragEvent, trackId: string) => {
      e.preventDefault();

      const cachedDuration =
        externalDrag?.duration ?? dragDurationCacheRef.current?.duration;

      dragCounterRef.current = 0;
      setExternalDrag(null);

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

      const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addClip(trackId, mediaFile.file, Math.max(0, startTime), mediaFile.duration, mediaFileId);
          return;
        }
      }

      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (
          file.type.startsWith('video/') ||
          file.type.startsWith('audio/') ||
          file.type.startsWith('image/')
        ) {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          addClip(trackId, file, Math.max(0, startTime), cachedDuration);
        }
      }
    },
    [scrollX, pixelToTime, addCompClip, addClip, externalDrag]
  );

  // Zoom with mouse wheel
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (e.ctrlKey || e.altKey) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -5 : 5;
        setZoom(zoom + delta);
      } else {
        setScrollX(scrollX + e.deltaX);
      }
    },
    [zoom, scrollX, setZoom, setScrollX]
  );

  // Render keyframe diamonds
  const renderKeyframeDiamonds = useCallback(
    (trackId: string, property: AnimatableProperty) => {
      return (
        <TimelineKeyframes
          trackId={trackId}
          property={property}
          clips={clips}
          selectedKeyframeIds={selectedKeyframeIds}
          getClipKeyframes={getClipKeyframes}
          onSelectKeyframe={selectKeyframe}
          timeToPixel={timeToPixel}
        />
      );
    },
    [clips, selectedKeyframeIds, getClipKeyframes, selectKeyframe, timeToPixel]
  );

  // Render a clip
  const renderClip = useCallback(
    (clip: TimelineClipType, trackId: string) => {
      const track = tracks.find((t) => t.id === trackId);
      if (!track) return null;

      const isDragging = clipDrag?.clipId === clip.id;
      const isTrimming = clipTrim?.clipId === clip.id;

      const draggedClip = clipDrag
        ? clips.find((c) => c.id === clipDrag.clipId)
        : null;
      const trimmedClip = clipTrim
        ? clips.find((c) => c.id === clipTrim.clipId)
        : null;

      const isLinkedToDragging =
        clipDrag &&
        draggedClip &&
        (clip.linkedClipId === clipDrag.clipId ||
          draggedClip.linkedClipId === clip.id);
      const isLinkedToTrimming =
        clipTrim &&
        !clipTrim.altKey &&
        trimmedClip &&
        (clip.linkedClipId === clipTrim.clipId ||
          trimmedClip.linkedClipId === clip.id);

      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(
        (f) =>
          f.id === clip.source?.mediaFileId ||
          f.name === clip.name ||
          f.name === clip.name.replace(' (Audio)', '')
      );

      return (
        <TimelineClip
          key={clip.id}
          clip={clip}
          trackId={trackId}
          track={track}
          tracks={tracks}
          clips={clips}
          isSelected={selectedClipId === clip.id}
          isDragging={isDragging}
          isTrimming={isTrimming}
          isLinkedToDragging={!!isLinkedToDragging}
          isLinkedToTrimming={!!isLinkedToTrimming}
          clipDrag={clipDrag}
          clipTrim={clipTrim}
          zoom={zoom}
          scrollX={scrollX}
          timelineRef={timelineRef}
          proxyEnabled={proxyEnabled}
          proxyStatus={mediaFile?.proxyStatus}
          proxyProgress={mediaFile?.proxyProgress || 0}
          onMouseDown={(e) => handleClipMouseDown(e, clip.id)}
          onContextMenu={(e) => handleClipContextMenu(e, clip.id)}
          onTrimStart={(e, edge) => handleTrimStart(e, clip.id, edge)}
          hasKeyframes={hasKeyframes}
          timeToPixel={timeToPixel}
          pixelToTime={pixelToTime}
          formatTime={formatTime}
        />
      );
    },
    [
      tracks,
      clips,
      selectedClipId,
      clipDrag,
      clipTrim,
      zoom,
      scrollX,
      proxyEnabled,
      handleClipMouseDown,
      handleClipContextMenu,
      handleTrimStart,
      hasKeyframes,
      timeToPixel,
      pixelToTime,
      formatTime,
    ]
  );

  // No active composition - show empty state
  if (openCompositions.length === 0) {
    return (
      <div className="timeline-container timeline-empty">
        <div className="timeline-empty-message">
          <p>No composition open</p>
          <p className="hint">Double-click a composition in the Media panel to open it</p>
        </div>
      </div>
    );
  }

  const anyVideoSolo = tracks.some((t) => t.type === 'video' && t.solo);
  const anyAudioSolo = tracks.some((t) => t.type === 'audio' && t.solo);

  return (
    <div className={`timeline-container ${clipDrag || clipTrim ? 'is-dragging' : ''}`}>
      <TimelineControls
        isPlaying={isPlaying}
        loopPlayback={loopPlayback}
        playheadPosition={playheadPosition}
        duration={duration}
        zoom={zoom}
        inPoint={inPoint}
        outPoint={outPoint}
        ramPreviewEnabled={ramPreviewEnabled}
        proxyEnabled={proxyEnabled}
        currentlyGeneratingProxyId={currentlyGeneratingProxyId}
        mediaFilesWithProxy={mediaFiles.filter((f) => f.proxyStatus === 'ready').length}
        onPlay={play}
        onPause={pause}
        onStop={stop}
        onToggleLoop={toggleLoopPlayback}
        onSetZoom={setZoom}
        onSetInPoint={setInPointAtPlayhead}
        onSetOutPoint={setOutPointAtPlayhead}
        onClearInOut={clearInOut}
        onToggleRamPreview={toggleRamPreviewEnabled}
        onToggleProxy={() => setProxyEnabled(!proxyEnabled)}
        onAddVideoTrack={() => addTrack('video')}
        onAddAudioTrack={() => addTrack('audio')}
        formatTime={formatTime}
      />

      <div className="timeline-body">
        <div className="track-headers">
          <div className="ruler-header">Time</div>
          {tracks.map((track) => {
            const isDimmed =
              (track.type === 'video' && anyVideoSolo && !track.solo) ||
              (track.type === 'audio' && anyAudioSolo && !track.solo);
            const isExpanded = isTrackExpanded(track.id);
            const dynamicHeight = getExpandedTrackHeight(track.id, track.height);

            return (
              <TimelineHeader
                key={track.id}
                track={track}
                isDimmed={isDimmed}
                isExpanded={isExpanded}
                dynamicHeight={dynamicHeight}
                hasKeyframes={trackHasKeyframes(track.id)}
                selectedClipId={selectedClipId}
                clips={clips}
                onToggleExpand={() => toggleTrackExpanded(track.id)}
                onToggleSolo={() =>
                  useTimelineStore.getState().setTrackSolo(track.id, !track.solo)
                }
                onToggleMuted={() =>
                  useTimelineStore.getState().setTrackMuted(track.id, !track.muted)
                }
                onToggleVisible={() =>
                  useTimelineStore.getState().setTrackVisible(track.id, !track.visible)
                }
                onWheel={(e) => handleTrackHeaderWheel(e, track.id)}
                isTrackPropertyGroupExpanded={isTrackPropertyGroupExpanded}
                toggleTrackPropertyGroupExpanded={toggleTrackPropertyGroupExpanded}
                hasPropertyKeyframes={hasKeyframes}
              />
            );
          })}
        </div>

        <div
          ref={(el) => {
            (timelineRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
            (trackLanesRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
          }}
          className={`timeline-tracks ${clipDrag ? 'dragging-clip' : ''}`}
          onWheel={handleWheel}
          style={{ transform: `translateX(-${scrollX}px)` }}
        >
          <TimelineRuler
            duration={duration}
            zoom={zoom}
            scrollX={scrollX}
            onRulerMouseDown={handleRulerMouseDown}
            formatTime={formatTime}
          />

          {tracks.map((track) => {
            const isDimmed =
              (track.type === 'video' && anyVideoSolo && !track.solo) ||
              (track.type === 'audio' && anyAudioSolo && !track.solo);
            const isExpanded = isTrackExpanded(track.id);
            const dynamicHeight = getExpandedTrackHeight(track.id, track.height);

            return (
              <TimelineTrack
                key={track.id}
                track={track}
                clips={clips}
                isDimmed={isDimmed}
                isExpanded={isExpanded}
                dynamicHeight={dynamicHeight}
                isDragTarget={clipDrag?.currentTrackId === track.id}
                isExternalDragTarget={
                  externalDrag?.trackId === track.id ||
                  externalDrag?.audioTrackId === track.id
                }
                selectedClipId={selectedClipId}
                clipDrag={clipDrag}
                clipTrim={clipTrim}
                externalDrag={externalDrag}
                zoom={zoom}
                scrollX={scrollX}
                timelineRef={timelineRef}
                onClipMouseDown={handleClipMouseDown}
                onClipContextMenu={handleClipContextMenu}
                onTrimStart={handleTrimStart}
                onDrop={(e) => handleTrackDrop(e, track.id)}
                onDragOver={(e) => handleTrackDragOver(e, track.id)}
                onDragEnter={(e) => handleTrackDragEnter(e, track.id)}
                onDragLeave={handleTrackDragLeave}
                renderClip={renderClip}
                isTrackPropertyGroupExpanded={isTrackPropertyGroupExpanded}
                hasPropertyKeyframes={hasKeyframes}
                renderKeyframeDiamonds={renderKeyframeDiamonds}
                timeToPixel={timeToPixel}
                pixelToTime={pixelToTime}
              />
            );
          })}

          {externalDrag &&
            externalDrag.isVideo &&
            externalDrag.audioTrackId === '__new_audio_track__' && (
              <div className="track-lane audio new-track-preview" style={{ height: 40 }}>
                <div
                  className="timeline-clip-preview audio"
                  style={{
                    left: timeToPixel(externalDrag.startTime),
                    width: timeToPixel(externalDrag.duration ?? 5),
                  }}
                >
                  <div className="clip-content">
                    <span className="clip-name">+ New Audio Track</span>
                  </div>
                </div>
              </div>
            )}

          {clipDrag?.isSnapping && clipDrag.snappedTime !== null && (
            <div className="snap-line" style={{ left: timeToPixel(clipDrag.snappedTime) }} />
          )}

          {(inPoint !== null || outPoint !== null) && (
            <>
              {inPoint !== null && inPoint > 0 && (
                <div
                  className="work-area-overlay before"
                  style={{
                    left: 0,
                    width: timeToPixel(inPoint),
                  }}
                />
              )}
              {outPoint !== null && (
                <div
                  className="work-area-overlay after"
                  style={{
                    left: timeToPixel(outPoint),
                    width: timeToPixel(duration - outPoint),
                  }}
                />
              )}
            </>
          )}

          {isRamPreviewing && ramPreviewProgress !== null && (
            <div
              className="ram-preview-progress-text"
              style={{
                left: timeToPixel(playheadPosition) + 10,
              }}
            >
              {Math.round(ramPreviewProgress)}%
            </div>
          )}

          {getCachedRanges().map((range, i) => (
            <div
              key={i}
              className="playback-cache-indicator"
              style={{
                left: timeToPixel(range.start),
                width: Math.max(2, timeToPixel(range.end - range.start)),
              }}
              title={`Cached: ${formatTime(range.start)} - ${formatTime(range.end)}`}
            />
          ))}

          {inPoint !== null && (
            <div
              className={`in-out-marker in-marker ${markerDrag?.type === 'in' ? 'dragging' : ''}`}
              style={{ left: timeToPixel(inPoint) }}
              title={`In: ${formatTime(inPoint)} (drag to move)`}
            >
              <div
                className="marker-flag"
                onMouseDown={(e) => handleMarkerMouseDown(e, 'in')}
              >
                I
              </div>
              <div className="marker-line" />
            </div>
          )}

          {outPoint !== null && (
            <div
              className={`in-out-marker out-marker ${markerDrag?.type === 'out' ? 'dragging' : ''}`}
              style={{ left: timeToPixel(outPoint) }}
              title={`Out: ${formatTime(outPoint)} (drag to move)`}
            >
              <div
                className="marker-flag"
                onMouseDown={(e) => handleMarkerMouseDown(e, 'out')}
              >
                O
              </div>
              <div className="marker-line" />
            </div>
          )}

          <div
            className="playhead"
            style={{ left: timeToPixel(playheadPosition) }}
            onMouseDown={handlePlayheadMouseDown}
          >
            <div className="playhead-head" />
            <div className="playhead-line" />
          </div>
        </div>
      </div>

      {contextMenu &&
        (() => {
          const mediaFile = getMediaFileForClip(contextMenu.clipId);
          const clip = clips.find((c) => c.id === contextMenu.clipId);
          const isVideo = clip?.source?.type === 'video';
          const isGenerating = mediaFile?.proxyStatus === 'generating';
          const hasProxyContextMenu = mediaFile?.proxyStatus === 'ready';

          return (
            <div
              className="timeline-context-menu"
              style={{
                position: 'fixed',
                left: contextMenu.x,
                top: contextMenu.y,
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
                      className={`context-menu-item ${!hasProxyContextMenu ? 'disabled' : ''}`}
                      onClick={() => hasProxyContextMenu && handleShowInExplorer('proxy')}
                    >
                      Proxy{' '}
                      {!hasProxyContextMenu
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
                  ) : hasProxyContextMenu ? (
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
        })()}
    </div>
  );
}
