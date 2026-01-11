// Timeline component - Main orchestrator for video editing timeline
// Composes TimelineRuler, TimelineControls, TimelineHeader, TimelineTrack, TimelineClip, TimelineKeyframes

import { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { AnimatableProperty, Effect, TimelineClip as TimelineClipType, Layer, NestedCompositionData } from '../../types';
import { useMediaStore } from '../../stores/mediaStore';
import { useMixerStore } from '../../stores/mixerStore';
import { engine } from '../../engine/WebGPUEngine';
import { proxyFrameCache } from '../../services/proxyFrameCache';
import { audioManager, audioStatusTracker } from '../../services/audioManager';
import { playheadState } from '../../services/layerBuilder';

import { TimelineRuler } from './TimelineRuler';
import { TimelineControls } from './TimelineControls';
import { TimelineHeader } from './TimelineHeader';
import { TimelineTrack } from './TimelineTrack';
import { TimelineClip } from './TimelineClip';
import { TimelineKeyframes } from './TimelineKeyframes';
import { MulticamDialog } from './MulticamDialog';
import { ParentChildLink } from './ParentChildLink';
import { PhysicsCable } from './PhysicsCable';
import { TimelineNavigator } from './TimelineNavigator';
import { useTimelineKeyboard } from './hooks/useTimelineKeyboard';
import { useTimelineZoom } from './hooks/useTimelineZoom';
import { usePlayheadDrag } from './hooks/usePlayheadDrag';
import { TimelineContextMenu, useClipContextMenu } from './TimelineContextMenu';
import { useMarqueeSelection } from './hooks/useMarqueeSelection';
import { useClipTrim } from './hooks/useClipTrim';
import { useClipDrag } from './hooks/useClipDrag';
import { useLayerSync } from './hooks/useLayerSync';
import {
  RAM_PREVIEW_IDLE_DELAY,
  PROXY_IDLE_DELAY,
  DURATION_CHECK_TIMEOUT,
} from './constants';
import { MIN_ZOOM, MAX_ZOOM, PROPERTY_ROW_HEIGHT } from '../../stores/timeline/constants';
import type {
  ClipDragState,
  ClipTrimState,
  MarkerDragState,
  ExternalDragState,
  ContextMenuState,
  MarqueeState,
  PickWhipDragState,
} from './types';

export function Timeline() {
  const {
    tracks,
    clips,
    playheadPosition,
    duration,
    zoom,
    scrollX,
    snappingEnabled,
    toggleSnapping,
    isPlaying,
    selectedClipIds,
    inPoint,
    outPoint,
    addTrack,
    addClip,
    addCompClip,
    moveClip,
    trimClip,
    removeClip,
    selectClip,
    unlinkGroup,
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
    getPositionWithResistance,
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
    getExpandedTrackHeight,
    getClipKeyframes,
    selectKeyframe,
    deselectAllKeyframes,
    selectedKeyframeIds,
    hasKeyframes,
    trackHasKeyframes,
    clipKeyframes,
    addKeyframe,
    moveKeyframe,
    updateKeyframe,
    removeKeyframe,
    setPropertyValue,
    expandedCurveProperties,
    toggleCurveExpanded,
    updateBezierHandle,
    thumbnailsEnabled,
    waveformsEnabled,
    toggleThumbnailsEnabled,
    toggleWaveformsEnabled,
    generateWaveformForClip,
    setDuration,
    setClipParent,
    setTrackParent,
    getSourceTimeForClip,
    getInterpolatedSpeed,
    addTextClip,
  } = useTimelineStore();

  const {
    getActiveComposition,
    getOpenCompositions,
    openCompositionTab,
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
  const timelineBodyRef = useRef<HTMLDivElement>(null);
  const trackLanesRef = useRef<HTMLDivElement>(null);

  // Performance: Create lookup maps for O(1) clip/track access (must be before hooks that use them)
  const clipMap = useMemo(() => new Map(clips.map(c => [c.id, c])), [clips]);
  const trackMap = useMemo(() => new Map(tracks.map(t => [t.id, t])), [tracks]);

  // Time conversion helpers (must be before hooks that use them)
  const timeToPixel = useCallback((time: number) => time * zoom, [zoom]);
  const pixelToTime = useCallback((pixel: number) => pixel / zoom, [zoom]);

  // Clip dragging - extracted to hook
  const { clipDrag, clipDragRef, handleClipMouseDown, handleClipDoubleClick } = useClipDrag({
    trackLanesRef,
    timelineRef,
    clips,
    tracks,
    clipMap,
    selectedClipIds,
    scrollX,
    selectClip,
    moveClip,
    openCompositionTab,
    pixelToTime,
    getSnappedPosition,
    getPositionWithResistance,
  });

  // Clip trimming - extracted to hook
  const { clipTrim, clipTrimRef, handleTrimStart } = useClipTrim({
    clipMap,
    selectClip,
    trimClip,
    moveClip,
    pixelToTime,
  });

  // Playhead and marker dragging - extracted to hook
  const { markerDrag, handleRulerMouseDown, handlePlayheadMouseDown, handleMarkerMouseDown } = usePlayheadDrag({
    timelineRef,
    scrollX,
    duration,
    inPoint,
    outPoint,
    isRamPreviewing,
    setPlayheadPosition,
    setDraggingPlayhead,
    setInPoint,
    setOutPoint,
    cancelRamPreview,
    pixelToTime,
  });

  // External file drag preview state
  const [externalDrag, setExternalDrag] = useState<ExternalDragState | null>(null);

  // Context menu state for clip right-click
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const handleClipContextMenu = useClipContextMenu(selectedClipIds, selectClip, setContextMenu);
  const dragCounterRef = useRef(0);

  // Transcript markers visibility toggle
  const [showTranscriptMarkers, setShowTranscriptMarkers] = useState(true);
  const dragDurationCacheRef = useRef<{ url: string; duration: number } | null>(null);

  // Multicam dialog state
  const [multicamDialogOpen, setMulticamDialogOpen] = useState(false);

  // Marquee selection - extracted to hook
  const { marquee, handleMarqueeMouseDown } = useMarqueeSelection({
    trackLanesRef,
    scrollX,
    clips,
    tracks,
    selectedClipIds,
    selectedKeyframeIds,
    clipKeyframes,
    clipDrag,
    clipTrim,
    markerDrag,
    isDraggingPlayhead,
    selectClip,
    selectKeyframe,
    deselectAllKeyframes,
    pixelToTime,
    isTrackExpanded,
    getExpandedTrackHeight,
  });

  // Pick whip drag state for clip parenting
  const [pickWhipDrag, setPickWhipDrag] = useState<PickWhipDragState | null>(null);

  // Pick whip drag state for track/layer parenting
  const [trackPickWhipDrag, setTrackPickWhipDrag] = useState<PickWhipDragState | null>(null);

  // Performance: Memoize video/audio track filtering and solo state
  const { videoTracks, audioTracks, anyVideoSolo, anyAudioSolo } = useMemo(() => {
    const vTracks = tracks.filter(t => t.type === 'video');
    const aTracks = tracks.filter(t => t.type === 'audio');
    return {
      videoTracks: vTracks,
      audioTracks: aTracks,
      anyVideoSolo: vTracks.some(t => t.solo),
      anyAudioSolo: aTracks.some(t => t.solo),
    };
  }, [tracks]);

  // Performance: Memoize track visibility check functions
  const isVideoTrackVisible = useCallback((track: typeof tracks[0]) => {
    if (!track.visible) return false;
    if (anyVideoSolo) return track.solo;
    return true;
  }, [anyVideoSolo]);

  const isAudioTrackMuted = useCallback((track: typeof tracks[0]) => {
    if (track.muted) return true;
    if (anyAudioSolo) return !track.solo;
    return false;
  }, [anyAudioSolo]);

  // Performance: Memoize proxy-ready file count
  const mediaFilesWithProxyCount = useMemo(
    () => mediaFiles.filter((f) => f.proxyStatus === 'ready').length,
    [mediaFiles]
  );

  // Format time as MM:SS.ms
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  }, []);

  // Parse time string (MM:SS.ms or SS.ms or just seconds) back to seconds
  const parseTime = useCallback((timeStr: string): number | null => {
    const trimmed = timeStr.trim();
    if (!trimmed) return null;

    // Try MM:SS.ms format
    const match = trimmed.match(/^(\d+):(\d+)(?:\.(\d+))?$/);
    if (match) {
      const mins = parseInt(match[1], 10);
      const secs = parseInt(match[2], 10);
      const ms = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
      return mins * 60 + secs + ms / 100;
    }

    // Try SS.ms or just seconds
    const num = parseFloat(trimmed);
    if (!isNaN(num) && num >= 0) {
      return num;
    }

    return null;
  }, []);

  // Get clips at time helper
  const getClipsAtTime = useCallback(
    (time: number) => {
      return clips.filter((c) => time >= c.startTime && time < c.startTime + c.duration);
    },
    [clips]
  );

  // Keyboard shortcuts - extracted to hook
  useTimelineKeyboard({
    isPlaying,
    play,
    pause,
    setInPointAtPlayhead,
    setOutPointAtPlayhead,
    clearInOut,
    toggleLoopPlayback,
    selectedClipIds,
    selectedKeyframeIds,
    removeClip,
    removeKeyframe,
    splitClipAtPlayhead,
    updateClipTransform,
    clipMap,
    activeComposition,
    playheadPosition,
    duration,
    setPlayheadPosition,
  });

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

  // Layer sync - extracted to hook
  useLayerSync({
    timelineRef,
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
    clipMap,
    videoTracks,
    audioTracks,
    getClipsAtTime,
    getInterpolatedTransform,
    getInterpolatedEffects,
    getInterpolatedSpeed,
    getSourceTimeForClip,
    isVideoTrackVisible,
    isAudioTrackMuted,
  });

  // Preload upcoming video clips - seek videos and force buffering before playhead hits them
  // This prevents stuttering when playback transitions to a new clip
  useEffect(() => {
    if (!isPlaying || isDraggingPlayhead) return;

    const LOOKAHEAD_TIME = 2.0; // Look 2 seconds ahead
    const lookaheadPosition = playheadPosition + LOOKAHEAD_TIME;

    // Helper to preload a video element - seeks and forces buffering
    const preloadVideo = (video: HTMLVideoElement, targetTime: number, _clipName: string) => {
      const timeDiff = Math.abs(video.currentTime - targetTime);

      // Only preload if significantly different (avoid repeated preloading)
      if (timeDiff > 0.1) {
        video.currentTime = Math.max(0, targetTime);

        // Force buffer by briefly playing then pausing
        // This triggers the browser to actually fetch the video data
        const wasPlaying = !video.paused;
        if (!wasPlaying) {
          video.play()
            .then(() => {
              // Immediately pause after play starts buffering
              setTimeout(() => {
                if (!wasPlaying) video.pause();
              }, 50);
            })
            .catch(() => {
              // Ignore play errors (e.g., autoplay policy)
            });
        }
        // console.log(`[Preload] Pre-buffering ${clipName} at ${targetTime.toFixed(2)}s`);
      }
    };

    // Find clips that will start playing soon (not currently playing, but will be soon)
    const upcomingClips = clips.filter(clip => {
      // Clip starts after current position but within lookahead window
      const startsInLookahead = clip.startTime > playheadPosition && clip.startTime <= lookaheadPosition;
      // Has a video element to preload
      const hasVideo = clip.source?.videoElement;
      return startsInLookahead && hasVideo;
    });

    // Pre-buffer upcoming regular clips
    for (const clip of upcomingClips) {
      if (clip.source?.videoElement) {
        preloadVideo(clip.source.videoElement, clip.inPoint, clip.name);
      }
    }

    // Also preload nested composition clips
    const upcomingNestedClips = clips.filter(clip => {
      const startsInLookahead = clip.startTime > playheadPosition && clip.startTime <= lookaheadPosition;
      const hasNestedClips = clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0;
      return startsInLookahead && hasNestedClips;
    });

    for (const compClip of upcomingNestedClips) {
      if (!compClip.nestedClips) continue;

      // Find the nested video clip that would play at the start of this comp clip
      const compStartTime = compClip.inPoint; // Time within the composition

      for (const nestedClip of compClip.nestedClips) {
        if (!nestedClip.source?.videoElement) continue;

        // Check if this nested clip would be playing at comp start
        if (compStartTime >= nestedClip.startTime && compStartTime < nestedClip.startTime + nestedClip.duration) {
          const nestedLocalTime = compStartTime - nestedClip.startTime;
          const targetTime = nestedClip.reversed
            ? nestedClip.outPoint - nestedLocalTime
            : nestedLocalTime + nestedClip.inPoint;

          preloadVideo(nestedClip.source.videoElement, targetTime, nestedClip.name);
        }
      }
    }
  }, [isPlaying, isDraggingPlayhead, playheadPosition, clips]);

  // Playback loop - using requestAnimationFrame for smooth playback
  // PERFORMANCE: Uses playheadState for high-frequency updates, only updates store at throttled interval
  useEffect(() => {
    if (!isPlaying) {
      // Disable internal position tracking when not playing
      playheadState.isUsingInternalPosition = false;
      return;
    }

    let rafId: number;
    let lastTime = performance.now();
    let lastStateUpdate = 0;
    const STATE_UPDATE_INTERVAL = 33; // Update store every 33ms (~30fps for UI/subscribers)

    // Initialize internal position from store and enable high-frequency mode
    playheadState.position = useTimelineStore.getState().playheadPosition;
    playheadState.isUsingInternalPosition = true;

    const getActiveVideoClip = () => {
      const state = useTimelineStore.getState();
      const pos = playheadState.position;
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

    const updatePlayhead = (currentTime: number) => {
      // Calculate actual elapsed time for smooth playback
      const deltaTime = (currentTime - lastTime) / 1000; // Convert to seconds
      lastTime = currentTime;

      // Cap delta to prevent huge jumps if tab was inactive
      const cappedDelta = Math.min(deltaTime, 0.1);

      const state = useTimelineStore.getState();
      const { duration: dur, inPoint: ip, outPoint: op, loopPlayback: lp, pause: ps } = state;
      const effectiveEnd = op !== null ? op : dur;
      const effectiveStart = ip !== null ? ip : 0;

      // Update high-frequency position (no store update = no subscriber triggers)
      let newPosition = playheadState.position + cappedDelta;

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
          playheadState.position = newPosition;
          playheadState.isUsingInternalPosition = false;
          useTimelineStore.setState({ playheadPosition: newPosition });
          return;
        }
      }

      // Update high-frequency position for render loop to read
      playheadState.position = newPosition;

      // PERFORMANCE: Only update store at throttled interval
      // This prevents subscriber cascade (effects, re-renders) every frame
      if (currentTime - lastStateUpdate >= STATE_UPDATE_INTERVAL) {
        useTimelineStore.setState({ playheadPosition: newPosition });
        lastStateUpdate = currentTime;
      }

      rafId = requestAnimationFrame(updatePlayhead);
    };

    rafId = requestAnimationFrame(updatePlayhead);

    return () => {
      cancelAnimationFrame(rafId);
      playheadState.isUsingInternalPosition = false;
    };
  }, [isPlaying]);

  // Handle shift+mousewheel on track header to resize height
  const handleTrackHeaderWheel = useCallback(
    (e: React.WheelEvent, trackId: string) => {
      const track = trackMap.get(trackId);
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
    [trackMap]
  );

  // Zoom handling - extracted to hook
  const { handleSetZoom, handleFitToWindow } = useTimelineZoom({
    timelineBodyRef: timelineBodyRef,
    zoom,
    scrollX,
    duration,
    playheadPosition,
    setZoom,
    setScrollX,
  });

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
          duration: prev?.duration ?? dragDurationCacheRef.current?.duration ?? 5,
          newTrackType: trackType,
          isVideo: trackType === 'video',
          isAudio: trackType === 'audio',
        }));
      }
    },
    [scrollX, pixelToTime]
  );

  // Handle drop on "new track" zone - creates new track and adds clip
  const handleNewTrackDrop = useCallback(
    async (e: React.DragEvent, trackType: 'video' | 'audio') => {
      e.preventDefault();
      e.stopPropagation();

      const cachedDuration =
        externalDrag?.duration ?? dragDurationCacheRef.current?.duration;

      dragCounterRef.current = 0;
      setExternalDrag(null);

      // Helper to check if file is audio
      const audioExtensions = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'aiff', 'opus'];
      const isAudioFile = (file: File) => {
        const ext = file.name?.split('.').pop()?.toLowerCase() || '';
        return file.type.startsWith('audio/') || audioExtensions.includes(ext);
      };

      // Validate file type matches track type BEFORE creating track
      const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
      if (mediaFileId) {
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find((f) => f.id === mediaFileId);
        if (mediaFile?.file) {
          const fileIsAudio = isAudioFile(mediaFile.file);
          if (fileIsAudio && trackType === 'video') {
            console.log('[Timeline] Audio files can only be dropped on audio tracks');
            return;
          }
          if (!fileIsAudio && trackType === 'audio') {
            console.log('[Timeline] Video/image files can only be dropped on video tracks');
            return;
          }
        }
      }

      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        const fileIsAudio = isAudioFile(file);
        if (fileIsAudio && trackType === 'video') {
          console.log('[Timeline] Audio files can only be dropped on audio tracks');
          return;
        }
        if (!fileIsAudio && trackType === 'audio') {
          console.log('[Timeline] Video/image files can only be dropped on video tracks');
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
      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (
          file.type.startsWith('video/') ||
          file.type.startsWith('audio/') ||
          file.type.startsWith('image/')
        ) {
          const mediaStore = useMediaStore.getState();
          const importedFile = await mediaStore.importFile(file);
          const newMediaFileId = importedFile?.id;
          addClip(newTrackId, file, startTime, cachedDuration, newMediaFileId);
        }
      }
    },
    [scrollX, pixelToTime, addTrack, addCompClip, addClip, externalDrag]
  );

  // Handle external file drop on track
  const handleTrackDrop = useCallback(
    async (e: React.DragEvent, trackId: string) => {
      e.preventDefault();

      const cachedDuration =
        externalDrag?.duration ?? dragDurationCacheRef.current?.duration;

      dragCounterRef.current = 0;
      setExternalDrag(null);

      // Get track type for validation
      const targetTrack = tracks.find((t) => t.id === trackId);
      const isVideoTrack = targetTrack?.type === 'video';
      const isAudioTrack = targetTrack?.type === 'audio';

      // Helper to check if file is audio
      const audioExtensions = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'aiff', 'opus'];
      const isAudioFile = (file: File) => {
        const ext = file.name?.split('.').pop()?.toLowerCase() || '';
        return file.type.startsWith('audio/') || audioExtensions.includes(ext);
      };

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
          // Simple validation: audio files only on audio tracks, video/image only on video tracks
          const fileIsAudio = isAudioFile(mediaFile.file);
          if (fileIsAudio && isVideoTrack) {
            console.log('[Timeline] Audio files can only be dropped on audio tracks');
            return;
          }
          if (!fileIsAudio && isAudioTrack) {
            console.log('[Timeline] Video/image files can only be dropped on video tracks');
            return;
          }

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
          // Simple validation: audio files only on audio tracks, video/image only on video tracks
          const fileIsAudio = isAudioFile(file);
          if (fileIsAudio && isVideoTrack) {
            console.log('[Timeline] Audio files can only be dropped on audio tracks');
            return;
          }
          if (!fileIsAudio && isAudioTrack) {
            console.log('[Timeline] Video/image files can only be dropped on video tracks');
            return;
          }

          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left + scrollX;
          const startTime = pixelToTime(x);
          // Import file to media store first to get mediaFileId (needed for multicam sync)
          const mediaStore = useMediaStore.getState();
          const importedFile = await mediaStore.importFile(file);
          const newMediaFileId = importedFile?.id;
          addClip(trackId, file, Math.max(0, startTime), cachedDuration, newMediaFileId);
        }
      }
    },
    [scrollX, pixelToTime, addCompClip, addClip, externalDrag, tracks]
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
          clipKeyframes={clipKeyframes}
          clipDrag={clipDrag}
          scrollX={scrollX}
          timelineRef={timelineRef}
          onSelectKeyframe={selectKeyframe}
          onMoveKeyframe={moveKeyframe}
          onUpdateKeyframe={updateKeyframe}
          timeToPixel={timeToPixel}
          pixelToTime={pixelToTime}
        />
      );
    },
    [clips, selectedKeyframeIds, clipKeyframes, clipDrag, scrollX, selectKeyframe, moveKeyframe, updateKeyframe, timeToPixel, pixelToTime]
  );

  // Pick whip drag handlers for layer parenting
  const handlePickWhipDragStart = useCallback((clipId: string, startX: number, startY: number) => {
    setPickWhipDrag({
      sourceClipId: clipId,
      startX,
      startY,
      currentX: startX,
      currentY: startY,
    });

    const handleMouseMove = (e: MouseEvent) => {
      setPickWhipDrag(prev => prev ? {
        ...prev,
        currentX: e.clientX,
        currentY: e.clientY,
      } : null);
    };

    const handleMouseUp = (e: MouseEvent) => {
      // Find clip at drop position
      const target = document.elementFromPoint(e.clientX, e.clientY);
      const clipElement = target?.closest('.timeline-clip');
      if (clipElement) {
        const targetClipId = clipElement.getAttribute('data-clip-id');
        if (targetClipId && targetClipId !== clipId) {
          setClipParent(clipId, targetClipId);
        }
      }
      setPickWhipDrag(null);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [setClipParent]);

  const handlePickWhipDragEnd = useCallback(() => {
    setPickWhipDrag(null);
  }, []);

  // Track pick whip drag handlers for layer parenting
  const handleTrackPickWhipDragStart = useCallback((trackId: string, startX: number, startY: number) => {
    setTrackPickWhipDrag({
      sourceClipId: trackId, // Using clipId field to store trackId
      startX,
      startY,
      currentX: startX,
      currentY: startY,
    });

    const handleMouseMove = (e: MouseEvent) => {
      setTrackPickWhipDrag(prev => prev ? {
        ...prev,
        currentX: e.clientX,
        currentY: e.clientY,
      } : null);
    };

    const handleMouseUp = (e: MouseEvent) => {
      // Find track header at drop position
      const target = document.elementFromPoint(e.clientX, e.clientY);
      const trackHeader = target?.closest('.track-header');
      if (trackHeader) {
        // Find the track-pick-whip with data-track-id inside the header
        const pickWhip = trackHeader.querySelector('.track-pick-whip');
        const targetTrackId = pickWhip?.getAttribute('data-track-id');
        if (targetTrackId && targetTrackId !== trackId) {
          setTrackParent(trackId, targetTrackId);
        }
      }
      setTrackPickWhipDrag(null);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [setTrackParent]);

  const handleTrackPickWhipDragEnd = useCallback(() => {
    setTrackPickWhipDrag(null);
  }, []);

  // Render a clip
  const renderClip = useCallback(
    (clip: TimelineClipType, trackId: string) => {
      const track = trackMap.get(trackId);
      if (!track) return null;

      const isDragging = clipDrag?.clipId === clip.id;
      const isTrimming = clipTrim?.clipId === clip.id;

      const draggedClip = clipDrag
        ? clipMap.get(clipDrag.clipId)
        : undefined;
      const trimmedClip = clipTrim
        ? clipMap.get(clipTrim.clipId)
        : undefined;

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

      // Use mediaFiles from hook state instead of getState() for render-time lookups
      const mediaFile = mediaFiles.find(
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
          isSelected={selectedClipIds.has(clip.id)}
          isInLinkedGroup={!!clip.linkedGroupId}
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
          showTranscriptMarkers={showTranscriptMarkers}
          onMouseDown={(e) => handleClipMouseDown(e, clip.id)}
          onDoubleClick={(e) => handleClipDoubleClick(e, clip.id)}
          onContextMenu={(e) => handleClipContextMenu(e, clip.id)}
          onTrimStart={(e, edge) => handleTrimStart(e, clip.id, edge)}
          hasKeyframes={hasKeyframes}
          timeToPixel={timeToPixel}
          pixelToTime={pixelToTime}
          formatTime={formatTime}
          onPickWhipDragStart={handlePickWhipDragStart}
          onPickWhipDragEnd={handlePickWhipDragEnd}
          onSetClipParent={setClipParent}
        />
      );
    },
    [
      trackMap,
      clipMap,
      clips,
      selectedClipIds,
      clipDrag,
      clipTrim,
      zoom,
      scrollX,
      proxyEnabled,
      mediaFiles,
      showTranscriptMarkers,
      handleClipMouseDown,
      handleClipDoubleClick,
      handleClipContextMenu,
      handleTrimStart,
      hasKeyframes,
      timeToPixel,
      pixelToTime,
      formatTime,
      tracks,
      handlePickWhipDragStart,
      handlePickWhipDragEnd,
      setClipParent,
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

  // anyVideoSolo and anyAudioSolo are already memoized at the top of the component

  return (
    <div className={`timeline-container ${clipDrag || clipTrim ? 'is-dragging' : ''}`}>
      <TimelineControls
        isPlaying={isPlaying}
        loopPlayback={loopPlayback}
        playheadPosition={playheadPosition}
        duration={duration}
        zoom={zoom}
        snappingEnabled={snappingEnabled}
        inPoint={inPoint}
        outPoint={outPoint}
        ramPreviewEnabled={ramPreviewEnabled}
        proxyEnabled={proxyEnabled}
        currentlyGeneratingProxyId={currentlyGeneratingProxyId}
        mediaFilesWithProxy={mediaFilesWithProxyCount}
        showTranscriptMarkers={showTranscriptMarkers}
        thumbnailsEnabled={thumbnailsEnabled}
        waveformsEnabled={waveformsEnabled}
        onPlay={play}
        onPause={pause}
        onStop={stop}
        onToggleLoop={toggleLoopPlayback}
        onSetZoom={handleSetZoom}
        onToggleSnapping={toggleSnapping}
        onSetInPoint={setInPointAtPlayhead}
        onSetOutPoint={setOutPointAtPlayhead}
        onClearInOut={clearInOut}
        onToggleRamPreview={toggleRamPreviewEnabled}
        onToggleProxy={() => setProxyEnabled(!proxyEnabled)}
        onToggleTranscriptMarkers={() => setShowTranscriptMarkers(!showTranscriptMarkers)}
        onToggleThumbnails={toggleThumbnailsEnabled}
        onToggleWaveforms={toggleWaveformsEnabled}
        onAddVideoTrack={() => addTrack('video')}
        onAddAudioTrack={() => addTrack('audio')}
        onAddTextClip={() => {
          // Add text clip at playhead position on topmost video track
          const videoTrack = tracks.find(t => t.type === 'video');
          if (videoTrack) {
            addTextClip(videoTrack.id, playheadPosition);
          }
        }}
        onSetDuration={setDuration}
        onFitToWindow={handleFitToWindow}
        formatTime={formatTime}
        parseTime={parseTime}
      />

      <div className="timeline-body" ref={timelineBodyRef}>
        <div className="timeline-header-row">
          <div className="ruler-header">Time</div>
          <div className="time-ruler-wrapper">
            <TimelineRuler
              duration={duration}
              zoom={zoom}
              scrollX={scrollX}
              onRulerMouseDown={handleRulerMouseDown}
              formatTime={formatTime}
            />
          </div>
        </div>
        <div className="timeline-content-row">
          <div className="track-headers">
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
                  tracks={tracks}
                  isDimmed={isDimmed}
                  isExpanded={isExpanded}
                  dynamicHeight={dynamicHeight}
                  hasKeyframes={trackHasKeyframes(track.id)}
                  selectedClipIds={selectedClipIds}
                  clips={clips}
                  playheadPosition={playheadPosition}
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
                  onRenameTrack={(name) =>
                    useTimelineStore.getState().renameTrack(track.id, name)
                  }
                  onWheel={(e) => handleTrackHeaderWheel(e, track.id)}
                  clipKeyframes={clipKeyframes}
                  getClipKeyframes={getClipKeyframes}
                  getInterpolatedTransform={getInterpolatedTransform}
                  getInterpolatedEffects={getInterpolatedEffects}
                  addKeyframe={addKeyframe}
                  setPlayheadPosition={setPlayheadPosition}
                  setPropertyValue={setPropertyValue}
                  expandedCurveProperties={expandedCurveProperties}
                  onToggleCurveExpanded={toggleCurveExpanded}
                  onSetTrackParent={setTrackParent}
                  onTrackPickWhipDragStart={handleTrackPickWhipDragStart}
                  onTrackPickWhipDragEnd={handleTrackPickWhipDragEnd}
                />
              );
            })}
          </div>

          <div
            ref={(el) => {
              (timelineRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
              (trackLanesRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
            }}
            className={`timeline-tracks ${clipDrag ? 'dragging-clip' : ''} ${marquee ? 'marquee-selecting' : ''}`}
            onMouseDown={handleMarqueeMouseDown}
          >
            <div className="track-lanes-scroll" style={{
              transform: `translateX(-${scrollX}px)`,
              minWidth: Math.max(duration * zoom + 500, 2000), // Ensure background extends beyond visible content
            }}>
              {/* New Video Track drop zone - at TOP above video tracks */}
              {externalDrag && (
                <div
                  className={`new-track-drop-zone video ${externalDrag.newTrackType === 'video' ? 'active' : ''}`}
                  onDragOver={(e) => handleNewTrackDragOver(e, 'video')}
                  onDragEnter={(e) => {
                    e.preventDefault();
                    dragCounterRef.current++;
                  }}
                  onDragLeave={handleTrackDragLeave}
                  onDrop={(e) => handleNewTrackDrop(e, 'video')}
                >
                  <span className="drop-zone-label">+ Drop to create new Video Track</span>
                  {externalDrag.newTrackType === 'video' && (
                    <div
                      className="timeline-clip-preview video"
                      style={{
                        left: timeToPixel(externalDrag.startTime),
                        width: timeToPixel(externalDrag.duration ?? 5),
                      }}
                    >
                      <div className="clip-content">
                        <span className="clip-name">New clip</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

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
                selectedClipIds={selectedClipIds}
                selectedKeyframeIds={selectedKeyframeIds}
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
                clipKeyframes={clipKeyframes}
                renderKeyframeDiamonds={renderKeyframeDiamonds}
                timeToPixel={timeToPixel}
                pixelToTime={pixelToTime}
                expandedCurveProperties={expandedCurveProperties}
                onSelectKeyframe={selectKeyframe}
                onMoveKeyframe={moveKeyframe}
                onUpdateBezierHandle={updateBezierHandle}
              />
            );
          })}

          {/* New audio track preview for linked video audio */}
          {externalDrag &&
            externalDrag.isVideo &&
            externalDrag.audioTrackId === '__new_audio_track__' &&
            externalDrag.newTrackType !== 'video' && (
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

          {/* New Audio Track drop zone - at BOTTOM below audio tracks */}
          {externalDrag && (
            <div
              className={`new-track-drop-zone audio ${externalDrag.newTrackType === 'audio' ? 'active' : ''}`}
              onDragOver={(e) => handleNewTrackDragOver(e, 'audio')}
              onDragEnter={(e) => {
                e.preventDefault();
                dragCounterRef.current++;
              }}
              onDragLeave={handleTrackDragLeave}
              onDrop={(e) => handleNewTrackDrop(e, 'audio')}
            >
              <span className="drop-zone-label">+ Drop to create new Audio Track</span>
              {externalDrag.newTrackType === 'audio' && (
                <div
                  className="timeline-clip-preview audio"
                  style={{
                    left: timeToPixel(externalDrag.startTime),
                    width: timeToPixel(externalDrag.duration ?? 5),
                  }}
                >
                  <div className="clip-content">
                    <span className="clip-name">New clip</span>
                  </div>
                </div>
              )}
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

              {/* Marquee selection rectangle */}
              {marquee && (
                <div
                  className="marquee-selection"
                  style={{
                    left: Math.min(marquee.startX, marquee.currentX),
                    top: Math.min(marquee.startY, marquee.currentY),
                    width: Math.abs(marquee.currentX - marquee.startX),
                    height: Math.abs(marquee.currentY - marquee.startY),
                  }}
                />
              )}

              {/* Parent-child link lines overlay */}
              <svg
                className="parent-child-links-overlay"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                  overflow: 'visible',
                }}
              >
                {clips.filter(c => c.parentClipId).map(childClip => {
                  const parentClip = clips.find(c => c.id === childClip.parentClipId);
                  if (!parentClip) return null;

                  // Apply drag offset for real-time updates during drag
                  let adjustedChildClip = childClip;
                  let adjustedParentClip = parentClip;

                  if (clipDrag) {
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
                      if (clipDrag.clipId === childClip.id) {
                        adjustedChildClip = { ...childClip, startTime: tempStartTime, trackId: clipDrag.currentTrackId };
                      }
                      if (clipDrag.clipId === parentClip.id) {
                        adjustedParentClip = { ...parentClip, startTime: tempStartTime, trackId: clipDrag.currentTrackId };
                      }
                    }
                  }

                  // Calculate Y position for track
                  const getTrackYPosition = (trackId: string): number => {
                    let y = 24; // Offset for new track drop zone
                    for (const track of tracks) {
                      if (track.id === trackId) {
                        return y + track.height / 2;
                      }
                      y += getExpandedTrackHeight(track.id, track.height);
                    }
                    return y;
                  };

                  return (
                    <ParentChildLink
                      key={childClip.id}
                      childClip={adjustedChildClip}
                      parentClip={adjustedParentClip}
                      tracks={tracks}
                      zoom={zoom}
                      scrollX={0} // Already in scrolled container
                      trackHeaderWidth={0} // Already offset
                      getTrackYPosition={getTrackYPosition}
                    />
                  );
                })}
              </svg>


            </div>{/* track-lanes-scroll */}
          </div>{/* timeline-tracks */}
        </div>{/* timeline-content-row */}

        {/* Playhead - spans from ruler through all tracks */}
        <div
          className="playhead"
          style={{ left: timeToPixel(playheadPosition) - scrollX + 150 }}
          onMouseDown={handlePlayheadMouseDown}
        >
          <div className="playhead-head" />
          <div className="playhead-line" />
        </div>
      </div>{/* timeline-body */}

      {/* Timeline Navigator - horizontal scrollbar with zoom handles */}
      <TimelineNavigator
        duration={duration}
        scrollX={scrollX}
        zoom={zoom}
        viewportWidth={timelineBodyRef.current?.querySelector('.track-lanes-scroll')?.parentElement?.clientWidth ?? 800}
        minZoom={MIN_ZOOM}
        maxZoom={MAX_ZOOM}
        onScrollChange={setScrollX}
        onZoomChange={handleSetZoom}
      />

      {/* Pick whip drag line - physics cable (clip parenting) */}
      {pickWhipDrag && (
        <svg
          className="pick-whip-drag-overlay"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            pointerEvents: 'none',
            zIndex: 9999,
          }}
        >
          <PhysicsCable
            startX={pickWhipDrag.startX}
            startY={pickWhipDrag.startY}
            endX={pickWhipDrag.currentX}
            endY={pickWhipDrag.currentY}
            isPreview={true}
          />
        </svg>
      )}

      {/* Track pick whip drag line - physics cable (layer parenting) */}
      {trackPickWhipDrag && (
        <svg
          className="pick-whip-drag-overlay"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            pointerEvents: 'none',
            zIndex: 9999,
          }}
        >
          <PhysicsCable
            startX={trackPickWhipDrag.startX}
            startY={trackPickWhipDrag.startY}
            endX={trackPickWhipDrag.currentX}
            endY={trackPickWhipDrag.currentY}
            isPreview={true}
          />
        </svg>
      )}

      <TimelineContextMenu
        contextMenu={contextMenu}
        setContextMenu={setContextMenu}
        clipMap={clipMap}
        selectedClipIds={selectedClipIds}
        selectClip={selectClip}
        removeClip={removeClip}
        splitClipAtPlayhead={splitClipAtPlayhead}
        toggleClipReverse={toggleClipReverse}
        unlinkGroup={unlinkGroup}
        generateWaveformForClip={generateWaveformForClip}
        setMulticamDialogOpen={setMulticamDialogOpen}
        proxyFolderName={proxyFolderName}
        showInExplorer={showInExplorer}
        pickProxyFolder={pickProxyFolder}
      />

      {/* Multicam Dialog */}
      <MulticamDialog
        open={multicamDialogOpen}
        onClose={() => setMulticamDialogOpen(false)}
        selectedClipIds={selectedClipIds}
      />
    </div>
  );
}
