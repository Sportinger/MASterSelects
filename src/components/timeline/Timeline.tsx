// Timeline component - Main orchestrator for video editing timeline
// Composes TimelineRuler, TimelineControls, TimelineHeader, TimelineTrack, TimelineClip, TimelineKeyframes

import { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import {
  // Core data selectors (subscribe individually for optimal re-renders)
  selectTracks,
  selectClips,
  selectPlayheadPosition,
  selectDuration,
  selectZoom,
  selectScrollX,
  selectIsPlaying,
  selectSelectedClipIds,
  selectMarkers,
  // UI state selectors
  selectSnappingEnabled,
  selectInPoint,
  selectOutPoint,
  selectLoopPlayback,
  selectToolMode,
  selectThumbnailsEnabled,
  selectWaveformsEnabled,
  selectIsDraggingPlayhead,
  // Preview/export selectors
  selectRamPreviewEnabled,
  selectRamPreviewProgress,
  selectRamPreviewRange,
  selectIsRamPreviewing,
  selectIsExporting,
  selectExportProgress,
  selectExportRange,
  selectIsProxyCaching,
  selectProxyCacheProgress,
  // Keyframe selectors
  selectSelectedKeyframeIds,
  selectClipKeyframes,
  selectExpandedCurveProperties,
  // Action selectors (stable references, grouped for convenience)
  selectPlaybackActions,
  selectTrackActions,
  selectClipActions,
  selectTransformGetters,
  selectKeyframeActions,
  selectInOutActions,
  selectViewActions,
  selectPreviewActions,
  selectToolActions,
  selectMarkerActions,
  selectClipboardActions,
} from '../../stores/timeline/selectors';
import type { AnimatableProperty, TimelineClip as TimelineClipType } from '../../types';
import { useMediaStore } from '../../stores/mediaStore';

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
import { VerticalScrollbar } from './VerticalScrollbar';
import { useTimelineKeyboard } from './hooks/useTimelineKeyboard';
import { useTimelineZoom } from './hooks/useTimelineZoom';
import { usePlayheadDrag } from './hooks/usePlayheadDrag';
import { TimelineContextMenu, useClipContextMenu } from './TimelineContextMenu';
import { useMarqueeSelection } from './hooks/useMarqueeSelection';
import { useClipTrim } from './hooks/useClipTrim';
import { useClipDrag } from './hooks/useClipDrag';
import { useClipFade } from './hooks/useClipFade';
import { useLayerSync } from './hooks/useLayerSync';
import { usePlaybackLoop } from './hooks/usePlaybackLoop';
import { useVideoPreload } from './hooks/useVideoPreload';
import { useAutoFeatures } from './hooks/useAutoFeatures';
import { useExternalDrop } from './hooks/useExternalDrop';
import { useTransitionDrop } from './hooks/useTransitionDrop';
import { usePickWhipDrag } from './hooks/usePickWhipDrag';
import { useTimelineHelpers } from './hooks/useTimelineHelpers';
import { usePlayheadSnap } from './hooks/usePlayheadSnap';
import { useMarkerDrag } from './hooks/useMarkerDrag';
import { MIN_ZOOM, MAX_ZOOM } from '../../stores/timeline/constants';
import type { ContextMenuState } from './types';

export function Timeline() {
  // ===========================================
  // OPTIMIZED STORE SUBSCRIPTIONS
  // Each selector creates a separate subscription, so the component
  // only re-renders when the specific value it uses changes.
  // ===========================================

  // Core data (frequently changing)
  const tracks = useTimelineStore(selectTracks);
  const clips = useTimelineStore(selectClips);
  const playheadPosition = useTimelineStore(selectPlayheadPosition);
  const duration = useTimelineStore(selectDuration);
  const zoom = useTimelineStore(selectZoom);
  const scrollX = useTimelineStore(selectScrollX);
  const isPlaying = useTimelineStore(selectIsPlaying);
  const selectedClipIds = useTimelineStore(selectSelectedClipIds);
  const markers = useTimelineStore(selectMarkers);

  // UI state (less frequent changes)
  const snappingEnabled = useTimelineStore(selectSnappingEnabled);
  const inPoint = useTimelineStore(selectInPoint);
  const outPoint = useTimelineStore(selectOutPoint);
  const loopPlayback = useTimelineStore(selectLoopPlayback);
  const toolMode = useTimelineStore(selectToolMode);
  const thumbnailsEnabled = useTimelineStore(selectThumbnailsEnabled);
  const waveformsEnabled = useTimelineStore(selectWaveformsEnabled);
  const isDraggingPlayhead = useTimelineStore(selectIsDraggingPlayhead);

  // Preview/export state
  const ramPreviewEnabled = useTimelineStore(selectRamPreviewEnabled);
  const ramPreviewProgress = useTimelineStore(selectRamPreviewProgress);
  const ramPreviewRange = useTimelineStore(selectRamPreviewRange);
  const isRamPreviewing = useTimelineStore(selectIsRamPreviewing);
  const isExporting = useTimelineStore(selectIsExporting);
  const exportProgress = useTimelineStore(selectExportProgress);
  const exportRange = useTimelineStore(selectExportRange);
  const isProxyCaching = useTimelineStore(selectIsProxyCaching);
  const proxyCacheProgress = useTimelineStore(selectProxyCacheProgress);

  // Keyframe state
  const selectedKeyframeIds = useTimelineStore(selectSelectedKeyframeIds);
  const clipKeyframes = useTimelineStore(selectClipKeyframes);
  const expandedCurveProperties = useTimelineStore(selectExpandedCurveProperties);

  // ===========================================
  // STABLE ACTION REFERENCES
  // Actions don't change, so grouping them doesn't cause re-renders.
  // ===========================================

  const {
    play, pause, stop, playForward, playReverse,
    setPlayheadPosition, setDraggingPlayhead,
  } = useTimelineStore(selectPlaybackActions);

  const {
    addTrack, isTrackExpanded, toggleTrackExpanded,
    getExpandedTrackHeight, trackHasKeyframes, setTrackParent,
  } = useTimelineStore(selectTrackActions);

  const {
    addClip, addCompClip, addTextClip, moveClip, trimClip,
    removeClip, selectClip, unlinkGroup, splitClip, splitClipAtPlayhead,
    toggleClipReverse, updateClipTransform, setClipParent, generateWaveformForClip,
  } = useTimelineStore(selectClipActions);

  const {
    getInterpolatedTransform, getInterpolatedEffects, getInterpolatedSpeed,
    getSourceTimeForClip, getSnappedPosition, getPositionWithResistance,
  } = useTimelineStore(selectTransformGetters);

  const {
    getClipKeyframes, selectKeyframe, deselectAllKeyframes, hasKeyframes,
    addKeyframe, moveKeyframe, updateKeyframe, removeKeyframe,
    setPropertyValue, toggleCurveExpanded, updateBezierHandle,
  } = useTimelineStore(selectKeyframeActions);

  const {
    setInPoint, setOutPoint, setInPointAtPlayhead,
    setOutPointAtPlayhead, clearInOut,
  } = useTimelineStore(selectInOutActions);

  const {
    setZoom, setScrollX, setDuration, toggleSnapping,
  } = useTimelineStore(selectViewActions);

  const {
    toggleLoopPlayback, toggleRamPreviewEnabled, startRamPreview,
    cancelRamPreview, getCachedRanges, getProxyCachedRanges,
    startProxyCachePreload, cancelProxyCachePreload,
  } = useTimelineStore(selectPreviewActions);

  const {
    setToolMode, toggleCutTool, toggleThumbnailsEnabled, toggleWaveformsEnabled,
  } = useTimelineStore(selectToolActions);

  const { addMarker, moveMarker, removeMarker } = useTimelineStore(selectMarkerActions);
  const { copyClips, pasteClips } = useTimelineStore(selectClipboardActions);

  const {
    getActiveComposition,
    getOpenCompositions,
    openCompositionTab,
    proxyEnabled,
    setProxyEnabled,
    files: mediaFiles,
    currentlyGeneratingProxyId,
    showInExplorer,
  } = useMediaStore();
  const activeComposition = getActiveComposition() ?? null;
  const openCompositions = getOpenCompositions();

  const timelineRef = useRef<HTMLDivElement>(null);
  const timelineBodyRef = useRef<HTMLDivElement>(null);
  const trackLanesRef = useRef<HTMLDivElement>(null);
  const scrollWrapperRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Cut tool hover state (shared across linked clips)
  const [cutHoverInfo, setCutHoverInfo] = useState<{ clipId: string; time: number } | null>(null);
  const handleCutHover = useCallback((clipId: string | null, time: number | null) => {
    if (clipId && time !== null) {
      setCutHoverInfo({ clipId, time });
    } else {
      setCutHoverInfo(null);
    }
  }, []);

  // Cut at position handler - splits clip and returns to select mode
  const handleCutAtPosition = useCallback((clipId: string, time: number) => {
    splitClip(clipId, time);
    setToolMode('select');
  }, [splitClip, setToolMode]);

  // Performance: Create lookup maps for O(1) clip/track access (must be before hooks that use them)
  const clipMap = useMemo(() => new Map(clips.map(c => [c.id, c])), [clips]);
  const trackMap = useMemo(() => new Map(tracks.map(t => [t.id, t])), [tracks]);

  // Time helpers - extracted to hook
  const {
    timeToPixel,
    pixelToTime,
    gridSize,
    formatTime,
    parseTime,
    getClipsAtTime,
    getSnapTargetTimes,
  } = useTimelineHelpers({ zoom, clips, getClipKeyframes });

  // Clip dragging - extracted to hook
  const { clipDrag, handleClipMouseDown, handleClipDoubleClick } = useClipDrag({
    trackLanesRef,
    timelineRef,
    clips,
    tracks,
    clipMap,
    selectedClipIds,
    scrollX,
    snappingEnabled,
    selectClip,
    moveClip,
    openCompositionTab,
    pixelToTime,
    getSnappedPosition,
    getPositionWithResistance,
  });

  // Clip trimming - extracted to hook
  const { clipTrim, handleTrimStart } = useClipTrim({
    clipMap,
    selectClip,
    trimClip,
    moveClip,
    pixelToTime,
  });

  // Clip fade (fade-in/out handles) - extracted to hook
  const { clipFade, handleFadeStart, getFadeInDuration, getFadeOutDuration } = useClipFade({
    clipMap,
    addKeyframe,
    removeKeyframe,
    moveKeyframe,
    getClipKeyframes,
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
    isPlaying,
    setPlayheadPosition,
    setDraggingPlayhead,
    setInPoint,
    setOutPoint,
    cancelRamPreview,
    pause,
    pixelToTime,
  });

  // External file drag & drop - extracted to hook
  const {
    externalDrag,
    dragCounterRef,
    handleTrackDragEnter,
    handleTrackDragOver,
    handleTrackDragLeave,
    handleTrackDrop,
    handleNewTrackDragOver,
    handleNewTrackDrop,
  } = useExternalDrop({
    timelineRef,
    scrollX,
    tracks,
    clips,
    pixelToTime,
    addTrack,
    addClip,
    addCompClip,
    addTextClip,
  });

  // Transition drop handling for drag-and-drop transitions between clips
  const {
    activeJunction,
    handleDragOver: handleTransitionDragOver,
    handleDrop: handleTransitionDrop,
    handleDragLeave: handleTransitionDragLeave,
    isTransitionDrag,
  } = useTransitionDrop();

  // Combined drag handlers that check for transition drops first
  const handleCombinedDragOver = useCallback((e: React.DragEvent, trackId: string) => {
    if (isTransitionDrag(e)) {
      const rect = timelineRef.current?.getBoundingClientRect();
      if (rect) {
        const mouseX = e.clientX - rect.left + scrollX;
        const mouseTime = pixelToTime(mouseX);
        handleTransitionDragOver(e, trackId, mouseTime);
      }
    } else {
      handleTrackDragOver(e, trackId);
    }
  }, [isTransitionDrag, handleTransitionDragOver, handleTrackDragOver, scrollX, pixelToTime]);

  const handleCombinedDrop = useCallback((e: React.DragEvent, trackId: string) => {
    if (isTransitionDrag(e)) {
      const rect = timelineRef.current?.getBoundingClientRect();
      if (rect) {
        const mouseX = e.clientX - rect.left + scrollX;
        const mouseTime = pixelToTime(mouseX);
        handleTransitionDrop(e, trackId, mouseTime);
      }
    } else {
      handleTrackDrop(e, trackId);
    }
  }, [isTransitionDrag, handleTransitionDrop, handleTrackDrop, scrollX, pixelToTime]);

  const handleCombinedDragLeave = useCallback((e: React.DragEvent) => {
    handleTransitionDragLeave();
    handleTrackDragLeave(e);
  }, [handleTransitionDragLeave, handleTrackDragLeave]);

  // Vertical scroll position (custom scrollbar)
  const [scrollY, setScrollY] = useState(0);

  // Context menu state for clip right-click
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const handleClipContextMenu = useClipContextMenu(selectedClipIds, selectClip, setContextMenu);

  // Transcript markers visibility toggle
  const [showTranscriptMarkers, setShowTranscriptMarkers] = useState(true);

  // Multicam dialog state
  const [multicamDialogOpen, setMulticamDialogOpen] = useState(false);

  // Marker drag operations - extracted to hook
  const {
    timelineMarkerDrag,
    markerCreateDrag,
    handleTimelineMarkerMouseDown,
    handleMarkerButtonDragStart,
  } = useMarkerDrag({
    timelineRef,
    timelineBodyRef,
    markers,
    scrollX,
    snappingEnabled,
    duration,
    playheadPosition,
    inPoint,
    outPoint,
    pixelToTime,
    getSnapTargetTimes,
    moveMarker,
    addMarker,
  });

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

  // Pick whip drag - extracted to hook
  const {
    pickWhipDrag,
    handlePickWhipDragStart,
    handlePickWhipDragEnd,
    trackPickWhipDrag,
    handleTrackPickWhipDragStart,
    handleTrackPickWhipDragEnd,
  } = usePickWhipDrag({ setClipParent, setTrackParent });

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

  // Calculate total content height for vertical scrollbar
  const contentHeight = useMemo(() => {
    let totalHeight = 0;
    for (const track of tracks) {
      const isExpanded = isTrackExpanded(track.id);
      totalHeight += isExpanded ? getExpandedTrackHeight(track.id, track.height) : track.height;
    }
    return totalHeight;
  }, [tracks, isTrackExpanded, getExpandedTrackHeight]);

  // Track viewport height for scrollbar
  const [viewportHeight, setViewportHeight] = useState(300);

  // Update viewport height on resize
  useEffect(() => {
    const updateViewportHeight = () => {
      if (scrollWrapperRef.current) {
        setViewportHeight(scrollWrapperRef.current.clientHeight);
      }
    };
    updateViewportHeight();
    window.addEventListener('resize', updateViewportHeight);
    return () => window.removeEventListener('resize', updateViewportHeight);
  }, []);

  // Performance: Memoize proxy-ready file count
  const mediaFilesWithProxyCount = useMemo(
    () => mediaFiles.filter((f) => f.proxyStatus === 'ready').length,
    [mediaFiles]
  );

  // Keyboard shortcuts - extracted to hook
  useTimelineKeyboard({
    isPlaying,
    play,
    pause,
    playForward,
    playReverse,
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
    copyClips,
    pasteClips,
    toolMode,
    toggleCutTool,
    clipMap,
    activeComposition,
    playheadPosition,
    duration,
    setPlayheadPosition,
    addMarker,
  });

  // Auto-start RAM preview and proxy generation - extracted to hook
  useAutoFeatures({
    ramPreviewEnabled,
    proxyEnabled,
    isPlaying,
    isDraggingPlayhead,
    isRamPreviewing,
    currentlyGeneratingProxyId,
    inPoint,
    outPoint,
    ramPreviewRange,
    clips,
    startRamPreview,
    cancelRamPreview,
  });

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

  // Preload upcoming video clips - extracted to hook
  useVideoPreload({
    isPlaying,
    isDraggingPlayhead,
    playheadPosition,
    clips,
  });

  // Audio master clock playback loop - extracted to hook
  usePlaybackLoop({ isPlaying });

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
    scrollY,
    duration,
    playheadPosition,
    contentHeight,
    viewportHeight,
    setZoom,
    setScrollX,
    setScrollY,
  });

  // Playhead snapping during drag - extracted to hook
  usePlayheadSnap({
    isDraggingPlayhead,
    timelineRef,
    scrollX,
    duration,
    snappingEnabled,
    pixelToTime,
    getSnapTargetTimes,
    setPlayheadPosition,
    setDraggingPlayhead,
  });

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

  // Render a clip
  const renderClip = useCallback(
    (clip: TimelineClipType, trackId: string) => {
      const track = trackMap.get(trackId);
      if (!track) return null;

      const isDragging = clipDrag?.clipId === clip.id;
      const isTrimming = clipTrim?.clipId === clip.id;
      const isFading = clipFade?.clipId === clip.id;

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
          f.id === clip.mediaFileId ||
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
          isFading={isFading}
          isLinkedToDragging={!!isLinkedToDragging}
          isLinkedToTrimming={!!isLinkedToTrimming}
          clipDrag={clipDrag}
          clipTrim={clipTrim}
          clipFade={clipFade}
          zoom={zoom}
          scrollX={scrollX}
          timelineRef={timelineRef}
          proxyEnabled={proxyEnabled}
          proxyStatus={mediaFile?.proxyStatus}
          proxyProgress={mediaFile?.proxyProgress || 0}
          showTranscriptMarkers={showTranscriptMarkers}
          toolMode={toolMode}
          snappingEnabled={snappingEnabled}
          playheadPosition={playheadPosition}
          cutHoverInfo={cutHoverInfo}
          onCutHover={handleCutHover}
          onMouseDown={(e) => handleClipMouseDown(e, clip.id)}
          onDoubleClick={(e) => handleClipDoubleClick(e, clip.id)}
          onContextMenu={(e) => handleClipContextMenu(e, clip.id)}
          onTrimStart={(e, edge) => handleTrimStart(e, clip.id, edge)}
          onFadeStart={(e, edge) => handleFadeStart(e, clip.id, edge)}
          onCutAtPosition={handleCutAtPosition}
          hasKeyframes={hasKeyframes}
          fadeInDuration={getFadeInDuration(clip.id)}
          fadeOutDuration={getFadeOutDuration(clip.id)}
          opacityKeyframes={getClipKeyframes(clip.id).filter(k => k.property === 'opacity')}
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
      clipFade,
      zoom,
      scrollX,
      proxyEnabled,
      mediaFiles,
      showTranscriptMarkers,
      toolMode,
      snappingEnabled,
      playheadPosition,
      cutHoverInfo,
      handleCutHover,
      handleClipMouseDown,
      handleClipDoubleClick,
      handleClipContextMenu,
      handleTrimStart,
      handleFadeStart,
      handleCutAtPosition,
      hasKeyframes,
      getFadeInDuration,
      getFadeOutDuration,
      getClipKeyframes,
      clipKeyframes,
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
        toolMode={toolMode}
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
        isProxyCaching={isProxyCaching}
        proxyCacheProgress={proxyCacheProgress}
        onStartProxyCachePreload={startProxyCachePreload}
        onCancelProxyCachePreload={cancelProxyCachePreload}
        onToggleTranscriptMarkers={() => setShowTranscriptMarkers(!showTranscriptMarkers)}
        onToggleThumbnails={toggleThumbnailsEnabled}
        onToggleWaveforms={toggleWaveformsEnabled}
        onToggleCutTool={toggleCutTool}
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
        <div className="timeline-body-content">
          <div className="timeline-header-row">
            <div className="ruler-header">
              <span>Time</span>
              <button
                className={`add-marker-btn ${markerCreateDrag?.isDragging ? 'dragging' : ''}`}
                onMouseDown={handleMarkerButtonDragStart}
                title="Drag to place marker, or press M"
              >
                M
              </button>
            </div>
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
          <div className="timeline-scroll-wrapper" ref={scrollWrapperRef}>
            <div className="timeline-content-row" ref={contentRef} style={{ transform: `translateY(-${scrollY}px)` }}>
          <div className="track-headers">
            {/* New video track preview header - appears when dragging over new track zone */}
            {externalDrag && (
              <div
                className={`track-header-preview video ${externalDrag.newTrackType === 'video' ? 'active' : ''}`}
                style={{ height: 60 }}
              >
                <span className="track-header-preview-label">+ New Video Track</span>
              </div>
            )}
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
            {/* New audio track preview header - appears when dragging over new track zone or linked audio needs new track */}
            {/* Only show if the video has audio */}
            {externalDrag && externalDrag.hasAudio && (
              <div
                className={`track-header-preview audio ${
                  externalDrag.newTrackType === 'audio' ||
                  (externalDrag.newTrackType === 'video' && externalDrag.hasAudio) ||
                  (externalDrag.isVideo && externalDrag.audioTrackId === '__new_audio_track__')
                    ? 'active'
                    : ''
                }`}
                style={{ height: 40 }}
              >
                <span className="track-header-preview-label">+ New Audio Track</span>
              </div>
            )}
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
              ['--grid-size' as string]: `${gridSize}px`,
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
                onDrop={(e) => handleCombinedDrop(e, track.id)}
                onDragOver={(e) => handleCombinedDragOver(e, track.id)}
                onDragEnter={(e) => handleTrackDragEnter(e, track.id)}
                onDragLeave={handleCombinedDragLeave}
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

          {/* Junction highlight for transition drop */}
          {activeJunction && (
            <div
              className="transition-junction-highlight"
              style={{
                position: 'absolute',
                left: timeToPixel(activeJunction.junctionTime) - 15,
                width: 30,
                top: 0,
                bottom: 0,
                background: 'linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.4), transparent)',
                pointerEvents: 'none',
                zIndex: 100,
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: '#3b82f6',
                  color: 'white',
                  padding: '4px 10px',
                  borderRadius: 4,
                  fontSize: 11,
                  fontWeight: 500,
                  whiteSpace: 'nowrap',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                }}
              >
                Drop transition
              </div>
            </div>
          )}

          {/* Render existing transitions as junction elements */}
          {clips.filter(c => c.transitionOut).map(clipA => {
            const clipB = clips.find(c => c.id === clipA.transitionOut?.linkedClipId);
            if (!clipB || !clipA.transitionOut) return null;

            const track = tracks.find(t => t.id === clipA.trackId);
            if (!track) return null;

            // Calculate track position
            const trackIndex = tracks.indexOf(track);
            const trackTop = tracks
              .slice(0, trackIndex)
              .reduce((sum, t) => sum + (isTrackExpanded(t.id) ? getExpandedTrackHeight(t.id, t.height) : t.height), 0);
            const trackHeight = isTrackExpanded(track.id) ? getExpandedTrackHeight(track.id, track.height) : track.height;

            // Transition spans from clipB.startTime to clipA.startTime + clipA.duration
            const transitionStart = clipB.startTime;
            const transitionEnd = clipA.startTime + clipA.duration;
            const transitionWidth = timeToPixel(transitionEnd - transitionStart);
            const transitionLeft = timeToPixel(transitionStart);

            return (
              <div
                key={clipA.transitionOut.id}
                className="timeline-transition"
                style={{
                  position: 'absolute',
                  left: transitionLeft,
                  top: trackTop,
                  width: Math.max(transitionWidth, 20),
                  height: trackHeight,
                  pointerEvents: 'none',
                  zIndex: 50,
                }}
              >
                {/* Transition visual */}
                <div
                  style={{
                    position: 'absolute',
                    inset: 4,
                    background: 'linear-gradient(90deg, rgba(74, 158, 255, 0.3), rgba(255, 107, 74, 0.3))',
                    borderRadius: 4,
                    border: '1px solid rgba(255,255,255,0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.6 }}>
                    <path d="M7 4v16M17 4v16M7 12h10" stroke="white" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </div>
              </div>
            );
          })}

          {/* New audio track preview for linked video audio */}
          {/* Only show if video has audio (hasAudio !== false) */}
          {externalDrag &&
            externalDrag.isVideo &&
            externalDrag.hasAudio !== false &&
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
          {/* Only show for audio files OR videos with audio (hasAudio !== false) */}
          {externalDrag && (externalDrag.newTrackType === 'audio' || externalDrag.hasAudio !== false) && (
            <div
              className={`new-track-drop-zone audio ${
                externalDrag.newTrackType === 'audio' ||
                (externalDrag.newTrackType === 'video' && externalDrag.hasAudio !== false)
                  ? 'active'
                  : ''
              }`}
              onDragOver={(e) => handleNewTrackDragOver(e, 'audio')}
              onDragEnter={(e) => {
                e.preventDefault();
                dragCounterRef.current++;
              }}
              onDragLeave={handleTrackDragLeave}
              onDrop={(e) => handleNewTrackDrop(e, 'audio')}
            >
              <span className="drop-zone-label">+ Drop to create new Audio Track</span>
              {/* Show audio preview when dropping audio OR when dropping video with audio */}
              {(externalDrag.newTrackType === 'audio' ||
                (externalDrag.newTrackType === 'video' && externalDrag.hasAudio !== false)) && (
                <div
                  className="timeline-clip-preview audio"
                  style={{
                    left: timeToPixel(externalDrag.startTime),
                    width: timeToPixel(externalDrag.duration ?? 5),
                  }}
                >
                  <div className="clip-content">
                    <span className="clip-name">
                      {externalDrag.newTrackType === 'video' ? 'Audio (linked)' : 'New clip'}
                    </span>
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

          {/* Export Progress Overlay */}
          {isExporting && exportRange && (
            <>
              {/* Progress bar - grows based on percentage (0-100%) */}
              <div
                className="timeline-export-overlay"
                style={{
                  left: timeToPixel(exportRange.start),
                  width: timeToPixel((exportRange.end - exportRange.start) * ((exportProgress ?? 0) / 100)),
                }}
              />
              {/* Percentage display - at end of progress bar */}
              <div
                className="timeline-export-text"
                style={{
                  left: timeToPixel(exportRange.start + (exportRange.end - exportRange.start) * ((exportProgress ?? 0) / 100)) - 10,
                  transform: 'translateX(-100%)',
                }}
              >
                {Math.round(exportProgress ?? 0)}%
              </div>
            </>
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

          {/* Proxy frame cache indicator (yellow) */}
          {(() => {
            const ranges = getProxyCachedRanges();
            if (ranges.length > 0) console.log('[Timeline] Proxy cached ranges:', ranges);
            return ranges;
          })().map((range, i) => (
            <div
              key={`proxy-${i}`}
              className="proxy-cache-indicator"
              style={{
                left: timeToPixel(range.start),
                width: Math.max(2, timeToPixel(range.end - range.start)),
              }}
              title={`Proxy cached: ${formatTime(range.start)} - ${formatTime(range.end)}`}
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
          </div>{/* timeline-scroll-wrapper */}

          {/* Playhead - spans from ruler through all tracks */}
          <div
            className="playhead"
            style={{ left: timeToPixel(playheadPosition) - scrollX + 150 }}
            onMouseDown={handlePlayheadMouseDown}
          >
            <div className="playhead-head" />
            <div className="playhead-line" />
          </div>

          {/* Timeline markers - span from ruler through all tracks like playhead */}
          {markers.map(marker => (
            <div
              key={marker.id}
              className={`timeline-marker ${timelineMarkerDrag?.markerId === marker.id ? 'dragging' : ''}`}
              style={{
                left: timeToPixel(marker.time) - scrollX + 150,
                '--marker-color': marker.color,
              } as React.CSSProperties}
              title={`${marker.label || 'Marker'}: ${formatTime(marker.time)} (drag to move, right-click to delete)`}
              onMouseDown={(e) => handleTimelineMarkerMouseDown(e, marker.id)}
              onContextMenu={(e) => {
                e.preventDefault();
                e.stopPropagation();
                removeMarker(marker.id);
              }}
            >
              <div className="timeline-marker-head">M</div>
              <div className="timeline-marker-line" />
            </div>
          ))}

          {/* Ghost marker for drag-to-create */}
          {markerCreateDrag && markerCreateDrag.isOverTimeline && (
            <div
              className={`timeline-marker ghost ${markerCreateDrag.dropAnimating ? 'drop-animation' : ''}`}
              style={{
                left: timeToPixel(markerCreateDrag.currentTime) - scrollX + 150,
                '--marker-color': '#00d4ff',
              } as React.CSSProperties}
            >
              <div className="timeline-marker-head">M</div>
              <div className="timeline-marker-line" />
            </div>
          )}
        </div>{/* timeline-body-content */}

        {/* Vertical Scrollbar */}
        <VerticalScrollbar
          scrollY={scrollY}
          contentHeight={contentHeight}
          viewportHeight={viewportHeight}
          onScrollChange={setScrollY}
        />
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
        showInExplorer={showInExplorer}
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
