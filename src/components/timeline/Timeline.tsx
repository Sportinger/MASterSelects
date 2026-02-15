// Timeline component - Main orchestrator for video editing timeline
// Composes TimelineRuler, TimelineControls, TimelineHeader, TimelineTrack, TimelineClip, TimelineKeyframes

import { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timeline';
import {
  // Grouped state selectors (6 subscriptions instead of 29)
  selectCoreData,
  selectPlaybackState,
  selectViewState,
  selectUISettings,
  selectPreviewExportState,
  selectKeyframeState,
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
import { TimelineNavigator } from './TimelineNavigator';
import { TimelineOverlays } from './components/TimelineOverlays';
import { TransitionOverlays } from './components/TransitionOverlays';
import { PickWhipCables } from './components/PickWhipCables';
import { ParentChildLinksOverlay } from './components/ParentChildLinksOverlay';
import { VerticalScrollbar } from './VerticalScrollbar';
import { SlotGrid } from './SlotGrid';
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
  // GROUPED STORE SUBSCRIPTIONS (6 instead of 29)
  // useShallow does shallow comparison on each key, so we only
  // re-render when an actual value changes within the group.
  // ===========================================

  // Core timeline structure (changes on edits)
  const { tracks, clips, duration, selectedClipIds, markers } =
    useTimelineStore(useShallow(selectCoreData));

  // Playback state (changes every frame during playback)
  const { playheadPosition, isPlaying, isDraggingPlayhead } =
    useTimelineStore(useShallow(selectPlaybackState));

  // View state (changes on zoom/scroll)
  const { zoom, scrollX } =
    useTimelineStore(useShallow(selectViewState));

  // Slot grid progress - direct selector for reliable reactivity
  const slotGridProgress = useTimelineStore(state => state.slotGridProgress);

  // UI settings (rarely changes)
  const { snappingEnabled, inPoint, outPoint, loopPlayback, toolMode, thumbnailsEnabled, waveformsEnabled } =
    useTimelineStore(useShallow(selectUISettings));

  // Preview/export state
  const { ramPreviewEnabled, ramPreviewProgress, ramPreviewRange, isRamPreviewing, isExporting, exportProgress, exportRange, isProxyCaching, proxyCacheProgress } =
    useTimelineStore(useShallow(selectPreviewExportState));

  // Keyframe state
  const { selectedKeyframeIds, clipKeyframes, expandedCurveProperties } =
    useTimelineStore(useShallow(selectKeyframeState));

  // ===========================================
  // STABLE ACTION REFERENCES
  // Actions are stable functions - get them once from getState() to avoid
  // creating new object references that would cause infinite re-renders.
  // ===========================================

  // Get actions once - they're stable and don't change
  const store = useTimelineStore.getState();

  // Playback actions
  const { play, pause, stop, playForward, playReverse, setPlayheadPosition, setDraggingPlayhead } = store;

  // Track actions
  const { addTrack, isTrackExpanded, toggleTrackExpanded, getExpandedTrackHeight, trackHasKeyframes, setTrackParent } = store;

  // Clip actions
  const {
    addClip, addCompClip, addTextClip, addSolidClip, moveClip, trimClip,
    removeClip, selectClip, unlinkGroup, splitClip, splitClipAtPlayhead,
    toggleClipReverse, updateClipTransform, setClipParent, generateWaveformForClip,
    addClipEffect,
  } = store;

  // Transform getters
  const {
    getInterpolatedTransform, getInterpolatedEffects, getInterpolatedSpeed,
    getSourceTimeForClip, getSnappedPosition, getPositionWithResistance,
  } = store;

  // Keyframe actions
  const {
    getClipKeyframes, selectKeyframe, deselectAllKeyframes, hasKeyframes,
    addKeyframe, moveKeyframe, updateKeyframe, removeKeyframe,
    setPropertyValue, toggleCurveExpanded, updateBezierHandle,
  } = store;

  // In/out point actions
  const { setInPoint, setOutPoint, setInPointAtPlayhead, setOutPointAtPlayhead, clearInOut } = store;

  // View actions
  const { setZoom, setScrollX, setDuration, toggleSnapping } = store;

  // Preview actions
  const {
    toggleLoopPlayback, toggleRamPreviewEnabled, startRamPreview,
    cancelRamPreview, getCachedRanges, getProxyCachedRanges,
    startProxyCachePreload, cancelProxyCachePreload,
  } = store;

  // Tool actions
  const { setToolMode, toggleCutTool, toggleThumbnailsEnabled, toggleWaveformsEnabled } = store;

  // Marker actions
  const { addMarker, moveMarker, removeMarker } = store;

  // Clipboard actions
  const { copyClips, pasteClips, copyKeyframes, pasteKeyframes } = store;

  const getActiveComposition = useMediaStore(state => state.getActiveComposition);
  const getOpenCompositions = useMediaStore(state => state.getOpenCompositions);
  const openCompositionTab = useMediaStore(state => state.openCompositionTab);
  const proxyEnabled = useMediaStore(state => state.proxyEnabled);
  const mediaFiles = useMediaStore(state => state.files);
  const currentlyGeneratingProxyId = useMediaStore(state => state.currentlyGeneratingProxyId);
  const showInExplorer = useMediaStore(state => state.showInExplorer);
  const activeComposition = getActiveComposition() ?? null;
  const openCompositions = getOpenCompositions();

  const timelineRef = useRef<HTMLDivElement>(null);
  const timelineBodyRef = useRef<HTMLDivElement>(null);
  const trackLanesRef = useRef<HTMLDivElement>(null);
  const scrollWrapperRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Composition switch animation phase for tracks/ruler
  const clipAnimationPhase = useTimelineStore(s => s.clipAnimationPhase);

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

  // Stable callbacks for TimelineControls (avoids re-renders from inline arrows)
  const toggleProxyEnabled = useMediaStore(state => state.toggleProxyEnabled);

  // Use store toggle directly (no useCallback needed - stable store reference)

  const handleAddVideoTrack = useCallback(() => addTrack('video'), [addTrack]);
  const handleAddAudioTrack = useCallback(() => addTrack('audio'), [addTrack]);

  const handleAddTextClip = useCallback(() => {
    const state = useTimelineStore.getState();
    const videoTrack = state.tracks.find(t => t.type === 'video');
    if (videoTrack) {
      addTextClip(videoTrack.id, state.playheadPosition);
    }
  }, [addTextClip]);

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
    tracks,
    addKeyframe,
    removeKeyframe,
    moveKeyframe,
    getClipKeyframes,
    addClipEffect,
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
    addSolidClip,
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

  // Transcript markers visibility toggle (from store for persistence)
  const showTranscriptMarkers = useTimelineStore(s => s.showTranscriptMarkers);
  const toggleTranscriptMarkers = useTimelineStore(s => s.toggleTranscriptMarkers);

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

  // Subscribe to curveEditorHeight for snap position recalculation
  const curveEditorHeight = useTimelineStore(s => s.curveEditorHeight);

  // Calculate total content height and track snap positions for vertical scrollbar
  // Dependencies: tracks, expansion state, curve editor state, selected clips (affects property rows)
  const { contentHeight, trackSnapPositions } = useMemo(() => {
    let totalHeight = 0;
    const snapPositions: number[] = [0];
    for (const track of tracks) {
      const isExpanded = isTrackExpanded(track.id);
      totalHeight += isExpanded ? getExpandedTrackHeight(track.id, track.height) : track.height;
      snapPositions.push(totalHeight);
    }
    return { contentHeight: totalHeight, trackSnapPositions: snapPositions };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, isTrackExpanded, getExpandedTrackHeight, expandedCurveProperties, curveEditorHeight, selectedClipIds, clipKeyframes]);

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
    copyKeyframes,
    pasteKeyframes,
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

  // Video preloading is handled by VideoSyncManager.preloadUpcomingClips()
  // in the render loop — it correctly skips shared video elements (split clips)
  // and uses smart two-phase preroll. No React hook needed.

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
        // Smooth scaling: small multiplier so each wheel notch (~100 deltaY) = ~5px
        const delta = -e.deltaY * 0.05;
        useTimelineStore.getState().scaleTracksOfType(track.type, delta);
      } else if (e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        const delta = -e.deltaY * 0.05;
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
    trackSnapPositions,
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
          allKeyframeTimes={[...new Set(getClipKeyframes(clip.id).map(k => k.time))]}
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
    <div
      className={`timeline-container ${clipDrag || clipTrim ? 'is-dragging' : ''}`}
      onMouseDown={() => {
        if (useMediaStore.getState().sourceMonitorFileId) {
          useMediaStore.getState().setSourceMonitorFile(null);
        }
      }}
    >
      {slotGridProgress < 1 && <TimelineControls
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
        onToggleProxy={toggleProxyEnabled}
        isProxyCaching={isProxyCaching}
        proxyCacheProgress={proxyCacheProgress}
        onStartProxyCachePreload={startProxyCachePreload}
        onCancelProxyCachePreload={cancelProxyCachePreload}
        onToggleTranscriptMarkers={toggleTranscriptMarkers}
        onToggleThumbnails={toggleThumbnailsEnabled}
        onToggleWaveforms={toggleWaveformsEnabled}
        onToggleCutTool={toggleCutTool}
        onAddVideoTrack={handleAddVideoTrack}
        onAddAudioTrack={handleAddAudioTrack}
        onAddTextClip={handleAddTextClip}
        onSetDuration={setDuration}
        onFitToWindow={handleFitToWindow}
        formatTime={formatTime}
        parseTime={parseTime}
      />}

      <div className="timeline-body" ref={timelineBodyRef}>
        {/* SlotGrid — fades in over timeline */}
        {slotGridProgress > 0 && (
          <SlotGrid opacity={slotGridProgress} />
        )}
        {/* Timeline content — fades out with subtle scale-back */}
        <div className="timeline-body-content" style={slotGridProgress > 0 ? {
          opacity: 1 - slotGridProgress,
          transform: `scale(${1 - slotGridProgress * 0.05})`,
          transformOrigin: 'center center',
          pointerEvents: (slotGridProgress >= 0.5 ? 'none' : 'auto') as React.CSSProperties['pointerEvents'],
          display: slotGridProgress >= 1 ? 'none' as const : undefined,
        } : undefined}>
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
            <div className={`time-ruler-wrapper ${clipAnimationPhase !== 'idle' ? 'comp-switching' : ''}`}>
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
          <div className={`track-headers ${clipAnimationPhase === 'exiting' ? 'phase-exiting' : clipAnimationPhase === 'entering' ? 'phase-entering' : ''}`}>
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
            <div className={`track-lanes-scroll ${clipAnimationPhase === 'exiting' ? 'phase-exiting' : clipAnimationPhase === 'entering' ? 'phase-entering' : ''}`} style={{
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

          <TransitionOverlays
            activeJunction={activeJunction}
            clips={clips}
            tracks={tracks}
            timeToPixel={timeToPixel}
            isTrackExpanded={isTrackExpanded}
            getExpandedTrackHeight={getExpandedTrackHeight}
          />

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

          <TimelineOverlays
            timeToPixel={timeToPixel}
            formatTime={formatTime}
            inPoint={inPoint}
            outPoint={outPoint}
            duration={duration}
            markerDrag={markerDrag}
            onMarkerMouseDown={handleMarkerMouseDown}
            clipDrag={clipDrag}
            isRamPreviewing={isRamPreviewing}
            ramPreviewProgress={ramPreviewProgress}
            playheadPosition={playheadPosition}
            isExporting={isExporting}
            exportProgress={exportProgress}
            exportRange={exportRange}
            getCachedRanges={getCachedRanges}
            getProxyCachedRanges={getProxyCachedRanges}
          />

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

              <ParentChildLinksOverlay
                clips={clips}
                tracks={tracks}
                clipDrag={clipDrag}
                timelineRef={timelineRef}
                scrollX={scrollX}
                zoom={zoom}
                getExpandedTrackHeight={getExpandedTrackHeight}
              />


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
                '--marker-color': '#2997E5',
              } as React.CSSProperties}
            >
              <div className="timeline-marker-head">M</div>
              <div className="timeline-marker-line" />
            </div>
          )}
        </div>{/* timeline-body-content */}

        {/* Vertical Scrollbar — hide when slot grid is active */}
        {slotGridProgress < 1 && (
          <VerticalScrollbar
            scrollY={scrollY}
            contentHeight={contentHeight}
            viewportHeight={viewportHeight}
            onScrollChange={setScrollY}
          />
        )}
      </div>{/* timeline-body */}

      {/* Timeline Navigator - horizontal scrollbar with zoom handles — hide when slot grid is active */}
      {slotGridProgress < 1 && (
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
      )}

      <PickWhipCables pickWhipDrag={pickWhipDrag} trackPickWhipDrag={trackPickWhipDrag} />

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
