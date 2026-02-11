// TimelineClip component - Clip rendering within tracks

import { memo, useRef } from 'react';
import type { TimelineClipProps } from './types';
import { THUMB_WIDTH } from './constants';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { getLabelHex } from '../panels/MediaPanel';
// PickWhip disabled
import { Logger } from '../../services/logger';
import { ClipWaveform } from './components/ClipWaveform';
import { ClipAnalysisOverlay } from './components/ClipAnalysisOverlay';
import { FadeCurve } from './components/FadeCurve';

const log = Logger.create('TimelineClip');

function TimelineClipComponent({
  clip,
  trackId,
  track,
  tracks,
  clips,
  isSelected,
  isInLinkedGroup,
  isDragging,
  isTrimming,
  isFading,
  isLinkedToDragging,
  isLinkedToTrimming,
  clipDrag,
  clipTrim,
  clipFade: _clipFade,
  zoom,
  scrollX,
  timelineRef,
  proxyEnabled,
  proxyStatus,
  proxyProgress,
  showTranscriptMarkers,
  toolMode,
  snappingEnabled,
  cutHoverInfo,
  onCutHover,
  onMouseDown,
  onDoubleClick,
  onContextMenu,
  onTrimStart,
  onFadeStart,
  onCutAtPosition,
  hasKeyframes,
  fadeInDuration,
  fadeOutDuration,
  opacityKeyframes,
  allKeyframeTimes,
  timeToPixel,
  pixelToTime,
  formatTime,
}: TimelineClipProps) {
  const thumbnails = clip.thumbnails || [];
  const thumbnailsEnabled = useTimelineStore(s => s.thumbnailsEnabled);
  const waveformsEnabled = useTimelineStore(s => s.waveformsEnabled);

  // Subscribe to playhead position only when cut tool is active (avoids re-renders during playback)
  const playheadPosition = useTimelineStore((state) =>
    toolMode === 'cut' ? state.playheadPosition : 0
  );

  // Look up media label color from mediaStore
  const mediaLabelHex = useMediaStore(s => {
    const mediaFileId = clip.mediaFileId || clip.source?.mediaFileId;
    if (clip.compositionId) {
      const comp = s.compositions.find(c => c.id === clip.compositionId);
      if (comp?.labelColor && comp.labelColor !== 'none') return getLabelHex(comp.labelColor);
    }
    if (mediaFileId) {
      const file = s.files.find(f => f.id === mediaFileId);
      if (file?.labelColor && file.labelColor !== 'none') return getLabelHex(file.labelColor);
    }
    // Check solid items and text items by matching clip name as fallback
    if (clip.source?.type === 'solid') {
      const solid = s.solidItems.find(si => si.id === mediaFileId);
      if (solid?.labelColor && solid.labelColor !== 'none') return getLabelHex(solid.labelColor);
    }
    if (clip.source?.type === 'text') {
      const text = s.textItems.find(ti => ti.id === mediaFileId);
      if (text?.labelColor && text.labelColor !== 'none') return getLabelHex(text.labelColor);
    }
    return null;
  });

  // Animation phase for enter/exit transitions
  const clipAnimationPhase = useTimelineStore(s => s.clipAnimationPhase);
  const clipEntranceKey = useTimelineStore(s => s.clipEntranceAnimationKey);
  const mountKeyRef = useRef(clipEntranceKey);

  // Calculate stagger delay: sort all clips by track order + startTime, then 20ms per clip
  const clipStaggerIndex = (() => {
    const sorted = [...clips].sort((a, b) => {
      const aTrack = tracks.findIndex(t => t.id === a.trackId);
      const bTrack = tracks.findIndex(t => t.id === b.trackId);
      if (aTrack !== bTrack) return aTrack - bTrack;
      return a.startTime - b.startTime;
    });
    return sorted.findIndex(c => c.id === clip.id);
  })();
  const animationDelay = Math.max(0, clipStaggerIndex) * 0.02;

  // Determine animation class:
  // - 'exiting': apply exit animation
  // - 'entering' + new clips: apply entrance animation (only during composition switch)
  // - Otherwise: no animation
  const isNewClip = mountKeyRef.current === clipEntranceKey && clipEntranceKey > 0;
  const animationClass = clipAnimationPhase === 'exiting'
    ? 'exit-animate'
    : (clipAnimationPhase === 'entering' && isNewClip)
      ? 'entrance-animate'
      : '';

  // Check if this clip should show cut indicator (either directly hovered or linked to hovered clip)
  const isDirectlyHovered = cutHoverInfo?.clipId === clip.id;
  const linkedClip = clip.linkedClipId ? clips.find(c => c.id === clip.linkedClipId) : null;
  const isLinkedToHovered = linkedClip && cutHoverInfo?.clipId === linkedClip.id;
  // Also check reverse link - if another clip links to this one
  const reverseLinkedClip = clips.find(c => c.linkedClipId === clip.id);
  const isReverseLinkedToHovered = reverseLinkedClip && cutHoverInfo?.clipId === reverseLinkedClip.id;
  const shouldShowCutIndicator = toolMode === 'cut' && cutHoverInfo && (isDirectlyHovered || isLinkedToHovered || isReverseLinkedToHovered);

  // Determine if this is an audio clip (check source type, MIME type, or extension as fallback)
  const audioExtensions = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'aiff', 'opus'];
  const fileExt = (clip.file?.name || clip.name || '').split('.').pop()?.toLowerCase() || '';
  const isAudioClip = clip.source?.type === 'audio' ||
    clip.file?.type?.startsWith('audio/') ||
    audioExtensions.includes(fileExt);

  // Determine if this is a text clip
  const isTextClip = clip.source?.type === 'text';

  // Determine if this is a solid clip
  const isSolidClip = clip.source?.type === 'solid';

  const isGeneratingProxy = proxyStatus === 'generating';
  const hasProxy = proxyStatus === 'ready';

  // Check if this clip is linked to the dragging/trimming clip
  const draggedClip = clipDrag
    ? clips.find((c) => c.id === clipDrag.clipId)
    : null;
  const trimmedClip = clipTrim
    ? clips.find((c) => c.id === clipTrim.clipId)
    : null;

  // Calculate live trim values (including inPoint/outPoint for waveform/thumbnail rendering)
  let displayStartTime = clip.startTime;
  let displayDuration = clip.duration;
  let displayInPoint = clip.inPoint;
  let displayOutPoint = clip.outPoint;

  if (isTrimming && clipTrim) {
    const deltaX = clipTrim.currentX - clipTrim.startX;
    const deltaTime = pixelToTime(deltaX);
    const sourceType = clip.source?.type;
    const isInfiniteClip = sourceType === 'text' || sourceType === 'image' || sourceType === 'solid';
    const maxDuration = isInfiniteClip
      ? Number.MAX_SAFE_INTEGER
      : (clip.source?.naturalDuration || clip.duration);

    if (clipTrim.edge === 'left') {
      const maxTrim = clipTrim.originalDuration - 0.1;
      const minTrim = isInfiniteClip
        ? -clipTrim.originalStartTime
        : -clipTrim.originalInPoint;
      const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));
      displayStartTime = clipTrim.originalStartTime + clampedDelta;
      displayDuration = clipTrim.originalDuration - clampedDelta;
      // Update inPoint when trimming left edge
      displayInPoint = clipTrim.originalInPoint + clampedDelta;
    } else {
      const maxExtend = maxDuration - clipTrim.originalOutPoint;
      const minTrim = -(clipTrim.originalDuration - 0.1);
      const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
      displayDuration = clipTrim.originalDuration + clampedDelta;
      // Update outPoint when trimming right edge
      displayOutPoint = clipTrim.originalOutPoint + clampedDelta;
    }
  } else if (isLinkedToTrimming && clipTrim && trimmedClip) {
    // Apply same trim to linked clip visually
    const deltaX = clipTrim.currentX - clipTrim.startX;
    const deltaTime = pixelToTime(deltaX);
    const maxDuration = clip.source?.naturalDuration || clip.duration;

    if (clipTrim.edge === 'left') {
      const maxTrim = clip.duration - 0.1;
      const minTrim = -clip.inPoint;
      const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));
      displayStartTime = clip.startTime + clampedDelta;
      displayDuration = clip.duration - clampedDelta;
      displayInPoint = clip.inPoint + clampedDelta;
    } else {
      const maxExtend = maxDuration - clip.outPoint;
      const minTrim = -(clip.duration - 0.1);
      const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
      displayDuration = clip.duration + clampedDelta;
      displayOutPoint = clip.outPoint + clampedDelta;
    }
  }

  const width = timeToPixel(displayDuration);

  // Calculate position - if dragging, use the computed position (with snapping/resistance)
  let left = timeToPixel(displayStartTime);
  if (isDragging && clipDrag && timelineRef.current) {
    // Always use snappedTime when available - it contains the position with snapping and resistance applied
    if (clipDrag.snappedTime !== null) {
      left = timeToPixel(clipDrag.snappedTime);
    } else {
      const rect = timelineRef.current.getBoundingClientRect();
      const x = clipDrag.currentX - rect.left + scrollX - clipDrag.grabOffsetX;
      left = Math.max(0, x);
    }
  } else if (isLinkedToDragging && clipDrag && timelineRef.current && draggedClip) {
    // Move linked clip in sync - use computed position (snapped + resistance) if available
    let newDragTime: number;
    if (clipDrag.snappedTime !== null) {
      newDragTime = clipDrag.snappedTime;
    } else {
      const rect = timelineRef.current.getBoundingClientRect();
      const dragX =
        clipDrag.currentX - rect.left + scrollX - clipDrag.grabOffsetX;
      newDragTime = pixelToTime(Math.max(0, dragX));
    }
    const timeDelta = newDragTime - draggedClip.startTime;
    left = timeToPixel(Math.max(0, clip.startTime + timeDelta));
  } else if (clipDrag?.multiSelectClipIds?.includes(clip.id) && clipDrag.multiSelectTimeDelta !== undefined) {
    // This clip is part of multi-select drag (but not the primary dragged clip)
    left = timeToPixel(Math.max(0, clip.startTime + clipDrag.multiSelectTimeDelta));
  }

  // Calculate how many thumbnails to show based on clip width
  const visibleThumbs = Math.max(1, Math.ceil(width / THUMB_WIDTH));

  // Track filtering
  if (isDragging && clipDrag && clipDrag.currentTrackId !== trackId) {
    return null;
  }
  if (!isDragging && !isLinkedToDragging && clip.trackId !== trackId) {
    return null;
  }
  if (clip.trackId !== trackId && !isDragging) {
    return null;
  }

  // Determine clip type class (audio, video, text, or image)
  const clipTypeClass = isSolidClip ? 'solid' : isTextClip ? 'text' : isAudioClip ? 'audio' : (clip.source?.type || 'video');

  // Check if this clip is part of a multi-select drag
  const isInMultiSelectDrag = clipDrag?.multiSelectClipIds?.includes(clip.id) && clipDrag.multiSelectTimeDelta !== undefined;

  const clipClass = [
    'timeline-clip',
    isSelected ? 'selected' : '',
    isInLinkedGroup ? 'linked-group' : '',
    isDragging ? 'dragging' : '',
    isInMultiSelectDrag ? 'dragging multiselect-dragging' : '',
    isLinkedToDragging ? 'linked-dragging' : '',
    isTrimming ? 'trimming' : '',
    isLinkedToTrimming ? 'linked-trimming' : '',
    isFading ? 'fading' : '',
    isDragging && clipDrag?.forcingOverlap ? 'forcing-overlap' : '',
    clipTypeClass,
    clip.isLoading ? 'loading' : '',
    clip.needsReload ? 'needs-reload' : '',
    hasProxy ? 'has-proxy' : '',
    isGeneratingProxy ? 'generating-proxy' : '',
    hasKeyframes(clip.id) ? 'has-keyframes' : '',
    clip.reversed ? 'reversed' : '',
    clip.transcriptStatus === 'ready' ? 'has-transcript' : '',
    clip.waveformGenerating ? 'generating-waveform' : '',
    clip.parentClipId ? 'has-parent' : '',
    clip.isPendingDownload ? 'pending-download' : '',
    clip.downloadError ? 'download-error' : '',
    clip.isComposition ? 'composition' : '',
  ]
    .filter(Boolean)
    .join(' ');

  // Cut tool snapping helper
  const snapCutTime = (rawTime: number, shouldSnap: boolean): number => {
    log.debug('CUT SNAP', { shouldSnap, snappingEnabled, rawTime, zoom, playheadPosition });
    if (!shouldSnap) return rawTime;

    const snapThresholdPixels = 10;
    const snapThresholdTime = snapThresholdPixels / zoom;

    // Collect snap targets: playhead and all clip edges
    const snapTargets: number[] = [playheadPosition];
    clips.forEach(c => {
      snapTargets.push(c.startTime);
      snapTargets.push(c.startTime + c.duration);
    });

    log.debug('CUT SNAP targets:', { snapTargets, threshold: snapThresholdTime });

    // Find nearest snap target
    let nearestTarget = rawTime;
    let nearestDistance = Infinity;
    for (const target of snapTargets) {
      const distance = Math.abs(target - rawTime);
      if (distance < nearestDistance && distance <= snapThresholdTime) {
        nearestDistance = distance;
        nearestTarget = target;
      }
    }

    log.debug('CUT SNAP result:', { nearestTarget, nearestDistance, snapped: nearestTarget !== rawTime });
    return nearestTarget;
  };

  // Cut tool handlers
  const handleMouseMove = (e: React.MouseEvent) => {
    if (toolMode !== 'cut') {
      if (cutHoverInfo?.clipId === clip.id) onCutHover(null, null);
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    // Convert pixel position to time
    const rawCutTime = displayStartTime + (x / width) * displayDuration;
    // When snapping enabled: snap by default, Alt temporarily disables
    // When snapping disabled: don't snap, Alt temporarily enables
    const shouldSnap = snappingEnabled !== e.altKey;
    const cutTime = snapCutTime(rawCutTime, shouldSnap);
    onCutHover(clip.id, cutTime);
  };

  const handleMouseLeave = () => {
    if (cutHoverInfo?.clipId === clip.id) onCutHover(null, null);
  };

  const handleClick = (e: React.MouseEvent) => {
    if (toolMode !== 'cut') return;
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    // Convert pixel position to time within clip
    const rawCutTime = displayStartTime + (x / width) * displayDuration;
    // When snapping enabled: snap by default, Alt temporarily disables
    // When snapping disabled: don't snap, Alt temporarily enables
    const shouldSnap = snappingEnabled !== e.altKey;
    const cutTime = snapCutTime(rawCutTime, shouldSnap);
    onCutAtPosition(clip.id, cutTime);
    onCutHover(null, null);
  };

  // Calculate cut indicator position for this clip
  const cutIndicatorX = shouldShowCutIndicator && cutHoverInfo
    ? ((cutHoverInfo.time - displayStartTime) / displayDuration) * width
    : null;

  return (
    <div
      className={`${clipClass}${toolMode === 'cut' ? ' cut-mode' : ''} ${animationClass}`}
      style={{
        left,
        width,
        cursor: toolMode === 'cut' ? 'crosshair' : undefined,
        animationDelay: `${animationDelay}s`,
        ...(isSolidClip && clip.solidColor ? {
          background: clip.solidColor,
          borderColor: clip.solidColor,
        } : mediaLabelHex ? {
          background: mediaLabelHex,
          borderColor: mediaLabelHex,
        } : {}),
      }}
      data-clip-id={clip.id}
      onMouseDown={toolMode === 'cut' ? undefined : onMouseDown}
      onDoubleClick={toolMode === 'cut' ? undefined : onDoubleClick}
      onContextMenu={onContextMenu}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {/* Cut indicator line */}
      {shouldShowCutIndicator && cutIndicatorX !== null && cutIndicatorX >= 0 && cutIndicatorX <= width && (
        <div
          className="cut-indicator"
          style={{ left: cutIndicatorX }}
        />
      )}
      {/* YouTube pending download preview */}
      {clip.isPendingDownload && clip.youtubeThumbnail && (
        <div
          className="clip-youtube-preview"
          style={{ backgroundImage: `url(${clip.youtubeThumbnail})` }}
        />
      )}
      {/* Download progress bar */}
      {clip.isPendingDownload && !clip.downloadError && (
        <>
          <div className="clip-download-progress">
            <div
              className="clip-download-progress-bar"
              style={{ width: `${clip.downloadProgress || 0}%` }}
            />
          </div>
          <div className="clip-download-status">
            <div className="download-spinner" />
            <span>Downloading {clip.downloadProgress || 0}%</span>
          </div>
        </>
      )}
      {/* Download error badge */}
      {clip.downloadError && (
        <div className="clip-download-error-badge" title={clip.downloadError}>
          Error
        </div>
      )}
      {/* Proxy generating indicator - fill badge */}
      {isGeneratingProxy && (
        <div className="clip-proxy-generating" title={`Generating proxy: ${proxyProgress}%`}>
          <span className="proxy-fill-badge">
            <span className="proxy-fill-bg">P</span>
            <span
              className="proxy-fill-progress"
              style={{ height: `${proxyProgress}%` }}
            >P</span>
          </span>
          <span className="proxy-percent">{proxyProgress}%</span>
        </div>
      )}
      {/* Proxy ready indicator */}
      {hasProxy && proxyEnabled && !isGeneratingProxy && (
        <div className="clip-proxy-badge" title="Proxy ready">
          P
        </div>
      )}
      {/* Reversed indicator */}
      {clip.reversed && (
        <div className="clip-reversed-badge" title="Reversed playback">
          {'\u27F2'}
        </div>
      )}
      {/* Linked group indicator */}
      {isInLinkedGroup && (
        <div className="clip-linked-group-badge" title="Multicam linked group">
          {'\u26D3'}
        </div>
      )}
      {/* Waveform generation progress indicator */}
      {clip.waveformGenerating && (
        <div className="clip-waveform-indicator">
          <div className="waveform-progress" style={{ width: `${clip.waveformProgress || 50}%` }} />
        </div>
      )}
      {/* Audio waveform */}
      {waveformsEnabled && isAudioClip && clip.waveform && clip.waveform.length > 0 && (
        <div className="clip-waveform">
          <ClipWaveform
            waveform={clip.waveform}
            width={width}
            height={Math.max(20, track.height - 12)}
            inPoint={displayInPoint}
            outPoint={displayOutPoint}
            naturalDuration={clip.source?.naturalDuration || clip.duration}
          />
        </div>
      )}
      {/* Nested composition mixdown waveform - shown overlaid on thumbnails */}
      {waveformsEnabled && clip.isComposition && clip.mixdownWaveform && clip.mixdownWaveform.length > 0 && (
        <div className="clip-mixdown-waveform">
          <ClipWaveform
            waveform={clip.mixdownWaveform}
            width={width}
            height={Math.min(30, Math.max(16, track.height / 3))}
            inPoint={displayInPoint}
            outPoint={displayOutPoint}
            naturalDuration={clip.duration}
          />
        </div>
      )}
      {/* Nested composition mixdown generating indicator */}
      {clip.isComposition && clip.mixdownGenerating && (
        <div className="clip-mixdown-indicator">
          <span>Generating audio...</span>
        </div>
      )}
      {/* Segment-based thumbnails for nested compositions */}
      {thumbnailsEnabled && clip.isComposition && clip.clipSegments && clip.clipSegments.length > 0 && !isAudioClip && (
        <div className="clip-thumbnails clip-thumbnails-segments">
          {clip.clipSegments.map((segment, segIdx) => {
            const segmentWidth = (segment.endNorm - segment.startNorm) * 100;
            const segmentLeft = segment.startNorm * 100;
            // Calculate how many thumbnails fit in this segment
            const segmentThumbCount = Math.max(1, Math.ceil((segmentWidth / 100) * visibleThumbs));

            return (
              <div
                key={segIdx}
                className="clip-segment"
                style={{
                  position: 'absolute',
                  left: `${segmentLeft}%`,
                  width: `${segmentWidth}%`,
                  height: '100%',
                  display: 'flex',
                  overflow: 'hidden',
                }}
              >
                {segment.thumbnails.length > 0 ? (
                  Array.from({ length: segmentThumbCount }).map((_, i) => {
                    const thumbIndex = Math.floor((i / segmentThumbCount) * segment.thumbnails.length);
                    const thumb = segment.thumbnails[Math.min(thumbIndex, segment.thumbnails.length - 1)];
                    return (
                      <img
                        key={i}
                        src={thumb}
                        alt=""
                        className="clip-thumb"
                        draggable={false}
                        style={{ flex: '1 0 auto', minWidth: 0, objectFit: 'cover' }}
                      />
                    );
                  })
                ) : (
                  <div className="clip-segment-empty" style={{ width: '100%', height: '100%', background: '#1a1a1a' }} />
                )}
              </div>
            );
          })}
        </div>
      )}
      {/* Regular thumbnail filmstrip - for non-composition clips */}
      {thumbnailsEnabled && thumbnails.length > 0 && !isAudioClip && !(clip.isComposition && clip.clipSegments && clip.clipSegments.length > 0) && (
        <div className="clip-thumbnails">
          {Array.from({ length: visibleThumbs }).map((_, i) => {
            // Calculate thumbnail index based on displayInPoint/displayOutPoint (trim-aware, live during trim)
            const naturalDuration = clip.source?.naturalDuration || clip.duration;
            const startRatio = displayInPoint / naturalDuration;
            const endRatio = displayOutPoint / naturalDuration;
            // Map visible position to the trimmed range in source media
            const positionInTrimmed = i / visibleThumbs;
            const sourceRatio = startRatio + positionInTrimmed * (endRatio - startRatio);
            const thumbIndex = Math.floor(sourceRatio * thumbnails.length);
            const thumb = thumbnails[Math.min(Math.max(0, thumbIndex), thumbnails.length - 1)];
            return (
              <img
                key={i}
                src={thumb}
                alt=""
                className="clip-thumb"
                draggable={false}
              />
            );
          })}
        </div>
      )}
      {/* Nested composition clip boundary markers */}
      {clip.isComposition && clip.nestedClipBoundaries && clip.nestedClipBoundaries.length > 0 && (
        <div className="nested-clip-boundaries">
          {clip.nestedClipBoundaries.map((boundary, i) => (
            <div
              key={i}
              className="nested-boundary-line"
              style={{ left: `${boundary * 100}%` }}
            />
          ))}
        </div>
      )}
      {/* Needs reload indicator */}
      {clip.needsReload && (
        <div className="clip-reload-badge" title="Click media file to reload">
          !
        </div>
      )}
      <div className="clip-content">
        {clip.isLoading && <div className="clip-loading-spinner" />}
        <div className="clip-name-row">
          {isSolidClip && (
            <span className="clip-solid-swatch" title="Solid Clip" style={{ background: clip.solidColor || '#fff' }} />
          )}
          {isTextClip && (
            <span className="clip-text-icon" title="Text Clip">T</span>
          )}
          <span className="clip-name">{isTextClip && clip.textProperties ? clip.textProperties.text.slice(0, 30) || 'Text' : clip.name}</span>
          {/* PickWhip disabled */}
        </div>
        <span className="clip-duration">{formatTime(displayDuration)}</span>
      </div>
      {/* Transcript word markers */}
      {showTranscriptMarkers && clip.transcript && clip.transcript.length > 0 && (
        <div className="clip-transcript-markers">
          {clip.transcript.map((word) => {
            // Word times are relative to clip's inPoint
            const wordStartInClip = word.start - clip.inPoint;
            const wordEndInClip = word.end - clip.inPoint;

            // Only show markers that are visible within the clip's current trim
            if (wordEndInClip < 0 || wordStartInClip > displayDuration) {
              return null;
            }

            // Calculate marker position and width
            const markerStart = Math.max(0, wordStartInClip);
            const markerEnd = Math.min(displayDuration, wordEndInClip);
            const markerLeft = (markerStart / displayDuration) * 100;
            const markerWidth = ((markerEnd - markerStart) / displayDuration) * 100;

            return (
              <div
                key={word.id}
                className="transcript-marker"
                style={{
                  left: `${markerLeft}%`,
                  width: `${Math.max(0.5, markerWidth)}%`,
                }}
                title={word.text}
              />
            );
          })}
        </div>
      )}
      {/* Transcribing indicator */}
      {clip.transcriptStatus === 'transcribing' && (
        <div className="clip-transcribing-indicator">
          <div className="transcribing-progress" style={{ width: `${clip.transcriptProgress || 0}%` }} />
        </div>
      )}
      {/* Analysis overlay - graph showing focus/motion (renders during analysis and when ready) */}
      {/* Only show analysis overlay for video clips, not audio */}
      {!isAudioClip && clip.analysis && (clip.analysisStatus === 'ready' || clip.analysisStatus === 'analyzing') && (
        <>
          <div className="analysis-legend-labels">
            <span className="legend-focus">Focus</span>
            <span className="legend-motion">Motion</span>
            {clip.analysisStatus === 'analyzing' && (
              <span className="legend-progress">{clip.analysisProgress || 0}%</span>
            )}
          </div>
          <div className="clip-analysis-overlay">
            <ClipAnalysisOverlay
              analysis={clip.analysis}
              clipDuration={displayDuration}
              clipInPoint={clip.inPoint}
              clipStartTime={displayStartTime}
              width={width}
              height={track.height}
            />
          </div>
        </>
      )}
      {/* Analyzing indicator (thin progress bar at bottom) */}
      {clip.analysisStatus === 'analyzing' && (
        <div className="clip-analyzing-indicator">
          <div className="analyzing-progress" style={{ width: `${clip.analysisProgress || 0}%` }} />
        </div>
      )}
      {/* Keyframe tick marks on clip bar */}
      {allKeyframeTimes.length > 0 && (
        <div className="clip-keyframe-ticks">
          {allKeyframeTimes.map((time, i) => {
            const xPercent = (time / displayDuration) * 100;
            if (xPercent < 0 || xPercent > 100) return null;
            return (
              <div
                key={i}
                className="keyframe-tick"
                style={{ left: `${xPercent}%` }}
              />
            );
          })}
        </div>
      )}
      {/* Fade curve - SVG bezier curve showing opacity animation */}
      {opacityKeyframes.length >= 2 && (
        <div className="fade-curve-container">
          <FadeCurve
            key={opacityKeyframes.map(k => `${k.id}:${k.time.toFixed(3)}:${k.value}:${k.handleIn?.x ?? ''}:${k.handleIn?.y ?? ''}:${k.handleOut?.x ?? ''}:${k.handleOut?.y ?? ''}`).join('|')}
            keyframes={opacityKeyframes}
            clipDuration={displayDuration}
            width={width}
            height={track.height}
          />
        </div>
      )}
      {/* Fade handles - corner handles for adjusting fade-in/out */}
      <div
        className={`fade-handle left${fadeInDuration > 0 ? ' active' : ''}`}
        style={fadeInDuration > 0 ? { left: timeToPixel(fadeInDuration) - 6 } : undefined}
        onMouseDown={(e) => {
          e.stopPropagation();
          onFadeStart(e, 'left');
        }}
        title={fadeInDuration > 0 ? `Fade In: ${fadeInDuration.toFixed(2)}s` : 'Drag to add fade in'}
      />
      <div
        className={`fade-handle right${fadeOutDuration > 0 ? ' active' : ''}`}
        style={fadeOutDuration > 0 ? { right: timeToPixel(fadeOutDuration) - 6 } : undefined}
        onMouseDown={(e) => {
          e.stopPropagation();
          onFadeStart(e, 'right');
        }}
        title={fadeOutDuration > 0 ? `Fade Out: ${fadeOutDuration.toFixed(2)}s` : 'Drag to add fade out'}
      />
      {/* Trim handles */}
      <div
        className="trim-handle left"
        onMouseDown={(e) => {
          e.stopPropagation();
          onTrimStart(e, 'left');
        }}
      />
      <div
        className="trim-handle right"
        onMouseDown={(e) => {
          e.stopPropagation();
          onTrimStart(e, 'right');
        }}
      />
    </div>
  );
}

export const TimelineClip = memo(TimelineClipComponent);
