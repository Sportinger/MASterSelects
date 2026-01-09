// TimelineClip component - Clip rendering within tracks

import { memo } from 'react';
import type { TimelineClipProps } from './types';
import { THUMB_WIDTH } from './constants';

// Render waveform for audio clips
function Waveform({
  waveform,
  width,
  height,
}: {
  waveform: number[];
  width: number;
  height: number;
}) {
  if (!waveform || waveform.length === 0) return null;

  const barWidth = Math.max(1, width / waveform.length);
  const bars = waveform.map((value, i) => {
    const barHeight = Math.max(2, value * (height - 8));
    return (
      <div
        key={i}
        className="waveform-bar"
        style={{
          left: i * barWidth,
          height: barHeight,
          width: Math.max(1, barWidth - 1),
        }}
      />
    );
  });

  return (
    <div className="waveform-container" style={{ width, height }}>
      {bars}
    </div>
  );
}

function TimelineClipComponent({
  clip,
  trackId,
  track,
  clips,
  isSelected,
  isDragging,
  isTrimming,
  isLinkedToDragging,
  isLinkedToTrimming,
  clipDrag,
  clipTrim,
  scrollX,
  timelineRef,
  proxyEnabled,
  proxyStatus,
  proxyProgress,
  onMouseDown,
  onContextMenu,
  onTrimStart,
  hasKeyframes,
  timeToPixel,
  pixelToTime,
  formatTime,
}: TimelineClipProps) {
  const thumbnails = clip.thumbnails || [];

  const isGeneratingProxy = proxyStatus === 'generating';
  const hasProxy = proxyStatus === 'ready';

  // Check if this clip is linked to the dragging/trimming clip
  const draggedClip = clipDrag
    ? clips.find((c) => c.id === clipDrag.clipId)
    : null;
  const trimmedClip = clipTrim
    ? clips.find((c) => c.id === clipTrim.clipId)
    : null;

  // Calculate live trim values
  let displayStartTime = clip.startTime;
  let displayDuration = clip.duration;

  if (isTrimming && clipTrim) {
    const deltaX = clipTrim.currentX - clipTrim.startX;
    const deltaTime = pixelToTime(deltaX);
    const maxDuration = clip.source?.naturalDuration || clip.duration;

    if (clipTrim.edge === 'left') {
      const maxTrim = clipTrim.originalDuration - 0.1;
      const minTrim = -clipTrim.originalInPoint;
      const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));
      displayStartTime = clipTrim.originalStartTime + clampedDelta;
      displayDuration = clipTrim.originalDuration - clampedDelta;
    } else {
      const maxExtend = maxDuration - clipTrim.originalOutPoint;
      const minTrim = -(clipTrim.originalDuration - 0.1);
      const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
      displayDuration = clipTrim.originalDuration + clampedDelta;
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
    } else {
      const maxExtend = maxDuration - clip.outPoint;
      const minTrim = -(clip.duration - 0.1);
      const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
      displayDuration = clip.duration + clampedDelta;
    }
  }

  const width = timeToPixel(displayDuration);

  // Calculate position - if dragging, use snapped position if available
  let left = timeToPixel(displayStartTime);
  if (isDragging && clipDrag && timelineRef.current) {
    // Use snapped time if snapping, otherwise raw position
    if (clipDrag.isSnapping && clipDrag.snappedTime !== null) {
      left = timeToPixel(clipDrag.snappedTime);
    } else {
      const rect = timelineRef.current.getBoundingClientRect();
      const x = clipDrag.currentX - rect.left + scrollX - clipDrag.grabOffsetX;
      left = Math.max(0, x);
    }
  } else if (isLinkedToDragging && clipDrag && timelineRef.current && draggedClip) {
    // Move linked clip in sync - use snapped position if available
    let newDragTime: number;
    if (clipDrag.isSnapping && clipDrag.snappedTime !== null) {
      newDragTime = clipDrag.snappedTime;
    } else {
      const rect = timelineRef.current.getBoundingClientRect();
      const dragX =
        clipDrag.currentX - rect.left + scrollX - clipDrag.grabOffsetX;
      newDragTime = pixelToTime(Math.max(0, dragX));
    }
    const timeDelta = newDragTime - draggedClip.startTime;
    left = timeToPixel(Math.max(0, clip.startTime + timeDelta));
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

  const clipClass = [
    'timeline-clip',
    isSelected ? 'selected' : '',
    isDragging ? 'dragging' : '',
    isLinkedToDragging ? 'linked-dragging' : '',
    isTrimming ? 'trimming' : '',
    isLinkedToTrimming ? 'linked-trimming' : '',
    clip.source?.type || 'video',
    clip.isLoading ? 'loading' : '',
    hasProxy ? 'has-proxy' : '',
    isGeneratingProxy ? 'generating-proxy' : '',
    hasKeyframes(clip.id) ? 'has-keyframes' : '',
    clip.reversed ? 'reversed' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div
      className={clipClass}
      style={{ left, width }}
      onMouseDown={onMouseDown}
      onContextMenu={onContextMenu}
    >
      {/* Proxy progress bar */}
      {isGeneratingProxy && (
        <div className="clip-proxy-progress">
          <div
            className="clip-proxy-progress-bar"
            style={{ width: `${proxyProgress}%` }}
          />
        </div>
      )}
      {/* Proxy ready indicator */}
      {hasProxy && proxyEnabled && (
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
      {/* Audio waveform */}
      {clip.source?.type === 'audio' &&
        clip.waveform &&
        clip.waveform.length > 0 && (
          <div className="clip-waveform">
            <Waveform
              waveform={clip.waveform}
              width={width}
              height={Math.max(20, track.height - 12)}
            />
          </div>
        )}
      {/* Thumbnail filmstrip */}
      {thumbnails.length > 0 && clip.source?.type !== 'audio' && (
        <div className="clip-thumbnails">
          {Array.from({ length: visibleThumbs }).map((_, i) => {
            const thumbIndex = Math.floor((i / visibleThumbs) * thumbnails.length);
            const thumb = thumbnails[Math.min(thumbIndex, thumbnails.length - 1)];
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
      <div className="clip-content">
        {clip.isLoading && <div className="clip-loading-spinner" />}
        <span className="clip-name">{clip.name}</span>
        <span className="clip-duration">{formatTime(displayDuration)}</span>
      </div>
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
