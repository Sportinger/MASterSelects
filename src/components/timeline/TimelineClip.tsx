// TimelineClip component - Clip rendering within tracks

import { memo, useRef, useEffect } from 'react';
import type { TimelineClipProps } from './types';
import { THUMB_WIDTH } from './constants';
import type { ClipAnalysis } from '../../types';
import { useTimelineStore } from '../../stores/timeline';
import { PickWhip } from './PickWhip';

// Render waveform for audio clips using canvas for better performance
const Waveform = memo(function Waveform({
  waveform,
  width,
  height,
}: {
  waveform: number[];
  width: number;
  height: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveform || waveform.length === 0 || width <= 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Limit canvas size to browser maximum (16384 is safe for most browsers)
    const MAX_CANVAS_WIDTH = 16384;
    const canvasWidth = Math.min(width, MAX_CANVAS_WIDTH);

    // Set canvas size (account for device pixel ratio for sharpness)
    const dpr = window.devicePixelRatio || 1;
    // Also limit by dpr to avoid exceeding canvas limits
    const effectiveDpr = Math.min(dpr, MAX_CANVAS_WIDTH / canvasWidth);

    canvas.width = canvasWidth * effectiveDpr;
    canvas.height = height * effectiveDpr;
    ctx.scale(effectiveDpr, effectiveDpr);

    // Clear
    ctx.clearRect(0, 0, canvasWidth, height);

    // Determine number of bars to draw (max 2 per pixel for detail)
    const maxBars = Math.floor(canvasWidth * 2);
    const samplesPerBar = Math.max(1, Math.floor(waveform.length / maxBars));
    const numBars = Math.ceil(waveform.length / samplesPerBar);
    const barWidth = canvasWidth / numBars;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';

    // Draw bars, using peak value for each segment
    for (let i = 0; i < numBars; i++) {
      const startIdx = i * samplesPerBar;
      const endIdx = Math.min(startIdx + samplesPerBar, waveform.length);

      // Get peak value for this segment
      let peak = 0;
      for (let j = startIdx; j < endIdx; j++) {
        if (waveform[j] > peak) peak = waveform[j];
      }

      const barHeight = Math.max(2, peak * (height - 4));
      const x = i * barWidth;
      const y = (height - barHeight) / 2;
      ctx.fillRect(x, y, Math.max(1, barWidth - 0.5), barHeight);
    }
  }, [waveform, width, height]);

  if (!waveform || waveform.length === 0) return null;

  return (
    <canvas
      ref={canvasRef}
      className="waveform-canvas"
      style={{ width, height }}
    />
  );
});

// Render analysis overlay as line graphs (focus, motion) with real-time position indicator
const AnalysisOverlay = memo(function AnalysisOverlay({
  analysis,
  clipDuration,
  clipInPoint,
  clipStartTime,
  width,
  height: containerHeight,
}: {
  analysis: ClipAnalysis;
  clipDuration: number;
  clipInPoint: number;
  clipStartTime: number;
  width: number;
  height: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const indicatorRef = useRef<HTMLDivElement>(null);

  // Subscribe to playhead position for real-time indicator
  const playheadPosition = useTimelineStore((state) => state.playheadPosition);

  // Calculate if playhead is within this clip
  const playheadInClip = playheadPosition >= clipStartTime && playheadPosition < clipStartTime + clipDuration;
  const relativePlayhead = playheadPosition - clipStartTime; // Time within clip
  const playheadX = playheadInClip ? (relativePlayhead / clipDuration) * width : -1;

  // Find current analysis values at playhead
  const currentValues = playheadInClip ? (() => {
    const sourceTime = clipInPoint + relativePlayhead;
    // Find nearest frame
    let nearest = analysis.frames[0];
    let minDiff = Math.abs(nearest?.timestamp - sourceTime);
    for (const frame of analysis.frames) {
      const diff = Math.abs(frame.timestamp - sourceTime);
      if (diff < minDiff) {
        minDiff = diff;
        nearest = frame;
      }
    }
    return nearest;
  })() : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analysis?.frames.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const height = containerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Filter frames within clip range and sort by timestamp
    const visibleFrames = analysis.frames
      .filter(frame => {
        const frameInClip = frame.timestamp - clipInPoint;
        return frameInClip >= 0 && frameInClip <= clipDuration;
      })
      .sort((a, b) => a.timestamp - b.timestamp);

    if (visibleFrames.length < 2) return;

    // Draw filled area + line for Focus (green) - from bottom
    // Lower multiplier so focus line sits lower on the graph
    const focusMultiplier = 1.0;
    const heightScale = 0.6; // Use 60% of height
    ctx.beginPath();
    ctx.moveTo(0, height); // Start at bottom-left

    for (let i = 0; i < visibleFrames.length; i++) {
      const frame = visibleFrames[i];
      const frameInClip = frame.timestamp - clipInPoint;
      const x = (frameInClip / clipDuration) * width;
      // Focus: 0 = bottom, 1 = top (inverted Y)
      const amplifiedFocus = Math.min(1, frame.focus * focusMultiplier);
      const y = height - (amplifiedFocus * height * heightScale);

      if (i === 0) {
        ctx.lineTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    // Close the path to bottom-right
    const lastFrame = visibleFrames[visibleFrames.length - 1];
    const lastX = ((lastFrame.timestamp - clipInPoint) / clipDuration) * width;
    ctx.lineTo(lastX, height);
    ctx.closePath();

    // Fill with semi-transparent green
    ctx.fillStyle = 'rgba(34, 197, 94, 0.3)';
    ctx.fill();

    // Draw the focus line on top with gradient coloring based on threshold
    // Green when good (>70% focus), red when bad (<40% focus)
    const focusRedThreshold = 0.4;   // Below this = full red
    const focusGreenThreshold = 0.7; // Above this = full green
    ctx.lineWidth = 1.5;

    for (let i = 0; i < visibleFrames.length - 1; i++) {
      const frame = visibleFrames[i];
      const nextFrame = visibleFrames[i + 1];

      const frameInClip = frame.timestamp - clipInPoint;
      const nextFrameInClip = nextFrame.timestamp - clipInPoint;

      const x1 = (frameInClip / clipDuration) * width;
      const x2 = (nextFrameInClip / clipDuration) * width;

      const amplifiedFocus1 = Math.min(1, frame.focus * focusMultiplier);
      const amplifiedFocus2 = Math.min(1, nextFrame.focus * focusMultiplier);

      const y1 = height - (amplifiedFocus1 * height * heightScale);
      const y2 = height - (amplifiedFocus2 * height * heightScale);

      // Calculate color based on average value of segment
      // t=0 at 40% or below (red), t=1 at 70% or above (green)
      const avgFocus = (frame.focus + nextFrame.focus) / 2;
      const t = Math.min(1, Math.max(0, (avgFocus - focusRedThreshold) / (focusGreenThreshold - focusRedThreshold)));

      // Interpolate from red (low) to green (high)
      const r = Math.round(239 - t * (239 - 34));   // 239 -> 34
      const g = Math.round(68 + t * (197 - 68));    // 68 -> 197
      const b = Math.round(68 + t * (94 - 68));     // 68 -> 94

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
      ctx.stroke();
    }

    // Draw filled area + line for Motion (blue) - from bottom
    // Use globalMotion (camera/scene motion) for display
    const motionMultiplier = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, height);

    for (let i = 0; i < visibleFrames.length; i++) {
      const frame = visibleFrames[i];
      const frameInClip = frame.timestamp - clipInPoint;
      const x = (frameInClip / clipDuration) * width;
      // Use globalMotion if available, fallback to total motion
      const motionValue = frame.globalMotion ?? frame.motion;
      const amplifiedMotion = Math.min(1, motionValue * motionMultiplier);
      const y = height - (amplifiedMotion * height * heightScale);

      if (i === 0) {
        ctx.lineTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.lineTo(lastX, height);
    ctx.closePath();

    // Fill with semi-transparent blue
    ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
    ctx.fill();

    // Draw the motion line on top with gradient coloring based on threshold
    // Blue when stable (low motion), red when shaky (high motion)
    const motionThreshold = 0.4; // Above this = red
    ctx.lineWidth = 1.5;

    for (let i = 0; i < visibleFrames.length - 1; i++) {
      const frame = visibleFrames[i];
      const nextFrame = visibleFrames[i + 1];

      const frameInClip = frame.timestamp - clipInPoint;
      const nextFrameInClip = nextFrame.timestamp - clipInPoint;

      const x1 = (frameInClip / clipDuration) * width;
      const x2 = (nextFrameInClip / clipDuration) * width;

      const motionValue1 = frame.globalMotion ?? frame.motion;
      const motionValue2 = nextFrame.globalMotion ?? nextFrame.motion;

      const amplifiedMotion1 = Math.min(1, motionValue1 * motionMultiplier);
      const amplifiedMotion2 = Math.min(1, motionValue2 * motionMultiplier);

      const y1 = height - (amplifiedMotion1 * height * heightScale);
      const y2 = height - (amplifiedMotion2 * height * heightScale);

      // Calculate color based on average value of segment
      // Higher motion = more red (inverted from focus)
      const avgMotion = (motionValue1 + motionValue2) / 2;
      const t = Math.min(1, Math.max(0, avgMotion / motionThreshold));

      // Interpolate from blue (low/stable) to red (high/shaky)
      const r = Math.round(59 + t * (239 - 59));    // 59 -> 239
      const g = Math.round(130 - t * (130 - 68));   // 130 -> 68
      const b = Math.round(246 - t * (246 - 68));   // 246 -> 68

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
      ctx.stroke();
    }

    // Draw face indicators as small yellow dots at the top
    for (const frame of visibleFrames) {
      if (frame.faceCount > 0) {
        const frameInClip = frame.timestamp - clipInPoint;
        const x = (frameInClip / clipDuration) * width;
        ctx.beginPath();
        ctx.arc(x, 4, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(234, 179, 8, 0.9)';
        ctx.fill();
      }
    }
  }, [analysis, clipDuration, clipInPoint, width, containerHeight]);

  if (!analysis?.frames.length) return null;

  return (
    <>
      <canvas
        ref={canvasRef}
        className="analysis-overlay-canvas"
        style={{ width, height: containerHeight }}
      />
      {/* Real-time position indicator */}
      {playheadInClip && currentValues && (
        <div
          ref={indicatorRef}
          className="analysis-position-indicator"
          style={{ left: playheadX }}
        >
          <div className="analysis-indicator-line" />
          <div className="analysis-indicator-values">
            <span className="focus-value" title="Focus/Sharpness">
              {Math.round(currentValues.focus * 100)}%
            </span>
            <span className="motion-value" title="Motion">
              {Math.round(currentValues.motion * 100)}%
            </span>
          </div>
        </div>
      )}
    </>
  );
});

function TimelineClipComponent({
  clip,
  trackId,
  track,
  clips,
  isSelected,
  isInLinkedGroup,
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
  showTranscriptMarkers,
  onMouseDown,
  onContextMenu,
  onTrimStart,
  hasKeyframes,
  timeToPixel,
  pixelToTime,
  formatTime,
  onPickWhipDragStart,
  onPickWhipDragEnd,
  onSetClipParent,
}: TimelineClipProps) {
  const thumbnails = clip.thumbnails || [];

  // Determine if this is an audio clip (check source type, MIME type, or extension as fallback)
  const audioExtensions = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'aiff', 'opus'];
  const fileExt = (clip.file?.name || clip.name || '').split('.').pop()?.toLowerCase() || '';
  const isAudioClip = clip.source?.type === 'audio' ||
    clip.file?.type?.startsWith('audio/') ||
    audioExtensions.includes(fileExt);

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

  // Determine clip type class (audio, video, or image)
  const clipTypeClass = isAudioClip ? 'audio' : (clip.source?.type || 'video');

  const clipClass = [
    'timeline-clip',
    isSelected ? 'selected' : '',
    isInLinkedGroup ? 'linked-group' : '',
    isDragging ? 'dragging' : '',
    isLinkedToDragging ? 'linked-dragging' : '',
    isTrimming ? 'trimming' : '',
    isLinkedToTrimming ? 'linked-trimming' : '',
    isDragging && clipDrag?.forcingOverlap ? 'forcing-overlap' : '',
    clipTypeClass,
    clip.isLoading ? 'loading' : '',
    hasProxy ? 'has-proxy' : '',
    isGeneratingProxy ? 'generating-proxy' : '',
    hasKeyframes(clip.id) ? 'has-keyframes' : '',
    clip.reversed ? 'reversed' : '',
    clip.transcriptStatus === 'ready' ? 'has-transcript' : '',
    clip.waveformGenerating ? 'generating-waveform' : '',
    clip.parentClipId ? 'has-parent' : '',
  ]
    .filter(Boolean)
    .join(' ');

  // Get parent clip name for tooltip
  const parentClip = clip.parentClipId ? clips.find(c => c.id === clip.parentClipId) : null;

  return (
    <div
      className={clipClass}
      style={{ left, width }}
      data-clip-id={clip.id}
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
      {isAudioClip && clip.waveform && clip.waveform.length > 0 && (
        <div className="clip-waveform">
          <Waveform
            waveform={clip.waveform}
            width={width}
            height={Math.max(20, track.height - 12)}
          />
        </div>
      )}
      {/* Thumbnail filmstrip - only for non-audio clips */}
      {thumbnails.length > 0 && !isAudioClip && (
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
        <PickWhip
          clipId={clip.id}
          clipName={clip.name}
          parentClipId={clip.parentClipId}
          parentClipName={parentClip?.name}
          onSetParent={onSetClipParent}
          onDragStart={onPickWhipDragStart}
          onDragEnd={onPickWhipDragEnd}
        />
        <span className="clip-name">{clip.name}</span>
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
            <AnalysisOverlay
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
