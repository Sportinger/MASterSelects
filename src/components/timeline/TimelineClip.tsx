// TimelineClip component - Clip rendering within tracks

import { memo, useRef, useEffect } from 'react';
import type { TimelineClipProps } from './types';
import { THUMB_WIDTH } from './constants';
import type { ClipAnalysis } from '../../types';
import { useTimelineStore } from '../../stores/timeline';
import { PickWhip } from './PickWhip';
import { Logger } from '../../services/logger';

const log = Logger.create('TimelineClip');

// Render waveform for audio clips using canvas for better performance
// Supports trimming: only displays the portion of waveform between inPoint and outPoint
const Waveform = memo(function Waveform({
  waveform,
  width,
  height,
  inPoint,
  outPoint,
  naturalDuration,
}: {
  waveform: number[];
  width: number;
  height: number;
  inPoint: number;
  outPoint: number;
  naturalDuration: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveform || waveform.length === 0 || width <= 0 || naturalDuration <= 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate which portion of the waveform to display based on trim points
    const startRatio = inPoint / naturalDuration;
    const endRatio = outPoint / naturalDuration;
    const startSample = Math.floor(startRatio * waveform.length);
    const endSample = Math.ceil(endRatio * waveform.length);

    // Extract the visible portion of the waveform
    const visibleWaveform = waveform.slice(startSample, endSample);
    if (visibleWaveform.length === 0) return;

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
    const samplesPerBar = Math.max(1, Math.floor(visibleWaveform.length / maxBars));
    const numBars = Math.ceil(visibleWaveform.length / samplesPerBar);
    const barWidth = canvasWidth / numBars;

    ctx.fillStyle = 'rgba(200, 200, 200, 0.5)';

    // Draw bars, using peak value for each segment
    for (let i = 0; i < numBars; i++) {
      const startIdx = i * samplesPerBar;
      const endIdx = Math.min(startIdx + samplesPerBar, visibleWaveform.length);

      // Get peak value for this segment
      let peak = 0;
      for (let j = startIdx; j < endIdx; j++) {
        if (visibleWaveform[j] > peak) peak = visibleWaveform[j];
      }

      const barHeight = Math.max(2, peak * (height - 4));
      const x = i * barWidth;
      const y = (height - barHeight) / 2;
      ctx.fillRect(x, y, Math.max(1, barWidth - 0.5), barHeight);
    }
  }, [waveform, width, height, inPoint, outPoint, naturalDuration]);

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
        ctx.fillStyle = 'rgba(180, 180, 120, 0.7)';
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

// FadeCurve - Renders SVG bezier curve showing opacity fade
// Note: Not using memo() here to ensure re-render on keyframe changes
function FadeCurve({
  keyframes,
  clipDuration,
  width,
  height,
}: {
  keyframes: Array<{
    time: number;
    value: number;
    easing: string;
    handleIn?: { x: number; y: number };
    handleOut?: { x: number; y: number };
  }>;
  clipDuration: number;
  width: number;
  height: number;
}) {
  if (keyframes.length < 2 || width <= 0 || height <= 0) return null;

  // Sort keyframes by time
  const sorted = [...keyframes].sort((a, b) => a.time - b.time);

  // Build SVG path
  const timeToX = (t: number) => (t / clipDuration) * width;
  const valueToY = (v: number) => height - v * height; // Invert Y (0 at bottom, 1 at top)

  // Generate path segments between keyframes
  const pathSegments: string[] = [];

  // Start from the first keyframe
  const firstKf = sorted[0];
  pathSegments.push(`M ${timeToX(firstKf.time)} ${valueToY(firstKf.value)}`);

  for (let i = 0; i < sorted.length - 1; i++) {
    const kf1 = sorted[i];
    const kf2 = sorted[i + 1];

    const x1 = timeToX(kf1.time);
    const y1 = valueToY(kf1.value);
    const x2 = timeToX(kf2.time);
    const y2 = valueToY(kf2.value);

    const duration = kf2.time - kf1.time;

    // Determine control points based on easing type
    let cp1x: number, cp1y: number, cp2x: number, cp2y: number;

    if (kf1.easing === 'bezier' && kf1.handleOut && kf2.handleIn) {
      // Custom bezier handles
      cp1x = timeToX(kf1.time + kf1.handleOut.x);
      cp1y = valueToY(kf1.value + kf1.handleOut.y);
      cp2x = timeToX(kf2.time + kf2.handleIn.x);
      cp2y = valueToY(kf2.value + kf2.handleIn.y);
    } else {
      // Standard easing curves (cubic-bezier approximations)
      switch (kf1.easing) {
        case 'ease-in':
          // Slow start, fast end: (0.42, 0, 1, 1)
          cp1x = x1 + duration * 0.42 * (width / clipDuration);
          cp1y = y1;
          cp2x = x2;
          cp2y = y2;
          break;
        case 'ease-out':
          // Fast start, slow end: (0, 0, 0.58, 1)
          cp1x = x1;
          cp1y = y1;
          cp2x = x1 + duration * 0.58 * (width / clipDuration);
          cp2y = y2;
          break;
        case 'ease-in-out':
          // Slow start and end: (0.42, 0, 0.58, 1)
          cp1x = x1 + duration * 0.42 * (width / clipDuration);
          cp1y = y1;
          cp2x = x1 + duration * 0.58 * (width / clipDuration);
          cp2y = y2;
          break;
        case 'linear':
        default:
          // Linear: straight line (use same point for both control points)
          cp1x = x1 + (x2 - x1) / 3;
          cp1y = y1 + (y2 - y1) / 3;
          cp2x = x1 + (x2 - x1) * 2 / 3;
          cp2y = y1 + (y2 - y1) * 2 / 3;
          break;
      }
    }

    pathSegments.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`);
  }

  const curvePath = pathSegments.join(' ');

  // Create filled area path (curve + bottom edge)
  const lastKf = sorted[sorted.length - 1];
  const fillPath = `${curvePath} L ${timeToX(lastKf.time)} ${height} L ${timeToX(firstKf.time)} ${height} Z`;

  return (
    <svg
      className="fade-curve-svg"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
    >
      {/* Filled area under curve */}
      <path
        d={fillPath}
        fill="rgba(0, 0, 0, 0.4)"
        stroke="none"
      />
      {/* Curve line */}
      <path
        d={curvePath}
        fill="none"
        stroke="rgba(140, 180, 220, 0.8)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Keyframe dots */}
      {sorted.map((kf, i) => (
        <circle
          key={i}
          cx={timeToX(kf.time)}
          cy={valueToY(kf.value)}
          r="3"
          fill="rgba(140, 180, 220, 1)"
        />
      ))}
    </svg>
  );
}

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
  timeToPixel,
  pixelToTime,
  formatTime,
  onPickWhipDragStart,
  onPickWhipDragEnd,
  onSetClipParent,
}: TimelineClipProps) {
  const thumbnails = clip.thumbnails || [];

  // Animation phase for enter/exit transitions
  const clipAnimationPhase = useTimelineStore(s => s.clipAnimationPhase);
  const clipEntranceKey = useTimelineStore(s => s.clipEntranceAnimationKey);
  const mountKeyRef = useRef(clipEntranceKey);

  // Calculate stagger delay based on track index (vertical) + startTime (horizontal)
  const trackIndex = track ? tracks.findIndex(t => t.id === track.id) : 0;
  // 80ms per track + 20ms per second of timeline position
  const animationDelay = (trackIndex * 0.08) + Math.min(clip.startTime * 0.02, 0.5);

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
    const maxDuration = clip.source?.naturalDuration || clip.duration;

    if (clipTrim.edge === 'left') {
      const maxTrim = clipTrim.originalDuration - 0.1;
      const minTrim = -clipTrim.originalInPoint;
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

  // Get parent clip name for tooltip
  const parentClip = clip.parentClipId ? clips.find(c => c.id === clip.parentClipId) : null;

  // Subscribe to playhead position only when cut tool is active (avoids re-renders during playback)
  const playheadPosition = useTimelineStore((state) =>
    toolMode === 'cut' ? state.playheadPosition : 0
  );

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
      {isAudioClip && clip.waveform && clip.waveform.length > 0 && (
        <div className="clip-waveform">
          <Waveform
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
      {clip.isComposition && clip.mixdownWaveform && clip.mixdownWaveform.length > 0 && (
        <div className="clip-mixdown-waveform">
          <Waveform
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
      {clip.isComposition && clip.clipSegments && clip.clipSegments.length > 0 && !isAudioClip && (
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
      {thumbnails.length > 0 && !isAudioClip && !(clip.isComposition && clip.clipSegments && clip.clipSegments.length > 0) && (
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
          <PickWhip
            clipId={clip.id}
            clipName={clip.name}
            parentClipId={clip.parentClipId}
            parentClipName={parentClip?.name}
            onSetParent={onSetClipParent}
            onDragStart={onPickWhipDragStart}
            onDragEnd={onPickWhipDragEnd}
          />
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
