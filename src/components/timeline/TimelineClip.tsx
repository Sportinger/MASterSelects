// TimelineClip component - Clip rendering within tracks

import { memo, useRef, useEffect } from 'react';
import type { TimelineClipProps } from './types';
import { THUMB_WIDTH } from './constants';
import type { ClipAnalysis } from '../../types';

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
    if (!canvas || !waveform || waveform.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size (account for device pixel ratio for sharpness)
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Draw waveform bars
    const barWidth = width / waveform.length;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'; // Match existing .waveform-bar color for audio clips

    waveform.forEach((value, i) => {
      const barHeight = Math.max(2, value * (height - 8));
      const x = i * barWidth;
      const y = (height - barHeight) / 2;
      ctx.fillRect(x, y, Math.max(1, barWidth - 1), barHeight);
    });
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

// Render analysis overlay (focus, motion, face indicators)
const AnalysisOverlay = memo(function AnalysisOverlay({
  analysis,
  clipDuration,
  clipInPoint,
  width,
}: {
  analysis: ClipAnalysis;
  clipDuration: number;
  clipInPoint: number;
  width: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analysis?.frames.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const height = 12; // Fixed height for overlay
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const barHeight = 4;

    // Draw bars for each frame in the visible clip range
    for (const frame of analysis.frames) {
      // Frame timestamp is relative to source, adjust for clip's inPoint
      const frameInClip = frame.timestamp - clipInPoint;
      if (frameInClip < 0 || frameInClip > clipDuration) continue;

      const x = (frameInClip / clipDuration) * width;
      const barWidth = Math.max(2, (analysis.sampleInterval / 1000 / clipDuration) * width);

      // Focus bar (green) - top row
      if (frame.focus > 0.3) {
        ctx.fillStyle = `rgba(34, 197, 94, ${frame.focus})`;
        ctx.fillRect(x, 0, barWidth, barHeight);
      }

      // Motion bar (blue) - middle row
      if (frame.motion > 0.1) {
        ctx.fillStyle = `rgba(59, 130, 246, ${frame.motion})`;
        ctx.fillRect(x, barHeight, barWidth, barHeight);
      }

      // Face indicator (yellow) - bottom row
      if (frame.faceCount > 0) {
        ctx.fillStyle = `rgba(234, 179, 8, 0.8)`;
        ctx.fillRect(x, barHeight * 2, barWidth, barHeight);
      }
    }
  }, [analysis, clipDuration, clipInPoint, width]);

  if (!analysis?.frames.length) return null;

  return (
    <canvas
      ref={canvasRef}
      className="analysis-overlay-canvas"
      style={{ width, height: 12 }}
    />
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
}: TimelineClipProps) {
  const thumbnails = clip.thumbnails || [];

  // Determine if this is an audio clip (check source type, MIME type, or extension as fallback)
  const audioExtensions = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'aiff', 'opus'];
  const fileExt = clip.file?.name?.split('.').pop()?.toLowerCase() || '';
  const isAudioClip = clip.source?.type === 'audio' ||
    clip.file?.type?.startsWith('audio/') ||
    audioExtensions.includes(fileExt);

  // Debug: log audio clip waveform status
  if (isAudioClip) {
    console.log(`[TimelineClip] "${clip.name}" - isAudio: ${isAudioClip}, waveform: ${clip.waveform?.length || 0}, generating: ${clip.waveformGenerating}, thumbnails: ${thumbnails.length}`);
  }

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
    clipTypeClass,
    clip.isLoading ? 'loading' : '',
    hasProxy ? 'has-proxy' : '',
    isGeneratingProxy ? 'generating-proxy' : '',
    hasKeyframes(clip.id) ? 'has-keyframes' : '',
    clip.reversed ? 'reversed' : '',
    clip.transcriptStatus === 'ready' ? 'has-transcript' : '',
    clip.waveformGenerating ? 'generating-waveform' : '',
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
      {/* Analysis overlay */}
      {clip.analysis && clip.analysisStatus === 'ready' && (
        <div className="clip-analysis-overlay">
          <AnalysisOverlay
            analysis={clip.analysis}
            clipDuration={displayDuration}
            clipInPoint={clip.inPoint}
            width={width}
          />
        </div>
      )}
      {/* Analyzing indicator */}
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
