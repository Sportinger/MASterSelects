// Render analysis overlay as line graphs (focus, motion) with real-time position indicator

import { memo, useRef, useEffect } from 'react';
import type { ClipAnalysis } from '../../../types';
import { useTimelineStore } from '../../../stores/timeline';

export const ClipAnalysisOverlay = memo(function ClipAnalysisOverlay({
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
    const focusMultiplier = 1.0;
    const heightScale = 0.6; // Use 60% of height
    ctx.beginPath();
    ctx.moveTo(0, height); // Start at bottom-left

    for (let i = 0; i < visibleFrames.length; i++) {
      const frame = visibleFrames[i];
      const frameInClip = frame.timestamp - clipInPoint;
      const x = (frameInClip / clipDuration) * width;
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
    const focusRedThreshold = 0.4;
    const focusGreenThreshold = 0.7;
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

      const avgFocus = (frame.focus + nextFrame.focus) / 2;
      const t = Math.min(1, Math.max(0, (avgFocus - focusRedThreshold) / (focusGreenThreshold - focusRedThreshold)));

      const r = Math.round(239 - t * (239 - 34));
      const g = Math.round(68 + t * (197 - 68));
      const b = Math.round(68 + t * (94 - 68));

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
      ctx.stroke();
    }

    // Draw filled area + line for Motion (blue) - from bottom
    const motionMultiplier = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, height);

    for (let i = 0; i < visibleFrames.length; i++) {
      const frame = visibleFrames[i];
      const frameInClip = frame.timestamp - clipInPoint;
      const x = (frameInClip / clipDuration) * width;
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
    const motionThreshold = 0.4;
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

      const avgMotion = (motionValue1 + motionValue2) / 2;
      const t = Math.min(1, Math.max(0, avgMotion / motionThreshold));

      const r = Math.round(59 + t * (239 - 59));
      const g = Math.round(130 - t * (130 - 68));
      const b = Math.round(246 - t * (246 - 68));

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
