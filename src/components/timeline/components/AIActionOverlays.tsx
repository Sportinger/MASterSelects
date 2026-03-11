// AI Action Visual Feedback Overlays
// Renders transient visual effects for AI tool actions:
// - Split glow lines at cut positions
// - Delete ghost clips fading out
// - Trim edge highlights

import { useEffect, useLayoutEffect, useRef } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import type { TimelineTrack } from '../../../types';
import type { AIActionOverlay } from '../../../stores/timeline/types';

interface AIActionOverlaysProps {
  tracks: TimelineTrack[];
  timeToPixel: (time: number) => number;
  isTrackExpanded: (trackId: string) => boolean;
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number;
}

type TrackLayout = {
  top: number;
  height: number;
};

function buildTrackLayouts(
  tracks: TimelineTrack[],
  isTrackExpanded: (trackId: string) => boolean,
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number
): Map<string, TrackLayout> {
  let top = 0;
  const layouts = new Map<string, TrackLayout>();

  for (const track of tracks) {
    const height = isTrackExpanded(track.id)
      ? getExpandedTrackHeight(track.id, track.height)
      : track.height;
    layouts.set(track.id, { top, height });
    top += height;
  }

  return layouts;
}

function syncCanvasSize(canvas: HTMLCanvasElement): { width: number; height: number; dpr: number } {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  const nextWidth = Math.max(1, Math.round(width * dpr));
  const nextHeight = Math.max(1, Math.round(height * dpr));

  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }

  return { width, height, dpr };
}

function getSplitGlowOpacity(progress: number): number {
  if (progress <= 0.05) return progress / 0.05;
  if (progress <= 0.3) return 1;
  return Math.max(0, 1 - (progress - 0.3) / 0.7);
}

function getSplitGlowScale(progress: number): number {
  if (progress <= 0.05) return 0.8 + (progress / 0.05) * 0.2;
  if (progress <= 0.3) return 1;
  return Math.max(0.95, 1 - ((progress - 0.3) / 0.7) * 0.05);
}

function drawSplitGlow(
  ctx: CanvasRenderingContext2D,
  x: number,
  top: number,
  height: number,
  progress: number
): void {
  const opacity = getSplitGlowOpacity(progress);
  if (opacity <= 0) return;

  const scale = getSplitGlowScale(progress);
  const centerY = top + height / 2;
  const scaledHeight = Math.max(12, height * scale);
  const y1 = centerY - scaledHeight / 2;
  const y2 = centerY + scaledHeight / 2;
  const crispX = Math.round(x) + 0.5;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  ctx.globalAlpha = opacity * 0.28;
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 10;
  ctx.shadowColor = 'rgba(59, 130, 246, 0.9)';
  ctx.shadowBlur = 26;
  ctx.beginPath();
  ctx.moveTo(crispX, y1);
  ctx.lineTo(crispX, y2);
  ctx.stroke();

  ctx.globalAlpha = opacity * 0.9;
  ctx.strokeStyle = '#93c5fd';
  ctx.lineWidth = 2.5;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.moveTo(crispX, y1);
  ctx.lineTo(crispX, y2);
  ctx.stroke();

  ctx.globalAlpha = opacity * 0.75;
  ctx.shadowBlur = 12;
  ctx.fillStyle = '#3b82f6';

  ctx.beginPath();
  ctx.moveTo(crispX, y1 - 1);
  ctx.lineTo(crispX - 5, y1 + 7);
  ctx.lineTo(crispX + 5, y1 + 7);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(crispX, y2 + 1);
  ctx.lineTo(crispX - 5, y2 - 7);
  ctx.lineTo(crispX + 5, y2 - 7);
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}

function SplitGlowCanvas({
  overlays,
  trackLayouts,
  timeToPixel,
}: {
  overlays: AIActionOverlay[];
  trackLayouts: Map<string, TrackLayout>;
  timeToPixel: (time: number) => number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const resize = () => {
      syncCanvasSize(canvas);
    };

    resize();

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(resize);
      observer.observe(canvas);
      return () => observer.disconnect();
    }

    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || overlays.length === 0) return undefined;

    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    let frameId = 0;

    const render = () => {
      const { width, height, dpr } = syncCanvasSize(canvas);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const now = Date.now();
      let hasPendingAnimation = false;

      for (const overlay of overlays) {
        const layout = trackLayouts.get(overlay.trackId);
        if (!layout) continue;

        const elapsed = now - overlay.createdAt - (overlay.animationDelay || 0);
        if (elapsed < overlay.duration) {
          hasPendingAnimation = true;
        }
        if (elapsed < 0 || elapsed > overlay.duration) {
          continue;
        }

        drawSplitGlow(
          ctx,
          timeToPixel(overlay.timePosition),
          layout.top,
          layout.height,
          elapsed / overlay.duration
        );
      }

      if (hasPendingAnimation) {
        frameId = requestAnimationFrame(render);
      }
    };

    render();

    return () => {
      cancelAnimationFrame(frameId);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [overlays, timeToPixel, trackLayouts]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}

function OverlayElement({
  overlay,
  trackLayouts,
  timeToPixel,
}: {
  overlay: AIActionOverlay;
  trackLayouts: Map<string, TrackLayout>;
  timeToPixel: (time: number) => number;
}) {
  const layout = trackLayouts.get(overlay.trackId);
  if (!layout) return null;

  switch (overlay.type) {
    case 'delete-ghost': {
      const widthPx = overlay.width
        ? timeToPixel(overlay.timePosition + overlay.width) - timeToPixel(overlay.timePosition)
        : 4;
      return (
        <div
          className="ai-delete-ghost"
          style={{
            left: timeToPixel(overlay.timePosition),
            width: Math.max(widthPx, 4),
            top: layout.top + 4,
            height: layout.height - 8,
            backgroundColor: overlay.clipColor ? `${overlay.clipColor}66` : undefined,
          }}
        >
          {overlay.clipName && (
            <span className="ghost-name">{overlay.clipName}</span>
          )}
        </div>
      );
    }

    case 'trim-highlight':
      return (
        <div
          className="ai-trim-highlight"
          style={{
            left: timeToPixel(overlay.timePosition),
            top: layout.top,
            height: layout.height,
          }}
        />
      );

    case 'silent-zone': {
      const zoneWidth = overlay.width
        ? timeToPixel(overlay.timePosition + overlay.width) - timeToPixel(overlay.timePosition)
        : 4;
      return (
        <div
          className="ai-silent-zone"
          style={{
            left: timeToPixel(overlay.timePosition),
            width: Math.max(zoneWidth, 4),
            top: layout.top + 2,
            height: layout.height - 4,
          }}
        />
      );
    }

    case 'low-quality-zone': {
      const lqWidth = overlay.width
        ? timeToPixel(overlay.timePosition + overlay.width) - timeToPixel(overlay.timePosition)
        : 4;
      return (
        <div
          className="ai-low-quality-zone"
          style={{
            left: timeToPixel(overlay.timePosition),
            width: Math.max(lqWidth, 4),
            top: layout.top + 2,
            height: layout.height - 4,
          }}
        />
      );
    }

    default:
      return null;
  }
}

export function AIActionOverlays({
  tracks,
  timeToPixel,
  isTrackExpanded,
  getExpandedTrackHeight,
}: AIActionOverlaysProps) {
  const overlays = useTimelineStore(s => s.aiActionOverlays);

  if (overlays.length === 0) return null;

  const trackLayouts = buildTrackLayouts(tracks, isTrackExpanded, getExpandedTrackHeight);
  const splitGlowOverlays = overlays.filter(overlay => overlay.type === 'split-glow');
  const otherOverlays = overlays.filter(overlay => overlay.type !== 'split-glow');

  return (
    <div
      className="ai-action-overlays"
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 200,
      }}
    >
      {splitGlowOverlays.length > 0 && (
        <SplitGlowCanvas
          overlays={splitGlowOverlays}
          trackLayouts={trackLayouts}
          timeToPixel={timeToPixel}
        />
      )}

      {otherOverlays.map(overlay => (
        <OverlayElement
          key={overlay.id}
          overlay={overlay}
          trackLayouts={trackLayouts}
          timeToPixel={timeToPixel}
        />
      ))}
    </div>
  );
}
