// AI Action Visual Feedback Overlays
// Renders transient visual effects for AI tool actions:
// - Split glow lines at cut positions
// - Delete ghost clips fading out
// - Trim edge highlights

import { useTimelineStore } from '../../../stores/timeline';
import type { TimelineTrack } from '../../../types';
import type { AIActionOverlay } from '../../../stores/timeline/types';

interface AIActionOverlaysProps {
  tracks: TimelineTrack[];
  timeToPixel: (time: number) => number;
  isTrackExpanded: (trackId: string) => boolean;
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number;
}

function getTrackLayout(
  trackId: string,
  tracks: TimelineTrack[],
  isTrackExpanded: (trackId: string) => boolean,
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number
): { top: number; height: number } | null {
  const track = tracks.find(t => t.id === trackId);
  if (!track) return null;

  const trackIndex = tracks.indexOf(track);
  const top = tracks
    .slice(0, trackIndex)
    .reduce((sum, t) => sum + (isTrackExpanded(t.id) ? getExpandedTrackHeight(t.id, t.height) : t.height), 0);
  const height = isTrackExpanded(track.id) ? getExpandedTrackHeight(track.id, track.height) : track.height;

  return { top, height };
}

function OverlayElement({
  overlay,
  tracks,
  timeToPixel,
  isTrackExpanded,
  getExpandedTrackHeight,
}: {
  overlay: AIActionOverlay;
  tracks: TimelineTrack[];
  timeToPixel: (time: number) => number;
  isTrackExpanded: (trackId: string) => boolean;
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number;
}) {
  const layout = getTrackLayout(overlay.trackId, tracks, isTrackExpanded, getExpandedTrackHeight);
  if (!layout) return null;

  switch (overlay.type) {
    case 'split-glow':
      return (
        <div
          className="ai-split-glow"
          style={{
            left: timeToPixel(overlay.timePosition),
            top: layout.top,
            height: layout.height,
            animationDelay: overlay.animationDelay ? `${overlay.animationDelay}ms` : undefined,
          }}
        />
      );

    case 'delete-ghost': {
      // Width needs to be calculated as pixel difference, not absolute timeToPixel
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
      {overlays.map(overlay => (
        <OverlayElement
          key={overlay.id}
          overlay={overlay}
          tracks={tracks}
          timeToPixel={timeToPixel}
          isTrackExpanded={isTrackExpanded}
          getExpandedTrackHeight={getExpandedTrackHeight}
        />
      ))}
    </div>
  );
}
