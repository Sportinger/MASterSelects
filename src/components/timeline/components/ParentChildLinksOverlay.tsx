// Parent-child link lines overlay (SVG cables between parented clips)

import type { TimelineClip, TimelineTrack as Track } from '../../../types';
import type { ClipDragState } from '../types';
import { ParentChildLink } from '../ParentChildLink';

interface ParentChildLinksOverlayProps {
  clips: TimelineClip[];
  tracks: Track[];
  clipDrag: ClipDragState | null;
  timelineRef: React.RefObject<HTMLDivElement | null>;
  scrollX: number;
  zoom: number;
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number;
}

export function ParentChildLinksOverlay({
  clips,
  tracks,
  clipDrag,
  timelineRef,
  scrollX,
  zoom,
  getExpandedTrackHeight,
}: ParentChildLinksOverlayProps) {
  return (
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
  );
}
