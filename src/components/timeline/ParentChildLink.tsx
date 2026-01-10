// ParentChildLink component - Visual connection line between parent and child clips
// Renders a dashed line from child clip to parent clip in the timeline

import type { TimelineClip, TimelineTrack } from '../../types';

interface ParentChildLinkProps {
  childClip: TimelineClip;
  parentClip: TimelineClip;
  tracks: TimelineTrack[]; // Used for type checking, passed for future extensibility
  zoom: number;
  scrollX: number;
  trackHeaderWidth: number;
  getTrackYPosition: (trackId: string) => number;
}

// Not memoized - needs to update in realtime when clips move
export function ParentChildLink({
  childClip,
  parentClip,
  tracks: _tracks,
  zoom,
  scrollX,
  trackHeaderWidth,
  getTrackYPosition,
}: ParentChildLinkProps) {
  // Calculate positions directly (no useMemo to ensure realtime updates)
  // Child position (start of clip)
  const childX = trackHeaderWidth + (childClip.startTime * zoom) - scrollX;
  const childY = getTrackYPosition(childClip.trackId);

  // Parent position (center of clip for better visibility)
  const parentCenterTime = parentClip.startTime + parentClip.duration / 2;
  const parentX = trackHeaderWidth + (parentCenterTime * zoom) - scrollX;
  const parentY = getTrackYPosition(parentClip.trackId);

  // Don't render if both are off-screen
  if (childX < -100 && parentX < -100) return null;
  if (childX > window.innerWidth + 100 && parentX > window.innerWidth + 100) return null;

  // Calculate control points for a smooth curve
  const midX = (childX + parentX) / 2;
  const midY = (childY + parentY) / 2;
  const curveOffset = Math.abs(childY - parentY) * 0.3;

  // Create a curved path
  const pathD = `M ${childX} ${childY} Q ${midX} ${midY - curveOffset} ${parentX} ${parentY}`;

  return (
    <g className="parent-child-link-group">
      {/* Main connection line */}
      <path
        className="parent-child-link"
        d={pathD}
        fill="none"
      />
      {/* Arrow at parent end */}
      <circle
        cx={parentX}
        cy={parentY}
        r="4"
        className="parent-child-link-endpoint"
      />
      {/* Small circle at child end */}
      <circle
        cx={childX}
        cy={childY}
        r="3"
        className="parent-child-link-start"
      />
    </g>
  );
}
