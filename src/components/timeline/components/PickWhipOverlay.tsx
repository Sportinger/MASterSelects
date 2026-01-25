// Pick whip drag overlay for layer parenting

import { PhysicsCable } from '../PhysicsCable';
import type { PickWhipDragState } from '../types';

interface PickWhipOverlayProps {
  dragState: PickWhipDragState | null;
}

export function PickWhipOverlay({ dragState }: PickWhipOverlayProps) {
  if (!dragState) return null;

  return (
    <svg
      className="pick-whip-drag-overlay"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
        zIndex: 9999,
      }}
    >
      <PhysicsCable
        startX={dragState.startX}
        startY={dragState.startY}
        endX={dragState.currentX}
        endY={dragState.currentY}
        isPreview={true}
      />
    </svg>
  );
}
