// Pick whip drag overlays (clip and track parenting cables)

import { PhysicsCable } from '../PhysicsCable';

interface DragState {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

interface PickWhipCablesProps {
  pickWhipDrag: DragState | null;
  trackPickWhipDrag: DragState | null;
}

export function PickWhipCables({ pickWhipDrag, trackPickWhipDrag }: PickWhipCablesProps) {
  return (
    <>
      {/* Pick whip drag line - physics cable (clip parenting) */}
      {pickWhipDrag && (
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
            startX={pickWhipDrag.startX}
            startY={pickWhipDrag.startY}
            endX={pickWhipDrag.currentX}
            endY={pickWhipDrag.currentY}
            isPreview={true}
          />
        </svg>
      )}

      {/* Track pick whip drag line - physics cable (layer parenting) */}
      {trackPickWhipDrag && (
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
            startX={trackPickWhipDrag.startX}
            startY={trackPickWhipDrag.startY}
            endX={trackPickWhipDrag.currentX}
            endY={trackPickWhipDrag.currentY}
            isPreview={true}
          />
        </svg>
      )}
    </>
  );
}
