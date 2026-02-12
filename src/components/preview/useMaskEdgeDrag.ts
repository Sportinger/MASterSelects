// Edge segment dragging (move two adjacent vertices together)

import { useRef, useCallback } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { ClipMask, TimelineClip } from '../../types';

export function useMaskEdgeDrag(
  svgRef: React.RefObject<SVGSVGElement | null>,
  canvasWidth: number,
  canvasHeight: number,
  selectedClip: TimelineClip | undefined,
  activeMask: ClipMask | undefined,
) {
  const { setMaskDragging } = useTimelineStore();

  const edgeDragState = useRef<{
    isDragging: boolean;
    startX: number;
    startY: number;
    vertexA: { id: string; x: number; y: number };
    vertexB: { id: string; x: number; y: number };
  }>({ isDragging: false, startX: 0, startY: 0, vertexA: { id: '', x: 0, y: 0 }, vertexB: { id: '', x: 0, y: 0 } });

  const handleEdgeMouseDown = useCallback((e: React.MouseEvent, vertexIdA: string, vertexIdB: string) => {
    e.stopPropagation();
    e.preventDefault();
    if (!activeMask || !selectedClip) return;

    const vA = activeMask.vertices.find(v => v.id === vertexIdA);
    const vB = activeMask.vertices.find(v => v.id === vertexIdB);
    if (!vA || !vB) return;

    setMaskDragging(true);
    edgeDragState.current = {
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY,
      vertexA: { id: vA.id, x: vA.x, y: vA.y },
      vertexB: { id: vB.id, x: vB.x, y: vB.y },
    };

    let lastUpdate = 0;
    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!edgeDragState.current.isDragging || !selectedClip || !activeMask) return;
      const now = performance.now();
      if (now - lastUpdate < 16) return;
      lastUpdate = now;

      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const scaleX = canvasWidth / rect.width;
      const scaleY = canvasHeight / rect.height;

      const dx = (moveEvent.clientX - edgeDragState.current.startX) * scaleX / canvasWidth;
      const dy = (moveEvent.clientY - edgeDragState.current.startY) * scaleY / canvasHeight;

      const { vertexA, vertexB } = edgeDragState.current;
      const newAx = Math.max(0, Math.min(1, vertexA.x + dx));
      const newAy = Math.max(0, Math.min(1, vertexA.y + dy));
      const newBx = Math.max(0, Math.min(1, vertexB.x + dx));
      const newBy = Math.max(0, Math.min(1, vertexB.y + dy));

      const { clips } = useTimelineStore.getState();
      const updatedClips = clips.map(c => {
        if (c.id !== selectedClip.id) return c;
        return {
          ...c,
          masks: (c.masks || []).map(m => {
            if (m.id !== activeMask.id) return m;
            return {
              ...m,
              vertices: m.vertices.map(v => {
                if (v.id === vertexA.id) return { ...v, x: newAx, y: newAy };
                if (v.id === vertexB.id) return { ...v, x: newBx, y: newBy };
                return v;
              }),
            };
          }),
        };
      });
      useTimelineStore.setState({ clips: updatedClips });
    };

    const handleMouseUp = () => {
      edgeDragState.current = { isDragging: false, startX: 0, startY: 0, vertexA: { id: '', x: 0, y: 0 }, vertexB: { id: '', x: 0, y: 0 } };
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      setMaskDragging(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [activeMask, selectedClip, canvasWidth, canvasHeight, setMaskDragging]);

  return { handleEdgeMouseDown };
}
