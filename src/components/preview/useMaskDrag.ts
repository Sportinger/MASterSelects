// Whole-mask dragging (move all vertices together)

import { useRef, useCallback } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { ClipMask, TimelineClip } from '../../types';

export function useMaskDrag(
  svgRef: React.RefObject<SVGSVGElement | null>,
  canvasWidth: number,
  canvasHeight: number,
  selectedClip: TimelineClip | undefined,
  activeMask: ClipMask | undefined,
) {
  const { setMaskDragging } = useTimelineStore();

  const maskDragState = useRef<{
    isDragging: boolean;
    startX: number;
    startY: number;
    startVertices: Array<{ id: string; x: number; y: number }>;
  }>({
    isDragging: false,
    startX: 0,
    startY: 0,
    startVertices: [],
  });

  const handleMaskDragStart = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();

    if (!activeMask || !selectedClip || !activeMask.visible) return;

    maskDragState.current = {
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY,
      startVertices: activeMask.vertices.map(v => ({ id: v.id, x: v.x, y: v.y })),
    };

    setMaskDragging(true);

    let lastMaskUpdate = 0;
    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!maskDragState.current.isDragging) return;
      if (!selectedClip || !activeMask) return;

      const now = performance.now();
      if (now - lastMaskUpdate < 16) return;
      lastMaskUpdate = now;

      const svg = svgRef.current;
      if (!svg) return;

      const rect = svg.getBoundingClientRect();
      const scaleX = canvasWidth / rect.width;
      const scaleY = canvasHeight / rect.height;

      const dx = (moveEvent.clientX - maskDragState.current.startX) * scaleX;
      const dy = (moveEvent.clientY - maskDragState.current.startY) * scaleY;

      const normalizedDx = dx / canvasWidth;
      const normalizedDy = dy / canvasHeight;

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
                const startVertex = maskDragState.current.startVertices.find(sv => sv.id === v.id);
                if (!startVertex) return v;
                return {
                  ...v,
                  x: Math.max(0, Math.min(1, startVertex.x + normalizedDx)),
                  y: Math.max(0, Math.min(1, startVertex.y + normalizedDy)),
                };
              }),
            };
          }),
        };
      });
      useTimelineStore.setState({ clips: updatedClips });
    };

    const handleMouseUp = () => {
      maskDragState.current = {
        isDragging: false,
        startX: 0,
        startY: 0,
        startVertices: [],
      };
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      useTimelineStore.getState().setMaskDragging(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [activeMask, selectedClip, canvasWidth, canvasHeight, setMaskDragging]);

  return { handleMaskDragStart };
}
