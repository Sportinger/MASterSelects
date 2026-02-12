// Mask vertex/handle dragging with document-level listeners

import { useRef, useCallback } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { MaskVertex, ClipMask, TimelineClip } from '../../types';
import { throttle } from './maskUtils';

export function useMaskVertexDrag(
  svgRef: React.RefObject<SVGSVGElement | null>,
  canvasWidth: number,
  canvasHeight: number,
  selectedClip: TimelineClip | undefined,
  activeMask: ClipMask | undefined,
) {
  const {
    selectVertex,
    updateVertex,
    setMaskDragging,
  } = useTimelineStore();

  const dragState = useRef<{
    vertexId: string | null;
    handleType: 'vertex' | 'handleIn' | 'handleOut' | null;
    startX: number;
    startY: number;
    startVertexX: number;
    startVertexY: number;
    startHandleX: number;
    startHandleY: number;
    lastShiftState: boolean;
    shiftStartX: number;
    shiftStartVertexX: number;
    shiftStartVertexY: number;
    startHandleInX: number;
    startHandleInY: number;
    startHandleOutX: number;
    startHandleOutY: number;
  }>({
    vertexId: null,
    handleType: null,
    startX: 0,
    startY: 0,
    startVertexX: 0,
    startVertexY: 0,
    startHandleX: 0,
    startHandleY: 0,
    lastShiftState: false,
    shiftStartX: 0,
    shiftStartVertexX: 0,
    shiftStartVertexY: 0,
    startHandleInX: 0,
    startHandleInY: 0,
    startHandleOutX: 0,
    startHandleOutY: 0,
  });

  const throttledUpdateVertex = useRef(
    throttle(
      (clipId: string, maskId: string, vertexId: string, updates: Partial<MaskVertex>) => {
        updateVertex(clipId, maskId, vertexId, updates, true);
      },
      16
    ) as (clipId: string, maskId: string, vertexId: string, updates: Partial<MaskVertex>) => void
  ).current;

  const handleVertexMouseDown = useCallback((
    e: React.MouseEvent,
    vertexId: string,
    handleType: 'vertex' | 'handleIn' | 'handleOut'
  ) => {
    e.stopPropagation();
    e.preventDefault();

    if (!activeMask || !selectedClip) return;

    const vertex = activeMask.vertices.find(v => v.id === vertexId);
    if (!vertex) return;

    selectVertex(vertexId, false);
    setMaskDragging(true);

    dragState.current = {
      vertexId,
      handleType,
      startX: e.clientX,
      startY: e.clientY,
      startVertexX: vertex.x,
      startVertexY: vertex.y,
      startHandleX: handleType === 'handleIn' ? vertex.handleIn.x : vertex.handleOut.x,
      startHandleY: handleType === 'handleIn' ? vertex.handleIn.y : vertex.handleOut.y,
      lastShiftState: false,
      shiftStartX: e.clientX,
      shiftStartVertexX: vertex.x,
      shiftStartVertexY: vertex.y,
      startHandleInX: vertex.handleIn.x,
      startHandleInY: vertex.handleIn.y,
      startHandleOutX: vertex.handleOut.x,
      startHandleOutY: vertex.handleOut.y,
    };

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragState.current.vertexId || !dragState.current.handleType) return;
      if (!selectedClip || !activeMask) return;

      const svg = svgRef.current;
      if (!svg) return;

      const rect = svg.getBoundingClientRect();
      const scaleX = canvasWidth / rect.width;
      const scaleY = canvasHeight / rect.height;

      const isShiftPressed = moveEvent.shiftKey;

      if (isShiftPressed && !dragState.current.lastShiftState) {
        dragState.current.shiftStartX = moveEvent.clientX;
        const { clips } = useTimelineStore.getState();
        const clip = clips.find(c => c.id === selectedClip.id);
        const mask = clip?.masks?.find(m => m.id === activeMask.id);
        const currentVertex = mask?.vertices.find(v => v.id === dragState.current.vertexId);
        if (currentVertex) {
          dragState.current.shiftStartVertexX = currentVertex.x;
          dragState.current.shiftStartVertexY = currentVertex.y;
        }
      }
      dragState.current.lastShiftState = isShiftPressed;

      if (dragState.current.handleType === 'vertex') {
        if (isShiftPressed) {
          const shiftDx = (moveEvent.clientX - dragState.current.shiftStartX) * scaleX;
          const normalizedShiftDx = shiftDx / canvasWidth;
          const scaleFactor = 1 + normalizedShiftDx * 5;

          throttledUpdateVertex(selectedClip.id, activeMask.id, dragState.current.vertexId, {
            x: dragState.current.shiftStartVertexX,
            y: dragState.current.shiftStartVertexY,
            handleIn: {
              x: dragState.current.startHandleInX * scaleFactor,
              y: dragState.current.startHandleInY * scaleFactor,
            },
            handleOut: {
              x: dragState.current.startHandleOutX * scaleFactor,
              y: dragState.current.startHandleOutY * scaleFactor,
            },
          });
        } else {
          const dx = (moveEvent.clientX - dragState.current.startX) * scaleX;
          const dy = (moveEvent.clientY - dragState.current.startY) * scaleY;
          const normalizedDx = dx / canvasWidth;
          const normalizedDy = dy / canvasHeight;

          const newX = Math.max(0, Math.min(1, dragState.current.startVertexX + normalizedDx));
          const newY = Math.max(0, Math.min(1, dragState.current.startVertexY + normalizedDy));
          throttledUpdateVertex(selectedClip.id, activeMask.id, dragState.current.vertexId, {
            x: newX,
            y: newY,
          });
        }
      } else {
        const dx = (moveEvent.clientX - dragState.current.startX) * scaleX;
        const dy = (moveEvent.clientY - dragState.current.startY) * scaleY;
        const normalizedDx = dx / canvasWidth;
        const normalizedDy = dy / canvasHeight;

        const handleKey = dragState.current.handleType;
        throttledUpdateVertex(selectedClip.id, activeMask.id, dragState.current.vertexId, {
          [handleKey]: {
            x: dragState.current.startHandleX + normalizedDx,
            y: dragState.current.startHandleY + normalizedDy,
          },
        });
      }
    };

    const handleMouseUp = () => {
      dragState.current = {
        vertexId: null,
        handleType: null,
        startX: 0,
        startY: 0,
        startVertexX: 0,
        startVertexY: 0,
        startHandleX: 0,
        startHandleY: 0,
        lastShiftState: false,
        shiftStartX: 0,
        shiftStartVertexX: 0,
        shiftStartVertexY: 0,
        startHandleInX: 0,
        startHandleInY: 0,
        startHandleOutX: 0,
        startHandleOutY: 0,
      };
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      setMaskDragging(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [activeMask, selectedClip, selectVertex, canvasWidth, canvasHeight, setMaskDragging, throttledUpdateVertex]);

  return { handleVertexMouseDown };
}
