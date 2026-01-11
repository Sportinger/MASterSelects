// usePlayheadDrag - Playhead and In/Out marker dragging
// Extracted from Timeline.tsx for better maintainability

import { useState, useCallback, useEffect } from 'react';
import type { MarkerDragState } from '../types';

interface UsePlayheadDragProps {
  // Refs
  timelineRef: React.RefObject<HTMLDivElement | null>;

  // State
  scrollX: number;
  duration: number;
  inPoint: number | null;
  outPoint: number | null;
  isRamPreviewing: boolean;

  // Actions
  setPlayheadPosition: (time: number) => void;
  setDraggingPlayhead: (dragging: boolean) => void;
  setInPoint: (time: number | null) => void;
  setOutPoint: (time: number | null) => void;
  cancelRamPreview: () => void;
  pixelToTime: (pixel: number) => number;
}

interface UsePlayheadDragReturn {
  markerDrag: MarkerDragState | null;
  handleRulerMouseDown: (e: React.MouseEvent) => void;
  handlePlayheadMouseDown: (e: React.MouseEvent) => void;
  handleMarkerMouseDown: (e: React.MouseEvent, type: 'in' | 'out') => void;
}

export function usePlayheadDrag({
  timelineRef,
  scrollX,
  duration,
  inPoint,
  outPoint,
  isRamPreviewing,
  setPlayheadPosition,
  setDraggingPlayhead,
  setInPoint,
  setOutPoint,
  cancelRamPreview,
  pixelToTime,
}: UsePlayheadDragProps): UsePlayheadDragReturn {
  // In/Out marker drag state
  const [markerDrag, setMarkerDrag] = useState<MarkerDragState | null>(null);

  // Handle time ruler mousedown
  const handleRulerMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();

      if (isRamPreviewing) {
        cancelRamPreview();
      }

      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const time = pixelToTime(x);
      setPlayheadPosition(Math.max(0, Math.min(time, duration)));

      setDraggingPlayhead(true);
    },
    [
      isRamPreviewing,
      cancelRamPreview,
      scrollX,
      pixelToTime,
      duration,
      setPlayheadPosition,
      setDraggingPlayhead,
    ]
  );

  // Handle playhead drag
  const handlePlayheadMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (isRamPreviewing) {
        cancelRamPreview();
      }
      setDraggingPlayhead(true);
    },
    [isRamPreviewing, cancelRamPreview, setDraggingPlayhead]
  );

  // Handle In/Out marker drag
  const handleMarkerMouseDown = useCallback(
    (e: React.MouseEvent, type: 'in' | 'out') => {
      e.stopPropagation();
      e.preventDefault();
      const originalTime = type === 'in' ? inPoint : outPoint;
      if (originalTime === null) return;

      setMarkerDrag({
        type,
        startX: e.clientX,
        originalTime,
      });
    },
    [inPoint, outPoint]
  );

  // Handle marker dragging
  useEffect(() => {
    if (!markerDrag) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!timelineRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const time = Math.max(0, Math.min(pixelToTime(x), duration));

      if (markerDrag.type === 'in') {
        const maxTime = outPoint !== null ? outPoint : duration;
        setInPoint(Math.min(time, maxTime));
      } else {
        const minTime = inPoint !== null ? inPoint : 0;
        setOutPoint(Math.max(time, minTime));
      }
    };

    const handleMouseUp = () => {
      setMarkerDrag(null);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [markerDrag, timelineRef, scrollX, duration, inPoint, outPoint, setInPoint, setOutPoint, pixelToTime]);

  return {
    markerDrag,
    handleRulerMouseDown,
    handlePlayheadMouseDown,
    handleMarkerMouseDown,
  };
}
