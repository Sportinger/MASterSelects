// Pick Whip Drag Hook - handles clip and track parenting via drag

import { useState, useCallback } from 'react';
import type { PickWhipDragState } from '../types';

interface UsePickWhipDragProps {
  setClipParent: (clipId: string, parentClipId: string | null) => void;
  setTrackParent: (trackId: string, parentTrackId: string | null) => void;
}

export function usePickWhipDrag({ setClipParent, setTrackParent }: UsePickWhipDragProps) {
  // Pick whip disabled
  const noop = useCallback(() => {}, []);
  return {
    pickWhipDrag: null,
    handlePickWhipDragStart: noop as any,
    handlePickWhipDragEnd: noop,
    trackPickWhipDrag: null,
    handleTrackPickWhipDragStart: noop as any,
    handleTrackPickWhipDragEnd: noop,
  };

  // Pick whip drag state for clip parenting
  const [pickWhipDrag, setPickWhipDrag] = useState<PickWhipDragState | null>(null);

  // Pick whip drag state for track/layer parenting
  const [trackPickWhipDrag, setTrackPickWhipDrag] = useState<PickWhipDragState | null>(null);

  // Clip pick whip handlers
  const handlePickWhipDragStart = useCallback((clipId: string, startX: number, startY: number) => {
    setPickWhipDrag({
      sourceClipId: clipId,
      startX,
      startY,
      currentX: startX,
      currentY: startY,
    });

    const handleMouseMove = (e: MouseEvent) => {
      setPickWhipDrag(prev => prev ? {
        ...prev,
        currentX: e.clientX,
        currentY: e.clientY,
      } : null);
    };

    const handleMouseUp = (e: MouseEvent) => {
      // Find clip at drop position
      const target = document.elementFromPoint(e.clientX, e.clientY);
      const clipElement = target?.closest('.timeline-clip');
      if (clipElement) {
        const targetClipId = clipElement.getAttribute('data-clip-id');
        if (targetClipId && targetClipId !== clipId) {
          setClipParent(clipId, targetClipId);
        }
      }
      setPickWhipDrag(null);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [setClipParent]);

  const handlePickWhipDragEnd = useCallback(() => {
    setPickWhipDrag(null);
  }, []);

  // Track pick whip handlers
  const handleTrackPickWhipDragStart = useCallback((trackId: string, startX: number, startY: number) => {
    setTrackPickWhipDrag({
      sourceClipId: trackId, // Using clipId field to store trackId
      startX,
      startY,
      currentX: startX,
      currentY: startY,
    });

    const handleMouseMove = (e: MouseEvent) => {
      setTrackPickWhipDrag(prev => prev ? {
        ...prev,
        currentX: e.clientX,
        currentY: e.clientY,
      } : null);
    };

    const handleMouseUp = (e: MouseEvent) => {
      // Find track header at drop position
      const target = document.elementFromPoint(e.clientX, e.clientY);
      const trackHeader = target?.closest('.track-header');
      if (trackHeader) {
        // Find the track-pick-whip with data-track-id inside the header
        const pickWhip = trackHeader.querySelector('.track-pick-whip');
        const targetTrackId = pickWhip?.getAttribute('data-track-id');
        if (targetTrackId && targetTrackId !== trackId) {
          setTrackParent(trackId, targetTrackId);
        }
      }
      setTrackPickWhipDrag(null);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [setTrackParent]);

  const handleTrackPickWhipDragEnd = useCallback(() => {
    setTrackPickWhipDrag(null);
  }, []);

  return {
    // Clip pick whip
    pickWhipDrag,
    handlePickWhipDragStart,
    handlePickWhipDragEnd,
    // Track pick whip
    trackPickWhipDrag,
    handleTrackPickWhipDragStart,
    handleTrackPickWhipDragEnd,
  };
}
