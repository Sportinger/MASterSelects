// Playhead Snap Effect Hook - handles playhead dragging with snapping

import { useEffect } from 'react';

interface UsePlayheadSnapProps {
  isDraggingPlayhead: boolean;
  timelineRef: React.RefObject<HTMLDivElement>;
  scrollX: number;
  duration: number;
  snappingEnabled: boolean;
  pixelToTime: (pixel: number) => number;
  getSnapTargetTimes: () => number[];
  setPlayheadPosition: (position: number) => void;
  setDraggingPlayhead: (dragging: boolean) => void;
}

export function usePlayheadSnap({
  isDraggingPlayhead,
  timelineRef,
  scrollX,
  duration,
  snappingEnabled,
  pixelToTime,
  getSnapTargetTimes,
  setPlayheadPosition,
  setDraggingPlayhead,
}: UsePlayheadSnapProps) {
  useEffect(() => {
    if (!isDraggingPlayhead) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!timelineRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      let time = pixelToTime(x);

      // Snapping with Alt-key toggle:
      // - When snapping enabled: snap by default, Alt temporarily disables
      // - When snapping disabled: don't snap, Alt temporarily enables
      const shouldSnap = snappingEnabled !== e.altKey;

      if (shouldSnap) {
        const snapTimes = getSnapTargetTimes();
        const snapThreshold = pixelToTime(10);

        let closestSnap: number | null = null;
        let closestDistance = Infinity;

        for (const snapTime of snapTimes) {
          const distance = Math.abs(time - snapTime);
          if (distance < closestDistance && distance < snapThreshold) {
            closestDistance = distance;
            closestSnap = snapTime;
          }
        }

        if (closestSnap !== null) {
          time = closestSnap;
        }
      }

      setPlayheadPosition(Math.max(0, Math.min(time, duration)));
    };

    const handleMouseUp = () => {
      setDraggingPlayhead(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [
    isDraggingPlayhead,
    scrollX,
    duration,
    snappingEnabled,
    setPlayheadPosition,
    setDraggingPlayhead,
    pixelToTime,
    getSnapTargetTimes,
    timelineRef,
  ]);
}
