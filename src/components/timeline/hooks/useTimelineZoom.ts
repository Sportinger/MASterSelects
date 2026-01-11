// useTimelineZoom - Zoom, scroll, and wheel handling for timeline
// Extracted from Timeline.tsx for better maintainability

import { useEffect, useCallback } from 'react';
import { MIN_ZOOM, MAX_ZOOM } from '../../../stores/timeline/constants';

interface UseTimelineZoomProps {
  // Refs
  timelineBodyRef: React.RefObject<HTMLDivElement | null>;

  // State
  zoom: number;
  scrollX: number;
  scrollY: number;
  duration: number;
  playheadPosition: number;
  contentHeight: number;
  viewportHeight: number;

  // Actions
  setZoom: (zoom: number) => void;
  setScrollX: (scrollX: number) => void;
  setScrollY: (scrollY: number) => void;
}

interface UseTimelineZoomReturn {
  handleSetZoom: (newZoom: number) => void;
  handleFitToWindow: () => void;
}

export function useTimelineZoom({
  timelineBodyRef,
  zoom,
  scrollX,
  scrollY,
  duration,
  playheadPosition,
  contentHeight,
  viewportHeight,
  setZoom,
  setScrollX,
  setScrollY,
}: UseTimelineZoomProps): UseTimelineZoomReturn {
  // Fit composition to window - calculate zoom to show entire duration
  const handleFitToWindow = useCallback(() => {
    const trackLanes = timelineBodyRef.current?.querySelector('.track-lanes-scroll');
    const viewportWidth = trackLanes?.parentElement?.clientWidth ?? 800;
    // Calculate zoom: viewportWidth = duration * zoom, so zoom = viewportWidth / duration
    // Subtract some padding (50px) to not be right at the edge
    const targetZoom = Math.max(MIN_ZOOM, (viewportWidth - 50) / duration);
    setZoom(targetZoom);
    setScrollX(0); // Reset scroll to start
  }, [timelineBodyRef, duration, setZoom, setScrollX]);

  // Wrapper for setZoom that enforces MIN_ZOOM/MAX_ZOOM bounds
  const handleSetZoom = useCallback((newZoom: number) => {
    setZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom)));
  }, [setZoom]);

  // Zoom with mouse wheel, also handle vertical scroll
  // Use native event listener with { passive: false } to allow preventDefault
  useEffect(() => {
    const el = timelineBodyRef.current;
    if (!el) return;

    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.altKey) {
        e.preventDefault();
        // Get the track lanes container width for accurate centering
        const trackLanes = el.querySelector('.track-lanes');
        const viewportWidth = trackLanes?.clientWidth ?? el.clientWidth - 120; // 120 = track headers width

        // Adjust delta based on current zoom level for smoother zooming
        // Use smaller steps at low zoom levels for precision
        const zoomFactor = zoom < 1 ? 0.1 : zoom < 10 ? 1 : 5;
        const delta = e.deltaY > 0 ? -zoomFactor : zoomFactor;
        const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom + delta));

        // Calculate playhead position in pixels with new zoom
        const playheadPixel = playheadPosition * newZoom;

        // Calculate scrollX to center playhead in viewport
        const newScrollX = Math.max(0, playheadPixel - viewportWidth / 2);

        setZoom(newZoom);
        setScrollX(newScrollX);
      } else if (e.shiftKey) {
        // Shift+scroll = horizontal scroll (use deltaY since mouse wheel is vertical)
        e.preventDefault();
        setScrollX(Math.max(0, scrollX + e.deltaY));
      } else {
        // Handle horizontal scroll (e.g., trackpad horizontal gesture)
        if (e.deltaX !== 0) {
          setScrollX(Math.max(0, scrollX + e.deltaX));
        }
        // Handle vertical scroll with custom scrollbar
        if (e.deltaY !== 0 && !e.shiftKey && !e.ctrlKey && !e.altKey) {
          e.preventDefault();
          const maxScrollY = Math.max(0, contentHeight - viewportHeight);
          setScrollY(Math.max(0, Math.min(maxScrollY, scrollY + e.deltaY)));
        }
      }
    };

    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [timelineBodyRef, zoom, scrollX, scrollY, playheadPosition, duration, contentHeight, viewportHeight, setZoom, setScrollX, setScrollY]);

  return {
    handleSetZoom,
    handleFitToWindow,
  };
}
