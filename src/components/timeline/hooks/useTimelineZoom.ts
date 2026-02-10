// useTimelineZoom - Zoom, scroll, and wheel handling for timeline
// Extracted from Timeline.tsx for better maintainability

import { useEffect, useCallback, useRef } from 'react';
import { MIN_ZOOM, MAX_ZOOM } from '../../../stores/timeline/constants';
import { animateSlotGrid } from '../slotGridAnimation';

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
  trackSnapPositions: number[];

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
  trackSnapPositions,
  setZoom,
  setScrollX,
  setScrollY,
}: UseTimelineZoomProps): UseTimelineZoomReturn {
  // Ref to avoid stale closure for scrollY in wheel handler
  const scrollYRef = useRef(scrollY);
  scrollYRef.current = scrollY;

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

  // Padding in pixels to show beyond the end of the composition
  const END_PADDING = 100;

  // Calculate dynamic minimum zoom to prevent zooming out too far beyond duration
  const getDynamicMinZoom = useCallback(() => {
    const trackLanes = timelineBodyRef.current?.querySelector('.track-lanes');
    const viewportWidth = trackLanes?.clientWidth ?? 800;
    // Allow some padding at the end to see the end marker
    // Min zoom ensures duration * zoom >= viewportWidth - END_PADDING
    return Math.max(MIN_ZOOM, (viewportWidth - END_PADDING) / duration);
  }, [timelineBodyRef, duration]);

  // Wrapper for setZoom that enforces dynamic min zoom
  const handleSetZoom = useCallback((newZoom: number) => {
    const dynamicMinZoom = getDynamicMinZoom();
    setZoom(Math.max(dynamicMinZoom, Math.min(MAX_ZOOM, newZoom)));
  }, [setZoom, getDynamicMinZoom]);

  // Clamp zoom and scrollX when duration or viewport changes
  useEffect(() => {
    const trackLanes = timelineBodyRef.current?.querySelector('.track-lanes');
    const viewportWidth = trackLanes?.clientWidth ?? 800;
    const dynamicMinZoom = Math.max(MIN_ZOOM, (viewportWidth - END_PADDING) / duration);

    // Clamp zoom to dynamic minimum
    if (zoom < dynamicMinZoom) {
      setZoom(dynamicMinZoom);
    }

    // Clamp scrollX to max (allow scrolling up to END_PADDING past duration)
    const maxScrollX = Math.max(0, duration * zoom - viewportWidth + END_PADDING);
    if (scrollX > maxScrollX) {
      setScrollX(maxScrollX);
    }
  }, [timelineBodyRef, zoom, duration, scrollX, setZoom, setScrollX]);

  // Zoom with mouse wheel, also handle vertical scroll
  // Use native event listener with { passive: false } to allow preventDefault
  useEffect(() => {
    const el = timelineBodyRef.current;
    if (!el) return;

    const handleWheel = (e: WheelEvent) => {
      // Don't zoom when hovering over track headers (first column for height adjustment)
      const target = e.target as HTMLElement;
      const isOverTrackHeaders = target.closest('.track-headers') !== null;

      // Ctrl+Shift+Scroll: toggle slot grid view (single trigger, 250ms animation)
      if (e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        animateSlotGrid(e.deltaY > 0 ? 1 : 0);
        return;
      }

      if ((e.ctrlKey || e.altKey) && !isOverTrackHeaders) {
        e.preventDefault();
        // Get the track lanes container width for accurate centering
        const trackLanes = el.querySelector('.track-lanes');
        const viewportWidth = trackLanes?.clientWidth ?? el.clientWidth - 120; // 120 = track headers width

        // Calculate dynamic minimum zoom with padding to see end marker
        const dynamicMinZoom = Math.max(MIN_ZOOM, (viewportWidth - END_PADDING) / duration);

        // Exponential zoom: each scroll step changes zoom by a constant ratio
        // This feels consistent at all zoom levels (same % change per step)
        const zoomMultiplier = 1.08; // 8% per scroll step
        const newZoom = Math.max(dynamicMinZoom, Math.min(MAX_ZOOM,
          e.deltaY > 0 ? zoom / zoomMultiplier : zoom * zoomMultiplier
        ));

        // Calculate max scroll with padding
        const maxScrollX = Math.max(0, duration * newZoom - viewportWidth + END_PADDING);

        // Calculate playhead position in pixels with new zoom
        const playheadPixel = playheadPosition * newZoom;

        // Calculate scrollX to center playhead in viewport, clamped to valid range
        const newScrollX = Math.max(0, Math.min(maxScrollX, playheadPixel - viewportWidth / 2));

        setZoom(newZoom);
        setScrollX(newScrollX);
      } else if (e.shiftKey && !isOverTrackHeaders) {
        // Shift+scroll = horizontal scroll (use deltaY since mouse wheel is vertical)
        e.preventDefault();
        const trackLanes = el.querySelector('.track-lanes');
        const viewportWidth = trackLanes?.clientWidth ?? el.clientWidth - 120;
        const maxScrollX = Math.max(0, duration * zoom - viewportWidth + END_PADDING);
        setScrollX(Math.max(0, Math.min(maxScrollX, scrollX + e.deltaY)));
      } else {
        // Handle horizontal scroll (e.g., trackpad horizontal gesture)
        if (e.deltaX !== 0) {
          const trackLanes = el.querySelector('.track-lanes');
          const viewportWidth = trackLanes?.clientWidth ?? el.clientWidth - 120;
          const maxScrollX = Math.max(0, duration * zoom - viewportWidth + END_PADDING);
          setScrollX(Math.max(0, Math.min(maxScrollX, scrollX + e.deltaX)));
        }
        // Handle vertical scroll — snap to track boundaries (1 track per step)
        if (e.deltaY !== 0 && !e.shiftKey && !e.ctrlKey && !e.altKey) {
          e.preventDefault();
          const maxScrollY = Math.max(0, contentHeight - viewportHeight);
          const currentY = scrollYRef.current;
          if (trackSnapPositions.length > 1) {
            // Find current snap index
            let currentIdx = 0;
            for (let i = trackSnapPositions.length - 1; i >= 0; i--) {
              if (trackSnapPositions[i] <= currentY + 1) {
                currentIdx = i;
                break;
              }
            }
            const nextIdx = e.deltaY > 0
              ? Math.min(currentIdx + 1, trackSnapPositions.length - 1)
              : Math.max(currentIdx - 1, 0);
            const newY = Math.max(0, Math.min(maxScrollY, trackSnapPositions[nextIdx]));
            scrollYRef.current = newY;
            setScrollY(newY);
          } else {
            setScrollY(Math.max(0, Math.min(maxScrollY, currentY + e.deltaY)));
          }
        }
      }
    };

    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [timelineBodyRef, zoom, scrollX, playheadPosition, duration, contentHeight, viewportHeight, trackSnapPositions, setZoom, setScrollX, setScrollY]);

  return {
    handleSetZoom,
    handleFitToWindow,
  };
}
