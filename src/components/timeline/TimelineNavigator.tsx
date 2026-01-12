// TimelineNavigator - Horizontal scrollbar with zoom handles
// Shows the visible range and allows dragging/resizing

import { useRef, useState, useCallback, useEffect } from 'react';

// Padding in pixels to show beyond the end of the composition (must match useTimelineZoom)
const END_PADDING = 100;

interface TimelineNavigatorProps {
  duration: number;
  scrollX: number;
  zoom: number;
  viewportWidth: number;
  minZoom: number;
  maxZoom: number;
  onScrollChange: (scrollX: number) => void;
  onZoomChange: (zoom: number) => void;
}

export function TimelineNavigator({
  duration,
  scrollX,
  zoom,
  viewportWidth,
  minZoom,
  maxZoom,
  onScrollChange,
  onZoomChange,
}: TimelineNavigatorProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState<'thumb' | 'left' | 'right' | null>(null);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartScrollX, setDragStartScrollX] = useState(0);
  const [dragStartZoom, setDragStartZoom] = useState(0);
  const [dragStartThumbLeft, setDragStartThumbLeft] = useState(0);
  const [dragStartThumbWidth, setDragStartThumbWidth] = useState(0);

  // Calculate thumb position and size
  const totalContentWidth = duration * zoom;
  const trackWidth = trackRef.current?.clientWidth ?? 200;

  // Thumb width represents the viewport as a fraction of total content
  const thumbWidthRatio = Math.min(1, viewportWidth / Math.max(1, totalContentWidth));
  const thumbWidth = Math.max(40, thumbWidthRatio * trackWidth); // Min 40px width

  // Thumb position represents scrollX as a fraction of scrollable area
  const maxScrollX = Math.max(0, totalContentWidth - viewportWidth);
  const scrollRatio = maxScrollX > 0 ? scrollX / maxScrollX : 0;
  const thumbLeft = scrollRatio * (trackWidth - thumbWidth);

  const handleMouseDown = useCallback((e: React.MouseEvent, type: 'thumb' | 'left' | 'right') => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(type);
    setDragStartX(e.clientX);
    setDragStartScrollX(scrollX);
    setDragStartZoom(zoom);
    setDragStartThumbLeft(thumbLeft);
    setDragStartThumbWidth(thumbWidth);
  }, [scrollX, zoom, thumbLeft, thumbWidth]);

  const handleTrackClick = useCallback((e: React.MouseEvent) => {
    if (!trackRef.current || isDragging) return;

    const rect = trackRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickRatio = clickX / rect.width;

    // Jump to clicked position (center the thumb there)
    const newScrollRatio = Math.max(0, Math.min(1, clickRatio - thumbWidthRatio / 2));
    const newScrollX = newScrollRatio * maxScrollX;
    onScrollChange(Math.max(0, newScrollX));
  }, [isDragging, thumbWidthRatio, maxScrollX, onScrollChange]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - dragStartX;
      const trackWidth = trackRef.current?.clientWidth ?? 200;

      if (isDragging === 'thumb') {
        // Drag the thumb to scroll
        const scrollableTrackWidth = trackWidth - thumbWidth;
        if (scrollableTrackWidth <= 0) return;

        const deltaRatio = deltaX / scrollableTrackWidth;
        const newScrollX = dragStartScrollX + deltaRatio * maxScrollX;
        onScrollChange(Math.max(0, Math.min(maxScrollX, newScrollX)));
      } else if (isDragging === 'left') {
        // Resize from left - handle follows mouse 1:1
        // New thumb width = original width minus the delta (drag left = wider thumb)
        // Cap at trackWidth to prevent zooming out beyond duration (with padding)
        const newThumbWidth = Math.max(40, Math.min(trackWidth, dragStartThumbWidth - deltaX));

        // Calculate zoom from thumb width: thumbWidth = (viewportWidth / (duration * zoom)) * trackWidth
        // Rearranged: zoom = viewportWidth * trackWidth / (duration * thumbWidth)
        // Use padding to allow seeing end marker
        const dynamicMinZoom = (viewportWidth - END_PADDING) / duration;
        const newZoom = Math.max(dynamicMinZoom, Math.min(maxZoom, (viewportWidth * trackWidth) / (duration * newThumbWidth)));
        onZoomChange(newZoom);

        // Adjust scroll to keep right edge stable (with padding)
        const newTotalWidth = duration * newZoom;
        const newMaxScrollX = Math.max(0, newTotalWidth - viewportWidth + END_PADDING);
        const rightEdge = dragStartScrollX + viewportWidth;
        const newScrollX = Math.max(0, Math.min(newMaxScrollX, rightEdge * (newZoom / dragStartZoom) - viewportWidth));
        onScrollChange(newScrollX);
      } else if (isDragging === 'right') {
        // Resize from right - handle follows mouse 1:1
        // New thumb width = original width plus the delta (drag right = wider thumb)
        // Cap at trackWidth to prevent zooming out beyond duration (with padding)
        const newThumbWidth = Math.max(40, Math.min(trackWidth, dragStartThumbWidth + deltaX));

        // Calculate zoom from thumb width (with padding)
        const dynamicMinZoom = (viewportWidth - END_PADDING) / duration;
        const newZoom = Math.max(dynamicMinZoom, Math.min(maxZoom, (viewportWidth * trackWidth) / (duration * newThumbWidth)));
        onZoomChange(newZoom);

        // Keep left edge stable (with padding)
        const newTotalWidth = duration * newZoom;
        const newMaxScrollX = Math.max(0, newTotalWidth - viewportWidth + END_PADDING);
        const newScrollX = Math.min(newMaxScrollX, dragStartScrollX * (newZoom / dragStartZoom));
        onScrollChange(Math.max(0, newScrollX));
      }
    };

    const handleMouseUp = () => {
      setIsDragging(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragStartX, dragStartScrollX, dragStartZoom, dragStartThumbWidth, thumbWidth, trackWidth, maxScrollX, duration, viewportWidth, minZoom, maxZoom, onScrollChange, onZoomChange]);

  return (
    <div className="timeline-navigator">
      <div
        className="timeline-navigator-track"
        ref={trackRef}
        onClick={handleTrackClick}
      >
        <div
          className={`timeline-navigator-thumb ${isDragging ? 'dragging' : ''}`}
          style={{
            left: thumbLeft,
            width: thumbWidth,
          }}
          onMouseDown={(e) => handleMouseDown(e, 'thumb')}
        >
          <div
            className="timeline-navigator-handle timeline-navigator-handle-left"
            onMouseDown={(e) => handleMouseDown(e, 'left')}
          />
          <div className="timeline-navigator-grip" />
          <div
            className="timeline-navigator-handle timeline-navigator-handle-right"
            onMouseDown={(e) => handleMouseDown(e, 'right')}
          />
        </div>
      </div>
    </div>
  );
}
