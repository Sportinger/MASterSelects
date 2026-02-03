// useMarkerDrag - Custom hook for timeline marker drag operations
// Consolidates useState and useEffect for marker dragging and creation

import { useState, useCallback, useEffect, type RefObject } from 'react';
import type { TimelineMarker } from '../../../stores/timeline/types';

interface TimelineMarkerDragState {
  markerId: string;
  startX: number;
  originalTime: number;
}

interface MarkerCreateDragState {
  isDragging: boolean;
  currentTime: number;
  isOverTimeline: boolean;
  dropAnimating: boolean;
}

interface UseMarkerDragParams {
  timelineRef: RefObject<HTMLDivElement | null>;
  timelineBodyRef: RefObject<HTMLDivElement | null>;
  markers: TimelineMarker[];
  scrollX: number;
  snappingEnabled: boolean;
  duration: number;
  playheadPosition: number;
  inPoint: number | null;
  outPoint: number | null;
  pixelToTime: (px: number) => number;
  getSnapTargetTimes: () => number[];
  moveMarker: (markerId: string, time: number) => void;
  addMarker: (time: number) => void;
}

interface UseMarkerDragReturn {
  /** Current marker drag state (dragging existing marker) */
  timelineMarkerDrag: TimelineMarkerDragState | null;
  /** Current marker create drag state (drag-to-create) */
  markerCreateDrag: MarkerCreateDragState | null;
  /** Mouse down handler for existing timeline markers */
  handleTimelineMarkerMouseDown: (e: React.MouseEvent, markerId: string) => void;
  /** Mouse down handler for the "Add Marker" button drag */
  handleMarkerButtonDragStart: (e: React.MouseEvent) => void;
}

/**
 * Custom hook for handling marker drag operations in the timeline.
 * Consolidates marker-related useState and useEffect calls.
 *
 * Handles two types of marker interactions:
 * 1. Dragging existing markers to reposition them
 * 2. Drag-to-create: dragging from the "M" button to place a new marker
 */
export function useMarkerDrag({
  timelineRef,
  timelineBodyRef,
  markers,
  scrollX,
  snappingEnabled,
  duration,
  playheadPosition,
  inPoint,
  outPoint,
  pixelToTime,
  getSnapTargetTimes,
  moveMarker,
  addMarker,
}: UseMarkerDragParams): UseMarkerDragReturn {
  // Timeline marker drag state (dragging existing markers)
  const [timelineMarkerDrag, setTimelineMarkerDrag] = useState<TimelineMarkerDragState | null>(null);

  // Drag-to-create marker state
  const [markerCreateDrag, setMarkerCreateDrag] = useState<MarkerCreateDragState | null>(null);

  // Handle timeline marker drag start
  const handleTimelineMarkerMouseDown = useCallback((e: React.MouseEvent, markerId: string) => {
    e.preventDefault();
    e.stopPropagation();
    const marker = markers.find(m => m.id === markerId);
    if (!marker) return;

    setTimelineMarkerDrag({
      markerId,
      startX: e.clientX,
      originalTime: marker.time,
    });
  }, [markers]);

  // Handle drag-to-create marker - start dragging from button
  const handleMarkerButtonDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setMarkerCreateDrag({
      isDragging: true,
      currentTime: playheadPosition,
      isOverTimeline: false,
      dropAnimating: false,
    });
  }, [playheadPosition]);

  // Effect: Handle timeline marker drag (existing markers)
  useEffect(() => {
    if (!timelineMarkerDrag) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!timelineRef.current) return;

      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      let time = pixelToTime(x);

      // Apply snapping if enabled (and not holding Alt)
      const shouldSnap = snappingEnabled !== e.altKey;
      if (shouldSnap) {
        const snapTimes = getSnapTargetTimes();
        // Also snap to playhead
        snapTimes.push(playheadPosition);
        // Snap to in/out points if set
        if (inPoint !== null) snapTimes.push(inPoint);
        if (outPoint !== null) snapTimes.push(outPoint);

        const snapThresholdTime = pixelToTime(10); // 10 pixels threshold
        let closestSnap = time;
        let minDist = Infinity;

        for (const snapTime of snapTimes) {
          const dist = Math.abs(time - snapTime);
          if (dist < minDist && dist < snapThresholdTime) {
            minDist = dist;
            closestSnap = snapTime;
          }
        }
        time = closestSnap;
      }

      // Clamp to valid range
      time = Math.max(0, Math.min(time, duration));
      moveMarker(timelineMarkerDrag.markerId, time);
    };

    const handleMouseUp = () => {
      setTimelineMarkerDrag(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [timelineMarkerDrag, scrollX, snappingEnabled, duration, pixelToTime, getSnapTargetTimes, moveMarker, playheadPosition, inPoint, outPoint, timelineRef]);

  // Effect: Handle drag-to-create marker
  useEffect(() => {
    if (!markerCreateDrag || !markerCreateDrag.isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      // Use timelineBodyRef for wider detection (includes ruler area)
      const bodyRef = timelineBodyRef.current;
      const trackRef = timelineRef.current;
      if (!bodyRef || !trackRef) return;

      const bodyRect = bodyRef.getBoundingClientRect();
      const trackRect = trackRef.getBoundingClientRect();

      // Calculate time from X position (using track area for proper offset)
      const x = e.clientX - trackRect.left + scrollX;
      let time = pixelToTime(x);

      // Apply snapping if enabled (and not holding Alt)
      const shouldSnap = snappingEnabled !== e.altKey;
      if (shouldSnap) {
        const snapTimes = getSnapTargetTimes();
        snapTimes.push(playheadPosition);
        if (inPoint !== null) snapTimes.push(inPoint);
        if (outPoint !== null) snapTimes.push(outPoint);

        const snapThresholdTime = pixelToTime(10);
        let closestSnap = time;
        let minDist = Infinity;

        for (const snapTime of snapTimes) {
          const dist = Math.abs(time - snapTime);
          if (dist < minDist && dist < snapThresholdTime) {
            minDist = dist;
            closestSnap = snapTime;
          }
        }
        time = closestSnap;
      }

      time = Math.max(0, Math.min(time, duration));

      // Check if mouse is over the timeline body (more generous area)
      const isOverTimeline = e.clientX >= bodyRect.left && e.clientX <= bodyRect.right &&
                             e.clientY >= bodyRect.top && e.clientY <= bodyRect.bottom;

      setMarkerCreateDrag(prev => prev ? { ...prev, currentTime: time, isOverTimeline } : null);
    };

    const handleMouseUp = (e: MouseEvent) => {
      const bodyRef = timelineBodyRef.current;
      if (!bodyRef || !markerCreateDrag) {
        setMarkerCreateDrag(null);
        return;
      }

      const bodyRect = bodyRef.getBoundingClientRect();
      const isOverTimeline = e.clientX >= bodyRect.left && e.clientX <= bodyRect.right &&
                             e.clientY >= bodyRect.top && e.clientY <= bodyRect.bottom;

      if (isOverTimeline) {
        // Add the marker and trigger drop animation
        addMarker(markerCreateDrag.currentTime);
        setMarkerCreateDrag(prev => prev ? { ...prev, isDragging: false, dropAnimating: true } : null);

        // Clear animation after it completes
        setTimeout(() => {
          setMarkerCreateDrag(null);
        }, 300);
      } else {
        setMarkerCreateDrag(null);
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [markerCreateDrag, scrollX, snappingEnabled, duration, pixelToTime, getSnapTargetTimes, addMarker, playheadPosition, inPoint, outPoint, timelineBodyRef, timelineRef]);

  return {
    timelineMarkerDrag,
    markerCreateDrag,
    handleTimelineMarkerMouseDown,
    handleMarkerButtonDragStart,
  };
}
