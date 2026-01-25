// useClipDrag - Premiere-style clip dragging with snapping
// Extracted from Timeline.tsx for better maintainability

import { useState, useCallback, useRef } from 'react';
import type { TimelineClip, TimelineTrack } from '../../../types';
import type { ClipDragState } from '../types';

interface UseClipDragProps {
  // Refs
  trackLanesRef: React.RefObject<HTMLDivElement | null>;
  timelineRef: React.RefObject<HTMLDivElement | null>;

  // State
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  clipMap: Map<string, TimelineClip>;
  selectedClipIds: Set<string>;
  scrollX: number;
  snappingEnabled: boolean;

  // Actions
  selectClip: (clipId: string | null, addToSelection?: boolean) => void;
  moveClip: (clipId: string, newStartTime: number, trackId: string, skipLinked?: boolean, skipGroup?: boolean) => void;
  openCompositionTab: (compositionId: string) => void;

  // Helpers
  pixelToTime: (pixel: number) => number;
  getSnappedPosition: (clipId: string, rawTime: number, trackId: string) => { startTime: number; snapped: boolean };
  getPositionWithResistance: (clipId: string, rawTime: number, trackId: string, duration: number) => { startTime: number; forcingOverlap: boolean };
}

interface UseClipDragReturn {
  clipDrag: ClipDragState | null;
  clipDragRef: React.MutableRefObject<ClipDragState | null>;
  handleClipMouseDown: (e: React.MouseEvent, clipId: string) => void;
  handleClipDoubleClick: (e: React.MouseEvent, clipId: string) => void;
}

export function useClipDrag({
  trackLanesRef,
  timelineRef,
  clips: _clips,
  tracks,
  clipMap,
  selectedClipIds,
  scrollX,
  snappingEnabled,
  selectClip,
  moveClip,
  openCompositionTab,
  pixelToTime,
  getSnappedPosition,
  getPositionWithResistance,
}: UseClipDragProps): UseClipDragReturn {
  const [clipDrag, setClipDrag] = useState<ClipDragState | null>(null);
  const clipDragRef = useRef<ClipDragState | null>(clipDrag);
  clipDragRef.current = clipDrag;

  // Premiere-style clip drag
  const handleClipMouseDown = useCallback(
    (e: React.MouseEvent, clipId: string) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();

      const clip = clipMap.get(clipId);
      if (!clip) return;

      // Shift+Click: Toggle selection (add/remove from multi-selection)
      if (e.shiftKey) {
        selectClip(clipId, true); // addToSelection = true
        return; // Don't start drag on shift+click
      }

      // If clip is not selected, select only this clip
      // If clip is already selected (part of multi-selection), keep selection
      if (!selectedClipIds.has(clipId)) {
        selectClip(clipId);
      }

      const clipElement = e.currentTarget as HTMLElement;
      const clipRect = clipElement.getBoundingClientRect();
      const grabOffsetX = e.clientX - clipRect.left;

      const initialDrag: ClipDragState = {
        clipId,
        originalStartTime: clip.startTime,
        originalTrackId: clip.trackId,
        grabOffsetX,
        currentX: e.clientX,
        currentTrackId: clip.trackId,
        snappedTime: null,
        isSnapping: false,
        altKeyPressed: e.altKey, // Capture Alt state for independent drag
        forcingOverlap: false,
        dragStartTime: Date.now(), // Track when drag started for track-change delay
      };
      setClipDrag(initialDrag);
      clipDragRef.current = initialDrag;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const drag = clipDragRef.current;
        if (!drag || !trackLanesRef.current || !timelineRef.current) return;

        const lanesRect = trackLanesRef.current.getBoundingClientRect();
        const mouseY = moveEvent.clientY - lanesRect.top;

        // Only allow track changes after 300ms of dragging (prevents accidental track switches)
        const trackChangeAllowed = Date.now() - drag.dragStartTime >= 300;

        let currentY = 24;
        let newTrackId = drag.currentTrackId; // Keep current track by default
        for (const track of tracks) {
          if (mouseY >= currentY && mouseY < currentY + track.height) {
            // Only change to a different track if the delay has passed
            if (trackChangeAllowed || track.id === drag.originalTrackId) {
              newTrackId = track.id;
            }
            break;
          }
          currentY += track.height;
        }

        const rect = timelineRef.current.getBoundingClientRect();
        const x = moveEvent.clientX - rect.left + scrollX - drag.grabOffsetX;
        const rawTime = Math.max(0, pixelToTime(x));

        // Snapping with Alt-key toggle:
        // - When snapping enabled: snap by default, Alt temporarily disables
        // - When snapping disabled: don't snap, Alt temporarily enables
        const shouldSnap = snappingEnabled !== moveEvent.altKey;

        // First check for edge snapping (only if snapping should be active)
        const { startTime: snappedTime, snapped } = shouldSnap
          ? getSnappedPosition(drag.clipId, rawTime, newTrackId)
          : { startTime: rawTime, snapped: false };

        // Then apply resistance for overlap prevention
        const draggedClip = clipMap.get(drag.clipId);
        const clipDuration = draggedClip?.duration || 0;
        const baseTime = snapped ? snappedTime : rawTime;

        let { startTime: resistedTime, forcingOverlap } = getPositionWithResistance(
          drag.clipId,
          baseTime,
          newTrackId,
          clipDuration
        );

        // Also check linked clip (audio) for resistance on its track
        if (draggedClip?.linkedClipId && !moveEvent.altKey) {
          const linkedClip = clipMap.get(draggedClip.linkedClipId);
          if (linkedClip) {
            const timeDelta = resistedTime - draggedClip.startTime;
            const linkedNewTime = linkedClip.startTime + timeDelta;
            const linkedResult = getPositionWithResistance(
              linkedClip.id,
              linkedNewTime,
              linkedClip.trackId,
              linkedClip.duration
            );
            // If linked clip has more resistance, use that position
            const linkedTimeDelta = linkedResult.startTime - linkedClip.startTime;
            if (Math.abs(linkedTimeDelta) < Math.abs(timeDelta)) {
              // Linked clip is more constrained - adjust main clip position
              resistedTime = draggedClip.startTime + linkedTimeDelta;
              forcingOverlap = linkedResult.forcingOverlap || forcingOverlap;
            }
          }
        }

        const newDrag: ClipDragState = {
          ...drag,
          currentX: moveEvent.clientX,
          currentTrackId: newTrackId,
          snappedTime: resistedTime,
          isSnapping: snapped && !forcingOverlap,
          altKeyPressed: moveEvent.altKey, // Update Alt state dynamically
          forcingOverlap,
        };
        setClipDrag(newDrag);
        clipDragRef.current = newDrag;
      };

      const handleMouseUp = (upEvent: MouseEvent) => {
        const drag = clipDragRef.current;
        if (drag && timelineRef.current) {
          const rect = timelineRef.current.getBoundingClientRect();
          const x = upEvent.clientX - rect.left + scrollX - drag.grabOffsetX;
          const newStartTime = Math.max(0, pixelToTime(x));
          // Pass skipGroup (altKeyPressed) to moveClip for independent drag
          moveClip(drag.clipId, newStartTime, drag.currentTrackId, false, drag.altKeyPressed);
        }
        setClipDrag(null);
        clipDragRef.current = null;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [trackLanesRef, timelineRef, clipMap, tracks, scrollX, snappingEnabled, pixelToTime, selectClip, selectedClipIds, getSnappedPosition, getPositionWithResistance, moveClip]
  );

  // Handle double-click on clip - open composition if it's a nested comp
  const handleClipDoubleClick = useCallback(
    (e: React.MouseEvent, clipId: string) => {
      e.stopPropagation();
      e.preventDefault();

      const clip = clipMap.get(clipId);
      if (!clip) return;

      // If this clip is a composition, open it in a new tab and switch to it
      if (clip.isComposition && clip.compositionId) {
        console.log('[Timeline] Double-click on composition clip, opening:', clip.compositionId);
        openCompositionTab(clip.compositionId);
      }
    },
    [clipMap, openCompositionTab]
  );

  return {
    clipDrag,
    clipDragRef,
    handleClipMouseDown,
    handleClipDoubleClick,
  };
}
