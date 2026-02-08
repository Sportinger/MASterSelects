// useClipDrag - Premiere-style clip dragging with snapping
// Extracted from Timeline.tsx for better maintainability

import { useState, useCallback, useRef } from 'react';
import type { TimelineClip, TimelineTrack } from '../../../types';
import type { ClipDragState } from '../types';
import { Logger } from '../../../services/logger';

const log = Logger.create('useClipDrag');

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
  selectClip: (clipId: string | null, addToSelection?: boolean, setPrimaryOnly?: boolean) => void;
  moveClip: (clipId: string, newStartTime: number, trackId: string, skipLinked?: boolean, skipGroup?: boolean, skipTrim?: boolean, excludeClipIds?: string[]) => void;
  openCompositionTab: (compositionId: string) => void;

  // Helpers
  pixelToTime: (pixel: number) => number;
  getSnappedPosition: (clipId: string, rawTime: number, trackId: string) => { startTime: number; snapped: boolean };
  getPositionWithResistance: (clipId: string, rawTime: number, trackId: string, duration: number, zoom?: number, excludeClipIds?: string[]) => { startTime: number; forcingOverlap: boolean; noFreeSpace?: boolean };
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

  // Keep refs to current values for use in event handlers (avoid stale closures)
  const selectedClipIdsRef = useRef<Set<string>>(selectedClipIds);
  selectedClipIdsRef.current = selectedClipIds;
  const clipMapRef = useRef<Map<string, TimelineClip>>(clipMap);
  clipMapRef.current = clipMap;

  // Premiere-style clip drag
  const handleClipMouseDown = useCallback(
    (e: React.MouseEvent, clipId: string) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();

      // Use ref for current clipMap to avoid stale closure
      const currentClipMap = clipMapRef.current;
      const clip = currentClipMap.get(clipId);
      if (!clip) return;

      // Shift+Click: Toggle selection (add/remove from multi-selection)
      if (e.shiftKey) {
        selectClip(clipId, true); // addToSelection = true
        return; // Don't start drag on shift+click
      }

      // Use ref for current selection to avoid stale closure
      const currentSelectedIds = selectedClipIdsRef.current;

      // If clip is not selected, select it (+ linked clip)
      // If already selected, keep selection but update primary for Properties panel
      if (!currentSelectedIds.has(clipId)) {
        selectClip(clipId);
      } else {
        selectClip(clipId, false, true); // setPrimaryOnly: keep existing selection, just update primary
      }

      // Capture other selected clip IDs for multi-select drag (re-read after potential selection change)
      const finalSelectedIds = selectedClipIdsRef.current;
      const otherSelectedIds = finalSelectedIds.size > 1 && finalSelectedIds.has(clipId)
        ? [...finalSelectedIds].filter(id => id !== clipId)
        : [];

      const clipElement = e.currentTarget as HTMLElement;
      const clipRect = clipElement.getBoundingClientRect();
      const grabOffsetX = e.clientX - clipRect.left;
      const lanesRectInit = trackLanesRef.current?.getBoundingClientRect();
      const grabY = lanesRectInit ? e.clientY - lanesRectInit.top : 0;

      const initialDrag: ClipDragState = {
        clipId,
        originalStartTime: clip.startTime,
        originalTrackId: clip.trackId,
        grabOffsetX,
        grabY,
        currentX: e.clientX,
        currentTrackId: clip.trackId,
        snappedTime: null,
        isSnapping: false,
        altKeyPressed: e.altKey, // Capture Alt state for independent drag
        forcingOverlap: false,
        dragStartTime: Date.now(), // Track when drag started for track-change delay
        // Multi-select support
        multiSelectClipIds: otherSelectedIds.length > 0 ? otherSelectedIds : undefined,
        multiSelectTimeDelta: 0,
      };
      setClipDrag(initialDrag);
      clipDragRef.current = initialDrag;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const drag = clipDragRef.current;
        if (!drag || !trackLanesRef.current || !timelineRef.current) return;

        const lanesRect = trackLanesRef.current.getBoundingClientRect();
        const mouseY = moveEvent.clientY - lanesRect.top;

        // Track change requires BOTH a time delay (300ms) AND a vertical distance (20px from grab point)
        const TRACK_CHANGE_DELAY_MS = 300;
        const TRACK_CHANGE_RESISTANCE_PX = 20;
        const trackChangeAllowed = Date.now() - drag.dragStartTime >= TRACK_CHANGE_DELAY_MS
          && Math.abs(mouseY - drag.grabY) >= TRACK_CHANGE_RESISTANCE_PX;

        let currentY = 24;
        let newTrackId = drag.currentTrackId; // Keep current track by default
        for (const track of tracks) {
          if (mouseY >= currentY && mouseY < currentY + track.height) {
            // Only change to a different track if both delay and distance thresholds are met
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

        // Get all selected clip IDs AND their linked clips (for excluding from collision detection)
        // This ensures video+audio pairs move as a unit
        const allSelectedIds = drag.multiSelectClipIds
          ? [drag.clipId, ...drag.multiSelectClipIds]
          : [drag.clipId];

        // Also collect all linked clips of selected clips
        const allExcludedIds = [...allSelectedIds];
        for (const selId of allSelectedIds) {
          const selClip = clipMap.get(selId);
          if (selClip?.linkedClipId && !allExcludedIds.includes(selClip.linkedClipId)) {
            allExcludedIds.push(selClip.linkedClipId);
          }
        }

        // Check primary clip with all related clips excluded
        let { startTime: resistedTime, forcingOverlap, noFreeSpace } = getPositionWithResistance(
          drag.clipId,
          baseTime,
          newTrackId,
          clipDuration,
          undefined, // zoom
          allExcludedIds // exclude all selected clips and their linked clips
        );

        // If no free space on target track (cross-track move), try other tracks of same type
        if (noFreeSpace && newTrackId !== drag.originalTrackId) {
          const targetTrack = tracks.find(t => t.id === newTrackId);
          if (targetTrack) {
            const altTracks = tracks.filter(t =>
              t.type === targetTrack.type && t.id !== newTrackId && t.id !== drag.originalTrackId
            );
            for (const alt of altTracks) {
              const altResult = getPositionWithResistance(
                drag.clipId, baseTime, alt.id, clipDuration, undefined, allExcludedIds
              );
              if (!altResult.noFreeSpace) {
                newTrackId = alt.id;
                resistedTime = altResult.startTime;
                forcingOverlap = altResult.forcingOverlap;
                break;
              }
            }
          }
        }

        let timeDelta = resistedTime - (draggedClip?.startTime ?? drag.originalStartTime);

        // Check ALL linked clips for resistance (not just the primary dragged clip's linked clip)
        if (!moveEvent.altKey) {
          for (const selId of allSelectedIds) {
            const selClip = clipMap.get(selId);
            if (!selClip?.linkedClipId) continue;

            const linkedClip = clipMap.get(selClip.linkedClipId);
            if (!linkedClip) continue;

            const linkedNewTime = linkedClip.startTime + timeDelta;
            const linkedResult = getPositionWithResistance(
              linkedClip.id,
              linkedNewTime,
              linkedClip.trackId,
              linkedClip.duration,
              undefined,
              allExcludedIds
            );
            // If linked clip has more resistance, use that position
            const linkedTimeDelta = linkedResult.startTime - linkedClip.startTime;
            if (Math.abs(linkedTimeDelta) < Math.abs(timeDelta)) {
              // Linked clip is more constrained - adjust for the whole group
              timeDelta = linkedTimeDelta;
              resistedTime = (draggedClip?.startTime ?? drag.originalStartTime) + timeDelta;
              forcingOverlap = linkedResult.forcingOverlap || forcingOverlap;
            }
          }
        }

        // For multi-select: check all other selected clips for resistance too
        // The whole group should stop if ANY clip hits an obstacle
        if (drag.multiSelectClipIds?.length && !forcingOverlap) {
          for (const selectedId of drag.multiSelectClipIds) {
            const selectedClip = clipMap.get(selectedId);
            if (!selectedClip) continue;

            const selectedNewTime = selectedClip.startTime + timeDelta;
            const selectedResult = getPositionWithResistance(
              selectedClip.id,
              selectedNewTime,
              selectedClip.trackId,
              selectedClip.duration,
              undefined,
              allExcludedIds
            );

            // If this clip is more constrained, reduce the timeDelta for the whole group
            const selectedActualDelta = selectedResult.startTime - selectedClip.startTime;
            if (Math.abs(selectedActualDelta) < Math.abs(timeDelta)) {
              timeDelta = selectedActualDelta;
              resistedTime = (draggedClip?.startTime ?? drag.originalStartTime) + timeDelta;
              forcingOverlap = selectedResult.forcingOverlap || forcingOverlap;
            }
          }
        }

        // Calculate time delta for multi-select preview
        const multiSelectTimeDelta = drag.multiSelectClipIds?.length
          ? timeDelta
          : undefined;

        const newDrag: ClipDragState = {
          ...drag,
          currentX: moveEvent.clientX,
          currentTrackId: newTrackId,
          snappedTime: resistedTime,
          isSnapping: snapped && !forcingOverlap,
          altKeyPressed: moveEvent.altKey, // Update Alt state dynamically
          forcingOverlap,
          multiSelectTimeDelta,
        };
        setClipDrag(newDrag);
        clipDragRef.current = newDrag;
      };

      const handleMouseUp = (upEvent: MouseEvent) => {
        const drag = clipDragRef.current;
        if (drag && timelineRef.current) {
          // Use refs to get current values (avoid stale closures)
          const currentSelectedIds = selectedClipIdsRef.current;
          const currentClipMap = clipMapRef.current;

          // For multi-select: use the already-calculated snappedTime and timeDelta from drag state
          // This ensures we use the same constrained position shown in the preview
          const isMultiSelect = currentSelectedIds.size > 1 && currentSelectedIds.has(drag.clipId);

          let finalStartTime: number;
          let timeDelta: number;

          if (isMultiSelect && drag.snappedTime !== null && drag.multiSelectTimeDelta !== undefined) {
            // Use the pre-calculated constrained position from the drag preview
            finalStartTime = drag.snappedTime;
            timeDelta = drag.multiSelectTimeDelta;
          } else {
            // Single clip or no snapped position - calculate from mouse
            const rect = timelineRef.current.getBoundingClientRect();
            const x = upEvent.clientX - rect.left + scrollX - drag.grabOffsetX;
            finalStartTime = Math.max(0, pixelToTime(x));
            const draggedClip = currentClipMap.get(drag.clipId);
            timeDelta = finalStartTime - (draggedClip?.startTime ?? drag.originalStartTime);
          }

          log.debug('Multi-select drag check', {
            selectedCount: currentSelectedIds.size,
            selectedIds: [...currentSelectedIds],
            dragClipId: drag.clipId,
            hasDragClip: currentSelectedIds.has(drag.clipId),
            timeDelta,
            finalStartTime,
            usedSnappedTime: isMultiSelect && drag.snappedTime !== null,
          });

          // If multiple clips are selected, move them all by the same delta
          if (isMultiSelect) {
            log.debug('Moving multiple clips', { count: currentSelectedIds.size });

            // Collect all clips that should be excluded from resistance (selected + linked)
            const allExcludedIds: string[] = [...currentSelectedIds];
            for (const selId of currentSelectedIds) {
              const selClip = currentClipMap.get(selId);
              if (selClip?.linkedClipId && !allExcludedIds.includes(selClip.linkedClipId)) {
                allExcludedIds.push(selClip.linkedClipId);
              }
            }

            // Track which clips we've already moved (to avoid double-moving linked clips)
            const movedClipIds = new Set<string>();

            // Move the dragged clip first (this handles snapping)
            // skipLinked depends on whether linked clip is also selected
            const draggedClip = currentClipMap.get(drag.clipId);
            const draggedLinkedInSelection = !!(draggedClip?.linkedClipId && currentSelectedIds.has(draggedClip.linkedClipId));
            moveClip(drag.clipId, finalStartTime, drag.currentTrackId, draggedLinkedInSelection, drag.altKeyPressed, true, allExcludedIds);
            movedClipIds.add(drag.clipId);
            // If linked clip was moved via skipLinked=false, mark it as moved
            if (draggedClip?.linkedClipId && !draggedLinkedInSelection && !drag.altKeyPressed) {
              movedClipIds.add(draggedClip.linkedClipId);
            }

            // Move other selected clips by the same delta
            for (const selectedId of currentSelectedIds) {
              if (movedClipIds.has(selectedId)) continue; // Skip already-moved clips
              const selectedClip = currentClipMap.get(selectedId);
              if (selectedClip) {
                const newTime = Math.max(0, selectedClip.startTime + timeDelta);
                // If linked clip is also selected, skip it (will be moved in its own iteration)
                // If linked clip is NOT selected, move it with this clip (skipLinked=false)
                const linkedInSelection = !!(selectedClip.linkedClipId && currentSelectedIds.has(selectedClip.linkedClipId));
                moveClip(selectedId, newTime, selectedClip.trackId, linkedInSelection, true, true, allExcludedIds); // skipGroup, skipTrim always true
                movedClipIds.add(selectedId);
                // If linked clip was moved via skipLinked=false, mark it as moved
                if (selectedClip.linkedClipId && !linkedInSelection) {
                  movedClipIds.add(selectedClip.linkedClipId);
                }
              }
            }
          } else {
            // Single clip drag - normal behavior
            moveClip(drag.clipId, finalStartTime, drag.currentTrackId, false, drag.altKeyPressed);
          }
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
        log.debug('Double-click on composition clip, opening:', clip.compositionId);
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
