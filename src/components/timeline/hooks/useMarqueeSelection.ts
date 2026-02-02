// useMarqueeSelection - Rectangle selection for clips and keyframes
// Extracted from Timeline.tsx for better maintainability

import { useState, useCallback, useEffect, useRef } from 'react';
import type { TimelineClip, TimelineTrack, AnimatableProperty } from '../../../types';
import type { MarqueeState, ClipDragState, ClipTrimState, MarkerDragState } from '../types';
import { PROPERTY_ROW_HEIGHT } from '../../../stores/timeline/constants';
import { useTimelineStore } from '../../../stores/timeline';

interface UseMarqueeSelectionProps {
  // Refs
  trackLanesRef: React.RefObject<HTMLDivElement | null>;

  // State
  scrollX: number;
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  selectedClipIds: Set<string>;
  selectedKeyframeIds: Set<string>;
  clipKeyframes: Map<string, Array<{ id: string; clipId: string; time: number; property: AnimatableProperty; value: number; easing: string }>>;

  // Drag states (to prevent marquee during other operations)
  clipDrag: ClipDragState | null;
  clipTrim: ClipTrimState | null;
  markerDrag: MarkerDragState | null;
  isDraggingPlayhead: boolean;

  // Actions
  selectClip: (clipId: string | null, addToSelection?: boolean) => void;
  selectKeyframe: (keyframeId: string, addToSelection?: boolean) => void;
  deselectAllKeyframes: () => void;

  // Helpers
  pixelToTime: (pixel: number) => number;
  isTrackExpanded: (trackId: string) => boolean;
  getExpandedTrackHeight: (trackId: string, baseHeight: number) => number;
}

interface UseMarqueeSelectionReturn {
  marquee: MarqueeState | null;
  handleMarqueeMouseDown: (e: React.MouseEvent) => void;
}

export function useMarqueeSelection({
  trackLanesRef,
  scrollX,
  clips,
  tracks,
  selectedClipIds,
  selectedKeyframeIds,
  clipKeyframes,
  clipDrag,
  clipTrim,
  markerDrag,
  isDraggingPlayhead,
  selectClip,
  selectKeyframe,
  deselectAllKeyframes,
  pixelToTime,
  isTrackExpanded,
  getExpandedTrackHeight,
}: UseMarqueeSelectionProps): UseMarqueeSelectionReturn {
  const [marquee, setMarquee] = useState<MarqueeState | null>(null);
  const marqueeRef = useRef(marquee);
  marqueeRef.current = marquee;

  // Helper: Calculate which clips intersect with a rectangle
  const getClipsInRect = useCallback(
    (left: number, right: number, top: number, bottom: number): Set<string> => {
      const result = new Set<string>();

      // Convert pixel bounds to time
      const startTime = pixelToTime(left);
      const endTime = pixelToTime(right);

      // Calculate which tracks are covered by the rectangle
      let currentY = 0;
      const coveredTrackIds = new Set<string>();

      for (const track of tracks) {
        const trackHeight = getExpandedTrackHeight(track.id, track.height);
        const trackTop = currentY;
        const trackBottom = currentY + trackHeight;

        // Check if rectangle overlaps with this track
        if (bottom > trackTop && top < trackBottom) {
          coveredTrackIds.add(track.id);
        }

        currentY += trackHeight;
      }

      // Find all clips that intersect with the selection rectangle
      for (const clip of clips) {
        // Check if clip's track is in covered tracks
        if (!coveredTrackIds.has(clip.trackId)) continue;

        // Check if clip's time range overlaps with selection time range
        const clipEnd = clip.startTime + clip.duration;
        if (clip.startTime < endTime && clipEnd > startTime) {
          result.add(clip.id);
        }
      }

      return result;
    },
    [pixelToTime, tracks, clips, getExpandedTrackHeight]
  );

  // Helper: Calculate which keyframes intersect with a rectangle
  const getKeyframesInRect = useCallback(
    (left: number, right: number, top: number, bottom: number): Set<string> => {
      const result = new Set<string>();

      // Convert pixel bounds to time
      const startTime = pixelToTime(left);
      const endTime = pixelToTime(right);

      // Property order for calculating Y position within expanded track
      const propertyOrder = ['opacity', 'position.x', 'position.y', 'position.z', 'scale.x', 'scale.y', 'rotation.x', 'rotation.y', 'rotation.z'];

      // Calculate track positions
      let currentY = 0;

      for (const track of tracks) {
        if (track.type !== 'video') {
          currentY += track.height;
          continue;
        }

        const baseHeight = track.height;
        const isExpanded = isTrackExpanded(track.id);

        // Get the selected clip in this track for keyframe display
        const trackClips = clips.filter(c => c.trackId === track.id);
        const selectedTrackClip = trackClips.find(c => selectedClipIds.has(c.id));

        if (!isExpanded || !selectedTrackClip) {
          currentY += baseHeight;
          continue;
        }

        // Get keyframes for the selected clip
        const keyframes = clipKeyframes.get(selectedTrackClip.id) || [];
        if (keyframes.length === 0) {
          currentY += baseHeight;
          continue;
        }

        // Get unique properties with keyframes in sorted order
        const uniqueProps = [...new Set(keyframes.map(k => k.property))];
        const sortedProps = uniqueProps.sort((a, b) => {
          const aIdx = propertyOrder.indexOf(a);
          const bIdx = propertyOrder.indexOf(b);
          if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
          if (aIdx !== -1) return -1;
          if (bIdx !== -1) return 1;
          return a.localeCompare(b);
        });

        // Calculate Y position for each property row
        // Property rows start after the base track height
        const propertyRowStart = currentY + baseHeight;

        for (let propIndex = 0; propIndex < sortedProps.length; propIndex++) {
          const prop = sortedProps[propIndex];
          const rowTop = propertyRowStart + propIndex * PROPERTY_ROW_HEIGHT;
          const rowBottom = rowTop + PROPERTY_ROW_HEIGHT;

          // Check if this row is within the marquee Y bounds
          if (bottom > rowTop && top < rowBottom) {
            // Find keyframes in this property that are within the time range
            const propKeyframes = keyframes.filter(k => k.property === prop);

            for (const kf of propKeyframes) {
              const absTime = selectedTrackClip.startTime + kf.time;
              // Check if keyframe time is within selection
              if (absTime >= startTime && absTime <= endTime) {
                result.add(kf.id);
              }
            }
          }
        }

        // Move to next track (include expanded height)
        currentY += getExpandedTrackHeight(track.id, baseHeight);
      }

      return result;
    },
    [pixelToTime, tracks, clips, selectedClipIds, clipKeyframes, isTrackExpanded, getExpandedTrackHeight]
  );

  // Marquee selection: mouse down on empty area starts selection
  const handleMarqueeMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Only start marquee on left mouse button and empty area
      if (e.button !== 0) return;
      // Don't start if clicking on a clip or interactive element
      const target = e.target as HTMLElement;
      if (
        target.closest('.timeline-clip') ||
        target.closest('.playhead') ||
        target.closest('.in-out-marker') ||
        target.closest('.trim-handle') ||
        target.closest('.fade-handle') ||
        target.closest('.track-header') ||
        target.closest('.keyframe-diamond')
      ) {
        return;
      }

      // Don't start if any other drag operation is in progress
      if (clipDrag || clipTrim || markerDrag || isDraggingPlayhead) {
        return;
      }

      const rect = trackLanesRef.current?.getBoundingClientRect();
      if (!rect) return;

      const startX = e.clientX - rect.left + scrollX;
      const startY = e.clientY - rect.top;

      // Check if we're starting in the keyframe area
      const isInKeyframeArea = target.closest('.keyframe-track-row') !== null;

      // Clear selection unless shift is held
      // But if in keyframe area, keep clip selection to prevent keyframe rows from collapsing
      if (!e.shiftKey) {
        if (!isInKeyframeArea) {
          selectClip(null, false);
        }
        deselectAllKeyframes();
      }

      // Store the initial selection (for shift+drag to add to it)
      // If in keyframe area, always preserve current clip selection
      const initialSelection = (e.shiftKey || isInKeyframeArea) ? new Set(selectedClipIds) : new Set<string>();
      const initialKeyframeSelection = e.shiftKey ? new Set(selectedKeyframeIds) : new Set<string>();

      setMarquee({
        startX,
        startY,
        currentX: startX,
        currentY: startY,
        startScrollX: scrollX,
        initialSelection,
        initialKeyframeSelection,
      });

      e.preventDefault();
    },
    [trackLanesRef, clipDrag, clipTrim, markerDrag, isDraggingPlayhead, scrollX, selectClip, selectedClipIds, deselectAllKeyframes, selectedKeyframeIds]
  );

  // Marquee selection: mouse move and mouse up handlers
  useEffect(() => {
    if (!marquee) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = trackLanesRef.current?.getBoundingClientRect();
      if (!rect) return;

      const currentX = e.clientX - rect.left + scrollX;
      const currentY = e.clientY - rect.top;

      // Update marquee position
      setMarquee((prev) =>
        prev ? { ...prev, currentX, currentY } : null
      );

      // Calculate rectangle bounds
      const m = marqueeRef.current;
      if (!m) return;

      const left = Math.min(m.startX, currentX);
      const right = Math.max(m.startX, currentX);
      const top = Math.min(m.startY, currentY);
      const bottom = Math.max(m.startY, currentY);

      // Get clips that intersect with the rectangle
      const intersectingClips = getClipsInRect(left, right, top, bottom);

      // Combine with initial selection (for shift+drag)
      const newClipSelection = new Set([...m.initialSelection, ...intersectingClips]);

      // Update clip selection
      const currentClipSelection = useTimelineStore.getState().selectedClipIds;
      const clipSelectionChanged =
        newClipSelection.size !== currentClipSelection.size ||
        [...newClipSelection].some(id => !currentClipSelection.has(id));

      if (clipSelectionChanged) {
        selectClip(null, false);
        for (const clipId of newClipSelection) {
          selectClip(clipId, true);
        }
      }

      // Get keyframes that intersect with the rectangle
      const intersectingKeyframes = getKeyframesInRect(left, right, top, bottom);

      // Combine with initial keyframe selection (for shift+drag)
      const newKeyframeSelection = new Set([...m.initialKeyframeSelection, ...intersectingKeyframes]);

      // Update keyframe selection
      const currentKeyframeSelection = useTimelineStore.getState().selectedKeyframeIds;
      const keyframeSelectionChanged =
        newKeyframeSelection.size !== currentKeyframeSelection.size ||
        [...newKeyframeSelection].some(id => !currentKeyframeSelection.has(id));

      if (keyframeSelectionChanged) {
        deselectAllKeyframes();
        for (const kfId of newKeyframeSelection) {
          selectKeyframe(kfId, true);
        }
      }
    };

    const handleMouseUp = () => {
      // Selection is already applied live, just clear marquee
      setMarquee(null);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [marquee, trackLanesRef, scrollX, selectClip, getClipsInRect, getKeyframesInRect, selectKeyframe, deselectAllKeyframes]);

  return {
    marquee,
    handleMarqueeMouseDown,
  };
}
