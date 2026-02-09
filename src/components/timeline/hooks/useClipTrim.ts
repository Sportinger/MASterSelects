// useClipTrim - Clip edge trimming (in/out point adjustment)
// Extracted from Timeline.tsx for better maintainability

import { useState, useCallback, useRef } from 'react';
import type { TimelineClip } from '../../../types';
import type { ClipTrimState } from '../types';

interface UseClipTrimProps {
  // Clip data
  clipMap: Map<string, TimelineClip>;

  // Actions
  selectClip: (clipId: string | null, addToSelection?: boolean) => void;
  trimClip: (clipId: string, inPoint: number, outPoint: number) => void;
  moveClip: (clipId: string, newStartTime: number, trackId: string, skipLinked?: boolean) => void;

  // Helpers
  pixelToTime: (pixel: number) => number;
}

interface UseClipTrimReturn {
  clipTrim: ClipTrimState | null;
  clipTrimRef: React.MutableRefObject<ClipTrimState | null>;
  handleTrimStart: (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => void;
}

export function useClipTrim({
  clipMap,
  selectClip,
  trimClip,
  moveClip,
  pixelToTime,
}: UseClipTrimProps): UseClipTrimReturn {
  const [clipTrim, setClipTrim] = useState<ClipTrimState | null>(null);
  const clipTrimRef = useRef<ClipTrimState | null>(clipTrim);
  clipTrimRef.current = clipTrim;

  const handleTrimStart = useCallback(
    (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => {
      e.stopPropagation();
      e.preventDefault();

      const clip = clipMap.get(clipId);
      if (!clip) return;

      selectClip(clipId);

      const initialTrim: ClipTrimState = {
        clipId,
        edge,
        originalStartTime: clip.startTime,
        originalDuration: clip.duration,
        originalInPoint: clip.inPoint,
        originalOutPoint: clip.outPoint,
        startX: e.clientX,
        currentX: e.clientX,
        altKey: e.altKey,
      };
      setClipTrim(initialTrim);
      clipTrimRef.current = initialTrim;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const newTrim = clipTrimRef.current;
        if (!newTrim) return;
        const updated = {
          ...newTrim,
          currentX: moveEvent.clientX,
          altKey: moveEvent.altKey,
        };
        setClipTrim(updated);
        clipTrimRef.current = updated;
      };

      const handleMouseUp = () => {
        const trim = clipTrimRef.current;
        if (!trim) {
          setClipTrim(null);
          clipTrimRef.current = null;
          document.removeEventListener('mousemove', handleMouseMove);
          document.removeEventListener('mouseup', handleMouseUp);
          return;
        }

        const clipToTrim = clipMap.get(trim.clipId);
        if (!clipToTrim) {
          setClipTrim(null);
          clipTrimRef.current = null;
          document.removeEventListener('mousemove', handleMouseMove);
          document.removeEventListener('mouseup', handleMouseUp);
          return;
        }

        const deltaX = trim.currentX - trim.startX;
        const deltaTime = pixelToTime(deltaX);

        // Text, image, and solid clips can be extended infinitely (no natural duration limit)
        const sourceType = clipToTrim.source?.type;
        const isInfiniteClip = sourceType === 'text' || sourceType === 'image' || sourceType === 'solid';
        const maxDuration = isInfiniteClip
          ? Number.MAX_SAFE_INTEGER
          : (clipToTrim.source?.naturalDuration || clipToTrim.duration);

        let newStartTime = trim.originalStartTime;
        let newInPoint = trim.originalInPoint;
        let newOutPoint = trim.originalOutPoint;

        if (trim.edge === 'left') {
          const maxTrim = trim.originalDuration - 0.1;
          // For infinite clips (text/image), allow extending left up to timeline start (0)
          // For video/audio, limit to existing in-point (can't reveal non-existent media)
          const minTrim = isInfiniteClip
            ? -trim.originalStartTime  // Can extend left until startTime reaches 0
            : -trim.originalInPoint;
          const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));
          newStartTime = trim.originalStartTime + clampedDelta;
          newInPoint = trim.originalInPoint + clampedDelta;
        } else {
          const maxExtend = maxDuration - trim.originalOutPoint;
          const minTrim = -(trim.originalDuration - 0.1);
          const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
          newOutPoint = trim.originalOutPoint + clampedDelta;
        }

        trimClip(clipToTrim.id, newInPoint, newOutPoint);
        if (trim.edge === 'left') {
          moveClip(clipToTrim.id, Math.max(0, newStartTime), clipToTrim.trackId, trim.altKey);
        }

        // Handle linked clip (audio/video pair)
        if (!trim.altKey && clipToTrim.linkedClipId) {
          const linkedClip = clipMap.get(clipToTrim.linkedClipId);
          if (linkedClip) {
            const linkedMaxDuration =
              linkedClip.source?.naturalDuration || linkedClip.duration;
            if (trim.edge === 'left') {
              const linkedNewInPoint = Math.max(
                0,
                Math.min(linkedMaxDuration - 0.1, newInPoint)
              );
              trimClip(linkedClip.id, linkedNewInPoint, linkedClip.outPoint);
              moveClip(
                linkedClip.id,
                Math.max(0, newStartTime),
                linkedClip.trackId,
                true
              );
            } else {
              const linkedNewOutPoint = Math.max(
                0.1,
                Math.min(linkedMaxDuration, newOutPoint)
              );
              trimClip(linkedClip.id, linkedClip.inPoint, linkedNewOutPoint);
            }
          }
        }

        setClipTrim(null);
        clipTrimRef.current = null;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [clipMap, pixelToTime, selectClip, trimClip, moveClip]
  );

  return {
    clipTrim,
    clipTrimRef,
    handleTrimStart,
  };
}
