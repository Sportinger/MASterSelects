// Timeline positioning utilities - snap, overlap, resistance, trimming
// Extracted from index.ts for maintainability

import type { SliceCreator, TimelineClip, TimelineUtils } from './types';
import { SNAP_THRESHOLD_SECONDS } from './constants';

type PositioningUtils = Pick<
  TimelineUtils,
  'getSnappedPosition' | 'findNonOverlappingPosition' | 'getPositionWithResistance' | 'trimOverlappingClips'
>;

export const createPositioningUtils: SliceCreator<PositioningUtils> = (set, get) => ({
  getSnappedPosition: (clipId: string, desiredStartTime: number, trackId: string) => {
    const { clips } = get();
    const movingClip = clips.find(c => c.id === clipId);
    if (!movingClip) return { startTime: desiredStartTime, snapped: false };

    // Note: Caller decides whether to call this based on snappingEnabled + Alt key
    // This function always attempts to snap when called

    const clipDuration = movingClip.duration;
    const desiredEndTime = desiredStartTime + clipDuration;

    // Get other clips on the same track (excluding the moving clip and its linked clip)
    const otherClips = clips.filter(c =>
      c.trackId === trackId &&
      c.id !== clipId &&
      c.id !== movingClip.linkedClipId &&
      c.linkedClipId !== clipId
    );

    let snappedStart = desiredStartTime;
    let snapped = false;
    let minSnapDistance = SNAP_THRESHOLD_SECONDS;

    // Check snap points
    for (const clip of otherClips) {
      const clipEnd = clip.startTime + clip.duration;

      // Snap start of moving clip to end of other clip
      const distToEnd = Math.abs(desiredStartTime - clipEnd);
      if (distToEnd < minSnapDistance) {
        snappedStart = clipEnd;
        minSnapDistance = distToEnd;
        snapped = true;
      }

      // Snap start of moving clip to start of other clip
      const distToStart = Math.abs(desiredStartTime - clip.startTime);
      if (distToStart < minSnapDistance) {
        snappedStart = clip.startTime;
        minSnapDistance = distToStart;
        snapped = true;
      }

      // Snap end of moving clip to start of other clip
      const distEndToStart = Math.abs(desiredEndTime - clip.startTime);
      if (distEndToStart < minSnapDistance) {
        snappedStart = clip.startTime - clipDuration;
        minSnapDistance = distEndToStart;
        snapped = true;
      }

      // Snap end of moving clip to end of other clip
      const distEndToEnd = Math.abs(desiredEndTime - clipEnd);
      if (distEndToEnd < minSnapDistance) {
        snappedStart = clipEnd - clipDuration;
        minSnapDistance = distEndToEnd;
        snapped = true;
      }
    }

    // Also snap to timeline start (0)
    if (Math.abs(desiredStartTime) < SNAP_THRESHOLD_SECONDS) {
      snappedStart = 0;
      snapped = true;
    }

    return { startTime: Math.max(0, snappedStart), snapped };
  },

  findNonOverlappingPosition: (clipId: string, desiredStartTime: number, trackId: string, duration: number) => {
    const { clips } = get();
    const movingClip = clips.find(c => c.id === clipId);

    // Get other clips on the same track (excluding the moving clip and its linked clip)
    const otherClips = clips.filter(c =>
      c.trackId === trackId &&
      c.id !== clipId &&
      (movingClip ? c.id !== movingClip.linkedClipId && c.linkedClipId !== clipId : true)
    ).sort((a, b) => a.startTime - b.startTime);

    const desiredEndTime = desiredStartTime + duration;

    // Check if desired position overlaps with any clip
    let overlappingClip: TimelineClip | null = null;
    for (const clip of otherClips) {
      const clipEnd = clip.startTime + clip.duration;
      // Check if time ranges overlap
      if (!(desiredEndTime <= clip.startTime || desiredStartTime >= clipEnd)) {
        overlappingClip = clip;
        break;
      }
    }

    // If no overlap, use desired position
    if (!overlappingClip) {
      return Math.max(0, desiredStartTime);
    }

    // There's an overlap - push clip to the nearest edge
    const overlappingEnd = overlappingClip.startTime + overlappingClip.duration;

    // Check which side is closer
    const distToStart = Math.abs(desiredStartTime - overlappingClip.startTime);
    const distToEnd = Math.abs(desiredStartTime - overlappingEnd);

    if (distToStart < distToEnd) {
      // Push to left side (end at overlapping clip's start)
      const newStart = overlappingClip.startTime - duration;

      // Check if this position overlaps with another clip
      const wouldOverlap = otherClips.some(c => {
        if (c.id === overlappingClip!.id) return false;
        const cEnd = c.startTime + c.duration;
        const newEnd = newStart + duration;
        return !(newEnd <= c.startTime || newStart >= cEnd);
      });

      if (!wouldOverlap && newStart >= 0) {
        return newStart;
      }
    }

    // Push to right side (start at overlapping clip's end)
    const newStart = overlappingEnd;

    // Check if this position overlaps with another clip
    const wouldOverlap = otherClips.some(c => {
      if (c.id === overlappingClip!.id) return false;
      const cEnd = c.startTime + c.duration;
      const newEnd = newStart + duration;
      return !(newEnd <= c.startTime || newStart >= cEnd);
    });

    if (!wouldOverlap) {
      return newStart;
    }

    // As a fallback, return the desired position (shouldn't happen often)
    return Math.max(0, desiredStartTime);
  },

  // Apply magnetic resistance at clip edges during drag
  // Returns position with resistance applied, and whether user has "broken through" to force overlap
  // Uses PIXEL-based resistance so it works regardless of clip duration
  getPositionWithResistance: (clipId: string, desiredStartTime: number, trackId: string, duration: number, _zoom?: number, excludeClipIds?: string[]) => {
    const { clips } = get();
    const movingClip = clips.find(c => c.id === clipId);
    const excludeSet = new Set(excludeClipIds || []);
    const isTrackChange = movingClip ? movingClip.trackId !== trackId : false;

    // Get other clips on the TARGET track (excluding the moving clip, its linked clip, and any excluded clips)
    const otherClips = clips.filter(c =>
      c.trackId === trackId &&
      c.id !== clipId &&
      !excludeSet.has(c.id) &&
      (movingClip ? c.id !== movingClip.linkedClipId && c.linkedClipId !== clipId : true)
    ).sort((a, b) => a.startTime - b.startTime);

    const desiredEndTime = desiredStartTime + duration;

    // Find the clip that would be overlapped
    let overlappingClip: TimelineClip | null = null;
    for (const clip of otherClips) {
      const clipEnd = clip.startTime + clip.duration;
      if (!(desiredEndTime <= clip.startTime || desiredStartTime >= clipEnd)) {
        overlappingClip = clip;
        break;
      }
    }

    // No overlap - return desired position
    if (!overlappingClip) {
      return { startTime: Math.max(0, desiredStartTime), forcingOverlap: false };
    }

    // Cross-track moves: never allow overlap, find closest free position
    if (isTrackChange) {
      // Generate candidate positions at edges of every clip on the track (+ timeline start)
      const candidates: number[] = [0];
      for (const c of otherClips) {
        candidates.push(c.startTime - duration); // right before clip
        candidates.push(c.startTime + c.duration); // right after clip
      }

      let bestPos: number | null = null;
      let bestDist = Infinity;
      for (const pos of candidates) {
        if (pos < 0) continue;
        const posEnd = pos + duration;
        const posOverlaps = otherClips.some(c => {
          const cEnd = c.startTime + c.duration;
          return !(posEnd <= c.startTime || pos >= cEnd);
        });
        if (!posOverlaps) {
          const dist = Math.abs(pos - desiredStartTime);
          if (dist < bestDist) {
            bestDist = dist;
            bestPos = pos;
          }
        }
      }

      if (bestPos !== null) {
        return { startTime: bestPos, forcingOverlap: false };
      }

      // No valid position found — track is fully packed
      return { startTime: Math.max(0, desiredStartTime), forcingOverlap: false, noFreeSpace: true };
    }

    // Same-track moves: free movement, overlap trimmed on drop
    return { startTime: Math.max(0, desiredStartTime), forcingOverlap: true };
  },

  // Trim any clips that the placed clip overlaps with
  trimOverlappingClips: (clipId: string, startTime: number, trackId: string, duration: number) => {
    const { clips, invalidateCache } = get();
    const movingClip = clips.find(c => c.id === clipId);

    // Get other clips on the same track (excluding the moving clip and its linked clip)
    const otherClips = clips.filter(c =>
      c.trackId === trackId &&
      c.id !== clipId &&
      (movingClip ? c.id !== movingClip.linkedClipId && c.linkedClipId !== clipId : true)
    );

    const endTime = startTime + duration;
    const clipsToModify: { id: string; action: 'trim-start' | 'trim-end' | 'delete' | 'split'; trimAmount?: number; splitTime?: number }[] = [];

    for (const clip of otherClips) {
      const clipEnd = clip.startTime + clip.duration;

      // Check if this clip overlaps with the placed clip
      if (!(endTime <= clip.startTime || startTime >= clipEnd)) {
        // There's overlap - determine how to handle it

        // Case 1: Placed clip completely covers this clip -> delete it
        if (startTime <= clip.startTime && endTime >= clipEnd) {
          clipsToModify.push({ id: clip.id, action: 'delete' });
        }
        // Case 2: Placed clip covers the start of this clip -> trim start
        else if (startTime <= clip.startTime && endTime < clipEnd) {
          const trimAmount = endTime - clip.startTime;
          clipsToModify.push({ id: clip.id, action: 'trim-start', trimAmount });
        }
        // Case 3: Placed clip covers the end of this clip -> trim end
        else if (startTime > clip.startTime && endTime >= clipEnd) {
          const trimAmount = clipEnd - startTime;
          clipsToModify.push({ id: clip.id, action: 'trim-end', trimAmount });
        }
        // Case 4: Placed clip is in the middle of this clip -> split and trim
        else if (startTime > clip.startTime && endTime < clipEnd) {
          // For now, just trim the end at the placed clip's start
          // (the "hole" in the middle - user can manually handle this)
          clipsToModify.push({ id: clip.id, action: 'trim-end', trimAmount: clipEnd - startTime });
        }
      }
    }

    // Apply modifications
    if (clipsToModify.length === 0) return;

    const clipIdsToDelete = new Set(clipsToModify.filter(m => m.action === 'delete').map(m => m.id));

    set({
      clips: clips
        .filter(c => !clipIdsToDelete.has(c.id))
        .map(c => {
          const modification = clipsToModify.find(m => m.id === c.id);
          if (!modification || modification.action === 'delete') return c;

          if (modification.action === 'trim-start' && modification.trimAmount) {
            // Trim start: move startTime forward, adjust inPoint
            const newStartTime = c.startTime + modification.trimAmount;
            const newInPoint = c.inPoint + modification.trimAmount;
            const newDuration = c.duration - modification.trimAmount;
            return {
              ...c,
              startTime: newStartTime,
              inPoint: newInPoint,
              duration: newDuration,
            };
          }

          if (modification.action === 'trim-end' && modification.trimAmount) {
            // Trim end: reduce duration and outPoint
            const newDuration = c.duration - modification.trimAmount;
            const newOutPoint = c.outPoint - modification.trimAmount;
            return {
              ...c,
              duration: newDuration,
              outPoint: newOutPoint,
            };
          }

          return c;
        }),
    });

    invalidateCache();
  },
});
