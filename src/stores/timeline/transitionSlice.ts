// Transition-related actions slice
// Transitions are handled independently from keyframes - the compositor blends clips during rendering

import type { TransitionActions, SliceCreator } from './types';
import type { TimelineTransition } from '../../types';
import { getTransition, type TransitionType } from '../../transitions';
import { Logger } from '../../services/logger';

const log = Logger.create('Transitions');

// Generate unique transition ID
const generateTransitionId = () => `transition-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

export const createTransitionSlice: SliceCreator<TransitionActions> = (set, get) => ({
  /**
   * Apply a transition between two adjacent clips
   * Creates overlap by moving clipB earlier - NO keyframes, compositor handles blending
   */
  applyTransition: (clipAId: string, clipBId: string, type: string, duration: number) => {
    const { clips, invalidateCache } = get();

    const clipA = clips.find(c => c.id === clipAId);
    const clipB = clips.find(c => c.id === clipBId);

    if (!clipA || !clipB) {
      log.warn('Cannot apply transition: clips not found', { clipAId, clipBId });
      return;
    }

    // Verify clips are on the same track
    if (clipA.trackId !== clipB.trackId) {
      log.warn('Cannot apply transition: clips on different tracks');
      return;
    }

    // Verify clipB comes after clipA
    if (clipB.startTime < clipA.startTime) {
      log.warn('Cannot apply transition: clipB must come after clipA');
      return;
    }

    // Get transition definition for validation
    const transitionDef = getTransition(type as TransitionType);
    if (!transitionDef) {
      log.warn('Unknown transition type', { type });
      return;
    }

    // Clamp duration to valid range
    const effectiveDuration = Math.min(
      Math.max(duration, transitionDef.minDuration),
      Math.min(transitionDef.maxDuration, clipA.duration * 0.5, clipB.duration * 0.5)
    );

    // Calculate new positions - move clipB's start earlier to create overlap
    const clipAEnd = clipA.startTime + clipA.duration;
    const newClipBStart = clipAEnd - effectiveDuration;

    // Create transition objects
    const transitionId = generateTransitionId();

    const transitionOut: TimelineTransition = {
      id: transitionId,
      type,
      duration: effectiveDuration,
      linkedClipId: clipBId,
    };

    const transitionIn: TimelineTransition = {
      id: transitionId,
      type,
      duration: effectiveDuration,
      linkedClipId: clipAId,
    };

    // Update clips with new positions and transition data
    set({
      clips: clips.map(c => {
        if (c.id === clipAId) {
          return { ...c, transitionOut };
        }
        if (c.id === clipBId) {
          return { ...c, startTime: newClipBStart, transitionIn };
        }
        return c;
      }),
    });

    invalidateCache();
    log.info('Applied transition', { type, duration: effectiveDuration, clipA: clipA.name, clipB: clipB.name });
  },

  /**
   * Remove a transition from a clip
   */
  removeTransition: (clipId: string, edge: 'in' | 'out') => {
    const { clips, invalidateCache } = get();

    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    const transition = edge === 'in' ? clip.transitionIn : clip.transitionOut;
    if (!transition) return;

    const linkedClip = clips.find(c => c.id === transition.linkedClipId);
    const duration = transition.duration;

    if (edge === 'in') {
      // Move clip back to non-overlapping position
      const linkedClipEnd = linkedClip ? linkedClip.startTime + linkedClip.duration : clip.startTime;
      const newStartTime = linkedClipEnd; // Remove overlap

      set({
        clips: clips.map(c => {
          if (c.id === clipId) {
            return { ...c, startTime: newStartTime, transitionIn: undefined };
          }
          if (c.id === transition.linkedClipId) {
            return { ...c, transitionOut: undefined };
          }
          return c;
        }),
      });
    } else {
      // Move linked clip back
      set({
        clips: clips.map(c => {
          if (c.id === clipId) {
            return { ...c, transitionOut: undefined };
          }
          if (c.id === transition.linkedClipId) {
            const newStart = clip.startTime + clip.duration;
            return { ...c, startTime: newStart, transitionIn: undefined };
          }
          return c;
        }),
      });
    }

    invalidateCache();
    log.info('Removed transition', { clipId, edge, duration });
  },

  /**
   * Update the duration of an existing transition
   */
  updateTransitionDuration: (clipId: string, edge: 'in' | 'out', newDuration: number) => {
    const { clips } = get();

    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    const transition = edge === 'in' ? clip.transitionIn : clip.transitionOut;
    if (!transition) return;

    // Remove and re-apply with new duration
    const linkedClipId = transition.linkedClipId;
    const type = transition.type;

    // Determine which is clipA and clipB
    const isClipB = edge === 'in';
    const clipAId = isClipB ? linkedClipId : clipId;
    const clipBId = isClipB ? clipId : linkedClipId;

    // First remove the existing transition
    get().removeTransition(clipId, edge);

    // Then apply new transition with updated duration
    get().applyTransition(clipAId, clipBId, type, newDuration);
  },

  /**
   * Find a junction between two clips at a given time
   * Returns the two clips if found, null otherwise
   */
  findClipJunction: (trackId: string, time: number, threshold: number = 0.5) => {
    const { clips } = get();

    // Get clips on this track, sorted by start time
    const trackClips = clips
      .filter(c => c.trackId === trackId)
      .sort((a, b) => a.startTime - b.startTime);

    // Find adjacent clip pairs
    for (let i = 0; i < trackClips.length - 1; i++) {
      const clipA = trackClips[i];
      const clipB = trackClips[i + 1];

      const clipAEnd = clipA.startTime + clipA.duration;
      const gap = clipB.startTime - clipAEnd;

      // Check if clips are touching (small gap) or already have a transition (negative gap/overlap)
      if (Math.abs(gap) < 0.1 || (clipA.transitionOut && clipB.transitionIn)) {
        // For clips with transition, junction is at the transition center
        const junctionTime = clipA.transitionOut
          ? clipAEnd - clipA.transitionOut.duration / 2
          : clipAEnd;

        // Check if the given time is near this junction
        if (Math.abs(time - junctionTime) < threshold) {
          return { clipA, clipB, junctionTime };
        }
      }
    }

    return null;
  },
});
