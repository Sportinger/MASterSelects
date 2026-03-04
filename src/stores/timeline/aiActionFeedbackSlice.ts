// AI Action Feedback Slice
// Manages transient visual feedback state for AI tool actions
// (split glow lines, delete ghosts, move animations, trim highlights)

import type { AIActionFeedbackActions, SliceCreator } from './types';

export const createAIActionFeedbackSlice: SliceCreator<AIActionFeedbackActions> = (set, get) => ({
  addAIOverlay: (overlay) => {
    const id = `ai-overlay-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
    const entry = { ...overlay, id, createdAt: Date.now() };

    const current = get().aiActionOverlays;
    set({ aiActionOverlays: [...current, entry] });

    // Auto-cleanup after animation duration + buffer
    setTimeout(() => {
      const overlays = get().aiActionOverlays;
      set({ aiActionOverlays: overlays.filter(o => o.id !== id) });
    }, overlay.duration + 50);

    return id;
  },

  removeAIOverlay: (id) => {
    const overlays = get().aiActionOverlays;
    set({ aiActionOverlays: overlays.filter(o => o.id !== id) });
  },

  setAIMovingClip: (clipId, fromStartTime, animationDuration = 200) => {
    const current = get().aiMovingClips;
    const next = new Map(current);
    next.set(clipId, { clipId, fromStartTime, animationDuration, startedAt: Date.now() });
    set({ aiMovingClips: next });

    // Auto-cleanup after animation completes
    setTimeout(() => {
      const map = get().aiMovingClips;
      if (map.has(clipId)) {
        const next = new Map(map);
        next.delete(clipId);
        set({ aiMovingClips: next });
      }
    }, animationDuration + 50);
  },

  clearAIMovingClip: (clipId) => {
    const map = get().aiMovingClips;
    if (map.has(clipId)) {
      const next = new Map(map);
      next.delete(clipId);
      set({ aiMovingClips: next });
    }
  },
});
