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

    // Auto-cleanup after animation delay + duration + buffer
    const totalTime = (overlay.animationDelay || 0) + overlay.duration + 50;
    setTimeout(() => {
      const overlays = get().aiActionOverlays;
      set({ aiActionOverlays: overlays.filter(o => o.id !== id) });
    }, totalTime);

    return id;
  },

  addAIOverlaysBatch: (overlays) => {
    const now = Date.now();
    const entries = overlays.map((overlay, i) => ({
      ...overlay,
      id: `ai-overlay-${now}-${i}-${Math.random().toString(36).substr(2, 3)}`,
      createdAt: now,
    }));

    const current = get().aiActionOverlays;
    set({ aiActionOverlays: [...current, ...entries] });

    // Single cleanup: remove all after the longest animation completes
    const maxTime = Math.max(...entries.map(e => (e.animationDelay || 0) + e.duration + 50));
    setTimeout(() => {
      const ids = new Set(entries.map(e => e.id));
      const remaining = get().aiActionOverlays.filter(o => !ids.has(o.id));
      set({ aiActionOverlays: remaining });
    }, maxTime);
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
