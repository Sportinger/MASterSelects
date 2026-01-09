// Selection-related actions slice

import type { SelectionActions, SliceCreator, Keyframe } from './types';

export const createSelectionSlice: SliceCreator<SelectionActions> = (set, get) => ({
  selectKeyframe: (keyframeId, addToSelection = false) => {
    const { selectedKeyframeIds } = get();

    if (addToSelection) {
      const newSet = new Set(selectedKeyframeIds);
      if (newSet.has(keyframeId)) {
        newSet.delete(keyframeId);
      } else {
        newSet.add(keyframeId);
      }
      set({ selectedKeyframeIds: newSet });
    } else {
      set({ selectedKeyframeIds: new Set([keyframeId]) });
    }
  },

  deselectAllKeyframes: () => {
    set({ selectedKeyframeIds: new Set() });
  },

  deleteSelectedKeyframes: () => {
    const { selectedKeyframeIds, clipKeyframes, invalidateCache } = get();
    if (selectedKeyframeIds.size === 0) return;

    const newMap = new Map<string, Keyframe[]>();

    clipKeyframes.forEach((keyframes, clipId) => {
      const filtered = keyframes.filter(k => !selectedKeyframeIds.has(k.id));
      if (filtered.length > 0) {
        newMap.set(clipId, filtered);
      }
    });

    set({
      clipKeyframes: newMap,
      selectedKeyframeIds: new Set(),
    });
    invalidateCache();
  },
});
