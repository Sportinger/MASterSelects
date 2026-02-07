// Selection-related actions slice

import type { SelectionActions, SliceCreator, Keyframe } from './types';

export const createSelectionSlice: SliceCreator<SelectionActions> = (set, get) => ({
  // Clip selection (multi-select support)
  selectClip: (id, addToSelection = false, setPrimaryOnly = false) => {
    const { selectedClipIds, expandedCurveProperties, clips } = get();

    // setPrimaryOnly: just update which clip is "focused" for Properties panel
    if (setPrimaryOnly && id !== null) {
      set({ primarySelectedClipId: id });
      return;
    }

    // Check if a specific clip has a curve editor open on its track
    const clipHasCurveEditorOpen = (clipId: string) => {
      if (expandedCurveProperties.size === 0) return false;
      const clip = clips.find(c => c.id === clipId);
      if (!clip) return false;
      const trackProps = expandedCurveProperties.get(clip.trackId);
      return trackProps && trackProps.size > 0;
    };

    // Check if any currently selected clip has curve editor open
    const hasAnyCurveEditorOpen = () => {
      if (expandedCurveProperties.size === 0) return false;
      const selectedClips = clips.filter(c => selectedClipIds.has(c.id));
      for (const clip of selectedClips) {
        const trackProps = expandedCurveProperties.get(clip.trackId);
        if (trackProps && trackProps.size > 0) {
          return true;
        }
      }
      return false;
    };

    if (id === null) {
      // Don't clear selection if curve editor is open
      if (hasAnyCurveEditorOpen()) return;
      set({ selectedClipIds: new Set(), primarySelectedClipId: null });
      return;
    }

    if (addToSelection) {
      // Shift+click: toggle only the clicked clip (independent selection)
      const newSet = new Set(selectedClipIds);
      if (newSet.has(id)) {
        // Trying to toggle off - prevent if this clip has curve editor open
        if (clipHasCurveEditorOpen(id)) return;
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      set({ selectedClipIds: newSet, primarySelectedClipId: id });
    } else {
      // Normal click: select clip + its linked clip
      // Prevent if any curve editor is open (unless clicking on already selected clip)
      if (!selectedClipIds.has(id) && hasAnyCurveEditorOpen()) {
        return;
      }
      const clip = clips.find(c => c.id === id);
      const linkedId = clip?.linkedClipId;
      const newSelection = linkedId ? new Set([id, linkedId]) : new Set([id]);
      set({ selectedClipIds: newSelection, primarySelectedClipId: id });
    }
  },

  selectClips: (ids) => {
    const { expandedCurveProperties, clips, selectedClipIds } = get();

    // Check if any currently selected clip has curve editor open
    const hasAnyCurveEditorOpen = () => {
      if (expandedCurveProperties.size === 0) return false;
      const selectedClips = clips.filter(c => selectedClipIds.has(c.id));
      for (const clip of selectedClips) {
        const trackProps = expandedCurveProperties.get(clip.trackId);
        if (trackProps && trackProps.size > 0) {
          return true;
        }
      }
      return false;
    };

    // Prevent if curve editor is open and would deselect clips
    if (hasAnyCurveEditorOpen()) {
      // Check if all currently selected clips are still in the new selection
      const currentSelected = Array.from(selectedClipIds);
      const wouldDeselect = currentSelected.some(clipId => !ids.includes(clipId));
      if (wouldDeselect) return;
    }

    set({ selectedClipIds: new Set(ids), primarySelectedClipId: ids.length > 0 ? ids[0] : null });
  },

  addClipToSelection: (id) => {
    const { selectedClipIds } = get();
    const newSet = new Set(selectedClipIds);
    newSet.add(id);
    set({ selectedClipIds: newSet, primarySelectedClipId: id });
  },

  removeClipFromSelection: (id) => {
    const { selectedClipIds, expandedCurveProperties, clips } = get();

    // Check if this clip has curve editor open
    const clip = clips.find(c => c.id === id);
    if (clip) {
      const trackProps = expandedCurveProperties.get(clip.trackId);
      if (trackProps && trackProps.size > 0) {
        return; // Don't allow removing clip with open curve editor
      }
    }

    const newSet = new Set(selectedClipIds);
    newSet.delete(id);
    set({ selectedClipIds: newSet });
  },

  clearClipSelection: () => {
    const { expandedCurveProperties, clips, selectedClipIds } = get();

    // Check if any currently selected clip has curve editor open
    if (expandedCurveProperties.size > 0) {
      const selectedClips = clips.filter(c => selectedClipIds.has(c.id));
      for (const clip of selectedClips) {
        const trackProps = expandedCurveProperties.get(clip.trackId);
        if (trackProps && trackProps.size > 0) {
          return; // Don't clear if curve editor is open
        }
      }
    }

    set({ selectedClipIds: new Set(), primarySelectedClipId: null });
  },

  // Keyframe selection
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
