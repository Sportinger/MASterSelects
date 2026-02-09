// Composition CRUD and tab management

import type { Composition, MediaSliceCreator, MediaState } from '../types';
import { generateId } from '../helpers/importPipeline';
import { useTimelineStore } from '../../timeline';
import { useSettingsStore } from '../../settingsStore';
import { compositionRenderer } from '../../../services/compositionRenderer';

export interface CompositionActions {
  createComposition: (name: string, settings?: Partial<Composition>) => Composition;
  duplicateComposition: (id: string) => Composition | null;
  removeComposition: (id: string) => void;
  updateComposition: (id: string, updates: Partial<Composition>) => void;
  setActiveComposition: (id: string | null) => void;
  getActiveComposition: () => Composition | undefined;
  openCompositionTab: (id: string) => void;
  closeCompositionTab: (id: string) => void;
  getOpenCompositions: () => Composition[];
  reorderCompositionTabs: (fromIndex: number, toIndex: number) => void;
}

export const createCompositionSlice: MediaSliceCreator<CompositionActions> = (set, get) => ({
  createComposition: (name: string, settings?: Partial<Composition>) => {
    const { outputResolution } = useSettingsStore.getState();
    const comp: Composition = {
      id: generateId(),
      name,
      type: 'composition',
      parentId: null,
      createdAt: Date.now(),
      width: settings?.width ?? outputResolution.width,
      height: settings?.height ?? outputResolution.height,
      frameRate: settings?.frameRate ?? 30,
      duration: settings?.duration ?? 60,
      backgroundColor: settings?.backgroundColor ?? '#000000',
    };

    set((state) => ({ compositions: [...state.compositions, comp] }));
    return comp;
  },

  duplicateComposition: (id: string) => {
    const original = get().compositions.find((c) => c.id === id);
    if (!original) return null;

    const duplicate: Composition = {
      ...original,
      id: generateId(),
      name: `${original.name} Copy`,
      createdAt: Date.now(),
    };

    set((state) => ({ compositions: [...state.compositions, duplicate] }));
    return duplicate;
  },

  removeComposition: (id: string) => {
    set((state) => ({
      compositions: state.compositions.filter((c) => c.id !== id),
      selectedIds: state.selectedIds.filter((sid) => sid !== id),
      activeCompositionId: state.activeCompositionId === id ? null : state.activeCompositionId,
      openCompositionIds: state.openCompositionIds.filter((cid) => cid !== id),
    }));
  },

  updateComposition: (id: string, updates: Partial<Composition>) => {
    const oldComp = get().compositions.find((c) => c.id === id);
    if (oldComp && (updates.width !== undefined || updates.height !== undefined)) {
      const newW = updates.width ?? oldComp.width;
      const newH = updates.height ?? oldComp.height;
      if (newW !== oldComp.width || newH !== oldComp.height) {
        adjustClipTransformsOnResize(get, id, oldComp.width, oldComp.height, newW, newH, updates);
      }
    }
    set((state) => ({
      compositions: state.compositions.map((c) =>
        c.id === id ? { ...c, ...updates } : c
      ),
    }));
  },

  setActiveComposition: (id: string | null) => {
    const { activeCompositionId, compositions } = get();
    doSetActiveComposition(set, get, activeCompositionId, id, compositions);
  },

  getActiveComposition: () => {
    const { compositions, activeCompositionId } = get();
    return compositions.find((c) => c.id === activeCompositionId);
  },

  openCompositionTab: (id: string) => {
    const { openCompositionIds, activeCompositionId, compositions } = get();
    if (!openCompositionIds.includes(id)) {
      set({ openCompositionIds: [...openCompositionIds, id] });
    }
    // Inline setActiveComposition logic
    doSetActiveComposition(set, get, activeCompositionId, id, compositions);
  },

  closeCompositionTab: (id: string) => {
    const { openCompositionIds, activeCompositionId, compositions } = get();
    const newOpenIds = openCompositionIds.filter((cid) => cid !== id);
    set({ openCompositionIds: newOpenIds });

    if (activeCompositionId === id && newOpenIds.length > 0) {
      const closedIndex = openCompositionIds.indexOf(id);
      const newActiveIndex = Math.min(closedIndex, newOpenIds.length - 1);
      doSetActiveComposition(set, get, activeCompositionId, newOpenIds[newActiveIndex], compositions);
    } else if (newOpenIds.length === 0) {
      doSetActiveComposition(set, get, activeCompositionId, null, compositions);
    }
  },

  getOpenCompositions: () => {
    const { compositions, openCompositionIds } = get();
    return openCompositionIds
      .map((id) => compositions.find((c) => c.id === id))
      .filter((c): c is Composition => c !== undefined);
  },

  reorderCompositionTabs: (fromIndex: number, toIndex: number) => {
    const { openCompositionIds } = get();
    if (fromIndex < 0 || fromIndex >= openCompositionIds.length) return;
    if (toIndex < 0 || toIndex >= openCompositionIds.length) return;
    if (fromIndex === toIndex) return;

    const newOrder = [...openCompositionIds];
    const [moved] = newOrder.splice(fromIndex, 1);
    newOrder.splice(toIndex, 0, moved);
    set({ openCompositionIds: newOrder });
  },
});

/**
 * Adjust clip transforms when a composition is resized so content stays at
 * the same pixel position (more canvas space around it, no scaling).
 *
 * Position is in normalized space (1.0 = full canvas). When canvas grows,
 * the same normalized position maps to a different pixel location.
 * Rescale position by oldRes/newRes to keep pixel coords stable.
 */
function adjustClipTransformsOnResize(
  get: () => MediaState,
  compId: string,
  oldW: number,
  oldH: number,
  newW: number,
  newH: number,
  updates: Partial<Composition>
): void {
  const scaleX = oldW / newW;
  const scaleY = oldH / newH;

  const { activeCompositionId } = get();

  if (compId === activeCompositionId) {
    // Active comp: modify live timeline store
    const timelineStore = useTimelineStore.getState();
    const { clips, clipKeyframes } = timelineStore;

    const updatedClips = clips.map(clip => ({
      ...clip,
      transform: {
        ...clip.transform,
        position: {
          x: clip.transform.position.x * scaleX,
          y: clip.transform.position.y * scaleY,
          z: clip.transform.position.z,
        },
      },
    }));

    // Adjust keyframes for position properties
    const updatedKeyframes = new Map<string, import('../../../types').Keyframe[]>();
    clipKeyframes.forEach((keyframes: import('../../../types').Keyframe[], clipId: string) => {
      updatedKeyframes.set(clipId, keyframes.map(kf => {
        if (kf.property === 'position.x') {
          return { ...kf, value: kf.value * scaleX };
        }
        if (kf.property === 'position.y') {
          return { ...kf, value: kf.value * scaleY };
        }
        return kf;
      }));
    });

    useTimelineStore.setState({ clips: updatedClips, clipKeyframes: updatedKeyframes });
  } else {
    // Non-active comp: modify serialized timelineData via the updates object
    // so the subsequent set() in updateComposition picks it up
    const comp = get().compositions.find(c => c.id === compId);
    if (!comp?.timelineData) return;

    const updatedClips = comp.timelineData.clips.map(clip => ({
      ...clip,
      transform: {
        ...clip.transform,
        position: {
          x: clip.transform.position.x * scaleX,
          y: clip.transform.position.y * scaleY,
          z: clip.transform.position.z,
        },
      },
      keyframes: clip.keyframes?.map(kf => {
        if (kf.property === 'position.x') {
          return { ...kf, value: kf.value * scaleX };
        }
        if (kf.property === 'position.y') {
          return { ...kf, value: kf.value * scaleY };
        }
        return kf;
      }),
    }));

    // Fold adjusted timelineData into the updates object
    updates.timelineData = { ...comp.timelineData, clips: updatedClips };
  }
}

/**
 * Calculate synced playhead for nested composition navigation.
 */
function calculateSyncedPlayhead(
  fromCompId: string | null,
  toCompId: string | null,
  compositions: Composition[],
  timelineStore: ReturnType<typeof useTimelineStore.getState>
): number | null {
  if (!fromCompId || !toCompId) return null;

  const currentPlayhead = timelineStore.playheadPosition;
  const currentClips = timelineStore.clips;

  // Check if navigating into nested comp
  const nestedClip = currentClips.find(
    (c) => c.isComposition && c.compositionId === toCompId
  );
  if (nestedClip) {
    const clipStart = nestedClip.startTime;
    const clipEnd = clipStart + nestedClip.duration;
    const inPoint = nestedClip.inPoint || 0;

    if (currentPlayhead >= clipStart && currentPlayhead < clipEnd) {
      return (currentPlayhead - clipStart) + inPoint;
    }
  }

  // Check if navigating to parent comp
  const toComp = compositions.find((c) => c.id === toCompId);
  if (toComp?.timelineData?.clips) {
    const parentClip = toComp.timelineData.clips.find(
      (c: { isComposition?: boolean; compositionId?: string; startTime: number; inPoint?: number }) =>
        c.isComposition && c.compositionId === fromCompId
    );
    if (parentClip) {
      return parentClip.startTime + (currentPlayhead - (parentClip.inPoint || 0));
    }
  }

  return null;
}

/**
 * Internal helper to set active composition (avoids calling get().setActiveComposition).
 * Handles exit/enter animations for smooth transitions.
 */
function doSetActiveComposition(
  set: (partial: Partial<MediaState> | ((state: MediaState) => Partial<MediaState>)) => void,
  get: () => MediaState,
  currentActiveId: string | null,
  newId: string | null,
  compositions: Composition[]
): void {
  const timelineStore = useTimelineStore.getState();

  // Calculate synced playhead for nested composition navigation
  const syncedPlayhead = calculateSyncedPlayhead(
    currentActiveId,
    newId,
    compositions,
    timelineStore
  );

  // Save current timeline to current composition
  const savedCompId = currentActiveId;
  if (currentActiveId) {
    const timelineData = timelineStore.getSerializableState();
    set((state) => ({
      compositions: state.compositions.map((c) =>
        c.id === currentActiveId ? { ...c, timelineData } : c
      ),
    }));
    compositionRenderer.invalidateCompositionAndParents(currentActiveId);
  }

  // Trigger exit animation for current clips
  const hasExistingClips = timelineStore.clips.length > 0;
  if (hasExistingClips && newId !== currentActiveId) {
    // Set exit animation phase
    timelineStore.setClipAnimationPhase('exiting');

    // Wait for exit animation, then load new composition
    setTimeout(async () => {
      await finishCompositionSwitch(set, get, newId, savedCompId, syncedPlayhead);
    }, 350); // Exit animation duration
  } else {
    // No existing clips or same comp, load immediately
    finishCompositionSwitch(set, get, newId, savedCompId, syncedPlayhead);
  }
}

/**
 * Complete the composition switch after exit animation
 */
async function finishCompositionSwitch(
  set: (partial: Partial<MediaState> | ((state: MediaState) => Partial<MediaState>)) => void,
  get: () => MediaState,
  newId: string | null,
  savedCompId: string | null,
  syncedPlayhead: number | null
): Promise<void> {
  const timelineStore = useTimelineStore.getState();

  // Update active composition
  set({ activeCompositionId: newId });

  // Load new composition's timeline
  if (newId) {
    const freshCompositions = get().compositions;
    const newComp = freshCompositions.find((c) => c.id === newId);
    await timelineStore.loadState(newComp?.timelineData);

    if (syncedPlayhead !== null && syncedPlayhead >= 0) {
      timelineStore.setPlayheadPosition(syncedPlayhead);
    }
    // zoom and scrollX are restored by loadState() from composition's timelineData

    // Refresh nested clips in the NEW timeline that reference the OLD composition
    // This ensures comp clips show updated content when source composition changes
    if (savedCompId) {
      timelineStore.refreshCompClipNestedData(savedCompId);
    }

    // Trigger entrance animation for new clips
    timelineStore.setClipAnimationPhase('entering');

    // Reset to idle after entrance animation completes
    setTimeout(() => {
      timelineStore.setClipAnimationPhase('idle');
    }, 700); // Entrance animation duration (0.6s + buffer)
  } else {
    timelineStore.clearTimeline();
    timelineStore.setClipAnimationPhase('idle');
  }
}
