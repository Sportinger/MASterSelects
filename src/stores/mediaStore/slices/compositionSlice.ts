// Composition CRUD and tab management

import type { Composition, MediaSliceCreator, MediaState } from '../types';
import { generateId } from '../helpers/importPipeline';
import { useTimelineStore } from '../../timeline';
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
    const comp: Composition = {
      id: generateId(),
      name,
      type: 'composition',
      parentId: null,
      createdAt: Date.now(),
      width: settings?.width ?? 1920,
      height: settings?.height ?? 1080,
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
  if (currentActiveId) {
    const timelineData = timelineStore.getSerializableState();
    set((state) => ({
      compositions: state.compositions.map((c) =>
        c.id === currentActiveId ? { ...c, timelineData } : c
      ),
    }));
    compositionRenderer.invalidateCompositionAndParents(currentActiveId);

    // Refresh nested clips in all parent compositions that contain this one
    // This ensures comp clips show updated content when source composition changes
    timelineStore.refreshCompClipNestedData(currentActiveId);
  }

  // Update active composition
  set({ activeCompositionId: newId });

  // Load new composition's timeline
  if (newId) {
    const freshCompositions = get().compositions;
    const newComp = freshCompositions.find((c) => c.id === newId);
    timelineStore.loadState(newComp?.timelineData);

    if (syncedPlayhead !== null && syncedPlayhead >= 0) {
      timelineStore.setPlayheadPosition(syncedPlayhead);
    }
    // zoom and scrollX are restored by loadState() from composition's timelineData
  } else {
    timelineStore.clearTimeline();
  }
}
