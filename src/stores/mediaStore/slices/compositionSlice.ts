// Composition CRUD and tab management

import type { Composition, MediaSliceCreator, MediaState } from '../types';
import { generateId } from '../helpers/importPipeline';
import { useTimelineStore } from '../../timeline';
import { useSettingsStore } from '../../settingsStore';
import { compositionRenderer } from '../../../services/compositionRenderer';
import { playheadState } from '../../../services/layerBuilder';

export interface CompositionSwitchOptions {
  skipAnimation?: boolean;
  playFromStart?: boolean;
}

export interface CompositionActions {
  createComposition: (name: string, settings?: Partial<Composition>) => Composition;
  duplicateComposition: (id: string) => Composition | null;
  removeComposition: (id: string) => void;
  updateComposition: (id: string, updates: Partial<Composition>) => void;
  setActiveComposition: (id: string | null) => void;
  getActiveComposition: () => Composition | undefined;
  openCompositionTab: (id: string, options?: CompositionSwitchOptions) => void;
  closeCompositionTab: (id: string) => void;
  getOpenCompositions: () => Composition[];
  reorderCompositionTabs: (fromIndex: number, toIndex: number) => void;
  moveSlot: (compId: string, toSlotIndex: number) => void;
  unassignSlot: (compId: string) => void;
  assignMediaFileToSlot: (mediaFileId: string, slotIndex: number) => void;
  setPreviewComposition: (id: string | null) => void;
  getSlotMap: (totalSlots: number) => (Composition | null)[];
  // Multi-layer playback (Resolume-style)
  activateOnLayer: (compositionId: string, layerIndex: number) => void;
  deactivateLayer: (layerIndex: number) => void;
  activateColumn: (colIndex: number) => void;
  deactivateAllLayers: () => void;
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
    set((state) => {
      const newAssignments = { ...state.slotAssignments };
      delete newAssignments[id];
      return {
        compositions: state.compositions.filter((c) => c.id !== id),
        selectedIds: state.selectedIds.filter((sid) => sid !== id),
        activeCompositionId: state.activeCompositionId === id ? null : state.activeCompositionId,
        openCompositionIds: state.openCompositionIds.filter((cid) => cid !== id),
        slotAssignments: newAssignments,
      };
    });
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

  openCompositionTab: (id: string, options?: CompositionSwitchOptions) => {
    const { openCompositionIds, activeCompositionId, compositions } = get();
    if (!openCompositionIds.includes(id)) {
      set({ openCompositionIds: [...openCompositionIds, id] });
    }
    // Same comp already active + playFromStart → just restart playback (no reload)
    if (id === activeCompositionId && options?.playFromStart) {
      const ts = useTimelineStore.getState();
      // Stop first to reset everything cleanly, then restart
      ts.pause();
      ts.setPlayheadPosition(0);
      // Reset the high-frequency playhead and audio master
      playheadState.position = 0;
      playheadState.hasMasterAudio = false;
      playheadState.masterAudioElement = null;
      playheadState.playbackJustStarted = true;
      // Seek all video/audio elements back to their in-points
      for (const clip of ts.clips) {
        if (clip.source?.videoElement) {
          clip.source.videoElement.currentTime = clip.inPoint;
        }
        if (clip.source?.audioElement) {
          clip.source.audioElement.currentTime = clip.inPoint;
        }
      }
      ts.play();
      return;
    }
    // Inline setActiveComposition logic
    doSetActiveComposition(set, get, activeCompositionId, id, compositions, options);
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

  moveSlot: (compId: string, toSlotIndex: number) => {
    const { slotAssignments } = get();
    const newAssignments = { ...slotAssignments };
    // Remove any comp currently at the target slot
    for (const [id, idx] of Object.entries(newAssignments)) {
      if (idx === toSlotIndex && id !== compId) {
        // Swap: move displaced comp to the dragged comp's old slot
        const oldSlot = newAssignments[compId];
        if (oldSlot !== undefined) {
          newAssignments[id] = oldSlot;
        } else {
          delete newAssignments[id];
        }
        break;
      }
    }
    newAssignments[compId] = toSlotIndex;
    set({ slotAssignments: newAssignments });
  },

  unassignSlot: (compId: string) => {
    const { slotAssignments } = get();
    const newAssignments = { ...slotAssignments };
    delete newAssignments[compId];
    set({ slotAssignments: newAssignments });
  },

  assignMediaFileToSlot: (mediaFileId: string, slotIndex: number) => {
    const { files } = get();
    const mediaFile = files.find(f => f.id === mediaFileId);
    if (!mediaFile) return;

    // Create composition from media file (inline createComposition logic)
    const { outputResolution } = useSettingsStore.getState();
    const nameWithoutExt = mediaFile.name.replace(/\.[^.]+$/, '');
    const comp: Composition = {
      id: generateId(),
      name: nameWithoutExt,
      type: 'composition',
      parentId: null,
      createdAt: Date.now(),
      width: mediaFile.width || outputResolution.width,
      height: mediaFile.height || outputResolution.height,
      frameRate: 30,
      duration: mediaFile.duration || 60,
      backgroundColor: '#000000',
    };
    set((state) => ({ compositions: [...state.compositions, comp] }));

    // Assign to slot (inline moveSlot logic)
    const { slotAssignments } = get();
    const newAssignments = { ...slotAssignments };
    for (const [id, idx] of Object.entries(newAssignments)) {
      if (idx === slotIndex && id !== comp.id) {
        delete newAssignments[id];
        break;
      }
    }
    newAssignments[comp.id] = slotIndex;
    set({ slotAssignments: newAssignments });

    // Open the composition tab (loads empty timeline)
    const { activeCompositionId, compositions } = get();
    if (!get().openCompositionIds.includes(comp.id)) {
      set({ openCompositionIds: [...get().openCompositionIds, comp.id] });
    }
    doSetActiveComposition(set, get, activeCompositionId, comp.id, compositions, { skipAnimation: true });

    // After short delay (let loadState settle for empty comp), add media as a clip
    // then flush timeline state back to composition so MiniTimeline shows correct preview
    setTimeout(async () => {
      const ts = useTimelineStore.getState();
      const videoTrack = ts.tracks.find(t => t.type === 'video');
      const audioTrack = ts.tracks.find(t => t.type === 'audio');

      if (mediaFile.file) {
        if ((mediaFile.type === 'video' || mediaFile.type === 'image') && videoTrack) {
          await ts.addClip(videoTrack.id, mediaFile.file, 0, mediaFile.duration, mediaFile.id);
        } else if (mediaFile.type === 'audio' && audioTrack) {
          await ts.addClip(audioTrack.id, mediaFile.file, 0, mediaFile.duration, mediaFile.id);
        }
      }

      // Save timeline state back to composition's timelineData for MiniTimeline preview
      const timelineData = useTimelineStore.getState().getSerializableState();
      set((state) => ({
        compositions: state.compositions.map((c) =>
          c.id === comp.id ? { ...c, timelineData } : c
        ),
      }));
    }, 100);
  },

  setPreviewComposition: (id: string | null) => {
    set({ previewCompositionId: id });
  },

  getSlotMap: (totalSlots: number) => {
    const { compositions, slotAssignments } = get();
    const map: (Composition | null)[] = new Array(totalSlots).fill(null);
    const assigned = new Set<string>();

    // Place explicitly assigned compositions
    for (const [compId, slotIdx] of Object.entries(slotAssignments)) {
      if (slotIdx >= 0 && slotIdx < totalSlots) {
        const comp = compositions.find((c: Composition) => c.id === compId);
        if (comp) {
          map[slotIdx] = comp;
          assigned.add(compId);
        }
      }
    }

    return map;
  },

  // === Multi-layer playback (Resolume-style) ===

  activateOnLayer: (compositionId: string, layerIndex: number) => {
    const { activeLayerSlots } = get();
    // If same comp already on this layer, it'll be restarted by the caller
    const newSlots = { ...activeLayerSlots };
    // Remove this comp from any other layer it might be on
    for (const [key, val] of Object.entries(newSlots)) {
      if (val === compositionId) {
        delete newSlots[Number(key)];
      }
    }
    newSlots[layerIndex] = compositionId;
    set({ activeLayerSlots: newSlots });
  },

  deactivateLayer: (layerIndex: number) => {
    const { activeLayerSlots } = get();
    const newSlots = { ...activeLayerSlots };
    delete newSlots[layerIndex];
    set({ activeLayerSlots: newSlots });
  },

  activateColumn: (colIndex: number) => {
    const GRID_COLS = 12;
    const GRID_ROWS = 4;
    const TOTAL_SLOTS = GRID_COLS * GRID_ROWS;
    // Inline getSlotMap logic (avoids unknown type from index signature)
    const { compositions, slotAssignments } = get();
    const slotMap: (Composition | null)[] = new Array(TOTAL_SLOTS).fill(null);
    for (const [compId, slotIdx] of Object.entries(slotAssignments)) {
      if (slotIdx >= 0 && slotIdx < TOTAL_SLOTS) {
        const comp = compositions.find(c => c.id === compId);
        if (comp) { slotMap[slotIdx] = comp; }
      }
    }

    const newSlots: Record<number, string | null> = {};
    for (let row = 0; row < GRID_ROWS; row++) {
      const slotIndex = row * GRID_COLS + colIndex;
      const comp = slotMap[slotIndex];
      if (comp) {
        newSlots[row] = comp.id;
      }
    }
    set({ activeLayerSlots: newSlots });
  },

  deactivateAllLayers: () => {
    set({ activeLayerSlots: {} });
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
        scale: {
          x: clip.transform.scale.x * scaleX,
          y: clip.transform.scale.y * scaleY,
        },
      },
    }));

    // Adjust keyframes for position and scale properties
    const updatedKeyframes = new Map<string, import('../../../types').Keyframe[]>();
    clipKeyframes.forEach((keyframes: import('../../../types').Keyframe[], clipId: string) => {
      updatedKeyframes.set(clipId, keyframes.map(kf => {
        if (kf.property === 'position.x' || kf.property === 'scale.x') {
          return { ...kf, value: kf.value * scaleX };
        }
        if (kf.property === 'position.y' || kf.property === 'scale.y') {
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
        scale: {
          x: clip.transform.scale.x * scaleX,
          y: clip.transform.scale.y * scaleY,
        },
      },
      keyframes: clip.keyframes?.map(kf => {
        if (kf.property === 'position.x' || kf.property === 'scale.x') {
          return { ...kf, value: kf.value * scaleX };
        }
        if (kf.property === 'position.y' || kf.property === 'scale.y') {
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
  compositions: Composition[],
  options?: CompositionSwitchOptions
): void {
  const timelineStore = useTimelineStore.getState();
  const skipAnimation = options?.skipAnimation ?? false;

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
    // Sync high-frequency playhead position back to store before serializing
    // (rAF loop updates playheadState.position but not the Zustand store)
    if (playheadState.isUsingInternalPosition) {
      timelineStore.setPlayheadPosition(playheadState.position);
    }
    const timelineData = timelineStore.getSerializableState();
    set((state) => ({
      compositions: state.compositions.map((c) =>
        c.id === currentActiveId ? { ...c, timelineData } : c
      ),
    }));
    compositionRenderer.invalidateCompositionAndParents(currentActiveId);
  }

  if (skipAnimation) {
    // Skip exit/enter animations entirely
    finishCompositionSwitch(set, get, newId, savedCompId, syncedPlayhead, options);
    return;
  }

  // Trigger exit animation for current clips
  const hasExistingClips = timelineStore.clips.length > 0;
  if (hasExistingClips && newId !== currentActiveId) {
    // Set exit animation phase
    timelineStore.setClipAnimationPhase('exiting');

    // Wait for exit animation, then load new composition
    setTimeout(async () => {
      await finishCompositionSwitch(set, get, newId, savedCompId, syncedPlayhead, options);
    }, 350); // Exit animation duration
  } else {
    // No existing clips or same comp, load immediately
    finishCompositionSwitch(set, get, newId, savedCompId, syncedPlayhead, options);
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
  syncedPlayhead: number | null,
  options?: CompositionSwitchOptions
): Promise<void> {
  const timelineStore = useTimelineStore.getState();
  const skipAnimation = options?.skipAnimation ?? false;
  const playFromStart = options?.playFromStart ?? false;

  // Update active composition
  set({ activeCompositionId: newId });

  // Load new composition's timeline
  if (newId) {
    const freshCompositions = get().compositions;
    const newComp = freshCompositions.find((c) => c.id === newId);
    await timelineStore.loadState(newComp?.timelineData);

    if (playFromStart) {
      timelineStore.setPlayheadPosition(0);
      timelineStore.play();
    } else if (syncedPlayhead !== null && syncedPlayhead >= 0) {
      timelineStore.setPlayheadPosition(syncedPlayhead);
    }
    // zoom and scrollX are restored by loadState() from composition's timelineData

    // Refresh nested clips in the NEW timeline that reference the OLD composition
    // This ensures comp clips show updated content when source composition changes
    if (savedCompId) {
      timelineStore.refreshCompClipNestedData(savedCompId);
    }

    if (skipAnimation) {
      // Skip entrance animation — go straight to idle
      timelineStore.setClipAnimationPhase('idle');
    } else {
      // Trigger entrance animation for new clips
      timelineStore.setClipAnimationPhase('entering');

      // Reset to idle after entrance animation completes
      setTimeout(() => {
        timelineStore.setClipAnimationPhase('idle');
      }, 700); // Entrance animation duration (0.6s + buffer)
    }
  } else {
    timelineStore.clearTimeline();
    timelineStore.setClipAnimationPhase('idle');
  }
}
