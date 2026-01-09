// Global undo/redo history store
// Captures snapshots of all undoable state across stores

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// Snapshot of undoable state from all stores
interface StateSnapshot {
  timestamp: number;
  label: string; // Description of the action (for debugging)

  // Timeline state
  timeline: {
    clips: any[];
    tracks: any[];
    selectedClipIds: string[];
    zoom: number;
    scrollX: number;
  };

  // Mixer state
  mixer: {
    layers: any[];
    selectedLayerId: string | null;
    gridColumns: number;
    gridRows: number;
    slotGroups: any[];
    outputResolution: { width: number; height: number };
  };

  // Media state
  media: {
    files: any[];
    compositions: any[];
    folders: any[];
    selectedIds: string[];
    expandedFolderIds: string[];
  };

  // Dock layout state
  dock: {
    layout: any;
  };
}

interface HistoryState {
  // Undo/redo stacks
  undoStack: StateSnapshot[];
  redoStack: StateSnapshot[];

  // Current state (for comparison to avoid duplicate snapshots)
  currentSnapshot: StateSnapshot | null;

  // Maximum history size
  maxHistorySize: number;

  // Whether we're currently applying undo/redo (to prevent capturing)
  isApplying: boolean;

  // Batch tracking - for grouping multiple changes into one undo step
  batchId: number | null;
  batchLabel: string | null;

  // Actions
  captureSnapshot: (label: string) => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // Batch operations
  startBatch: (label: string) => void;
  endBatch: () => void;

  // Internal
  setIsApplying: (value: boolean) => void;
  clearHistory: () => void;
}

// Import stores dynamically to avoid circular dependencies
let getTimelineState: () => any;
let setTimelineState: (state: any) => void;
let getMixerState: () => any;
let setMixerState: (state: any) => void;
let getMediaState: () => any;
let setMediaState: (state: any) => void;
let getDockState: () => any;
let setDockState: (state: any) => void;

// Initialize store references (called from App.tsx)
export function initHistoryStoreRefs(stores: {
  timeline: { getState: () => any; setState: (state: any) => void };
  mixer: { getState: () => any; setState: (state: any) => void };
  media: { getState: () => any; setState: (state: any) => void };
  dock: { getState: () => any; setState: (state: any) => void };
}) {
  getTimelineState = stores.timeline.getState;
  setTimelineState = stores.timeline.setState;
  getMixerState = stores.mixer.getState;
  setMixerState = stores.mixer.setState;
  getMediaState = stores.media.getState;
  setMediaState = stores.media.setState;
  getDockState = stores.dock.getState;
  setDockState = stores.dock.setState;
}

// Deep clone helper (handles most objects, excluding DOM elements and functions)
function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as any;
  if (Array.isArray(obj)) return obj.map(deepClone) as any;

  // Skip cloning DOM elements, HTMLMediaElements, File objects, etc.
  if (obj instanceof Element || obj instanceof HTMLMediaElement || obj instanceof File) {
    return obj; // Return reference, don't clone
  }

  const cloned: any = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const value = (obj as any)[key];
      // Skip functions and DOM elements
      if (typeof value === 'function') continue;
      if (value instanceof Element || value instanceof HTMLMediaElement) {
        cloned[key] = value; // Keep reference
      } else {
        cloned[key] = deepClone(value);
      }
    }
  }
  return cloned;
}

// Create snapshot from current state
function createSnapshot(label: string): StateSnapshot {
  const timeline = getTimelineState?.() || {};
  const mixer = getMixerState?.() || {};
  const media = getMediaState?.() || {};
  const dock = getDockState?.() || {};

  return {
    timestamp: Date.now(),
    label,
    timeline: {
      clips: deepClone(timeline.clips || []),
      tracks: deepClone(timeline.tracks || []),
      selectedClipIds: timeline.selectedClipIds ? [...timeline.selectedClipIds] : [],
      zoom: timeline.zoom || 50,
      scrollX: timeline.scrollX || 0,
    },
    mixer: {
      layers: deepClone(mixer.layers || []),
      selectedLayerId: mixer.selectedLayerId || null,
      gridColumns: mixer.gridColumns || 4,
      gridRows: mixer.gridRows || 4,
      slotGroups: deepClone(mixer.slotGroups || []),
      outputResolution: mixer.outputResolution || { width: 1920, height: 1080 },
    },
    media: {
      files: deepClone(media.files || []),
      compositions: deepClone(media.compositions || []),
      folders: deepClone(media.folders || []),
      selectedIds: [...(media.selectedIds || [])],
      expandedFolderIds: [...(media.expandedFolderIds || [])],
    },
    dock: {
      layout: deepClone(dock.layout || {}),
    },
  };
}

// Apply a snapshot to all stores
function applySnapshot(snapshot: StateSnapshot) {
  if (!snapshot) return;

  // Apply timeline state
  if (setTimelineState) {
    setTimelineState({
      clips: deepClone(snapshot.timeline.clips),
      tracks: deepClone(snapshot.timeline.tracks),
      selectedClipIds: new Set(snapshot.timeline.selectedClipIds || []),
      zoom: snapshot.timeline.zoom,
      scrollX: snapshot.timeline.scrollX,
    });
  }

  // Apply mixer state (preserve source references)
  if (setMixerState) {
    const currentMixer = getMixerState();
    const restoredLayers = snapshot.mixer.layers.map((layer: any) => {
      // Find current layer to preserve video/image source references
      const currentLayer = currentMixer.layers?.find((l: any) => l.id === layer.id);
      return {
        ...deepClone(layer),
        source: currentLayer?.source || layer.source, // Preserve source reference
      };
    });

    setMixerState({
      layers: restoredLayers,
      selectedLayerId: snapshot.mixer.selectedLayerId,
      gridColumns: snapshot.mixer.gridColumns,
      gridRows: snapshot.mixer.gridRows,
      slotGroups: deepClone(snapshot.mixer.slotGroups),
      outputResolution: snapshot.mixer.outputResolution,
    });
  }

  // Apply media state (preserve file references)
  if (setMediaState) {
    const currentMedia = getMediaState();
    const restoredFiles = snapshot.media.files.map((file: any) => {
      const currentFile = currentMedia.files?.find((f: any) => f.id === file.id);
      return {
        ...deepClone(file),
        file: currentFile?.file || file.file, // Preserve File reference
      };
    });

    setMediaState({
      files: restoredFiles,
      compositions: deepClone(snapshot.media.compositions),
      folders: deepClone(snapshot.media.folders),
      selectedIds: [...snapshot.media.selectedIds],
      expandedFolderIds: [...snapshot.media.expandedFolderIds],
    });
  }

  // Apply dock state
  if (setDockState) {
    setDockState({
      layout: deepClone(snapshot.dock.layout),
    });
  }
}

export const useHistoryStore = create<HistoryState>()(
  subscribeWithSelector((set, get) => ({
    undoStack: [],
    redoStack: [],
    currentSnapshot: null,
    maxHistorySize: 50,
    isApplying: false,
    batchId: null,
    batchLabel: null,

    captureSnapshot: (label: string) => {
      const { isApplying, undoStack, currentSnapshot, maxHistorySize, batchId } = get();

      // Don't capture during undo/redo application
      if (isApplying) return;

      // If batching, don't create new snapshots until batch ends
      if (batchId !== null) return;

      const newSnapshot = createSnapshot(label);

      // Push current state to undo stack (if exists)
      if (currentSnapshot) {
        const newUndoStack = [...undoStack, currentSnapshot];
        // Limit history size
        if (newUndoStack.length > maxHistorySize) {
          newUndoStack.shift();
        }
        set({
          undoStack: newUndoStack,
          redoStack: [], // Clear redo stack on new action
          currentSnapshot: newSnapshot,
        });
      } else {
        set({ currentSnapshot: newSnapshot });
      }
    },

    undo: () => {
      const { undoStack, currentSnapshot, redoStack } = get();

      if (undoStack.length === 0) return;

      set({ isApplying: true });

      // Pop from undo stack
      const newUndoStack = [...undoStack];
      const previousSnapshot = newUndoStack.pop()!;

      // Push current to redo stack
      const newRedoStack = currentSnapshot
        ? [...redoStack, currentSnapshot]
        : redoStack;

      // Apply previous state
      applySnapshot(previousSnapshot);

      set({
        undoStack: newUndoStack,
        redoStack: newRedoStack,
        currentSnapshot: previousSnapshot,
        isApplying: false,
      });

      console.log(`[History] Undo: ${previousSnapshot.label}`);
    },

    redo: () => {
      const { redoStack, currentSnapshot, undoStack } = get();

      if (redoStack.length === 0) return;

      set({ isApplying: true });

      // Pop from redo stack
      const newRedoStack = [...redoStack];
      const nextSnapshot = newRedoStack.pop()!;

      // Push current to undo stack
      const newUndoStack = currentSnapshot
        ? [...undoStack, currentSnapshot]
        : undoStack;

      // Apply next state
      applySnapshot(nextSnapshot);

      set({
        undoStack: newUndoStack,
        redoStack: newRedoStack,
        currentSnapshot: nextSnapshot,
        isApplying: false,
      });

      console.log(`[History] Redo: ${nextSnapshot.label}`);
    },

    canUndo: () => get().undoStack.length > 0,
    canRedo: () => get().redoStack.length > 0,

    startBatch: (label: string) => {
      const { batchId, currentSnapshot } = get();
      if (batchId !== null) return; // Already batching

      // Capture initial state before batch
      if (!currentSnapshot) {
        set({ currentSnapshot: createSnapshot('initial') });
      }

      set({
        batchId: Date.now(),
        batchLabel: label,
      });
    },

    endBatch: () => {
      const { batchId, batchLabel, undoStack, currentSnapshot, maxHistorySize } = get();
      if (batchId === null) return;

      // Create final snapshot with batch label
      const finalSnapshot = createSnapshot(batchLabel || 'batch');

      // Push previous state to undo stack
      if (currentSnapshot) {
        const newUndoStack = [...undoStack, currentSnapshot];
        if (newUndoStack.length > maxHistorySize) {
          newUndoStack.shift();
        }
        set({
          undoStack: newUndoStack,
          redoStack: [],
          currentSnapshot: finalSnapshot,
          batchId: null,
          batchLabel: null,
        });
      } else {
        set({
          currentSnapshot: finalSnapshot,
          batchId: null,
          batchLabel: null,
        });
      }
    },

    setIsApplying: (value: boolean) => set({ isApplying: value }),

    clearHistory: () => set({
      undoStack: [],
      redoStack: [],
      currentSnapshot: null,
    }),
  }))
);

// Export convenience functions
export const captureSnapshot = (label: string) => useHistoryStore.getState().captureSnapshot(label);
export const undo = () => useHistoryStore.getState().undo();
export const redo = () => useHistoryStore.getState().redo();
export const startBatch = (label: string) => useHistoryStore.getState().startBatch(label);
export const endBatch = () => useHistoryStore.getState().endBatch();
