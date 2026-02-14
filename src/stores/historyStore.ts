// Global undo/redo history store
// Captures snapshots of all undoable state across stores

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { Logger } from '../services/logger';
import type { TimelineClip, TimelineTrack, Layer, Keyframe } from '../types';
import type { MediaFile, Composition, MediaFolder, TextItem, SolidItem } from './mediaStore/types';
import type { TimelineMarker } from './timeline/types';
import type { DockNode } from '../types/dock';
import { SnapshotManager } from '../engine/structuralSharing/SnapshotManager';
import type { HistorySnapshotV2 } from '../engine/structuralSharing/types';

const log = Logger.create('History');

// Singleton SnapshotManager for structural sharing
const snapshotManager = new SnapshotManager();
let lastV2Snapshot: HistorySnapshotV2 | undefined;

// Snapshot of undoable state from all stores
interface StateSnapshot {
  timestamp: number;
  label: string; // Description of the action (for debugging)

  // Timeline state (including layers since they moved here from mixerStore)
  timeline: {
    clips: TimelineClip[];
    tracks: TimelineTrack[];
    selectedClipIds: string[];
    zoom: number;
    scrollX: number;
    layers: Layer[];
    selectedLayerId: string | null;
    clipKeyframes: Record<string, Keyframe[]>;
    markers: TimelineMarker[];
  };

  // Media state
  media: {
    files: MediaFile[];
    compositions: Composition[];
    folders: MediaFolder[];
    selectedIds: string[];
    expandedFolderIds: string[];
    textItems: TextItem[];
    solidItems: SolidItem[];
  };

  // Dock layout state
  dock: {
    layout: DockNode | null;
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

// Store state types for dynamic references
interface TimelineStoreState {
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  selectedClipIds: Set<string>;
  zoom: number;
  scrollX: number;
  layers: Layer[];
  selectedLayerId: string | null;
  clipKeyframes: Map<string, Keyframe[]>;
  markers: TimelineMarker[];
}

interface MediaStoreState {
  files: MediaFile[];
  compositions: Composition[];
  folders: MediaFolder[];
  selectedIds: string[];
  expandedFolderIds: string[];
  textItems: TextItem[];
  solidItems: SolidItem[];
}

// Callback to flush pending debounced captures before undo/redo (set by useGlobalHistory)
// Flush = execute the pending capture immediately so its state isn't lost
let flushPendingCaptureCallback: (() => void) | null = null;
let suppressCapturesCallback: (() => void) | null = null;
export function setHistoryCallbacks(callbacks: {
  flushPendingCapture: () => void;
  suppressCaptures: () => void;
}) {
  flushPendingCaptureCallback = callbacks.flushPendingCapture;
  suppressCapturesCallback = callbacks.suppressCaptures;
}

// Import stores dynamically to avoid circular dependencies
let getTimelineState: (() => TimelineStoreState) | undefined;
let setTimelineState: ((state: Partial<TimelineStoreState>) => void) | undefined;
let getMediaState: (() => MediaStoreState) | undefined;
let setMediaState: ((state: Partial<MediaStoreState>) => void) | undefined;
let getDockState: (() => any) | undefined;
let setDockState: ((state: any) => void) | undefined;

// Initialize store references (called from useGlobalHistory)
export function initHistoryStoreRefs(stores: {
  timeline: { getState: () => TimelineStoreState; setState: (state: Partial<TimelineStoreState>) => void };
  media: { getState: () => MediaStoreState; setState: (state: Partial<MediaStoreState>) => void };
  dock: { getState: () => any; setState: (state: any) => void };
}) {
  getTimelineState = stores.timeline.getState;
  setTimelineState = stores.timeline.setState;
  getMediaState = stores.media.getState;
  setMediaState = stores.media.setState;
  getDockState = stores.dock.getState;
  setDockState = stores.dock.setState;
}

// Deep clone helper (handles most objects, excluding DOM elements and functions)
// Uses a WeakSet to detect circular references and avoid infinite recursion
function deepClone<T>(obj: T, seen?: WeakSet<object>): T {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as T;

  // Skip cloning DOM elements, HTMLMediaElements, File objects, etc.
  if (obj instanceof Element || obj instanceof HTMLMediaElement || obj instanceof File) {
    return obj; // Return reference, don't clone
  }

  // Circular reference detection
  if (!seen) seen = new WeakSet();
  if (seen.has(obj as object)) return obj; // Break cycle, return reference
  seen.add(obj as object);

  if (Array.isArray(obj)) return obj.map(item => deepClone(item, seen)) as T;

  const cloned = {} as T;
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const value = obj[key];
      // Skip functions and DOM elements
      if (typeof value === 'function') continue;
      if (value instanceof Element || value instanceof HTMLMediaElement) {
        cloned[key] = value; // Keep reference
      } else {
        cloned[key] = deepClone(value, seen);
      }
    }
  }
  return cloned;
}

// Create snapshot from current state
function createSnapshot(label: string): StateSnapshot {
  const timeline = getTimelineState?.() || ({} as any);
  const media = getMediaState?.() || ({} as any);
  const dock = getDockState?.() || ({} as any);

  // Structural sharing: only clone changed clips, share unchanged refs
  const currentKeyframes: Map<string, Keyframe[]> = timeline.clipKeyframes instanceof Map
    ? timeline.clipKeyframes
    : new Map();

  const v2Snapshot = snapshotManager.createSnapshot(
    label,
    timeline.clips || [],
    timeline.tracks || [],
    currentKeyframes,
    timeline.markers || [],
    lastV2Snapshot
  );
  lastV2Snapshot = v2Snapshot;

  return {
    timestamp: v2Snapshot.timestamp,
    label,
    timeline: {
      clips: v2Snapshot.clips as any, // SerializedClipState[] used as TimelineClip[] (serialized subset)
      tracks: v2Snapshot.tracks,
      selectedClipIds: timeline.selectedClipIds ? [...timeline.selectedClipIds] : [],
      zoom: timeline.zoom || 50,
      scrollX: timeline.scrollX || 0,
      layers: deepClone((timeline.layers || []).filter(Boolean)),
      selectedLayerId: timeline.selectedLayerId || null,
      clipKeyframes: v2Snapshot.clipKeyframes,
      markers: v2Snapshot.markers,
    },
    media: {
      files: deepClone(media.files || []),
      compositions: deepClone(media.compositions || []),
      folders: deepClone(media.folders || []),
      selectedIds: [...(media.selectedIds || [])],
      expandedFolderIds: [...(media.expandedFolderIds || [])],
      textItems: deepClone(media.textItems || []),
      solidItems: deepClone(media.solidItems || []),
    },
    dock: {
      layout: deepClone(dock.layout || {}),
    },
  };
}

// Apply a snapshot to all stores
function applySnapshot(snapshot: StateSnapshot) {
  if (!snapshot) return;

  // Apply timeline state (including layers)
  if (setTimelineState && getTimelineState) {
    const currentTimeline = getTimelineState();
    // Preserve source references for layers (filter out undefined entries from snapshots)
    const restoredLayers = (snapshot.timeline.layers || []).filter(Boolean).map((layer) => {
      const currentLayer = (currentTimeline.layers || []).find((l) => l?.id === layer.id);
      return {
        ...deepClone(layer),
        source: currentLayer?.source || layer.source,
      };
    });

    // Convert plain object back to Map<string, Keyframe[]>
    const restoredKeyframes = new Map<string, Keyframe[]>();
    if (snapshot.timeline.clipKeyframes) {
      for (const [clipId, kfs] of Object.entries(snapshot.timeline.clipKeyframes)) {
        restoredKeyframes.set(clipId, deepClone(kfs));
      }
    }

    setTimelineState({
      clips: deepClone(snapshot.timeline.clips),
      tracks: deepClone(snapshot.timeline.tracks),
      selectedClipIds: new Set(snapshot.timeline.selectedClipIds || []),
      zoom: snapshot.timeline.zoom,
      scrollX: snapshot.timeline.scrollX,
      layers: restoredLayers,
      selectedLayerId: snapshot.timeline.selectedLayerId,
      clipKeyframes: restoredKeyframes,
      markers: deepClone(snapshot.timeline.markers || []),
    });
  }

  // Apply media state (preserve file references)
  if (setMediaState && getMediaState) {
    const currentMedia = getMediaState();
    const restoredFiles = (snapshot.media.files || []).filter(Boolean).map((file) => {
      const currentFile = (currentMedia.files || []).find((f) => f?.id === file.id);
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
      textItems: deepClone(snapshot.media.textItems || []),
      solidItems: deepClone(snapshot.media.solidItems || []),
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
      // End any stuck batch first (safety: lost mouseup etc.)
      if (get().batchId !== null) {
        get().endBatch();
      }

      // Flush pending debounced capture so we don't lose the latest state
      flushPendingCaptureCallback?.();

      // Re-read stacks after flush may have pushed new entries
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

      // Suppress auto-captures for 200ms to prevent cascading state changes from re-capturing
      suppressCapturesCallback?.();

      log.debug(`Undo: ${previousSnapshot.label} (stack: ${newUndoStack.length})`);
    },

    redo: () => {
      // End any stuck batch first
      if (get().batchId !== null) {
        get().endBatch();
      }

      // Flush pending debounced capture
      flushPendingCaptureCallback?.();

      // Re-read stacks after flush
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

      // Suppress auto-captures for 200ms
      suppressCapturesCallback?.();

      log.debug(`Redo: ${nextSnapshot.label} (stack: ${newRedoStack.length})`);
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

/** Track a clip as changed for structural sharing (call before captureSnapshot) */
export const trackClipChange = (clipId: string) => snapshotManager.trackChange(clipId);

/** Track multiple clip changes for structural sharing */
export const trackClipChanges = (clipIds: string[]) => snapshotManager.trackChanges(clipIds);
