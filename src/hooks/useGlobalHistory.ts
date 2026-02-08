// Global history hook - initializes undo/redo system and keyboard shortcuts

import { useEffect, useRef } from 'react';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { useDockStore } from '../stores/dockStore';
import {
  useHistoryStore,
  initHistoryStoreRefs,
  setHistoryCallbacks,
  captureSnapshot,
  undo,
  redo,
} from '../stores/historyStore';
import { Logger } from '../services/logger';

const log = Logger.create('History');

// Shallow equality for subscription selectors — prevents callback from firing
// on unrelated store changes (e.g. playheadPosition updates at 60fps)
function shallowEqual<T extends Record<string, unknown>>(a: T, b: T): boolean {
  if (a === b) return true;
  const keysA = Object.keys(a);
  if (keysA.length !== Object.keys(b).length) return false;
  for (const key of keysA) {
    if (!Object.is(a[key], b[key])) return false;
  }
  return true;
}

export function useGlobalHistory() {
  const initialized = useRef(false);
  const lastCaptureTime = useRef(0);
  const pendingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingLabel = useRef('');
  const suppressUntil = useRef(0);

  // Initialize store references
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    // Initialize history store with store references
    initHistoryStoreRefs({
      timeline: {
        getState: useTimelineStore.getState,
        setState: useTimelineStore.setState,
      },
      media: {
        getState: useMediaStore.getState,
        setState: useMediaStore.setState,
      },
      dock: {
        getState: useDockStore.getState,
        setState: useDockStore.setState,
      },
    });

    // Register callbacks so undo/redo can flush pending captures
    setHistoryCallbacks({
      flushPendingCapture: () => {
        if (pendingTimer.current) {
          clearTimeout(pendingTimer.current);
          const label = pendingLabel.current;
          pendingTimer.current = null;
          pendingLabel.current = '';
          // Execute the capture immediately so the state isn't lost
          lastCaptureTime.current = Date.now();
          captureSnapshot(label || 'pending');
        }
      },
      suppressCaptures: () => {
        suppressUntil.current = Date.now() + 250;
      },
    });

    // Capture initial state
    captureSnapshot('initial');

    log.info('Undo/redo system initialized');
  }, []);

  // Subscribe to store changes and capture snapshots
  useEffect(() => {
    // Debounced capture — stores timer ID and label so undo/redo can flush it
    const debouncedCapture = (label: string) => {
      if (pendingTimer.current) clearTimeout(pendingTimer.current);
      pendingLabel.current = label;
      pendingTimer.current = setTimeout(() => {
        pendingTimer.current = null;
        pendingLabel.current = '';

        // Suppress captures shortly after undo/redo to prevent cascade re-captures
        if (Date.now() < suppressUntil.current) return;

        // Don't capture during undo/redo application
        if (useHistoryStore.getState().isApplying) return;

        const now = Date.now();
        // Minimum 100ms between captures
        if (now - lastCaptureTime.current < 100) return;
        lastCaptureTime.current = now;
        captureSnapshot(label);
      }, 150);
    };

    // Subscribe to timeline changes (clips, tracks, keyframes, markers)
    // Using shallowEqual so callback only fires when these specific properties change,
    // not on every store update (playheadPosition, isPlaying, etc.)
    const unsubTimeline = useTimelineStore.subscribe(
      (state) => ({
        clips: state.clips,
        tracks: state.tracks,
        clipKeyframes: state.clipKeyframes,
        markers: state.markers,
      }),
      (curr, prev) => {
        if (useHistoryStore.getState().isApplying) return;

        // Skip captures during mask dragging — vertex updates fire at 60fps
        // and would cause expensive deep-clone snapshots every 150ms
        if (useTimelineStore.getState().maskDragging) return;

        if (curr.clips !== prev.clips) {
          if (curr.clips.length !== prev.clips.length) {
            debouncedCapture(curr.clips.length > prev.clips.length ? 'Add clip' : 'Remove clip');
          } else {
            debouncedCapture('Modify clip');
          }
        } else if (curr.tracks !== prev.tracks) {
          debouncedCapture('Modify track');
        } else if (curr.clipKeyframes !== prev.clipKeyframes) {
          debouncedCapture('Modify keyframes');
        } else if (curr.markers !== prev.markers) {
          debouncedCapture('Modify markers');
        }
      },
      { equalityFn: shallowEqual, fireImmediately: false }
    );

    // Subscribe to media changes
    const unsubMedia = useMediaStore.subscribe(
      (state) => ({
        files: state.files,
        compositions: state.compositions,
        folders: state.folders,
        textItems: state.textItems,
        solidItems: state.solidItems,
      }),
      (curr, prev) => {
        if (useHistoryStore.getState().isApplying) return;

        if (curr.files !== prev.files) {
          debouncedCapture(curr.files.length > prev.files.length ? 'Import file' : 'Remove file');
        } else if (curr.compositions !== prev.compositions) {
          debouncedCapture('Modify composition');
        } else if (curr.folders !== prev.folders) {
          debouncedCapture('Modify folder');
        } else if (curr.textItems !== prev.textItems) {
          debouncedCapture('Modify text items');
        } else if (curr.solidItems !== prev.solidItems) {
          debouncedCapture('Modify solid items');
        }
      },
      { equalityFn: shallowEqual, fireImmediately: false }
    );

    // Subscribe to dock changes
    const unsubDock = useDockStore.subscribe(
      (state) => state.layout,
      (curr, prev) => {
        if (useHistoryStore.getState().isApplying) return;
        if (curr !== prev) {
          debouncedCapture('Change layout');
        }
      },
      { fireImmediately: false }
    );

    return () => {
      if (pendingTimer.current) {
        clearTimeout(pendingTimer.current);
        pendingTimer.current = null;
      }
      unsubTimeline();
      unsubMedia();
      unsubDock();
    };
  }, []);

  // Global keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target as HTMLElement)?.isContentEditable
      ) {
        return;
      }

      // Ctrl+Z or Cmd+Z for undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
        return;
      }

      // Ctrl+Shift+Z or Cmd+Shift+Z for redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        redo();
        return;
      }

      // Ctrl+Y or Cmd+Y for redo (alternative)
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault();
        redo();
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return {
    undo,
    redo,
    canUndo: useHistoryStore((state) => state.undoStack.length > 0),
    canRedo: useHistoryStore((state) => state.redoStack.length > 0),
  };
}
