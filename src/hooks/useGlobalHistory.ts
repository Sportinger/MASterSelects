// Global history hook - initializes undo/redo system and keyboard shortcuts

import { useEffect, useRef } from 'react';
import { useTimelineStore } from '../stores/timeline';
import { useMixerStore } from '../stores/mixerStore';
import { useMediaStore } from '../stores/mediaStore';
import { useDockStore } from '../stores/dockStore';
import {
  useHistoryStore,
  initHistoryStoreRefs,
  captureSnapshot,
  undo,
  redo,
} from '../stores/historyStore';

// Debounce helper for rapid state changes
function debounce<T extends (...args: any[]) => void>(fn: T, delay: number): T {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  return ((...args: any[]) => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  }) as T;
}

export function useGlobalHistory() {
  const initialized = useRef(false);
  const lastCaptureTime = useRef(0);

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
      mixer: {
        getState: useMixerStore.getState,
        setState: useMixerStore.setState,
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

    // Capture initial state
    captureSnapshot('initial');

    console.log('[History] Undo/redo system initialized');
  }, []);

  // Subscribe to store changes and capture snapshots
  useEffect(() => {
    // Debounced capture to avoid too many snapshots during rapid changes
    const debouncedCapture = debounce((label: string) => {
      const now = Date.now();
      // Minimum 100ms between captures
      if (now - lastCaptureTime.current < 100) return;
      lastCaptureTime.current = now;
      captureSnapshot(label);
    }, 150);

    // Subscribe to timeline changes
    const unsubTimeline = useTimelineStore.subscribe(
      (state) => ({
        clips: state.clips,
        tracks: state.tracks,
        selectedClipIds: state.selectedClipIds,
      }),
      (curr, prev) => {
        // Check if this is during undo/redo
        if (useHistoryStore.getState().isApplying) return;

        // Determine what changed
        if (curr.clips !== prev.clips) {
          if (curr.clips.length !== prev.clips.length) {
            debouncedCapture(curr.clips.length > prev.clips.length ? 'Add clip' : 'Remove clip');
          } else {
            debouncedCapture('Modify clip');
          }
        } else if (curr.tracks !== prev.tracks) {
          debouncedCapture('Modify track');
        } else if (curr.selectedClipIds !== prev.selectedClipIds) {
          // Selection changes are not captured as separate undo steps
        }
      },
      { fireImmediately: false }
    );

    // Subscribe to mixer changes
    const unsubMixer = useMixerStore.subscribe(
      (state) => ({
        layers: state.layers,
        selectedLayerId: state.selectedLayerId,
        gridColumns: state.gridColumns,
        gridRows: state.gridRows,
        slotGroups: state.slotGroups,
      }),
      (curr, prev) => {
        if (useHistoryStore.getState().isApplying) return;

        if (curr.layers !== prev.layers) {
          // Check what changed in layers
          const currLayerState = JSON.stringify(curr.layers.map(l => ({
            id: l?.id,
            opacity: l?.opacity,
            blendMode: l?.blendMode,
            visible: l?.visible,
            position: l?.position,
            scale: l?.scale,
            rotation: l?.rotation,
            effects: l?.effects?.length,
            hasSource: !!l?.source,
          })));
          const prevLayerState = JSON.stringify(prev.layers.map(l => ({
            id: l?.id,
            opacity: l?.opacity,
            blendMode: l?.blendMode,
            visible: l?.visible,
            position: l?.position,
            scale: l?.scale,
            rotation: l?.rotation,
            effects: l?.effects?.length,
            hasSource: !!l?.source,
          })));

          if (currLayerState !== prevLayerState) {
            debouncedCapture('Modify layer');
          }
        } else if (curr.gridColumns !== prev.gridColumns || curr.gridRows !== prev.gridRows) {
          debouncedCapture('Change grid');
        } else if (curr.slotGroups !== prev.slotGroups) {
          debouncedCapture('Modify slot groups');
        }
      },
      { fireImmediately: false }
    );

    // Subscribe to media changes
    const unsubMedia = useMediaStore.subscribe(
      (state) => ({
        files: state.files,
        compositions: state.compositions,
        folders: state.folders,
      }),
      (curr, prev) => {
        if (useHistoryStore.getState().isApplying) return;

        if (curr.files !== prev.files) {
          debouncedCapture(curr.files.length > prev.files.length ? 'Import file' : 'Remove file');
        } else if (curr.compositions !== prev.compositions) {
          debouncedCapture('Modify composition');
        } else if (curr.folders !== prev.folders) {
          debouncedCapture('Modify folder');
        }
      },
      { fireImmediately: false }
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
      unsubTimeline();
      unsubMixer();
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
        if (useHistoryStore.getState().canUndo()) {
          undo();
        }
        return;
      }

      // Ctrl+Shift+Z or Cmd+Shift+Z for redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        if (useHistoryStore.getState().canRedo()) {
          redo();
        }
        return;
      }

      // Ctrl+Y or Cmd+Y for redo (alternative)
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault();
        if (useHistoryStore.getState().canRedo()) {
          redo();
        }
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
