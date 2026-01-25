// MediaStore initialization and auto-save
// NOTE: This module is imported by index.ts for side effects
// We use a lazy getter to avoid circular dependencies

import { useTimelineStore } from '../timeline';
import { fileSystemService } from '../../services/fileSystemService';
import type { Composition, MediaState } from './types';
import { Logger } from '../../services/logger';

const log = Logger.create('MediaStore');

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MediaStore = any;

// Lazy getter to avoid circular dependency
const getMediaStore = (): MediaStore => {
  // Dynamic import at runtime after index.ts has finished initializing
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { useMediaStore } = require('./index');
  return useMediaStore;
};

/**
 * Save current timeline to active composition.
 */
function saveTimelineToActiveComposition(): void {
  const useMediaStore = getMediaStore();
  const { activeCompositionId } = useMediaStore.getState();
  if (activeCompositionId) {
    const timelineStore = useTimelineStore.getState();
    const timelineData = timelineStore.getSerializableState();
    useMediaStore.setState((state: MediaState) => ({
      compositions: state.compositions.map((c: Composition) =>
        c.id === activeCompositionId ? { ...c, timelineData } : c
      ),
    }));
  }
}

/**
 * Trigger timeline save (exported for external use).
 */
export function triggerTimelineSave(): void {
  saveTimelineToActiveComposition();
  log.info('Timeline saved to composition');
}

/**
 * Initialize media store from IndexedDB and file handles.
 */
async function initializeStore(): Promise<void> {
  const useMediaStore = getMediaStore();

  // Initialize file system service
  await fileSystemService.init();

  // Update proxy folder name if restored
  const proxyFolderName = fileSystemService.getProxyFolderName();
  if (proxyFolderName) {
    useMediaStore.setState({ proxyFolderName });
  }

  // Initialize media from IndexedDB
  await useMediaStore.getState().initFromDB();

  // Restore active composition's timeline
  const { activeCompositionId, compositions } = useMediaStore.getState();
  if (activeCompositionId) {
    const activeComp = compositions.find((c: Composition) => c.id === activeCompositionId);
    if (activeComp?.timelineData) {
      log.info('Restoring timeline for:', activeComp.name);
      await useTimelineStore.getState().loadState(activeComp.timelineData);
    }
  }
}

/**
 * Set up auto-save interval.
 */
function setupAutoSave(): void {
  setInterval(() => {
    if ((window as unknown as { __CLEARING_CACHE__?: boolean }).__CLEARING_CACHE__) return;
    saveTimelineToActiveComposition();
  }, 30000); // Every 30 seconds
}

/**
 * Set up beforeunload handler.
 */
function setupBeforeUnload(): void {
  window.addEventListener('beforeunload', () => {
    if ((window as unknown as { __CLEARING_CACHE__?: boolean }).__CLEARING_CACHE__) return;
    saveTimelineToActiveComposition();
  });
}

// Auto-initialize on app load
if (typeof window !== 'undefined') {
  setTimeout(() => {
    initializeStore();
    setupAutoSave();
    setupBeforeUnload();
  }, 100);
}
