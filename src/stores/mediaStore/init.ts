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

// Cached store reference - populated after first access
let cachedMediaStore: MediaStore | null = null;

// Lazy getter to avoid circular dependency
const getMediaStore = (): MediaStore | null => {
  if (cachedMediaStore) return cachedMediaStore;

  // Try to get the store - it may not be ready yet during initial load
  try {
    // Use dynamic import workaround for ESM
    // The store is accessed through the global module cache
    const module = (globalThis as any).__mediaStoreModule;
    if (module?.useMediaStore) {
      cachedMediaStore = module.useMediaStore;
      return cachedMediaStore;
    }
  } catch {
    // Store not ready yet
  }
  return null;
};

/**
 * Save current timeline to active composition.
 */
function saveTimelineToActiveComposition(): void {
  const useMediaStore = getMediaStore();
  if (!useMediaStore) return; // Store not ready yet
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
  if (!useMediaStore) {
    log.warn('Media store not ready during initialization');
    return;
  }

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

      // Sync transcript and analysis status from restored clips to MediaFiles (for badge display)
      syncStatusFromClips(useMediaStore);
    }
  }
}

/**
 * Scan timeline clips for transcripts and analysis and propagate status to MediaFiles.
 * This ensures the "T" and "A" badges show correctly after project reload.
 */
function syncStatusFromClips(useMediaStore: MediaStore): void {
  const clips = useTimelineStore.getState().clips;
  const transcriptMap = new Map<string, import('../../types').TranscriptWord[]>();
  const analysisSet = new Set<string>();

  for (const clip of clips) {
    const mediaFileId = clip.source?.mediaFileId || clip.mediaFileId;
    if (!mediaFileId) continue;

    // Transcript sync
    if (clip.transcriptStatus === 'ready' && clip.transcript?.length) {
      const existing = transcriptMap.get(mediaFileId);
      if (existing) {
        for (const word of clip.transcript) {
          const dup = existing.some(
            (w: import('../../types').TranscriptWord) => Math.abs(w.start - word.start) < 0.05 && Math.abs(w.end - word.end) < 0.05
          );
          if (!dup) existing.push(word);
        }
      } else {
        transcriptMap.set(mediaFileId, [...clip.transcript]);
      }
    }

    // Analysis sync
    if (clip.analysisStatus === 'ready' || clip.sceneDescriptionStatus === 'ready') {
      analysisSet.add(mediaFileId);
    }
  }

  if (transcriptMap.size === 0 && analysisSet.size === 0) return;

  // Sort each transcript by start time
  for (const [, words] of transcriptMap) {
    words.sort((a, b) => a.start - b.start);
  }

  useMediaStore.setState((state: MediaState) => ({
    files: state.files.map((f: { id: string; analysisStatus?: string }) => {
      const transcript = transcriptMap.get(f.id);
      const hasAnalysis = analysisSet.has(f.id);
      if (!transcript && !hasAnalysis) return f;
      return {
        ...f,
        ...(transcript && { transcriptStatus: 'ready' as const, transcript }),
        ...(hasAnalysis && f.analysisStatus !== 'ready' && { analysisStatus: 'ready' as const }),
      };
    }),
  }));

  const total = transcriptMap.size + analysisSet.size;
  log.info(`Synced status for ${total} media file(s) (T:${transcriptMap.size}, A:${analysisSet.size})`);
}

/**
 * Persist textItems and solidItems to localStorage on change.
 */
function setupItemPersistence(): void {
  const useMediaStore = getMediaStore();
  if (!useMediaStore) return;

  // Subscribe to textItems changes
  useMediaStore.subscribe(
    (state: MediaState) => state.textItems,
    (textItems: MediaState['textItems']) => {
      try {
        localStorage.setItem('ms-textItems', JSON.stringify(textItems));
      } catch { /* quota exceeded or unavailable */ }
    }
  );

  // Subscribe to solidItems changes
  useMediaStore.subscribe(
    (state: MediaState) => state.solidItems,
    (solidItems: MediaState['solidItems']) => {
      try {
        localStorage.setItem('ms-solidItems', JSON.stringify(solidItems));
      } catch { /* quota exceeded or unavailable */ }
    }
  );

  log.info('Item persistence setup complete');
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
    setupItemPersistence();
  }, 100);
}
