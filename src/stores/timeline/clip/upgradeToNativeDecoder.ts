// Upgrade existing video clips to use NativeDecoder when helper connects
// Also handles downgrade when helper disconnects
// Watches for clip changes to upgrade newly added clips automatically

import type { TimelineClip } from '../../../types';
import { NativeDecoder } from '../../../services/nativeHelper/NativeDecoder';
import { NativeHelperClient } from '../../../services/nativeHelper/NativeHelperClient';
import { useMediaStore } from '../../mediaStore';
import { useTimelineStore } from '../index';
import { Logger } from '../../../services/logger';

const log = Logger.create('NativeUpgrade');

let upgradeInProgress = false;
// Track clips that failed to upgrade (no path found) — don't retry endlessly
const failedClipIds = new Set<string>();

/**
 * Upgrade all video clips to NativeDecoder.
 * Called when native helper connects + turbo mode is on,
 * and re-called automatically when new clips appear.
 */
export async function upgradeAllClipsToNativeDecoder(): Promise<void> {
  if (upgradeInProgress) return;
  upgradeInProgress = true;

  try {
    const clips = useTimelineStore.getState().clips;
    const mediaStore = useMediaStore.getState();

    const videoClips = clips.filter(
      (c) => c.source?.type === 'video' && !c.source.nativeDecoder && !failedClipIds.has(c.id)
    );

    if (videoClips.length === 0) {
      return;
    }
    log.info(`Upgrading ${videoClips.length} video clips to NativeDecoder`);

    for (const clip of videoClips) {
      if (!NativeHelperClient.isConnected()) break;

      const filePath = await resolveFilePath(clip, mediaStore);
      if (!filePath) {
        log.warn(`[${clip.name}] Cannot resolve file path — skipping`);
        failedClipIds.add(clip.id);
        continue;
      }

      try {
        const nativeDecoder = await NativeDecoder.open(filePath);
        // Decode frame 0 so preview isn't black
        await nativeDecoder.seekToFrame(0);

        // Update clip in store
        const currentClips = useTimelineStore.getState().clips;
        useTimelineStore.setState({
          clips: currentClips.map((c) => {
            if (c.id !== clip.id || !c.source) return c;
            return {
              ...c,
              source: { ...c.source, nativeDecoder, filePath },
            };
          }),
        });
        log.info(`Upgraded [${clip.name}] to NH (${filePath})`);
      } catch (e) {
        log.warn(`Failed to upgrade [${clip.name}]`, e);
        failedClipIds.add(clip.id);
      }
    }
  } finally {
    upgradeInProgress = false;
  }
}

/**
 * Remove NativeDecoder from all clips (fallback to WC/HTML).
 * Called when native helper disconnects or turbo mode is turned off.
 */
export function downgradeAllClipsFromNativeDecoder(): void {
  const clips = useTimelineStore.getState().clips;
  const hasNH = clips.some((c) => c.source?.nativeDecoder);
  if (!hasNH) return;

  log.info('Downgrading all clips from NativeDecoder');
  const currentClips = useTimelineStore.getState().clips;
  useTimelineStore.setState({
    clips: currentClips.map((c) => {
      if (!c.source?.nativeDecoder) return c;
      // Close the decoder
      c.source.nativeDecoder.close().catch(() => {});
      // Remove nativeDecoder but keep filePath for future upgrades
      const { nativeDecoder: _, ...restSource } = c.source;
      return { ...c, source: restSource };
    }),
  });
  // Clear failed set so re-upgrade can be attempted
  failedClipIds.clear();
}

// --- Auto-watch: subscribe to clip changes and upgrade new clips ---

let watcherActive = false;
let unsubscribe: (() => void) | null = null;

/**
 * Start watching for new video clips and auto-upgrade them.
 * Called when helper connects + turbo mode on.
 */
export function startClipWatcher(): void {
  if (watcherActive) return;
  watcherActive = true;

  let prevClipCount = useTimelineStore.getState().clips.length;

  unsubscribe = useTimelineStore.subscribe((state) => {
    const clips = state.clips;
    // Only re-check when clip count changes (new clip added or project loaded)
    if (clips.length !== prevClipCount) {
      prevClipCount = clips.length;
      const hasUnupgraded = clips.some(
        (c) => c.source?.type === 'video' && !c.source.nativeDecoder && !failedClipIds.has(c.id)
      );
      if (hasUnupgraded && NativeHelperClient.isConnected()) {
        void upgradeAllClipsToNativeDecoder();
      }
    }
  });

  log.info('Clip watcher started — will auto-upgrade new video clips');
}

/**
 * Stop watching for clip changes.
 * Called when helper disconnects or turbo mode off.
 */
export function stopClipWatcher(): void {
  if (!watcherActive) return;
  watcherActive = false;
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
  log.info('Clip watcher stopped');
}

/**
 * Resolve the absolute file path for a clip.
 */
async function resolveFilePath(
  clip: TimelineClip,
  mediaStore: ReturnType<typeof useMediaStore.getState>
): Promise<string | undefined> {
  // 1. Already stored in source
  if (clip.source?.filePath) {
    log.info(`[${clip.name}] path from source.filePath: ${clip.source.filePath}`);
    return clip.source.filePath;
  }

  // 2. From media store
  const mediaFile = clip.source?.mediaFileId
    ? mediaStore.files.find((f) => f.id === clip.source!.mediaFileId)
    : null;
  const fromMedia = mediaFile?.absolutePath;
  if (fromMedia && isAbsolutePath(fromMedia)) {
    log.info(`[${clip.name}] path from mediaStore: ${fromMedia}`);
    return fromMedia;
  }

  // 3. From File object (Electron/browser path property)
  const fromFile = (clip.file as any)?.path as string | undefined;
  if (fromFile && isAbsolutePath(fromFile)) {
    log.info(`[${clip.name}] path from File.path: ${fromFile}`);
    return fromFile;
  }

  log.info(`[${clip.name}] no local path — trying locate (recursive search)...`);

  // 4. Ask native helper to locate by filename (searches recursively in user dirs)
  try {
    const located = await NativeHelperClient.locateFile(clip.name);
    if (located) {
      log.info(`[${clip.name}] located via helper: ${located}`);
      return located;
    }
    log.warn(`[${clip.name}] file not found by helper`);
  } catch (e) {
    log.warn(`[${clip.name}] locate failed:`, e);
  }

  return undefined;
}

function isAbsolutePath(p: string): boolean {
  return p.startsWith('/') || /^[A-Za-z]:[/\\]/.test(p);
}
