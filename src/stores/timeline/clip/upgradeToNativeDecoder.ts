// Upgrade existing video clips to use NativeDecoder when helper connects
// Also handles downgrade when helper disconnects

import type { TimelineClip } from '../../../types';
import { NativeDecoder } from '../../../services/nativeHelper/NativeDecoder';
import { NativeHelperClient } from '../../../services/nativeHelper/NativeHelperClient';
import { useMediaStore } from '../../mediaStore';
import { useTimelineStore } from '../index';
import { Logger } from '../../../services/logger';

const log = Logger.create('NativeUpgrade');

let upgradeInProgress = false;

/**
 * Upgrade all video clips to NativeDecoder.
 * Called when native helper connects + turbo mode is on.
 */
export async function upgradeAllClipsToNativeDecoder(): Promise<void> {
  if (upgradeInProgress) return;
  upgradeInProgress = true;

  try {
    const clips = useTimelineStore.getState().clips;
    const mediaStore = useMediaStore.getState();
    const videoClips = clips.filter(
      (c) => c.source?.type === 'video' && !c.source.nativeDecoder
    );

    if (videoClips.length === 0) return;
    log.info(`Upgrading ${videoClips.length} clips to NativeDecoder`);

    for (const clip of videoClips) {
      if (!NativeHelperClient.isConnected()) break;

      const filePath = await resolveFilePath(clip, mediaStore);
      if (!filePath) {
        log.warn('Cannot resolve file path for clip', { name: clip.name });
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
        log.info('Upgraded clip to NH', { name: clip.name });
      } catch (e) {
        log.warn('Failed to upgrade clip', { name: clip.name, error: e });
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
}

/**
 * Resolve the absolute file path for a clip.
 */
async function resolveFilePath(
  clip: TimelineClip,
  mediaStore: ReturnType<typeof useMediaStore.getState>
): Promise<string | undefined> {
  // 1. Already stored in source
  if (clip.source?.filePath) return clip.source.filePath;

  // 2. From media store
  const mediaFile = clip.source?.mediaFileId
    ? mediaStore.files.find((f) => f.id === clip.source!.mediaFileId)
    : null;
  const fromMedia = mediaFile?.absolutePath;
  if (fromMedia && isAbsolutePath(fromMedia)) return fromMedia;

  // 3. From File object (Electron/browser path property)
  const fromFile = (clip.file as any)?.path as string | undefined;
  if (fromFile && isAbsolutePath(fromFile)) return fromFile;

  // 4. Ask native helper to locate by filename
  try {
    const located = await NativeHelperClient.locateFile(clip.name);
    if (located) return located;
  } catch {
    // ignore
  }

  return undefined;
}

function isAbsolutePath(p: string): boolean {
  return p.startsWith('/') || /^[A-Za-z]:[/\\]/.test(p);
}
