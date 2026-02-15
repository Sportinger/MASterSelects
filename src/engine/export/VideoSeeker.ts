// Video seeking and ready-state management for export

import { Logger } from '../../services/logger';
import type { ExportClipState, FrameContext } from './types';

const log = Logger.create('VideoSeeker');
import { ParallelDecodeManager } from '../ParallelDecodeManager';

/**
 * Seek all clips to the specified time for frame export.
 * Uses FrameContext for O(1) lookups instead of repeated getState() calls.
 */
export async function seekAllClipsToTime(
  ctx: FrameContext,
  clipStates: Map<string, ExportClipState>,
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Promise<void> {
  const { time } = ctx;

  // PARALLEL DECODE MODE - no HTMLVideoElement seeking needed!
  // ParallelDecoder provides VideoFrames directly, much faster than seeking videos
  if (useParallelDecode && parallelDecoder) {
    await parallelDecoder.prefetchFramesForTime(time);
    parallelDecoder.advanceToTime(time);
    return;
  }

  // SEQUENTIAL MODE (single clip only)
  await seekSequentialMode(ctx, clipStates);
}

async function seekSequentialMode(
  ctx: FrameContext,
  clipStates: Map<string, ExportClipState>
): Promise<void> {
  const { time, clipsAtTime, trackMap, getSourceTimeForClip, getInterpolatedSpeed } = ctx;
  const seekPromises: Promise<void>[] = [];

  for (const clip of clipsAtTime) {
    const track = trackMap.get(clip.trackId);
    if (!track?.visible) continue;

    // Handle nested composition clips
    if (clip.isComposition && clip.nestedClips && clip.nestedTracks) {
      const clipLocalTime = time - clip.startTime;
      const nestedTime = clipLocalTime + (clip.inPoint || 0);

      for (const nestedClip of clip.nestedClips) {
        if (nestedTime >= nestedClip.startTime && nestedTime < nestedClip.startTime + nestedClip.duration) {
          const nestedLocalTime = nestedTime - nestedClip.startTime;
          const nestedClipTime = nestedClip.reversed
            ? nestedClip.outPoint - nestedLocalTime
            : nestedLocalTime + nestedClip.inPoint;

          if (nestedClip.source?.webCodecsPlayer && !nestedClip.source.videoElement) {
            // WebCodecs Full Mode: seek via player
            seekPromises.push(nestedClip.source.webCodecsPlayer.seekAsync(nestedClipTime).then(() => {}));
          } else if (nestedClip.source?.videoElement) {
            seekPromises.push(seekVideo(nestedClip.source.videoElement, nestedClipTime));
          }
        }
      }
      continue;
    }

    // Handle regular video clips
    if (clip.source?.type === 'video' && (clip.source.videoElement || clip.source.webCodecsPlayer)) {
      const clipLocalTime = time - clip.startTime;

      // Calculate clip time (handles speed keyframes and reversed clips)
      let clipTime: number;
      try {
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
      } catch {
        clipTime = clip.reversed
          ? clip.outPoint - clipLocalTime
          : clipLocalTime + clip.inPoint;
        clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, clipTime));
      }

      const clipState = clipStates.get(clip.id);

      if (clipState?.isSequential && clipState.webCodecsPlayer) {
        // FAST MODE: WebCodecs sequential decoding
        seekPromises.push(clipState.webCodecsPlayer.seekDuringExport(clipTime));
      } else if (clip.source.webCodecsPlayer && !clip.source.videoElement) {
        // WebCodecs Full Mode: seek via player (no HTMLVideoElement)
        seekPromises.push(clip.source.webCodecsPlayer.seekAsync(clipTime).then(() => {}));
      } else if (clip.source.videoElement) {
        // PRECISE MODE: HTMLVideoElement seeking
        seekPromises.push(seekVideo(clip.source.videoElement, clipTime));
      }
    }
  }

  if (seekPromises.length > 0) {
    await Promise.all(seekPromises);
  }
}

/**
 * Seek a video element to a specific time with frame-accurate waiting.
 */
export function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const targetTime = Math.max(0, Math.min(time, video.duration || 0));

    const timeout = setTimeout(() => {
      log.warn(`Seek timeout at ${targetTime}`);
      resolve();
    }, 500); // 500ms for AV1 and other slow-decoding codecs

    const waitForFrame = () => {
      // Check for requestVideoFrameCallback without type narrowing issues
      const hasRvfc = typeof (video as any).requestVideoFrameCallback === 'function';
      if (hasRvfc) {
        (video as any).requestVideoFrameCallback(() => {
          clearTimeout(timeout);
          resolve();
        });
      } else {
        // Fallback: wait for readyState using setTimeout for export reliability
        let retries = 0;
        const maxRetries = 30;

        const waitForReady = () => {
          retries++;
          if (!video.seeking && video.readyState >= 3) {
            clearTimeout(timeout);
            // Use setTimeout instead of requestAnimationFrame for export
            setTimeout(() => {
              setTimeout(() => resolve(), 16);
            }, 16);
          } else if (retries < maxRetries) {
            setTimeout(waitForReady, 16);
          } else {
            clearTimeout(timeout);
            resolve();
          }
        };
        waitForReady();
      }
    };

    // If already at correct time, still wait for frame callback
    if (Math.abs(video.currentTime - targetTime) < 0.01 && !video.seeking && video.readyState >= 3) {
      waitForFrame();
      return;
    }

    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      waitForFrame();
    };

    video.addEventListener('seeked', onSeeked);
    video.currentTime = targetTime;
  });
}

/**
 * Wait for all video clips at a given time to have their frames ready.
 * Uses FrameContext for O(1) lookups.
 */
export async function waitForAllVideosReady(
  ctx: FrameContext,
  clipStates: Map<string, ExportClipState>,
  _parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Promise<void> {
  // PARALLEL DECODE MODE - frames are already ready from prefetchFramesForTime
  // No need to wait for HTMLVideoElement
  if (useParallelDecode) {
    return;
  }

  // SEQUENTIAL MODE - wait for WebCodecs player (not HTMLVideoElement)
  const { clipsAtTime, trackMap } = ctx;

  const videoClips = clipsAtTime.filter(clip => {
    const track = trackMap.get(clip.trackId);
    return track?.visible && clip.source?.type === 'video';
  });

  if (videoClips.length === 0) return;

  // Only wait for sequential WebCodecs clips
  for (const clip of videoClips) {
    const clipState = clipStates.get(clip.id);
    if (clipState?.isSequential && clipState.webCodecsPlayer) {
      // WebCodecs player handles its own frame readiness
      continue;
    }
  }
}
