// Video seeking and ready-state management for export

import type { ExportClipState, FrameContext } from './types';
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
  const { time, clipsAtTime, trackMap } = ctx;

  // PARALLEL DECODE MODE
  if (useParallelDecode && parallelDecoder) {
    await parallelDecoder.prefetchFramesForTime(time);

    // Handle composition clips not in parallel decode
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
            if (nestedClip.source?.videoElement) {
              // Skip if parallel decoder handles this
              if (parallelDecoder.hasClip(nestedClip.id)) continue;

              const nestedLocalTime = nestedTime - nestedClip.startTime;
              const nestedClipTime = nestedClip.reversed
                ? nestedClip.outPoint - nestedLocalTime
                : nestedLocalTime + nestedClip.inPoint;
              seekPromises.push(seekVideo(nestedClip.source.videoElement, nestedClipTime));
            }
          }
        }
      }
    }

    if (seekPromises.length > 0) {
      await Promise.all(seekPromises);
    }

    parallelDecoder.advanceToTime(time);
    return;
  }

  // SEQUENTIAL MODE
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
          if (nestedClip.source?.videoElement) {
            const nestedLocalTime = nestedTime - nestedClip.startTime;
            const nestedClipTime = nestedClip.reversed
              ? nestedClip.outPoint - nestedLocalTime
              : nestedLocalTime + nestedClip.inPoint;
            seekPromises.push(seekVideo(nestedClip.source.videoElement, nestedClipTime));
          }
        }
      }
      continue;
    }

    // Handle regular video clips
    if (clip.source?.type === 'video' && clip.source.videoElement) {
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
      } else {
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
      console.warn('[FrameExporter] Seek timeout at', targetTime);
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
  parallelDecoder: ParallelDecodeManager | null,
  useParallelDecode: boolean
): Promise<void> {
  const { clipsAtTime, trackMap } = ctx;

  const videoClips = clipsAtTime.filter(clip => {
    const track = trackMap.get(clip.trackId);
    return track?.visible && clip.source?.type === 'video' && clip.source.videoElement;
  });

  if (videoClips.length === 0) return;

  // Filter out clips using WebCodecs or parallel decode
  const htmlVideoClips = videoClips.filter(clip => {
    const clipState = clipStates.get(clip.id);
    if (clipState?.isSequential) return false;
    if (useParallelDecode && parallelDecoder?.hasClip(clip.id)) return false;
    return true;
  });

  if (htmlVideoClips.length === 0) return;

  // Wait for HTMLVideoElement clips using setTimeout for export reliability
  const maxWaitTime = 100;
  const startWait = performance.now();

  while (performance.now() - startWait < maxWaitTime) {
    let allReady = true;

    for (const clip of htmlVideoClips) {
      const video = clip.source!.videoElement!;
      if (video.readyState < 2 || video.seeking) {
        allReady = false;
        break;
      }
    }

    if (allReady) {
      await new Promise(r => setTimeout(r, 16));
      return;
    }

    await new Promise(r => setTimeout(r, 16));
  }

  console.warn('[FrameExporter] Timeout waiting for videos to be ready at time', ctx.time);
}
