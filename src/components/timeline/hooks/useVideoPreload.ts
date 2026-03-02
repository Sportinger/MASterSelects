// Video preloading - seeks and buffers upcoming clips before playhead reaches them

import { useEffect, useRef } from 'react';
import { playheadState } from '../../../services/layerBuilder';
import type { TimelineClip } from '../../../types';

interface UseVideoPreloadProps {
  isPlaying: boolean;
  isDraggingPlayhead: boolean;
  playheadPosition: number;
  clips: TimelineClip[];
}

/**
 * Preload upcoming video clips - seek videos and force buffering before playhead hits them
 * This prevents stuttering when playback transitions to a new clip
 * PERFORMANCE: Throttled to run every 500ms instead of every frame
 */
export function useVideoPreload({
  isPlaying,
  isDraggingPlayhead,
  playheadPosition,
  clips,
}: UseVideoPreloadProps) {
  const lastPreloadCheckRef = useRef(0);

  useEffect(() => {
    if (!isPlaying || isDraggingPlayhead) return;

    // Throttle preload checks to every 500ms (no need to check every frame for 2s lookahead)
    const now = performance.now();
    if (now - lastPreloadCheckRef.current < 500) return;
    lastPreloadCheckRef.current = now;

    const LOOKAHEAD_TIME = 2.0; // Look 2 seconds ahead
    // Use high-frequency playhead position during playback
    const currentPosition = playheadState.isUsingInternalPosition
      ? playheadState.position
      : playheadPosition;
    const lookaheadPosition = currentPosition + LOOKAHEAD_TIME;

    // Helper to preload a video element - seeks and forces buffering
    const preloadVideo = (
      video: HTMLVideoElement,
      targetTime: number,
      _clipName: string
    ) => {
      const timeDiff = Math.abs(video.currentTime - targetTime);

      // Only preload if significantly different (avoid repeated preloading)
      if (timeDiff > 0.1) {
        video.currentTime = Math.max(0, targetTime);

        // Force buffer by briefly playing then pausing
        // This triggers the browser to actually fetch the video data
        const wasPlaying = !video.paused;
        if (!wasPlaying) {
          video
            .play()
            .then(() => {
              // Immediately pause after play starts buffering
              setTimeout(() => {
                if (!wasPlaying) video.pause();
              }, 50);
            })
            .catch(() => {
              // Ignore play errors (e.g., autoplay policy)
            });
        }
      }
    };

    // Find clips that will start playing soon (not currently playing, but will be soon)
    const upcomingClips = clips.filter((clip) => {
      // Clip starts after current position but within lookahead window
      const startsInLookahead =
        clip.startTime > currentPosition && clip.startTime <= lookaheadPosition;
      // Has a video element to preload
      const hasVideo = clip.source?.videoElement;
      return startsInLookahead && hasVideo;
    });

    // Pre-buffer upcoming regular clips
    for (const clip of upcomingClips) {
      if (clip.source?.videoElement) {
        preloadVideo(clip.source.videoElement, clip.inPoint, clip.name);
      }
    }

    // Also preload nested composition clips
    const upcomingNestedClips = clips.filter((clip) => {
      const startsInLookahead =
        clip.startTime > currentPosition && clip.startTime <= lookaheadPosition;
      const hasNestedClips =
        clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0;
      return startsInLookahead && hasNestedClips;
    });

    for (const compClip of upcomingNestedClips) {
      if (!compClip.nestedClips) continue;

      // Find the nested video clip that would play at the start of this comp clip
      const compStartTime = compClip.inPoint; // Time within the composition

      for (const nestedClip of compClip.nestedClips) {
        if (!nestedClip.source?.videoElement) continue;

        // Check if this nested clip would be playing at comp start
        if (
          compStartTime >= nestedClip.startTime &&
          compStartTime < nestedClip.startTime + nestedClip.duration
        ) {
          const nestedLocalTime = compStartTime - nestedClip.startTime;
          const targetTime = nestedClip.reversed
            ? nestedClip.outPoint - nestedLocalTime
            : nestedLocalTime + nestedClip.inPoint;

          preloadVideo(nestedClip.source.videoElement, targetTime, nestedClip.name);
        }
      }
    }
  }, [isPlaying, isDraggingPlayhead, playheadPosition, clips]);
}
