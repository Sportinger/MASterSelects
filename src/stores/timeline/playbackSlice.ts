// Playback-related actions slice

import type { PlaybackActions, SliceCreator } from './types';
import { MIN_ZOOM, MAX_ZOOM } from './constants';
import { useMediaStore } from '../mediaStore';

// Module-level counter to track play/pause request ordering.
// Incremented by both play() and pause() so that a pending play()
// can detect that a newer pause() (or another play()) has been issued.
let playRequestId = 0;

// Playback actions only (RAM preview and proxy cache in separate slices)
export const createPlaybackSlice: SliceCreator<PlaybackActions> = (set, get) => ({
  // Playback actions
  setPlayheadPosition: (position) => {
    const { duration } = get();
    set({ playheadPosition: Math.max(0, Math.min(position, duration)) });
  },

  setDraggingPlayhead: (dragging) => {
    set({ isDraggingPlayhead: dragging });
  },

  play: async () => {
    const currentRequestId = ++playRequestId;
    const { clips, playheadPosition } = get();

    // Wait for WebCodecs players at playhead to be ready (samples loaded + first frame decoded).
    // Without this, audio starts immediately while the video decoder is still loading samples,
    // causing audio to race ahead and video to show a frozen frame for seconds.
    const wcClipsAtPlayhead = clips.filter(clip => {
      const isAtPlayhead = playheadPosition >= clip.startTime &&
                           playheadPosition < clip.startTime + clip.duration;
      const hasWcPlayer = !!clip.source?.webCodecsPlayer && !clip.source.videoElement;
      return isAtPlayhead && hasWcPlayer;
    });

    if (wcClipsAtPlayhead.length > 0) {
      const waitForWcReady = (clip: typeof wcClipsAtPlayhead[0]): Promise<void> => {
        const player = clip.source!.webCodecsPlayer!;
        if (player.ready) return Promise.resolve();
        return new Promise<void>((resolve) => {
          let iterations = 0;
          const check = () => {
            if (player.ready || ++iterations > 40) { resolve(); return; }
            setTimeout(check, 50);
          };
          check();
        });
      };
      await Promise.race([
        Promise.all(wcClipsAtPlayhead.map(waitForWcReady)),
        new Promise(resolve => setTimeout(resolve, 2000)), // Max 2s wait
      ]);
    }

    // Guard: if pause() or another play() was called during the WC wait
    if (playRequestId !== currentRequestId) return;

    // Find all video clips at current playhead position that need HTMLVideoElement readiness check
    const clipsAtPlayhead = clips.filter(clip => {
      const isAtPlayhead = playheadPosition >= clip.startTime &&
                           playheadPosition < clip.startTime + clip.duration;
      const hasLegacyVideo = clip.source?.videoElement && !clip.source.webCodecsPlayer;
      return isAtPlayhead && hasLegacyVideo;
    });

    // Also check nested composition clips (legacy HTMLVideoElement only)
    const nestedVideos: HTMLVideoElement[] = [];
    for (const clip of clips) {
      if (clip.isComposition && clip.nestedClips) {
        const isAtPlayhead = playheadPosition >= clip.startTime &&
                             playheadPosition < clip.startTime + clip.duration;
        if (isAtPlayhead) {
          const compTime = playheadPosition - clip.startTime + clip.inPoint;
          for (const nestedClip of clip.nestedClips) {
            if (nestedClip.source?.videoElement && !nestedClip.source.webCodecsPlayer) {
              const isNestedAtTime = compTime >= nestedClip.startTime &&
                                     compTime < nestedClip.startTime + nestedClip.duration;
              if (isNestedAtTime) {
                nestedVideos.push(nestedClip.source.videoElement);
              }
            }
          }
        }
      }
    }

    // Collect legacy HTMLVideoElement videos that need readiness check
    const videosToCheck = [
      ...clipsAtPlayhead.map(c => c.source!.videoElement!),
      ...nestedVideos
    ];

    if (videosToCheck.length > 0) {
      // Wait for all videos to be ready (readyState >= 3 means HAVE_FUTURE_DATA)
      const waitForReady = async (video: HTMLVideoElement): Promise<void> => {
        if (video.readyState >= 3) return;

        return new Promise((resolve) => {
          let iterations = 0;
          const maxIterations = 20; // 20 iterations * ~50ms = ~1000ms max

          const checkReady = () => {
            if (video.readyState >= 3) {
              resolve();
              return;
            }

            // Bail out if we've exceeded max iterations
            if (++iterations > maxIterations) {
              resolve();
              return;
            }

            // Trigger buffering by briefly playing
            video.play().then(() => {
              setTimeout(() => {
                video.pause();
                if (video.readyState >= 3) {
                  resolve();
                } else {
                  // Check again after a short delay
                  setTimeout(checkReady, 50);
                }
              }, 50);
            }).catch(() => {
              // If play fails, just wait for canplaythrough
              video.addEventListener('canplaythrough', () => resolve(), { once: true });
              setTimeout(resolve, 500); // Timeout fallback
            });
          };
          checkReady();
        });
      };

      // Wait for all videos in parallel with a timeout
      await Promise.race([
        Promise.all(videosToCheck.map(waitForReady)),
        new Promise(resolve => setTimeout(resolve, 1000)) // Max 1 second wait
      ]);
    }

    // Guard: if pause() or another play() was called during the async wait,
    // this request is stale — do not override the newer state.
    if (playRequestId !== currentRequestId) return;

    set({ isPlaying: true });
  },

  pause: () => {
    // Invalidate any pending play() that is still awaiting video readiness
    playRequestId++;
    // Reset playback speed to normal when pausing
    // So that Space (play/pause toggle) plays forward again
    set({ isPlaying: false, playbackSpeed: 1 });
  },

  stop: () => {
    // Invalidate any pending play() that is still awaiting video readiness
    playRequestId++;
    set({ isPlaying: false, playheadPosition: 0 });
  },

  // View actions
  setZoom: (zoom) => {
    set({ zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom)) });
  },

  toggleSnapping: () => {
    set((state) => ({ snappingEnabled: !state.snappingEnabled }));
  },

  setScrollX: (scrollX) => {
    set({ scrollX: Math.max(0, scrollX) });
  },

  // In/Out marker actions
  setInPoint: (time) => {
    const { outPoint, duration } = get();
    if (time === null) {
      set({ inPoint: null });
      return;
    }
    // Ensure in point doesn't exceed out point or duration
    const clampedTime = Math.max(0, Math.min(time, outPoint ?? duration));
    set({ inPoint: clampedTime });
  },

  setOutPoint: (time) => {
    const { inPoint, duration } = get();
    if (time === null) {
      set({ outPoint: null });
      return;
    }
    // Ensure out point doesn't precede in point and doesn't exceed duration
    const clampedTime = Math.max(inPoint ?? 0, Math.min(time, duration));
    set({ outPoint: clampedTime });
  },

  clearInOut: () => {
    set({ inPoint: null, outPoint: null });
  },

  setInPointAtPlayhead: () => {
    const { playheadPosition, setInPoint } = get();
    setInPoint(playheadPosition);
  },

  setOutPointAtPlayhead: () => {
    const { playheadPosition, setOutPoint } = get();
    setOutPoint(playheadPosition);
  },

  setLoopPlayback: (loop) => {
    set({ loopPlayback: loop });
  },

  toggleLoopPlayback: () => {
    set({ loopPlayback: !get().loopPlayback });
  },

  setPlaybackSpeed: (speed: number) => {
    set({ playbackSpeed: speed });
  },

  // JKL playback control - L for forward play
  playForward: () => {
    const { isPlaying, playbackSpeed, play } = get();
    if (!isPlaying) {
      // Start playing forward at normal speed
      set({ playbackSpeed: 1 });
      play();
    } else if (playbackSpeed < 0) {
      // Was playing reverse, switch to forward
      set({ playbackSpeed: 1 });
    } else {
      // Already playing forward, increase speed (1 -> 2 -> 4 -> 8)
      const newSpeed = playbackSpeed >= 8 ? 8 : playbackSpeed * 2;
      set({ playbackSpeed: newSpeed });
    }
  },

  // JKL playback control - J for reverse play
  playReverse: () => {
    const { isPlaying, playbackSpeed, play } = get();
    if (!isPlaying) {
      // Start playing reverse at normal speed
      set({ playbackSpeed: -1 });
      play();
    } else if (playbackSpeed > 0) {
      // Was playing forward, switch to reverse
      set({ playbackSpeed: -1 });
    } else {
      // Already playing reverse, increase reverse speed (-1 -> -2 -> -4 -> -8)
      const newSpeed = playbackSpeed <= -8 ? -8 : playbackSpeed * 2;
      set({ playbackSpeed: newSpeed });
    }
  },

  setDuration: (duration: number) => {
    // Manually set duration and lock it so it won't auto-update
    const clampedDuration = Math.max(1, duration); // Minimum 1 second
    set({ duration: clampedDuration, durationLocked: true });

    // Sync to composition in media store so it persists
    const { activeCompositionId, updateComposition } = useMediaStore.getState();
    if (activeCompositionId) {
      updateComposition(activeCompositionId, { duration: clampedDuration });
    }

    // Clamp playhead if it's beyond new duration
    const { playheadPosition, inPoint, outPoint } = get();
    if (playheadPosition > clampedDuration) {
      set({ playheadPosition: clampedDuration });
    }
    // Clamp in/out points if needed
    if (inPoint !== null && inPoint > clampedDuration) {
      set({ inPoint: clampedDuration });
    }
    if (outPoint !== null && outPoint > clampedDuration) {
      set({ outPoint: clampedDuration });
    }
  },

  // Performance toggles
  toggleThumbnailsEnabled: () => {
    set({ thumbnailsEnabled: !get().thumbnailsEnabled });
  },

  toggleWaveformsEnabled: () => {
    set({ waveformsEnabled: !get().waveformsEnabled });
  },

  setThumbnailsEnabled: (enabled: boolean) => {
    set({ thumbnailsEnabled: enabled });
  },

  setWaveformsEnabled: (enabled: boolean) => {
    set({ waveformsEnabled: enabled });
  },

  toggleTranscriptMarkers: () => {
    set({ showTranscriptMarkers: !get().showTranscriptMarkers });
  },

  setShowTranscriptMarkers: (enabled: boolean) => {
    set({ showTranscriptMarkers: enabled });
  },

  // Tool mode actions
  setToolMode: (mode) => {
    set({ toolMode: mode });
  },

  toggleCutTool: () => {
    const { toolMode } = get();
    set({ toolMode: toolMode === 'cut' ? 'select' : 'cut' });
  },

  // Clip animation phase for composition transitions
  setClipAnimationPhase: (phase: 'idle' | 'exiting' | 'entering') => {
    set({ clipAnimationPhase: phase });
  },

  // Slot grid view progress
  setSlotGridProgress: (progress: number) => {
    set({ slotGridProgress: Math.max(0, Math.min(1, progress)) });
  },
});
