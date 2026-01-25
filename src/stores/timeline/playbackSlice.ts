// Playback-related actions slice

import type { PlaybackActions, RamPreviewActions, SliceCreator } from './types';
import type { Layer } from '../../types';
import { RAM_PREVIEW_FPS, FRAME_TOLERANCE, MIN_ZOOM, MAX_ZOOM } from './constants';
import { quantizeTime } from './utils';
import { Logger } from '../../services/logger';

const log = Logger.create('PlaybackSlice');

// Combined playback and RAM preview actions
export type PlaybackAndRamPreviewActions = PlaybackActions & RamPreviewActions;

export const createPlaybackSlice: SliceCreator<PlaybackAndRamPreviewActions> = (set, get) => ({
  // Playback actions
  setPlayheadPosition: (position) => {
    const { duration } = get();
    set({ playheadPosition: Math.max(0, Math.min(position, duration)) });
  },

  setDraggingPlayhead: (dragging) => {
    set({ isDraggingPlayhead: dragging });
  },

  play: async () => {
    const { clips, playheadPosition } = get();

    // Find all video clips at current playhead position that need to be ready
    const clipsAtPlayhead = clips.filter(clip => {
      const isAtPlayhead = playheadPosition >= clip.startTime &&
                           playheadPosition < clip.startTime + clip.duration;
      const hasVideo = clip.source?.videoElement;
      return isAtPlayhead && hasVideo;
    });

    // Also check nested composition clips
    const nestedVideos: HTMLVideoElement[] = [];
    for (const clip of clips) {
      if (clip.isComposition && clip.nestedClips) {
        const isAtPlayhead = playheadPosition >= clip.startTime &&
                             playheadPosition < clip.startTime + clip.duration;
        if (isAtPlayhead) {
          const compTime = playheadPosition - clip.startTime + clip.inPoint;
          for (const nestedClip of clip.nestedClips) {
            if (nestedClip.source?.videoElement) {
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

    // Collect all videos that need to be ready
    const videosToCheck = [
      ...clipsAtPlayhead.map(c => c.source!.videoElement!),
      ...nestedVideos
    ];

    if (videosToCheck.length > 0) {
      // Wait for all videos to be ready (readyState >= 3 means HAVE_FUTURE_DATA)
      const waitForReady = async (video: HTMLVideoElement): Promise<void> => {
        if (video.readyState >= 3) return;

        return new Promise((resolve) => {
          const checkReady = () => {
            if (video.readyState >= 3) {
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

    set({ isPlaying: true });
  },

  pause: () => {
    set({ isPlaying: false });
  },

  stop: () => {
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

  setDuration: (duration: number) => {
    // Manually set duration and lock it so it won't auto-update
    const clampedDuration = Math.max(1, duration); // Minimum 1 second
    set({ duration: clampedDuration, durationLocked: true });

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

  // RAM Preview actions
  toggleRamPreviewEnabled: () => {
    const { ramPreviewEnabled } = get();
    if (ramPreviewEnabled) {
      // Turning OFF - cancel any running preview and clear cache
      set({ ramPreviewEnabled: false, isRamPreviewing: false, ramPreviewProgress: null });
      import('../../engine/WebGPUEngine').then(({ engine }) => {
        engine.setGeneratingRamPreview(false);
        engine.clearCompositeCache();
      });
      set({ ramPreviewRange: null, cachedFrameTimes: new Set() });
    } else {
      // Turning ON - enable automatic RAM preview
      set({ ramPreviewEnabled: true });
    }
  },

  startRamPreview: async () => {
    const { inPoint, outPoint, duration, clips, tracks, isRamPreviewing, playheadPosition, addCachedFrame, ramPreviewEnabled } = get();
    // Don't start if RAM Preview is disabled or already running
    if (!ramPreviewEnabled || isRamPreviewing) return;

    log.debug('RAM Preview starting generation');

    // Determine range to preview (use In/Out or clips extent)
    const start = inPoint ?? 0;
    const end = outPoint ?? (clips.length > 0
      ? Math.max(...clips.map(c => c.startTime + c.duration))
      : duration);

    if (end <= start) return;

    // Import engine dynamically to avoid circular dependency
    const { engine } = await import('../../engine/WebGPUEngine');

    // Tell engine to skip preview updates for efficiency
    engine.setGeneratingRamPreview(true);

    set({
      isRamPreviewing: true,
      ramPreviewProgress: 0,
      ramPreviewRange: null
    });

    const fps = RAM_PREVIEW_FPS;
    const frameInterval = 1 / fps;

    // Helper: check if there's a video clip at a given time
    const hasVideoAt = (time: number) => {
      return clips.some(c =>
        time >= c.startTime &&
        time < c.startTime + c.duration &&
        (c.source?.type === 'video' || c.source?.type === 'image')
      );
    };

    // Generate frame times spreading outward from playhead
    // Only include times where there are video clips
    const centerTime = Math.max(start, Math.min(end, playheadPosition));
    const frameTimes: number[] = [];

    // Add center frame if it has video
    if (hasVideoAt(centerTime)) {
      frameTimes.push(centerTime);
    }

    // Alternate left and right from center, only adding frames with video
    let offset = frameInterval;
    while (offset <= (end - start)) {
      const rightTime = centerTime + offset;
      const leftTime = centerTime - offset;

      if (rightTime <= end && hasVideoAt(rightTime)) {
        frameTimes.push(rightTime);
      }
      if (leftTime >= start && hasVideoAt(leftTime)) {
        frameTimes.push(leftTime);
      }

      offset += frameInterval;
    }

    // No frames to render
    if (frameTimes.length === 0) {
      engine.setGeneratingRamPreview(false);
      set({ isRamPreviewing: false, ramPreviewProgress: null });
      return;
    }

    const totalFrames = frameTimes.length;
    let cancelled = false;

    // Store cancel function
    const checkCancelled = () => !get().isRamPreviewing;

    try {
      for (let frame = 0; frame < totalFrames; frame++) {
        if (checkCancelled()) {
          cancelled = true;
          break;
        }

        const time = frameTimes[frame];

        // Skip frames that are already cached (reuse existing work)
        const quantizedTime = quantizeTime(time);
        if (get().cachedFrameTimes.has(quantizedTime)) {
          // Update progress even for skipped frames
          const progress = ((frame + 1) / totalFrames) * 100;
          set({ ramPreviewProgress: progress });
          continue;
        }

        // Get clips at this time
        const clipsAtTime = clips.filter(c =>
          time >= c.startTime && time < c.startTime + c.duration
        );

        // Build layers for this frame
        const videoTracks = tracks.filter(t => t.type === 'video');
        const layers: Layer[] = [];

        // Seek all videos and build layers
        for (const clip of clipsAtTime) {
          const track = tracks.find(t => t.id === clip.trackId);
          if (!track?.visible || track.type !== 'video') continue;

          if (clip.source?.type === 'video' && clip.source.videoElement) {
            const video = clip.source.videoElement;
            const clipLocalTime = time - clip.startTime;
            // Calculate source time using speed integration (handles keyframes)
            const sourceTime = get().getSourceTimeForClip(clip.id, clipLocalTime);
            // Determine start point based on INITIAL speed (speed at t=0), not clip.speed
            // This is important when keyframes change speed throughout the clip
            const initialSpeed = get().getInterpolatedSpeed(clip.id, 0);
            const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
            const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

            // Robust seek with verification and retry
            const seekWithVerify = async (targetTime: number, maxRetries = 3): Promise<boolean> => {
              for (let attempt = 0; attempt < maxRetries; attempt++) {
                // Check if cancelled
                if (checkCancelled()) return false;

                // Seek to target time
                await new Promise<void>((resolve) => {
                  const timeout = setTimeout(() => {
                    video.removeEventListener('seeked', onSeeked);
                    resolve();
                  }, 500);

                  const onSeeked = () => {
                    clearTimeout(timeout);
                    video.removeEventListener('seeked', onSeeked);
                    resolve();
                  };

                  video.addEventListener('seeked', onSeeked);
                  video.currentTime = targetTime;
                });

                // Wait for video to be fully ready (not seeking, has data)
                await new Promise<void>((resolve) => {
                  const checkReady = () => {
                    if (!video.seeking && video.readyState >= 2) {
                      resolve();
                    } else {
                      requestAnimationFrame(checkReady);
                    }
                  };
                  checkReady();
                  // Timeout fallback
                  setTimeout(resolve, 200);
                });

                // Verify position is correct (within 1 frame tolerance at 30fps)
                if (Math.abs(video.currentTime - targetTime) < FRAME_TOLERANCE) {
                  return true; // Success
                }

                // Position wrong (user scrubbed?), retry
                if (checkCancelled()) return false;
              }
              return false; // Failed after retries
            };

            // Perform seek with verification
            const seekSuccess = await seekWithVerify(clipTime);
            if (!seekSuccess || checkCancelled()) {
              continue; // Skip this clip if seek failed or cancelled
            }

            // Add to layers (with defensive defaults for transform properties)
            const pos = clip.transform?.position ?? { x: 0, y: 0, z: 0 };
            const scl = clip.transform?.scale ?? { x: 1, y: 1 };
            const rot = clip.transform?.rotation ?? { x: 0, y: 0, z: 0 };
            layers.push({
              id: clip.id,
              name: clip.name,
              visible: true,
              opacity: clip.transform?.opacity ?? 1,
              blendMode: clip.transform?.blendMode ?? 'normal',
              source: { type: 'video', videoElement: video },
              effects: [],
              position: { x: pos.x, y: pos.y, z: pos.z },
              scale: { x: scl.x, y: scl.y },
              rotation: { x: rot.x * (Math.PI / 180), y: rot.y * (Math.PI / 180), z: rot.z * (Math.PI / 180) },
            });
          } else if (clip.source?.type === 'image' && clip.source.imageElement) {
            const imgPos = clip.transform?.position ?? { x: 0, y: 0, z: 0 };
            const imgScl = clip.transform?.scale ?? { x: 1, y: 1 };
            const imgRot = clip.transform?.rotation ?? { x: 0, y: 0, z: 0 };
            layers.push({
              id: clip.id,
              name: clip.name,
              visible: true,
              opacity: clip.transform?.opacity ?? 1,
              blendMode: clip.transform?.blendMode ?? 'normal',
              source: { type: 'image', imageElement: clip.source.imageElement },
              effects: [],
              position: { x: imgPos.x, y: imgPos.y, z: imgPos.z },
              scale: { x: imgScl.x, y: imgScl.y },
              rotation: { x: imgRot.x * (Math.PI / 180), y: imgRot.y * (Math.PI / 180), z: imgRot.z * (Math.PI / 180) },
            });
          }
        }

        // Sort layers by track order
        const trackOrder = new Map(videoTracks.map((t, i) => [t.id, i]));
        layers.sort((a, b) => {
          const clipA = clipsAtTime.find(c => c.id === a.id);
          const clipB = clipsAtTime.find(c => c.id === b.id);
          const orderA = clipA ? (trackOrder.get(clipA.trackId) ?? 0) : 0;
          const orderB = clipB ? (trackOrder.get(clipB.trackId) ?? 0) : 0;
          return orderA - orderB;
        });

        // Final verification: ensure all videos are still at correct position before rendering
        // This catches cases where user interaction changed position between seek and render
        let allPositionsCorrect = true;
        for (const clip of clipsAtTime) {
          if (clip.source?.type === 'video' && clip.source.videoElement) {
            const video = clip.source.videoElement;
            const localTime = time - clip.startTime;
            // Use speed integration for expected time (must match seek logic above)
            const sourceTime = get().getSourceTimeForClip(clip.id, localTime);
            // Must match seek logic: use initial speed for start point
            const initialSpeed = get().getInterpolatedSpeed(clip.id, 0);
            const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
            const expectedTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
            if (Math.abs(video.currentTime - expectedTime) > FRAME_TOLERANCE) {
              allPositionsCorrect = false;
              break;
            }
          }
        }

        // Skip this frame if positions are wrong (user scrubbed) or cancelled
        if (!allPositionsCorrect || checkCancelled()) {
          continue;
        }

        // Render and cache this frame
        if (layers.length > 0) {
          engine.render(layers);
        }
        await engine.cacheCompositeFrame(time);

        // Add to cached frames set (shows green indicator immediately)
        addCachedFrame(time);

        // Update progress percentage
        const progress = ((frame + 1) / totalFrames) * 100;
        set({ ramPreviewProgress: progress });

        // Yield to allow UI updates (every frame for smooth green dot updates)
        if (frame % 3 === 0) {
          await new Promise(resolve => setTimeout(resolve, 0));
        }
      }

      if (!cancelled) {
        // Set the preview range for cache hit detection
        set({
          ramPreviewRange: { start, end },
          ramPreviewProgress: null
        });
        log.debug('RAM Preview complete', { totalFrames, start: start.toFixed(1), end: end.toFixed(1) });
      } else {
        log.debug('RAM Preview cancelled');
      }
    } catch (error) {
      log.error('RAM Preview error', error);
    } finally {
      engine.setGeneratingRamPreview(false);
      set({ isRamPreviewing: false, ramPreviewProgress: null });
    }
  },

  cancelRamPreview: () => {
    // IMMEDIATELY set state to cancel the loop - this must be synchronous!
    // The RAM preview loop checks !get().isRamPreviewing to know when to stop
    set({ isRamPreviewing: false, ramPreviewProgress: null });
    // Then async cleanup the engine
    import('../../engine/WebGPUEngine').then(({ engine }) => {
      engine.setGeneratingRamPreview(false);
    });
  },

  clearRamPreview: async () => {
    const { engine } = await import('../../engine/WebGPUEngine');
    engine.clearCompositeCache();
    set({ ramPreviewRange: null, ramPreviewProgress: null, cachedFrameTimes: new Set() });
  },

  // Playback frame caching (green line like After Effects)
  addCachedFrame: (time: number) => {
    const quantized = quantizeTime(time);
    const { cachedFrameTimes } = get();
    if (!cachedFrameTimes.has(quantized)) {
      const newSet = new Set(cachedFrameTimes);
      newSet.add(quantized);
      set({ cachedFrameTimes: newSet });
    }
  },

  getCachedRanges: () => {
    const { cachedFrameTimes } = get();
    if (cachedFrameTimes.size === 0) return [];

    // Convert set to sorted array
    const times = Array.from(cachedFrameTimes).sort((a, b) => a - b);
    const ranges: Array<{ start: number; end: number }> = [];
    const frameInterval = 1 / RAM_PREVIEW_FPS;
    const gap = frameInterval * 2; // Allow gap of 2 frames

    let rangeStart = times[0];
    let rangeEnd = times[0];

    for (let i = 1; i < times.length; i++) {
      if (times[i] - rangeEnd <= gap) {
        // Continue range
        rangeEnd = times[i];
      } else {
        // End range and start new one
        ranges.push({ start: rangeStart, end: rangeEnd + frameInterval });
        rangeStart = times[i];
        rangeEnd = times[i];
      }
    }

    // Add final range
    ranges.push({ start: rangeStart, end: rangeEnd + frameInterval });

    return ranges;
  },

  // Invalidate cache when content changes (clip moved, trimmed, etc.)
  invalidateCache: () => {
    // Cancel any ongoing RAM preview
    set({ isRamPreviewing: false });
    // Then async cleanup the engine
    import('../../engine/WebGPUEngine').then(({ engine }) => {
      engine.setGeneratingRamPreview(false);
      engine.clearCompositeCache();
    });
    // Clear cached frame times
    set({ cachedFrameTimes: new Set(), ramPreviewRange: null, ramPreviewProgress: null });
  },

  // Performance toggles
  toggleThumbnailsEnabled: () => {
    set({ thumbnailsEnabled: !get().thumbnailsEnabled });
  },

  toggleWaveformsEnabled: () => {
    set({ waveformsEnabled: !get().waveformsEnabled });
  },

  // Tool mode actions
  setToolMode: (mode) => {
    set({ toolMode: mode });
  },

  toggleCutTool: () => {
    const { toolMode } = get();
    set({ toolMode: toolMode === 'cut' ? 'select' : 'cut' });
  },
});
