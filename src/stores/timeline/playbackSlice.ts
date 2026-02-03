// Playback-related actions slice

import type { PlaybackActions, RamPreviewActions, SliceCreator } from './types';
import type { Layer, NestedCompositionData } from '../../types';
import { RAM_PREVIEW_FPS, FRAME_TOLERANCE, MIN_ZOOM, MAX_ZOOM } from './constants';
import { quantizeTime } from './utils';
import { Logger } from '../../services/logger';
import { engine } from '../../engine/WebGPUEngine';
import { layerBuilder } from '../../services/layerBuilder';
import { proxyFrameCache } from '../../services/proxyFrameCache';
import { useMediaStore } from '../mediaStore';

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
    // Reset playback speed to normal when pausing
    // So that Space (play/pause toggle) plays forward again
    set({ isPlaying: false, playbackSpeed: 1 });
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

    // Helper: check if there's a video/image/composition clip at a given time
    const hasVideoAt = (time: number) => {
      return clips.some(c =>
        time >= c.startTime &&
        time < c.startTime + c.duration &&
        (c.source?.type === 'video' || c.source?.type === 'image' || c.isComposition)
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
          } else if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
            // Handle nested compositions
            const clipLocalTime = time - clip.startTime;
            const clipTime = clipLocalTime + clip.inPoint;

            // Get nested video tracks
            const nestedVideoTracks = clip.nestedTracks?.filter(t => t.type === 'video' && t.visible) || [];
            const nestedLayers: Layer[] = [];

            // Seek all nested videos and build nested layers
            for (const nestedTrack of nestedVideoTracks) {
              const nestedClip = clip.nestedClips.find(nc =>
                nc.trackId === nestedTrack.id &&
                clipTime >= nc.startTime &&
                clipTime < nc.startTime + nc.duration
              );

              if (!nestedClip) continue;

              const nestedLocalTime = clipTime - nestedClip.startTime;
              const nestedClipTime = nestedClip.reversed
                ? nestedClip.outPoint - nestedLocalTime
                : nestedLocalTime + nestedClip.inPoint;

              if (nestedClip.source?.videoElement) {
                const nestedVideo = nestedClip.source.videoElement;

                // Seek nested video with verification
                await new Promise<void>((resolve) => {
                  const timeout = setTimeout(() => {
                    nestedVideo.removeEventListener('seeked', onSeeked);
                    resolve();
                  }, 300);

                  const onSeeked = () => {
                    clearTimeout(timeout);
                    nestedVideo.removeEventListener('seeked', onSeeked);
                    resolve();
                  };

                  nestedVideo.addEventListener('seeked', onSeeked);
                  nestedVideo.currentTime = nestedClipTime;
                });

                // Wait for video to be ready
                await new Promise<void>((resolve) => {
                  const checkReady = () => {
                    if (!nestedVideo.seeking && nestedVideo.readyState >= 2) {
                      resolve();
                    } else {
                      requestAnimationFrame(checkReady);
                    }
                  };
                  checkReady();
                  setTimeout(resolve, 100);
                });

                const nestedPos = nestedClip.transform?.position ?? { x: 0, y: 0, z: 0 };
                const nestedScl = nestedClip.transform?.scale ?? { x: 1, y: 1 };
                const nestedRot = nestedClip.transform?.rotation ?? { x: 0, y: 0, z: 0 };

                nestedLayers.push({
                  id: `nested-${nestedClip.id}`,
                  name: nestedClip.name,
                  visible: true,
                  opacity: nestedClip.transform?.opacity ?? 1,
                  blendMode: nestedClip.transform?.blendMode ?? 'normal',
                  source: { type: 'video', videoElement: nestedVideo },
                  effects: nestedClip.effects || [],
                  position: { x: nestedPos.x, y: nestedPos.y, z: nestedPos.z },
                  scale: { x: nestedScl.x, y: nestedScl.y },
                  rotation: { x: nestedRot.x * (Math.PI / 180), y: nestedRot.y * (Math.PI / 180), z: nestedRot.z * (Math.PI / 180) },
                });
              } else if (nestedClip.source?.imageElement) {
                const nestedPos = nestedClip.transform?.position ?? { x: 0, y: 0, z: 0 };
                const nestedScl = nestedClip.transform?.scale ?? { x: 1, y: 1 };
                const nestedRot = nestedClip.transform?.rotation ?? { x: 0, y: 0, z: 0 };

                nestedLayers.push({
                  id: `nested-${nestedClip.id}`,
                  name: nestedClip.name,
                  visible: true,
                  opacity: nestedClip.transform?.opacity ?? 1,
                  blendMode: nestedClip.transform?.blendMode ?? 'normal',
                  source: { type: 'image', imageElement: nestedClip.source.imageElement },
                  effects: nestedClip.effects || [],
                  position: { x: nestedPos.x, y: nestedPos.y, z: nestedPos.z },
                  scale: { x: nestedScl.x, y: nestedScl.y },
                  rotation: { x: nestedRot.x * (Math.PI / 180), y: nestedRot.y * (Math.PI / 180), z: nestedRot.z * (Math.PI / 180) },
                });
              }
            }

            if (nestedLayers.length > 0) {
              // Get composition dimensions
              const mediaStore = useMediaStore.getState();
              const composition = mediaStore.compositions.find(c => c.id === clip.compositionId);
              const compWidth = composition?.width || 1920;
              const compHeight = composition?.height || 1080;

              const nestedCompData: NestedCompositionData = {
                compositionId: clip.compositionId || clip.id,
                layers: nestedLayers,
                width: compWidth,
                height: compHeight,
              };

              const compPos = clip.transform?.position ?? { x: 0, y: 0, z: 0 };
              const compScl = clip.transform?.scale ?? { x: 1, y: 1 };
              const compRot = clip.transform?.rotation ?? { x: 0, y: 0, z: 0 };

              layers.push({
                id: clip.id,
                name: clip.name,
                visible: true,
                opacity: clip.transform?.opacity ?? 1,
                blendMode: clip.transform?.blendMode ?? 'normal',
                source: { type: 'image', nestedComposition: nestedCompData },
                effects: clip.effects || [],
                position: { x: compPos.x, y: compPos.y, z: compPos.z },
                scale: { x: compScl.x, y: compScl.y },
                rotation: { x: compRot.x * (Math.PI / 180), y: compRot.y * (Math.PI / 180), z: compRot.z * (Math.PI / 180) },
              });
            }
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

  // Get proxy frame cached ranges (for yellow timeline indicator)
  // Returns ranges in timeline time coordinates
  getProxyCachedRanges: () => {
    const { clips } = get();
    const mediaFiles = useMediaStore.getState().files;
    const allRanges: Array<{ start: number; end: number }> = [];

    // Process all video clips with proxy enabled
    for (const clip of clips) {
      // Check if clip has video source and mediaFileId
      if (clip.source?.type !== 'video') continue;

      // Try to get mediaFileId from clip or from source
      const mediaFileId = clip.mediaFileId || clip.source?.mediaFileId;
      if (!mediaFileId) continue;

      const mediaFile = mediaFiles.find(f => f.id === mediaFileId);
      if (!mediaFile?.proxyFps || mediaFile.proxyStatus !== 'ready') continue;

      // Get cached ranges for this media file (in media time)
      const mediaCachedRanges = proxyFrameCache.getCachedRanges(mediaFileId, mediaFile.proxyFps);

      if (mediaCachedRanges.length > 0) {
      }

      // Convert media time ranges to timeline time ranges
      const playbackRate = clip.speed || 1;
      for (const range of mediaCachedRanges) {
        // Media time is relative to inPoint
        const mediaStart = range.start;
        const mediaEnd = range.end;

        // Only include ranges that overlap with the visible clip portion
        const clipMediaStart = clip.inPoint;
        const clipMediaEnd = clip.inPoint + clip.duration * playbackRate;

        if (mediaEnd < clipMediaStart || mediaStart > clipMediaEnd) continue;

        // Clamp to visible portion
        const visibleMediaStart = Math.max(mediaStart, clipMediaStart);
        const visibleMediaEnd = Math.min(mediaEnd, clipMediaEnd);

        // Convert to timeline time
        const timelineStart = clip.startTime + (visibleMediaStart - clip.inPoint) / playbackRate;
        const timelineEnd = clip.startTime + (visibleMediaEnd - clip.inPoint) / playbackRate;

        allRanges.push({ start: timelineStart, end: timelineEnd });
      }

      // Also process nested clips if this is a composition
      if (clip.isComposition && clip.nestedClips) {
        for (const nestedClip of clip.nestedClips) {
          if (nestedClip.source?.type !== 'video' || !nestedClip.mediaFileId) continue;

          const nestedMediaFile = mediaFiles.find(f => f.id === nestedClip.mediaFileId);
          if (!nestedMediaFile?.proxyFps || nestedMediaFile.proxyStatus !== 'ready') continue;

          const nestedCachedRanges = proxyFrameCache.getCachedRanges(nestedMediaFile.id, nestedMediaFile.proxyFps);
          const nestedPlaybackRate = nestedClip.speed || 1;

          for (const range of nestedCachedRanges) {
            // Convert nested clip media time to parent clip timeline time
            const mediaStart = range.start;
            const mediaEnd = range.end;

            const nestedMediaStart = nestedClip.inPoint;
            const nestedMediaEnd = nestedClip.inPoint + nestedClip.duration * nestedPlaybackRate;

            if (mediaEnd < nestedMediaStart || mediaStart > nestedMediaEnd) continue;

            const visibleMediaStart = Math.max(mediaStart, nestedMediaStart);
            const visibleMediaEnd = Math.min(mediaEnd, nestedMediaEnd);

            // First convert to nested clip's local time
            const nestedLocalStart = nestedClip.startTime + (visibleMediaStart - nestedClip.inPoint) / nestedPlaybackRate;
            const nestedLocalEnd = nestedClip.startTime + (visibleMediaEnd - nestedClip.inPoint) / nestedPlaybackRate;

            // Then convert to parent timeline time (accounting for composition's inPoint)
            const compInPoint = clip.inPoint;
            if (nestedLocalEnd < compInPoint || nestedLocalStart > compInPoint + clip.duration) continue;

            const visibleNestedStart = Math.max(nestedLocalStart, compInPoint);
            const visibleNestedEnd = Math.min(nestedLocalEnd, compInPoint + clip.duration);

            const timelineStart = clip.startTime + (visibleNestedStart - compInPoint);
            const timelineEnd = clip.startTime + (visibleNestedEnd - compInPoint);

            allRanges.push({ start: timelineStart, end: timelineEnd });
          }
        }
      }
    }

    // Merge overlapping ranges
    if (allRanges.length === 0) return [];

    allRanges.sort((a, b) => a.start - b.start);
    const merged: Array<{ start: number; end: number }> = [allRanges[0]];

    for (let i = 1; i < allRanges.length; i++) {
      const last = merged[merged.length - 1];
      const current = allRanges[i];

      if (current.start <= last.end + 0.05) { // Allow 50ms gap
        last.end = Math.max(last.end, current.end);
      } else {
        merged.push(current);
      }
    }

    return merged;
  },

  // Invalidate cache when content changes (clip moved, trimmed, etc.)
  invalidateCache: () => {
    // Cancel any ongoing RAM preview
    set({ isRamPreviewing: false, cachedFrameTimes: new Set(), ramPreviewRange: null, ramPreviewProgress: null });
    // Immediately clear all caches and request render
    layerBuilder.invalidateCache(); // Force layer rebuild
    engine.setGeneratingRamPreview(false);
    engine.clearCompositeCache();
    engine.requestRender(); // Wake up render loop to show changes immediately
  },

  // Video warmup - seek through all videos to fill browser cache for smooth scrubbing
  startProxyCachePreload: async () => {
    const { clips, isProxyCaching } = get();

    if (isProxyCaching) return;

    // Collect all video elements (including from nested compositions)
    const videoClips: Array<{ video: HTMLVideoElement; duration: number; name: string }> = [];

    const collectVideos = (clipList: typeof clips) => {
      for (const clip of clipList) {
        if (clip.source?.videoElement) {
          videoClips.push({
            video: clip.source.videoElement,
            duration: clip.source.naturalDuration || clip.duration,
            name: clip.name,
          });
        }
        // Also collect from nested compositions
        if (clip.isComposition && clip.nestedClips) {
          collectVideos(clip.nestedClips);
        }
      }
    };

    collectVideos(clips);

    if (videoClips.length === 0) {
      log.info('No video clips to warmup');
      return;
    }

    set({ isProxyCaching: true, proxyCacheProgress: 0 });
    log.info(`Starting video warmup for ${videoClips.length} clips`);

    try {
      const SEEK_INTERVAL = 0.5; // Seek every 0.5 seconds
      let totalSeeks = 0;
      let completedSeeks = 0;

      // Calculate total seeks needed
      for (const clip of videoClips) {
        totalSeeks += Math.ceil(clip.duration / SEEK_INTERVAL);
      }

      // Warmup each video
      for (const clip of videoClips) {
        const video = clip.video;
        const duration = clip.duration;
        const seekCount = Math.ceil(duration / SEEK_INTERVAL);

        for (let i = 0; i < seekCount; i++) {
          // Check if cancelled
          if (!get().isProxyCaching) {
            log.info('Video warmup cancelled');
            return;
          }

          const seekTime = Math.min(i * SEEK_INTERVAL, duration - 0.1);

          // Seek to position
          video.currentTime = seekTime;

          // Wait for seek to complete
          await new Promise<void>((resolve) => {
            const onSeeked = () => {
              video.removeEventListener('seeked', onSeeked);
              resolve();
            };
            video.addEventListener('seeked', onSeeked);
            // Timeout fallback
            setTimeout(resolve, 200);
          });

          completedSeeks++;
          const progress = Math.round((completedSeeks / totalSeeks) * 100);
          set({ proxyCacheProgress: progress });

          // Small delay to not overwhelm the browser
          await new Promise(r => setTimeout(r, 10));
        }
      }

      log.info('Video warmup complete');
    } catch (e) {
      log.error('Video warmup failed', e);
    } finally {
      set({ isProxyCaching: false, proxyCacheProgress: null });
    }
  },

  cancelProxyCachePreload: () => {
    proxyFrameCache.cancelPreload();
    set({ isProxyCaching: false, proxyCacheProgress: null });
    log.info('Proxy cache preload cancelled');
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

  // Clip animation phase for composition transitions
  setClipAnimationPhase: (phase: 'idle' | 'exiting' | 'entering') => {
    set({ clipAnimationPhase: phase });
  },
});
