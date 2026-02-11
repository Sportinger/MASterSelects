// Playback-related actions slice

import type { PlaybackActions, RamPreviewActions, SliceCreator } from './types';
import { RAM_PREVIEW_FPS, MIN_ZOOM, MAX_ZOOM } from './constants';
import { quantizeTime } from './utils';
import { Logger } from '../../services/logger';
import { engine } from '../../engine/WebGPUEngine';
import { RamPreviewEngine } from '../../services/ramPreviewEngine';
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
    if (!ramPreviewEnabled || isRamPreviewing) return;

    log.debug('RAM Preview starting generation');

    const start = inPoint ?? 0;
    const end = outPoint ?? (clips.length > 0
      ? Math.max(...clips.map(c => c.startTime + c.duration))
      : duration);
    if (end <= start) return;

    const { engine } = await import('../../engine/WebGPUEngine');
    engine.setGeneratingRamPreview(true);
    set({ isRamPreviewing: true, ramPreviewProgress: 0, ramPreviewRange: null });

    try {
      const preview = new RamPreviewEngine(engine);
      const result = await preview.generate(
        { start, end, centerTime: playheadPosition, clips, tracks },
        {
          isCancelled: () => !get().isRamPreviewing,
          isFrameCached: (qt) => get().cachedFrameTimes.has(qt),
          getSourceTimeForClip: (id, t) => get().getSourceTimeForClip(id, t),
          getInterpolatedSpeed: (id, t) => get().getInterpolatedSpeed(id, t),
          getCompositionDimensions: (compId) => {
            const comp = useMediaStore.getState().compositions.find(c => c.id === compId);
            return { width: comp?.width || 1920, height: comp?.height || 1080 };
          },
          onFrameCached: (time) => addCachedFrame(time),
          onProgress: (percent) => set({ ramPreviewProgress: percent }),
        }
      );

      if (result.completed) {
        set({ ramPreviewRange: { start, end }, ramPreviewProgress: null });
        log.debug('RAM Preview complete', { totalFrames: result.frameCount, start: start.toFixed(1), end: end.toFixed(1) });
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
