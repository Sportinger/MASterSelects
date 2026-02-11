// Proxy cache and invalidation actions slice

import type { ProxyCacheActions, SliceCreator } from './types';
import { Logger } from '../../services/logger';
import { engine } from '../../engine/WebGPUEngine';
import { layerBuilder } from '../../services/layerBuilder';
import { proxyFrameCache } from '../../services/proxyFrameCache';
import { useMediaStore } from '../mediaStore';

const log = Logger.create('ProxyCacheSlice');

export const createProxyCacheSlice: SliceCreator<ProxyCacheActions> = (set, get) => ({
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
});
