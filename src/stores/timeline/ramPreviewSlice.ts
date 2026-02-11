// RAM Preview actions slice

import type { RamPreviewActions, SliceCreator } from './types';
import { RAM_PREVIEW_FPS } from './constants';
import { quantizeTime } from './utils';
import { Logger } from '../../services/logger';
import { RamPreviewEngine } from '../../services/ramPreviewEngine';
import { useMediaStore } from '../mediaStore';

const log = Logger.create('RamPreviewSlice');

export const createRamPreviewSlice: SliceCreator<RamPreviewActions> = (set, get) => ({
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
});
