// Solid clip actions slice - extracted from clipSlice

import type { TimelineClip } from '../../types';
import type { SolidClipActions, SliceCreator } from './types';
import { DEFAULT_TRANSFORM } from './constants';
import { engine } from '../../engine/WebGPUEngine';
import { layerBuilder } from '../../services/layerBuilder';
import { generateSolidClipId } from './helpers/idGenerator';
import { useMediaStore } from '../mediaStore';
import { Logger } from '../../services/logger';

const log = Logger.create('SolidClipSlice');

export const createSolidClipSlice: SliceCreator<SolidClipActions> = (set, get) => ({
  addSolidClip: (trackId, startTime, color = '#ffffff', duration = 5, skipMediaItem = false) => {
    const { clips, tracks, updateDuration, invalidateCache } = get();
    const track = tracks.find(t => t.id === trackId);

    if (!track || track.type !== 'video') {
      log.warn('Solid clips can only be added to video tracks');
      return null;
    }

    const clipId = generateSolidClipId();

    const activeComp = useMediaStore.getState().getActiveComposition();
    const compWidth = activeComp?.width || 1920;
    const compHeight = activeComp?.height || 1080;

    const canvas = document.createElement('canvas');
    canvas.width = compWidth;
    canvas.height = compHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, compWidth, compHeight);

    const solidClip: TimelineClip = {
      id: clipId,
      trackId,
      name: `Solid ${color}`,
      file: new File([], 'solid-clip.dat', { type: 'application/octet-stream' }),
      startTime,
      duration,
      inPoint: 0,
      outPoint: duration,
      source: { type: 'solid', textCanvas: canvas, naturalDuration: duration },
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      solidColor: color,
      isLoading: false,
    };

    set({ clips: [...clips, solidClip] });
    updateDuration();
    invalidateCache();

    if (!skipMediaItem) {
      const mediaStore = useMediaStore.getState();
      const solidFolderId = mediaStore.getOrCreateSolidFolder();
      mediaStore.createSolidItem(`Solid ${color}`, color, solidFolderId);
    }

    log.debug('Created solid clip', { clipId, color });
    return clipId;
  },

  updateSolidColor: (clipId, color) => {
    const { clips, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip || clip.source?.type !== 'solid') return;

    const canvas = clip.source.textCanvas;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = color;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      const texMgr = engine.getTextureManager();
      if (texMgr) {
        texMgr.updateCanvasTexture(canvas);
      }
    }

    set({
      clips: clips.map(c => c.id !== clipId ? c : {
        ...c,
        solidColor: color,
        name: `Solid ${color}`,
        source: { ...c.source!, textCanvas: canvas },
      }),
    });
    invalidateCache();

    try {
      layerBuilder.invalidateCache();
      const layers = layerBuilder.buildLayersFromStore();
      engine.render(layers);
    } catch (e) {
      log.debug('Direct render after solid color update failed', e);
    }
  },
});
