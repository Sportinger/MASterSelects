// Text clip actions slice - extracted from clipSlice

import type { TimelineClip, TextClipProperties } from '../../types';
import type { TextClipActions, SliceCreator } from './types';
import { DEFAULT_TRANSFORM, DEFAULT_TEXT_PROPERTIES, DEFAULT_TEXT_DURATION } from './constants';
import { textRenderer } from '../../services/textRenderer';
import { googleFontsService } from '../../services/googleFontsService';
import { engine } from '../../engine/WebGPUEngine';
import { layerBuilder } from '../../services/layerBuilder';
import { generateTextClipId } from './helpers/idGenerator';
import { useMediaStore } from '../mediaStore';
import { Logger } from '../../services/logger';

const log = Logger.create('TextClipSlice');

export const createTextClipSlice: SliceCreator<TextClipActions> = (set, get) => ({
  addTextClip: async (trackId, startTime, duration = DEFAULT_TEXT_DURATION, skipMediaItem = false) => {
    const { clips, tracks, updateDuration, invalidateCache } = get();
    const track = tracks.find(t => t.id === trackId);

    if (!track || track.type !== 'video') {
      log.warn('Text clips can only be added to video tracks');
      return null;
    }

    const clipId = generateTextClipId();
    await googleFontsService.loadFont(DEFAULT_TEXT_PROPERTIES.fontFamily, DEFAULT_TEXT_PROPERTIES.fontWeight);

    const canvas = textRenderer.createCanvas(1920, 1080);
    textRenderer.render(DEFAULT_TEXT_PROPERTIES, canvas);

    const textClip: TimelineClip = {
      id: clipId,
      trackId,
      name: 'Text',
      file: new File([], 'text-clip.txt', { type: 'text/plain' }),
      startTime,
      duration,
      inPoint: 0,
      outPoint: duration,
      source: { type: 'text', textCanvas: canvas, naturalDuration: duration },
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      textProperties: { ...DEFAULT_TEXT_PROPERTIES },
      isLoading: false,
    };

    set({ clips: [...clips, textClip] });
    updateDuration();
    invalidateCache();

    if (!skipMediaItem) {
      const mediaStore = useMediaStore.getState();
      const textFolderId = mediaStore.getOrCreateTextFolder();
      mediaStore.createTextItem('Text', textFolderId);
    }

    log.debug('Created text clip', { clipId });
    return clipId;
  },

  updateTextProperties: (clipId, props) => {
    const { clips, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip?.textProperties) return;

    const newProps: TextClipProperties = { ...clip.textProperties, ...props };

    const canvas = clip.source?.textCanvas || textRenderer.createCanvas(1920, 1080);
    textRenderer.render(newProps, canvas);

    const texMgr = engine.getTextureManager();
    if (texMgr) {
      if (!texMgr.updateCanvasTexture(canvas)) {
        log.debug('Canvas texture not cached yet, will create on render');
      }
    }

    set({
      clips: clips.map(c => c.id !== clipId ? c : {
        ...c,
        textProperties: newProps,
        source: { ...c.source!, textCanvas: canvas },
        name: newProps.text.substring(0, 20) || 'Text',
      }),
    });
    invalidateCache();

    try {
      layerBuilder.invalidateCache();
      const layers = layerBuilder.buildLayersFromStore();
      engine.render(layers);
    } catch (e) {
      log.debug('Direct render after text update failed', e);
    }

    if (props.fontFamily || props.fontWeight) {
      const fontFamily = props.fontFamily || newProps.fontFamily;
      const fontWeight = props.fontWeight || newProps.fontWeight;
      googleFontsService.loadFont(fontFamily, fontWeight).then(() => {
        const { clips: currentClips, invalidateCache: inv } = get();
        const currentClip = currentClips.find(cl => cl.id === clipId);
        if (!currentClip?.textProperties) return;

        const currentCanvas = currentClip.source?.textCanvas;
        if (currentCanvas) {
          textRenderer.render(currentClip.textProperties, currentCanvas);
          engine.getTextureManager()?.updateCanvasTexture(currentCanvas);
        }
        inv();

        try {
          layerBuilder.invalidateCache();
          const layers = layerBuilder.buildLayersFromStore();
          engine.render(layers);
        } catch (e) {
          log.debug('Direct render after font load failed', e);
        }
      });
    }
  },
});
