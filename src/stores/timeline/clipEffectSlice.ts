// Clip effect actions slice - extracted from clipSlice

import type { Effect, EffectType } from '../../types';
import type { ClipEffectActions, SliceCreator } from './types';
import { getDefaultEffectParams } from './utils';
import { generateEffectId } from './helpers/idGenerator';

export const createClipEffectSlice: SliceCreator<ClipEffectActions> = (set, get) => ({
  addClipEffect: (clipId, effectType) => {
    const { clips, invalidateCache } = get();
    const effect: Effect = {
      id: generateEffectId(),
      name: effectType,
      type: effectType as EffectType,
      enabled: true,
      params: getDefaultEffectParams(effectType),
    };
    set({ clips: clips.map(c => c.id === clipId ? { ...c, effects: [...(c.effects || []), effect] } : c) });
    invalidateCache();
  },

  removeClipEffect: (clipId, effectId) => {
    const { clips, invalidateCache } = get();
    set({ clips: clips.map(c => c.id === clipId ? { ...c, effects: c.effects.filter(e => e.id !== effectId) } : c) });
    invalidateCache();
  },

  updateClipEffect: (clipId, effectId, params) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId
          ? { ...c, effects: c.effects.map(e => e.id === effectId ? { ...e, params: { ...e.params, ...params } as Effect['params'] } : e) }
          : c
      ),
    });
    invalidateCache();
  },

  setClipEffectEnabled: (clipId, effectId, enabled) => {
    const { clips, invalidateCache } = get();
    set({
      clips: clips.map(c =>
        c.id === clipId
          ? { ...c, effects: c.effects.map(e => e.id === effectId ? { ...e, enabled } : e) }
          : c
      ),
    });
    invalidateCache();
  },
});
