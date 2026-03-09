import { useTimelineStore } from '../../../stores/timeline';
import { getAllEffects, getDefaultParams, hasEffect, getCategoriesWithEffects } from '../../../effects';
import type { ToolResult } from '../types';
import { selectClipAndOpenTab } from '../aiFeedback';

type TimelineStore = ReturnType<typeof useTimelineStore.getState>;

export async function handleListEffects(): Promise<ToolResult> {
  const categories = getCategoriesWithEffects();
  const data = categories.map(({ category, effects }) => ({
    category,
    effects: effects.map(e => ({
      id: e.id,
      name: e.name,
      params: Object.entries(e.params).map(([key, param]) => ({
        name: key,
        type: param.type,
        default: param.default,
        min: param.min,
        max: param.max,
        step: param.step,
        description: param.label || key,
      })),
    })),
  }));

  return {
    success: true,
    data: { totalEffects: getAllEffects().length, categories: data },
  };
}

export async function handleAddEffect(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const effectType = args.effectType as string;
  const customParams = args.params as Record<string, unknown> | undefined;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) return { success: false, error: `Clip not found: ${clipId}` };

  if (!hasEffect(effectType)) {
    const available = getAllEffects().map(e => e.id).join(', ');
    return { success: false, error: `Unknown effect type: ${effectType}. Available: ${available}` };
  }

  const { addClipEffect, updateClipEffect, invalidateCache } = useTimelineStore.getState();
  addClipEffect(clipId, effectType);

  // Visual feedback: select clip and open effects tab
  selectClipAndOpenTab(clipId, 'effects');

  // Find the newly added effect (last one on the clip)
  const updatedClip = useTimelineStore.getState().clips.find(c => c.id === clipId);
  const newEffect = updatedClip?.effects[updatedClip.effects.length - 1];

  // Apply custom params if provided
  if (newEffect && customParams) {
    updateClipEffect(clipId, newEffect.id, customParams as Partial<Record<string, string | number | boolean>>);
  }

  invalidateCache();

  return {
    success: true,
    data: {
      clipId,
      effectId: newEffect?.id,
      effectType,
      params: newEffect ? { ...getDefaultParams(effectType), ...customParams } : getDefaultParams(effectType),
    },
  };
}

export async function handleRemoveEffect(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const effectId = args.effectId as string;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) return { success: false, error: `Clip not found: ${clipId}` };

  const effect = clip.effects.find(e => e.id === effectId);
  if (!effect) return { success: false, error: `Effect not found: ${effectId}` };

  const { removeClipEffect, invalidateCache } = useTimelineStore.getState();
  removeClipEffect(clipId, effectId);
  invalidateCache();

  // Visual feedback: select clip and open effects tab
  selectClipAndOpenTab(clipId, 'effects');

  return {
    success: true,
    data: { clipId, removedEffectId: effectId, removedEffectType: effect.type },
  };
}

export async function handleUpdateEffect(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const effectId = args.effectId as string;
  const params = args.params as Record<string, unknown>;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) return { success: false, error: `Clip not found: ${clipId}` };

  const effect = clip.effects.find(e => e.id === effectId);
  if (!effect) return { success: false, error: `Effect not found: ${effectId}` };

  const { updateClipEffect, invalidateCache } = useTimelineStore.getState();
  updateClipEffect(clipId, effectId, params as Partial<Record<string, string | number | boolean>>);
  invalidateCache();

  // Visual feedback: select clip and open effects tab
  selectClipAndOpenTab(clipId, 'effects');

  return {
    success: true,
    data: { clipId, effectId, updatedParams: Object.keys(params) },
  };
}
