import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import type { ToolResult } from '../types';

type TimelineStore = ReturnType<typeof useTimelineStore.getState>;

export async function handleSetTransform(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  // Get composition resolution for pixel → normalized conversion
  const activeComp = useMediaStore.getState().getActiveComposition();
  const compWidth = activeComp?.width ?? 1920;
  const compHeight = activeComp?.height ?? 1080;

  const updates: Record<string, unknown> = {};
  const hasPosition = args.x !== undefined || args.y !== undefined;
  const hasScale = args.scaleX !== undefined || args.scaleY !== undefined;
  const hasRotation = args.rotation !== undefined;

  if (hasPosition) {
    const currentPos = clip.transform?.position || { x: 0, y: 0, z: 0 };
    updates.position = {
      x: args.x !== undefined ? (args.x as number) / compWidth : currentPos.x,
      y: args.y !== undefined ? (args.y as number) / compHeight : currentPos.y,
      z: currentPos.z,
    };
  }
  if (hasScale) {
    const currentScale = clip.transform?.scale || { x: 1, y: 1 };
    updates.scale = {
      x: args.scaleX !== undefined ? args.scaleX as number : currentScale.x,
      y: args.scaleY !== undefined ? args.scaleY as number : currentScale.y,
    };
  }
  if (hasRotation) {
    const currentRot = clip.transform?.rotation || { x: 0, y: 0, z: 0 };
    updates.rotation = { ...currentRot, z: args.rotation as number };
  }
  if (args.opacity !== undefined) updates.opacity = args.opacity as number;
  if (args.blendMode !== undefined) updates.blendMode = args.blendMode as string;

  if (Object.keys(updates).length === 0) {
    return { success: false, error: 'No transform properties provided' };
  }

  const { updateClipTransform, invalidateCache } = useTimelineStore.getState();
  updateClipTransform(clipId, updates);
  invalidateCache();

  return {
    success: true,
    data: {
      clipId,
      updatedProperties: Object.keys(updates),
    },
  };
}
