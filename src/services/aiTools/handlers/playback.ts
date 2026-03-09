import { useTimelineStore } from '../../../stores/timeline';
import { undo as historyUndo, redo as historyRedo } from '../../../stores/historyStore';
import type { ToolResult } from '../types';
import { flashPreviewCanvas, animateMarker } from '../aiFeedback';

type TimelineStore = ReturnType<typeof useTimelineStore.getState>;

export async function handlePlay(
  _args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  await timelineStore.play();
  return { success: true, data: { playing: true } };
}

export async function handlePause(
  _args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  timelineStore.pause();
  return { success: true, data: { playing: false } };
}

export async function handleSetClipSpeed(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) return { success: false, error: `Clip not found: ${clipId}` };

  const store = useTimelineStore.getState();

  if (args.speed !== undefined) {
    const speed = args.speed as number;
    if (speed <= 0) return { success: false, error: 'Speed must be positive. Use "reverse: true" for reverse playback.' };

    // Use keyframe system for speed
    store.setPropertyValue(clipId, 'speed' as any, speed);
  }

  if (args.reverse !== undefined) {
    store.toggleClipReverse(clipId);
    // If already in desired state, toggle back
    const updated = useTimelineStore.getState().clips.find(c => c.id === clipId);
    if (updated && updated.reversed !== (args.reverse as boolean)) {
      useTimelineStore.getState().toggleClipReverse(clipId);
    }
  }

  if (args.preservePitch !== undefined) {
    store.setClipPreservesPitch(clipId, args.preservePitch as boolean);
  }

  store.invalidateCache();

  const finalClip = useTimelineStore.getState().clips.find(c => c.id === clipId);
  return {
    success: true,
    data: {
      clipId,
      speed: finalClip?.speed ?? 1,
      reversed: finalClip?.reversed ?? false,
      preservesPitch: finalClip?.preservesPitch ?? true,
    },
  };
}

export async function handleUndo(): Promise<ToolResult> {
  historyUndo();
  flashPreviewCanvas('undo');
  return { success: true, data: { action: 'undo' } };
}

export async function handleRedo(): Promise<ToolResult> {
  historyRedo();
  flashPreviewCanvas('redo');
  return { success: true, data: { action: 'redo' } };
}

export async function handleAddMarker(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const time = args.time as number;
  const label = args.label as string | undefined;
  const color = args.color as string | undefined;

  const markerId = timelineStore.addMarker(time, label, color);

  // Visual feedback: marker pop animation
  animateMarker(markerId, 'add');

  return {
    success: true,
    data: { markerId, time, label, color },
  };
}

export async function handleGetMarkers(
  _args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const markers = timelineStore.markers || [];
  return {
    success: true,
    data: {
      markers: markers.map(m => ({
        id: m.id,
        time: m.time,
        label: m.label,
        color: m.color,
      })),
    },
  };
}

export async function handleRemoveMarker(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const markerId = args.markerId as string;

  // Visual feedback: marker fade animation before removal
  animateMarker(markerId, 'remove');

  timelineStore.removeMarker(markerId);
  return { success: true, data: { removedMarkerId: markerId } };
}
