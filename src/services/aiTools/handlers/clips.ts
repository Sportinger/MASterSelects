// Clip Tool Handlers

import { useTimelineStore } from '../../../stores/timeline';
import type { ToolResult } from '../types';
import { formatClipInfo } from '../utils';

type TimelineStore = ReturnType<typeof useTimelineStore.getState>;

export async function handleGetClipDetails(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }
  const track = timelineStore.tracks.find(t => t.id === clip.trackId);

  return {
    success: true,
    data: {
      ...formatClipInfo(clip, track),
      effects: clip.effects || [],
      masks: clip.masks || [],
      transcript: clip.transcript,
      analysisStatus: clip.analysisStatus,
    },
  };
}

export async function handleGetClipsInTimeRange(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const startTime = args.startTime as number;
  const endTime = args.endTime as number;
  const trackType = (args.trackType as string) || 'all';

  const { clips, tracks } = timelineStore;

  const filteredClips = clips.filter(clip => {
    const clipEnd = clip.startTime + clip.duration;
    const overlaps = clip.startTime < endTime && clipEnd > startTime;
    if (!overlaps) return false;

    if (trackType === 'all') return true;
    const track = tracks.find(t => t.id === clip.trackId);
    return track?.type === trackType;
  });

  return {
    success: true,
    data: {
      clips: filteredClips.map(c => {
        const track = tracks.find(t => t.id === c.trackId);
        return formatClipInfo(c, track);
      }),
      count: filteredClips.length,
    },
  };
}

export async function handleSplitClip(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const splitTime = args.splitTime as number;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  const clipEnd = clip.startTime + clip.duration;
  if (splitTime <= clip.startTime || splitTime >= clipEnd) {
    return { success: false, error: `Split time ${splitTime}s is outside clip range (${clip.startTime}s - ${clipEnd}s)` };
  }

  timelineStore.splitClip(clipId, splitTime);
  return { success: true, data: { splitAt: splitTime, originalClipId: clipId } };
}

export async function handleDeleteClip(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  timelineStore.removeClip(clipId);
  return { success: true, data: { deletedClipId: clipId, clipName: clip.name } };
}

export async function handleDeleteClips(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipIds = args.clipIds as string[];
  const deleted: string[] = [];
  const notFound: string[] = [];

  for (const clipId of clipIds) {
    const clip = timelineStore.clips.find(c => c.id === clipId);
    if (clip) {
      timelineStore.removeClip(clipId);
      deleted.push(clipId);
    } else {
      notFound.push(clipId);
    }
  }

  return {
    success: true,
    data: { deleted, notFound, deletedCount: deleted.length },
  };
}

export async function handleCutRangesFromClip(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const ranges = args.ranges as Array<{ timelineStart: number; timelineEnd: number }>;

  // Get initial clip info
  const initialClip = timelineStore.clips.find(c => c.id === clipId);
  if (!initialClip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  const trackId = initialClip.trackId;
  const results: Array<{ range: { start: number; end: number }; status: string }> = [];

  // Sort ranges from END to START (so we don't shift positions)
  const sortedRanges = [...ranges].sort((a, b) => b.timelineStart - a.timelineStart);

  for (const range of sortedRanges) {
    const { timelineStart, timelineEnd } = range;

    // Find the clip that currently contains this range
    // (clip IDs change after splits, so we need to find by position)
    const currentClips = useTimelineStore.getState().clips;
    const targetClip = currentClips.find(c =>
      c.trackId === trackId &&
      c.startTime <= timelineStart &&
      c.startTime + c.duration >= timelineEnd
    );

    if (!targetClip) {
      results.push({ range: { start: timelineStart, end: timelineEnd }, status: 'skipped - no clip at this position' });
      continue;
    }

    const clipEnd = targetClip.startTime + targetClip.duration;

    try {
      // Split at the end of the range (if not at clip boundary)
      if (timelineEnd < clipEnd - 0.01) {
        timelineStore.splitClip(targetClip.id, timelineEnd);
      }

      // Find the clip again (it may have changed after the split)
      const clipsAfterEndSplit = useTimelineStore.getState().clips;
      const clipForStartSplit = clipsAfterEndSplit.find(c =>
        c.trackId === trackId &&
        c.startTime <= timelineStart &&
        c.startTime + c.duration >= timelineStart + 0.01
      );

      if (!clipForStartSplit) {
        results.push({ range: { start: timelineStart, end: timelineEnd }, status: 'error - lost clip after end split' });
        continue;
      }

      // Split at the start of the range (if not at clip boundary)
      if (timelineStart > clipForStartSplit.startTime + 0.01) {
        timelineStore.splitClip(clipForStartSplit.id, timelineStart);
      }

      // Find and delete the middle clip (the unwanted section)
      const clipsAfterSplits = useTimelineStore.getState().clips;
      const clipToDelete = clipsAfterSplits.find(c =>
        c.trackId === trackId &&
        Math.abs(c.startTime - timelineStart) < 0.1
      );

      if (clipToDelete) {
        timelineStore.removeClip(clipToDelete.id);
        results.push({ range: { start: timelineStart, end: timelineEnd }, status: 'removed' });
      } else {
        results.push({ range: { start: timelineStart, end: timelineEnd }, status: 'error - could not find section to delete' });
      }
    } catch (err) {
      results.push({ range: { start: timelineStart, end: timelineEnd }, status: `error: ${err}` });
    }
  }

  const removedCount = results.filter(r => r.status === 'removed').length;
  return {
    success: true,
    data: {
      originalClipId: clipId,
      rangesProcessed: ranges.length,
      rangesRemoved: removedCount,
      results,
    },
  };
}

export async function handleMoveClip(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const newStartTime = args.newStartTime as number;
  const newTrackId = args.newTrackId as string | undefined;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  if (newTrackId) {
    const track = timelineStore.tracks.find(t => t.id === newTrackId);
    if (!track) {
      return { success: false, error: `Track not found: ${newTrackId}` };
    }
  }

  timelineStore.moveClip(clipId, newStartTime, newTrackId);
  return {
    success: true,
    data: {
      clipId,
      newStartTime,
      newTrackId: newTrackId || clip.trackId,
    },
  };
}

export async function handleTrimClip(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const inPoint = args.inPoint as number;
  const outPoint = args.outPoint as number;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  if (inPoint >= outPoint) {
    return { success: false, error: 'In point must be less than out point' };
  }

  timelineStore.trimClip(clipId, inPoint, outPoint);
  return { success: true, data: { clipId, inPoint, outPoint, newDuration: outPoint - inPoint } };
}

export async function handleSplitClipEvenly(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const parts = args.parts as number;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }
  if (parts < 2 || !Number.isInteger(parts)) {
    return { success: false, error: `Parts must be an integer >= 2, got: ${parts}` };
  }

  const trackId = clip.trackId;
  const clipStart = clip.startTime;
  const clipDuration = clip.duration;
  const clipName = clip.name;
  const partDuration = clipDuration / parts;

  // Calculate split times (N-1 splits for N parts)
  const splitTimes: number[] = [];
  for (let i = 1; i < parts; i++) {
    splitTimes.push(clipStart + partDuration * i);
  }

  // Split from END to START so earlier positions remain valid
  for (let i = splitTimes.length - 1; i >= 0; i--) {
    const splitTime = splitTimes[i];
    const currentClips = useTimelineStore.getState().clips;
    const targetClip = currentClips.find(c =>
      c.trackId === trackId &&
      c.startTime <= splitTime - 0.001 &&
      c.startTime + c.duration >= splitTime + 0.001
    );

    if (targetClip) {
      useTimelineStore.getState().splitClip(targetClip.id, splitTime);
    }
  }

  return {
    success: true,
    data: { parts, splitTimes, clipName, partDuration },
  };
}

export async function handleSplitClipAtTimes(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const times = args.times as number[];

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  const trackId = clip.trackId;
  const clipStart = clip.startTime;
  const clipEnd = clip.startTime + clip.duration;

  // Sort and filter to valid times within clip range
  const validTimes = [...times]
    .sort((a, b) => a - b)
    .filter(t => t > clipStart + 0.001 && t < clipEnd - 0.001);

  if (validTimes.length === 0) {
    return { success: false, error: `No valid split times within clip range (${clipStart}s - ${clipEnd}s)` };
  }

  // Split from END to START so earlier positions remain valid
  for (let i = validTimes.length - 1; i >= 0; i--) {
    const splitTime = validTimes[i];
    const currentClips = useTimelineStore.getState().clips;
    const targetClip = currentClips.find(c =>
      c.trackId === trackId &&
      c.startTime <= splitTime - 0.001 &&
      c.startTime + c.duration >= splitTime + 0.001
    );

    if (targetClip) {
      useTimelineStore.getState().splitClip(targetClip.id, splitTime);
    }
  }

  return {
    success: true,
    data: { splitCount: validTimes.length, splitTimes: validTimes, resultingParts: validTimes.length + 1 },
  };
}

export async function handleSelectClips(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipIds = args.clipIds as string[];
  timelineStore.selectClips(clipIds);
  return { success: true, data: { selectedClipIds: clipIds } };
}

export async function handleClearSelection(
  _args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  timelineStore.clearClipSelection();
  return { success: true, data: { message: 'Selection cleared' } };
}
