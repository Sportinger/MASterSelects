// Clip Tool Handlers

import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import { createVideoElement, createAudioElement } from '../../../stores/timeline/helpers/webCodecsHelpers';
import type { TimelineClip } from '../../../types';
import type { ToolResult } from '../types';
import { formatClipInfo } from '../utils';
import { isAIExecutionActive, consumeStaggerDelay } from '../executionState';
import { activateDockPanel } from '../aiFeedback';
import { Logger } from '../../../services/logger';

const log = Logger.create('AITool:Clips');

/** Resolve clip background color for ghost overlays */
function getClipColor(clip: TimelineClip): string {
  if (clip.source?.type === 'audio') return '#2d6b4a';
  if (clip.source?.type === 'text') return '#5c3d7a';
  if (clip.source?.type === 'solid' && clip.solidColor) return clip.solidColor;
  return '#3d5a80';
}

type TimelineStore = ReturnType<typeof useTimelineStore.getState>;

function getHeapSnapshot():
  | {
      heapUsedMB: number;
      heapTotalMB: number;
      heapLimitMB: number;
    }
  | undefined {
  const perf = performance as Performance & {
    memory?: {
      usedJSHeapSize: number;
      totalJSHeapSize: number;
      jsHeapSizeLimit: number;
    };
  };
  const memory = perf.memory;
  if (!memory) return undefined;

  return {
    heapUsedMB: Math.round(memory.usedJSHeapSize / (1024 * 1024)),
    heapTotalMB: Math.round(memory.totalJSHeapSize / (1024 * 1024)),
    heapLimitMB: Math.round(memory.jsHeapSizeLimit / (1024 * 1024)),
  };
}

function logSplitCheckpoint(
  stage: string,
  clip: TimelineClip,
  splitCount: number,
  withLinked: boolean
): void {
  const state = useTimelineStore.getState();
  log.warn(`[split-checkpoint:${stage}] ${clip.id}`, {
    clipId: clip.id,
    clipName: clip.name,
    splitCount,
    withLinked,
    aiExecutionActive: isAIExecutionActive(),
    totalClips: state.clips.length,
    totalTracks: state.tracks.length,
    selectedClipIds: state.selectedClipIds.size,
    ...getHeapSnapshot(),
  });
}

/**
 * Deep clone serializable clip properties (effects, masks, transforms, etc.)
 * Same logic as deepCloneClipProps in clipSlice.ts
 */
function deepCloneClipProps(clip: TimelineClip): Partial<TimelineClip> {
  return {
    transform: structuredClone(clip.transform),
    effects: clip.effects.map(e => structuredClone(e)),
    ...(clip.masks ? { masks: clip.masks.map(m => structuredClone(m)) } : {}),
    ...(clip.textProperties ? { textProperties: structuredClone(clip.textProperties) } : {}),
    ...(clip.analysis ? { analysis: structuredClone(clip.analysis) } : {}),
  };
}

function cloneVideoElementForSplit(clip: TimelineClip): HTMLVideoElement {
  const existingSrc = clip.source?.videoElement?.src;
  if (existingSrc) {
    const video = document.createElement('video');
    video.src = existingSrc;
    video.preload = 'none';
    video.muted = true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';
    return video;
  }

  const video = createVideoElement(clip.file);
  video.preload = 'none';
  return video;
}

function cloneAudioElementForSplit(
  clip: Pick<TimelineClip, 'file' | 'source'>
): HTMLAudioElement {
  const existingSrc = clip.source?.audioElement?.src;
  if (existingSrc) {
    const audio = document.createElement('audio');
    audio.src = existingSrc;
    audio.preload = 'none';
    return audio;
  }

  const audio = createAudioElement(clip.file);
  audio.preload = 'none';
  return audio;
}

/**
 * Clone video/audio source for a new clip part.
 * Creates fresh HTMLMediaElements for independent seeking while reusing the
 * existing decoder/runtime state from the original source where possible.
 */
function cloneSourceForPart(clip: TimelineClip): TimelineClip['source'] {
  if (clip.source?.type === 'video' && clip.source.videoElement && clip.file) {
    return {
      ...clip.source,
      videoElement: cloneVideoElementForSplit(clip),
      webCodecsPlayer: clip.source.webCodecsPlayer,
    };
  } else if (clip.source?.type === 'audio' && clip.source.audioElement && clip.file) {
    return {
      ...clip.source,
      audioElement: cloneAudioElementForSplit(clip),
    };
  }
  return clip.source;
}

/**
 * Clone linked audio source for a new clip part.
 */
function cloneLinkedSourceForPart(
  linkedClip: TimelineClip,
  partClipId: string
): TimelineClip['source'] {
  if (linkedClip.source?.type === 'audio' && linkedClip.source.audioElement) {
    if (linkedClip.mixdownBuffer) {
      // Async create audio from mixdown buffer
      import('../../../services/compositionAudioMixer').then(({ compositionAudioMixer }) => {
        const newAudio = compositionAudioMixer.createAudioElement(linkedClip.mixdownBuffer!);
        newAudio.preload = 'none';
        const { clips: currentClips } = useTimelineStore.getState();
        useTimelineStore.setState({
          clips: currentClips.map(c => {
            if (c.id !== partClipId || !c.source) return c;
            return { ...c, source: { ...c.source, audioElement: newAudio } };
          }),
        });
      });
      return { ...linkedClip.source };
    } else if (linkedClip.file && linkedClip.file.size > 0) {
      return {
        ...linkedClip.source,
        audioElement: cloneAudioElementForSplit(linkedClip),
      };
    }
  }
  return linkedClip.source;
}

/**
 * Split a clip into N parts at the given sorted split times (timeline-absolute).
 * Creates all clips at once and applies with a single setState() call.
 * Handles linked audio clips and source element cloning.
 */
function splitClipBatch(clip: TimelineClip, splitTimes: number[], withLinked = true): void {
  const state = useTimelineStore.getState();
  const allClips = state.clips;
  const linkedClip = withLinked && clip.linkedClipId
    ? allClips.find(c => c.id === clip.linkedClipId)
    : undefined;

  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substr(2, 5);

  // Build boundaries: [clipStart, split1, split2, ..., clipEnd]
  const boundaries = [clip.startTime, ...splitTimes, clip.startTime + clip.duration];
  const newParts: TimelineClip[] = [];
  const newLinkedParts: TimelineClip[] = [];

  for (let i = 0; i < boundaries.length - 1; i++) {
    const partStart = boundaries[i];
    const partEnd = boundaries[i + 1];
    const partDuration = partEnd - partStart;
    const partInPoint = clip.inPoint + (partStart - clip.startTime);
    const partOutPoint = partInPoint + partDuration;
    const partId = `clip-${timestamp}-${randomSuffix}-p${i}`;
    const linkedPartId = linkedClip ? `clip-${timestamp}-${randomSuffix}-lp${i}` : undefined;

    // First part keeps the original source; subsequent parts get cloned sources
    const partSource = i === 0 ? clip.source : cloneSourceForPart(clip);

    const partClip: TimelineClip = {
      ...clip,
      ...deepCloneClipProps(clip),
      id: partId,
      startTime: partStart,
      duration: partDuration,
      inPoint: partInPoint,
      outPoint: partOutPoint,
      linkedClipId: linkedClip ? linkedPartId : undefined,
      source: partSource,
      transitionIn: i === 0 ? clip.transitionIn : undefined,
      transitionOut: i === boundaries.length - 2 ? clip.transitionOut : undefined,
    };
    newParts.push(partClip);

    // Create matching linked audio part
    if (linkedClip && linkedPartId) {
      const linkedInPoint = linkedClip.inPoint + (partStart - clip.startTime);
      const linkedSource = i === 0 ? linkedClip.source : cloneLinkedSourceForPart(linkedClip, linkedPartId);

      const linkedPartClip: TimelineClip = {
        ...linkedClip,
        ...deepCloneClipProps(linkedClip),
        id: linkedPartId,
        startTime: partStart,
        duration: partDuration,
        inPoint: linkedInPoint,
        outPoint: linkedInPoint + partDuration,
        linkedClipId: partId,
        source: linkedSource,
      };
      newLinkedParts.push(linkedPartClip);
    }
  }

  // Remove original clip (and linked) and add all new parts
  const removedIds = new Set([clip.id, ...(linkedClip ? [linkedClip.id] : [])]);
  const remainingClips = allClips.filter(c => !removedIds.has(c.id));
  const finalClips = [...remainingClips, ...newParts, ...newLinkedParts];

  // Single setState() call — no stack overflow possible
  useTimelineStore.setState({
    clips: finalClips,
    selectedClipIds: new Set([newParts[newParts.length - 1].id]),
  });
  useTimelineStore.getState().updateDuration();
  useTimelineStore.getState().invalidateCache();
}

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
  const withLinked = (args.withLinked as boolean | undefined) ?? true;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  const clipEnd = clip.startTime + clip.duration;
  if (splitTime <= clip.startTime || splitTime >= clipEnd) {
    return { success: false, error: `Split time ${splitTime}s is outside clip range (${clip.startTime}s - ${clipEnd}s)` };
  }

  // Use splitClipBatch to respect withLinked parameter
  splitClipBatch(clip, [splitTime], withLinked);

  // Visual feedback: split glow at cut position
  if (isAIExecutionActive()) {
    const store = useTimelineStore.getState();
    store.addAIOverlay({ type: 'split-glow', trackId: clip.trackId, timePosition: splitTime, duration: 1000 });
    // Also show on linked audio track
    if (withLinked && clip.linkedClipId) {
      const linked = store.clips.find(c => c.linkedClipId === clip.linkedClipId || c.id === clip.linkedClipId);
      if (linked && linked.trackId !== clip.trackId) {
        store.addAIOverlay({ type: 'split-glow', trackId: linked.trackId, timePosition: splitTime, duration: 1000 });
      }
    }
  }

  return { success: true, data: { splitAt: splitTime, originalClipId: clipId, withLinked } };
}

export async function handleDeleteClip(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const withLinked = (args.withLinked as boolean | undefined) ?? true;
  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  // Visual feedback: delete ghost before removing
  if (isAIExecutionActive()) {
    const store = useTimelineStore.getState();
    store.addAIOverlay({
      type: 'delete-ghost', trackId: clip.trackId,
      timePosition: clip.startTime, width: clip.duration,
      clipName: clip.name, clipColor: getClipColor(clip), duration: 350,
    });
    if (withLinked && clip.linkedClipId) {
      const linked = timelineStore.clips.find(c => c.id === clip.linkedClipId);
      if (linked) {
        store.addAIOverlay({
          type: 'delete-ghost', trackId: linked.trackId,
          timePosition: linked.startTime, width: linked.duration,
          clipName: linked.name, clipColor: getClipColor(linked), duration: 350,
        });
      }
    }
  }

  // removeClip() only deletes linked clip if it's in selectedClipIds
  // So we select the linked clip first when withLinked is true
  if (withLinked && clip.linkedClipId) {
    const linkedClip = timelineStore.clips.find(c => c.id === clip.linkedClipId);
    if (linkedClip) {
      useTimelineStore.setState({
        selectedClipIds: new Set([clipId, clip.linkedClipId]),
      });
    }
  }

  timelineStore.removeClip(clipId);
  return { success: true, data: { deletedClipId: clipId, clipName: clip.name, withLinked } };
}

export async function handleDeleteClips(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipIds = args.clipIds as string[];
  const withLinked = (args.withLinked as boolean | undefined) ?? true;
  const deleted: string[] = [];
  const notFound: string[] = [];

  // When withLinked is true, collect all linked IDs and select them so removeClip deletes them
  if (withLinked) {
    const allLinkedIds = new Set<string>();
    for (const clipId of clipIds) {
      allLinkedIds.add(clipId);
      const clip = useTimelineStore.getState().clips.find(c => c.id === clipId);
      if (clip?.linkedClipId) {
        allLinkedIds.add(clip.linkedClipId);
      }
    }
    useTimelineStore.setState({ selectedClipIds: allLinkedIds });
  }

  for (const clipId of clipIds) {
    const clip = useTimelineStore.getState().clips.find(c => c.id === clipId);
    if (clip) {
      // Visual feedback: delete ghost
      if (isAIExecutionActive()) {
        useTimelineStore.getState().addAIOverlay({
          type: 'delete-ghost', trackId: clip.trackId,
          timePosition: clip.startTime, width: clip.duration,
          clipName: clip.name, clipColor: getClipColor(clip), duration: 350,
        });
      }
      timelineStore.removeClip(clipId);
      deleted.push(clipId);
    } else {
      notFound.push(clipId);
    }
  }

  return {
    success: true,
    data: { deleted, notFound, deletedCount: deleted.length, withLinked },
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
  const newStartTime = (args.newStartTime ?? args.startTime) as number;
  const newTrackId = (args.newTrackId ?? args.trackId) as string | undefined;
  const withLinked = (args.withLinked as boolean | undefined) ?? true;

  if (newStartTime == null || isNaN(newStartTime)) {
    return { success: false, error: 'newStartTime is required and must be a valid number' };
  }

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

  // Visual feedback: animate move from old to new position
  const oldStartTime = clip.startTime;
  if (isAIExecutionActive() && Math.abs(oldStartTime - newStartTime) > 0.01) {
    const store = useTimelineStore.getState();
    store.setAIMovingClip(clipId, oldStartTime, 200);
    // Also animate linked clip
    if (withLinked && clip.linkedClipId) {
      store.setAIMovingClip(clip.linkedClipId, oldStartTime, 200);
    }
  }

  // skipLinked is the inverse of withLinked
  timelineStore.moveClip(clipId, newStartTime, newTrackId, !withLinked);
  return {
    success: true,
    data: {
      clipId,
      newStartTime,
      newTrackId: newTrackId || clip.trackId,
      withLinked,
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

  const oldInPoint = clip.inPoint;
  const oldOutPoint = clip.outPoint;
  timelineStore.trimClip(clipId, inPoint, outPoint);

  // Visual feedback: trim highlight at the changed edge
  if (isAIExecutionActive()) {
    const store = useTimelineStore.getState();
    const trimmedClip = store.clips.find(c => c.id === clipId);
    if (trimmedClip) {
      // Show highlight at left edge if inPoint changed, right edge if outPoint changed
      if (Math.abs(inPoint - oldInPoint) > 0.01) {
        store.addAIOverlay({ type: 'trim-highlight', trackId: trimmedClip.trackId, timePosition: trimmedClip.startTime, duration: 400 });
      }
      if (Math.abs(outPoint - oldOutPoint) > 0.01) {
        store.addAIOverlay({ type: 'trim-highlight', trackId: trimmedClip.trackId, timePosition: trimmedClip.startTime + trimmedClip.duration, duration: 400 });
      }
    }
  }

  return { success: true, data: { clipId, inPoint, outPoint, newDuration: outPoint - inPoint } };
}

export async function handleSplitClipEvenly(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const parts = args.parts as number;
  const withLinked = (args.withLinked as boolean | undefined) ?? true;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }
  if (parts < 2 || !Number.isInteger(parts)) {
    return { success: false, error: `Parts must be an integer >= 2, got: ${parts}` };
  }

  const clipStart = clip.startTime;
  const clipDuration = clip.duration;
  const clipName = clip.name;
  const partDuration = clipDuration / parts;

  // Calculate N-1 split times
  const splitTimes: number[] = [];
  for (let i = 1; i < parts; i++) {
    splitTimes.push(clipStart + partDuration * i);
  }

  if (isAIExecutionActive()) {
    logSplitCheckpoint('split-evenly:start', clip, splitTimes.length, withLinked);
    const trackId = clip.trackId;
    // Bulk split: single state update for all cuts at once
    splitClipBatch(clip, splitTimes, withLinked);
    logSplitCheckpoint('split-evenly:after-batch', clip, splitTimes.length, withLinked);
    // Staggered overlays via CSS animation-delay (single state update, no JS timers)
    const totalAnimMs = Math.min(3000, splitTimes.length * 100);
    const delayStep = splitTimes.length <= 1 ? 0 : totalAnimMs / (splitTimes.length - 1);
    useTimelineStore.getState().addAIOverlaysBatch(
      splitTimes.map((t, i) => ({
        type: 'split-glow' as const, trackId, timePosition: t,
        duration: 1000, animationDelay: Math.round(i * delayStep),
      }))
    );
    logSplitCheckpoint('split-evenly:after-overlays', clip, splitTimes.length, withLinked);
  } else {
    splitClipBatch(clip, splitTimes, withLinked);
  }

  return {
    success: true,
    data: { parts, splitTimes, clipName, partDuration, withLinked },
  };
}

export async function handleSplitClipAtTimes(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipId = args.clipId as string;
  const times = args.times as number[];
  const withLinked = (args.withLinked as boolean | undefined) ?? true;

  const clip = timelineStore.clips.find(c => c.id === clipId);
  if (!clip) {
    return { success: false, error: `Clip not found: ${clipId}` };
  }

  const clipStart = clip.startTime;
  const clipEnd = clip.startTime + clip.duration;

  // Sort and filter to valid times within clip range
  const validTimes = [...times]
    .sort((a, b) => a - b)
    .filter(t => t > clipStart + 0.001 && t < clipEnd - 0.001);

  if (validTimes.length === 0) {
    return { success: false, error: `No valid split times within clip range (${clipStart}s - ${clipEnd}s)` };
  }

  if (isAIExecutionActive()) {
    logSplitCheckpoint('split-at-times:start', clip, validTimes.length, withLinked);
    const trackId = clip.trackId;
    // Bulk split: single state update for all cuts at once
    splitClipBatch(clip, validTimes, withLinked);
    logSplitCheckpoint('split-at-times:after-batch', clip, validTimes.length, withLinked);
    // Staggered overlays via CSS animation-delay (single state update, no JS timers)
    const totalAnimMs = Math.min(3000, validTimes.length * 100);
    const delayStep = validTimes.length <= 1 ? 0 : totalAnimMs / (validTimes.length - 1);
    useTimelineStore.getState().addAIOverlaysBatch(
      validTimes.map((t, i) => ({
        type: 'split-glow' as const, trackId, timePosition: t,
        duration: 1000, animationDelay: Math.round(i * delayStep),
      }))
    );
    logSplitCheckpoint('split-at-times:after-overlays', clip, validTimes.length, withLinked);
  } else {
    splitClipBatch(clip, validTimes, withLinked);
  }

  return {
    success: true,
    data: { splitCount: validTimes.length, splitTimes: validTimes, resultingParts: validTimes.length + 1, withLinked },
  };
}

export async function handleReorderClips(
  args: Record<string, unknown>,
  _timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipIds = args.clipIds as string[];
  const withLinked = (args.withLinked as boolean | undefined) ?? true;

  if (!clipIds || clipIds.length < 2) {
    return { success: false, error: 'Need at least 2 clip IDs to reorder' };
  }

  // Get fresh state
  const state = useTimelineStore.getState();
  const allClips = state.clips;

  // Resolve all clips and validate
  const orderedClips = clipIds.map(id => allClips.find(c => c.id === id));
  const missing = clipIds.filter((_id, i) => !orderedClips[i]);
  if (missing.length > 0) {
    return { success: false, error: `Clips not found: ${missing.join(', ')}` };
  }

  // Find the earliest startTime among the clips to reorder
  const startPosition = Math.min(...orderedClips.map(c => c!.startTime));

  // Build a map of new positions: clipId -> newStartTime
  const newPositions = new Map<string, number>();
  let currentTime = startPosition;

  for (const clip of orderedClips) {
    newPositions.set(clip!.id, currentTime);
    currentTime += clip!.duration;
  }

  // Also move linked audio clips (same delta as their video clip)
  if (withLinked) {
    for (const clip of orderedClips) {
      if (clip!.linkedClipId) {
        const linkedClip = allClips.find(c => c.id === clip!.linkedClipId);
        if (linkedClip && !newPositions.has(linkedClip.id)) {
          const delta = newPositions.get(clip!.id)! - clip!.startTime;
          newPositions.set(linkedClip.id, linkedClip.startTime + delta);
        }
      }
    }
  }

  // Staggered reorder: move clips one by one with visual feedback
  if (isAIExecutionActive()) {
    // Build ordered list of moves (only clips that actually move)
    const moves: { clipId: string; linkedId?: string; newStart: number; linkedNewStart?: number }[] = [];
    for (const clip of orderedClips) {
      const newStart = newPositions.get(clip!.id)!;
      if (Math.abs(clip!.startTime - newStart) > 0.01) {
        const linkedId = withLinked && clip!.linkedClipId ? clip!.linkedClipId : undefined;
        const linkedNewStart = linkedId ? newPositions.get(linkedId) : undefined;
        moves.push({ clipId: clip!.id, linkedId, newStart, linkedNewStart });
      }
    }

    for (let i = 0; i < moves.length; i++) {
      const { clipId, linkedId, newStart, linkedNewStart } = moves[i];
      const store = useTimelineStore.getState();

      // Set FLIP animation data before moving
      const currentClip = store.clips.find(c => c.id === clipId);
      if (currentClip) {
        store.setAIMovingClip(clipId, currentClip.startTime, 200);
      }
      if (linkedId) {
        const linkedClip = store.clips.find(c => c.id === linkedId);
        if (linkedClip) {
          store.setAIMovingClip(linkedId, linkedClip.startTime, 200);
        }
      }

      // Move this clip (and linked) to new position
      useTimelineStore.setState({
        clips: store.clips.map(c => {
          if (c.id === clipId) return { ...c, startTime: Math.max(0, newStart) };
          if (c.id === linkedId && linkedNewStart !== undefined) return { ...c, startTime: Math.max(0, linkedNewStart) };
          return c;
        }),
      });

      if (i < moves.length - 1) {
        await new Promise(resolve => setTimeout(resolve, consumeStaggerDelay(moves.length - 1 - i)));
      }
    }
  } else {
    // Non-AI: apply all at once
    useTimelineStore.setState({
      clips: allClips.map(c => {
        const newStart = newPositions.get(c.id);
        if (newStart !== undefined) {
          return { ...c, startTime: Math.max(0, newStart) };
        }
        return c;
      }),
    });
  }

  // Update duration and invalidate cache once
  useTimelineStore.getState().updateDuration();
  useTimelineStore.getState().invalidateCache();

  return {
    success: true,
    data: {
      reorderedCount: clipIds.length,
      withLinked,
      newOrder: clipIds.map((id, i) => ({
        clipId: id,
        newStartTime: newPositions.get(id),
        position: i + 1,
      })),
    },
  };
}

export async function handleSelectClips(
  args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  const clipIds = args.clipIds as string[];
  timelineStore.selectClips(clipIds);

  // Visual feedback: activate properties panel
  activateDockPanel('clip-properties');

  return { success: true, data: { selectedClipIds: clipIds } };
}

export async function handleClearSelection(
  _args: Record<string, unknown>,
  timelineStore: TimelineStore
): Promise<ToolResult> {
  timelineStore.clearClipSelection();
  return { success: true, data: { message: 'Selection cleared' } };
}

/**
 * Add a clip segment from the media pool with specific in/out points.
 * Self-contained handler — fetches both stores internally.
 */
export async function handleAddClipSegment(
  args: Record<string, unknown>,
): Promise<ToolResult> {
  const mediaFileId = args.mediaFileId as string;
  const trackId = args.trackId as string;
  const startTime = args.startTime as number;
  const inPoint = args.inPoint as number;
  const outPoint = args.outPoint as number;

  if (inPoint >= outPoint) {
    return { success: false, error: 'inPoint must be less than outPoint' };
  }
  if (isNaN(startTime) || isNaN(inPoint) || isNaN(outPoint)) {
    return { success: false, error: 'startTime, inPoint, and outPoint must be valid numbers' };
  }

  const mediaStore = useMediaStore.getState();
  const timelineStore = useTimelineStore.getState();

  // Find media file
  const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
  if (!mediaFile) {
    return { success: false, error: `Media file not found: ${mediaFileId}` };
  }
  if (!mediaFile.file) {
    return { success: false, error: `File object not available for media: ${mediaFileId}. Try re-importing the file.` };
  }

  // Validate track
  const track = timelineStore.tracks.find(t => t.id === trackId);
  if (!track) {
    return { success: false, error: `Track not found: ${trackId}` };
  }

  const duration = outPoint - inPoint;

  // Snapshot clip count before adding
  const clipsBefore = new Set(timelineStore.clips.map(c => c.id));

  // Add the clip (this creates video + linked audio for video files)
  await timelineStore.addClip(trackId, mediaFile.file, startTime, duration, mediaFileId);

  // Find newly created clips
  const clipsAfter = useTimelineStore.getState().clips;
  const newClips = clipsAfter.filter(c => !clipsBefore.has(c.id));

  if (newClips.length === 0) {
    return { success: false, error: 'Failed to create clip' };
  }

  // Trim all new clips (video + linked audio) to the desired segment
  const ts = useTimelineStore.getState();
  for (const clip of newClips) {
    ts.trimClip(clip.id, inPoint, outPoint);
  }

  // Return info about created clips
  const createdClips = useTimelineStore.getState().clips.filter(c => newClips.some(n => n.id === c.id));
  return {
    success: true,
    data: {
      clipCount: createdClips.length,
      clips: createdClips.map(c => ({
        id: c.id,
        trackId: c.trackId,
        startTime: c.startTime,
        duration: c.duration,
        inPoint: c.inPoint,
        outPoint: c.outPoint,
        linkedClipId: c.linkedClipId,
      })),
    },
  };
}
