// Clip state update helpers - reduces repetitive clips.map() patterns
// and provides batched updates for better performance

import type { TimelineClip } from '../../../types';

/**
 * Update a single clip by ID.
 * Avoids repeated get().clips.map() pattern.
 */
export function updateClipById(
  clips: TimelineClip[],
  id: string,
  updates: Partial<TimelineClip>
): TimelineClip[] {
  return clips.map(c => c.id === id ? { ...c, ...updates } : c);
}

/**
 * Update multiple clips by ID in a single pass.
 * More efficient than chaining multiple updateClipById calls.
 */
export function updateClipsById(
  clips: TimelineClip[],
  updates: Map<string, Partial<TimelineClip>>
): TimelineClip[] {
  if (updates.size === 0) return clips;

  return clips.map(c => {
    const clipUpdates = updates.get(c.id);
    return clipUpdates ? { ...c, ...clipUpdates } : c;
  });
}

/**
 * Update clips matching a predicate.
 */
export function updateClipsWhere(
  clips: TimelineClip[],
  predicate: (clip: TimelineClip) => boolean,
  updates: Partial<TimelineClip> | ((clip: TimelineClip) => Partial<TimelineClip>)
): TimelineClip[] {
  return clips.map(c => {
    if (!predicate(c)) return c;
    const clipUpdates = typeof updates === 'function' ? updates(c) : updates;
    return { ...c, ...clipUpdates };
  });
}

/**
 * Batch update builder for complex multi-clip updates.
 * Collects updates and applies them in a single pass.
 */
export class ClipUpdateBatch {
  private updates = new Map<string, Partial<TimelineClip>>();

  /**
   * Queue an update for a clip.
   */
  update(clipId: string, updates: Partial<TimelineClip>): this {
    const existing = this.updates.get(clipId) || {};
    this.updates.set(clipId, { ...existing, ...updates });
    return this;
  }

  /**
   * Queue updates for multiple clips.
   */
  updateMany(clipIds: string[], updates: Partial<TimelineClip>): this {
    for (const id of clipIds) {
      this.update(id, updates);
    }
    return this;
  }

  /**
   * Apply all queued updates to clips array.
   */
  apply(clips: TimelineClip[]): TimelineClip[] {
    return updateClipsById(clips, this.updates);
  }

  /**
   * Check if any updates are queued.
   */
  hasUpdates(): boolean {
    return this.updates.size > 0;
  }

  /**
   * Clear all queued updates.
   */
  clear(): void {
    this.updates.clear();
  }
}

/**
 * Create a new batch update builder.
 */
export function createUpdateBatch(): ClipUpdateBatch {
  return new ClipUpdateBatch();
}
