// SnapshotManager — creates undo/redo snapshots with structural sharing.
// Only changed clips are cloned; unchanged clips share references with the previous snapshot.
// DOM refs are NOT cloned — they live in the DomRefRegistry.

import type {
  SerializedClipState,
  HistorySnapshotV2,
} from './types.ts';
import type { TimelineClip, TimelineTrack, Keyframe } from '../../types/index.ts';
import type { TimelineMarker } from '../../stores/timeline/types.ts';
import { Logger } from '../../services/logger.ts';

const log = Logger.create('SnapshotManager');

export class SnapshotManager {
  /** Set of clip IDs that have changed since the last snapshot */
  private changedClipIds = new Set<string>();
  /** Previous clip object references for auto-detection */
  private prevClipRefs = new Map<string, TimelineClip>();

  /**
   * Track a clip as changed. Call this whenever a clip is modified.
   */
  trackChange(clipId: string): void {
    this.changedClipIds.add(clipId);
  }

  /**
   * Track multiple clip changes at once.
   */
  trackChanges(clipIds: string[]): void {
    for (const id of clipIds) {
      this.changedClipIds.add(id);
    }
  }

  /**
   * Create a snapshot with structural sharing.
   * Only clips in changedClipIds are serialized fresh.
   * Unchanged clips share references from the previous snapshot.
   */
  createSnapshot(
    label: string,
    currentClips: TimelineClip[],
    currentTracks: TimelineTrack[],
    currentKeyframes: Map<string, Keyframe[]>,
    currentMarkers: TimelineMarker[],
    prev?: HistorySnapshotV2
  ): HistorySnapshotV2 {
    // Auto-detect changed clips via reference comparison (Zustand immutable updates)
    if (this.prevClipRefs.size > 0) {
      for (const clip of currentClips) {
        if (this.prevClipRefs.get(clip.id) !== clip) {
          this.changedClipIds.add(clip.id);
        }
      }
      // New clips not in prev
      for (const clip of currentClips) {
        if (!this.prevClipRefs.has(clip.id)) {
          this.changedClipIds.add(clip.id);
        }
      }
    }

    // Update reference map for next diff
    this.prevClipRefs.clear();
    for (const clip of currentClips) {
      this.prevClipRefs.set(clip.id, clip);
    }

    const changedIds = new Set(this.changedClipIds);

    // Build previous clip lookup for sharing
    const prevClipById = new Map<string, SerializedClipState>();
    if (prev) {
      for (const clip of prev.clips) {
        prevClipById.set(clip.id, clip);
      }
    }

    // Build clips array with structural sharing
    const clips: SerializedClipState[] = currentClips.map(clip => {
      if (changedIds.has(clip.id) || !prevClipById.has(clip.id)) {
        // Changed or new — serialize fresh (without DOM refs)
        return this.serializeClip(clip);
      }
      // Unchanged — share reference from previous snapshot
      return prevClipById.get(clip.id)!;
    });

    // Convert keyframes Map to plain object
    const clipKeyframes: Record<string, Keyframe[]> = {};
    currentKeyframes.forEach((kfs, clipId) => {
      if (changedIds.has(clipId) || !prev?.clipKeyframes[clipId]) {
        // Clone keyframes for changed clips
        clipKeyframes[clipId] = kfs.map(k => ({ ...k }));
      } else {
        // Share reference for unchanged clips
        clipKeyframes[clipId] = prev.clipKeyframes[clipId];
      }
    });

    const snapshot: HistorySnapshotV2 = {
      timestamp: Date.now(),
      label,
      clips,
      tracks: currentTracks.map(t => ({ ...t })),
      clipKeyframes,
      markers: currentMarkers.map(m => ({ ...m })),
      changedClipIds: [...changedIds],
    };

    // Clear change tracking
    this.changedClipIds.clear();

    log.debug('Snapshot created', {
      label,
      totalClips: clips.length,
      changedClips: changedIds.size,
      sharedClips: clips.length - changedIds.size,
    });

    return snapshot;
  }

  /**
   * Apply a snapshot to restore state.
   * Returns the deserialized clips (caller must re-link DOM refs via DomRefRegistry).
   */
  applySnapshot(snapshot: HistorySnapshotV2): {
    clips: SerializedClipState[];
    tracks: TimelineTrack[];
    clipKeyframes: Map<string, Keyframe[]>;
    markers: TimelineMarker[];
  } {
    // Convert clipKeyframes back to Map
    const keyframesMap = new Map<string, Keyframe[]>();
    for (const [clipId, kfs] of Object.entries(snapshot.clipKeyframes)) {
      keyframesMap.set(clipId, kfs.map(k => ({ ...k })));
    }

    return {
      clips: snapshot.clips.map(c => ({ ...c })),
      tracks: snapshot.tracks.map(t => ({ ...t })),
      clipKeyframes: keyframesMap,
      markers: snapshot.markers.map(m => ({ ...m })),
    };
  }

  /**
   * Get the number of pending changes.
   */
  getPendingChangeCount(): number {
    return this.changedClipIds.size;
  }

  /**
   * Clear pending changes without creating a snapshot.
   */
  clearPendingChanges(): void {
    this.changedClipIds.clear();
  }

  // === Private ===

  private serializeClip(clip: TimelineClip): SerializedClipState {
    return {
      id: clip.id,
      trackId: clip.trackId,
      name: clip.name,
      startTime: clip.startTime,
      duration: clip.duration,
      inPoint: clip.inPoint,
      outPoint: clip.outPoint,
      sourceType: clip.source?.type ?? 'video',
      mediaFileId: clip.source?.mediaFileId ?? clip.mediaFileId,
      transform: clip.transform ? {
        opacity: clip.transform.opacity,
        blendMode: clip.transform.blendMode,
        position: { ...clip.transform.position },
        scale: { ...clip.transform.scale },
        rotation: { ...clip.transform.rotation },
      } : undefined!,
      effects: clip.effects?.map(e => ({
        id: e.id,
        name: e.name,
        type: e.type,
        enabled: e.enabled,
        params: { ...e.params },
      })) ?? [],
      speed: clip.speed,
      reversed: clip.reversed,
      isComposition: clip.isComposition,
      compositionId: clip.compositionId,
      masks: clip.masks?.map(m => ({ ...m, vertices: m.vertices.map(v => ({ ...v, handleIn: { ...v.handleIn }, handleOut: { ...v.handleOut } })) })),
      textProperties: clip.textProperties ? { ...clip.textProperties } : undefined,
      solidColor: clip.solidColor,
      transitionIn: clip.transitionIn ? { ...clip.transitionIn } : undefined,
      transitionOut: clip.transitionOut ? { ...clip.transitionOut } : undefined,
    };
  }
}
