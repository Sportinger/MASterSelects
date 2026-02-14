// SceneGraphBuilder — builds a SceneGraph from the current timeline + media state.
// Reads from stores once, builds a recursive tree of SceneNodes.
// Caches the graph when clips/tracks reference-identity is unchanged.

import type { TimelineClip, TimelineTrack } from '../../types/index.ts';
import type { SceneNode, SceneGraph, SceneNodeType } from './types.ts';
import { useTimelineStore } from '../../stores/timeline/index.ts';
import { useMediaStore } from '../../stores/mediaStore/index.ts';
import { DEFAULT_TRANSFORM } from '../../stores/timeline/constants.ts';

export class SceneGraphBuilder {
  // Cache invalidation: keep previous input references
  private prevClips: TimelineClip[] | null = null;
  private prevTracks: TimelineTrack[] | null = null;
  private prevActiveCompId: string | null = null;
  private cachedGraph: SceneGraph | null = null;
  private nextVersion = 1;

  /**
   * Build (or return cached) scene graph from current store state.
   */
  build(): SceneGraph {
    const timelineState = useTimelineStore.getState();
    const mediaState = useMediaStore.getState();

    const { clips, tracks } = timelineState;
    const activeCompId = mediaState.activeCompositionId || 'default';

    // Reference-identity cache check
    if (
      this.cachedGraph &&
      this.prevClips === clips &&
      this.prevTracks === tracks &&
      this.prevActiveCompId === activeCompId
    ) {
      return this.cachedGraph;
    }

    // Build fresh graph
    const nodeById = new Map<string, SceneNode>();
    const videoTracks = tracks.filter(t => t.type === 'video');
    const roots: SceneNode[] = [];

    for (let trackIdx = 0; trackIdx < videoTracks.length; trackIdx++) {
      const track = videoTracks[trackIdx];
      const trackClips = clips.filter(c => c.trackId === track.id);

      for (const clip of trackClips) {
        const node = this.buildNode(clip, trackIdx, track.id, timelineState, 0);
        roots.push(node);
        this.registerNodes(node, nodeById);
      }
    }

    const graph: SceneGraph = {
      roots,
      nodeById,
      version: this.nextVersion++,
    };

    // Update cache
    this.prevClips = clips;
    this.prevTracks = tracks;
    this.prevActiveCompId = activeCompId;
    this.cachedGraph = graph;

    return graph;
  }

  /**
   * Force cache invalidation (e.g. on composition switch).
   */
  invalidate(): void {
    this.prevClips = null;
    this.prevTracks = null;
    this.prevActiveCompId = null;
    this.cachedGraph = null;
  }

  // === Private ===

  private buildNode(
    clip: TimelineClip,
    trackIndex: number,
    trackId: string,
    timelineState: ReturnType<typeof useTimelineStore.getState>,
    depth: number
  ): SceneNode {
    const type = this.resolveNodeType(clip);
    const hasKeyframes = timelineState.hasKeyframes(clip.id);

    const node: SceneNode = {
      id: `sg_${clip.id}`,
      type,
      clipId: clip.id,
      mediaFileId: clip.source?.mediaFileId ?? clip.mediaFileId,
      compositionId: clip.isComposition ? (clip.compositionId ?? undefined) : undefined,

      // Temporal
      timelineStart: clip.startTime,
      duration: clip.duration,
      inPoint: clip.inPoint,
      outPoint: clip.outPoint,
      speed: clip.speed ?? 1,
      reversed: clip.reversed ?? false,

      // Static transform (merge with defaults — clip.transform may be partial)
      transform: {
        opacity: clip.transform?.opacity ?? DEFAULT_TRANSFORM.opacity,
        blendMode: clip.transform?.blendMode ?? DEFAULT_TRANSFORM.blendMode,
        position: {
          x: clip.transform?.position?.x ?? DEFAULT_TRANSFORM.position.x,
          y: clip.transform?.position?.y ?? DEFAULT_TRANSFORM.position.y,
          z: clip.transform?.position?.z ?? DEFAULT_TRANSFORM.position.z,
        },
        scale: {
          x: clip.transform?.scale?.x ?? DEFAULT_TRANSFORM.scale.x,
          y: clip.transform?.scale?.y ?? DEFAULT_TRANSFORM.scale.y,
        },
        rotation: {
          x: clip.transform?.rotation?.x ?? DEFAULT_TRANSFORM.rotation.x,
          y: clip.transform?.rotation?.y ?? DEFAULT_TRANSFORM.rotation.y,
          z: clip.transform?.rotation?.z ?? DEFAULT_TRANSFORM.rotation.z,
        },
      },
      effects: clip.effects ?? [],
      masks: clip.masks,

      // Hierarchy
      children: [],
      childTrackIndex: trackIndex,
      parentNode: null,

      // Versions — use simple hash-like counters
      version: this.nextVersion,
      transformVersion: this.nextVersion,
      effectsVersion: this.nextVersion,
      structureVersion: this.nextVersion,

      // Source metadata
      sourceType: clip.source?.type ?? (clip.isComposition ? 'video' : 'video'),
      hasKeyframes,
      trackId,

      // Transition data
      transitionProgress: undefined,
      transitionOpacityMultiplier: undefined,
    };

    // Build children for composition nodes
    if (type === 'composition' && clip.nestedClips && clip.nestedTracks) {
      node.children = this.buildNestedChildren(clip, timelineState, depth + 1);
      for (const child of node.children) {
        child.parentNode = node;
      }
    }

    return node;
  }

  private buildNestedChildren(
    compClip: TimelineClip,
    timelineState: ReturnType<typeof useTimelineStore.getState>,
    depth: number
  ): SceneNode[] {
    const nestedClips = compClip.nestedClips!;
    const nestedTracks = compClip.nestedTracks!;
    const videoTracks = nestedTracks.filter(t => t.type === 'video' && t.visible !== false);

    const children: SceneNode[] = [];

    for (let trackIdx = 0; trackIdx < videoTracks.length; trackIdx++) {
      const track = videoTracks[trackIdx];
      const trackClips = nestedClips.filter(c => c.trackId === track.id);

      for (const clip of trackClips) {
        const node = this.buildNode(clip, trackIdx, track.id, timelineState, depth);
        children.push(node);
      }
    }

    return children;
  }

  private resolveNodeType(clip: TimelineClip): SceneNodeType {
    if (clip.isComposition) return 'composition';
    switch (clip.source?.type) {
      case 'image': return 'image';
      case 'text': return 'text';
      case 'solid': return 'solid';
      default: return 'video';
    }
  }

  private registerNodes(node: SceneNode, map: Map<string, SceneNode>): void {
    map.set(node.id, node);
    for (const child of node.children) {
      this.registerNodes(child, map);
    }
  }
}
