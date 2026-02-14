// SceneGraphEvaluator — evaluates a SceneGraph at a specific time.
// Performs visibility culling, keyframe interpolation, and outputs
// a flat array of EvaluatedNodes sorted by render order.

import type { SceneNode, SceneGraph, EvaluatedNode, ResolvedTransform } from './types.ts';
import type { ClipTransform, BlendMode } from '../../types/index.ts';
import { useTimelineStore } from '../../stores/timeline/index.ts';
import { DirtyTracker } from '../dirtyTracking/DirtyTracker.ts';

export class SceneGraphEvaluator {
  private dirtyTracker = new DirtyTracker();
  /**
   * Evaluate the scene graph at a given time.
   * Returns a flat list of EvaluatedNodes, sorted by render order (bottom to top).
   */
  evaluate(graph: SceneGraph, currentTime: number): EvaluatedNode[] {
    const results: EvaluatedNode[] = [];
    const timelineState = useTimelineStore.getState();

    // Update dirty tracking before evaluation
    this.dirtyTracker.update(graph, currentTime);

    for (const root of graph.roots) {
      this.evaluateNode(root, currentTime, 0, undefined, results, timelineState);
    }

    return results;
  }

  /**
   * Evaluate nested composition children at a specific composition-local time.
   * Used by the adapter when building NestedCompositionData layers.
   */
  evaluateNestedChildren(
    parentNode: SceneNode,
    compLocalTime: number,
    depth: number,
    parentCompId: string | undefined
  ): EvaluatedNode[] {
    const results: EvaluatedNode[] = [];
    const timelineState = useTimelineStore.getState();

    for (const child of parentNode.children) {
      this.evaluateNode(child, compLocalTime, depth, parentCompId, results, timelineState);
    }

    return results;
  }

  // === Private ===

  private evaluateNode(
    node: SceneNode,
    currentTime: number,
    depth: number,
    parentCompId: string | undefined,
    results: EvaluatedNode[],
    timelineState: ReturnType<typeof useTimelineStore.getState>
  ): void {
    // Visibility culling: is this node active at the current time?
    const nodeEnd = node.timelineStart + node.duration;
    if (currentTime < node.timelineStart || currentTime >= nodeEnd) {
      return;
    }

    // Dirty tracking: skip interpolation for clean nodes
    const cached = this.dirtyTracker.getOrReuse(node, currentTime);
    if (cached) {
      results.push(cached);
      return;
    }

    // Calculate local time within the clip
    const clipLocalTime = currentTime - node.timelineStart;

    // Calculate source time (handles speed + reverse)
    const sourceTime = this.calculateSourceTime(node, clipLocalTime, timelineState);

    // Interpolate transform using keyframes
    const resolvedTransform = this.resolveTransform(node, clipLocalTime, timelineState);

    // Interpolate effect parameters
    const resolvedEffects = this.resolveEffects(node, clipLocalTime, timelineState);

    const evaluated: EvaluatedNode = {
      sceneNode: node,
      resolvedTransform,
      resolvedEffects,
      sourceTime,
      localTime: clipLocalTime,
      compositionDepth: depth,
      parentCompositionId: parentCompId,
    };

    // For composition nodes, recursively evaluate children
    if (node.type === 'composition' && node.children.length > 0) {
      // The children exist at nested-comp time (sourceTime mapped into the comp)
      const nestedChildren: EvaluatedNode[] = [];
      for (const child of node.children) {
        this.evaluateNode(
          child,
          sourceTime,
          depth + 1,
          node.compositionId,
          nestedChildren,
          timelineState
        );
      }

      // Attach children to the evaluated node (adapter will use these)
      (evaluated as any).nestedEvaluatedChildren = nestedChildren;
    }

    // Cache the evaluation for dirty tracking reuse in subsequent frames
    this.dirtyTracker.cacheEvaluation(node.id, evaluated);

    results.push(evaluated);
  }

  private resolveTransform(
    node: SceneNode,
    clipLocalTime: number,
    timelineState: ReturnType<typeof useTimelineStore.getState>
  ): ResolvedTransform {
    let transform: ClipTransform;

    if (node.hasKeyframes) {
      // Use the store's interpolation which handles parenting
      transform = timelineState.getInterpolatedTransform(node.clipId, clipLocalTime);
    } else {
      transform = node.transform;
    }

    // Defensive: ensure sub-objects exist (clip.transform can be partial)
    const pos = transform?.position;
    const scl = transform?.scale;
    const rot = transform?.rotation;

    // Convert to resolved format (rotation: degrees → radians)
    return {
      opacity: transform?.opacity ?? 1,
      blendMode: (transform?.blendMode ?? 'normal') as BlendMode,
      position: {
        x: pos?.x ?? 0,
        y: pos?.y ?? 0,
        z: pos?.z ?? 0,
      },
      scale: {
        x: scl?.x ?? 1,
        y: scl?.y ?? 1,
      },
      rotation: {
        x: ((rot?.x ?? 0) * Math.PI) / 180,
        y: ((rot?.y ?? 0) * Math.PI) / 180,
        z: ((rot?.z ?? 0) * Math.PI) / 180,
      },
    };
  }

  private resolveEffects(
    node: SceneNode,
    clipLocalTime: number,
    timelineState: ReturnType<typeof useTimelineStore.getState>
  ): typeof node.effects {
    // Delegate to the store's interpolation (handles effect keyframes)
    return timelineState.getInterpolatedEffects(node.clipId, clipLocalTime);
  }

  private calculateSourceTime(
    node: SceneNode,
    clipLocalTime: number,
    timelineState: ReturnType<typeof useTimelineStore.getState>
  ): number {
    // Use the store's source time calculation (handles speed keyframes + integration)
    const sourceTime = timelineState.getSourceTimeForClip(node.clipId, clipLocalTime);

    // Compute the initial speed to determine direction
    const initialSpeed = timelineState.getInterpolatedSpeed(node.clipId, 0);
    const startPoint = initialSpeed >= 0 ? node.inPoint : node.outPoint;

    return Math.max(node.inPoint, Math.min(node.outPoint, startPoint + sourceTime));
  }
}
