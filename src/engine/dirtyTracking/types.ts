// Dirty Tracking types — per-node change detection for frame-to-frame optimization

import type { EvaluatedNode } from '../sceneGraph/types.ts';

/** Flags indicating what changed on a node since last frame */
export interface DirtyFlags {
  transform: boolean;
  effects: boolean;
  source: boolean;
  structure: boolean;
  time: boolean;
  /** True if any flag is dirty */
  any: boolean;
}

/** Tracked state for a single node */
export interface TrackedNodeState {
  nodeId: string;
  version: number;
  transformVersion: number;
  effectsVersion: number;
  structureVersion: number;
  lastTime: number;
  lastEvaluation: EvaluatedNode | null;
}

/** Statistics for dirty tracking effectiveness */
export interface DirtyTrackingStats {
  totalNodes: number;
  dirtyNodes: number;
  cleanNodes: number;
  /** Percentage of nodes that could be skipped (0-1) */
  skipRate: number;
  /** Total interpolation calls saved this frame */
  interpolationsSaved: number;
}
