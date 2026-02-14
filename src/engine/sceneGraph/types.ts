// Scene Graph types — pure data tree representing the composition hierarchy

import type { ClipTransform, Effect, ClipMask, BlendMode } from '../../types/index.ts';

/** Source type for scene nodes */
export type SceneNodeType = 'video' | 'image' | 'text' | 'solid' | 'composition';

/**
 * SceneNode — pure data object, no DOM refs, no GPU handles.
 * Represents a single renderable item in the composition tree.
 */
export interface SceneNode {
  id: string;
  type: SceneNodeType;
  clipId: string;
  mediaFileId?: string;
  compositionId?: string;

  // Temporal
  timelineStart: number;
  duration: number;
  inPoint: number;
  outPoint: number;
  speed: number;
  reversed: boolean;

  // Static transform (before keyframe interpolation)
  transform: ClipTransform;
  effects: Effect[];
  masks?: ClipMask[];

  // Hierarchy
  children: SceneNode[];
  childTrackIndex: number;
  parentNode: SceneNode | null;

  // Versionning (for dirty tracking)
  version: number;
  transformVersion: number;
  effectsVersion: number;
  structureVersion: number;

  // Source metadata (for adapter to locate DOM refs)
  sourceType: 'video' | 'audio' | 'image' | 'text' | 'solid';
  hasKeyframes: boolean;
  trackId: string;

  // Transition data
  transitionProgress?: number;
  transitionOpacityMultiplier?: number;
}

/**
 * Resolved transform after keyframe interpolation.
 * Values are in render-ready units (rotation in radians, etc.)
 */
export interface ResolvedTransform {
  opacity: number;
  blendMode: BlendMode;
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number }; // radians
}

/**
 * EvaluatedNode — result of evaluating a SceneNode at a specific time.
 * Flat, sorted by render order, with all transforms resolved.
 */
export interface EvaluatedNode {
  sceneNode: SceneNode;
  resolvedTransform: ResolvedTransform;
  resolvedEffects: Effect[];
  sourceTime: number;
  localTime: number;
  compositionDepth: number;
  parentCompositionId?: string;
}

/**
 * The full scene graph — tree structure with fast lookup.
 */
export interface SceneGraph {
  roots: SceneNode[];
  nodeById: Map<string, SceneNode>;
  version: number;
}
