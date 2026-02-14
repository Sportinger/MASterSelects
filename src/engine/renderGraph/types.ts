// Render Graph types — DAG of render passes for GPU execution

import type { BlendMode } from '../../types/index.ts';

/** Types of render passes */
export type RenderPassType =
  | 'clear'
  | 'composite'
  | 'effect'
  | 'nestedComp'
  | 'output'
  | 'slice';

/** Handle to a GPU resource managed by the render graph */
export interface ResourceHandle {
  id: string;
  type: 'texture' | 'externalTexture' | 'buffer';
  persistent: boolean;
}

/** A single render pass node in the DAG */
export interface RenderPassNode {
  id: string;
  type: RenderPassType;
  inputs: ResourceHandle[];
  outputs: ResourceHandle[];
  dependsOn: string[];
  config: RenderPassConfig;
}

/** Union of all pass configs */
export type RenderPassConfig =
  | ClearPassConfig
  | CompositePassConfig
  | EffectPassConfig
  | NestedCompPassConfig
  | OutputPassConfig
  | SlicePassConfig;

export interface ClearPassConfig {
  type: 'clear';
  clearColor: { r: number; g: number; b: number; a: number };
}

export interface CompositePassConfig {
  type: 'composite';
  layerId: string;
  blendMode: BlendMode;
  opacity: number;
  sourceAspect: number;
  outputAspect: number;
  hasMask: boolean;
  isVideo: boolean;
}

export interface EffectPassConfig {
  type: 'effect';
  layerId: string;
  effectId: string;
  effectType: string;
  params: Record<string, unknown>;
}

export interface NestedCompPassConfig {
  type: 'nestedComp';
  compositionId: string;
  width: number;
  height: number;
  childPasses: RenderPassNode[];
}

export interface OutputPassConfig {
  type: 'output';
  targetId: string;
  showTransparencyGrid: boolean;
}

export interface SlicePassConfig {
  type: 'slice';
  targetId: string;
  sliceConfig: unknown;
}

/**
 * The full render graph — DAG of passes with topological execution order.
 */
export interface RenderGraph {
  passes: RenderPassNode[];
  executionOrder: string[];
  resources: Map<string, ResourceHandle>;
  version: number;
}
