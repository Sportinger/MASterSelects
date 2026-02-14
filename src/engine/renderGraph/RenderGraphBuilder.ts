// RenderGraphBuilder — builds a RenderGraph DAG from EvaluatedNode[] + render targets.
// Each node in the graph represents a GPU render pass.

import type {
  RenderGraph,
  RenderPassNode,
  ResourceHandle,
  ClearPassConfig,
  CompositePassConfig,
  EffectPassConfig,
  NestedCompPassConfig,
  OutputPassConfig,
} from './types.ts';
import type { EvaluatedNode } from '../sceneGraph/types.ts';
import { useRenderTargetStore } from '../../stores/renderTargetStore.ts';

let passIdCounter = 0;
function nextPassId(): string {
  return `pass_${++passIdCounter}`;
}

function makeResourceHandle(id: string, type: ResourceHandle['type'], persistent = false): ResourceHandle {
  return { id, type, persistent };
}

export class RenderGraphBuilder {
  private nextVersion = 1;

  /**
   * Build a render graph from evaluated scene nodes.
   */
  build(
    evaluated: EvaluatedNode[],
    outputWidth: number,
    outputHeight: number
  ): RenderGraph {
    const passes: RenderPassNode[] = [];
    const resources = new Map<string, ResourceHandle>();

    // Register persistent resources
    const pingRes = makeResourceHandle('pingBuffer', 'texture', true);
    const pongRes = makeResourceHandle('pongBuffer', 'texture', true);
    resources.set(pingRes.id, pingRes);
    resources.set(pongRes.id, pongRes);

    // 1. Clear pass
    const clearPass = this.buildClearPass(pingRes);
    passes.push(clearPass);

    let prevPassId = clearPass.id;
    let readRes = pingRes;
    let writeRes = pongRes;

    // 2. Composite each layer
    for (const node of evaluated) {
      if (node.sceneNode.type === 'composition') {
        // Build nested composition passes
        const nestedPassResult = this.buildNestedCompPasses(
          node, resources, outputWidth, outputHeight, prevPassId
        );
        passes.push(...nestedPassResult.passes);

        // Composite the nested result onto the accumulator
        const compositePass = this.buildCompositePass(
          node, readRes, writeRes, nestedPassResult.outputRes, prevPassId
        );
        passes.push(compositePass);
        prevPassId = compositePass.id;
      } else {
        // Build source resource
        const sourceRes = this.getSourceResource(node, resources);

        // Apply effects if any
        let effectSourceRes = sourceRes;
        if (node.resolvedEffects.length > 0) {
          const effectResult = this.buildEffectPasses(
            node, sourceRes, resources, prevPassId
          );
          passes.push(...effectResult.passes);
          effectSourceRes = effectResult.outputRes;
          prevPassId = effectResult.lastPassId;
        }

        // Composite onto accumulator
        const compositePass = this.buildCompositePass(
          node, readRes, writeRes, effectSourceRes, prevPassId
        );
        passes.push(compositePass);
        prevPassId = compositePass.id;
      }

      // Swap read/write
      [readRes, writeRes] = [writeRes, readRes];
    }

    // 3. Output passes
    const outputPasses = this.buildOutputPasses(readRes, prevPassId);
    passes.push(...outputPasses);

    // Build execution order (already topologically sorted by construction)
    const executionOrder = passes.map(p => p.id);

    return {
      passes,
      resources,
      executionOrder,
      version: this.nextVersion++,
    };
  }

  // === Private pass builders ===

  private buildClearPass(targetRes: ResourceHandle): RenderPassNode {
    return {
      id: nextPassId(),
      type: 'clear',
      inputs: [],
      outputs: [targetRes],
      dependsOn: [],
      config: {
        type: 'clear',
        clearColor: { r: 0, g: 0, b: 0, a: 0 },
      } as ClearPassConfig,
    };
  }

  private buildCompositePass(
    node: EvaluatedNode,
    readRes: ResourceHandle,
    writeRes: ResourceHandle,
    sourceRes: ResourceHandle,
    dependsOn: string
  ): RenderPassNode {
    const { resolvedTransform, sceneNode } = node;
    const isVideo = sceneNode.type === 'video';
    const hasMask = !!(sceneNode.masks && sceneNode.masks.length > 0);

    return {
      id: nextPassId(),
      type: 'composite',
      inputs: [readRes, sourceRes],
      outputs: [writeRes],
      dependsOn: [dependsOn],
      config: {
        type: 'composite',
        layerId: sceneNode.clipId,
        blendMode: resolvedTransform.blendMode,
        opacity: resolvedTransform.opacity,
        sourceAspect: 1, // Will be resolved at execution time
        outputAspect: 1,
        hasMask,
        isVideo,
      } as CompositePassConfig,
    };
  }

  private buildEffectPasses(
    node: EvaluatedNode,
    sourceRes: ResourceHandle,
    resources: Map<string, ResourceHandle>,
    prevPassId: string
  ): { passes: RenderPassNode[]; outputRes: ResourceHandle; lastPassId: string } {
    const passes: RenderPassNode[] = [];
    const effectTempA = makeResourceHandle('effectTempA', 'texture', true);
    const effectTempB = makeResourceHandle('effectTempB', 'texture', true);
    resources.set(effectTempA.id, effectTempA);
    resources.set(effectTempB.id, effectTempB);

    let currentInput = sourceRes;
    let currentOutput = effectTempA;
    let lastId = prevPassId;

    for (const effect of node.resolvedEffects) {
      if (!effect.enabled) continue;

      const pass: RenderPassNode = {
        id: nextPassId(),
        type: 'effect',
        inputs: [currentInput],
        outputs: [currentOutput],
        dependsOn: [lastId],
        config: {
          type: 'effect',
          layerId: node.sceneNode.clipId,
          effectId: effect.id,
          effectType: effect.type,
          params: effect.params,
        } as EffectPassConfig,
      };

      passes.push(pass);
      lastId = pass.id;

      // Swap for ping-pong
      [currentInput, currentOutput] = [currentOutput, currentInput === sourceRes ? effectTempB : currentInput];
    }

    return { passes, outputRes: currentInput, lastPassId: lastId };
  }

  private buildNestedCompPasses(
    node: EvaluatedNode,
    resources: Map<string, ResourceHandle>,
    outputWidth: number,
    outputHeight: number,
    prevPassId: string
  ): { passes: RenderPassNode[]; outputRes: ResourceHandle } {
    const compId = node.sceneNode.compositionId ?? node.sceneNode.clipId;
    const outputRes = makeResourceHandle(`nestedComp_${compId}`, 'texture', false);
    resources.set(outputRes.id, outputRes);

    // Build child passes inline (the executor will handle them)
    const childPasses: RenderPassNode[] = [];

    // Build clear + composite chain for children
    const childClear: RenderPassNode = {
      id: nextPassId(),
      type: 'clear',
      inputs: [],
      outputs: [outputRes],
      dependsOn: [prevPassId],
      config: { type: 'clear', clearColor: { r: 0, g: 0, b: 0, a: 0 } } as ClearPassConfig,
    };
    childPasses.push(childClear);

    const nestedPass: RenderPassNode = {
      id: nextPassId(),
      type: 'nestedComp',
      inputs: [],
      outputs: [outputRes],
      dependsOn: [prevPassId],
      config: {
        type: 'nestedComp',
        compositionId: compId,
        width: outputWidth,
        height: outputHeight,
        childPasses,
      } as NestedCompPassConfig,
    };

    return { passes: [nestedPass], outputRes };
  }

  private buildOutputPasses(
    finalRes: ResourceHandle,
    prevPassId: string
  ): RenderPassNode[] {
    const passes: RenderPassNode[] = [];
    const activeTargets = useRenderTargetStore.getState().getActiveCompTargets();

    for (const target of activeTargets) {
      passes.push({
        id: nextPassId(),
        type: 'output',
        inputs: [finalRes],
        outputs: [],
        dependsOn: [prevPassId],
        config: {
          type: 'output',
          targetId: target.id,
          showTransparencyGrid: target.showTransparencyGrid,
        } as OutputPassConfig,
      });
    }

    return passes;
  }

  private getSourceResource(
    node: EvaluatedNode,
    resources: Map<string, ResourceHandle>
  ): ResourceHandle {
    const id = `source_${node.sceneNode.clipId}`;
    const isVideo = node.sceneNode.type === 'video';
    const resType = isVideo ? 'externalTexture' as const : 'texture' as const;
    const res = makeResourceHandle(id, resType, false);
    resources.set(id, res);
    return res;
  }
}
