// RenderGraphExecutor — walks the render graph in execution order and dispatches GPU commands.
// Delegates actual rendering to existing pipelines (CompositorPipeline, EffectsPipeline, etc.)

import type { RenderGraph, RenderPassNode, NestedCompPassConfig, OutputPassConfig } from './types.ts';
import type { RenderDeps } from '../render/RenderDispatcher.ts';
import { Logger } from '../../services/logger.ts';

const log = Logger.create('RenderGraphExecutor');

export class RenderGraphExecutor {
  /**
   * Execute the full render graph for one frame.
   * Uses a single CommandEncoder for the entire frame, then submits.
   */
  execute(graph: RenderGraph, deps: RenderDeps): void {
    const device = deps.getDevice();
    if (!device || !deps.compositorPipeline || !deps.outputPipeline || !deps.sampler) return;
    if (!deps.renderTargetManager || !deps.layerCollector || !deps.compositor) return;

    const pingView = deps.renderTargetManager.getPingView();
    const pongView = deps.renderTargetManager.getPongView();
    if (!pingView || !pongView) return;

    deps.compositorPipeline.beginFrame();

    const commandEncoder = device.createCommandEncoder();

    let readView = pingView;
    let writeView = pongView;

    for (const passId of graph.executionOrder) {
      const pass = graph.passes.find(p => p.id === passId);
      if (!pass) continue;

      switch (pass.config.type) {
        case 'clear':
          this.executeClearPass(commandEncoder, readView);
          break;

        case 'composite':
          // The actual compositing is delegated to the existing Compositor
          // This is a simplified placeholder — full integration will use
          // LayerCollector + Compositor.composite() from existing pipeline
          [readView, writeView] = [writeView, readView];
          break;

        case 'effect':
          // Delegated to EffectsPipeline
          break;

        case 'nestedComp':
          this.executeNestedCompPass(pass, commandEncoder, deps);
          break;

        case 'output':
          this.executeOutputPass(pass, commandEncoder, readView, deps);
          break;
      }
    }

    try {
      device.queue.submit([commandEncoder.finish()]);
    } catch (e) {
      log.error('GPU submit failed in render graph executor', e);
    }
  }

  // === Private pass executors ===

  private executeClearPass(
    commandEncoder: GPUCommandEncoder,
    targetView: GPUTextureView
  ): void {
    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: targetView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear' as const,
        storeOp: 'store' as const,
      }],
    });
    clearPass.end();
  }

  private executeNestedCompPass(
    pass: RenderPassNode,
    _commandEncoder: GPUCommandEncoder,
    _deps: RenderDeps
  ): void {
    const config = pass.config as NestedCompPassConfig;
    // Delegate to existing NestedCompRenderer when available
    // Full integration happens when the render graph replaces the main render path
    log.debug('NestedComp pass', { compositionId: config.compositionId });
  }

  private executeOutputPass(
    pass: RenderPassNode,
    commandEncoder: GPUCommandEncoder,
    finalView: GPUTextureView,
    deps: RenderDeps
  ): void {
    const config = pass.config as OutputPassConfig;
    if (!deps.outputPipeline || !deps.sampler) return;

    const ctx = deps.targetCanvases.get(config.targetId)?.context;
    if (!ctx) return;

    const bindGroup = deps.outputPipeline.createOutputBindGroup(
      deps.sampler,
      finalView,
      config.showTransparencyGrid
    );
    deps.outputPipeline.renderToCanvas(commandEncoder, ctx, bindGroup);
  }
}
