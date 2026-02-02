// Ping-pong compositing with effects

import type { LayerRenderData, CompositeResult } from '../core/types';
import type { CompositorPipeline } from '../pipeline/CompositorPipeline';
import type { EffectsPipeline } from '../../effects/EffectsPipeline';
import type { MaskTextureManager } from '../texture/MaskTextureManager';

export interface CompositorState {
  device: GPUDevice;
  sampler: GPUSampler;
  pingView: GPUTextureView;
  pongView: GPUTextureView;
  outputWidth: number;
  outputHeight: number;
  // Additional textures for effect pre-processing
  effectTempTexture?: GPUTexture;
  effectTempView?: GPUTextureView;
  effectTempTexture2?: GPUTexture;
  effectTempView2?: GPUTextureView;
}

export class Compositor {
  private compositorPipeline: CompositorPipeline;
  private effectsPipeline: EffectsPipeline;
  private maskTextureManager: MaskTextureManager;
  private lastRenderWasPing = false;

  constructor(
    compositorPipeline: CompositorPipeline,
    effectsPipeline: EffectsPipeline,
    maskTextureManager: MaskTextureManager
  ) {
    this.compositorPipeline = compositorPipeline;
    this.effectsPipeline = effectsPipeline;
    this.maskTextureManager = maskTextureManager;
  }

  composite(
    layerData: LayerRenderData[],
    commandEncoder: GPUCommandEncoder,
    state: CompositorState
  ): CompositeResult {
    let readView = state.pingView;
    let writeView = state.pongView;
    let usePing = true;

    // Clear first buffer to transparent
    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: readView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    clearPass.end();

    // Composite each layer
    for (let i = 0; i < layerData.length; i++) {
      const data = layerData[i];
      const layer = data.layer;

      // Get uniform buffer
      const uniformBuffer = this.compositorPipeline.getOrCreateUniformBuffer(layer.id);

      // Calculate aspect ratios
      const sourceAspect = data.sourceWidth / data.sourceHeight;
      const outputAspect = state.outputWidth / state.outputHeight;

      // Get mask texture (single lookup instead of two)
      const maskLookupId = layer.maskClipId || layer.id;
      const maskInfo = this.maskTextureManager.getMaskInfo(maskLookupId);
      const hasMask = maskInfo.hasMask;
      const maskTextureView = maskInfo.view;

      this.maskTextureManager.logMaskState(maskLookupId, hasMask);

      // Update uniforms
      this.compositorPipeline.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      // Track which ping-pong buffer we're reading from for cache key
      const isPingBase = readView === state.pingView;

      // Determine the source texture/view to use for compositing
      let sourceTextureView = data.textureView;
      let sourceExternalTexture = data.externalTexture;
      let useExternalTexture = data.isVideo && !!data.externalTexture;

      // IMPORTANT: Apply effects to the SOURCE layer BEFORE compositing
      // This ensures effects only affect this layer, not the accumulated background
      if (layer.effects && layer.effects.length > 0 && state.effectTempView && state.effectTempView2) {
        // First, we need to copy/render the source into a temp texture so we can apply effects to it
        // For video (external texture), render it to temp texture first
        if (useExternalTexture && sourceExternalTexture) {
          // Render external texture to effectTempView using a simple copy pass
          const copyPipeline = this.compositorPipeline.getExternalCopyPipeline?.();
          if (copyPipeline) {
            const copyBindGroup = this.compositorPipeline.createExternalCopyBindGroup?.(
              state.sampler,
              sourceExternalTexture,
              layer.id
            );
            if (copyBindGroup) {
              const copyPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                  view: state.effectTempView,
                  loadOp: 'clear',
                  storeOp: 'store',
                }],
              });
              copyPass.setPipeline(copyPipeline);
              copyPass.setBindGroup(0, copyBindGroup);
              copyPass.draw(6);
              copyPass.end();

              // Now apply effects to the copied texture
              const effectResult = this.effectsPipeline.applyEffects(
                commandEncoder,
                layer.effects,
                state.sampler,
                state.effectTempView,
                state.effectTempView2,
                state.effectTempView,
                state.effectTempView2,
                state.outputWidth,
                state.outputHeight
              );

              // Use the effected texture for compositing (as regular texture, not external)
              sourceTextureView = effectResult.finalView;
              useExternalTexture = false;
              sourceExternalTexture = null;
            }
          }
        } else if (sourceTextureView) {
          // For regular textures, apply effects directly
          // First copy to temp
          const copyPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
              view: state.effectTempView,
              loadOp: 'clear',
              storeOp: 'store',
            }],
          });
          const copyPipeline = this.compositorPipeline.getCopyPipeline?.();
          if (copyPipeline) {
            const copyBindGroup = this.compositorPipeline.createCopyBindGroup?.(
              state.sampler,
              sourceTextureView,
              layer.id
            );
            if (copyBindGroup) {
              copyPass.setPipeline(copyPipeline);
              copyPass.setBindGroup(0, copyBindGroup);
              copyPass.draw(6);
            }
          }
          copyPass.end();

          // Apply effects
          const effectResult = this.effectsPipeline.applyEffects(
            commandEncoder,
            layer.effects,
            state.sampler,
            state.effectTempView,
            state.effectTempView2,
            state.effectTempView,
            state.effectTempView2,
            state.outputWidth,
            state.outputHeight
          );

          sourceTextureView = effectResult.finalView;
        }
      }

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (useExternalTexture && sourceExternalTexture) {
        pipeline = this.compositorPipeline.getExternalCompositePipeline()!;
        bindGroup = this.compositorPipeline.createExternalCompositeBindGroup(
          state.sampler,
          readView,
          sourceExternalTexture,
          uniformBuffer,
          maskTextureView,
          layer.id,
          isPingBase
        );
      } else if (sourceTextureView) {
        pipeline = this.compositorPipeline.getCompositePipeline()!;
        bindGroup = this.compositorPipeline.createCompositeBindGroup(
          state.sampler,
          readView,
          sourceTextureView,
          uniformBuffer,
          maskTextureView,
          layer.id,
          isPingBase
        );
      } else {
        continue;
      }

      // Render pass - composite the (possibly effected) layer onto the accumulated result
      const compositePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: writeView,
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      compositePass.setPipeline(pipeline);
      compositePass.setBindGroup(0, bindGroup);
      compositePass.draw(6);
      compositePass.end();

      // Swap buffers
      const temp = readView;
      readView = writeView;
      writeView = temp;
      usePing = !usePing;
    }

    this.lastRenderWasPing = usePing;

    return {
      finalView: readView,
      usedPing: !usePing,
      layerCount: layerData.length,
    };
  }

  getLastRenderWasPing(): boolean {
    return this.lastRenderWasPing;
  }
}
