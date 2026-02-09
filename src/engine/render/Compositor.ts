// Ping-pong compositing with effects

import type { LayerRenderData, CompositeResult } from '../core/types';
import type { CompositorPipeline, InlineEffectParams } from '../pipeline/CompositorPipeline';
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

      // Classify effects: inlineable effects (brightness, contrast, saturation, invert)
      // are handled directly in the composite shader via uniforms - no extra render passes.
      // Complex effects (blur, pixelate, etc.) still need separate render passes.
      const inlineEffects: InlineEffectParams = { brightness: 0, contrast: 1, saturation: 1, invert: false };
      let complexEffects: typeof layer.effects | undefined;

      if (layer.effects && layer.effects.length > 0) {
        const complex: typeof layer.effects = [];
        for (const effect of layer.effects) {
          if (!effect.enabled || effect.type.startsWith('audio-')) continue;
          switch (effect.type) {
            case 'brightness':
              inlineEffects.brightness = (effect.params.amount as number) ?? 0;
              break;
            case 'contrast':
              inlineEffects.contrast = (effect.params.amount as number) ?? 1;
              break;
            case 'saturation':
              inlineEffects.saturation = (effect.params.amount as number) ?? 1;
              break;
            case 'invert':
              inlineEffects.invert = true;
              break;
            default:
              complex.push(effect);
              break;
          }
        }
        if (complex.length > 0) {
          complexEffects = complex;
        }
      }

      // Update uniforms (includes inline effect params)
      this.compositorPipeline.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer, inlineEffects);

      // Track which ping-pong buffer we're reading from for cache key
      const isPingBase = readView === state.pingView;

      // Determine the source texture/view to use for compositing
      let sourceTextureView = data.textureView;
      let sourceExternalTexture = data.externalTexture;
      let useExternalTexture = data.isVideo && !!data.externalTexture;

      // Apply complex effects to the SOURCE layer BEFORE compositing (skip if only inline effects)
      // Inline effects (brightness, contrast, saturation, invert) are handled in the composite shader
      if (complexEffects && complexEffects.length > 0 && state.effectTempView && state.effectTempView2) {
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

              // Now apply complex effects to the copied texture
              const effectResult = this.effectsPipeline.applyEffects(
                commandEncoder,
                complexEffects,
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

          // Apply complex effects
          const effectResult = this.effectsPipeline.applyEffects(
            commandEncoder,
            complexEffects,
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
        // When complex effects are applied, the final texture view alternates between
        // effectTempView/effectTempView2 depending on effect count parity.
        // Don't use the bind group cache in this case - it would return a stale bind group
        // pointing to the wrong texture view.
        const cacheLayerId = complexEffects ? undefined : layer.id;
        if (complexEffects) {
          // Invalidate any stale cached bind group for this layer
          this.compositorPipeline.invalidateBindGroupCache(layer.id);
        }
        bindGroup = this.compositorPipeline.createCompositeBindGroup(
          state.sampler,
          readView,
          sourceTextureView,
          uniformBuffer,
          maskTextureView,
          cacheLayerId,
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
