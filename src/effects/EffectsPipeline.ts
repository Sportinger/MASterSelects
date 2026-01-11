// Effects Pipeline - GPU effect processing using the modular effect registry

import { EFFECT_REGISTRY, getEffect } from './index';
import type { EffectDefinition } from './types';
import commonShader from './_shared/common.wgsl?raw';

// Effect instance interface (runtime data attached to clips)
interface EffectInstance {
  id: string;
  type: string;
  name: string;
  enabled: boolean;
  params: Record<string, number | boolean | string>;
}

export class EffectsPipeline {
  private device: GPUDevice;
  private pipelines = new Map<string, GPURenderPipeline>();
  private bindGroupLayouts = new Map<string, GPUBindGroupLayout>();
  private shaderModules = new Map<string, GPUShaderModule>();
  private initialized = false;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Initialize pipelines for all registered effects
   */
  async createPipelines(): Promise<void> {
    if (this.initialized) return;

    for (const [id, effect] of EFFECT_REGISTRY) {
      await this.createEffectPipeline(id, effect);
    }

    this.initialized = true;
    console.log(`[EffectsPipeline] Created ${this.pipelines.size} effect pipelines`);
  }

  /**
   * Create GPU pipeline for a single effect
   */
  private async createEffectPipeline(id: string, effect: EffectDefinition): Promise<void> {
    try {
      // Combine common shader with effect shader
      const shaderCode = `${commonShader}\n${effect.shader}`;

      const shaderModule = this.device.createShaderModule({
        label: `effect-${id}`,
        code: shaderCode,
      });
      this.shaderModules.set(id, shaderModule);

      // Create bind group layout
      const entries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ];

      if (effect.uniformSize > 0) {
        entries.push({
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        });
      }

      const bindGroupLayout = this.device.createBindGroupLayout({
        label: `effect-${id}-layout`,
        entries,
      });
      this.bindGroupLayouts.set(id, bindGroupLayout);

      // Create render pipeline
      const pipeline = this.device.createRenderPipeline({
        label: `effect-${id}-pipeline`,
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout],
        }),
        vertex: {
          module: shaderModule,
          entryPoint: 'vertexMain',
        },
        fragment: {
          module: shaderModule,
          entryPoint: effect.entryPoint,
          targets: [{ format: 'rgba8unorm' }],
        },
        primitive: { topology: 'triangle-list' },
      });

      this.pipelines.set(id, pipeline);
    } catch (error) {
      console.error(`[EffectsPipeline] Failed to create pipeline for ${id}:`, error);
    }
  }

  /**
   * Get pipeline for an effect type
   */
  getEffectPipeline(effectType: string): GPURenderPipeline | undefined {
    return this.pipelines.get(effectType);
  }

  /**
   * Get bind group layout for an effect type
   */
  getEffectBindGroupLayout(effectType: string): GPUBindGroupLayout | undefined {
    return this.bindGroupLayouts.get(effectType);
  }

  /**
   * Create uniform data for an effect using its packUniforms function
   */
  createEffectUniformData(
    effect: EffectInstance,
    outputWidth: number,
    outputHeight: number
  ): Float32Array | null {
    const definition = getEffect(effect.type);
    if (!definition) return null;

    return definition.packUniforms(effect.params, outputWidth, outputHeight);
  }

  /**
   * Create bind group for an effect
   */
  createEffectBindGroup(
    effectType: string,
    sampler: GPUSampler,
    inputView: GPUTextureView,
    uniformBuffer?: GPUBuffer
  ): GPUBindGroup | null {
    const layout = this.bindGroupLayouts.get(effectType);
    if (!layout) return null;

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: sampler },
      { binding: 1, resource: inputView },
    ];

    if (uniformBuffer) {
      entries.push({ binding: 2, resource: { buffer: uniformBuffer } });
    }

    return this.device.createBindGroup({
      layout,
      entries,
    });
  }

  /**
   * Apply effects to a texture using ping-pong rendering
   */
  applyEffects(
    commandEncoder: GPUCommandEncoder,
    effects: EffectInstance[],
    sampler: GPUSampler,
    inputView: GPUTextureView,
    outputView: GPUTextureView,
    _pingView: GPUTextureView,
    _pongView: GPUTextureView,
    outputWidth: number,
    outputHeight: number
  ): { finalView: GPUTextureView; swapped: boolean } {
    const enabledEffects = effects.filter(e => e.enabled);
    if (enabledEffects.length === 0) {
      return { finalView: inputView, swapped: false };
    }

    let effectInput = inputView;
    let effectOutput = outputView;
    let swapped = false;

    for (const effect of enabledEffects) {
      const pipeline = this.pipelines.get(effect.type);
      const bindGroupLayout = this.bindGroupLayouts.get(effect.type);

      if (!pipeline || !bindGroupLayout) {
        console.warn(`[EffectsPipeline] No pipeline for effect type: ${effect.type}`);
        continue;
      }

      // Create uniform buffer for effect parameters
      const effectParams = this.createEffectUniformData(effect, outputWidth, outputHeight);
      let effectUniformBuffer: GPUBuffer | null = null;

      if (effectParams) {
        effectUniformBuffer = this.device.createBuffer({
          size: effectParams.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(effectUniformBuffer, 0, effectParams.buffer);
      }

      // Create bind group
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: sampler },
        { binding: 1, resource: effectInput },
      ];

      if (effectUniformBuffer) {
        entries.push({ binding: 2, resource: { buffer: effectUniformBuffer } });
      }

      const effectBindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries,
      });

      // Render effect pass
      const effectPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: effectOutput,
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      effectPass.setPipeline(pipeline);
      effectPass.setBindGroup(0, effectBindGroup);
      effectPass.draw(6);
      effectPass.end();

      // Swap buffers for next effect in chain
      const tempView = effectInput;
      effectInput = effectOutput;
      effectOutput = tempView;
      swapped = !swapped;
    }

    // effectInput now contains the final result
    return { finalView: effectInput, swapped };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.pipelines.clear();
    this.bindGroupLayouts.clear();
    this.shaderModules.clear();
    this.initialized = false;
  }

  /**
   * Check if pipeline is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get number of registered effect pipelines
   */
  getPipelineCount(): number {
    return this.pipelines.size;
  }
}
