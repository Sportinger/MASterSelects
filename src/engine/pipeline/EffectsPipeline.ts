// Effect processing pipelines

import effectsShader from '../../shaders/effects.wgsl?raw';
import { EFFECT_CONFIGS } from '../core/types';
import type { Effect } from '../core/types';

export class EffectsPipeline {
  private device: GPUDevice;

  // Effect pipelines and bind group layouts
  private effectPipelines: Map<string, GPURenderPipeline> = new Map();
  private effectBindGroupLayouts: Map<string, GPUBindGroupLayout> = new Map();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async createPipelines(): Promise<void> {
    const effectModule = this.device.createShaderModule({ code: effectsShader });

    for (const [effectType, config] of Object.entries(EFFECT_CONFIGS)) {
      // Create bind group layout
      const entries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ];

      if (config.needsUniform) {
        entries.push({ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } });
      }

      const bindGroupLayout = this.device.createBindGroupLayout({ entries });
      this.effectBindGroupLayouts.set(effectType, bindGroupLayout);

      // Create pipeline
      const pipeline = this.device.createRenderPipeline({
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout],
        }),
        vertex: { module: effectModule, entryPoint: 'vertexMain' },
        fragment: {
          module: effectModule,
          entryPoint: config.entryPoint,
          targets: [{ format: 'rgba8unorm' }],
        },
        primitive: { topology: 'triangle-list' },
      });

      this.effectPipelines.set(effectType, pipeline);
    }
  }

  getEffectPipeline(effectType: string): GPURenderPipeline | undefined {
    return this.effectPipelines.get(effectType);
  }

  getEffectBindGroupLayout(effectType: string): GPUBindGroupLayout | undefined {
    return this.effectBindGroupLayouts.get(effectType);
  }

  // Create uniform data for an effect
  createEffectUniformData(effect: Effect, outputWidth: number, outputHeight: number): Float32Array | null {
    const params = effect.params;

    switch (effect.type) {
      case 'hue-shift':
        return new Float32Array([
          params.shift as number || 0,
          0, 0, 0, // padding
        ]);

      case 'brightness':
      case 'contrast':
      case 'saturation': {
        // ColorAdjust shader uses: brightness, contrast, saturation
        const brightness = effect.type === 'brightness' ? (params.amount as number || 0) : 0;
        const contrast = effect.type === 'contrast' ? (params.amount as number || 1) : 1;
        const saturation = effect.type === 'saturation' ? (params.amount as number || 1) : 1;
        return new Float32Array([
          brightness,
          contrast,
          saturation,
          0, // padding
        ]);
      }

      case 'pixelate':
        return new Float32Array([
          params.size as number || 8,
          outputWidth,
          outputHeight,
          0, // padding
        ]);

      case 'kaleidoscope':
        return new Float32Array([
          params.segments as number || 6,
          params.rotation as number || 0,
          0, 0, // padding
        ]);

      case 'mirror':
        return new Float32Array([
          params.horizontal ? 1 : 0,
          params.vertical ? 1 : 0,
          0, 0, // padding
        ]);

      case 'rgb-split':
        return new Float32Array([
          params.amount as number || 0.01,
          params.angle as number || 0,
          0, 0, // padding
        ]);

      case 'levels':
        return new Float32Array([
          params.inputBlack as number || 0,
          params.inputWhite as number || 1,
          params.gamma as number || 1,
          params.outputBlack as number || 0,
          params.outputWhite as number || 1,
          0, 0, 0, // padding
        ]);

      case 'invert':
        return null; // No uniforms needed

      default:
        return null;
    }
  }

  // Create bind group for an effect
  createEffectBindGroup(
    effectType: string,
    sampler: GPUSampler,
    inputView: GPUTextureView,
    uniformBuffer?: GPUBuffer
  ): GPUBindGroup | null {
    const layout = this.effectBindGroupLayouts.get(effectType);
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

  // Apply effects to a render pass
  applyEffects(
    commandEncoder: GPUCommandEncoder,
    effects: Effect[],
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
      const pipeline = this.effectPipelines.get(effect.type);
      const bindGroupLayout = this.effectBindGroupLayouts.get(effect.type);

      if (!pipeline || !bindGroupLayout) continue;

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

      // Render effect
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

      // Swap for next effect
      const tempView = effectInput;
      effectInput = effectOutput;
      effectOutput = tempView;
      swapped = !swapped;
    }

    // effectInput contains the final result
    return { finalView: effectInput, swapped };
  }

  destroy(): void {
    this.effectPipelines.clear();
    this.effectBindGroupLayouts.clear();
  }
}
