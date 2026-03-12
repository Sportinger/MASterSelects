// Output to canvas pipeline with optional transparency grid
// Uses dual uniform buffers (grid-on / grid-off) so different render targets
// within the same command encoder can have different transparency states.

import outputShader from '../../shaders/output.wgsl?raw';

export class OutputPipeline {
  private device: GPUDevice;

  // Output pipeline
  private outputPipeline: GPURenderPipeline | null = null;

  // Bind group layout
  private outputBindGroupLayout: GPUBindGroupLayout | null = null;

  // Uniform buffers per output mode
  private uniformBufferGridOn: GPUBuffer | null = null;
  private uniformBufferGridOff: GPUBuffer | null = null;
  private uniformBufferStackedAlpha: GPUBuffer | null = null;

  // Separate caches per mode (buffer is baked into bind group)
  private bindGroupCacheGridOn = new Map<GPUTextureView, GPUBindGroup>();
  private bindGroupCacheGridOff = new Map<GPUTextureView, GPUBindGroup>();
  private bindGroupCacheStackedAlpha = new Map<GPUTextureView, GPUBindGroup>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async createPipeline(): Promise<void> {
    // Create uniform buffers (16 bytes each: u32 + f32 + f32 + padding)
    this.uniformBufferGridOn = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniformBufferGridOff = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniformBufferStackedAlpha = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Output bind group layout with uniform buffer
    this.outputBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    // Output pipeline
    const outputModule = this.device.createShaderModule({ code: outputShader });

    this.outputPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.outputBindGroupLayout],
      }),
      vertex: { module: outputModule, entryPoint: 'vertexMain' },
      fragment: {
        module: outputModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  // Write resolution into all uniform buffers (mode: 0=off, 1=grid, 2=stackedAlpha)
  updateResolution(outputWidth: number, outputHeight: number): void {
    if (!this.uniformBufferGridOn || !this.uniformBufferGridOff) return;

    // Grid ON buffer (mode=1)
    const dataOn = new ArrayBuffer(16);
    const viewOn = new DataView(dataOn);
    viewOn.setUint32(0, 1, true);
    viewOn.setFloat32(4, outputWidth, true);
    viewOn.setFloat32(8, outputHeight, true);
    viewOn.setFloat32(12, 0, true);
    this.device.queue.writeBuffer(this.uniformBufferGridOn, 0, dataOn);

    // Grid OFF buffer (mode=0)
    const dataOff = new ArrayBuffer(16);
    const viewOff = new DataView(dataOff);
    viewOff.setUint32(0, 0, true);
    viewOff.setFloat32(4, outputWidth, true);
    viewOff.setFloat32(8, outputHeight, true);
    viewOff.setFloat32(12, 0, true);
    this.device.queue.writeBuffer(this.uniformBufferGridOff, 0, dataOff);

    // Stacked Alpha buffer (mode=2)
    if (this.uniformBufferStackedAlpha) {
      const dataSA = new ArrayBuffer(16);
      const viewSA = new DataView(dataSA);
      viewSA.setUint32(0, 2, true);
      viewSA.setFloat32(4, outputWidth, true);
      viewSA.setFloat32(8, outputHeight, true);
      viewSA.setFloat32(12, 0, true);
      this.device.queue.writeBuffer(this.uniformBufferStackedAlpha, 0, dataSA);
    }
  }

  getOutputPipeline(): GPURenderPipeline | null {
    return this.outputPipeline;
  }

  getOutputBindGroupLayout(): GPUBindGroupLayout | null {
    return this.outputBindGroupLayout;
  }

  // Create output bind group for a texture view, selecting the appropriate uniform buffer
  // mode: 'normal' = no grid, 'grid' = transparency grid, 'stackedAlpha' = stacked alpha export
  createOutputBindGroup(sampler: GPUSampler, textureView: GPUTextureView, mode: 'normal' | 'grid' | 'stackedAlpha' = 'normal'): GPUBindGroup {
    const cache = mode === 'grid' ? this.bindGroupCacheGridOn
      : mode === 'stackedAlpha' ? this.bindGroupCacheStackedAlpha
      : this.bindGroupCacheGridOff;
    const uniformBuffer = mode === 'grid' ? this.uniformBufferGridOn!
      : mode === 'stackedAlpha' ? this.uniformBufferStackedAlpha!
      : this.uniformBufferGridOff!;

    // Check cache first
    let bindGroup = cache.get(textureView);
    if (bindGroup) return bindGroup;

    // Create new bind group
    bindGroup = this.device.createBindGroup({
      layout: this.outputBindGroupLayout!,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: textureView },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    // Cache it
    cache.set(textureView, bindGroup);
    return bindGroup;
  }

  // Legacy method for compatibility
  getOutputBindGroup(
    sampler: GPUSampler,
    textureView: GPUTextureView,
    _isPing: boolean
  ): GPUBindGroup {
    return this.createOutputBindGroup(sampler, textureView, 'normal');
  }

  // Invalidate cached bind groups (when textures are recreated)
  invalidateCache(): void {
    this.bindGroupCacheGridOn.clear();
    this.bindGroupCacheGridOff.clear();
    this.bindGroupCacheStackedAlpha.clear();
  }

  // Render to a canvas context
  renderToCanvas(
    commandEncoder: GPUCommandEncoder,
    context: GPUCanvasContext,
    bindGroup: GPUBindGroup
  ): void {
    if (!this.outputPipeline) return;

    // getCurrentTexture() can throw if canvas context is lost/unconfigured
    let canvasView: GPUTextureView;
    try {
      canvasView = context.getCurrentTexture().createView();
    } catch {
      return; // Canvas context lost - skip this canvas, render loop continues
    }

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasView,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    renderPass.setPipeline(this.outputPipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6);
    renderPass.end();
  }

  destroy(): void {
    this.invalidateCache();
    this.uniformBufferGridOn?.destroy();
    this.uniformBufferGridOn = null;
    this.uniformBufferGridOff?.destroy();
    this.uniformBufferGridOff = null;
    this.uniformBufferStackedAlpha?.destroy();
    this.uniformBufferStackedAlpha = null;
  }
}
