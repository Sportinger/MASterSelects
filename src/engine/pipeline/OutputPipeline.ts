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

  // Dual uniform buffers: one with grid enabled, one without
  private uniformBufferGridOn: GPUBuffer | null = null;
  private uniformBufferGridOff: GPUBuffer | null = null;

  // Separate caches per grid state (buffer is baked into bind group)
  private bindGroupCacheGridOn = new Map<GPUTextureView, GPUBindGroup>();
  private bindGroupCacheGridOff = new Map<GPUTextureView, GPUBindGroup>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async createPipeline(): Promise<void> {
    // Create dual uniform buffers (16 bytes each: u32 + f32 + f32 + padding)
    this.uniformBufferGridOn = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniformBufferGridOff = this.device.createBuffer({
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

  // Write resolution into both uniform buffers (grid=1 and grid=0)
  updateResolution(outputWidth: number, outputHeight: number): void {
    if (!this.uniformBufferGridOn || !this.uniformBufferGridOff) return;

    // Grid ON buffer
    const dataOn = new ArrayBuffer(16);
    const viewOn = new DataView(dataOn);
    viewOn.setUint32(0, 1, true);              // showTransparencyGrid = true
    viewOn.setFloat32(4, outputWidth, true);
    viewOn.setFloat32(8, outputHeight, true);
    viewOn.setFloat32(12, 0, true);            // padding
    this.device.queue.writeBuffer(this.uniformBufferGridOn, 0, dataOn);

    // Grid OFF buffer
    const dataOff = new ArrayBuffer(16);
    const viewOff = new DataView(dataOff);
    viewOff.setUint32(0, 0, true);             // showTransparencyGrid = false
    viewOff.setFloat32(4, outputWidth, true);
    viewOff.setFloat32(8, outputHeight, true);
    viewOff.setFloat32(12, 0, true);           // padding
    this.device.queue.writeBuffer(this.uniformBufferGridOff, 0, dataOff);
  }

  getOutputPipeline(): GPURenderPipeline | null {
    return this.outputPipeline;
  }

  getOutputBindGroupLayout(): GPUBindGroupLayout | null {
    return this.outputBindGroupLayout;
  }

  // Create output bind group for a texture view, selecting the appropriate uniform buffer
  createOutputBindGroup(sampler: GPUSampler, textureView: GPUTextureView, showGrid: boolean = false): GPUBindGroup {
    const cache = showGrid ? this.bindGroupCacheGridOn : this.bindGroupCacheGridOff;
    const uniformBuffer = showGrid ? this.uniformBufferGridOn! : this.uniformBufferGridOff!;

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
    return this.createOutputBindGroup(sampler, textureView, false);
  }

  // Invalidate cached bind groups (when textures are recreated)
  invalidateCache(): void {
    this.bindGroupCacheGridOn.clear();
    this.bindGroupCacheGridOff.clear();
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
  }
}
