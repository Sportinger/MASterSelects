// Output to canvas pipeline

import outputShader from '../../shaders/output.wgsl?raw';

export class OutputPipeline {
  private device: GPUDevice;

  // Output pipeline
  private outputPipeline: GPURenderPipeline | null = null;

  // Bind group layout
  private outputBindGroupLayout: GPUBindGroupLayout | null = null;

  // Cached output bind groups
  private cachedOutputBindGroupPing: GPUBindGroup | null = null;
  private cachedOutputBindGroupPong: GPUBindGroup | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async createPipeline(): Promise<void> {
    // Output bind group layout
    this.outputBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
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
        targets: [{ format: 'bgra8unorm' }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  getOutputPipeline(): GPURenderPipeline | null {
    return this.outputPipeline;
  }

  getOutputBindGroupLayout(): GPUBindGroupLayout | null {
    return this.outputBindGroupLayout;
  }

  // Get or create output bind group for ping texture
  getOutputBindGroup(
    sampler: GPUSampler,
    textureView: GPUTextureView,
    isPing: boolean
  ): GPUBindGroup {
    // Check cache
    const cached = isPing ? this.cachedOutputBindGroupPing : this.cachedOutputBindGroupPong;
    if (cached) return cached;

    // Create new bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.outputBindGroupLayout!,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: textureView },
      ],
    });

    // Cache it
    if (isPing) {
      this.cachedOutputBindGroupPing = bindGroup;
    } else {
      this.cachedOutputBindGroupPong = bindGroup;
    }

    return bindGroup;
  }

  // Create a fresh output bind group (no caching)
  createOutputBindGroup(sampler: GPUSampler, textureView: GPUTextureView): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.outputBindGroupLayout!,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: textureView },
      ],
    });
  }

  // Invalidate cached bind groups (when textures are recreated)
  invalidateCache(): void {
    this.cachedOutputBindGroupPing = null;
    this.cachedOutputBindGroupPong = null;
  }

  // Render to a canvas context
  renderToCanvas(
    commandEncoder: GPUCommandEncoder,
    context: GPUCanvasContext,
    bindGroup: GPUBindGroup
  ): void {
    if (!this.outputPipeline) return;

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
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
  }
}
