// Output to canvas pipeline with optional transparency grid

import outputShader from '../../shaders/output.wgsl?raw';

export class OutputPipeline {
  private device: GPUDevice;

  // Output pipeline
  private outputPipeline: GPURenderPipeline | null = null;

  // Bind group layout
  private outputBindGroupLayout: GPUBindGroupLayout | null = null;

  // Uniform buffer for output settings
  private uniformBuffer: GPUBuffer | null = null;

  // Cached output bind groups (keyed by texture view)
  private bindGroupCache = new Map<GPUTextureView, GPUBindGroup>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async createPipeline(): Promise<void> {
    // Create uniform buffer (16 bytes: u32 + f32 + f32 + padding)
    this.uniformBuffer = this.device.createBuffer({
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

  // Update uniform buffer with current settings
  updateUniforms(showTransparencyGrid: boolean, outputWidth: number, outputHeight: number): void {
    if (!this.uniformBuffer) return;

    const data = new ArrayBuffer(16);
    const view = new DataView(data);
    view.setUint32(0, showTransparencyGrid ? 1 : 0, true);  // showTransparencyGrid
    view.setFloat32(4, outputWidth, true);   // outputWidth
    view.setFloat32(8, outputHeight, true);  // outputHeight
    view.setFloat32(12, 0, true);            // padding

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  getOutputPipeline(): GPURenderPipeline | null {
    return this.outputPipeline;
  }

  getOutputBindGroupLayout(): GPUBindGroupLayout | null {
    return this.outputBindGroupLayout;
  }

  // Create output bind group for a texture view
  createOutputBindGroup(sampler: GPUSampler, textureView: GPUTextureView): GPUBindGroup {
    // Check cache first
    let bindGroup = this.bindGroupCache.get(textureView);
    if (bindGroup) return bindGroup;

    // Create new bind group
    bindGroup = this.device.createBindGroup({
      layout: this.outputBindGroupLayout!,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: textureView },
        { binding: 2, resource: { buffer: this.uniformBuffer! } },
      ],
    });

    // Cache it
    this.bindGroupCache.set(textureView, bindGroup);
    return bindGroup;
  }

  // Legacy method for compatibility - same as createOutputBindGroup now
  getOutputBindGroup(
    sampler: GPUSampler,
    textureView: GPUTextureView,
    _isPing: boolean
  ): GPUBindGroup {
    return this.createOutputBindGroup(sampler, textureView);
  }

  // Invalidate cached bind groups (when textures are recreated)
  invalidateCache(): void {
    this.bindGroupCache.clear();
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
    this.uniformBuffer?.destroy();
    this.uniformBuffer = null;
  }
}
