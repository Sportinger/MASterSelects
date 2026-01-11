// GPU-accelerated proxy frame resize pipeline
// Renders VideoFrames to a texture atlas for batch readback

import proxyResizeShader from '../../shaders/proxy-resize.wgsl?raw';

// Configuration
const ATLAS_GRID_SIZE = 4; // 4x4 = 16 frames per atlas
const BATCH_SIZE = ATLAS_GRID_SIZE * ATLAS_GRID_SIZE; // 16 frames

export interface ProxyResizeConfig {
  maxWidth: number;
  maxHeight: number;
}

export interface BatchReadbackResult {
  pixels: Uint8Array;
  frameWidth: number;
  frameHeight: number;
  frameCount: number;
}

export class ProxyResizePipeline {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private sampler: GPUSampler | null = null;

  // Atlas texture for batch rendering
  private atlasTexture: GPUTexture | null = null;
  private atlasView: GPUTextureView | null = null;

  // Frame dimensions (calculated based on source aspect ratio)
  private frameWidth = 0;
  private frameHeight = 0;
  private atlasWidth = 0;
  private atlasHeight = 0;

  // Uniform buffer for tile params
  private uniformBuffer: GPUBuffer | null = null;

  // Staging buffer for readback
  private stagingBuffer: GPUBuffer | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.createPipeline();
  }

  private createPipeline(): void {
    // Create shader module
    const shaderModule = this.device.createShaderModule({
      label: 'Proxy Resize Shader',
      code: proxyResizeShader,
    });

    // Create sampler with linear filtering for quality resize
    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    // Create bind group layout
    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'Proxy Resize Bind Group Layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, externalTexture: {} },
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    // Create uniform buffer (TileParams struct = 8 floats = 32 bytes)
    this.uniformBuffer = this.device.createBuffer({
      label: 'Proxy Resize Uniform Buffer',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create pipeline layout
    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Proxy Resize Pipeline Layout',
      bindGroupLayouts: [this.bindGroupLayout],
    });

    // Create render pipeline
    this.pipeline = this.device.createRenderPipeline({
      label: 'Proxy Resize Pipeline',
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vertexMain',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: 'rgba8unorm' }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });

    console.log('[ProxyResizePipeline] Pipeline created');
  }

  /**
   * Initialize atlas for a given frame size
   * Call this before processing frames to set up the correct dimensions
   */
  initializeAtlas(sourceWidth: number, sourceHeight: number, maxWidth: number): void {
    // Calculate output frame dimensions (maintain aspect ratio)
    const aspectRatio = sourceHeight / sourceWidth;
    this.frameWidth = Math.min(sourceWidth, maxWidth);
    this.frameHeight = Math.round(this.frameWidth * aspectRatio);

    // Ensure even dimensions for video encoding compatibility
    this.frameWidth = Math.floor(this.frameWidth / 2) * 2;
    this.frameHeight = Math.floor(this.frameHeight / 2) * 2;

    // Calculate atlas dimensions
    this.atlasWidth = this.frameWidth * ATLAS_GRID_SIZE;
    this.atlasHeight = this.frameHeight * ATLAS_GRID_SIZE;

    // Destroy existing atlas if present
    this.atlasTexture?.destroy();
    this.stagingBuffer?.destroy();

    // Create atlas texture
    this.atlasTexture = this.device.createTexture({
      label: 'Proxy Atlas Texture',
      size: [this.atlasWidth, this.atlasHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    this.atlasView = this.atlasTexture.createView();

    // Create staging buffer for readback
    // WebGPU requires bytesPerRow to be aligned to 256 bytes
    const bytesPerRow = Math.ceil((this.atlasWidth * 4) / 256) * 256;
    const bufferSize = bytesPerRow * this.atlasHeight;

    this.stagingBuffer = this.device.createBuffer({
      label: 'Proxy Atlas Staging Buffer',
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    console.log(`[ProxyResizePipeline] Atlas initialized: ${this.atlasWidth}x${this.atlasHeight} (${BATCH_SIZE} frames of ${this.frameWidth}x${this.frameHeight})`);
  }

  /**
   * Render a single video frame to a tile in the atlas
   */
  renderFrameToAtlas(frame: VideoFrame, tileIndex: number, commandEncoder: GPUCommandEncoder): void {
    if (!this.pipeline || !this.atlasView || !this.sampler || !this.uniformBuffer || !this.bindGroupLayout) {
      console.warn('[ProxyResizePipeline] Pipeline not initialized');
      return;
    }

    const tileX = tileIndex % ATLAS_GRID_SIZE;
    const tileY = Math.floor(tileIndex / ATLAS_GRID_SIZE);

    // Import video frame as external texture
    let externalTexture: GPUExternalTexture;
    try {
      externalTexture = this.device.importExternalTexture({ source: frame });
    } catch (e) {
      console.warn('[ProxyResizePipeline] Failed to import video frame:', e);
      return;
    }

    // Update uniform buffer with tile params
    const uniformData = new Float32Array([
      tileX,                    // tileX (u32, but stored as f32)
      tileY,                    // tileY
      this.frameWidth,          // tileWidth
      this.frameHeight,         // tileHeight
      this.atlasWidth,          // atlasWidth
      this.atlasHeight,         // atlasHeight
      frame.displayWidth,       // srcWidth
      frame.displayHeight,      // srcHeight
    ]);

    // Reinterpret first two floats as u32
    const uniformView = new DataView(uniformData.buffer);
    uniformView.setUint32(0, tileX, true);
    uniformView.setUint32(4, tileY, true);

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Create bind group for this frame
    const bindGroup = this.device.createBindGroup({
      label: `Proxy Resize Bind Group (tile ${tileIndex})`,
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: externalTexture },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    });

    // Render to atlas
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.atlasView,
        loadOp: tileIndex === 0 ? 'clear' : 'load', // Clear on first tile only
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6);
    renderPass.end();
  }

  /**
   * Read back the entire atlas as raw pixel data
   * Returns an array of Uint8Array, one per frame
   */
  async readBackAtlas(frameCount: number): Promise<Uint8Array[]> {
    if (!this.atlasTexture || !this.stagingBuffer) {
      throw new Error('Atlas not initialized');
    }

    // Copy atlas to staging buffer
    const bytesPerRow = Math.ceil((this.atlasWidth * 4) / 256) * 256;

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyTextureToBuffer(
      { texture: this.atlasTexture },
      { buffer: this.stagingBuffer, bytesPerRow, rowsPerImage: this.atlasHeight },
      [this.atlasWidth, this.atlasHeight]
    );
    this.device.queue.submit([commandEncoder.finish()]);

    // Map and read
    await this.stagingBuffer.mapAsync(GPUMapMode.READ);
    const mappedData = new Uint8Array(this.stagingBuffer.getMappedRange());

    // Extract individual frames from atlas
    const frames: Uint8Array[] = [];
    const frameSize = this.frameWidth * this.frameHeight * 4;

    for (let i = 0; i < frameCount; i++) {
      const tileX = i % ATLAS_GRID_SIZE;
      const tileY = Math.floor(i / ATLAS_GRID_SIZE);

      // Calculate start position in atlas
      const startX = tileX * this.frameWidth;
      const startY = tileY * this.frameHeight;

      // Extract frame pixels row by row (handling bytesPerRow alignment)
      const framePixels = new Uint8Array(frameSize);
      const frameRowBytes = this.frameWidth * 4;

      for (let y = 0; y < this.frameHeight; y++) {
        const srcOffset = (startY + y) * bytesPerRow + startX * 4;
        const dstOffset = y * frameRowBytes;
        framePixels.set(mappedData.subarray(srcOffset, srcOffset + frameRowBytes), dstOffset);
      }

      frames.push(framePixels);
    }

    this.stagingBuffer.unmap();

    return frames;
  }

  /**
   * Process a batch of frames: render to atlas and read back
   * This is the main entry point for batch processing
   */
  async processBatch(frames: VideoFrame[]): Promise<Uint8Array[]> {
    if (frames.length === 0) return [];
    if (frames.length > BATCH_SIZE) {
      throw new Error(`Batch size ${frames.length} exceeds maximum ${BATCH_SIZE}`);
    }

    const commandEncoder = this.device.createCommandEncoder();

    // Render all frames to atlas
    for (let i = 0; i < frames.length; i++) {
      this.renderFrameToAtlas(frames[i], i, commandEncoder);
    }

    this.device.queue.submit([commandEncoder.finish()]);

    // Read back all frames
    return this.readBackAtlas(frames.length);
  }

  /**
   * Get frame dimensions (after resize)
   */
  getFrameDimensions(): { width: number; height: number } {
    return { width: this.frameWidth, height: this.frameHeight };
  }

  /**
   * Get batch size (maximum frames per batch)
   */
  static getBatchSize(): number {
    return BATCH_SIZE;
  }

  destroy(): void {
    this.atlasTexture?.destroy();
    this.stagingBuffer?.destroy();
    this.uniformBuffer?.destroy();
    console.log('[ProxyResizePipeline] Destroyed');
  }
}
