// WebGPU device, adapter, and queue initialization

export class WebGPUContext {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initPromise: Promise<boolean> | null = null;
  private isInitialized = false;

  async initialize(): Promise<boolean> {
    // Prevent multiple initializations with promise-based lock
    if (this.isInitialized && this.device) {
      console.log('[WebGPU] Already initialized, skipping');
      return true;
    }

    // If initialization is in progress, wait for it
    if (this.initPromise) {
      console.log('[WebGPU] Initialization in progress, waiting...');
      return this.initPromise;
    }

    if (!navigator.gpu) {
      console.error('WebGPU not supported');
      return false;
    }

    // Create the initialization promise
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<boolean> {
    try {
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!this.adapter) {
        console.error('Failed to get GPU adapter');
        return false;
      }

      this.device = await this.adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxTextureDimension2D: 4096,
        },
      });

      this.device.lost.then((info) => {
        console.error('WebGPU device lost:', info.message);
        this.isInitialized = false;
      });

      this.isInitialized = true;
      console.log('[WebGPU] Context initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize WebGPU:', error);
      this.initPromise = null;
      return false;
    }
  }

  getDevice(): GPUDevice | null {
    return this.device;
  }

  getAdapter(): GPUAdapter | null {
    return this.adapter;
  }

  get initialized(): boolean {
    return this.isInitialized;
  }

  // Create and configure a canvas context
  configureCanvas(canvas: HTMLCanvasElement): GPUCanvasContext | null {
    if (!this.device) return null;

    const context = canvas.getContext('webgpu');
    if (context) {
      context.configure({
        device: this.device,
        format: 'bgra8unorm',
        alphaMode: 'premultiplied',
      });
    }
    return context;
  }

  // Create a sampler with standard settings
  createSampler(): GPUSampler | null {
    if (!this.device) return null;
    return this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  // Create a uniform buffer
  createUniformBuffer(size: number): GPUBuffer | null {
    if (!this.device) return null;
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // Create a single-pixel texture with a solid color
  createSolidColorTexture(r: number, g: number, b: number, a: number): GPUTexture | null {
    if (!this.device) return null;

    const texture = this.device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture },
      new Uint8Array([r, g, b, a]),
      { bytesPerRow: 4 },
      [1, 1]
    );

    return texture;
  }

  destroy(): void {
    this.device?.destroy();
    this.device = null;
    this.adapter = null;
    this.isInitialized = false;
    this.initPromise = null;
  }
}
