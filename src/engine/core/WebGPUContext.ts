// WebGPU device, adapter, and queue initialization

export type DeviceLostCallback = (reason: string) => void;
export type DeviceRestoredCallback = () => void;

export class WebGPUContext {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initPromise: Promise<boolean> | null = null;
  private isInitialized = false;

  // Callbacks for device loss/restore events
  private deviceLostCallbacks: Set<DeviceLostCallback> = new Set();
  private deviceRestoredCallbacks: Set<DeviceRestoredCallback> = new Set();

  // Track if we're recovering from a device loss
  private isRecovering = false;

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

        // Notify listeners about device loss BEFORE attempting recovery
        for (const callback of this.deviceLostCallbacks) {
          try {
            callback(info.message);
          } catch (e) {
            console.error('[WebGPU] Error in device lost callback:', e);
          }
        }

        // Attempt auto-recovery after a short delay
        if (info.reason !== 'destroyed') {
          console.log('[WebGPU] Attempting device recovery...');
          this.initPromise = null;
          this.isRecovering = true;
          setTimeout(async () => {
            const success = await this.initialize();
            if (success) {
              this.isRecovering = false;
              // Notify listeners that device was restored
              for (const callback of this.deviceRestoredCallbacks) {
                try {
                  callback();
                } catch (e) {
                  console.error('[WebGPU] Error in device restored callback:', e);
                }
              }
            }
          }, 100);
        }
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

  /**
   * Get GPU info (vendor, device name, etc.)
   */
  getGPUInfo(): { vendor: string; device: string; description: string } | null {
    if (!this.adapter) return null;

    // adapter.info is available in Chrome 114+
    const info = (this.adapter as any).info;
    if (info) {
      return {
        vendor: info.vendor || 'Unknown',
        device: info.device || '',
        description: info.description || '',
      };
    }
    return null;
  }

  // Get the preferred canvas format for this GPU
  getPreferredCanvasFormat(): GPUTextureFormat {
    return navigator.gpu.getPreferredCanvasFormat();
  }

  // Create and configure a canvas context
  configureCanvas(canvas: HTMLCanvasElement): GPUCanvasContext | null {
    if (!this.device) return null;

    const context = canvas.getContext('webgpu');
    if (context) {
      // Use the GPU's preferred format to avoid extra copies
      const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({
        device: this.device,
        format: preferredFormat,
        alphaMode: 'premultiplied',
      });
      console.log(`[WebGPU] Canvas configured with preferred format: ${preferredFormat}`);
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

  /**
   * Register a callback to be notified when the device is lost
   */
  onDeviceLost(callback: DeviceLostCallback): void {
    this.deviceLostCallbacks.add(callback);
  }

  /**
   * Remove a device lost callback
   */
  offDeviceLost(callback: DeviceLostCallback): void {
    this.deviceLostCallbacks.delete(callback);
  }

  /**
   * Register a callback to be notified when the device is restored after loss
   */
  onDeviceRestored(callback: DeviceRestoredCallback): void {
    this.deviceRestoredCallbacks.add(callback);
  }

  /**
   * Remove a device restored callback
   */
  offDeviceRestored(callback: DeviceRestoredCallback): void {
    this.deviceRestoredCallbacks.delete(callback);
  }

  /**
   * Check if the context is currently recovering from a device loss
   */
  get recovering(): boolean {
    return this.isRecovering;
  }

  destroy(): void {
    this.device?.destroy();
    this.device = null;
    this.adapter = null;
    this.isInitialized = false;
    this.initPromise = null;
    this.deviceLostCallbacks.clear();
    this.deviceRestoredCallbacks.clear();
  }
}
