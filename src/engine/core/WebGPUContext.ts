// WebGPU device, adapter, and queue initialization

import { Logger } from '../../services/logger';

const log = Logger.create('WebGPUContext');

export type DeviceLostCallback = (reason: string) => void;
export type DeviceRestoredCallback = () => void;
export type GPUPowerPreference = 'high-performance' | 'low-power';

export class WebGPUContext {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initPromise: Promise<boolean> | null = null;
  private isInitialized = false;
  private currentPowerPreference: GPUPowerPreference = 'high-performance';

  // Callbacks for device loss/restore events
  private deviceLostCallbacks: Set<DeviceLostCallback> = new Set();
  private deviceRestoredCallbacks: Set<DeviceRestoredCallback> = new Set();

  // Track if we're recovering from a device loss
  private isRecovering = false;

  async initialize(powerPreference?: GPUPowerPreference): Promise<boolean> {
    // Store the preference if provided
    if (powerPreference) {
      this.currentPowerPreference = powerPreference;
    }
    // Prevent multiple initializations with promise-based lock
    if (this.isInitialized && this.device) {
      log.debug('Already initialized, skipping');
      return true;
    }

    // If initialization is in progress, wait for it
    if (this.initPromise) {
      log.debug('Initialization in progress, waiting...');
      return this.initPromise;
    }

    if (!navigator.gpu) {
      log.error('WebGPU not supported');
      return false;
    }

    // Create the initialization promise
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<boolean> {
    try {
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: this.currentPowerPreference,
      });
      log.info(`Requested adapter with powerPreference: ${this.currentPowerPreference}`);

      if (!this.adapter) {
        log.error('Failed to get GPU adapter');
        return false;
      }

      this.device = await this.adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxTextureDimension2D: 4096,
        },
      });

      this.device.lost.then((info) => {
        log.error('Device lost', info.message);
        this.isInitialized = false;

        // Notify listeners about device loss BEFORE attempting recovery
        for (const callback of this.deviceLostCallbacks) {
          try {
            callback(info.message);
          } catch (e) {
            log.error('Error in device lost callback', e);
          }
        }

        // Attempt auto-recovery after a short delay
        if (info.reason !== 'destroyed') {
          log.info('Attempting device recovery...');
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
                  log.error('Error in device restored callback', e);
                }
              }
            }
          }, 100);
        }
      });

      this.isInitialized = true;

      // Log detailed GPU adapter info to help debug iGPU vs dGPU selection
      const adapterInfo = (this.adapter as any).info || (this.device as any).adapterInfo;
      if (adapterInfo) {
        const isIntegrated = adapterInfo.description?.toLowerCase().includes('intel') ||
                            adapterInfo.description?.toLowerCase().includes('integrated') ||
                            adapterInfo.vendor?.toLowerCase().includes('intel');
        const gpuType = isIntegrated ? 'INTEGRATED' : 'DISCRETE';
        log.info(`${gpuType} GPU detected`);
        log.info('GPU Info', {
          vendor: adapterInfo.vendor || 'unknown',
          architecture: adapterInfo.architecture || 'unknown',
          device: adapterInfo.device || 'unknown',
          description: adapterInfo.description || 'unknown',
          powerPreference: this.currentPowerPreference,
        });
        if (isIntegrated && this.currentPowerPreference === 'high-performance') {
          log.warn('high-performance was requested but integrated GPU was selected! To fix: Open Windows Graphics Settings > Add Chrome/Edge > Options > High Performance');
        }
      }

      log.info('Context initialized successfully');
      return true;
    } catch (error) {
      log.error('Failed to initialize WebGPU', error);
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
      log.debug(`Canvas configured with preferred format: ${preferredFormat}`);
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

  /**
   * Get the current power preference
   */
  getPowerPreference(): GPUPowerPreference {
    return this.currentPowerPreference;
  }

  /**
   * Reinitialize with a new power preference
   * This destroys the current device and creates a new one
   */
  async reinitializeWithPreference(preference: GPUPowerPreference): Promise<boolean> {
    log.info(`Reinitializing with powerPreference: ${preference}`);

    // Skip if preference hasn't changed
    if (preference === this.currentPowerPreference && this.isInitialized) {
      log.debug('Power preference unchanged, skipping reinit');
      return true;
    }

    // Destroy current device
    this.device?.destroy();
    this.device = null;
    this.adapter = null;
    this.isInitialized = false;
    this.initPromise = null;

    // Store new preference
    this.currentPowerPreference = preference;

    // Reinitialize
    return this.initialize(preference);
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
