// Manages ping-pong render targets for compositing

import { Logger } from '../../services/logger';
import type { RenderTargets } from './types';

const log = Logger.create('RenderTargetManager');

export class RenderTargetManager {
  private device: GPUDevice;
  private pingTexture: GPUTexture | null = null;
  private pongTexture: GPUTexture | null = null;
  private pingView: GPUTextureView | null = null;
  private pongView: GPUTextureView | null = null;
  private independentPingTexture: GPUTexture | null = null;
  private independentPongTexture: GPUTexture | null = null;
  private independentPingView: GPUTextureView | null = null;
  private independentPongView: GPUTextureView | null = null;
  private blackTexture: GPUTexture | null = null;

  private outputWidth = 640;
  private outputHeight = 360;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  createPingPongTextures(): void {
    // Destroy existing textures first to free memory
    this.pingTexture?.destroy();
    this.pongTexture?.destroy();
    this.independentPingTexture?.destroy();
    this.independentPongTexture?.destroy();

    // Reset all references
    this.pingTexture = null;
    this.pongTexture = null;
    this.independentPingTexture = null;
    this.independentPongTexture = null;
    this.pingView = null;
    this.pongView = null;
    this.independentPingView = null;
    this.independentPongView = null;

    log.info(`Creating ping-pong textures at ${this.outputWidth}x${this.outputHeight}`);

    try {
      // Main render loop ping-pong buffers
      log.debug('Creating ping texture...');
      this.pingTexture = this.device.createTexture({
        size: [this.outputWidth, this.outputHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });

      log.debug('Creating pong texture...');
      this.pongTexture = this.device.createTexture({
        size: [this.outputWidth, this.outputHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });

      // Independent preview ping-pong buffers
      log.debug('Creating independent ping texture...');
      this.independentPingTexture = this.device.createTexture({
        size: [this.outputWidth, this.outputHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });

      log.debug('Creating independent pong texture...');
      this.independentPongTexture = this.device.createTexture({
        size: [this.outputWidth, this.outputHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });

      // Cache views
      if (this.pingTexture && this.pongTexture) {
        this.pingView = this.pingTexture.createView();
        this.pongView = this.pongTexture.createView();
      }
      if (this.independentPingTexture && this.independentPongTexture) {
        this.independentPingView = this.independentPingTexture.createView();
        this.independentPongView = this.independentPongTexture.createView();
      }

      log.info('Ping-pong textures created successfully');
    } catch (e) {
      log.error('Failed to create ping-pong textures', e);
    }
  }

  createBlackTexture(createSolidColorTexture: (r: number, g: number, b: number, a: number) => GPUTexture | null): void {
    this.blackTexture = createSolidColorTexture(0, 0, 0, 255);
  }

  setResolution(width: number, height: number): boolean {
    if (this.outputWidth === width && this.outputHeight === height) {
      return false;
    }
    this.outputWidth = width;
    this.outputHeight = height;
    this.createPingPongTextures();
    return true;
  }

  getResolution(): { width: number; height: number } {
    return { width: this.outputWidth, height: this.outputHeight };
  }

  getTargets(): RenderTargets {
    return {
      pingTexture: this.pingTexture,
      pongTexture: this.pongTexture,
      pingView: this.pingView,
      pongView: this.pongView,
      independentPingTexture: this.independentPingTexture,
      independentPongTexture: this.independentPongTexture,
      independentPingView: this.independentPingView,
      independentPongView: this.independentPongView,
      blackTexture: this.blackTexture,
    };
  }

  getPingView(): GPUTextureView | null { return this.pingView; }
  getPongView(): GPUTextureView | null { return this.pongView; }
  getPingTexture(): GPUTexture | null { return this.pingTexture; }
  getPongTexture(): GPUTexture | null { return this.pongTexture; }
  getIndependentPingView(): GPUTextureView | null { return this.independentPingView; }
  getIndependentPongView(): GPUTextureView | null { return this.independentPongView; }
  getBlackTexture(): GPUTexture | null { return this.blackTexture; }

  clearAll(): void {
    this.pingTexture = null;
    this.pongTexture = null;
    this.pingView = null;
    this.pongView = null;
    this.independentPingTexture = null;
    this.independentPongTexture = null;
    this.independentPingView = null;
    this.independentPongView = null;
    this.blackTexture = null;
  }

  destroy(): void {
    this.pingTexture?.destroy();
    this.pongTexture?.destroy();
    this.independentPingTexture?.destroy();
    this.independentPongTexture?.destroy();
    this.blackTexture?.destroy();
    this.clearAll();
  }
}
