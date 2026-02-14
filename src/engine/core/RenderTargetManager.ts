// Manages ping-pong render targets for compositing

import { Logger } from '../../services/logger';
import type { RenderTargets } from './types';
import type { GpuMemoryManager } from '../gpuMemory/GpuMemoryManager';

const log = Logger.create('RenderTargetManager');

export class RenderTargetManager {
  private device: GPUDevice;
  private gpuMemoryManager: GpuMemoryManager | null = null;
  private pingTexture: GPUTexture | null = null;
  private pongTexture: GPUTexture | null = null;
  private pingView: GPUTextureView | null = null;
  private pongView: GPUTextureView | null = null;
  private independentPingTexture: GPUTexture | null = null;
  private independentPongTexture: GPUTexture | null = null;
  private independentPingView: GPUTextureView | null = null;
  private independentPongView: GPUTextureView | null = null;
  private blackTexture: GPUTexture | null = null;

  // Effect pre-processing temp textures (for applying effects to source before compositing)
  private effectTempTexture: GPUTexture | null = null;
  private effectTempTexture2: GPUTexture | null = null;
  private effectTempView: GPUTextureView | null = null;
  private effectTempView2: GPUTextureView | null = null;

  private outputWidth = 640;
  private outputHeight = 360;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  setGpuMemoryManager(manager: GpuMemoryManager): void {
    this.gpuMemoryManager = manager;
  }

  createPingPongTextures(): void {
    // During re-creation we intentionally DON'T call .destroy() on old textures.
    // Calling destroy() while GPU commands are still in-flight causes
    // "Destroyed texture used in a submit" warnings and black preview.
    // Instead we null the references and let GC reclaim them after the GPU is done.
    // Explicit .destroy() only happens in the destroy() method during final teardown.

    // Reset all references (old textures will be GC'd when GPU is done)
    this.pingTexture = null;
    this.pongTexture = null;
    this.independentPingTexture = null;
    this.independentPongTexture = null;
    this.pingView = null;
    this.pongView = null;
    this.independentPingView = null;
    this.independentPongView = null;
    this.effectTempTexture = null;
    this.effectTempTexture2 = null;
    this.effectTempView = null;
    this.effectTempView2 = null;

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

      // Effect pre-processing temp textures (for applying effects to source before compositing)
      log.debug('Creating effect temp textures...');
      this.effectTempTexture = this.device.createTexture({
        size: [this.outputWidth, this.outputHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });

      this.effectTempTexture2 = this.device.createTexture({
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
      if (this.effectTempTexture && this.effectTempTexture2) {
        this.effectTempView = this.effectTempTexture.createView();
        this.effectTempView2 = this.effectTempTexture2.createView();
      }

      // Register all textures with GPU memory manager for VRAM tracking
      this.registerTexturesWithMemoryManager();

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
  getEffectTempTexture(): GPUTexture | null { return this.effectTempTexture; }
  getEffectTempTexture2(): GPUTexture | null { return this.effectTempTexture2; }
  getEffectTempView(): GPUTextureView | null { return this.effectTempView; }
  getEffectTempView2(): GPUTextureView | null { return this.effectTempView2; }

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
    this.effectTempTexture = null;
    this.effectTempTexture2 = null;
    this.effectTempView = null;
    this.effectTempView2 = null;
  }

  private registerTexturesWithMemoryManager(): void {
    if (!this.gpuMemoryManager) return;
    const w = this.outputWidth;
    const h = this.outputHeight;
    const texBytes = w * h * 4; // rgba8unorm = 4 bpp

    // Unregister old first
    for (const id of ['rt_ping', 'rt_pong', 'rt_indPing', 'rt_indPong', 'rt_effectTmp1', 'rt_effectTmp2']) {
      this.gpuMemoryManager.unregisterExternal(id);
    }

    if (this.pingTexture) this.gpuMemoryManager.registerExternal('rt_ping', this.pingTexture, texBytes, 'pingPong');
    if (this.pongTexture) this.gpuMemoryManager.registerExternal('rt_pong', this.pongTexture, texBytes, 'pingPong');
    if (this.independentPingTexture) this.gpuMemoryManager.registerExternal('rt_indPing', this.independentPingTexture, texBytes, 'pingPong');
    if (this.independentPongTexture) this.gpuMemoryManager.registerExternal('rt_indPong', this.independentPongTexture, texBytes, 'pingPong');
    if (this.effectTempTexture) this.gpuMemoryManager.registerExternal('rt_effectTmp1', this.effectTempTexture, texBytes, 'effectTemp');
    if (this.effectTempTexture2) this.gpuMemoryManager.registerExternal('rt_effectTmp2', this.effectTempTexture2, texBytes, 'effectTemp');
  }

  destroy(): void {
    this.pingTexture?.destroy();
    this.pongTexture?.destroy();
    this.independentPingTexture?.destroy();
    this.independentPongTexture?.destroy();
    this.blackTexture?.destroy();
    this.effectTempTexture?.destroy();
    this.effectTempTexture2?.destroy();
    this.clearAll();
  }
}
