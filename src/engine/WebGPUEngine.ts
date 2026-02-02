// WebGPU Rendering Engine - Thin Facade
// Orchestrates: PerformanceStats, RenderTargetManager, OutputWindowManager,
//               RenderLoop, LayerCollector, Compositor, NestedCompRenderer

import type { Layer, OutputWindow, EngineStats, LayerRenderData } from './core/types';
import { WebGPUContext, type GPUPowerPreference } from './core/WebGPUContext';
import { TextureManager } from './texture/TextureManager';
import { MaskTextureManager } from './texture/MaskTextureManager';
import { ScrubbingCache } from './texture/ScrubbingCache';
import { CompositorPipeline } from './pipeline/CompositorPipeline';
import { EffectsPipeline } from '../effects/EffectsPipeline';
import { OutputPipeline } from './pipeline/OutputPipeline';
import { VideoFrameManager } from './video/VideoFrameManager';
import { useMediaStore } from '../stores/mediaStore';
import { useSettingsStore } from '../stores/settingsStore';
import { reportRenderTime } from '../services/performanceMonitor';
import { Logger } from '../services/logger';

const log = Logger.create('WebGPUEngine');

// New modules
import { PerformanceStats } from './stats/PerformanceStats';
import { RenderTargetManager } from './core/RenderTargetManager';
import { OutputWindowManager } from './managers/OutputWindowManager';
import { RenderLoop } from './render/RenderLoop';
import { LayerCollector } from './render/LayerCollector';
import { Compositor } from './render/Compositor';
import { NestedCompRenderer } from './render/NestedCompRenderer';

export class WebGPUEngine {
  // Core context
  private context: WebGPUContext;

  // Extracted modules
  private performanceStats: PerformanceStats;
  private renderTargetManager: RenderTargetManager | null = null;
  private outputWindowManager: OutputWindowManager | null = null;
  private renderLoop: RenderLoop | null = null;
  private layerCollector: LayerCollector | null = null;
  private compositor: Compositor | null = null;
  private nestedCompRenderer: NestedCompRenderer | null = null;

  // Existing managers (unchanged)
  private textureManager: TextureManager | null = null;
  private maskTextureManager: MaskTextureManager | null = null;
  private scrubbingCache: ScrubbingCache | null = null;
  private videoFrameManager: VideoFrameManager;

  // Pipelines
  private compositorPipeline: CompositorPipeline | null = null;
  private effectsPipeline: EffectsPipeline | null = null;
  private outputPipeline: OutputPipeline | null = null;

  // Resources
  private sampler: GPUSampler | null = null;

  // Canvas management (kept inline - simple Map operations)
  private previewContext: GPUCanvasContext | null = null;
  private previewCanvases: Map<string, GPUCanvasContext> = new Map();
  private independentPreviewCanvases: Map<string, GPUCanvasContext> = new Map();
  private independentCanvasCompositions: Map<string, string> = new Map();
  private previewCanvasElements: Map<string, HTMLCanvasElement> = new Map();
  private independentCanvasElements: Map<string, HTMLCanvasElement> = new Map();
  private mainPreviewCanvas: HTMLCanvasElement | null = null;

  // State flags
  private isRecoveringFromDeviceLoss = false;
  private isGeneratingRamPreview = false;
  private isExporting = false;
  private showTransparencyGrid = false;

  // Video time tracking (for optimization)
  private lastVideoTime: Map<string, number> = new Map();

  // RAM preview playback
  private ramPlaybackCanvas: HTMLCanvasElement | null = null;
  private ramPlaybackCtx: CanvasRenderingContext2D | null = null;

  // Export canvas (OffscreenCanvas for zero-copy VideoFrame creation)
  private exportCanvas: OffscreenCanvas | null = null;
  private exportCanvasContext: GPUCanvasContext | null = null;

  constructor() {
    this.context = new WebGPUContext();
    this.videoFrameManager = new VideoFrameManager();
    this.performanceStats = new PerformanceStats();

    // Device recovery handlers
    this.context.onDeviceLost((reason) => {
      log.warn('Device lost', { reason });
      this.isRecoveringFromDeviceLoss = true;
      this.handleDeviceLost();
    });

    this.context.onDeviceRestored(() => {
      log.info('Device restored');
      this.handleDeviceRestored();
      this.isRecoveringFromDeviceLoss = false;
    });
  }

  // === INITIALIZATION ===

  async initialize(): Promise<boolean> {
    const preference = useSettingsStore.getState().gpuPowerPreference;
    const success = await this.context.initialize(preference);
    if (!success) return false;

    await this.createResources();
    log.info('Engine initialized');
    return true;
  }

  private async createResources(): Promise<void> {
    const device = this.context.getDevice();
    if (!device) return;

    // Initialize managers
    this.textureManager = new TextureManager(device);
    this.maskTextureManager = new MaskTextureManager(device);
    this.scrubbingCache = new ScrubbingCache(device);

    // Create sampler
    this.sampler = this.context.createSampler();

    // Create pipelines
    this.compositorPipeline = new CompositorPipeline(device);
    this.effectsPipeline = new EffectsPipeline(device);
    this.outputPipeline = new OutputPipeline(device);
    await this.compositorPipeline.createPipelines();
    await this.effectsPipeline.createPipelines();
    await this.outputPipeline.createPipeline();

    // Small delay to let Vulkan memory manager settle after pipeline creation
    await new Promise(resolve => setTimeout(resolve, 100));

    // Initialize extracted modules
    this.renderTargetManager = new RenderTargetManager(device);

    // Create black texture first (tiny - 1x1 pixel)
    this.renderTargetManager.createBlackTexture((r, g, b, a) =>
      this.context.createSolidColorTexture(r, g, b, a)
    );

    // Another small delay before large texture allocation
    // Critical for Vulkan on Linux - memory manager needs time to settle
    await new Promise(resolve => setTimeout(resolve, 50));

    // Create ping-pong textures last (largest memory allocation)
    this.renderTargetManager.createPingPongTextures();

    const { width, height } = this.renderTargetManager.getResolution();
    this.outputWindowManager = new OutputWindowManager(width, height);

    this.layerCollector = new LayerCollector();

    this.compositor = new Compositor(
      this.compositorPipeline,
      this.effectsPipeline,
      this.maskTextureManager
    );

    this.nestedCompRenderer = new NestedCompRenderer(
      device,
      this.compositorPipeline,
      this.effectsPipeline,
      this.textureManager,
      this.maskTextureManager
    );

    this.renderLoop = new RenderLoop(this.performanceStats, {
      isRecovering: () => this.isRecoveringFromDeviceLoss || this.context.recovering,
      isExporting: () => this.isExporting,
      onRender: () => {}, // Set by start()
    });
  }

  // === DEVICE RECOVERY ===

  private handleDeviceLost(): void {
    this.renderLoop?.stop();

    // Clear GPU resources
    this.renderTargetManager?.clearAll();
    this.previewContext = null;
    this.previewCanvases.clear();
    this.independentPreviewCanvases.clear();
    this.lastVideoTime.clear();

    // Clear managers
    this.textureManager = null;
    this.maskTextureManager = null;
    this.scrubbingCache = null;
    this.compositorPipeline = null;
    this.effectsPipeline = null;
    this.outputPipeline = null;

    log.debug('Resources cleaned after device loss');
  }

  private async handleDeviceRestored(): Promise<void> {
    await this.createResources();

    // Reconfigure canvases
    if (this.mainPreviewCanvas) {
      this.previewContext = this.context.configureCanvas(this.mainPreviewCanvas);
    }
    for (const [id, canvas] of this.previewCanvasElements) {
      const ctx = this.context.configureCanvas(canvas);
      if (ctx) this.previewCanvases.set(id, ctx);
    }
    for (const [id, canvas] of this.independentCanvasElements) {
      const ctx = this.context.configureCanvas(canvas);
      if (ctx) this.independentPreviewCanvases.set(id, ctx);
    }

    this.renderLoop?.start();
    this.requestRender();
    log.info('Recovery complete');
  }

  // === CANVAS MANAGEMENT ===

  setPreviewCanvas(canvas: HTMLCanvasElement): void {
    this.mainPreviewCanvas = canvas;
    this.previewContext = this.context.configureCanvas(canvas);
  }

  registerPreviewCanvas(id: string, canvas: HTMLCanvasElement): void {
    this.previewCanvasElements.set(id, canvas);
    const ctx = this.context.configureCanvas(canvas);
    if (ctx) {
      this.previewCanvases.set(id, ctx);
      log.debug('Registered preview canvas', { id });
    }
  }

  unregisterPreviewCanvas(id: string): void {
    this.previewCanvases.delete(id);
    this.previewCanvasElements.delete(id);
    log.debug('Unregistered preview canvas', { id });
  }

  registerIndependentPreviewCanvas(id: string, canvas: HTMLCanvasElement, compositionId?: string): void {
    this.independentCanvasElements.set(id, canvas);
    const ctx = this.context.configureCanvas(canvas);
    if (ctx) {
      this.independentPreviewCanvases.set(id, ctx);
      if (compositionId) this.independentCanvasCompositions.set(id, compositionId);
      log.debug('Registered independent preview canvas', { id, compositionId });
    }
  }

  unregisterIndependentPreviewCanvas(id: string): void {
    this.independentPreviewCanvases.delete(id);
    this.independentCanvasElements.delete(id);
    this.independentCanvasCompositions.delete(id);
    log.debug('Unregistered independent preview canvas', { id });
  }

  setIndependentCanvasComposition(canvasId: string, compositionId: string): void {
    this.independentCanvasCompositions.set(canvasId, compositionId);
  }

  /** @deprecated Use setIndependentCanvasComposition instead */
  setCanvasMirrorsActiveComp(_canvasId: string, _mirrors: boolean): void {
    // Kept for backward compatibility - no-op
  }

  // === OUTPUT WINDOWS ===

  createOutputWindow(id: string, name: string): OutputWindow | null {
    const device = this.context.getDevice();
    if (!device || !this.outputWindowManager) return null;
    return this.outputWindowManager.createOutputWindow(id, name, device);
  }

  closeOutputWindow(id: string): void {
    this.outputWindowManager?.closeOutputWindow(id);
  }

  // === MASK MANAGEMENT ===

  updateMaskTexture(layerId: string, imageData: ImageData | null): void {
    this.maskTextureManager?.updateMaskTexture(layerId, imageData);
  }

  removeMaskTexture(layerId: string): void {
    this.maskTextureManager?.removeMaskTexture(layerId);
  }

  hasMaskTexture(layerId: string): boolean {
    return this.maskTextureManager?.hasMaskTexture(layerId) ?? false;
  }

  // === VIDEO MANAGEMENT ===

  registerVideo(video: HTMLVideoElement): void {
    this.videoFrameManager.registerVideo(video);
  }

  setActiveVideo(video: HTMLVideoElement | null): void {
    this.videoFrameManager.setActiveVideo(video);
  }

  cleanupVideo(video: HTMLVideoElement): void {
    this.scrubbingCache?.cleanupVideo(video);
    this.videoFrameManager.cleanupVideo(video);
    if (video.src) this.lastVideoTime.delete(video.src);
    log.debug('Cleaned up video resources');
  }

  setHasActiveVideo(hasVideo: boolean): void {
    this.renderLoop?.setHasActiveVideo(hasVideo);
  }

  // === TEXTURE MANAGEMENT ===

  createImageTexture(image: HTMLImageElement): GPUTexture | null {
    return this.textureManager?.createImageTexture(image) ?? null;
  }

  importVideoTexture(source: HTMLVideoElement | VideoFrame): GPUExternalTexture | null {
    return this.textureManager?.importVideoTexture(source) ?? null;
  }

  // === CACHING ===

  clearCaches(): void {
    this.scrubbingCache?.clearAll();
    this.textureManager?.clearCaches();
    log.debug('Cleared all caches');
  }

  clearVideoCache(): void {
    this.lastVideoTime.clear();
    log.debug('Cleared video texture cache');
  }

  cacheFrameAtTime(video: HTMLVideoElement, time: number): void {
    this.scrubbingCache?.cacheFrameAtTime(video, time);
  }

  getCachedFrame(videoSrc: string, time: number): GPUTextureView | null {
    return this.scrubbingCache?.getCachedFrame(videoSrc, time) ?? null;
  }

  getScrubbingCacheStats(): { count: number; maxCount: number } {
    return this.scrubbingCache?.getScrubbingCacheStats() ?? { count: 0, maxCount: 0 };
  }

  clearScrubbingCache(videoSrc?: string): void {
    this.scrubbingCache?.clearScrubbingCache(videoSrc);
  }

  // === RAM PREVIEW CACHE ===

  async cacheCompositeFrame(time: number): Promise<void> {
    if (!this.scrubbingCache) return;
    if (this.scrubbingCache.hasCompositeCacheFrame(time)) return;

    const pixels = await this.readPixels();
    if (!pixels) return;

    const { width, height } = this.renderTargetManager?.getResolution() ?? { width: 640, height: 360 };

    if (this.scrubbingCache.getCompositeCacheStats(width, height).count === 0) {
      let nonZero = 0;
      for (let i = 0; i < Math.min(1000, pixels.length); i++) {
        if (pixels[i] !== 0) nonZero++;
      }
      log.debug('RAM Preview first frame', { nonZero, width, height });
    }

    const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
    this.scrubbingCache.cacheCompositeFrame(time, imageData);
  }

  getCachedCompositeFrame(time: number): ImageData | null {
    return this.scrubbingCache?.getCachedCompositeFrame(time) ?? null;
  }

  hasCompositeCacheFrame(time: number): boolean {
    return this.scrubbingCache?.hasCompositeCacheFrame(time) ?? false;
  }

  clearCompositeCache(): void {
    this.scrubbingCache?.clearCompositeCache();
    this.ramPlaybackCanvas = null;
    this.ramPlaybackCtx = null;
    log.debug('Composite cache cleared');
  }

  getCompositeCacheStats(): { count: number; maxFrames: number; memoryMB: number } {
    const { width, height } = this.renderTargetManager?.getResolution() ?? { width: 640, height: 360 };
    return this.scrubbingCache?.getCompositeCacheStats(width, height) ?? { count: 0, maxFrames: 0, memoryMB: 0 };
  }

  setGeneratingRamPreview(generating: boolean): void {
    this.isGeneratingRamPreview = generating;
  }

  setExporting(exporting: boolean): void {
    this.isExporting = exporting;
    if (exporting) this.lastVideoTime.clear();
    log.info('Export mode', { enabled: exporting });
  }

  getIsExporting(): boolean {
    return this.isExporting;
  }

  /**
   * Initialize export canvas for zero-copy VideoFrame creation.
   * Call this before starting export with the target resolution.
   */
  initExportCanvas(width: number, height: number): boolean {
    const device = this.context.getDevice();
    if (!device) {
      log.error('Cannot init export canvas: no device');
      return false;
    }

    // Create OffscreenCanvas at export resolution
    this.exportCanvas = new OffscreenCanvas(width, height);
    const ctx = this.exportCanvas.getContext('webgpu');
    if (!ctx) {
      log.error('Failed to get WebGPU context from OffscreenCanvas');
      this.exportCanvas = null;
      return false;
    }

    // Configure with same settings as preview canvases
    const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({
      device,
      format: preferredFormat,
      alphaMode: 'premultiplied',
    });

    this.exportCanvasContext = ctx;
    log.info('Export canvas initialized', { width, height, format: preferredFormat });
    return true;
  }

  /**
   * Create VideoFrame directly from the export canvas (zero-copy path).
   * Must call render() first to populate the canvas.
   * Waits for GPU work to complete before capturing the frame.
   */
  async createVideoFrameFromExport(timestamp: number, duration: number): Promise<VideoFrame | null> {
    if (!this.exportCanvas) {
      log.error('Export canvas not initialized');
      return null;
    }

    const device = this.context.getDevice();
    if (!device) {
      log.error('No GPU device');
      return null;
    }

    // CRITICAL: Wait for GPU to finish rendering before capturing frame
    await device.queue.onSubmittedWorkDone();

    try {
      // Create VideoFrame directly from OffscreenCanvas - browser handles GPU→VideoFrame
      const frame = new VideoFrame(this.exportCanvas, {
        timestamp,
        duration,
        alpha: 'discard', // We don't need alpha channel in export
      });
      return frame;
    } catch (e) {
      log.error('Failed to create VideoFrame from export canvas', e);
      return null;
    }
  }

  /**
   * Cleanup export canvas after export completes.
   */
  cleanupExportCanvas(): void {
    this.exportCanvasContext = null;
    this.exportCanvas = null;
    log.debug('Export canvas cleaned up');
  }

  // === RENDER LOOP ===

  requestRender(): void {
    this.renderLoop?.requestRender();
  }

  getIsIdle(): boolean {
    return this.renderLoop?.getIsIdle() ?? false;
  }

  updatePlayheadTracking(playhead: number): boolean {
    return this.renderLoop?.updatePlayheadTracking(playhead) ?? false;
  }

  start(renderCallback: () => void): void {
    if (!this.performanceStats) return;

    // Create new loop with the callback
    this.renderLoop = new RenderLoop(this.performanceStats, {
      isRecovering: () => this.isRecoveringFromDeviceLoss || this.context.recovering,
      isExporting: () => this.isExporting,
      onRender: renderCallback,
    });
    this.renderLoop.start();
  }

  stop(): void {
    this.renderLoop?.stop();
  }

  // === MAIN RENDER ===

  render(layers: Layer[]): void {
    if (this.isRecoveringFromDeviceLoss || this.context.recovering) return;

    const device = this.context.getDevice();
    if (!device || !this.compositorPipeline || !this.outputPipeline || !this.sampler) return;
    if (!this.renderTargetManager || !this.layerCollector || !this.compositor) return;

    const pingView = this.renderTargetManager.getPingView();
    const pongView = this.renderTargetManager.getPongView();
    if (!pingView || !pongView) return;

    // Clear frame-scoped caches (external texture bind groups)
    this.compositorPipeline.beginFrame();

    const t0 = performance.now();
    const { width, height } = this.renderTargetManager.getResolution();

    // Collect layer data
    const t1 = performance.now();
    const layerData = this.layerCollector.collect(layers, {
      textureManager: this.textureManager!,
      scrubbingCache: this.scrubbingCache,
      getLastVideoTime: (key) => this.lastVideoTime.get(key),
      setLastVideoTime: (key, time) => this.lastVideoTime.set(key, time),
      isExporting: this.isExporting,
    });
    const importTime = performance.now() - t1;

    // Update stats
    this.performanceStats.setDecoder(this.layerCollector.getDecoder());
    this.renderLoop?.setHasActiveVideo(this.layerCollector.hasActiveVideo());

    // Handle empty layers
    if (layerData.length === 0) {
      this.renderEmptyFrame(device);
      this.performanceStats.setLayerCount(0);
      return;
    }

    // Pre-render nested compositions (batched with main composite)
    const commandBuffers: GPUCommandBuffer[] = [];
    let hasNestedComps = false;

    const preRenderEncoder = device.createCommandEncoder();
    for (const data of layerData) {
      if (data.layer.source?.nestedComposition) {
        hasNestedComps = true;
        const nc = data.layer.source.nestedComposition;
        const view = this.nestedCompRenderer!.preRender(
          nc.compositionId, nc.layers, nc.width, nc.height, preRenderEncoder, this.sampler, nc.currentTime
        );
        if (view) data.textureView = view;
      }
    }
    if (hasNestedComps) {
      commandBuffers.push(preRenderEncoder.finish());
    }

    // Composite
    const t2 = performance.now();
    const commandEncoder = device.createCommandEncoder();
    const result = this.compositor.composite(layerData, commandEncoder, {
      device, sampler: this.sampler, pingView, pongView, outputWidth: width, outputHeight: height,
    });
    const renderTime = performance.now() - t2;

    // Output
    const finalIsPing = result.usedPing;
    const outputBindGroup = this.outputPipeline!.getOutputBindGroup(this.sampler, result.finalView, finalIsPing);
    this.outputPipeline!.updateUniforms(this.showTransparencyGrid, width, height);

    const skipCanvas = this.isGeneratingRamPreview || this.isExporting;
    if (!skipCanvas) {
      if (this.previewContext) {
        this.outputPipeline!.renderToCanvas(commandEncoder, this.previewContext, outputBindGroup);
      }
      for (const ctx of this.previewCanvases.values()) {
        this.outputPipeline!.renderToCanvas(commandEncoder, ctx, outputBindGroup);
      }
      // Independent canvases showing active comp
      const activeCompId = useMediaStore.getState().activeCompositionId;
      for (const [canvasId, compId] of this.independentCanvasCompositions) {
        if (compId === activeCompId) {
          const ctx = this.independentPreviewCanvases.get(canvasId);
          if (ctx) this.outputPipeline!.renderToCanvas(commandEncoder, ctx, outputBindGroup);
        }
      }
      // Output windows
      for (const output of this.outputWindowManager!.getOutputWindows().values()) {
        if (output.context) this.outputPipeline!.renderToCanvas(commandEncoder, output.context, outputBindGroup);
      }
    }

    // Render to export canvas for zero-copy VideoFrame creation
    if (this.isExporting && this.exportCanvasContext) {
      // Disable transparency grid for export
      this.outputPipeline!.updateUniforms(false, width, height);
      this.outputPipeline!.renderToCanvas(commandEncoder, this.exportCanvasContext, outputBindGroup);
    }

    // Batch submit all command buffers in single call
    commandBuffers.push(commandEncoder.finish());
    const t3 = performance.now();
    device.queue.submit(commandBuffers);
    const submitTime = performance.now() - t3;

    // Cleanup after submit
    if (hasNestedComps) {
      this.nestedCompRenderer!.cleanupPendingTextures();
    }

    // Stats
    const totalTime = performance.now() - t0;
    this.performanceStats.recordRenderTiming({
      importTexture: importTime,
      createBindGroup: 0,
      renderPass: renderTime,
      submit: submitTime,
      total: totalTime,
    });
    this.performanceStats.setLayerCount(result.layerCount);
    this.performanceStats.updateStats();
    reportRenderTime(totalTime);
  }

  private renderEmptyFrame(device: GPUDevice): void {
    const commandEncoder = device.createCommandEncoder();
    if (this.previewContext) {
      const pass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: this.previewContext.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.end();
    }
    for (const ctx of this.previewCanvases.values()) {
      const pass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.end();
    }
    device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Render specific layers to a specific preview canvas
   * Used for multi-composition preview where each preview shows different content
   */
  renderToPreviewCanvas(canvasId: string, layers: Layer[]): void {
    if (this.isRecoveringFromDeviceLoss || this.context.recovering) return;

    const device = this.context.getDevice();
    const canvasContext = this.independentPreviewCanvases.get(canvasId);
    if (!device || !canvasContext || !this.compositorPipeline || !this.outputPipeline || !this.sampler) return;

    const indPingView = this.renderTargetManager?.getIndependentPingView();
    const indPongView = this.renderTargetManager?.getIndependentPongView();
    if (!indPingView || !indPongView) return;

    // Prepare layer data
    const layerData: LayerRenderData[] = [];
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer?.visible || !layer.source || layer.opacity === 0) continue;

      if (layer.source.videoElement) {
        const video = layer.source.videoElement;
        if (video.readyState >= 2) {
          const extTex = this.textureManager?.importVideoTexture(video);
          if (extTex) {
            layerData.push({ layer, isVideo: true, externalTexture: extTex, textureView: null, sourceWidth: video.videoWidth, sourceHeight: video.videoHeight });
            continue;
          }
        }
      }
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = this.textureManager?.getCachedImageTexture(img);
        if (!texture) texture = this.textureManager?.createImageTexture(img) ?? undefined;
        if (texture) {
          layerData.push({ layer, isVideo: false, externalTexture: null, textureView: this.textureManager!.getImageView(texture), sourceWidth: img.naturalWidth, sourceHeight: img.naturalHeight });
        }
      }
      if (layer.source.textCanvas) {
        const canvas = layer.source.textCanvas;
        const texture = this.textureManager?.createCanvasTexture(canvas);
        if (texture) {
          layerData.push({ layer, isVideo: false, externalTexture: null, textureView: this.textureManager!.getImageView(texture), sourceWidth: canvas.width, sourceHeight: canvas.height });
        }
      }
    }

    const { width, height } = this.renderTargetManager!.getResolution();

    if (layerData.length === 0) {
      const commandEncoder = device.createCommandEncoder();
      const blackTex = this.renderTargetManager!.getBlackTexture();
      if (blackTex) {
        const blackView = blackTex.createView();
        const blackBindGroup = this.outputPipeline.createOutputBindGroup(this.sampler, blackView);
        this.outputPipeline.renderToCanvas(commandEncoder, canvasContext, blackBindGroup);
      }
      device.queue.submit([commandEncoder.finish()]);
      return;
    }

    const commandEncoder = device.createCommandEncoder();

    // Ping-pong compositing using independent buffers
    let readView = indPingView;
    let writeView = indPongView;
    let usePing = true;

    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{ view: readView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
    });
    clearPass.end();

    for (const data of layerData) {
      const layer = data.layer;
      const uniformBuffer = this.compositorPipeline!.getOrCreateUniformBuffer(layer.id);
      const sourceAspect = data.sourceWidth / data.sourceHeight;
      const outputAspect = width / height;
      const maskLookupId = layer.maskClipId || layer.id;
      // Get mask info - maskTextureManager should always exist during rendering
      const maskManager = this.maskTextureManager!;
      const maskInfo = maskManager.getMaskInfo(maskLookupId) ?? { hasMask: false, view: maskManager.getWhiteMaskView() };
      const hasMask = maskInfo.hasMask;
      const maskTextureView = maskInfo.view;

      this.compositorPipeline!.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = this.compositorPipeline!.getExternalCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createExternalCompositeBindGroup(this.sampler!, readView, data.externalTexture, uniformBuffer, maskTextureView);
      } else if (data.textureView) {
        pipeline = this.compositorPipeline!.getCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createCompositeBindGroup(this.sampler!, readView, data.textureView, uniformBuffer, maskTextureView);
      } else {
        continue;
      }

      const compositePass = commandEncoder.beginRenderPass({
        colorAttachments: [{ view: writeView, loadOp: 'clear', storeOp: 'store' }],
      });
      compositePass.setPipeline(pipeline);
      compositePass.setBindGroup(0, bindGroup);
      compositePass.draw(6);
      compositePass.end();

      [readView, writeView] = [writeView, readView];
      usePing = !usePing;
    }

    const finalIsPing = !usePing;
    const outputBindGroup = this.outputPipeline!.getOutputBindGroup(this.sampler!, readView, finalIsPing);
    this.outputPipeline!.renderToCanvas(commandEncoder, canvasContext, outputBindGroup);

    device.queue.submit([commandEncoder.finish()]);
  }

  renderCachedFrame(time: number): boolean {
    const device = this.context.getDevice();
    if (!this.previewContext || !device || !this.scrubbingCache || !this.outputPipeline || !this.sampler) {
      return false;
    }

    const gpuCached = this.scrubbingCache.getGpuCachedFrame(time);
    if (gpuCached) {
      log.debug('RAM Preview cache hit (GPU)', { time: time.toFixed(3) });
      const commandEncoder = device.createCommandEncoder();
      this.outputPipeline.renderToCanvas(commandEncoder, this.previewContext, gpuCached.bindGroup);
      for (const previewCtx of this.previewCanvases.values()) {
        this.outputPipeline.renderToCanvas(commandEncoder, previewCtx, gpuCached.bindGroup);
      }
      for (const output of this.outputWindowManager!.getOutputWindows().values()) {
        if (output.context) this.outputPipeline.renderToCanvas(commandEncoder, output.context, gpuCached.bindGroup);
      }
      device.queue.submit([commandEncoder.finish()]);
      return true;
    }

    const imageData = this.scrubbingCache.getCachedCompositeFrame(time);
    if (!imageData) {
      // Only log occasionally to avoid spam
      if (Math.random() < 0.05) {
        log.debug('RAM Preview cache miss', { time: time.toFixed(3), cacheSize: this.scrubbingCache.getCompositeCacheStats(1920, 1080).count });
      }
      return false;
    }
    log.debug('RAM Preview cache hit (ImageData→GPU)', { time: time.toFixed(3) });

    try {
      const { width, height } = { width: imageData.width, height: imageData.height };

      if (!this.ramPlaybackCanvas || !this.ramPlaybackCtx) {
        this.ramPlaybackCanvas = document.createElement('canvas');
        this.ramPlaybackCanvas.width = width;
        this.ramPlaybackCanvas.height = height;
        this.ramPlaybackCtx = this.ramPlaybackCanvas.getContext('2d', { willReadFrequently: false });
      } else if (this.ramPlaybackCanvas.width !== width || this.ramPlaybackCanvas.height !== height) {
        this.ramPlaybackCanvas.width = width;
        this.ramPlaybackCanvas.height = height;
      }

      if (!this.ramPlaybackCtx) return false;

      this.ramPlaybackCtx.putImageData(imageData, 0, 0);

      const texture = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      device.queue.copyExternalImageToTexture({ source: this.ramPlaybackCanvas }, { texture }, [width, height]);

      const view = texture.createView();
      const bindGroup = this.outputPipeline.createOutputBindGroup(this.sampler, view);

      this.scrubbingCache.addToGpuCache(time, { texture, view, bindGroup });

      const commandEncoder = device.createCommandEncoder();
      this.outputPipeline.renderToCanvas(commandEncoder, this.previewContext, bindGroup);
      for (const previewCtx of this.previewCanvases.values()) {
        this.outputPipeline.renderToCanvas(commandEncoder, previewCtx, bindGroup);
      }
      for (const output of this.outputWindowManager!.getOutputWindows().values()) {
        if (output.context) this.outputPipeline.renderToCanvas(commandEncoder, output.context, bindGroup);
      }
      device.queue.submit([commandEncoder.finish()]);
      return true;
    } catch (e) {
      log.warn('Failed to render cached frame', e);
      return false;
    }
  }

  // === NESTED COMPOSITION HELPERS ===

  hasNestedCompTexture(compositionId: string): boolean {
    return this.nestedCompRenderer?.hasTexture(compositionId) ?? false;
  }

  cacheActiveCompOutput(compositionId: string): void {
    const pingTex = this.renderTargetManager?.getPingTexture();
    const pongTex = this.renderTargetManager?.getPongTexture();
    if (!pingTex || !pongTex || !this.nestedCompRenderer) return;

    const { width, height } = this.renderTargetManager!.getResolution();
    const finalIsPing = !this.compositor?.getLastRenderWasPing();
    const sourceTexture = finalIsPing ? pingTex : pongTex;

    this.nestedCompRenderer.cacheActiveCompOutput(compositionId, sourceTexture, width, height);
  }

  copyMainOutputToPreview(canvasId: string): boolean {
    const device = this.context.getDevice();
    const canvasContext = this.independentPreviewCanvases.get(canvasId);
    const pingView = this.renderTargetManager?.getPingView();
    const pongView = this.renderTargetManager?.getPongView();

    if (!device || !canvasContext || !this.outputPipeline || !this.sampler || !pingView || !pongView) return false;

    const finalIsPing = !this.compositor?.getLastRenderWasPing();
    const finalView = finalIsPing ? pingView : pongView;

    const commandEncoder = device.createCommandEncoder();
    const outputBindGroup = this.outputPipeline.getOutputBindGroup(this.sampler, finalView, finalIsPing);
    this.outputPipeline.renderToCanvas(commandEncoder, canvasContext, outputBindGroup);
    device.queue.submit([commandEncoder.finish()]);
    return true;
  }

  copyNestedCompTextureToPreview(canvasId: string, compositionId: string): boolean {
    const device = this.context.getDevice();
    const canvasContext = this.independentPreviewCanvases.get(canvasId);
    const compTexture = this.nestedCompRenderer?.getTexture(compositionId);

    if (!device || !canvasContext || !compTexture || !this.outputPipeline || !this.sampler) return false;

    const commandEncoder = device.createCommandEncoder();
    const outputBindGroup = this.outputPipeline.createOutputBindGroup(this.sampler, compTexture.view);
    this.outputPipeline.renderToCanvas(commandEncoder, canvasContext, outputBindGroup);
    device.queue.submit([commandEncoder.finish()]);
    return true;
  }

  cleanupNestedCompTexture(compositionId: string): void {
    this.nestedCompRenderer?.cleanupTexture(compositionId);
  }

  // === RESOLUTION ===

  setResolution(width: number, height: number): void {
    if (this.renderTargetManager?.setResolution(width, height)) {
      this.scrubbingCache?.clearCompositeCache();
      this.scrubbingCache?.clearScrubbingCache();
      this.outputWindowManager?.updateResolution(width, height);
      this.outputPipeline?.invalidateCache();
      this.compositorPipeline?.invalidateBindGroupCache();
      log.debug('Caches cleared for resolution change', { width, height });
    }
  }

  setShowTransparencyGrid(show: boolean): void {
    this.showTransparencyGrid = show;
  }

  clearFrame(): void {
    const device = this.context.getDevice();
    const pingView = this.renderTargetManager?.getPingView();
    const pongView = this.renderTargetManager?.getPongView();
    if (!device || !pingView || !pongView) return;

    const commandEncoder = device.createCommandEncoder();

    const clearPing = commandEncoder.beginRenderPass({
      colorAttachments: [{ view: pingView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
    });
    clearPing.end();

    const clearPong = commandEncoder.beginRenderPass({
      colorAttachments: [{ view: pongView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
    });
    clearPong.end();

    const { width, height } = this.renderTargetManager!.getResolution();
    this.outputPipeline?.updateUniforms(this.showTransparencyGrid, width, height);
    const outputBindGroup = this.outputPipeline?.createOutputBindGroup(this.sampler!, pingView);
    if (outputBindGroup) {
      if (this.previewContext) {
        this.outputPipeline!.renderToCanvas(commandEncoder, this.previewContext, outputBindGroup);
      }
      for (const previewCtx of this.previewCanvases.values()) {
        this.outputPipeline!.renderToCanvas(commandEncoder, previewCtx, outputBindGroup);
      }
    }

    device.queue.submit([commandEncoder.finish()]);
  }

  getOutputDimensions(): { width: number; height: number } {
    return this.renderTargetManager?.getResolution() ?? { width: 640, height: 360 };
  }

  // === STATS ===

  getStats(): EngineStats {
    return this.performanceStats.getStats(this.getIsIdle());
  }

  // === ACCESSORS ===

  getDevice(): GPUDevice | null {
    return this.context.getDevice();
  }

  isDeviceValid(): boolean {
    return this.context.initialized && this.context.getDevice() !== null;
  }

  getGPUInfo(): { vendor: string; device: string; description: string } | null {
    return this.context.getGPUInfo();
  }

  getPowerPreference(): GPUPowerPreference {
    return this.context.getPowerPreference();
  }

  async reinitializeWithPreference(preference: GPUPowerPreference): Promise<boolean> {
    log.info('Reinitializing with preference', { preference });
    this.stop();
    this.handleDeviceLost();
    const success = await this.context.reinitializeWithPreference(preference);
    if (!success) {
      log.error('Failed to reinitialize with new preference');
      return false;
    }
    await this.createResources();

    if (this.mainPreviewCanvas) {
      this.previewContext = this.context.configureCanvas(this.mainPreviewCanvas);
    }
    for (const [id, canvas] of this.previewCanvasElements) {
      const ctx = this.context.configureCanvas(canvas);
      if (ctx) this.previewCanvases.set(id, ctx);
    }
    for (const [id, canvas] of this.independentCanvasElements) {
      const ctx = this.context.configureCanvas(canvas);
      if (ctx) this.independentPreviewCanvases.set(id, ctx);
    }

    this.requestRender();
    log.info('Reinitialize complete');
    return true;
  }

  // === PIXEL READBACK ===

  async readPixels(): Promise<Uint8ClampedArray | null> {
    const device = this.context.getDevice();
    const pingTex = this.renderTargetManager?.getPingTexture();
    const pongTex = this.renderTargetManager?.getPongTexture();
    if (!device || !pingTex || !pongTex) return null;

    const { width, height } = this.renderTargetManager!.getResolution();
    const sourceTexture = this.compositor?.getLastRenderWasPing() ? pingTex : pongTex;

    const bytesPerPixel = 4;
    const unalignedBytesPerRow = width * bytesPerPixel;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;
    const bufferSize = bytesPerRow * height;

    const stagingBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyTextureToBuffer(
      { texture: sourceTexture },
      { buffer: stagingBuffer, bytesPerRow, rowsPerImage: height },
      [width, height]
    );
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();
    const result = new Uint8ClampedArray(width * height * bytesPerPixel);
    const srcView = new Uint8Array(arrayBuffer);

    if (bytesPerRow === unalignedBytesPerRow) {
      result.set(srcView.subarray(0, result.length));
    } else {
      for (let y = 0; y < height; y++) {
        const srcOffset = y * bytesPerRow;
        const dstOffset = y * unalignedBytesPerRow;
        result.set(srcView.subarray(srcOffset, srcOffset + unalignedBytesPerRow), dstOffset);
      }
    }

    stagingBuffer.unmap();
    stagingBuffer.destroy();
    return result;
  }

  // === CLEANUP ===

  destroy(): void {
    this.stop();
    this.outputWindowManager?.destroy();
    this.renderTargetManager?.destroy();
    this.nestedCompRenderer?.destroy();
    this.textureManager?.destroy();
    this.maskTextureManager?.destroy();
    this.scrubbingCache?.destroy();
    this.videoFrameManager.destroy();
    this.compositorPipeline?.destroy();
    this.effectsPipeline?.destroy();
    this.outputPipeline?.destroy();
    this.context.destroy();
    this.lastVideoTime.clear();
  }
}

// === HMR SINGLETON ===

let engineInstance: WebGPUEngine;

const hot = typeof import.meta !== 'undefined'
  ? (import.meta as { hot?: { data: Record<string, unknown> } }).hot
  : undefined;

const hmrLog = Logger.create('WebGPU-HMR');

if (hot) {
  const existing = hot.data.engine as WebGPUEngine | undefined;
  if (existing) {
    hmrLog.debug('Reusing engine from HMR');
    existing.clearVideoCache();
    engineInstance = existing;
  } else {
    hmrLog.debug('Creating new engine');
    engineInstance = new WebGPUEngine();
    hot.data.engine = engineInstance;
  }
} else {
  engineInstance = new WebGPUEngine();
}

export const engine = engineInstance;
