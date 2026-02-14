// WebGPU Rendering Engine - Thin Facade
// Orchestrates: PerformanceStats, RenderTargetManager, OutputWindowManager,
//               RenderLoop, LayerCollector, Compositor, NestedCompRenderer

import type { Layer, EngineStats } from './core/types';
// OutputWindow type no longer needed — state lives in renderTargetStore
import { WebGPUContext, type GPUPowerPreference } from './core/WebGPUContext';
import { TextureManager } from './texture/TextureManager';
import { MaskTextureManager } from './texture/MaskTextureManager';
import { CacheManager } from './managers/CacheManager';
import { ExportCanvasManager } from './managers/ExportCanvasManager';
import { CompositorPipeline } from './pipeline/CompositorPipeline';
import { EffectsPipeline } from '../effects/EffectsPipeline';
import { OutputPipeline } from './pipeline/OutputPipeline';
import { SlicePipeline } from './pipeline/SlicePipeline';
import { VideoFrameManager } from './video/VideoFrameManager';
import { useSettingsStore } from '../stores/settingsStore';
import { useRenderTargetStore } from '../stores/renderTargetStore';
import { getSavedTargetMeta } from '../stores/sliceStore';
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
import { RenderDispatcher } from './render/RenderDispatcher';
import { GpuMemoryManager } from './gpuMemory/GpuMemoryManager';

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
  private renderDispatcher: RenderDispatcher | null = null;

  // Existing managers (unchanged)
  private textureManager: TextureManager | null = null;
  private maskTextureManager: MaskTextureManager | null = null;
  private cacheManager: CacheManager = new CacheManager();
  private exportCanvasManager: ExportCanvasManager = new ExportCanvasManager();
  private videoFrameManager: VideoFrameManager;

  // Pipelines
  private compositorPipeline: CompositorPipeline | null = null;
  private effectsPipeline: EffectsPipeline | null = null;
  private outputPipeline: OutputPipeline | null = null;
  private slicePipeline: SlicePipeline | null = null;

  // GPU memory tracking
  private gpuMemoryManager: GpuMemoryManager | null = null;

  // Resources
  private sampler: GPUSampler | null = null;

  // Unified canvas management - single Map replaces 6 old Maps
  private targetCanvases: Map<string, { canvas: HTMLCanvasElement; context: GPUCanvasContext }> = new Map();
  // Legacy: kept for backward compat during migration
  private mainPreviewCanvas: HTMLCanvasElement | null = null;
  private previewContext: GPUCanvasContext | null = null;

  // State flags
  private isRecoveringFromDeviceLoss = false;

  // Track whether play has ever been pressed — persists across RenderLoop recreations.
  // Before first play, idle detection is suppressed so video GPU surfaces stay warm.
  private hasEverPlayed = false;

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

    // GPU memory tracking
    this.gpuMemoryManager = new GpuMemoryManager(device);
    this.performanceStats.setGpuMemoryManager(this.gpuMemoryManager);

    // Initialize managers
    this.textureManager = new TextureManager(device);
    this.textureManager.setGpuMemoryManager(this.gpuMemoryManager);
    this.maskTextureManager = new MaskTextureManager(device);
    this.cacheManager.initialize(device);

    // Create sampler
    this.sampler = this.context.createSampler();

    // Create pipelines
    this.compositorPipeline = new CompositorPipeline(device);
    this.effectsPipeline = new EffectsPipeline(device);
    this.outputPipeline = new OutputPipeline(device);
    this.slicePipeline = new SlicePipeline(device);
    await this.compositorPipeline.createPipelines();
    await this.effectsPipeline.createPipelines();
    await this.outputPipeline.createPipeline();
    await this.slicePipeline.createPipeline();

    // Small delay to let Vulkan memory manager settle after pipeline creation
    await new Promise(resolve => setTimeout(resolve, 100));

    // Initialize extracted modules
    this.renderTargetManager = new RenderTargetManager(device);
    this.renderTargetManager.setGpuMemoryManager(this.gpuMemoryManager);

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
      this.maskTextureManager,
      this.cacheManager.getScrubbingCache()
    );

    this.renderLoop = new RenderLoop(this.performanceStats, {
      isRecovering: () => this.isRecoveringFromDeviceLoss || this.context.recovering,
      isExporting: () => this.exportCanvasManager.getIsExporting(),
      onRender: () => {}, // Set by start()
    });

    // Create render dispatcher with live deps (getters close over engine instance)
    const eng = this;
    this.renderDispatcher = new RenderDispatcher({
      getDevice: () => eng.context.getDevice(),
      isRecovering: () => eng.isRecoveringFromDeviceLoss || eng.context.recovering,
      get sampler() { return eng.sampler; },
      get previewContext() { return eng.previewContext; },
      get targetCanvases() { return eng.targetCanvases; },
      get compositorPipeline() { return eng.compositorPipeline; },
      get outputPipeline() { return eng.outputPipeline; },
      get slicePipeline() { return eng.slicePipeline; },
      get textureManager() { return eng.textureManager; },
      get maskTextureManager() { return eng.maskTextureManager; },
      get renderTargetManager() { return eng.renderTargetManager; },
      get layerCollector() { return eng.layerCollector; },
      get compositor() { return eng.compositor; },
      get nestedCompRenderer() { return eng.nestedCompRenderer; },
      get cacheManager() { return eng.cacheManager; },
      get exportCanvasManager() { return eng.exportCanvasManager; },
      get performanceStats() { return eng.performanceStats; },
      get renderLoop() { return eng.renderLoop; },
    });
  }

  // === DEVICE RECOVERY ===

  private handleDeviceLost(): void {
    this.renderLoop?.stop();

    // Clear GPU resources
    this.renderTargetManager?.clearAll();
    this.previewContext = null;
    this.targetCanvases.clear();
    this.cacheManager.handleDeviceLost();

    // Clear managers
    this.textureManager = null;
    this.maskTextureManager = null;
    this.compositorPipeline = null;
    this.effectsPipeline = null;
    this.outputPipeline = null;
    this.slicePipeline = null;

    log.debug('Resources cleaned after device loss');
  }

  private async handleDeviceRestored(): Promise<void> {
    await this.createResources();

    // Reconfigure main preview canvas
    if (this.mainPreviewCanvas) {
      this.previewContext = this.context.configureCanvas(this.mainPreviewCanvas);
    }

    // Reconfigure all target canvases from unified map
    for (const [id, entry] of this.targetCanvases) {
      const ctx = this.context.configureCanvas(entry.canvas);
      if (ctx) {
        this.targetCanvases.set(id, { canvas: entry.canvas, context: ctx });
        // Also update the store's context reference
        useRenderTargetStore.getState().setTargetCanvas(id, entry.canvas, ctx);
      }
    }

    this.renderLoop?.start();
    this.requestRender();
    log.info('Recovery complete');
  }

  // === CANVAS MANAGEMENT (Unified) ===

  setPreviewCanvas(canvas: HTMLCanvasElement): void {
    this.mainPreviewCanvas = canvas;
    this.previewContext = this.context.configureCanvas(canvas);
  }

  /**
   * Register a canvas as a render target. Configures WebGPU context and stores in unified map.
   * Returns the GPU context or null on failure.
   */
  registerTargetCanvas(targetId: string, canvas: HTMLCanvasElement): GPUCanvasContext | null {
    const ctx = this.context.configureCanvas(canvas);
    if (ctx) {
      this.targetCanvases.set(targetId, { canvas, context: ctx });
      log.debug('Registered target canvas', { targetId });
      return ctx;
    }
    return null;
  }

  /** Remove a canvas from the unified target map */
  unregisterTargetCanvas(targetId: string): void {
    this.targetCanvases.delete(targetId);
    log.debug('Unregistered target canvas', { targetId });
  }

  /** Lookup GPU context for a target */
  getTargetContext(targetId: string): GPUCanvasContext | null {
    return this.targetCanvases.get(targetId)?.context ?? null;
  }

  // === OUTPUT WINDOWS ===

  /**
   * Create an output window, register it as a render target, and configure WebGPU.
   * The window will automatically receive frames based on its source (default: activeComp).
   */
  createOutputWindow(id: string, name: string): { id: string; name: string } | null {
    if (!this.outputWindowManager) return null;

    const result = this.outputWindowManager.createWindow(id, name);
    if (!result) return null;

    // Register canvas with engine (creates WebGPU context)
    const gpuContext = this.registerTargetCanvas(id, result.canvas);
    if (!gpuContext) {
      result.window.close();
      return null;
    }

    // Register as render target in store (default source: activeComp)
    useRenderTargetStore.getState().registerTarget({
      id,
      name,
      source: { type: 'activeComp' },
      destinationType: 'window',
      enabled: true,
      showTransparencyGrid: false,
      canvas: result.canvas,
      context: gpuContext,
      window: result.window,
      isFullscreen: false,
    });

    return { id, name };
  }

  closeOutputWindow(id: string): void {
    const target = useRenderTargetStore.getState().targets.get(id);
    if (target?.window && !target.window.closed) {
      target.window.close();
    }
    this.unregisterTargetCanvas(id);
    useRenderTargetStore.getState().deactivateTarget(id);
  }

  restoreOutputWindow(id: string): boolean {
    if (!this.outputWindowManager) return false;

    const target = useRenderTargetStore.getState().targets.get(id);
    if (!target || target.destinationType !== 'window') return false;

    // Look up saved geometry from localStorage
    const savedTargets = getSavedTargetMeta();
    const savedMeta = savedTargets.find((t) => t.id === id);
    const geometry = savedMeta ? {
      screenX: savedMeta.screenX,
      screenY: savedMeta.screenY,
      outerWidth: savedMeta.outerWidth,
      outerHeight: savedMeta.outerHeight,
    } : undefined;

    const result = this.outputWindowManager.createWindow(id, target.name, geometry);
    if (!result) return false;

    const gpuContext = this.registerTargetCanvas(id, result.canvas);
    if (!gpuContext) {
      result.window.close();
      return false;
    }

    // Update the existing store entry with new runtime refs
    const store = useRenderTargetStore.getState();
    store.setTargetCanvas(id, result.canvas, gpuContext);
    store.setTargetWindow(id, result.window);
    store.setTargetEnabled(id, true);

    // Restore fullscreen if it was previously fullscreen
    if (savedMeta?.isFullscreen || target.isFullscreen) {
      result.canvas.requestFullscreen().catch(() => {});
    }

    return true;
  }

  removeOutputTarget(id: string): void {
    this.unregisterTargetCanvas(id);
    useRenderTargetStore.getState().unregisterTarget(id);
  }

  /**
   * After page refresh, try to reconnect to existing output windows by name.
   * Takes an array of {id, name, source} from saved metadata.
   */
  reconnectOutputWindows(savedTargets: Array<{ id: string; name: string; source: import('../types/renderTarget').RenderSource }>): number {
    if (!this.outputWindowManager) return 0;

    let reconnected = 0;
    for (const saved of savedTargets) {
      const result = this.outputWindowManager.reconnectWindow(saved.id);
      if (!result) continue;

      // Re-register canvas with WebGPU
      const gpuContext = this.registerTargetCanvas(saved.id, result.canvas);
      if (!gpuContext) continue;

      // Register as render target
      useRenderTargetStore.getState().registerTarget({
        id: saved.id,
        name: saved.name,
        source: saved.source,
        destinationType: 'window',
        enabled: true,
        showTransparencyGrid: false,
        canvas: result.canvas,
        context: gpuContext,
        window: result.window,
        isFullscreen: false,
      });

      reconnected++;
    }

    return reconnected;
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
    this.cacheManager.cleanupVideoCache(video);
    this.videoFrameManager.cleanupVideo(video);
    if (video.src) {
      // Release video element resources to free memory
      video.pause();
      video.removeAttribute('src');
      video.load(); // Forces release of media resources
    }
    log.debug('Cleaned up video resources');
  }

  setHasActiveVideo(hasVideo: boolean): void {
    this.renderLoop?.setHasActiveVideo(hasVideo);
  }

  setIsPlaying(playing: boolean): void {
    if (playing) this.hasEverPlayed = true;
    this.renderLoop?.setIsPlaying(playing);
  }

  setIsScrubbing(scrubbing: boolean): void {
    this.renderLoop?.setIsScrubbing(scrubbing);
  }

  // Called by RVFC when a new decoded frame is ready - bypasses scrub rate limiter
  requestNewFrameRender(): void {
    this.renderLoop?.requestNewFrameRender();
  }

  // === TEXTURE MANAGEMENT ===

  createImageTexture(image: HTMLImageElement): GPUTexture | null {
    return this.textureManager?.createImageTexture(image) ?? null;
  }

  importVideoTexture(source: HTMLVideoElement | VideoFrame): GPUExternalTexture | null {
    return this.textureManager?.importVideoTexture(source) ?? null;
  }

  // === CACHING (delegated to CacheManager) ===

  clearCaches(): void {
    this.cacheManager.clearAll();
    this.textureManager?.clearCaches();
  }

  clearVideoCache(): void {
    this.cacheManager.clearVideoTimeTracking();
  }

  cacheFrameAtTime(video: HTMLVideoElement, time: number): void {
    this.cacheManager.cacheFrameAtTime(video, time);
  }

  getCachedFrame(videoSrc: string, time: number): GPUTextureView | null {
    return this.cacheManager.getCachedFrame(videoSrc, time);
  }

  getScrubbingCacheStats(): { count: number; maxCount: number } {
    return this.cacheManager.getScrubbingCacheStats();
  }

  clearScrubbingCache(videoSrc?: string): void {
    this.cacheManager.clearScrubbingCache(videoSrc);
  }

  // === RAM PREVIEW CACHE ===

  async cacheCompositeFrame(time: number): Promise<void> {
    const getResolution = () => this.renderTargetManager?.getResolution() ?? { width: 640, height: 360 };
    await this.cacheManager.cacheCompositeFrame(time, () => this.readPixels(), getResolution);
  }

  getCachedCompositeFrame(time: number): ImageData | null {
    return this.cacheManager.getCachedCompositeFrame(time);
  }

  hasCompositeCacheFrame(time: number): boolean {
    return this.cacheManager.hasCompositeCacheFrame(time);
  }

  clearCompositeCache(): void {
    this.cacheManager.clearCompositeCache();
  }

  getCompositeCacheStats(): { count: number; maxFrames: number; memoryMB: number } {
    const getResolution = () => this.renderTargetManager?.getResolution() ?? { width: 640, height: 360 };
    return this.cacheManager.getCompositeCacheStats(getResolution);
  }

  setGeneratingRamPreview(generating: boolean): void {
    this.exportCanvasManager.setGeneratingRamPreview(generating);
  }

  setExporting(exporting: boolean): void {
    this.exportCanvasManager.setExporting(exporting);
    if (exporting) this.cacheManager.clearVideoTimeTracking();
  }

  getIsExporting(): boolean {
    return this.exportCanvasManager.getIsExporting();
  }

  initExportCanvas(width: number, height: number): boolean {
    const device = this.context.getDevice();
    if (!device) {
      log.error('Cannot init export canvas: no device');
      return false;
    }
    return this.exportCanvasManager.initExportCanvas(device, width, height);
  }

  async createVideoFrameFromExport(timestamp: number, duration: number): Promise<VideoFrame | null> {
    const device = this.context.getDevice();
    if (!device) return null;
    return this.exportCanvasManager.createVideoFrameFromExport(device, timestamp, duration);
  }

  cleanupExportCanvas(): void {
    this.exportCanvasManager.cleanupExportCanvas();
  }

  // === RENDER LOOP ===

  requestRender(): void {
    this.renderLoop?.requestRender();
  }

  getIsIdle(): boolean {
    return this.renderLoop?.getIsIdle() ?? false;
  }

  /**
   * Ensure the scrubbing cache has at least one frame for this video.
   * Called before seeking to provide a fallback frame during seek.
   */
  ensureVideoFrameCached(video: HTMLVideoElement): void {
    this.cacheManager.ensureVideoFrameCached(video);
  }

  /**
   * Pre-cache a video frame using createImageBitmap (async forced decode).
   * This is the ONLY way to get a real frame from a never-played video after reload.
   * Call from canplaythrough handlers during project restore.
   */
  async preCacheVideoFrame(video: HTMLVideoElement): Promise<boolean> {
    const success = await this.cacheManager.preCacheVideoFrame(video);
    if (success) {
      this.requestRender();
    }
    return success;
  }

  updatePlayheadTracking(playhead: number): boolean {
    return this.renderLoop?.updatePlayheadTracking(playhead) ?? false;
  }

  start(renderCallback: () => void): void {
    if (!this.performanceStats) return;

    // Stop any existing loop first to prevent multiple RAF loops accumulating
    this.renderLoop?.stop();

    // Create new loop with the callback
    this.renderLoop = new RenderLoop(this.performanceStats, {
      isRecovering: () => this.isRecoveringFromDeviceLoss || this.context.recovering,
      isExporting: () => this.exportCanvasManager.getIsExporting(),
      onRender: renderCallback,
    });

    // Suppress idle until user presses play for the first time.
    // After page reload, video GPU surfaces are empty and need the render loop
    // running continuously so syncClipVideo warmup can complete.
    if (!this.hasEverPlayed) {
      this.renderLoop.suppressIdle();
    }

    this.renderLoop.start();
  }

  stop(): void {
    this.renderLoop?.stop();
  }

  // === MAIN RENDER (delegated to RenderDispatcher) ===

  render(layers: Layer[]): void {
    this.renderDispatcher?.render(layers);
  }

  renderToPreviewCanvas(canvasId: string, layers: Layer[]): void {
    this.renderDispatcher?.renderToPreviewCanvas(canvasId, layers);
  }

  renderCachedFrame(time: number): boolean {
    return this.renderDispatcher?.renderCachedFrame(time) ?? false;
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
    const canvasContext = this.targetCanvases.get(canvasId)?.context;
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
    const canvasContext = this.targetCanvases.get(canvasId)?.context;
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

  /**
   * Render sliced output to a specific canvas using the main composited output.
   * Used by TargetPreview to preview sliced output for a target.
   */
  renderSlicedToCanvas(canvasId: string, slices: import('../types/outputSlice').OutputSlice[]): boolean {
    const device = this.context.getDevice();
    const canvasContext = this.targetCanvases.get(canvasId)?.context;
    const pingView = this.renderTargetManager?.getPingView();
    const pongView = this.renderTargetManager?.getPongView();

    if (!device || !canvasContext || !this.slicePipeline || !this.sampler || !pingView || !pongView) return false;

    const enabledSlices = slices.filter((s) => s.enabled);
    if (enabledSlices.length === 0) return false;

    const finalIsPing = !this.compositor?.getLastRenderWasPing();
    const finalView = finalIsPing ? pingView : pongView;

    this.slicePipeline.buildVertexBuffer(enabledSlices);

    const commandEncoder = device.createCommandEncoder();
    this.slicePipeline.renderSlicedOutput(commandEncoder, canvasContext, finalView, this.sampler);
    device.queue.submit([commandEncoder.finish()]);
    return true;
  }

  // === RESOLUTION ===

  setResolution(width: number, height: number): void {
    if (this.renderTargetManager?.setResolution(width, height)) {
      this.cacheManager.clearCompositeCache();
      this.cacheManager.clearScrubbingCache();
      this.outputWindowManager?.updateResolution(width, height);
      this.outputPipeline?.invalidateCache();
      this.compositorPipeline?.invalidateBindGroupCache();
      log.debug('Caches cleared for resolution change', { width, height });
    }
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
    this.outputPipeline?.updateResolution(width, height);
    if (this.outputPipeline && this.sampler) {
      if (this.previewContext) {
        const mainBindGroup = this.outputPipeline.createOutputBindGroup(this.sampler, pingView, false);
        this.outputPipeline.renderToCanvas(commandEncoder, this.previewContext, mainBindGroup);
      }
      const activeTargets = useRenderTargetStore.getState().getActiveCompTargets();
      for (const target of activeTargets) {
        const ctx = this.targetCanvases.get(target.id)?.context;
        if (!ctx) continue;
        const targetBindGroup = this.outputPipeline.createOutputBindGroup(this.sampler, pingView, target.showTransparencyGrid);
        this.outputPipeline.renderToCanvas(commandEncoder, ctx, targetBindGroup);
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

  getLastRenderedTexture(): GPUTexture | null {
    if (!this.renderTargetManager || !this.compositor) return null;
    if (!this.renderDispatcher?.lastRenderHadContent) return null;
    return this.compositor.getLastRenderWasPing()
      ? this.renderTargetManager.getPingTexture()
      : this.renderTargetManager.getPongTexture();
  }

  getTextureManager(): TextureManager | null {
    return this.textureManager;
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
    // Reconfigure all target canvases
    for (const [id, entry] of this.targetCanvases) {
      const ctx = this.context.configureCanvas(entry.canvas);
      if (ctx) {
        this.targetCanvases.set(id, { canvas: entry.canvas, context: ctx });
        useRenderTargetStore.getState().setTargetCanvas(id, entry.canvas, ctx);
      }
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
    this.cacheManager.destroy();
    this.exportCanvasManager.destroy();
    this.videoFrameManager.destroy();
    this.compositorPipeline?.destroy();
    this.effectsPipeline?.destroy();
    this.outputPipeline?.destroy();
    this.slicePipeline?.destroy();
    this.context.destroy();
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
