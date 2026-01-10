// WebGPU Rendering Engine for WebVJ Mixer - Facade Pattern
// This is the main engine class that orchestrates all the manager modules

import type { Layer, OutputWindow, EngineStats, LayerRenderData, DetailedStats, ProfileData } from './core/types';
import { WebGPUContext } from './core/WebGPUContext';
import { TextureManager } from './texture/TextureManager';
import { MaskTextureManager } from './texture/MaskTextureManager';
import { ScrubbingCache } from './texture/ScrubbingCache';
import { CompositorPipeline } from './pipeline/CompositorPipeline';
import { EffectsPipeline } from './pipeline/EffectsPipeline';
import { OutputPipeline } from './pipeline/OutputPipeline';
import { VideoFrameManager } from './video/VideoFrameManager';
import { audioStatusTracker } from '../services/audioManager';

export class WebGPUEngine {
  // Core context
  private context: WebGPUContext;

  // Managers
  private textureManager: TextureManager | null = null;
  private maskTextureManager: MaskTextureManager | null = null;
  private scrubbingCache: ScrubbingCache | null = null;
  private videoFrameManager: VideoFrameManager;

  // Pipelines
  private compositorPipeline: CompositorPipeline | null = null;
  private effectsPipeline: EffectsPipeline | null = null;
  private outputPipeline: OutputPipeline | null = null;

  // Main preview canvas (legacy - kept for backward compatibility)
  private previewContext: GPUCanvasContext | null = null;

  // Multiple preview canvases (inline previews in dock panels)
  // These are rendered by the MAIN render loop (active composition)
  private previewCanvases: Map<string, GPUCanvasContext> = new Map();

  // Independent preview canvases - NOT rendered by main loop
  // These have their own render loop for different compositions
  private independentPreviewCanvases: Map<string, GPUCanvasContext> = new Map();

  // Render targets - ping pong buffers (main render loop)
  private pingTexture: GPUTexture | null = null;
  private pongTexture: GPUTexture | null = null;
  private blackTexture: GPUTexture | null = null;

  // Cached texture views (main render loop)
  private pingView: GPUTextureView | null = null;
  private pongView: GPUTextureView | null = null;

  // Separate ping-pong buffers for independent preview rendering
  // This prevents flickering when main loop and independent previews run simultaneously
  private independentPingTexture: GPUTexture | null = null;
  private independentPongTexture: GPUTexture | null = null;
  private independentPingView: GPUTextureView | null = null;
  private independentPongView: GPUTextureView | null = null;

  // Resources
  private sampler: GPUSampler | null = null;

  // Output windows
  private outputWindows: Map<string, OutputWindow> = new Map();

  // Stats
  private frameCount = 0;
  private fps = 0;
  private fpsUpdateTime = 0;

  // Detailed stats tracking
  private detailedStats: DetailedStats = {
    rafGap: 0,
    importTexture: 0,
    renderPass: 0,
    submit: 0,
    total: 0,
    dropsTotal: 0,
    dropsLastSecond: 0,
    dropsThisSecond: 0,
    lastDropReason: 'none',
    lastRafTime: 0,
    decoder: 'none',
  };
  private readonly TARGET_FRAME_TIME = 16.67; // 60fps target

  // Animation
  private animationId: number | null = null;
  private isRunning = false;

  // Performance profiling
  private profileData: ProfileData = {
    importTexture: 0,
    createBindGroup: 0,
    renderPass: 0,
    submit: 0,
    total: 0,
  };
  private profileCounter = 0;
  private lastProfileTime = 0;

  // Resolution
  private outputWidth = 1920;
  private outputHeight = 1080;

  // Ring buffer for frame times (avoids O(n) shift)
  private frameTimeBuffer = new Float32Array(60);
  private frameTimeIndex = 0;
  private frameTimeCount = 0;

  // Track which texture has the final composited frame (for export)
  private lastRenderWasPing = false;

  // Reusable layer data array to avoid allocations
  private layerRenderData: LayerRenderData[] = [];

  // Track active video for frame rate limiting
  private hasActiveVideo = false;
  private lastRenderTime = 0;
  private readonly VIDEO_FRAME_TIME = 33.33; // ~30fps when video is playing

  // Performance tracking
  private lastFrameStart = 0;
  private statsCounter = 0;
  private lastLayerCount = 0;

  // Flag to skip preview updates during RAM preview generation
  private isGeneratingRamPreview = false;

  // Reusable resources for RAM Preview playback
  private ramPlaybackCanvas: HTMLCanvasElement | null = null;
  private ramPlaybackCtx: CanvasRenderingContext2D | null = null;

  // Track video currentTime to skip texture import for unchanged frames
  private lastVideoTime: Map<string, number> = new Map();
  private cachedExternalTexture: Map<string, GPUExternalTexture> = new Map();

  // Batched uniform buffer updates
  private pendingUniformUpdates: Array<{buffer: GPUBuffer; data: Float32Array}> = [];

  // Nested composition pre-render textures
  // Map of compositionId -> {texture, view} for pre-rendered nested compositions
  private nestedCompTextures: Map<string, { texture: GPUTexture; view: GPUTextureView }> = new Map();

  constructor() {
    this.context = new WebGPUContext();
    this.videoFrameManager = new VideoFrameManager();
  }

  async initialize(): Promise<boolean> {
    const success = await this.context.initialize();
    if (!success) return false;

    await this.createResources();
    console.log('[WebGPU] Engine initialized successfully');
    return true;
  }

  private async createResources(): Promise<void> {
    const device = this.context.getDevice();
    if (!device) return;

    // Initialize managers
    this.textureManager = new TextureManager(device);
    this.maskTextureManager = new MaskTextureManager(device);
    this.scrubbingCache = new ScrubbingCache(device);

    // Initialize pipelines
    this.compositorPipeline = new CompositorPipeline(device);
    this.effectsPipeline = new EffectsPipeline(device);
    this.outputPipeline = new OutputPipeline(device);

    // Create sampler
    this.sampler = this.context.createSampler();

    // Create black texture
    this.blackTexture = this.context.createSolidColorTexture(0, 0, 0, 255);

    this.createPingPongTextures();
    await this.createPipelines();
  }

  private createPingPongTextures(): void {
    const device = this.context.getDevice();
    if (!device) return;

    // Destroy existing textures
    this.pingTexture?.destroy();
    this.pongTexture?.destroy();
    this.independentPingTexture?.destroy();
    this.independentPongTexture?.destroy();

    // Main render loop ping-pong buffers
    this.pingTexture = device.createTexture({
      size: [this.outputWidth, this.outputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    this.pongTexture = device.createTexture({
      size: [this.outputWidth, this.outputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    // Independent preview ping-pong buffers (prevents flickering)
    this.independentPingTexture = device.createTexture({
      size: [this.outputWidth, this.outputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    this.independentPongTexture = device.createTexture({
      size: [this.outputWidth, this.outputHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    // Cache views
    this.pingView = this.pingTexture.createView();
    this.pongView = this.pongTexture.createView();
    this.independentPingView = this.independentPingTexture.createView();
    this.independentPongView = this.independentPongTexture.createView();

    // Invalidate bind group caches (textures changed)
    this.outputPipeline?.invalidateCache();
    this.compositorPipeline?.invalidateBindGroupCache();
  }

  private async createPipelines(): Promise<void> {
    await this.compositorPipeline?.createPipelines();
    await this.effectsPipeline?.createPipelines();
    await this.outputPipeline?.createPipeline();
  }

  setPreviewCanvas(canvas: HTMLCanvasElement): void {
    this.previewContext = this.context.configureCanvas(canvas);
  }

  // === MULTIPLE PREVIEW CANVAS MANAGEMENT ===

  registerPreviewCanvas(id: string, canvas: HTMLCanvasElement): void {
    const context = this.context.configureCanvas(canvas);
    if (context) {
      this.previewCanvases.set(id, context);
      console.log(`[Engine] Registered preview canvas: ${id}`);
    }
  }

  unregisterPreviewCanvas(id: string): void {
    this.previewCanvases.delete(id);
    console.log(`[Engine] Unregistered preview canvas: ${id}`);
  }

  // Register canvas for independent composition rendering (NOT rendered by main loop)
  registerIndependentPreviewCanvas(id: string, canvas: HTMLCanvasElement): void {
    const context = this.context.configureCanvas(canvas);
    if (context) {
      this.independentPreviewCanvases.set(id, context);
      console.log(`[Engine] Registered INDEPENDENT preview canvas: ${id}`);
    }
  }

  unregisterIndependentPreviewCanvas(id: string): void {
    this.independentPreviewCanvases.delete(id);
    console.log(`[Engine] Unregistered INDEPENDENT preview canvas: ${id}`);
  }

  // === MASK TEXTURE MANAGEMENT ===

  updateMaskTexture(layerId: string, imageData: ImageData | null): void {
    this.maskTextureManager?.updateMaskTexture(layerId, imageData);
  }

  removeMaskTexture(layerId: string): void {
    this.maskTextureManager?.removeMaskTexture(layerId);
  }

  hasMaskTexture(layerId: string): boolean {
    return this.maskTextureManager?.hasMaskTexture(layerId) ?? false;
  }

  // === OUTPUT WINDOW MANAGEMENT ===

  createOutputWindow(id: string, name: string): OutputWindow | null {
    const device = this.context.getDevice();
    const outputWindow = window.open(
      '',
      `output_${id}`,
      'width=960,height=540,menubar=no,toolbar=no,location=no,status=no'
    );

    if (!outputWindow) {
      console.error('Failed to open output window (popup blocked?)');
      return null;
    }

    outputWindow.document.title = `WebVJ Output - ${name}`;
    outputWindow.document.body.style.cssText =
      'margin:0;padding:0;background:#000;overflow:hidden;width:100vw;height:100vh;';

    const canvas = outputWindow.document.createElement('canvas');
    canvas.width = this.outputWidth;
    canvas.height = this.outputHeight;
    canvas.style.cssText = 'display:block;background:#000;';
    outputWindow.document.body.appendChild(canvas);

    // Aspect ratio locking
    const aspectRatio = this.outputWidth / this.outputHeight;
    let lastWidth = outputWindow.innerWidth;
    let lastHeight = outputWindow.innerHeight;
    let resizing = false;

    const enforceAspectRatio = () => {
      if (resizing) return;
      resizing = true;

      const currentWidth = outputWindow.innerWidth;
      const currentHeight = outputWindow.innerHeight;
      const widthDelta = Math.abs(currentWidth - lastWidth);
      const heightDelta = Math.abs(currentHeight - lastHeight);

      let newWidth: number;
      let newHeight: number;

      if (widthDelta >= heightDelta) {
        newWidth = currentWidth;
        newHeight = Math.round(currentWidth / aspectRatio);
      } else {
        newHeight = currentHeight;
        newWidth = Math.round(currentHeight * aspectRatio);
      }

      if (newWidth !== currentWidth || newHeight !== currentHeight) {
        outputWindow.resizeTo(newWidth + (outputWindow.outerWidth - currentWidth),
                              newHeight + (outputWindow.outerHeight - currentHeight));
      }

      canvas.style.width = '100%';
      canvas.style.height = '100%';

      lastWidth = newWidth;
      lastHeight = newHeight;

      setTimeout(() => { resizing = false; }, 50);
    };

    canvas.style.width = '100%';
    canvas.style.height = '100%';
    outputWindow.addEventListener('resize', enforceAspectRatio);

    let context: GPUCanvasContext | null = null;

    if (device) {
      context = canvas.getContext('webgpu');
      if (context) {
        context.configure({
          device,
          format: 'bgra8unorm',
          alphaMode: 'premultiplied',
        });
      }
    }

    const fullscreenBtn = outputWindow.document.createElement('button');
    fullscreenBtn.textContent = 'Fullscreen';
    fullscreenBtn.style.cssText =
      'position:fixed;top:10px;right:10px;padding:8px 16px;cursor:pointer;z-index:1000;opacity:0.7;';
    fullscreenBtn.onclick = () => {
      canvas.requestFullscreen();
    };
    outputWindow.document.body.appendChild(fullscreenBtn);

    outputWindow.document.addEventListener('fullscreenchange', () => {
      fullscreenBtn.style.display = outputWindow.document.fullscreenElement ? 'none' : 'block';
    });

    outputWindow.onbeforeunload = () => {
      this.outputWindows.delete(id);
    };

    const output: OutputWindow = {
      id,
      name,
      window: outputWindow,
      canvas,
      context,
      isFullscreen: false,
    };

    this.outputWindows.set(id, output);
    return output;
  }

  closeOutputWindow(id: string): void {
    const output = this.outputWindows.get(id);
    if (output?.window) {
      output.window.close();
    }
    this.outputWindows.delete(id);
  }

  // === TEXTURE MANAGEMENT ===

  createImageTexture(image: HTMLImageElement): GPUTexture | null {
    return this.textureManager?.createImageTexture(image) ?? null;
  }

  importVideoTexture(source: HTMLVideoElement | VideoFrame): GPUExternalTexture | null {
    return this.textureManager?.importVideoTexture(source) ?? null;
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
    // Clear video time tracking
    const videoKey = video.src;
    if (videoKey) {
      this.lastVideoTime.delete(videoKey);
      this.cachedExternalTexture.delete(videoKey);
    }
    console.log('[WebGPU] Cleaned up video resources');
  }

  setHasActiveVideo(hasVideo: boolean): void {
    this.hasActiveVideo = hasVideo;
  }

  // === CACHING ===

  clearCaches(): void {
    this.scrubbingCache?.clearAll();
    this.textureManager?.clearCaches();
    console.log('[WebGPU] Cleared all caches');
  }

  cacheFrameAtTime(video: HTMLVideoElement, time: number): void {
    this.scrubbingCache?.cacheFrameAtTime(video, time);
  }

  getCachedFrame(videoSrc: string, time: number): GPUTextureView | null {
    return this.scrubbingCache?.getCachedFrame(videoSrc, time) ?? null;
  }

  preCacheFrames(_video: HTMLVideoElement, _centerTime: number, _radiusSeconds: number = 2, _fps: number = 30): void {
    // Implemented in ScrubbingCache but not exposed - for future use
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

    // Debug: check if pixels have data
    if (this.scrubbingCache.getCompositeCacheStats(this.outputWidth, this.outputHeight).count === 0) {
      let nonZero = 0;
      for (let i = 0; i < Math.min(1000, pixels.length); i++) {
        if (pixels[i] !== 0) nonZero++;
      }
      console.log(`[RAM Preview] First frame: ${nonZero} non-zero pixels in first 1000, size: ${this.outputWidth}x${this.outputHeight}`);
    }

    const imageData = new ImageData(
      new Uint8ClampedArray(pixels),
      this.outputWidth,
      this.outputHeight
    );

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
    console.log('[WebGPU] Composite cache cleared');
  }

  getCompositeCacheStats(): { count: number; maxFrames: number; memoryMB: number } {
    return this.scrubbingCache?.getCompositeCacheStats(this.outputWidth, this.outputHeight) ??
      { count: 0, maxFrames: 0, memoryMB: 0 };
  }

  setGeneratingRamPreview(generating: boolean): void {
    this.isGeneratingRamPreview = generating;
  }

  renderCachedFrame(time: number): boolean {
    const device = this.context.getDevice();
    if (!this.previewContext || !device || !this.scrubbingCache || !this.outputPipeline || !this.sampler) {
      return false;
    }

    // First, check GPU cache for instant playback
    const gpuCached = this.scrubbingCache.getGpuCachedFrame(time);
    if (gpuCached) {
      const commandEncoder = device.createCommandEncoder();
      this.outputPipeline.renderToCanvas(commandEncoder, this.previewContext, gpuCached.bindGroup);
      // Render to all inline preview canvases
      for (const previewCtx of this.previewCanvases.values()) {
        this.outputPipeline.renderToCanvas(commandEncoder, previewCtx, gpuCached.bindGroup);
      }
      for (const output of this.outputWindows.values()) {
        if (output.context) {
          this.outputPipeline.renderToCanvas(commandEncoder, output.context, gpuCached.bindGroup);
        }
      }
      device.queue.submit([commandEncoder.finish()]);
      return true;
    }

    // Fall back to CPU cache and upload to GPU
    const imageData = this.scrubbingCache.getCachedCompositeFrame(time);
    if (!imageData) {
      return false;
    }

    try {
      const width = imageData.width;
      const height = imageData.height;

      // Reuse or create canvas for ImageData -> GPU transfer
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

      // Create a new GPU texture for this frame
      const texture = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      device.queue.copyExternalImageToTexture(
        { source: this.ramPlaybackCanvas },
        { texture },
        [width, height]
      );

      const view = texture.createView();
      const bindGroup = this.outputPipeline.createOutputBindGroup(this.sampler, view);

      // Add to GPU cache
      this.scrubbingCache.addToGpuCache(time, { texture, view, bindGroup });

      // Render to preview
      const commandEncoder = device.createCommandEncoder();
      this.outputPipeline.renderToCanvas(commandEncoder, this.previewContext, bindGroup);

      // Render to all inline preview canvases
      for (const previewCtx of this.previewCanvases.values()) {
        this.outputPipeline.renderToCanvas(commandEncoder, previewCtx, bindGroup);
      }

      for (const output of this.outputWindows.values()) {
        if (output.context) {
          this.outputPipeline.renderToCanvas(commandEncoder, output.context, bindGroup);
        }
      }

      device.queue.submit([commandEncoder.finish()]);
      return true;
    } catch (e) {
      console.warn('[WebGPU] Failed to render cached frame:', e);
      return false;
    }
  }

  // === MAIN RENDER ===

  render(layers: Layer[]): void {
    const device = this.context.getDevice();
    if (!device || !this.compositorPipeline || !this.outputPipeline || !this.sampler) return;
    if (!this.pingView || !this.pongView) return;

    const t0 = performance.now();

    // Reuse array, just clear it
    this.layerRenderData.length = 0;

    // Prepare layer data - import textures (reverse order: lower slots render on top)
    const t1 = performance.now();
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      // Skip invisible layers and layers with zero opacity (saves GPU overhead)
      if (!layer || !layer.visible || !layer.source || layer.opacity === 0) continue;

      // Try WebCodecs VideoFrame first (if available)
      if (layer.source.webCodecsPlayer) {
        const frame = layer.source.webCodecsPlayer.getCurrentFrame();
        if (frame) {
          const extTex = this.textureManager?.importVideoTexture(frame);
          if (extTex) {
            this.detailedStats.decoder = 'WebCodecs';
            this.layerRenderData.push({
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: frame.displayWidth,
              sourceHeight: frame.displayHeight,
            });
            continue;
          }
        }
      }

      // HTMLVideoElement - optimized with frame tracking
      if (layer.source.videoElement) {
        const video = layer.source.videoElement;
        const videoKey = video.src || layer.id;

        if (video.readyState >= 2) {
          // Check if video time has changed since last frame (optimization for paused videos)
          const lastTime = this.lastVideoTime.get(videoKey);
          const currentTime = video.currentTime;
          const videoTimeChanged = lastTime === undefined || Math.abs(currentTime - lastTime) > 0.001;

          // OPTIMIZATION: If video time hasn't changed, use cached frame instead of re-importing
          // This saves significant GPU overhead for paused videos
          if (!videoTimeChanged) {
            const lastFrame = this.scrubbingCache?.getLastFrame(video);
            if (lastFrame) {
              this.detailedStats.decoder = 'HTMLVideo(paused-cache)';
              this.layerRenderData.push({
                layer,
                isVideo: false,
                externalTexture: null,
                textureView: lastFrame.view,
                sourceWidth: lastFrame.width,
                sourceHeight: lastFrame.height,
              });
              continue;
            }
          }

          // Video time changed or no cache - import external texture
          const extTex = this.textureManager?.importVideoTexture(video);
          if (extTex) {
            // Update time tracking
            this.lastVideoTime.set(videoKey, currentTime);

            // Cache frame for pause/seek fallback
            const now = performance.now();
            const lastCapture = this.scrubbingCache?.getLastCaptureTime(video) || 0;
            if (now - lastCapture > 200) { // Cache more frequently for smoother pause
              this.scrubbingCache?.captureVideoFrame(video);
              this.scrubbingCache?.setLastCaptureTime(video, now);
            }
            this.detailedStats.decoder = 'HTMLVideo';

            this.layerRenderData.push({
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: video.videoWidth,
              sourceHeight: video.videoHeight,
            });
            continue;
          }

          // Import failed - use cached frame
          const lastFrame = this.scrubbingCache?.getLastFrame(video);
          if (lastFrame) {
            this.detailedStats.decoder = 'HTMLVideo(cached)';
            this.layerRenderData.push({
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: lastFrame.view,
              sourceWidth: lastFrame.width,
              sourceHeight: lastFrame.height,
            });
            continue;
          }
        } else {
          // Video not ready - try last frame cache
          const lastFrame = this.scrubbingCache?.getLastFrame(video);
          if (lastFrame) {
            this.layerRenderData.push({
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: lastFrame.view,
              sourceWidth: lastFrame.width,
              sourceHeight: lastFrame.height,
            });
            continue;
          }
        }
      }

      // Images
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = this.textureManager?.getCachedImageTexture(img);
        if (!texture) {
          texture = this.textureManager?.createImageTexture(img) ?? undefined;
        }
        if (texture) {
          const imageView = this.textureManager!.getImageView(texture);
          this.layerRenderData.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: imageView,
            sourceWidth: img.naturalWidth,
            sourceHeight: img.naturalHeight,
          });
          continue;
        }
      }

      // Nested compositions - will be pre-rendered below
      if (layer.source.nestedComposition) {
        const nestedComp = layer.source.nestedComposition;
        // Mark for pre-rendering - actual texture will be created in pre-render pass
        this.layerRenderData.push({
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: null, // Will be set after pre-render
          sourceWidth: nestedComp.width,
          sourceHeight: nestedComp.height,
        });
      }
    }

    // Pre-render nested compositions (must happen before main compositing)
    // Need to create command encoder early for pre-rendering
    const preRenderCommandEncoder = device.createCommandEncoder();
    for (const data of this.layerRenderData) {
      if (data.layer.source?.nestedComposition) {
        const nestedComp = data.layer.source.nestedComposition;
        const preRenderedView = this.preRenderNestedComposition(
          nestedComp.compositionId,
          nestedComp.layers,
          nestedComp.width,
          nestedComp.height,
          preRenderCommandEncoder
        );
        if (preRenderedView) {
          data.textureView = preRenderedView;
        }
      }
    }
    // Submit pre-render commands
    device.queue.submit([preRenderCommandEncoder.finish()]);

    this.profileData.importTexture = performance.now() - t1;

    // Update video flag for frame rate limiting
    this.hasActiveVideo = this.layerRenderData.some(d => d.isVideo);

    // Early exit if nothing to render - clear all canvases to black
    if (this.layerRenderData.length === 0) {
      const commandEncoder = device.createCommandEncoder();

      // Clear main preview context
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

      // Clear all registered preview canvases (from Preview components)
      for (const previewCtx of this.previewCanvases.values()) {
        const pass = commandEncoder.beginRenderPass({
          colorAttachments: [{
            view: previewCtx.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        pass.end();
      }

      device.queue.submit([commandEncoder.finish()]);
      this.lastLayerCount = 0;
      return;
    }

    const t2 = performance.now();
    const commandEncoder = device.createCommandEncoder();

    // Ping-pong compositing
    let readView = this.pingView;
    let writeView = this.pongView;
    let usePing = true;

    // Clear first buffer
    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: readView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    clearPass.end();

    // Composite each visible layer
    for (let i = 0; i < this.layerRenderData.length; i++) {
      const data = this.layerRenderData[i];
      const layer = data.layer;

      // Get or create per-layer uniform buffer
      const uniformBuffer = this.compositorPipeline!.getOrCreateUniformBuffer(layer.id);

      // Calculate aspect ratios
      const sourceAspect = data.sourceWidth / data.sourceHeight;
      const outputAspect = this.outputWidth / this.outputHeight;

      // Get mask texture view for this layer
      const hasMask = this.maskTextureManager?.hasMaskTexture(layer.id) ?? false;
      const maskTextureView = this.maskTextureManager?.getMaskTextureView(layer.id) ??
                             this.maskTextureManager?.getWhiteMaskView()!;

      // Debug logging for mask state
      this.maskTextureManager?.logMaskState(layer.id, hasMask);

      // Update uniforms
      this.compositorPipeline!.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = this.compositorPipeline!.getExternalCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createExternalCompositeBindGroup(
          this.sampler!,
          readView,
          data.externalTexture,
          uniformBuffer,
          maskTextureView
        );
      } else if (data.textureView) {
        pipeline = this.compositorPipeline!.getCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createCompositeBindGroup(
          this.sampler!,
          readView,
          data.textureView,
          uniformBuffer,
          maskTextureView
        );
      } else {
        continue;
      }

      const compositePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: writeView,
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      compositePass.setPipeline(pipeline);
      compositePass.setBindGroup(0, bindGroup);
      compositePass.draw(6);
      compositePass.end();

      // Apply effects to the layer if any
      if (layer.effects && layer.effects.length > 0 && this.effectsPipeline) {
        const result = this.effectsPipeline.applyEffects(
          commandEncoder,
          layer.effects,
          this.sampler!,
          writeView,
          readView,
          this.pingView!,
          this.pongView!,
          this.outputWidth,
          this.outputHeight
        );

        if (result.swapped) {
          // Adjust for effect swaps
          const tempView = readView;
          readView = writeView;
          writeView = tempView;
          usePing = !usePing;
        }
      }

      // Swap buffers
      const temp = readView;
      readView = writeView;
      writeView = temp;
      usePing = !usePing;
    }
    this.profileData.renderPass = performance.now() - t2;

    this.lastLayerCount = this.layerRenderData.length;
    this.lastRenderWasPing = usePing;

    // Get output bind group
    const finalIsPing = !usePing;
    const outputBindGroup = this.outputPipeline!.getOutputBindGroup(this.sampler!, readView, finalIsPing);

    // Render to preview (skip during RAM preview generation)
    if (this.previewContext && !this.isGeneratingRamPreview) {
      this.outputPipeline!.renderToCanvas(commandEncoder, this.previewContext, outputBindGroup);
    }

    // Render to all inline preview canvases
    if (!this.isGeneratingRamPreview) {
      for (const previewCtx of this.previewCanvases.values()) {
        this.outputPipeline!.renderToCanvas(commandEncoder, previewCtx, outputBindGroup);
      }
    }

    // Render to output windows
    if (!this.isGeneratingRamPreview) {
      for (const output of this.outputWindows.values()) {
        if (output.context) {
          this.outputPipeline!.renderToCanvas(commandEncoder, output.context, outputBindGroup);
        }
      }
    }

    const t3 = performance.now();
    device.queue.submit([commandEncoder.finish()]);
    this.profileData.submit = performance.now() - t3;

    this.profileData.total = performance.now() - t0;

    // Update detailed stats
    this.detailedStats.importTexture = this.detailedStats.importTexture * 0.9 + this.profileData.importTexture * 0.1;
    this.detailedStats.renderPass = this.detailedStats.renderPass * 0.9 + this.profileData.renderPass * 0.1;
    this.detailedStats.submit = this.detailedStats.submit * 0.9 + this.profileData.submit * 0.1;
    this.detailedStats.total = this.detailedStats.total * 0.9 + this.profileData.total * 0.1;

    // Detect drops caused by slow render
    if (this.profileData.total > this.TARGET_FRAME_TIME) {
      if (this.profileData.importTexture > this.TARGET_FRAME_TIME * 0.5) {
        this.detailedStats.lastDropReason = 'slow_import';
      } else {
        this.detailedStats.lastDropReason = 'slow_render';
      }
    }

    // Profile counter for internal use
    this.profileCounter++;
    const now = performance.now();
    if (now - this.lastProfileTime >= 1000) {
      this.profileCounter = 0;
      this.lastProfileTime = now;
    }

    this.updateStats();
  }

  /**
   * Render specific layers to a specific preview canvas
   * Used for multi-composition preview where each preview shows different content
   */
  renderToPreviewCanvas(canvasId: string, layers: Layer[]): void {
    const device = this.context.getDevice();
    // Look in INDEPENDENT canvases map (not rendered by main loop)
    const canvasContext = this.independentPreviewCanvases.get(canvasId);

    if (!device || !canvasContext || !this.compositorPipeline || !this.outputPipeline || !this.sampler) {
      return;
    }
    // Use independent ping-pong buffers to avoid conflicts with main render loop
    if (!this.independentPingView || !this.independentPongView) return;

    // Prepare layer data
    const layerData: LayerRenderData[] = [];

    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer || !layer.visible || !layer.source || layer.opacity === 0) continue;

      // HTMLVideoElement
      if (layer.source.videoElement) {
        const video = layer.source.videoElement;
        if (video.readyState >= 2) {
          const extTex = this.textureManager?.importVideoTexture(video);
          if (extTex) {
            layerData.push({
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: video.videoWidth,
              sourceHeight: video.videoHeight,
            });
            continue;
          }
        }
      }

      // Image
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = this.textureManager?.getCachedImageTexture(img);
        if (!texture) {
          texture = this.textureManager?.createImageTexture(img) ?? undefined;
        }
        if (texture) {
          const imageView = this.textureManager!.getImageView(texture);
          layerData.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: imageView,
            sourceWidth: img.naturalWidth,
            sourceHeight: img.naturalHeight,
          });
        }
      }
    }

    // If no layers, clear to black
    if (layerData.length === 0) {
      const commandEncoder = device.createCommandEncoder();
      if (this.blackTexture) {
        const blackView = this.blackTexture.createView();
        const blackBindGroup = this.outputPipeline.createOutputBindGroup(this.sampler, blackView);
        this.outputPipeline.renderToCanvas(commandEncoder, canvasContext, blackBindGroup);
      }
      device.queue.submit([commandEncoder.finish()]);
      return;
    }

    const commandEncoder = device.createCommandEncoder();

    // Ping-pong compositing using INDEPENDENT buffers (avoids main loop conflicts)
    let readView = this.independentPingView!;
    let writeView = this.independentPongView!;
    let usePing = true;

    // Clear first buffer
    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: readView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    clearPass.end();

    // Composite each layer
    for (let i = 0; i < layerData.length; i++) {
      const data = layerData[i];
      const layer = data.layer;

      // Get or create per-layer uniform buffer
      const uniformBuffer = this.compositorPipeline!.getOrCreateUniformBuffer(layer.id);

      // Calculate aspect ratios
      const sourceAspect = data.sourceWidth / data.sourceHeight;
      const outputAspect = this.outputWidth / this.outputHeight;

      // Get mask texture view for this layer
      const hasMask = this.maskTextureManager?.hasMaskTexture(layer.id) ?? false;
      const maskTextureView = this.maskTextureManager?.getMaskTextureView(layer.id) ??
                             this.maskTextureManager?.getWhiteMaskView()!;

      // Update uniforms
      this.compositorPipeline!.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = this.compositorPipeline!.getExternalCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createExternalCompositeBindGroup(
          this.sampler!,
          readView,
          data.externalTexture,
          uniformBuffer,
          maskTextureView
        );
      } else if (data.textureView) {
        pipeline = this.compositorPipeline!.getCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createCompositeBindGroup(
          this.sampler!,
          readView,
          data.textureView,
          uniformBuffer,
          maskTextureView
        );
      } else {
        continue;
      }

      const compositePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: writeView,
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      compositePass.setPipeline(pipeline);
      compositePass.setBindGroup(0, bindGroup);
      compositePass.draw(6);
      compositePass.end();

      // Swap buffers
      const temp = readView;
      readView = writeView;
      writeView = temp;
      usePing = !usePing;
    }

    // Output to this specific canvas
    const finalIsPing = !usePing;
    const outputBindGroup = this.outputPipeline!.getOutputBindGroup(this.sampler!, readView, finalIsPing);
    this.outputPipeline!.renderToCanvas(commandEncoder, canvasContext, outputBindGroup);

    device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Pre-render a nested composition to an offscreen texture
   * Returns the texture view to be used as the source for the parent layer
   */
  preRenderNestedComposition(
    compositionId: string,
    nestedLayers: Layer[],
    width: number,
    height: number,
    commandEncoder: GPUCommandEncoder
  ): GPUTextureView | null {
    const device = this.context.getDevice();
    if (!device || !this.compositorPipeline || !this.sampler) return null;

    // Get or create offscreen texture for this composition
    let compTexture = this.nestedCompTextures.get(compositionId);
    if (!compTexture || compTexture.texture.width !== width || compTexture.texture.height !== height) {
      // Clean up old texture if it exists
      if (compTexture) {
        compTexture.texture.destroy();
      }

      // Create new texture
      const texture = device.createTexture({
        size: { width, height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      const view = texture.createView();
      compTexture = { texture, view };
      this.nestedCompTextures.set(compositionId, compTexture);
    }

    // Create temporary ping-pong textures for nested comp rendering
    const nestedPingTexture = device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const nestedPongTexture = device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const nestedPingView = nestedPingTexture.createView();
    const nestedPongView = nestedPongTexture.createView();

    // Prepare layer data for nested composition (reverse order: lower slots render on top)
    const nestedLayerData: LayerRenderData[] = [];
    for (let i = nestedLayers.length - 1; i >= 0; i--) {
      const layer = nestedLayers[i];
      if (!layer || !layer.visible || !layer.source || layer.opacity === 0) continue;

      // Try WebCodecs VideoFrame first
      if (layer.source.webCodecsPlayer) {
        const frame = layer.source.webCodecsPlayer.getCurrentFrame();
        if (frame) {
          const extTex = this.textureManager?.importVideoTexture(frame);
          if (extTex) {
            nestedLayerData.push({
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: frame.displayWidth,
              sourceHeight: frame.displayHeight,
            });
            continue;
          }
        }
      }

      // HTMLVideoElement
      if (layer.source.videoElement) {
        const video = layer.source.videoElement;
        if (video.readyState >= 2) {
          const extTex = this.textureManager?.importVideoTexture(video);
          if (extTex) {
            nestedLayerData.push({
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: video.videoWidth,
              sourceHeight: video.videoHeight,
            });
            continue;
          }
        }
      }

      // Images
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = this.textureManager?.getCachedImageTexture(img);
        if (!texture) {
          texture = this.textureManager?.createImageTexture(img) ?? undefined;
        }
        if (texture) {
          const imageView = this.textureManager!.getImageView(texture);
          nestedLayerData.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: imageView,
            sourceWidth: img.naturalWidth,
            sourceHeight: img.naturalHeight,
          });
        }
      }
    }

    // If no layers to render, return black texture
    if (nestedLayerData.length === 0) {
      // Clear the output texture to black
      const clearPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: compTexture.view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      clearPass.end();
      nestedPingTexture.destroy();
      nestedPongTexture.destroy();
      return compTexture.view;
    }

    // Ping-pong compositing for nested layers
    let readView = nestedPingView;
    let writeView = nestedPongView;

    // Clear first buffer
    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: readView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    clearPass.end();

    // Composite each nested layer
    const nestedOutputAspect = width / height;
    for (let i = 0; i < nestedLayerData.length; i++) {
      const data = nestedLayerData[i];
      const layer = data.layer;

      const uniformBuffer = this.compositorPipeline!.getOrCreateUniformBuffer(`nested-${compositionId}-${layer.id}`);
      const sourceAspect = data.sourceWidth / data.sourceHeight;

      // Get mask texture view (use white mask if none)
      const hasMask = this.maskTextureManager?.hasMaskTexture(layer.id) ?? false;
      const maskTextureView = this.maskTextureManager?.getMaskTextureView(layer.id) ??
                             this.maskTextureManager?.getWhiteMaskView()!;

      this.compositorPipeline!.updateLayerUniforms(layer, sourceAspect, nestedOutputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = this.compositorPipeline!.getExternalCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createExternalCompositeBindGroup(
          this.sampler!,
          readView,
          data.externalTexture,
          uniformBuffer,
          maskTextureView
        );
      } else if (data.textureView) {
        pipeline = this.compositorPipeline!.getCompositePipeline()!;
        bindGroup = this.compositorPipeline!.createCompositeBindGroup(
          this.sampler!,
          readView,
          data.textureView,
          uniformBuffer,
          maskTextureView
        );
      } else {
        continue;
      }

      const compositePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: writeView,
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      compositePass.setPipeline(pipeline);
      compositePass.setBindGroup(0, bindGroup);
      compositePass.draw(6);
      compositePass.end();

      // Apply effects
      if (layer.effects && layer.effects.length > 0 && this.effectsPipeline) {
        const result = this.effectsPipeline.applyEffects(
          commandEncoder,
          layer.effects,
          this.sampler!,
          writeView,
          readView,
          nestedPingView,
          nestedPongView
        );
        readView = result.outputView;
        writeView = result.outputView === nestedPingView ? nestedPongView : nestedPingView;
      } else {
        // Swap buffers
        const temp = readView;
        readView = writeView;
        writeView = temp;
      }
    }

    // Copy final result to the composition texture
    const copyPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: compTexture.view,
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    // Use output pipeline to copy from readView to compTexture
    const copyBindGroup = this.outputPipeline!.getOutputBindGroup(this.sampler!, readView, true);
    copyPass.setPipeline(this.outputPipeline!.getOutputPipeline()!);
    copyPass.setBindGroup(0, copyBindGroup);
    copyPass.draw(6);
    copyPass.end();

    // Clean up temporary textures
    nestedPingTexture.destroy();
    nestedPongTexture.destroy();

    return compTexture.view;
  }

  /**
   * Clean up nested composition textures
   */
  cleanupNestedCompTexture(compositionId: string): void {
    const compTexture = this.nestedCompTextures.get(compositionId);
    if (compTexture) {
      compTexture.texture.destroy();
      this.nestedCompTextures.delete(compositionId);
    }
  }

  private updateStats(): void {
    this.frameCount++;
    this.statsCounter++;

    if (this.statsCounter >= 10) {
      this.statsCounter = 0;
      const now = performance.now();

      if (this.lastFrameStart > 0) {
        const frameTime = (now - this.lastFrameStart) / 10;
        this.frameTimeBuffer[this.frameTimeIndex] = frameTime;
        this.frameTimeIndex = (this.frameTimeIndex + 1) % 60;
        if (this.frameTimeCount < 60) this.frameTimeCount++;
      }
      this.lastFrameStart = now;

      if (now - this.fpsUpdateTime >= 1000) {
        this.fps = this.frameCount;
        this.frameCount = 0;
        this.fpsUpdateTime = now;
      }
    }
  }

  getStats(): EngineStats {
    let sum = 0;
    for (let i = 0; i < this.frameTimeCount; i++) {
      sum += this.frameTimeBuffer[i];
    }
    const avgFrameTime = this.frameTimeCount > 0 ? sum / this.frameTimeCount : 0;
    return {
      fps: this.fps,
      frameTime: avgFrameTime,
      gpuMemory: 0,
      timing: {
        rafGap: this.detailedStats.rafGap,
        importTexture: this.detailedStats.importTexture,
        renderPass: this.detailedStats.renderPass,
        submit: this.detailedStats.submit,
        total: this.detailedStats.total,
      },
      drops: {
        count: this.detailedStats.dropsTotal,
        lastSecond: this.detailedStats.dropsLastSecond,
        reason: this.detailedStats.lastDropReason,
      },
      layerCount: this.lastLayerCount,
      targetFps: 60,
      decoder: this.detailedStats.decoder,
      audio: audioStatusTracker.getStatus(),
    };
  }

  // === ANIMATION LOOP ===

  start(renderCallback: () => void): void {
    if (this.isRunning) return;
    this.isRunning = true;
    console.log('[WebGPU] Starting render loop');

    let lastTimestamp = 0;
    let lastFpsReset = 0;

    const loop = (timestamp: number) => {
      if (!this.isRunning) return;

      const rafGap = lastTimestamp > 0 ? timestamp - lastTimestamp : 0;
      lastTimestamp = timestamp;

      // Frame rate limiting when video is playing
      if (this.hasActiveVideo) {
        const timeSinceLastRender = timestamp - this.lastRenderTime;
        if (timeSinceLastRender < this.VIDEO_FRAME_TIME) {
          this.animationId = requestAnimationFrame(loop);
          return;
        }
        this.lastRenderTime = timestamp;
      }

      this.detailedStats.rafGap = this.detailedStats.rafGap * 0.9 + rafGap * 0.1;

      // Detect frame drops
      if (rafGap > this.TARGET_FRAME_TIME * 1.5 && lastTimestamp > 0) {
        const missedFrames = Math.floor(rafGap / this.TARGET_FRAME_TIME) - 1;
        this.detailedStats.dropsTotal += missedFrames;
        this.detailedStats.dropsThisSecond += missedFrames;
        this.detailedStats.lastDropReason = 'slow_raf';
      }

      // Reset per-second drop counter
      if (timestamp - lastFpsReset >= 1000) {
        this.detailedStats.dropsLastSecond = this.detailedStats.dropsThisSecond;
        this.detailedStats.dropsThisSecond = 0;
        lastFpsReset = timestamp;
      }

      renderCallback();

      this.animationId = requestAnimationFrame(loop);
    };

    this.animationId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.isRunning = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  setResolution(width: number, height: number): void {
    // Skip if resolution hasn't changed
    if (this.outputWidth === width && this.outputHeight === height) return;

    this.outputWidth = width;
    this.outputHeight = height;
    this.createPingPongTextures();

    // Clear caches since they contain frames at old resolution
    if (this.scrubbingCache) {
      this.scrubbingCache.clearCompositeCache();
      this.scrubbingCache.clearScrubbingCache();
      console.log(`[Engine] Caches cleared for resolution change to ${width}×${height}`);
    }
  }

  getDevice(): GPUDevice | null {
    return this.context.getDevice();
  }

  getOutputDimensions(): { width: number; height: number } {
    return { width: this.outputWidth, height: this.outputHeight };
  }

  async readPixels(): Promise<Uint8ClampedArray | null> {
    const device = this.context.getDevice();
    if (!device || !this.pingTexture || !this.pongTexture) return null;

    const sourceTexture = this.lastRenderWasPing ? this.pingTexture : this.pongTexture;

    const bytesPerPixel = 4;
    const unalignedBytesPerRow = this.outputWidth * bytesPerPixel;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / 256) * 256;
    const bufferSize = bytesPerRow * this.outputHeight;

    const stagingBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyTextureToBuffer(
      { texture: sourceTexture },
      { buffer: stagingBuffer, bytesPerRow, rowsPerImage: this.outputHeight },
      [this.outputWidth, this.outputHeight]
    );
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();

    const result = new Uint8ClampedArray(this.outputWidth * this.outputHeight * bytesPerPixel);
    const srcView = new Uint8Array(arrayBuffer);

    if (bytesPerRow === unalignedBytesPerRow) {
      result.set(srcView.subarray(0, result.length));
    } else {
      for (let y = 0; y < this.outputHeight; y++) {
        const srcOffset = y * bytesPerRow;
        const dstOffset = y * unalignedBytesPerRow;
        result.set(srcView.subarray(srcOffset, srcOffset + unalignedBytesPerRow), dstOffset);
      }
    }

    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return result;
  }

  destroy(): void {
    this.stop();

    for (const output of this.outputWindows.values()) {
      output.window?.close();
    }
    this.outputWindows.clear();

    this.pingTexture?.destroy();
    this.pongTexture?.destroy();
    this.blackTexture?.destroy();

    this.textureManager?.destroy();
    this.maskTextureManager?.destroy();
    this.scrubbingCache?.destroy();
    this.videoFrameManager.destroy();
    this.compositorPipeline?.destroy();
    this.effectsPipeline?.destroy();
    this.outputPipeline?.destroy();
    this.context.destroy();

    // Clear optimization caches
    this.lastVideoTime.clear();
    this.cachedExternalTexture.clear();
    this.pendingUniformUpdates.length = 0;
  }
}

// Preserve singleton across HMR to prevent multiple GPU device creation
let engineInstance: WebGPUEngine;

// Check if we're in a Vite HMR context
declare const import_meta_hot: { data: Record<string, unknown> } | undefined;
const hot = typeof import.meta !== 'undefined' ? (import.meta as { hot?: { data: Record<string, unknown> } }).hot : undefined;

if (hot) {
  // Preserve engine instance across HMR
  const existing = hot.data.engine as WebGPUEngine | undefined;
  if (existing) {
    console.log('[WebGPU] Reusing existing engine instance from HMR');
    engineInstance = existing;
  } else {
    console.log('[WebGPU] Creating new engine instance');
    engineInstance = new WebGPUEngine();
    hot.data.engine = engineInstance;
  }
} else {
  engineInstance = new WebGPUEngine();
}

export const engine = engineInstance;
