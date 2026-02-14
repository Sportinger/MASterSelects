// RenderDispatcher — extracted render methods from WebGPUEngine
// Handles: render(), renderEmptyFrame(), renderToPreviewCanvas(), renderCachedFrame()

import type { Layer, LayerRenderData } from '../core/types';
import type { TextureManager } from '../texture/TextureManager';
import type { MaskTextureManager } from '../texture/MaskTextureManager';
import type { CacheManager } from '../managers/CacheManager';
import type { ExportCanvasManager } from '../managers/ExportCanvasManager';
import type { CompositorPipeline } from '../pipeline/CompositorPipeline';
import type { OutputPipeline } from '../pipeline/OutputPipeline';
import type { SlicePipeline } from '../pipeline/SlicePipeline';
import type { RenderTargetManager } from '../core/RenderTargetManager';
import type { LayerCollector } from './LayerCollector';
import type { Compositor } from './Compositor';
import type { NestedCompRenderer } from './NestedCompRenderer';
import type { PerformanceStats } from '../stats/PerformanceStats';
import type { RenderLoop } from './RenderLoop';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { useSliceStore } from '../../stores/sliceStore';
import { reportRenderTime } from '../../services/performanceMonitor';
import { Logger } from '../../services/logger';
import { flags } from '../featureFlags';
import { SceneGraphBuilder } from '../sceneGraph/SceneGraphBuilder';
import { SceneGraphEvaluator } from '../sceneGraph/SceneGraphEvaluator';
import { RenderGraphBuilder } from '../renderGraph/RenderGraphBuilder';
import { RenderGraphExecutor } from '../renderGraph/RenderGraphExecutor';
import { useTimelineStore } from '../../stores/timeline';

const log = Logger.create('RenderDispatcher');

/**
 * Mutable deps bag — the engine updates these references as they change
 * (e.g. after device loss/restore, canvas attach/detach).
 */
export interface RenderDeps {
  getDevice: () => GPUDevice | null;
  isRecovering: () => boolean;
  sampler: GPUSampler | null;
  previewContext: GPUCanvasContext | null;
  targetCanvases: Map<string, { canvas: HTMLCanvasElement; context: GPUCanvasContext }>;
  compositorPipeline: CompositorPipeline | null;
  outputPipeline: OutputPipeline | null;
  slicePipeline: SlicePipeline | null;
  textureManager: TextureManager | null;
  maskTextureManager: MaskTextureManager | null;
  renderTargetManager: RenderTargetManager | null;
  layerCollector: LayerCollector | null;
  compositor: Compositor | null;
  nestedCompRenderer: NestedCompRenderer | null;
  cacheManager: CacheManager;
  exportCanvasManager: ExportCanvasManager;
  performanceStats: PerformanceStats;
  renderLoop: RenderLoop | null;
}

export class RenderDispatcher {
  /** Whether the last render() call produced visible content */
  lastRenderHadContent = false;
  private deps: RenderDeps;

  // Render Graph instances (lazy-initialized when flag is enabled)
  private sceneGraphBuilder = new SceneGraphBuilder();
  private sceneGraphEvaluator = new SceneGraphEvaluator();
  private renderGraphBuilder = new RenderGraphBuilder();
  private renderGraphExecutor = new RenderGraphExecutor();

  constructor(deps: RenderDeps) {
    this.deps = deps;
  }

  // === MAIN RENDER ===

  render(layers: Layer[]): void {
    const d = this.deps;
    if (d.isRecovering()) return;

    // Render Graph path: build scene graph → evaluate → build render graph → execute
    if (flags.useRenderGraph) {
      this.renderViaRenderGraph();
      return;
    }

    const device = d.getDevice();
    if (!device || !d.compositorPipeline || !d.outputPipeline || !d.sampler) return;
    if (!d.renderTargetManager || !d.layerCollector || !d.compositor || !d.textureManager) return;

    const pingView = d.renderTargetManager.getPingView();
    const pongView = d.renderTargetManager.getPongView();
    if (!pingView || !pongView) return;

    // Clear frame-scoped caches (external texture bind groups)
    d.compositorPipeline.beginFrame();

    const t0 = performance.now();
    const { width, height } = d.renderTargetManager.getResolution();

    // Collect layer data
    const t1 = performance.now();
    const layerData = d.layerCollector.collect(layers, {
      textureManager: d.textureManager!,
      scrubbingCache: d.cacheManager.getScrubbingCache(),
      getLastVideoTime: (key) => d.cacheManager.getLastVideoTime(key),
      setLastVideoTime: (key, time) => d.cacheManager.setLastVideoTime(key, time),
      isExporting: d.exportCanvasManager.getIsExporting(),
    });
    const importTime = performance.now() - t1;

    // Update stats
    d.performanceStats.setDecoder(d.layerCollector.getDecoder());
    d.renderLoop?.setHasActiveVideo(d.layerCollector.hasActiveVideo());

    // Handle empty layers
    if (layerData.length === 0) {
      this.lastRenderHadContent = false;
      this.renderEmptyFrame(device);
      d.performanceStats.setLayerCount(0);
      return;
    }
    this.lastRenderHadContent = true;

    // Pre-render nested compositions (batched with main composite)
    const commandBuffers: GPUCommandBuffer[] = [];
    let hasNestedComps = false;

    const preRenderEncoder = device.createCommandEncoder();
    for (const data of layerData) {
      if (data.layer.source?.nestedComposition) {
        hasNestedComps = true;
        const nc = data.layer.source.nestedComposition;
        const view = d.nestedCompRenderer!.preRender(
          nc.compositionId, nc.layers, nc.width, nc.height, preRenderEncoder, d.sampler, nc.currentTime
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

    // Get effect temp textures for pre-processing effects on source layers
    const effectTempTexture = d.renderTargetManager.getEffectTempTexture() ?? undefined;
    const effectTempView = d.renderTargetManager.getEffectTempView() ?? undefined;
    const effectTempTexture2 = d.renderTargetManager.getEffectTempTexture2() ?? undefined;
    const effectTempView2 = d.renderTargetManager.getEffectTempView2() ?? undefined;

    const result = d.compositor.composite(layerData, commandEncoder, {
      device, sampler: d.sampler, pingView, pongView, outputWidth: width, outputHeight: height,
      effectTempTexture, effectTempView, effectTempTexture2, effectTempView2,
    });
    const renderTime = performance.now() - t2;

    // Output
    d.outputPipeline!.updateResolution(width, height);

    const skipCanvas = d.exportCanvasManager.shouldSkipPreviewOutput();
    if (!skipCanvas) {
      // Output to main preview canvas (legacy — no grid)
      if (d.previewContext) {
        const mainBindGroup = d.outputPipeline!.createOutputBindGroup(d.sampler, result.finalView, false);
        d.outputPipeline!.renderToCanvas(commandEncoder, d.previewContext, mainBindGroup);
      }
      // Output to all activeComp render targets (from unified store)
      const activeTargets = useRenderTargetStore.getState().getActiveCompTargets();
      const sliceState = useSliceStore.getState();
      const sliceConfigs = sliceState.configs;
      for (const target of activeTargets) {
        const ctx = d.targetCanvases.get(target.id)?.context;
        if (!ctx) continue;

        // For the OM preview canvas, use the previewed target's slices (if in output mode)
        let sliceLookupId = target.id;
        if (target.id === '__om_preview__' && sliceState.previewingTargetId) {
          if (sliceState.activeTab === 'output') {
            sliceLookupId = sliceState.previewingTargetId;
          }
        }

        const config = sliceConfigs.get(sliceLookupId);
        const enabledSlices = config?.slices.filter((s) => s.enabled) ?? [];

        if (enabledSlices.length > 0 && d.slicePipeline) {
          d.slicePipeline.buildVertexBuffer(enabledSlices);
          d.slicePipeline.renderSlicedOutput(commandEncoder, ctx, result.finalView, d.sampler!);
        } else {
          const targetBindGroup = d.outputPipeline!.createOutputBindGroup(d.sampler, result.finalView, target.showTransparencyGrid);
          d.outputPipeline!.renderToCanvas(commandEncoder, ctx, targetBindGroup);
        }
      }
    }

    // Render to export canvas for zero-copy VideoFrame creation (never show grid)
    const exportCtx = d.exportCanvasManager.getExportCanvasContext();
    if (d.exportCanvasManager.getIsExporting() && exportCtx) {
      const exportBindGroup = d.outputPipeline!.createOutputBindGroup(d.sampler, result.finalView, false);
      d.outputPipeline!.renderToCanvas(commandEncoder, exportCtx, exportBindGroup);
    }

    // Batch submit all command buffers in single call
    commandBuffers.push(commandEncoder.finish());
    const t3 = performance.now();
    try {
      device.queue.submit(commandBuffers);
    } catch (e) {
      // GPU submit failed - likely device lost or validation error
      log.error('GPU submit failed', e);
      return;
    }
    const submitTime = performance.now() - t3;

    // Cleanup after submit
    if (hasNestedComps) {
      d.nestedCompRenderer!.cleanupPendingTextures();
    }

    // Stats
    const totalTime = performance.now() - t0;
    d.performanceStats.recordRenderTiming({
      importTexture: importTime,
      createBindGroup: 0,
      renderPass: renderTime,
      submit: submitTime,
      total: totalTime,
    });
    d.performanceStats.setLayerCount(result.layerCount);
    d.performanceStats.updateStats();
    reportRenderTime(totalTime);
  }

  renderEmptyFrame(device: GPUDevice): void {
    const d = this.deps;
    const commandEncoder = device.createCommandEncoder();
    const pingView = d.renderTargetManager?.getPingView();

    // Use output pipeline to render empty frame (allows shader to generate checkerboard)
    if (pingView && d.outputPipeline && d.sampler) {
      // Clear ping texture to transparent
      const clearPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: pingView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      clearPass.end();

      const { width, height } = d.renderTargetManager!.getResolution();
      d.outputPipeline.updateResolution(width, height);

      // Render through output pipeline to main preview (no grid) + all activeComp targets
      if (d.previewContext) {
        const mainBindGroup = d.outputPipeline.createOutputBindGroup(d.sampler, pingView, false);
        d.outputPipeline.renderToCanvas(commandEncoder, d.previewContext, mainBindGroup);
      }
      const activeTargets = useRenderTargetStore.getState().getActiveCompTargets();
      for (const target of activeTargets) {
        const ctx = d.targetCanvases.get(target.id)?.context;
        if (!ctx) continue;
        const targetBindGroup = d.outputPipeline.createOutputBindGroup(d.sampler, pingView, target.showTransparencyGrid);
        d.outputPipeline.renderToCanvas(commandEncoder, ctx, targetBindGroup);
      }
    } else {
      // Fallback: direct clear
      if (d.previewContext) {
        try {
          const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
              view: d.previewContext.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: 'clear',
              storeOp: 'store',
            }],
          });
          pass.end();
        } catch {
          // Canvas context lost - skip
        }
      }
    }
    // Also clear export canvas when exporting (needed for empty frames at export boundaries)
    const emptyExportCtx = d.exportCanvasManager.getExportCanvasContext();
    if (d.exportCanvasManager.getIsExporting() && emptyExportCtx) {
      try {
        const pass = commandEncoder.beginRenderPass({
          colorAttachments: [{
            view: emptyExportCtx.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        pass.end();
      } catch {
        // Export canvas context lost - skip
      }
    }
    device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Render specific layers to a specific target canvas
   * Used for multi-composition preview where each preview shows different content
   */
  renderToPreviewCanvas(canvasId: string, layers: Layer[]): void {
    const d = this.deps;
    if (d.isRecovering()) return;

    const device = d.getDevice();
    const canvasContext = d.targetCanvases.get(canvasId)?.context;
    if (!device || !canvasContext || !d.compositorPipeline || !d.outputPipeline || !d.sampler) return;

    const indPingView = d.renderTargetManager?.getIndependentPingView();
    const indPongView = d.renderTargetManager?.getIndependentPongView();
    if (!indPingView || !indPongView) return;

    // Prepare layer data
    const layerData: LayerRenderData[] = [];
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer?.visible || !layer.source || layer.opacity === 0) continue;

      if (layer.source.videoElement) {
        const video = layer.source.videoElement;
        if (video.readyState >= 2) {
          const extTex = d.textureManager?.importVideoTexture(video);
          if (extTex) {
            layerData.push({ layer, isVideo: true, externalTexture: extTex, textureView: null, sourceWidth: video.videoWidth, sourceHeight: video.videoHeight });
            continue;
          }
        }
      }
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = d.textureManager?.getCachedImageTexture(img);
        if (!texture) texture = d.textureManager?.createImageTexture(img) ?? undefined;
        if (texture) {
          layerData.push({ layer, isVideo: false, externalTexture: null, textureView: d.textureManager!.getImageView(texture), sourceWidth: img.naturalWidth, sourceHeight: img.naturalHeight });
        }
      }
      if (layer.source.textCanvas) {
        const canvas = layer.source.textCanvas;
        const texture = d.textureManager?.createCanvasTexture(canvas);
        if (texture) {
          layerData.push({ layer, isVideo: false, externalTexture: null, textureView: d.textureManager!.getImageView(texture), sourceWidth: canvas.width, sourceHeight: canvas.height });
        }
      }
    }

    const { width, height } = d.renderTargetManager!.getResolution();

    // Read per-target transparency flag
    const target = useRenderTargetStore.getState().targets.get(canvasId);
    const showGrid = target?.showTransparencyGrid ?? false;

    // Ensure resolution is up to date for this render
    d.outputPipeline.updateResolution(width, height);

    if (layerData.length === 0) {
      const commandEncoder = device.createCommandEncoder();
      const blackTex = d.renderTargetManager!.getBlackTexture();
      if (blackTex) {
        const blackView = blackTex.createView();
        const blackBindGroup = d.outputPipeline.createOutputBindGroup(d.sampler, blackView, showGrid);
        d.outputPipeline.renderToCanvas(commandEncoder, canvasContext, blackBindGroup);
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
      const uniformBuffer = d.compositorPipeline!.getOrCreateUniformBuffer(layer.id);
      const sourceAspect = data.sourceWidth / data.sourceHeight;
      const outputAspect = width / height;
      const maskLookupId = layer.maskClipId || layer.id;
      // Get mask info
      const maskManager = d.maskTextureManager!;
      const maskInfo = maskManager.getMaskInfo(maskLookupId) ?? { hasMask: false, view: maskManager.getWhiteMaskView() };
      const hasMask = maskInfo.hasMask;
      const maskTextureView = maskInfo.view;

      d.compositorPipeline!.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = d.compositorPipeline!.getExternalCompositePipeline()!;
        bindGroup = d.compositorPipeline!.createExternalCompositeBindGroup(d.sampler!, readView, data.externalTexture, uniformBuffer, maskTextureView);
      } else if (data.textureView) {
        pipeline = d.compositorPipeline!.getCompositePipeline()!;
        bindGroup = d.compositorPipeline!.createCompositeBindGroup(d.sampler!, readView, data.textureView, uniformBuffer, maskTextureView);
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

    const outputBindGroup = d.outputPipeline!.createOutputBindGroup(d.sampler!, readView, showGrid);
    d.outputPipeline!.renderToCanvas(commandEncoder, canvasContext, outputBindGroup);

    device.queue.submit([commandEncoder.finish()]);
  }

  renderCachedFrame(time: number): boolean {
    const d = this.deps;
    const device = d.getDevice();
    const scrubbingCache = d.cacheManager.getScrubbingCache();
    if (!d.previewContext || !device || !scrubbingCache || !d.outputPipeline || !d.sampler) {
      return false;
    }

    const gpuCached = scrubbingCache.getGpuCachedFrame(time);
    if (gpuCached) {
      log.debug('RAM Preview cache hit (GPU)', { time: time.toFixed(3) });
      const commandEncoder = device.createCommandEncoder();
      d.outputPipeline.renderToCanvas(commandEncoder, d.previewContext, gpuCached.bindGroup);
      // Output to all activeComp targets
      const activeTargets = useRenderTargetStore.getState().getActiveCompTargets();
      for (const target of activeTargets) {
        const ctx = d.targetCanvases.get(target.id)?.context;
        if (ctx) d.outputPipeline.renderToCanvas(commandEncoder, ctx, gpuCached.bindGroup);
      }
      device.queue.submit([commandEncoder.finish()]);
      return true;
    }

    const imageData = scrubbingCache.getCachedCompositeFrame(time);
    if (!imageData) {
      // Only log occasionally to avoid spam
      if (Math.random() < 0.05) {
        log.debug('RAM Preview cache miss', { time: time.toFixed(3), cacheSize: scrubbingCache.getCompositeCacheStats(1920, 1080).count });
      }
      return false;
    }
    log.debug('RAM Preview cache hit (ImageData→GPU)', { time: time.toFixed(3) });

    try {
      const { width, height } = { width: imageData.width, height: imageData.height };

      let canvas = d.cacheManager.getRamPlaybackCanvas();
      let ctx = d.cacheManager.getRamPlaybackCtx();

      if (!canvas || !ctx) {
        canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        ctx = canvas.getContext('2d', { willReadFrequently: false });
        if (!ctx) return false;
        d.cacheManager.setRamPlaybackCanvas(canvas, ctx);
      } else if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.putImageData(imageData, 0, 0);

      const texture = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      device.queue.copyExternalImageToTexture({ source: canvas }, { texture }, [width, height]);

      const view = texture.createView();
      const bindGroup = d.outputPipeline.createOutputBindGroup(d.sampler, view);

      scrubbingCache.addToGpuCache(time, { texture, view, bindGroup });

      const commandEncoder = device.createCommandEncoder();
      d.outputPipeline.renderToCanvas(commandEncoder, d.previewContext, bindGroup);
      // Output to all activeComp targets
      const cachedActiveTargets = useRenderTargetStore.getState().getActiveCompTargets();
      for (const target of cachedActiveTargets) {
        const ctx = d.targetCanvases.get(target.id)?.context;
        if (ctx) d.outputPipeline.renderToCanvas(commandEncoder, ctx, bindGroup);
      }
      device.queue.submit([commandEncoder.finish()]);
      return true;
    } catch (e) {
      log.warn('Failed to render cached frame', e);
      return false;
    }
  }

  // === RENDER GRAPH PATH ===

  private renderViaRenderGraph(): void {
    const d = this.deps;
    const device = d.getDevice();
    if (!device || !d.renderTargetManager) return;

    const { width, height } = d.renderTargetManager.getResolution();
    const t0 = performance.now();

    // Build scene graph from current store state
    const graph = this.sceneGraphBuilder.build();

    // Evaluate at current playhead position
    const playheadPosition = useTimelineStore.getState().playheadPosition;
    const evaluated = this.sceneGraphEvaluator.evaluate(graph, playheadPosition);

    if (evaluated.length === 0) {
      this.lastRenderHadContent = false;
      this.renderEmptyFrame(device);
      d.performanceStats.setLayerCount(0);
      return;
    }
    this.lastRenderHadContent = true;

    // Build and execute render graph
    const renderGraph = this.renderGraphBuilder.build(evaluated, width, height);
    this.renderGraphExecutor.execute(renderGraph, d);

    // Stats
    const totalTime = performance.now() - t0;
    d.performanceStats.setLayerCount(evaluated.length);
    d.performanceStats.updateStats();
    reportRenderTime(totalTime);
  }
}
