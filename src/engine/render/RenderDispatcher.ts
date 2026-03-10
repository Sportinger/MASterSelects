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
import { useTimelineStore } from '../../stores/timeline';
import { reportRenderTime } from '../../services/performanceMonitor';
import { Logger } from '../../services/logger';
import { scrubSettleState } from '../../services/scrubSettleState';
import { vfPipelineMonitor } from '../../services/vfPipelineMonitor';
import { getCopiedHtmlVideoPreviewFrame } from './htmlVideoPreviewFallback';

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
  private lastPreviewSignature = '';
  private lastPreviewTargetTimeMs?: number;

  constructor(deps: RenderDeps) {
    this.deps = deps;
  }

  private getTargetVideoTime(layer: Layer, video: HTMLVideoElement): number {
    return layer.source?.mediaTime ?? video.currentTime;
  }

  private getSafePreviewFallback(
    layer: Layer,
    video: HTMLVideoElement
  ): { view: GPUTextureView; width: number; height: number; mediaTime?: number } | null {
    const scrubbingCache = this.deps.cacheManager.getScrubbingCache();
    if (!scrubbingCache) {
      return null;
    }
    const isDragging = useTimelineStore.getState().isDraggingPlayhead;
    const tolerance = video.seeking || isDragging ? 0.35 : 0.2;
    return scrubbingCache.getLastFrameNearTime(
      video,
      this.getTargetVideoTime(layer, video),
      tolerance,
      layer.sourceClipId
    );
  }

  private isFrameNearTarget(
    frame: { mediaTime?: number } | null | undefined,
    targetTime: number,
    maxDeltaSeconds: number = 0.35
  ): boolean {
    return (
      typeof frame?.mediaTime === 'number' &&
      Number.isFinite(frame.mediaTime) &&
      Math.abs(frame.mediaTime - targetTime) <= maxDeltaSeconds
    );
  }

  private toMediaTimeMs(time?: number): number | undefined {
    if (typeof time !== 'number' || !Number.isFinite(time)) {
      return undefined;
    }
    return Math.round(time * 1000);
  }

  private recordMainPreviewFrame(
    mode: string,
    layerData?: LayerRenderData[],
    fallback?: { clipId?: string; targetTimeMs?: number; displayedTimeMs?: number }
  ): void {
    const primary = layerData?.find((data) => data.layer.source?.type === 'video') ?? layerData?.[0];
    const clipId = fallback?.clipId ?? primary?.layer.sourceClipId ?? primary?.layer.id;
    const targetTimeMs =
      fallback?.targetTimeMs ??
      this.toMediaTimeMs(primary?.targetMediaTime);
    const displayedTimeMs =
      fallback?.displayedTimeMs ??
      this.toMediaTimeMs(primary?.displayedMediaTime ?? primary?.targetMediaTime);
    const previewPath =
      primary?.previewPath ??
      primary?.layer.source?.type ??
      mode;
    const signature = layerData && layerData.length > 0
      ? layerData
        .slice(0, 4)
        .map((data) => {
          const id = data.layer.sourceClipId ?? data.layer.id;
          const mediaTimeMs = this.toMediaTimeMs(data.displayedMediaTime ?? data.targetMediaTime) ?? -1;
          return `${id}:${data.previewPath ?? data.layer.source?.type ?? 'layer'}:${mediaTimeMs}`;
        })
        .join('|')
      : `${mode}:${clipId ?? 'none'}:${displayedTimeMs ?? -1}`;
    const changed = signature !== this.lastPreviewSignature;
    const targetMoved =
      targetTimeMs !== undefined &&
      this.lastPreviewTargetTimeMs !== undefined &&
      Math.abs(targetTimeMs - this.lastPreviewTargetTimeMs) >= 12;
    const driftMs =
      targetTimeMs !== undefined && displayedTimeMs !== undefined
        ? Math.abs(targetTimeMs - displayedTimeMs)
        : undefined;

    vfPipelineMonitor.record('vf_preview_frame', {
      mode,
      changed: changed ? 'true' : 'false',
      targetMoved: targetMoved ? 'true' : 'false',
      previewPath,
      ...(clipId ? { clipId } : {}),
      ...(targetTimeMs !== undefined ? { targetTimeMs } : {}),
      ...(displayedTimeMs !== undefined ? { displayedTimeMs } : {}),
      ...(driftMs !== undefined ? { driftMs } : {}),
    });

    this.lastPreviewSignature = signature;
    this.lastPreviewTargetTimeMs = targetTimeMs;
  }

  // === MAIN RENDER ===

  render(layers: Layer[]): void {
    const d = this.deps;
    if (d.isRecovering()) return;

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
    const skipEffects = useTimelineStore.getState().isDraggingPlayhead;

    // Collect layer data
    const t1 = performance.now();
    const layerData = d.layerCollector.collect(layers, {
      textureManager: d.textureManager!,
      scrubbingCache: d.cacheManager.getScrubbingCache(),
      getLastVideoTime: (key) => d.cacheManager.getLastVideoTime(key),
      setLastVideoTime: (key, time) => d.cacheManager.setLastVideoTime(key, time),
      isExporting: d.exportCanvasManager.getIsExporting(),
      isPlaying: d.renderLoop?.getIsPlaying() ?? false,
    });
    const importTime = performance.now() - t1;

    // Update stats
    d.performanceStats.setDecoder(d.layerCollector.getDecoder());
    d.performanceStats.setWebCodecsInfo(d.layerCollector.getWebCodecsInfo());
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
          nc.compositionId, nc.layers, nc.width, nc.height, preRenderEncoder, d.sampler, nc.currentTime, 0, skipEffects
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
      skipEffects,
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
        this.recordMainPreviewFrame('composite', layerData);
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
      if (!d.previewContext && activeTargets.length > 0) {
        this.recordMainPreviewFrame('target-composite', layerData);
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
        this.recordMainPreviewFrame('empty');
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
        const scrubbingCache = d.cacheManager.getScrubbingCache();
        const targetTime = this.getTargetVideoTime(layer, video);
        const isDragging = useTimelineStore.getState().isDraggingPlayhead;
        const isSettling = scrubSettleState.isPending(layer.sourceClipId);
        const isPausedSettle = !(d.renderLoop?.getIsPlaying() ?? false) && !isDragging && isSettling;
        const lastPresentedTime = scrubbingCache?.getLastPresentedTime(video);
        const lastPresentedOwner = scrubbingCache?.getLastPresentedOwner(video);
        const hasPresentedOwnerMismatch =
          !!layer.sourceClipId &&
          !!lastPresentedOwner &&
          lastPresentedOwner !== layer.sourceClipId;
        const hasConfirmedPresentedFrame =
          !hasPresentedOwnerMismatch &&
          typeof lastPresentedTime === 'number' &&
          Number.isFinite(lastPresentedTime);
        const displayedTime = hasConfirmedPresentedFrame ? lastPresentedTime : undefined;
        const hasFreshPresentedFrame =
          hasConfirmedPresentedFrame &&
          Math.abs(lastPresentedTime - targetTime) <= 0.12;
        const awaitingPausedTargetFrame =
          hasPresentedOwnerMismatch ||
          !(d.renderLoop?.getIsPlaying() ?? false) &&
          !isDragging &&
          (!isSettling &&
            (!hasConfirmedPresentedFrame || Math.abs(lastPresentedTime - targetTime) > 0.05));
        const cacheSearchDistanceFrames = isDragging ? 12 : 6;
        const lastSameClipFrame = layer.sourceClipId
          ? scrubbingCache?.getLastFrame(video, layer.sourceClipId) ?? null
          : null;
        const dragHoldFrame = isDragging
          ? lastSameClipFrame
          : (isSettling || awaitingPausedTargetFrame) && this.isFrameNearTarget(lastSameClipFrame, targetTime)
            ? lastSameClipFrame
            : null;
        const emergencyHoldFrame = isDragging && layer.sourceClipId
          ? lastSameClipFrame
          : dragHoldFrame;
        const safeFallback = this.getSafePreviewFallback(layer, video) ?? dragHoldFrame;
        const allowLiveVideoImport = !hasPresentedOwnerMismatch && (isPausedSettle
          ? hasFreshPresentedFrame
          : !awaitingPausedTargetFrame &&
            (((!isDragging && !isSettling) || hasFreshPresentedFrame || !safeFallback)));
        const allowConfirmedFrameCaching = !hasPresentedOwnerMismatch && (isPausedSettle
          ? hasFreshPresentedFrame
          : !awaitingPausedTargetFrame &&
            (((!isDragging && !isSettling) || hasFreshPresentedFrame)));
        const captureOwnerId = allowConfirmedFrameCaching ? layer.sourceClipId : undefined;
        if (video.readyState >= 2) {
          if ((video.seeking || awaitingPausedTargetFrame) && scrubbingCache) {
            const cachedView =
              scrubbingCache.getCachedFrame(video.src, targetTime) ??
              scrubbingCache.getNearestCachedFrame(video.src, targetTime, cacheSearchDistanceFrames);
            if (cachedView) {
              layerData.push({
                layer,
                isVideo: false,
                externalTexture: null,
                textureView: cachedView,
                sourceWidth: video.videoWidth,
                sourceHeight: video.videoHeight,
              });
              continue;
            }
            if (!allowLiveVideoImport && safeFallback) {
              layerData.push({
                layer,
                isVideo: false,
                externalTexture: null,
                textureView: safeFallback.view,
                sourceWidth: safeFallback.width,
                sourceHeight: safeFallback.height,
              });
              continue;
            }
            if (!allowLiveVideoImport && emergencyHoldFrame) {
              layerData.push({
                layer,
                isVideo: false,
                externalTexture: null,
                textureView: emergencyHoldFrame.view,
                sourceWidth: emergencyHoldFrame.width,
                sourceHeight: emergencyHoldFrame.height,
              });
              continue;
            }
          }

          const copiedFrame = getCopiedHtmlVideoPreviewFrame(
            video,
            scrubbingCache,
            targetTime,
            layer.sourceClipId,
            captureOwnerId
          );
          if (allowLiveVideoImport && copiedFrame) {
            layerData.push({
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: copiedFrame.view,
              sourceWidth: copiedFrame.width,
              sourceHeight: copiedFrame.height,
            });
            continue;
          }

          const extTex = allowLiveVideoImport
            ? d.textureManager?.importVideoTexture(video)
            : null;
          if (extTex) {
            if (scrubbingCache && allowConfirmedFrameCaching && !(d.renderLoop?.getIsPlaying() ?? false)) {
              const now = performance.now();
              const lastCapture = scrubbingCache.getLastCaptureTime(video);
              if (now - lastCapture > 50) {
                scrubbingCache.captureVideoFrame(video, captureOwnerId);
                scrubbingCache.setLastCaptureTime(video, now);
              }
              scrubbingCache.cacheFrameAtTime(video, targetTime);
            } else if (scrubbingCache && !(d.renderLoop?.getIsPlaying() ?? false)) {
              if (typeof displayedTime === 'number' && Number.isFinite(displayedTime)) {
                scrubbingCache.captureVideoFrameIfCloser(
                  video,
                  targetTime,
                  displayedTime,
                  layer.sourceClipId
                );
              }
              if (typeof displayedTime === 'number' && Number.isFinite(displayedTime)) {
                scrubbingCache.cacheFrameAtTime(video, displayedTime);
              }
            }
            layerData.push({ layer, isVideo: true, externalTexture: extTex, textureView: null, sourceWidth: video.videoWidth, sourceHeight: video.videoHeight });
            continue;
        }

        const notReadyCachedFrame =
          scrubbingCache?.getCachedFrameEntry(video.src, targetTime) ??
          scrubbingCache?.getNearestCachedFrameEntry(video.src, targetTime, cacheSearchDistanceFrames);
        if (notReadyCachedFrame) {
          layerData.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: notReadyCachedFrame.view,
            sourceWidth: video.videoWidth,
            sourceHeight: video.videoHeight,
            displayedMediaTime: notReadyCachedFrame.mediaTime,
            targetMediaTime: targetTime,
            previewPath: 'not-ready-scrub-cache',
          });
          continue;
        }

        if (safeFallback) {
          layerData.push({
            layer,
              isVideo: false,
              externalTexture: null,
              textureView: safeFallback.view,
              sourceWidth: safeFallback.width,
              sourceHeight: safeFallback.height,
            });
            continue;
          }
          if (emergencyHoldFrame) {
            layerData.push({
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: emergencyHoldFrame.view,
              sourceWidth: emergencyHoldFrame.width,
              sourceHeight: emergencyHoldFrame.height,
            });
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
        this.recordMainPreviewFrame('target-empty');
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
    this.recordMainPreviewFrame('target-canvas', layerData);

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
      const commandEncoder = device.createCommandEncoder();
      d.outputPipeline.renderToCanvas(commandEncoder, d.previewContext, gpuCached.bindGroup);
      this.recordMainPreviewFrame('ram-gpu-cache', undefined, {
        targetTimeMs: Math.round(time * 1000),
        displayedTimeMs: Math.round(time * 1000),
      });
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
      return false;
    }
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
      this.recordMainPreviewFrame('ram-cpu-cache', undefined, {
        targetTimeMs: Math.round(time * 1000),
        displayedTimeMs: Math.round(time * 1000),
      });
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
}
