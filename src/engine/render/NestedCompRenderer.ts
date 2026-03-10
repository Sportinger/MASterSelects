// Pre-renders nested compositions to offscreen textures

import type { Layer, LayerRenderData } from '../core/types';
import type { CompositorPipeline } from '../pipeline/CompositorPipeline';
import type { EffectsPipeline } from '../../effects/EffectsPipeline';
import type { TextureManager } from '../texture/TextureManager';
import type { MaskTextureManager } from '../texture/MaskTextureManager';
import type { ScrubbingCache } from '../texture/ScrubbingCache';
import { flags } from '../featureFlags';
import { MAX_NESTING_DEPTH } from '../../stores/timeline/constants';
import { Logger } from '../../services/logger';
import {
  getRuntimeFrameProvider,
  readRuntimeFrameForSource,
} from '../../services/mediaRuntime/runtimePlayback';
import { scrubSettleState } from '../../services/scrubSettleState';
import { wcPipelineMonitor } from '../../services/wcPipelineMonitor';
import { useTimelineStore } from '../../stores/timeline';
import { getCopiedHtmlVideoPreviewFrame } from './htmlVideoPreviewFallback';

const log = Logger.create('NestedCompRenderer');
const ENABLE_VISUAL_HTML_VIDEO_FALLBACK = false;

interface NestedCompTexture {
  texture: GPUTexture;
  view: GPUTextureView;
}

interface PooledTexturePair {
  pingTexture: GPUTexture;
  pongTexture: GPUTexture;
  pingView: GPUTextureView;
  pongView: GPUTextureView;
  width: number;
  height: number;
  inUse: boolean;
}

export class NestedCompRenderer {
  private device: GPUDevice;
  private compositorPipeline: CompositorPipeline;
  private effectsPipeline: EffectsPipeline;
  private textureManager: TextureManager;
  private maskTextureManager: MaskTextureManager;
  private scrubbingCache: ScrubbingCache | null;
  private nestedCompTextures: Map<string, NestedCompTexture> = new Map();

  // Texture pool for ping-pong buffers, keyed by "widthxheight"
  private texturePool: Map<string, PooledTexturePair[]> = new Map();

  // Frame caching: track last render time to skip redundant re-renders
  private lastRenderTime: Map<string, number> = new Map();
  private lastLayerCount: Map<string, number> = new Map();
  private providerIds = new WeakMap<object, number>();
  private nextProviderId = 1;
  private lastSuccessfulVideoProviderKey = new Map<string, string>();
  private lastCollectorState = new Map<string, 'render' | 'hold' | 'drop'>();

  private getSafeLastFrameFallback(layer: Layer, video: HTMLVideoElement, targetTime: number) {
    if (!this.scrubbingCache) {
      return null;
    }
    const isDragging = useTimelineStore.getState().isDraggingPlayhead;
    const tolerance = video.seeking || isDragging ? 0.35 : 0.2;
    return this.scrubbingCache.getLastFrameNearTime(video, targetTime, tolerance, layer.sourceClipId);
  }

  private getTargetVideoTime(layer: Layer, video: HTMLVideoElement): number {
    return layer.source?.mediaTime ?? video.currentTime;
  }

  private getDragHoldFrame(layer: Layer, video: HTMLVideoElement) {
    if (!layer.sourceClipId) {
      return null;
    }
    return this.scrubbingCache?.getLastFrame(video, layer.sourceClipId) ?? null;
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

  private isPendingWebCodecsFrameStable(
    provider: NonNullable<Layer['source']>['webCodecsPlayer'] | undefined
  ): boolean {
    if (!provider) {
      return true;
    }

    const pendingTarget = provider.getPendingSeekTime?.();
    if (pendingTarget == null) {
      return true;
    }

    const fps = provider.getFrameRate?.() ?? 30;
    const tolerance = Math.max(1.5 / Math.max(fps, 1), 0.05);
    return Math.abs(pendingTarget - provider.currentTime) <= tolerance;
  }

  private getProviderObjectId(provider: object): number {
    const existing = this.providerIds.get(provider);
    if (existing !== undefined) {
      return existing;
    }
    const next = this.nextProviderId++;
    this.providerIds.set(provider, next);
    return next;
  }

  private getVideoProviderKey(
    layer: Layer,
    frameProvider: NonNullable<Layer['source']>['webCodecsPlayer'] | null,
    runtimeProvider: NonNullable<Layer['source']>['webCodecsPlayer'] | null
  ): string | null {
    if (!frameProvider) {
      return null;
    }
    if (
      runtimeProvider &&
      frameProvider === runtimeProvider &&
      layer.source?.runtimeSourceId &&
      layer.source.runtimeSessionKey
    ) {
      return `runtime:${layer.source.runtimeSourceId}:${layer.source.runtimeSessionKey}`;
    }
    return `provider:${this.getProviderObjectId(frameProvider as object)}`;
  }

  private canReuseLastSuccessfulVideoFrame(layerId: string, providerKey: string | null): boolean {
    return !!providerKey && this.lastSuccessfulVideoProviderKey.get(layerId) === providerKey;
  }

  private setCollectorState(
    layerId: string,
    state: 'render' | 'hold' | 'drop',
    detail?: Record<string, number | string>
  ): void {
    if (this.lastCollectorState.get(layerId) === state) {
      return;
    }
    this.lastCollectorState.set(layerId, state);
    if (state === 'hold') {
      wcPipelineMonitor.record('collector_hold', detail);
    } else if (state === 'drop') {
      wcPipelineMonitor.record('collector_drop', detail);
    }
  }

  constructor(
    device: GPUDevice,
    compositorPipeline: CompositorPipeline,
    effectsPipeline: EffectsPipeline,
    textureManager: TextureManager,
    maskTextureManager: MaskTextureManager,
    scrubbingCache: ScrubbingCache | null = null
  ) {
    this.device = device;
    this.compositorPipeline = compositorPipeline;
    this.effectsPipeline = effectsPipeline;
    this.textureManager = textureManager;
    this.maskTextureManager = maskTextureManager;
    this.scrubbingCache = scrubbingCache;
  }

  // Acquire a ping-pong texture pair from pool or create new
  private acquireTexturePair(width: number, height: number): PooledTexturePair {
    const key = `${width}x${height}`;
    let pool = this.texturePool.get(key);
    if (!pool) {
      pool = [];
      this.texturePool.set(key, pool);
    }

    // Find available pair
    for (const pair of pool) {
      if (!pair.inUse) {
        pair.inUse = true;
        return pair;
      }
    }

    // Create new pair
    const pingTexture = this.device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const pongTexture = this.device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const pair: PooledTexturePair = {
      pingTexture,
      pongTexture,
      pingView: pingTexture.createView(),
      pongView: pongTexture.createView(),
      width,
      height,
      inUse: true,
    };
    pool.push(pair);
    return pair;
  }

  // Release texture pair back to pool
  private releaseTexturePair(pair: PooledTexturePair): void {
    pair.inUse = false;
  }

  preRender(
    compositionId: string,
    nestedLayers: Layer[],
    width: number,
    height: number,
    commandEncoder: GPUCommandEncoder,
    sampler: GPUSampler,
    currentTime?: number,
    depth: number = 0,
    skipEffects = false
  ): GPUTextureView | null {
    if (depth >= MAX_NESTING_DEPTH) {
      log.warn('Max nesting depth reached in preRender', { compositionId, depth });
      return null;
    }
    // Get or create output texture
    let compTexture = this.nestedCompTextures.get(compositionId);
    if (!compTexture || compTexture.texture.width !== width || compTexture.texture.height !== height) {
      // Don't destroy immediately - let GC handle to avoid GPU conflicts

      const texture = this.device.createTexture({
        size: { width, height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      });
      compTexture = { texture, view: texture.createView() };
      this.nestedCompTextures.set(compositionId, compTexture);
    }

    // Frame caching: skip re-render if same time and layer count
    // Quantize time to ~60fps frames to avoid floating point issues
    const quantizedTime = currentTime !== undefined ? Math.round(currentTime * 60) : -1;
    const lastTime = this.lastRenderTime.get(compositionId);
    const lastCount = this.lastLayerCount.get(compositionId);

    if (quantizedTime >= 0 && lastTime === quantizedTime && lastCount === nestedLayers.length) {
      // Same frame, return cached texture
      return compTexture.view;
    }

    // Update cache tracking
    this.lastRenderTime.set(compositionId, quantizedTime);
    this.lastLayerCount.set(compositionId, nestedLayers.length);

    // Acquire ping-pong textures from pool
    const texturePair = this.acquireTexturePair(width, height);
    const nestedPingView = texturePair.pingView;
    const nestedPongView = texturePair.pongView;

    // Collect layer data (including sub-nested compositions)
    const nestedLayerData = this.collectNestedLayerData(nestedLayers, commandEncoder, sampler, depth, skipEffects);

    // Handle empty composition
    if (nestedLayerData.length === 0) {
      if (nestedLayers.length > 0) {
        // Input layers exist but none could be collected (transient decode gap)
        // Retain the existing texture which holds the last good frame
        this.releaseTexturePair(texturePair);
        return compTexture.view;
      }
      // Genuinely empty composition - clear to transparent
      const clearPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: compTexture.view,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      clearPass.end();
      this.releaseTexturePair(texturePair);
      return compTexture.view;
    }

    // Ping-pong compositing
    let readView = nestedPingView;
    let writeView = nestedPongView;

    // Clear first buffer
    const clearPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: readView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    clearPass.end();

    // Composite nested layers
    const outputAspect = width / height;
    for (const data of nestedLayerData) {
      const layer = data.layer;
      const uniformBuffer = this.compositorPipeline.getOrCreateUniformBuffer(`nested-${compositionId}-${layer.id}`);
      const sourceAspect = data.sourceWidth / data.sourceHeight;

      const maskLookupId = layer.maskClipId || layer.id;
      const maskInfo = this.maskTextureManager.getMaskInfo(maskLookupId);
      const hasMask = maskInfo.hasMask;
      const maskTextureView = maskInfo.view;

      this.compositorPipeline.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = this.compositorPipeline.getExternalCompositePipeline()!;
        bindGroup = this.compositorPipeline.createExternalCompositeBindGroup(
          sampler, readView, data.externalTexture, uniformBuffer, maskTextureView
        );
      } else if (data.textureView) {
        pipeline = this.compositorPipeline.getCompositePipeline()!;
        bindGroup = this.compositorPipeline.createCompositeBindGroup(
          sampler, readView, data.textureView, uniformBuffer, maskTextureView
        );
      } else {
        continue;
      }

      const pass = commandEncoder.beginRenderPass({
        colorAttachments: [{ view: writeView, loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6);
      pass.end();

      // Apply effects
      if (!skipEffects && layer.effects?.length && this.effectsPipeline) {
        const result = this.effectsPipeline.applyEffects(
          commandEncoder, layer.effects, sampler,
          writeView, readView, nestedPingView, nestedPongView, width, height
        );
        if (result.swapped) {
          [readView, writeView] = [writeView, readView];
        }
      }

      // Swap
      [readView, writeView] = [writeView, readView];
    }

    // Copy result to output texture using efficient GPU copy
    // Determine which texture readView came from
    const sourceTexture = readView === nestedPingView ? texturePair.pingTexture : texturePair.pongTexture;
    commandEncoder.copyTextureToTexture(
      { texture: sourceTexture },
      { texture: compTexture.texture },
      { width, height }
    );

    // Release textures back to pool
    this.releaseTexturePair(texturePair);

    return compTexture.view;
  }

  private collectNestedLayerData(
    layers: Layer[],
    commandEncoder?: GPUCommandEncoder,
    sampler?: GPUSampler,
    depth: number = 0,
    skipEffects = false
  ): LayerRenderData[] {
    const result: LayerRenderData[] = [];

    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer?.visible || !layer.source || layer.opacity === 0) continue;

      // Sub-nested composition (Level 3+)
      if (layer.source.nestedComposition && commandEncoder && sampler) {
        const nc = layer.source.nestedComposition;
        const subTextureView = this.preRender(
          nc.compositionId,
          nc.layers,
          nc.width,
          nc.height,
          commandEncoder,
          sampler,
          nc.currentTime,
          depth + 1,
          skipEffects
        );
        if (subTextureView) {
          result.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: subTextureView,
            sourceWidth: nc.width,
            sourceHeight: nc.height,
          });
        }
        continue;
      }

      // NativeDecoder (turbo mode — ImageBitmap-based)
      if (layer.source.nativeDecoder) {
        const bitmap = layer.source.nativeDecoder.getCurrentFrame();
        if (bitmap) {
          const texture = this.textureManager.createImageBitmapTexture(bitmap, layer.id);
          if (texture) {
            result.push({
              layer, isVideo: false, externalTexture: null,
              textureView: this.textureManager.getDynamicTextureView(layer.id) ?? texture.createView(),
              sourceWidth: bitmap.width, sourceHeight: bitmap.height,
            });
            continue;
          }
        }
      }

      // VideoFrame
      if (layer.source.videoFrame) {
        const frame = layer.source.videoFrame;
        const extTex = this.textureManager.importVideoTexture(frame);
        if (extTex) {
          result.push({
            layer, isVideo: true, externalTexture: extTex, textureView: null,
            sourceWidth: frame.displayWidth, sourceHeight: frame.displayHeight,
          });
          continue;
        }
      }

      const allowHtmlScrubPreview =
        (useTimelineStore.getState().isDraggingPlayhead || scrubSettleState.isPending(layer.sourceClipId)) &&
        !!layer.source.videoElement;
      const allowHtmlVideoPreview =
        !!layer.source.videoElement &&
        (!flags.useFullWebCodecsPlayback ||
          ENABLE_VISUAL_HTML_VIDEO_FALLBACK ||
          allowHtmlScrubPreview);

      if (allowHtmlVideoPreview) {
        const video = layer.source.videoElement!;
        const targetTime = this.getTargetVideoTime(layer, video);
        const isDragging = useTimelineStore.getState().isDraggingPlayhead;
        const isSettling = scrubSettleState.isPending(layer.sourceClipId);
        const isPausedSettle = !useTimelineStore.getState().isPlaying && !isDragging && isSettling;
        const lastPresentedTime = this.scrubbingCache?.getLastPresentedTime(video);
        const lastPresentedOwner = this.scrubbingCache?.getLastPresentedOwner(video);
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
          !useTimelineStore.getState().isPlaying &&
          !isDragging &&
          (!isSettling &&
            (!hasConfirmedPresentedFrame || Math.abs(lastPresentedTime - targetTime) > 0.05));
        const cacheSearchDistanceFrames = isDragging ? 12 : 6;
        const lastSameClipFrame = this.getDragHoldFrame(layer, video);
        const dragHoldFrame = isDragging
          ? lastSameClipFrame
          : (isSettling || awaitingPausedTargetFrame) && this.isFrameNearTarget(lastSameClipFrame, targetTime)
            ? lastSameClipFrame
            : null;
        const emergencyHoldFrame = isDragging ? lastSameClipFrame : dragHoldFrame;
        const safeFallback = this.getSafeLastFrameFallback(layer, video, targetTime) ?? dragHoldFrame;
        const allowLiveVideoImport = !hasPresentedOwnerMismatch && (isPausedSettle
          ? hasFreshPresentedFrame
          : !awaitingPausedTargetFrame &&
            (((!isDragging && !isSettling) || hasFreshPresentedFrame || !safeFallback)));
        const allowConfirmedFrameCaching = !hasPresentedOwnerMismatch && (isPausedSettle
          ? hasFreshPresentedFrame
          : !awaitingPausedTargetFrame &&
            (((!isDragging && !isSettling) || hasFreshPresentedFrame)));
        const captureOwnerId = allowConfirmedFrameCaching ? layer.sourceClipId : undefined;
        if ((video.seeking || awaitingPausedTargetFrame) && this.scrubbingCache) {
          const cachedView =
            this.scrubbingCache.getCachedFrame(video.src, targetTime) ??
            this.scrubbingCache.getNearestCachedFrame(video.src, targetTime, cacheSearchDistanceFrames);
          if (cachedView) {
            result.push({
              layer, isVideo: false, externalTexture: null, textureView: cachedView,
              sourceWidth: video.videoWidth, sourceHeight: video.videoHeight,
            });
            continue;
          }
          if (!allowLiveVideoImport) {
            if (safeFallback) {
              result.push({
                layer, isVideo: false, externalTexture: null, textureView: safeFallback.view,
                sourceWidth: safeFallback.width, sourceHeight: safeFallback.height,
              });
              continue;
            }
            if (emergencyHoldFrame) {
              result.push({
                layer, isVideo: false, externalTexture: null, textureView: emergencyHoldFrame.view,
                sourceWidth: emergencyHoldFrame.width, sourceHeight: emergencyHoldFrame.height,
              });
              continue;
            }
            continue;
          }
        }
        if (video.readyState >= 2) {
          if (allowLiveVideoImport) {
            const copiedFrame = getCopiedHtmlVideoPreviewFrame(
              video,
              this.scrubbingCache,
              targetTime,
              layer.sourceClipId,
              captureOwnerId
            );
            if (copiedFrame) {
              result.push({
                layer, isVideo: false, externalTexture: null, textureView: copiedFrame.view,
                sourceWidth: copiedFrame.width, sourceHeight: copiedFrame.height,
              });
              continue;
            }
          }

          const extTex = allowLiveVideoImport
            ? this.textureManager.importVideoTexture(video)
            : null;
          if (extTex) {
            if (this.scrubbingCache) {
              const now = performance.now();
              const lastCapture = this.scrubbingCache.getLastCaptureTime(video);
              if (allowConfirmedFrameCaching && now - lastCapture > 50) {
                this.scrubbingCache.captureVideoFrame(video, captureOwnerId);
                this.scrubbingCache.setLastCaptureTime(video, now);
              }
              if (allowConfirmedFrameCaching) {
                this.scrubbingCache.cacheFrameAtTime(video, targetTime);
              } else {
                if (typeof displayedTime === 'number' && Number.isFinite(displayedTime)) {
                  this.scrubbingCache.captureVideoFrameIfCloser(
                    video,
                    targetTime,
                    displayedTime,
                    layer.sourceClipId
                  );
                }
                if (typeof displayedTime === 'number' && Number.isFinite(displayedTime)) {
                  this.scrubbingCache.cacheFrameAtTime(video, displayedTime);
                }
              }
            }
            result.push({
              layer, isVideo: true, externalTexture: extTex, textureView: null,
              sourceWidth: video.videoWidth, sourceHeight: video.videoHeight,
            });
            continue;
          } else {
            log.warn('Failed to import video texture', { layerId: layer.id });
          }
        }

        const notReadyCachedFrame =
          this.scrubbingCache?.getCachedFrameEntry(video.src, targetTime) ??
          this.scrubbingCache?.getNearestCachedFrameEntry(video.src, targetTime, cacheSearchDistanceFrames);
        if (notReadyCachedFrame) {
          result.push({
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
          log.debug('Using cached frame fallback for nested video', { layerId: layer.id });
          result.push({
            layer, isVideo: false, externalTexture: null, textureView: safeFallback.view,
            sourceWidth: safeFallback.width, sourceHeight: safeFallback.height,
          });
          continue;
        }
        if (emergencyHoldFrame) {
          result.push({
            layer, isVideo: false, externalTexture: null, textureView: emergencyHoldFrame.view,
            sourceWidth: emergencyHoldFrame.width, sourceHeight: emergencyHoldFrame.height,
          });
          continue;
        }
      }

      const runtimeProvider = getRuntimeFrameProvider(layer.source, 'background');
      const clipProvider = layer.source.webCodecsPlayer?.isFullMode()
        ? layer.source.webCodecsPlayer
        : null;
      const runtimeProviderStable = this.isPendingWebCodecsFrameStable(runtimeProvider ?? undefined);
      const runtimeHasFrame =
        (runtimeProvider?.hasFrame?.() ?? false) ||
        !!runtimeProvider?.getCurrentFrame?.();
      const allowPendingScrubFrame = useTimelineStore.getState().isDraggingPlayhead;
      const shouldPreferRuntimeProvider =
        !!runtimeProvider?.isFullMode() &&
        runtimeProvider !== clipProvider &&
        runtimeProviderStable &&
        runtimeHasFrame;
      const frameProvider =
        shouldPreferRuntimeProvider
          ? runtimeProvider
          : clipProvider ?? (runtimeProvider?.isFullMode()
            ? runtimeProvider
            : null);
      const providerKey = this.getVideoProviderKey(layer, frameProvider, runtimeProvider);
      const canReuseLastFrame = this.canReuseLastSuccessfulVideoFrame(layer.id, providerKey);
      const frameProviderStable = this.isPendingWebCodecsFrameStable(frameProvider ?? undefined);
      const holdingFrame = !frameProviderStable && canReuseLastFrame;
      const canReadRuntimeFrame =
        !!layer.source.runtimeSourceId &&
        !!layer.source.runtimeSessionKey &&
        !!runtimeProvider?.isFullMode() &&
        (!frameProvider || frameProvider === runtimeProvider) &&
        (runtimeProviderStable || canReuseLastFrame || allowPendingScrubFrame);
      const runtimeFrameRead = canReadRuntimeFrame
        ? readRuntimeFrameForSource(layer.source, 'background')
        : null;
      const runtimeFrame = runtimeFrameRead?.frameHandle?.frame;
      if (
        runtimeFrame &&
        'displayWidth' in runtimeFrame &&
        'displayHeight' in runtimeFrame
      ) {
        const extTex = this.textureManager.importVideoTexture(runtimeFrame);
        if (extTex) {
          if (providerKey) {
            this.lastSuccessfulVideoProviderKey.set(layer.id, providerKey);
          }
          this.setCollectorState(layer.id, holdingFrame ? 'hold' : 'render', {
            reason: holdingFrame ? 'same_provider_pending' : 'runtime_frame',
          });
          result.push({
            layer, isVideo: true, externalTexture: extTex, textureView: null,
            sourceWidth: runtimeFrame.displayWidth, sourceHeight: runtimeFrame.displayHeight,
          });
          continue;
        }
      }

      // WebCodecs
      if (frameProvider?.isFullMode()) {
        if (!frameProviderStable && !canReuseLastFrame && !allowPendingScrubFrame) {
          this.setCollectorState(layer.id, 'drop', {
            reason: 'pending_unstable',
          });
          continue;
        }
        const frame = frameProvider.getCurrentFrame();
        if (frame) {
          const extTex = this.textureManager.importVideoTexture(frame);
          if (extTex) {
            if (providerKey) {
              this.lastSuccessfulVideoProviderKey.set(layer.id, providerKey);
            }
            this.setCollectorState(layer.id, holdingFrame ? 'hold' : 'render', {
              reason: holdingFrame ? 'same_provider_pending' : 'provider_frame',
            });
            result.push({
              layer, isVideo: true, externalTexture: extTex, textureView: null,
              sourceWidth: frame.displayWidth, sourceHeight: frame.displayHeight,
            });
            continue;
          }
          this.setCollectorState(layer.id, 'drop', {
            reason: 'import_failed',
          });
        } else {
          // WebCodecs has no frame yet - normal during decode startup
          this.setCollectorState(layer.id, 'drop', {
            reason: 'no_frame',
          });
        }
      }

      // Image
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = this.textureManager.getCachedImageTexture(img);
        if (!texture) texture = this.textureManager.createImageTexture(img) ?? undefined;
        if (texture) {
          result.push({
            layer, isVideo: false, externalTexture: null,
            textureView: this.textureManager.getImageView(texture),
            sourceWidth: img.naturalWidth, sourceHeight: img.naturalHeight,
          });
          continue;
        }
      }

      // Text
      if (layer.source.textCanvas) {
        const canvas = layer.source.textCanvas;
        const texture = this.textureManager.createCanvasTexture(canvas);
        if (texture) {
          result.push({
            layer, isVideo: false, externalTexture: null,
            textureView: this.textureManager.getImageView(texture),
            sourceWidth: canvas.width, sourceHeight: canvas.height,
          });
        }
      }
    }

    return result;
  }

  hasTexture(compositionId: string): boolean {
    return this.nestedCompTextures.has(compositionId);
  }

  getTexture(compositionId: string): NestedCompTexture | undefined {
    return this.nestedCompTextures.get(compositionId);
  }

  cleanupPendingTextures(): void {
    // No-op - textures are now managed by the pool
  }

  cleanupTexture(compositionId: string): void {
    // Just remove from map - don't destroy, let GC handle to avoid GPU conflicts
    this.nestedCompTextures.delete(compositionId);
  }

  /**
   * Cache the current main render output for a composition
   */
  cacheActiveCompOutput(compositionId: string, sourceTexture: GPUTexture, width: number, height: number): void {
    let compTexture = this.nestedCompTextures.get(compositionId);
    if (!compTexture || compTexture.texture.width !== width || compTexture.texture.height !== height) {
      // Don't destroy immediately - let GC handle to avoid GPU conflicts

      const texture = this.device.createTexture({
        size: { width, height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      compTexture = { texture, view: texture.createView() };
      this.nestedCompTextures.set(compositionId, compTexture);
    }

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyTextureToTexture(
      { texture: sourceTexture },
      { texture: compTexture.texture },
      { width, height }
    );
    this.device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Invalidate frame cache for a specific composition or all
   */
  invalidateCache(compositionId?: string): void {
    if (compositionId) {
      this.lastRenderTime.delete(compositionId);
      this.lastLayerCount.delete(compositionId);
    } else {
      this.lastRenderTime.clear();
      this.lastLayerCount.clear();
    }
  }

  destroy(): void {
    // Clear frame cache
    this.lastRenderTime.clear();
    this.lastLayerCount.clear();

    // Destroy nested comp textures
    for (const tex of this.nestedCompTextures.values()) {
      tex.texture.destroy();
    }
    this.nestedCompTextures.clear();

    // Destroy texture pool
    for (const pool of this.texturePool.values()) {
      for (const pair of pool) {
        pair.pingTexture.destroy();
        pair.pongTexture.destroy();
      }
    }
    this.texturePool.clear();
  }
}
