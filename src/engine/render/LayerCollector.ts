// Collects layer render data by importing textures from various sources

import type { Layer, LayerRenderData, DetailedStats } from '../core/types';
import type { TextureManager } from '../texture/TextureManager';
import type { ScrubbingCache } from '../texture/ScrubbingCache';
import { flags } from '../featureFlags';
import { Logger } from '../../services/logger';
import {
  getRuntimeFrameProvider,
  readRuntimeFrameForSource,
} from '../../services/mediaRuntime/runtimePlayback';
import { vfPipelineMonitor } from '../../services/vfPipelineMonitor';
import { wcPipelineMonitor } from '../../services/wcPipelineMonitor';
import { scrubSettleState } from '../../services/scrubSettleState';
import { useTimelineStore } from '../../stores/timeline';
import { getCopiedHtmlVideoPreviewFrame } from './htmlVideoPreviewFallback';

const log = Logger.create('LayerCollector');
const ENABLE_VISUAL_HTML_VIDEO_FALLBACK = false;

export interface LayerCollectorDeps {
  textureManager: TextureManager;
  scrubbingCache: ScrubbingCache | null;
  getLastVideoTime: (key: string) => number | undefined;
  setLastVideoTime: (key: string, time: number) => void;
  isExporting: boolean;
  isPlaying: boolean;
}

export class LayerCollector {
  private layerRenderData: LayerRenderData[] = [];
  private currentDecoder: DetailedStats['decoder'] = 'none';
  private currentWebCodecsInfo?: DetailedStats['webCodecsInfo'];
  private hasVideo = false;
  private lastCollectedCount = -1;
  private providerIds = new WeakMap<object, number>();
  private nextProviderId = 1;
  private videoIds = new WeakMap<HTMLVideoElement, number>();
  private nextVideoId = 1;
  private lastSuccessfulVideoProviderKey = new Map<string, string>();
  private lastCollectorState = new Map<string, 'render' | 'hold' | 'drop'>();
  private lastScrubTrace = new Map<string, string>();
  // Grace period: keep HTMLVideo scrub preview path for a few frames after
  // scrub stops, so the settle-seek has time to complete before switching
  // to the WebCodecs path (which may not have the correct frame yet).
  private scrubGraceUntil = 0;
  private static readonly SCRUB_GRACE_MS = 150; // ~9 frames at 60fps

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

  private getVideoObjectId(video: HTMLVideoElement): number {
    const existing = this.videoIds.get(video);
    if (existing !== undefined) {
      return existing;
    }
    const next = this.nextVideoId++;
    this.videoIds.set(video, next);
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

  private getLayerReuseKey(layer: Layer): string {
    return layer.sourceClipId ? `${layer.id}:${layer.sourceClipId}` : layer.id;
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

  collect(layers: Layer[], deps: LayerCollectorDeps): LayerRenderData[] {
    this.layerRenderData.length = 0;
    this.hasVideo = false;
    this.currentDecoder = 'none';
    this.currentWebCodecsInfo = undefined;

    // Process layers in reverse order (lower slots render on top)
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer?.visible || !layer.source || layer.opacity === 0) {
        continue;
      }

      try {
        const data = this.collectLayerData(layer, deps);
        if (data) {
          this.layerRenderData.push(data);
        }
      } catch (err) {
        log.warn(`Layer ${layer.id} collect error, skipping`, err);
      }
    }

    // Only log when collected count changes (not per-frame)
    if (this.layerRenderData.length !== this.lastCollectedCount) {
      log.debug(`Layers collected: ${this.layerRenderData.length}/${layers.length}`);
      this.lastCollectedCount = this.layerRenderData.length;
    }
    return this.layerRenderData;
  }

  private collectLayerData(layer: Layer, deps: LayerCollectorDeps): LayerRenderData | null {
    const source = layer.source;
    if (!source) return null;

    // Fast path: use source.type to skip irrelevant checks
    const sourceType = source.type;

    // Image sources - skip video checks entirely
    if (sourceType === 'image') {
      if (source.imageElement) {
        return this.tryImage(layer, source.imageElement, deps);
      }
      // Nested compositions are also images
      if (source.nestedComposition) {
        const nestedComp = source.nestedComposition;
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: null, // Set after pre-render
          sourceWidth: nestedComp.width,
          sourceHeight: nestedComp.height,
        };
      }
      return null;
    }

    // Text/Solid sources - skip video/image checks
    if (sourceType === 'text' || sourceType === 'solid') {
      if (source.textCanvas) {
        return this.tryTextCanvas(layer, source.textCanvas, deps);
      }
      return null;
    }

    // Video sources - check decoders in priority order
    if (sourceType === 'video') {
      // 1. Try Native Helper decoder (turbo mode) - most efficient
      if (source.nativeDecoder) {
        const bitmap = source.nativeDecoder.getCurrentFrame();
        if (bitmap) {
          const texture = deps.textureManager.createImageBitmapTexture(bitmap, layer.id);
          if (texture) {
            this.currentDecoder = 'NativeHelper';
            return {
              layer,
              isVideo: false,
              isDynamic: true,
              externalTexture: null,
              textureView: deps.textureManager.getDynamicTextureView(layer.id) ?? texture.createView(),
              sourceWidth: bitmap.width,
              sourceHeight: bitmap.height,
            };
          }
        }
      }

      // 2. Try direct VideoFrame (parallel decoder)
      if (source.videoFrame) {
        const frame = source.videoFrame;
        const extTex = deps.textureManager.importVideoTexture(frame);
        if (extTex) {
          this.currentDecoder = 'ParallelDecode';
          this.hasVideo = true;
          return {
            layer,
            isVideo: true,
            externalTexture: extTex,
            textureView: null,
            sourceWidth: frame.displayWidth,
            sourceHeight: frame.displayHeight,
          };
        }
      }

      const isDragging = useTimelineStore.getState().isDraggingPlayhead;
      const runtimeProvider = getRuntimeFrameProvider(source);
      const clipProvider = source.webCodecsPlayer?.isFullMode()
        ? source.webCodecsPlayer
        : null;
      const hasFullWebCodecsPreview =
        flags.useFullWebCodecsPlayback &&
        (!!clipProvider || !!runtimeProvider?.isFullMode());
      // Extend grace period while dragging; after drag stops,
      // keep HTML preview path briefly so the settle-seek can complete.
      if (isDragging) {
        this.scrubGraceUntil = performance.now() + LayerCollector.SCRUB_GRACE_MS;
      }
      const inScrubGrace = !isDragging && performance.now() < this.scrubGraceUntil;
      const isSettling = scrubSettleState.isPending(layer.sourceClipId);
      const allowHtmlScrubPreview =
        !hasFullWebCodecsPreview &&
        !deps.isPlaying &&
        (isDragging || inScrubGrace || isSettling) &&
        !!source.videoElement;
      const allowHtmlVideoPreview =
        !!source.videoElement &&
        (!hasFullWebCodecsPreview ||
          ENABLE_VISUAL_HTML_VIDEO_FALLBACK ||
          allowHtmlScrubPreview);

      if (allowHtmlVideoPreview) {
        return this.tryHTMLVideo(layer, source.videoElement!, deps);
      }

      // 3. Try full WebCodecs VideoFrame.
      const layerReuseKey = this.getLayerReuseKey(layer);
      const runtimeProviderStable = this.isPendingWebCodecsFrameStable(runtimeProvider ?? undefined);
      const runtimeHasFrame =
        (runtimeProvider?.hasFrame?.() ?? false) ||
        !!runtimeProvider?.getCurrentFrame?.();
      const allowPendingScrubFrame =
        !deps.isPlaying &&
        useTimelineStore.getState().isDraggingPlayhead;
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
      const canReuseLastFrame = this.canReuseLastSuccessfulVideoFrame(layerReuseKey, providerKey);
      const frameProviderStable = this.isPendingWebCodecsFrameStable(frameProvider ?? undefined);
      const holdingFrame = !frameProviderStable && canReuseLastFrame;

      const canReadRuntimeFrame =
        !!source.runtimeSourceId &&
        !!source.runtimeSessionKey &&
        !!runtimeProvider?.isFullMode() &&
        (!frameProvider || frameProvider === runtimeProvider) &&
        (runtimeProviderStable || canReuseLastFrame || allowPendingScrubFrame);

      const runtimeFrameRead = canReadRuntimeFrame
        ? readRuntimeFrameForSource(source)
        : null;
      const runtimeFrame = runtimeFrameRead?.frameHandle?.frame;

      if (
        runtimeFrame &&
        'displayWidth' in runtimeFrame &&
        'displayHeight' in runtimeFrame
      ) {
        const extTex = deps.textureManager.importVideoTexture(runtimeFrame);
        if (extTex) {
          wcPipelineMonitor.record('frame_read', {
            frameTs: runtimeFrame.timestamp,
          });
          if (providerKey) {
            this.lastSuccessfulVideoProviderKey.set(layerReuseKey, providerKey);
          }
          this.setCollectorState(layerReuseKey, holdingFrame ? 'hold' : 'render', {
            reason: holdingFrame ? 'same_provider_pending' : 'runtime_frame',
          });
          this.currentDecoder = 'WebCodecs';
          this.currentWebCodecsInfo = frameProvider?.getDebugInfo?.() ?? undefined;
          this.hasVideo = true;
          return {
            layer,
            isVideo: true,
            externalTexture: extTex,
            textureView: null,
            sourceWidth: runtimeFrame.displayWidth,
            sourceHeight: runtimeFrame.displayHeight,
          };
        }
      }

      if (frameProvider && typeof frameProvider.getCurrentFrame === 'function') {
        if (!frameProviderStable && !canReuseLastFrame && !allowPendingScrubFrame) {
          this.setCollectorState(layerReuseKey, 'drop', {
            reason: 'pending_unstable',
          });
          return null;
        }
        const frame = frameProvider.getCurrentFrame();
        if (frame) {
          const extTex = deps.textureManager.importVideoTexture(frame);
          if (extTex) {
            wcPipelineMonitor.record('frame_read', {
              frameTs: frame.timestamp,
            });
            if (providerKey) {
              this.lastSuccessfulVideoProviderKey.set(layerReuseKey, providerKey);
            }
            this.setCollectorState(layerReuseKey, holdingFrame ? 'hold' : 'render', {
              reason: holdingFrame ? 'same_provider_pending' : 'provider_frame',
            });
            this.currentDecoder = 'WebCodecs';
            this.currentWebCodecsInfo = frameProvider.getDebugInfo?.() ?? undefined;
            this.hasVideo = true;
            return {
              layer,
              isVideo: true,
              externalTexture: extTex,
              textureView: null,
              sourceWidth: frame.displayWidth,
              sourceHeight: frame.displayHeight,
            };
          }
          this.setCollectorState(layerReuseKey, 'drop', {
            reason: 'import_failed',
          });
        } else {
          this.setCollectorState(layerReuseKey, 'drop', {
            reason: 'no_frame',
          });
        }
      }

      // HTMLVideo preview is handled above when enabled.
    }

    return null;
  }

  // Track videos where importExternalTexture produces valid (non-black) frames
  // After page reload, importExternalTexture returns black until the video is played
  private videoGpuReady = new WeakSet<HTMLVideoElement>();

  private getSafeLastFrameFallback(
    layer: Layer,
    video: HTMLVideoElement,
    deps: LayerCollectorDeps,
    targetTime: number
  ) {
    const isDragging = useTimelineStore.getState().isDraggingPlayhead;
    const tolerance = video.seeking || isDragging ? 0.35 : 0.2;
    return deps.scrubbingCache?.getLastFrameNearTime(video, targetTime, tolerance, layer.sourceClipId) ?? null;
  }

  private getDragHoldFrame(layer: Layer, video: HTMLVideoElement, deps: LayerCollectorDeps) {
    if (!layer.sourceClipId) {
      return null;
    }
    return deps.scrubbingCache?.getLastFrame(video, layer.sourceClipId) ?? null;
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

  private getTargetVideoTime(layer: Layer, video: HTMLVideoElement): number {
    return layer.source?.mediaTime ?? video.currentTime;
  }

  private traceScrubPath(
    layer: Layer,
    path: string,
    video: HTMLVideoElement,
    targetTime: number,
    lastPresentedTime?: number
  ): void {
    if (!useTimelineStore.getState().isDraggingPlayhead) {
      return;
    }
    const traceKey = this.getLayerReuseKey(layer);
    const signature = [
      path,
      Math.round(targetTime * 1000),
      Math.round(video.currentTime * 1000),
      Math.round((lastPresentedTime ?? -1) * 1000),
      video.seeking ? '1' : '0',
    ].join(':');
    if (this.lastScrubTrace.get(traceKey) === signature) {
      return;
    }
    this.lastScrubTrace.set(traceKey, signature);
    vfPipelineMonitor.record('vf_scrub_path', {
      clipId: layer.sourceClipId ?? layer.id,
      path,
      targetTimeMs: Math.round(targetTime * 1000),
      currentTimeMs: Math.round(video.currentTime * 1000),
      presentedTimeMs: Math.round((lastPresentedTime ?? -1) * 1000),
      seeking: video.seeking ? 'true' : 'false',
      readyState: video.readyState,
    });
  }

  private tryHTMLVideo(layer: Layer, video: HTMLVideoElement, deps: LayerCollectorDeps): LayerRenderData | null {
    const videoKey = `video:${this.getVideoObjectId(video)}`;

    log.debug(`tryHTMLVideo: readyState=${video.readyState}, seeking=${video.seeking}, videoWidth=${video.videoWidth}, videoHeight=${video.videoHeight}`);

    if (video.readyState >= 2) {
      const currentTime = video.currentTime;
      const targetTime = this.getTargetVideoTime(layer, video);
      const isDragging = useTimelineStore.getState().isDraggingPlayhead;
      const isSettling = scrubSettleState.isPending(layer.sourceClipId);
      const isPausedSettle = !deps.isPlaying && !isDragging && isSettling;
      const lastPresentedTime = deps.scrubbingCache?.getLastPresentedTime(video);
      const lastPresentedOwner = deps.scrubbingCache?.getLastPresentedOwner(video);
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
        !deps.isPlaying &&
        !isDragging &&
        (!isSettling &&
          (!hasConfirmedPresentedFrame || Math.abs(lastPresentedTime - targetTime) > 0.05));
      const cacheSearchDistanceFrames = isDragging ? 12 : 6;
      const lastSameClipFrame = this.getDragHoldFrame(layer, video, deps);
      const dragHoldFrame = isDragging
        ? lastSameClipFrame
        : (isSettling || awaitingPausedTargetFrame) && this.isFrameNearTarget(lastSameClipFrame, targetTime)
          ? lastSameClipFrame
        : null;
      const emergencyHoldFrame = isDragging
        ? lastSameClipFrame
        : dragHoldFrame;
      const safeFallback = this.getSafeLastFrameFallback(layer, video, deps, targetTime) ?? dragHoldFrame;
      const allowLiveVideoImport = !hasPresentedOwnerMismatch && (isPausedSettle
        ? hasFreshPresentedFrame
        : !awaitingPausedTargetFrame &&
          (((!isDragging && !isSettling) || hasFreshPresentedFrame || !safeFallback)));
      const allowConfirmedFrameCaching = !hasPresentedOwnerMismatch && (isPausedSettle
        ? hasFreshPresentedFrame
        : !awaitingPausedTargetFrame &&
          (((!isDragging && !isSettling) || hasFreshPresentedFrame)));
      const captureOwnerId = allowConfirmedFrameCaching ? layer.sourceClipId : undefined;

      // Keep using the live HTML video frame even when currentTime is unchanged.
      // Re-importing the decoded frame is more reliable than holding onto a
      // copied fallback texture across seeks and clip switches.
      // Falling through to importExternalTexture with an unchanged currentTime is
      // harmless — it just re-imports the same decoded frame.

      // If video is seeking, try per-time scrubbing cache first (exact frame for this position),
      // then fall back to generic last-frame cache
      if ((video.seeking || awaitingPausedTargetFrame) && !deps.isExporting) {
        // Try per-time cache: if we've visited this position before, show the exact frame
        const cachedFrame =
          deps.scrubbingCache?.getCachedFrameEntry(video.src, targetTime) ??
          deps.scrubbingCache?.getNearestCachedFrameEntry(video.src, targetTime, cacheSearchDistanceFrames);
        if (cachedFrame) {
          this.traceScrubPath(layer, 'scrub-cache', video, targetTime, lastPresentedTime);
          this.currentDecoder = 'HTMLVideo(scrub-cache)';
          return {
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: cachedFrame.view,
            sourceWidth: video.videoWidth,
            sourceHeight: video.videoHeight,
            displayedMediaTime: cachedFrame.mediaTime,
            targetMediaTime: targetTime,
            previewPath: 'scrub-cache',
          };
        }
        // Only hold a copied fallback if it was captured for essentially the
        // same media time. Otherwise we risk flashing a frame from the previous
        // clip/seek position while the new seek is still decoding.
        if (safeFallback) {
          this.traceScrubPath(layer, 'seeking-cache', video, targetTime, lastPresentedTime);
          this.currentDecoder = 'HTMLVideo(seeking-cache)';
          return {
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: safeFallback.view,
            sourceWidth: safeFallback.width,
            sourceHeight: safeFallback.height,
            displayedMediaTime: safeFallback.mediaTime,
            targetMediaTime: targetTime,
            previewPath: 'seeking-cache',
          };
        }
        if (emergencyHoldFrame) {
          this.traceScrubPath(layer, 'emergency-hold', video, targetTime, lastPresentedTime);
          this.currentDecoder = 'HTMLVideo(cached)';
          return {
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: emergencyHoldFrame.view,
            sourceWidth: emergencyHoldFrame.width,
            sourceHeight: emergencyHoldFrame.height,
            displayedMediaTime: emergencyHoldFrame.mediaTime,
            targetMediaTime: targetTime,
            previewPath: 'emergency-hold',
          };
        }
      }

      // After page reload, importExternalTexture returns a valid GPUExternalTexture
      // but the frame data is black/empty because the GPU decoder hasn't presented a frame.
      // syncClipVideo triggers a brief play() to activate the GPU surface.
      // Once the video plays, the next render sees it playing → sets videoGpuReady.
      if (!this.videoGpuReady.has(video) && !deps.isExporting) {
        // Video is actively playing (warmup in progress) — GPU decoder is now active
        if (!video.paused && !video.seeking) {
          this.videoGpuReady.add(video);
        } else if (!deps.isPlaying) {
          // When paused: use cached frame if available, or skip rendering
          if (safeFallback) {
            this.traceScrubPath(layer, 'gpu-cached', video, targetTime, lastPresentedTime);
            deps.setLastVideoTime(videoKey, currentTime);
            this.currentDecoder = 'HTMLVideo(cached)';
            return {
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: safeFallback.view,
              sourceWidth: safeFallback.width,
              sourceHeight: safeFallback.height,
              displayedMediaTime: safeFallback.mediaTime,
              targetMediaTime: targetTime,
              previewPath: 'gpu-cached',
            };
          }
          this.traceScrubPath(layer, 'gpu-not-ready-drop', video, targetTime, lastPresentedTime);
          // No cached frame yet — warmup not started or still in progress
          return null;
        }
        // During playback: fall through to importExternalTexture anyway.
        // May show a brief flash but avoids black frame at cut boundaries.
        // Proactive warmup should prevent this in most cases.
      }

      if (allowLiveVideoImport) {
        const copiedFrame = getCopiedHtmlVideoPreviewFrame(
          video,
          deps.scrubbingCache,
          targetTime,
          layer.sourceClipId,
          captureOwnerId
        );
        if (copiedFrame) {
          this.traceScrubPath(layer, 'copied-preview', video, targetTime, lastPresentedTime);
          deps.setLastVideoTime(videoKey, currentTime);
          this.currentDecoder = 'HTMLVideo';
          this.hasVideo = true;
          return {
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: copiedFrame.view,
            sourceWidth: copiedFrame.width,
            sourceHeight: copiedFrame.height,
            displayedMediaTime: copiedFrame.mediaTime ?? displayedTime,
            targetMediaTime: targetTime,
            previewPath: 'copied-preview',
          };
        }
      }

      // Import external texture (zero-copy GPU path)
      const extTex = allowLiveVideoImport
        ? deps.textureManager.importVideoTexture(video)
        : null;
      if (extTex) {
        deps.setLastVideoTime(videoKey, currentTime);
        // Cache frame for pause/seek fallback — skip during playback to save GPU bandwidth.
        // With 4+ videos, the GPU copies (copyExternalImageToTexture per video at 20fps)
        // waste significant bandwidth that's needed for rendering + effects.
        if (!deps.isPlaying && allowLiveVideoImport) {
          const now = performance.now();
          const lastCapture = deps.scrubbingCache?.getLastCaptureTime(video) || 0;
          if (allowConfirmedFrameCaching && now - lastCapture > 50) {
            deps.scrubbingCache?.captureVideoFrame(video, captureOwnerId);
            deps.scrubbingCache?.setLastCaptureTime(video, now);
          }

          // Populate per-time scrubbing cache: store this frame indexed by video time
          if (allowConfirmedFrameCaching) {
            deps.scrubbingCache?.cacheFrameAtTime(video, targetTime);
          } else {
            if (typeof displayedTime === 'number' && Number.isFinite(displayedTime)) {
              deps.scrubbingCache?.captureVideoFrameIfCloser(
                video,
                targetTime,
                displayedTime,
                layer.sourceClipId
              );
            }
            if (typeof displayedTime === 'number' && Number.isFinite(displayedTime)) {
              deps.scrubbingCache?.cacheFrameAtTime(video, displayedTime);
            }
          }
        }

        this.traceScrubPath(layer, 'live-import', video, targetTime, lastPresentedTime);
        this.currentDecoder = 'HTMLVideo';
        this.hasVideo = true;
        return {
          layer,
          isVideo: true,
          externalTexture: extTex,
          textureView: null,
          sourceWidth: video.videoWidth,
          sourceHeight: video.videoHeight,
          displayedMediaTime: displayedTime,
          targetMediaTime: targetTime,
          previewPath: 'live-import',
        };
      }

      // Fallback to cache
      if (safeFallback) {
        this.traceScrubPath(layer, 'final-cache', video, targetTime, lastPresentedTime);
        this.currentDecoder = 'HTMLVideo(cached)';
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: safeFallback.view,
          sourceWidth: safeFallback.width,
          sourceHeight: safeFallback.height,
          displayedMediaTime: safeFallback.mediaTime,
          targetMediaTime: targetTime,
          previewPath: 'final-cache',
        };
      }
      if (emergencyHoldFrame) {
        this.traceScrubPath(layer, 'emergency-hold', video, targetTime, lastPresentedTime);
        this.currentDecoder = 'HTMLVideo(cached)';
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: emergencyHoldFrame.view,
          sourceWidth: emergencyHoldFrame.width,
          sourceHeight: emergencyHoldFrame.height,
          displayedMediaTime: emergencyHoldFrame.mediaTime,
          targetMediaTime: targetTime,
          previewPath: 'emergency-hold',
        };
      }
      this.traceScrubPath(layer, 'final-drop', video, targetTime, lastPresentedTime);
    } else {
      // Video not ready - try cache
      const targetTime = this.getTargetVideoTime(layer, video);
      const isDragging = useTimelineStore.getState().isDraggingPlayhead;
      const cacheSearchDistanceFrames = isDragging ? 12 : 6;
      const isSettling = scrubSettleState.isPending(layer.sourceClipId);
      const dragHoldFrame = isSettling
        ? this.getDragHoldFrame(layer, video, deps)
        : null;
      const emergencyHoldFrame = isDragging
        ? this.getDragHoldFrame(layer, video, deps)
        : dragHoldFrame;
      const safeFallback = this.getSafeLastFrameFallback(layer, video, deps, targetTime) ?? dragHoldFrame;
      const cachedFrame =
        deps.scrubbingCache?.getCachedFrameEntry(video.src, targetTime) ??
        deps.scrubbingCache?.getNearestCachedFrameEntry(video.src, targetTime, cacheSearchDistanceFrames);
      if (cachedFrame) {
        this.traceScrubPath(layer, 'not-ready-scrub-cache', video, targetTime, deps.scrubbingCache?.getLastPresentedTime(video));
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: cachedFrame.view,
          sourceWidth: video.videoWidth,
          sourceHeight: video.videoHeight,
          displayedMediaTime: cachedFrame.mediaTime,
          targetMediaTime: targetTime,
          previewPath: 'not-ready-scrub-cache',
        };
      }
      if (safeFallback) {
        this.traceScrubPath(layer, 'not-ready-cache', video, targetTime, deps.scrubbingCache?.getLastPresentedTime(video));
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: safeFallback.view,
          sourceWidth: safeFallback.width,
          sourceHeight: safeFallback.height,
          displayedMediaTime: safeFallback.mediaTime,
          targetMediaTime: targetTime,
          previewPath: 'not-ready-cache',
        };
      }
      if (emergencyHoldFrame) {
        this.traceScrubPath(layer, 'emergency-hold', video, targetTime, deps.scrubbingCache?.getLastPresentedTime(video));
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: emergencyHoldFrame.view,
          sourceWidth: emergencyHoldFrame.width,
          sourceHeight: emergencyHoldFrame.height,
          displayedMediaTime: emergencyHoldFrame.mediaTime,
          targetMediaTime: targetTime,
          previewPath: 'emergency-hold',
        };
      }
      this.traceScrubPath(layer, 'not-ready-drop', video, targetTime, deps.scrubbingCache?.getLastPresentedTime(video));
    }

    return null;
  }

  private tryImage(layer: Layer, img: HTMLImageElement, deps: LayerCollectorDeps): LayerRenderData | null {
    let texture = deps.textureManager.getCachedImageTexture(img);
    if (!texture) {
      texture = deps.textureManager.createImageTexture(img) ?? undefined;
    }
    if (texture) {
      return {
        layer,
        isVideo: false,
        externalTexture: null,
        textureView: deps.textureManager.getImageView(texture),
        sourceWidth: img.naturalWidth,
        sourceHeight: img.naturalHeight,
      };
    }
    return null;
  }

  private tryTextCanvas(layer: Layer, canvas: HTMLCanvasElement, deps: LayerCollectorDeps): LayerRenderData | null {
    const texture = deps.textureManager.createCanvasTexture(canvas);
    if (texture) {
      return {
        layer,
        isVideo: false,
        externalTexture: null,
        textureView: deps.textureManager.getImageView(texture),
        sourceWidth: canvas.width,
        sourceHeight: canvas.height,
      };
    }
    return null;
  }

  getDecoder(): DetailedStats['decoder'] {
    return this.currentDecoder;
  }

  getWebCodecsInfo(): DetailedStats['webCodecsInfo'] {
    return this.currentWebCodecsInfo;
  }

  hasActiveVideo(): boolean {
    return this.hasVideo;
  }

  isVideoGpuReady(video: HTMLVideoElement): boolean {
    return this.videoGpuReady.has(video);
  }

  markVideoGpuReady(video: HTMLVideoElement): void {
    this.videoGpuReady.add(video);
  }

  resetVideoGpuReady(video: HTMLVideoElement): void {
    this.videoGpuReady.delete(video);
  }
}
