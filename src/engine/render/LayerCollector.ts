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
import { wcPipelineMonitor } from '../../services/wcPipelineMonitor';
import { useTimelineStore } from '../../stores/timeline';

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
      // Extend grace period while dragging; after drag stops,
      // keep HTML preview path briefly so the settle-seek can complete.
      if (isDragging) {
        this.scrubGraceUntil = performance.now() + LayerCollector.SCRUB_GRACE_MS;
      }
      const inScrubGrace = !isDragging && performance.now() < this.scrubGraceUntil;
      const allowHtmlScrubPreview =
        !deps.isPlaying &&
        (isDragging || inScrubGrace) &&
        !!source.videoElement;
      const allowHtmlVideoPreview =
        !!source.videoElement &&
        (!flags.useFullWebCodecsPlayback ||
          ENABLE_VISUAL_HTML_VIDEO_FALLBACK ||
          allowHtmlScrubPreview);

      if (allowHtmlVideoPreview) {
        return this.tryHTMLVideo(layer, source.videoElement!, deps);
      }

      // 3. Try full WebCodecs VideoFrame.
      const runtimeProvider = getRuntimeFrameProvider(source);
      const clipProvider = source.webCodecsPlayer?.isFullMode()
        ? source.webCodecsPlayer
        : null;
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
      const canReuseLastFrame = this.canReuseLastSuccessfulVideoFrame(layer.id, providerKey);
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
            this.lastSuccessfulVideoProviderKey.set(layer.id, providerKey);
          }
          this.setCollectorState(layer.id, holdingFrame ? 'hold' : 'render', {
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
          this.setCollectorState(layer.id, 'drop', {
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
              this.lastSuccessfulVideoProviderKey.set(layer.id, providerKey);
            }
            this.setCollectorState(layer.id, holdingFrame ? 'hold' : 'render', {
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
          this.setCollectorState(layer.id, 'drop', {
            reason: 'import_failed',
          });
        } else {
          this.setCollectorState(layer.id, 'drop', {
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

  private tryHTMLVideo(layer: Layer, video: HTMLVideoElement, deps: LayerCollectorDeps): LayerRenderData | null {
    const videoKey = `video:${this.getVideoObjectId(video)}`;

    log.debug(`tryHTMLVideo: readyState=${video.readyState}, seeking=${video.seeking}, videoWidth=${video.videoWidth}, videoHeight=${video.videoHeight}`);

    if (video.readyState >= 2) {
      const lastTime = deps.getLastVideoTime(videoKey);
      const currentTime = video.currentTime;
      const videoTimeChanged = lastTime === undefined || Math.abs(currentTime - lastTime) > 0.001;

      // Use cache for paused videos (skip during export and during playback).
      // During playback the cache may contain a stale frame from before play started
      // (captureVideoFrame is skipped while playing to save GPU bandwidth).
      // Falling through to importExternalTexture with an unchanged currentTime is
      // harmless — it just re-imports the same decoded frame.
      if (!videoTimeChanged && !deps.isExporting && !deps.isPlaying) {
        const lastFrame = deps.scrubbingCache?.getLastFrame(video);
        if (lastFrame) {
          this.currentDecoder = 'HTMLVideo(paused-cache)';
          return {
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: lastFrame.view,
            sourceWidth: lastFrame.width,
            sourceHeight: lastFrame.height,
          };
        }
      }

      // If video is seeking, try per-time scrubbing cache first (exact frame for this position),
      // then fall back to generic last-frame cache
      if (video.seeking && !deps.isExporting) {
        // Try per-time cache: if we've visited this position before, show the exact frame
        const cachedView = deps.scrubbingCache?.getCachedFrame(video.src, currentTime);
        if (cachedView) {
          this.currentDecoder = 'HTMLVideo(scrub-cache)';
          return {
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: cachedView,
            sourceWidth: video.videoWidth,
            sourceHeight: video.videoHeight,
          };
        }
        // Fall back to generic last-frame cache
        const lastFrame = deps.scrubbingCache?.getLastFrame(video);
        if (lastFrame) {
          this.currentDecoder = 'HTMLVideo(seeking-cache)';
          return {
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: lastFrame.view,
            sourceWidth: lastFrame.width,
            sourceHeight: lastFrame.height,
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
          const cachedFrame = deps.scrubbingCache?.getLastFrame(video);
          if (cachedFrame) {
            deps.setLastVideoTime(videoKey, currentTime);
            this.currentDecoder = 'HTMLVideo(cached)';
            return {
              layer,
              isVideo: false,
              externalTexture: null,
              textureView: cachedFrame.view,
              sourceWidth: cachedFrame.width,
              sourceHeight: cachedFrame.height,
            };
          }
          // No cached frame yet — warmup not started or still in progress
          return null;
        }
        // During playback: fall through to importExternalTexture anyway.
        // May show a brief flash but avoids black frame at cut boundaries.
        // Proactive warmup should prevent this in most cases.
      }

      // Import external texture (zero-copy GPU path)
      const extTex = deps.textureManager.importVideoTexture(video);
      if (extTex) {
        deps.setLastVideoTime(videoKey, currentTime);

        // Cache frame for pause/seek fallback — skip during playback to save GPU bandwidth.
        // With 4+ videos, the GPU copies (copyExternalImageToTexture per video at 20fps)
        // waste significant bandwidth that's needed for rendering + effects.
        if (!deps.isPlaying) {
          const now = performance.now();
          const lastCapture = deps.scrubbingCache?.getLastCaptureTime(video) || 0;
          if (now - lastCapture > 50) {
            deps.scrubbingCache?.captureVideoFrame(video);
            deps.scrubbingCache?.setLastCaptureTime(video, now);
          }

          // Populate per-time scrubbing cache: store this frame indexed by video time
          deps.scrubbingCache?.cacheFrameAtTime(video, currentTime);
        }

        this.currentDecoder = 'HTMLVideo';
        this.hasVideo = true;
        return {
          layer,
          isVideo: true,
          externalTexture: extTex,
          textureView: null,
          sourceWidth: video.videoWidth,
          sourceHeight: video.videoHeight,
        };
      }

      // Fallback to cache
      const lastFrame = deps.scrubbingCache?.getLastFrame(video);
      if (lastFrame) {
        this.currentDecoder = 'HTMLVideo(cached)';
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: lastFrame.view,
          sourceWidth: lastFrame.width,
          sourceHeight: lastFrame.height,
        };
      }
    } else {
      // Video not ready - try cache
      const lastFrame = deps.scrubbingCache?.getLastFrame(video);
      if (lastFrame) {
        return {
          layer,
          isVideo: false,
          externalTexture: null,
          textureView: lastFrame.view,
          sourceWidth: lastFrame.width,
          sourceHeight: lastFrame.height,
        };
      }
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

  resetVideoGpuReady(video: HTMLVideoElement): void {
    this.videoGpuReady.delete(video);
  }
}
