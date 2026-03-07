// Collects layer render data by importing textures from various sources

import type { Layer, LayerRenderData, DetailedStats } from '../core/types';
import type { TextureManager } from '../texture/TextureManager';
import type { ScrubbingCache } from '../texture/ScrubbingCache';
import { Logger } from '../../services/logger';
import { wcPipelineMonitor } from '../../services/wcPipelineMonitor';
import { vfPipelineMonitor } from '../../services/vfPipelineMonitor';

const log = Logger.create('LayerCollector');

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

      // 3. Try WebCodecs VideoFrame
      // Skip for videos that haven't been played yet — after page reload,
      // VideoFrame from a never-played video produces black/empty frames.
      // Fall through to tryHTMLVideo which has a canvas-based fallback.
      if (source.webCodecsPlayer && typeof source.webCodecsPlayer.getCurrentFrame === 'function') {
        // Full mode: frames come from VideoDecoder, no GPU-ready guard needed
        // Simple mode: frames come from HTMLVideoElement, needs GPU surface ready
        const needsGpuReady = source.webCodecsPlayer.isSimpleMode();
        if (!needsGpuReady || !source.videoElement || this.videoGpuReady.has(source.videoElement)) {
          const frame = source.webCodecsPlayer.getCurrentFrame();
          if (frame) {
            const extTex = deps.textureManager.importVideoTexture(frame);
            if (extTex) {
              if (source.webCodecsPlayer.isFullMode()) {
                wcPipelineMonitor.record('frame_read', {
                  frameTs: frame.timestamp,
                });
              } else {
                vfPipelineMonitor.record('vf_read', {
                  frameTs: frame.timestamp,
                });
              }
              this.currentDecoder = source.webCodecsPlayer.isFullMode()
                ? 'WebCodecs'       // Echtes WebCodecs (VideoDecoder API)
                : 'HTMLVideo(VF)';  // VideoFrame-Wrapper um HTMLVideo
              this.currentWebCodecsInfo = source.webCodecsPlayer.getDebugInfo() ?? undefined;
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
        }
      }

      // 4. HTMLVideoElement fallback — DISABLED for testing.
      // In full WebCodecs mode, we want to see pure WebCodecs output
      // to identify timing/decode issues without HTMLVideo masking them.
      if (source.videoElement && !source.webCodecsPlayer?.isFullMode()) {
        return this.tryHTMLVideo(layer, source.videoElement, deps);
      }
    }

    return null;
  }

  // Track videos where importExternalTexture produces valid (non-black) frames
  // After page reload, importExternalTexture returns black until the video is played
  private videoGpuReady = new WeakSet<HTMLVideoElement>();

  private tryHTMLVideo(layer: Layer, video: HTMLVideoElement, deps: LayerCollectorDeps): LayerRenderData | null {
    const videoKey = video.src || layer.id;

    log.debug(`tryHTMLVideo: readyState=${video.readyState}, seeking=${video.seeking}, videoWidth=${video.videoWidth}, videoHeight=${video.videoHeight}`);

    if (video.readyState >= 2) {
      const lastTime = deps.getLastVideoTime(videoKey);
      const currentTime = video.currentTime;
      const videoTimeChanged = lastTime === undefined || Math.abs(currentTime - lastTime) > 0.001;

      // Use cache for paused videos (skip during export)
      if (!videoTimeChanged && !deps.isExporting) {
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
