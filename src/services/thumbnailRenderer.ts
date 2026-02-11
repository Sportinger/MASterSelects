// ThumbnailRendererService - Generates WebGPU-rendered thumbnails for nested compositions
// Shows all layers with effects, not just the first video

import { Logger } from './logger';
import { compositionRenderer } from './compositionRenderer';
import type { Layer } from '../types';
import { useMediaStore } from '../stores/mediaStore';

const log = Logger.create('ThumbnailRenderer');

// Minimal subset of WebGPU resources needed for thumbnail rendering
interface ThumbnailResources {
  device: GPUDevice;
  sampler: GPUSampler;
  compositorPipeline: import('../engine/pipeline/CompositorPipeline').CompositorPipeline;
  effectsPipeline: import('../effects/EffectsPipeline').EffectsPipeline;
  outputPipeline: import('../engine/pipeline/OutputPipeline').OutputPipeline;
  textureManager: import('../engine/texture/TextureManager').TextureManager;
  maskTextureManager: import('../engine/texture/MaskTextureManager').MaskTextureManager;
}

interface ThumbnailOptions {
  count?: number;
  width?: number;
  height?: number;
  /** Pre-calculated boundary positions (normalized 0-1) for segment-based thumbnails */
  boundaries?: number[];
}

const DEFAULT_OPTIONS: Required<ThumbnailOptions> = {
  count: 10,
  width: 160,
  height: 90,
  boundaries: [],
};

class ThumbnailRendererService {
  private resources: ThumbnailResources | null = null;
  private isInitialized = false;
  private initPromise: Promise<boolean> | null = null;

  // Ping-pong textures for compositing
  private pingTexture: GPUTexture | null = null;
  private pongTexture: GPUTexture | null = null;
  private pingView: GPUTextureView | null = null;
  private pongView: GPUTextureView | null = null;
  private currentWidth = 0;
  private currentHeight = 0;

  // OffscreenCanvas for capturing
  private canvas: OffscreenCanvas | null = null;
  private canvasContext: GPUCanvasContext | null = null;

  async initialize(): Promise<boolean> {
    // Singleton init pattern
    if (this.isInitialized) return true;
    if (this.initPromise) return this.initPromise;

    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<boolean> {
    try {
      // Check WebGPU support
      if (!navigator.gpu) {
        log.warn('WebGPU not supported - thumbnail renderer disabled');
        return false;
      }

      // Request adapter and device
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'low-power', // Use efficiency GPU for background work
      });
      if (!adapter) {
        log.warn('No GPU adapter available');
        return false;
      }

      const device = await adapter.requestDevice();

      // Import pipeline classes
      const { CompositorPipeline } = await import('../engine/pipeline/CompositorPipeline');
      const { EffectsPipeline } = await import('../effects/EffectsPipeline');
      const { OutputPipeline } = await import('../engine/pipeline/OutputPipeline');
      const { TextureManager } = await import('../engine/texture/TextureManager');
      const { MaskTextureManager } = await import('../engine/texture/MaskTextureManager');

      // Create pipelines
      const compositorPipeline = new CompositorPipeline(device);
      const effectsPipeline = new EffectsPipeline(device);
      const outputPipeline = new OutputPipeline(device);

      await compositorPipeline.createPipelines();
      await effectsPipeline.createPipelines();
      await outputPipeline.createPipeline();

      // Create sampler
      const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });

      // Create texture managers
      const textureManager = new TextureManager(device);
      const maskTextureManager = new MaskTextureManager(device);

      this.resources = {
        device,
        sampler,
        compositorPipeline,
        effectsPipeline,
        outputPipeline,
        textureManager,
        maskTextureManager,
      };

      this.isInitialized = true;
      log.info('ThumbnailRenderer initialized');
      return true;
    } catch (e) {
      log.error('Failed to initialize ThumbnailRenderer', e);
      this.initPromise = null;
      return false;
    }
  }

  private ensurePingPongTextures(width: number, height: number): boolean {
    if (!this.resources) return false;

    if (this.pingTexture && this.currentWidth === width && this.currentHeight === height) {
      return true;
    }

    // Cleanup old textures
    this.pingTexture?.destroy();
    this.pongTexture?.destroy();

    const { device } = this.resources;

    this.pingTexture = device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    this.pongTexture = device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    this.pingView = this.pingTexture.createView();
    this.pongView = this.pongTexture.createView();
    this.currentWidth = width;
    this.currentHeight = height;

    return true;
  }

  private ensureCanvas(width: number, height: number): boolean {
    if (!this.resources) return false;

    if (this.canvas && this.canvas.width === width && this.canvas.height === height) {
      return true;
    }

    const { device } = this.resources;

    this.canvas = new OffscreenCanvas(width, height);
    const ctx = this.canvas.getContext('webgpu');
    if (!ctx) {
      log.error('Failed to get WebGPU context from OffscreenCanvas');
      return false;
    }

    const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({
      device,
      format: preferredFormat,
      alphaMode: 'premultiplied',
    });

    this.canvasContext = ctx;
    return true;
  }

  async generateCompositionThumbnails(
    compositionId: string,
    duration: number,
    options?: ThumbnailOptions
  ): Promise<string[]> {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    const { count, width, height, boundaries } = opts;

    // Initialize if needed
    if (!await this.initialize()) {
      log.warn('ThumbnailRenderer not available, returning empty thumbnails');
      return [];
    }

    // Prepare the composition sources
    log.info(`Preparing composition for thumbnails: ${compositionId}`);
    const prepared = await compositionRenderer.prepareComposition(compositionId);
    if (!prepared) {
      log.warn(`Failed to prepare composition ${compositionId}`);
      return [];
    }
    log.info(`Composition prepared: ${compositionId}`);

    // Ensure textures and canvas
    if (!this.ensurePingPongTextures(width, height)) {
      return [];
    }
    if (!this.ensureCanvas(width, height)) {
      return [];
    }

    // Get sample times - use boundaries if provided, otherwise content-aware
    log.info(`Thumbnail generation - boundaries provided: ${boundaries?.length ?? 0}, duration: ${duration}`);
    if (boundaries && boundaries.length > 0) {
      log.info(`Using segment-based sampling with boundaries: ${boundaries.map(b => (b * 100).toFixed(1) + '%').join(', ')}`);
    }
    const sampleTimes = boundaries && boundaries.length > 0
      ? this.getSegmentSampleTimes(boundaries, duration, count)
      : this.getContentAwareSampleTimes(compositionId, duration, count);

    const thumbnails: string[] = [];

    // Generate frames at sample positions
    for (const time of sampleTimes) {
      // Clamp to slightly before end to avoid seek issues
      const clampedTime = Math.min(time, duration - 0.01);

      try {
        const dataUrl = await this.renderFrameAt(compositionId, clampedTime, width, height);
        if (dataUrl) {
          thumbnails.push(dataUrl);
        }
      } catch (e) {
        log.warn(`Failed to render thumbnail at ${clampedTime}s`, e);
      }

      // Yield to main thread between frames
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    log.debug(`Generated ${thumbnails.length} thumbnails for composition ${compositionId}`);
    return thumbnails;
  }

  /**
   * Get sample times based on pre-calculated segment boundaries.
   * Each segment between boundaries gets at least one thumbnail at its midpoint.
   */
  private getSegmentSampleTimes(boundaries: number[], duration: number, targetCount: number): number[] {
    // Build full boundary list including 0 and 1
    const allBoundaries = [0, ...boundaries.filter(b => b > 0 && b < 1), 1].sort((a, b) => a - b);

    // Remove duplicates (boundaries very close to each other)
    const uniqueBoundaries: number[] = [allBoundaries[0]];
    for (let i = 1; i < allBoundaries.length; i++) {
      if (allBoundaries[i] - uniqueBoundaries[uniqueBoundaries.length - 1] > 0.01) {
        uniqueBoundaries.push(allBoundaries[i]);
      }
    }

    log.debug(`Segment boundaries (${uniqueBoundaries.length}):`, uniqueBoundaries.map(b => (b * 100).toFixed(1) + '%'));

    // Calculate segment count
    const segmentCount = uniqueBoundaries.length - 1;
    if (segmentCount <= 0) {
      return this.getEvenSampleTimes(duration, targetCount);
    }

    // Calculate how many samples per segment to reach target count
    const samplesPerSegment = Math.max(1, Math.ceil(targetCount / segmentCount));
    const times: number[] = [];

    for (let i = 0; i < segmentCount; i++) {
      const segStart = uniqueBoundaries[i];
      const segEnd = uniqueBoundaries[i + 1];
      const segDuration = segEnd - segStart;

      // Generate samples within this segment
      for (let j = 0; j < samplesPerSegment; j++) {
        // Position samples evenly within the segment, avoiding edges
        const t = samplesPerSegment > 1
          ? (j + 0.5) / samplesPerSegment
          : 0.5; // Single sample at midpoint

        const normalizedTime = segStart + t * segDuration;
        times.push(normalizedTime * duration);
      }
    }

    // Limit to target count if we have too many
    if (times.length > targetCount) {
      const step = times.length / targetCount;
      const sampled: number[] = [];
      for (let i = 0; i < targetCount; i++) {
        sampled.push(times[Math.floor(i * step)]);
      }
      log.info(`Segment-based sample times (${sampled.length}):`, sampled.map(t => t.toFixed(2) + 's'));
      return sampled;
    }

    log.info(`Segment-based sample times (${times.length}):`, times.map(t => t.toFixed(2) + 's'));
    return times;
  }

  /**
   * Get content-aware sample times that reflect clip boundaries.
   * This ensures thumbnails show where clips start/end and where gaps exist.
   */
  private getContentAwareSampleTimes(compositionId: string, duration: number, count: number): number[] {
    // Get composition's clip layout
    const composition = useMediaStore.getState().compositions.find(
      (c: { id: string }) => c.id === compositionId
    );

    if (!composition?.timelineData?.clips || composition.timelineData.clips.length === 0) {
      log.debug(`No clips found for ${compositionId}, using even distribution`);
      return this.getEvenSampleTimes(duration, count);
    }

    const clips = composition.timelineData.clips;
    const tracks = composition.timelineData.tracks || [];

    log.debug(`Analyzing ${clips.length} clips, ${tracks.length} tracks for ${compositionId}`);

    // Get visible video tracks
    const videoTrackIds = new Set(
      tracks
        .filter((t: { type: string; visible?: boolean }) => t.type === 'video' && t.visible !== false)
        .map((t: { id: string }) => t.id)
    );

    // Get all clips on visible video tracks (include all types that might render)
    // Note: sourceType might be undefined, so we include clips without it too
    const videoClips = clips.filter((c: { trackId: string; sourceType?: string }) => {
      const isOnVideoTrack = videoTrackIds.has(c.trackId);
      const isVisualType = !c.sourceType || c.sourceType === 'video' || c.sourceType === 'image' || c.sourceType === 'text';
      return isOnVideoTrack && isVisualType;
    });

    log.debug(`Found ${videoClips.length} visual clips on video tracks`);

    if (videoClips.length === 0) {
      return this.getEvenSampleTimes(duration, count);
    }

    // Collect all important time points (clip boundaries)
    const timePoints = new Set<number>();

    // Always add start and end
    timePoints.add(0);

    // Add clip boundaries
    for (const clip of videoClips) {
      const clipStart = clip.startTime ?? 0;
      const clipDuration = clip.duration ?? 0;
      const clipEnd = clipStart + clipDuration;

      log.debug(`Clip boundary: ${clipStart.toFixed(2)}s - ${clipEnd.toFixed(2)}s`);

      // Add clip start (with small offset to be inside the clip)
      if (clipStart >= 0 && clipStart < duration) {
        timePoints.add(clipStart);
        // Add point just inside clip start
        const insideStart = Math.min(clipStart + 0.05, clipEnd - 0.05);
        if (insideStart > clipStart) {
          timePoints.add(insideStart);
        }
      }

      // Add clip end (with small offset to be inside the clip)
      if (clipEnd > 0 && clipEnd <= duration) {
        // Add point just inside clip end
        const insideEnd = Math.max(clipEnd - 0.05, clipStart + 0.05);
        if (insideEnd < clipEnd) {
          timePoints.add(insideEnd);
        }
        timePoints.add(clipEnd);
      }
    }

    // Always add duration end
    timePoints.add(Math.max(0, duration - 0.01));

    // Convert to sorted array
    let sortedTimes = Array.from(timePoints).sort((a, b) => a - b);

    log.debug(`Initial time points (${sortedTimes.length}):`, sortedTimes.map(t => t.toFixed(2)));

    // Fill in additional points to reach target count
    while (sortedTimes.length < count) {
      // Find the largest gap
      let maxGap = 0;
      let maxGapIndex = 0;

      for (let i = 0; i < sortedTimes.length - 1; i++) {
        const gap = sortedTimes[i + 1] - sortedTimes[i];
        if (gap > maxGap) {
          maxGap = gap;
          maxGapIndex = i;
        }
      }

      if (maxGap < 0.1) break; // Stop if all gaps are tiny

      // Insert midpoint in the largest gap
      const midpoint = (sortedTimes[maxGapIndex] + sortedTimes[maxGapIndex + 1]) / 2;
      sortedTimes.splice(maxGapIndex + 1, 0, midpoint);
    }

    // If we have too many points, sample evenly from them
    if (sortedTimes.length > count) {
      const sampled: number[] = [sortedTimes[0]]; // Always include first
      for (let i = 1; i < count - 1; i++) {
        const idx = Math.round((i / (count - 1)) * (sortedTimes.length - 1));
        sampled.push(sortedTimes[idx]);
      }
      sampled.push(sortedTimes[sortedTimes.length - 1]); // Always include last
      sortedTimes = sampled;
    }

    log.info(`Content-aware sample times for ${compositionId} (${sortedTimes.length} points):`,
      sortedTimes.map(t => t.toFixed(2)).join(', '));
    return sortedTimes;
  }

  /**
   * Get evenly distributed sample times (fallback).
   */
  private getEvenSampleTimes(duration: number, count: number): number[] {
    const times: number[] = [];
    for (let i = 0; i < count; i++) {
      times.push(count > 1 ? (i / (count - 1)) * duration : 0);
    }
    return times;
  }

  private async renderFrameAt(
    compositionId: string,
    time: number,
    width: number,
    height: number
  ): Promise<string | null> {
    if (!this.resources || !this.pingView || !this.pongView || !this.canvasContext || !this.canvas) {
      return null;
    }

    const { device, sampler, compositorPipeline, effectsPipeline, outputPipeline, maskTextureManager } = this.resources;

    // Evaluate composition at the given time
    const layers = compositionRenderer.evaluateAtTime(compositionId, time);

    log.debug(`evaluateAtTime result for ${compositionId} at ${time.toFixed(2)}s: ${layers.length} layers`);

    if (layers.length === 0) {
      // Return transparent/black thumbnail for empty frames
      log.debug(`No layers at time ${time.toFixed(2)}s - returning black thumbnail`);
      return this.createBlackThumbnail(width, height);
    }

    // Seek videos to correct time and wait for frames
    await this.seekAndWaitForLayers(layers, time);

    // Collect layer data
    const layerData = this.collectLayerData(layers);

    if (layerData.length === 0) {
      return this.createBlackThumbnail(width, height);
    }

    // Clear frame-scoped caches
    compositorPipeline.beginFrame();

    // Create command encoder
    const commandEncoder = device.createCommandEncoder();

    // Ping-pong compositing
    let readView = this.pingView;
    let writeView = this.pongView;

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

    // Composite layers
    const outputAspect = width / height;
    for (const data of layerData) {
      const layer = data.layer;
      const uniformBuffer = compositorPipeline.getOrCreateUniformBuffer(`thumb-${layer.id}`);
      const sourceAspect = data.sourceWidth / data.sourceHeight;

      const maskLookupId = layer.maskClipId || layer.id;
      const maskInfo = maskTextureManager.getMaskInfo(maskLookupId);
      const hasMask = maskInfo.hasMask;
      const maskTextureView = maskInfo.view;

      compositorPipeline.updateLayerUniforms(layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = compositorPipeline.getExternalCompositePipeline()!;
        bindGroup = compositorPipeline.createExternalCompositeBindGroup(
          sampler, readView, data.externalTexture, uniformBuffer, maskTextureView
        );
      } else if (data.textureView) {
        pipeline = compositorPipeline.getCompositePipeline()!;
        bindGroup = compositorPipeline.createCompositeBindGroup(
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
      if (layer.effects?.length && effectsPipeline) {
        const result = effectsPipeline.applyEffects(
          commandEncoder, layer.effects, sampler,
          writeView, readView, this.pingView!, this.pongView!, width, height
        );
        if (result.swapped) {
          [readView, writeView] = [writeView, readView];
        }
      }

      // Swap for next layer
      [readView, writeView] = [writeView, readView];
    }

    // Output to canvas
    outputPipeline.updateResolution(width, height);
    const outputBindGroup = outputPipeline.createOutputBindGroup(sampler, readView);
    outputPipeline.renderToCanvas(commandEncoder, this.canvasContext, outputBindGroup);

    // Submit and wait
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Convert to data URL
    try {
      const blob = await this.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.6 });
      return await this.blobToDataURL(blob);
    } catch (e) {
      log.warn('Failed to convert thumbnail to data URL', e);
      return null;
    }
  }

  private async seekAndWaitForLayers(layers: Layer[], time: number): Promise<void> {
    const seekPromises: Promise<void>[] = [];

    for (const layer of layers) {
      if (layer.source?.videoElement) {
        const video = layer.source.videoElement;
        seekPromises.push(this.seekVideoAndWait(video, time));
      }
    }

    await Promise.all(seekPromises);
  }

  private seekVideoAndWait(video: HTMLVideoElement, time: number): Promise<void> {
    return new Promise((resolve) => {
      // Already at correct time and ready?
      if (Math.abs(video.currentTime - time) < 0.05 && video.readyState >= 2) {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        resolve(); // Don't block forever
      }, 1000);

      const onSeeked = () => {
        clearTimeout(timeout);
        video.removeEventListener('seeked', onSeeked);
        // Wait a bit for frame to decode
        requestAnimationFrame(() => resolve());
      };

      video.addEventListener('seeked', onSeeked);
      video.currentTime = time;
    });
  }

  private collectLayerData(layers: Layer[]): Array<{
    layer: Layer;
    isVideo: boolean;
    externalTexture: GPUExternalTexture | null;
    textureView: GPUTextureView | null;
    sourceWidth: number;
    sourceHeight: number;
  }> {
    if (!this.resources) return [];
    const { textureManager } = this.resources;

    const result: Array<{
      layer: Layer;
      isVideo: boolean;
      externalTexture: GPUExternalTexture | null;
      textureView: GPUTextureView | null;
      sourceWidth: number;
      sourceHeight: number;
    }> = [];

    // Process layers in reverse for correct compositing order
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer?.visible || !layer.source || layer.opacity === 0) continue;

      // Video element
      if (layer.source.videoElement) {
        const video = layer.source.videoElement;
        if (video.readyState >= 2) {
          const extTex = textureManager.importVideoTexture(video);
          if (extTex) {
            result.push({
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

      // Image element
      if (layer.source.imageElement) {
        const img = layer.source.imageElement;
        let texture = textureManager.getCachedImageTexture(img);
        if (!texture) {
          texture = textureManager.createImageTexture(img) ?? undefined;
        }
        if (texture) {
          result.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: textureManager.getImageView(texture),
            sourceWidth: img.naturalWidth,
            sourceHeight: img.naturalHeight,
          });
          continue;
        }
      }

      // Text canvas
      if (layer.source.textCanvas) {
        const canvas = layer.source.textCanvas;
        const texture = textureManager.createCanvasTexture(canvas);
        if (texture) {
          result.push({
            layer,
            isVideo: false,
            externalTexture: null,
            textureView: textureManager.getImageView(texture),
            sourceWidth: canvas.width,
            sourceHeight: canvas.height,
          });
        }
      }
    }

    return result;
  }

  private createBlackThumbnail(width: number, height: number): string {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, width, height);

    // Convert to data URL synchronously via canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return '';
    tempCtx.drawImage(canvas, 0, 0);
    return tempCanvas.toDataURL('image/jpeg', 0.6);
  }

  private blobToDataURL(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Generate thumbnails for a single clip with its effects applied.
   * This renders the clip through WebGPU so effects are visible in thumbnails.
   */
  async generateClipThumbnails(
    clip: {
      id: string;
      name: string;
      source: {
        type: string;
        videoElement?: HTMLVideoElement;
        imageElement?: HTMLImageElement;
        naturalDuration?: number;
      } | null;
      transform?: {
        position?: { x: number; y: number; z?: number };
        scale?: { x: number; y: number };
        rotation?: number | { x?: number; y?: number; z?: number };
        opacity?: number;
      };
      effects?: Array<{ id: string; type: string; enabled: boolean; params: Record<string, unknown> }>;
      inPoint: number;
      outPoint: number;
    },
    options?: ThumbnailOptions
  ): Promise<string[]> {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    const { count, width, height } = opts;

    if (!clip.source) {
      log.warn('No source for clip thumbnail generation');
      return [];
    }

    // Initialize if needed
    if (!await this.initialize()) {
      log.warn('ThumbnailRenderer not available');
      return [];
    }

    // Ensure textures and canvas
    if (!this.ensurePingPongTextures(width, height)) {
      return [];
    }
    if (!this.ensureCanvas(width, height)) {
      return [];
    }

    const clipDuration = (clip.outPoint - clip.inPoint) || clip.source.naturalDuration || 5;
    const thumbnails: string[] = [];

    log.info(`Generating ${count} thumbnails for clip ${clip.name} with effects`);

    // Generate frames from 0% to 100% of clip duration
    for (let i = 0; i < count; i++) {
      const progress = count > 1 ? i / (count - 1) : 0;
      const clipTime = clip.inPoint + progress * clipDuration;

      try {
        const dataUrl = await this.renderClipFrameAt(clip, clipTime, width, height);
        if (dataUrl) {
          thumbnails.push(dataUrl);
        }
      } catch (e) {
        log.warn(`Failed to render clip thumbnail at ${clipTime.toFixed(2)}s`, e);
      }

      // Yield to main thread between frames
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    log.debug(`Generated ${thumbnails.length} thumbnails for clip ${clip.name}`);
    return thumbnails;
  }

  /**
   * Render a single clip frame with effects applied.
   */
  private async renderClipFrameAt(
    clip: {
      id: string;
      source: {
        type: string;
        videoElement?: HTMLVideoElement;
        imageElement?: HTMLImageElement;
      } | null;
      transform?: {
        position?: { x: number; y: number; z?: number };
        scale?: { x: number; y: number };
        rotation?: number | { x?: number; y?: number; z?: number };
        opacity?: number;
      };
      effects?: Array<{ id: string; type: string; enabled: boolean; params: Record<string, unknown> }>;
    },
    time: number,
    width: number,
    height: number
  ): Promise<string | null> {
    if (!this.resources || !this.pingView || !this.pongView || !this.canvasContext || !this.canvas) {
      return null;
    }

    const { device, sampler, compositorPipeline, effectsPipeline, outputPipeline, textureManager: _textureManager, maskTextureManager } = this.resources;

    if (!clip.source) return null;

    // Build a layer from the clip
    const transform = clip.transform || {};
    const layer: Layer = {
      id: `thumb-${clip.id}`,
      name: clip.id,
      visible: true,
      opacity: transform.opacity ?? 1,
      blendMode: 'normal',
      source: clip.source as Layer['source'],
      effects: (clip.effects || []) as any,
      position: { x: transform.position?.x || 0, y: transform.position?.y || 0, z: transform.position?.z || 0 },
      scale: transform.scale || { x: 1, y: 1 },
      rotation: typeof transform.rotation === 'number'
        ? transform.rotation
        : (transform.rotation?.z || 0),
    };

    // Seek video to correct time
    if (clip.source.videoElement) {
      await this.seekVideoAndWait(clip.source.videoElement, time);
    }

    // Collect layer data
    const layerData = this.collectLayerData([layer]);
    if (layerData.length === 0) {
      return this.createBlackThumbnail(width, height);
    }

    // Clear frame-scoped caches
    compositorPipeline.beginFrame();

    // Create command encoder
    const commandEncoder = device.createCommandEncoder();

    // Ping-pong compositing
    let readView = this.pingView;
    let writeView = this.pongView;

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

    // Composite the single layer
    const outputAspect = width / height;
    for (const data of layerData) {
      const uniformBuffer = compositorPipeline.getOrCreateUniformBuffer(`clip-thumb-${data.layer.id}`);
      const sourceAspect = data.sourceWidth / data.sourceHeight;

      const maskLookupId = data.layer.id;
      const maskInfo = maskTextureManager.getMaskInfo(maskLookupId);
      const hasMask = maskInfo.hasMask;
      const maskTextureView = maskInfo.view;

      compositorPipeline.updateLayerUniforms(data.layer, sourceAspect, outputAspect, hasMask, uniformBuffer);

      let pipeline: GPURenderPipeline;
      let bindGroup: GPUBindGroup;

      if (data.isVideo && data.externalTexture) {
        pipeline = compositorPipeline.getExternalCompositePipeline()!;
        bindGroup = compositorPipeline.createExternalCompositeBindGroup(
          sampler, readView, data.externalTexture, uniformBuffer, maskTextureView
        );
      } else if (data.textureView) {
        pipeline = compositorPipeline.getCompositePipeline()!;
        bindGroup = compositorPipeline.createCompositeBindGroup(
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
      if (data.layer.effects?.length && effectsPipeline) {
        const result = effectsPipeline.applyEffects(
          commandEncoder, data.layer.effects, sampler,
          writeView, readView, this.pingView!, this.pongView!, width, height
        );
        if (result.swapped) {
          [readView, writeView] = [writeView, readView];
        }
      }

      // Swap for next layer
      [readView, writeView] = [writeView, readView];
    }

    // Output to canvas
    outputPipeline.updateResolution(width, height);
    const outputBindGroup = outputPipeline.createOutputBindGroup(sampler, readView);
    outputPipeline.renderToCanvas(commandEncoder, this.canvasContext, outputBindGroup);

    // Submit and wait
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Convert to data URL
    try {
      const blob = await this.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.6 });
      return await this.blobToDataURL(blob);
    } catch (e) {
      log.warn('Failed to convert clip thumbnail to data URL', e);
      return null;
    }
  }

  dispose(): void {
    this.pingTexture?.destroy();
    this.pongTexture?.destroy();
    this.pingTexture = null;
    this.pongTexture = null;
    this.pingView = null;
    this.pongView = null;
    this.canvas = null;
    this.canvasContext = null;

    if (this.resources) {
      this.resources.compositorPipeline.destroy();
      this.resources.effectsPipeline.destroy();
      this.resources.outputPipeline.destroy();
      this.resources.textureManager.destroy();
      this.resources.maskTextureManager.destroy();
      this.resources.device.destroy();
      this.resources = null;
    }

    this.isInitialized = false;
    this.initPromise = null;
    log.debug('ThumbnailRenderer disposed');
  }
}

// Singleton instance
export const thumbnailRenderer = new ThumbnailRendererService();
