// Texture creation and caching for images and video frames

import { Logger } from '../../services/logger';

const log = Logger.create('TextureManager');

export class TextureManager {
  private device: GPUDevice;

  // Cached image textures (created from HTMLImageElement)
  private imageTextures: Map<HTMLImageElement, GPUTexture> = new Map();

  // Cached canvas textures (created from HTMLCanvasElement - for text clips)
  // Canvas reference changes when text properties change, so caching by reference is safe
  private canvasTextures: Map<HTMLCanvasElement, GPUTexture> = new Map();

  // Cached image texture views
  private cachedImageViews: Map<GPUTexture, GPUTextureView> = new Map();

  // Reusable dynamic textures keyed by layer ID (for NativeDecoder frames)
  private dynamicTextures: Map<string, { texture: GPUTexture; view: GPUTextureView; width: number; height: number }> = new Map();

  // Video frame textures (rendered from external textures)
  private videoFrameTextures: Map<string, GPUTexture> = new Map();
  private videoFrameViews: Map<string, GPUTextureView> = new Map();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  // Create GPU texture from HTMLImageElement
  createImageTexture(image: HTMLImageElement): GPUTexture | null {
    // Use naturalWidth/naturalHeight for images not added to DOM (like proxy frames)
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;

    if (width === 0 || height === 0) return null;

    // Check cache first
    const cached = this.imageTextures.get(image);
    if (cached) return cached;

    try {
      const texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      this.device.queue.copyExternalImageToTexture(
        { source: image },
        { texture },
        [width, height]
      );

      this.imageTextures.set(image, texture);
      return texture;
    } catch (e) {
      log.error('Failed to create image texture', e);
      return null;
    }
  }

  // Create GPU texture from HTMLCanvasElement (for text clips)
  // Cached by canvas reference - text clips create new canvas when properties change
  createCanvasTexture(canvas: HTMLCanvasElement): GPUTexture | null {
    const width = canvas.width;
    const height = canvas.height;

    if (width === 0 || height === 0) return null;

    // Check cache first
    const cached = this.canvasTextures.get(canvas);
    if (cached) return cached;

    try {
      const texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      this.device.queue.copyExternalImageToTexture(
        { source: canvas },
        { texture },
        [width, height]
      );

      this.canvasTextures.set(canvas, texture);
      return texture;
    } catch (e) {
      log.error('Failed to create canvas texture', e);
      return null;
    }
  }

  // Re-upload canvas content to an existing cached GPU texture
  // Used when canvas pixels changed (e.g. text re-rendered) but reference is the same
  updateCanvasTexture(canvas: HTMLCanvasElement): boolean {
    const width = canvas.width;
    const height = canvas.height;
    if (width === 0 || height === 0) return false;

    const texture = this.canvasTextures.get(canvas);
    if (!texture) return false;

    try {
      this.device.queue.copyExternalImageToTexture(
        { source: canvas },
        { texture },
        [width, height]
      );
      return true;
    } catch (e) {
      log.error('Failed to update canvas texture', e);
      return false;
    }
  }

  // Get cached canvas texture
  getCachedCanvasTexture(canvas: HTMLCanvasElement): GPUTexture | undefined {
    return this.canvasTextures.get(canvas);
  }

  // Create or reuse GPU texture from ImageBitmap (for native helper decoded frames).
  // When a key is provided, reuses the existing texture if dimensions match,
  // avoiding GPU memory growth from creating 30+ textures/second.
  createImageBitmapTexture(bitmap: ImageBitmap, key?: string): GPUTexture | null {
    const width = bitmap.width;
    const height = bitmap.height;

    if (width === 0 || height === 0) return null;

    // Fast path: reuse existing texture for this key
    if (key) {
      const existing = this.dynamicTextures.get(key);
      if (existing && existing.width === width && existing.height === height) {
        try {
          this.device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: existing.texture },
            [width, height]
          );
          return existing.texture;
        } catch {
          // Texture became invalid (device lost, etc.) — fall through to create new
          this.dynamicTextures.delete(key);
        }
      } else if (existing) {
        // Dimensions changed — destroy old texture
        existing.texture.destroy();
        this.dynamicTextures.delete(key);
      }
    }

    try {
      const texture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      this.device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture },
        [width, height]
      );

      // Cache for reuse if key provided
      if (key) {
        const view = texture.createView();
        this.dynamicTextures.set(key, { texture, view, width, height });
      }

      return texture;
    } catch (e) {
      log.error('Failed to create ImageBitmap texture', e);
      return null;
    }
  }

  // Get cached view for a dynamic texture (avoids creating new view every frame)
  getDynamicTextureView(key: string): GPUTextureView | null {
    return this.dynamicTextures.get(key)?.view ?? null;
  }

  // Remove a dynamic texture entry (e.g. when clip is removed or downgraded from NH)
  removeDynamicTexture(key: string): void {
    const entry = this.dynamicTextures.get(key);
    if (entry) {
      entry.texture.destroy();
      this.dynamicTextures.delete(key);
    }
  }

  // Get or create a view for a texture
  getImageView(texture: GPUTexture): GPUTextureView {
    let view = this.cachedImageViews.get(texture);
    if (!view) {
      view = texture.createView();
      this.cachedImageViews.set(texture, view);
    }
    return view;
  }

  // Get cached image texture
  getCachedImageTexture(image: HTMLImageElement): GPUTexture | undefined {
    return this.imageTextures.get(image);
  }

  // Import external texture - true zero-copy from video decoder
  // Supports both HTMLVideoElement and VideoFrame (from WebCodecs)
  importVideoTexture(source: HTMLVideoElement | VideoFrame): GPUExternalTexture | null {
    // Check if source is valid
    if (source instanceof HTMLVideoElement) {
      // readyState >= 2 means HAVE_CURRENT_DATA (has at least one frame)
      if (source.readyState < 2 || source.videoWidth === 0 || source.videoHeight === 0) {
        log.debug('Video not ready', { readyState: source.readyState, width: source.videoWidth, height: source.videoHeight });
        return null;
      }
    } else if (source instanceof VideoFrame) {
      if (source.codedWidth === 0 || source.codedHeight === 0) {
        log.debug('VideoFrame has zero dimensions');
        return null;
      }
    } else {
      return null;
    }

    try {
      const texture = this.device.importExternalTexture({ source });
      log.debug('External texture imported successfully');
      return texture;
    } catch (e) {
      // Log the actual error - this is critical for debugging Linux/Vulkan issues
      log.error('Failed to import external texture', e);
      return null;
    }
  }

  // Create a render target texture
  createRenderTexture(width: number, height: number): GPUTexture {
    return this.device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
  }

  // Create a texture from ImageData
  createTextureFromImageData(imageData: ImageData): GPUTexture {
    const texture = this.device.createTexture({
      size: [imageData.width, imageData.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.device.queue.writeTexture(
      { texture },
      imageData.data,
      {
        bytesPerRow: imageData.width * 4,
        rowsPerImage: imageData.height,
      },
      [imageData.width, imageData.height]
    );

    return texture;
  }

  // Clear all caches
  clearCaches(): void {
    // Don't destroy textures - let GC handle to avoid GPU conflicts
    // WebGPU will automatically release GPU resources when JS objects are GC'd
    // AND the GPU is done using them
    this.imageTextures.clear();
    this.canvasTextures.clear();
    this.cachedImageViews.clear();
    this.videoFrameTextures.clear();
    this.videoFrameViews.clear();
    // Dynamic textures are explicitly destroyed since we own them
    for (const entry of this.dynamicTextures.values()) {
      entry.texture.destroy();
    }
    this.dynamicTextures.clear();
  }

  // Remove a specific image from cache
  removeImageTexture(image: HTMLImageElement): void {
    const texture = this.imageTextures.get(image);
    if (texture) {
      // Don't destroy - let GC handle to avoid GPU conflicts
      this.imageTextures.delete(image);
      this.cachedImageViews.delete(texture);
    }
  }

  // Remove a specific canvas from cache
  removeCanvasTexture(canvas: HTMLCanvasElement): void {
    const texture = this.canvasTextures.get(canvas);
    if (texture) {
      // Don't destroy - let GC handle to avoid GPU conflicts
      this.canvasTextures.delete(canvas);
      this.cachedImageViews.delete(texture);
    }
  }

  destroy(): void {
    this.clearCaches();
  }
}
