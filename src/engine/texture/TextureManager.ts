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

  // Get cached canvas texture
  getCachedCanvasTexture(canvas: HTMLCanvasElement): GPUTexture | undefined {
    return this.canvasTextures.get(canvas);
  }

  // Create GPU texture from ImageBitmap (for native helper decoded frames)
  // NOT cached - ImageBitmaps change every frame and are closed after use
  createImageBitmapTexture(bitmap: ImageBitmap): GPUTexture | null {
    const width = bitmap.width;
    const height = bitmap.height;

    if (width === 0 || height === 0) return null;

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

      return texture;
    } catch (e) {
      log.error('Failed to create ImageBitmap texture', e);
      return null;
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
      // Also check we're not in middle of seeking which can cause blank frames
      if (source.readyState < 2 || source.videoWidth === 0 || source.videoHeight === 0) {
        log.debug('Video not ready', { readyState: source.readyState, width: source.videoWidth, height: source.videoHeight });
        return null;
      }
      // Skip if video is seeking - frame might not be ready
      if (source.seeking) {
        log.debug('Video is seeking, skipping frame');
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
