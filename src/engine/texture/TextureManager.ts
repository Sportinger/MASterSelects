// Texture creation and caching for images and video frames

export class TextureManager {
  private device: GPUDevice;

  // Cached image textures (created from HTMLImageElement)
  private imageTextures: Map<HTMLImageElement, GPUTexture> = new Map();

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
      console.error('Failed to create image texture:', e);
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
        return null;
      }
      // Skip if video is seeking - frame might not be ready
      if (source.seeking) {
        return null;
      }
    } else if (source instanceof VideoFrame) {
      if (source.codedWidth === 0 || source.codedHeight === 0) {
        return null;
      }
    } else {
      return null;
    }

    try {
      return this.device.importExternalTexture({ source });
    } catch {
      // Silently fail - video may not be ready yet
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
    // Destroy image textures
    for (const texture of this.imageTextures.values()) {
      texture.destroy();
    }
    this.imageTextures.clear();
    this.cachedImageViews.clear();

    // Destroy video frame textures
    for (const texture of this.videoFrameTextures.values()) {
      texture.destroy();
    }
    this.videoFrameTextures.clear();
    this.videoFrameViews.clear();
  }

  // Remove a specific image from cache
  removeImageTexture(image: HTMLImageElement): void {
    const texture = this.imageTextures.get(image);
    if (texture) {
      texture.destroy();
      this.imageTextures.delete(image);
      this.cachedImageViews.delete(texture);
    }
  }

  destroy(): void {
    this.clearCaches();
  }
}
