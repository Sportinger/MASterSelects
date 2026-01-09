// Mask texture handling for layer masking

export class MaskTextureManager {
  private device: GPUDevice;

  // Mask textures per layer
  private maskTextures: Map<string, GPUTexture> = new Map();
  private maskTextureViews: Map<string, GPUTextureView> = new Map();
  private lastMaskDebugLog: number | null = null;

  // Fallback mask texture (fully white = no masking)
  private whiteMaskTexture: GPUTexture | null = null;
  private whiteMaskView: GPUTextureView | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.createWhiteMaskTexture();
  }

  private createWhiteMaskTexture(): void {
    this.whiteMaskTexture = this.device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture: this.whiteMaskTexture },
      new Uint8Array([255, 255, 255, 255]),  // Pure white = fully visible
      { bytesPerRow: 4 },
      [1, 1]
    );

    this.whiteMaskView = this.whiteMaskTexture.createView();
  }

  // Update mask texture for a layer
  updateMaskTexture(layerId: string, imageData: ImageData | null): void {
    // Remove existing mask texture
    const existingTexture = this.maskTextures.get(layerId);
    if (existingTexture) {
      existingTexture.destroy();
      this.maskTextures.delete(layerId);
      this.maskTextureViews.delete(layerId);
    }

    // If no imageData, layer will use white fallback (no masking)
    if (!imageData) {
      console.log(`[MaskTextureManager] No mask data for layer ${layerId}, using white fallback`);
      return;
    }

    // Create new mask texture
    const maskTexture = this.device.createTexture({
      size: [imageData.width, imageData.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Upload mask data
    this.device.queue.writeTexture(
      { texture: maskTexture },
      imageData.data,
      {
        bytesPerRow: imageData.width * 4,
        rowsPerImage: imageData.height,
      },
      [imageData.width, imageData.height]
    );

    // Cache texture and view
    this.maskTextures.set(layerId, maskTexture);
    this.maskTextureViews.set(layerId, maskTexture.createView());

    console.log(`[MaskTextureManager] Uploaded mask texture for layer ${layerId}: ${imageData.width}x${imageData.height}`);
  }

  // Remove mask texture for a layer
  removeMaskTexture(layerId: string): void {
    const existingTexture = this.maskTextures.get(layerId);
    if (existingTexture) {
      existingTexture.destroy();
      this.maskTextures.delete(layerId);
      this.maskTextureViews.delete(layerId);
    }
  }

  // Check if a layer has a mask texture
  hasMaskTexture(layerId: string): boolean {
    return this.maskTextureViews.has(layerId);
  }

  // Get mask texture view for a layer (returns white fallback if no mask)
  getMaskTextureView(layerId: string): GPUTextureView {
    return this.maskTextureViews.get(layerId) || this.whiteMaskView!;
  }

  // Get the fallback white mask view
  getWhiteMaskView(): GPUTextureView {
    return this.whiteMaskView!;
  }

  // Log mask state for debugging (throttled)
  logMaskState(layerId: string, hasMask: boolean): void {
    if (hasMask && (!this.lastMaskDebugLog || Date.now() - this.lastMaskDebugLog > 1000)) {
      console.log(`[MaskTextureManager] Rendering layer ${layerId} WITH mask`);
      this.lastMaskDebugLog = Date.now();
    }
  }

  // Clear all mask textures
  clearAll(): void {
    for (const texture of this.maskTextures.values()) {
      texture.destroy();
    }
    this.maskTextures.clear();
    this.maskTextureViews.clear();
  }

  destroy(): void {
    this.clearAll();
    this.whiteMaskTexture?.destroy();
    this.whiteMaskTexture = null;
    this.whiteMaskView = null;
  }
}
