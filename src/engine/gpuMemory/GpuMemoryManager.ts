// GpuMemoryManager — wraps device.createTexture() to track VRAM usage,
// enforce a budget, and evict unused textures via LRU.

import type {
  GpuAllocation,
  GpuAllocationCategory,
  GpuMemoryBudget,
  GpuMemoryConfig,
} from './types.ts';
import { EVICTION_PRIORITY, DEFAULT_GPU_MEMORY_CONFIG } from './types.ts';
import { Logger } from '../../services/logger.ts';

const log = Logger.create('GpuMemory');

export class GpuMemoryManager {
  private device: GPUDevice;
  private config: GpuMemoryConfig;
  private allocations = new Map<string, GpuAllocation>();
  private usedBytes = 0;
  private frameCounter = 0;

  constructor(device: GPUDevice, config?: Partial<GpuMemoryConfig>) {
    this.device = device;
    this.config = { ...DEFAULT_GPU_MEMORY_CONFIG, ...config };
  }

  /**
   * Create a GPU texture through the manager.
   * Tracks the allocation and may evict if budget is exceeded.
   */
  createTexture(
    desc: GPUTextureDescriptor,
    category: GpuAllocationCategory,
    id: string,
    pinned = false
  ): GPUTexture {
    const byteSize = this.calcBytes(desc);

    // Evict if needed (only for non-pinned allocations)
    if (!pinned && this.usedBytes + byteSize > this.config.budgetBytes) {
      this.evict(byteSize);
    }

    const texture = this.device.createTexture(desc);
    this.track(id, texture, byteSize, category, pinned);

    return texture;
  }

  /**
   * Release a tracked allocation.
   */
  release(id: string): void {
    const alloc = this.allocations.get(id);
    if (!alloc) return;

    alloc.texture.destroy();
    this.usedBytes -= alloc.byteSize;
    this.allocations.delete(id);
  }

  /**
   * Mark an allocation as accessed this frame (updates LRU).
   */
  touch(id: string): void {
    const alloc = this.allocations.get(id);
    if (alloc) {
      alloc.lastAccessFrame = this.frameCounter;
    }
  }

  /**
   * Call once per frame to age allocations.
   */
  tick(): void {
    this.frameCounter++;
  }

  /**
   * Get current VRAM usage in MB.
   */
  getUsageMB(): number {
    return this.usedBytes / (1024 * 1024);
  }

  /**
   * Get detailed budget and usage statistics.
   */
  getStats(): GpuMemoryBudget {
    const byCategory: GpuMemoryBudget['byCategory'] = {
      pingPong: { count: 0, bytes: 0 },
      effectTemp: { count: 0, bytes: 0 },
      nestedComp: { count: 0, bytes: 0 },
      scrubCache: { count: 0, bytes: 0 },
      imageTexture: { count: 0, bytes: 0 },
      dynamicTexture: { count: 0, bytes: 0 },
    };

    for (const alloc of this.allocations.values()) {
      const cat = byCategory[alloc.category];
      cat.count++;
      cat.bytes += alloc.byteSize;
    }

    return {
      totalBudget: this.config.budgetBytes,
      usedBytes: this.usedBytes,
      usagePercent: this.usedBytes / this.config.budgetBytes,
      byCategory,
      allocationCount: this.allocations.size,
    };
  }

  /**
   * Check if an allocation exists.
   */
  has(id: string): boolean {
    return this.allocations.has(id);
  }

  /**
   * Get the GPUTexture for a tracked allocation.
   */
  getTexture(id: string): GPUTexture | undefined {
    return this.allocations.get(id)?.texture;
  }

  /**
   * Register an externally-created texture for stats tracking only.
   * The manager will NOT destroy it — the caller owns the lifecycle.
   */
  registerExternal(id: string, texture: GPUTexture, byteSize: number, category: GpuAllocationCategory): void {
    if (this.allocations.has(id)) {
      this.unregisterExternal(id);
    }
    this.allocations.set(id, {
      id,
      texture,
      byteSize,
      category,
      pinned: true, // external textures are never evicted
      lastAccessFrame: this.frameCounter,
      createdAt: performance.now(),
    });
    this.usedBytes += byteSize;
  }

  /**
   * Remove tracking for an external texture (does NOT destroy it).
   */
  unregisterExternal(id: string): void {
    const alloc = this.allocations.get(id);
    if (!alloc) return;
    this.usedBytes -= alloc.byteSize;
    this.allocations.delete(id);
  }

  /**
   * Destroy all tracked allocations.
   */
  destroy(): void {
    for (const alloc of this.allocations.values()) {
      alloc.texture.destroy();
    }
    this.allocations.clear();
    this.usedBytes = 0;
  }

  // === Private ===

  private track(
    id: string,
    texture: GPUTexture,
    byteSize: number,
    category: GpuAllocationCategory,
    pinned: boolean
  ): void {
    // Release existing allocation with same ID if present
    if (this.allocations.has(id)) {
      this.release(id);
    }

    this.allocations.set(id, {
      id,
      texture,
      byteSize,
      category,
      pinned,
      lastAccessFrame: this.frameCounter,
      createdAt: performance.now(),
    });
    this.usedBytes += byteSize;
  }

  private evict(neededBytes: number): void {
    let freedBytes = 0;

    // Evict in priority order (scrubCache first, then dynamic, etc.)
    for (const category of EVICTION_PRIORITY) {
      if (freedBytes >= neededBytes) break;

      // Collect candidates in this category, sorted by LRU (oldest first)
      const candidates: GpuAllocation[] = [];
      for (const alloc of this.allocations.values()) {
        if (alloc.category === category && !alloc.pinned) {
          candidates.push(alloc);
        }
      }
      candidates.sort((a, b) => a.lastAccessFrame - b.lastAccessFrame);

      for (const candidate of candidates) {
        if (freedBytes >= neededBytes) break;

        log.debug('Evicting', { id: candidate.id, category, sizeMB: (candidate.byteSize / (1024 * 1024)).toFixed(1) });
        candidate.texture.destroy();
        this.usedBytes -= candidate.byteSize;
        freedBytes += candidate.byteSize;
        this.allocations.delete(candidate.id);
      }
    }

    if (freedBytes < neededBytes) {
      log.warn('Could not free enough VRAM', {
        neededMB: (neededBytes / (1024 * 1024)).toFixed(1),
        freedMB: (freedBytes / (1024 * 1024)).toFixed(1),
        usedMB: (this.usedBytes / (1024 * 1024)).toFixed(1),
      });
    }
  }

  private calcBytes(desc: GPUTextureDescriptor): number {
    const size = desc.size as GPUExtent3DDict;
    const w = size.width ?? 1;
    const h = size.height ?? 1;
    const d = size.depthOrArrayLayers ?? 1;

    // bytes per pixel based on format
    let bpp = 4; // rgba8unorm default
    const format = desc.format;
    if (format === 'rgba16float') bpp = 8;
    else if (format === 'rgba32float') bpp = 16;
    else if (format === 'r8unorm' || format === 'r8uint') bpp = 1;
    else if (format === 'rg8unorm') bpp = 2;

    return w * h * d * bpp;
  }
}
