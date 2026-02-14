// GPU Memory Manager types — VRAM budget tracking and eviction

/** Categories of GPU allocations with different eviction policies */
export type GpuAllocationCategory =
  | 'pingPong'
  | 'effectTemp'
  | 'nestedComp'
  | 'scrubCache'
  | 'imageTexture'
  | 'dynamicTexture';

/** A tracked GPU memory allocation */
export interface GpuAllocation {
  id: string;
  texture: GPUTexture;
  byteSize: number;
  category: GpuAllocationCategory;
  pinned: boolean;
  lastAccessFrame: number;
  createdAt: number;
}

/** Budget and usage statistics */
export interface GpuMemoryBudget {
  totalBudget: number;
  usedBytes: number;
  usagePercent: number;
  allocationCount: number;
  byCategory: Record<GpuAllocationCategory, { count: number; bytes: number }>;
}

/** Eviction priority — ordered array, first category evicted first */
export const EVICTION_PRIORITY: GpuAllocationCategory[] = [
  'scrubCache',      // Evicted first
  'dynamicTexture',
  'nestedComp',
  'imageTexture',
  // effectTemp and pingPong are always pinned, never evicted
];

/** Configuration for the GPU memory manager */
export interface GpuMemoryConfig {
  /** Total VRAM budget in bytes (default: 2GB) */
  budgetBytes: number;
  /** Enable LRU eviction when budget exceeded */
  enableEviction: boolean;
  /** Log eviction events */
  logEvictions: boolean;
}

/** Default config */
export const DEFAULT_GPU_MEMORY_CONFIG: GpuMemoryConfig = {
  budgetBytes: 2 * 1024 * 1024 * 1024, // 2GB
  enableEviction: true,
  logEvictions: true,
};
