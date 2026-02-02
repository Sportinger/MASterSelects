/**
 * FrameCacheManager - LRU cache for decoded VideoFrames
 *
 * Features:
 * - Per-file LRU cache with configurable size
 * - Automatic memory tracking and eviction
 * - Cache statistics for monitoring
 * - Hard memory limit protection
 */

import { Logger } from '../../../services/logger'
import type { DecodedFrameData, CacheStats, FileCacheStats } from './types'

const log = Logger.create('FrameCache')

interface CacheEntry {
  data: DecodedFrameData
  lastAccessed: number      // timestamp for LRU
  accessCount: number       // for stats
}

interface FileCache {
  fileHash: string
  entries: Map<number, CacheEntry>  // timestamp -> entry
  sortedTimestamps: number[]         // for O(log n) lookup
  maxFrames: number
  stats: {
    hits: number
    misses: number
    evictions: number
  }
}

export class FrameCacheManager {
  private caches: Map<string, FileCache> = new Map()
  private totalMemoryBytes = 0
  private maxMemoryBytes: number
  private defaultMaxFramesPerFile: number

  constructor(options: {
    maxMemoryMB?: number
    defaultMaxFramesPerFile?: number
  } = {}) {
    this.maxMemoryBytes = (options.maxMemoryMB ?? 1000) * 1024 * 1024 // Default 1GB
    this.defaultMaxFramesPerFile = options.defaultMaxFramesPerFile ?? 60
    log.info(`Initialized with ${(this.maxMemoryBytes / 1024 / 1024).toFixed(0)}MB limit, ${this.defaultMaxFramesPerFile} frames/file default`)
  }

  /**
   * Store frame in cache
   */
  put(data: DecodedFrameData): void {
    let fileCache = this.caches.get(data.fileHash)

    if (!fileCache) {
      fileCache = {
        fileHash: data.fileHash,
        entries: new Map(),
        sortedTimestamps: [],
        maxFrames: this.defaultMaxFramesPerFile,
        stats: { hits: 0, misses: 0, evictions: 0 }
      }
      this.caches.set(data.fileHash, fileCache)
      log.debug(`Created cache for file ${data.fileHash}`)
    }

    // Check if already exists (shouldn't happen, but handle it)
    if (fileCache.entries.has(data.timestamp)) {
      const existing = fileCache.entries.get(data.timestamp)!
      existing.data.frame.close()
      fileCache.entries.delete(data.timestamp)
      this.removeFromSorted(fileCache.sortedTimestamps, data.timestamp)
    }

    // Add to cache
    const entry: CacheEntry = {
      data,
      lastAccessed: performance.now(),
      accessCount: 0
    }
    fileCache.entries.set(data.timestamp, entry)

    // Maintain sorted timestamps for binary search
    const insertIdx = this.binarySearchInsertPosition(fileCache.sortedTimestamps, data.timestamp)
    fileCache.sortedTimestamps.splice(insertIdx, 0, data.timestamp)

    // Estimate memory (rough estimate: 1920x1080 RGBA = 8MB)
    const estimatedBytes = (data.frame.codedWidth || 1920) * (data.frame.codedHeight || 1080) * 4
    this.totalMemoryBytes += estimatedBytes

    // Evict if needed
    this.evictIfNeeded(fileCache)
    this.evictGlobalIfNeeded()
  }

  /**
   * Get frame from cache
   */
  get(fileHash: string, timestamp: number, toleranceMicros: number): DecodedFrameData | null {
    const fileCache = this.caches.get(fileHash)
    if (!fileCache) {
      return null
    }

    // Binary search for closest timestamp
    const idx = this.findClosestTimestamp(fileCache.sortedTimestamps, timestamp, toleranceMicros)
    if (idx === -1) {
      fileCache.stats.misses++
      return null
    }

    const closestTimestamp = fileCache.sortedTimestamps[idx]
    const entry = fileCache.entries.get(closestTimestamp)
    if (!entry) {
      fileCache.stats.misses++
      return null
    }

    // Update LRU
    entry.lastAccessed = performance.now()
    entry.accessCount++
    fileCache.stats.hits++

    return entry.data
  }

  /**
   * Check if frame exists in cache
   */
  has(fileHash: string, timestamp: number, toleranceMicros: number): boolean {
    return this.get(fileHash, timestamp, toleranceMicros) !== null
  }

  /**
   * Set max frames for a specific file (for heavy usage files)
   */
  setFileMaxFrames(fileHash: string, maxFrames: number): void {
    const fileCache = this.caches.get(fileHash)
    if (fileCache) {
      fileCache.maxFrames = maxFrames
      log.debug(`Set max frames for ${fileHash}: ${maxFrames}`)
      this.evictIfNeeded(fileCache)
    }
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    let totalFrames = 0
    let totalHits = 0
    let totalMisses = 0
    let totalEvictions = 0
    const perFileStats = new Map<string, FileCacheStats>()

    for (const [fileHash, fileCache] of this.caches) {
      totalFrames += fileCache.entries.size
      totalHits += fileCache.stats.hits
      totalMisses += fileCache.stats.misses
      totalEvictions += fileCache.stats.evictions

      perFileStats.set(fileHash, {
        fileHash,
        frames: fileCache.entries.size,
        memoryMB: this.estimateFileCacheMemory(fileCache) / 1024 / 1024,
        hits: fileCache.stats.hits,
        misses: fileCache.stats.misses,
        evictions: fileCache.stats.evictions
      })
    }

    const total = totalHits + totalMisses
    const hitRate = total > 0 ? totalHits / total : 0
    const missRate = total > 0 ? totalMisses / total : 0

    return {
      totalFrames,
      totalMemoryMB: this.totalMemoryBytes / 1024 / 1024,
      hitRate,
      missRate,
      evictions: totalEvictions,
      perFileStats
    }
  }

  /**
   * Clear cache for specific file or all
   */
  clear(fileHash?: string): void {
    if (fileHash) {
      const fileCache = this.caches.get(fileHash)
      if (fileCache) {
        for (const entry of fileCache.entries.values()) {
          entry.data.frame.close()
        }
        const memory = this.estimateFileCacheMemory(fileCache)
        this.totalMemoryBytes -= memory
        this.caches.delete(fileHash)
        log.debug(`Cleared cache for ${fileHash}`)
      }
    } else {
      for (const fileCache of this.caches.values()) {
        for (const entry of fileCache.entries.values()) {
          entry.data.frame.close()
        }
      }
      this.caches.clear()
      this.totalMemoryBytes = 0
      log.info('Cleared all caches')
    }
  }

  /**
   * Cleanup all resources
   */
  dispose(): void {
    this.clear()
  }

  // Private methods

  private evictIfNeeded(fileCache: FileCache): void {
    while (fileCache.entries.size > fileCache.maxFrames) {
      this.evictLRU(fileCache)
    }
  }

  private evictGlobalIfNeeded(): void {
    if (this.totalMemoryBytes <= this.maxMemoryBytes) {
      return
    }

    log.warn(`Memory limit exceeded (${(this.totalMemoryBytes / 1024 / 1024).toFixed(1)}MB > ${(this.maxMemoryBytes / 1024 / 1024).toFixed(0)}MB), evicting...`)

    // Evict LRU entries across all caches until under limit
    const allEntries: Array<{ fileHash: string; timestamp: number; entry: CacheEntry }> = []

    for (const [fileHash, fileCache] of this.caches) {
      for (const [timestamp, entry] of fileCache.entries) {
        allEntries.push({ fileHash, timestamp, entry })
      }
    }

    // Sort by lastAccessed (oldest first)
    allEntries.sort((a, b) => a.entry.lastAccessed - b.entry.lastAccessed)

    // Evict oldest until under limit
    let evicted = 0
    for (const item of allEntries) {
      if (this.totalMemoryBytes <= this.maxMemoryBytes) {
        break
      }

      const fileCache = this.caches.get(item.fileHash)!
      const entry = fileCache.entries.get(item.timestamp)!

      entry.data.frame.close()
      fileCache.entries.delete(item.timestamp)
      this.removeFromSorted(fileCache.sortedTimestamps, item.timestamp)
      fileCache.stats.evictions++

      const estimatedBytes = (entry.data.frame.codedWidth || 1920) * (entry.data.frame.codedHeight || 1080) * 4
      this.totalMemoryBytes -= estimatedBytes
      evicted++
    }

    log.info(`Evicted ${evicted} frames to stay under memory limit`)
  }

  private evictLRU(fileCache: FileCache): void {
    if (fileCache.entries.size === 0) return

    // Find LRU entry
    let oldestTimestamp = -1
    let oldestAccessTime = Infinity

    for (const [timestamp, entry] of fileCache.entries) {
      if (entry.lastAccessed < oldestAccessTime) {
        oldestAccessTime = entry.lastAccessed
        oldestTimestamp = timestamp
      }
    }

    if (oldestTimestamp === -1) return

    const entry = fileCache.entries.get(oldestTimestamp)!
    entry.data.frame.close()
    fileCache.entries.delete(oldestTimestamp)
    this.removeFromSorted(fileCache.sortedTimestamps, oldestTimestamp)
    fileCache.stats.evictions++

    const estimatedBytes = (entry.data.frame.codedWidth || 1920) * (entry.data.frame.codedHeight || 1080) * 4
    this.totalMemoryBytes -= estimatedBytes
  }

  private estimateFileCacheMemory(fileCache: FileCache): number {
    let total = 0
    for (const entry of fileCache.entries.values()) {
      total += (entry.data.frame.codedWidth || 1920) * (entry.data.frame.codedHeight || 1080) * 4
    }
    return total
  }

  private binarySearchInsertPosition(arr: number[], target: number): number {
    let left = 0
    let right = arr.length

    while (left < right) {
      const mid = Math.floor((left + right) / 2)
      if (arr[mid] < target) {
        left = mid + 1
      } else {
        right = mid
      }
    }

    return left
  }

  private findClosestTimestamp(arr: number[], target: number, tolerance: number): number {
    if (arr.length === 0) return -1

    let left = 0
    let right = arr.length - 1

    while (left < right) {
      const mid = Math.floor((left + right) / 2)
      if (arr[mid] < target) {
        left = mid + 1
      } else {
        right = mid
      }
    }

    // Check left and left-1 for closest
    let closestIdx = left
    if (left > 0) {
      const diffLeft = Math.abs(arr[left] - target)
      const diffPrev = Math.abs(arr[left - 1] - target)
      if (diffPrev < diffLeft) {
        closestIdx = left - 1
      }
    }

    // Check if within tolerance
    if (Math.abs(arr[closestIdx] - target) <= tolerance) {
      return closestIdx
    }

    return -1
  }

  private removeFromSorted(arr: number[], value: number): void {
    const idx = arr.indexOf(value)
    if (idx !== -1) {
      arr.splice(idx, 1)
    }
  }
}
