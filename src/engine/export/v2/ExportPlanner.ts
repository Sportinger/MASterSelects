/**
 * ExportPlanner - Analyzes timeline and optimizes decode scheduling
 *
 * Features:
 * - Analyzes full export range to understand file usage patterns
 * - Detects heavy usage files (>20% of export) for larger cache
 * - Plans look-ahead decode scheduling (2-3 seconds ahead)
 * - Minimizes decoder switches and seeks
 * - Groups clips by file for efficient decoding
 */

import { Logger } from '../../../services/logger'
import type { TimelineClip, TimelineTrack } from '../../../stores/timeline/types'
import type { Composition } from '../../../stores/mediaStore/types'
import type {
  DecodeSchedule,
  UsagePattern,
  TimeRange,
  FrameRequest
} from './types'

const log = Logger.create('ExportPlanner')

export class ExportPlanner {
  private startTime: number
  private endTime: number
  private fps: number
  private clips: TimelineClip[]
  private tracks: TimelineTrack[]
  // compositions stored for future nested comp export support
  // private compositions: Map<string, Composition>

  constructor(options: {
    startTime: number
    endTime: number
    fps: number
    clips: TimelineClip[]
    tracks: TimelineTrack[]
    compositions: Composition[]
  }) {
    this.startTime = options.startTime
    this.endTime = options.endTime
    this.fps = options.fps
    this.clips = options.clips
    this.tracks = options.tracks
    // Store compositions for future nested comp support
    void options.compositions

    log.info(`ExportPlanner created: ${(this.endTime - this.startTime).toFixed(1)}s @ ${this.fps}fps`)
  }

  /**
   * Analyze timeline and create decode schedule
   */
  async createSchedule(): Promise<DecodeSchedule> {
    const endAnalysis = log.time('createSchedule')

    // Step 1: Find all video clips (including nested)
    const allVideoClips = this.collectAllVideoClips()
    log.debug(`Found ${allVideoClips.length} total video clips (including nested)`)

    // Step 2: Group clips by fileHash
    const fileUsageMap = new Map<string, {
      clipIds: string[]
      timeRanges: TimeRange[]
      clips: TimelineClip[]
    }>()

    for (const clip of allVideoClips) {
      const fileHash = this.getClipFileHash(clip)
      if (!fileHash) continue

      let usage = fileUsageMap.get(fileHash)
      if (!usage) {
        usage = {
          clipIds: [],
          timeRanges: [],
          clips: []
        }
        fileUsageMap.set(fileHash, usage)
      }

      usage.clipIds.push(clip.id)
      usage.clips.push(clip)

      // Calculate time range for this clip
      const clipStart = clip.startTime
      const clipEnd = clip.startTime + clip.duration

      // Only include if overlaps with export range
      if (clipEnd > this.startTime && clipStart < this.endTime) {
        const rangeStart = Math.max(clipStart, this.startTime)
        const rangeEnd = Math.min(clipEnd, this.endTime)
        usage.timeRanges.push({ start: rangeStart, end: rangeEnd })
      }
    }

    // Step 3: Process each file's usage pattern
    const totalExportDuration = this.endTime - this.startTime
    const fileUsage = new Map<string, UsagePattern>()

    for (const [fileHash, usage] of fileUsageMap) {
      // Merge overlapping time ranges
      const mergedRanges = this.mergeTimeRanges(usage.timeRanges)

      // Calculate total duration this file is used
      const totalUsageDuration = mergedRanges.reduce((sum, range) => {
        return sum + (range.end - range.start)
      }, 0)

      // Calculate total frames needed
      const totalFrames = Math.ceil(totalUsageDuration * this.fps)

      // Determine if heavy usage (>20% of export)
      const usagePercentage = totalUsageDuration / totalExportDuration
      const isHeavyUsage = usagePercentage > 0.2

      const pattern: UsagePattern = {
        fileHash,
        clipIds: [...new Set(usage.clipIds)], // Unique clip IDs
        timeRanges: mergedRanges,
        totalFrames,
        isHeavyUsage
      }

      fileUsage.set(fileHash, pattern)

      log.debug(`File ${fileHash.substring(0, 8)}: ${totalFrames} frames, ${(usagePercentage * 100).toFixed(1)}% usage, heavy=${isHeavyUsage}`)
    }

    const totalFrames = Math.ceil(totalExportDuration * this.fps)
    const estimatedTime = this.estimateExportTime(totalFrames, fileUsage.size)

    const schedule: DecodeSchedule = {
      fileUsage,
      totalFrames,
      estimatedTime
    }

    endAnalysis()
    log.info(`Schedule created: ${fileUsage.size} files, ${totalFrames} frames, ~${Math.ceil(estimatedTime / 60)}min estimated`)

    return schedule
  }

  /**
   * Get next batch of frames to decode (look-ahead)
   * Called during export to plan upcoming decodes
   */
  getNextDecodeBatch(currentTime: number, schedule: DecodeSchedule): FrameRequest[] {
    const requests: FrameRequest[] = []
    const lookAheadDuration = 3.0 // 3 seconds look-ahead
    const lookAheadEndTime = Math.min(currentTime + lookAheadDuration, this.endTime)

    // For each file in use, determine which frames we need
    for (const [fileHash, pattern] of schedule.fileUsage) {
      // Check if file is active in look-ahead window
      const activeInWindow = pattern.timeRanges.some(range =>
        range.start < lookAheadEndTime && range.end > currentTime
      )

      if (!activeInWindow) continue

      // Find clips for this file active in look-ahead window
      const activeClips = this.clips.filter(clip => {
        if (this.getClipFileHash(clip) !== fileHash) return false
        const clipEnd = clip.startTime + clip.duration
        return clip.startTime < lookAheadEndTime && clipEnd > currentTime
      })

      // For each active clip, calculate source times needed
      for (const clip of activeClips) {
        const clipStart = Math.max(clip.startTime, currentTime)
        const clipEnd = Math.min(clip.startTime + clip.duration, lookAheadEndTime)

        // Sample every ~10 frames to avoid too many requests
        const sampleInterval = 10 / this.fps // ~10 frames
        for (let t = clipStart; t < clipEnd; t += sampleInterval) {
          const clipLocalTime = t - clip.startTime
          const sourceTime = this.calculateSourceTime(clip, clipLocalTime)

          // Priority: closer to current time = higher priority
          const timeDelta = t - currentTime
          const priority = Math.max(0, 100 - timeDelta * 10)

          requests.push({
            fileHash,
            clipId: clip.id,
            sourceTime,
            priority,
            isNestedComp: clip.isComposition || false,
            nestedDepth: 0 // TODO: Calculate actual nesting depth
          })
        }
      }
    }

    // Sort by priority (highest first)
    requests.sort((a, b) => b.priority - a.priority)

    // Limit batch size to avoid overwhelming decoder
    const maxBatchSize = 50
    return requests.slice(0, maxBatchSize)
  }

  // Private methods

  /**
   * Collect all video clips including nested clips
   */
  private collectAllVideoClips(): TimelineClip[] {
    const allClips: TimelineClip[] = []

    // Get visible video tracks
    const videoTracks = this.tracks.filter(t => t.type === 'video' && t.visible)

    for (const track of videoTracks) {
      const trackClips = this.clips.filter(c => c.trackId === track.id)

      for (const clip of trackClips) {
        if (clip.source?.type === 'video') {
          allClips.push(clip)
        }

        // If nested composition, collect nested clips recursively
        if (clip.isComposition && clip.nestedClips) {
          const nestedVideoClips = clip.nestedClips.filter(nc => nc.source?.type === 'video')
          allClips.push(...nestedVideoClips)

          // TODO: Handle deeper nesting (triple-nested, etc.)
          // For now, only handle one level deep
        }
      }
    }

    return allClips
  }

  /**
   * Get file hash for a clip (for grouping)
   */
  private getClipFileHash(clip: TimelineClip): string | null {
    // For now, use mediaFileId as fileHash
    // TODO: Calculate actual content hash of file
    return clip.source?.mediaFileId || null
  }

  /**
   * Merge overlapping or adjacent time ranges
   */
  private mergeTimeRanges(ranges: TimeRange[]): TimeRange[] {
    if (ranges.length === 0) return []

    // Sort by start time
    const sorted = [...ranges].sort((a, b) => a.start - b.start)
    const merged: TimeRange[] = [sorted[0]]

    for (let i = 1; i < sorted.length; i++) {
      const current = sorted[i]
      const last = merged[merged.length - 1]

      // If overlapping or adjacent (within 0.1s), merge
      if (current.start <= last.end + 0.1) {
        last.end = Math.max(last.end, current.end)
      } else {
        merged.push(current)
      }
    }

    return merged
  }

  /**
   * Calculate source video time from timeline clip time
   */
  private calculateSourceTime(clip: TimelineClip, clipLocalTime: number): number {
    // Handle reversed clips
    if (clip.reversed) {
      return clip.outPoint - clipLocalTime
    }

    // Normal forward playback
    return clip.inPoint + clipLocalTime
  }

  /**
   * Estimate export time based on project complexity
   */
  private estimateExportTime(totalFrames: number, fileCount: number): number {
    // Rough estimates based on benchmarks:
    // - Simple (1-3 files): ~0.5x realtime (2min video = 1min export)
    // - Medium (4-8 files): ~1x realtime (2min video = 2min export)
    // - Complex (9+ files): ~1.5x realtime (2min video = 3min export)

    const realtimeDuration = totalFrames / this.fps

    let multiplier = 0.5
    if (fileCount > 8) {
      multiplier = 1.5
    } else if (fileCount > 3) {
      multiplier = 1.0
    }

    return realtimeDuration * multiplier
  }
}
