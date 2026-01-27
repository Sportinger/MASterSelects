/**
 * SystemSelector - Chooses best export system (V1 or V2) based on project complexity
 *
 * Decision Matrix:
 * - Simple projects (≤8 files, ≤5 nested clips) → V1 (proven, stable)
 * - Complex projects (>8 files or >5 nested clips) → V2 (shared decoders)
 * - Manual override available in export settings
 */

import { Logger } from '../../../services/logger'
import type { TimelineClip, TimelineTrack } from '../../../stores/timeline/types'
import type { Composition } from '../../../stores/mediaStore/types'

const log = Logger.create('SystemSelector')

export type ExportSystemVersion = 'v1' | 'v2'

export interface SystemSelectionResult {
  version: ExportSystemVersion
  reason: string
  stats: {
    totalClips: number
    videoClips: number
    uniqueFiles: number
    nestedClips: number
    nestedCompositions: number
    maxNestingDepth: number
  }
}

export interface SelectionCriteria {
  clips: TimelineClip[]
  tracks: TimelineTrack[]
  compositions: Composition[]
  startTime: number
  endTime: number
}

export class SystemSelector {
  /**
   * Automatically select best system based on project complexity
   */
  static selectSystem(criteria: SelectionCriteria): SystemSelectionResult {
    const stats = this.analyzeComplexity(criteria)

    log.info(`Project complexity: ${stats.uniqueFiles} files, ${stats.nestedClips} nested clips, depth=${stats.maxNestingDepth}`)

    // Decision logic
    let version: ExportSystemVersion
    let reason: string

    // Rule 1: Very simple projects → V1
    if (stats.uniqueFiles <= 3 && stats.nestedClips === 0) {
      version = 'v1'
      reason = 'Simple project (≤3 files, no nesting) - using proven V1 system'
    }
    // Rule 2: Medium projects with no/light nesting → V1
    else if (stats.uniqueFiles <= 8 && stats.nestedClips <= 5) {
      version = 'v1'
      reason = 'Medium complexity (≤8 files, ≤5 nested) - V1 can handle efficiently'
    }
    // Rule 3: Deep nesting → V2
    else if (stats.maxNestingDepth >= 3) {
      version = 'v2'
      reason = 'Deep nesting detected (depth ≥3) - V2 needed for shared decoders'
    }
    // Rule 4: Many nested clips → V2
    else if (stats.nestedClips > 5) {
      version = 'v2'
      reason = `Many nested clips (${stats.nestedClips}) - V2 prevents decoder duplication`
    }
    // Rule 5: Many files → V2
    else if (stats.uniqueFiles > 8) {
      version = 'v2'
      reason = `Complex project (${stats.uniqueFiles} files) - V2 needed for shared decoders`
    }
    // Default: V1 for safety
    else {
      version = 'v1'
      reason = 'Default to V1 for reliability'
    }

    log.info(`Selected: ${version.toUpperCase()} - ${reason}`)

    return {
      version,
      reason,
      stats
    }
  }

  /**
   * Check if V2 is required (will fail if forced to use V1)
   */
  static isV2Required(criteria: SelectionCriteria): boolean {
    const stats = this.analyzeComplexity(criteria)

    // V2 is required if:
    // - More than 15 unique files (V1 will be too slow)
    // - Nesting depth > 3 (V1 will have decoder conflicts)
    // - More than 10 nested clips (too many duplicate decoders in V1)

    if (stats.uniqueFiles > 15) return true
    if (stats.maxNestingDepth > 3) return true
    if (stats.nestedClips > 10) return true

    return false
  }

  /**
   * Analyze project complexity
   */
  private static analyzeComplexity(criteria: SelectionCriteria): SystemSelectionResult['stats'] {
    const { clips, tracks, compositions, startTime, endTime } = criteria

    // Get visible video tracks
    const videoTracks = tracks.filter(t => t.type === 'video' && t.visible)

    // Get clips in export range
    const clipsInRange = clips.filter(clip => {
      const clipEnd = clip.startTime + clip.duration
      return clip.startTime < endTime && clipEnd > startTime
    })

    const totalClips = clipsInRange.length

    // Count video clips
    const videoClips = clipsInRange.filter(c =>
      c.source?.type === 'video' && videoTracks.some(t => t.id === c.trackId)
    )

    // Count unique files (by mediaFileId)
    const uniqueFileIds = new Set<string>()
    const processClip = (clip: TimelineClip) => {
      if (clip.source?.type === 'video' && clip.source.mediaFileId) {
        uniqueFileIds.add(clip.source.mediaFileId)
      }
    }

    // Process main timeline clips
    videoClips.forEach(processClip)

    // Count nested clips and compositions
    let nestedClips = 0
    let nestedCompositions = 0
    let maxNestingDepth = 0

    const processComposition = (clip: TimelineClip, depth: number) => {
      if (!clip.isComposition || !clip.nestedClips) return

      nestedCompositions++
      maxNestingDepth = Math.max(maxNestingDepth, depth)

      for (const nestedClip of clip.nestedClips) {
        if (nestedClip.source?.type === 'video') {
          nestedClips++
          processClip(nestedClip)

          // Check for deeper nesting
          if (nestedClip.isComposition) {
            processComposition(nestedClip, depth + 1)
          }
        }
      }
    }

    // Process compositions
    for (const clip of clipsInRange) {
      if (clip.isComposition) {
        processComposition(clip, 1)
      }
    }

    const uniqueFiles = uniqueFileIds.size

    return {
      totalClips,
      videoClips: videoClips.length,
      uniqueFiles,
      nestedClips,
      nestedCompositions,
      maxNestingDepth
    }
  }

  /**
   * Get human-readable recommendation
   */
  static getRecommendation(criteria: SelectionCriteria): string {
    const result = this.selectSystem(criteria)
    const { stats } = result

    if (result.version === 'v1') {
      return `Your project has ${stats.uniqueFiles} unique video file${stats.uniqueFiles !== 1 ? 's' : ''} and ${stats.nestedClips} nested clip${stats.nestedClips !== 1 ? 's' : ''}. The Legacy System (V1) is recommended for optimal stability.`
    } else {
      return `Your project has ${stats.uniqueFiles} unique video file${stats.uniqueFiles !== 1 ? 's' : ''} and ${stats.nestedClips} nested clip${stats.nestedClips !== 1 ? 's' : ''}. The Shared Decoder System (V2) is required for optimal performance with this complexity.`
    }
  }
}
