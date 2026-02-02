/**
 * Type definitions for Shared Decoder Export System V2
 */

export interface FrameRequest {
  fileHash: string
  clipId: string
  sourceTime: number        // Time in source video (seconds)
  priority: number          // Higher = needed sooner (0-100)
  isNestedComp: boolean
  nestedDepth: number       // 0 = main timeline, 1+ = nested
}

export interface DecodedFrameData {
  frame: VideoFrame
  sourceTime: number        // Time in source video (seconds)
  timestamp: number         // Original timestamp from VideoFrame (microseconds)
  fileHash: string
}

export interface SharedDecoderConfig {
  fileHash: string
  fileData: ArrayBuffer
  codecConfig: VideoDecoderConfig
  videoTrack: MP4VideoTrack
  samples: Sample[]
}

export interface CacheStats {
  totalFrames: number
  totalMemoryMB: number
  hitRate: number
  missRate: number
  evictions: number
  perFileStats: Map<string, FileCacheStats>
}

export interface FileCacheStats {
  fileHash: string
  frames: number
  memoryMB: number
  hits: number
  misses: number
  evictions: number
}

export interface UsagePattern {
  fileHash: string
  clipIds: string[]
  timeRanges: TimeRange[]
  totalFrames: number
  isHeavyUsage: boolean     // > 20% of export
}

export interface TimeRange {
  start: number             // seconds
  end: number               // seconds
}

export interface DecodeSchedule {
  fileUsage: Map<string, UsagePattern>
  totalFrames: number
  estimatedTime: number     // seconds
}

// MP4Box types (minimal definitions)
export interface Sample {
  number: number
  track_id: number
  data: ArrayBuffer
  size: number
  cts: number
  dts: number
  duration: number
  is_sync: boolean
  timescale: number
}

export interface MP4VideoTrack {
  id: number
  codec: string
  duration: number
  timescale: number
  nb_samples: number
  video: { width: number; height: number }
}

// Clip metadata for V2ExportBridge
export interface ClipMetadata {
  clip: import('../../../stores/timeline/types').TimelineClip
  fileHash: string
  fileName: string
  mediaFileId: string | null
  isNested: boolean
  parentClipId: string | null
}

// Error types
export type ExportErrorComponent = 'SharedDecoder' | 'FrameCache' | 'Worker' | 'NestedRenderer'

export class ExportError extends Error {
  component: ExportErrorComponent
  clipName?: string
  fileHash?: string
  detailedMessage: string
  suggestedAction: string

  constructor(options: {
    component: ExportErrorComponent
    message: string
    clipName?: string
    fileHash?: string
    detailedMessage: string
    suggestedAction: string
  }) {
    super(options.message)
    this.name = 'ExportError'
    this.component = options.component
    this.clipName = options.clipName
    this.fileHash = options.fileHash
    this.detailedMessage = options.detailedMessage
    this.suggestedAction = options.suggestedAction
  }
}
