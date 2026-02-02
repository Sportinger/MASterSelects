/**
 * SharedDecoderPool - Manages shared VideoDecoder instances per unique file
 *
 * Key Features:
 * - One decoder instance per unique file (not per clip instance)
 * - Decoder reuse via reset() + configure()
 * - Integrated frame cache for decoded frames
 * - Smart position tracking to minimize seeks
 *
 * Phase 1: Single-threaded implementation (no workers yet)
 * Phase 2: Will add workers for true parallelism
 */

import { Logger } from '../../../services/logger'
import { FrameCacheManager } from './FrameCacheManager'
import type {
  FrameRequest,
  DecodedFrameData,
  SharedDecoderConfig,
  Sample,
  MP4VideoTrack
} from './types'
import { ExportError as ExportErrorClass } from './types'

// MP4Box import - kept for future direct demuxing support
// import * as MP4BoxModule from 'mp4box'
// const MP4Box = (MP4BoxModule as any).default || MP4BoxModule

const log = Logger.create('SharedDecoderPool')

interface SharedDecoder {
  fileHash: string
  fileName: string
  decoder: VideoDecoder
  codecConfig: VideoDecoderConfig
  samples: Sample[]
  videoTrack: MP4VideoTrack
  currentSampleIndex: number
  currentPosition: number      // Current decode position in seconds
  isDecoding: boolean
  pendingDecode: Promise<void> | null
  needsKeyframe: boolean
  stats: {
    totalDecoded: number
    seeks: number
    resets: number
  }
}

const BUFFER_AHEAD_FRAMES = 30
const DECODE_BATCH_SIZE = 60

export class SharedDecoderPool {
  private decoders: Map<string, SharedDecoder> = new Map()
  private frameCache: FrameCacheManager
  private isActive = false

  constructor(options: {
    maxCacheMemoryMB?: number
    defaultMaxFramesPerFile?: number
  } = {}) {
    this.frameCache = new FrameCacheManager({
      maxMemoryMB: options.maxCacheMemoryMB ?? 1000,
      defaultMaxFramesPerFile: options.defaultMaxFramesPerFile ?? 60
    })
    log.info('SharedDecoderPool initialized')
  }

  /**
   * Initialize pool and prepare decoders for files
   */
  async initialize(configs: SharedDecoderConfig[]): Promise<void> {
    const endInit = log.time('initialize')
    this.isActive = true

    log.info(`Initializing ${configs.length} decoders...`)

    // Initialize all decoders in parallel
    const initPromises = configs.map(config => this.initializeDecoder(config))
    await Promise.all(initPromises)

    log.info(`All ${configs.length} decoders initialized`)
    endInit()
  }

  /**
   * Request frame for a specific file at a specific time
   */
  async requestFrame(request: FrameRequest): Promise<VideoFrame> {
    if (!this.isActive) {
      throw new ExportErrorClass({
        component: 'SharedDecoder',
        message: 'Pool not initialized',
        detailedMessage: 'SharedDecoderPool.initialize() must be called before requesting frames',
        suggestedAction: 'This is a bug - report to developers'
      })
    }

    const decoder = this.decoders.get(request.fileHash)
    if (!decoder) {
      throw new ExportErrorClass({
        component: 'SharedDecoder',
        message: `No decoder for file ${request.fileHash}`,
        fileHash: request.fileHash,
        detailedMessage: `Decoder not found for fileHash ${request.fileHash}`,
        suggestedAction: 'This is a bug - report to developers'
      })
    }

    // Check cache first
    const targetTimestamp = request.sourceTime * 1_000_000
    const tolerance = 50_000 // 50ms in microseconds
    const cached = this.frameCache.get(request.fileHash, targetTimestamp, tolerance)
    if (cached) {
      return cached.frame
    }

    // Not in cache - need to decode
    await this.ensureFrameDecoded(decoder, request.sourceTime)

    // Try cache again after decode
    const decoded = this.frameCache.get(request.fileHash, targetTimestamp, tolerance)
    if (decoded) {
      return decoded.frame
    }

    // Still not available - error
    throw new ExportErrorClass({
      component: 'SharedDecoder',
      message: `Frame not available after decode`,
      clipName: request.clipId,
      fileHash: request.fileHash,
      detailedMessage: `Failed to decode frame at ${request.sourceTime.toFixed(3)}s for file ${decoder.fileName}`,
      suggestedAction: 'Try Legacy System (V1) in Export Settings'
    })
  }

  /**
   * Request multiple frames in a batch (for look-ahead)
   */
  async requestFrameBatch(requests: FrameRequest[]): Promise<void> {
    // Group by fileHash
    const byFile = new Map<string, FrameRequest[]>()
    for (const req of requests) {
      const existing = byFile.get(req.fileHash) || []
      existing.push(req)
      byFile.set(req.fileHash, existing)
    }

    // Process each file's requests
    const promises: Promise<void>[] = []
    for (const [fileHash, fileRequests] of byFile) {
      const decoder = this.decoders.get(fileHash)
      if (!decoder) continue

      // Sort by sourceTime
      fileRequests.sort((a, b) => a.sourceTime - b.sourceTime)

      // Decode range
      const minTime = fileRequests[0].sourceTime
      const maxTime = fileRequests[fileRequests.length - 1].sourceTime

      promises.push(this.decodeRange(decoder, minTime, maxTime))
    }

    await Promise.all(promises)
  }

  /**
   * Check if decoder exists for file
   */
  hasDecoder(fileHash: string): boolean {
    return this.decoders.has(fileHash)
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return this.frameCache.getStats()
  }

  /**
   * Cleanup all resources
   */
  dispose(): void {
    this.isActive = false

    for (const decoder of this.decoders.values()) {
      try {
        decoder.decoder.close()
      } catch (e) {
        // Ignore
      }
    }

    this.decoders.clear()
    this.frameCache.dispose()
    log.info('Disposed')
  }

  // Private methods

  private async initializeDecoder(config: SharedDecoderConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new ExportErrorClass({
          component: 'SharedDecoder',
          message: `Initialization timeout for ${config.fileHash}`,
          fileHash: config.fileHash,
          detailedMessage: `Decoder initialization timed out after 5 seconds`,
          suggestedAction: 'Try Legacy System (V1) in Export Settings'
        }))
      }, 5000)

      try {
        const decoder = new VideoDecoder({
          output: (frame) => this.handleDecodedFrame(config.fileHash, frame),
          error: (e) => {
            log.error(`Decoder error for ${config.fileHash}: ${e.message || e}`)
          }
        })

        decoder.configure(config.codecConfig)

        const sharedDecoder: SharedDecoder = {
          fileHash: config.fileHash,
          fileName: config.fileHash.substring(0, 8), // TODO: Store actual filename
          decoder,
          codecConfig: config.codecConfig,
          samples: config.samples,
          videoTrack: config.videoTrack,
          currentSampleIndex: 0,
          currentPosition: 0,
          isDecoding: false,
          pendingDecode: null,
          needsKeyframe: false,
          stats: {
            totalDecoded: 0,
            seeks: 0,
            resets: 0
          }
        }

        this.decoders.set(config.fileHash, sharedDecoder)

        clearTimeout(timeout)
        log.info(`Decoder initialized for ${config.fileHash}: ${config.videoTrack.video.width}x${config.videoTrack.video.height}`)
        resolve()
      } catch (e) {
        clearTimeout(timeout)
        reject(new ExportErrorClass({
          component: 'SharedDecoder',
          message: `Failed to create decoder`,
          fileHash: config.fileHash,
          detailedMessage: `VideoDecoder creation failed: ${e}`,
          suggestedAction: 'Try Legacy System (V1) in Export Settings'
        }))
      }
    })
  }

  private handleDecodedFrame(fileHash: string, frame: VideoFrame): void {
    const timestamp = frame.timestamp
    const sourceTime = timestamp / 1_000_000

    const data: DecodedFrameData = {
      frame,
      sourceTime,
      timestamp,
      fileHash
    }

    this.frameCache.put(data)

    const decoder = this.decoders.get(fileHash)
    if (decoder) {
      decoder.stats.totalDecoded++
      decoder.currentPosition = sourceTime
    }
  }

  private async ensureFrameDecoded(decoder: SharedDecoder, targetTime: number): Promise<void> {
    // Check cache first (might have been decoded by another request)
    const targetTimestamp = targetTime * 1_000_000
    if (this.frameCache.has(decoder.fileHash, targetTimestamp, 50_000)) {
      return
    }

    // Need to decode
    const targetSampleIndex = this.findSampleIndexForTime(decoder, targetTime)
    await this.decodeToSample(decoder, targetSampleIndex + BUFFER_AHEAD_FRAMES)
  }

  private async decodeRange(decoder: SharedDecoder, startTime: number, endTime: number): Promise<void> {
    // startTime used for future partial range decoding optimization
    void startTime
    const endSampleIndex = this.findSampleIndexForTime(decoder, endTime)

    await this.decodeToSample(decoder, endSampleIndex + BUFFER_AHEAD_FRAMES)
  }

  private async decodeToSample(decoder: SharedDecoder, targetSampleIndex: number): Promise<void> {
    if (decoder.isDecoding) {
      await decoder.pendingDecode
      return
    }

    decoder.isDecoding = true

    decoder.pendingDecode = (async () => {
      try {
        // Check if need to seek (target is far ahead)
        const needsSeek = targetSampleIndex > decoder.currentSampleIndex + 30

        if (needsSeek) {
          await this.seekToSample(decoder, targetSampleIndex)
        }

        // Decode frames
        const endIndex = Math.min(targetSampleIndex, decoder.samples.length)
        let framesToDecode = endIndex - decoder.currentSampleIndex

        if (framesToDecode <= 0) {
          return
        }

        framesToDecode = Math.min(framesToDecode, DECODE_BATCH_SIZE)

        log.debug(`${decoder.fileName}: Decoding ${framesToDecode} frames (${decoder.currentSampleIndex} -> ${decoder.currentSampleIndex + framesToDecode})`)

        for (let i = 0; i < framesToDecode && decoder.currentSampleIndex < decoder.samples.length; i++) {
          const sample = decoder.samples[decoder.currentSampleIndex]
          decoder.currentSampleIndex++

          const chunk = new EncodedVideoChunk({
            type: sample.is_sync ? 'key' : 'delta',
            timestamp: (sample.cts * 1_000_000) / sample.timescale,
            duration: (sample.duration * 1_000_000) / sample.timescale,
            data: sample.data
          })

          decoder.decoder.decode(chunk)
        }

        // Flush to get frames out
        await decoder.decoder.flush()
        decoder.needsKeyframe = true
      } catch (e) {
        throw new ExportErrorClass({
          component: 'SharedDecoder',
          message: `Decode failed`,
          fileHash: decoder.fileHash,
          detailedMessage: `Decoding failed: ${e}`,
          suggestedAction: 'Try Legacy System (V1) in Export Settings'
        })
      } finally {
        decoder.isDecoding = false
        decoder.pendingDecode = null
      }
    })()

    await decoder.pendingDecode
  }

  private async seekToSample(decoder: SharedDecoder, targetSampleIndex: number): Promise<void> {
    // Find nearest keyframe before target
    let keyframeIndex = targetSampleIndex
    for (let i = targetSampleIndex; i >= 0; i--) {
      if (decoder.samples[i].is_sync) {
        keyframeIndex = i
        break
      }
    }

    log.info(`${decoder.fileName}: Seeking to keyframe at sample ${keyframeIndex} (target=${targetSampleIndex})`)
    decoder.stats.seeks++

    // Reset decoder
    try {
      decoder.decoder.reset()
      decoder.stats.resets++
      decoder.decoder.configure(decoder.codecConfig)
      decoder.currentSampleIndex = keyframeIndex
      decoder.needsKeyframe = false
    } catch (e) {
      throw new ExportErrorClass({
        component: 'SharedDecoder',
        message: `Seek failed`,
        fileHash: decoder.fileHash,
        detailedMessage: `Decoder reset/configure failed during seek: ${e}`,
        suggestedAction: 'This codec may not support decoder reuse. Try Legacy System (V1) in Export Settings'
      })
    }
  }

  private findSampleIndexForTime(decoder: SharedDecoder, sourceTime: number): number {
    const targetTime = sourceTime * decoder.videoTrack.timescale
    const samples = decoder.samples

    if (samples.length === 0) return 0

    // Linear search for closest CTS (samples may be out of order due to B-frames)
    let targetIndex = 0
    let closestDiff = Infinity

    for (let i = 0; i < samples.length; i++) {
      const diff = Math.abs(samples[i].cts - targetTime)
      if (diff < closestDiff) {
        closestDiff = diff
        targetIndex = i
      }
    }

    return targetIndex
  }
}
