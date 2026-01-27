/**
 * V2ExportBridge - Bridge between FrameExporter and V2 Export System
 *
 * Responsibilities:
 * - Initialize V2 components (SharedDecoderPool, ExportPlanner)
 * - Prepare file data and decoder configs
 * - Coordinate frame requests during export
 * - Handle cleanup and errors
 */

import { Logger } from '../../../services/logger'
import { SharedDecoderPool } from './SharedDecoderPool'
import { ExportPlanner } from './ExportPlanner'
import type { TimelineClip } from '../../../stores/timeline/types'
import type { SharedDecoderConfig, FrameRequest, DecodeSchedule } from './types'
import { ExportError as ExportErrorClass } from './types'
import { loadClipFileData } from '../ClipPreparation'
import { useMediaStore } from '../../../stores/mediaStore'

import * as MP4BoxModule from 'mp4box'
const MP4Box = (MP4BoxModule as any).default || MP4BoxModule

const log = Logger.create('V2ExportBridge')

export class V2ExportBridge {
  private decoderPool: SharedDecoderPool | null = null
  private planner: ExportPlanner | null = null
  private schedule: DecodeSchedule | null = null
  private isInitialized = false

  constructor(private options: {
    maxCacheMemoryMB?: number
    defaultMaxFramesPerFile?: number
  } = {}) {}

  /**
   * Initialize V2 export system
   */
  async initialize(
    clips: TimelineClip[],
    tracks: any[],
    compositions: any[],
    startTime: number,
    endTime: number,
    fps: number
  ): Promise<void> {
    const endInit = log.time('V2 initialization')

    try {
      // Step 1: Create planner and analyze timeline
      log.info('Step 1: Creating export plan...')
      this.planner = new ExportPlanner({
        startTime,
        endTime,
        fps,
        clips,
        tracks,
        compositions
      })

      this.schedule = await this.planner.createSchedule()
      log.info(`Schedule created: ${this.schedule.fileUsage.size} files`)

      // Step 2: Load file data and parse MP4
      log.info('Step 2: Loading and parsing video files...')
      const decoderConfigs = await this.prepareDecoderConfigs(this.schedule)
      log.info(`Prepared ${decoderConfigs.length} decoder configs`)

      // Step 3: Initialize decoder pool
      log.info('Step 3: Initializing decoder pool...')
      this.decoderPool = new SharedDecoderPool({
        maxCacheMemoryMB: this.options.maxCacheMemoryMB ?? 1000,
        defaultMaxFramesPerFile: this.options.defaultMaxFramesPerFile ?? 60
      })

      await this.decoderPool.initialize(decoderConfigs)

      // Step 4: Adjust cache sizes for heavy usage files
      for (const [fileHash, pattern] of this.schedule.fileUsage) {
        if (pattern.isHeavyUsage) {
          // Heavy usage files get 150 frames cache (vs default 60)
          // TODO: Make this configurable
          log.debug(`Setting larger cache for heavy usage file: ${fileHash.substring(0, 8)}`)
        }
      }

      this.isInitialized = true
      endInit()
      log.info('✅ V2 export system initialized successfully')
    } catch (error) {
      endInit()
      log.error('❌ V2 initialization failed:', error)
      this.cleanup()
      throw error
    }
  }

  /**
   * Get frame for a clip at specific time
   */
  async getFrame(clipId: string, timelineTime: number): Promise<VideoFrame> {
    if (!this.isInitialized || !this.decoderPool) {
      throw new ExportErrorClass({
        component: 'SharedDecoder',
        message: 'V2 system not initialized',
        detailedMessage: 'getFrame called before initialize()',
        suggestedAction: 'This is a bug - report to developers'
      })
    }

    // Find clip to get file hash
    // TODO: Store clip metadata during initialization for faster lookup
    const fileHash = 'TODO' // Need to map clipId -> fileHash

    const request: FrameRequest = {
      fileHash,
      clipId,
      sourceTime: timelineTime, // TODO: Calculate actual source time
      priority: 100,
      isNestedComp: false,
      nestedDepth: 0
    }

    return await this.decoderPool.requestFrame(request)
  }

  /**
   * Pre-fetch frames for upcoming timeline position (look-ahead)
   */
  async prefetchFrames(currentTime: number): Promise<void> {
    if (!this.isInitialized || !this.decoderPool || !this.planner || !this.schedule) {
      return
    }

    // Get next batch of frames to decode
    const requests = this.planner.getNextDecodeBatch(currentTime, this.schedule)

    if (requests.length > 0) {
      log.debug(`Prefetching ${requests.length} frames for time ${currentTime.toFixed(2)}s`)
      await this.decoderPool.requestFrameBatch(requests)
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return this.decoderPool?.getCacheStats() || null
  }

  /**
   * Get export schedule
   */
  getSchedule(): DecodeSchedule | null {
    return this.schedule
  }

  /**
   * Cleanup all resources
   */
  cleanup(): void {
    if (this.decoderPool) {
      this.decoderPool.dispose()
      this.decoderPool = null
    }
    this.planner = null
    this.schedule = null
    this.isInitialized = false
    log.info('V2 export system cleaned up')
  }

  // Private methods

  /**
   * Prepare decoder configs for all files in schedule
   */
  private async prepareDecoderConfigs(schedule: DecodeSchedule): Promise<SharedDecoderConfig[]> {
    const configs: SharedDecoderConfig[] = []
    const mediaFiles = useMediaStore.getState().files

    for (const [fileHash, pattern] of schedule.fileUsage) {
      // Get first clip for this file to access file data
      const clipId = pattern.clipIds[0]
      const clip = this.findClip(clipId)
      if (!clip) {
        log.warn(`Could not find clip ${clipId} for file ${fileHash}`)
        continue
      }

      // Load file data
      const mediaFileId = clip.source?.mediaFileId
      const mediaFile = mediaFileId ? mediaFiles.find(f => f.id === mediaFileId) : null

      const fileData = await loadClipFileData(clip, mediaFile)
      if (!fileData) {
        throw new ExportErrorClass({
          component: 'SharedDecoder',
          message: `Could not load file data`,
          clipName: clip.name,
          fileHash,
          detailedMessage: `Failed to load file data for clip "${clip.name}"`,
          suggestedAction: 'Check if file is accessible and not corrupted'
        })
      }

      // Parse MP4 and extract codec config
      const config = await this.parseMP4AndExtractConfig(fileHash, fileData, clip.name)
      configs.push(config)
    }

    return configs
  }

  /**
   * Parse MP4 file and extract codec configuration
   */
  private async parseMP4AndExtractConfig(
    fileHash: string,
    fileData: ArrayBuffer,
    fileName: string
  ): Promise<SharedDecoderConfig> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new ExportErrorClass({
          component: 'SharedDecoder',
          message: 'MP4 parsing timeout',
          clipName: fileName,
          fileHash,
          detailedMessage: `MP4Box parsing timed out after 10 seconds for "${fileName}"`,
          suggestedAction: 'File may be corrupted or in unsupported format'
        }))
      }, 10000)

      const mp4File = MP4Box.createFile()
      let videoTrack: any = null
      const samples: any[] = []

      mp4File.onReady = (info: any) => {
        videoTrack = info.videoTracks[0]
        if (!videoTrack) {
          clearTimeout(timeout)
          reject(new ExportErrorClass({
            component: 'SharedDecoder',
            message: 'No video track found',
            clipName: fileName,
            fileHash,
            detailedMessage: `File "${fileName}" contains no video track`,
            suggestedAction: 'Check if file is a valid video file'
          }))
          return
        }

        // Build codec config
        const codec = videoTrack.codec
        let description: ArrayBuffer | undefined

        try {
          const trak = (mp4File as any).getTrackById(videoTrack.id)
          if (trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]) {
            const entry = trak.mdia.minf.stbl.stsd.entries[0]
            const configBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C
            if (configBox) {
              const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN)
              configBox.write(stream)
              description = stream.buffer.slice(8)
            }
          }
        } catch (e) {
          log.warn(`Failed to extract codec description for ${fileName}: ${e}`)
        }

        const codecConfig: VideoDecoderConfig = {
          codec,
          codedWidth: videoTrack.video.width,
          codedHeight: videoTrack.video.height,
          hardwareAcceleration: 'prefer-software',
          optimizeForLatency: true,
          description
        }

        // Start sample extraction
        mp4File.setExtractionOptions(videoTrack.id, null, { nbSamples: Infinity })
        mp4File.start()
      }

      mp4File.onSamples = (_trackId: number, _ref: any, newSamples: any[]) => {
        samples.push(...newSamples)

        // Resolve once we have all samples (or at least some)
        // For large files, we might want to resolve earlier
        if (samples.length >= 100) {
          clearTimeout(timeout)
          resolve({
            fileHash,
            fileData,
            codecConfig: null as any, // Will be set by onReady
            videoTrack,
            samples
          })
        }
      }

      mp4File.onError = (e: string) => {
        clearTimeout(timeout)
        reject(new ExportErrorClass({
          component: 'SharedDecoder',
          message: 'MP4 parsing error',
          clipName: fileName,
          fileHash,
          detailedMessage: `MP4Box error: ${e}`,
          suggestedAction: 'File may be corrupted or in unsupported format'
        }))
      }

      // Feed buffer to MP4Box
      const mp4Buffer = fileData as any
      mp4Buffer.fileStart = 0
      try {
        mp4File.appendBuffer(mp4Buffer)
        mp4File.flush()
      } catch (e) {
        clearTimeout(timeout)
        reject(new ExportErrorClass({
          component: 'SharedDecoder',
          message: 'MP4Box appendBuffer failed',
          clipName: fileName,
          fileHash,
          detailedMessage: `Failed to parse file: ${e}`,
          suggestedAction: 'File may be corrupted'
        }))
      }
    })
  }

  /**
   * Find clip by ID (helper method - should be optimized with a map)
   */
  private findClip(clipId: string): TimelineClip | null {
    // TODO: Store clips in a map during initialization
    // For now, this is a placeholder
    return null
  }
}
