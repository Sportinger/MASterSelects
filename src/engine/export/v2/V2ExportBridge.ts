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
import type { SharedDecoderConfig, FrameRequest, DecodeSchedule, ClipMetadata } from './types'
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
  private allClips: TimelineClip[] = []
  private clipMetadata: Map<string, ClipMetadata> = new Map()
  private options: {
    maxCacheMemoryMB?: number
    defaultMaxFramesPerFile?: number
  }

  constructor(options: {
    maxCacheMemoryMB?: number
    defaultMaxFramesPerFile?: number
  } = {}) {
    this.options = options
  }

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
      // Step 0: Store clips for later lookup
      this.allClips = clips
      this.clipMetadata = new Map()

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

      // Build clip metadata for fast lookup
      this.buildClipMetadata(this.schedule)

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

    // Get clip metadata for fileHash and source time calculation
    const metadata = this.clipMetadata.get(clipId)
    if (!metadata) {
      throw new ExportErrorClass({
        component: 'SharedDecoder',
        message: 'Clip not found in metadata',
        detailedMessage: `Clip ${clipId} not found in clipMetadata. This may indicate the clip was not part of the export schedule.`,
        suggestedAction: 'Ensure the clip is visible in the export range'
      })
    }

    const sourceTime = this.calculateSourceTime(metadata.clip, timelineTime)

    const request: FrameRequest = {
      fileHash: metadata.fileHash,
      clipId,
      sourceTime,
      priority: 100,
      isNestedComp: metadata.isNested,
      nestedDepth: metadata.isNested ? 1 : 0
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
    this.allClips = []
    this.clipMetadata.clear()
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
          detailedMessage: `MP4Box parsing timed out after 30 seconds for "${fileName}"`,
          suggestedAction: 'File may be corrupted or in unsupported format'
        }))
      }, 30000) // Increased timeout for large files

      const mp4File = MP4Box.createFile()
      let videoTrack: any = null
      let codecConfig: VideoDecoderConfig | null = null
      const samples: any[] = []
      let resolved = false

      const tryResolve = () => {
        if (resolved) return
        if (!videoTrack || !codecConfig) return

        // Wait for all samples (or at least enough for the file)
        const expectedSamples = videoTrack.nb_samples
        if (samples.length < expectedSamples) {
          log.debug(`Waiting for samples: ${samples.length}/${expectedSamples} for ${fileName}`)
          return
        }

        resolved = true
        clearTimeout(timeout)
        log.info(`MP4 parsed: ${fileName} - ${samples.length} samples, ${videoTrack.video.width}x${videoTrack.video.height}`)
        resolve({
          fileHash,
          fileData,
          codecConfig,
          videoTrack,
          samples
        })
      }

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

        codecConfig = {
          codec,
          codedWidth: videoTrack.video.width,
          codedHeight: videoTrack.video.height,
          hardwareAcceleration: 'prefer-software',
          optimizeForLatency: true,
          description
        }

        log.debug(`Starting sample extraction for ${fileName}: expecting ${videoTrack.nb_samples} samples`)

        // Start sample extraction
        mp4File.setExtractionOptions(videoTrack.id, null, { nbSamples: Infinity })
        mp4File.start()
      }

      mp4File.onSamples = (_trackId: number, _ref: any, newSamples: any[]) => {
        samples.push(...newSamples)
        tryResolve()
      }

      // onFlush is called when all data has been processed
      mp4File.onFlush = () => {
        if (resolved) return
        if (!videoTrack || !codecConfig) {
          clearTimeout(timeout)
          reject(new ExportErrorClass({
            component: 'SharedDecoder',
            message: 'MP4 flush without ready',
            clipName: fileName,
            fileHash,
            detailedMessage: `MP4Box flushed before onReady for "${fileName}"`,
            suggestedAction: 'File may be corrupted'
          }))
          return
        }

        // Force resolve even if we don't have all samples (edge case)
        resolved = true
        clearTimeout(timeout)
        log.info(`MP4 parsed (flush): ${fileName} - ${samples.length}/${videoTrack.nb_samples} samples`)
        resolve({
          fileHash,
          fileData,
          codecConfig,
          videoTrack,
          samples
        })
      }

      mp4File.onError = (e: string) => {
        if (resolved) return
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
   * Find clip by ID using metadata map
   */
  private findClip(clipId: string): TimelineClip | null {
    return this.clipMetadata.get(clipId)?.clip || null
  }

  /**
   * Build clip metadata map from schedule for fast O(1) lookup
   */
  private buildClipMetadata(schedule: DecodeSchedule): void {
    for (const [fileHash, pattern] of schedule.fileUsage) {
      for (const clipId of pattern.clipIds) {
        const clip = this.allClips.find(c => c.id === clipId)
        if (!clip) {
          log.warn(`Clip ${clipId} not found in timeline during metadata build`)
          continue
        }

        this.clipMetadata.set(clipId, {
          clip,
          fileHash,
          fileName: clip.name,
          mediaFileId: clip.source?.mediaFileId || null,
          isNested: false, // TODO: Detect from composition structure
          parentClipId: clip.parentClipId || null
        })
      }
    }

    log.info(`Built metadata for ${this.clipMetadata.size} clips`)
  }

  /**
   * Calculate source time from timeline time
   * Handles reversed clips and in/out points
   */
  private calculateSourceTime(clip: TimelineClip, timelineTime: number): number {
    // Convert timeline time to clip local time
    const clipLocalTime = timelineTime - clip.startTime

    // Handle speed adjustment (if any)
    const speed = clip.speed ?? 1.0
    const adjustedLocalTime = clipLocalTime * Math.abs(speed)

    // Handle reversed clips
    if (clip.reversed || speed < 0) {
      return clip.outPoint - adjustedLocalTime
    }

    // Normal forward playback
    return clip.inPoint + adjustedLocalTime
  }
}
