/**
 * ParallelDecodeManager - Parallel video decoding for multi-clip exports
 *
 * Problem: Sequential decoding of multiple videos is slow because each video
 * waits for the previous one to decode before proceeding.
 *
 * Solution: Pre-decode frames in parallel using separate VideoDecoder instances
 * per clip, with a frame buffer that stays ahead of the render position.
 */

import * as MP4BoxModule from 'mp4box';
const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

// MP4Box types
interface MP4ArrayBuffer extends ArrayBuffer {
  fileStart: number;
}

interface Sample {
  number: number;
  track_id: number;
  data: ArrayBuffer;
  size: number;
  cts: number;
  dts: number;
  duration: number;
  is_sync: boolean;
  timescale: number;
}

interface MP4VideoTrack {
  id: number;
  codec: string;
  duration: number;
  timescale: number;
  nb_samples: number;
  video: { width: number; height: number };
}

interface MP4File {
  onReady: (info: { videoTracks: MP4VideoTrack[] }) => void;
  onSamples: (trackId: number, ref: any, samples: Sample[]) => void;
  onError: (error: string) => void;
  appendBuffer: (buffer: MP4ArrayBuffer) => number;
  start: () => void;
  flush: () => void;
  setExtractionOptions: (trackId: number, user: any, options: { nbSamples: number }) => void;
}

interface ClipInfo {
  clipId: string;
  clipName: string;
  fileData: ArrayBuffer;
  startTime: number;      // Clip start on timeline (or within composition for nested)
  duration: number;       // Clip duration on timeline
  inPoint: number;        // Source in point
  outPoint: number;       // Source out point
  reversed: boolean;
  // Nested clip properties
  isNested?: boolean;
  parentClipId?: string;
  parentStartTime?: number;   // Parent composition's start on main timeline
  parentInPoint?: number;     // Parent composition's in point
}

interface DecodedFrame {
  frame: VideoFrame;
  sourceTime: number;     // Time in source video (seconds)
  timestamp: number;      // Original timestamp from VideoFrame (microseconds)
}

interface ClipDecoder {
  clipId: string;
  clipName: string;
  decoder: VideoDecoder;
  samples: Sample[];
  sampleIndex: number;
  videoTrack: MP4VideoTrack;
  codecConfig: VideoDecoderConfig;
  frameBuffer: Map<number, DecodedFrame>;  // timestamp (μs) -> decoded frame
  sortedTimestamps: number[];              // Sorted list for O(log n) lookup
  oldestTimestamp: number;                 // Track bounds for quick rejection
  newestTimestamp: number;                 // Track bounds for quick rejection
  lastDecodedTimestamp: number;            // Track last decoded timestamp
  clipInfo: ClipInfo;
  isDecoding: boolean;
  pendingDecode: Promise<void> | null;
}

// Buffer settings - tuned for speed like After Effects
const BUFFER_AHEAD_FRAMES = 30;   // Pre-decode this many frames ahead (1 second at 30fps)
const MAX_BUFFER_SIZE = 60;       // Maximum frames to keep in buffer
const DECODE_BATCH_SIZE = 60;     // Decode this many frames per batch - large for initial catchup

export class ParallelDecodeManager {
  private clipDecoders: Map<string, ClipDecoder> = new Map();
  private isActive = false;
  private decodePromises: Map<string, Promise<void>> = new Map();
  private exportFps = 30;
  private frameTolerance = 50_000;  // Default 50ms in microseconds

  /**
   * Initialize the manager with clips to decode
   */
  async initialize(clips: ClipInfo[], exportFps: number): Promise<void> {
    this.isActive = true;
    this.exportFps = exportFps;
    // FPS-based tolerance: 1.5 frame duration
    this.frameTolerance = Math.round((1_000_000 / exportFps) * 1.5);

    console.log(`[ParallelDecode] Initializing ${clips.length} clips at ${exportFps}fps (tolerance: ${this.frameTolerance}μs)...`);

    // Parse all clips in parallel
    const initPromises = clips.map(clip => this.initializeClip(clip));
    await Promise.all(initPromises);

    console.log(`[ParallelDecode] All ${clips.length} clips initialized`);
  }

  /**
   * Initialize a single clip decoder
   */
  private async initializeClip(clipInfo: ClipInfo): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`MP4 parsing timeout for clip "${clipInfo.clipName}"`));
      }, 15000);

      const mp4File = MP4Box.createFile() as MP4File;
      let samples: Sample[] = [];
      let videoTrack: MP4VideoTrack | null = null;
      let codecConfig: VideoDecoderConfig | null = null;

      mp4File.onReady = (info) => {
        videoTrack = info.videoTracks[0];
        if (!videoTrack) {
          clearTimeout(timeout);
          reject(new Error(`No video track in clip "${clipInfo.clipName}"`));
          return;
        }

        // Build codec config
        const codec = this.getCodecString(videoTrack);
        let description: ArrayBuffer | undefined;

        try {
          const trak = (mp4File as any).getTrackById(videoTrack.id);
          if (trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]) {
            const entry = trak.mdia.minf.stbl.stsd.entries[0];
            const configBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
            if (configBox) {
              const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
              configBox.write(stream);
              description = stream.buffer.slice(8);
            }
          }
        } catch (e) {
          console.warn(`[ParallelDecode] Failed to extract codec description for ${clipInfo.clipName}:`, e);
        }

        codecConfig = {
          codec,
          codedWidth: videoTrack.video.width,
          codedHeight: videoTrack.video.height,
          hardwareAcceleration: 'prefer-hardware',
          optimizeForLatency: true,
          description,
        };

        mp4File.setExtractionOptions(videoTrack.id, null, { nbSamples: Infinity });
        mp4File.start();
      };

      mp4File.onSamples = (_trackId, _ref, newSamples) => {
        samples.push(...newSamples);

        // Once we have samples, create the decoder
        if (samples.length > 0 && videoTrack && codecConfig && !this.clipDecoders.has(clipInfo.clipId)) {
          clearTimeout(timeout);

          // Create VideoDecoder for this clip
          const decoder = new VideoDecoder({
            output: (frame) => {
              const clipDecoder = this.clipDecoders.get(clipInfo.clipId);
              if (clipDecoder) {
                this.handleDecodedFrame(clipDecoder, frame);
              } else {
                frame.close();
              }
            },
            error: (e) => {
              console.error(`[ParallelDecode] Decoder error for ${clipInfo.clipName}:`, e);
            },
          });

          decoder.configure(codecConfig);

          const clipDecoder: ClipDecoder = {
            clipId: clipInfo.clipId,
            clipName: clipInfo.clipName,
            decoder,
            samples,
            sampleIndex: 0,
            videoTrack,
            codecConfig,
            frameBuffer: new Map(),
            sortedTimestamps: [],
            oldestTimestamp: Infinity,
            newestTimestamp: -Infinity,
            lastDecodedTimestamp: 0,
            clipInfo,
            isDecoding: false,
            pendingDecode: null,
          };

          this.clipDecoders.set(clipInfo.clipId, clipDecoder);

          console.log(`[ParallelDecode] Clip "${clipInfo.clipName}" ready: ${videoTrack.video.width}x${videoTrack.video.height}, ${samples.length} samples`);
          resolve();
        }
      };

      mp4File.onError = (e) => {
        clearTimeout(timeout);
        reject(new Error(`MP4 parsing error for "${clipInfo.clipName}": ${e}`));
      };

      // Feed buffer to MP4Box
      const mp4Buffer = clipInfo.fileData as MP4ArrayBuffer;
      mp4Buffer.fileStart = 0;
      try {
        mp4File.appendBuffer(mp4Buffer);
        mp4File.flush();
      } catch (e) {
        clearTimeout(timeout);
        reject(new Error(`MP4Box appendBuffer failed for "${clipInfo.clipName}": ${e}`));
      }
    });
  }

  /**
   * Handle a decoded frame from VideoDecoder output callback
   * Uses the frame's timestamp directly for accurate time mapping
   * Optimized: maintains sorted timestamp list for O(log n) lookups
   */
  private handleDecodedFrame(clipDecoder: ClipDecoder, frame: VideoFrame): void {
    const timestamp = frame.timestamp;  // microseconds
    const sourceTime = timestamp / 1_000_000;  // convert to seconds

    // Store frame by its timestamp
    clipDecoder.frameBuffer.set(timestamp, {
      frame,
      sourceTime,
      timestamp,
    });

    // Maintain sorted timestamp list with binary insertion - O(log n)
    const insertIdx = this.binarySearchInsertPosition(clipDecoder.sortedTimestamps, timestamp);
    clipDecoder.sortedTimestamps.splice(insertIdx, 0, timestamp);

    // Update bounds - O(1)
    if (timestamp < clipDecoder.oldestTimestamp) {
      clipDecoder.oldestTimestamp = timestamp;
    }
    if (timestamp > clipDecoder.newestTimestamp) {
      clipDecoder.newestTimestamp = timestamp;
    }

    clipDecoder.lastDecodedTimestamp = timestamp;

    // Cleanup if buffer too large - remove oldest (no sorting needed)
    if (clipDecoder.frameBuffer.size > MAX_BUFFER_SIZE) {
      const oldestTs = clipDecoder.sortedTimestamps.shift()!;
      const oldFrame = clipDecoder.frameBuffer.get(oldestTs);
      if (oldFrame) {
        oldFrame.frame.close();
        clipDecoder.frameBuffer.delete(oldestTs);
      }
      // Update oldest bound
      clipDecoder.oldestTimestamp = clipDecoder.sortedTimestamps[0] ?? Infinity;
    }
  }

  /**
   * Binary search for insert position - O(log n)
   */
  private binarySearchInsertPosition(arr: number[], target: number): number {
    let left = 0;
    let right = arr.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (arr[mid] < target) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    return left;
  }

  /**
   * Convert timeline time to source video time
   * For nested clips, timelineTime is the MAIN timeline time
   */
  private timelineToSourceTime(clipInfo: ClipInfo, timelineTime: number): number {
    let clipLocalTime: number;

    if (clipInfo.isNested && clipInfo.parentStartTime !== undefined) {
      // Convert main timeline time to composition time
      const compTime = timelineTime - clipInfo.parentStartTime - (clipInfo.parentInPoint || 0);
      // Then to clip local time
      clipLocalTime = compTime - clipInfo.startTime;
    } else {
      clipLocalTime = timelineTime - clipInfo.startTime;
    }

    if (clipInfo.reversed) {
      return clipInfo.outPoint - clipLocalTime;
    }
    return clipInfo.inPoint + clipLocalTime;
  }

  /**
   * Check if a timeline time falls within this clip's range
   */
  private isTimeInClipRange(clipInfo: ClipInfo, timelineTime: number): boolean {
    if (clipInfo.isNested && clipInfo.parentStartTime !== undefined) {
      // Convert main timeline time to composition time
      const compTime = timelineTime - clipInfo.parentStartTime - (clipInfo.parentInPoint || 0);
      // Check if within nested clip's range in the composition
      return compTime >= clipInfo.startTime && compTime < clipInfo.startTime + clipInfo.duration;
    }

    // Regular clip check
    return timelineTime >= clipInfo.startTime && timelineTime < clipInfo.startTime + clipInfo.duration;
  }

  /**
   * Pre-decode frames for a specific timeline time across all clips
   * Optimized for speed: fires decode ahead in background, only waits if frame is missing
   */
  async prefetchFramesForTime(timelineTime: number): Promise<void> {
    if (!this.isActive) return;

    const clipsNeedingFlush: ClipDecoder[] = [];

    for (const [, clipDecoder] of this.clipDecoders) {
      const clipInfo = clipDecoder.clipInfo;

      // Skip if timeline time is outside this clip's range (handles nested clips too)
      if (!this.isTimeInClipRange(clipInfo, timelineTime)) {
        continue;
      }

      // Calculate target source time and sample index
      const sourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
      const targetSampleIndex = this.findSampleIndexForTime(clipDecoder, sourceTime);

      // Check if frame is already in buffer (fast path)
      const targetTimestamp = sourceTime * 1_000_000;
      let frameInBuffer = false;
      for (const [, decodedFrame] of clipDecoder.frameBuffer) {
        if (Math.abs(decodedFrame.timestamp - targetTimestamp) < 50_000) { // 50ms tolerance
          frameInBuffer = true;
          break;
        }
      }

      // Trigger decode ahead - await if we need the frame NOW
      const needsDecoding = clipDecoder.sampleIndex < targetSampleIndex + BUFFER_AHEAD_FRAMES;
      if (needsDecoding && !clipDecoder.isDecoding) {
        if (!frameInBuffer) {
          // Need frame NOW - await the decode with flush
          await this.decodeAhead(clipDecoder, targetSampleIndex + BUFFER_AHEAD_FRAMES, true);
        } else {
          // Fire and forget for frames already in buffer
          this.decodeAhead(clipDecoder, targetSampleIndex + BUFFER_AHEAD_FRAMES, false);
        }
      }

      // Track clips that still need their frames
      if (!frameInBuffer) {
        clipsNeedingFlush.push(clipDecoder);
      }
    }

    // Wait for clips that don't have their frame yet - keep decoding until we have it
    for (const clipDecoder of clipsNeedingFlush) {
      const clipInfo = clipDecoder.clipInfo;
      const sourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
      const targetTimestamp = sourceTime * 1_000_000;
      const targetSampleIndex = this.findSampleIndexForTime(clipDecoder, sourceTime);

      // Loop until frame is in buffer (max 10 attempts)
      for (let attempt = 0; attempt < 10; attempt++) {
        // Wait for pending decode
        if (clipDecoder.pendingDecode) {
          await clipDecoder.pendingDecode;
        }

        // Check if frame is now in buffer
        let frameFound = false;
        for (const [, decodedFrame] of clipDecoder.frameBuffer) {
          if (Math.abs(decodedFrame.timestamp - targetTimestamp) < 100_000) {
            frameFound = true;
            break;
          }
        }

        if (frameFound) break;

        // Still missing - decode more frames
        if (clipDecoder.decoder.decodeQueueSize > 0) {
          await clipDecoder.decoder.flush();
        }

        // Trigger another decode batch if needed
        if (clipDecoder.sampleIndex <= targetSampleIndex && !clipDecoder.isDecoding) {
          await this.decodeAhead(clipDecoder, targetSampleIndex + BUFFER_AHEAD_FRAMES, true);
        }
      }
    }
  }

  /**
   * Decode frames ahead to fill buffer - optimized for throughput
   * Does NOT flush after every batch - frames arrive via output callback asynchronously
   */
  private async decodeAhead(clipDecoder: ClipDecoder, targetSampleIndex: number, forceFlush: boolean = false): Promise<void> {
    if (clipDecoder.isDecoding) {
      return; // Let current decode continue, don't wait
    }

    clipDecoder.isDecoding = true;

    clipDecoder.pendingDecode = (async () => {
      try {
        const endIndex = Math.min(targetSampleIndex, clipDecoder.samples.length);
        let framesToDecode = endIndex - clipDecoder.sampleIndex;

        if (framesToDecode <= 0) {
          return;
        }

        // Decode in larger batches for throughput
        framesToDecode = Math.min(framesToDecode, DECODE_BATCH_SIZE);

        // Check if we need to seek (target is far ahead of current position)
        if (targetSampleIndex > clipDecoder.sampleIndex + 30) {
          // Need to seek - find nearest keyframe before target
          let keyframeIndex = targetSampleIndex;
          for (let i = targetSampleIndex; i >= 0; i--) {
            if (clipDecoder.samples[i].is_sync) {
              keyframeIndex = i;
              break;
            }
          }

          // Reset decoder and decode from keyframe
          await clipDecoder.decoder.flush(); // Flush before reset
          clipDecoder.decoder.reset();
          clipDecoder.decoder.configure(clipDecoder.codecConfig);
          clipDecoder.sampleIndex = keyframeIndex;

          // Clear buffer since we're seeking - also reset sorted list and bounds
          for (const [, decodedFrame] of clipDecoder.frameBuffer) {
            decodedFrame.frame.close();
          }
          clipDecoder.frameBuffer.clear();
          clipDecoder.sortedTimestamps = [];
          clipDecoder.oldestTimestamp = Infinity;
          clipDecoder.newestTimestamp = -Infinity;
        }

        // Queue frames for decode (non-blocking - output callback handles results)
        for (let i = 0; i < framesToDecode && clipDecoder.sampleIndex < clipDecoder.samples.length; i++) {
          const sample = clipDecoder.samples[clipDecoder.sampleIndex];
          clipDecoder.sampleIndex++;

          const chunk = new EncodedVideoChunk({
            type: sample.is_sync ? 'key' : 'delta',
            timestamp: (sample.cts * 1_000_000) / sample.timescale,
            duration: (sample.duration * 1_000_000) / sample.timescale,
            data: sample.data,
          });

          clipDecoder.decoder.decode(chunk);
        }

        // Only flush if explicitly requested (when we need frames NOW)
        if (forceFlush) {
          await clipDecoder.decoder.flush();
        }
      } catch (e) {
        console.error(`[ParallelDecode] Decode error for ${clipDecoder.clipName}:`, e);
      } finally {
        clipDecoder.isDecoding = false;
        clipDecoder.pendingDecode = null;
      }
    })();

    await clipDecoder.pendingDecode;
  }

  /**
   * Find sample index for a given source time - O(log n) binary search
   */
  private findSampleIndexForTime(clipDecoder: ClipDecoder, sourceTime: number): number {
    const targetTime = sourceTime * clipDecoder.videoTrack.timescale;
    const samples = clipDecoder.samples;

    // Add half-frame tolerance to handle floating point precision issues
    // e.g., at 30fps: 1/30 = 0.0333... but floating point may give slightly less
    const halfFrame = (samples[0]?.duration ?? 1) / 2;
    const targetTimeWithTolerance = targetTime + halfFrame;

    // Binary search for the sample with cts <= targetTimeWithTolerance
    let left = 0;
    let right = samples.length - 1;

    while (left < right) {
      const mid = Math.floor((left + right + 1) / 2);
      if (samples[mid].cts <= targetTimeWithTolerance) {
        left = mid;
      } else {
        right = mid - 1;
      }
    }

    return left;
  }

  /**
   * Get the decoded frame for a clip at a specific timeline time
   * Returns null if frame isn't ready (shouldn't happen if prefetch was called)
   * Optimized: O(log n) binary search instead of O(n) linear scan
   */
  getFrameForClip(clipId: string, timelineTime: number): VideoFrame | null {
    const clipDecoder = this.clipDecoders.get(clipId);
    if (!clipDecoder) return null;

    const clipInfo = clipDecoder.clipInfo;

    // Check if time is within clip range (handles nested clips too)
    if (!this.isTimeInClipRange(clipInfo, timelineTime)) {
      return null;
    }

    const targetSourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
    const targetTimestamp = targetSourceTime * 1_000_000;  // Convert to microseconds

    // Quick bounds check - O(1) rejection
    if (targetTimestamp < clipDecoder.oldestTimestamp - this.frameTolerance ||
        targetTimestamp > clipDecoder.newestTimestamp + this.frameTolerance) {
      return null;
    }

    // Binary search for closest timestamp - O(log n) instead of O(n)
    const timestamps = clipDecoder.sortedTimestamps;
    if (timestamps.length === 0) return null;

    let left = 0;
    let right = timestamps.length - 1;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (timestamps[mid] < targetTimestamp) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    // Check closest candidates (left and left-1)
    let closestIdx = left;
    if (left > 0) {
      const diffLeft = Math.abs(timestamps[left] - targetTimestamp);
      const diffPrev = Math.abs(timestamps[left - 1] - targetTimestamp);
      if (diffPrev < diffLeft) {
        closestIdx = left - 1;
      }
    }

    const closestTimestamp = timestamps[closestIdx];
    const closestDiff = Math.abs(closestTimestamp - targetTimestamp);

    if (closestDiff < this.frameTolerance) {
      const decodedFrame = clipDecoder.frameBuffer.get(closestTimestamp);
      if (decodedFrame) {
        // Debug: log if diff is significant (> half frame duration)
        const halfFrameDuration = 500_000 / this.exportFps;
        if (closestDiff > halfFrameDuration) {
          console.log(`[ParallelDecode] ${clipDecoder.clipName}: frame diff ${(closestDiff/1000).toFixed(1)}ms`);
        }
        return decodedFrame.frame;
      }
    }

    console.warn(`[ParallelDecode] ${clipDecoder.clipName}: No frame within tolerance at ${(targetTimestamp/1_000_000).toFixed(3)}s`);
    return null;
  }

  /**
   * Get all frames for the current timeline time
   * Returns Map of clipId -> VideoFrame
   */
  async getFramesAtTime(timelineTime: number): Promise<Map<string, VideoFrame>> {
    // First prefetch to ensure frames are decoded
    await this.prefetchFramesForTime(timelineTime);

    const frames = new Map<string, VideoFrame>();

    for (const [clipId] of this.clipDecoders) {
      const frame = this.getFrameForClip(clipId, timelineTime);
      if (frame) {
        frames.set(clipId, frame);
      }
    }

    return frames;
  }

  /**
   * Advance buffer position after rendering a frame
   * Call this after successfully rendering to clean up old frames
   */
  advanceToTime(timelineTime: number): void {
    for (const [, clipDecoder] of this.clipDecoders) {
      const clipInfo = clipDecoder.clipInfo;

      // Skip if time is not in this clip's range
      if (!this.isTimeInClipRange(clipInfo, timelineTime)) {
        continue;
      }

      const sourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
      const currentTimestamp = sourceTime * 1_000_000;  // Convert to microseconds

      // Clean up frames that are significantly behind current position (> 200ms behind)
      const timestampsToRemove: number[] = [];
      for (const [timestamp, decodedFrame] of clipDecoder.frameBuffer) {
        if (timestamp < currentTimestamp - 200_000) {  // 200ms behind
          decodedFrame.frame.close();
          timestampsToRemove.push(timestamp);
        }
      }

      for (const timestamp of timestampsToRemove) {
        clipDecoder.frameBuffer.delete(timestamp);
      }
    }
  }

  /**
   * Check if a clip is managed by this decoder
   */
  hasClip(clipId: string): boolean {
    return this.clipDecoders.has(clipId);
  }

  /**
   * Get codec string from video track
   */
  private getCodecString(track: MP4VideoTrack): string {
    const codec = track.codec;

    // H.264/AVC
    if (codec.startsWith('avc1') || codec.startsWith('avc3')) {
      return codec;
    }

    // H.265/HEVC
    if (codec.startsWith('hvc1') || codec.startsWith('hev1')) {
      return codec;
    }

    // VP9
    if (codec.startsWith('vp09')) {
      return codec;
    }

    // AV1
    if (codec.startsWith('av01')) {
      return codec;
    }

    return codec;
  }

  /**
   * Cleanup all resources
   */
  cleanup(): void {
    this.isActive = false;

    for (const [, clipDecoder] of this.clipDecoders) {
      // Close all buffered frames
      for (const [, decodedFrame] of clipDecoder.frameBuffer) {
        decodedFrame.frame.close();
      }
      clipDecoder.frameBuffer.clear();

      // Close decoder
      try {
        clipDecoder.decoder.close();
      } catch (e) {
        // Ignore close errors
      }
    }

    this.clipDecoders.clear();
    this.decodePromises.clear();
    console.log('[ParallelDecode] Cleaned up');
  }
}
