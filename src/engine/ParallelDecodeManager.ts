/**
 * ParallelDecodeManager - Parallel video decoding for multi-clip exports
 *
 * Problem: Sequential decoding of multiple videos is slow because each video
 * waits for the previous one to decode before proceeding.
 *
 * Solution: Pre-decode frames in parallel using separate VideoDecoder instances
 * per clip, with a frame buffer that stays ahead of the render position.
 */

import { Logger } from '../services/logger';
const log = Logger.create('ParallelDecode');

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
  needsKeyframe: boolean;                  // True after flush - must start from keyframe
}

// Buffer settings - tuned for speed like After Effects
const BUFFER_AHEAD_FRAMES = 60;   // Pre-decode this many frames ahead (1 second at 60fps)
const MAX_BUFFER_SIZE = 300;      // Maximum frames to keep in buffer (5s at 60fps - must cover keyframe intervals)
const DECODE_BATCH_SIZE = 90;     // Decode this many frames per batch - large for initial catchup
const SEEK_BATCH_MULTIPLIER = 5;  // Multiplier for batch size after seeks (5x = 450 frames)

export class ParallelDecodeManager {
  private clipDecoders: Map<string, ClipDecoder> = new Map();
  private isActive = false;
  private decodePromises: Map<string, Promise<void>> = new Map();
  private frameTolerance = 50_000;  // Default 50ms in microseconds

  /**
   * Initialize the manager with clips to decode
   */
  async initialize(clips: ClipInfo[], exportFps: number): Promise<void> {
    const endInit = log.time('initialize');
    this.isActive = true;
    // FPS-based tolerance: 1.5 frame duration
    this.frameTolerance = Math.round((1_000_000 / exportFps) * 1.5);

    log.info(`Initializing ${clips.length} clips:`, clips.map(c => c.clipName));

    // Parse all clips in parallel
    const initPromises = clips.map(clip => this.initializeClip(clip));
    await Promise.all(initPromises);

    log.info(`All ${clips.length} clips initialized`);
    endInit();
  }

  /**
   * Initialize a single clip decoder - LAZY MODE
   * Phase 1: Parse MP4 synchronously (onReady must be sync to not break MP4Box pipeline)
   * Phase 2: Async hwAccel probing
   * Phase 3: Configure decoder and start sample extraction
   */
  private async initializeClip(clipInfo: ClipInfo): Promise<void> {
    // Phase 1: Parse MP4 and extract track info (sync onReady)
    const parseResult = await this.parseMP4TrackInfo(clipInfo);

    // Phase 2: Async hardware acceleration probing
    const hwAccel = await this.findSupportedHwAccel(parseResult.baseConfig, clipInfo.clipName);

    const codecConfig: VideoDecoderConfig = {
      ...parseResult.baseConfig,
      hardwareAcceleration: hwAccel,
    };

    // Phase 3: Configure decoder
    const decoder = new VideoDecoder({
      output: (frame) => {
        if (!this.isActive) {
          frame.close();
          return;
        }
        const clipDecoder = this.clipDecoders.get(clipInfo.clipId);
        if (clipDecoder) {
          this.handleDecodedFrame(clipDecoder, frame);
        } else {
          log.warn(`Frame output for unknown clip ${clipInfo.clipId}`);
          frame.close();
        }
      },
      error: (e) => {
        if (!this.isActive) return;
        log.error(`Decoder error for ${clipInfo.clipName}: ${e.message || e}`);
      },
    });

    try {
      decoder.configure(codecConfig);
      log.info(`Decoder configured for "${clipInfo.clipName}": ${codecConfig.codec} ${parseResult.videoTrack.video.width}x${parseResult.videoTrack.video.height} (hwAccel=${hwAccel})`);
    } catch (e) {
      log.error(`Failed to configure decoder for "${clipInfo.clipName}":`, e);
      throw e;
    }

    const clipDecoder: ClipDecoder = {
      clipId: clipInfo.clipId,
      clipName: clipInfo.clipName,
      decoder,
      samples: parseResult.samples,
      sampleIndex: 0,
      videoTrack: parseResult.videoTrack,
      codecConfig,
      frameBuffer: new Map(),
      sortedTimestamps: [],
      oldestTimestamp: Infinity,
      newestTimestamp: -Infinity,
      lastDecodedTimestamp: 0,
      clipInfo,
      isDecoding: false,
      pendingDecode: null,
      needsKeyframe: true, // Decoder was just configure()'d — first chunk must be a keyframe
    };

    this.clipDecoders.set(clipInfo.clipId, clipDecoder);
    log.info(`Clip "${clipInfo.clipName}" initialized: ${parseResult.videoTrack.video.width}x${parseResult.videoTrack.video.height} (${parseResult.samples.length} samples ready)`);
  }

  /**
   * Parse MP4 file and extract track info + samples synchronously.
   * onReady MUST be sync — MP4Box calls it during appendBuffer, and an async
   * callback would yield control before setExtractionOptions/start/flush,
   * causing MP4Box to never deliver samples.
   */
  private parseMP4TrackInfo(clipInfo: ClipInfo): Promise<{
    videoTrack: MP4VideoTrack;
    baseConfig: VideoDecoderConfig;
    samples: Sample[];
  }> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`MP4 parsing timeout for clip "${clipInfo.clipName}"`));
      }, 5000);

      const mp4File = MP4Box.createFile() as MP4File;
      const collectedSamples: Sample[] = [];

      // SYNC onReady — no await allowed here!
      mp4File.onReady = (info) => {
        const videoTrack = info.videoTracks[0];
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
          log.warn(`Failed to extract codec description for ${clipInfo.clipName}: ${e}`);
        }

        const baseConfig: VideoDecoderConfig = {
          codec,
          codedWidth: videoTrack.video.width,
          codedHeight: videoTrack.video.height,
          optimizeForLatency: true,
          description,
        };

        // Start sample extraction SYNCHRONOUSLY before appendBuffer returns
        mp4File.setExtractionOptions(videoTrack.id, null, { nbSamples: Infinity });
        mp4File.start();

        // Resolve will happen after appendBuffer+flush complete (samples collected via onSamples)
        // Use setTimeout(0) to resolve after the synchronous appendBuffer+flush chain completes
        setTimeout(() => {
          clearTimeout(timeout);
          log.info(`"${clipInfo.clipName}" parsed: ${codec} ${videoTrack.video.width}x${videoTrack.video.height}, ${collectedSamples.length} samples`);
          resolve({ videoTrack, baseConfig, samples: collectedSamples });
        }, 0);
      };

      mp4File.onSamples = (_trackId, _ref, newSamples) => {
        collectedSamples.push(...newSamples);
      };

      mp4File.onError = (e) => {
        clearTimeout(timeout);
        reject(new Error(`MP4 parsing error for "${clipInfo.clipName}": ${e}`));
      };

      // Feed entire buffer — onReady fires sync during appendBuffer,
      // which sets up extraction before flush() signals end-of-data
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
    // If cleanup has started, immediately close the frame
    if (!this.isActive) {
      frame.close();
      return;
    }

    const timestamp = frame.timestamp;  // microseconds
    const sourceTime = timestamp / 1_000_000;  // convert to seconds

    // Log first 5 frames for debugging
    if (clipDecoder.frameBuffer.size < 5) {
      log.debug(`"${clipDecoder.clipName}": Frame ${clipDecoder.frameBuffer.size + 1} decoded at ${sourceTime.toFixed(3)}s (timestamp=${timestamp}µs)`);
    }

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
    log.debug(`prefetchFramesForTime(${timelineTime.toFixed(3)}) - isActive=${this.isActive}, decoders=${this.clipDecoders.size}`);
    if (!this.isActive) return;

    const clipsNeedingFlush: ClipDecoder[] = [];

    for (const [, clipDecoder] of this.clipDecoders) {
      const clipInfo = clipDecoder.clipInfo;

      // Skip if timeline time is outside this clip's range (handles nested clips too)
      if (!this.isTimeInClipRange(clipInfo, timelineTime)) {
        log.debug(`"${clipInfo.clipName}": Skipped - not in range (start=${clipInfo.startTime}, dur=${clipInfo.duration}, nested=${clipInfo.isNested})`);
        continue;
      }

      log.debug(`"${clipInfo.clipName}": Processing at time ${timelineTime.toFixed(3)}s - samples=${clipDecoder.samples.length}, buffer=${clipDecoder.frameBuffer.size}, decoderState=${clipDecoder.decoder.state}`);

      // Wait for samples if lazy loading hasn't delivered them yet
      if (clipDecoder.samples.length === 0) {
        log.debug(`"${clipInfo.clipName}": Waiting for samples...`);
        const maxWaitMs = 10000; // 10 second max wait per clip (increased for large files)
        const startWait = performance.now();
        while (clipDecoder.samples.length === 0 && performance.now() - startWait < maxWaitMs) {
          await new Promise(r => setTimeout(r, 50));
        }
        if (clipDecoder.samples.length === 0) {
          const errorMsg = `"${clipInfo.clipName}" has no samples after waiting ${maxWaitMs}ms`;
          log.error(errorMsg);
          throw new Error(`Parallel decode initialization failed: ${errorMsg}`);
        }
        log.info(`"${clipInfo.clipName}" samples ready: ${clipDecoder.samples.length} (waited ${(performance.now() - startWait).toFixed(0)}ms)`);
      }

      // Calculate target source time and sample index
      const sourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
      const targetSampleIndex = this.findSampleIndexForTime(clipDecoder, sourceTime);

      // Check if frame is already in buffer (fast path)
      const targetTimestamp = sourceTime * 1_000_000;
      let frameInBuffer = false;
      const checkTolerance = this.frameTolerance * 2; // Double tolerance for buffer check

      // Get buffer time range for logging
      const bufferTimes = Array.from(clipDecoder.frameBuffer.values())
        .map(f => f.timestamp)
        .sort((a, b) => a - b);
      const bufferStart = bufferTimes.length > 0 ? (bufferTimes[0] / 1_000_000).toFixed(3) : 'N/A';
      const bufferEnd = bufferTimes.length > 0 ? (bufferTimes[bufferTimes.length - 1] / 1_000_000).toFixed(3) : 'N/A';

      for (const [, decodedFrame] of clipDecoder.frameBuffer) {
        if (Math.abs(decodedFrame.timestamp - targetTimestamp) < checkTolerance) {
          frameInBuffer = true;
          break;
        }
      }

      log.debug(`"${clipInfo.clipName}": Frame check - target=${(targetTimestamp/1_000_000).toFixed(3)}s, buffer=${clipDecoder.frameBuffer.size} frames [${bufferStart}s-${bufferEnd}s], frameInBuffer=${frameInBuffer}, tolerance=${(checkTolerance/1000).toFixed(1)}ms`);

      // Trigger decode ahead - ALWAYS await if we're behind the target sample
      // Also need to decode if frame is not in buffer (we might be too far ahead and need to seek back)
      const needsDecodingAhead = clipDecoder.sampleIndex < targetSampleIndex + BUFFER_AHEAD_FRAMES;
      const needsDecodingBack = !frameInBuffer && clipDecoder.sampleIndex > targetSampleIndex + 30; // Too far ahead
      const needsDecoding = needsDecodingAhead || needsDecodingBack;
      const isBehindTarget = clipDecoder.sampleIndex <= targetSampleIndex; // Are we behind the current target?

      if (needsDecoding && !clipDecoder.isDecoding) {
        const decodeTarget = targetSampleIndex + BUFFER_AHEAD_FRAMES;
        log.debug(`"${clipInfo.clipName}": Triggering decode - samples=${clipDecoder.samples.length}, targetIdx=${targetSampleIndex}, currentIdx=${clipDecoder.sampleIndex}, decodeTarget=${decodeTarget}, frameInBuffer=${frameInBuffer}, isBehindTarget=${isBehindTarget}, needsBackSeek=${needsDecodingBack}`);

        // Decode WITHOUT flush first — let output callback deliver frames async.
        // This avoids the expensive flush→needsKeyframe→reset→re-decode-from-keyframe cycle.
        // The retry loop below will flush only if the frame hasn't appeared.
        if (!frameInBuffer) {
          log.debug(`"${clipInfo.clipName}": Decode without flush (frame not in buffer)`);
          await this.decodeAhead(clipDecoder, decodeTarget, false, 0, targetSampleIndex);
        } else {
          // Frame already in buffer - background decode for future frames
          log.debug(`"${clipInfo.clipName}": Background decode (frame in buffer)`);
          this.decodeAhead(clipDecoder, decodeTarget, false);
        }
      }

      // Track clips that still need their frames
      if (!frameInBuffer) {
        clipsNeedingFlush.push(clipDecoder);
      }
    }

    // Wait for clips that still need their frame — escalating strategy:
    // 1. Wait for async output callback (no flush)
    // 2. Flush only if frame hasn't appeared
    // 3. Re-decode with flush as last resort
    for (const clipDecoder of clipsNeedingFlush) {
      const clipInfo = clipDecoder.clipInfo;
      const sourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
      const targetTimestamp = sourceTime * 1_000_000;
      const targetSampleIndex = this.findSampleIndexForTime(clipDecoder, sourceTime);

      for (let attempt = 0; attempt < 10; attempt++) {
        // Wait for pending decode to complete
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

        if (frameFound) {
          if (attempt > 0) {
            log.debug(`"${clipDecoder.clipName}": Frame found after ${attempt + 1} attempts`);
          }
          break;
        }

        // Escalating strategy
        if (attempt < 2) {
          // First 2 attempts: just wait briefly for async output callback
          await new Promise(r => setTimeout(r, 8));
        } else if (clipDecoder.decoder.decodeQueueSize > 0) {
          // Flush decoder queue to force output — with timeout to prevent infinite hang
          log.debug(`"${clipDecoder.clipName}": Flushing decoder (attempt ${attempt + 1}, queue=${clipDecoder.decoder.decodeQueueSize})`);
          try {
            await Promise.race([
              clipDecoder.decoder.flush(),
              new Promise<void>((_, reject) => setTimeout(() => reject(new Error('flush timeout')), 3000))
            ]);
          } catch {
            log.warn(`"${clipDecoder.clipName}": Decoder flush timed out after 3s, resetting decoder`);
            try {
              clipDecoder.decoder.reset();
              await clipDecoder.decoder.configure(clipDecoder.codecConfig);
            } catch {
              log.warn(`"${clipDecoder.clipName}": Reset failed, recreating decoder`);
              await this.recreateDecoder(clipDecoder);
            }
          }
          clipDecoder.needsKeyframe = true;
        } else if (!clipDecoder.isDecoding) {
          // Queue is empty but frame not found — re-decode with forceFlush
          log.debug(`"${clipDecoder.clipName}": Re-decode with flush (attempt ${attempt + 1}, queue empty)`);
          const decodeTarget = Math.max(targetSampleIndex + BUFFER_AHEAD_FRAMES, BUFFER_AHEAD_FRAMES);
          await this.decodeAhead(clipDecoder, decodeTarget, true, 0, targetSampleIndex);
        } else {
          // Decoder is busy, wait for it
          await new Promise(r => setTimeout(r, 10));
        }
      }

      // Final check - warn but don't crash if frame is missing.
      // getFrameForClip() will return the closest available frame,
      // so a slightly off frame is better than aborting the entire export.
      let finalCheck = false;
      const finalTolerance = this.frameTolerance * 3; // 3x tolerance for final check
      for (const [, decodedFrame] of clipDecoder.frameBuffer) {
        if (Math.abs(decodedFrame.timestamp - targetTimestamp) < finalTolerance) {
          finalCheck = true;
          break;
        }
      }
      if (!finalCheck) {
        const availableFrames = Array.from(clipDecoder.frameBuffer.values())
          .map(f => (f.timestamp / 1_000_000).toFixed(3))
          .sort()
          .slice(0, 10)
          .join(', ');

        log.warn(`"${clipDecoder.clipName}": Frame at ${(targetTimestamp/1_000_000).toFixed(3)}s not exact match after all attempts (buffer: ${clipDecoder.frameBuffer.size} frames, decoderState: ${clipDecoder.decoder.state}, nearby: [${availableFrames}...]). Using closest available frame.`);
        // Don't throw — getFrameForClip will return closest match
      }
    }
  }

  /**
   * Recreate a decoder that has entered the permanent 'closed' state due to an error.
   * WebCodecs decoders cannot be reset() once closed - a full recreate is needed.
   * Re-checks hardware acceleration support since the original mode may have been the cause.
   */
  private async recreateDecoder(clipDecoder: ClipDecoder): Promise<void> {
    log.warn(`${clipDecoder.clipName}: Recreating closed decoder`);

    // Re-check hardware acceleration — the original mode may have caused the failure
    const hwAccel = await this.findSupportedHwAccel(clipDecoder.codecConfig, clipDecoder.clipName);
    const newConfig: VideoDecoderConfig = { ...clipDecoder.codecConfig, hardwareAcceleration: hwAccel };

    // Create new decoder with same callbacks
    const newDecoder = new VideoDecoder({
      output: (frame) => {
        if (!this.isActive) {
          frame.close();
          return;
        }
        const cd = this.clipDecoders.get(clipDecoder.clipId);
        if (cd) {
          this.handleDecodedFrame(cd, frame);
        } else {
          frame.close();
        }
      },
      error: (e) => {
        if (!this.isActive) return;
        log.error(`Decoder error for ${clipDecoder.clipName}: ${e.message || e}`);
      },
    });

    // Configure with updated codec config
    try {
      newDecoder.configure(newConfig);
    } catch (e) {
      log.error(`${clipDecoder.clipName}: Failed to configure recreated decoder: ${e}`);
      throw e;
    }

    // Replace decoder and config
    clipDecoder.decoder = newDecoder;
    clipDecoder.codecConfig = newConfig;
    clipDecoder.needsKeyframe = true;
    clipDecoder.sampleIndex = 0;

    // Clear stale buffer
    for (const [, decodedFrame] of clipDecoder.frameBuffer) {
      try { decodedFrame.frame.close(); } catch (_) { /* already closed */ }
    }
    clipDecoder.frameBuffer.clear();
    clipDecoder.sortedTimestamps = [];
    clipDecoder.oldestTimestamp = Infinity;
    clipDecoder.newestTimestamp = -Infinity;

    log.info(`${clipDecoder.clipName}: Decoder recreated successfully (hwAccel=${hwAccel})`);
  }

  /**
   * Decode frames ahead to fill buffer - optimized for throughput
   * Does NOT flush after every batch - frames arrive via output callback asynchronously
   * @param seekTargetSampleIndex - If provided, use this for seek keyframe calculation instead of targetSampleIndex
   *                                This is important when targetSampleIndex includes buffer-ahead frames
   */
  private async decodeAhead(clipDecoder: ClipDecoder, targetSampleIndex: number, forceFlush: boolean = false, recursionDepth: number = 0, seekTargetSampleIndex?: number): Promise<void> {
    // Prevent infinite recursion
    if (recursionDepth > 3) {
      log.warn(`${clipDecoder.clipName}: Max recursion depth reached (${recursionDepth}), stopping`);
      return;
    }

    if (clipDecoder.isDecoding) {
      log.debug(`${clipDecoder.clipName}: Already decoding, skipping`);
      return; // Let current decode continue, don't wait
    }

    // Check if decoder is still valid - recreate if closed
    if (!clipDecoder.decoder || clipDecoder.decoder.state === 'closed') {
      log.warn(`${clipDecoder.clipName}: Decoder is ${clipDecoder.decoder?.state || 'null'}, recreating...`);
      await this.recreateDecoder(clipDecoder);
    }

    clipDecoder.isDecoding = true;

    clipDecoder.pendingDecode = (async () => {
      try {
        // Double-check decoder state inside async block - recreate if closed
        if (!clipDecoder.decoder || clipDecoder.decoder.state === 'closed') {
          log.warn(`${clipDecoder.clipName}: Decoder closed during decode setup, recreating...`);
          await this.recreateDecoder(clipDecoder);
        }
        // Check if we need to seek (target is far from current position - either ahead OR behind)
        // But ONLY seek if forceFlush is true (we actually need the frame now)
        // Background decodes should just continue forward, not seek
        const isTooFarAhead = targetSampleIndex > clipDecoder.sampleIndex + 30;
        // When seekTargetSampleIndex is provided, we need a specific frame NOW.
        // If we've already decoded past that sample, we must seek back since
        // decoders can only decode forward. Without seekTargetSampleIndex,
        // use the original tolerance-based check.
        const isTooFarBehind = seekTargetSampleIndex !== undefined
          ? clipDecoder.sampleIndex > seekTargetSampleIndex
          : clipDecoder.sampleIndex > targetSampleIndex + 30;
        const needsSeek = forceFlush && (isTooFarAhead || isTooFarBehind);

        // IMPORTANT: Do seek FIRST before calculating framesToDecode
        // Otherwise if we're past the target, framesToDecode will be negative and we'll return early
        if (needsSeek) {
          // Need to seek - find nearest keyframe before the ACTUAL target we need
          const seekTarget = seekTargetSampleIndex ?? targetSampleIndex;
          // Find keyframe candidates by CTS (display time), not decode order.
          // Due to B-frame reordering, a keyframe earlier in decode order
          // can have a LATER CTS than the target, causing wrong frames to be decoded.
          const targetCTS = clipDecoder.samples[seekTarget].cts;
          const keyframeCandidates: number[] = [];
          for (let i = 0; i < clipDecoder.samples.length; i++) {
            if (clipDecoder.samples[i].is_sync) {
              if (clipDecoder.samples[i].cts <= targetCTS) {
                keyframeCandidates.push(i);
              } else {
                break; // Keyframe CTS values increase monotonically
              }
            }
          }
          if (keyframeCandidates.length === 0) keyframeCandidates.push(0);

          const exportConfig = clipDecoder.codecConfig;

          // Try keyframes from closest to earliest - some samples marked is_sync
          // by MP4Box aren't real IDR keyframes (e.g. open-GOP recovery points).
          // The decoder rejects these, so we fall back to earlier keyframes.
          const maxAttempts = Math.min(keyframeCandidates.length, 5);
          for (let k = keyframeCandidates.length - 1; k >= keyframeCandidates.length - maxAttempts; k--) {
            const candidateIndex = keyframeCandidates[k];
            const candidateSample = clipDecoder.samples[candidateIndex];
            const candidateCTS = (candidateSample.cts / clipDecoder.videoTrack.timescale).toFixed(3);

            clipDecoder.decoder.reset();
            clipDecoder.decoder.configure(exportConfig);

            const chunk = new EncodedVideoChunk({
              type: 'key',
              timestamp: (candidateSample.cts * 1_000_000) / candidateSample.timescale,
              duration: (candidateSample.duration * 1_000_000) / candidateSample.timescale,
              data: candidateSample.data,
            });

            try {
              clipDecoder.decoder.decode(chunk);
              clipDecoder.sampleIndex = candidateIndex + 1; // Already decoded this one
              log.debug(`${clipDecoder.clipName}: Seek keyframe accepted at sample ${candidateIndex} (CTS=${candidateCTS}s, targetCTS=${(targetCTS / clipDecoder.videoTrack.timescale).toFixed(3)}s, bufferTarget=${targetSampleIndex})`);
              break;
            } catch (e) {
              log.debug(`${clipDecoder.clipName}: Seek keyframe REJECTED at sample ${candidateIndex} (CTS=${candidateCTS}s) - not a real IDR, trying earlier`);
              if (k === keyframeCandidates.length - maxAttempts) {
                // Last attempt failed - reset and start from first sample
                clipDecoder.decoder.reset();
                clipDecoder.decoder.configure(exportConfig);
                clipDecoder.sampleIndex = 0;
                log.warn(`${clipDecoder.clipName}: No valid keyframe found after ${maxAttempts} attempts, starting from sample 0`);
              }
            }
          }

          clipDecoder.needsKeyframe = false;

          // Clear buffer since we're seeking
          for (const [, decodedFrame] of clipDecoder.frameBuffer) {
            decodedFrame.frame.close();
          }
          clipDecoder.frameBuffer.clear();
          clipDecoder.sortedTimestamps = [];
          clipDecoder.oldestTimestamp = Infinity;
          clipDecoder.newestTimestamp = -Infinity;
        }

        // Calculate frames to decode AFTER potential seek (sampleIndex may have changed)
        const endIndex = Math.min(targetSampleIndex, clipDecoder.samples.length);
        let framesToDecode = endIndex - clipDecoder.sampleIndex;

        if (framesToDecode <= 0) {
          log.debug(`${clipDecoder.clipName}: No frames to decode (sampleIndex=${clipDecoder.sampleIndex}, target=${targetSampleIndex})`);
          return;
        }

        // Decode in larger batches for throughput
        // Use much larger batch for seeks to reach target in one go
        const batchSize = needsSeek ? DECODE_BATCH_SIZE * SEEK_BATCH_MULTIPLIER : DECODE_BATCH_SIZE;
        framesToDecode = Math.min(framesToDecode, batchSize);

        log.debug(`${clipDecoder.clipName}: Decoding ${framesToDecode} frames (from sample ${clipDecoder.sampleIndex} to ${clipDecoder.sampleIndex + framesToDecode}), forceFlush=${forceFlush}, needsSeek=${needsSeek} (ahead=${isTooFarAhead}, behind=${isTooFarBehind}), batchSize=${batchSize}`);

        // After flush/configure, decoder requires next chunk to be a keyframe.
        // Reset decoder and start from nearest keyframe (same approach as seek path).
        if (clipDecoder.needsKeyframe && !needsSeek) {
          let keyframeIndex = clipDecoder.sampleIndex;
          // Find nearest keyframe at or before current position
          if (!clipDecoder.samples[keyframeIndex]?.is_sync) {
            for (let i = keyframeIndex - 1; i >= 0; i--) {
              if (clipDecoder.samples[i].is_sync) {
                keyframeIndex = i;
                break;
              }
            }
          }
          // Reset decoder to clean state and start from keyframe
          clipDecoder.decoder.reset();
          clipDecoder.decoder.configure(clipDecoder.codecConfig);
          clipDecoder.sampleIndex = keyframeIndex;
          clipDecoder.needsKeyframe = false;
          log.debug(`${clipDecoder.clipName}: needsKeyframe - reset decoder, starting from keyframe at sample ${keyframeIndex}`);
        }

        // Queue frames for decode (non-blocking - output callback handles results)
        let decodedCount = 0;
        let needsKeyframeRecovery = false;
        for (let i = 0; i < framesToDecode && clipDecoder.sampleIndex < clipDecoder.samples.length; i++) {
          const sample = clipDecoder.samples[clipDecoder.sampleIndex];

          // Safety: if decoder rejected a delta frame, skip until next keyframe
          if (needsKeyframeRecovery && !sample.is_sync) {
            clipDecoder.sampleIndex++;
            continue;
          }
          if (needsKeyframeRecovery && sample.is_sync) {
            // Found a keyframe — reset decoder to clean state before feeding it
            clipDecoder.decoder.reset();
            clipDecoder.decoder.configure(clipDecoder.codecConfig);
            needsKeyframeRecovery = false;
            log.debug(`${clipDecoder.clipName}: keyframe recovery at sample ${clipDecoder.sampleIndex}`);
          }

          clipDecoder.sampleIndex++;

          const chunk = new EncodedVideoChunk({
            type: sample.is_sync ? 'key' : 'delta',
            timestamp: (sample.cts * 1_000_000) / sample.timescale,
            duration: (sample.duration * 1_000_000) / sample.timescale,
            data: sample.data,
          });

          try {
            clipDecoder.decoder.decode(chunk);
            decodedCount++;
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            if (msg.includes('key frame')) {
              // Decoder needs a keyframe — skip delta frames until we find one
              needsKeyframeRecovery = true;
              log.debug(`${clipDecoder.clipName}: key frame required at sample ${clipDecoder.sampleIndex - 1}, scanning for next keyframe`);
            } else {
              log.warn(`${clipDecoder.clipName}: decode error at sample ${clipDecoder.sampleIndex - 1}: ${e}`);
            }
          }
        }

        log.debug(`${clipDecoder.clipName}: Queued ${decodedCount} chunks to decoder, decodeQueueSize=${clipDecoder.decoder.decodeQueueSize}`);

        // Only flush if explicitly requested (when we need frames NOW)
        if (forceFlush) {
          await clipDecoder.decoder.flush();
          clipDecoder.needsKeyframe = true; // After flush, next decode needs keyframe
        }
      } catch (e) {
        log.error(`Decode error for ${clipDecoder.clipName}: ${e}`);
      } finally {
        clipDecoder.isDecoding = false;
        clipDecoder.pendingDecode = null;
      }
    })();

    await clipDecoder.pendingDecode;

    // If we're still behind target after the batch, decode more recursively
    // BUT: Don't recurse if we just did a seek (needsSeek), as the seek resets sampleIndex
    // and would cause infinite recursion. Instead, let the next prefetch call handle it.
    const stillBehind = clipDecoder.sampleIndex < targetSampleIndex;
    // Check if a seek happened (either direction) - recompute same logic as above
    const didSeekAhead = targetSampleIndex > clipDecoder.sampleIndex + 30;
    const didSeekBehind = clipDecoder.sampleIndex > targetSampleIndex + 30;
    const didSeek = didSeekAhead || didSeekBehind;

    if (forceFlush && stillBehind && !didSeek && recursionDepth < 3) {
      const remainingFrames = targetSampleIndex - clipDecoder.sampleIndex;
      log.debug(`${clipDecoder.clipName}: Still behind target (sampleIndex=${clipDecoder.sampleIndex}, targetIdx=${targetSampleIndex}, remaining=${remainingFrames}), decoding additional batch (recursion ${recursionDepth + 1}/3)`);
      await this.decodeAhead(clipDecoder, targetSampleIndex, true, recursionDepth + 1);
    } else if (stillBehind) {
      log.debug(`${clipDecoder.clipName}: Still behind target (sampleIndex=${clipDecoder.sampleIndex}, targetIdx=${targetSampleIndex}), stopping (${didSeek ? 'after seek' : 'max recursion'})`);
    }
  }

  /**
   * Find sample index for a given source time.
   * Handles B-frame reordering by searching for closest CTS match.
   * IMPORTANT: Samples are in DECODE order (DTS), not presentation order (CTS)
   * due to B-frame reordering. Binary search doesn't work here.
   */
  private findSampleIndexForTime(clipDecoder: ClipDecoder, sourceTime: number): number {
    const targetTime = sourceTime * clipDecoder.videoTrack.timescale;
    const samples = clipDecoder.samples;

    if (samples.length === 0) return 0;

    // Linear search for sample with CTS closest to target time
    let targetIndex = 0;
    let closestDiff = Infinity;

    for (let i = 0; i < samples.length; i++) {
      const diff = Math.abs(samples[i].cts - targetTime);
      if (diff < closestDiff) {
        closestDiff = diff;
        targetIndex = i;
      }
    }

    return targetIndex;
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

    // Quick bounds check - return first/last frame if target is outside buffer range
    // This handles videos where first frame isn't at exactly 0 or clip extends beyond video
    const bufferEmpty = clipDecoder.sortedTimestamps.length === 0;
    if (bufferEmpty) {
      log.warn(`${clipDecoder.clipName}: Buffer empty for target ${(targetTimestamp/1_000_000).toFixed(3)}s`);
      return null;
    }

    const useLastFrame = targetTimestamp > clipDecoder.newestTimestamp + this.frameTolerance;
    if (useLastFrame) {
      // Return last frame for targets beyond video end
      const lastTimestamp = clipDecoder.sortedTimestamps[clipDecoder.sortedTimestamps.length - 1];
      const lastFrame = clipDecoder.frameBuffer.get(lastTimestamp);
      if (lastFrame) {
        log.debug(`${clipDecoder.clipName}: using last frame for target ${(targetTimestamp/1_000_000).toFixed(3)}s (video ends at ${(lastTimestamp/1_000_000).toFixed(3)}s)`);
        return lastFrame.frame;
      }
      log.warn(`${clipDecoder.clipName}: No last frame available`);
      return null;
    }

    // If target is before first frame, use first frame (common for clips starting at 0)
    const useFirstFrame = targetTimestamp < clipDecoder.oldestTimestamp - this.frameTolerance;
    if (useFirstFrame) {
      const firstTimestamp = clipDecoder.sortedTimestamps[0];
      const firstFrame = clipDecoder.frameBuffer.get(firstTimestamp);
      if (firstFrame) {
        log.debug(`${clipDecoder.clipName}: using first frame for target ${(targetTimestamp/1_000_000).toFixed(3)}s (video starts at ${(firstTimestamp/1_000_000).toFixed(3)}s)`);
        return firstFrame.frame;
      }
      log.warn(`${clipDecoder.clipName}: No first frame available`);
      return null;
    }

    // Binary search for closest timestamp - O(log n) instead of O(n)
    const timestamps = clipDecoder.sortedTimestamps;

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

    const frameTimestamp = timestamps[closestIdx];
    const frameDiff = Math.abs(frameTimestamp - targetTimestamp);

    const decodedFrame = clipDecoder.frameBuffer.get(frameTimestamp);
    if (decodedFrame) {
      // Log warning if outside tolerance but still return the closest frame
      // Better to have a slightly off frame than fail the export
      if (frameDiff >= this.frameTolerance) {
        log.debug(`${clipDecoder.clipName}: Using nearest frame at ${(targetTimestamp/1_000_000).toFixed(3)}s - diff=${(frameDiff/1000).toFixed(1)}ms exceeds tolerance=${(this.frameTolerance/1000).toFixed(1)}ms`);
      }
      return decodedFrame.frame;
    }

    // No frame found at all
    log.warn(`${clipDecoder.clipName}: No frame available at ${(targetTimestamp/1_000_000).toFixed(3)}s - buffer=${clipDecoder.frameBuffer.size} frames`);
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
   * Find a supported hardwareAcceleration mode for the given config.
   * Tries prefer-software first (most reliable for export), then prefer-hardware, then no-preference.
   */
  private async findSupportedHwAccel(
    baseConfig: VideoDecoderConfig,
    clipName: string
  ): Promise<HardwareAcceleration> {
    const modes: HardwareAcceleration[] = ['prefer-software', 'prefer-hardware', 'no-preference'];

    for (const mode of modes) {
      try {
        const result = await VideoDecoder.isConfigSupported({ ...baseConfig, hardwareAcceleration: mode });
        if (result.supported) {
          if (mode !== 'prefer-software') {
            log.info(`"${clipName}": prefer-software not supported, using ${mode}`);
          }
          return mode;
        }
      } catch {
        // isConfigSupported threw — skip this mode
      }
    }

    // None explicitly supported — fall back to no-preference and let configure() decide
    log.warn(`"${clipName}": No hwAccel mode reported as supported for codec ${baseConfig.codec}, trying no-preference`);
    return 'no-preference';
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
    // Set inactive first - this ensures handleDecodedFrame closes any new frames
    this.isActive = false;

    for (const [, clipDecoder] of this.clipDecoders) {
      // Reset decoder first to stop any pending decode operations
      // This will cause output callback to fire for any buffered frames
      try {
        if (clipDecoder.decoder.state !== 'closed') {
          clipDecoder.decoder.reset();
        }
      } catch (e) {
        // Ignore reset errors
      }

      // Close all buffered frames
      for (const [, decodedFrame] of clipDecoder.frameBuffer) {
        try {
          decodedFrame.frame.close();
        } catch (e) {
          // Frame may already be closed
        }
      }
      clipDecoder.frameBuffer.clear();
      clipDecoder.sortedTimestamps = [];

      // Close decoder
      try {
        if (clipDecoder.decoder.state !== 'closed') {
          clipDecoder.decoder.close();
        }
      } catch (e) {
        // Ignore close errors
      }
    }

    this.clipDecoders.clear();
    this.decodePromises.clear();
    log.info('Cleaned up');
  }
}
