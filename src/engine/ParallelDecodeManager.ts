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
  lastDecodedTimestamp: number;            // Track last decoded timestamp
  clipInfo: ClipInfo;
  isDecoding: boolean;
  pendingDecode: Promise<void> | null;
}

// Buffer settings
const BUFFER_AHEAD_FRAMES = 15;  // Pre-decode this many frames ahead
const MAX_BUFFER_SIZE = 30;       // Maximum frames to keep in buffer
const DECODE_BATCH_SIZE = 5;      // Decode this many frames per batch

export class ParallelDecodeManager {
  private clipDecoders: Map<string, ClipDecoder> = new Map();
  private isActive = false;
  private decodePromises: Map<string, Promise<void>> = new Map();

  /**
   * Initialize the manager with clips to decode
   */
  async initialize(clips: ClipInfo[], exportFps: number): Promise<void> {
    this.isActive = true;

    console.log(`[ParallelDecode] Initializing ${clips.length} clips at ${exportFps}fps...`);

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
   */
  private handleDecodedFrame(clipDecoder: ClipDecoder, frame: VideoFrame): void {
    // Use the frame's timestamp directly (preserved from EncodedVideoChunk)
    const timestamp = frame.timestamp;  // microseconds
    const sourceTime = timestamp / 1_000_000;  // convert to seconds

    // Store frame by its timestamp for accurate retrieval
    clipDecoder.frameBuffer.set(timestamp, {
      frame,
      sourceTime,
      timestamp,
    });

    clipDecoder.lastDecodedTimestamp = timestamp;

    // Cleanup old frames if buffer is too large
    if (clipDecoder.frameBuffer.size > MAX_BUFFER_SIZE) {
      const timestamps = [...clipDecoder.frameBuffer.keys()].sort((a, b) => a - b);
      const oldestTimestamp = timestamps[0];
      const oldFrame = clipDecoder.frameBuffer.get(oldestTimestamp);
      if (oldFrame) {
        oldFrame.frame.close();
        clipDecoder.frameBuffer.delete(oldestTimestamp);
      }
    }
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
   * Call this before rendering each frame to ensure frames are ready
   */
  async prefetchFramesForTime(timelineTime: number): Promise<void> {
    if (!this.isActive) return;

    const promises: Promise<void>[] = [];

    for (const [, clipDecoder] of this.clipDecoders) {
      const clipInfo = clipDecoder.clipInfo;

      // Skip if timeline time is outside this clip's range (handles nested clips too)
      if (!this.isTimeInClipRange(clipInfo, timelineTime)) {
        continue;
      }

      // Calculate target source time and sample index
      const sourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
      const targetSampleIndex = this.findSampleIndexForTime(clipDecoder, sourceTime);

      // Check if we need to decode more frames
      // We need to decode if sampleIndex (next to decode) is not far enough ahead of target
      const needsDecoding = clipDecoder.sampleIndex < targetSampleIndex + BUFFER_AHEAD_FRAMES;

      if (needsDecoding && !clipDecoder.isDecoding) {
        promises.push(this.decodeAhead(clipDecoder, targetSampleIndex + BUFFER_AHEAD_FRAMES));
      }
    }

    if (promises.length > 0) {
      await Promise.all(promises);
    }
  }

  /**
   * Decode frames ahead to fill buffer
   */
  private async decodeAhead(clipDecoder: ClipDecoder, targetSampleIndex: number): Promise<void> {
    if (clipDecoder.isDecoding) {
      // Wait for current decode to finish
      if (clipDecoder.pendingDecode) {
        await clipDecoder.pendingDecode;
      }
      return;
    }

    clipDecoder.isDecoding = true;

    clipDecoder.pendingDecode = (async () => {
      try {
        const startIndex = clipDecoder.sampleIndex;
        const endIndex = Math.min(targetSampleIndex, clipDecoder.samples.length);
        const framesToDecode = Math.min(endIndex - startIndex, DECODE_BATCH_SIZE);

        if (framesToDecode <= 0) {
          return;
        }

        // Check if we need to seek (find nearest keyframe)
        if (startIndex > clipDecoder.sampleIndex + 10) {
          // Need to seek - find keyframe
          let keyframeIndex = startIndex;
          for (let i = startIndex; i >= 0; i--) {
            if (clipDecoder.samples[i].is_sync) {
              keyframeIndex = i;
              break;
            }
          }

          // Reset decoder and decode from keyframe
          clipDecoder.decoder.reset();
          clipDecoder.decoder.configure(clipDecoder.codecConfig);
          clipDecoder.sampleIndex = keyframeIndex;

          // Clear buffer since we're seeking
          for (const [, decodedFrame] of clipDecoder.frameBuffer) {
            decodedFrame.frame.close();
          }
          clipDecoder.frameBuffer.clear();
        }

        // Decode frames
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

        // Wait for decoder to flush
        await clipDecoder.decoder.flush();
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
   * Find sample index for a given source time
   */
  private findSampleIndexForTime(clipDecoder: ClipDecoder, sourceTime: number): number {
    const targetTime = sourceTime * clipDecoder.videoTrack.timescale;

    for (let i = 0; i < clipDecoder.samples.length; i++) {
      if (clipDecoder.samples[i].cts > targetTime) {
        return Math.max(0, i - 1);
      }
    }
    return clipDecoder.samples.length - 1;
  }

  /**
   * Get the decoded frame for a clip at a specific timeline time
   * Returns null if frame isn't ready (shouldn't happen if prefetch was called)
   */
  getFrameForClip(clipId: string, timelineTime: number): VideoFrame | null {
    const clipDecoder = this.clipDecoders.get(clipId);
    if (!clipDecoder) return null;

    const clipInfo = clipDecoder.clipInfo;

    // Check if time is within clip range (handles nested clips too)
    if (!this.isTimeInClipRange(clipInfo, timelineTime)) {
      return null;
    }

    // Find the closest frame in buffer by source time
    const targetSourceTime = this.timelineToSourceTime(clipInfo, timelineTime);
    const targetTimestamp = targetSourceTime * 1_000_000;  // Convert to microseconds

    // Find the frame with closest timestamp
    let closestFrame: DecodedFrame | null = null;
    let closestDiff = Infinity;

    for (const [, decodedFrame] of clipDecoder.frameBuffer) {
      const diff = Math.abs(decodedFrame.timestamp - targetTimestamp);
      if (diff < closestDiff) {
        closestDiff = diff;
        closestFrame = decodedFrame;
      }
    }

    // Return frame if within reasonable range (100ms = 100000μs)
    if (closestFrame && closestDiff < 100_000) {
      return closestFrame.frame;
    }

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
