// WebCodecsExportMode - Sequential export frame decoding extracted from WebCodecsPlayer
// Manages pre-decoded frame buffer for frame-accurate video export

import { Logger } from '../services/logger';
import type { Sample } from './webCodecsTypes';

const log = Logger.create('WebCodecsExportMode');

/**
 * Interface for the player internals that export mode needs access to.
 * WebCodecsPlayer implements this to expose its internal state.
 */
export interface ExportModePlayer {
  getDecoder(): VideoDecoder | null;
  getSamples(): Sample[];
  getSampleIndex(): number;
  setSampleIndex(index: number): void;
  getVideoTrackTimescale(): number | null;
  getCodecConfig(): VideoDecoderConfig | null;
  getFrameRate(): number;
  getCurrentFrame(): VideoFrame | null;
  setCurrentFrame(frame: VideoFrame | null): void;
  isSimpleMode(): boolean;
  seekAsync(time: number): Promise<void>;
}

export class WebCodecsExportMode {
  private player: ExportModePlayer;

  // Export mode state
  private isActive = false;
  private exportFrameBuffer: Map<number, VideoFrame> = new Map(); // CTS (μs) -> VideoFrame
  private exportFramesCts: number[] = []; // Sorted CTS values for index-based lookup
  private exportCurrentIndex = 0;

  constructor(player: ExportModePlayer) {
    this.player = player;
  }

  /**
   * Handle decoder output during export mode - buffers all frames by CTS
   */
  handleDecoderOutput(frame: VideoFrame): void {
    const cts = frame.timestamp;
    this.exportFrameBuffer.set(cts, frame);
  }

  /**
   * Check if currently in export mode
   */
  get isInExportMode(): boolean {
    return this.isActive;
  }

  /**
   * Prepare for sequential export - pre-decodes frames for the export range.
   * Uses flush() to ensure all frames are output before continuing.
   */
  async prepareForSequentialExport(startTimeSeconds: number): Promise<void> {
    const endPrepare = log.time('prepareForSequentialExport');

    // Simple mode: browser handles decoding
    if (this.player.isSimpleMode()) {
      this.isActive = true;
      endPrepare();
      return;
    }

    const samples = this.player.getSamples();

    // Wait for samples to load (lazy loading means they might not be ready yet)
    if (samples.length === 0) {
      const endWaitSamples = log.time('waitForSamples');
      log.info('Waiting for samples to load...');
      const maxWaitMs = 10000;
      const startWait = performance.now();
      while (this.player.getSamples().length === 0 && performance.now() - startWait < maxWaitMs) {
        await new Promise(r => setTimeout(r, 50));
      }
      endWaitSamples();
      const loadedSamples = this.player.getSamples();
      if (loadedSamples.length === 0) {
        log.error('Timeout waiting for samples');
        endPrepare();
        return;
      }
      log.info(`Samples ready: ${loadedSamples.length} (waited ${(performance.now() - startWait).toFixed(0)}ms)`);
    } else {
      log.info(`Samples already loaded: ${samples.length}`);
    }

    const decoder = this.player.getDecoder();
    const timescale = this.player.getVideoTrackTimescale();
    if (timescale === null || !decoder) {
      endPrepare();
      return;
    }

    // Clear any existing export state
    this.cleanupExportBuffer();
    this.exportFramesCts = [];
    this.exportCurrentIndex = 0;

    // Close currentFrame from normal mode
    const currentFrame = this.player.getCurrentFrame();
    if (currentFrame) {
      try { currentFrame.close(); } catch {}
      this.player.setCurrentFrame(null);
    }

    // Enter export mode BEFORE decoding
    this.isActive = true;

    // Find the sample closest to start time
    const allSamples = this.player.getSamples();
    const targetTimeInTimescale = startTimeSeconds * timescale;
    let startSampleIndex = 0;
    let closestDiff = Infinity;
    for (let i = 0; i < allSamples.length; i++) {
      const diff = Math.abs(allSamples[i].cts - targetTimeInTimescale);
      if (diff < closestDiff) {
        closestDiff = diff;
        startSampleIndex = i;
      }
    }

    // Find keyframe before start
    let keyframeIndex = 0;
    for (let i = 0; i <= startSampleIndex; i++) {
      if (allSamples[i].is_sync) {
        keyframeIndex = i;
      }
    }

    // Reset decoder with software acceleration for reliable export
    const codecConfig = this.player.getCodecConfig();
    decoder.reset();
    const exportConfig: VideoDecoderConfig = {
      ...codecConfig!,
      hardwareAcceleration: 'prefer-software',
    };
    decoder.configure(exportConfig);
    log.debug('Configured decoder with prefer-software for export');
    this.player.setSampleIndex(keyframeIndex);

    // Find NEXT keyframe after start position - this is the natural decode boundary
    let nextKeyframeIndex = allSamples.length;
    for (let i = startSampleIndex + 1; i < allSamples.length; i++) {
      if (allSamples[i].is_sync) {
        nextKeyframeIndex = i;
        break;
      }
    }

    // Decode from current keyframe to next keyframe (or end if no more keyframes)
    const BUFFER_BEYOND_KEYFRAME = 15;
    const decodeEnd = Math.min(nextKeyframeIndex + BUFFER_BEYOND_KEYFRAME, allSamples.length);

    log.info(`Preparing: keyframe=${keyframeIndex}, start=${startSampleIndex}, nextKeyframe=${nextKeyframeIndex}, decoding ${decodeEnd - keyframeIndex} samples (total: ${allSamples.length})`);

    // Decode from keyframe to start position + buffer
    const endDecode = log.time('decodeInitialSamples');
    let needsKeyframeRecovery = false;
    for (let i = keyframeIndex; i < decodeEnd; i++) {
      const sample = allSamples[i];

      // After a "key frame required" error, skip delta frames until next keyframe
      if (needsKeyframeRecovery) {
        if (!sample.is_sync) continue;
        // Found keyframe — reset decoder and resume
        decoder.reset();
        decoder.configure(exportConfig);
        needsKeyframeRecovery = false;
        log.debug(`Keyframe recovery at sample ${i}`);
      }

      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts * 1_000_000) / sample.timescale,
        duration: (sample.duration * 1_000_000) / sample.timescale,
        data: sample.data,
      });
      try {
        decoder.decode(chunk);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg.includes('key frame')) {
          needsKeyframeRecovery = true;
          log.debug(`Key frame required at sample ${i}, scanning for next keyframe`);
        } else {
          log.warn(`Decode error at sample ${i}: ${e}`);
        }
      }
    }
    this.player.setSampleIndex(decodeEnd);
    endDecode();

    const samplesDecoded = decodeEnd - keyframeIndex;
    log.info(`Queued ${samplesDecoded} samples, queue size: ${decoder.decodeQueueSize}`);

    // Wait for decoder to output frames
    const flushTimeout = Math.max(5000, samplesDecoded * 10);
    const endFlush = log.time('waitForDecoderFlush');
    await this.waitForDecoderFlush(flushTimeout);
    endFlush();

    // Build sorted CTS array for index-based access
    this.exportFramesCts = Array.from(this.exportFrameBuffer.keys()).sort((a, b) => a - b);

    // Set currentFrame to first frame
    if (this.exportFramesCts.length > 0) {
      this.player.setCurrentFrame(this.exportFrameBuffer.get(this.exportFramesCts[0]) || null);
    }

    log.info(`Ready: ${this.exportFrameBuffer.size} frames buffered, CTS range: ${this.exportFramesCts[0]?.toFixed(0)} - ${this.exportFramesCts[this.exportFramesCts.length - 1]?.toFixed(0)}`);
    endPrepare();
  }

  /**
   * Wait for decoder to flush with timeout fallback
   */
  private async waitForDecoderFlush(timeoutMs: number): Promise<void> {
    const decoder = this.player.getDecoder();
    if (!decoder) return;

    const startTime = performance.now();
    const startBufferSize = this.exportFrameBuffer.size;

    const flushPromise = decoder.flush().catch(e => {
      log.warn(`Flush error: ${e}`);
    });

    const timeoutPromise = new Promise<void>((resolve) => {
      setTimeout(resolve, timeoutMs);
    });

    await Promise.race([flushPromise, timeoutPromise]);

    // Check if decoder is still valid after async wait
    const decoderAfter = this.player.getDecoder();
    if (!decoderAfter) {
      log.warn('Decoder was closed during flush');
      return;
    }

    if (decoderAfter.decodeQueueSize > 0) {
      log.warn(`Flush timeout, waiting for queue (${decoderAfter.decodeQueueSize} remaining)...`);
      let waitCount = 0;
      while (this.player.getDecoder() && this.player.getDecoder()!.decodeQueueSize > 0 && waitCount < 100) {
        await new Promise(r => setTimeout(r, 20));
        waitCount++;
      }
    }

    const decoderFinal = this.player.getDecoder();
    if (!decoderFinal) {
      log.warn('Decoder was closed during queue drain');
      return;
    }

    const elapsed = performance.now() - startTime;
    const framesOutput = this.exportFrameBuffer.size - startBufferSize;
    log.debug(`Flush complete: ${framesOutput} frames output in ${elapsed.toFixed(0)}ms, buffer now ${this.exportFrameBuffer.size}`);
  }

  /**
   * Clean up export frame buffer
   */
  cleanupExportBuffer(): void {
    const currentFrame = this.player.getCurrentFrame();
    for (const frame of this.exportFrameBuffer.values()) {
      if (frame !== currentFrame) {
        try { frame.close(); } catch {}
      }
    }
    this.exportFrameBuffer.clear();
  }

  /**
   * Get frame for export at specified time.
   * Simple approach: find closest frame in buffer, decode more if needed.
   */
  async seekDuringExport(timeSeconds: number): Promise<void> {
    if (this.player.isSimpleMode()) {
      await this.player.seekAsync(timeSeconds);
      return;
    }

    if (!this.isActive) {
      log.warn(`seekDuringExport: not in export mode at ${timeSeconds.toFixed(3)}s`);
      return;
    }

    const decoder = this.player.getDecoder();
    const timescale = this.player.getVideoTrackTimescale();
    if (timescale === null || !decoder) {
      log.warn(`seekDuringExport: missing videoTrack/decoder at ${timeSeconds.toFixed(3)}s`);
      return;
    }

    const targetCts = timeSeconds * 1_000_000;
    const frameDuration = 1_000_000 / this.player.getFrameRate();

    // Find closest frame in sorted CTS array using binary search
    let bestIndex = this.findClosestFrameIndex(targetCts);

    if (bestIndex >= 0 && bestIndex < this.exportFramesCts.length) {
      const cts = this.exportFramesCts[bestIndex];
      const diff = Math.abs(cts - targetCts);

      if (diff < frameDuration * 1.5) {
        const foundFrame = this.exportFrameBuffer.get(cts);
        if (foundFrame) {
          this.player.setCurrentFrame(foundFrame);
          this.exportCurrentIndex = bestIndex;

          // Decode ahead if we're getting close to buffer end
          const framesRemaining = this.exportFramesCts.length - bestIndex;
          if (framesRemaining < 30 && this.player.getSampleIndex() < this.player.getSamples().length) {
            log.debug(`Decoding ahead: ${framesRemaining} frames remaining, sampleIndex=${this.player.getSampleIndex()}/${this.player.getSamples().length}`);
            await this.decodeMoreFrames(30);
          }

          // Clean up frames far behind current position (keep 10 behind)
          this.cleanupOldFrames(bestIndex - 10);

          return;
        } else {
          log.warn(`Frame CTS ${cts} in list but not in buffer at ${timeSeconds.toFixed(3)}s`);
        }
      }
    }

    // Need to decode more frames
    const maxCtsInBuffer = this.exportFramesCts.length > 0
      ? this.exportFramesCts[this.exportFramesCts.length - 1]
      : 0;
    const minCtsInBuffer = this.exportFramesCts.length > 0
      ? this.exportFramesCts[0]
      : 0;

    log.warn(`Frame not in buffer: target=${targetCts.toFixed(0)}, range=[${minCtsInBuffer.toFixed(0)}-${maxCtsInBuffer.toFixed(0)}], bufferSize=${this.exportFramesCts.length}`);

    // Decode more frames — either target is ahead of buffer, or within range but missing
    if (this.player.getSamples().length > 0) {
      if (targetCts > maxCtsInBuffer) {
        log.info(`Decoding more: target ahead of buffer by ${((targetCts - maxCtsInBuffer)/1000).toFixed(1)}ms`);
      } else {
        // Target is within buffer range but frame is missing (likely failed to decode)
        // Find the right sample position and re-decode that section
        log.info(`Decoding more: target within buffer range but missing, re-decoding section`);
        const allSamples = this.player.getSamples();
        const timescale = this.player.getVideoTrackTimescale()!;
        const targetTimeInTimescale = timeSeconds * timescale;

        // Find sample closest to target
        let targetSampleIdx = 0;
        let closestDiff = Infinity;
        for (let i = 0; i < allSamples.length; i++) {
          const diff = Math.abs(allSamples[i].cts - targetTimeInTimescale);
          if (diff < closestDiff) {
            closestDiff = diff;
            targetSampleIdx = i;
          }
        }

        // Back up to keyframe before target
        let kfIdx = 0;
        for (let i = targetSampleIdx; i >= 0; i--) {
          if (allSamples[i].is_sync) {
            kfIdx = i;
            break;
          }
        }

        // Set sample index to this keyframe so decodeMoreFrames starts from there
        this.player.setSampleIndex(kfIdx);
      }

      await this.decodeMoreFrames(90);

      // Try again with wider tolerance
      bestIndex = this.findClosestFrameIndex(targetCts);
      if (bestIndex >= 0 && bestIndex < this.exportFramesCts.length) {
        const cts = this.exportFramesCts[bestIndex];
        const diff = Math.abs(cts - targetCts);
        if (diff < frameDuration * 3) {
          this.player.setCurrentFrame(this.exportFrameBuffer.get(cts) || null);
          this.exportCurrentIndex = bestIndex;
          return;
        }
      }
    }

    // Fallback: use closest available frame (better than last frame)
    if (this.exportFramesCts.length > 0) {
      bestIndex = this.findClosestFrameIndex(targetCts);
      const fallbackCts = this.exportFramesCts[Math.max(0, bestIndex)];
      this.player.setCurrentFrame(this.exportFrameBuffer.get(fallbackCts) || null);
      log.warn(`Using fallback frame at CTS ${fallbackCts.toFixed(0)} for target ${targetCts.toFixed(0)}`);
    } else {
      log.error(`No frames in buffer for seek to ${timeSeconds.toFixed(3)}s`);
    }
  }

  /**
   * Binary search to find closest frame index for a target CTS
   */
  private findClosestFrameIndex(targetCts: number): number {
    const arr = this.exportFramesCts;
    if (arr.length === 0) return -1;

    let left = 0;
    let right = arr.length - 1;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (arr[mid] < targetCts) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    if (left > 0 && Math.abs(arr[left - 1] - targetCts) < Math.abs(arr[left] - targetCts)) {
      return left - 1;
    }
    return left;
  }

  /**
   * Decode more frames and add to buffer.
   * Decodes until next keyframe to ensure B-frames can be resolved.
   */
  private async decodeMoreFrames(minCount: number): Promise<void> {
    const decoder = this.player.getDecoder();
    const samples = this.player.getSamples();
    const sampleIndex = this.player.getSampleIndex();

    if (!decoder || this.player.getVideoTrackTimescale() === null || sampleIndex >= samples.length) {
      log.debug(`decodeMoreFrames: nothing to decode (sampleIndex=${sampleIndex}/${samples.length})`);
      return;
    }

    // After flush, decoder needs a keyframe. Find the previous keyframe if current isn't one.
    let startIndex = sampleIndex;
    if (!samples[startIndex].is_sync) {
      for (let i = startIndex - 1; i >= 0; i--) {
        if (samples[i].is_sync) {
          startIndex = i;
          log.debug(`decodeMoreFrames: backed up to keyframe at sample ${i}`);
          break;
        }
      }
      if (!samples[startIndex].is_sync) {
        startIndex = 0;
        log.debug(`decodeMoreFrames: no keyframe found, starting from sample 0`);
      }
    }

    const bufferBefore = this.exportFrameBuffer.size;

    // Find next keyframe after minCount samples
    let endIndex = Math.min(startIndex + minCount, samples.length);
    for (let i = endIndex; i < samples.length; i++) {
      if (samples[i].is_sync) {
        endIndex = i + 15;
        break;
      }
      endIndex = i + 1;
    }
    endIndex = Math.min(endIndex, samples.length);

    log.info(`decodeMoreFrames: decoding samples ${startIndex}-${endIndex} (${endIndex - startIndex} samples)`);

    // Reset and reconfigure decoder to ensure clean state after previous flush
    const codecConfig = this.player.getCodecConfig();
    if (decoder && codecConfig) {
      decoder.reset();
      const exportConfig: VideoDecoderConfig = {
        ...codecConfig,
        hardwareAcceleration: 'prefer-software',
      };
      decoder.configure(exportConfig);
      log.debug('decodeMoreFrames: decoder reset and reconfigured with prefer-software');
    }

    let needsKeyframeRecovery = false;
    for (let i = startIndex; i < endIndex; i++) {
      const currentDecoder = this.player.getDecoder();
      if (!currentDecoder) {
        log.warn(`decodeMoreFrames: decoder closed at sample ${i}`);
        break;
      }
      const sample = samples[i];

      // After a "key frame required" error, skip delta frames until next keyframe
      if (needsKeyframeRecovery) {
        if (!sample.is_sync) continue;
        // Found keyframe — reset decoder and resume
        currentDecoder.reset();
        const exportConfig: VideoDecoderConfig = {
          ...this.player.getCodecConfig()!,
          hardwareAcceleration: 'prefer-software',
        };
        currentDecoder.configure(exportConfig);
        needsKeyframeRecovery = false;
        log.debug(`decodeMoreFrames: keyframe recovery at sample ${i}`);
      }

      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts * 1_000_000) / sample.timescale,
        duration: (sample.duration * 1_000_000) / sample.timescale,
        data: sample.data,
      });
      try {
        currentDecoder.decode(chunk);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg.includes('key frame')) {
          needsKeyframeRecovery = true;
          log.debug(`decodeMoreFrames: key frame required at sample ${i}, scanning for next keyframe`);
        } else {
          log.warn(`Decode error at sample ${i}: ${e}`);
        }
      }
    }
    this.player.setSampleIndex(endIndex);

    // Wait for frames to be output (with timeout)
    const sampleCount = endIndex - startIndex;
    await this.waitForDecoderFlush(Math.max(2000, sampleCount * 10));

    // Update sorted CTS array
    this.exportFramesCts = Array.from(this.exportFrameBuffer.keys()).sort((a, b) => a - b);

    const framesAdded = this.exportFrameBuffer.size - bufferBefore;
    log.info(`decodeMoreFrames: ${framesAdded} new frames added, buffer now ${this.exportFrameBuffer.size}`);
  }

  /**
   * Clean up frames before a certain index to free memory
   */
  private cleanupOldFrames(keepFromIndex: number): void {
    if (keepFromIndex <= 0) return;

    const currentFrame = this.player.getCurrentFrame();
    const toRemove = this.exportFramesCts.slice(0, keepFromIndex);
    for (const cts of toRemove) {
      const frame = this.exportFrameBuffer.get(cts);
      if (frame && frame !== currentFrame) {
        try { frame.close(); } catch {}
      }
      this.exportFrameBuffer.delete(cts);
    }
    this.exportFramesCts = this.exportFramesCts.slice(keepFromIndex);
    this.exportCurrentIndex = Math.max(0, this.exportCurrentIndex - keepFromIndex);
  }

  /**
   * End sequential export mode and clean up
   */
  endSequentialExport(): void {
    this.isActive = false;
    this.cleanupExportBuffer();
    this.exportFramesCts = [];
    this.exportCurrentIndex = 0;
    log.info('Export mode ended');
  }

  /**
   * Destroy and clean up all export buffers
   */
  destroy(): void {
    for (const frame of this.exportFrameBuffer.values()) {
      try { frame.close(); } catch {}
    }
    this.exportFrameBuffer.clear();
    this.exportFramesCts = [];
    this.isActive = false;
  }
}
