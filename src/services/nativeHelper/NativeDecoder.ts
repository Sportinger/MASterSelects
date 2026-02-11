/**
 * Native Decoder
 *
 * Provides a decoder interface using the native helper for decoding.
 * Returns ImageBitmap objects for WebGPU textures.
 *
 * Key optimization: frame ring buffer with look-ahead decoding.
 * During playback, frames are pre-decoded ahead of the playhead
 * so they're instantly available when the render loop needs them.
 */

import { Logger } from '../logger';
import { NativeHelperClient } from './NativeHelperClient';
import type { FileMetadata } from './protocol';

const log = Logger.create('NativeDecoder');

export interface NativeDecoderOptions {
  /** Use scaled preview during scrubbing */
  scrubScale?: number;
  /** How many frames to decode ahead */
  bufferAhead?: number;
  /** How many frames to keep behind current position */
  bufferBehind?: number;
}

const DEFAULT_BUFFER_AHEAD = 8;
const DEFAULT_BUFFER_BEHIND = 3;

/**
 * Native video decoder with frame pre-buffering
 */
export class NativeDecoder {
  private fileId: string;
  private filePath: string;
  private metadata: FileMetadata;
  private options: Required<NativeDecoderOptions>;

  private currentFrame: ImageBitmap | null = null;
  private currentFrameNum = -1;
  private closed = false;
  private reopening = false;

  // Frame ring buffer: pre-decoded frames keyed by frame number
  private frameBuffer: Map<number, ImageBitmap> = new Map();
  // Frames currently being decoded (prevent duplicate requests)
  private pendingFrames: Set<number> = new Set();
  // Single pending decode for the current frame (seekToFrame awaits this)
  private currentDecode: Promise<void> | null = null;

  private constructor(fileId: string, filePath: string, metadata: FileMetadata, options: NativeDecoderOptions) {
    this.fileId = fileId;
    this.filePath = filePath;
    this.metadata = metadata;
    this.options = {
      scrubScale: options.scrubScale ?? 0.5,
      bufferAhead: options.bufferAhead ?? DEFAULT_BUFFER_AHEAD,
      bufferBehind: options.bufferBehind ?? DEFAULT_BUFFER_BEHIND,
    };
  }

  /**
   * Open a video file and create a decoder
   */
  static async open(filePath: string, options?: NativeDecoderOptions): Promise<NativeDecoder> {
    log.debug('Opening file:', filePath);

    if (!NativeHelperClient.isConnected()) {
      const connected = await NativeHelperClient.connect();
      if (!connected) {
        throw new Error('Failed to connect to native helper');
      }
    }

    const metadata = await NativeHelperClient.openFile(filePath);
    return new NativeDecoder(metadata.file_id, filePath, metadata, options ?? {});
  }

  getMetadata(): FileMetadata { return this.metadata; }
  get width(): number { return this.metadata.width; }
  get height(): number { return this.metadata.height; }
  get fps(): number { return this.metadata.fps; }
  get frameCount(): number { return this.metadata.frame_count; }
  get duration(): number { return this.metadata.duration_ms / 1000; }

  /**
   * Get current frame as ImageBitmap for rendering
   */
  getCurrentFrame(): ImageBitmap | null {
    return this.currentFrame;
  }

  getCurrentFrameNum(): number {
    return this.currentFrameNum;
  }

  /**
   * Seek to a specific frame.
   * Fast path: if frame is in buffer, returns immediately.
   * Slow path: decodes frame, then kicks off look-ahead buffer fill.
   */
  async seekToFrame(frame: number, fastScrub = false): Promise<void> {
    if (this.closed) throw new Error('Decoder is closed');

    frame = Math.max(0, Math.min(frame, this.metadata.frame_count - 1));

    // Already showing this frame
    if (frame === this.currentFrameNum && this.currentFrame) {
      return;
    }

    // Fast path: frame is already in buffer
    const buffered = this.frameBuffer.get(frame);
    if (buffered) {
      this.setCurrentFrame(buffered, frame);
      // Kick off look-ahead in background (don't await)
      this.fillBufferAhead(frame, fastScrub);
      return;
    }

    // Wait for any in-flight decode of the same frame
    if (this.pendingFrames.has(frame)) {
      // Wait a bit for it to land in buffer
      await this.waitForFrame(frame, 500);
      const arrived = this.frameBuffer.get(frame);
      if (arrived) {
        this.setCurrentFrame(arrived, frame);
        this.fillBufferAhead(frame, fastScrub);
        return;
      }
    }

    // Slow path: decode this frame now
    if (this.currentDecode) {
      await this.currentDecode;
    }

    this.currentDecode = this.decodeSingleFrame(frame, fastScrub);
    try {
      await this.currentDecode;
    } finally {
      this.currentDecode = null;
    }

    // Set from buffer (decodeSingleFrame puts it there)
    const decoded = this.frameBuffer.get(frame);
    if (decoded) {
      this.setCurrentFrame(decoded, frame);
    }

    // Kick off look-ahead
    this.fillBufferAhead(frame, fastScrub);
  }

  async seekToTime(time: number, fastScrub = false): Promise<void> {
    const frame = Math.round(time * this.metadata.fps);
    await this.seekToFrame(frame, fastScrub);
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;

    // Release all buffered frames
    for (const bitmap of this.frameBuffer.values()) {
      bitmap.close();
    }
    this.frameBuffer.clear();
    this.pendingFrames.clear();

    if (this.currentFrame) {
      this.currentFrame.close();
      this.currentFrame = null;
    }

    try {
      await NativeHelperClient.closeFile(this.fileId);
    } catch { /* ignore */ }
  }

  isClosed(): boolean {
    return this.closed;
  }

  // --- Private ---

  private setCurrentFrame(bitmap: ImageBitmap, frameNum: number): void {
    // Don't close the old currentFrame — it's still in the buffer
    this.currentFrame = bitmap;
    this.currentFrameNum = frameNum;
    this.evictOldFrames(frameNum);
  }

  /**
   * Evict frames outside the retention window
   */
  private evictOldFrames(currentFrame: number): void {
    const minKeep = currentFrame - this.options.bufferBehind;
    const maxKeep = currentFrame + this.options.bufferAhead + 2;

    for (const [num, bitmap] of this.frameBuffer) {
      if (num < minKeep || num > maxKeep) {
        // Don't close the bitmap that's currently displayed
        if (bitmap !== this.currentFrame) {
          bitmap.close();
        }
        this.frameBuffer.delete(num);
      }
    }
  }

  /**
   * Decode a single frame and put it in the buffer
   */
  private async decodeSingleFrame(frame: number, fastScrub: boolean): Promise<void> {
    if (this.frameBuffer.has(frame) || this.pendingFrames.has(frame)) return;

    this.pendingFrames.add(frame);
    try {
      const bitmap = await this.fetchAndCreateBitmap(frame, fastScrub);
      if (bitmap && !this.closed) {
        this.frameBuffer.set(frame, bitmap);
      }
    } catch (err) {
      // Try reopen on session errors
      const msg = err instanceof Error ? err.message : '';
      if (msg.includes('not open') || msg.includes('NOT_OPEN') || msg.includes('Connection lost')) {
        const reopened = await this.tryReopen();
        if (reopened) {
          const bitmap = await this.fetchAndCreateBitmap(frame, fastScrub);
          if (bitmap && !this.closed) {
            this.frameBuffer.set(frame, bitmap);
          }
          return;
        }
      }
      log.error(`Decode frame ${frame} failed`, err);
    } finally {
      this.pendingFrames.delete(frame);
    }
  }

  /**
   * Fire-and-forget: decode frames ahead of current position
   */
  private fillBufferAhead(fromFrame: number, fastScrub: boolean): void {
    const maxFrame = this.metadata.frame_count - 1;
    const ahead = fastScrub ? 3 : this.options.bufferAhead;

    for (let i = 1; i <= ahead; i++) {
      const f = fromFrame + i;
      if (f > maxFrame) break;
      if (this.frameBuffer.has(f) || this.pendingFrames.has(f)) continue;

      // Fire and forget — don't await
      this.decodeInBackground(f, fastScrub);
    }
  }

  /**
   * Background decode — errors are silently logged
   */
  private async decodeInBackground(frame: number, fastScrub: boolean): Promise<void> {
    if (this.closed) return;
    this.pendingFrames.add(frame);
    try {
      const bitmap = await this.fetchAndCreateBitmap(frame, fastScrub);
      if (bitmap && !this.closed) {
        this.frameBuffer.set(frame, bitmap);
      }
    } catch {
      // Silently fail — background prefetch is best-effort
    } finally {
      this.pendingFrames.delete(frame);
    }
  }

  /**
   * Core: request frame from helper, create ImageBitmap
   */
  private async fetchAndCreateBitmap(frame: number, fastScrub: boolean): Promise<ImageBitmap | null> {
    const scale = fastScrub ? this.options.scrubScale : 1.0;

    const decoded = await NativeHelperClient.decodeFrame(this.fileId, frame, {
      format: 'rgba8',
      scale,
    });

    if (decoded.isJpeg) {
      // JPEG path: browser decodes natively — much faster than manual ImageData
      const blob = new Blob([new Uint8Array(decoded.data)], { type: 'image/jpeg' });
      return createImageBitmap(blob);
    }

    // Raw RGBA fallback
    const pixelData = new Uint8ClampedArray(decoded.data);
    const imageData = new ImageData(pixelData, decoded.width, decoded.height);
    return createImageBitmap(imageData);
  }

  /**
   * Wait for a pending frame to arrive in buffer
   */
  private waitForFrame(frame: number, timeoutMs: number): Promise<void> {
    return new Promise((resolve) => {
      const start = performance.now();
      const check = () => {
        if (this.frameBuffer.has(frame) || performance.now() - start > timeoutMs) {
          resolve();
          return;
        }
        setTimeout(check, 5);
      };
      check();
    });
  }

  private async tryReopen(): Promise<boolean> {
    if (this.reopening) return false;
    this.reopening = true;
    try {
      log.info('Reopening file after session reset:', this.filePath);
      if (!NativeHelperClient.isConnected()) {
        const ok = await NativeHelperClient.connect();
        if (!ok) return false;
      }
      const metadata = await NativeHelperClient.openFile(this.filePath);
      this.fileId = metadata.file_id;
      this.metadata = metadata;
      return true;
    } catch (e) {
      log.error('Reopen failed', e);
      return false;
    } finally {
      this.reopening = false;
    }
  }
}

/**
 * Check if native helper is available
 */
export async function isNativeHelperAvailable(): Promise<boolean> {
  try {
    if (NativeHelperClient.isConnected()) return true;
    return await Promise.race([
      NativeHelperClient.connect(),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 2000)),
    ]);
  } catch {
    return false;
  }
}

/**
 * Get supported codecs from native helper
 */
export async function getNativeCodecs(): Promise<string[]> {
  try {
    await NativeHelperClient.getInfo();
    return ['prores', 'dnxhd', 'dnxhr', 'ffv1', 'utvideo', 'mjpeg', 'h264', 'h265', 'vp9'];
  } catch {
    return [];
  }
}
