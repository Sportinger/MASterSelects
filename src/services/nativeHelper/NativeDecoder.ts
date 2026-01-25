/**
 * Native Decoder
 *
 * Provides a decoder interface similar to WebCodecsPlayer that uses
 * the native helper for decoding. Returns VideoFrame or ImageBitmap
 * objects that can be used directly with WebGPU textures.
 */

import { Logger } from '../logger';
import { NativeHelperClient } from './NativeHelperClient';
import type { FileMetadata } from './protocol';

const log = Logger.create('NativeDecoder');

export interface NativeDecoderOptions {
  /** Use scaled preview during scrubbing */
  scrubScale?: number;
  /** Prefetch radius in frames */
  prefetchRadius?: number;
}

/**
 * Native video decoder that communicates with the helper application
 */
export class NativeDecoder {
  private fileId: string;
  private metadata: FileMetadata;
  private options: Required<NativeDecoderOptions>;

  private currentFrame: ImageBitmap | null = null;
  private currentFrameNum = -1;
  private pendingDecode: Promise<void> | null = null;
  private lastPrefetchFrame = -1;
  private closed = false;

  private constructor(fileId: string, metadata: FileMetadata, options: NativeDecoderOptions) {
    this.fileId = fileId;
    this.metadata = metadata;
    this.options = {
      scrubScale: options.scrubScale ?? 0.5,
      prefetchRadius: options.prefetchRadius ?? 30,
    };
  }

  /**
   * Open a video file and create a decoder
   */
  static async open(filePath: string, options?: NativeDecoderOptions): Promise<NativeDecoder> {
    log.debug('Opening file:', filePath);

    // Ensure connected
    if (!NativeHelperClient.isConnected()) {
      log.debug('Not connected, connecting...');
      const connected = await NativeHelperClient.connect();
      if (!connected) {
        throw new Error('Failed to connect to native helper');
      }
      log.debug('Connected successfully');
    }

    // Open file
    log.debug('Sending open command...');
    const metadata = await NativeHelperClient.openFile(filePath);
    log.debug('Got metadata:', metadata);

    return new NativeDecoder(metadata.file_id, metadata, options ?? {});
  }

  /**
   * Get file metadata
   */
  getMetadata(): FileMetadata {
    return this.metadata;
  }

  /**
   * Get video dimensions
   */
  get width(): number {
    return this.metadata.width;
  }

  get height(): number {
    return this.metadata.height;
  }

  /**
   * Get frame rate
   */
  get fps(): number {
    return this.metadata.fps;
  }

  /**
   * Get total frame count
   */
  get frameCount(): number {
    return this.metadata.frame_count;
  }

  /**
   * Get duration in seconds
   */
  get duration(): number {
    return this.metadata.duration_ms / 1000;
  }

  /**
   * Get current frame as ImageBitmap (compatible with WebGPU textures)
   * Returns null if no frame is ready
   */
  getCurrentFrame(): ImageBitmap | null {
    return this.currentFrame;
  }

  /**
   * Get current frame number
   */
  getCurrentFrameNum(): number {
    return this.currentFrameNum;
  }

  /**
   * Seek to a specific frame
   * @param frame Frame number
   * @param fastScrub Use lower resolution for faster scrubbing
   */
  async seekToFrame(frame: number, fastScrub = false): Promise<void> {
    if (this.closed) {
      throw new Error('Decoder is closed');
    }

    // Clamp frame number
    frame = Math.max(0, Math.min(frame, this.metadata.frame_count - 1));

    // Skip if already at this frame
    if (frame === this.currentFrameNum && this.currentFrame) {
      return;
    }

    // Wait for any pending decode
    if (this.pendingDecode) {
      await this.pendingDecode;
    }

    // Start decode
    this.pendingDecode = this.decodeFrame(frame, fastScrub);

    try {
      await this.pendingDecode;
    } finally {
      this.pendingDecode = null;
    }

    // Trigger prefetch if moved significantly
    if (Math.abs(frame - this.lastPrefetchFrame) > this.options.prefetchRadius / 2) {
      this.lastPrefetchFrame = frame;
      NativeHelperClient.prefetch(this.fileId, frame, this.options.prefetchRadius);
    }
  }

  /**
   * Seek to a specific time
   * @param time Time in seconds
   * @param fastScrub Use lower resolution for faster scrubbing
   */
  async seekToTime(time: number, fastScrub = false): Promise<void> {
    const frame = Math.round(time * this.metadata.fps);
    await this.seekToFrame(frame, fastScrub);
  }

  /**
   * Close the decoder and release resources
   */
  async close(): Promise<void> {
    if (this.closed) return;

    this.closed = true;

    // Release current frame
    if (this.currentFrame) {
      this.currentFrame.close();
      this.currentFrame = null;
    }

    // Close file on native side
    try {
      await NativeHelperClient.closeFile(this.fileId);
    } catch {
      // Ignore close errors
    }
  }

  /**
   * Check if decoder is closed
   */
  isClosed(): boolean {
    return this.closed;
  }

  // Private methods

  private async decodeFrame(frame: number, fastScrub: boolean): Promise<void> {
    try {
      const scale = fastScrub ? this.options.scrubScale : 1.0;

      const decoded = await NativeHelperClient.decodeFrame(this.fileId, frame, {
        format: 'rgba8',
        scale,
      });

      // Create ImageBitmap from decoded data
      // Ensure we have a proper Uint8ClampedArray with ArrayBuffer (not SharedArrayBuffer)
      const pixelData = new Uint8ClampedArray(decoded.data);
      const imageData = new ImageData(pixelData, decoded.width, decoded.height);
      const bitmap = await createImageBitmap(imageData);

      // Release old frame
      if (this.currentFrame) {
        this.currentFrame.close();
      }

      this.currentFrame = bitmap;
      this.currentFrameNum = frame;
    } catch (err) {
      log.error('Decode failed', err);
      throw err;
    }
  }
}

/**
 * Check if native helper is available
 */
export async function isNativeHelperAvailable(): Promise<boolean> {
  try {
    if (NativeHelperClient.isConnected()) {
      return true;
    }

    // Try to connect with a short timeout
    const connected = await Promise.race([
      NativeHelperClient.connect(),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 2000)),
    ]);

    return connected;
  } catch {
    return false;
  }
}

/**
 * Get supported codecs from native helper
 */
export async function getNativeCodecs(): Promise<string[]> {
  try {
    // Check if helper is available by calling getInfo
    await NativeHelperClient.getInfo();
    // The native helper supports these codecs via FFmpeg
    return ['prores', 'dnxhd', 'dnxhr', 'ffv1', 'utvideo', 'mjpeg', 'h264', 'h265', 'vp9'];
  } catch {
    return [];
  }
}
