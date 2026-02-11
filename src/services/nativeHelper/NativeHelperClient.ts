/**
 * Native Helper WebSocket Client
 *
 * Manages the connection to the native helper application and provides
 * methods for video decoding and encoding operations.
 */

import { Logger } from '../logger';
import type {
  Command,
  Response,
  FileMetadata,
  SystemInfo,
  EncodeOutput,
  VideoInfo,
} from './protocol';

import {
  parseFrameHeader,
  isCompressed,
  isJpeg,
} from './protocol';

// LZ4 decompression (we'll use a simple implementation or skip for now)
// In production, use a proper LZ4 library like 'lz4js'

const log = Logger.create('NativeHelper');

export interface NativeHelperConfig {
  port?: number;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  token?: string;
  /** Only reconnect if we were previously connected */
  onlyReconnectIfWasConnected?: boolean;
}

export interface DecodedFrame {
  width: number;
  height: number;
  frameNum: number;
  data: Uint8ClampedArray;
  requestId: number;
  /** If true, data contains JPEG bytes — use createImageBitmap(Blob) instead of ImageData */
  isJpeg?: boolean;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

type ResponseCallback = (response: Response) => void;
type FrameCallback = (frame: DecodedFrame) => void;

/**
 * Singleton client for communicating with the Native Helper
 */
class NativeHelperClientImpl {
  private ws: WebSocket | null = null;
  private config: Required<NativeHelperConfig>;
  private status: ConnectionStatus = 'disconnected';
  private requestId = 0;
  private pendingRequests = new Map<string, ResponseCallback>();
  private progressCallbacks = new Map<string, (percent: number) => void>();
  private frameCallbacks = new Map<string, FrameCallback>();
  private statusListeners = new Set<(status: ConnectionStatus) => void>();
  private reconnectTimer: number | null = null;
  private wasEverConnected = false;

  constructor() {
    this.config = {
      port: 9876,
      autoReconnect: true,
      reconnectInterval: 10000, // 10 seconds between reconnect attempts
      token: '',
      onlyReconnectIfWasConnected: true, // Don't spam reconnects if never connected
    };
  }

  /**
   * Configure the client
   */
  configure(config: NativeHelperConfig): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current connection status
   */
  getStatus(): ConnectionStatus {
    return this.status;
  }

  /**
   * Add a status change listener
   */
  onStatusChange(listener: (status: ConnectionStatus) => void): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  /**
   * Connect to the native helper
   */
  async connect(): Promise<boolean> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return true;
    }

    this.setStatus('connecting');

    return new Promise((resolve) => {
      try {
        this.ws = new WebSocket(`ws://127.0.0.1:${this.config.port}`);
        this.ws.binaryType = 'arraybuffer'; // Ensure binary data comes as ArrayBuffer, not Blob

        this.ws.onopen = async () => {
          log.info('Connected to native helper');
          this.wasEverConnected = true;

          // Authenticate if token provided
          if (this.config.token) {
            try {
              await this.send({ cmd: 'auth', id: this.nextId(), token: this.config.token });
            } catch {
              log.warn('Auth failed');
            }
          }

          this.setStatus('connected');
          resolve(true);
        };

        this.ws.onclose = () => {
          if (this.wasEverConnected) {
            log.info('Disconnected');
          }
          this.setStatus('disconnected');
          this.handleDisconnect();
          if (this.status === 'connecting') {
            resolve(false);
          }
        };

        this.ws.onerror = () => {
          // Don't log errors when helper isn't running - it's optional
          this.setStatus('disconnected');
          if (this.status === 'connecting') {
            resolve(false);
          }
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };
      } catch {
        // Silent fail - helper is optional
        this.setStatus('disconnected');
        resolve(false);
      }
    });
  }

  /**
   * Disconnect from the native helper
   */
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.setStatus('disconnected');
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Open a video file
   */
  async openFile(path: string): Promise<FileMetadata> {
    const id = this.nextId();
    const response = await this.send({ cmd: 'open', id, path });

    if (!response.ok) {
      throw new Error((response as any).error?.message || 'Failed to open file');
    }

    return response as unknown as FileMetadata;
  }

  /**
   * Decode a single frame
   */
  async decodeFrame(
    fileId: string,
    frame: number,
    options?: {
      format?: 'rgba8' | 'rgb8' | 'yuv420';
      scale?: number;
      compression?: 'lz4';
    }
  ): Promise<DecodedFrame> {
    const id = this.nextId();

    return new Promise((resolve, reject) => {
      let settled = false;

      const cleanup = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.frameCallbacks.delete(id);
        this.pendingRequests.delete(id);
      };

      // Set timeout
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error('Decode timeout'));
      }, 10000);

      // Register frame callback (for binary success response)
      this.frameCallbacks.set(id, (frame) => {
        cleanup();
        resolve(frame);
      });

      // Also register in pendingRequests (for JSON error responses)
      this.pendingRequests.set(id, (response) => {
        cleanup();
        const err = (response as any).error;
        reject(new Error(err?.message || 'Decode failed'));
      });

      // Send decode command
      const cmd: Command = {
        cmd: 'decode',
        id,
        file_id: fileId,
        frame,
        format: options?.format,
        scale: options?.scale,
        compression: options?.compression,
      };

      this.sendRaw(JSON.stringify(cmd)).catch((err) => {
        cleanup();
        reject(err);
      });
    });
  }

  /**
   * Prefetch frames around a position (fire and forget)
   */
  prefetch(fileId: string, aroundFrame: number, radius = 50): void {
    if (!this.isConnected()) return;

    const cmd: Command = {
      cmd: 'prefetch',
      file_id: fileId,
      around_frame: aroundFrame,
      radius,
    };

    this.sendRaw(JSON.stringify(cmd)).catch(() => {
      // Ignore prefetch errors
    });
  }

  /**
   * Start an encode job
   */
  async startEncode(output: EncodeOutput, frameCount: number): Promise<string> {
    const id = this.nextId();
    const response = await this.send({ cmd: 'start_encode', id, output, frame_count: frameCount });

    if (!response.ok) {
      throw new Error((response as any).error?.message || 'Failed to start encode');
    }

    return id;
  }

  /**
   * Send a frame for encoding
   */
  async encodeFrame(encodeId: string, frameNum: number, frameData: Uint8Array): Promise<void> {
    // Send text command first
    const cmd: Command = {
      cmd: 'encode_frame',
      id: encodeId,
      frame_num: frameNum,
    };

    await this.sendRaw(JSON.stringify(cmd));

    // Then send binary frame data
    await this.sendRaw(frameData);
  }

  /**
   * Finish encoding
   */
  async finishEncode(encodeId: string): Promise<string> {
    const response = await this.send({ cmd: 'finish_encode', id: encodeId });

    if (!response.ok) {
      throw new Error((response as any).error?.message || 'Failed to finish encode');
    }

    return (response as any).output_path;
  }

  /**
   * Cancel encoding
   */
  async cancelEncode(encodeId: string): Promise<void> {
    await this.send({ cmd: 'cancel_encode', id: encodeId });
  }

  /**
   * Close a file
   */
  async closeFile(fileId: string): Promise<void> {
    const id = this.nextId();
    await this.send({ cmd: 'close', id, file_id: fileId });
  }

  /**
   * Get system info
   */
  async getInfo(): Promise<SystemInfo> {
    const id = this.nextId();
    const response = await this.send({ cmd: 'info', id });

    if (!response.ok) {
      throw new Error((response as any).error?.message || 'Failed to get info');
    }

    return response as unknown as SystemInfo;
  }

  /**
   * Ping the server
   */
  async ping(): Promise<boolean> {
    try {
      const id = this.nextId();
      const response = await this.send({ cmd: 'ping', id });
      return response.ok === true;
    } catch {
      return false;
    }
  }

  /**
   * List available formats for a video URL (YouTube, TikTok, Instagram, etc.)
   */
  async listFormats(url: string): Promise<VideoInfo | null> {
    const id = this.nextId();

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        resolve(null);
      }, 30000);

      this.pendingRequests.set(id, (response: any) => {
        clearTimeout(timeout);
        if (response.ok) {
          resolve({
            title: response.title,
            thumbnail: response.thumbnail,
            duration: response.duration,
            uploader: response.uploader,
            platform: response.platform,
            recommendations: response.recommendations,
            allFormats: response.allFormats,
          });
        } else {
          resolve(null);
        }
      });

      const cmd = {
        cmd: 'list_formats',
        id,
        url,
      };

      this.sendRaw(JSON.stringify(cmd)).catch(() => {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        resolve(null);
      });
    });
  }

  /**
   * Download a YouTube video using yt-dlp
   */
  async downloadYouTube(
    url: string,
    formatId?: string,
    onProgress?: (percent: number) => void
  ): Promise<{ success: boolean; path?: string; error?: string }> {
    const id = this.nextId();

    return new Promise((resolve, reject) => {
      // Set timeout (10 minutes for large videos)
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        this.progressCallbacks.delete(id);
        reject(new Error('Download timeout'));
      }, 600000);

      // Register progress callback if provided
      if (onProgress) {
        this.progressCallbacks.set(id, onProgress);
      }

      // Register completion callback
      this.pendingRequests.set(id, (response: any) => {
        // Check if this is a progress message
        if (response.type === 'progress' && response.percent !== undefined) {
          // Don't resolve yet - this is just progress
          const progressCb = this.progressCallbacks.get(id);
          if (progressCb) {
            progressCb(response.percent);
          }
          return; // Keep waiting for final response
        }

        // Final response
        clearTimeout(timeout);
        this.progressCallbacks.delete(id);
        if (response.ok) {
          resolve({
            success: true,
            path: response.path,
          });
        } else {
          resolve({
            success: false,
            error: response.error?.message || 'Download failed',
          });
        }
      });

      // Send download command
      const cmd: any = {
        cmd: 'download_youtube',
        id,
        url,
      };

      if (formatId) {
        cmd.format_id = formatId;
      }

      this.sendRaw(JSON.stringify(cmd)).catch((err) => {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        this.progressCallbacks.delete(id);
        reject(err);
      });
    });
  }

  /**
   * Download a video from any yt-dlp-supported platform
   */
  async download(
    url: string,
    formatId?: string,
    onProgress?: (percent: number) => void
  ): Promise<{ success: boolean; path?: string; error?: string }> {
    const id = this.nextId();

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        this.progressCallbacks.delete(id);
        reject(new Error('Download timeout'));
      }, 600000);

      if (onProgress) {
        this.progressCallbacks.set(id, onProgress);
      }

      this.pendingRequests.set(id, (response: any) => {
        if (response.type === 'progress' && response.percent !== undefined) {
          const progressCb = this.progressCallbacks.get(id);
          if (progressCb) {
            progressCb(response.percent);
          }
          return;
        }

        clearTimeout(timeout);
        this.progressCallbacks.delete(id);
        if (response.ok) {
          resolve({
            success: true,
            path: response.path,
          });
        } else {
          resolve({
            success: false,
            error: response.error?.message || 'Download failed',
          });
        }
      });

      const cmd: any = {
        cmd: 'download',
        id,
        url,
      };

      if (formatId) {
        cmd.format_id = formatId;
      }

      this.sendRaw(JSON.stringify(cmd)).catch((err) => {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        this.progressCallbacks.delete(id);
        reject(err);
      });
    });
  }

  /**
   * Locate a file by name in common directories (Desktop, Downloads, Videos, Documents, Home)
   * Returns the absolute path if found, or null if not found.
   */
  async locateFile(filename: string, searchDirs?: string[]): Promise<string | null> {
    const id = this.nextId();
    const cmd: any = { cmd: 'locate', id, filename };
    if (searchDirs?.length) {
      cmd.search_dirs = searchDirs;
    }
    const response = await this.send(cmd);
    if (!response.ok) return null;
    const data = response as any;
    if (data.found && data.path) {
      return data.path as string;
    }
    return null;
  }

  /**
   * Get a downloaded file from the Native Helper via HTTP (fast) or WebSocket fallback
   */
  async getDownloadedFile(path: string): Promise<ArrayBuffer | null> {
    // Try HTTP first (much faster than WebSocket base64)
    const httpPort = this.config.port + 1; // HTTP on port+1 (9877)
    try {
      log.debug('Fetching file via HTTP:', path);
      const response = await fetch(`http://127.0.0.1:${httpPort}/file?path=${encodeURIComponent(path)}`);
      if (response.ok) {
        const buffer = await response.arrayBuffer();
        log.debug('File received via HTTP:', buffer.byteLength + ' bytes');
        return buffer;
      }
    } catch (e) {
      log.warn('HTTP fetch failed, falling back to WebSocket', e);
    }

    // Fallback to WebSocket (slower but more compatible)
    const id = this.nextId();

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        resolve(null);
      }, 120000); // 120 seconds for large files via WebSocket

      // For file requests, we expect base64 data in the response
      this.pendingRequests.set(id, async (response: any) => {
        clearTimeout(timeout);
        if (response.ok && response.data) {
          // Decode base64 to ArrayBuffer using fetch (much faster than manual loop)
          try {
            const dataUrl = `data:application/octet-stream;base64,${response.data}`;
            const fetchResponse = await fetch(dataUrl);
            const buffer = await fetchResponse.arrayBuffer();
            resolve(buffer);
          } catch (e) {
            log.error('Failed to decode base64 data', e);
            resolve(null);
          }
        } else {
          resolve(null);
        }
      });

      const cmd = {
        cmd: 'get_file',
        id,
        path,
      };

      this.sendRaw(JSON.stringify(cmd)).catch(() => {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        resolve(null);
      });
    });
  }

  // Private methods

  private nextId(): string {
    return `req_${++this.requestId}`;
  }

  private setStatus(status: ConnectionStatus): void {
    if (this.status !== status) {
      this.status = status;
      this.statusListeners.forEach((listener) => listener(status));
    }
  }

  private handleDisconnect(): void {
    // Reject all pending requests (including decode error handlers)
    this.pendingRequests.forEach((callback) => {
      callback({ id: '', ok: false, error: { code: 'DISCONNECTED', message: 'Connection lost' } });
    });
    this.pendingRequests.clear();
    // Frame callbacks are cleaned up by the pendingRequests error handler above
    // (decodeFrame registers in both maps, cleanup removes from both)
    this.frameCallbacks.clear();

    // Auto-reconnect only if:
    // 1. autoReconnect is enabled
    // 2. Not already trying to connect
    // 3. Either we were connected before OR onlyReconnectIfWasConnected is false
    const shouldReconnect =
      this.config.autoReconnect &&
      this.status !== 'connecting' &&
      (!this.config.onlyReconnectIfWasConnected || this.wasEverConnected);

    if (shouldReconnect) {
      this.reconnectTimer = window.setTimeout(() => {
        log.debug('Attempting reconnect...');
        this.connect();
      }, this.config.reconnectInterval);
    }
  }

  private async handleMessage(data: string | ArrayBuffer): Promise<void> {
    if (typeof data === 'string') {
      // JSON response
      try {
        const response: Response = JSON.parse(data);
        const callback = this.pendingRequests.get(response.id);

        if (callback) {
          // Check if this is a progress message - don't delete callback yet
          const isProgress = (response as any).type === 'progress';
          if (!isProgress) {
            this.pendingRequests.delete(response.id);
          }
          callback(response);
        }
      } catch (err) {
        log.error('Failed to parse response', err);
      }
    } else {
      // Binary frame data
      const header = parseFrameHeader(data);

      if (!header) {
        log.error('Invalid frame header');
        return;
      }

      // Extract payload
      const payloadStart = 16;
      const payload = new Uint8Array(data, payloadStart);

      const jpegFrame = isJpeg(header.flags);

      // Decompress if needed (LZ4 — not used when JPEG is active)
      if (!jpegFrame && isCompressed(header.flags)) {
        log.warn('LZ4 decompression not implemented, using raw data');
      }

      const frame: DecodedFrame = {
        width: header.width,
        height: header.height,
        frameNum: header.frameNum,
        data: new Uint8ClampedArray(payload),
        requestId: header.requestId,
        isJpeg: jpegFrame,
      };

      // Find callback by request ID pattern
      // The request ID in the header maps to our string IDs
      for (const [id, callback] of this.frameCallbacks) {
        // Match by checking if any pending decode could be this frame
        callback(frame);
        this.frameCallbacks.delete(id);
        break;
      }
    }
  }

  private async send(cmd: Command): Promise<Response> {
    if (!this.isConnected()) {
      throw new Error('Not connected');
    }

    return new Promise((resolve, reject) => {
      const id = (cmd as any).id;

      // Set timeout
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error('Request timeout'));
      }, 30000);

      // Register callback
      this.pendingRequests.set(id, (response) => {
        clearTimeout(timeout);
        resolve(response);
      });

      // Send command
      this.sendRaw(JSON.stringify(cmd)).catch((err) => {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        reject(err);
      });
    });
  }

  private async sendRaw(data: string | ArrayBuffer | Uint8Array): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected');
    }

    this.ws.send(data);
  }
}

// Singleton instance
export const NativeHelperClient = new NativeHelperClientImpl();

// Also export the class for testing
export { NativeHelperClientImpl };
