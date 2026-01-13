/**
 * Native Helper Protocol Types
 *
 * Defines the message types for communication with the native helper
 * via WebSocket.
 */

// Message type bytes
export const MESSAGE_TYPES = {
  COMMAND: 0x01,
  FRAME: 0x02,
  RESPONSE: 0x03,
  ERROR: 0x04,
  PROGRESS: 0x05,
} as const;

// Frame flags
export const FRAME_FLAGS = {
  COMPRESSED: 0x01,
  SCALED: 0x02,
  DELTA: 0x04,
} as const;

// Magic bytes
export const MAGIC = new Uint8Array([0x4D, 0x48]); // "MH"

// Commands
export interface AuthCommand {
  cmd: 'auth';
  id: string;
  token: string;
}

export interface OpenCommand {
  cmd: 'open';
  id: string;
  path: string;
}

export interface DecodeCommand {
  cmd: 'decode';
  id: string;
  file_id: string;
  frame: number;
  format?: 'rgba8' | 'rgb8' | 'yuv420';
  scale?: number;
  compression?: 'lz4';
}

export interface DecodeRangeCommand {
  cmd: 'decode_range';
  id: string;
  file_id: string;
  start_frame: number;
  end_frame: number;
  priority?: 'low' | 'normal' | 'high';
}

export interface PrefetchCommand {
  cmd: 'prefetch';
  file_id: string;
  around_frame: number;
  radius?: number;
}

export interface StartEncodeCommand {
  cmd: 'start_encode';
  id: string;
  output: EncodeOutput;
  frame_count: number;
}

export interface EncodeFrameCommand {
  cmd: 'encode_frame';
  id: string;
  frame_num: number;
}

export interface FinishEncodeCommand {
  cmd: 'finish_encode';
  id: string;
}

export interface CancelEncodeCommand {
  cmd: 'cancel_encode';
  id: string;
}

export interface CloseCommand {
  cmd: 'close';
  id: string;
  file_id: string;
}

export interface InfoCommand {
  cmd: 'info';
  id: string;
}

export interface PingCommand {
  cmd: 'ping';
  id: string;
}

export interface DownloadYouTubeCommand {
  cmd: 'download_youtube';
  id: string;
  url: string;
  output_dir?: string;
}

export interface GetFileCommand {
  cmd: 'get_file';
  id: string;
  path: string;
}

export type Command =
  | AuthCommand
  | OpenCommand
  | DecodeCommand
  | DecodeRangeCommand
  | PrefetchCommand
  | StartEncodeCommand
  | EncodeFrameCommand
  | FinishEncodeCommand
  | CancelEncodeCommand
  | CloseCommand
  | InfoCommand
  | PingCommand
  | DownloadYouTubeCommand
  | GetFileCommand;

// Encode settings
export interface EncodeOutput {
  path: string;
  codec: 'prores' | 'dnxhd' | 'h264' | 'h265' | 'vp9' | 'ffv1' | 'utvideo' | 'mjpeg';
  profile?: string;
  width: number;
  height: number;
  fps: number;
  audio?: AudioSettings;
}

export interface AudioSettings {
  codec: 'aac' | 'flac' | 'pcm' | 'alac';
  sample_rate: number;
  channels: number;
  bitrate?: number;
}

// Responses
export interface OkResponse {
  id: string;
  ok: true;
  [key: string]: unknown;
}

export interface ErrorResponse {
  id: string;
  ok: false;
  error: {
    code: string;
    message: string;
  };
}

export interface ProgressResponse {
  id: string;
  progress: number;
  frames_done: number;
  frames_total: number;
  eta_ms?: number;
}

export type Response = OkResponse | ErrorResponse | ProgressResponse;

// File metadata
export interface FileMetadata {
  file_id: string;
  width: number;
  height: number;
  fps: number;
  duration_ms: number;
  frame_count: number;
  codec: string;
  profile?: string;
  color_space?: string;
  audio_tracks: number;
  hw_accel?: string;
}

// System info
export interface SystemInfo {
  version: string;
  ffmpeg_version: string;
  hw_accel: string[];
  cache_used_mb: number;
  cache_max_mb: number;
  open_files: number;
}

// Frame header (16 bytes)
export interface FrameHeader {
  type: number;
  flags: number;
  width: number;
  height: number;
  frameNum: number;
  requestId: number;
}

/**
 * Parse a binary frame message header
 */
export function parseFrameHeader(data: ArrayBuffer): FrameHeader | null {
  if (data.byteLength < 16) {
    return null;
  }

  const view = new DataView(data);

  // Check magic bytes
  if (view.getUint8(0) !== 0x4D || view.getUint8(1) !== 0x48) {
    return null;
  }

  return {
    type: view.getUint8(2),
    flags: view.getUint8(3),
    width: view.getUint16(4, true),
    height: view.getUint16(6, true),
    frameNum: view.getUint32(8, true),
    requestId: view.getUint32(12, true),
  };
}

/**
 * Check if frame is compressed
 */
export function isCompressed(flags: number): boolean {
  return (flags & FRAME_FLAGS.COMPRESSED) !== 0;
}

/**
 * Check if frame is scaled
 */
export function isScaled(flags: number): boolean {
  return (flags & FRAME_FLAGS.SCALED) !== 0;
}

// Error codes
export const ERROR_CODES = {
  AUTH_REQUIRED: 'AUTH_REQUIRED',
  INVALID_TOKEN: 'INVALID_TOKEN',
  FILE_NOT_FOUND: 'FILE_NOT_FOUND',
  PERMISSION_DENIED: 'PERMISSION_DENIED',
  UNSUPPORTED_CODEC: 'UNSUPPORTED_CODEC',
  DECODE_ERROR: 'DECODE_ERROR',
  ENCODE_ERROR: 'ENCODE_ERROR',
  OUT_OF_MEMORY: 'OUT_OF_MEMORY',
  INVALID_FRAME: 'INVALID_FRAME',
  INVALID_PATH: 'INVALID_PATH',
  FILE_NOT_OPEN: 'FILE_NOT_OPEN',
  ENCODE_NOT_STARTED: 'ENCODE_NOT_STARTED',
  INTERNAL_ERROR: 'INTERNAL_ERROR',
} as const;
