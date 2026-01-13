// src/engine/ffmpeg/types.ts
// Type definitions for FFmpeg WASM integration

// ProRes profiles (Apple intermediate codec)
export type ProResProfile = 'proxy' | 'lt' | 'standard' | 'hq' | '4444' | '4444xq';

// HAP formats (GPU-accelerated VJ codec)
export type HapFormat = 'hap' | 'hap_alpha' | 'hap_q';

// DNxHR profiles (Avid broadcast codec)
export type DnxhrProfile = 'dnxhr_lb' | 'dnxhr_sq' | 'dnxhr_hq' | 'dnxhr_hqx' | 'dnxhr_444';

// Video codecs available in FFmpeg WASM (ASYNCIFY build)
// NOTE: This build only includes native FFmpeg encoders
// External libs (libx264, libvpx, libsnappy) require pkg-config which fails in Emscripten
export type FFmpegVideoCodec =
  | 'prores'      // Apple ProRes (professional)
  | 'dnxhd'       // Avid DNxHR (broadcast)
  | 'ffv1'        // FFV1 (lossless archival)
  | 'utvideo'     // Ut Video (fast lossless)
  | 'mjpeg';      // Motion JPEG

// Audio codecs available in ASYNCIFY build
// NOTE: libopus and libmp3lame require pkg-config
export type FFmpegAudioCodec =
  | 'aac'         // AAC (universal)
  | 'flac'        // FLAC (lossless)
  | 'alac'        // Apple Lossless
  | 'pcm_s16le'   // PCM 16-bit
  | 'pcm_s24le'   // PCM 24-bit
  | 'pcm_f32le'   // PCM 32-bit float
  | 'ac3'         // Dolby AC-3
  | 'none';       // No audio

// Container formats
// NOTE: webm removed - requires VP9/VP8 which needs libvpx
export type FFmpegContainer =
  | 'mov'   // QuickTime (Apple/Pro)
  | 'mkv'   // Matroska (open)
  | 'avi'   // AVI (legacy)
  | 'mxf';  // MXF (broadcast)

// Image sequence formats
export type FFmpegImageFormat =
  | 'png'   // PNG (lossless)
  | 'jpeg'  // JPEG (lossy)
  | 'tiff'  // TIFF (print)
  | 'exr'   // OpenEXR (VFX)
  | 'dpx'   // DPX (film)
  | 'webp'  // WebP (web)
  | 'tga';  // TGA (games)

// Pixel formats
export type PixelFormat =
  | 'yuv420p'       // 8-bit 4:2:0 (delivery)
  | 'yuv422p'       // 8-bit 4:2:2 (broadcast)
  | 'yuv444p'       // 8-bit 4:4:4 (high quality)
  | 'yuv420p10le'   // 10-bit 4:2:0 (HDR delivery)
  | 'yuv422p10le'   // 10-bit 4:2:2 (ProRes/DNxHR)
  | 'yuv444p10le'   // 10-bit 4:4:4 (high quality)
  | 'yuva420p'      // 8-bit 4:2:0 + alpha (VP9)
  | 'yuva444p10le'  // 10-bit 4:4:4 + alpha
  | 'rgba'          // 8-bit RGBA (HAP/UtVideo)
  | 'rgb48le'       // 16-bit RGB
  | 'gbrp10le'      // 10-bit planar RGB
  | 'gbrpf32le';    // 32-bit float RGB (EXR)

// Color space
export type ColorSpace = 'bt709' | 'bt2020' | 'srgb';

// Main export settings interface
export interface FFmpegExportSettings {
  // Video basics
  codec: FFmpegVideoCodec;
  container: FFmpegContainer;
  width: number;
  height: number;
  fps: number;

  // Quality control
  bitrate?: number;      // Target bitrate (bps)
  quality?: number;      // CRF value (0-51, lower = better)

  // Pixel format & color
  pixelFormat?: PixelFormat;
  colorSpace?: ColorSpace;

  // ProRes-specific
  proresProfile?: ProResProfile;

  // HAP-specific
  hapFormat?: HapFormat;
  hapChunks?: number;           // Parallel decode chunks (1-64)
  hapCompressor?: 'snappy' | 'none';

  // DNxHR-specific
  dnxhrProfile?: DnxhrProfile;

  // Audio settings
  audioCodec?: FFmpegAudioCodec;
  audioSampleRate?: 44100 | 48000;
  audioBitrate?: number;
  audioChannels?: 1 | 2 | 6;

  // Time range
  startTime: number;
  endTime: number;
}

// Image sequence export settings
export interface ImageSequenceSettings {
  format: FFmpegImageFormat;
  width: number;
  height: number;
  fps: number;
  startFrame: number;
  endFrame: number;
  padding: number;              // Zero-padding for frame numbers
  filenamePattern: string;      // e.g., "frame_%06d"

  // Format-specific options
  jpegQuality?: number;         // 1-100
  pngCompression?: number;      // 0-9
  exrCompression?: 'none' | 'piz' | 'zip' | 'zips' | 'rle' | 'pxr24' | 'b44' | 'dwaa' | 'dwab';
  bitDepth?: 8 | 16 | 32;
}

// Progress reporting
export interface FFmpegProgress {
  frame: number;
  fps: number;
  time: number;           // Encoded time in seconds
  speed: number;          // Encoding speed multiplier
  bitrate: number;        // Current bitrate
  size: number;           // Output size in bytes
  percent: number;        // 0-100
  eta: number;            // Estimated time remaining (seconds)
}

// Log entry from FFmpeg
export interface FFmpegLogEntry {
  type: 'info' | 'warning' | 'error';
  message: string;
  timestamp: number;
}

// Codec metadata
export interface CodecInfo {
  id: string;
  name: string;
  description: string;
  category: 'professional' | 'realtime' | 'lossless' | 'delivery';
  containers: FFmpegContainer[];
  supportsAlpha: boolean;
  supports10bit: boolean;
  defaultPixelFormat: PixelFormat;
}

// Platform preset
export interface PlatformPreset {
  name: string;
  codec: FFmpegVideoCodec;
  container: FFmpegContainer;
  pixelFormat?: PixelFormat;
  quality?: number;
  bitrate?: number;
  audioCodec: FFmpegAudioCodec;
  audioBitrate?: number;
  proresProfile?: ProResProfile;
  dnxhrProfile?: DnxhrProfile;
  hapFormat?: HapFormat;
}
