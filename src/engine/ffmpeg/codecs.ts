// src/engine/ffmpeg/codecs.ts
// Codec definitions, profiles, and platform presets

import type { CodecInfo, FFmpegContainer, PlatformPreset } from './types';

// All available video codecs with metadata
// NOTE: This build uses ASYNCIFY (single-threaded) and includes only native FFmpeg encoders
// External libs (libx264, libvpx, libsnappy) require pkg-config which doesn't work in Emscripten
export const FFMPEG_CODECS: CodecInfo[] = [
  // === Professional Intermediate ===
  {
    id: 'prores',
    name: 'Apple ProRes',
    description: 'Industry standard intermediate codec for editing',
    category: 'professional',
    containers: ['mov'],
    supportsAlpha: true,
    supports10bit: true,
    defaultPixelFormat: 'yuv422p10le',
  },
  {
    id: 'dnxhd',
    name: 'Avid DNxHR',
    description: 'Broadcast and Avid workflow codec',
    category: 'professional',
    containers: ['mxf', 'mov'],
    supportsAlpha: false,
    supports10bit: true,
    defaultPixelFormat: 'yuv422p',
  },

  // === Real-time / Intermediate ===
  {
    id: 'mjpeg',
    name: 'Motion JPEG',
    description: 'Simple frame-by-frame compression, widely compatible',
    category: 'realtime',
    containers: ['mov', 'avi', 'mkv'],
    supportsAlpha: false,
    supports10bit: false,
    defaultPixelFormat: 'yuv422p',
  },

  // === Lossless ===
  {
    id: 'ffv1',
    name: 'FFV1',
    description: 'Open lossless codec for archival',
    category: 'lossless',
    containers: ['mkv', 'avi'],
    supportsAlpha: true,
    supports10bit: true,
    defaultPixelFormat: 'yuv444p10le',
  },
  {
    id: 'utvideo',
    name: 'Ut Video',
    description: 'Fast lossless codec with alpha support',
    category: 'lossless',
    containers: ['avi', 'mkv', 'mov'],
    supportsAlpha: true,
    supports10bit: false,
    defaultPixelFormat: 'rgba',
  },
];

// ProRes profile definitions
export const PRORES_PROFILES = [
  { id: 'proxy', name: 'ProRes Proxy', profile: 0, bitrate: 45, description: 'Offline editing' },
  { id: 'lt', name: 'ProRes LT', profile: 1, bitrate: 102, description: 'Light editing' },
  { id: 'standard', name: 'ProRes 422', profile: 2, bitrate: 147, description: 'Standard quality' },
  { id: 'hq', name: 'ProRes 422 HQ', profile: 3, bitrate: 220, description: 'High quality' },
  { id: '4444', name: 'ProRes 4444', profile: 4, bitrate: 330, description: 'With alpha channel' },
  { id: '4444xq', name: 'ProRes 4444 XQ', profile: 5, bitrate: 500, description: 'Maximum quality' },
] as const;

// DNxHR profile definitions
export const DNXHR_PROFILES = [
  { id: 'dnxhr_lb', name: 'DNxHR LB', description: 'Low Bandwidth - offline' },
  { id: 'dnxhr_sq', name: 'DNxHR SQ', description: 'Standard Quality' },
  { id: 'dnxhr_hq', name: 'DNxHR HQ', description: 'High Quality' },
  { id: 'dnxhr_hqx', name: 'DNxHR HQX', description: '10-bit High Quality' },
  { id: 'dnxhr_444', name: 'DNxHR 444', description: '10-bit 4:4:4 RGB' },
] as const;

// HAP format variants (NOT AVAILABLE in ASYNCIFY build - requires snappy)
// Kept for type compatibility but should not be used
export const HAP_FORMATS = [
  { id: 'hap', name: 'HAP', description: 'Not available' },
  { id: 'hap_alpha', name: 'HAP Alpha', description: 'Not available' },
  { id: 'hap_q', name: 'HAP Q', description: 'Not available' },
] as const;

// Container format info
// NOTE: webm removed - requires VP9/VP8 which needs libvpx (pkg-config issue)
export const CONTAINER_FORMATS = [
  { id: 'mov', name: 'QuickTime (.mov)', description: 'Apple/Professional - ProRes, DNxHR, MJPEG' },
  { id: 'mkv', name: 'Matroska (.mkv)', description: 'Open format - FFV1, MJPEG, UTVideo' },
  { id: 'avi', name: 'AVI (.avi)', description: 'Legacy - MJPEG, UTVideo, FFV1' },
  { id: 'mxf', name: 'MXF (.mxf)', description: 'Broadcast - DNxHR' },
] as const;

// Platform and workflow presets
// NOTE: Social media presets use ProRes - user should use WebCodecs for H.264 delivery
// FFmpeg ASYNCIFY build is optimized for professional intermediate codecs
export const PLATFORM_PRESETS: Record<string, PlatformPreset> = {
  // === Professional Workflows ===
  premiere: {
    name: 'Adobe Premiere',
    codec: 'prores',
    container: 'mov',
    proresProfile: 'hq',
    audioCodec: 'pcm_s24le',
  },
  davinci: {
    name: 'DaVinci Resolve',
    codec: 'dnxhd',
    container: 'mxf',
    dnxhrProfile: 'dnxhr_hq',
    audioCodec: 'pcm_s24le',
  },
  finalcut: {
    name: 'Final Cut Pro',
    codec: 'prores',
    container: 'mov',
    proresProfile: 'hq',
    audioCodec: 'pcm_s24le',
  },
  avid: {
    name: 'Avid Media Composer',
    codec: 'dnxhd',
    container: 'mxf',
    dnxhrProfile: 'dnxhr_hq',
    audioCodec: 'pcm_s24le',
  },

  // === Quality Levels ===
  prores_proxy: {
    name: 'ProRes Proxy',
    codec: 'prores',
    container: 'mov',
    proresProfile: 'proxy',
    audioCodec: 'aac',
    audioBitrate: 128000,
  },
  prores_lt: {
    name: 'ProRes LT',
    codec: 'prores',
    container: 'mov',
    proresProfile: 'lt',
    audioCodec: 'aac',
    audioBitrate: 192000,
  },
  prores_hq: {
    name: 'ProRes HQ',
    codec: 'prores',
    container: 'mov',
    proresProfile: 'hq',
    audioCodec: 'aac',
    audioBitrate: 256000,
  },
  prores_4444: {
    name: 'ProRes 4444 (Alpha)',
    codec: 'prores',
    container: 'mov',
    proresProfile: '4444',
    audioCodec: 'aac',
    audioBitrate: 256000,
  },

  // === Lossless / Archive ===
  archive: {
    name: 'Archive (Lossless)',
    codec: 'ffv1',
    container: 'mkv',
    pixelFormat: 'yuv444p10le',
    audioCodec: 'flac',
  },
  utvideo_alpha: {
    name: 'UTVideo (Alpha)',
    codec: 'utvideo',
    container: 'mov',
    pixelFormat: 'rgba',
    audioCodec: 'pcm_s16le',
  },

  // === Quick Preview ===
  mjpeg_preview: {
    name: 'MJPEG Preview',
    codec: 'mjpeg',
    container: 'mov',
    quality: 5,
    audioCodec: 'aac',
    audioBitrate: 128000,
  },
};

// Helper functions

/**
 * Get codec info by ID
 */
export function getCodecInfo(codecId: string): CodecInfo | undefined {
  return FFMPEG_CODECS.find(c => c.id === codecId);
}

/**
 * Get all codecs that support a specific container
 */
export function getCodecsForContainer(container: FFmpegContainer): CodecInfo[] {
  return FFMPEG_CODECS.filter(c => c.containers.includes(container));
}

/**
 * Get all containers that support a specific codec
 */
export function getContainersForCodec(codecId: string): FFmpegContainer[] {
  const codec = getCodecInfo(codecId);
  return codec?.containers || [];
}

/**
 * Get codecs by category
 */
export function getCodecsByCategory(category: CodecInfo['category']): CodecInfo[] {
  return FFMPEG_CODECS.filter(c => c.category === category);
}

/**
 * Check if a codec supports alpha channel
 */
export function codecSupportsAlpha(codecId: string): boolean {
  return getCodecInfo(codecId)?.supportsAlpha ?? false;
}

/**
 * Check if a codec supports 10-bit
 */
export function codecSupports10bit(codecId: string): boolean {
  return getCodecInfo(codecId)?.supports10bit ?? false;
}

/**
 * Get category label for display
 */
export function getCategoryLabel(category: CodecInfo['category']): string {
  const labels: Record<CodecInfo['category'], string> = {
    professional: 'Professional',
    realtime: 'Real-time / VJ',
    lossless: 'Lossless',
    delivery: 'Delivery',
  };
  return labels[category] || category;
}
