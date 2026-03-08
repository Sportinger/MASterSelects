// Export-related types and interfaces

import type { TimelineClip, TimelineTrack } from '../../stores/timeline/types';
import type { BlendMode } from '../../types';
import type { WebCodecsPlayer } from '../WebCodecsPlayer';

// ============ VIDEO CODECS ============

export type VideoCodec = 'h264' | 'h265' | 'vp9' | 'av1';
export type ContainerFormat = 'mp4' | 'webm';
export type ExportMode = 'fast' | 'precise';

// ============ EXPORT SETTINGS ============

export interface ExportSettings {
  width: number;
  height: number;
  fps: number;
  codec: VideoCodec;
  container: ContainerFormat;
  bitrate: number;
  startTime: number;
  endTime: number;
  // Audio settings
  includeAudio?: boolean;
  audioSampleRate?: 44100 | 48000;
  audioBitrate?: number;  // 128000 - 320000
  normalizeAudio?: boolean;
  // Export mode
  exportMode?: ExportMode;  // 'fast' = WebCodecs sequential, 'precise' = HTMLVideoElement
}

export interface FullExportSettings extends ExportSettings {
  filename?: string;
}

// ============ PROGRESS ============

export interface ExportProgress {
  phase: 'video' | 'audio' | 'muxing';
  currentFrame: number;
  totalFrames: number;
  percent: number;
  estimatedTimeRemaining: number;
  currentTime: number;
  audioPhase?: 'extracting' | 'processing' | 'effects' | 'mixing' | 'encoding' | 'complete';
  audioPercent?: number;
}

// ============ INTERNAL STATE ============

export interface ExportClipState {
  clipId: string;
  webCodecsPlayer: WebCodecsPlayer | null;
  lastSampleIndex: number;
  isSequential: boolean; // true if using sequential decoding
  runtimeSource?: TimelineClip['source'];
  runtimeOwnerId?: string;
}

// ============ PRESETS ============

export interface ResolutionPreset {
  label: string;
  width: number;
  height: number;
}

export interface FrameRatePreset {
  label: string;
  fps: number;
}

export interface ContainerFormatOption {
  id: ContainerFormat;
  label: string;
  extension: string;
}

export interface VideoCodecOption {
  id: VideoCodec;
  label: string;
  description: string;
}

// ============ LAYER BUILDING ============

export interface LayerTransformData {
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number };
  opacity: number;
  blendMode: string;
}

export interface BaseLayerProps {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  blendMode: BlendMode;
  effects: any[];
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number };
}

// ============ FRAME CONTEXT (Performance Optimization) ============

/**
 * Cached context for a single frame - avoids repeated getState() calls.
 * Create once per frame, pass to all functions.
 */
export interface FrameContext {
  time: number;
  fps: number;
  frameTolerance: number;
  clipsAtTime: TimelineClip[];
  trackMap: Map<string, TimelineTrack>;
  clipsByTrack: Map<string, TimelineClip>;
  getInterpolatedTransform: (clipId: string, localTime: number) => any;
  getInterpolatedEffects: (clipId: string, localTime: number) => any[];
  getSourceTimeForClip: (clipId: string, localTime: number) => number;
  getInterpolatedSpeed: (clipId: string, localTime: number) => number;
}

// ============ FPS-BASED CONSTANTS ============

/**
 * Get frame tolerance in microseconds based on fps.
 * Uses 1.5 frame duration for tolerance.
 */
export function getFrameTolerance(fps: number): number {
  return Math.round((1_000_000 / fps) * 1.5);
}

/**
 * Get keyframe interval (frames between keyframes).
 * Default: 1 keyframe per second.
 */
export function getKeyframeInterval(fps: number): number {
  return Math.round(fps);
}
