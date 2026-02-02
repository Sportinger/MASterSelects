// LayerBuilder Types - Shared interfaces for the layerBuilder module

import type {
  TimelineClip,
  TimelineTrack,
  Effect,
} from '../../types';

/**
 * Clip time calculation result - memoized per clip per frame
 */
export interface ClipTimeInfo {
  clipLocalTime: number;
  sourceTime: number;
  clipTime: number;
  speed: number;
  absSpeed: number;
}

/**
 * Frame context - computed once per frame, passed to all methods
 * Eliminates duplicate store reads and array filtering
 */
export interface FrameContext {
  // Raw store data
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  isPlaying: boolean;
  isDraggingPlayhead: boolean;
  playheadPosition: number;
  activeCompId: string;
  proxyEnabled: boolean;

  // Store functions
  getInterpolatedTransform: (clipId: string, localTime: number) => any;
  getInterpolatedEffects: (clipId: string, localTime: number) => Effect[];
  getInterpolatedSpeed: (clipId: string, localTime: number) => number;
  getSourceTimeForClip: (clipId: string, localTime: number) => number;
  hasKeyframes: (clipId: string) => boolean;

  // Timing
  now: number;
  frameNumber: number;

  // Pre-computed track data (lazy, cached via getters)
  videoTracks: TimelineTrack[];
  audioTracks: TimelineTrack[];
  visibleVideoTrackIds: Set<string>;
  unmutedAudioTrackIds: Set<string>;
  anyVideoSolo: boolean;
  anyAudioSolo: boolean;

  // Pre-computed clip data (lazy, cached via getters)
  clipsAtTime: TimelineClip[];
  clipsByTrackId: Map<string, TimelineClip>;

  // Media lookups (lazy, cached via getters)
  mediaFiles: any[];
  mediaFileById: Map<string, any>;
  mediaFileByName: Map<string, any>;
  compositionById: Map<string, any>;
}

/**
 * Audio sync state - tracks playing audio and master clock
 */
export interface AudioSyncState {
  audioPlayingCount: number;
  maxAudioDrift: number;
  hasAudioError: boolean;
  masterSet: boolean;
}

/**
 * Audio sync target - unified interface for all audio sources
 */
export interface AudioSyncTarget {
  element: HTMLAudioElement | HTMLVideoElement;
  clip: TimelineClip;
  clipTime: number;
  absSpeed: number;
  isMuted: boolean;
  canBeMaster: boolean;
  type: 'audioTrack' | 'audioProxy' | 'videoElement' | 'mixdown';
}

/**
 * Cached transform data for object reuse
 */
export interface CachedTransform {
  position: { x: number; y: number; z: number };
  scale: { x: number; y: number };
  rotation: { x: number; y: number; z: number };
  opacity: number;
  blendMode: string;
  sourceRef: any; // Reference to detect changes
}

/**
 * Native decoder seek state
 */
export interface NativeDecoderState {
  lastSeekTime: number;
  lastSeekFrame: number;
  isPending: boolean;
}

/**
 * Constants for the layer builder
 */
export const LAYER_BUILDER_CONSTANTS = {
  // Frame rate for playhead quantization
  FRAME_RATE: 30,

  // Native decoder throttling
  NATIVE_SEEK_THROTTLE_MS: 16,

  // Audio sync interval
  AUDIO_SYNC_INTERVAL: 50,

  // Lookahead for nested comp preloading
  LOOKAHEAD_INTERVAL: 100,
  LOOKAHEAD_SECONDS: 3.0,

  // Scrubbing
  SCRUB_AUDIO_DURATION: 150,
  SCRUB_TRIGGER_INTERVAL: 30,

  // Cache limits
  MAX_CLIP_TIME_CACHE: 100,
  MAX_TRANSFORM_CACHE: 50,
} as const;
