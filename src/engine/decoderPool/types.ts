// Decoder Pool types — shared video decoder management

import type { WebCodecsPlayer } from '../WebCodecsPlayer.ts';
import type { NativeDecoder } from '../../services/nativeHelper/index.ts';

/** Priority levels for decoder requests */
export type DecoderPriority = 'playback' | 'scrub' | 'preload';

/** Decoder backend type */
export type DecoderType = 'HTMLVideoElement' | 'WebCodecs' | 'NativeDecoder';

/** Handle to a shared decoder instance */
export interface DecoderHandle {
  id: string;
  mediaFileId: string;
  decoderType: DecoderType;

  /** The underlying decoder */
  videoElement?: HTMLVideoElement;
  webCodecsPlayer?: WebCodecsPlayer;
  nativeDecoder?: NativeDecoder;

  /** Current state */
  currentTime: number;
  refCount: number;
  lastAccessFrame: number;
  priority: DecoderPriority;

  /** Whether this handle is shared (multiple consumers) */
  isShared: boolean;
}

/** Request for a decoder from the pool */
export interface DecoderRequest {
  mediaFileId: string;
  sourceTime: number;
  priority: DecoderPriority;
  clipId: string;
  /** Tolerance in seconds for sharing (default: 1 frame at 30fps) */
  shareTolerance?: number;
}

/** Statistics for the decoder pool */
export interface DecoderPoolStats {
  activeDecoders: number;
  idleDecoders: number;
  sharedDecoders: number;
  totalCreated: number;
  totalEvicted: number;
  totalShares: number;
}

/** Configuration for the decoder pool */
export interface DecoderPoolConfig {
  maxDecoders: number;
  shareToleranceFrames: number;
  idleTimeoutMs: number;
}
