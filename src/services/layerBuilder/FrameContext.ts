// FrameContext - Single store read with lazy cached computations
// Eliminates duplicate store reads and repeated array filtering

import type { TimelineClip, TimelineTrack } from '../../types';
import type { FrameContext, ClipTimeInfo } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { getPlayheadPosition } from './PlayheadState';

/**
 * Create a FrameContext with lazy-computed cached values
 * All store reads happen once here, then values are reused
 */
export function createFrameContext(): FrameContext {
  // === SINGLE STORE READS ===
  const timelineState = useTimelineStore.getState();
  const mediaState = useMediaStore.getState();
  const now = performance.now();

  const {
    clips,
    tracks,
    isPlaying,
    isDraggingPlayhead,
    playheadPosition: storePlayheadPosition,
    playbackSpeed,
    getInterpolatedTransform,
    getInterpolatedEffects,
    getInterpolatedSpeed,
    getSourceTimeForClip,
    hasKeyframes,
  } = timelineState;

  const playheadPosition = getPlayheadPosition(storePlayheadPosition);
  const activeCompId = mediaState.activeCompositionId || 'default';
  const proxyEnabled = mediaState.proxyEnabled;
  const frameNumber = Math.floor(playheadPosition * LAYER_BUILDER_CONSTANTS.FRAME_RATE);

  // === LAZY CACHED VALUES ===
  // These are computed on first access and cached

  let _videoTracks: TimelineTrack[] | null = null;
  let _audioTracks: TimelineTrack[] | null = null;
  let _visibleVideoTrackIds: Set<string> | null = null;
  let _unmutedAudioTrackIds: Set<string> | null = null;
  let _anyVideoSolo: boolean | null = null;
  let _anyAudioSolo: boolean | null = null;
  let _clipsAtTime: TimelineClip[] | null = null;
  let _clipsByTrackId: Map<string, TimelineClip> | null = null;
  let _mediaFileById: Map<string, any> | null = null;
  let _mediaFileByName: Map<string, any> | null = null;
  let _compositionById: Map<string, any> | null = null;

  const context: FrameContext = {
    // Raw data
    clips,
    tracks,
    isPlaying,
    isDraggingPlayhead,
    playheadPosition,
    playbackSpeed,
    activeCompId,
    proxyEnabled,

    // Store functions
    getInterpolatedTransform,
    getInterpolatedEffects,
    getInterpolatedSpeed,
    getSourceTimeForClip,
    hasKeyframes,

    // Timing
    now,
    frameNumber,

    // Media files reference
    mediaFiles: mediaState.files,

    // === LAZY GETTERS ===

    get videoTracks(): TimelineTrack[] {
      if (_videoTracks === null) {
        _videoTracks = tracks.filter(t => t.type === 'video' && t.visible !== false);
      }
      return _videoTracks;
    },

    get audioTracks(): TimelineTrack[] {
      if (_audioTracks === null) {
        _audioTracks = tracks.filter(t => t.type === 'audio');
      }
      return _audioTracks;
    },

    get anyVideoSolo(): boolean {
      if (_anyVideoSolo === null) {
        _anyVideoSolo = this.videoTracks.some(t => t.solo);
      }
      return _anyVideoSolo;
    },

    get anyAudioSolo(): boolean {
      if (_anyAudioSolo === null) {
        _anyAudioSolo = this.audioTracks.some(t => t.solo);
      }
      return _anyAudioSolo;
    },

    get visibleVideoTrackIds(): Set<string> {
      if (_visibleVideoTrackIds === null) {
        _visibleVideoTrackIds = new Set();
        const anyVideoSolo = this.anyVideoSolo;
        for (const track of this.videoTracks) {
          if (track.visible && (!anyVideoSolo || track.solo)) {
            _visibleVideoTrackIds.add(track.id);
          }
        }
      }
      return _visibleVideoTrackIds;
    },

    get unmutedAudioTrackIds(): Set<string> {
      if (_unmutedAudioTrackIds === null) {
        _unmutedAudioTrackIds = new Set();
        const anyAudioSolo = this.anyAudioSolo;
        for (const track of this.audioTracks) {
          if (!track.muted && (!anyAudioSolo || track.solo)) {
            _unmutedAudioTrackIds.add(track.id);
          }
        }
      }
      return _unmutedAudioTrackIds;
    },

    get clipsAtTime(): TimelineClip[] {
      if (_clipsAtTime === null) {
        _clipsAtTime = clips.filter(
          c => playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
        );
      }
      return _clipsAtTime;
    },

    get clipsByTrackId(): Map<string, TimelineClip> {
      if (_clipsByTrackId === null) {
        _clipsByTrackId = new Map();
        for (const clip of this.clipsAtTime) {
          _clipsByTrackId.set(clip.trackId, clip);
        }
      }
      return _clipsByTrackId;
    },

    get mediaFileById(): Map<string, any> {
      if (_mediaFileById === null) {
        _mediaFileById = new Map();
        for (const file of mediaState.files) {
          _mediaFileById.set(file.id, file);
        }
      }
      return _mediaFileById;
    },

    get mediaFileByName(): Map<string, any> {
      if (_mediaFileByName === null) {
        _mediaFileByName = new Map();
        for (const file of mediaState.files) {
          if (file.name) {
            _mediaFileByName.set(file.name, file);
          }
        }
      }
      return _mediaFileByName;
    },

    get compositionById(): Map<string, any> {
      if (_compositionById === null) {
        _compositionById = new Map();
        for (const comp of mediaState.compositions) {
          _compositionById.set(comp.id, comp);
        }
      }
      return _compositionById;
    },
  };

  return context;
}

/**
 * Get media file for a clip - O(1) lookup
 */
export function getMediaFileForClip(ctx: FrameContext, clip: TimelineClip): any | undefined {
  // Try by ID first
  if (clip.mediaFileId) {
    const byId = ctx.mediaFileById.get(clip.mediaFileId);
    if (byId) return byId;
  }

  // Try source.mediaFileId (survives project reload even when top-level mediaFileId doesn't)
  if (clip.source?.mediaFileId && clip.source.mediaFileId !== clip.mediaFileId) {
    const bySourceId = ctx.mediaFileById.get(clip.source.mediaFileId);
    if (bySourceId) return bySourceId;
  }

  // Fall back to name
  if (clip.name) {
    return ctx.mediaFileByName.get(clip.name);
  }

  return undefined;
}

/**
 * Check if a video track is visible (considering solo)
 */
export function isVideoTrackVisible(ctx: FrameContext, trackId: string): boolean {
  return ctx.visibleVideoTrackIds.has(trackId);
}

/**
 * Check if an audio track is muted (considering solo)
 */
export function isAudioTrackMuted(ctx: FrameContext, trackId: string): boolean {
  return !ctx.unmutedAudioTrackIds.has(trackId);
}

/**
 * Get clip at playhead for a track - O(1) lookup
 */
export function getClipForTrack(ctx: FrameContext, trackId: string): TimelineClip | undefined {
  return ctx.clipsByTrackId.get(trackId);
}

// === CLIP TIME CALCULATION MEMOIZATION ===

const clipTimeCache = new Map<string, ClipTimeInfo>();
let lastCacheFrame = -1;

/**
 * Get clip time info with memoization
 * Eliminates repeated calculations of the same clip time
 */
export function getClipTimeInfo(ctx: FrameContext, clip: TimelineClip): ClipTimeInfo {
  // Clear cache on new frame
  if (ctx.frameNumber !== lastCacheFrame) {
    clipTimeCache.clear();
    lastCacheFrame = ctx.frameNumber;
  }

  // Check cache
  const cached = clipTimeCache.get(clip.id);
  if (cached) return cached;

  // Calculate
  const clipLocalTime = ctx.playheadPosition - clip.startTime;
  const speed = ctx.getInterpolatedSpeed(clip.id, clipLocalTime);
  const absSpeed = Math.abs(speed);
  const sourceTime = ctx.getSourceTimeForClip(clip.id, clipLocalTime);
  const initialSpeed = ctx.getInterpolatedSpeed(clip.id, 0);
  const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
  const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

  const info: ClipTimeInfo = {
    clipLocalTime,
    sourceTime,
    clipTime,
    speed,
    absSpeed,
  };

  // Cache and return
  clipTimeCache.set(clip.id, info);

  // Limit cache size
  if (clipTimeCache.size > LAYER_BUILDER_CONSTANTS.MAX_CLIP_TIME_CACHE) {
    const firstKey = clipTimeCache.keys().next().value;
    if (firstKey) clipTimeCache.delete(firstKey);
  }

  return info;
}
