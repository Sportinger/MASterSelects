// AudioTrackSyncManager - Handles audio element synchronization with playhead
// Extracted from LayerBuilderService for separation of concerns

import type { TimelineClip, Effect } from '../../types';
import type { FrameContext, AudioSyncState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { playheadState } from './PlayheadState';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip, getClipForTrack, isVideoTrackVisible } from './FrameContext';
import { AudioSyncHandler, createAudioSyncState, finalizeAudioSync, resumeAudioContextIfNeeded } from './AudioSyncHandler';
import { audioRoutingManager } from '../audioRoutingManager';
import { proxyFrameCache } from '../proxyFrameCache';
import { layerPlaybackManager } from '../layerPlaybackManager';

/**
 * Get interpolated volume for a clip from audio-volume effect
 */
function getClipVolume(ctx: FrameContext, clip: TimelineClip, clipLocalTime: number): number {
  const effects = ctx.getInterpolatedEffects(clip.id, clipLocalTime);
  const volumeEffect = effects.find((e: Effect) => e.type === 'audio-volume');
  return (volumeEffect?.params?.volume as number) ?? 1;
}

// EQ band parameter names (matching audio-eq effect)
const EQ_BAND_PARAMS = [
  'band31', 'band62', 'band125', 'band250', 'band500',
  'band1k', 'band2k', 'band4k', 'band8k', 'band16k'
];

/**
 * Get interpolated EQ gains for a clip from audio-eq effect
 * Returns array of 10 gain values in dB, or undefined if no EQ effect
 */
function getClipEQGains(ctx: FrameContext, clip: TimelineClip, clipLocalTime: number): number[] | undefined {
  const effects = ctx.getInterpolatedEffects(clip.id, clipLocalTime);
  const eqEffect = effects.find((e: Effect) => e.type === 'audio-eq');
  if (!eqEffect) return undefined;

  return EQ_BAND_PARAMS.map(param => (eqEffect.params?.[param] as number) ?? 0);
}

export class AudioTrackSyncManager {
  // Sub-module
  private audioSyncHandler = new AudioSyncHandler();

  // Audio sync throttling
  private lastAudioSyncTime = 0;
  private playbackStartFrames = 0;

  // Active audio proxies tracking
  private activeAudioProxies = new Map<string, HTMLAudioElement>();

  /**
   * Sync audio elements to current playhead
   */
  syncAudioElements(): void {
    const ctx = createFrameContext();

    // At non-standard playback speeds (reverse or fast-forward), mute all audio
    // Audio can't play backwards and fast-forward sounds bad
    if (ctx.playbackSpeed !== 1 && ctx.isPlaying) {
      this.muteAllAudio(ctx);
      return;
    }

    // Handle playback start
    const isStartup = playheadState.playbackJustStarted;
    if (isStartup) {
      this.playbackStartFrames++;
      if (this.playbackStartFrames > 10) {
        playheadState.playbackJustStarted = false;
        this.playbackStartFrames = 0;
      }
    } else {
      // Throttle audio sync
      if (ctx.now - this.lastAudioSyncTime < LAYER_BUILDER_CONSTANTS.AUDIO_SYNC_INTERVAL) {
        return;
      }
    }
    this.lastAudioSyncTime = ctx.now;

    // Resume audio context if needed
    resumeAudioContextIfNeeded(ctx.isPlaying, ctx.isDraggingPlayhead);

    // Create sync state
    const state = createAudioSyncState();

    // Sync audio track clips
    this.syncAudioTrackClips(ctx, state);

    // Sync video clip audio (proxies and elements)
    this.syncVideoClipAudio(ctx, state);

    // Sync nested comp mixdown
    this.syncNestedCompMixdown(ctx, state);

    // Pause inactive audio
    this.pauseInactiveAudio(ctx);

    // Sync background layer audio elements
    layerPlaybackManager.syncAudioElements(ctx.playheadPosition, ctx.isPlaying);

    // Finalize
    finalizeAudioSync(state, ctx.isPlaying);
  }

  /**
   * Sync audio track clips
   */
  private syncAudioTrackClips(ctx: FrameContext, state: AudioSyncState): void {
    for (const track of ctx.audioTracks) {
      const clip = getClipForTrack(ctx, track.id);
      if (!clip?.source?.audioElement) continue;

      // Skip audio elements without a valid source (e.g., empty audio from nested comps without audio)
      const audio = clip.source.audioElement;
      if (!audio.src && audio.readyState === 0) continue;

      const timeInfo = getClipTimeInfo(ctx, clip);
      const isMuted = !ctx.unmutedAudioTrackIds.has(track.id);

      this.audioSyncHandler.syncAudioElement({
        element: clip.source.audioElement,
        clip,
        clipTime: timeInfo.clipTime,
        absSpeed: timeInfo.absSpeed,
        isMuted,
        canBeMaster: true,
        type: 'audioTrack',
        volume: getClipVolume(ctx, clip, timeInfo.clipLocalTime),
        eqGains: getClipEQGains(ctx, clip, timeInfo.clipLocalTime),
      }, ctx, state);
    }
  }

  /**
   * Sync video clip audio (proxies and varispeed scrubbing)
   */
  private syncVideoClipAudio(ctx: FrameContext, state: AudioSyncState): void {
    const activeVideoClipIds = new Set<string>();

    for (const track of ctx.videoTracks) {
      const clip = getClipForTrack(ctx, track.id);
      if (!clip?.source?.videoElement || clip.isComposition) continue;

      const mediaFile = getMediaFileForClip(ctx, clip);
      const timeInfo = getClipTimeInfo(ctx, clip);
      const isMuted = !isVideoTrackVisible(ctx, track.id);
      const mediaFileId = mediaFile?.id || clip.mediaFileId || clip.id;

      // Varispeed scrubbing for all clips
      if (ctx.isDraggingPlayhead && !isMuted) {
        const video = clip.source.videoElement;
        if (!video.muted) video.muted = true;
        proxyFrameCache.playScrubAudio(mediaFileId, timeInfo.clipTime, undefined, video.currentSrc || video.src);
      } else if (!ctx.isDraggingPlayhead) {
        proxyFrameCache.stopScrubAudio();
      }

      // Audio proxy handling
      const shouldUseAudioProxy = ctx.proxyEnabled &&
        mediaFile?.hasProxyAudio &&
        (mediaFile.proxyStatus === 'ready' || mediaFile.proxyStatus === 'generating');

      if (shouldUseAudioProxy && mediaFile) {
        activeVideoClipIds.add(clip.id);

        const video = clip.source.videoElement;
        if (!video.muted) video.muted = true;

        const audioProxy = proxyFrameCache.getCachedAudioProxy(mediaFile.id);
        if (audioProxy) {
          this.activeAudioProxies.set(clip.id, audioProxy);

          this.audioSyncHandler.syncAudioElement({
            element: audioProxy,
            clip,
            clipTime: timeInfo.clipTime,
            absSpeed: timeInfo.absSpeed,
            isMuted,
            canBeMaster: !state.masterSet,
            type: 'audioProxy',
            volume: getClipVolume(ctx, clip, timeInfo.clipLocalTime),
            eqGains: getClipEQGains(ctx, clip, timeInfo.clipLocalTime),
          }, ctx, state);
        } else {
          // Trigger preload
          proxyFrameCache.preloadAudioProxy(mediaFile.id);
          proxyFrameCache.getAudioBuffer(mediaFile.id);
        }
      }
    }

    // Fade out and pause inactive audio proxies (prevents audio pop)
    for (const [clipId, audioProxy] of this.activeAudioProxies) {
      if (!activeVideoClipIds.has(clipId) && !audioProxy.paused) {
        audioRoutingManager.fadeOutAndPause(audioProxy);
        this.activeAudioProxies.delete(clipId);
      }
    }
  }

  /**
   * Sync nested composition mixdown audio
   */
  private syncNestedCompMixdown(ctx: FrameContext, state: AudioSyncState): void {
    for (const clip of ctx.clipsAtTime) {
      if (!clip.isComposition || !clip.mixdownAudio || !clip.hasMixdownAudio) continue;

      const timeInfo = getClipTimeInfo(ctx, clip);
      const track = ctx.videoTracks.find(t => t.id === clip.trackId);
      const isMuted = track ? !isVideoTrackVisible(ctx, track.id) : false;

      this.audioSyncHandler.syncAudioElement({
        element: clip.mixdownAudio,
        clip,
        clipTime: timeInfo.clipTime,
        absSpeed: timeInfo.absSpeed,
        isMuted,
        canBeMaster: !state.masterSet,
        type: 'mixdown',
        volume: getClipVolume(ctx, clip, timeInfo.clipLocalTime),
        eqGains: getClipEQGains(ctx, clip, timeInfo.clipLocalTime),
      }, ctx, state);
    }
  }

  /**
   * Pause audio not at playhead
   */
  private pauseInactiveAudio(ctx: FrameContext): void {
    for (const clip of ctx.clips) {
      const isAtPlayhead = ctx.clipsAtTime.some(c => c.id === clip.id);

      if (clip.source?.audioElement && !isAtPlayhead && !clip.source.audioElement.paused) {
        // Don't pause if an active clip shares this audio element (split clips)
        const audio = clip.source.audioElement;
        const sharedByActiveClip = ctx.clipsAtTime.some(
          active => active.source?.audioElement === audio
        );
        if (!sharedByActiveClip) {
          audioRoutingManager.fadeOutAndPause(audio);
        }
      }

      // Pause WebCodecs audio players for inactive clips
      if (clip.source?.audioPlayer && !isAtPlayhead) {
        clip.source.audioPlayer.pause();
      }

      if (clip.mixdownAudio && !isAtPlayhead && !clip.mixdownAudio.paused) {
        audioRoutingManager.fadeOutAndPause(clip.mixdownAudio);
      }
    }
  }

  /**
   * Mute all audio during non-standard playback (reverse or fast-forward)
   * Audio can't play backwards and fast-forward audio sounds bad
   */
  private muteAllAudio(ctx: FrameContext): void {
    // Clear master audio since we're not using audio sync
    playheadState.hasMasterAudio = false;
    playheadState.masterAudioElement = null;

    // Fade out and pause all audio elements to prevent pops
    for (const clip of ctx.clips) {
      if (clip.source?.audioElement && !clip.source.audioElement.paused) {
        audioRoutingManager.fadeOutAndPause(clip.source.audioElement);
      }
      if (clip.source?.videoElement && !clip.source.videoElement.muted) {
        clip.source.videoElement.muted = true;
      }
      // Pause WebCodecs audio players
      if (clip.source?.audioPlayer) {
        clip.source.audioPlayer.pause();
      }
      if (clip.mixdownAudio && !clip.mixdownAudio.paused) {
        audioRoutingManager.fadeOutAndPause(clip.mixdownAudio);
      }
    }
  }
}
