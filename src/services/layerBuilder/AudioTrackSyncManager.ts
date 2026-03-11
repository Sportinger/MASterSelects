// AudioTrackSyncManager - Handles audio element synchronization with playhead
// Extracted from LayerBuilderService for separation of concerns

import type { TimelineClip, Effect } from '../../types';
import type { FrameContext, AudioSyncState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { playheadState } from './PlayheadState';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip, getClipForTrack, isVideoTrackVisible } from './FrameContext';
import { AudioSyncHandler, createAudioSyncState, finalizeAudioSync, resumeAudioContextIfNeeded } from './AudioSyncHandler';
import { proxyFrameCache } from '../proxyFrameCache';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { Logger } from '../logger';

const log = Logger.create('CutTransition');

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

  // Seamless audio cut transition: keep old audio element playing through cuts
  // (same approach as video handoff in VideoSyncManager)
  private lastAudioTrackState = new Map<string, {
    clipId: string;
    fileId: string;
    file: File;
    audioElement: HTMLAudioElement;
    outPoint: number;
  }>();
  private audioHandoffs = new Map<string, HTMLAudioElement>();
  private audioHandoffElements = new Set<HTMLAudioElement>();

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

    // Compute audio handoffs for seamless cut transitions
    this.computeAudioHandoffs(ctx);

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

    // Update audio track state for next frame's handoff detection
    this.updateLastAudioTrackState(ctx);

    // Pre-buffer audio for upcoming clips (audio lookahead)
    this.preBufferUpcomingAudio(ctx);

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

      if (this.shouldSuppressLinkedAudioClipScrub(ctx, clip)) {
        if (!clip.source.audioElement.paused) {
          clip.source.audioElement.pause();
        }
        continue;
      }

      // Skip audio elements without a valid source (e.g., empty audio from nested comps without audio)
      const audio = clip.source.audioElement;
      if (!audio.src && audio.readyState === 0) continue;

      const timeInfo = getClipTimeInfo(ctx, clip);
      const isMuted = !ctx.unmutedAudioTrackIds.has(track.id);

      // Use handoff element if available (seamless cut transition)
      const handoffAudio = this.audioHandoffs.get(clip.id);
      this.audioSyncHandler.syncAudioElement({
        element: handoffAudio ?? clip.source.audioElement,
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
    let hasScrubAudioSource = false;

    for (const track of ctx.videoTracks) {
      const clip = getClipForTrack(ctx, track.id);
      if (!clip?.source?.videoElement || clip.isComposition) continue;

      const mediaFile = getMediaFileForClip(ctx, clip);
      const timeInfo = getClipTimeInfo(ctx, clip);
      const isMuted = !isVideoTrackVisible(ctx, track.id);
      const mediaFileId = mediaFile?.id || clip.mediaFileId || clip.id;
      const linkedAudioClip = this.getLinkedAudioClipAtPlayhead(ctx, clip);
      const audioSettingsClip = linkedAudioClip ?? clip;
      const audioSettingsTimeInfo = linkedAudioClip ? getClipTimeInfo(ctx, linkedAudioClip) : timeInfo;

      // Varispeed scrubbing should follow the effective audio clip settings.
      const clipVolume = getClipVolume(ctx, audioSettingsClip, audioSettingsTimeInfo.clipLocalTime);
      const eqGains = getClipEQGains(ctx, audioSettingsClip, audioSettingsTimeInfo.clipLocalTime);
      let audioMuted = isMuted || clipVolume <= 0.01;

      if (!audioMuted && linkedAudioClip) {
        const linkedTrackMuted = !ctx.unmutedAudioTrackIds.has(linkedAudioClip.trackId);
        if (linkedTrackMuted) audioMuted = true;
      } else if (!audioMuted && clip.linkedClipId && !ctx.clips.some(c => c.id === clip.linkedClipId)) {
        // Linked audio clip was deleted - mute scrub audio.
        audioMuted = true;
      }

      const useVarispeedScrubAudio = ctx.isDraggingPlayhead && !audioMuted && proxyFrameCache.hasAudioBuffer(mediaFileId);

      if (ctx.isDraggingPlayhead && !audioMuted) {
        hasScrubAudioSource = true;
        const video = clip.source.videoElement;
        if (!video.muted) video.muted = true;
        proxyFrameCache.playScrubAudio(
          mediaFileId,
          timeInfo.clipTime,
          undefined,
          video.currentSrc || video.src,
          { volume: clipVolume, eqGains }
        );
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

          const shouldUseAudioProxyScrubFallback =
            ctx.isDraggingPlayhead &&
            !linkedAudioClip &&
            !useVarispeedScrubAudio;

          if (!ctx.isDraggingPlayhead || shouldUseAudioProxyScrubFallback) {
            this.audioSyncHandler.syncAudioElement({
              element: audioProxy,
              clip,
              clipTime: timeInfo.clipTime,
              absSpeed: timeInfo.absSpeed,
              isMuted,
              canBeMaster: !state.masterSet,
              type: 'audioProxy',
              volume: clipVolume,
              eqGains,
            }, ctx, state);
          } else if (!audioProxy.paused) {
            audioProxy.pause();
          }
        } else {
          // Trigger preload
          proxyFrameCache.preloadAudioProxy(mediaFile.id);
          proxyFrameCache.getAudioBuffer(mediaFile.id);
        }
      }
    }

    // Pause inactive audio proxies
    for (const [clipId, audioProxy] of this.activeAudioProxies) {
      if (!activeVideoClipIds.has(clipId) && !audioProxy.paused) {
        audioProxy.pause();
        this.activeAudioProxies.delete(clipId);
      }
    }

    if (!ctx.isDraggingPlayhead || !hasScrubAudioSource) {
      proxyFrameCache.stopScrubAudio();
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

      if (clip.source?.audioElement && !isAtPlayhead && !clip.source.audioElement.paused
          && !this.audioHandoffElements.has(clip.source.audioElement)) {
        clip.source.audioElement.pause();
      }

      if (clip.mixdownAudio && !isAtPlayhead && !clip.mixdownAudio.paused) {
        clip.mixdownAudio.pause();
      }
    }
  }

  /**
   * Detect same-source sequential audio clips for seamless handoff.
   * Same logic as video handoff: compare mediaFileId (not blob URL).
   */
  private computeAudioHandoffs(ctx: FrameContext): void {
    this.audioHandoffs.clear();
    this.audioHandoffElements.clear();

    // Handoffs are needed during playback (seamless cut transitions)
    // Only skip during scrubbing where we don't need seamless audio
    if (ctx.isDraggingPlayhead) return;

    for (const track of ctx.audioTracks) {
      const clip = getClipForTrack(ctx, track.id);
      if (!clip?.source?.audioElement) continue;

      const prev = this.lastAudioTrackState.get(track.id);
      if (!prev) continue;

      if (prev.clipId === clip.id) {
        // Same clip as last frame - persist handoff if we were using one.
        // Without this, the handoff only lasts 1 frame and then the clip's
        // cold element takes over (causing an audio click/gap).
        if (prev.audioElement !== clip.source.audioElement) {
          this.audioHandoffs.set(clip.id, prev.audioElement);
          this.audioHandoffElements.add(prev.audioElement);
        }
        continue;
      }

      // Different clip - detect same-source sequential cut for new handoff
      const clipFileId = clip.source.mediaFileId || clip.mediaFileId;
      const sameSource = clipFileId
        ? clipFileId === prev.fileId
        : clip.file === prev.file;
      if (!sameSource) {
        log.debug('Audio handoff SKIP: different source', { track: track.id });
        continue;
      }

      const inOutGap = Math.abs(clip.inPoint - prev.outPoint);
      if (inOutGap > 0.1) {
        log.debug('Audio handoff SKIP: non-continuous', { gap: inOutGap.toFixed(3) });
        continue;
      }

      const elemDrift = Math.abs(prev.audioElement.currentTime - clip.inPoint);
      if (elemDrift > 0.5) {
        log.debug('Audio handoff SKIP: element too far', {
          elementTime: prev.audioElement.currentTime.toFixed(3),
          inPoint: clip.inPoint.toFixed(3),
          drift: elemDrift.toFixed(3),
        });
        continue;
      }

      log.info('Audio handoff START', {
        track: track.id.slice(-6),
        prevClip: prev.clipId.slice(-6),
        newClip: clip.id.slice(-6),
        drift: elemDrift.toFixed(3),
      });
      this.audioHandoffs.set(clip.id, prev.audioElement);
      this.audioHandoffElements.add(prev.audioElement);
    }
  }

  /**
   * Update per-track audio state for next frame's handoff detection
   */
  private updateLastAudioTrackState(ctx: FrameContext): void {
    for (const track of ctx.audioTracks) {
      const clip = getClipForTrack(ctx, track.id);
      if (!clip?.source?.audioElement) continue;

      const handoffElement = this.audioHandoffs.get(clip.id);
      const audio = handoffElement ?? clip.source.audioElement;
      const fileId = clip.source.mediaFileId || clip.mediaFileId || '';

      this.lastAudioTrackState.set(track.id, {
        clipId: clip.id,
        fileId,
        file: clip.file,
        audioElement: audio,
        outPoint: clip.outPoint,
      });
    }
  }

  // Audio lookahead: pre-buffer upcoming audio elements
  private static readonly AUDIO_LOOKAHEAD_TIME = 1.0; // seconds
  private preBufferedAudio = new WeakSet<HTMLAudioElement>();

  /**
   * Pre-buffer audio elements for clips about to become active.
   * Without this, audio starts cold at cut points causing a 100-500ms gap.
   * We seek the audio element to the correct inPoint and call load() to
   * ensure the browser has decoded audio data ready.
   */
  private preBufferUpcomingAudio(ctx: FrameContext): void {
    if (!ctx.isPlaying || ctx.isDraggingPlayhead) return;

    const lookaheadEnd = ctx.playheadPosition + AudioTrackSyncManager.AUDIO_LOOKAHEAD_TIME;

    for (const clip of ctx.clips) {
      // Only audio clips with audio elements
      if (!clip.source?.audioElement) continue;

      const audio = clip.source.audioElement;
      const clipStart = clip.startTime;

      // Is this clip about to become active? (starts within lookahead, not yet active)
      if (clipStart <= ctx.playheadPosition || clipStart > lookaheadEnd) continue;

      // Skip if already pre-buffered
      if (this.preBufferedAudio.has(audio)) continue;

      // Skip if no source loaded
      if (!audio.src && audio.readyState === 0) continue;

      // Pre-seek to inPoint so audio data is buffered and ready
      const targetTime = clip.inPoint;
      if (Math.abs(audio.currentTime - targetTime) > 0.1) {
        audio.currentTime = targetTime;
      }

      this.preBufferedAudio.add(audio);
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

    // Pause all audio elements
    for (const clip of ctx.clips) {
      if (clip.source?.audioElement && !clip.source.audioElement.paused) {
        clip.source.audioElement.pause();
      }
      if (clip.source?.videoElement && !clip.source.videoElement.muted) {
        clip.source.videoElement.muted = true;
      }
      if (clip.mixdownAudio && !clip.mixdownAudio.paused) {
        clip.mixdownAudio.pause();
      }
    }
  }

  private getLinkedAudioClipAtPlayhead(ctx: FrameContext, clip: TimelineClip): TimelineClip | undefined {
    if (!clip.linkedClipId) return undefined;

    const linkedClip = ctx.clipsAtTime.find(c => c.id === clip.linkedClipId);
    return linkedClip?.source?.type === 'audio' ? linkedClip : undefined;
  }

  private getLinkedVideoClipAtPlayhead(ctx: FrameContext, clip: TimelineClip): TimelineClip | undefined {
    if (!clip.linkedClipId) return undefined;

    const linkedClip = ctx.clipsAtTime.find(c => c.id === clip.linkedClipId);
    return linkedClip?.source?.videoElement && !linkedClip.isComposition ? linkedClip : undefined;
  }

  private shouldSuppressLinkedAudioClipScrub(ctx: FrameContext, clip: TimelineClip): boolean {
    if (!ctx.isDraggingPlayhead || !clip.linkedClipId) {
      return false;
    }

    const linkedVideoClip = this.getLinkedVideoClipAtPlayhead(ctx, clip);
    if (!linkedVideoClip) {
      return false;
    }

    const mediaFile = getMediaFileForClip(ctx, linkedVideoClip);
    const mediaFileId = mediaFile?.id || linkedVideoClip.mediaFileId || linkedVideoClip.id;
    return proxyFrameCache.hasAudioBuffer(mediaFileId);
  }
}
