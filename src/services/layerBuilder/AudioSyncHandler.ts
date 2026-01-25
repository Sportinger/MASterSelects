// AudioSyncHandler - Unified audio synchronization for all audio sources
// Consolidates 4 similar 80-line blocks into one reusable handler

import { Logger } from '../logger';
import type { TimelineClip } from '../../types';
import type { FrameContext, AudioSyncState, AudioSyncTarget } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { playheadState, setMasterAudio } from './PlayheadState';
import { audioManager, audioStatusTracker } from '../audioManager';

const log = Logger.create('AudioSyncHandler');

/**
 * AudioSyncHandler - Manages audio synchronization for all audio sources
 */
export class AudioSyncHandler {
  // Scrub audio state
  private lastScrubPosition = -1;
  private lastScrubTime = 0;
  private scrubAudioTimeout: ReturnType<typeof setTimeout> | null = null;

  /**
   * Sync a single audio element with unified logic
   */
  syncAudioElement(
    target: AudioSyncTarget,
    ctx: FrameContext,
    state: AudioSyncState
  ): void {
    const { element, clip, clipTime, absSpeed, isMuted, canBeMaster, type } = target;

    // Set muted state
    element.muted = isMuted;

    // Set pitch preservation
    this.setPitchPreservation(element, clip.preservesPitch !== false);

    const shouldPlay = ctx.isPlaying && !isMuted && !ctx.isDraggingPlayhead && absSpeed > 0.1;

    // Handle scrubbing
    if (ctx.isDraggingPlayhead && !isMuted) {
      this.handleScrub(element, clipTime, ctx);
    } else if (shouldPlay) {
      this.handlePlayback(element, clipTime, absSpeed, clip, canBeMaster, type, state);
    } else {
      this.pauseIfPlaying(element);
    }
  }

  /**
   * Handle audio scrubbing - play short snippet at current position
   */
  private handleScrub(
    element: HTMLAudioElement | HTMLVideoElement,
    clipTime: number,
    ctx: FrameContext
  ): void {
    const timeSinceLastScrub = ctx.now - this.lastScrubTime;
    const positionChanged = Math.abs(ctx.playheadPosition - this.lastScrubPosition) > 0.005;

    if (positionChanged && timeSinceLastScrub > LAYER_BUILDER_CONSTANTS.SCRUB_TRIGGER_INTERVAL) {
      this.lastScrubPosition = ctx.playheadPosition;
      this.lastScrubTime = ctx.now;
      element.playbackRate = 1;
      this.playScrubAudio(element, clipTime);
    }
  }

  /**
   * Play short audio snippet for scrubbing feedback
   */
  private playScrubAudio(element: HTMLAudioElement | HTMLVideoElement, time: number): void {
    element.currentTime = time;
    element.volume = 0.8;
    element.play().catch(() => {});

    // Only set new timeout if none active
    if (!this.scrubAudioTimeout) {
      this.scrubAudioTimeout = setTimeout(() => {
        element.pause();
        this.scrubAudioTimeout = null;
      }, LAYER_BUILDER_CONSTANTS.SCRUB_AUDIO_DURATION);
    }
  }

  /**
   * Handle normal audio playback
   */
  private handlePlayback(
    element: HTMLAudioElement | HTMLVideoElement,
    clipTime: number,
    absSpeed: number,
    clip: TimelineClip,
    canBeMaster: boolean,
    type: AudioSyncTarget['type'],
    state: AudioSyncState
  ): void {
    // Set playback rate
    const targetRate = absSpeed > 0.1 ? absSpeed : 1;
    if (Math.abs(element.playbackRate - targetRate) > 0.01) {
      element.playbackRate = Math.max(0.25, Math.min(4, targetRate));
    }

    // Reset volume after scrubbing
    if (element.volume !== 1) {
      element.volume = 1;
    }

    // Start playback if paused
    if (element.paused) {
      element.currentTime = clipTime;
      element.play().catch(err => {
        log.warn(`[Audio ${type}] Failed to play: ${err.message}`);
        state.hasAudioError = true;
      });
    }

    // Set as master audio if eligible
    if (!state.masterSet && canBeMaster && !element.paused) {
      setMasterAudio(element, clip.startTime, clip.inPoint, absSpeed);
      state.masterSet = true;
    }

    // Track drift for stats (informational only)
    const timeDiff = element.currentTime - clipTime;
    if (Math.abs(timeDiff) > state.maxAudioDrift) {
      state.maxAudioDrift = Math.abs(timeDiff);
    }

    // Count playing audio
    if (!element.paused) {
      state.audioPlayingCount++;
    }
  }

  /**
   * Pause element if currently playing
   */
  private pauseIfPlaying(element: HTMLAudioElement | HTMLVideoElement): void {
    if (!element.paused) {
      element.pause();
    }
  }

  /**
   * Set pitch preservation on audio element
   */
  private setPitchPreservation(element: HTMLAudioElement | HTMLVideoElement, preserve: boolean): void {
    const el = element as HTMLAudioElement & { preservesPitch?: boolean };
    if (el.preservesPitch !== preserve) {
      el.preservesPitch = preserve;
    }
  }

  /**
   * Reset scrub state (call when not scrubbing)
   */
  resetScrubState(): void {
    this.lastScrubPosition = -1;
  }

  /**
   * Stop scrub audio (call when scrubbing ends)
   */
  stopScrubAudio(): void {
    if (this.scrubAudioTimeout) {
      clearTimeout(this.scrubAudioTimeout);
      this.scrubAudioTimeout = null;
    }
  }
}

/**
 * Create initial audio sync state for a frame
 */
export function createAudioSyncState(): AudioSyncState {
  return {
    audioPlayingCount: 0,
    maxAudioDrift: 0,
    hasAudioError: false,
    masterSet: false,
  };
}

/**
 * Finalize audio sync state (call at end of sync)
 */
export function finalizeAudioSync(state: AudioSyncState, isPlaying: boolean): void {
  // Clear master audio if no master was set during playback
  if (!state.masterSet && isPlaying) {
    playheadState.hasMasterAudio = false;
    playheadState.masterAudioElement = null;
  }

  // Update audio status tracker
  audioStatusTracker.updateStatus(
    state.audioPlayingCount,
    state.maxAudioDrift,
    state.hasAudioError
  );
}

/**
 * Resume audio context if needed (browser autoplay policy)
 */
export async function resumeAudioContextIfNeeded(isPlaying: boolean, isDraggingPlayhead: boolean): Promise<void> {
  if (isPlaying && !isDraggingPlayhead) {
    await audioManager.resume().catch(() => {});
  }
}
