// AudioSyncHandler - Unified audio synchronization for all audio sources
// Consolidates 4 similar 80-line blocks into one reusable handler

import { Logger } from '../logger';
import type { TimelineClip } from '../../types';
import type { FrameContext, AudioSyncState, AudioSyncTarget } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { playheadState, setMasterAudio } from './PlayheadState';
import { audioManager, audioStatusTracker } from '../audioManager';
import { audioRoutingManager } from '../audioRoutingManager';
import { vfPipelineMonitor } from '../vfPipelineMonitor';

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
    const { element, clip, clipTime, absSpeed, isMuted, canBeMaster, type, volume = 1, eqGains } = target;
    const effectivelyMuted = isMuted || volume <= 0.01;

    // Set muted state
    element.muted = effectivelyMuted;

    // Set pitch preservation
    this.setPitchPreservation(element, clip.preservesPitch !== false);

    const shouldPlay = ctx.isPlaying && !effectivelyMuted && !ctx.isDraggingPlayhead && absSpeed > 0.1;

    // Handle scrubbing
    if (ctx.isDraggingPlayhead && !effectivelyMuted) {
      this.handleScrub(element, clipTime, ctx, volume, eqGains);
    } else if (shouldPlay) {
      this.handlePlayback(element, clipTime, absSpeed, clip, canBeMaster, type, state, volume, eqGains);
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
    ctx: FrameContext,
    volume: number,
    eqGains?: number[]
  ): void {
    const timeSinceLastScrub = ctx.now - this.lastScrubTime;
    const positionChanged = Math.abs(ctx.playheadPosition - this.lastScrubPosition) > 0.005;

    if (positionChanged && timeSinceLastScrub > LAYER_BUILDER_CONSTANTS.SCRUB_TRIGGER_INTERVAL) {
      this.lastScrubPosition = ctx.playheadPosition;
      this.lastScrubTime = ctx.now;
      element.playbackRate = 1;
      this.applyScrubEffects(element, volume, eqGains);
      this.playScrubAudio(element, clipTime);
    }
  }

  /**
   * Standalone audio clips scrub via their media element fallback, so the
   * element still needs the clip's current volume/EQ applied while dragging.
   */
  private applyScrubEffects(
    element: HTMLAudioElement | HTMLVideoElement,
    volume: number,
    eqGains?: number[]
  ): void {
    const hasEQ = eqGains?.some(g => Math.abs(g) > 0.01) ?? false;

    if (hasEQ || volume > 1) {
      void audioRoutingManager.applyEffects(element, volume, eqGains ?? new Array(10).fill(0));
      return;
    }

    const targetVolume = Math.max(0, Math.min(1, volume));
    if (Math.abs(element.volume - targetVolume) > 0.01) {
      element.volume = targetVolume;
    }
  }

  /**
   * Play short audio snippet for scrubbing feedback
   */
  private playScrubAudio(element: HTMLAudioElement | HTMLVideoElement, time: number): void {
    element.currentTime = time;
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
    state: AudioSyncState,
    volume: number = 1,
    eqGains?: number[]
  ): void {
    // Set playback rate
    const targetRate = absSpeed > 0.1 ? absSpeed : 1;
    if (Math.abs(element.playbackRate - targetRate) > 0.01) {
      element.playbackRate = Math.max(0.25, Math.min(4, targetRate));
      vfPipelineMonitor.record('audio_rate_change', {
        type,
        rate: Math.round(targetRate * 100) / 100,
        clipId: clip.id,
      });
    }

    // Check if we have EQ to apply (any non-zero gain)
    const hasEQ = eqGains && eqGains.some(g => Math.abs(g) > 0.01);

    if (hasEQ) {
      // Use Web Audio routing for volume + EQ
      // This handles both volume and EQ through the audio graph
      audioRoutingManager.applyEffects(element, volume, eqGains!);
    } else {
      // Simple volume-only path (no Web Audio overhead)
      // HTMLMediaElement.volume only accepts [0, 1] range - clamp to prevent errors
      const targetVolume = Math.max(0, Math.min(1, volume));
      if (Math.abs(element.volume - targetVolume) > 0.01) {
        element.volume = targetVolume;
      }
    }

    // Start playback if paused
    if (element.paused) {
      // Only seek before play if the element is significantly out of sync.
      // After a clean pause, the element is already at the correct position —
      // an unnecessary seek forces the browser to re-decode from the last
      // keyframe, causing a 100-400ms startup delay.
      const currentDrift = Math.abs(element.currentTime - clipTime);
      if (currentDrift > 0.1) {
        element.currentTime = clipTime;
      }
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

    // Audio drift correction: if audio drifts > 0.3s from expected position, re-sync.
    // Without this, audio can drift indefinitely (user reported 1300ms delay).
    const timeDiff = element.currentTime - clipTime;
    if (Math.abs(timeDiff) > 0.3) {
      vfPipelineMonitor.record('audio_drift_correct', {
        type,
        driftMs: Math.round(timeDiff * 1000),
        clipId: clip.id,
      });
      element.currentTime = clipTime;
    } else if (Math.abs(timeDiff) > 0.05) {
      vfPipelineMonitor.record('audio_drift', {
        type,
        driftMs: Math.round(timeDiff * 1000),
        clipId: clip.id,
      });
    }
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

  // Record to VF pipeline monitor
  const audioStatus = audioStatusTracker.getStatus();
  vfPipelineMonitor.record('audio_status', {
    status: audioStatus.status,
    playing: audioStatus.playing,
    driftMs: audioStatus.drift,
  });
}

/**
 * Resume audio context if needed (browser autoplay policy)
 */
export async function resumeAudioContextIfNeeded(isPlaying: boolean, isDraggingPlayhead: boolean): Promise<void> {
  if (isPlaying && !isDraggingPlayhead) {
    await audioManager.resume().catch(() => {});
  }
}
