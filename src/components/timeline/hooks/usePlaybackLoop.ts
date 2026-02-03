// Playback loop with audio master clock synchronization

import { useEffect } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { playheadState } from '../../../services/layerBuilder';

interface UsePlaybackLoopProps {
  isPlaying: boolean;
}

/**
 * Audio Master Clock playback loop
 * Audio runs freely without correction, playhead follows audio time
 * This eliminates audio drift and clicking from constant seeks
 */
export function usePlaybackLoop({ isPlaying }: UsePlaybackLoopProps) {
  useEffect(() => {
    if (!isPlaying) {
      // Disable internal position tracking when not playing
      playheadState.isUsingInternalPosition = false;
      playheadState.hasMasterAudio = false;
      playheadState.masterAudioElement = null;
      return;
    }

    let rafId: number;
    let lastTime = performance.now();
    let lastStateUpdate = 0;
    const STATE_UPDATE_INTERVAL = 33; // Update store every 33ms (~30fps for UI/subscribers)

    // Initialize internal position from store and enable high-frequency mode
    playheadState.position = useTimelineStore.getState().playheadPosition;
    playheadState.isUsingInternalPosition = true;
    playheadState.playbackJustStarted = true; // Signal for initial audio sync

    const updatePlayhead = (currentTime: number) => {
      const state = useTimelineStore.getState();
      const {
        duration: dur,
        inPoint: ip,
        outPoint: op,
        loopPlayback: lp,
        pause: ps,
        clips,
        playbackSpeed,
      } = state;
      const effectiveEnd = op !== null ? op : dur;
      const effectiveStart = ip !== null ? ip : 0;

      let newPosition: number;

      // AUDIO MASTER CLOCK: If we have an active audio element, derive playhead from its time
      // Only use audio master for normal forward playback (speed === 1)
      if (playheadState.hasMasterAudio && playheadState.masterAudioElement && playbackSpeed === 1) {
        const audio = playheadState.masterAudioElement;
        if (!audio.paused && audio.readyState >= 2) {
          // Calculate timeline position from audio's current time
          // audioTime = clipInPoint + (timelinePosition - clipStartTime) * speed
          // So: timelinePosition = clipStartTime + (audioTime - clipInPoint) / speed
          const audioTime = audio.currentTime;
          const speed = playheadState.masterClipSpeed || 1;
          newPosition =
            playheadState.masterClipStartTime +
            (audioTime - playheadState.masterClipInPoint) / speed;
        } else {
          // Audio paused or not ready, fall back to system time
          const deltaTime = (currentTime - lastTime) / 1000;
          const cappedDelta = Math.min(deltaTime, 0.1);
          newPosition = playheadState.position + cappedDelta * playbackSpeed;
        }
      } else {
        // No audio master or non-standard speed - use system time with playback speed
        const deltaTime = (currentTime - lastTime) / 1000;
        const cappedDelta = Math.min(deltaTime, 0.1);
        newPosition = playheadState.position + cappedDelta * playbackSpeed;
      }
      lastTime = currentTime;

      // Handle end of timeline / looping (forward playback)
      if (newPosition >= effectiveEnd && playbackSpeed > 0) {
        if (lp) {
          newPosition = effectiveStart;
          // Reset audio master - will be re-established by syncAudioElements
          playheadState.hasMasterAudio = false;
          playheadState.masterAudioElement = null;
          // Seek all audio/video to start
          clips.forEach((clip) => {
            if (clip.source?.audioElement) {
              clip.source.audioElement.currentTime = clip.inPoint;
            }
            if (clip.source?.videoElement) {
              clip.source.videoElement.currentTime = clip.reversed
                ? clip.outPoint
                : clip.inPoint;
            }
          });
        } else {
          newPosition = effectiveEnd;
          ps();
          // Reset playback speed to normal when stopping
          useTimelineStore.setState({ playbackSpeed: 1 });
          playheadState.position = newPosition;
          playheadState.isUsingInternalPosition = false;
          playheadState.hasMasterAudio = false;
          playheadState.masterAudioElement = null;
          useTimelineStore.setState({ playheadPosition: newPosition });
          return;
        }
      }

      // Handle start of timeline (reverse playback)
      if (newPosition <= effectiveStart && playbackSpeed < 0) {
        if (lp) {
          newPosition = effectiveEnd;
          // Reset audio master
          playheadState.hasMasterAudio = false;
          playheadState.masterAudioElement = null;
          // Seek all audio/video to end
          clips.forEach((clip) => {
            if (clip.source?.audioElement) {
              clip.source.audioElement.currentTime = clip.outPoint;
            }
            if (clip.source?.videoElement) {
              clip.source.videoElement.currentTime = clip.reversed
                ? clip.inPoint
                : clip.outPoint;
            }
          });
        } else {
          newPosition = effectiveStart;
          ps();
          // Reset playback speed to normal when stopping
          useTimelineStore.setState({ playbackSpeed: 1 });
          playheadState.position = newPosition;
          playheadState.isUsingInternalPosition = false;
          playheadState.hasMasterAudio = false;
          playheadState.masterAudioElement = null;
          useTimelineStore.setState({ playheadPosition: newPosition });
          return;
        }
      }

      // Clamp to bounds (for edge cases)
      if (newPosition < effectiveStart) {
        newPosition = effectiveStart;
      }
      if (newPosition > effectiveEnd) {
        newPosition = effectiveEnd;
      }

      // Update high-frequency position for render loop to read
      playheadState.position = newPosition;

      // PERFORMANCE: Only update store at throttled interval
      if (currentTime - lastStateUpdate >= STATE_UPDATE_INTERVAL) {
        useTimelineStore.setState({ playheadPosition: newPosition });
        lastStateUpdate = currentTime;
      }

      rafId = requestAnimationFrame(updatePlayhead);
    };

    rafId = requestAnimationFrame(updatePlayhead);

    return () => {
      cancelAnimationFrame(rafId);
      playheadState.isUsingInternalPosition = false;
      playheadState.hasMasterAudio = false;
      playheadState.masterAudioElement = null;
    };
  }, [isPlaying]);
}
