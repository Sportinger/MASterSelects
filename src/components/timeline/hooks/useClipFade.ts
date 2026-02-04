// useClipFade - Fade-in/out handle dragging with real-time keyframe generation
// Creates opacity keyframes (video) or volume keyframes (audio) as the user drags
// Preserves existing bezier handles when adjusting fade duration

import { useState, useCallback, useRef } from 'react';
import type { TimelineClip, TimelineTrack, AnimatableProperty, EasingType } from '../../../types';
import { createEffectProperty } from '../../../types';
import type { ClipFadeState } from '../types';

interface KeyframeData {
  id: string;
  clipId: string;
  time: number;
  property: string;
  value: number;
  easing: string;
  handleIn?: { x: number; y: number };
  handleOut?: { x: number; y: number };
}

interface UseClipFadeProps {
  // Clip and track data
  clipMap: Map<string, TimelineClip>;
  tracks: TimelineTrack[];

  // Keyframe actions
  addKeyframe: (clipId: string, property: AnimatableProperty, value: number, time?: number, easing?: EasingType) => void;
  removeKeyframe: (keyframeId: string) => void;
  moveKeyframe: (keyframeId: string, newTime: number) => void;
  getClipKeyframes: (clipId: string) => KeyframeData[];

  // Audio effect management
  addClipEffect: (clipId: string, effectType: string) => void;

  // Helpers
  pixelToTime: (pixel: number) => number;
}

interface UseClipFadeReturn {
  clipFade: ClipFadeState | null;
  clipFadeRef: React.MutableRefObject<ClipFadeState | null>;
  handleFadeStart: (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => void;
  getFadeInDuration: (clipId: string) => number;
  getFadeOutDuration: (clipId: string) => number;
}

export function useClipFade({
  clipMap,
  tracks,
  addKeyframe,
  removeKeyframe,
  moveKeyframe,
  getClipKeyframes,
  addClipEffect,
  pixelToTime,
}: UseClipFadeProps): UseClipFadeReturn {
  const [clipFade, setClipFade] = useState<ClipFadeState | null>(null);
  const clipFadeRef = useRef<ClipFadeState | null>(clipFade);
  clipFadeRef.current = clipFade;

  // Store the keyframe IDs we're working with during a drag
  const fadeKeyframeIdsRef = useRef<{
    startKeyframeId?: string;  // The keyframe at start/end of fade (value 0)
    endKeyframeId?: string;    // The keyframe at fade point (value 1)
  }>({});

  // Helper to check if clip is on an audio track
  const isAudioClip = useCallback((clipId: string): boolean => {
    const clip = clipMap.get(clipId);
    if (!clip) return false;
    const track = tracks.find(t => t.id === clip.trackId);
    return track?.type === 'audio';
  }, [clipMap, tracks]);

  // Helper to get the fade property for a clip (opacity for video, volume for audio)
  const getFadeProperty = useCallback((clipId: string): AnimatableProperty => {
    const clip = clipMap.get(clipId);
    if (!clip) return 'opacity';

    if (isAudioClip(clipId)) {
      // For audio clips, use the audio-volume effect's volume parameter
      const volumeEffect = clip.effects?.find(e => e.type === 'audio-volume');
      if (volumeEffect) {
        return createEffectProperty(volumeEffect.id, 'volume');
      }
      // If no volume effect exists, we'll need to create one first
      return 'opacity'; // Fallback, but we'll handle this in handleFadeStart
    }

    return 'opacity';
  }, [clipMap, isAudioClip]);

  // Ensure audio clip has a volume effect and return its property
  const ensureAudioVolumeEffect = useCallback((clipId: string): AnimatableProperty => {
    const clip = clipMap.get(clipId);
    if (!clip) return 'opacity';

    if (!isAudioClip(clipId)) return 'opacity';

    let volumeEffect = clip.effects?.find(e => e.type === 'audio-volume');
    if (!volumeEffect) {
      // Add the audio-volume effect
      addClipEffect(clipId, 'audio-volume');
      // Get the updated clip with the new effect
      const updatedClip = clipMap.get(clipId);
      volumeEffect = updatedClip?.effects?.find(e => e.type === 'audio-volume');
    }

    if (volumeEffect) {
      return createEffectProperty(volumeEffect.id, 'volume');
    }

    return 'opacity';
  }, [clipMap, isAudioClip, addClipEffect]);

  // Calculate fade-in duration from keyframes (opacity for video, volume for audio)
  const getFadeInDuration = useCallback((clipId: string): number => {
    const keyframes = getClipKeyframes(clipId);
    const fadeProperty = getFadeProperty(clipId);

    const fadeKeyframes = keyframes
      .filter(k => k.property === fadeProperty || (isAudioClip(clipId) && k.property.includes('.volume')))
      .sort((a, b) => a.time - b.time);

    if (fadeKeyframes.length < 2) return 0;

    // Fade-in: First keyframe should be at time 0 with value 0,
    // and we look for the next keyframe with value 1
    const firstKf = fadeKeyframes[0];
    if (firstKf.time !== 0 || firstKf.value !== 0) return 0;

    // Find the first keyframe with value 1 (or near 1)
    for (const kf of fadeKeyframes) {
      if (kf.value >= 0.99 && kf.time > 0) {
        return kf.time;
      }
    }

    return 0;
  }, [getClipKeyframes, getFadeProperty, isAudioClip]);

  // Calculate fade-out duration from keyframes (opacity for video, volume for audio)
  const getFadeOutDuration = useCallback((clipId: string): number => {
    const clip = clipMap.get(clipId);
    if (!clip) return 0;

    const keyframes = getClipKeyframes(clipId);
    const fadeProperty = getFadeProperty(clipId);

    const fadeKeyframes = keyframes
      .filter(k => k.property === fadeProperty || (isAudioClip(clipId) && k.property.includes('.volume')))
      .sort((a, b) => a.time - b.time);

    if (fadeKeyframes.length < 2) return 0;

    // Fade-out: Last keyframe should be at clip.duration with value 0,
    // and we look for the previous keyframe with value 1
    const lastKf = fadeKeyframes[fadeKeyframes.length - 1];
    const tolerance = 0.01; // 10ms tolerance for floating point
    if (Math.abs(lastKf.time - clip.duration) > tolerance || lastKf.value !== 0) return 0;

    // Find the last keyframe with value 1 (before the final 0)
    for (let i = fadeKeyframes.length - 2; i >= 0; i--) {
      const kf = fadeKeyframes[i];
      if (kf.value >= 0.99) {
        return clip.duration - kf.time;
      }
    }

    return 0;
  }, [clipMap, getClipKeyframes, getFadeProperty, isAudioClip]);

  const handleFadeStart = useCallback(
    (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => {
      e.stopPropagation();
      e.preventDefault();

      const clip = clipMap.get(clipId);
      if (!clip) return;

      // For audio clips, ensure the volume effect exists
      const fadeProperty = isAudioClip(clipId)
        ? ensureAudioVolumeEffect(clipId)
        : 'opacity' as AnimatableProperty;

      // Get existing fade duration
      const originalFadeDuration = edge === 'left'
        ? getFadeInDuration(clipId)
        : getFadeOutDuration(clipId);

      // Find existing keyframes for this fade
      const keyframes = getClipKeyframes(clipId);
      const fadeKeyframes = keyframes
        .filter(k => k.property === fadeProperty || (isAudioClip(clipId) && k.property.includes('.volume')))
        .sort((a, b) => a.time - b.time);

      // Reset keyframe IDs for this drag session
      fadeKeyframeIdsRef.current = {};

      if (edge === 'left') {
        // Fade-in: Look for keyframe at 0 (value 0) and next one (value 1)
        const startKf = fadeKeyframes.find(k => k.time === 0 && k.value === 0);
        const endKf = fadeKeyframes.find(k => k.value >= 0.99 && k.time > 0 && k.time < clip.duration * 0.5);

        if (startKf && endKf) {
          fadeKeyframeIdsRef.current.startKeyframeId = startKf.id;
          fadeKeyframeIdsRef.current.endKeyframeId = endKf.id;
        }
      } else {
        // Fade-out: Look for keyframe at end (value 0) and previous one (value 1)
        const endKf = fadeKeyframes.find(k => Math.abs(k.time - clip.duration) < 0.01 && k.value === 0);
        const startKf = fadeKeyframes.find(k => k.value >= 0.99 && k.time > clip.duration * 0.5);

        if (startKf && endKf) {
          fadeKeyframeIdsRef.current.startKeyframeId = startKf.id;
          fadeKeyframeIdsRef.current.endKeyframeId = endKf.id;
        }
      }

      const initialFade: ClipFadeState = {
        clipId,
        edge,
        startX: e.clientX,
        currentX: e.clientX,
        clipDuration: clip.duration,
        originalFadeDuration,
      };
      setClipFade(initialFade);
      clipFadeRef.current = initialFade;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const fade = clipFadeRef.current;
        if (!fade) return;

        const currentClip = clipMap.get(fade.clipId);
        if (!currentClip) return;

        // Determine the property to use for this fade (opacity for video, volume for audio)
        const currentFadeProperty = isAudioClip(fade.clipId)
          ? ensureAudioVolumeEffect(fade.clipId)
          : 'opacity' as AnimatableProperty;

        // Calculate new fade duration based on mouse movement
        const deltaX = moveEvent.clientX - fade.startX;
        const deltaTime = pixelToTime(Math.abs(deltaX));

        let newFadeDuration: number;
        if (fade.edge === 'left') {
          // For fade-in: dragging right increases duration
          newFadeDuration = fade.originalFadeDuration + (deltaX > 0 ? deltaTime : -deltaTime);
        } else {
          // For fade-out: dragging left increases duration
          newFadeDuration = fade.originalFadeDuration + (deltaX < 0 ? deltaTime : -deltaTime);
        }

        // Clamp fade duration (min 0, max half of clip duration)
        const maxFade = currentClip.duration * 0.5;
        newFadeDuration = Math.max(0, Math.min(newFadeDuration, maxFade));

        // Update keyframes FIRST (before triggering React re-render)
        const { startKeyframeId, endKeyframeId } = fadeKeyframeIdsRef.current;

        if (fade.edge === 'left') {
          // Fade-in: move the end keyframe (value 1) to new position
          if (startKeyframeId && endKeyframeId) {
            // Just move the existing keyframe - preserves all bezier handles
            moveKeyframe(endKeyframeId, newFadeDuration);
          } else if (newFadeDuration > 0.01) {
            // No existing fade - create new keyframes
            addKeyframe(fade.clipId, currentFadeProperty, 0, 0, 'ease-out');
            addKeyframe(fade.clipId, currentFadeProperty, 1, newFadeDuration, 'linear');

            // Get the newly created keyframe IDs for future moves
            const newKeyframes = getClipKeyframes(fade.clipId).filter(k =>
              k.property === currentFadeProperty || (isAudioClip(fade.clipId) && k.property.includes('.volume'))
            );
            const newStartKf = newKeyframes.find(k => k.time === 0 && k.value === 0);
            const newEndKf = newKeyframes.find(k => k.value >= 0.99 && k.time > 0);
            if (newStartKf && newEndKf) {
              fadeKeyframeIdsRef.current.startKeyframeId = newStartKf.id;
              fadeKeyframeIdsRef.current.endKeyframeId = newEndKf.id;
            }
          }
        } else {
          // Fade-out: move the start keyframe (value 1) to new position
          if (startKeyframeId && endKeyframeId) {
            // Just move the existing keyframe - preserves all bezier handles
            const fadeStartTime = currentClip.duration - newFadeDuration;
            moveKeyframe(startKeyframeId, fadeStartTime);
          } else if (newFadeDuration > 0.01) {
            // No existing fade - create new keyframes
            const fadeStartTime = currentClip.duration - newFadeDuration;
            addKeyframe(fade.clipId, currentFadeProperty, 1, fadeStartTime, 'ease-in');
            addKeyframe(fade.clipId, currentFadeProperty, 0, currentClip.duration, 'linear');

            // Get the newly created keyframe IDs for future moves
            const newKeyframes = getClipKeyframes(fade.clipId).filter(k =>
              k.property === currentFadeProperty || (isAudioClip(fade.clipId) && k.property.includes('.volume'))
            );
            const newStartKf = newKeyframes.find(k => k.value >= 0.99 && k.time > currentClip.duration * 0.5);
            const newEndKf = newKeyframes.find(k => Math.abs(k.time - currentClip.duration) < 0.01 && k.value === 0);
            if (newStartKf && newEndKf) {
              fadeKeyframeIdsRef.current.startKeyframeId = newStartKf.id;
              fadeKeyframeIdsRef.current.endKeyframeId = newEndKf.id;
            }
          }
        }

        // Handle removing fade when duration goes to 0
        if (newFadeDuration <= 0.01 && startKeyframeId && endKeyframeId) {
          removeKeyframe(startKeyframeId);
          removeKeyframe(endKeyframeId);
          fadeKeyframeIdsRef.current = {};
        }

        // Now update local state to trigger re-render with the fresh keyframe data
        const updated = {
          ...fade,
          currentX: moveEvent.clientX,
        };
        setClipFade(updated);
        clipFadeRef.current = updated;
      };

      const handleMouseUp = () => {
        setClipFade(null);
        clipFadeRef.current = null;
        fadeKeyframeIdsRef.current = {};
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [clipMap, getFadeInDuration, getFadeOutDuration, getClipKeyframes, pixelToTime, addKeyframe, moveKeyframe, removeKeyframe, isAudioClip, ensureAudioVolumeEffect]
  );

  return {
    clipFade,
    clipFadeRef,
    handleFadeStart,
    getFadeInDuration,
    getFadeOutDuration,
  };
}
