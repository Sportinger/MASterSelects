// useClipFade - Fade-in/out handle dragging with real-time keyframe generation
// Creates opacity keyframes as the user drags the fade handles

import { useState, useCallback, useRef } from 'react';
import type { TimelineClip } from '../../../types';
import type { ClipFadeState } from '../types';

interface UseClipFadeProps {
  // Clip data
  clipMap: Map<string, TimelineClip>;

  // Keyframe actions
  addKeyframe: (clipId: string, property: 'opacity', value: number, time: number) => void;
  removeKeyframe: (keyframeId: string) => void;
  getClipKeyframes: (clipId: string) => Array<{
    id: string;
    clipId: string;
    time: number;
    property: string;
    value: number;
    easing: string;
  }>;

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
  addKeyframe,
  removeKeyframe,
  getClipKeyframes,
  pixelToTime,
}: UseClipFadeProps): UseClipFadeReturn {
  const [clipFade, setClipFade] = useState<ClipFadeState | null>(null);
  const clipFadeRef = useRef<ClipFadeState | null>(clipFade);
  clipFadeRef.current = clipFade;

  // Calculate fade-in duration from opacity keyframes
  const getFadeInDuration = useCallback((clipId: string): number => {
    const keyframes = getClipKeyframes(clipId);
    const opacityKeyframes = keyframes
      .filter(k => k.property === 'opacity')
      .sort((a, b) => a.time - b.time);

    if (opacityKeyframes.length < 2) return 0;

    // Fade-in: First keyframe should be at time 0 with value 0,
    // and we look for the next keyframe with value 1
    const firstKf = opacityKeyframes[0];
    if (firstKf.time !== 0 || firstKf.value !== 0) return 0;

    // Find the first keyframe with opacity 1 (or near 1)
    for (const kf of opacityKeyframes) {
      if (kf.value >= 0.99 && kf.time > 0) {
        return kf.time;
      }
    }

    return 0;
  }, [getClipKeyframes]);

  // Calculate fade-out duration from opacity keyframes
  const getFadeOutDuration = useCallback((clipId: string): number => {
    const clip = clipMap.get(clipId);
    if (!clip) return 0;

    const keyframes = getClipKeyframes(clipId);
    const opacityKeyframes = keyframes
      .filter(k => k.property === 'opacity')
      .sort((a, b) => a.time - b.time);

    if (opacityKeyframes.length < 2) return 0;

    // Fade-out: Last keyframe should be at clip.duration with value 0,
    // and we look for the previous keyframe with value 1
    const lastKf = opacityKeyframes[opacityKeyframes.length - 1];
    const tolerance = 0.01; // 10ms tolerance for floating point
    if (Math.abs(lastKf.time - clip.duration) > tolerance || lastKf.value !== 0) return 0;

    // Find the last keyframe with opacity 1 (before the final 0)
    for (let i = opacityKeyframes.length - 2; i >= 0; i--) {
      const kf = opacityKeyframes[i];
      if (kf.value >= 0.99) {
        return clip.duration - kf.time;
      }
    }

    return 0;
  }, [clipMap, getClipKeyframes]);

  const handleFadeStart = useCallback(
    (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => {
      e.stopPropagation();
      e.preventDefault();

      const clip = clipMap.get(clipId);
      if (!clip) return;

      // Get existing fade duration
      const originalFadeDuration = edge === 'left'
        ? getFadeInDuration(clipId)
        : getFadeOutDuration(clipId);

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

        const clip = clipMap.get(fade.clipId);
        if (!clip) return;

        const updated = {
          ...fade,
          currentX: moveEvent.clientX,
        };
        setClipFade(updated);
        clipFadeRef.current = updated;

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
        const maxFade = clip.duration * 0.5;
        newFadeDuration = Math.max(0, Math.min(newFadeDuration, maxFade));

        // Update keyframes in real-time
        updateFadeKeyframes(fade.clipId, fade.edge, newFadeDuration, clip.duration);
      };

      const handleMouseUp = () => {
        setClipFade(null);
        clipFadeRef.current = null;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [clipMap, getFadeInDuration, getFadeOutDuration, pixelToTime]
  );

  // Helper function to update/create fade keyframes
  const updateFadeKeyframes = useCallback((
    clipId: string,
    edge: 'left' | 'right',
    fadeDuration: number,
    clipDuration: number
  ) => {
    const keyframes = getClipKeyframes(clipId);
    const opacityKeyframes = keyframes.filter(k => k.property === 'opacity');

    if (edge === 'left') {
      // Fade-in: keyframes at time 0 (opacity 0) and fadeDuration (opacity 1)

      // First, remove any existing fade-in keyframes (at time 0 and early times)
      // Keep any keyframes that are part of fade-out (at or after clip end - some buffer)
      const fadeOutBuffer = clipDuration * 0.5;
      const fadeInKeyframesToRemove = opacityKeyframes.filter(k => k.time < fadeOutBuffer);
      fadeInKeyframesToRemove.forEach(k => removeKeyframe(k.id));

      if (fadeDuration > 0.01) {
        // Add new fade-in keyframes
        addKeyframe(clipId, 'opacity', 0, 0);
        addKeyframe(clipId, 'opacity', 1, fadeDuration);
      }
    } else {
      // Fade-out: keyframes at (clipDuration - fadeDuration) (opacity 1) and clipDuration (opacity 0)

      // Remove any existing fade-out keyframes (near the end)
      const fadeInBuffer = clipDuration * 0.5;
      const fadeOutKeyframesToRemove = opacityKeyframes.filter(k => k.time > fadeInBuffer);
      fadeOutKeyframesToRemove.forEach(k => removeKeyframe(k.id));

      if (fadeDuration > 0.01) {
        // Add new fade-out keyframes
        const fadeStartTime = clipDuration - fadeDuration;
        addKeyframe(clipId, 'opacity', 1, fadeStartTime);
        addKeyframe(clipId, 'opacity', 0, clipDuration);
      }
    }
  }, [addKeyframe, removeKeyframe, getClipKeyframes]);

  return {
    clipFade,
    clipFadeRef,
    handleFadeStart,
    getFadeInDuration,
    getFadeOutDuration,
  };
}
