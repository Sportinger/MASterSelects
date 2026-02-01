// useClipFade - Fade-in/out handle dragging with real-time keyframe generation
// Creates opacity keyframes as the user drags the fade handles

import { useState, useCallback, useRef } from 'react';
import type { TimelineClip } from '../../../types';
import type { ClipFadeState } from '../types';

interface UseClipFadeProps {
  // Clip data
  clipMap: Map<string, TimelineClip>;

  // Keyframe actions
  addKeyframe: (clipId: string, property: 'opacity', value: number, time: number, easing?: string) => void;
  removeKeyframe: (keyframeId: string) => void;
  getClipKeyframes: (clipId: string) => Array<{
    id: string;
    clipId: string;
    time: number;
    property: string;
    value: number;
    easing: string;
    handleIn?: { x: number; y: number };
    handleOut?: { x: number; y: number };
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

      // Reset preserved easing for this drag session
      preservedEasingRef.current = {};

      // Preserve existing easing before any modifications
      const keyframes = getClipKeyframes(clipId);
      const opacityKeyframes = keyframes.filter(k => k.property === 'opacity').sort((a, b) => a.time - b.time);

      if (edge === 'left') {
        // Find the first keyframe (at time 0) for fade-in easing
        const firstKf = opacityKeyframes.find(k => k.time === 0);
        if (firstKf) {
          preservedEasingRef.current.fadeInEasing = firstKf.easing;
        }
      } else {
        // Find the second-to-last keyframe for fade-out easing
        const fadeOutStart = opacityKeyframes.find(k => k.value >= 0.99 && k.time > clip.duration * 0.5);
        if (fadeOutStart) {
          preservedEasingRef.current.fadeOutEasing = fadeOutStart.easing;
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

      let isFirstMove = true;

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
        updateFadeKeyframes(fade.clipId, fade.edge, newFadeDuration, clip.duration, isFirstMove);
        isFirstMove = false;
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
    [clipMap, getFadeInDuration, getFadeOutDuration, getClipKeyframes, pixelToTime, updateFadeKeyframes]
  );

  // Store preserved easing settings when starting fade drag
  const preservedEasingRef = useRef<{
    fadeInEasing?: string;
    fadeOutEasing?: string;
  }>({});

  // Helper function to update/create fade keyframes
  const updateFadeKeyframes = useCallback((
    clipId: string,
    edge: 'left' | 'right',
    fadeDuration: number,
    clipDuration: number,
    isFirstUpdate: boolean = false
  ) => {
    const keyframes = getClipKeyframes(clipId);
    const opacityKeyframes = keyframes.filter(k => k.property === 'opacity').sort((a, b) => a.time - b.time);

    if (edge === 'left') {
      // Fade-in: keyframes at time 0 (opacity 0) and fadeDuration (opacity 1)
      const fadeOutBuffer = clipDuration * 0.5;
      const fadeInKeyframes = opacityKeyframes.filter(k => k.time < fadeOutBuffer);

      // On first update, preserve the easing from existing keyframe at time 0
      if (isFirstUpdate && fadeInKeyframes.length > 0) {
        const firstKf = fadeInKeyframes.find(k => k.time === 0);
        if (firstKf) {
          preservedEasingRef.current.fadeInEasing = firstKf.easing;
        }
      }

      // Remove existing fade-in keyframes
      fadeInKeyframes.forEach(k => removeKeyframe(k.id));

      if (fadeDuration > 0.01) {
        // Add new fade-in keyframes with preserved easing
        const easing = preservedEasingRef.current.fadeInEasing || 'ease-out';
        addKeyframe(clipId, 'opacity', 0, 0, easing);
        addKeyframe(clipId, 'opacity', 1, fadeDuration, 'linear');
      }
    } else {
      // Fade-out: keyframes at (clipDuration - fadeDuration) (opacity 1) and clipDuration (opacity 0)
      const fadeInBuffer = clipDuration * 0.5;
      const fadeOutKeyframes = opacityKeyframes.filter(k => k.time > fadeInBuffer);

      // On first update, preserve the easing from the keyframe before the final one
      if (isFirstUpdate && fadeOutKeyframes.length >= 2) {
        const preLastKf = fadeOutKeyframes[fadeOutKeyframes.length - 2];
        if (preLastKf) {
          preservedEasingRef.current.fadeOutEasing = preLastKf.easing;
        }
      }

      // Remove existing fade-out keyframes
      fadeOutKeyframes.forEach(k => removeKeyframe(k.id));

      if (fadeDuration > 0.01) {
        // Add new fade-out keyframes with preserved easing
        const fadeStartTime = clipDuration - fadeDuration;
        const easing = preservedEasingRef.current.fadeOutEasing || 'ease-in';
        addKeyframe(clipId, 'opacity', 1, fadeStartTime, easing);
        addKeyframe(clipId, 'opacity', 0, clipDuration, 'linear');
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
