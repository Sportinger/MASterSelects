// useClipFade - Fade-in/out handle dragging with real-time keyframe generation
// Creates opacity keyframes as the user drags the fade handles
// Preserves existing bezier handles when adjusting fade duration

import { useState, useCallback, useRef } from 'react';
import type { TimelineClip } from '../../../types';
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
  // Clip data
  clipMap: Map<string, TimelineClip>;

  // Keyframe actions
  addKeyframe: (clipId: string, property: 'opacity', value: number, time: number, easing?: string) => void;
  removeKeyframe: (keyframeId: string) => void;
  moveKeyframe: (keyframeId: string, newTime: number) => void;
  getClipKeyframes: (clipId: string) => KeyframeData[];

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
  moveKeyframe,
  getClipKeyframes,
  pixelToTime,
}: UseClipFadeProps): UseClipFadeReturn {
  const [clipFade, setClipFade] = useState<ClipFadeState | null>(null);
  const clipFadeRef = useRef<ClipFadeState | null>(clipFade);
  clipFadeRef.current = clipFade;

  // Store the keyframe IDs we're working with during a drag
  const fadeKeyframeIdsRef = useRef<{
    startKeyframeId?: string;  // The keyframe at start/end of fade (opacity 0)
    endKeyframeId?: string;    // The keyframe at fade point (opacity 1)
  }>({});

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

      // Find existing keyframes for this fade
      const keyframes = getClipKeyframes(clipId);
      const opacityKeyframes = keyframes.filter(k => k.property === 'opacity').sort((a, b) => a.time - b.time);

      // Reset keyframe IDs for this drag session
      fadeKeyframeIdsRef.current = {};

      if (edge === 'left') {
        // Fade-in: Look for keyframe at 0 (opacity 0) and next one (opacity 1)
        const startKf = opacityKeyframes.find(k => k.time === 0 && k.value === 0);
        const endKf = opacityKeyframes.find(k => k.value >= 0.99 && k.time > 0 && k.time < clip.duration * 0.5);

        if (startKf && endKf) {
          fadeKeyframeIdsRef.current.startKeyframeId = startKf.id;
          fadeKeyframeIdsRef.current.endKeyframeId = endKf.id;
        }
      } else {
        // Fade-out: Look for keyframe at end (opacity 0) and previous one (opacity 1)
        const endKf = opacityKeyframes.find(k => Math.abs(k.time - clip.duration) < 0.01 && k.value === 0);
        const startKf = opacityKeyframes.find(k => k.value >= 0.99 && k.time > clip.duration * 0.5);

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
        const maxFade = currentClip.duration * 0.5;
        newFadeDuration = Math.max(0, Math.min(newFadeDuration, maxFade));

        // Update keyframes
        const { startKeyframeId, endKeyframeId } = fadeKeyframeIdsRef.current;

        if (fade.edge === 'left') {
          // Fade-in: move the end keyframe (opacity 1) to new position
          if (startKeyframeId && endKeyframeId) {
            // Just move the existing keyframe - preserves all bezier handles
            moveKeyframe(endKeyframeId, newFadeDuration);
          } else if (newFadeDuration > 0.01) {
            // No existing fade - create new keyframes
            addKeyframe(fade.clipId, 'opacity', 0, 0, 'ease-out');
            addKeyframe(fade.clipId, 'opacity', 1, newFadeDuration, 'linear');

            // Get the newly created keyframe IDs for future moves
            const newKeyframes = getClipKeyframes(fade.clipId).filter(k => k.property === 'opacity');
            const newStartKf = newKeyframes.find(k => k.time === 0 && k.value === 0);
            const newEndKf = newKeyframes.find(k => k.value >= 0.99 && k.time > 0);
            if (newStartKf && newEndKf) {
              fadeKeyframeIdsRef.current.startKeyframeId = newStartKf.id;
              fadeKeyframeIdsRef.current.endKeyframeId = newEndKf.id;
            }
          }
        } else {
          // Fade-out: move the start keyframe (opacity 1) to new position
          if (startKeyframeId && endKeyframeId) {
            // Just move the existing keyframe - preserves all bezier handles
            const fadeStartTime = currentClip.duration - newFadeDuration;
            moveKeyframe(startKeyframeId, fadeStartTime);
          } else if (newFadeDuration > 0.01) {
            // No existing fade - create new keyframes
            const fadeStartTime = currentClip.duration - newFadeDuration;
            addKeyframe(fade.clipId, 'opacity', 1, fadeStartTime, 'ease-in');
            addKeyframe(fade.clipId, 'opacity', 0, currentClip.duration, 'linear');

            // Get the newly created keyframe IDs for future moves
            const newKeyframes = getClipKeyframes(fade.clipId).filter(k => k.property === 'opacity');
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
    [clipMap, getFadeInDuration, getFadeOutDuration, getClipKeyframes, pixelToTime, addKeyframe, moveKeyframe, removeKeyframe]
  );

  return {
    clipFade,
    clipFadeRef,
    handleFadeStart,
    getFadeInDuration,
    getFadeOutDuration,
  };
}
