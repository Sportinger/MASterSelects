// useTimelineKeyboard - Global keyboard shortcuts for timeline
// Extracted from Timeline.tsx for better maintainability

import { useEffect } from 'react';
import type { TimelineClip, ClipTransform } from '../../../types';
import type { Composition } from '../../../stores/mediaStore';
import { ALL_BLEND_MODES } from '../constants';

interface UseTimelineKeyboardProps {
  // Playback
  isPlaying: boolean;
  play: () => void;
  pause: () => void;

  // In/Out points
  setInPointAtPlayhead: () => void;
  setOutPointAtPlayhead: () => void;
  clearInOut: () => void;
  toggleLoopPlayback: () => void;

  // Selection
  selectedClipIds: Set<string>;
  selectedKeyframeIds: Set<string>;

  // Clip operations
  removeClip: (id: string) => void;
  removeKeyframe: (id: string) => void;
  splitClipAtPlayhead: () => void;
  updateClipTransform: (id: string, transform: Partial<ClipTransform>) => void;

  // Tool mode
  toolMode: 'select' | 'cut';
  toggleCutTool: () => void;

  // Clip lookup
  clipMap: Map<string, TimelineClip>;

  // Playhead navigation
  activeComposition: Composition | null;
  playheadPosition: number;
  duration: number;
  setPlayheadPosition: (time: number) => void;

  // Markers
  addMarker?: (time: number) => string;
}

export function useTimelineKeyboard({
  isPlaying,
  play,
  pause,
  setInPointAtPlayhead,
  setOutPointAtPlayhead,
  clearInOut,
  toggleLoopPlayback,
  selectedClipIds,
  selectedKeyframeIds,
  removeClip,
  removeKeyframe,
  splitClipAtPlayhead,
  updateClipTransform,
  toolMode,
  toggleCutTool,
  clipMap,
  activeComposition,
  playheadPosition,
  duration,
  setPlayheadPosition,
  addMarker,
}: UseTimelineKeyboardProps): void {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not typing in a text input
      const isTextInput =
        e.target instanceof HTMLTextAreaElement ||
        (e.target instanceof HTMLInputElement &&
          e.target.type !== 'range' &&
          e.target.type !== 'checkbox' &&
          e.target.type !== 'radio');

      if (isTextInput) {
        return;
      }

      // Space: toggle play/pause (also blur any focused slider/checkbox)
      if (e.code === 'Space' || e.key === ' ') {
        if (e.target instanceof HTMLInputElement) {
          e.target.blur();
        }
        e.preventDefault();
        if (isPlaying) {
          pause();
        } else {
          play();
        }
        return;
      }

      // I: set In point at playhead
      if (e.key === 'i' || e.key === 'I') {
        e.preventDefault();
        setInPointAtPlayhead();
        return;
      }

      // O: set Out point at playhead
      if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        setOutPointAtPlayhead();
        return;
      }

      // X: clear In/Out points
      if (e.key === 'x' || e.key === 'X') {
        e.preventDefault();
        clearInOut();
        return;
      }

      // L: toggle loop playback
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        toggleLoopPlayback();
        return;
      }

      // Delete/Backspace: remove selected keyframes first, then clips
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        // First check if any keyframes are selected
        if (selectedKeyframeIds.size > 0) {
          // Remove all selected keyframes
          [...selectedKeyframeIds].forEach(keyframeId => removeKeyframe(keyframeId));
          return;
        }
        // Otherwise remove selected clips
        if (selectedClipIds.size > 0) {
          [...selectedClipIds].forEach(clipId => removeClip(clipId));
        }
        return;
      }

      // C: Toggle cut tool mode / Shift+C: Split clip at playhead position
      if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        if (e.shiftKey) {
          // Shift+C: Split clip at playhead position (legacy behavior)
          splitClipAtPlayhead();
        } else {
          // C: Toggle cut tool mode
          toggleCutTool();
        }
        return;
      }

      // Escape: Exit cut tool mode (return to select)
      if (e.key === 'Escape' && toolMode === 'cut') {
        e.preventDefault();
        toggleCutTool();
        return;
      }

      // Shift + "+": Cycle through blend modes (forward)
      // Shift + "-": Cycle through blend modes (backward)
      // Supports: numpad +/-, Shift+=/- on main keyboard
      const isNumpadPlus = e.code === 'NumpadAdd';
      const isNumpadMinus = e.code === 'NumpadSubtract';
      const isMainPlus = e.shiftKey && (e.key === '+' || e.key === '=');
      const isMainMinus = e.shiftKey && (e.key === '-' || e.key === '_' || e.code === 'Minus');
      const isPlus = isNumpadPlus || isMainPlus;
      const isMinus = isNumpadMinus || isMainMinus;

      if (isPlus || isMinus) {
        e.preventDefault();
        const firstSelectedId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;
        if (!firstSelectedId) return;

        const clip = clipMap.get(firstSelectedId);
        if (!clip) return;

        const currentMode = clip.transform?.blendMode || 'normal';
        const currentIndex = ALL_BLEND_MODES.indexOf(currentMode);
        const direction = isPlus ? 1 : -1;
        const nextIndex =
          (currentIndex + direction + ALL_BLEND_MODES.length) %
          ALL_BLEND_MODES.length;
        const nextMode = ALL_BLEND_MODES[nextIndex];

        // Apply to all selected clips
        [...selectedClipIds].forEach(clipId => {
          updateClipTransform(clipId, { blendMode: nextMode });
        });
        return;
      }

      // Arrow Left: Move playhead one frame backward
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (activeComposition) {
          const frameDuration = 1 / activeComposition.frameRate;
          const newPosition = Math.max(0, playheadPosition - frameDuration);
          setPlayheadPosition(newPosition);
        }
        return;
      }

      // Arrow Right: Move playhead one frame forward
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (activeComposition) {
          const frameDuration = 1 / activeComposition.frameRate;
          const newPosition = Math.min(duration, playheadPosition + frameDuration);
          setPlayheadPosition(newPosition);
        }
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    isPlaying,
    play,
    pause,
    setInPointAtPlayhead,
    setOutPointAtPlayhead,
    clearInOut,
    toggleLoopPlayback,
    selectedClipIds,
    selectedKeyframeIds,
    removeClip,
    removeKeyframe,
    splitClipAtPlayhead,
    clipMap,
    updateClipTransform,
    toolMode,
    toggleCutTool,
    activeComposition,
    playheadPosition,
    duration,
    setPlayheadPosition,
  ]);
}
