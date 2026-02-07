// Shared components for Properties Panel tabs
import { useRef, useCallback } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import type { AnimatableProperty, BlendMode } from '../../../types';
import { createEffectProperty } from '../../../types';

// EQ band parameter names
export const EQ_BAND_PARAMS = ['band31', 'band62', 'band125', 'band250', 'band500', 'band1k', 'band2k', 'band4k', 'band8k', 'band16k'];

// Organized by category like After Effects
export const BLEND_MODE_GROUPS: { label: string; modes: BlendMode[] }[] = [
  { label: 'Normal', modes: ['normal', 'dissolve', 'dancing-dissolve'] },
  { label: 'Darken', modes: ['darken', 'multiply', 'color-burn', 'classic-color-burn', 'linear-burn', 'darker-color'] },
  { label: 'Lighten', modes: ['add', 'lighten', 'screen', 'color-dodge', 'classic-color-dodge', 'linear-dodge', 'lighter-color'] },
  { label: 'Contrast', modes: ['overlay', 'soft-light', 'hard-light', 'linear-light', 'vivid-light', 'pin-light', 'hard-mix'] },
  { label: 'Inversion', modes: ['difference', 'classic-difference', 'exclusion', 'subtract', 'divide'] },
  { label: 'Component', modes: ['hue', 'saturation', 'color', 'luminosity'] },
  { label: 'Stencil', modes: ['stencil-alpha', 'stencil-luma', 'silhouette-alpha', 'silhouette-luma', 'alpha-add'] },
];

export const formatBlendModeName = (mode: BlendMode): string => {
  return mode.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

// Keyframe toggle button
interface KeyframeToggleProps {
  clipId: string;
  property: AnimatableProperty;
  value: number;
}

export function KeyframeToggle({ clipId, property, value }: KeyframeToggleProps) {
  // Use getState() for actions - they're stable and don't need subscriptions
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore.getState();
  const recording = isRecording(clipId, property);
  const hasKfs = hasKeyframes(clipId, property);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!recording && !hasKfs) {
      addKeyframe(clipId, property, value);
    }
    toggleKeyframeRecording(clipId, property);
  };

  return (
    <button
      className={`keyframe-toggle ${recording ? 'recording' : ''} ${hasKfs ? 'has-keyframes' : ''}`}
      onClick={handleClick}
      title={recording ? 'Stop recording keyframes' : hasKfs ? 'Enable keyframe recording' : 'Add keyframe'}
    >
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="13" r="7" />
        <line x1="12" y1="13" x2="12" y2="9" />
        <line x1="12" y1="2" x2="12" y2="5" />
        <line x1="9" y1="3" x2="15" y2="3" />
      </svg>
    </button>
  );
}

// Master keyframe toggle for Scale X and Y together
export function ScaleKeyframeToggle({ clipId, scaleX, scaleY }: { clipId: string; scaleX: number; scaleY: number }) {
  // Use getState() for actions - they're stable and don't need subscriptions
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore.getState();

  const xRecording = isRecording(clipId, 'scale.x');
  const yRecording = isRecording(clipId, 'scale.y');
  const xHasKfs = hasKeyframes(clipId, 'scale.x');
  const yHasKfs = hasKeyframes(clipId, 'scale.y');

  const anyRecording = xRecording || yRecording;
  const anyHasKfs = xHasKfs || yHasKfs;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!anyRecording && !anyHasKfs) {
      addKeyframe(clipId, 'scale.x', scaleX);
      addKeyframe(clipId, 'scale.y', scaleY);
    }
    toggleKeyframeRecording(clipId, 'scale.x');
    toggleKeyframeRecording(clipId, 'scale.y');
  };

  return (
    <button
      className={`keyframe-toggle ${anyRecording ? 'recording' : ''} ${anyHasKfs ? 'has-keyframes' : ''}`}
      onClick={handleClick}
      title={anyRecording ? 'Stop recording scale keyframes' : anyHasKfs ? 'Enable scale keyframe recording' : 'Add scale keyframes'}
    >
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="13" r="7" />
        <line x1="12" y1="13" x2="12" y2="9" />
        <line x1="12" y1="2" x2="12" y2="5" />
        <line x1="9" y1="3" x2="15" y2="3" />
      </svg>
    </button>
  );
}

// Precision slider with modifier key support
interface PrecisionSliderProps {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  defaultValue?: number;
  onDragStart?: () => void;
  onDragEnd?: () => void;
}

export function PrecisionSlider({ min, max, step, value, onChange, defaultValue, onDragStart, onDragEnd }: PrecisionSliderProps) {
  const sliderRef = useRef<HTMLDivElement>(null);
  const accumulatedDelta = useRef(0);
  const startValue = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    accumulatedDelta.current = 0;
    startValue.current = value;
    onDragStart?.();

    const element = sliderRef.current;
    if (element) element.requestPointerLock();

    const handleMouseMove = (e: MouseEvent) => {
      if (!sliderRef.current) return;
      const rect = sliderRef.current.getBoundingClientRect();
      const range = max - min;
      const pixelsPerUnit = rect.width / range;
      let speedMultiplier = 1;
      if (e.ctrlKey) speedMultiplier = 0.01;
      else if (e.shiftKey) speedMultiplier = 0.1;

      accumulatedDelta.current += e.movementX * speedMultiplier;
      const deltaValue = accumulatedDelta.current / pixelsPerUnit;
      const newValue = Math.max(min, Math.min(max, startValue.current + deltaValue));
      const preciseValue = Math.round(newValue * 1000000) / 1000000;
      onChange(preciseValue);
    };

    const handleMouseUp = () => {
      document.exitPointerLock();
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      onDragEnd?.();
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [value, min, max, step, onChange, onDragStart, onDragEnd]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    if (defaultValue !== undefined) onChange(defaultValue);
  }, [defaultValue, onChange]);

  const fillPercent = ((value - min) / (max - min)) * 100;

  return (
    <div
      ref={sliderRef}
      className="precision-slider"
      onMouseDown={handleMouseDown}
      onContextMenu={handleContextMenu}
      title={defaultValue !== undefined ? "Right-click to reset to default" : undefined}
    >
      <div className="precision-slider-track">
        <div className="precision-slider-fill" style={{ width: `${fillPercent}%` }} />
        <div className="precision-slider-thumb" style={{ left: `${fillPercent}%` }} />
      </div>
    </div>
  );
}

// Draggable number input
interface DraggableNumberProps {
  value: number;
  onChange: (value: number) => void;
  defaultValue?: number;
  sensitivity?: number;
  decimals?: number;
  suffix?: string;
  min?: number;
  max?: number;
  onDragStart?: () => void;
  onDragEnd?: () => void;
}

export function DraggableNumber({ value, onChange, defaultValue, sensitivity = 2, decimals = 2, suffix = '', min, max, onDragStart, onDragEnd }: DraggableNumberProps) {
  const inputRef = useRef<HTMLSpanElement>(null);
  const accumulatedDelta = useRef(0);
  const startValue = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    accumulatedDelta.current = 0;
    startValue.current = value;
    onDragStart?.();

    const element = inputRef.current;
    if (element) element.requestPointerLock();

    const handleMouseMove = (e: MouseEvent) => {
      let speedMultiplier = 1;
      if (e.ctrlKey) speedMultiplier = 0.01;
      else if (e.shiftKey) speedMultiplier = 0.1;

      accumulatedDelta.current += e.movementX * speedMultiplier;
      const deltaValue = accumulatedDelta.current / sensitivity;
      let newValue = startValue.current + deltaValue;
      // Clamp to min/max if specified
      if (min !== undefined) newValue = Math.max(min, newValue);
      if (max !== undefined) newValue = Math.min(max, newValue);
      const preciseValue = Math.round(newValue * Math.pow(10, decimals + 2)) / Math.pow(10, decimals + 2);
      onChange(preciseValue);
    };

    const handleMouseUp = () => {
      document.exitPointerLock();
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      onDragEnd?.();
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [value, sensitivity, decimals, onChange, min, max, onDragStart, onDragEnd]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    if (defaultValue !== undefined) onChange(defaultValue);
  }, [defaultValue, onChange]);

  return (
    <span
      ref={inputRef}
      className="draggable-number"
      onMouseDown={handleMouseDown}
      onContextMenu={handleContextMenu}
      title={defaultValue !== undefined ? "Drag to change, right-click to reset" : "Drag to change"}
    >
      {value.toFixed(decimals)}{suffix}
    </span>
  );
}

// Effect keyframe toggle
export function EffectKeyframeToggle({ clipId, effectId, paramName, value }: { clipId: string; effectId: string; paramName: string; value: number }) {
  // Use getState() for actions - they're stable and don't need subscriptions
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore.getState();
  const property = createEffectProperty(effectId, paramName);
  const recording = isRecording(clipId, property);
  const hasKfs = hasKeyframes(clipId, property);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!recording && !hasKfs) addKeyframe(clipId, property, value);
    toggleKeyframeRecording(clipId, property);
  };

  return (
    <button className={`keyframe-toggle ${recording ? 'recording' : ''} ${hasKfs ? 'has-keyframes' : ''}`}
      onClick={handleClick} title={recording ? 'Stop recording keyframes' : hasKfs ? 'Enable keyframe recording' : 'Add keyframe'}>
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="13" r="7" /><line x1="12" y1="13" x2="12" y2="9" />
        <line x1="12" y1="2" x2="12" y2="5" /><line x1="9" y1="3" x2="15" y2="3" />
      </svg>
    </button>
  );
}

// Master keyframe toggle for all 10 EQ bands at once
export function EQKeyframeToggle({ clipId, effectId, eqBands }: { clipId: string; effectId: string; eqBands: number[] }) {
  // Use getState() for actions - they're stable and don't need subscriptions
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore.getState();

  // Check if any band is recording or has keyframes
  const anyRecording = EQ_BAND_PARAMS.some(param => isRecording(clipId, createEffectProperty(effectId, param)));
  const anyHasKfs = EQ_BAND_PARAMS.some(param => hasKeyframes(clipId, createEffectProperty(effectId, param)));

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    // Toggle all 10 bands at once
    EQ_BAND_PARAMS.forEach((param, index) => {
      const property = createEffectProperty(effectId, param);
      if (!anyRecording && !anyHasKfs) {
        // Add keyframe for each band with current value
        addKeyframe(clipId, property, eqBands[index]);
      }
      toggleKeyframeRecording(clipId, property);
    });
  };

  return (
    <button className={`keyframe-toggle ${anyRecording ? 'recording' : ''} ${anyHasKfs ? 'has-keyframes' : ''}`}
      onClick={handleClick} title={anyRecording ? 'Stop recording EQ keyframes' : anyHasKfs ? 'Enable EQ keyframe recording' : 'Add EQ keyframes'}>
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="13" r="7" /><line x1="12" y1="13" x2="12" y2="9" />
        <line x1="12" y1="2" x2="12" y2="5" /><line x1="9" y1="3" x2="15" y2="3" />
      </svg>
    </button>
  );
}
