// Curve Editor Header component for Y-axis labels

import React, { useMemo } from 'react';
import type { AnimatableProperty, Keyframe } from '../../types';
import { useTimelineStore } from '../../stores/timeline';

export interface CurveEditorHeaderProps {
  property: AnimatableProperty;
  keyframes: Keyframe[];
  onClose: () => void;
}

// Get default range for a property type (used as fallback when no keyframes exist)
function getPropertyDefaults(property: AnimatableProperty): { min: number; max: number; fallbackPad: number } {
  if (property === 'opacity') {
    return { min: 0, max: 1, fallbackPad: 0.05 };
  }
  if (property.startsWith('scale.')) {
    return { min: 0, max: 2, fallbackPad: 0.05 };
  }
  if (property.startsWith('rotation.')) {
    return { min: -360, max: 360, fallbackPad: 5 };
  }
  if (property.startsWith('position.')) {
    return { min: -1000, max: 1000, fallbackPad: 10 };
  }
  // Effect properties
  return { min: -100, max: 100, fallbackPad: 5 };
}

// Calculate a "nice" step size for grid lines that produces clean label values
function niceStep(range: number, targetLines: number = 5): number {
  if (range <= 0) return 1;
  const roughStep = range / targetLines;
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const normalized = roughStep / magnitude;

  let nice: number;
  if (normalized <= 1) nice = 1;
  else if (normalized <= 2) nice = 2;
  else if (normalized <= 5) nice = 5;
  else nice = 10;

  return nice * magnitude;
}

// Compute auto-range that always fits tightly to actual keyframe values
function computeAutoRange(keyframes: Keyframe[], property: AnimatableProperty): { min: number; max: number } {
  const defaults = getPropertyDefaults(property);

  if (keyframes.length === 0) {
    return { min: defaults.min, max: defaults.max };
  }

  const values = keyframes.map(k => k.value);
  let min = Math.min(...values);
  let max = Math.max(...values);

  const range = max - min;

  if (range > 0) {
    // Add 10% padding so curve doesn't touch top/bottom edges
    const pad = range * 0.1;
    min -= pad;
    max += pad;
  } else {
    // All values identical — create a small range around the value
    const pad = Math.max(Math.abs(min) * 0.1, defaults.fallbackPad) || 1;
    min -= pad;
    max += pad;
  }

  return { min, max };
}

// Format value for display
function formatValue(value: number, property: AnimatableProperty): string {
  if (property === 'opacity') {
    return `${(value * 100).toFixed(0)}%`;
  }
  if (property.startsWith('scale.')) {
    return `${(value * 100).toFixed(0)}%`;
  }
  if (property.startsWith('rotation.')) {
    return `${value.toFixed(0)}°`;
  }
  return value.toFixed(0);
}

export const CurveEditorHeader: React.FC<CurveEditorHeaderProps> = ({
  property,
  keyframes,
  onClose,
}) => {
  const height = useTimelineStore(s => s.curveEditorHeight);
  const padding = { top: 20, bottom: 20 };

  // Compute value range
  const valueRange = useMemo(() =>
    computeAutoRange(keyframes, property),
    [keyframes, property]
  );

  // Convert value to Y position
  const valueToY = (value: number): number => {
    const range = valueRange.max - valueRange.min;
    const normalized = (value - valueRange.min) / range;
    return height - padding.bottom - normalized * (height - padding.top - padding.bottom);
  };

  // Generate tick values with adaptive step size
  const ticks = useMemo(() => {
    const tickValues: number[] = [];
    const range = valueRange.max - valueRange.min;
    const step = niceStep(range);

    for (let value = Math.ceil(valueRange.min / step) * step; value <= valueRange.max; value += step) {
      tickValues.push(value);
    }

    return tickValues;
  }, [valueRange]);

  return (
    <div className="curve-editor-header" style={{ height }}>
      <button
        className="curve-editor-close-btn"
        onClick={onClose}
        title="Close curve editor"
      >
        ×
      </button>
      <div className="curve-editor-y-axis">
        {ticks.map((value, i) => (
          <div
            key={i}
            className="curve-editor-tick"
            style={{ top: valueToY(value) }}
          >
            <span className="curve-editor-tick-label">
              {formatValue(value, property)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CurveEditorHeader;
