// Curve Editor Header component for Y-axis labels

import React, { useMemo } from 'react';
import type { AnimatableProperty, Keyframe } from '../../types';
import { CURVE_EDITOR_HEIGHT } from '../../stores/timeline/constants';

export interface CurveEditorHeaderProps {
  property: AnimatableProperty;
  keyframes: Keyframe[];
  onClose: () => void;
}

// Get value range for Y axis based on property type
function getPropertyRange(property: AnimatableProperty): { min: number; max: number; step: number } {
  if (property === 'opacity') {
    return { min: 0, max: 1, step: 0.1 };
  }
  if (property.startsWith('scale.')) {
    return { min: 0, max: 2, step: 0.1 };
  }
  if (property.startsWith('rotation.')) {
    return { min: -360, max: 360, step: 15 };
  }
  if (property.startsWith('position.')) {
    return { min: -1000, max: 1000, step: 100 };
  }
  // Effect properties
  return { min: -100, max: 100, step: 10 };
}

// Compute auto-range based on keyframe values
function computeAutoRange(keyframes: Keyframe[], property: AnimatableProperty): { min: number; max: number } {
  const defaultRange = getPropertyRange(property);

  if (keyframes.length === 0) {
    return defaultRange;
  }

  const values = keyframes.map(k => k.value);
  let min = Math.min(...values);
  let max = Math.max(...values);

  // Add padding
  const range = max - min;
  const padding = range > 0 ? range * 0.2 : defaultRange.step;
  min -= padding;
  max += padding;

  // Ensure we have a reasonable range
  if (max - min < defaultRange.step) {
    const mid = (min + max) / 2;
    min = mid - defaultRange.step;
    max = mid + defaultRange.step;
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
  const height = CURVE_EDITOR_HEIGHT;
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

  // Generate tick values
  const ticks = useMemo(() => {
    const tickValues: number[] = [];
    const range = valueRange.max - valueRange.min;
    const defaultRange = getPropertyRange(property);
    const step = defaultRange.step;

    // Calculate nice step size
    const numLines = Math.ceil(range / step);
    const adjustedStep = numLines > 10 ? step * 2 : step;

    for (let value = Math.ceil(valueRange.min / adjustedStep) * adjustedStep; value <= valueRange.max; value += adjustedStep) {
      tickValues.push(value);
    }

    return tickValues;
  }, [valueRange, property]);

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
