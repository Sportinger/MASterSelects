// Clip Properties Panel - Shows transform controls for selected timeline clip

import { useRef, useCallback } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import type { BlendMode } from '../types';

const BLEND_MODES: BlendMode[] = [
  'normal',
  'add',
  'multiply',
  'screen',
  'overlay',
  'difference',
];

// Precision slider with modifier key support
// Shift = half speed, Ctrl = super slow (10x slower)
interface PrecisionSliderProps {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}

function PrecisionSlider({ min, max, step, value, onChange }: PrecisionSliderProps) {
  const sliderRef = useRef<HTMLDivElement>(null);
  const accumulatedDelta = useRef(0);
  const startValue = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    accumulatedDelta.current = 0;
    startValue.current = value;

    // Request pointer lock for infinite dragging
    const element = sliderRef.current;
    if (element) {
      element.requestPointerLock();
    }

    const handleMouseMove = (e: MouseEvent) => {
      if (!sliderRef.current) return;

      const rect = sliderRef.current.getBoundingClientRect();
      const range = max - min;
      const pixelsPerUnit = rect.width / range;

      // Calculate speed multiplier based on modifier keys
      let speedMultiplier = 1;
      if (e.ctrlKey) {
        speedMultiplier = 0.01; // Ultra fine (1%)
      } else if (e.shiftKey) {
        speedMultiplier = 0.1; // Slow (10%)
      }

      // Use movementX for pointer lock (raw delta, not position)
      accumulatedDelta.current += e.movementX * speedMultiplier;
      const deltaValue = accumulatedDelta.current / pixelsPerUnit;
      const newValue = Math.max(min, Math.min(max, startValue.current + deltaValue));

      // Use full float precision (round to 6 decimal places to avoid float errors)
      const preciseValue = Math.round(newValue * 1000000) / 1000000;
      onChange(preciseValue);
    };

    const handleMouseUp = () => {
      // Exit pointer lock
      document.exitPointerLock();
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [value, min, max, step, onChange]);

  // Calculate fill percentage
  const fillPercent = ((value - min) / (max - min)) * 100;

  return (
    <div
      ref={sliderRef}
      className="precision-slider"
      onMouseDown={handleMouseDown}
    >
      <div className="precision-slider-track">
        <div
          className="precision-slider-fill"
          style={{ width: `${fillPercent}%` }}
        />
        <div
          className="precision-slider-thumb"
          style={{ left: `${fillPercent}%` }}
        />
      </div>
    </div>
  );
}

export function ClipPropertiesPanel() {
  const { clips, selectedClipId, updateClipTransform } = useTimelineStore();
  const selectedClip = clips.find(c => c.id === selectedClipId);

  if (!selectedClip) {
    return (
      <div className="clip-properties-panel">
        <div className="panel-header">
          <h3>Properties</h3>
        </div>
        <div className="panel-empty">
          <p>Select a clip to edit properties</p>
        </div>
      </div>
    );
  }

  const { transform } = selectedClip;

  const updateTransform = (updates: Parameters<typeof updateClipTransform>[1]) => {
    updateClipTransform(selectedClip.id, updates);
  };

  // Calculate uniform scale (average of X and Y)
  const uniformScale = (transform.scale.x + transform.scale.y) / 2;

  const handleUniformScaleChange = (value: number) => {
    updateTransform({ scale: { x: value, y: value } });
  };

  return (
    <div className="clip-properties-panel">
      <div className="panel-header">
        <h3>{selectedClip.name}</h3>
      </div>

      <div className="properties-content">
        {/* Blend Mode & Opacity */}
        <div className="properties-section">
          <h4>Appearance</h4>
          <div className="control-row">
            <label>Blend Mode</label>
            <select
              value={transform.blendMode}
              onChange={(e) => updateTransform({ blendMode: e.target.value as BlendMode })}
            >
              {BLEND_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </option>
              ))}
            </select>
          </div>
          <div className="control-row">
            <label>Opacity</label>
            <PrecisionSlider
              min={0}
              max={1}
              step={0.0001}
              value={transform.opacity}
              onChange={(v) => updateTransform({ opacity: v })}
            />
            <span className="value">{(transform.opacity * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Scale */}
        <div className="properties-section">
          <h4>Scale</h4>
          <div className="control-row">
            <label>Uniform</label>
            <PrecisionSlider
              min={0.1}
              max={3}
              step={0.0001}
              value={uniformScale}
              onChange={handleUniformScaleChange}
            />
            <span className="value">{uniformScale.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <label>X</label>
            <PrecisionSlider
              min={0.1}
              max={3}
              step={0.0001}
              value={transform.scale.x}
              onChange={(v) => updateTransform({ scale: { x: v } })}
            />
            <span className="value">{transform.scale.x.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <label>Y</label>
            <PrecisionSlider
              min={0.1}
              max={3}
              step={0.0001}
              value={transform.scale.y}
              onChange={(v) => updateTransform({ scale: { y: v } })}
            />
            <span className="value">{transform.scale.y.toFixed(3)}</span>
          </div>
        </div>

        {/* Position */}
        <div className="properties-section">
          <h4>Position</h4>
          <div className="control-row">
            <label>X</label>
            <PrecisionSlider
              min={-1}
              max={1}
              step={0.0001}
              value={transform.position.x}
              onChange={(v) => updateTransform({ position: { x: v } })}
            />
            <span className="value">{transform.position.x.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <label>Y</label>
            <PrecisionSlider
              min={-1}
              max={1}
              step={0.0001}
              value={transform.position.y}
              onChange={(v) => updateTransform({ position: { y: v } })}
            />
            <span className="value">{transform.position.y.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <label>Z</label>
            <PrecisionSlider
              min={-1}
              max={1}
              step={0.0001}
              value={transform.position.z}
              onChange={(v) => updateTransform({ position: { z: v } })}
            />
            <span className="value">{transform.position.z.toFixed(3)}</span>
          </div>
        </div>

        {/* Rotation */}
        <div className="properties-section">
          <h4>Rotation</h4>
          <div className="control-row">
            <label>X</label>
            <PrecisionSlider
              min={-180}
              max={180}
              step={0.01}
              value={transform.rotation.x}
              onChange={(v) => updateTransform({ rotation: { x: v } })}
            />
            <span className="value">{transform.rotation.x.toFixed(1)}°</span>
          </div>
          <div className="control-row">
            <label>Y</label>
            <PrecisionSlider
              min={-180}
              max={180}
              step={0.01}
              value={transform.rotation.y}
              onChange={(v) => updateTransform({ rotation: { y: v } })}
            />
            <span className="value">{transform.rotation.y.toFixed(1)}°</span>
          </div>
          <div className="control-row">
            <label>Z</label>
            <PrecisionSlider
              min={-180}
              max={180}
              step={0.01}
              value={transform.rotation.z}
              onChange={(v) => updateTransform({ rotation: { z: v } })}
            />
            <span className="value">{transform.rotation.z.toFixed(1)}°</span>
          </div>
        </div>

        {/* Reset Button */}
        <div className="properties-actions">
          <button
            className="btn btn-sm"
            onClick={() => updateTransform({
              opacity: 1,
              blendMode: 'normal',
              position: { x: 0, y: 0, z: 0 },
              scale: { x: 1, y: 1 },
              rotation: { x: 0, y: 0, z: 0 },
            })}
          >
            Reset All
          </button>
        </div>
      </div>
    </div>
  );
}
