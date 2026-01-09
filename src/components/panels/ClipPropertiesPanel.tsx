// Clip Properties Panel - Shows transform controls for selected timeline clip

import { useRef, useCallback, useState } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { BlendMode, AnimatableProperty, MaskMode, ClipMask } from '../../types';

// Organized by category like After Effects
const BLEND_MODE_GROUPS: { label: string; modes: BlendMode[] }[] = [
  {
    label: 'Normal',
    modes: ['normal', 'dissolve', 'dancing-dissolve'],
  },
  {
    label: 'Darken',
    modes: ['darken', 'multiply', 'color-burn', 'classic-color-burn', 'linear-burn', 'darker-color'],
  },
  {
    label: 'Lighten',
    modes: ['add', 'lighten', 'screen', 'color-dodge', 'classic-color-dodge', 'linear-dodge', 'lighter-color'],
  },
  {
    label: 'Contrast',
    modes: ['overlay', 'soft-light', 'hard-light', 'linear-light', 'vivid-light', 'pin-light', 'hard-mix'],
  },
  {
    label: 'Inversion',
    modes: ['difference', 'classic-difference', 'exclusion', 'subtract', 'divide'],
  },
  {
    label: 'Component',
    modes: ['hue', 'saturation', 'color', 'luminosity'],
  },
  {
    label: 'Stencil',
    modes: ['stencil-alpha', 'stencil-luma', 'silhouette-alpha', 'silhouette-luma', 'alpha-add'],
  },
];

// Format blend mode name for display
const formatBlendModeName = (mode: BlendMode): string => {
  return mode
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// Keyframe toggle button component
interface KeyframeToggleProps {
  clipId: string;
  property: AnimatableProperty;
  value: number;
}

function KeyframeToggle({ clipId, property, value }: KeyframeToggleProps) {
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore();

  const recording = isRecording(clipId, property);
  const hasKfs = hasKeyframes(clipId, property);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();

    if (!recording && !hasKfs) {
      // Turning ON for first time - create initial keyframe
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

// Precision slider with modifier key support
// Shift = half speed, Ctrl = super slow (10x slower)
// Right-click to reset to default value
interface PrecisionSliderProps {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  defaultValue?: number;
}

function PrecisionSlider({ min, max, step, value, onChange, defaultValue }: PrecisionSliderProps) {
  const sliderRef = useRef<HTMLDivElement>(null);
  const accumulatedDelta = useRef(0);
  const startValue = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only handle left click
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

  // Handle right-click to reset to default
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    if (defaultValue !== undefined) {
      onChange(defaultValue);
    }
  }, [defaultValue, onChange]);

  // Calculate fill percentage
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

// Draggable number input - no caps, supports negative values
// Drag left/right to change value, right-click to reset
interface DraggableNumberProps {
  value: number;
  onChange: (value: number) => void;
  defaultValue?: number;
  sensitivity?: number; // How many pixels per unit (default: 2)
  decimals?: number; // Number of decimal places to display (default: 2)
  suffix?: string; // Optional suffix like "px" or "%"
}

function DraggableNumber({ value, onChange, defaultValue, sensitivity = 2, decimals = 2, suffix = '' }: DraggableNumberProps) {
  const inputRef = useRef<HTMLSpanElement>(null);
  const accumulatedDelta = useRef(0);
  const startValue = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only handle left click
    e.preventDefault();
    accumulatedDelta.current = 0;
    startValue.current = value;

    // Request pointer lock for infinite dragging
    const element = inputRef.current;
    if (element) {
      element.requestPointerLock();
    }

    const handleMouseMove = (e: MouseEvent) => {
      // Calculate speed multiplier based on modifier keys
      let speedMultiplier = 1;
      if (e.ctrlKey) {
        speedMultiplier = 0.01; // Ultra fine (1%)
      } else if (e.shiftKey) {
        speedMultiplier = 0.1; // Slow (10%)
      }

      // Use movementX for pointer lock (raw delta, not position)
      accumulatedDelta.current += e.movementX * speedMultiplier;
      const deltaValue = accumulatedDelta.current / sensitivity;
      const newValue = startValue.current + deltaValue;

      // Round to avoid float errors
      const preciseValue = Math.round(newValue * Math.pow(10, decimals + 2)) / Math.pow(10, decimals + 2);
      onChange(preciseValue);
    };

    const handleMouseUp = () => {
      document.exitPointerLock();
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [value, sensitivity, decimals, onChange]);

  // Handle right-click to reset to default
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    if (defaultValue !== undefined) {
      onChange(defaultValue);
    }
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

// Mask mode options
const MASK_MODES: { value: MaskMode; label: string }[] = [
  { value: 'add', label: 'Add' },
  { value: 'subtract', label: 'Subtract' },
  { value: 'intersect', label: 'Intersect' },
];

// Individual mask item component
interface MaskItemProps {
  clipId: string;
  mask: ClipMask;
  isActive: boolean;
  onSelect: () => void;
}

function MaskItem({ clipId, mask, isActive, onSelect }: MaskItemProps) {
  const { updateMask, removeMask, setActiveMask, setMaskEditMode } = useTimelineStore();
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(mask.name);

  const handleNameDoubleClick = () => {
    setIsEditing(true);
    setEditName(mask.name);
  };

  const handleNameChange = () => {
    if (editName.trim()) {
      updateMask(clipId, mask.id, { name: editName.trim() });
    }
    setIsEditing(false);
  };

  const handleEditMask = () => {
    onSelect();
    setActiveMask(clipId, mask.id);
    setMaskEditMode('editing');
  };

  return (
    <div className={`mask-item ${isActive ? 'active' : ''} ${mask.expanded ? 'expanded' : ''}`}>
      <div className="mask-item-header" onClick={onSelect}>
        <button
          className="mask-expand-btn"
          onClick={(e) => {
            e.stopPropagation();
            updateMask(clipId, mask.id, { expanded: !mask.expanded });
          }}
        >
          {mask.expanded ? '\u25BC' : '\u25B6'}
        </button>

        {isEditing ? (
          <input
            type="text"
            className="mask-name-input"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={handleNameChange}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleNameChange();
              if (e.key === 'Escape') setIsEditing(false);
            }}
            autoFocus
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span className="mask-name" onDoubleClick={handleNameDoubleClick}>
            {mask.name}
          </span>
        )}

        <select
          className="mask-mode-select"
          value={mask.mode}
          onChange={(e) => updateMask(clipId, mask.id, { mode: e.target.value as MaskMode })}
          onClick={(e) => e.stopPropagation()}
        >
          {MASK_MODES.map(m => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>

        <button
          className="mask-visible-btn"
          onClick={(e) => {
            e.stopPropagation();
            updateMask(clipId, mask.id, { visible: !mask.visible });
          }}
          title={mask.visible ? "Hide mask outline" : "Show mask outline"}
          style={{ opacity: mask.visible ? 1 : 0.5 }}
        >
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            {mask.visible ? (
              <>
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </>
            ) : (
              <>
                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                <line x1="1" y1="1" x2="23" y2="23" />
              </>
            )}
          </svg>
        </button>

        <button
          className="mask-edit-btn"
          onClick={(e) => {
            e.stopPropagation();
            handleEditMask();
          }}
          title="Edit mask path"
        >
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
          </svg>
        </button>

        <button
          className="mask-delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            removeMask(clipId, mask.id);
          }}
          title="Delete mask"
        >
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {mask.expanded && (
        <div className="mask-item-properties">
          <div className="control-row">
            <label>Opacity</label>
            <DraggableNumber
              value={mask.opacity * 100}
              onChange={(v) => updateMask(clipId, mask.id, { opacity: v / 100 })}
              defaultValue={100}
              sensitivity={1}
              decimals={0}
              suffix="%"
            />
          </div>

          <div className="control-row">
            <label>Feather</label>
            <DraggableNumber
              value={mask.feather}
              onChange={(v) => updateMask(clipId, mask.id, { feather: v })}
              defaultValue={0}
              sensitivity={1}
              decimals={1}
              suffix="px"
            />
          </div>

          <div className="control-row">
            <label>Quality</label>
            <DraggableNumber
              value={mask.featherQuality ?? 50}
              onChange={(v) => updateMask(clipId, mask.id, { featherQuality: Math.max(1, Math.round(v)) })}
              defaultValue={50}
              sensitivity={1}
              decimals={0}
            />
          </div>

          <div className="control-row">
            <label>Position X</label>
            <DraggableNumber
              value={mask.position.x}
              onChange={(v) => updateMask(clipId, mask.id, { position: { ...mask.position, x: v } })}
              defaultValue={0}
              sensitivity={100}
              decimals={3}
            />
          </div>

          <div className="control-row">
            <label>Position Y</label>
            <DraggableNumber
              value={mask.position.y}
              onChange={(v) => updateMask(clipId, mask.id, { position: { ...mask.position, y: v } })}
              defaultValue={0}
              sensitivity={100}
              decimals={3}
            />
          </div>

          <div className="control-row">
            <label>Inverted</label>
            <input
              type="checkbox"
              checked={mask.inverted}
              onChange={(e) => updateMask(clipId, mask.id, { inverted: e.target.checked })}
            />
          </div>

          <div className="mask-info">
            {mask.vertices.length} vertices | {mask.closed ? 'Closed' : 'Open'}
          </div>
        </div>
      )}
    </div>
  );
}

export function ClipPropertiesPanel() {
  const {
    clips,
    selectedClipIds,
    setPropertyValue,
    playheadPosition,
    getInterpolatedTransform,
    addRectangleMask,
    addEllipseMask,
    activeMaskId,
    setActiveMask,
    maskEditMode,
    setMaskEditMode,
    updateClipTransform,
  } = useTimelineStore();
  // Get first selected clip for properties panel
  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;
  const selectedClip = clips.find(c => c.id === selectedClipId);
  const [showMaskMenu, setShowMaskMenu] = useState(false);

  // Handle starting a shape drawing mode
  const handleStartDrawMode = (mode: 'drawingRect' | 'drawingEllipse' | 'drawingPen') => {
    if (selectedClip) {
      setMaskEditMode(mode);
    }
  };

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

  // Get interpolated transform at current playhead position
  const clipLocalTime = playheadPosition - selectedClip.startTime;
  const transform = getInterpolatedTransform(selectedClip.id, clipLocalTime);

  const handlePropertyChange = (property: AnimatableProperty, value: number) => {
    setPropertyValue(selectedClip.id, property, value);
  };

  // Calculate uniform scale (average of X and Y)
  const uniformScale = (transform.scale.x + transform.scale.y) / 2;

  const handleUniformScaleChange = (value: number) => {
    handlePropertyChange('scale.x', value);
    handlePropertyChange('scale.y', value);
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
              onChange={(e) => {
                // Blend mode is not animatable, update directly
                updateClipTransform(selectedClip.id, {
                  blendMode: e.target.value as BlendMode
                });
              }}
            >
              {BLEND_MODE_GROUPS.map((group) => (
                <optgroup key={group.label} label={group.label}>
                  {group.modes.map((mode) => (
                    <option key={mode} value={mode}>
                      {formatBlendModeName(mode)}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="opacity" value={transform.opacity} />
            <label>Opacity</label>
            <PrecisionSlider
              min={0}
              max={1}
              step={0.0001}
              value={transform.opacity}
              onChange={(v) => handlePropertyChange('opacity', v)}
              defaultValue={1}
            />
            <span className="value">{(transform.opacity * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Scale */}
        <div className="properties-section">
          <h4>Scale</h4>
          <div className="control-row">
            <span className="keyframe-toggle-placeholder" />
            <label>Uniform</label>
            <PrecisionSlider
              min={0.1}
              max={3}
              step={0.0001}
              value={uniformScale}
              onChange={handleUniformScaleChange}
              defaultValue={1}
            />
            <span className="value">{uniformScale.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="scale.x" value={transform.scale.x} />
            <label>X</label>
            <PrecisionSlider
              min={0.1}
              max={3}
              step={0.0001}
              value={transform.scale.x}
              onChange={(v) => handlePropertyChange('scale.x', v)}
              defaultValue={1}
            />
            <span className="value">{transform.scale.x.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="scale.y" value={transform.scale.y} />
            <label>Y</label>
            <PrecisionSlider
              min={0.1}
              max={3}
              step={0.0001}
              value={transform.scale.y}
              onChange={(v) => handlePropertyChange('scale.y', v)}
              defaultValue={1}
            />
            <span className="value">{transform.scale.y.toFixed(3)}</span>
          </div>
        </div>

        {/* Position */}
        <div className="properties-section">
          <h4>Position</h4>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="position.x" value={transform.position.x} />
            <label>X</label>
            <PrecisionSlider
              min={-1}
              max={1}
              step={0.0001}
              value={transform.position.x}
              onChange={(v) => handlePropertyChange('position.x', v)}
              defaultValue={0}
            />
            <span className="value">{transform.position.x.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="position.y" value={transform.position.y} />
            <label>Y</label>
            <PrecisionSlider
              min={-1}
              max={1}
              step={0.0001}
              value={transform.position.y}
              onChange={(v) => handlePropertyChange('position.y', v)}
              defaultValue={0}
            />
            <span className="value">{transform.position.y.toFixed(3)}</span>
          </div>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="position.z" value={transform.position.z} />
            <label>Z</label>
            <PrecisionSlider
              min={-1}
              max={1}
              step={0.0001}
              value={transform.position.z}
              onChange={(v) => handlePropertyChange('position.z', v)}
              defaultValue={0}
            />
            <span className="value">{transform.position.z.toFixed(3)}</span>
          </div>
        </div>

        {/* Rotation */}
        <div className="properties-section">
          <h4>Rotation</h4>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="rotation.x" value={transform.rotation.x} />
            <label>X</label>
            <PrecisionSlider
              min={-180}
              max={180}
              step={0.01}
              value={transform.rotation.x}
              onChange={(v) => handlePropertyChange('rotation.x', v)}
              defaultValue={0}
            />
            <span className="value">{transform.rotation.x.toFixed(1)}°</span>
          </div>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="rotation.y" value={transform.rotation.y} />
            <label>Y</label>
            <PrecisionSlider
              min={-180}
              max={180}
              step={0.01}
              value={transform.rotation.y}
              onChange={(v) => handlePropertyChange('rotation.y', v)}
              defaultValue={0}
            />
            <span className="value">{transform.rotation.y.toFixed(1)}°</span>
          </div>
          <div className="control-row">
            <KeyframeToggle clipId={selectedClip.id} property="rotation.z" value={transform.rotation.z} />
            <label>Z</label>
            <PrecisionSlider
              min={-180}
              max={180}
              step={0.01}
              value={transform.rotation.z}
              onChange={(v) => handlePropertyChange('rotation.z', v)}
              defaultValue={0}
            />
            <span className="value">{transform.rotation.z.toFixed(1)}°</span>
          </div>
        </div>

        {/* Masks */}
        <div className="properties-section masks-section">
          <div className="section-header-with-button">
            <h4>Masks</h4>
            <div className="mask-add-menu-container">
              <button
                className="btn btn-sm btn-add"
                onClick={() => setShowMaskMenu(!showMaskMenu)}
              >
                + Add
              </button>
              {showMaskMenu && (
                <div className="mask-add-menu">
                  <button
                    onClick={() => {
                      addRectangleMask(selectedClip.id);
                      setShowMaskMenu(false);
                    }}
                  >
                    Rectangle
                  </button>
                  <button
                    onClick={() => {
                      addEllipseMask(selectedClip.id);
                      setShowMaskMenu(false);
                    }}
                  >
                    Ellipse
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Shape drawing tools */}
          <div className="mask-shape-tools">
            <button
              className={`mask-tool-btn ${maskEditMode === 'drawingRect' ? 'active' : ''}`}
              onClick={() => handleStartDrawMode('drawingRect')}
              title="Draw Rectangle Mask (drag on preview)"
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="1" />
              </svg>
            </button>
            <button
              className={`mask-tool-btn ${maskEditMode === 'drawingEllipse' ? 'active' : ''}`}
              onClick={() => handleStartDrawMode('drawingEllipse')}
              title="Draw Ellipse Mask (drag on preview)"
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <ellipse cx="12" cy="12" rx="9" ry="9" />
              </svg>
            </button>
            <button
              className={`mask-tool-btn ${maskEditMode === 'drawingPen' ? 'active' : ''}`}
              onClick={() => handleStartDrawMode('drawingPen')}
              title="Pen Tool (click to add points)"
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 19l7-7 3 3-7 7-3-3z" />
                <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
                <path d="M2 2l7.586 7.586" />
                <circle cx="11" cy="11" r="2" />
              </svg>
            </button>
            {maskEditMode !== 'none' && maskEditMode !== 'editing' && (
              <button
                className="mask-tool-btn cancel"
                onClick={() => setMaskEditMode('none')}
                title="Cancel drawing (ESC)"
              >
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            )}
          </div>

          {maskEditMode !== 'none' && maskEditMode !== 'editing' && (
            <div className="mask-draw-hint">
              {maskEditMode === 'drawingRect' && 'Click and drag on preview to draw rectangle'}
              {maskEditMode === 'drawingEllipse' && 'Click and drag on preview to draw ellipse'}
              {maskEditMode === 'drawingPen' && 'Click to add points, click first point to close'}
            </div>
          )}

          {(!selectedClip.masks || selectedClip.masks.length === 0) ? (
            <div className="mask-empty">
              No masks. Use tools above or click "+ Add".
            </div>
          ) : (
            <div className="mask-list">
              {selectedClip.masks.map((mask) => (
                <MaskItem
                  key={mask.id}
                  clipId={selectedClip.id}
                  mask={mask}
                  isActive={activeMaskId === mask.id}
                  onSelect={() => setActiveMask(selectedClip.id, mask.id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Reset Button */}
        <div className="properties-actions">
          <button
            className="btn btn-sm"
            onClick={() => {
              updateClipTransform(selectedClip.id, {
                opacity: 1,
                blendMode: 'normal',
                position: { x: 0, y: 0, z: 0 },
                scale: { x: 1, y: 1 },
                rotation: { x: 0, y: 0, z: 0 },
              });
            }}
          >
            Reset All
          </button>
        </div>
      </div>
    </div>
  );
}
