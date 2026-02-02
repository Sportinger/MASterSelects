// Unified Properties Panel - Transform, Effects, Masks in one panel with sub-tabs
// Also handles Audio clips with Volume and EQ

import { useRef, useCallback, useState, useEffect, useMemo } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { BlendMode, AnimatableProperty, MaskMode, ClipMask, TranscriptWord, FrameAnalysisData } from '../../types';
import { createEffectProperty } from '../../types';
import { EQ_FREQUENCIES } from '../../services/audioManager';
import { TextTab } from './TextTab';
import { EFFECT_REGISTRY, getDefaultParams, getCategoriesWithEffects } from '../../effects';

// EQ band parameter names
const EQ_BAND_PARAMS = ['band31', 'band62', 'band125', 'band250', 'band500', 'band1k', 'band2k', 'band4k', 'band8k', 'band16k'];

// ============================================
// SHARED COMPONENTS
// ============================================

// Organized by category like After Effects
const BLEND_MODE_GROUPS: { label: string; modes: BlendMode[] }[] = [
  { label: 'Normal', modes: ['normal', 'dissolve', 'dancing-dissolve'] },
  { label: 'Darken', modes: ['darken', 'multiply', 'color-burn', 'classic-color-burn', 'linear-burn', 'darker-color'] },
  { label: 'Lighten', modes: ['add', 'lighten', 'screen', 'color-dodge', 'classic-color-dodge', 'linear-dodge', 'lighter-color'] },
  { label: 'Contrast', modes: ['overlay', 'soft-light', 'hard-light', 'linear-light', 'vivid-light', 'pin-light', 'hard-mix'] },
  { label: 'Inversion', modes: ['difference', 'classic-difference', 'exclusion', 'subtract', 'divide'] },
  { label: 'Component', modes: ['hue', 'saturation', 'color', 'luminosity'] },
  { label: 'Stencil', modes: ['stencil-alpha', 'stencil-luma', 'silhouette-alpha', 'silhouette-luma', 'alpha-add'] },
];

const formatBlendModeName = (mode: BlendMode): string => {
  return mode.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

// Keyframe toggle button
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
function ScaleKeyframeToggle({ clipId, scaleX, scaleY }: { clipId: string; scaleX: number; scaleY: number }) {
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore();

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
}

function PrecisionSlider({ min, max, step, value, onChange, defaultValue }: PrecisionSliderProps) {
  const sliderRef = useRef<HTMLDivElement>(null);
  const accumulatedDelta = useRef(0);
  const startValue = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    accumulatedDelta.current = 0;
    startValue.current = value;

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
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [value, min, max, step, onChange]);

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
}

function DraggableNumber({ value, onChange, defaultValue, sensitivity = 2, decimals = 2, suffix = '', min, max }: DraggableNumberProps) {
  const inputRef = useRef<HTMLSpanElement>(null);
  const accumulatedDelta = useRef(0);
  const startValue = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    accumulatedDelta.current = 0;
    startValue.current = value;

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
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [value, sensitivity, decimals, onChange, min, max]);

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

// ============================================
// TAB TYPE
// ============================================
type PropertiesTab = 'transform' | 'effects' | 'masks' | 'volume' | 'transcript' | 'analysis' | 'text';

// ============================================
// TRANSFORM TAB
// ============================================
interface TransformTabProps {
  clipId: string;
  transform: {
    opacity: number;
    blendMode: BlendMode;
    position: { x: number; y: number; z: number };
    scale: { x: number; y: number };
    rotation: { x: number; y: number; z: number };
  };
  speed?: number;
}

function TransformTab({ clipId, transform, speed = 1 }: TransformTabProps) {
  const { setPropertyValue, updateClipTransform } = useTimelineStore();

  const handlePropertyChange = (property: AnimatableProperty, value: number) => {
    setPropertyValue(clipId, property, value);
  };

  const uniformScale = (transform.scale.x + transform.scale.y) / 2;
  const handleUniformScaleChange = (value: number) => {
    handlePropertyChange('scale.x', value);
    handlePropertyChange('scale.y', value);
  };

  return (
    <div className="properties-tab-content">
      {/* Appearance */}
      <div className="properties-section">
        <h4>Appearance</h4>
        <div className="control-row">
          <label>Blend Mode</label>
          <select
            value={transform.blendMode}
            onChange={(e) => updateClipTransform(clipId, { blendMode: e.target.value as BlendMode })}
          >
            {BLEND_MODE_GROUPS.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.modes.map((mode) => (
                  <option key={mode} value={mode}>{formatBlendModeName(mode)}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="opacity" value={transform.opacity} />
          <label>Opacity</label>
          <PrecisionSlider min={0} max={1} step={0.0001} value={transform.opacity}
            onChange={(v) => handlePropertyChange('opacity', v)} defaultValue={1} />
          <span className="value">{(transform.opacity * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Time/Speed */}
      <div className="properties-section">
        <h4>Time</h4>
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="speed" value={speed} />
          <label>Speed</label>
          <PrecisionSlider min={-4} max={4} step={0.01} value={speed}
            onChange={(v) => handlePropertyChange('speed', v)} defaultValue={1} />
          <span className="value">{(speed * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Position */}
      <div className="properties-section">
        <h4>Position</h4>
        {(['x', 'y', 'z'] as const).map(axis => (
          <div className="control-row" key={axis}>
            <KeyframeToggle clipId={clipId} property={`position.${axis}`} value={transform.position[axis]} />
            <label>{axis.toUpperCase()}</label>
            <PrecisionSlider min={-1} max={1} step={0.0001} value={transform.position[axis]}
              onChange={(v) => handlePropertyChange(`position.${axis}`, v)} defaultValue={0} />
            <span className="value">{transform.position[axis].toFixed(3)}</span>
          </div>
        ))}
      </div>

      {/* Scale */}
      <div className="properties-section">
        <h4>Scale</h4>
        <div className="control-row">
          <ScaleKeyframeToggle clipId={clipId} scaleX={transform.scale.x} scaleY={transform.scale.y} />
          <label>Uniform</label>
          <PrecisionSlider min={0.1} max={3} step={0.0001} value={uniformScale}
            onChange={handleUniformScaleChange} defaultValue={1} />
          <span className="value">{uniformScale.toFixed(3)}</span>
        </div>
        {(['x', 'y'] as const).map(axis => (
          <div className="control-row" key={axis}>
            <span className="keyframe-toggle-placeholder" />
            <label>{axis.toUpperCase()}</label>
            <PrecisionSlider min={0.1} max={3} step={0.0001} value={transform.scale[axis]}
              onChange={(v) => handlePropertyChange(`scale.${axis}`, v)} defaultValue={1} />
            <span className="value">{transform.scale[axis].toFixed(3)}</span>
          </div>
        ))}
      </div>

      {/* Rotation */}
      <div className="properties-section">
        <h4>Rotation</h4>
        {(['x', 'y', 'z'] as const).map(axis => (
          <div className="control-row" key={axis}>
            <KeyframeToggle clipId={clipId} property={`rotation.${axis}`} value={transform.rotation[axis]} />
            <label>{axis.toUpperCase()}</label>
            <PrecisionSlider min={-180} max={180} step={0.01} value={transform.rotation[axis]}
              onChange={(v) => handlePropertyChange(`rotation.${axis}`, v)} defaultValue={0} />
            <span className="value">{transform.rotation[axis].toFixed(1)}°</span>
          </div>
        ))}
      </div>

      {/* Reset */}
      <div className="properties-actions">
        <button className="btn btn-sm" onClick={() => {
          updateClipTransform(clipId, {
            opacity: 1, blendMode: 'normal',
            position: { x: 0, y: 0, z: 0 }, scale: { x: 1, y: 1 }, rotation: { x: 0, y: 0, z: 0 },
          });
        }}>Reset All</button>
      </div>
    </div>
  );
}

// ============================================
// VOLUME TAB (Audio Clips)
// ============================================
interface VolumeTabProps {
  clipId: string;
  effects: Array<{ id: string; name: string; type: string; params: Record<string, number | boolean | string> }>;
}

function VolumeTab({ clipId, effects }: VolumeTabProps) {
  const { setPropertyValue, getInterpolatedEffects, playheadPosition, clips, addClipEffect, setClipPreservesPitch } = useTimelineStore();
  const clip = clips.find(c => c.id === clipId);
  const clipLocalTime = clip ? playheadPosition - clip.startTime : 0;
  const interpolatedEffects = getInterpolatedEffects(clipId, clipLocalTime);
  const preservesPitch = clip?.preservesPitch !== false; // default true

  // Auto-add audio effects if they don't exist
  useEffect(() => {
    const hasVolumeEffect = effects.some(e => e.type === 'audio-volume');
    const hasEQEffect = effects.some(e => e.type === 'audio-eq');
    if (!hasVolumeEffect) addClipEffect(clipId, 'audio-volume');
    if (!hasEQEffect) addClipEffect(clipId, 'audio-eq');
  }, [clipId, effects, addClipEffect]);

  // Get current values
  const volumeEffect = interpolatedEffects.find(e => e.type === 'audio-volume');
  const eqEffect = interpolatedEffects.find(e => e.type === 'audio-eq');
  const volume = (volumeEffect?.params?.volume as number) ?? 1;
  const eqBands = EQ_BAND_PARAMS.map(param => (eqEffect?.params?.[param] as number) ?? 0);

  const formatFreq = (freq: number) => freq >= 1000 ? `${freq / 1000}k` : `${freq}`;

  const handleVolumeChange = (value: number) => {
    if (!volumeEffect) return;
    const property = createEffectProperty(volumeEffect.id, 'volume');
    setPropertyValue(clipId, property, value);
  };

  const handleEQChange = (bandIndex: number, value: number) => {
    if (!eqEffect) return;
    const property = createEffectProperty(eqEffect.id, EQ_BAND_PARAMS[bandIndex]);
    setPropertyValue(clipId, property, value);
  };

  const handleResetEQ = () => {
    if (!eqEffect) return;
    EQ_BAND_PARAMS.forEach(param => {
      const property = createEffectProperty(eqEffect.id, param);
      setPropertyValue(clipId, property, 0);
    });
  };

  return (
    <div className="properties-tab-content volume-tab">
      {/* Volume Section */}
      <div className="properties-section">
        <div className="section-header-row">
          <h4>Volume</h4>
        </div>
        <div className="control-row">
          {volumeEffect && (
            <EffectKeyframeToggle clipId={clipId} effectId={volumeEffect.id} paramName="volume" value={volume} />
          )}
          <input type="range" min="0" max="2" step="0.01" value={volume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))} />
          <span className="value">{Math.round(volume * 100)}%</span>
        </div>
      </div>

      {/* Pitch Preservation Section */}
      <div className="properties-section">
        <div className="section-header-row">
          <h4>Speed Settings</h4>
        </div>
        <div className="control-row checkbox-row">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={preservesPitch}
              onChange={(e) => setClipPreservesPitch(clipId, e.target.checked)}
            />
            <span>Keep Pitch</span>
          </label>
          <span className="hint">When speed changes, maintain original pitch</span>
        </div>
      </div>

      {/* 10-Band EQ Section */}
      <div className="properties-section eq-section">
        <div className="section-header-row">
          <h4>10-Band Equalizer</h4>
          {eqEffect && (
            <EQKeyframeToggle clipId={clipId} effectId={eqEffect.id} eqBands={eqBands} />
          )}
          <button className="btn btn-sm" onClick={handleResetEQ}>Reset</button>
        </div>

        <div className="eq-bands">
          {EQ_FREQUENCIES.map((freq, index) => (
            <div key={freq} className="eq-band">
              <div className="eq-band-value">
                {eqBands[index] > 0 ? '+' : ''}{eqBands[index].toFixed(1)}
              </div>
              <input type="range" className="eq-slider" min="-12" max="12" step="0.5"
                value={eqBands[index]} onChange={(e) => handleEQChange(index, parseFloat(e.target.value))}
                title={`${formatFreq(freq)}Hz: ${eqBands[index].toFixed(1)}dB`} />
              <div className="eq-band-label">{formatFreq(freq)}</div>
            </div>
          ))}
        </div>

        <div className="eq-scale">
          <span>+12dB</span>
          <span>0dB</span>
          <span>-12dB</span>
        </div>
      </div>
    </div>
  );
}

// ============================================
// EFFECTS TAB
// ============================================
// Effects are now loaded from the modular effect registry
// See src/effects/ for effect definitions

function EffectKeyframeToggle({ clipId, effectId, paramName, value }: { clipId: string; effectId: string; paramName: string; value: number }) {
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore();
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
function EQKeyframeToggle({ clipId, effectId, eqBands }: { clipId: string; effectId: string; eqBands: number[] }) {
  const { isRecording, toggleKeyframeRecording, hasKeyframes, addKeyframe } = useTimelineStore();

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

interface EffectsTabProps {
  clipId: string;
  effects: Array<{ id: string; name: string; type: string; enabled: boolean; params: Record<string, number | boolean | string> }>;
}

function EffectsTab({ clipId, effects }: EffectsTabProps) {
  const { addClipEffect, removeClipEffect, updateClipEffect, setClipEffectEnabled, setPropertyValue, getInterpolatedEffects, playheadPosition, clips } = useTimelineStore();
  const clip = clips.find(c => c.id === clipId);
  const clipLocalTime = clip ? playheadPosition - clip.startTime : 0;
  const interpolatedEffects = getInterpolatedEffects(clipId, clipLocalTime);

  // Get effects grouped by category from registry
  const effectCategories = useMemo(() => getCategoriesWithEffects(), []);

  return (
    <div className="properties-tab-content effects-tab">
      <div className="effect-add-row">
        <select onChange={(e) => { if (e.target.value) { addClipEffect(clipId, e.target.value); e.target.value = ''; } }} defaultValue="">
          <option value="" disabled>+ Add Effect</option>
          {effectCategories.map(({ category, effects: catEffects }) => (
            <optgroup key={category} label={category.charAt(0).toUpperCase() + category.slice(1)}>
              {catEffects.map((effect) => (
                <option key={effect.id} value={effect.id}>{effect.name}</option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>

      {effects.length === 0 ? (
        <div className="panel-empty"><p>No effects applied</p></div>
      ) : (
        <div className="effects-list">
          {effects.map((effect) => {
            const interpolated = interpolatedEffects.find(e => e.id === effect.id) || effect;
            const isEnabled = effect.enabled !== false; // default to true if undefined
            return (
              <div key={effect.id} className={`effect-item ${!isEnabled ? 'bypassed' : ''}`}>
                <div className="effect-header">
                  <button
                    className={`effect-bypass-btn ${!isEnabled ? 'bypassed' : ''}`}
                    onClick={() => setClipEffectEnabled(clipId, effect.id, !isEnabled)}
                    title={isEnabled ? 'Bypass effect' : 'Enable effect'}
                  >
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      {isEnabled ? (
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      ) : (
                        <circle cx="12" cy="12" r="10" strokeDasharray="4 4" />
                      )}
                      {isEnabled && <polyline points="22 4 12 14.01 9 11.01" />}
                    </svg>
                  </button>
                  <span className="effect-name">{effect.name}</span>
                  <button className="btn btn-sm btn-danger" onClick={() => removeClipEffect(clipId, effect.id)}>×</button>
                </div>
                <div className="effect-params">
                  <EffectParams
                    effect={{ ...effect, params: interpolated.params }}
                    onChange={(params) => {
                      Object.entries(params).forEach(([paramName, value]) => {
                        if (typeof value === 'number') {
                          setPropertyValue(clipId, `effect.${effect.id}.${paramName}` as AnimatableProperty, value);
                        } else {
                          updateClipEffect(clipId, effect.id, { [paramName]: value });
                        }
                      });
                    }}
                    clipId={clipId}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Single parameter control renderer
function renderParamControl(
  paramName: string,
  paramDef: { type: string; label: string; default: number | boolean | string; min?: number; max?: number; step?: number; options?: { value: string; label: string }[]; animatable?: boolean },
  value: number | boolean | string,
  effect: { id: string; params: Record<string, number | boolean | string> },
  onChange: (params: Record<string, number | boolean | string>) => void,
  defaults: Record<string, number | boolean | string>,
  clipId?: string,
  noMaxLimit?: boolean
) {
  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    const defaultValue = defaults[paramName];
    if (defaultValue !== undefined) onChange({ ...effect.params, [paramName]: defaultValue });
  };

  const renderKfToggle = (val: number) => {
    if (!clipId) return null;
    return <EffectKeyframeToggle clipId={clipId} effectId={effect.id} paramName={paramName} value={val} />;
  };

  switch (paramDef.type) {
    case 'number': {
      const min = paramDef.min ?? 0;
      // For quality params with noMaxLimit, allow much higher values
      const max = noMaxLimit ? (paramDef.max ?? 1) * 10 : (paramDef.max ?? 1);
      const range = max - min;
      const decimals = paramDef.step && paramDef.step >= 1 ? 0 : paramDef.step && paramDef.step >= 0.1 ? 1 : 2;
      return (
        <div className="control-row" key={paramName} onContextMenu={handleContextMenu}>
          {paramDef.animatable && renderKfToggle(value as number)}
          <label>{paramDef.label}</label>
          <input
            type="range"
            min={min}
            max={max}
            step={paramDef.step ?? 0.01}
            value={value as number}
            onChange={(e) => onChange({ ...effect.params, [paramName]: parseFloat(e.target.value) })}
          />
          <DraggableNumber
            value={value as number}
            onChange={(v) => onChange({ ...effect.params, [paramName]: Math.max(min, v) })}
            defaultValue={paramDef.default as number}
            sensitivity={Math.max(0.5, range / 100)}
            decimals={decimals}
            min={min}
          />
        </div>
      );
    }

    case 'boolean':
      return (
        <div className="control-row checkbox-row" key={paramName}>
          <label>
            <input
              type="checkbox"
              checked={value as boolean}
              onChange={(e) => onChange({ ...effect.params, [paramName]: e.target.checked })}
            />
            {paramDef.label}
          </label>
        </div>
      );

    case 'select':
      return (
        <div className="control-row" key={paramName}>
          <label>{paramDef.label}</label>
          <select
            value={value as string}
            onChange={(e) => onChange({ ...effect.params, [paramName]: e.target.value })}
          >
            {paramDef.options?.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      );

    default:
      return null;
  }
}

// Effect parameters with collapsible Quality section
interface EffectParamsProps {
  effect: { id: string; type: string; params: Record<string, number | boolean | string> };
  onChange: (params: Record<string, number | boolean | string>) => void;
  clipId?: string;
}

function EffectParams({ effect, onChange, clipId }: EffectParamsProps) {
  const [qualityExpanded, setQualityExpanded] = useState(false);

  const effectDef = EFFECT_REGISTRY.get(effect.type);
  if (!effectDef) {
    return <p className="effect-info">Unknown effect type: {effect.type}</p>;
  }

  const defaults = getDefaultParams(effect.type);

  if (Object.keys(effectDef.params).length === 0) {
    return <p className="effect-info">No parameters</p>;
  }

  // Separate regular params from quality params
  const regularParams = Object.entries(effectDef.params).filter(([, def]) => !def.quality);
  const qualityParams = Object.entries(effectDef.params).filter(([, def]) => def.quality);

  const handleResetQuality = () => {
    const resetParams: Record<string, number | boolean | string> = { ...effect.params };
    qualityParams.forEach(([name, def]) => {
      resetParams[name] = def.default;
    });
    onChange(resetParams);
  };

  return (
    <>
      {/* Regular parameters */}
      {regularParams.map(([paramName, paramDef]) => {
        const value = effect.params[paramName] ?? paramDef.default;
        return renderParamControl(paramName, paramDef, value, effect, onChange, defaults, clipId, false);
      })}

      {/* Quality section (collapsible) */}
      {qualityParams.length > 0 && (
        <div className="effect-quality-section">
          <div className="effect-quality-header" onClick={() => setQualityExpanded(!qualityExpanded)}>
            <span className="effect-quality-toggle">{qualityExpanded ? '\u25BC' : '\u25B6'}</span>
            <span className="effect-quality-title">Quality</span>
            {qualityExpanded && (
              <button
                className="btn btn-xs effect-quality-reset"
                onClick={(e) => { e.stopPropagation(); handleResetQuality(); }}
                title="Reset quality to defaults"
              >
                Reset
              </button>
            )}
          </div>
          {qualityExpanded && (
            <div className="effect-quality-params">
              {qualityParams.map(([paramName, paramDef]) => {
                const value = effect.params[paramName] ?? paramDef.default;
                // Quality params have no max limit when dragging
                return renderParamControl(paramName, paramDef, value, effect, onChange, defaults, clipId, true);
              })}
              <div className="effect-quality-warning">
                High values may cause slowdowns
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}

// ============================================
// MASKS TAB
// ============================================
const MASK_MODES: { value: MaskMode; label: string }[] = [
  { value: 'add', label: 'Add' },
  { value: 'subtract', label: 'Subtract' },
  { value: 'intersect', label: 'Intersect' },
];

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

  const handleNameDoubleClick = () => { setIsEditing(true); setEditName(mask.name); };
  const handleNameChange = () => { if (editName.trim()) updateMask(clipId, mask.id, { name: editName.trim() }); setIsEditing(false); };
  const handleEditMask = () => { onSelect(); setActiveMask(clipId, mask.id); setMaskEditMode('editing'); };

  return (
    <div className={`mask-item ${isActive ? 'active' : ''} ${mask.expanded ? 'expanded' : ''}`}>
      <div className="mask-item-header" onClick={onSelect}>
        <button className="mask-expand-btn" onClick={(e) => { e.stopPropagation(); updateMask(clipId, mask.id, { expanded: !mask.expanded }); }}>
          {mask.expanded ? '\u25BC' : '\u25B6'}
        </button>
        {isEditing ? (
          <input type="text" className="mask-name-input" value={editName} onChange={(e) => setEditName(e.target.value)}
            onBlur={handleNameChange} onKeyDown={(e) => { if (e.key === 'Enter') handleNameChange(); if (e.key === 'Escape') setIsEditing(false); }}
            autoFocus onClick={(e) => e.stopPropagation()} />
        ) : (
          <span className="mask-name" onDoubleClick={handleNameDoubleClick}>{mask.name}</span>
        )}
        <select className="mask-mode-select" value={mask.mode} onChange={(e) => updateMask(clipId, mask.id, { mode: e.target.value as MaskMode })}
          onClick={(e) => e.stopPropagation()}>
          {MASK_MODES.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
        <button className="mask-visible-btn" onClick={(e) => { e.stopPropagation(); updateMask(clipId, mask.id, { visible: !mask.visible }); }}
          title={mask.visible ? "Hide mask outline" : "Show mask outline"} style={{ opacity: mask.visible ? 1 : 0.5 }}>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            {mask.visible ? (<><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></>)
              : (<><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" /><line x1="1" y1="1" x2="23" y2="23" /></>)}
          </svg>
        </button>
        <button className="mask-edit-btn" onClick={(e) => { e.stopPropagation(); handleEditMask(); }} title="Edit mask path">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
          </svg>
        </button>
        <button className="mask-delete-btn" onClick={(e) => { e.stopPropagation(); removeMask(clipId, mask.id); }} title="Delete mask">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
      {mask.expanded && (
        <div className="mask-item-properties">
          <div className="control-row"><label>Opacity</label>
            <DraggableNumber value={mask.opacity * 100} onChange={(v) => updateMask(clipId, mask.id, { opacity: v / 100 })}
              defaultValue={100} sensitivity={1} decimals={0} suffix="%" /></div>
          <div className="control-row"><label>Feather</label>
            <DraggableNumber value={mask.feather} onChange={(v) => updateMask(clipId, mask.id, { feather: v })}
              defaultValue={0} sensitivity={1} decimals={1} suffix="px" /></div>
          <div className="control-row"><label>Quality</label>
            <DraggableNumber value={mask.featherQuality ?? 50} onChange={(v) => updateMask(clipId, mask.id, { featherQuality: Math.min(100, Math.max(1, Math.round(v))) })}
              defaultValue={50} min={1} max={100} sensitivity={1} decimals={0} /></div>
          <div className="control-row"><label>Position X</label>
            <DraggableNumber value={mask.position.x} onChange={(v) => updateMask(clipId, mask.id, { position: { ...mask.position, x: v } })}
              defaultValue={0} sensitivity={100} decimals={3} /></div>
          <div className="control-row"><label>Position Y</label>
            <DraggableNumber value={mask.position.y} onChange={(v) => updateMask(clipId, mask.id, { position: { ...mask.position, y: v } })}
              defaultValue={0} sensitivity={100} decimals={3} /></div>
          <div className="control-row"><label>Inverted</label>
            <input type="checkbox" checked={mask.inverted} onChange={(e) => updateMask(clipId, mask.id, { inverted: e.target.checked })} /></div>
          <div className="mask-info">{mask.vertices.length} vertices | {mask.closed ? 'Closed' : 'Open'}</div>
        </div>
      )}
    </div>
  );
}

interface MasksTabProps {
  clipId: string;
  masks: ClipMask[] | undefined;
}

function MasksTab({ clipId, masks }: MasksTabProps) {
  const { addRectangleMask, addEllipseMask, activeMaskId, setActiveMask, maskEditMode, setMaskEditMode } = useTimelineStore();
  const [showMaskMenu, setShowMaskMenu] = useState(false);

  const handleStartDrawMode = (mode: 'drawingRect' | 'drawingEllipse' | 'drawingPen') => setMaskEditMode(mode);

  return (
    <div className="properties-tab-content masks-tab">
      <div className="section-header-with-button">
        <div className="mask-add-menu-container">
          <button className="btn btn-sm btn-add" onClick={() => setShowMaskMenu(!showMaskMenu)}>+ Add</button>
          {showMaskMenu && (
            <div className="mask-add-menu">
              <button onClick={() => { addRectangleMask(clipId); setShowMaskMenu(false); }}>Rectangle</button>
              <button onClick={() => { addEllipseMask(clipId); setShowMaskMenu(false); }}>Ellipse</button>
            </div>
          )}
        </div>
      </div>

      <div className="mask-shape-tools">
        <button className={`mask-tool-btn ${maskEditMode === 'drawingRect' ? 'active' : ''}`}
          onClick={() => handleStartDrawMode('drawingRect')} title="Draw Rectangle Mask">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="1" /></svg>
        </button>
        <button className={`mask-tool-btn ${maskEditMode === 'drawingEllipse' ? 'active' : ''}`}
          onClick={() => handleStartDrawMode('drawingEllipse')} title="Draw Ellipse Mask">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="12" rx="9" ry="9" /></svg>
        </button>
        <button className={`mask-tool-btn ${maskEditMode === 'drawingPen' ? 'active' : ''}`}
          onClick={() => handleStartDrawMode('drawingPen')} title="Pen Tool">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 19l7-7 3 3-7 7-3-3z" /><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
            <path d="M2 2l7.586 7.586" /><circle cx="11" cy="11" r="2" />
          </svg>
        </button>
        {maskEditMode !== 'none' && maskEditMode !== 'editing' && (
          <button className="mask-tool-btn cancel" onClick={() => setMaskEditMode('none')} title="Cancel drawing (ESC)">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
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

      {(!masks || masks.length === 0) ? (
        <div className="mask-empty">No masks. Use tools above or click "+ Add".</div>
      ) : (
        <div className="mask-list">
          {masks.map((mask) => (
            <MaskItem key={mask.id} clipId={clipId} mask={mask} isActive={activeMaskId === mask.id}
              onSelect={() => setActiveMask(clipId, mask.id)} />
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================
// TRANSCRIPT TAB
// ============================================

const LANGUAGES = [
  { code: 'de', name: 'Deutsch' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'fr', name: 'Français' },
  { code: 'it', name: 'Italiano' },
  { code: 'pt', name: 'Português' },
];

function formatTimeShort(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

interface TranscriptTabProps {
  clipId: string;
  transcript: TranscriptWord[];
  transcriptStatus: 'none' | 'transcribing' | 'ready' | 'error';
  transcriptProgress: number;
  clipStartTime: number;
  inPoint: number;
}

function TranscriptTab({ clipId, transcript, transcriptStatus, transcriptProgress, clipStartTime, inPoint }: TranscriptTabProps) {
  const { setPlayheadPosition, playheadPosition } = useTimelineStore();
  const [language, setLanguage] = useState(() => localStorage.getItem('transcriptLanguage') || 'de');
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate clip-local time for word matching
  const clipLocalTime = playheadPosition - clipStartTime + inPoint;

  // Find current word based on playhead position
  const currentWordId = useMemo(() => {
    if (clipLocalTime < 0 || transcript.length === 0) return null;
    for (const word of transcript) {
      if (clipLocalTime >= word.start && clipLocalTime <= word.end) {
        return word.id;
      }
    }
    return null;
  }, [transcript, clipLocalTime]);

  // Filter words by search query
  const filteredWords = useMemo(() => {
    if (!searchQuery.trim()) return transcript;
    const query = searchQuery.toLowerCase();
    return transcript.filter(w => w.text.toLowerCase().includes(query));
  }, [transcript, searchQuery]);

  const handleWordClick = useCallback((sourceTime: number) => {
    const timelinePosition = clipStartTime + (sourceTime - inPoint);
    setPlayheadPosition(Math.max(0, timelinePosition));
  }, [clipStartTime, inPoint, setPlayheadPosition]);

  const handleTranscribe = useCallback(async () => {
    const { transcribeClip } = await import('../../services/clipTranscriber');
    await transcribeClip(clipId, language);
  }, [clipId, language]);

  const handleCancel = useCallback(async () => {
    const { cancelTranscription } = await import('../../services/clipTranscriber');
    cancelTranscription();
  }, []);

  const handleDelete = useCallback(async () => {
    const { clearClipTranscript } = await import('../../services/clipTranscriber');
    clearClipTranscript(clipId);
  }, [clipId]);

  const handleLanguageChange = useCallback((newLanguage: string) => {
    setLanguage(newLanguage);
    localStorage.setItem('transcriptLanguage', newLanguage);
  }, []);

  return (
    <div className="properties-tab-content transcript-tab">
      {/* Language and actions */}
      <div className="properties-section">
        <div className="control-row">
          <label>Language</label>
          <select value={language} onChange={(e) => handleLanguageChange(e.target.value)}
            disabled={transcriptStatus === 'transcribing'}>
            {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.name}</option>)}
          </select>
        </div>
        <div className="transcript-tab-actions">
          {transcriptStatus !== 'ready' && transcriptStatus !== 'transcribing' && (
            <button className="btn btn-sm" onClick={handleTranscribe}>Transcribe</button>
          )}
          {transcriptStatus === 'transcribing' && (
            <button className="btn btn-sm btn-danger" onClick={handleCancel}>Cancel</button>
          )}
          {transcriptStatus === 'ready' && (
            <>
              <button className="btn btn-sm" onClick={handleTranscribe}>Re-transcribe</button>
              <button className="btn btn-sm btn-danger" onClick={handleDelete}>Delete</button>
            </>
          )}
        </div>
      </div>

      {/* Progress */}
      {transcriptStatus === 'transcribing' && (
        <div className="properties-section">
          <div className="transcript-progress-bar">
            <div className="transcript-progress-fill" style={{ width: `${transcriptProgress}%` }} />
          </div>
          <span className="transcript-progress-text">{transcriptProgress}%</span>
        </div>
      )}

      {/* Search */}
      {transcript.length > 0 && (
        <div className="properties-section">
          <input type="text" placeholder="Search transcript..." value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)} className="transcript-search-input" />
        </div>
      )}

      {/* Transcript content */}
      <div className="transcript-content-embedded" ref={containerRef}>
        {transcript.length === 0 ? (
          <div className="transcript-empty-state">
            {transcriptStatus === 'transcribing' ? 'Transcribing...' : 'No transcript. Click "Transcribe" to generate.'}
          </div>
        ) : (
          <div className="transcript-words-flow">
            {filteredWords.map(word => (
              <span
                key={word.id}
                className={`transcript-word-inline ${word.id === currentWordId ? 'active' : ''} ${searchQuery && word.text.toLowerCase().includes(searchQuery.toLowerCase()) ? 'highlighted' : ''}`}
                onClick={() => handleWordClick(word.start)}
                title={formatTimeShort(word.start)}
              >
                {word.text}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Status */}
      {transcriptStatus === 'ready' && (
        <div className="transcript-status-bar">
          {transcript.length} words
        </div>
      )}
    </div>
  );
}

// ============================================
// ANALYSIS TAB
// ============================================

interface AnalysisTabProps {
  clipId: string;
  analysis: { frames: FrameAnalysisData[] } | undefined;
  analysisStatus: 'none' | 'analyzing' | 'ready' | 'error';
  analysisProgress: number;
  clipStartTime: number;
  inPoint: number;
  outPoint: number;
}

function AnalysisTab({ clipId, analysis, analysisStatus, analysisProgress, clipStartTime, inPoint, outPoint }: AnalysisTabProps) {
  const { playheadPosition } = useTimelineStore();

  // Calculate current values at playhead
  const currentValues = useMemo((): FrameAnalysisData | null => {
    if (!analysis?.frames.length) return null;

    const clipEnd = clipStartTime + (outPoint - inPoint);
    if (playheadPosition < clipStartTime || playheadPosition > clipEnd) return null;

    const timeInClip = playheadPosition - clipStartTime;
    const sourceTime = inPoint + timeInClip;

    let closestFrame = analysis.frames[0];
    let closestDistance = Math.abs(closestFrame.timestamp - sourceTime);

    for (const frame of analysis.frames) {
      const distance = Math.abs(frame.timestamp - sourceTime);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestFrame = frame;
      }
    }
    return closestFrame;
  }, [analysis, clipStartTime, inPoint, outPoint, playheadPosition]);

  // Stats summary
  const stats = useMemo(() => {
    if (!analysis?.frames.length) return null;
    const frames = analysis.frames;
    return {
      avgFocus: Math.round(frames.reduce((s, f) => s + f.focus, 0) / frames.length * 100),
      avgMotion: Math.round(frames.reduce((s, f) => s + f.motion, 0) / frames.length * 100),
      maxFocus: Math.round(Math.max(...frames.map(f => f.focus)) * 100),
      maxMotion: Math.round(Math.max(...frames.map(f => f.motion)) * 100),
      totalFaces: frames.reduce((s, f) => s + f.faceCount, 0),
      frameCount: frames.length,
    };
  }, [analysis]);

  const handleAnalyze = useCallback(async () => {
    const { analyzeClip } = await import('../../services/clipAnalyzer');
    await analyzeClip(clipId);
  }, [clipId]);

  const handleCancel = useCallback(async () => {
    const { cancelAnalysis } = await import('../../services/clipAnalyzer');
    cancelAnalysis();
  }, []);

  const handleClear = useCallback(async () => {
    const { clearClipAnalysis } = await import('../../services/clipAnalyzer');
    clearClipAnalysis(clipId);
  }, [clipId]);

  return (
    <div className="properties-tab-content analysis-tab">
      {/* Actions */}
      <div className="properties-section">
        <div className="analysis-tab-actions">
          {analysisStatus !== 'ready' && analysisStatus !== 'analyzing' && (
            <button className="btn btn-sm" onClick={handleAnalyze}>Analyze Clip</button>
          )}
          {analysisStatus === 'analyzing' && (
            <button className="btn btn-sm btn-danger" onClick={handleCancel}>Cancel</button>
          )}
          {analysisStatus === 'ready' && (
            <>
              <button className="btn btn-sm" onClick={handleAnalyze}>Re-analyze</button>
              <button className="btn btn-sm btn-danger" onClick={handleClear}>Clear</button>
            </>
          )}
        </div>
      </div>

      {/* Progress */}
      {analysisStatus === 'analyzing' && (
        <div className="properties-section">
          <div className="analysis-progress-bar">
            <div className="analysis-progress-fill" style={{ width: `${analysisProgress}%` }} />
          </div>
          <span className="analysis-progress-text">{analysisProgress}%</span>
        </div>
      )}

      {/* Current values at playhead */}
      {currentValues && (
        <div className="properties-section">
          <h4>Current Frame</h4>
          <div className="analysis-realtime-grid">
            <div className="analysis-metric">
              <span className="metric-label">Focus</span>
              <div className="metric-bar"><div className="metric-fill focus" style={{ width: `${Math.round(currentValues.focus * 100)}%` }} /></div>
              <span className="metric-value">{Math.round(currentValues.focus * 100)}%</span>
            </div>
            <div className="analysis-metric">
              <span className="metric-label">Motion</span>
              <div className="metric-bar"><div className="metric-fill motion" style={{ width: `${Math.round(currentValues.motion * 100)}%` }} /></div>
              <span className="metric-value">{Math.round(currentValues.motion * 100)}%</span>
            </div>
            {currentValues.faceCount > 0 && (
              <div className="analysis-metric">
                <span className="metric-label">Faces</span>
                <span className="metric-value">{currentValues.faceCount}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stats summary */}
      {stats && (
        <div className="properties-section">
          <h4>Summary ({stats.frameCount} frames)</h4>
          <div className="analysis-stats-grid">
            <div className="stat-row"><span>Avg Focus:</span><span>{stats.avgFocus}%</span></div>
            <div className="stat-row"><span>Peak Focus:</span><span>{stats.maxFocus}%</span></div>
            <div className="stat-row"><span>Avg Motion:</span><span>{stats.avgMotion}%</span></div>
            <div className="stat-row"><span>Peak Motion:</span><span>{stats.maxMotion}%</span></div>
            <div className="stat-row"><span>Total Faces:</span><span>{stats.totalFaces}</span></div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {analysisStatus !== 'ready' && analysisStatus !== 'analyzing' && (
        <div className="analysis-empty-state">
          Click "Analyze Clip" to detect focus, motion, and faces.
        </div>
      )}
    </div>
  );
}

// ============================================
// MAIN PANEL
// ============================================
export function PropertiesPanel() {
  const { clips, tracks, selectedClipIds, playheadPosition, getInterpolatedTransform, getInterpolatedSpeed } = useTimelineStore();
  const [activeTab, setActiveTab] = useState<PropertiesTab>('transform');
  const [lastClipId, setLastClipId] = useState<string | null>(null);

  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;
  const selectedClip = clips.find(c => c.id === selectedClipId);

  // Check if it's an audio clip
  const selectedTrack = selectedClip ? tracks.find(t => t.id === selectedClip.trackId) : null;
  const isAudioClip = selectedTrack?.type === 'audio';

  // Check if it's a text clip
  const isTextClip = selectedClip?.source?.type === 'text';

  // Reset tab when switching between audio/video/text clips
  useEffect(() => {
    if (selectedClipId && selectedClipId !== lastClipId) {
      setLastClipId(selectedClipId);
      // Set appropriate default tab based on clip type
      if (isTextClip) {
        setActiveTab('text');
      } else if (isAudioClip && (activeTab === 'transform' || activeTab === 'masks' || activeTab === 'text')) {
        setActiveTab('volume');
      } else if (!isAudioClip && !isTextClip && (activeTab === 'volume' || activeTab === 'text')) {
        setActiveTab('transform');
      }
    }
  }, [selectedClipId, isAudioClip, isTextClip, lastClipId, activeTab]);

  if (!selectedClip) {
    return (
      <div className="properties-panel">
        <div className="panel-header"><h3>Properties</h3></div>
        <div className="panel-empty"><p>Select a clip to edit properties</p></div>
      </div>
    );
  }

  const clipLocalTime = playheadPosition - selectedClip.startTime;
  const transform = getInterpolatedTransform(selectedClip.id, clipLocalTime);
  const interpolatedSpeed = getInterpolatedSpeed(selectedClip.id, clipLocalTime);

  // Count non-audio effects for badge
  const visualEffects = (selectedClip.effects || []).filter(e => e.type !== 'audio-volume' && e.type !== 'audio-eq');

  return (
    <div className="properties-panel">
      <div className="properties-tabs">
        {isAudioClip ? (
          <>
            <button className={`tab-btn ${activeTab === 'volume' ? 'active' : ''}`} onClick={() => setActiveTab('volume')}>Volume</button>
            <button className={`tab-btn ${activeTab === 'effects' ? 'active' : ''}`} onClick={() => setActiveTab('effects')}>
              Effects {visualEffects.length > 0 && <span className="badge">{visualEffects.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>
              Transcript {selectedClip.transcript && selectedClip.transcript.length > 0 && <span className="badge">{selectedClip.transcript.length}</span>}
            </button>
          </>
        ) : isTextClip ? (
          <>
            <button className={`tab-btn ${activeTab === 'text' ? 'active' : ''}`} onClick={() => setActiveTab('text')}>Text</button>
            <button className={`tab-btn ${activeTab === 'transform' ? 'active' : ''}`} onClick={() => setActiveTab('transform')}>Transform</button>
            <button className={`tab-btn ${activeTab === 'effects' ? 'active' : ''}`} onClick={() => setActiveTab('effects')}>
              Effects {visualEffects.length > 0 && <span className="badge">{visualEffects.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'masks' ? 'active' : ''}`} onClick={() => setActiveTab('masks')}>
              Masks {selectedClip.masks && selectedClip.masks.length > 0 && <span className="badge">{selectedClip.masks.length}</span>}
            </button>
          </>
        ) : (
          <>
            <button className={`tab-btn ${activeTab === 'transform' ? 'active' : ''}`} onClick={() => setActiveTab('transform')}>Transform</button>
            <button className={`tab-btn ${activeTab === 'volume' ? 'active' : ''}`} onClick={() => setActiveTab('volume')}>Audio</button>
            <button className={`tab-btn ${activeTab === 'effects' ? 'active' : ''}`} onClick={() => setActiveTab('effects')}>
              Effects {visualEffects.length > 0 && <span className="badge">{visualEffects.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'masks' ? 'active' : ''}`} onClick={() => setActiveTab('masks')}>
              Masks {selectedClip.masks && selectedClip.masks.length > 0 && <span className="badge">{selectedClip.masks.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>
              Transcript {selectedClip.transcript && selectedClip.transcript.length > 0 && <span className="badge">{selectedClip.transcript.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>
              Analysis {selectedClip.analysisStatus === 'ready' && <span className="badge">✓</span>}
            </button>
          </>
        )}
      </div>

      <div className="properties-content">
        {activeTab === 'text' && isTextClip && selectedClip.textProperties && (
          <TextTab clipId={selectedClip.id} textProperties={selectedClip.textProperties} />
        )}
        {activeTab === 'transform' && !isAudioClip && <TransformTab clipId={selectedClip.id} transform={transform} speed={interpolatedSpeed} />}
        {activeTab === 'volume' && <VolumeTab clipId={selectedClip.id} effects={selectedClip.effects || []} />}
        {activeTab === 'effects' && <EffectsTab clipId={selectedClip.id} effects={selectedClip.effects || []} />}
        {activeTab === 'masks' && !isAudioClip && <MasksTab clipId={selectedClip.id} masks={selectedClip.masks} />}
        {activeTab === 'transcript' && (
          <TranscriptTab
            clipId={selectedClip.id}
            transcript={selectedClip.transcript || []}
            transcriptStatus={selectedClip.transcriptStatus || 'none'}
            transcriptProgress={selectedClip.transcriptProgress || 0}
            clipStartTime={selectedClip.startTime}
            inPoint={selectedClip.inPoint}
          />
        )}
        {activeTab === 'analysis' && !isAudioClip && (
          <AnalysisTab
            clipId={selectedClip.id}
            analysis={selectedClip.analysis}
            analysisStatus={selectedClip.analysisStatus || 'none'}
            analysisProgress={selectedClip.analysisProgress || 0}
            clipStartTime={selectedClip.startTime}
            inPoint={selectedClip.inPoint}
            outPoint={selectedClip.outPoint}
          />
        )}
      </div>
    </div>
  );
}
