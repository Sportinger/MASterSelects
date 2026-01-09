// Unified Properties Panel - Transform, Effects, Masks, Transcript, Analysis in one panel with sub-tabs
// Also handles Audio clips with Volume and EQ

import { useRef, useCallback, useState, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { BlendMode, AnimatableProperty, MaskMode, ClipMask, EffectType } from '../../types';
import { createEffectProperty } from '../../types';
import { EQ_FREQUENCIES } from '../../services/audioManager';
import { TranscriptPanel } from './TranscriptPanel';
import { AnalysisPanel } from './AnalysisPanel';

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
}

function DraggableNumber({ value, onChange, defaultValue, sensitivity = 2, decimals = 2, suffix = '' }: DraggableNumberProps) {
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
      const newValue = startValue.current + deltaValue;
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
type PropertiesTab = 'transform' | 'effects' | 'masks' | 'volume' | 'transcript' | 'analysis';

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
}

function TransformTab({ clipId, transform }: TransformTabProps) {
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
          <span className="keyframe-toggle-placeholder" />
          <label>Uniform</label>
          <PrecisionSlider min={0.1} max={3} step={0.0001} value={uniformScale}
            onChange={handleUniformScaleChange} defaultValue={1} />
          <span className="value">{uniformScale.toFixed(3)}</span>
        </div>
        {(['x', 'y'] as const).map(axis => (
          <div className="control-row" key={axis}>
            <KeyframeToggle clipId={clipId} property={`scale.${axis}`} value={transform.scale[axis]} />
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
  const { setPropertyValue, getInterpolatedEffects, playheadPosition, clips, addClipEffect } = useTimelineStore();
  const clip = clips.find(c => c.id === clipId);
  const clipLocalTime = clip ? playheadPosition - clip.startTime : 0;
  const interpolatedEffects = getInterpolatedEffects(clipId, clipLocalTime);

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
          {volumeEffect && (
            <EffectKeyframeToggle clipId={clipId} effectId={volumeEffect.id} paramName="volume" value={volume} />
          )}
        </div>
        <div className="control-row">
          <input type="range" min="0" max="2" step="0.01" value={volume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))} />
          <span className="value">{Math.round(volume * 100)}%</span>
        </div>
      </div>

      {/* 10-Band EQ Section */}
      <div className="properties-section eq-section">
        <div className="section-header-row">
          <h4>10-Band Equalizer</h4>
          <button className="btn btn-sm" onClick={handleResetEQ}>Reset</button>
        </div>

        <div className="eq-bands">
          {EQ_FREQUENCIES.map((freq, index) => (
            <div key={freq} className="eq-band">
              <div className="eq-band-kf">
                {eqEffect && (
                  <EffectKeyframeToggle clipId={clipId} effectId={eqEffect.id} paramName={EQ_BAND_PARAMS[index]} value={eqBands[index]} />
                )}
              </div>
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
const AVAILABLE_EFFECTS: { type: EffectType; name: string }[] = [
  { type: 'hue-shift', name: 'Hue Shift' },
  { type: 'brightness', name: 'Brightness' },
  { type: 'contrast', name: 'Contrast' },
  { type: 'saturation', name: 'Saturation' },
  { type: 'levels', name: 'Levels' },
  { type: 'pixelate', name: 'Pixelate' },
  { type: 'kaleidoscope', name: 'Kaleidoscope' },
  { type: 'mirror', name: 'Mirror' },
  { type: 'rgb-split', name: 'RGB Split' },
  { type: 'invert', name: 'Invert' },
];

const EFFECT_DEFAULTS: Record<string, Record<string, number | boolean | string>> = {
  'hue-shift': { shift: 0 },
  'saturation': { amount: 1 },
  'brightness': { amount: 0 },
  'contrast': { amount: 1 },
  'blur': { radius: 0 },
  'pixelate': { size: 8 },
  'kaleidoscope': { segments: 6, rotation: 0 },
  'mirror': { horizontal: true, vertical: false },
  'invert': {},
  'rgb-split': { amount: 0.01, angle: 0 },
  'levels': { inputBlack: 0, inputWhite: 1, gamma: 1, outputBlack: 0, outputWhite: 1 },
};

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

interface EffectsTabProps {
  clipId: string;
  effects: Array<{ id: string; name: string; type: string; params: Record<string, number | boolean | string> }>;
}

function EffectsTab({ clipId, effects }: EffectsTabProps) {
  const { addClipEffect, removeClipEffect, updateClipEffect, setPropertyValue, getInterpolatedEffects, playheadPosition, clips } = useTimelineStore();
  const clip = clips.find(c => c.id === clipId);
  const clipLocalTime = clip ? playheadPosition - clip.startTime : 0;
  const interpolatedEffects = getInterpolatedEffects(clipId, clipLocalTime);

  return (
    <div className="properties-tab-content effects-tab">
      <div className="effect-add-row">
        <select onChange={(e) => { if (e.target.value) { addClipEffect(clipId, e.target.value); e.target.value = ''; } }} defaultValue="">
          <option value="" disabled>+ Add Effect</option>
          {AVAILABLE_EFFECTS.map((effect) => (
            <option key={effect.type} value={effect.type}>{effect.name}</option>
          ))}
        </select>
      </div>

      {effects.length === 0 ? (
        <div className="panel-empty"><p>No effects applied</p></div>
      ) : (
        <div className="effects-list">
          {effects.map((effect) => {
            const interpolated = interpolatedEffects.find(e => e.id === effect.id) || effect;
            return (
              <div key={effect.id} className="effect-item">
                <div className="effect-header">
                  <span className="effect-name">{effect.name}</span>
                  <button className="btn btn-sm btn-danger" onClick={() => removeClipEffect(clipId, effect.id)}>×</button>
                </div>
                <div className="effect-params">
                  {renderEffectParams({ ...effect, params: interpolated.params }, (params) => {
                    Object.entries(params).forEach(([paramName, value]) => {
                      if (typeof value === 'number') {
                        setPropertyValue(clipId, `effect.${effect.id}.${paramName}` as AnimatableProperty, value);
                      } else {
                        updateClipEffect(clipId, effect.id, { [paramName]: value });
                      }
                    });
                  }, clipId)}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function renderEffectParams(
  effect: { id: string; type: string; params: Record<string, number | boolean | string> },
  onChange: (params: Record<string, number | boolean | string>) => void,
  clipId?: string
) {
  const handleContextMenu = (paramName: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    const defaults = EFFECT_DEFAULTS[effect.type] || {};
    const defaultValue = defaults[paramName];
    if (defaultValue !== undefined) onChange({ ...effect.params, [paramName]: defaultValue });
  };

  const renderKfToggle = (paramName: string, value: number) => {
    if (!clipId) return null;
    return <EffectKeyframeToggle clipId={clipId} effectId={effect.id} paramName={paramName} value={value} />;
  };

  switch (effect.type) {
    case 'hue-shift':
      return (<div className="control-row">{renderKfToggle('shift', effect.params.shift as number)}<label>Shift</label>
        <input type="range" min="0" max="1" step="0.01" value={effect.params.shift as number}
          onChange={(e) => onChange({ shift: parseFloat(e.target.value) })} onContextMenu={handleContextMenu('shift')} /></div>);
    case 'brightness':
      return (<div className="control-row">{renderKfToggle('amount', effect.params.amount as number)}<label>Amount</label>
        <input type="range" min="-1" max="1" step="0.01" value={effect.params.amount as number}
          onChange={(e) => onChange({ amount: parseFloat(e.target.value) })} onContextMenu={handleContextMenu('amount')} /></div>);
    case 'contrast':
    case 'saturation':
      return (<div className="control-row">{renderKfToggle('amount', effect.params.amount as number)}<label>Amount</label>
        <input type="range" min="0" max="3" step="0.01" value={effect.params.amount as number}
          onChange={(e) => onChange({ amount: parseFloat(e.target.value) })} onContextMenu={handleContextMenu('amount')} /></div>);
    case 'pixelate':
      return (<div className="control-row">{renderKfToggle('size', effect.params.size as number)}<label>Size</label>
        <input type="range" min="1" max="64" step="1" value={effect.params.size as number}
          onChange={(e) => onChange({ size: parseInt(e.target.value, 10) })} onContextMenu={handleContextMenu('size')} /></div>);
    case 'kaleidoscope':
      return (<>
        <div className="control-row">{renderKfToggle('segments', effect.params.segments as number)}<label>Segments</label>
          <input type="range" min="2" max="16" step="1" value={effect.params.segments as number}
            onChange={(e) => onChange({ ...effect.params, segments: parseInt(e.target.value, 10) })} onContextMenu={handleContextMenu('segments')} /></div>
        <div className="control-row">{renderKfToggle('rotation', effect.params.rotation as number)}<label>Rotation</label>
          <input type="range" min="0" max={Math.PI * 2} step="0.01" value={effect.params.rotation as number}
            onChange={(e) => onChange({ ...effect.params, rotation: parseFloat(e.target.value) })} onContextMenu={handleContextMenu('rotation')} /></div>
      </>);
    case 'mirror':
      return (<>
        <div className="control-row"><label><input type="checkbox" checked={effect.params.horizontal as boolean}
          onChange={(e) => onChange({ ...effect.params, horizontal: e.target.checked })} /> Horizontal</label></div>
        <div className="control-row"><label><input type="checkbox" checked={effect.params.vertical as boolean}
          onChange={(e) => onChange({ ...effect.params, vertical: e.target.checked })} /> Vertical</label></div>
      </>);
    case 'rgb-split':
      return (<>
        <div className="control-row">{renderKfToggle('amount', effect.params.amount as number)}<label>Amount</label>
          <input type="range" min="0" max="0.1" step="0.001" value={effect.params.amount as number}
            onChange={(e) => onChange({ ...effect.params, amount: parseFloat(e.target.value) })} onContextMenu={handleContextMenu('amount')} /></div>
        <div className="control-row">{renderKfToggle('angle', effect.params.angle as number)}<label>Angle</label>
          <input type="range" min="0" max={Math.PI * 2} step="0.01" value={effect.params.angle as number}
            onChange={(e) => onChange({ ...effect.params, angle: parseFloat(e.target.value) })} onContextMenu={handleContextMenu('angle')} /></div>
      </>);
    case 'levels':
      return (<>
        {['inputBlack', 'inputWhite', 'gamma', 'outputBlack', 'outputWhite'].map(param => (
          <div className="control-row" key={param}>
            {renderKfToggle(param, effect.params[param] as number)}<label>{param.replace(/([A-Z])/g, ' $1').trim()}</label>
            <input type="range" min={param === 'gamma' ? 0.1 : 0} max={param === 'gamma' ? 10 : 1} step={param === 'gamma' ? 0.1 : 0.01}
              value={effect.params[param] as number} onChange={(e) => onChange({ ...effect.params, [param]: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu(param)} /><span className="value">{(effect.params[param] as number).toFixed(2)}</span>
          </div>
        ))}
      </>);
    case 'invert':
      return <p className="effect-info">No parameters</p>;
    default:
      return null;
  }
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
            <DraggableNumber value={mask.featherQuality ?? 50} onChange={(v) => updateMask(clipId, mask.id, { featherQuality: Math.max(1, Math.round(v)) })}
              defaultValue={50} sensitivity={1} decimals={0} /></div>
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
// MAIN PANEL
// ============================================
export function PropertiesPanel() {
  const { clips, tracks, selectedClipIds, playheadPosition, getInterpolatedTransform } = useTimelineStore();
  const [activeTab, setActiveTab] = useState<PropertiesTab>('transform');
  const [lastClipId, setLastClipId] = useState<string | null>(null);

  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;
  const selectedClip = clips.find(c => c.id === selectedClipId);

  // Check if it's an audio clip
  const selectedTrack = selectedClip ? tracks.find(t => t.id === selectedClip.trackId) : null;
  const isAudioClip = selectedTrack?.type === 'audio';

  // Reset tab when switching between audio/video clips
  useEffect(() => {
    if (selectedClipId && selectedClipId !== lastClipId) {
      setLastClipId(selectedClipId);
      // Set appropriate default tab based on clip type
      if (isAudioClip && (activeTab === 'transform' || activeTab === 'masks')) {
        setActiveTab('volume');
      } else if (!isAudioClip && activeTab === 'volume') {
        setActiveTab('transform');
      }
    }
  }, [selectedClipId, isAudioClip, lastClipId, activeTab]);

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
            <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>Transcript</button>
          </>
        ) : (
          <>
            <button className={`tab-btn ${activeTab === 'transform' ? 'active' : ''}`} onClick={() => setActiveTab('transform')}>Transform</button>
            <button className={`tab-btn ${activeTab === 'effects' ? 'active' : ''}`} onClick={() => setActiveTab('effects')}>
              Effects {visualEffects.length > 0 && <span className="badge">{visualEffects.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'masks' ? 'active' : ''}`} onClick={() => setActiveTab('masks')}>
              Masks {selectedClip.masks && selectedClip.masks.length > 0 && <span className="badge">{selectedClip.masks.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>Transcript</button>
            <button className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>Analysis</button>
          </>
        )}
      </div>

      <div className="properties-content">
        {activeTab === 'transform' && !isAudioClip && <TransformTab clipId={selectedClip.id} transform={transform} />}
        {activeTab === 'volume' && isAudioClip && <VolumeTab clipId={selectedClip.id} effects={selectedClip.effects || []} />}
        {activeTab === 'effects' && <EffectsTab clipId={selectedClip.id} effects={selectedClip.effects || []} />}
        {activeTab === 'masks' && !isAudioClip && <MasksTab clipId={selectedClip.id} masks={selectedClip.masks} />}
        {activeTab === 'transcript' && <TranscriptPanel />}
        {activeTab === 'analysis' && !isAudioClip && <AnalysisPanel />}
      </div>
    </div>
  );
}
