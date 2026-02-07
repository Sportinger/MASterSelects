// Effects Tab - Add and configure visual effects
import { useState, useMemo, useCallback } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { startBatch, endBatch } from '../../../stores/historyStore';
import type { AnimatableProperty } from '../../../types';
import { EFFECT_REGISTRY, getDefaultParams, getCategoriesWithEffects } from '../../../effects';
import { EffectKeyframeToggle, DraggableNumber } from './shared';

// Single parameter control renderer
function renderParamControl(
  paramName: string,
  paramDef: { type: string; label: string; default: number | boolean | string; min?: number; max?: number; step?: number; options?: { value: string; label: string }[]; animatable?: boolean },
  value: number | boolean | string,
  effect: { id: string; params: Record<string, number | boolean | string> },
  onChange: (params: Record<string, number | boolean | string>) => void,
  defaults: Record<string, number | boolean | string>,
  clipId?: string,
  noMaxLimit?: boolean,
  onDragStart?: () => void,
  onDragEnd?: () => void,
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
            onDragStart={onDragStart}
            onDragEnd={onDragEnd}
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
  onDragStart?: () => void;
  onDragEnd?: () => void;
}

function EffectParams({ effect, onChange, clipId, onDragStart, onDragEnd }: EffectParamsProps) {
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
        return renderParamControl(paramName, paramDef, value, effect, onChange, defaults, clipId, false, onDragStart, onDragEnd);
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
                return renderParamControl(paramName, paramDef, value, effect, onChange, defaults, clipId, true, onDragStart, onDragEnd);
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

interface EffectsTabProps {
  clipId: string;
  effects: Array<{ id: string; name: string; type: string; enabled: boolean; params: Record<string, number | boolean | string> }>;
}

export function EffectsTab({ clipId, effects }: EffectsTabProps) {
  // Reactive data - subscribe to specific values only
  const playheadPosition = useTimelineStore(state => state.playheadPosition);
  const clips = useTimelineStore(state => state.clips);
  // Actions from getState() - stable, no subscription needed
  const { addClipEffect, removeClipEffect, updateClipEffect, setClipEffectEnabled, setPropertyValue, getInterpolatedEffects } = useTimelineStore.getState();

  const handleBatchStart = useCallback(() => startBatch('Adjust effect'), []);
  const handleBatchEnd = useCallback(() => endBatch(), []);
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
                    onDragStart={handleBatchStart}
                    onDragEnd={handleBatchEnd}
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
