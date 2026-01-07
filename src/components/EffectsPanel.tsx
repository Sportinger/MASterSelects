// Effects panel component

import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timelineStore';
import type { EffectType, BlendMode, Effect } from '../types';

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

// Default values for effect parameters (for right-click reset)
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

export function EffectsPanel() {
  // Mixer store (for live mixing layers)
  const { layers, selectedLayerId, addEffect: addLayerEffect, removeEffect: removeLayerEffect, updateEffect: updateLayerEffect, setLayerOpacity, setLayerBlendMode } =
    useMixerStore();

  // Timeline store (for timeline clips)
  const { clips, selectedClipId, addClipEffect, removeClipEffect, updateClipEffect, updateClipTransform } = useTimelineStore();

  // Check if a timeline clip is selected first
  const selectedClip = clips.find((c) => c.id === selectedClipId);
  const selectedLayer = layers.find((l) => l?.id === selectedLayerId);

  // Timeline clip takes priority
  if (selectedClip) {
    return (
      <div className="effects-panel">
        <div className="panel-header">
          <div className="effect-add">
            <select
              onChange={(e) => {
                if (e.target.value) {
                  addClipEffect(selectedClip.id, e.target.value);
                  e.target.value = '';
                }
              }}
              defaultValue=""
            >
              <option value="" disabled>
                + Add Effect
              </option>
              {AVAILABLE_EFFECTS.map((effect) => (
                <option key={effect.type} value={effect.type}>
                  {effect.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="layer-settings">
          <div className="control-row">
            <label>Opacity</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={selectedClip.transform.opacity}
              onChange={(e) => updateClipTransform(selectedClip.id, { opacity: parseFloat(e.target.value) })}
            />
            <span className="value">{(selectedClip.transform.opacity * 100).toFixed(0)}%</span>
          </div>

          <div className="control-row">
            <label>Blend Mode</label>
            <select
              value={selectedClip.transform.blendMode}
              onChange={(e) => updateClipTransform(selectedClip.id, { blendMode: e.target.value as BlendMode })}
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
        </div>

        {selectedClip.effects && selectedClip.effects.length > 0 ? (
          <div className="effects-list">
            {selectedClip.effects.map((effect) => (
              <div key={effect.id} className="effect-item">
                <div className="effect-header">
                  <span className="effect-name">{effect.name}</span>
                  <button
                    className="btn btn-sm btn-danger"
                    onClick={() => removeClipEffect(selectedClip.id, effect.id)}
                  >
                    ×
                  </button>
                </div>

                <div className="effect-params">
                  {renderEffectParams(effect, (params) =>
                    updateClipEffect(selectedClip.id, effect.id, params)
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="panel-empty">
            <p>No effects applied</p>
          </div>
        )}
      </div>
    );
  }

  // Fall back to mixer layer
  if (!selectedLayer) {
    return (
      <div className="effects-panel">
        <div className="panel-empty">
          <p>Select a layer or clip to add effects</p>
        </div>
      </div>
    );
  }

  return (
    <div className="effects-panel">
      <div className="panel-header">
        <div className="effect-add">
          <select
            onChange={(e) => {
              if (e.target.value) {
                addLayerEffect(selectedLayer.id, e.target.value);
                e.target.value = '';
              }
            }}
            defaultValue=""
          >
            <option value="" disabled>
              + Add Effect
            </option>
            {AVAILABLE_EFFECTS.map((effect) => (
              <option key={effect.type} value={effect.type}>
                {effect.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="layer-settings">
        <div className="control-row">
          <label>Opacity</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={selectedLayer.opacity}
            onChange={(e) => setLayerOpacity(selectedLayer.id, parseFloat(e.target.value))}
          />
          <span>{Math.round(selectedLayer.opacity * 100)}%</span>
        </div>
        <div className="control-row">
          <label>Blend</label>
          <select
            value={selectedLayer.blendMode}
            onChange={(e) => setLayerBlendMode(selectedLayer.id, e.target.value as BlendMode)}
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
      </div>

      <div className="effects-list">
        {selectedLayer.effects.length === 0 ? (
          <div className="panel-empty">
            <p>No effects added</p>
          </div>
        ) : (
          selectedLayer.effects.map((effect) => (
            <div key={effect.id} className="effect-item">
              <div className="effect-header">
                <span className="effect-name">{effect.name}</span>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => removeLayerEffect(selectedLayer.id, effect.id)}
                >
                  ×
                </button>
              </div>

              <div className="effect-params">
                {renderEffectParams(effect, (params) =>
                  updateLayerEffect(selectedLayer.id, effect.id, params)
                )}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="transform-section">
        <h4>Transform</h4>
        <div className="control-row">
          <label>Position X</label>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.01"
            value={selectedLayer.position.x}
            onChange={(e) =>
              useMixerStore.getState().setLayerTransform(selectedLayer.id, {
                position: { x: parseFloat(e.target.value), y: selectedLayer.position.y },
              })
            }
          />
          <span>{selectedLayer.position.x.toFixed(2)}</span>
        </div>
        <div className="control-row">
          <label>Position Y</label>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.01"
            value={selectedLayer.position.y}
            onChange={(e) =>
              useMixerStore.getState().setLayerTransform(selectedLayer.id, {
                position: { x: selectedLayer.position.x, y: parseFloat(e.target.value) },
              })
            }
          />
          <span>{selectedLayer.position.y.toFixed(2)}</span>
        </div>
        <div className="control-row">
          <label>Scale</label>
          <input
            type="range"
            min="0.1"
            max="3"
            step="0.01"
            value={selectedLayer.scale.x}
            onChange={(e) => {
              const scale = parseFloat(e.target.value);
              useMixerStore.getState().setLayerTransform(selectedLayer.id, {
                scale: { x: scale, y: scale },
              });
            }}
          />
          <span>{selectedLayer.scale.x.toFixed(2)}</span>
        </div>
        <div className="control-row">
          <label>Rotation</label>
          <input
            type="range"
            min="0"
            max={Math.PI * 2}
            step="0.01"
            value={selectedLayer.rotation}
            onChange={(e) =>
              useMixerStore.getState().setLayerTransform(selectedLayer.id, {
                rotation: parseFloat(e.target.value),
              })
            }
          />
          <span>{Math.round((selectedLayer.rotation * 180) / Math.PI)}°</span>
        </div>
      </div>
    </div>
  );
}

// Helper to reset a single parameter to default
function resetToDefault(
  effectType: string,
  paramName: string,
  currentParams: Record<string, number | boolean | string>,
  onChange: (params: Record<string, number | boolean | string>) => void
) {
  const defaults = EFFECT_DEFAULTS[effectType] || {};
  const defaultValue = defaults[paramName];
  if (defaultValue !== undefined) {
    onChange({ ...currentParams, [paramName]: defaultValue });
  }
}

function renderEffectParams(
  effect: { type: string; params: Record<string, number | boolean | string> },
  onChange: (params: Record<string, number | boolean | string>) => void
) {
  const handleContextMenu = (paramName: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    resetToDefault(effect.type, paramName, effect.params, onChange);
  };

  switch (effect.type) {
    case 'hue-shift':
      return (
        <div className="control-row">
          <label>Shift</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={effect.params.shift as number}
            onChange={(e) => onChange({ shift: parseFloat(e.target.value) })}
            onContextMenu={handleContextMenu('shift')}
            title="Right-click to reset"
          />
        </div>
      );

    case 'brightness':
      return (
        <div className="control-row">
          <label>Amount</label>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.01"
            value={effect.params.amount as number}
            onChange={(e) => onChange({ amount: parseFloat(e.target.value) })}
            onContextMenu={handleContextMenu('amount')}
            title="Right-click to reset"
          />
        </div>
      );

    case 'contrast':
      return (
        <div className="control-row">
          <label>Amount</label>
          <input
            type="range"
            min="0"
            max="3"
            step="0.01"
            value={effect.params.amount as number}
            onChange={(e) => onChange({ amount: parseFloat(e.target.value) })}
            onContextMenu={handleContextMenu('amount')}
            title="Right-click to reset"
          />
        </div>
      );

    case 'saturation':
      return (
        <div className="control-row">
          <label>Amount</label>
          <input
            type="range"
            min="0"
            max="3"
            step="0.01"
            value={effect.params.amount as number}
            onChange={(e) => onChange({ amount: parseFloat(e.target.value) })}
            onContextMenu={handleContextMenu('amount')}
            title="Right-click to reset"
          />
        </div>
      );

    case 'pixelate':
      return (
        <div className="control-row">
          <label>Size</label>
          <input
            type="range"
            min="1"
            max="64"
            step="1"
            value={effect.params.size as number}
            onChange={(e) => onChange({ size: parseInt(e.target.value, 10) })}
            onContextMenu={handleContextMenu('size')}
            title="Right-click to reset"
          />
        </div>
      );

    case 'kaleidoscope':
      return (
        <>
          <div className="control-row">
            <label>Segments</label>
            <input
              type="range"
              min="2"
              max="16"
              step="1"
              value={effect.params.segments as number}
              onChange={(e) => onChange({ ...effect.params, segments: parseInt(e.target.value, 10) })}
              onContextMenu={handleContextMenu('segments')}
              title="Right-click to reset"
            />
          </div>
          <div className="control-row">
            <label>Rotation</label>
            <input
              type="range"
              min="0"
              max={Math.PI * 2}
              step="0.01"
              value={effect.params.rotation as number}
              onChange={(e) => onChange({ ...effect.params, rotation: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('rotation')}
              title="Right-click to reset"
            />
          </div>
        </>
      );

    case 'mirror':
      return (
        <>
          <div className="control-row">
            <label>
              <input
                type="checkbox"
                checked={effect.params.horizontal as boolean}
                onChange={(e) => onChange({ ...effect.params, horizontal: e.target.checked })}
              />
              Horizontal
            </label>
          </div>
          <div className="control-row">
            <label>
              <input
                type="checkbox"
                checked={effect.params.vertical as boolean}
                onChange={(e) => onChange({ ...effect.params, vertical: e.target.checked })}
              />
              Vertical
            </label>
          </div>
        </>
      );

    case 'rgb-split':
      return (
        <>
          <div className="control-row">
            <label>Amount</label>
            <input
              type="range"
              min="0"
              max="0.1"
              step="0.001"
              value={effect.params.amount as number}
              onChange={(e) => onChange({ ...effect.params, amount: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('amount')}
              title="Right-click to reset"
            />
          </div>
          <div className="control-row">
            <label>Angle</label>
            <input
              type="range"
              min="0"
              max={Math.PI * 2}
              step="0.01"
              value={effect.params.angle as number}
              onChange={(e) => onChange({ ...effect.params, angle: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('angle')}
              title="Right-click to reset"
            />
          </div>
        </>
      );

    case 'levels':
      return (
        <>
          <div className="control-row">
            <label>Input Black</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={effect.params.inputBlack as number}
              onChange={(e) => onChange({ ...effect.params, inputBlack: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('inputBlack')}
              title="Right-click to reset"
            />
            <span className="value">{(effect.params.inputBlack as number).toFixed(2)}</span>
          </div>
          <div className="control-row">
            <label>Input White</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={effect.params.inputWhite as number}
              onChange={(e) => onChange({ ...effect.params, inputWhite: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('inputWhite')}
              title="Right-click to reset"
            />
            <span className="value">{(effect.params.inputWhite as number).toFixed(2)}</span>
          </div>
          <div className="control-row">
            <label>Gamma</label>
            <input
              type="range"
              min="0.1"
              max="10"
              step="0.1"
              value={effect.params.gamma as number}
              onChange={(e) => onChange({ ...effect.params, gamma: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('gamma')}
              title="Right-click to reset"
            />
            <span className="value">{(effect.params.gamma as number).toFixed(2)}</span>
          </div>
          <div className="control-row">
            <label>Output Black</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={effect.params.outputBlack as number}
              onChange={(e) => onChange({ ...effect.params, outputBlack: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('outputBlack')}
              title="Right-click to reset"
            />
            <span className="value">{(effect.params.outputBlack as number).toFixed(2)}</span>
          </div>
          <div className="control-row">
            <label>Output White</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={effect.params.outputWhite as number}
              onChange={(e) => onChange({ ...effect.params, outputWhite: parseFloat(e.target.value) })}
              onContextMenu={handleContextMenu('outputWhite')}
              title="Right-click to reset"
            />
            <span className="value">{(effect.params.outputWhite as number).toFixed(2)}</span>
          </div>
        </>
      );

    case 'invert':
      return <p className="effect-info">No parameters</p>;

    default:
      return null;
  }
}
