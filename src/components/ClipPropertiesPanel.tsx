// Clip Properties Panel - Shows transform controls for selected timeline clip

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
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={transform.opacity}
              onChange={(e) => updateTransform({ opacity: parseFloat(e.target.value) })}
            />
            <span className="value">{Math.round(transform.opacity * 100)}%</span>
          </div>
        </div>

        {/* Scale */}
        <div className="properties-section">
          <h4>Scale</h4>
          <div className="control-row">
            <label>X</label>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.01"
              value={transform.scale.x}
              onChange={(e) => updateTransform({ scale: { x: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.scale.x.toFixed(2)}</span>
          </div>
          <div className="control-row">
            <label>Y</label>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.01"
              value={transform.scale.y}
              onChange={(e) => updateTransform({ scale: { y: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.scale.y.toFixed(2)}</span>
          </div>
        </div>

        {/* Position */}
        <div className="properties-section">
          <h4>Position</h4>
          <div className="control-row">
            <label>X</label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={transform.position.x}
              onChange={(e) => updateTransform({ position: { x: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.position.x.toFixed(2)}</span>
          </div>
          <div className="control-row">
            <label>Y</label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={transform.position.y}
              onChange={(e) => updateTransform({ position: { y: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.position.y.toFixed(2)}</span>
          </div>
          <div className="control-row">
            <label>Z</label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={transform.position.z}
              onChange={(e) => updateTransform({ position: { z: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.position.z.toFixed(2)}</span>
          </div>
        </div>

        {/* Rotation */}
        <div className="properties-section">
          <h4>Rotation</h4>
          <div className="control-row">
            <label>X</label>
            <input
              type="range"
              min="-180"
              max="180"
              step="1"
              value={transform.rotation.x}
              onChange={(e) => updateTransform({ rotation: { x: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.rotation.x.toFixed(0)}°</span>
          </div>
          <div className="control-row">
            <label>Y</label>
            <input
              type="range"
              min="-180"
              max="180"
              step="1"
              value={transform.rotation.y}
              onChange={(e) => updateTransform({ rotation: { y: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.rotation.y.toFixed(0)}°</span>
          </div>
          <div className="control-row">
            <label>Z</label>
            <input
              type="range"
              min="-180"
              max="180"
              step="1"
              value={transform.rotation.z}
              onChange={(e) => updateTransform({ rotation: { z: parseFloat(e.target.value) } })}
            />
            <span className="value">{transform.rotation.z.toFixed(0)}°</span>
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
