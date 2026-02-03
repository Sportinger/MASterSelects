// Transform Tab - Position, Scale, Rotation, Opacity controls
import { useTimelineStore } from '../../../stores/timeline';
import type { BlendMode, AnimatableProperty } from '../../../types';
import {
  KeyframeToggle,
  ScaleKeyframeToggle,
  PrecisionSlider,
  BLEND_MODE_GROUPS,
  formatBlendModeName,
} from './shared';

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

export function TransformTab({ clipId, transform, speed = 1 }: TransformTabProps) {
  // Use getState() for actions - they're stable and don't need subscriptions
  const { setPropertyValue, updateClipTransform } = useTimelineStore.getState();

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
