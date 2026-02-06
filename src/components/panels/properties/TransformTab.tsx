// Transform Tab - Position, Scale, Rotation, Opacity controls (AE-style values)
import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import type { BlendMode, AnimatableProperty } from '../../../types';
import {
  KeyframeToggle,
  ScaleKeyframeToggle,
  DraggableNumber,
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
  const { setPropertyValue, updateClipTransform } = useTimelineStore.getState();

  // Get composition dimensions for pixel conversion
  const activeComp = useMediaStore.getState().getActiveComposition();
  const compWidth = activeComp?.width || 1920;
  const compHeight = activeComp?.height || 1080;

  const handlePropertyChange = (property: AnimatableProperty, value: number) => {
    setPropertyValue(clipId, property, value);
  };

  // Position: normalized (-1..1) → pixels, where 0 = center
  const posXPx = transform.position.x * (compWidth / 2);
  const posYPx = transform.position.y * (compHeight / 2);
  const posZPx = transform.position.z * (compWidth / 2); // Z uses width as reference

  const handlePosXChange = (px: number) => handlePropertyChange('position.x', px / (compWidth / 2));
  const handlePosYChange = (px: number) => handlePropertyChange('position.y', px / (compHeight / 2));
  const handlePosZChange = (px: number) => handlePropertyChange('position.z', px / (compWidth / 2));

  // Scale: multiplier → percentage
  const scaleXPct = transform.scale.x * 100;
  const scaleYPct = transform.scale.y * 100;
  const uniformScalePct = ((transform.scale.x + transform.scale.y) / 2) * 100;

  const handleScaleXChange = (pct: number) => handlePropertyChange('scale.x', pct / 100);
  const handleScaleYChange = (pct: number) => handlePropertyChange('scale.y', pct / 100);
  const handleUniformScaleChange = (pct: number) => {
    const v = pct / 100;
    handlePropertyChange('scale.x', v);
    handlePropertyChange('scale.y', v);
  };

  // Opacity: 0-1 → percentage
  const opacityPct = transform.opacity * 100;
  const handleOpacityChange = (pct: number) => handlePropertyChange('opacity', Math.max(0, Math.min(100, pct)) / 100);

  // Speed: multiplier → percentage
  const speedPct = speed * 100;
  const handleSpeedChange = (pct: number) => handlePropertyChange('speed', pct / 100);

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
          <DraggableNumber value={opacityPct} onChange={handleOpacityChange}
            defaultValue={100} decimals={1} suffix="%" min={0} max={100} sensitivity={1} />
        </div>
      </div>

      {/* Time/Speed */}
      <div className="properties-section">
        <h4>Time</h4>
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="speed" value={speed} />
          <label>Speed</label>
          <DraggableNumber value={speedPct} onChange={handleSpeedChange}
            defaultValue={100} decimals={0} suffix="%" min={-400} max={400} sensitivity={1} />
        </div>
      </div>

      {/* Position */}
      <div className="properties-section">
        <h4>Position</h4>
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="position.x" value={transform.position.x} />
          <label>X</label>
          <DraggableNumber value={posXPx} onChange={handlePosXChange}
            defaultValue={0} decimals={1} suffix=" px" sensitivity={0.5} />
        </div>
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="position.y" value={transform.position.y} />
          <label>Y</label>
          <DraggableNumber value={posYPx} onChange={handlePosYChange}
            defaultValue={0} decimals={1} suffix=" px" sensitivity={0.5} />
        </div>
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="position.z" value={transform.position.z} />
          <label>Z</label>
          <DraggableNumber value={posZPx} onChange={handlePosZChange}
            defaultValue={0} decimals={1} suffix=" px" sensitivity={0.5} />
        </div>
      </div>

      {/* Scale */}
      <div className="properties-section">
        <h4>Scale</h4>
        <div className="control-row">
          <ScaleKeyframeToggle clipId={clipId} scaleX={transform.scale.x} scaleY={transform.scale.y} />
          <label>Uniform</label>
          <DraggableNumber value={uniformScalePct} onChange={handleUniformScaleChange}
            defaultValue={100} decimals={1} suffix="%" min={1} sensitivity={1} />
        </div>
        {(['x', 'y'] as const).map(axis => (
          <div className="control-row" key={axis}>
            <span className="keyframe-toggle-placeholder" />
            <label>{axis.toUpperCase()}</label>
            <DraggableNumber
              value={axis === 'x' ? scaleXPct : scaleYPct}
              onChange={axis === 'x' ? handleScaleXChange : handleScaleYChange}
              defaultValue={100} decimals={1} suffix="%" min={1} sensitivity={1}
            />
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
            <DraggableNumber value={transform.rotation[axis]} onChange={(v) => handlePropertyChange(`rotation.${axis}`, v)}
              defaultValue={0} decimals={1} suffix="°" min={-180} max={180} sensitivity={0.5} />
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
