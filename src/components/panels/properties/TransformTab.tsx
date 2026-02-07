// Transform Tab - Position, Scale, Rotation, Opacity controls (AE-style compact layout)
import { useCallback } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import { startBatch, endBatch } from '../../../stores/historyStore';
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

// Labeled value cell: tiny axis label + draggable number
function LabeledValue({ label, ...props }: { label: string } & React.ComponentProps<typeof DraggableNumber>) {
  return (
    <div className="labeled-value">
      <span className="labeled-value-label">{label}</span>
      <DraggableNumber {...props} />
    </div>
  );
}

export function TransformTab({ clipId, transform, speed = 1 }: TransformTabProps) {
  const { setPropertyValue, updateClipTransform } = useTimelineStore.getState();

  const handleBatchStart = useCallback(() => startBatch('Adjust transform'), []);
  const handleBatchEnd = useCallback(() => endBatch(), []);

  const activeComp = useMediaStore.getState().getActiveComposition();
  const compWidth = activeComp?.width || 1920;
  const compHeight = activeComp?.height || 1080;

  const handlePropertyChange = (property: AnimatableProperty, value: number) => {
    setPropertyValue(clipId, property, value);
  };

  // Position: normalized → pixels
  const posXPx = transform.position.x * (compWidth / 2);
  const posYPx = transform.position.y * (compHeight / 2);
  const posZPx = transform.position.z * (compWidth / 2);
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

  const opacityPct = transform.opacity * 100;
  const handleOpacityChange = (pct: number) => handlePropertyChange('opacity', Math.max(0, Math.min(100, pct)) / 100);
  const speedPct = speed * 100;
  const handleSpeedChange = (pct: number) => handlePropertyChange('speed', pct / 100);

  return (
    <div className="properties-tab-content transform-tab-compact">
      {/* Appearance + Time */}
      <div className="properties-section">
        <div className="control-row">
          <label className="prop-label">Blend</label>
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
          <label className="prop-label">Opacity</label>
          <DraggableNumber value={opacityPct} onChange={handleOpacityChange}
            defaultValue={100} decimals={1} suffix="%" min={0} max={100} sensitivity={1}
            onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
        </div>
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="speed" value={speed} />
          <label className="prop-label">Speed</label>
          <DraggableNumber value={speedPct} onChange={handleSpeedChange}
            defaultValue={100} decimals={0} suffix="%" min={-400} max={400} sensitivity={1}
            onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
        </div>
      </div>

      {/* Position */}
      <div className="properties-section">
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="position.x" value={transform.position.x} />
          <label className="prop-label">Position</label>
          <div className="multi-value-row">
            <LabeledValue label="X" value={posXPx} onChange={handlePosXChange}
              defaultValue={0} decimals={1} sensitivity={0.5}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
            <LabeledValue label="Y" value={posYPx} onChange={handlePosYChange}
              defaultValue={0} decimals={1} sensitivity={0.5}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
            <LabeledValue label="Z" value={posZPx} onChange={handlePosZChange}
              defaultValue={0} decimals={1} sensitivity={0.5}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
          </div>
        </div>
      </div>

      {/* Scale */}
      <div className="properties-section">
        <div className="control-row">
          <ScaleKeyframeToggle clipId={clipId} scaleX={transform.scale.x} scaleY={transform.scale.y} />
          <label className="prop-label">Scale</label>
          <div className="multi-value-row">
            <LabeledValue label="All" value={uniformScalePct} onChange={handleUniformScaleChange}
              defaultValue={100} decimals={1} suffix="%" min={1} sensitivity={1}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
            <LabeledValue label="X" value={scaleXPct} onChange={handleScaleXChange}
              defaultValue={100} decimals={1} suffix="%" min={1} sensitivity={1}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
            <LabeledValue label="Y" value={scaleYPct} onChange={handleScaleYChange}
              defaultValue={100} decimals={1} suffix="%" min={1} sensitivity={1}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
          </div>
        </div>
      </div>

      {/* Rotation */}
      <div className="properties-section">
        <div className="control-row">
          <KeyframeToggle clipId={clipId} property="rotation.z" value={transform.rotation.z} />
          <label className="prop-label">Rotation</label>
          <div className="multi-value-row">
            <LabeledValue label="X" value={transform.rotation.x} onChange={(v) => handlePropertyChange('rotation.x', v)}
              defaultValue={0} decimals={1} suffix="°" min={-180} max={180} sensitivity={0.5}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
            <LabeledValue label="Y" value={transform.rotation.y} onChange={(v) => handlePropertyChange('rotation.y', v)}
              defaultValue={0} decimals={1} suffix="°" min={-180} max={180} sensitivity={0.5}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
            <LabeledValue label="Z" value={transform.rotation.z} onChange={(v) => handlePropertyChange('rotation.z', v)}
              defaultValue={0} decimals={1} suffix="°" min={-180} max={180} sensitivity={0.5}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} />
          </div>
        </div>
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
