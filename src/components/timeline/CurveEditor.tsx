// Curve Editor component for keyframe animation curves with bezier handles

import React, { useCallback, useRef, useState, useMemo } from 'react';
import type { AnimatableProperty, Keyframe, BezierHandle, EasingType } from '../../types';
import { PRESET_BEZIER } from '../../utils/keyframeInterpolation';
import { CURVE_EDITOR_HEIGHT, BEZIER_HANDLE_SIZE } from '../../stores/timeline/constants';

export interface CurveEditorProps {
  trackId: string;
  clipId: string;
  property: AnimatableProperty;
  keyframes: Keyframe[];
  clipStartTime: number;
  clipDuration: number;
  width: number;
  scrollX: number;
  selectedKeyframeIds: Set<string>;
  onSelectKeyframe: (id: string, addToSelection: boolean) => void;
  onMoveKeyframe: (id: string, newTime: number, newValue: number) => void;
  onUpdateBezierHandle: (keyframeId: string, handle: 'in' | 'out', position: BezierHandle) => void;
  timeToPixel: (time: number) => number;
  pixelToTime: (pixel: number) => number;
}

// Get value range for Y axis based on property type
function getPropertyRange(property: AnimatableProperty): { min: number; max: number; step: number } {
  if (property === 'opacity') {
    return { min: 0, max: 1, step: 0.1 };
  }
  if (property.startsWith('scale.')) {
    return { min: 0, max: 2, step: 0.1 };
  }
  if (property.startsWith('rotation.')) {
    return { min: -360, max: 360, step: 15 };
  }
  if (property.startsWith('position.')) {
    return { min: -1000, max: 1000, step: 100 };
  }
  // Effect properties
  return { min: -100, max: 100, step: 10 };
}

// Compute auto-range based on keyframe values
function computeAutoRange(keyframes: Keyframe[], property: AnimatableProperty): { min: number; max: number } {
  const defaultRange = getPropertyRange(property);

  if (keyframes.length === 0) {
    return defaultRange;
  }

  const values = keyframes.map(k => k.value);
  let min = Math.min(...values);
  let max = Math.max(...values);

  // Add padding
  const range = max - min;
  const padding = range > 0 ? range * 0.2 : defaultRange.step;
  min -= padding;
  max += padding;

  // Ensure we have a reasonable range
  if (max - min < defaultRange.step) {
    const mid = (min + max) / 2;
    min = mid - defaultRange.step;
    max = mid + defaultRange.step;
  }

  return { min, max };
}

// Generate SVG path for bezier curve between two keyframes
function generateBezierPath(
  prevKf: Keyframe,
  nextKf: Keyframe,
  timeToX: (time: number) => number,
  valueToY: (value: number) => number
): string {
  const x1 = timeToX(prevKf.time);
  const y1 = valueToY(prevKf.value);
  const x2 = timeToX(nextKf.time);
  const y2 = valueToY(nextKf.value);

  const timeDelta = nextKf.time - prevKf.time;
  const valueDelta = nextKf.value - prevKf.value;

  // Get control points
  let cp1x: number, cp1y: number, cp2x: number, cp2y: number;

  if (prevKf.easing === 'bezier' || prevKf.handleOut || nextKf.handleIn) {
    // Use custom handles
    const handleOut = prevKf.handleOut || { x: timeDelta / 3, y: valueDelta / 3 };
    const handleIn = nextKf.handleIn || { x: -timeDelta / 3, y: -valueDelta / 3 };

    cp1x = timeToX(prevKf.time + handleOut.x);
    cp1y = valueToY(prevKf.value + handleOut.y);
    cp2x = timeToX(nextKf.time + handleIn.x);
    cp2y = valueToY(nextKf.value + handleIn.y);
  } else {
    // Use preset easing
    const preset = PRESET_BEZIER[prevKf.easing as Exclude<EasingType, 'bezier'>] || PRESET_BEZIER.linear;

    cp1x = x1 + (x2 - x1) * preset.p1[0];
    cp1y = y1 + (y2 - y1) * preset.p1[1];
    cp2x = x1 + (x2 - x1) * preset.p2[0];
    cp2y = y1 + (y2 - y1) * preset.p2[1];
  }

  return `M ${x1} ${y1} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`;
}

export const CurveEditor: React.FC<CurveEditorProps> = ({
  trackId: _trackId,
  clipId: _clipId,
  property,
  keyframes,
  clipStartTime,
  clipDuration,
  width,
  scrollX,
  selectedKeyframeIds,
  onSelectKeyframe,
  onMoveKeyframe,
  onUpdateBezierHandle,
  timeToPixel,
  pixelToTime,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragState, setDragState] = useState<{
    type: 'keyframe' | 'handle-in' | 'handle-out';
    keyframeId: string;
    startX: number;
    startY: number;
    startTime: number;
    startValue: number;
    startHandleX?: number;
    startHandleY?: number;
  } | null>(null);

  const height = CURVE_EDITOR_HEIGHT;
  const padding = { top: 20, right: 10, bottom: 20, left: 10 };

  // Sort keyframes by time
  const sortedKeyframes = useMemo(() =>
    [...keyframes].sort((a, b) => a.time - b.time),
    [keyframes]
  );

  // Compute value range
  const valueRange = useMemo(() =>
    computeAutoRange(sortedKeyframes, property),
    [sortedKeyframes, property]
  );

  // Convert time to X position
  const timeToX = useCallback((time: number) => {
    return timeToPixel(clipStartTime + time) - scrollX;
  }, [timeToPixel, clipStartTime, scrollX]);

  // Convert X position to time
  const xToTime = useCallback((x: number) => {
    return pixelToTime(x + scrollX) - clipStartTime;
  }, [pixelToTime, scrollX, clipStartTime]);

  // Convert value to Y position (inverted because SVG Y goes down)
  const valueToY = useCallback((value: number) => {
    const range = valueRange.max - valueRange.min;
    const normalized = (value - valueRange.min) / range;
    return height - padding.bottom - normalized * (height - padding.top - padding.bottom);
  }, [valueRange, height, padding]);

  // Convert Y position to value
  const yToValue = useCallback((y: number) => {
    const range = valueRange.max - valueRange.min;
    const normalized = (height - padding.bottom - y) / (height - padding.top - padding.bottom);
    return valueRange.min + normalized * range;
  }, [valueRange, height, padding]);

  // Generate grid lines
  const gridLines = useMemo(() => {
    const lines: { y: number; value: number; major: boolean }[] = [];
    const range = valueRange.max - valueRange.min;
    const defaultRange = getPropertyRange(property);
    const step = defaultRange.step;

    // Calculate nice step size
    const numLines = Math.ceil(range / step);
    const adjustedStep = numLines > 10 ? step * 2 : step;

    for (let value = Math.ceil(valueRange.min / adjustedStep) * adjustedStep; value <= valueRange.max; value += adjustedStep) {
      lines.push({
        y: valueToY(value),
        value,
        major: Math.abs(value) < 0.001 || value % (adjustedStep * 2) === 0,
      });
    }

    return lines;
  }, [valueRange, valueToY, property]);

  // Handle mouse down on keyframe
  const handleKeyframeMouseDown = useCallback((e: React.MouseEvent, kf: Keyframe) => {
    e.stopPropagation();

    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;

    onSelectKeyframe(kf.id, e.shiftKey);

    setDragState({
      type: 'keyframe',
      keyframeId: kf.id,
      startX: e.clientX,
      startY: e.clientY,
      startTime: kf.time,
      startValue: kf.value,
    });
  }, [onSelectKeyframe]);

  // Handle mouse down on bezier handle
  const handleHandleMouseDown = useCallback((e: React.MouseEvent, kf: Keyframe, handleType: 'in' | 'out') => {
    e.stopPropagation();

    const prevKf = sortedKeyframes.find((_k, i) =>
      i < sortedKeyframes.length - 1 && sortedKeyframes[i + 1].id === kf.id
    );
    const nextKf = sortedKeyframes.find((_k, i) =>
      i > 0 && sortedKeyframes[i - 1].id === kf.id
    );

    let handle: BezierHandle;
    if (handleType === 'in') {
      const defaultX = prevKf ? -(kf.time - prevKf.time) / 3 : -0.1;
      const defaultY = prevKf ? -(kf.value - prevKf.value) / 3 : 0;
      handle = kf.handleIn || { x: defaultX, y: defaultY };
    } else {
      const defaultX = nextKf ? (nextKf.time - kf.time) / 3 : 0.1;
      const defaultY = nextKf ? (nextKf.value - kf.value) / 3 : 0;
      handle = kf.handleOut || { x: defaultX, y: defaultY };
    }

    setDragState({
      type: handleType === 'in' ? 'handle-in' : 'handle-out',
      keyframeId: kf.id,
      startX: e.clientX,
      startY: e.clientY,
      startTime: kf.time,
      startValue: kf.value,
      startHandleX: handle.x,
      startHandleY: handle.y,
    });
  }, [sortedKeyframes]);

  // Handle mouse move
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragState) return;

    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (dragState.type === 'keyframe') {
      // Move keyframe
      let newTime = xToTime(x);
      let newValue = yToValue(y);

      // Constrain to horizontal or vertical if shift held
      if (e.shiftKey) {
        const dx = Math.abs(e.clientX - dragState.startX);
        const dy = Math.abs(e.clientY - dragState.startY);
        if (dx > dy) {
          newValue = dragState.startValue;
        } else {
          newTime = dragState.startTime;
        }
      }

      // Clamp time to clip duration
      newTime = Math.max(0, Math.min(newTime, clipDuration));

      onMoveKeyframe(dragState.keyframeId, newTime, newValue);
    } else {
      // Move handle
      const kf = sortedKeyframes.find(k => k.id === dragState.keyframeId);
      if (!kf) return;

      const handleTime = xToTime(x) - kf.time;
      const handleValue = yToValue(y) - kf.value;

      // Constrain handle direction
      const isIn = dragState.type === 'handle-in';
      const constrainedTime = isIn
        ? Math.min(0, handleTime)  // In handle must be <= 0
        : Math.max(0, handleTime); // Out handle must be >= 0

      onUpdateBezierHandle(dragState.keyframeId, isIn ? 'in' : 'out', {
        x: constrainedTime,
        y: handleValue,
      });
    }
  }, [dragState, xToTime, yToValue, clipDuration, onMoveKeyframe, onUpdateBezierHandle, sortedKeyframes]);

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setDragState(null);
  }, []);

  // Handle click on empty area
  const handleSvgClick = useCallback((e: React.MouseEvent) => {
    if (e.target === svgRef.current) {
      onSelectKeyframe('', false); // Deselect
    }
  }, [onSelectKeyframe]);

  return (
    <svg
      ref={svgRef}
      className="curve-editor-svg"
      width={width}
      height={height}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={handleSvgClick}
    >
      {/* Background */}
      <rect x={0} y={0} width={width} height={height} fill="var(--bg-tertiary)" />

      {/* Grid lines */}
      {gridLines.map((line, i) => (
        <g key={i}>
          <line
            x1={0}
            y1={line.y}
            x2={width}
            y2={line.y}
            className={line.major ? 'curve-editor-grid-major' : 'curve-editor-grid'}
          />
          {line.major && (
            <text
              x={4}
              y={line.y - 2}
              className="curve-editor-value-label"
              fontSize={9}
              fill="var(--text-secondary)"
            >
              {property === 'opacity' ? `${(line.value * 100).toFixed(0)}%` :
               property.startsWith('scale.') ? `${(line.value * 100).toFixed(0)}%` :
               property.startsWith('rotation.') ? `${line.value.toFixed(0)}°` :
               line.value.toFixed(0)}
            </text>
          )}
        </g>
      ))}

      {/* Zero line (if in range) */}
      {valueRange.min <= 0 && valueRange.max >= 0 && (
        <line
          x1={0}
          y1={valueToY(0)}
          x2={width}
          y2={valueToY(0)}
          stroke="var(--text-secondary)"
          strokeWidth={1}
          opacity={0.5}
        />
      )}

      {/* Curves between keyframes */}
      {sortedKeyframes.map((kf, i) => {
        if (i === 0) return null;
        const prevKf = sortedKeyframes[i - 1];
        const path = generateBezierPath(prevKf, kf, timeToX, valueToY);
        return (
          <path
            key={`curve-${prevKf.id}-${kf.id}`}
            d={path}
            className="curve-editor-curve"
          />
        );
      })}

      {/* Keyframe points and handles */}
      {sortedKeyframes.map((kf, i) => {
        const x = timeToX(kf.time);
        const y = valueToY(kf.value);
        const isSelected = selectedKeyframeIds.has(kf.id);

        // Calculate handle positions
        const prevKf = i > 0 ? sortedKeyframes[i - 1] : null;
        const nextKf = i < sortedKeyframes.length - 1 ? sortedKeyframes[i + 1] : null;

        let handleInX = 0, handleInY = 0, handleOutX = 0, handleOutY = 0;
        let showHandleIn = false, showHandleOut = false;

        if (isSelected) {
          if (prevKf) {
            const defaultInX = -(kf.time - prevKf.time) / 3;
            const defaultInY = -(kf.value - prevKf.value) / 3;
            const handleIn = kf.handleIn || { x: defaultInX, y: defaultInY };
            handleInX = timeToX(kf.time + handleIn.x);
            handleInY = valueToY(kf.value + handleIn.y);
            showHandleIn = true;
          }

          if (nextKf) {
            const defaultOutX = (nextKf.time - kf.time) / 3;
            const defaultOutY = (nextKf.value - kf.value) / 3;
            const handleOut = kf.handleOut || { x: defaultOutX, y: defaultOutY };
            handleOutX = timeToX(kf.time + handleOut.x);
            handleOutY = valueToY(kf.value + handleOut.y);
            showHandleOut = true;
          }
        }

        return (
          <g key={kf.id}>
            {/* Handle lines (only for selected keyframes) */}
            {showHandleIn && (
              <line
                x1={x}
                y1={y}
                x2={handleInX}
                y2={handleInY}
                className="curve-editor-handle-line"
              />
            )}
            {showHandleOut && (
              <line
                x1={x}
                y1={y}
                x2={handleOutX}
                y2={handleOutY}
                className="curve-editor-handle-line"
              />
            )}

            {/* Handle circles (only for selected keyframes) */}
            {showHandleIn && (
              <circle
                cx={handleInX}
                cy={handleInY}
                r={BEZIER_HANDLE_SIZE / 2}
                className="curve-editor-handle"
                onMouseDown={(e) => handleHandleMouseDown(e, kf, 'in')}
              />
            )}
            {showHandleOut && (
              <circle
                cx={handleOutX}
                cy={handleOutY}
                r={BEZIER_HANDLE_SIZE / 2}
                className="curve-editor-handle"
                onMouseDown={(e) => handleHandleMouseDown(e, kf, 'out')}
              />
            )}

            {/* Keyframe point */}
            <circle
              cx={x}
              cy={y}
              r={5}
              className={`curve-editor-keyframe ${isSelected ? 'selected' : ''}`}
              onMouseDown={(e) => handleKeyframeMouseDown(e, kf)}
            />
          </g>
        );
      })}
    </svg>
  );
};

export default CurveEditor;
