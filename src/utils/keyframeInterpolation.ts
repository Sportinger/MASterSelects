import type { Keyframe, EasingType, AnimatableProperty, ClipTransform, BezierHandle } from '../types';

// Preset easing functions (for non-bezier easing types)
export const easingFunctions: Record<Exclude<EasingType, 'bezier'>, (t: number) => number> = {
  'linear': (t) => t,
  'ease-in': (t) => t * t,
  'ease-out': (t) => t * (2 - t),
  'ease-in-out': (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
};

// Standard CSS-equivalent bezier control points for presets
export const PRESET_BEZIER: Record<Exclude<EasingType, 'bezier'>, { p1: [number, number]; p2: [number, number] }> = {
  'linear': { p1: [0, 0], p2: [1, 1] },
  'ease-in': { p1: [0.42, 0], p2: [1, 1] },
  'ease-out': { p1: [0, 0], p2: [0.58, 1] },
  'ease-in-out': { p1: [0.42, 0], p2: [0.58, 1] },
};

/**
 * Solve cubic bezier for Y given X (time) using Newton-Raphson iteration.
 * Uses standard CSS cubic-bezier format where X controls timing and Y controls output.
 *
 * @param targetX - The input value (normalized time, 0-1)
 * @param p1x - First control point X
 * @param p1y - First control point Y
 * @param p2x - Second control point X
 * @param p2y - Second control point Y
 * @param epsilon - Precision for Newton-Raphson iteration
 * @returns The output value (eased time, 0-1)
 */
export function solveCubicBezierForX(
  targetX: number,
  p1x: number, p1y: number,
  p2x: number, p2y: number,
  epsilon: number = 0.0001
): number {
  // Edge cases
  if (targetX <= 0) return 0;
  if (targetX >= 1) return 1;

  // Bezier coefficients for X
  const cx = 3 * p1x;
  const bx = 3 * (p2x - p1x) - cx;
  const ax = 1 - cx - bx;

  // Bezier coefficients for Y
  const cy = 3 * p1y;
  const by = 3 * (p2y - p1y) - cy;
  const ay = 1 - cy - by;

  // Newton-Raphson iteration to find t where bezier_x(t) = targetX
  let t = targetX;  // Initial guess

  for (let i = 0; i < 10; i++) {
    const x = ((ax * t + bx) * t + cx) * t;
    const dx = (3 * ax * t + 2 * bx) * t + cx;

    const diff = x - targetX;
    if (Math.abs(diff) < epsilon) break;

    if (dx === 0) break;  // Avoid division by zero
    t -= diff / dx;
    t = Math.max(0, Math.min(1, t));  // Clamp to [0, 1]
  }

  // Evaluate Y at solved t
  return ((ay * t + by) * t + cy) * t;
}

/**
 * Interpolate between two keyframes using custom bezier handles.
 * The handles define the curve shape between the keyframes.
 *
 * @param prevKey - The keyframe at the start of the segment
 * @param nextKey - The keyframe at the end of the segment
 * @param t - Normalized time (0-1) between the two keyframes
 * @returns Interpolated value
 */
export function interpolateBezier(
  prevKey: Keyframe,
  nextKey: Keyframe,
  t: number
): number {
  const timeDelta = nextKey.time - prevKey.time;
  const valueDelta = nextKey.value - prevKey.value;

  // If no time difference, return target value
  if (timeDelta <= 0) return nextKey.value;

  // Default handles if not specified (equivalent to linear)
  const handleOut = prevKey.handleOut || { x: timeDelta / 3, y: valueDelta / 3 };
  const handleIn = nextKey.handleIn || { x: -timeDelta / 3, y: -valueDelta / 3 };

  // Convert relative handles to normalized 0-1 control points
  // handleOut.x is seconds from prevKey, convert to 0-1 range
  // handleOut.y is value offset from prevKey.value
  const p1x = Math.max(0, Math.min(1, handleOut.x / timeDelta));
  const p1y = valueDelta !== 0 ? handleOut.y / valueDelta : 0;

  // handleIn.x is seconds from nextKey (negative), convert to 0-1 range
  const p2x = Math.max(0, Math.min(1, 1 + handleIn.x / timeDelta));
  const p2y = valueDelta !== 0 ? 1 + handleIn.y / valueDelta : 1;

  // Solve bezier to get eased t
  const easedT = solveCubicBezierForX(t, p1x, p1y, p2x, p2y);

  // Return interpolated value
  return prevKey.value + valueDelta * easedT;
}

/**
 * Convert a preset easing type to bezier handles for a specific keyframe segment.
 * Useful when user wants to customize an existing preset.
 */
export function convertPresetToBezierHandles(
  easing: Exclude<EasingType, 'bezier'>,
  timeDelta: number,
  valueDelta: number
): { handleOut: BezierHandle; handleIn: BezierHandle } {
  const preset = PRESET_BEZIER[easing];
  return {
    handleOut: {
      x: preset.p1[0] * timeDelta,
      y: preset.p1[1] * valueDelta,
    },
    handleIn: {
      x: (preset.p2[0] - 1) * timeDelta,
      y: (preset.p2[1] - 1) * valueDelta,
    },
  };
}

// Get interpolated value for a single property at a given time
export function interpolateKeyframes(
  keyframes: Keyframe[],
  property: AnimatableProperty,
  time: number,
  defaultValue: number
): number {
  // Filter keyframes for this property and sort by time
  const propKeyframes = keyframes
    .filter(k => k.property === property)
    .sort((a, b) => a.time - b.time);

  // No keyframes - return default
  if (propKeyframes.length === 0) return defaultValue;

  // Single keyframe - return its value
  if (propKeyframes.length === 1) return propKeyframes[0].value;

  // Before first keyframe - return first value
  if (time <= propKeyframes[0].time) return propKeyframes[0].value;

  // After last keyframe - return last value
  const lastKeyframe = propKeyframes[propKeyframes.length - 1];
  if (time >= lastKeyframe.time) return lastKeyframe.value;

  // Find surrounding keyframes
  let prevKey = propKeyframes[0];
  let nextKey = propKeyframes[1];

  for (let i = 1; i < propKeyframes.length; i++) {
    if (propKeyframes[i].time >= time) {
      prevKey = propKeyframes[i - 1];
      nextKey = propKeyframes[i];
      break;
    }
  }

  // Calculate interpolation factor (0 to 1)
  const range = nextKey.time - prevKey.time;
  const localTime = time - prevKey.time;
  const t = range > 0 ? localTime / range : 0;

  // Use bezier interpolation if easing is 'bezier' or if keyframe has custom handles
  if (prevKey.easing === 'bezier' || prevKey.handleOut || nextKey.handleIn) {
    return interpolateBezier(prevKey, nextKey, t);
  }

  // Apply preset easing from the previous keyframe
  const easedT = easingFunctions[prevKey.easing](t);

  // Linear interpolation between values
  return prevKey.value + (nextKey.value - prevKey.value) * easedT;
}

// Get full interpolated transform at a given time
export function getInterpolatedClipTransform(
  keyframes: Keyframe[],
  time: number,
  baseTransform: ClipTransform
): ClipTransform {
  return {
    opacity: interpolateKeyframes(keyframes, 'opacity', time, baseTransform.opacity),
    blendMode: baseTransform.blendMode, // Not animatable
    position: {
      x: interpolateKeyframes(keyframes, 'position.x', time, baseTransform.position.x),
      y: interpolateKeyframes(keyframes, 'position.y', time, baseTransform.position.y),
      z: interpolateKeyframes(keyframes, 'position.z', time, baseTransform.position.z),
    },
    scale: {
      x: interpolateKeyframes(keyframes, 'scale.x', time, baseTransform.scale.x),
      y: interpolateKeyframes(keyframes, 'scale.y', time, baseTransform.scale.y),
    },
    rotation: {
      x: interpolateKeyframes(keyframes, 'rotation.x', time, baseTransform.rotation.x),
      y: interpolateKeyframes(keyframes, 'rotation.y', time, baseTransform.rotation.y),
      z: interpolateKeyframes(keyframes, 'rotation.z', time, baseTransform.rotation.z),
    },
  };
}

// Check if a property has keyframes
export function hasKeyframesForProperty(
  keyframes: Keyframe[],
  property: AnimatableProperty
): boolean {
  return keyframes.some(k => k.property === property);
}

// Get all unique properties that have keyframes
export function getAnimatedProperties(keyframes: Keyframe[]): AnimatableProperty[] {
  const properties = new Set<AnimatableProperty>();
  keyframes.forEach(k => properties.add(k.property));
  return Array.from(properties);
}

// Get keyframe at specific time for a property (for updating existing keyframes)
export function getKeyframeAtTime(
  keyframes: Keyframe[],
  property: AnimatableProperty,
  time: number,
  tolerance: number = 0.01 // 10ms tolerance
): Keyframe | undefined {
  return keyframes.find(
    k => k.property === property && Math.abs(k.time - time) < tolerance
  );
}

// Property path helpers for nested transform properties
export function getValueFromTransform(
  transform: ClipTransform,
  property: AnimatableProperty
): number {
  switch (property) {
    case 'opacity': return transform.opacity;
    case 'position.x': return transform.position.x;
    case 'position.y': return transform.position.y;
    case 'position.z': return transform.position.z;
    case 'scale.x': return transform.scale.x;
    case 'scale.y': return transform.scale.y;
    case 'rotation.x': return transform.rotation.x;
    case 'rotation.y': return transform.rotation.y;
    case 'rotation.z': return transform.rotation.z;
    default: return 0;
  }
}

export function setValueInTransform(
  transform: ClipTransform,
  property: AnimatableProperty,
  value: number
): ClipTransform {
  const newTransform = { ...transform };

  switch (property) {
    case 'opacity':
      newTransform.opacity = value;
      break;
    case 'position.x':
      newTransform.position = { ...transform.position, x: value };
      break;
    case 'position.y':
      newTransform.position = { ...transform.position, y: value };
      break;
    case 'position.z':
      newTransform.position = { ...transform.position, z: value };
      break;
    case 'scale.x':
      newTransform.scale = { ...transform.scale, x: value };
      break;
    case 'scale.y':
      newTransform.scale = { ...transform.scale, y: value };
      break;
    case 'rotation.x':
      newTransform.rotation = { ...transform.rotation, x: value };
      break;
    case 'rotation.y':
      newTransform.rotation = { ...transform.rotation, y: value };
      break;
    case 'rotation.z':
      newTransform.rotation = { ...transform.rotation, z: value };
      break;
  }

  return newTransform;
}
