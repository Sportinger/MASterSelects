import type { Keyframe, EasingType, AnimatableProperty, ClipTransform } from '../types';

// Easing functions
export const easingFunctions: Record<EasingType, (t: number) => number> = {
  'linear': (t) => t,
  'ease-in': (t) => t * t,
  'ease-out': (t) => t * (2 - t),
  'ease-in-out': (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
};

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

  // Apply easing from the previous keyframe
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
