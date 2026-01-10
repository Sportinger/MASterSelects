import type { Keyframe, AnimatableProperty } from '../types';
import { interpolateKeyframes } from './keyframeInterpolation';

/**
 * Speed Integration Utility
 *
 * For variable speed playback with keyframes, we need to INTEGRATE the speed curve
 * to calculate the source time. This is different from normal interpolation.
 *
 * Example: If speed is 2x from t=0 to t=1, and 0.5x from t=1 to t=2:
 * - At timeline t=1: sourceTime = 1 * 2 = 2 seconds
 * - At timeline t=2: sourceTime = 2 + 1 * 0.5 = 2.5 seconds
 *
 * For keyframed speed, we use trapezoidal integration for smooth results.
 */

/**
 * Calculate the source time for a given timeline local time by integrating the speed curve.
 *
 * @param keyframes - All keyframes for the clip
 * @param clipLocalTime - Time within the clip (0 to clip.duration)
 * @param defaultSpeed - Default speed if no keyframes (usually clip.speed ?? 1)
 * @returns The source time to seek to
 */
export function calculateSourceTime(
  keyframes: Keyframe[],
  clipLocalTime: number,
  defaultSpeed: number
): number {
  // Filter speed keyframes and sort by time
  const speedKeyframes = keyframes
    .filter(k => k.property === 'speed')
    .sort((a, b) => a.time - b.time);

  // No speed keyframes - simple multiplication
  if (speedKeyframes.length === 0) {
    return clipLocalTime * defaultSpeed;
  }

  // Single keyframe - use that constant speed
  if (speedKeyframes.length === 1) {
    return clipLocalTime * speedKeyframes[0].value;
  }

  // Multiple keyframes - integrate using trapezoidal rule
  return integrateSpeedCurve(keyframes, speedKeyframes, clipLocalTime, defaultSpeed);
}

/**
 * Get the interpolated speed at a specific time.
 * Uses the same interpolation as other keyframed properties.
 */
export function getSpeedAtTime(
  keyframes: Keyframe[],
  clipLocalTime: number,
  defaultSpeed: number
): number {
  return interpolateKeyframes(keyframes, 'speed' as AnimatableProperty, clipLocalTime, defaultSpeed);
}

/**
 * Integrate the speed curve from 0 to endTime using adaptive trapezoidal integration.
 *
 * The algorithm:
 * 1. Sample the speed curve at keyframe times and intermediate points
 * 2. Use trapezoidal rule: integral = sum of (speed_i + speed_{i+1}) / 2 * dt
 *
 * For smooth bezier curves, we sample more densely.
 */
function integrateSpeedCurve(
  keyframes: Keyframe[],
  speedKeyframes: Keyframe[],
  endTime: number,
  defaultSpeed: number
): number {
  // Edge cases
  if (endTime <= 0) return 0;

  // Collect all sample points (keyframe times + intermediate samples)
  const samplePoints: number[] = [0];

  // Add keyframe times up to endTime
  for (const kf of speedKeyframes) {
    if (kf.time > 0 && kf.time < endTime) {
      samplePoints.push(kf.time);
    }
  }

  // Add endTime
  samplePoints.push(endTime);

  // Add intermediate samples between keyframes for better accuracy
  // Especially important for bezier curves
  const samplesPerSegment = 10;
  const allSamplePoints: number[] = [];

  for (let i = 0; i < samplePoints.length - 1; i++) {
    const start = samplePoints[i];
    const end = samplePoints[i + 1];
    const dt = (end - start) / samplesPerSegment;

    for (let j = 0; j < samplesPerSegment; j++) {
      allSamplePoints.push(start + j * dt);
    }
  }
  allSamplePoints.push(endTime);

  // Remove duplicates and sort
  const uniqueSamples = [...new Set(allSamplePoints)].sort((a, b) => a - b);

  // Trapezoidal integration
  let integral = 0;

  for (let i = 0; i < uniqueSamples.length - 1; i++) {
    const t0 = uniqueSamples[i];
    const t1 = uniqueSamples[i + 1];
    const dt = t1 - t0;

    const speed0 = getSpeedAtTime(keyframes, t0, defaultSpeed);
    const speed1 = getSpeedAtTime(keyframes, t1, defaultSpeed);

    // Trapezoidal rule: area = (y0 + y1) / 2 * width
    integral += (speed0 + speed1) / 2 * dt;
  }

  return integral;
}

/**
 * Calculate the total source time that will be consumed by a clip with speed keyframes.
 * This is useful for determining if the clip will run out of source material.
 *
 * @param keyframes - All keyframes for the clip
 * @param timelineDuration - The timeline duration of the clip
 * @param defaultSpeed - Default speed
 * @returns Total source time consumed
 */
export function calculateTotalSourceTime(
  keyframes: Keyframe[],
  timelineDuration: number,
  defaultSpeed: number
): number {
  return calculateSourceTime(keyframes, timelineDuration, defaultSpeed);
}

/**
 * Calculate the timeline duration needed to play through a given source duration
 * with the specified speed keyframes. This is the inverse of calculateSourceTime.
 *
 * Uses binary search since the integral is monotonic (assuming positive speeds).
 *
 * @param keyframes - All keyframes for the clip
 * @param sourceDuration - The source duration to consume
 * @param defaultSpeed - Default speed
 * @param maxIterations - Maximum binary search iterations
 * @returns Timeline duration needed
 */
export function calculateTimelineDuration(
  keyframes: Keyframe[],
  sourceDuration: number,
  defaultSpeed: number,
  maxIterations: number = 50
): number {
  // Handle edge cases
  if (sourceDuration <= 0) return 0;

  const speedKeyframes = keyframes
    .filter(k => k.property === 'speed')
    .sort((a, b) => a.time - b.time);

  // No keyframes - simple division
  if (speedKeyframes.length === 0) {
    return Math.abs(defaultSpeed) > 0.001 ? sourceDuration / Math.abs(defaultSpeed) : Infinity;
  }

  // Binary search for the timeline duration that produces the target source time
  // Start with a reasonable range
  let low = 0;
  let high = sourceDuration * 10; // Generous upper bound (assumes speed >= 0.1x average)

  for (let i = 0; i < maxIterations; i++) {
    const mid = (low + high) / 2;
    const sourceTimeAtMid = calculateSourceTime(keyframes, mid, defaultSpeed);

    if (Math.abs(sourceTimeAtMid - sourceDuration) < 0.001) {
      return mid;
    }

    if (sourceTimeAtMid < sourceDuration) {
      low = mid;
    } else {
      high = mid;
    }
  }

  return (low + high) / 2;
}

/**
 * Check if speed keyframes result in reverse playback at any point.
 * Useful for determining if we need special handling for reverse.
 */
export function hasReverseSpeed(keyframes: Keyframe[], defaultSpeed: number): boolean {
  const speedKeyframes = keyframes.filter(k => k.property === 'speed');

  if (speedKeyframes.length === 0) {
    return defaultSpeed < 0;
  }

  return speedKeyframes.some(k => k.value < 0);
}

/**
 * Get the maximum absolute speed in the keyframe curve.
 * Useful for determining buffer requirements.
 */
export function getMaxSpeed(keyframes: Keyframe[], defaultSpeed: number): number {
  const speedKeyframes = keyframes.filter(k => k.property === 'speed');

  if (speedKeyframes.length === 0) {
    return Math.abs(defaultSpeed);
  }

  let maxSpeed = Math.abs(defaultSpeed);
  for (const kf of speedKeyframes) {
    maxSpeed = Math.max(maxSpeed, Math.abs(kf.value));
  }

  return maxSpeed;
}
