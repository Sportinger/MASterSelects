// Performance Monitor Service
// Monitors frame render times and automatically resets quality parameters when performance is too slow

import { useTimelineStore } from '../stores/timeline';
import { getDefaultParams, EFFECT_REGISTRY } from '../effects';

// Config
const SLOW_FRAME_THRESHOLD_MS = 100; // 10fps
const CONSECUTIVE_SLOW_FRAMES = 5;   // Need 5 slow frames before triggering
const CHECK_INTERVAL_MS = 500;       // Check every 500ms

let slowFrameCount = 0;
let lastCheckTime = 0;
let isMonitoring = false;
let onSlowPerformanceCallback: (() => void) | null = null;

// Get quality parameter names for an effect type
function getQualityParamNames(effectType: string): string[] {
  const effectDef = EFFECT_REGISTRY.get(effectType);
  if (!effectDef) return [];

  return Object.entries(effectDef.params)
    .filter(([, def]) => def.quality)
    .map(([name]) => name);
}

// Reset all quality parameters to defaults for all clips
export function resetAllQualityParams(): number {
  const { clips, updateClipEffect } = useTimelineStore.getState();
  let resetCount = 0;

  clips.forEach(clip => {
    (clip.effects || []).forEach(effect => {
      const qualityParams = getQualityParamNames(effect.type);
      if (qualityParams.length === 0) return;

      const defaults = getDefaultParams(effect.type);
      const updates: Record<string, number | boolean | string> = {};

      qualityParams.forEach(paramName => {
        const currentValue = effect.params[paramName];
        const defaultValue = defaults[paramName];
        if (currentValue !== defaultValue) {
          updates[paramName] = defaultValue;
          resetCount++;
        }
      });

      if (Object.keys(updates).length > 0) {
        updateClipEffect(clip.id, effect.id, updates);
      }
    });
  });

  return resetCount;
}

// Check frame time from engine
function checkPerformance(renderTimeMs: number) {
  const now = performance.now();
  if (now - lastCheckTime < CHECK_INTERVAL_MS) return;
  lastCheckTime = now;

  if (renderTimeMs > SLOW_FRAME_THRESHOLD_MS) {
    slowFrameCount++;

    if (slowFrameCount >= CONSECUTIVE_SLOW_FRAMES) {
      console.warn(`[Performance] Slow rendering detected (${renderTimeMs.toFixed(1)}ms). Resetting quality parameters...`);
      const resetCount = resetAllQualityParams();

      if (resetCount > 0) {
        console.log(`[Performance] Reset ${resetCount} quality parameters to defaults`);
        onSlowPerformanceCallback?.();
      }

      slowFrameCount = 0;
    }
  } else {
    // Reset counter if we have a fast frame
    slowFrameCount = Math.max(0, slowFrameCount - 1);
  }
}

// Start monitoring
export function startPerformanceMonitor() {
  if (isMonitoring) return;
  isMonitoring = true;
  slowFrameCount = 0;
  console.log('[Performance] Monitor started');
}

// Stop monitoring
export function stopPerformanceMonitor() {
  isMonitoring = false;
  slowFrameCount = 0;
  console.log('[Performance] Monitor stopped');
}

// Set callback for when slow performance triggers a reset
export function onSlowPerformance(callback: () => void) {
  onSlowPerformanceCallback = callback;
}

// Report render time (called from engine)
export function reportRenderTime(ms: number) {
  if (!isMonitoring) return;
  checkPerformance(ms);
}

// Check if monitoring is active
export function isPerformanceMonitorActive(): boolean {
  return isMonitoring;
}

// Auto-start monitoring when module is imported
startPerformanceMonitor();
