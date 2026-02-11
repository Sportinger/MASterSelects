// Timeline store utility functions

import type { EffectType } from '../../types';
import { getDefaultParams as getRegistryDefaultParams, hasEffect } from '../../effects';
// Helper to seek video and wait for it to be ready
export function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Seek timeout')), 3000);

    const onSeeked = () => {
      clearTimeout(timeout);
      video.removeEventListener('seeked', onSeeked);
      resolve();
    };

    video.addEventListener('seeked', onSeeked);
    video.currentTime = time;
  });
}

// Helper function to get default effect parameters
// Now uses the modular effect registry, with fallback for audio effects
export function getDefaultEffectParams(type: string | EffectType): Record<string, number | boolean | string> {
  // Check if effect exists in the new registry
  if (hasEffect(type)) {
    return getRegistryDefaultParams(type);
  }

  // Fallback for audio effects (not yet in the modular system)
  switch (type) {
    case 'audio-eq':
      return {
        band31: 0, band62: 0, band125: 0, band250: 0, band500: 0,
        band1k: 0, band2k: 0, band4k: 0, band8k: 0, band16k: 0
      };
    case 'audio-volume':
      return { volume: 1 };
    default:
      return {};
  }
}

// Quantize time to 30fps for caching
export function quantizeTime(time: number): number {
  return Math.round(time * 30) / 30;
}
