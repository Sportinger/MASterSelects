// Timeline store utility functions

import type { EffectType } from '../../types';
import { getDefaultParams as getRegistryDefaultParams, hasEffect } from '../../effects';
import { Logger } from '../../services/logger';

const log = Logger.create('TimelineUtils');

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

// Generate waveform data from audio file
// Uses ~50 samples per second for good visual resolution
// Supports optional progress callback for real-time updates
export async function generateWaveform(
  file: File,
  samplesPerSecond: number = 50,
  onProgress?: (progress: number, partialWaveform: number[]) => void
): Promise<number[]> {
  try {
    const audioContext = new AudioContext();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.getChannelData(0); // Use first channel
    const duration = audioBuffer.duration;

    // Calculate samples based on duration (more samples for longer files)
    const sampleCount = Math.max(200, Math.min(10000, Math.floor(duration * samplesPerSecond)));
    const blockSize = Math.floor(channelData.length / sampleCount);

    const samples: number[] = [];
    let runningMax = 0;

    for (let i = 0; i < sampleCount; i++) {
      const start = i * blockSize;
      const end = Math.min(start + blockSize, channelData.length);

      // Use peak value for better visual representation
      let peak = 0;
      for (let j = start; j < end; j++) {
        const abs = Math.abs(channelData[j]);
        if (abs > peak) peak = abs;
      }

      samples.push(peak);
      if (peak > runningMax) runningMax = peak;

      // Report progress with normalized partial waveform every 5%
      if (onProgress && (i % Math.max(1, Math.floor(sampleCount / 20)) === 0 || i === sampleCount - 1)) {
        const progress = Math.round(((i + 1) / sampleCount) * 100);
        // Normalize partial waveform with running max
        const normalizedPartial = runningMax > 0
          ? samples.map(s => s / runningMax)
          : samples;
        onProgress(progress, normalizedPartial);
        // Yield to UI
        await new Promise(r => setTimeout(r, 0));
      }
    }

    // Final normalization to 0-1 range
    const max = Math.max(...samples);
    await audioContext.close();

    if (max > 0) {
      return samples.map(s => s / max);
    }
    return samples;
  } catch (e) {
    log.warn('Failed to generate waveform', e);
    return [];
  }
}

// Generate waveform data from an already decoded AudioBuffer
// Synchronous version for use with pre-decoded buffers (e.g., composition mixdowns)
export function generateWaveformFromBuffer(
  audioBuffer: AudioBuffer,
  samplesPerSecond: number = 50
): number[] {
  try {
    const channelData = audioBuffer.getChannelData(0); // Use first channel
    const duration = audioBuffer.duration;

    // Calculate samples based on duration
    const sampleCount = Math.max(200, Math.min(10000, Math.floor(duration * samplesPerSecond)));
    const blockSize = Math.floor(channelData.length / sampleCount);

    const samples: number[] = [];

    for (let i = 0; i < sampleCount; i++) {
      const start = i * blockSize;
      const end = Math.min(start + blockSize, channelData.length);

      // Use peak value for better visual representation
      let peak = 0;
      for (let j = start; j < end; j++) {
        const abs = Math.abs(channelData[j]);
        if (abs > peak) peak = abs;
      }

      samples.push(peak);
    }

    // Normalize to 0-1 range
    const max = Math.max(...samples);
    if (max > 0) {
      return samples.map(s => s / max);
    }
    return samples;
  } catch (e) {
    log.warn('Failed to generate waveform from buffer', e);
    return [];
  }
}

// Generate thumbnail filmstrip from video
// Now covers 0% to 100% of duration (previously only 0% to 90%)
export async function generateThumbnails(video: HTMLVideoElement, duration: number, count: number = 10): Promise<string[]> {
  const thumbnails: string[] = [];
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return thumbnails;

  // Thumbnail dimensions (aspect ratio preserved)
  const thumbHeight = 40;
  const thumbWidth = Math.round((video.videoWidth / video.videoHeight) * thumbHeight);
  canvas.width = thumbWidth;
  canvas.height = thumbHeight;

  // Generate frames from 0% to 100% of duration
  // With count=10: 0%, 11.1%, 22.2%, ... 88.9%, 100%
  for (let i = 0; i < count; i++) {
    // Use (count - 1) as divisor so last thumbnail is at 100%
    // For count=1, just use time=0
    const time = count > 1 ? (i / (count - 1)) * duration : 0;
    // Clamp to slightly before end to avoid seek issues at exact duration
    const clampedTime = Math.min(time, duration - 0.01);
    try {
      await seekVideo(video, clampedTime);
      ctx.drawImage(video, 0, 0, thumbWidth, thumbHeight);
      thumbnails.push(canvas.toDataURL('image/jpeg', 0.6));
    } catch (e) {
      log.warn('Failed to generate thumbnail', { time: clampedTime, error: e });
    }
  }

  return thumbnails;
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
