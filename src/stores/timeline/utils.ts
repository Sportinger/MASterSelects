// Timeline store utility functions

import type { EffectType } from '../../types';

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
    console.warn('Failed to generate waveform:', e);
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
    console.warn('Failed to generate waveform from buffer:', e);
    return [];
  }
}

// Generate thumbnail filmstrip from video
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

  // Generate frames at regular intervals
  const interval = duration / count;

  for (let i = 0; i < count; i++) {
    const time = i * interval;
    try {
      await seekVideo(video, time);
      ctx.drawImage(video, 0, 0, thumbWidth, thumbHeight);
      thumbnails.push(canvas.toDataURL('image/jpeg', 0.6));
    } catch (e) {
      console.warn('Failed to generate thumbnail at', time, e);
    }
  }

  return thumbnails;
}

// Helper function to get default effect parameters
export function getDefaultEffectParams(type: string | EffectType): Record<string, number | boolean | string> {
  switch (type) {
    case 'hue-shift':
      return { shift: 0 };
    case 'saturation':
      return { amount: 1 };
    case 'brightness':
      return { amount: 0 };
    case 'contrast':
      return { amount: 1 };
    case 'blur':
      return { radius: 0 };
    case 'pixelate':
      return { size: 8 };
    case 'kaleidoscope':
      return { segments: 6, rotation: 0 };
    case 'mirror':
      return { horizontal: true, vertical: false };
    case 'invert':
      return {};
    case 'rgb-split':
      return { amount: 0.01, angle: 0 };
    case 'levels':
      return { inputBlack: 0, inputWhite: 1, gamma: 1, outputBlack: 0, outputWhite: 1 };
    // Audio effects
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
