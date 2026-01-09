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
export async function generateWaveform(file: File, sampleCount: number = 200): Promise<number[]> {
  try {
    const audioContext = new AudioContext();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.getChannelData(0); // Use first channel
    const samples: number[] = [];
    const blockSize = Math.floor(channelData.length / sampleCount);

    for (let i = 0; i < sampleCount; i++) {
      const start = i * blockSize;
      const end = start + blockSize;
      let sum = 0;

      for (let j = start; j < end; j++) {
        sum += Math.abs(channelData[j]);
      }

      samples.push(sum / blockSize);
    }

    // Normalize to 0-1 range
    const max = Math.max(...samples);
    if (max > 0) {
      await audioContext.close();
      return samples.map(s => s / max);
    }
    await audioContext.close();
    return samples;
  } catch (e) {
    console.warn('Failed to generate waveform:', e);
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
    default:
      return {};
  }
}

// Quantize time to 30fps for caching
export function quantizeTime(time: number): number {
  return Math.round(time * 30) / 30;
}
