// Waveform generation helper - centralizes waveform logic and file size checks
// Provides consistent thresholds and logging across clip loading

import { generateWaveform as baseGenerateWaveform } from '../utils';
import { Logger } from '../../../services/logger';

const log = Logger.create('WaveformHelpers');

// File size thresholds for waveform generation
// Video waveforms: skip if >500MB (video decode + audio decode is expensive)
export const VIDEO_WAVEFORM_THRESHOLD = 500 * 1024 * 1024; // 500MB

// Audio-only waveforms: can handle up to 4GB (just audio decode)
export const AUDIO_WAVEFORM_THRESHOLD = 4 * 1024 * 1024 * 1024; // 4GB

export interface WaveformGenerationOptions {
  samplesPerSecond?: number;
  onProgress?: (progress: number, partialWaveform: number[]) => void;
}

/**
 * Start waveform generation for a file.
 * Returns the complete waveform data.
 */
export async function generateWaveformForFile(
  file: File,
  options: WaveformGenerationOptions = {}
): Promise<number[]> {
  const { samplesPerSecond = 50, onProgress } = options;

  log.debug('Starting waveform generation', { file: file.name });

  const waveform = await baseGenerateWaveform(file, samplesPerSecond, onProgress);

  log.debug('Waveform complete', { samples: waveform.length, file: file.name });

  return waveform;
}

/**
 * Check if waveform generation should be skipped based on file size.
 * @param file - The file to check
 * @param isAudioOnly - True for audio-only files (higher threshold)
 */
export function shouldSkipWaveform(file: File, isAudioOnly: boolean = false): boolean {
  const threshold = isAudioOnly ? AUDIO_WAVEFORM_THRESHOLD : VIDEO_WAVEFORM_THRESHOLD;

  if (file.size > threshold) {
    const sizeMB = (file.size / 1024 / 1024).toFixed(0);
    log.debug('Skipping waveform for large file', { sizeMB, file: file.name });
    return true;
  }

  return false;
}

/**
 * Generate a flat (silent) waveform for a given duration.
 * Used for composition clips without audio.
 */
export function generateSilentWaveform(duration: number, samplesPerSecond: number = 50): number[] {
  return new Array(Math.max(1, Math.floor(duration * samplesPerSecond))).fill(0);
}

/**
 * Calculate expected waveform sample count for a duration.
 */
export function getExpectedWaveformSamples(duration: number, samplesPerSecond: number = 50): number {
  return Math.max(200, Math.min(10000, Math.floor(duration * samplesPerSecond)));
}
