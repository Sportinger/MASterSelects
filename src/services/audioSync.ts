// Audio Sync Service
// Synchronizes multiple camera angles using audio waveform cross-correlation

import { Logger } from './logger';
import { audioAnalyzer, type AudioFingerprint } from './audioAnalyzer';

const log = Logger.create('AudioSync');

/**
 * Cross-correlation algorithm to find the offset between two audio signals.
 * Returns the offset in milliseconds (positive = second signal is delayed)
 */
export function crossCorrelate(
  signal1: Float32Array,
  signal2: Float32Array,
  maxOffsetSamples: number
): { offset: number; correlation: number } {
  let bestOffset = 0;
  let bestCorrelation = -Infinity;

  // Search in both directions
  for (let offset = -maxOffsetSamples; offset <= maxOffsetSamples; offset++) {
    let correlation = 0;
    let count = 0;

    for (let i = 0; i < signal1.length; i++) {
      const j = i + offset;
      if (j >= 0 && j < signal2.length) {
        correlation += signal1[i] * signal2[j];
        count++;
      }
    }

    // Normalize by number of overlapping samples
    if (count > 0) {
      correlation /= count;
    }

    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestOffset = offset;
    }
  }

  return { offset: bestOffset, correlation: bestCorrelation };
}

/**
 * Normalize cross-correlation (Pearson correlation coefficient)
 * More accurate but slower
 */
function normalizedCrossCorrelate(
  signal1: Float32Array,
  signal2: Float32Array,
  maxOffsetSamples: number
): { offset: number; correlation: number } {
  let bestOffset = 0;
  let bestCorrelation = -Infinity;

  // Calculate means
  const mean1 = signal1.reduce((a, b) => a + b, 0) / signal1.length;
  const mean2 = signal2.reduce((a, b) => a + b, 0) / signal2.length;

  // Calculate standard deviations
  let std1 = 0, std2 = 0;
  for (let i = 0; i < signal1.length; i++) {
    std1 += (signal1[i] - mean1) ** 2;
  }
  std1 = Math.sqrt(std1 / signal1.length);

  for (let i = 0; i < signal2.length; i++) {
    std2 += (signal2[i] - mean2) ** 2;
  }
  std2 = Math.sqrt(std2 / signal2.length);

  // Search in both directions
  for (let offset = -maxOffsetSamples; offset <= maxOffsetSamples; offset++) {
    let correlation = 0;
    let count = 0;

    for (let i = 0; i < signal1.length; i++) {
      const j = i + offset;
      if (j >= 0 && j < signal2.length) {
        correlation += ((signal1[i] - mean1) / std1) * ((signal2[j] - mean2) / std2);
        count++;
      }
    }

    if (count > 0) {
      correlation /= count;
    }

    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestOffset = offset;
    }
  }

  return { offset: bestOffset, correlation: bestCorrelation };
}

// Clip info for sync (includes time bounds)
export interface ClipSyncInfo {
  mediaFileId: string;
  clipId: string;
  inPoint: number;  // Start time within source file (seconds)
  duration: number; // Duration of the clip (seconds)
}

class AudioSync {
  // Cache fingerprints - key includes time range for trimmed clips
  private fingerprintCache = new Map<string, AudioFingerprint>();

  /**
   * Generate cache key that includes time range
   */
  private getCacheKey(mediaFileId: string, startTime: number, duration: number): string {
    return `${mediaFileId}-${startTime.toFixed(2)}-${duration.toFixed(2)}`;
  }

  /**
   * Get or generate fingerprint for a media file (or portion of it)
   * @param mediaFileId - ID of the media file
   * @param startTime - Start time in seconds (for trimmed clips)
   * @param duration - Duration to analyze in seconds
   */
  private async getFingerprint(
    mediaFileId: string,
    startTime: number = 0,
    duration: number = 30
  ): Promise<AudioFingerprint | null> {
    const cacheKey = this.getCacheKey(mediaFileId, startTime, duration);

    // Check cache
    if (this.fingerprintCache.has(cacheKey)) {
      return this.fingerprintCache.get(cacheKey)!;
    }

    // Generate fingerprint with time bounds
    const fingerprint = await audioAnalyzer.generateFingerprint(
      mediaFileId,
      2000, // targetSampleRate
      startTime,
      duration
    );
    if (fingerprint) {
      this.fingerprintCache.set(cacheKey, fingerprint);
    }
    return fingerprint;
  }

  /**
   * Find the time offset between two media files based on their audio.
   * Returns offset in milliseconds.
   * Positive offset means the second file's audio starts later than the first.
   */
  async findOffset(
    masterMediaFileId: string,
    targetMediaFileId: string,
    maxOffsetSeconds: number = 30
  ): Promise<number> {
    log.info(`Finding offset between ${masterMediaFileId} and ${targetMediaFileId}`);

    // Get fingerprints
    const [masterFp, targetFp] = await Promise.all([
      this.getFingerprint(masterMediaFileId),
      this.getFingerprint(targetMediaFileId),
    ]);

    if (!masterFp || !targetFp) {
      log.warn('Could not generate fingerprints');
      return 0;
    }

    // Calculate max offset in samples
    const maxOffsetSamples = Math.floor(maxOffsetSeconds * masterFp.sampleRate);

    // Perform cross-correlation
    const result = normalizedCrossCorrelate(
      masterFp.data,
      targetFp.data,
      maxOffsetSamples
    );

    // Convert offset from samples to milliseconds
    const offsetMs = (result.offset / masterFp.sampleRate) * 1000;

    log.info(`Found offset: ${offsetMs.toFixed(2)}ms (correlation: ${result.correlation.toFixed(4)})`);

    return offsetMs;
  }

  /**
   * Sync multiple clips to a master clip using their audio.
   * Uses clip inPoint and duration to analyze only the visible portion.
   * Returns a map of clipId to offset in milliseconds.
   */
  async syncMultipleClips(
    masterClip: ClipSyncInfo,
    targetClips: ClipSyncInfo[],
    onProgress?: (progress: number) => void
  ): Promise<Map<string, number>> {
    const offsets = new Map<string, number>();
    offsets.set(masterClip.clipId, 0); // Master has zero offset

    const totalSteps = targetClips.length + 1; // +1 for master fingerprint
    let currentStep = 0;

    const reportProgress = () => {
      if (onProgress) {
        onProgress(Math.round((currentStep / totalSteps) * 100));
      }
    };

    // Generate master fingerprint using clip's time bounds
    log.info(`Generating master fingerprint (${masterClip.inPoint.toFixed(1)}s - ${(masterClip.inPoint + masterClip.duration).toFixed(1)}s)...`);
    reportProgress();
    const masterFp = await this.getFingerprint(
      masterClip.mediaFileId,
      masterClip.inPoint,
      Math.min(masterClip.duration, 30) // Limit to 30s for performance
    );
    currentStep++;
    reportProgress();

    if (!masterFp) {
      log.warn('Could not generate master fingerprint');
      return offsets;
    }

    // Process each target clip
    for (let i = 0; i < targetClips.length; i++) {
      const targetClip = targetClips[i];

      log.debug(`Processing target ${i + 1}/${targetClips.length} (${targetClip.inPoint.toFixed(1)}s - ${(targetClip.inPoint + targetClip.duration).toFixed(1)}s)...`);

      // Get target fingerprint using clip's time bounds
      const targetFp = await this.getFingerprint(
        targetClip.mediaFileId,
        targetClip.inPoint,
        Math.min(targetClip.duration, 30)
      );

      if (!targetFp) {
        log.warn('Could not generate fingerprint for clip', targetClip.clipId);
        currentStep++;
        reportProgress();
        continue;
      }

      // Calculate max offset in samples (10 seconds - sufficient for most multicam)
      const maxOffsetSamples = Math.floor(10 * masterFp.sampleRate);

      // Perform cross-correlation
      const result = normalizedCrossCorrelate(
        masterFp.data,
        targetFp.data,
        maxOffsetSamples
      );

      // Convert offset from samples to milliseconds
      // Also account for inPoint differences: if master starts at 5s and target at 10s,
      // the base offset is already (10-5)*1000 = 5000ms
      const correlationOffsetMs = (result.offset / masterFp.sampleRate) * 1000;
      const inPointDifferenceMs = (targetClip.inPoint - masterClip.inPoint) * 1000;
      const totalOffsetMs = correlationOffsetMs + inPointDifferenceMs;

      offsets.set(targetClip.clipId, totalOffsetMs);

      log.info(`Offset for ${targetClip.clipId}: ${totalOffsetMs.toFixed(1)}ms (correlation: ${result.correlation.toFixed(4)}, inPoint diff: ${inPointDifferenceMs.toFixed(1)}ms)`);

      currentStep++;
      reportProgress();
    }

    return offsets;
  }

  /**
   * Legacy method: Sync multiple cameras by mediaFileId only.
   * Uses full files (first 30s). For backward compatibility.
   */
  async syncMultiple(
    masterMediaFileId: string,
    targetMediaFileIds: string[],
    onProgress?: (progress: number) => void
  ): Promise<Map<string, number>> {
    const offsets = new Map<string, number>();
    offsets.set(masterMediaFileId, 0); // Master has zero offset

    const totalSteps = targetMediaFileIds.length + 1; // +1 for master fingerprint
    let currentStep = 0;

    const reportProgress = () => {
      if (onProgress) {
        onProgress(Math.round((currentStep / totalSteps) * 100));
      }
    };

    // Generate master fingerprint first (this is the slow part)
    log.info('Generating master fingerprint...');
    reportProgress();
    const masterFp = await this.getFingerprint(masterMediaFileId);
    currentStep++;
    reportProgress();

    if (!masterFp) {
      log.warn('Could not generate master fingerprint');
      return offsets;
    }

    // Process each target
    for (let i = 0; i < targetMediaFileIds.length; i++) {
      const targetId = targetMediaFileIds[i];
      if (targetId === masterMediaFileId) {
        currentStep++;
        reportProgress();
        continue;
      }

      log.debug(`Processing target ${i + 1}/${targetMediaFileIds.length}...`);

      // Get target fingerprint
      const targetFp = await this.getFingerprint(targetId);

      if (!targetFp) {
        log.warn('Could not generate fingerprint for', targetId);
        currentStep++;
        reportProgress();
        continue;
      }

      // Calculate max offset in samples (10 seconds - sufficient for most multicam)
      const maxOffsetSamples = Math.floor(10 * masterFp.sampleRate);

      // Perform cross-correlation
      const result = normalizedCrossCorrelate(
        masterFp.data,
        targetFp.data,
        maxOffsetSamples
      );

      // Convert offset from samples to milliseconds
      const offsetMs = (result.offset / masterFp.sampleRate) * 1000;
      offsets.set(targetId, offsetMs);

      log.info(`Offset for ${targetId}: ${offsetMs.toFixed(1)}ms (correlation: ${result.correlation.toFixed(4)})`);

      currentStep++;
      reportProgress();
    }

    return offsets;
  }

  /**
   * Clear the fingerprint cache
   */
  clearCache(): void {
    this.fingerprintCache.clear();
  }
}

// Singleton instance
export const audioSync = new AudioSync();
