// Audio Sync Service
// Synchronizes multiple camera angles using audio waveform cross-correlation

import { audioAnalyzer, type AudioFingerprint } from './audioAnalyzer';

/**
 * Cross-correlation algorithm to find the offset between two audio signals.
 * Returns the offset in milliseconds (positive = second signal is delayed)
 */
function crossCorrelate(
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

class AudioSync {
  // Cache fingerprints
  private fingerprintCache = new Map<string, AudioFingerprint>();

  /**
   * Get or generate fingerprint for a media file
   */
  private async getFingerprint(mediaFileId: string): Promise<AudioFingerprint | null> {
    // Check cache
    if (this.fingerprintCache.has(mediaFileId)) {
      return this.fingerprintCache.get(mediaFileId)!;
    }

    // Generate fingerprint
    const fingerprint = await audioAnalyzer.generateFingerprint(mediaFileId);
    if (fingerprint) {
      this.fingerprintCache.set(mediaFileId, fingerprint);
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
    console.log('[AudioSync] Finding offset between', masterMediaFileId, 'and', targetMediaFileId);

    // Get fingerprints
    const [masterFp, targetFp] = await Promise.all([
      this.getFingerprint(masterMediaFileId),
      this.getFingerprint(targetMediaFileId),
    ]);

    if (!masterFp || !targetFp) {
      console.warn('[AudioSync] Could not generate fingerprints');
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

    console.log(`[AudioSync] Found offset: ${offsetMs.toFixed(2)}ms (correlation: ${result.correlation.toFixed(4)})`);

    return offsetMs;
  }

  /**
   * Sync multiple cameras to a master camera.
   * Returns a map of camera ID to offset in milliseconds.
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
    console.log('[AudioSync] Generating master fingerprint...');
    reportProgress();
    const masterFp = await this.getFingerprint(masterMediaFileId);
    currentStep++;
    reportProgress();

    if (!masterFp) {
      console.warn('[AudioSync] Could not generate master fingerprint');
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

      console.log(`[AudioSync] Processing target ${i + 1}/${targetMediaFileIds.length}...`);

      // Get target fingerprint
      const targetFp = await this.getFingerprint(targetId);

      if (!targetFp) {
        console.warn('[AudioSync] Could not generate fingerprint for', targetId);
        currentStep++;
        reportProgress();
        continue;
      }

      // Calculate max offset in samples (30 seconds)
      const maxOffsetSamples = Math.floor(30 * masterFp.sampleRate);

      // Perform cross-correlation
      const result = normalizedCrossCorrelate(
        masterFp.data,
        targetFp.data,
        maxOffsetSamples
      );

      // Convert offset from samples to milliseconds
      const offsetMs = (result.offset / masterFp.sampleRate) * 1000;
      offsets.set(targetId, offsetMs);

      console.log(`[AudioSync] Offset for ${targetId}: ${offsetMs.toFixed(1)}ms (correlation: ${result.correlation.toFixed(4)})`);

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
