// Audio Analyzer Service
// Extracts audio levels and fingerprints from media files for sync and analysis

import { useMediaStore } from '../stores/mediaStore';
import { projectDB } from './projectDB';

export interface AudioLevel {
  timestamp: number; // ms
  level: number; // 0-1 (RMS)
}

export interface AudioCurve {
  mediaFileId: string;
  levels: AudioLevel[];
  sampleRate: number;
  windowSizeMs: number;
}

export interface AudioFingerprint {
  mediaFileId: string;
  // Downsampled audio data for comparison
  data: Float32Array;
  sampleRate: number;
}

class AudioAnalyzer {
  private audioContext: AudioContext | null = null;

  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
    }
    return this.audioContext;
  }

  /**
   * Extract audio buffer from a media file
   */
  async extractAudioBuffer(mediaFileId: string): Promise<AudioBuffer | null> {
    const mediaStore = useMediaStore.getState();
    const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);

    if (!mediaFile || !mediaFile.file) {
      console.warn('[AudioAnalyzer] Media file not found:', mediaFileId);
      return null;
    }

    try {
      const ctx = this.getAudioContext();
      const arrayBuffer = await mediaFile.file.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      return audioBuffer;
    } catch (error) {
      console.error('[AudioAnalyzer] Failed to decode audio:', error);
      return null;
    }
  }

  /**
   * Analyze audio levels (RMS) at regular intervals
   */
  async analyzeLevels(
    mediaFileId: string,
    windowSizeMs: number = 100
  ): Promise<AudioCurve | null> {
    const audioBuffer = await this.extractAudioBuffer(mediaFileId);
    if (!audioBuffer) return null;

    const sampleRate = audioBuffer.sampleRate;
    const windowSize = Math.floor((windowSizeMs / 1000) * sampleRate);
    const channelData = audioBuffer.getChannelData(0); // Use first channel

    const levels: AudioLevel[] = [];

    for (let i = 0; i < channelData.length; i += windowSize) {
      const end = Math.min(i + windowSize, channelData.length);
      const window = channelData.slice(i, end);

      // Calculate RMS (Root Mean Square)
      let sum = 0;
      for (let j = 0; j < window.length; j++) {
        sum += window[j] * window[j];
      }
      const rms = Math.sqrt(sum / window.length);

      // Normalize to 0-1 range (RMS is typically 0-1 for normalized audio)
      const level = Math.min(1, rms * 3); // Scale up a bit for visibility

      levels.push({
        timestamp: (i / sampleRate) * 1000,
        level,
      });
    }

    return {
      mediaFileId,
      levels,
      sampleRate,
      windowSizeMs,
    };
  }

  /**
   * Generate a fingerprint for audio sync comparison
   * Downsamples audio to a manageable size for cross-correlation
   */
  async generateFingerprint(
    mediaFileId: string,
    targetSampleRate: number = 8000
  ): Promise<AudioFingerprint | null> {
    const audioBuffer = await this.extractAudioBuffer(mediaFileId);
    if (!audioBuffer) return null;

    const originalSampleRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);

    // Downsample by averaging
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.floor(channelData.length / ratio);
    const downsampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.floor((i + 1) * ratio);
      let sum = 0;
      for (let j = start; j < end && j < channelData.length; j++) {
        sum += channelData[j];
      }
      downsampled[i] = sum / (end - start);
    }

    return {
      mediaFileId,
      data: downsampled,
      sampleRate: targetSampleRate,
    };
  }

  /**
   * Get peak audio level at a specific timestamp
   */
  async getLevelAtTime(mediaFileId: string, timestampMs: number): Promise<number> {
    const audioBuffer = await this.extractAudioBuffer(mediaFileId);
    if (!audioBuffer) return 0;

    const sampleRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);
    const sampleIndex = Math.floor((timestampMs / 1000) * sampleRate);

    if (sampleIndex < 0 || sampleIndex >= channelData.length) {
      return 0;
    }

    // Get RMS of a small window around the timestamp
    const windowSize = Math.floor(sampleRate * 0.05); // 50ms window
    const start = Math.max(0, sampleIndex - windowSize / 2);
    const end = Math.min(channelData.length, sampleIndex + windowSize / 2);

    let sum = 0;
    for (let i = start; i < end; i++) {
      sum += channelData[i] * channelData[i];
    }

    return Math.sqrt(sum / (end - start));
  }

  /**
   * Cleanup
   */
  dispose(): void {
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// Singleton instance
export const audioAnalyzer = new AudioAnalyzer();
