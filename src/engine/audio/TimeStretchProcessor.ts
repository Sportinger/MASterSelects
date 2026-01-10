/**
 * TimeStretchProcessor - Handle speed changes and pitch preservation
 *
 * Uses SoundTouchJS for high-quality time-stretching with independent
 * control over tempo and pitch.
 *
 * Features:
 * - Constant speed processing
 * - Variable speed with keyframes
 * - Pitch preservation option
 * - Chunked processing for memory efficiency
 */

import { SoundTouch, SimpleFilter } from 'soundtouch-ts';
import type { Keyframe } from '../../types';
import { interpolateKeyframes } from '../../utils/keyframeInterpolation';

export interface TimeStretchSettings {
  preservePitch: boolean;  // Keep original pitch when changing speed
  quality: 'fast' | 'normal' | 'high';  // Processing quality
}

export interface TimeStretchProgress {
  processedSamples: number;
  totalSamples: number;
  percent: number;
  currentSpeed: number;
}

export type TimeStretchProgressCallback = (progress: TimeStretchProgress) => void;

// Speed keyframe for internal use
interface SpeedPoint {
  time: number;
  speed: number;
}

export class TimeStretchProcessor {
  private settings: TimeStretchSettings;

  constructor(settings?: Partial<TimeStretchSettings>) {
    this.settings = {
      preservePitch: settings?.preservePitch ?? true,
      quality: settings?.quality ?? 'normal',
    };
  }

  /**
   * Process audio with constant speed
   * @param buffer - Source AudioBuffer
   * @param speed - Playback speed (0.1 to 10.0)
   * @param preservePitch - Override pitch preservation setting
   * @returns Processed AudioBuffer
   */
  async processConstantSpeed(
    buffer: AudioBuffer,
    speed: number,
    preservePitch?: boolean
  ): Promise<AudioBuffer> {
    const shouldPreservePitch = preservePitch ?? this.settings.preservePitch;

    // Clamp speed to valid range
    const clampedSpeed = Math.max(0.1, Math.min(10, speed));

    console.log(`[TimeStretch] Processing constant speed: ${clampedSpeed}x, preservePitch: ${shouldPreservePitch}`);

    // If speed is 1.0, no processing needed
    if (Math.abs(clampedSpeed - 1.0) < 0.001) {
      return buffer;
    }

    // If not preserving pitch, use simple resampling (faster)
    if (!shouldPreservePitch) {
      return this.resampleForSpeed(buffer, clampedSpeed);
    }

    // Use SoundTouch for pitch-preserved time-stretching
    return this.soundTouchProcess(buffer, clampedSpeed);
  }

  /**
   * Process audio with speed keyframes
   * @param buffer - Source AudioBuffer
   * @param keyframes - All keyframes for the clip
   * @param defaultSpeed - Default speed if no keyframes at a given time
   * @param clipDuration - Timeline duration of the clip
   * @param preservePitch - Override pitch preservation setting
   * @param onProgress - Optional progress callback
   * @returns Processed AudioBuffer
   */
  async processWithKeyframes(
    buffer: AudioBuffer,
    keyframes: Keyframe[],
    defaultSpeed: number,
    clipDuration: number,
    preservePitch?: boolean,
    onProgress?: TimeStretchProgressCallback
  ): Promise<AudioBuffer> {
    const shouldPreservePitch = preservePitch ?? this.settings.preservePitch;

    // Filter speed keyframes
    const speedKeyframes = keyframes
      .filter(k => k.property === 'speed')
      .sort((a, b) => a.time - b.time);

    // If no speed keyframes, use constant speed
    if (speedKeyframes.length === 0) {
      return this.processConstantSpeed(buffer, defaultSpeed, shouldPreservePitch);
    }

    // If single keyframe, use constant speed
    if (speedKeyframes.length === 1) {
      return this.processConstantSpeed(buffer, speedKeyframes[0].value, shouldPreservePitch);
    }

    console.log(`[TimeStretch] Processing with ${speedKeyframes.length} speed keyframes`);

    // For variable speed, we need to process in segments
    return this.processVariableSpeed(
      buffer,
      keyframes,
      defaultSpeed,
      clipDuration,
      shouldPreservePitch,
      onProgress
    );
  }

  /**
   * Process with variable speed using segmented approach
   */
  private async processVariableSpeed(
    buffer: AudioBuffer,
    keyframes: Keyframe[],
    defaultSpeed: number,
    clipDuration: number,
    preservePitch: boolean,
    onProgress?: TimeStretchProgressCallback
  ): Promise<AudioBuffer> {
    const sampleRate = buffer.sampleRate;
    const channels = buffer.numberOfChannels;

    // Calculate output duration based on speed integration
    // For variable speed, output duration = timeline duration (clipDuration)
    const outputSamples = Math.ceil(clipDuration * sampleRate);

    // Create output buffer
    const audioContext = new AudioContext();
    const outputBuffer = audioContext.createBuffer(channels, outputSamples, sampleRate);

    // Segment size for processing (100ms segments)
    const segmentDuration = 0.1; // seconds
    const numSegments = Math.ceil(clipDuration / segmentDuration);

    // Process each segment
    for (let segIdx = 0; segIdx < numSegments; segIdx++) {
      const segmentStart = segIdx * segmentDuration;
      const segmentEnd = Math.min((segIdx + 1) * segmentDuration, clipDuration);

      // Get average speed for this segment
      const midTime = (segmentStart + segmentEnd) / 2;
      const speed = this.getSpeedAtTime(keyframes, midTime, defaultSpeed);
      const absSpeed = Math.abs(speed);

      // Calculate source range for this segment
      const sourceStart = this.integrateSpeed(keyframes, 0, segmentStart, defaultSpeed);
      const sourceEnd = this.integrateSpeed(keyframes, 0, segmentEnd, defaultSpeed);

      // Handle reverse playback
      const actualSourceStart = Math.min(sourceStart, sourceEnd);
      const actualSourceEnd = Math.max(sourceStart, sourceEnd);

      // Extract source segment
      const sourceSamples = this.extractSegment(
        buffer,
        Math.max(0, actualSourceStart),
        Math.min(buffer.duration, actualSourceEnd)
      );

      if (sourceSamples.length === 0) continue;

      // Process segment with current speed
      let processedSegment: Float32Array[];
      if (preservePitch && Math.abs(absSpeed - 1.0) > 0.01) {
        processedSegment = await this.stretchSegment(sourceSamples, absSpeed, sampleRate, channels);
      } else if (!preservePitch && Math.abs(absSpeed - 1.0) > 0.01) {
        processedSegment = this.resampleSegment(sourceSamples, absSpeed, channels);
      } else {
        processedSegment = sourceSamples;
      }

      // If reverse, flip the segment
      if (speed < 0) {
        processedSegment = processedSegment.map(ch => {
          const reversed = new Float32Array(ch.length);
          for (let i = 0; i < ch.length; i++) {
            reversed[i] = ch[ch.length - 1 - i];
          }
          return reversed;
        });
      }

      // Copy to output buffer
      const outputStartSample = Math.floor(segmentStart * sampleRate);
      const outputEndSample = Math.floor(segmentEnd * sampleRate);
      const targetLength = outputEndSample - outputStartSample;

      for (let ch = 0; ch < channels; ch++) {
        const outputData = outputBuffer.getChannelData(ch);
        const segmentData = processedSegment[ch] || processedSegment[0];

        // Resample segment to fit target length if needed
        for (let i = 0; i < targetLength && outputStartSample + i < outputSamples; i++) {
          const srcIdx = Math.floor(i * segmentData.length / targetLength);
          if (srcIdx < segmentData.length) {
            outputData[outputStartSample + i] = segmentData[srcIdx];
          }
        }
      }

      // Progress callback
      if (onProgress) {
        onProgress({
          processedSamples: outputStartSample + targetLength,
          totalSamples: outputSamples,
          percent: Math.round(((segIdx + 1) / numSegments) * 100),
          currentSpeed: speed,
        });
      }

      // Yield to UI
      if (segIdx % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    audioContext.close();
    return outputBuffer;
  }

  /**
   * Extract a segment from buffer as Float32Array per channel
   */
  private extractSegment(
    buffer: AudioBuffer,
    startTime: number,
    endTime: number
  ): Float32Array[] {
    const startSample = Math.floor(startTime * buffer.sampleRate);
    const endSample = Math.ceil(endTime * buffer.sampleRate);
    const length = Math.max(0, Math.min(endSample - startSample, buffer.length - startSample));

    if (length === 0) return [];

    const segments: Float32Array[] = [];
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const channelData = buffer.getChannelData(ch);
      const segment = new Float32Array(length);
      for (let i = 0; i < length; i++) {
        segment[i] = channelData[startSample + i] || 0;
      }
      segments.push(segment);
    }
    return segments;
  }

  /**
   * Time-stretch a segment using SoundTouch
   */
  private async stretchSegment(
    segments: Float32Array[],
    speed: number,
    sampleRate: number,
    channels: number
  ): Promise<Float32Array[]> {
    const soundtouch = new SoundTouch();
    soundtouch.tempo = speed;
    soundtouch.pitch = 1.0; // Keep pitch

    // Configure quality
    switch (this.settings.quality) {
      case 'fast':
        soundtouch.setParameter(0, 8); // SEQUENCE_MS
        soundtouch.setParameter(1, 4); // SEEKWINDOW_MS
        break;
      case 'high':
        soundtouch.setParameter(0, 82); // SEQUENCE_MS
        soundtouch.setParameter(1, 28); // SEEKWINDOW_MS
        break;
    }

    // Create interleaved input
    const length = segments[0].length;
    const interleaved = new Float32Array(length * channels);
    for (let i = 0; i < length; i++) {
      for (let ch = 0; ch < channels; ch++) {
        interleaved[i * channels + ch] = segments[ch]?.[i] || 0;
      }
    }

    // Create simple filter for processing
    const filter = new SimpleFilter(
      {
        extract: (target: Float32Array, numFrames: number, position: number) => {
          const start = position * channels;
          const framesToCopy = Math.min(numFrames, Math.floor((interleaved.length - start) / channels));
          for (let i = 0; i < framesToCopy * channels; i++) {
            target[i] = interleaved[start + i] || 0;
          }
          return framesToCopy;
        },
      },
      soundtouch
    );

    // Process and collect output
    const outputLength = Math.ceil(length / speed);
    const outputInterleaved = new Float32Array(outputLength * channels);

    let outputPos = 0;
    const chunkSize = 4096;
    const chunk = new Float32Array(chunkSize * channels);

    while (outputPos < outputLength) {
      const framesRead = filter.extract(chunk, chunkSize);
      if (framesRead === 0) break;

      for (let i = 0; i < framesRead * channels && outputPos * channels + i < outputInterleaved.length; i++) {
        outputInterleaved[outputPos * channels + i] = chunk[i];
      }
      outputPos += framesRead;
    }

    // De-interleave output
    const actualLength = Math.min(outputPos, outputLength);
    const output: Float32Array[] = [];
    for (let ch = 0; ch < channels; ch++) {
      const channelData = new Float32Array(actualLength);
      for (let i = 0; i < actualLength; i++) {
        channelData[i] = outputInterleaved[i * channels + ch] || 0;
      }
      output.push(channelData);
    }

    return output;
  }

  /**
   * Simple resampling (changes pitch with speed)
   */
  private resampleSegment(
    segments: Float32Array[],
    speed: number,
    channels: number
  ): Float32Array[] {
    const inputLength = segments[0].length;
    const outputLength = Math.ceil(inputLength / speed);

    const output: Float32Array[] = [];
    for (let ch = 0; ch < channels; ch++) {
      const input = segments[ch] || segments[0];
      const channelOutput = new Float32Array(outputLength);

      for (let i = 0; i < outputLength; i++) {
        const srcIdx = i * speed;
        const srcIdxFloor = Math.floor(srcIdx);
        const frac = srcIdx - srcIdxFloor;

        // Linear interpolation
        const s1 = input[srcIdxFloor] || 0;
        const s2 = input[srcIdxFloor + 1] || s1;
        channelOutput[i] = s1 + (s2 - s1) * frac;
      }

      output.push(channelOutput);
    }

    return output;
  }

  /**
   * Get interpolated speed at a specific time
   */
  private getSpeedAtTime(keyframes: Keyframe[], time: number, defaultSpeed: number): number {
    return interpolateKeyframes(keyframes, 'speed' as any, time, defaultSpeed);
  }

  /**
   * Integrate speed from startTime to endTime (source time consumed)
   */
  private integrateSpeed(
    keyframes: Keyframe[],
    startTime: number,
    endTime: number,
    defaultSpeed: number
  ): number {
    const speedKeyframes = keyframes.filter(k => k.property === 'speed');

    if (speedKeyframes.length === 0) {
      return (endTime - startTime) * defaultSpeed;
    }

    // Trapezoidal integration
    const steps = 20;
    const dt = (endTime - startTime) / steps;
    let integral = 0;

    for (let i = 0; i < steps; i++) {
      const t0 = startTime + i * dt;
      const t1 = startTime + (i + 1) * dt;
      const s0 = this.getSpeedAtTime(keyframes, t0, defaultSpeed);
      const s1 = this.getSpeedAtTime(keyframes, t1, defaultSpeed);
      integral += (s0 + s1) / 2 * dt;
    }

    return integral;
  }

  /**
   * Process entire buffer with SoundTouch (for constant speed)
   */
  private async soundTouchProcess(buffer: AudioBuffer, speed: number): Promise<AudioBuffer> {
    const channels = buffer.numberOfChannels;
    const segments: Float32Array[] = [];

    for (let ch = 0; ch < channels; ch++) {
      segments.push(buffer.getChannelData(ch).slice());
    }

    const processed = await this.stretchSegment(segments, speed, buffer.sampleRate, channels);

    // Create output buffer
    const audioContext = new AudioContext();
    const outputLength = processed[0]?.length || 0;
    const outputBuffer = audioContext.createBuffer(channels, outputLength, buffer.sampleRate);

    for (let ch = 0; ch < channels; ch++) {
      const outputData = outputBuffer.getChannelData(ch);
      const srcData = processed[ch] || processed[0];
      for (let i = 0; i < outputLength; i++) {
        outputData[i] = srcData[i] || 0;
      }
    }

    audioContext.close();
    return outputBuffer;
  }

  /**
   * Simple resampling for speed without pitch preservation
   */
  private async resampleForSpeed(buffer: AudioBuffer, speed: number): Promise<AudioBuffer> {
    const outputLength = Math.ceil(buffer.length / speed);
    const audioContext = new AudioContext();
    const outputBuffer = audioContext.createBuffer(
      buffer.numberOfChannels,
      outputLength,
      buffer.sampleRate
    );

    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const input = buffer.getChannelData(ch);
      const output = outputBuffer.getChannelData(ch);

      for (let i = 0; i < outputLength; i++) {
        const srcIdx = i * speed;
        const srcIdxFloor = Math.floor(srcIdx);
        const frac = srcIdx - srcIdxFloor;

        const s1 = input[srcIdxFloor] || 0;
        const s2 = input[srcIdxFloor + 1] || s1;
        output[i] = s1 + (s2 - s1) * frac;
      }
    }

    audioContext.close();
    return outputBuffer;
  }

  /**
   * Update settings
   */
  updateSettings(settings: Partial<TimeStretchSettings>): void {
    this.settings = { ...this.settings, ...settings };
  }

  /**
   * Get current settings
   */
  getSettings(): TimeStretchSettings {
    return { ...this.settings };
  }
}

// Default instance
export const timeStretchProcessor = new TimeStretchProcessor();
