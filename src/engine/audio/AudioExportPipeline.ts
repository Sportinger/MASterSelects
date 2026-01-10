/**
 * AudioExportPipeline - Orchestrates the complete audio export process
 *
 * Coordinates:
 * 1. AudioExtractor - Decode audio from files
 * 2. TimeStretchProcessor - Handle speed/pitch changes
 * 3. AudioEffectRenderer - Apply EQ and volume
 * 4. AudioMixer - Mix all tracks
 * 5. AudioEncoder - Encode to AAC
 *
 * Returns encoded audio chunks ready for muxing with video
 */

import { AudioExtractor, audioExtractor } from './AudioExtractor';
import { AudioEncoderWrapper, type EncodedAudioResult } from './AudioEncoder';
import { AudioMixer, type AudioTrackData } from './AudioMixer';
import { TimeStretchProcessor, timeStretchProcessor } from './TimeStretchProcessor';
import { AudioEffectRenderer, audioEffectRenderer } from './AudioEffectRenderer';
import { useTimelineStore } from '../../stores/timeline';
import type { TimelineClip, TimelineTrack, Keyframe } from '../../types';

export interface AudioExportSettings {
  sampleRate: number;       // 44100 or 48000
  bitrate: number;          // 128000 - 320000
  normalize: boolean;       // Peak normalize output
}

export interface AudioExportProgress {
  phase: 'extracting' | 'processing' | 'effects' | 'mixing' | 'encoding' | 'complete';
  percent: number;
  currentClip?: string;
  message?: string;
}

export type AudioExportProgressCallback = (progress: AudioExportProgress) => void;

export class AudioExportPipeline {
  private extractor: AudioExtractor;
  private encoder: AudioEncoderWrapper | null = null;
  private mixer: AudioMixer;
  private timeStretch: TimeStretchProcessor;
  private effectRenderer: AudioEffectRenderer;
  private settings: AudioExportSettings;
  private cancelled = false;

  constructor(settings?: Partial<AudioExportSettings>) {
    this.settings = {
      sampleRate: settings?.sampleRate ?? 48000,
      bitrate: settings?.bitrate ?? 256000,
      normalize: settings?.normalize ?? false,
    };

    this.extractor = audioExtractor;
    this.mixer = new AudioMixer({
      sampleRate: this.settings.sampleRate,
      normalize: this.settings.normalize,
    });
    this.timeStretch = timeStretchProcessor;
    this.effectRenderer = audioEffectRenderer;
  }

  /**
   * Export all audio from timeline
   * @param startTime - Export start time
   * @param endTime - Export end time
   * @param onProgress - Progress callback
   * @returns Encoded audio result with chunks for muxing
   */
  async exportAudio(
    startTime: number,
    endTime: number,
    onProgress?: AudioExportProgressCallback
  ): Promise<EncodedAudioResult | null> {
    this.cancelled = false;

    const { clips, tracks, clipKeyframes } = useTimelineStore.getState();
    const duration = endTime - startTime;

    console.log(`[AudioExportPipeline] Starting export: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s (${duration.toFixed(2)}s)`);

    // 1. Find all clips with audio in the export range
    const audioClips = this.getClipsWithAudio(clips, tracks, startTime, endTime);

    if (audioClips.length === 0) {
      console.log('[AudioExportPipeline] No audio clips found in export range');
      return null;
    }

    console.log(`[AudioExportPipeline] Found ${audioClips.length} clips with audio`);

    try {
      // 2. Extract audio from all clips
      onProgress?.({ phase: 'extracting', percent: 0, message: 'Extracting audio...' });
      const extractedBuffers = await this.extractAllAudio(audioClips, onProgress);

      if (this.cancelled) return null;

      // 3. Process speed/pitch for each clip
      onProgress?.({ phase: 'processing', percent: 0, message: 'Processing speed changes...' });
      const processedBuffers = await this.processAllSpeed(
        audioClips,
        extractedBuffers,
        clipKeyframes,
        onProgress
      );

      if (this.cancelled) return null;

      // 4. Render effects for each clip
      onProgress?.({ phase: 'effects', percent: 0, message: 'Applying effects...' });
      const effectBuffers = await this.renderAllEffects(
        audioClips,
        processedBuffers,
        clipKeyframes,
        onProgress
      );

      if (this.cancelled) return null;

      // 5. Mix all tracks
      onProgress?.({ phase: 'mixing', percent: 0, message: 'Mixing tracks...' });
      const trackData = this.prepareTrackData(audioClips, effectBuffers, tracks, startTime);
      const mixedBuffer = await this.mixer.mixTracks(trackData, duration);

      if (this.cancelled) return null;

      // 6. Encode to AAC
      onProgress?.({ phase: 'encoding', percent: 0, message: 'Encoding audio...' });
      const result = await this.encodeAudio(mixedBuffer, onProgress);

      // 7. Cleanup
      this.extractor.clearCache();

      onProgress?.({ phase: 'complete', percent: 100, message: 'Audio export complete' });

      console.log(`[AudioExportPipeline] Export complete: ${result.chunks.length} chunks`);
      return result;

    } catch (error) {
      console.error('[AudioExportPipeline] Export failed:', error);
      this.extractor.clearCache();
      throw error;
    }
  }

  /**
   * Cancel the export
   */
  cancel(): void {
    this.cancelled = true;
    console.log('[AudioExportPipeline] Export cancelled');
  }

  /**
   * Get clips that have audio in the export range
   */
  private getClipsWithAudio(
    clips: TimelineClip[],
    tracks: TimelineTrack[],
    startTime: number,
    endTime: number
  ): TimelineClip[] {
    return clips.filter(clip => {
      // Check if clip is in range
      const clipEnd = clip.startTime + clip.duration;
      if (clipEnd <= startTime || clip.startTime >= endTime) {
        return false;
      }

      // Nested composition with mixdown audio
      if (clip.isComposition && clip.mixdownBuffer && clip.hasMixdownAudio) {
        // Check track is not muted (nested comps are on video tracks)
        const track = tracks.find(t => t.id === clip.trackId);
        if (track && !track.visible) return false;
        return true;
      }

      // Check if clip has audio source
      if (!clip.source?.audioElement && !clip.source?.videoElement) {
        return false;
      }

      // For video clips, we need the linked audio clip
      // For audio clips, we use them directly
      if (clip.source.type === 'audio') {
        // Check track is not muted
        const track = tracks.find(t => t.id === clip.trackId);
        if (track?.muted) return false;
        return true;
      }

      // Video clips don't have audio in this architecture
      // (audio is in separate linked clips)
      return false;
    });
  }

  /**
   * Extract audio from all clips
   */
  private async extractAllAudio(
    clips: TimelineClip[],
    onProgress?: AudioExportProgressCallback
  ): Promise<Map<string, AudioBuffer>> {
    const buffers = new Map<string, AudioBuffer>();

    for (let i = 0; i < clips.length; i++) {
      const clip = clips[i];

      if (this.cancelled) break;

      onProgress?.({
        phase: 'extracting',
        percent: Math.round((i / clips.length) * 100),
        currentClip: clip.name,
        message: `Extracting: ${clip.name}`,
      });

      try {
        let buffer: AudioBuffer;

        // Nested composition with pre-mixed audio buffer
        if (clip.isComposition && clip.mixdownBuffer) {
          buffer = clip.mixdownBuffer;
          console.log(`[AudioExportPipeline] Using mixdown buffer for nested comp ${clip.name}`);
        } else if (clip.source?.audioElement) {
          // Extract from audio element
          buffer = await this.extractor.extractFromElement(
            clip.source.audioElement,
            clip.id
          );
        } else if (clip.file) {
          // Extract from file
          buffer = await this.extractor.extractAudio(clip.file, clip.id);
        } else {
          console.warn(`[AudioExportPipeline] No audio source for clip ${clip.id}`);
          continue;
        }

        // Trim to clip's in/out points
        const trimmedBuffer = this.extractor.trimBuffer(
          buffer,
          clip.inPoint,
          clip.outPoint
        );

        buffers.set(clip.id, trimmedBuffer);
      } catch (error) {
        console.error(`[AudioExportPipeline] Failed to extract audio from ${clip.name}:`, error);
        // Create silent buffer as fallback
        buffers.set(clip.id, this.extractor.createSilentBuffer(clip.duration));
      }
    }

    return buffers;
  }

  /**
   * Process speed/pitch for all clips
   */
  private async processAllSpeed(
    clips: TimelineClip[],
    buffers: Map<string, AudioBuffer>,
    clipKeyframes: Map<string, Keyframe[]>,
    onProgress?: AudioExportProgressCallback
  ): Promise<Map<string, AudioBuffer>> {
    const processed = new Map<string, AudioBuffer>();

    for (let i = 0; i < clips.length; i++) {
      const clip = clips[i];
      const buffer = buffers.get(clip.id);

      if (!buffer || this.cancelled) continue;

      onProgress?.({
        phase: 'processing',
        percent: Math.round((i / clips.length) * 100),
        currentClip: clip.name,
        message: `Processing: ${clip.name}`,
      });

      const keyframes = clipKeyframes.get(clip.id) || [];
      const defaultSpeed = clip.speed ?? 1;
      const preservesPitch = clip.preservesPitch !== false;

      // Check if we have speed keyframes
      const speedKeyframes = keyframes.filter(k => k.property === 'speed');

      let processedBuffer: AudioBuffer;

      if (speedKeyframes.length > 0) {
        // Variable speed with keyframes
        processedBuffer = await this.timeStretch.processWithKeyframes(
          buffer,
          keyframes,
          defaultSpeed,
          clip.duration,
          preservesPitch
        );
      } else if (Math.abs(defaultSpeed - 1) > 0.01) {
        // Constant non-1x speed
        processedBuffer = await this.timeStretch.processConstantSpeed(
          buffer,
          Math.abs(defaultSpeed),
          preservesPitch
        );
      } else {
        // No speed change
        processedBuffer = buffer;
      }

      processed.set(clip.id, processedBuffer);
    }

    return processed;
  }

  /**
   * Render effects for all clips
   */
  private async renderAllEffects(
    clips: TimelineClip[],
    buffers: Map<string, AudioBuffer>,
    clipKeyframes: Map<string, Keyframe[]>,
    onProgress?: AudioExportProgressCallback
  ): Promise<Map<string, AudioBuffer>> {
    const processed = new Map<string, AudioBuffer>();

    for (let i = 0; i < clips.length; i++) {
      const clip = clips[i];
      const buffer = buffers.get(clip.id);

      if (!buffer || this.cancelled) continue;

      onProgress?.({
        phase: 'effects',
        percent: Math.round((i / clips.length) * 100),
        currentClip: clip.name,
        message: `Effects: ${clip.name}`,
      });

      const keyframes = clipKeyframes.get(clip.id) || [];
      const effects = clip.effects || [];

      const processedBuffer = await this.effectRenderer.renderEffects(
        buffer,
        effects,
        keyframes,
        clip.duration
      );

      processed.set(clip.id, processedBuffer);
    }

    return processed;
  }

  /**
   * Prepare track data for mixer
   */
  private prepareTrackData(
    clips: TimelineClip[],
    buffers: Map<string, AudioBuffer>,
    tracks: TimelineTrack[],
    exportStartTime: number
  ): AudioTrackData[] {
    const trackData: AudioTrackData[] = [];

    // Check for soloed tracks
    const hasSolo = tracks.some(t => t.type === 'audio' && t.solo);

    for (const clip of clips) {
      const buffer = buffers.get(clip.id);
      if (!buffer) continue;

      const track = tracks.find(t => t.id === clip.trackId);
      if (!track) continue;

      // Skip if track is muted, or if solo mode is active and this track isn't soloed
      if (track.muted || (hasSolo && !track.solo)) continue;

      trackData.push({
        clipId: clip.id,
        buffer,
        startTime: clip.startTime - exportStartTime, // Adjust for export range
        trackId: clip.trackId,
        trackMuted: track.muted,
        trackSolo: track.solo,
      });
    }

    return trackData;
  }

  /**
   * Encode mixed audio to AAC
   */
  private async encodeAudio(
    buffer: AudioBuffer,
    onProgress?: AudioExportProgressCallback
  ): Promise<EncodedAudioResult> {
    // Ensure stereo
    let stereoBuffer = buffer;
    if (buffer.numberOfChannels === 1) {
      stereoBuffer = this.extractor.convertToStereo(buffer);
    }

    // Resample if needed
    if (stereoBuffer.sampleRate !== this.settings.sampleRate) {
      stereoBuffer = await this.extractor.resampleBuffer(
        stereoBuffer,
        this.settings.sampleRate
      );
    }

    // Create encoder
    this.encoder = new AudioEncoderWrapper({
      sampleRate: this.settings.sampleRate,
      numberOfChannels: 2,
      bitrate: this.settings.bitrate,
    });

    const supported = await this.encoder.init();
    if (!supported) {
      throw new Error('AAC audio encoding is not supported in this browser');
    }

    // Encode with progress
    await this.encoder.encode(stereoBuffer, (progress) => {
      onProgress?.({
        phase: 'encoding',
        percent: progress.percent,
        message: `Encoding: ${progress.percent}%`,
      });
    });

    return await this.encoder.finalize();
  }

  /**
   * Get current settings
   */
  getSettings(): AudioExportSettings {
    return { ...this.settings };
  }

  /**
   * Update settings
   */
  updateSettings(settings: Partial<AudioExportSettings>): void {
    this.settings = { ...this.settings, ...settings };
    this.mixer.updateSettings({
      sampleRate: this.settings.sampleRate,
      normalize: this.settings.normalize,
    });
  }

  /**
   * Check if audio export is supported
   */
  static async isSupported(): Promise<boolean> {
    return await AudioEncoderWrapper.isSupported();
  }
}

// Default instance
export const audioExportPipeline = new AudioExportPipeline();
