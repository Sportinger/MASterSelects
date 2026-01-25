/**
 * CompositionAudioMixer - Renders audio from a composition to a mixdown buffer
 *
 * Used when nesting compositions to:
 * 1. Mix down all audio tracks in the nested comp
 * 2. Generate a waveform for display
 * 3. Create a playable audio element for timeline playback
 */

import { Logger } from './logger';
import { useMediaStore } from '../stores/mediaStore';

const log = Logger.create('CompositionAudioMixer');
import { useTimelineStore } from '../stores/timeline';
import { AudioMixer, type AudioTrackData } from '../engine/audio/AudioMixer';
import { audioExtractor } from '../engine/audio/AudioExtractor';
import type { TimelineClip, TimelineTrack, SerializableClip } from '../types';
import { generateWaveformFromBuffer } from '../stores/timeline/utils';

export interface CompositionMixdownResult {
  buffer: AudioBuffer;
  waveform: number[];
  duration: number;
  hasAudio: boolean;
}

export interface MixdownProgress {
  phase: 'loading' | 'extracting' | 'mixing' | 'waveform' | 'complete';
  percent: number;
  message?: string;
}

export type MixdownProgressCallback = (progress: MixdownProgress) => void;

class CompositionAudioMixerService {
  private audioContext: AudioContext | null = null;

  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: 48000 });
    }
    return this.audioContext;
  }

  /**
   * Mix down all audio from a composition to a single buffer
   */
  async mixdownComposition(
    compositionId: string,
    onProgress?: MixdownProgressCallback
  ): Promise<CompositionMixdownResult | null> {
    const { activeCompositionId, compositions, files } = useMediaStore.getState();
    const composition = compositions.find(c => c.id === compositionId);

    if (!composition) {
      console.warn(`[CompositionAudioMixer] Composition ${compositionId} not found`);
      return null;
    }

    onProgress?.({ phase: 'loading', percent: 0, message: 'Loading composition data...' });

    // Get clips and tracks - use timeline store if active, otherwise use serialized data
    const isActiveComp = compositionId === activeCompositionId;
    let clips: (SerializableClip | TimelineClip)[];
    let tracks: TimelineTrack[];

    if (isActiveComp) {
      const timelineState = useTimelineStore.getState();
      clips = timelineState.clips;
      tracks = timelineState.tracks;
    } else if (composition.timelineData) {
      clips = composition.timelineData.clips || [];
      tracks = composition.timelineData.tracks || [];
    } else {
      console.warn(`[CompositionAudioMixer] Composition ${compositionId} has no timeline data`);
      return null;
    }

    // Find audio tracks
    const audioTracks = tracks.filter(t => t.type === 'audio');
    if (audioTracks.length === 0) {
      console.log(`[CompositionAudioMixer] No audio tracks in composition ${composition.name}`);
      return {
        buffer: this.createSilentBuffer(composition.duration || 10),
        waveform: [],
        duration: composition.duration || 10,
        hasAudio: false,
      };
    }

    // Find clips on audio tracks
    const audioTrackIds = new Set(audioTracks.map(t => t.id));
    const audioClips = clips.filter(c => audioTrackIds.has(c.trackId));

    if (audioClips.length === 0) {
      console.log(`[CompositionAudioMixer] No audio clips in composition ${composition.name}`);
      return {
        buffer: this.createSilentBuffer(composition.duration || 10),
        waveform: [],
        duration: composition.duration || 10,
        hasAudio: false,
      };
    }

    console.log(`[CompositionAudioMixer] Processing ${audioClips.length} audio clips from ${composition.name}`);

    onProgress?.({ phase: 'extracting', percent: 10, message: 'Extracting audio...' });

    // Extract and decode audio from each clip
    const trackDataList: AudioTrackData[] = [];
    const duration = composition.duration || Math.max(...clips.map(c => c.startTime + c.duration), 10);

    for (let i = 0; i < audioClips.length; i++) {
      const clip = audioClips[i];
      const track = audioTracks.find(t => t.id === clip.trackId);

      // Find the source file
      let file: File | undefined;

      // Check if clip has file directly (TimelineClip has file property)
      if ('file' in clip && clip.file) {
        file = clip.file;
      } else {
        // Look up in media files
        const mediaFile = files.find(f => f.name === clip.name || f.id === (clip as any).mediaFileId);
        if (mediaFile?.file) {
          file = mediaFile.file;
        }
      }

      if (!file) {
        console.warn(`[CompositionAudioMixer] No file found for clip ${clip.name}`);
        continue;
      }

      try {
        // Extract audio from the file
        const extractedBuffer = await audioExtractor.extractAudio(file, clip.id);
        if (!extractedBuffer) {
          console.warn(`[CompositionAudioMixer] Failed to extract audio from ${clip.name}`);
          continue;
        }

        // Calculate what portion of the audio to use based on in/out points
        const inPoint = clip.inPoint || 0;
        const outPoint = clip.outPoint || extractedBuffer.duration;

        // Create a trimmed buffer if needed
        let processedBuffer = extractedBuffer;
        if (inPoint > 0 || outPoint < extractedBuffer.duration) {
          processedBuffer = this.trimBuffer(extractedBuffer, inPoint, outPoint);
        }

        trackDataList.push({
          clipId: clip.id,
          buffer: processedBuffer,
          startTime: clip.startTime,
          trackId: clip.trackId,
          trackMuted: track?.muted || false,
          trackSolo: track?.solo || false,
          clipVolume: clip.transform?.opacity ?? 1, // Use opacity as volume proxy
        });
      } catch (e) {
        console.error(`[CompositionAudioMixer] Error processing ${clip.name}:`, e);
      }

      onProgress?.({
        phase: 'extracting',
        percent: 10 + Math.round((i / audioClips.length) * 50),
        message: `Extracting ${clip.name}...`,
      });
    }

    if (trackDataList.length === 0) {
      console.log(`[CompositionAudioMixer] No audio could be extracted from composition ${composition.name}`);
      return {
        buffer: this.createSilentBuffer(duration),
        waveform: [],
        duration,
        hasAudio: false,
      };
    }

    onProgress?.({ phase: 'mixing', percent: 60, message: 'Mixing audio tracks...' });

    // Mix all tracks together
    const mixer = new AudioMixer({
      sampleRate: 48000,
      numberOfChannels: 2,
      normalize: true,
    });

    const mixedBuffer = await mixer.mixTracks(trackDataList, duration);

    onProgress?.({ phase: 'waveform', percent: 90, message: 'Generating waveform...' });

    // Generate waveform from mixed buffer
    const waveform = generateWaveformFromBuffer(mixedBuffer, 50);

    onProgress?.({ phase: 'complete', percent: 100, message: 'Complete' });

    console.log(`[CompositionAudioMixer] Mixdown complete: ${duration.toFixed(2)}s, ${waveform.length} waveform samples`);

    return {
      buffer: mixedBuffer,
      waveform,
      duration,
      hasAudio: true,
    };
  }

  /**
   * Create an audio element from an AudioBuffer for playback
   */
  createAudioElement(buffer: AudioBuffer): HTMLAudioElement {
    // Convert AudioBuffer to WAV blob
    const wavBlob = this.audioBufferToWav(buffer);
    const url = URL.createObjectURL(wavBlob);

    const audio = document.createElement('audio');
    audio.src = url;
    audio.preload = 'auto';

    return audio;
  }

  /**
   * Trim an AudioBuffer to a specific range
   */
  private trimBuffer(buffer: AudioBuffer, startTime: number, endTime: number): AudioBuffer {
    const ctx = this.getAudioContext();
    const startSample = Math.floor(startTime * buffer.sampleRate);
    const endSample = Math.floor(endTime * buffer.sampleRate);
    const length = endSample - startSample;

    const trimmed = ctx.createBuffer(
      buffer.numberOfChannels,
      length,
      buffer.sampleRate
    );

    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const sourceData = buffer.getChannelData(ch);
      const destData = trimmed.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        destData[i] = sourceData[startSample + i] || 0;
      }
    }

    return trimmed;
  }

  /**
   * Create a silent buffer of specified duration
   */
  private createSilentBuffer(duration: number): AudioBuffer {
    const ctx = this.getAudioContext();
    const sampleRate = 48000;
    const length = Math.ceil(duration * sampleRate);
    return ctx.createBuffer(2, length, sampleRate);
  }

  /**
   * Convert AudioBuffer to WAV Blob
   */
  private audioBufferToWav(buffer: AudioBuffer): Blob {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const length = buffer.length;
    const bytesPerSample = 2; // 16-bit
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = length * blockAlign;
    const bufferSize = 44 + dataSize;

    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, 1, true); // AudioFormat (PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytesPerSample * 8, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    // Interleave and convert to 16-bit
    const channels: Float32Array[] = [];
    for (let ch = 0; ch < numChannels; ch++) {
      channels.push(buffer.getChannelData(ch));
    }

    let offset = 44;
    for (let i = 0; i < length; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const sample = Math.max(-1, Math.min(1, channels[ch][i]));
        const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset, int16, true);
        offset += 2;
      }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }
}

export const compositionAudioMixer = new CompositionAudioMixerService();
