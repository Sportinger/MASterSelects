// Audio Extractor - Extract audio from video files using MP4Box.js (no FFmpeg needed)
// Much faster than FFmpeg WASM since we just copy without re-encoding

import { Logger } from './logger';
import * as MP4BoxModule from 'mp4box';

const log = Logger.create('AudioExtractor');

const MP4Box = (MP4BoxModule as unknown as { default: typeof MP4BoxModule }).default || MP4BoxModule;

// MP4Box types
interface MP4ArrayBuffer extends ArrayBuffer {
  fileStart: number;
}

interface Sample {
  number: number;
  track_id: number;
  data: Uint8Array;
  size: number;
  cts: number;
  dts: number;
  duration: number;
  is_sync: boolean;
  timescale: number;
}

interface MP4AudioTrack {
  id: number;
  type: string;
  codec: string;
  duration: number;
  timescale: number;
  nb_samples: number;
  bitrate: number;
  audio?: { sample_rate: number; channel_count: number };
}

interface MP4FileType {
  onReady: (info: { tracks: MP4AudioTrack[] }) => void;
  onSamples: (trackId: number, ref: unknown, samples: Sample[]) => void;
  onError: (error: string) => void;
  appendBuffer: (buffer: MP4ArrayBuffer) => number;
  start: () => void;
  flush: () => void;
  setExtractionOptions: (trackId: number, ref: unknown, options: { nbSamples: number }) => void;
}

interface AudioTrackInfo {
  id: number;
  codec: string;
  sampleRate: number;
  channelCount: number;
  bitrate?: number;
  timescale: number;
  duration: number;
  nbSamples: number;
}

interface ExtractedAudio {
  blob: Blob;
  codec: string;
  sampleRate: number;
  channelCount: number;
  duration: number;
}

/**
 * Extract audio track from a video file
 * Returns the audio as a playable blob
 */
export async function extractAudioFromVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<ExtractedAudio | null> {
  return new Promise((resolve) => {
    const mp4boxFile = MP4Box.createFile() as unknown as MP4FileType;
    let audioTrack: AudioTrackInfo | null = null;
    const audioSamples: Uint8Array[] = [];
    let totalSamples = 0;
    let processedSamples = 0;

    mp4boxFile.onReady = (info) => {
      console.log('[AudioExtractor] File info ready');

      // Find audio track
      const audioTrackInfo = info.tracks.find((t) => t.type === 'audio');
      if (!audioTrackInfo) {
        console.warn('[AudioExtractor] No audio track found');
        resolve(null);
        return;
      }

      audioTrack = {
        id: audioTrackInfo.id,
        codec: audioTrackInfo.codec,
        sampleRate: audioTrackInfo.audio?.sample_rate || 48000,
        channelCount: audioTrackInfo.audio?.channel_count || 2,
        bitrate: audioTrackInfo.bitrate,
        timescale: audioTrackInfo.timescale,
        duration: audioTrackInfo.duration,
        nbSamples: audioTrackInfo.nb_samples,
      };

      totalSamples = audioTrack.nbSamples;

      console.log(`[AudioExtractor] Found audio track: ${audioTrack.codec}, ${audioTrack.sampleRate}Hz, ${audioTrack.channelCount}ch, ${totalSamples} samples`);

      // Set up extraction
      mp4boxFile.setExtractionOptions(audioTrack.id, null, {
        nbSamples: 500, // Process in batches
      });

      mp4boxFile.start();
    };

    mp4boxFile.onSamples = (trackId: number, _ref: unknown, samples: Sample[]) => {
      if (!audioTrack || trackId !== audioTrack.id) return;

      // Collect audio samples
      for (const sample of samples) {
        // Copy sample data
        audioSamples.push(new Uint8Array(sample.data));
      }

      processedSamples += samples.length;
      const progress = Math.min(90, (processedSamples / Math.max(1, totalSamples)) * 90);
      onProgress?.(progress);
    };

    mp4boxFile.onError = (error: string) => {
      console.error('[AudioExtractor] MP4Box error:', error);
      resolve(null);
    };

    // When all samples are extracted, create the output file
    const finalize = () => {
      if (!audioTrack || audioSamples.length === 0) {
        console.warn('[AudioExtractor] No audio samples extracted');
        resolve(null);
        return;
      }

      onProgress?.(95);

      try {
        // Create audio file based on codec
        const blob = createAudioBlob(audioTrack, audioSamples);
        const duration = audioTrack.duration / audioTrack.timescale;

        onProgress?.(100);

        if (blob) {
          console.log(`[AudioExtractor] Created ${(blob.size / 1024).toFixed(1)}KB audio file`);
          resolve({
            blob,
            codec: audioTrack.codec,
            sampleRate: audioTrack.sampleRate,
            channelCount: audioTrack.channelCount,
            duration,
          });
        } else {
          resolve(null);
        }
      } catch (e) {
        console.error('[AudioExtractor] Failed to create audio file:', e);
        resolve(null);
      }
    };

    // Read file in chunks
    const reader = new FileReader();
    const chunkSize = 1024 * 1024; // 1MB chunks
    let offset = 0;

    const readNextChunk = () => {
      const slice = file.slice(offset, offset + chunkSize);
      reader.readAsArrayBuffer(slice);
    };

    reader.onload = (e) => {
      const buffer = e.target?.result as ArrayBuffer;
      if (!buffer || buffer.byteLength === 0) {
        // Done reading file
        mp4boxFile.flush();
        finalize();
        return;
      }

      // MP4Box needs fileStart property
      const mp4Buffer = buffer as MP4ArrayBuffer;
      mp4Buffer.fileStart = offset;
      mp4boxFile.appendBuffer(mp4Buffer);

      offset += buffer.byteLength;

      if (offset < file.size) {
        readNextChunk();
      } else {
        mp4boxFile.flush();
        finalize();
      }
    };

    reader.onerror = () => {
      console.error('[AudioExtractor] File read error');
      resolve(null);
    };

    onProgress?.(5);
    readNextChunk();
  });
}

/**
 * Create a playable audio blob from extracted samples
 * For AAC: create ADTS file (raw AAC with headers)
 * For other codecs: create raw blob
 */
function createAudioBlob(
  track: AudioTrackInfo,
  samples: Uint8Array[]
): Blob | null {
  const codecLower = track.codec.toLowerCase();

  if (codecLower.includes('mp4a') || codecLower.includes('aac')) {
    // AAC audio - create ADTS file (raw AAC with frame headers)
    return createADTSFile(track, samples);
  } else if (codecLower.includes('opus')) {
    // Opus - just concatenate (browser might not play this directly)
    console.log('[AudioExtractor] Opus audio - creating raw blob');
    const totalSize = samples.reduce((sum, s) => sum + s.byteLength, 0);
    const combined = new Uint8Array(totalSize);
    let offset = 0;
    for (const sample of samples) {
      combined.set(sample, offset);
      offset += sample.byteLength;
    }
    return new Blob([combined], { type: 'audio/opus' });
  } else {
    // Unknown codec - try raw blob
    console.log(`[AudioExtractor] Unknown codec ${track.codec} - creating raw blob`);
    const totalSize = samples.reduce((sum, s) => sum + s.byteLength, 0);
    const combined = new Uint8Array(totalSize);
    let offset = 0;
    for (const sample of samples) {
      combined.set(sample, offset);
      offset += sample.byteLength;
    }
    return new Blob([combined], { type: 'audio/mp4' });
  }
}

/**
 * Create ADTS AAC file (raw AAC with headers that browsers can play)
 */
function createADTSFile(
  track: AudioTrackInfo,
  samples: Uint8Array[]
): Blob {
  // ADTS header is 7 bytes per frame
  const headerSize = 7;
  const totalSampleSize = samples.reduce((sum, s) => sum + s.byteLength, 0);
  const totalSize = totalSampleSize + (samples.length * headerSize);

  const output = new Uint8Array(totalSize);
  let offset = 0;

  // Sample rate index for ADTS header
  const sampleRateIndex = getSampleRateIndex(track.sampleRate);

  for (const sample of samples) {
    const frameLength = headerSize + sample.byteLength;

    // Write ADTS header (7 bytes)
    // Syncword (12 bits): 0xFFF
    output[offset] = 0xFF;
    output[offset + 1] = 0xF1; // MPEG-4, Layer 0, no CRC

    // Profile (2 bits): AAC-LC = 1 (stored as profile - 1 = 0)
    // Sample rate index (4 bits)
    // Private bit (1 bit): 0
    // Channel config (3 bits, first 1 bit)
    const profile = 1; // AAC-LC
    output[offset + 2] = ((profile - 1) << 6) | (sampleRateIndex << 2) | ((track.channelCount >> 2) & 0x01);

    // Channel config (2 bits)
    // Original/copy (1 bit): 0
    // Home (1 bit): 0
    // Copyright ID (1 bit): 0
    // Copyright start (1 bit): 0
    // Frame length (13 bits, first 2 bits)
    output[offset + 3] = ((track.channelCount & 0x03) << 6) | ((frameLength >> 11) & 0x03);

    // Frame length (next 8 bits)
    output[offset + 4] = (frameLength >> 3) & 0xFF;

    // Frame length (last 3 bits)
    // Buffer fullness (11 bits, first 5 bits): 0x7FF (VBR)
    output[offset + 5] = ((frameLength & 0x07) << 5) | 0x1F;

    // Buffer fullness (last 6 bits): 0x3F
    // Number of frames - 1 (2 bits): 0
    output[offset + 6] = 0xFC;

    offset += headerSize;

    // Write sample data
    output.set(sample, offset);
    offset += sample.byteLength;
  }

  return new Blob([output], { type: 'audio/aac' });
}

/**
 * Get ADTS sample rate index
 */
function getSampleRateIndex(sampleRate: number): number {
  const rates = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];
  const index = rates.indexOf(sampleRate);
  if (index >= 0) return index;

  // Find closest rate
  let closest = 0;
  let minDiff = Math.abs(sampleRate - rates[0]);
  for (let i = 1; i < rates.length; i++) {
    const diff = Math.abs(sampleRate - rates[i]);
    if (diff < minDiff) {
      minDiff = diff;
      closest = i;
    }
  }
  return closest;
}
