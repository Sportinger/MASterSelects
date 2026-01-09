// Clip Transcriber Service
// Handles transcription of individual clips using Whisper in a Web Worker

import { useTimelineStore } from '../stores/timeline';
import { triggerTimelineSave } from '../stores/mediaStore';
import type { TranscriptWord, TranscriptStatus } from '../types';

// Worker instance
let worker: Worker | null = null;
let isTranscribing = false;
let currentClipId: string | null = null;

/**
 * Get or create the transcription worker
 */
function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(
      new URL('../workers/transcriptionWorker.ts', import.meta.url),
      { type: 'module' }
    );
  }
  return worker;
}

/**
 * Extract audio from a clip's file and transcribe it using Web Worker
 */
export async function transcribeClip(clipId: string, language: string = 'de'): Promise<void> {
  if (isTranscribing) {
    console.warn('[Transcribe] Already transcribing');
    return;
  }

  const store = useTimelineStore.getState();
  const clip = store.clips.find(c => c.id === clipId);

  if (!clip || !clip.file) {
    console.warn('[Transcribe] Clip not found or has no file:', clipId);
    return;
  }

  // Check if file has audio
  const hasAudio = clip.file.type.startsWith('video/') || clip.file.type.startsWith('audio/');
  if (!hasAudio) {
    console.warn('[Transcribe] File does not contain audio');
    return;
  }

  isTranscribing = true;
  currentClipId = clipId;

  console.log(`[Transcript] Starting transcription for ${clip.name}...`);

  // Update status to transcribing
  updateClipTranscript(clipId, {
    status: 'transcribing',
    progress: 0,
    message: 'Extracting audio...',
  });

  try {
    // Extract audio on main thread (AudioContext not available in workers)
    // Only extract the portion between inPoint and outPoint (the trimmed clip region)
    const inPoint = clip.inPoint || 0;
    const outPoint = clip.outPoint || clip.duration;
    const clipDuration = outPoint - inPoint;

    console.log(`[Transcribe] Extracting audio from ${inPoint.toFixed(1)}s to ${outPoint.toFixed(1)}s (${clipDuration.toFixed(1)}s)`);

    const audioBuffer = await extractAudioBuffer(clip.file, inPoint, outPoint);
    const audioData = await resampleAudio(audioBuffer, 16000);
    const audioDuration = audioBuffer.duration;

    console.log('[Transcribe] Audio extracted:', audioDuration.toFixed(1) + 's');

    updateClipTranscript(clipId, {
      progress: 5,
      message: 'Starting worker...',
    });

    // Run transcription in worker, passing inPoint for timestamp offset
    // Whisper returns timestamps starting at 0 (beginning of extracted audio)
    // The worker handler will offset them so they match the original source file
    const words = await runWorkerTranscription(clipId, audioData, language, audioDuration, inPoint);

    // Complete
    updateClipTranscript(clipId, {
      status: 'ready',
      progress: 100,
      words,
      message: undefined,
    });
    triggerTimelineSave();
    console.log(`[Transcript] Complete: ${words.length} words for ${clip.name}`);

  } catch (error) {
    console.error('[Transcribe] Failed:', error);
    updateClipTranscript(clipId, {
      status: 'error',
      progress: 0,
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  } finally {
    isTranscribing = false;
    currentClipId = null;
  }
}

/**
 * Run transcription in Web Worker
 * @param inPointOffset - Offset to add to word timestamps (for trimmed clips)
 */
function runWorkerTranscription(
  clipId: string,
  audioData: Float32Array,
  language: string,
  audioDuration: number,
  inPointOffset: number = 0
): Promise<TranscriptWord[]> {
  return new Promise((resolve, reject) => {
    const w = getWorker();

    // Helper to offset word timestamps
    const offsetWords = (words: TranscriptWord[]): TranscriptWord[] =>
      words.map(word => ({
        ...word,
        start: word.start + inPointOffset,
        end: word.end + inPointOffset,
      }));

    const handleMessage = (event: MessageEvent) => {
      const { type, progress, message, words, error } = event.data;

      switch (type) {
        case 'progress':
          updateClipTranscript(clipId, { progress, message });
          break;

        case 'words':
          // Offset partial results too
          updateClipTranscript(clipId, {
            words: offsetWords(words),
            message: `Transcribed ${words.length} words`,
          });
          break;

        case 'complete':
          w.removeEventListener('message', handleMessage);
          w.removeEventListener('error', handleError);
          // Offset final words before returning
          resolve(offsetWords(words));
          break;

        case 'error':
          w.removeEventListener('message', handleMessage);
          w.removeEventListener('error', handleError);
          reject(new Error(error));
          break;
      }
    };

    const handleError = (error: ErrorEvent) => {
      w.removeEventListener('message', handleMessage);
      w.removeEventListener('error', handleError);
      reject(new Error(error.message || 'Worker error'));
    };

    w.addEventListener('message', handleMessage);
    w.addEventListener('error', handleError);

    // Send audio data to worker (transferable for performance)
    w.postMessage(
      { type: 'transcribe', audioData, language, audioDuration },
      [audioData.buffer]
    );
  });
}

/**
 * Update clip transcript data in the timeline store
 */
function updateClipTranscript(
  clipId: string,
  data: {
    status?: TranscriptStatus;
    progress?: number;
    words?: TranscriptWord[];
    message?: string;
  }
): void {
  const store = useTimelineStore.getState();
  const clips = store.clips.map(clip => {
    if (clip.id !== clipId) return clip;

    return {
      ...clip,
      transcriptStatus: data.status ?? clip.transcriptStatus,
      transcriptProgress: data.progress ?? clip.transcriptProgress,
      transcript: data.words ?? clip.transcript,
      transcriptMessage: data.message,
    };
  });

  useTimelineStore.setState({ clips });
}

/**
 * Extract audio buffer from a media file, optionally slicing to a time range
 * @param file - The media file to extract audio from
 * @param startTime - Start time in seconds (optional, defaults to 0)
 * @param endTime - End time in seconds (optional, defaults to full duration)
 */
async function extractAudioBuffer(
  file: File,
  startTime?: number,
  endTime?: number
): Promise<AudioBuffer> {
  const audioContext = new AudioContext();
  const arrayBuffer = await file.arrayBuffer();
  const fullBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // If no time range specified, return full buffer
  if (startTime === undefined && endTime === undefined) {
    audioContext.close();
    return fullBuffer;
  }

  // Calculate sample range
  const sampleRate = fullBuffer.sampleRate;
  const startSample = Math.floor((startTime || 0) * sampleRate);
  const endSample = Math.min(
    Math.ceil((endTime || fullBuffer.duration) * sampleRate),
    fullBuffer.length
  );
  const sliceLength = endSample - startSample;

  // Create new buffer with sliced audio
  const slicedBuffer = audioContext.createBuffer(
    fullBuffer.numberOfChannels,
    sliceLength,
    sampleRate
  );

  // Copy each channel's data
  for (let channel = 0; channel < fullBuffer.numberOfChannels; channel++) {
    const sourceData = fullBuffer.getChannelData(channel);
    const destData = slicedBuffer.getChannelData(channel);
    for (let i = 0; i < sliceLength; i++) {
      destData[i] = sourceData[startSample + i];
    }
  }

  audioContext.close();
  return slicedBuffer;
}

/**
 * Resample audio to target sample rate (e.g., 16kHz for Whisper)
 */
async function resampleAudio(
  audioBuffer: AudioBuffer,
  targetSampleRate: number
): Promise<Float32Array> {
  const channelData = audioBuffer.getChannelData(0); // Mono
  const originalSampleRate = audioBuffer.sampleRate;

  if (originalSampleRate === targetSampleRate) {
    return channelData;
  }

  // Simple linear interpolation resampling
  const ratio = originalSampleRate / targetSampleRate;
  const newLength = Math.floor(channelData.length / ratio);
  const resampled = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const srcIndex = i * ratio;
    const srcIndexFloor = Math.floor(srcIndex);
    const srcIndexCeil = Math.min(srcIndexFloor + 1, channelData.length - 1);
    const t = srcIndex - srcIndexFloor;
    resampled[i] = channelData[srcIndexFloor] * (1 - t) + channelData[srcIndexCeil] * t;
  }

  return resampled;
}

/**
 * Clear transcript from a clip
 */
export function clearClipTranscript(clipId: string): void {
  updateClipTranscript(clipId, {
    status: 'none',
    progress: 0,
    words: undefined,
    message: undefined,
  });
  triggerTimelineSave();
}

/**
 * Cancel ongoing transcription
 */
export function cancelTranscription(): void {
  if (worker && isTranscribing) {
    worker.terminate();
    worker = null;
    if (currentClipId) {
      updateClipTranscript(currentClipId, {
        status: 'none',
        progress: 0,
        message: undefined,
      });
    }
    isTranscribing = false;
    currentClipId = null;
  }
}
