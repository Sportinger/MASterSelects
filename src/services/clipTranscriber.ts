// Clip Transcriber Service
// Handles transcription of individual clips using Whisper in a Web Worker

import { useTimelineStore } from '../stores/timeline';
import { triggerTimelineSave } from '../stores/mediaStore';
import type { TranscriptWord, TranscriptStatus } from '../types';

// Worker instance (reused across transcriptions)
let worker: Worker | null = null;
let currentClipId: string | null = null;

// Message types from worker
interface ProgressMessage {
  type: 'progress';
  stage: 'loading' | 'transcribing';
  progress: number;
  message: string;
}

interface PartialResultMessage {
  type: 'partial';
  words: TranscriptWord[];
  processedDuration: number;
  totalDuration: number;
}

interface CompleteMessage {
  type: 'complete';
  words: TranscriptWord[];
}

interface ErrorMessage {
  type: 'error';
  error: string;
}

type WorkerMessage = ProgressMessage | PartialResultMessage | CompleteMessage | ErrorMessage;

/**
 * Get or create worker instance
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
export async function transcribeClip(clipId: string): Promise<void> {
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

  // Set current clip ID for worker messages
  currentClipId = clipId;

  // Update status to transcribing
  updateClipTranscript(clipId, {
    status: 'transcribing',
    progress: 0,
    message: 'Extracting audio...',
  });

  try {
    // Extract audio on main thread (needs AudioContext)
    const audioBuffer = await extractAudioBuffer(clip.file);
    const audioData = await resampleAudio(audioBuffer, 16000);
    const audioDuration = audioBuffer.duration;

    console.log('[Transcribe] Audio extracted:', audioDuration.toFixed(1) + 's');

    updateClipTranscript(clipId, {
      progress: 5,
      message: 'Starting transcription...',
    });

    // Send to worker for transcription
    const transcriptWorker = getWorker();

    // Set up message handler
    const handleMessage = (event: MessageEvent<WorkerMessage>) => {
      const msg = event.data;

      // Only process messages for current clip
      if (currentClipId !== clipId) return;

      switch (msg.type) {
        case 'progress':
          updateClipTranscript(clipId, {
            progress: msg.progress,
            message: msg.message,
          });
          break;

        case 'partial':
          // Update with partial results - show words transcribed so far
          updateClipTranscript(clipId, {
            words: msg.words,
            progress: Math.round((msg.processedDuration / msg.totalDuration) * 100),
            message: `Transcribed ${msg.words.length} words (${Math.round(msg.processedDuration)}s / ${Math.round(msg.totalDuration)}s)`,
          });
          break;

        case 'complete':
          updateClipTranscript(clipId, {
            status: 'ready',
            progress: 100,
            words: msg.words,
            message: undefined,
          });
          triggerTimelineSave();
          console.log('[Transcribe] Done:', msg.words.length, 'words');
          transcriptWorker.removeEventListener('message', handleMessage);
          break;

        case 'error':
          console.error('[Transcribe] Worker error:', msg.error);
          updateClipTranscript(clipId, {
            status: 'error',
            progress: 0,
            message: msg.error,
          });
          transcriptWorker.removeEventListener('message', handleMessage);
          break;
      }
    };

    transcriptWorker.addEventListener('message', handleMessage);

    // Start transcription in worker
    transcriptWorker.postMessage({
      type: 'start',
      audioData,
      audioDuration,
    });

  } catch (error) {
    console.error('[Transcribe] Failed:', error);
    updateClipTranscript(clipId, {
      status: 'error',
      progress: 0,
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
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
 * Extract audio buffer from a media file
 */
async function extractAudioBuffer(file: File): Promise<AudioBuffer> {
  const audioContext = new AudioContext();
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  audioContext.close();
  return audioBuffer;
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
}

/**
 * Cancel ongoing transcription
 */
export function cancelTranscription(): void {
  if (worker) {
    worker.terminate();
    worker = null;
  }
  if (currentClipId) {
    updateClipTranscript(currentClipId, {
      status: 'none',
      progress: 0,
      message: undefined,
    });
    currentClipId = null;
  }
}
