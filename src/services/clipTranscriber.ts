// Clip Transcriber Service
// Handles transcription of individual clips using Whisper (local) or cloud APIs

import { useTimelineStore } from '../stores/timeline';
import { triggerTimelineSave } from '../stores/mediaStore';
import { useSettingsStore } from '../stores/settingsStore';
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
 * Extract audio from a clip's file and transcribe it
 * Uses the configured provider (local Whisper, OpenAI, AssemblyAI, or Deepgram)
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

  // Get transcription provider settings
  const { transcriptionProvider, apiKeys } = useSettingsStore.getState();
  const apiKey = transcriptionProvider !== 'local' ? apiKeys[transcriptionProvider] : null;

  // Validate API key if using cloud provider
  if (transcriptionProvider !== 'local' && !apiKey) {
    console.error('[Transcribe] No API key configured for', transcriptionProvider);
    updateClipTranscript(clipId, {
      status: 'error',
      progress: 0,
      message: `No API key configured for ${transcriptionProvider}. Go to Settings to add one.`,
    });
    return;
  }

  isTranscribing = true;
  currentClipId = clipId;

  const providerName = transcriptionProvider === 'local' ? 'Local Whisper' : transcriptionProvider.toUpperCase();
  console.log(`[Transcript] Starting transcription for ${clip.name} using ${providerName}...`);

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
    const audioDuration = audioBuffer.duration;

    console.log('[Transcribe] Audio extracted:', audioDuration.toFixed(1) + 's');

    let words: TranscriptWord[];

    if (transcriptionProvider === 'local') {
      // Local Whisper via Web Worker
      const audioData = await resampleAudio(audioBuffer, 16000);
      updateClipTranscript(clipId, {
        progress: 5,
        message: 'Starting local transcription...',
      });
      words = await runWorkerTranscription(clipId, audioData, language, audioDuration, inPoint);
    } else {
      // Cloud API transcription
      updateClipTranscript(clipId, {
        progress: 10,
        message: `Uploading to ${providerName}...`,
      });

      // Convert audio buffer to appropriate format for API
      const audioBlob = await audioBufferToWav(audioBuffer);

      switch (transcriptionProvider) {
        case 'openai':
          words = await transcribeWithOpenAI(clipId, audioBlob, language, apiKey!, inPoint);
          break;
        case 'assemblyai':
          words = await transcribeWithAssemblyAI(clipId, audioBlob, language, apiKey!, inPoint);
          break;
        case 'deepgram':
          words = await transcribeWithDeepgram(clipId, audioBlob, language, apiKey!, inPoint);
          break;
        default:
          throw new Error(`Unknown provider: ${transcriptionProvider}`);
      }
    }

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

// ============================================================================
// Cloud API Transcription Functions
// ============================================================================

/**
 * Convert AudioBuffer to WAV Blob for API upload
 */
async function audioBufferToWav(audioBuffer: AudioBuffer): Promise<Blob> {
  const numChannels = 1; // Mono
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  const channelData = audioBuffer.getChannelData(0);
  const samples = new Int16Array(channelData.length);

  // Convert float samples to 16-bit PCM
  for (let i = 0; i < channelData.length; i++) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    samples[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // WAV header
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
  view.setUint16(32, numChannels * (bitDepth / 8), true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  // Write PCM samples
  const dataView = new Int16Array(buffer, 44);
  dataView.set(samples);

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Transcribe using OpenAI Whisper API
 */
async function transcribeWithOpenAI(
  clipId: string,
  audioBlob: Blob,
  language: string,
  apiKey: string,
  inPointOffset: number
): Promise<TranscriptWord[]> {
  updateClipTranscript(clipId, {
    progress: 20,
    message: 'Sending to OpenAI...',
  });

  const formData = new FormData();
  formData.append('file', audioBlob, 'audio.wav');
  formData.append('model', 'whisper-1');
  formData.append('language', language);
  formData.append('response_format', 'verbose_json');
  formData.append('timestamp_granularities[]', 'word');

  const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: { message: response.statusText } }));
    throw new Error(`OpenAI API error: ${error.error?.message || response.statusText}`);
  }

  updateClipTranscript(clipId, {
    progress: 80,
    message: 'Processing response...',
  });

  const result = await response.json();

  // Convert OpenAI response to TranscriptWord[]
  const words: TranscriptWord[] = (result.words || []).map((word: any, index: number) => ({
    id: `word-${index}`,
    text: word.word,
    start: (word.start || 0) + inPointOffset,
    end: (word.end || word.start + 0.1) + inPointOffset,
    confidence: 1,
    speaker: 'Speaker 1',
  }));

  return words;
}

/**
 * Transcribe using AssemblyAI API
 */
async function transcribeWithAssemblyAI(
  clipId: string,
  audioBlob: Blob,
  language: string,
  apiKey: string,
  inPointOffset: number
): Promise<TranscriptWord[]> {
  // Step 1: Upload audio
  updateClipTranscript(clipId, {
    progress: 15,
    message: 'Uploading to AssemblyAI...',
  });

  const uploadResponse = await fetch('https://api.assemblyai.com/v2/upload', {
    method: 'POST',
    headers: {
      Authorization: apiKey,
      'Content-Type': 'application/octet-stream',
    },
    body: audioBlob,
  });

  if (!uploadResponse.ok) {
    throw new Error(`AssemblyAI upload failed: ${uploadResponse.statusText}`);
  }

  const { upload_url } = await uploadResponse.json();

  // Step 2: Start transcription
  updateClipTranscript(clipId, {
    progress: 30,
    message: 'Starting transcription...',
  });

  // Map common language codes to AssemblyAI format
  const languageMap: Record<string, string> = {
    de: 'de',
    en: 'en',
    es: 'es',
    fr: 'fr',
    it: 'it',
    pt: 'pt',
    nl: 'nl',
    pl: 'pl',
    ru: 'ru',
    ja: 'ja',
    zh: 'zh',
    ko: 'ko',
  };

  const transcriptResponse = await fetch('https://api.assemblyai.com/v2/transcript', {
    method: 'POST',
    headers: {
      Authorization: apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      audio_url: upload_url,
      language_code: languageMap[language] || language,
    }),
  });

  if (!transcriptResponse.ok) {
    throw new Error(`AssemblyAI transcription request failed: ${transcriptResponse.statusText}`);
  }

  const { id: transcriptId } = await transcriptResponse.json();

  // Step 3: Poll for completion
  let result: any;
  let attempts = 0;
  const maxAttempts = 120; // 2 minutes max

  while (attempts < maxAttempts) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    attempts++;

    const pollResponse = await fetch(`https://api.assemblyai.com/v2/transcript/${transcriptId}`, {
      headers: { Authorization: apiKey },
    });

    result = await pollResponse.json();

    if (result.status === 'completed') {
      break;
    } else if (result.status === 'error') {
      throw new Error(`AssemblyAI error: ${result.error}`);
    }

    // Update progress (30-80% during polling)
    const progress = 30 + Math.min(50, attempts * 0.5);
    updateClipTranscript(clipId, {
      progress,
      message: `Transcribing... (${result.status})`,
    });
  }

  if (!result || result.status !== 'completed') {
    throw new Error('AssemblyAI transcription timed out');
  }

  updateClipTranscript(clipId, {
    progress: 90,
    message: 'Processing response...',
  });

  // Convert AssemblyAI response to TranscriptWord[]
  const words: TranscriptWord[] = (result.words || []).map((word: any, index: number) => ({
    id: `word-${index}`,
    text: word.text,
    start: (word.start / 1000) + inPointOffset, // AssemblyAI uses milliseconds
    end: (word.end / 1000) + inPointOffset,
    confidence: word.confidence || 1,
    speaker: word.speaker || 'Speaker 1',
  }));

  return words;
}

/**
 * Transcribe using Deepgram API
 */
async function transcribeWithDeepgram(
  clipId: string,
  audioBlob: Blob,
  language: string,
  apiKey: string,
  inPointOffset: number
): Promise<TranscriptWord[]> {
  updateClipTranscript(clipId, {
    progress: 20,
    message: 'Sending to Deepgram...',
  });

  // Build query params
  const params = new URLSearchParams({
    model: 'nova-2',
    language: language,
    punctuate: 'true',
    utterances: 'false',
  });

  const response = await fetch(`https://api.deepgram.com/v1/listen?${params}`, {
    method: 'POST',
    headers: {
      Authorization: `Token ${apiKey}`,
      'Content-Type': 'audio/wav',
    },
    body: audioBlob,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(`Deepgram API error: ${error.error || error.err_msg || response.statusText}`);
  }

  updateClipTranscript(clipId, {
    progress: 80,
    message: 'Processing response...',
  });

  const result = await response.json();
  const channel = result.results?.channels?.[0];
  const alternative = channel?.alternatives?.[0];

  if (!alternative) {
    throw new Error('No transcription results from Deepgram');
  }

  // Convert Deepgram response to TranscriptWord[]
  const words: TranscriptWord[] = (alternative.words || []).map((word: any, index: number) => ({
    id: `word-${index}`,
    text: word.word,
    start: (word.start || 0) + inPointOffset,
    end: (word.end || word.start + 0.1) + inPointOffset,
    confidence: word.confidence || 1,
    speaker: word.speaker !== undefined ? `Speaker ${word.speaker + 1}` : 'Speaker 1',
  }));

  return words;
}
