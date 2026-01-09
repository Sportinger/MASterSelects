// Clip Transcriber Service
// Handles transcription of individual clips using Whisper

import { useTimelineStore } from '../stores/timeline';
import { triggerTimelineSave } from '../stores/mediaStore';
import type { TranscriptWord, TranscriptStatus } from '../types';

// Transcriber instance (cached)
let transcriber: any = null;
let loadedModel: string | null = null;
let isTranscribing = false;
let shouldCancel = false;
let transformersLoaded = false;
let pipelineFn: any = null;

interface TranscriptChunk {
  text: string;
  timestamp: [number, number | null];
}

/**
 * Get model name based on language
 */
function getModelName(language: string): string {
  if (language === 'en') {
    return 'onnx-community/whisper-tiny.en';
  }
  return 'onnx-community/whisper-tiny';
}

/**
 * Load transformers.js library dynamically
 */
async function loadTransformers(): Promise<void> {
  if (transformersLoaded) return;

  const transformers = await import('@huggingface/transformers');

  // Configure environment
  transformers.env.allowLocalModels = false;
  transformers.env.useBrowserCache = true;

  pipelineFn = transformers.pipeline;
  transformersLoaded = true;
}

/**
 * Load Whisper model
 */
async function loadModel(
  language: string,
  onProgress: (progress: number, message: string) => void
): Promise<any> {
  const modelName = getModelName(language);

  // Return cached model if same
  if (transcriber && loadedModel === modelName) {
    return transcriber;
  }

  // Clear old model
  transcriber = null;
  loadedModel = null;

  const langName = language === 'en' ? 'English' : 'multilingual';
  onProgress(0, `Loading Whisper model (${langName})...`);

  try {
    // Load transformers.js dynamically
    await loadTransformers();

    transcriber = await pipelineFn(
      'automatic-speech-recognition',
      modelName,
      {
        progress_callback: (data: any) => {
          if (data.status === 'progress' && data.progress) {
            onProgress(data.progress, `Model loading: ${Math.round(data.progress)}%`);
          }
        },
        revision: 'main',
      }
    );

    loadedModel = modelName;
    onProgress(100, 'Model loaded');
    return transcriber;
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    if (errorMsg.includes('<!doctype') || errorMsg.includes('Unexpected token')) {
      throw new Error('Model download failed - network error. Please refresh and try again.');
    }
    throw new Error(`Failed to load Whisper model: ${errorMsg}`);
  }
}

/**
 * Extract audio from a clip's file and transcribe it
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
  shouldCancel = false;

  // Update status to transcribing
  updateClipTranscript(clipId, {
    status: 'transcribing',
    progress: 0,
    message: 'Extracting audio...',
  });

  try {
    // Extract audio
    const audioBuffer = await extractAudioBuffer(clip.file);
    const audioData = await resampleAudio(audioBuffer, 16000);
    const audioDuration = audioBuffer.duration;

    console.log('[Transcribe] Audio extracted:', audioDuration.toFixed(1) + 's');

    if (shouldCancel) {
      updateClipTranscript(clipId, { status: 'none', progress: 0, message: undefined });
      isTranscribing = false;
      return;
    }

    // Load model
    const model = await loadModel(language, (progress, message) => {
      updateClipTranscript(clipId, {
        progress: progress * 0.3, // Model loading is 0-30%
        message,
      });
    });

    if (shouldCancel) {
      updateClipTranscript(clipId, { status: 'none', progress: 0, message: undefined });
      isTranscribing = false;
      return;
    }

    updateClipTranscript(clipId, {
      progress: 30,
      message: 'Starting transcription...',
    });

    // Process in segments
    const SEGMENT_DURATION = 30; // seconds
    const SAMPLE_RATE = 16000;
    const segmentSamples = SEGMENT_DURATION * SAMPLE_RATE;
    const totalSamples = audioData.length;
    const numSegments = Math.ceil(totalSamples / segmentSamples);

    const allWords: TranscriptWord[] = [];
    let wordIndex = 0;

    for (let segmentIdx = 0; segmentIdx < numSegments; segmentIdx++) {
      if (shouldCancel) {
        updateClipTranscript(clipId, { status: 'none', progress: 0, message: undefined });
        isTranscribing = false;
        return;
      }

      const startSample = segmentIdx * segmentSamples;
      const endSample = Math.min(startSample + segmentSamples, totalSamples);
      const segmentData = audioData.slice(startSample, endSample);
      const segmentStartTime = startSample / SAMPLE_RATE;

      // Update progress
      const transcriptionProgress = 30 + ((segmentIdx / numSegments) * 70);
      updateClipTranscript(clipId, {
        progress: transcriptionProgress,
        message: `Segment ${segmentIdx + 1}/${numSegments}...`,
      });

      // Yield to event loop
      await new Promise(resolve => setTimeout(resolve, 0));

      try {
        // English-only models don't support language/task parameters
        // Use chunk-level timestamps (word-level requires cross-attention outputs)
        const isEnglishOnly = language === 'en';
        const result = await model(segmentData, {
          return_timestamps: true,
          chunk_length_s: 30,
          stride_length_s: 5,
          ...(isEnglishOnly ? {} : { language, task: 'transcribe' }),
        });

        // Process chunks
        const chunks: TranscriptChunk[] = result.chunks || [];

        for (const chunk of chunks) {
          const chunkText = chunk.text?.trim();
          if (!chunkText) continue;

          const chunkStart = (chunk.timestamp[0] ?? 0) + segmentStartTime;
          const chunkEnd = (chunk.timestamp[1] ?? chunkStart + 0.5) + segmentStartTime;

          const chunkWords = chunkText.split(/\s+/).filter((w: string) => w.length > 0);

          if (chunkWords.length === 1) {
            allWords.push({
              id: `word-${wordIndex++}`,
              text: chunkText,
              start: chunkStart,
              end: chunkEnd,
              confidence: 1,
              speaker: 'Speaker 1',
            });
          } else {
            const duration = chunkEnd - chunkStart;
            const wordDuration = duration / chunkWords.length;

            for (let i = 0; i < chunkWords.length; i++) {
              allWords.push({
                id: `word-${wordIndex++}`,
                text: chunkWords[i],
                start: chunkStart + (i * wordDuration),
                end: chunkStart + ((i + 1) * wordDuration),
                confidence: 1,
                speaker: 'Speaker 1',
              });
            }
          }
        }

        // Fallback: if no chunks but have text
        if (chunks.length === 0 && result.text) {
          const segmentText = result.text.trim();
          const segmentWords = segmentText.split(/\s+/).filter((w: string) => w.length > 0);
          const segmentDuration = (endSample - startSample) / SAMPLE_RATE;
          const wordDuration = segmentDuration / Math.max(1, segmentWords.length);

          for (let i = 0; i < segmentWords.length; i++) {
            allWords.push({
              id: `word-${wordIndex++}`,
              text: segmentWords[i],
              start: segmentStartTime + (i * wordDuration),
              end: segmentStartTime + ((i + 1) * wordDuration),
              confidence: 1,
              speaker: 'Speaker 1',
            });
          }
        }

        // Update with partial results
        updateClipTranscript(clipId, {
          words: [...allWords],
          progress: Math.round((segmentIdx + 1) / numSegments * 100),
          message: `Transcribed ${allWords.length} words`,
        });

      } catch (err) {
        console.error('[Transcribe] Segment error:', err);
        // Continue with next segment
      }
    }

    // Complete
    updateClipTranscript(clipId, {
      status: 'ready',
      progress: 100,
      words: allWords,
      message: undefined,
    });
    triggerTimelineSave();
    console.log('[Transcribe] Done:', allWords.length, 'words');

  } catch (error) {
    console.error('[Transcribe] Failed:', error);
    updateClipTranscript(clipId, {
      status: 'error',
      progress: 0,
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  } finally {
    isTranscribing = false;
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
  triggerTimelineSave();
}

/**
 * Cancel ongoing transcription
 */
export function cancelTranscription(): void {
  shouldCancel = true;
}
