// Transcription Web Worker
// Runs Whisper transcription off the main thread with streaming results

import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js for worker
env.allowLocalModels = false;
env.useBrowserCache = true;

interface TranscriptChunk {
  text: string;
  timestamp: [number, number | null];
}

interface WorkerMessage {
  type: 'start';
  audioData: Float32Array;
  audioDuration: number;
  language: string;
}

interface ProgressMessage {
  type: 'progress';
  stage: 'loading' | 'transcribing';
  progress: number;
  message: string;
}

interface PartialResultMessage {
  type: 'partial';
  words: Array<{
    id: string;
    text: string;
    start: number;
    end: number;
    confidence: number;
    speaker: string;
  }>;
  processedDuration: number;
  totalDuration: number;
}

interface CompleteMessage {
  type: 'complete';
  words: Array<{
    id: string;
    text: string;
    start: number;
    end: number;
    confidence: number;
    speaker: string;
  }>;
}

interface ErrorMessage {
  type: 'error';
  error: string;
}

type OutgoingMessage = ProgressMessage | PartialResultMessage | CompleteMessage | ErrorMessage;

// Cache for loaded models
let transcriber: any = null;
let loadedModel: string | null = null;

// Get model name based on language
function getModelName(language: string): string {
  // Use English-only model for English (faster), multilingual for others
  if (language === 'en') {
    return 'Xenova/whisper-tiny.en';
  }
  // Use multilingual model for all other languages
  return 'Xenova/whisper-tiny';
}

// Load model (reload if language changed)
async function loadModel(
  language: string,
  onProgress: (progress: number, message: string) => void
) {
  const modelName = getModelName(language);

  // Return cached model if same language
  if (transcriber && loadedModel === modelName) {
    return transcriber;
  }

  // Clear old model
  transcriber = null;
  loadedModel = null;

  const langName = language === 'en' ? 'English' : 'multilingual';
  onProgress(0, `Lade Whisper Model (${langName})...`);

  transcriber = await pipeline(
    'automatic-speech-recognition',
    modelName,
    {
      progress_callback: (data: any) => {
        if (data.status === 'progress' && data.progress) {
          onProgress(data.progress, `Model laden: ${Math.round(data.progress)}%`);
        }
      },
      revision: 'main',
    }
  );

  loadedModel = modelName;
  onProgress(100, 'Model geladen');
  return transcriber;
}

// Process audio and stream results
async function transcribeAudio(
  audioData: Float32Array,
  audioDuration: number,
  language: string,
  postMessage: (msg: OutgoingMessage) => void
) {
  // Load model first
  const model = await loadModel(language, (progress, message) => {
    postMessage({
      type: 'progress',
      stage: 'loading',
      progress: progress * 0.3, // Model loading is 0-30%
      message,
    });
  });

  postMessage({
    type: 'progress',
    stage: 'transcribing',
    progress: 30,
    message: 'Starte Transkription...',
  });

  // For long audio, process in segments to get incremental results
  const SEGMENT_DURATION = 30; // seconds
  const SAMPLE_RATE = 16000;
  const segmentSamples = SEGMENT_DURATION * SAMPLE_RATE;
  const totalSamples = audioData.length;
  const numSegments = Math.ceil(totalSamples / segmentSamples);

  const allWords: Array<{
    id: string;
    text: string;
    start: number;
    end: number;
    confidence: number;
    speaker: string;
  }> = [];

  let wordIndex = 0;

  for (let segmentIdx = 0; segmentIdx < numSegments; segmentIdx++) {
    const startSample = segmentIdx * segmentSamples;
    const endSample = Math.min(startSample + segmentSamples, totalSamples);
    const segmentData = audioData.slice(startSample, endSample);
    const segmentStartTime = startSample / SAMPLE_RATE;

    // Update progress
    const transcriptionProgress = 30 + ((segmentIdx / numSegments) * 70);
    postMessage({
      type: 'progress',
      stage: 'transcribing',
      progress: transcriptionProgress,
      message: `Segment ${segmentIdx + 1}/${numSegments}...`,
    });

    try {
      // Transcribe this segment with language setting
      const result = await model(segmentData, {
        return_timestamps: 'word',
        chunk_length_s: 30,
        stride_length_s: 5,
        language: language,
        task: 'transcribe',
      });

      // Process chunks from this segment
      const chunks: TranscriptChunk[] = result.chunks || [];

      for (const chunk of chunks) {
        const chunkText = chunk.text?.trim();
        if (!chunkText) continue;

        const chunkStart = (chunk.timestamp[0] ?? 0) + segmentStartTime;
        const chunkEnd = (chunk.timestamp[1] ?? chunkStart + 0.5) + segmentStartTime;

        // Split into words if needed
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
        const wordDuration = segmentDuration / segmentWords.length;

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

      // Send partial results after each segment
      postMessage({
        type: 'partial',
        words: [...allWords],
        processedDuration: Math.min((segmentIdx + 1) * SEGMENT_DURATION, audioDuration),
        totalDuration: audioDuration,
      });

    } catch (err) {
      console.error('[Worker] Segment error:', err);
      // Continue with next segment
    }
  }

  // Send final complete message
  postMessage({
    type: 'complete',
    words: allWords,
  });
}

// Handle messages from main thread
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type, audioData, audioDuration, language } = event.data;

  if (type === 'start') {
    try {
      await transcribeAudio(audioData, audioDuration, language || 'de', (msg) => self.postMessage(msg));
    } catch (error) {
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      } as ErrorMessage);
    }
  }
};

export {};
