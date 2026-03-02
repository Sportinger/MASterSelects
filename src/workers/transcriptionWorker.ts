// Transcription Web Worker
// Runs Whisper model in background thread to avoid UI blocking

import { pipeline, env } from '@huggingface/transformers';

// Configure environment
env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriber: any = null;
let loadedModel: string | null = null;

interface TranscriptChunk {
  text: string;
  timestamp: [number, number | null];
}

interface TranscriptWord {
  id: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
  speaker: string;
}

function getModelName(language: string): string {
  if (language === 'en') {
    return 'Xenova/whisper-tiny.en';
  }
  return 'onnx-community/whisper-tiny';
}

function supportsWordTimestamps(language: string): boolean {
  return language === 'en';
}

/**
 * Detect repetitive/hallucinated text
 */
function isRepetitiveText(text: string): boolean {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (words.length < 4) return false;

  // Check for repeated phrases (3+ words repeating)
  for (let phraseLen = 2; phraseLen <= 4; phraseLen++) {
    const phrases: Record<string, number> = {};
    for (let i = 0; i <= words.length - phraseLen; i++) {
      const phrase = words.slice(i, i + phraseLen).join(' ');
      phrases[phrase] = (phrases[phrase] || 0) + 1;
    }

    // If any phrase repeats more than 3 times, it's repetitive
    for (const count of Object.values(phrases)) {
      if (count > 3) return true;
    }
  }

  // Check if same word repeats more than 50% of the text
  const wordCounts: Record<string, number> = {};
  for (const word of words) {
    wordCounts[word] = (wordCounts[word] || 0) + 1;
  }

  for (const count of Object.values(wordCounts)) {
    if (count > words.length * 0.5 && count > 4) return true;
  }

  return false;
}

async function loadModel(
  language: string,
  onProgress: (progress: number, message: string) => void
): Promise<any> {
  const modelName = getModelName(language);

  if (transcriber && loadedModel === modelName) {
    return transcriber;
  }

  transcriber = null;
  loadedModel = null;

  const langName = language === 'en' ? 'English' : 'multilingual';
  onProgress(0, `Loading Whisper model (${langName})...`);

  try {
    transcriber = await pipeline(
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

async function transcribe(
  audioData: Float32Array,
  language: string,
  _audioDuration: number,
  onProgress: (progress: number, message: string) => void,
  onWords: (words: TranscriptWord[]) => void
): Promise<TranscriptWord[]> {
  const model = await loadModel(language, (progress, message) => {
    onProgress(progress * 0.3, message);
  });

  onProgress(30, 'Starting transcription...');

  const SEGMENT_DURATION = 30;
  const SAMPLE_RATE = 16000;
  const segmentSamples = SEGMENT_DURATION * SAMPLE_RATE;
  const totalSamples = audioData.length;
  const numSegments = Math.ceil(totalSamples / segmentSamples);

  const allWords: TranscriptWord[] = [];
  let wordIndex = 0;

  const isEnglishOnly = language === 'en';
  const useWordTimestamps = supportsWordTimestamps(language);

  for (let segmentIdx = 0; segmentIdx < numSegments; segmentIdx++) {
    const startSample = segmentIdx * segmentSamples;
    const endSample = Math.min(startSample + segmentSamples, totalSamples);
    const segmentData = audioData.slice(startSample, endSample);
    const segmentStartTime = startSample / SAMPLE_RATE;

    const transcriptionProgress = 30 + ((segmentIdx / numSegments) * 70);
    onProgress(transcriptionProgress, `Segment ${segmentIdx + 1}/${numSegments}...`);

    try {
      const result = await model(segmentData, {
        return_timestamps: useWordTimestamps ? 'word' : true,
        chunk_length_s: 30,
        stride_length_s: 5,
        // Anti-hallucination settings
        no_repeat_ngram_size: 3,
        condition_on_previous_text: false,
        compression_ratio_threshold: 2.4,
        logprob_threshold: -1.0,
        ...(isEnglishOnly ? {} : { language, task: 'transcribe' }),
      });

      const chunks: TranscriptChunk[] = result.chunks || [];

      for (const chunk of chunks) {
        const chunkText = chunk.text?.trim();
        if (!chunkText) continue;

        // Skip repetitive/hallucinated chunks
        if (isRepetitiveText(chunkText)) {
          console.log('[Transcribe Worker] Skipping repetitive chunk:', chunkText.slice(0, 50));
          continue;
        }

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

      // Send partial results
      onWords([...allWords]);

    } catch (err) {
      console.error('[Transcribe Worker] Segment error:', err);
    }
  }

  return allWords;
}

// Handle messages from main thread
self.onmessage = async (event) => {
  const { type, audioData, language, audioDuration } = event.data;

  if (type === 'transcribe') {
    try {
      const words = await transcribe(
        audioData,
        language,
        audioDuration,
        (progress, message) => {
          self.postMessage({ type: 'progress', progress, message });
        },
        (words) => {
          self.postMessage({ type: 'words', words });
        }
      );

      self.postMessage({ type: 'complete', words });
    } catch (error) {
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
};
