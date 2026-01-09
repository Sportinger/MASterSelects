// Clip Transcriber Service
// Handles transcription of individual clips using Whisper

import { useTimelineStore } from '../stores/timeline';
import { triggerTimelineSave } from '../stores/mediaStore';
import type { TranscriptWord, TranscriptStatus } from '../types';

/**
 * Extract audio from a clip's file and transcribe it
 */
export async function transcribeClip(clipId: string): Promise<void> {
  const store = useTimelineStore.getState();
  const clip = store.clips.find(c => c.id === clipId);

  if (!clip || !clip.file) {
    console.warn('[ClipTranscriber] Clip not found or has no file:', clipId);
    return;
  }

  // Update status to transcribing
  updateClipTranscript(clipId, {
    status: 'transcribing',
    progress: 0,
  });

  try {
    // Extract audio and transcribe
    const transcript = await transcribeFile(clip.file, (progress) => {
      updateClipTranscript(clipId, { progress });
    });

    // Update clip with transcript
    updateClipTranscript(clipId, {
      status: 'ready',
      progress: 100,
      words: transcript,
    });

    // Save timeline immediately so transcript persists across refresh
    triggerTimelineSave();

    console.log('[ClipTranscriber] Transcription complete for', clip.name, '-', transcript.length, 'words');
  } catch (error) {
    console.error('[ClipTranscriber] Transcription failed:', error);
    updateClipTranscript(clipId, {
      status: 'error',
      progress: 0,
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
    };
  });

  useTimelineStore.setState({ clips });
}

/**
 * Transcribe a media file using Whisper
 */
async function transcribeFile(
  file: File,
  onProgress: (progress: number) => void
): Promise<TranscriptWord[]> {
  // Check if file has audio
  const hasAudio = file.type.startsWith('video/') || file.type.startsWith('audio/');
  if (!hasAudio) {
    throw new Error('File does not contain audio');
  }

  // Extract audio buffer
  onProgress(5);
  const audioBuffer = await extractAudioBuffer(file);
  onProgress(15);

  // Convert to float32 array at 16kHz for Whisper
  const audioData = await resampleAudio(audioBuffer, 16000);
  onProgress(25);

  // Dynamically import transformers.js
  let pipeline: any;
  let env: any;
  try {
    const transformers = await import('@xenova/transformers');
    pipeline = transformers.pipeline;
    env = transformers.env;

    // Configure environment for browser usage
    // Use CDN for model files and enable caching
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    // Use jsDelivr CDN as fallback (more reliable for browser)
    env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/';
  } catch (error) {
    throw new Error(
      'Whisper model requires @xenova/transformers. Install with: npm install @xenova/transformers'
    );
  }

  onProgress(30);

  // Load Whisper model
  console.log('[ClipTranscriber] Loading Whisper model...');
  const transcriber = await pipeline(
    'automatic-speech-recognition',
    'Xenova/whisper-tiny.en', // Use English-only model (smaller, faster)
    {
      progress_callback: (data: any) => {
        if (data.status === 'progress' && data.progress) {
          // Model loading progress (30-50%)
          const modelProgress = 30 + (data.progress * 0.2);
          onProgress(Math.round(modelProgress));
        }
        if (data.status === 'ready') {
          console.log('[ClipTranscriber] Whisper model ready');
        }
      },
      revision: 'main',
    }
  );

  onProgress(50);

  // Run transcription with word-level timestamps
  const result = await transcriber(audioData, {
    return_timestamps: 'word',
    chunk_length_s: 30,
    stride_length_s: 5,
    language: 'en',
    task: 'transcribe',
  });

  console.log('[ClipTranscriber] Raw result:', JSON.stringify(result, null, 2));
  console.log('[ClipTranscriber] Result keys:', Object.keys(result));
  console.log('[ClipTranscriber] Has chunks:', !!result.chunks, 'length:', result.chunks?.length);
  console.log('[ClipTranscriber] Has text:', !!result.text);
  onProgress(95);

  // Convert result to TranscriptWord array
  const words: TranscriptWord[] = [];
  let wordIndex = 0;

  // Handle different result formats from Whisper
  const chunks = result.chunks || result.output?.chunks || [];

  if (chunks.length > 0) {
    for (const chunk of chunks) {
      const chunkText = chunk.text?.trim();
      if (!chunkText) continue;

      const chunkStart = chunk.timestamp[0] ?? 0;
      const chunkEnd = chunk.timestamp[1] ?? chunkStart + 0.5;

      // Check if chunk contains multiple words (split by spaces)
      const chunkWords = chunkText.split(/\s+/).filter(w => w.length > 0);

      if (chunkWords.length === 1) {
        // Single word chunk - use as-is
        words.push({
          id: `word-${wordIndex++}`,
          text: chunkText,
          start: chunkStart,
          end: chunkEnd,
          confidence: 1,
          speaker: 'Speaker 1',
        });
      } else {
        // Multiple words in chunk - distribute time evenly
        const duration = chunkEnd - chunkStart;
        const wordDuration = duration / chunkWords.length;

        for (let i = 0; i < chunkWords.length; i++) {
          const wordStart = chunkStart + (i * wordDuration);
          const wordEnd = wordStart + wordDuration;

          words.push({
            id: `word-${wordIndex++}`,
            text: chunkWords[i],
            start: wordStart,
            end: wordEnd,
            confidence: 1,
            speaker: 'Speaker 1',
          });
        }
      }
    }
  }

  // Fallback: if no chunks but have text, split it
  const textResult = result.text || result.output?.text || '';
  if (words.length === 0 && textResult) {
    console.log('[ClipTranscriber] Using text fallback:', textResult);
    const allWords = textResult.trim().split(/\s+/).filter((w: string) => w.length > 0);
    const totalDuration = audioBuffer.duration;
    const wordDuration = totalDuration / allWords.length;

    for (let i = 0; i < allWords.length; i++) {
      words.push({
        id: `word-${i}`,
        text: allWords[i],
        start: i * wordDuration,
        end: (i + 1) * wordDuration,
        confidence: 1,
        speaker: 'Speaker 1',
      });
    }
  }

  console.log('[ClipTranscriber] Parsed', words.length, 'words');
  onProgress(100);
  return words;
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
  });
}
