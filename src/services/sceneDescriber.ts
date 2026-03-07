// Scene Describer Service
// Extracts frames in browser, sends all at once to Ollama for temporal video understanding
// Supports chunking with context carry-over for long videos

import { Logger } from './logger';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import type { SceneSegment, SceneDescriptionStatus } from '../types';

const log = Logger.create('SceneDescriber');

const OLLAMA_URL = 'http://localhost:11434';
const MODEL = 'qwen3-vl:8b';

// Frame extraction settings
const FRAME_WIDTH = 320;
const FRAME_HEIGHT = 180;
const JPEG_QUALITY = 0.6;

// Chunking settings
const MAX_FRAMES_PER_CHUNK = 24;  // Max frames per Ollama request
const FRAMES_PER_SECOND = 0.5;    // Default sample rate

// Cancellation state
let isDescribing = false;
let shouldCancel = false;

/**
 * Check if Ollama is available and model loaded
 */
export async function checkServerStatus(): Promise<{
  available: boolean;
  modelLoaded: boolean;
  error?: string;
}> {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) return { available: false, modelLoaded: false, error: 'Ollama not responding' };

    const data = await response.json();
    const models = data.models || [];
    const hasModel = models.some((m: { name: string }) => m.name.startsWith('qwen3-vl'));
    return {
      available: true,
      modelLoaded: hasModel,
      error: hasModel ? undefined : `Model not found. Run: ollama pull ${MODEL}`,
    };
  } catch {
    return { available: false, modelLoaded: false, error: 'Ollama not running. Install from ollama.com' };
  }
}

/**
 * Extract a single frame from video as base64 JPEG
 */
function extractFrame(
  video: HTMLVideoElement,
  timestampSec: number,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): Promise<string> {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
      resolve(dataUrl.split(',')[1]);
    };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = timestampSec;

    // Timeout fallback
    setTimeout(() => {
      video.removeEventListener('seeked', onSeeked);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
      resolve(dataUrl.split(',')[1]);
    }, 3000);
  });
}

/**
 * Parse model output into timestamped segments
 */
function parseSegments(rawText: string, duration: number, timeOffset: number = 0): SceneSegment[] {
  const segments: SceneSegment[] = [];

  // Pattern: [MM:SS-MM:SS] Description
  const rangePattern = /\[?\s*(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})\s*\]?\s*[:\-–]?\s*(.+?)(?=\n\[?\s*\d{1,2}:\d{2}|$)/gs;
  let match;
  while ((match = rangePattern.exec(rawText)) !== null) {
    const start = parseInt(match[1]) * 60 + parseInt(match[2]) + timeOffset;
    const end = parseInt(match[3]) * 60 + parseInt(match[4]) + timeOffset;
    let text = match[5].trim().replace(/\s+/g, ' ').replace(/\*\*(.+?)\*\*/g, '$1');
    if (text.endsWith('.')) text = text.slice(0, -1);
    if (text) {
      segments.push({ id: `scene-${segments.length}`, text, start, end });
    }
  }
  if (segments.length > 0) return segments;

  // Pattern: [MM:SS] or MM:SS - single timestamp
  const singlePattern = /[\[*]*(\d{1,2}):(\d{2})[\]*]*\s*[:\-–]?\s*(.+)/g;
  const timestamps: { time: number; text: string }[] = [];
  while ((match = singlePattern.exec(rawText)) !== null) {
    const t = parseInt(match[1]) * 60 + parseInt(match[2]) + timeOffset;
    let text = match[3].trim().replace(/\s+/g, ' ').replace(/\*\*(.+?)\*\*/g, '$1');
    if (text.endsWith('.')) text = text.slice(0, -1);
    if (text) timestamps.push({ time: t, text });
  }
  if (timestamps.length > 0) {
    for (let i = 0; i < timestamps.length; i++) {
      const end = i + 1 < timestamps.length ? timestamps[i + 1].time : duration + timeOffset;
      segments.push({ id: `scene-${i}`, text: timestamps[i].text, start: timestamps[i].time, end });
    }
    return segments;
  }

  // Fallback: whole text as one segment
  const text = rawText.trim();
  if (text) {
    segments.push({ id: 'scene-0', text, start: timeOffset, end: duration + timeOffset });
  }
  return segments;
}

/**
 * Send a batch of frames to Ollama and get scene descriptions
 */
async function describeFrameBatch(
  frames: string[],
  chunkStartTime: number,
  chunkEndTime: number,
  totalDuration: number,
  previousContext: string,
  signal: AbortSignal,
): Promise<{ rawText: string; segments: SceneSegment[] }> {
  const chunkDuration = chunkEndTime - chunkStartTime;
  const startLabel = formatTime(chunkStartTime);
  const endLabel = formatTime(chunkEndTime);

  let prompt = '/no_think\n';

  if (previousContext) {
    prompt += `This is part of a longer ${Math.round(totalDuration)}s video. `;
    prompt += `Previously (before ${startLabel}): ${previousContext}\n\n`;
    prompt += `Now analyzing ${startLabel} to ${endLabel}. `;
  }

  prompt += `These are ${frames.length} frames evenly sampled from `;
  prompt += previousContext
    ? `the ${startLabel}-${endLabel} portion of the video.\n\n`
    : `a ${Math.round(chunkDuration)}s video.\n\n`;

  prompt += `Output ONLY lines in this exact format:\n`;
  prompt += `[MM:SS-MM:SS] Description of what happens\n\n`;
  prompt += `Timestamps should be relative to the full video (starting from ${startLabel}). `;
  prompt += `Be specific about subjects, actions, camera movements. 1-2 sentences per scene. `;
  prompt += `Do not add any introduction or conclusion.`;

  const response = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'user', content: prompt, images: frames }],
      stream: false,
      options: { num_predict: 4096, temperature: 0.3, num_ctx: 32768 },
    }),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Ollama error: ${response.status} - ${text}`);
  }

  const data = await response.json();
  let rawText = data.message?.content?.trim() || '';

  // Fallback: use thinking field if content is empty
  if (!rawText && data.message?.thinking) {
    rawText = data.message.thinking.trim();
    log.info('Content empty, using thinking field');
  }

  // Clean thinking artifacts from output
  rawText = cleanThinkingText(rawText);

  const segments = parseSegments(rawText, chunkDuration, chunkStartTime);
  return { rawText, segments };
}

/**
 * Strip thinking/reasoning artifacts from model output.
 * Only keep lines matching exact format: [MM:SS-MM:SS] Description
 */
function cleanThinkingText(text: string): string {
  // Remove <think>...</think> blocks
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');

  // Strict pattern: line must start with [MM:SS-MM:SS] followed by a description
  const strictRange = /^\[?\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\s*\]?\s*.{5,}/;
  const lines = text.split('\n');
  const cleanLines = lines.filter(line => strictRange.test(line.trim()));

  if (cleanLines.length >= 2) {
    return cleanLines.join('\n');
  }

  // If strict matching fails, try to extract ONLY the [MM:SS-MM:SS] Description parts
  // from within a larger block of text (thinking mixed with output)
  const extractPattern = /\[\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\]\s*[A-Z][^[\n]{10,}/g;
  const extracted = text.match(extractPattern);
  if (extracted && extracted.length >= 2) {
    return extracted.join('\n');
  }

  // Last resort: return original stripped text
  return text.trim();
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/**
 * Update clip scene description data in the timeline store
 */
function updateClipSceneDescription(
  clipId: string,
  data: {
    status?: SceneDescriptionStatus;
    progress?: number;
    segments?: SceneSegment[];
    message?: string;
  }
): void {
  const store = useTimelineStore.getState();
  const clips = store.clips.map(clip => {
    if (clip.id !== clipId) return clip;
    return {
      ...clip,
      sceneDescriptionStatus: data.status ?? clip.sceneDescriptionStatus,
      sceneDescriptionProgress: data.progress ?? clip.sceneDescriptionProgress,
      sceneDescriptions: data.segments ?? clip.sceneDescriptions,
      sceneDescriptionMessage: data.message,
    };
  });
  useTimelineStore.setState({ clips });
}

/**
 * Describe a video clip — extracts frames in browser, sends to Ollama with temporal context
 */
export async function describeClip(clipId: string): Promise<void> {
  if (isDescribing) {
    log.warn('Already describing a clip');
    return;
  }

  const store = useTimelineStore.getState();
  const clip = store.clips.find(c => c.id === clipId);

  if (!clip || !clip.file) {
    log.warn('Clip not found or has no file', { clipId });
    return;
  }

  const isVideo = clip.file.type.startsWith('video/') ||
    /\.(mp4|webm|mov|avi|mkv|m4v|mxf)$/i.test(clip.file.name);
  if (!isVideo) {
    log.warn('Not a video file');
    return;
  }

  // Check Ollama
  const status = await checkServerStatus();
  if (!status.available || !status.modelLoaded) {
    updateClipSceneDescription(clipId, {
      status: 'error',
      progress: 0,
      message: status.error || 'Ollama not available',
    });
    return;
  }

  isDescribing = true;
  shouldCancel = false;
  const abortController = new AbortController();

  updateClipSceneDescription(clipId, {
    status: 'describing',
    progress: 5,
    message: 'Loading video...',
  });

  let videoUrl: string | null = null;

  try {
    // Create video element
    const video = document.createElement('video');
    videoUrl = URL.createObjectURL(clip.file);
    video.src = videoUrl;
    video.muted = true;
    video.preload = 'auto';

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to load video'));
      setTimeout(() => reject(new Error('Video load timeout')), 30000);
    });

    const canvas = document.createElement('canvas');
    canvas.width = FRAME_WIDTH;
    canvas.height = FRAME_HEIGHT;
    const ctx = canvas.getContext('2d')!;

    const inPoint = clip.inPoint ?? 0;
    const outPoint = clip.outPoint ?? clip.duration;
    const clipDuration = outPoint - inPoint;

    // Calculate all sample timestamps
    const sampleInterval = 1 / FRAMES_PER_SECOND;
    const allTimestamps: number[] = [];
    for (let t = inPoint; t < outPoint; t += sampleInterval) {
      allTimestamps.push(t);
    }
    // Ensure last frame
    if (allTimestamps.length > 0 && outPoint - allTimestamps[allTimestamps.length - 1] > 1) {
      allTimestamps.push(outPoint - 0.1);
    }

    const totalFrames = allTimestamps.length;
    log.info(`Extracting ${totalFrames} frames from ${clip.name} (${clipDuration.toFixed(1)}s)`);

    // Split into chunks
    const chunks: number[][] = [];
    for (let i = 0; i < allTimestamps.length; i += MAX_FRAMES_PER_CHUNK) {
      chunks.push(allTimestamps.slice(i, i + MAX_FRAMES_PER_CHUNK));
    }

    log.info(`${chunks.length} chunk(s) of max ${MAX_FRAMES_PER_CHUNK} frames`);

    const allSegments: SceneSegment[] = [];
    let previousContext = '';
    let framesExtracted = 0;

    for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
      if (shouldCancel) throw new Error('Cancelled');

      const chunkTimestamps = chunks[chunkIdx];
      const chunkStart = chunkTimestamps[0];
      const chunkEnd = chunkTimestamps[chunkTimestamps.length - 1] + sampleInterval;

      // Extract frames for this chunk
      const chunkFrames: string[] = [];
      for (let i = 0; i < chunkTimestamps.length; i++) {
        if (shouldCancel) throw new Error('Cancelled');

        const base64 = await extractFrame(video, chunkTimestamps[i], canvas, ctx);
        chunkFrames.push(base64);
        framesExtracted++;

        const extractPct = 5 + (35 * framesExtracted / totalFrames);
        updateClipSceneDescription(clipId, {
          progress: Math.round(extractPct),
          message: `Extracting frame ${framesExtracted}/${totalFrames}...`,
        });
      }

      if (shouldCancel) throw new Error('Cancelled');

      // Send chunk to Ollama
      const chunkLabel = chunks.length > 1
        ? `AI analyzing chunk ${chunkIdx + 1}/${chunks.length}...`
        : 'AI analyzing video...';

      const analyzePct = 40 + (55 * (chunkIdx + 1) / chunks.length);
      updateClipSceneDescription(clipId, {
        progress: Math.round(40 + (55 * chunkIdx / chunks.length)),
        message: chunkLabel,
      });

      log.info(`Chunk ${chunkIdx + 1}/${chunks.length}: ${chunkFrames.length} frames (${formatTime(chunkStart)}-${formatTime(chunkEnd)})`);

      const result = await describeFrameBatch(
        chunkFrames,
        chunkStart - inPoint,  // relative to clip start
        Math.min(chunkEnd, outPoint) - inPoint,
        clipDuration,
        previousContext,
        abortController.signal,
      );

      // Offset segments to source time (for playhead sync)
      const offsetSegments = result.segments.map((seg, i) => ({
        ...seg,
        id: `scene-${allSegments.length + i}`,
        start: seg.start + inPoint,
        end: Math.min(seg.end + inPoint, outPoint),
      }));

      allSegments.push(...offsetSegments);

      // Build context for next chunk
      const summaries = result.segments.map(s => `[${formatTime(s.start)}-${formatTime(s.end)}] ${s.text}`);
      previousContext = summaries.join('; ');

      // Update with partial results
      updateClipSceneDescription(clipId, {
        progress: Math.round(analyzePct),
        segments: [...allSegments],
      });
    }

    // Re-number IDs
    allSegments.forEach((seg, i) => { seg.id = `scene-${i}`; });

    updateClipSceneDescription(clipId, {
      status: 'ready',
      progress: 100,
      segments: allSegments,
      message: undefined,
    });

    // Propagate analysis status to MediaFile for badge display
    const mediaFileId = clip.source?.mediaFileId || clip.mediaFileId;
    if (mediaFileId) {
      try {
        const mediaState = useMediaStore.getState();
        const file = mediaState.files.find(f => f.id === mediaFileId);
        if (file && file.analysisStatus !== 'ready') {
          useMediaStore.setState({
            files: mediaState.files.map(f =>
              f.id === mediaFileId ? { ...f, analysisStatus: 'ready' as const } : f
            ),
          });
        }
      } catch { /* ignore */ }
    }

    log.info(`Scene description complete: ${allSegments.length} segments in ${chunks.length} chunk(s)`);

  } catch (error) {
    if (shouldCancel) {
      updateClipSceneDescription(clipId, {
        status: 'none',
        progress: 0,
        message: undefined,
        segments: undefined,
      });
      log.info('Scene description cancelled');
    } else {
      log.error('Scene description failed', error);
      updateClipSceneDescription(clipId, {
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  } finally {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    isDescribing = false;
  }
}

/**
 * Cancel ongoing scene description
 */
export function cancelDescription(): void {
  if (isDescribing) {
    shouldCancel = true;
    log.info('Cancel requested');
  }
}

/**
 * Clear scene descriptions from a clip
 */
export function clearSceneDescriptions(clipId: string): void {
  updateClipSceneDescription(clipId, {
    status: 'none',
    progress: 0,
    segments: undefined,
    message: undefined,
  });
}
