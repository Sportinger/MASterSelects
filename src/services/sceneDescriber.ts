// Scene Describer Service
// Uses local Ollama (Qwen3.5) to describe video content with timestamps

import { Logger } from './logger';
import { useTimelineStore } from '../stores/timeline';
import type { SceneSegment, SceneDescriptionStatus } from '../types';

const log = Logger.create('SceneDescriber');

const OLLAMA_URL = 'http://localhost:11434';
const MODEL = 'qwen3.5:9b';
// Sample every N seconds for frame extraction
const SAMPLE_INTERVAL_SEC = 3;
// How many frames to send per batch to the model
const FRAMES_PER_BATCH = 6;
// Canvas size for frame capture (smaller = faster, less VRAM)
const CAPTURE_WIDTH = 512;
const CAPTURE_HEIGHT = 288;

// Cancellation state
let isDescribing = false;
let shouldCancel = false;

/**
 * Check if Ollama is available and the model is loaded
 */
export async function checkOllamaStatus(): Promise<{ available: boolean; modelLoaded: boolean; error?: string }> {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) return { available: false, modelLoaded: false, error: 'Ollama not responding' };

    const data = await response.json();
    const models = data.models || [];
    const hasModel = models.some((m: { name: string }) => m.name.startsWith('qwen3.5'));

    return { available: true, modelLoaded: hasModel };
  } catch {
    return { available: false, modelLoaded: false, error: 'Ollama not running. Install from ollama.com and run: ollama pull qwen3.5:9b' };
  }
}

/**
 * Extract a single frame from video as base64 JPEG
 */
function extractFrameAsBase64(
  video: HTMLVideoElement,
  timestampSec: number,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): Promise<string> {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      // Get as JPEG base64 (smaller than PNG)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      // Strip the data:image/jpeg;base64, prefix
      resolve(dataUrl.split(',')[1]);
    };

    video.addEventListener('seeked', onSeeked);
    video.currentTime = timestampSec;

    // Timeout fallback
    setTimeout(() => {
      video.removeEventListener('seeked', onSeeked);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      resolve(dataUrl.split(',')[1]);
    }, 2000);
  });
}

/**
 * Send frames to Ollama and get scene description
 */
async function describeFrameBatch(
  frames: { base64: string; timestamp: number }[],
  isFirstBatch: boolean,
): Promise<string> {
  const timeLabels = frames.map(f => {
    const mins = Math.floor(f.timestamp / 60);
    const secs = Math.floor(f.timestamp % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }).join(', ');

  const prompt = isFirstBatch
    ? `These are ${frames.length} frames from a video at timestamps: ${timeLabels}. ` +
      `Describe what happens in each frame concisely. For each frame write exactly one line in this format:\n` +
      `[MM:SS] Description of what is visible.\n` +
      `Focus on actions, people, objects, camera movement, and scene changes. Be specific and brief (max 20 words per line). Do not add any other text.`
    : `Continuation - ${frames.length} more frames at timestamps: ${timeLabels}. ` +
      `Same format: [MM:SS] Description. One line per frame, max 20 words each. No other text.`;

  const response = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages: [{
        role: 'user',
        content: prompt,
        images: frames.map(f => f.base64),
      }],
      stream: false,
      think: false,
      options: {
        num_predict: 512,
        temperature: 0.3,
      },
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Ollama API error: ${response.status} - ${text}`);
  }

  const data = await response.json();
  return data.message?.content || '';
}

/**
 * Parse model response into SceneSegment objects
 */
function parseResponse(
  text: string,
  frameTimestamps: number[],
  segmentInterval: number,
): SceneSegment[] {
  const segments: SceneSegment[] = [];
  const lines = text.split('\n').filter(l => l.trim());

  for (const line of lines) {
    // Match patterns like [0:00], [00:00], [1:30], etc.
    const match = line.match(/\[(\d{1,2}):(\d{2})\]\s*(.+)/);
    if (!match) continue;

    const mins = parseInt(match[1], 10);
    const secs = parseInt(match[2], 10);
    const description = match[3].trim();
    const timestamp = mins * 60 + secs;

    // Find the closest frame timestamp
    let closest = frameTimestamps[0];
    let minDist = Math.abs(closest - timestamp);
    for (const ft of frameTimestamps) {
      const dist = Math.abs(ft - timestamp);
      if (dist < minDist) {
        minDist = dist;
        closest = ft;
      }
    }

    segments.push({
      id: `scene-${segments.length}`,
      text: description,
      start: closest,
      end: closest + segmentInterval,
    });
  }

  return segments;
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
 * Describe a video clip using Ollama AI
 */
export async function describeClip(clipId: string): Promise<void> {
  if (isDescribing) {
    log.warn('Already describing a clip');
    return;
  }

  // Check Ollama availability
  const status = await checkOllamaStatus();
  if (!status.available) {
    updateClipSceneDescription(clipId, {
      status: 'error',
      progress: 0,
      message: status.error || 'Ollama not available',
    });
    return;
  }
  if (!status.modelLoaded) {
    updateClipSceneDescription(clipId, {
      status: 'error',
      progress: 0,
      message: `Model ${MODEL} not found. Run: ollama pull ${MODEL}`,
    });
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

  isDescribing = true;
  shouldCancel = false;
  updateClipSceneDescription(clipId, {
    status: 'describing',
    progress: 0,
    message: 'Loading video...',
  });

  let videoUrl: string | null = null;

  try {
    // Create video element for frame extraction
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
    canvas.width = CAPTURE_WIDTH;
    canvas.height = CAPTURE_HEIGHT;
    const ctx = canvas.getContext('2d')!;

    const inPoint = clip.inPoint ?? 0;
    const outPoint = clip.outPoint ?? clip.duration;
    const clipDuration = outPoint - inPoint;

    // Calculate sample timestamps
    const timestamps: number[] = [];
    for (let t = inPoint; t < outPoint; t += SAMPLE_INTERVAL_SEC) {
      timestamps.push(t);
    }
    // Always include last frame if not already close
    if (timestamps.length > 0 && outPoint - timestamps[timestamps.length - 1] > 1) {
      timestamps.push(outPoint - 0.1);
    }

    const totalFrames = timestamps.length;
    log.info(`Extracting ${totalFrames} frames from ${clip.name} (${clipDuration.toFixed(1)}s)`);

    updateClipSceneDescription(clipId, {
      progress: 5,
      message: `Extracting ${totalFrames} frames...`,
    });

    // Extract all frames
    const frames: { base64: string; timestamp: number }[] = [];
    for (let i = 0; i < timestamps.length; i++) {
      if (shouldCancel) throw new Error('Cancelled');

      const base64 = await extractFrameAsBase64(video, timestamps[i], canvas, ctx);
      frames.push({ base64, timestamp: timestamps[i] });

      const extractProgress = 5 + (35 * (i + 1) / totalFrames);
      updateClipSceneDescription(clipId, {
        progress: Math.round(extractProgress),
        message: `Extracted frame ${i + 1}/${totalFrames}`,
      });
    }

    // Process in batches
    const allSegments: SceneSegment[] = [];
    const totalBatches = Math.ceil(frames.length / FRAMES_PER_BATCH);

    for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
      if (shouldCancel) throw new Error('Cancelled');

      const batchStart = batchIdx * FRAMES_PER_BATCH;
      const batchFrames = frames.slice(batchStart, batchStart + FRAMES_PER_BATCH);

      updateClipSceneDescription(clipId, {
        progress: Math.round(40 + (50 * batchIdx / totalBatches)),
        message: `AI analyzing batch ${batchIdx + 1}/${totalBatches}...`,
      });

      log.info(`Sending batch ${batchIdx + 1}/${totalBatches} (${batchFrames.length} frames) to Ollama`);

      const response = await describeFrameBatch(batchFrames, batchIdx === 0);
      const batchTimestamps = batchFrames.map(f => f.timestamp);
      const batchSegments = parseResponse(response, batchTimestamps, SAMPLE_INTERVAL_SEC);

      // If parsing failed, create segments from raw response lines
      if (batchSegments.length === 0 && response.trim()) {
        const lines = response.split('\n').filter(l => l.trim());
        for (let i = 0; i < Math.min(lines.length, batchFrames.length); i++) {
          batchSegments.push({
            id: `scene-${allSegments.length + i}`,
            text: lines[i].replace(/^\[.*?\]\s*/, '').trim(),
            start: batchFrames[i].timestamp,
            end: batchFrames[i].timestamp + SAMPLE_INTERVAL_SEC,
          });
        }
      }

      allSegments.push(...batchSegments);

      // Update with partial results
      updateClipSceneDescription(clipId, {
        segments: allSegments.map((s, i) => ({ ...s, id: `scene-${i}` })),
      });
    }

    // Finalize: adjust end times so segments don't overlap
    const finalSegments = allSegments.map((seg, i, arr) => ({
      ...seg,
      id: `scene-${i}`,
      end: i < arr.length - 1 ? arr[i + 1].start : outPoint,
    }));

    updateClipSceneDescription(clipId, {
      status: 'ready',
      progress: 100,
      segments: finalSegments,
      message: undefined,
    });

    log.info(`Scene description complete: ${finalSegments.length} segments`);

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
