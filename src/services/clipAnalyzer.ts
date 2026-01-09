// Clip Analyzer Service
// Analyzes individual clips for focus, motion, and face detection

import { useTimelineStore } from '../stores/timeline';
import { triggerTimelineSave } from '../stores/mediaStore';
import type { ClipAnalysis, FrameAnalysisData, AnalysisStatus } from '../types';

// Analysis sample interval in milliseconds
const SAMPLE_INTERVAL_MS = 500;

// Cancellation state
let isAnalyzing = false;
let shouldCancel = false;
let currentClipId: string | null = null;

/**
 * Analyze motion between two frames using pixel difference
 * Returns 0-1 (no motion to high motion)
 */
function analyzeMotion(
  currentFrame: ImageData,
  previousFrame: ImageData | null
): number {
  if (!previousFrame) return 0;

  const curr = currentFrame.data;
  const prev = previousFrame.data;
  let diff = 0;

  // Sample every 4th pixel for performance
  for (let i = 0; i < curr.length; i += 16) {
    const currLum = curr[i] * 0.299 + curr[i + 1] * 0.587 + curr[i + 2] * 0.114;
    const prevLum = prev[i] * 0.299 + prev[i + 1] * 0.587 + prev[i + 2] * 0.114;
    diff += Math.abs(currLum - prevLum);
  }

  const pixelCount = curr.length / 16;
  const normalizedDiff = diff / (pixelCount * 255);
  return Math.min(1, normalizedDiff * 5);
}

/**
 * Analyze sharpness/focus using Laplacian variance
 * Returns 0-1 (blurry to sharp)
 */
function analyzeSharpness(frame: ImageData): number {
  const { width, height, data } = frame;
  let variance = 0;
  let mean = 0;
  const values: number[] = [];

  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const idx = (y * width + x) * 4;
      const c = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;

      const t = data[((y - 1) * width + x) * 4] * 0.299 +
                data[((y - 1) * width + x) * 4 + 1] * 0.587 +
                data[((y - 1) * width + x) * 4 + 2] * 0.114;
      const b = data[((y + 1) * width + x) * 4] * 0.299 +
                data[((y + 1) * width + x) * 4 + 1] * 0.587 +
                data[((y + 1) * width + x) * 4 + 2] * 0.114;
      const l = data[(y * width + (x - 1)) * 4] * 0.299 +
                data[(y * width + (x - 1)) * 4 + 1] * 0.587 +
                data[(y * width + (x - 1)) * 4 + 2] * 0.114;
      const r = data[(y * width + (x + 1)) * 4] * 0.299 +
                data[(y * width + (x + 1)) * 4 + 1] * 0.587 +
                data[(y * width + (x + 1)) * 4 + 2] * 0.114;

      const lap = 4 * c - t - b - l - r;
      values.push(lap);
      mean += lap;
    }
  }

  mean /= values.length;
  for (const v of values) {
    variance += (v - mean) ** 2;
  }
  variance /= values.length;

  return Math.min(1, Math.sqrt(variance) / 50);
}

/**
 * Detect faces in a frame (placeholder - returns 0 for now)
 * TODO: Implement with TensorFlow.js face detection model
 */
function detectFaceCount(_frame: ImageData): number {
  return 0;
}

/**
 * Extract a frame from video at specific timestamp
 */
async function extractFrame(
  video: HTMLVideoElement,
  timestampSec: number,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): Promise<ImageData> {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      resolve(ctx.getImageData(0, 0, canvas.width, canvas.height));
    };

    video.addEventListener('seeked', onSeeked);
    video.currentTime = timestampSec;

    // Timeout fallback
    setTimeout(() => {
      video.removeEventListener('seeked', onSeeked);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      resolve(ctx.getImageData(0, 0, canvas.width, canvas.height));
    }, 1000);
  });
}

/**
 * Check if analysis is currently running
 */
export function isAnalysisRunning(): boolean {
  return isAnalyzing;
}

/**
 * Get the clip ID currently being analyzed
 */
export function getCurrentAnalyzingClipId(): string | null {
  return currentClipId;
}

/**
 * Cancel ongoing analysis
 */
export function cancelAnalysis(): void {
  if (isAnalyzing) {
    shouldCancel = true;
    console.log('[ClipAnalyzer] Cancel requested');
  }
}

/**
 * Analyze a clip for focus, motion, and faces
 * Only analyzes the trimmed portion (inPoint to outPoint)
 */
export async function analyzeClip(clipId: string): Promise<void> {
  // Prevent concurrent analysis
  if (isAnalyzing) {
    console.warn('[ClipAnalyzer] Already analyzing');
    return;
  }

  const store = useTimelineStore.getState();
  const clip = store.clips.find(c => c.id === clipId);

  if (!clip || !clip.file) {
    console.warn('[ClipAnalyzer] Clip not found or has no file:', clipId);
    return;
  }

  // Only analyze video files
  if (!clip.file.type.startsWith('video/')) {
    console.warn('[ClipAnalyzer] Not a video file:', clip.file.type);
    return;
  }

  // Set analyzing state
  isAnalyzing = true;
  shouldCancel = false;
  currentClipId = clipId;

  // Update status to analyzing
  updateClipAnalysis(clipId, { status: 'analyzing', progress: 0 });

  let videoUrl: string | null = null;

  try {
    // Create video element
    const video = document.createElement('video');
    videoUrl = URL.createObjectURL(clip.file);
    video.src = videoUrl;
    video.muted = true;
    video.preload = 'auto';

    // Wait for video to load
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to load video'));
      setTimeout(() => reject(new Error('Video load timeout')), 30000);
    });

    // Create canvas for frame extraction (lower res for speed)
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 180;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    if (!ctx) {
      throw new Error('Could not get canvas context');
    }

    // Use clip trim points (inPoint/outPoint) to analyze only the visible portion
    const inPoint = clip.inPoint ?? 0;
    const outPoint = clip.outPoint ?? video.duration;
    const analysisStartTime = inPoint;
    const analysisEndTime = Math.min(outPoint, video.duration);
    const analysisDuration = analysisEndTime - analysisStartTime;

    const totalSamples = Math.ceil((analysisDuration * 1000) / SAMPLE_INTERVAL_MS);
    const frames: FrameAnalysisData[] = [];
    let previousFrame: ImageData | null = null;

    console.log(`[ClipAnalyzer] Analyzing ${totalSamples} frames from ${analysisStartTime.toFixed(1)}s to ${analysisEndTime.toFixed(1)}s (${analysisDuration.toFixed(1)}s)`);

    // Analyze frames
    for (let i = 0; i < totalSamples; i++) {
      // Check for cancellation
      if (shouldCancel) {
        console.log('[ClipAnalyzer] Analysis cancelled');
        updateClipAnalysis(clipId, { status: 'none', progress: 0 });
        return;
      }

      // Calculate timestamp relative to analysis start (which is inPoint)
      const relativeTime = (i * SAMPLE_INTERVAL_MS) / 1000;
      const absoluteTime = analysisStartTime + relativeTime;

      const frame = await extractFrame(video, absoluteTime, canvas, ctx);

      const motion = analyzeMotion(frame, previousFrame);
      const focus = analyzeSharpness(frame);
      const faceCount = detectFaceCount(frame);

      frames.push({
        // Store timestamp relative to the source file (for overlay rendering)
        timestamp: absoluteTime,
        motion,
        focus,
        faceCount,
      });

      previousFrame = frame;

      // Update progress
      const progress = Math.round(((i + 1) / totalSamples) * 100);
      updateClipAnalysis(clipId, { progress });

      // Yield to UI every 5 frames
      if (i % 5 === 0) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    // Final cancellation check
    if (shouldCancel) {
      console.log('[ClipAnalyzer] Analysis cancelled');
      updateClipAnalysis(clipId, { status: 'none', progress: 0 });
      return;
    }

    // Store analysis results
    const analysis: ClipAnalysis = {
      frames,
      sampleInterval: SAMPLE_INTERVAL_MS,
    };

    updateClipAnalysis(clipId, {
      status: 'ready',
      progress: 100,
      analysis,
    });

    triggerTimelineSave();
    console.log('[ClipAnalyzer] Done:', frames.length, 'frames analyzed');

  } catch (error) {
    console.error('[ClipAnalyzer] Analysis failed:', error);
    if (!shouldCancel) {
      updateClipAnalysis(clipId, { status: 'error', progress: 0 });
    }
  } finally {
    // Clean up
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    isAnalyzing = false;
    shouldCancel = false;
    currentClipId = null;
  }
}

/**
 * Update clip analysis data in timeline store
 */
function updateClipAnalysis(
  clipId: string,
  data: {
    status?: AnalysisStatus;
    progress?: number;
    analysis?: ClipAnalysis;
  }
): void {
  const store = useTimelineStore.getState();
  const clips = store.clips.map(clip => {
    if (clip.id !== clipId) return clip;

    return {
      ...clip,
      analysisStatus: data.status ?? clip.analysisStatus,
      analysisProgress: data.progress ?? clip.analysisProgress,
      analysis: data.analysis ?? clip.analysis,
    };
  });

  useTimelineStore.setState({ clips });
}

/**
 * Clear analysis from a clip
 */
export function clearClipAnalysis(clipId: string): void {
  updateClipAnalysis(clipId, {
    status: 'none',
    progress: 0,
    analysis: undefined,
  });
}
