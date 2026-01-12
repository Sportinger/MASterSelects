// Clip Analyzer Service
// Analyzes individual clips for focus, motion, and face detection

import { useTimelineStore } from '../stores/timeline';
import { triggerTimelineSave } from '../stores/mediaStore';
import { projectFileService } from './projectFileService';
import { engine } from '../engine/WebGPUEngine';
import {
  OpticalFlowAnalyzer,
  getOpticalFlowAnalyzer,
  resetOpticalFlowAnalyzer,
  destroyOpticalFlowAnalyzer,
  type MotionResult,
} from '../engine/analysis/OpticalFlowAnalyzer';
import type { ClipAnalysis, FrameAnalysisData, AnalysisStatus } from '../types';

// Analysis sample interval in milliseconds
const SAMPLE_INTERVAL_MS = 500;

// Cancellation state
let isAnalyzing = false;
let shouldCancel = false;
let currentClipId: string | null = null;

// GPU optical flow analyzer instance
let flowAnalyzer: OpticalFlowAnalyzer | null = null;
let useGPUAnalysis = true; // Will be set to false if GPU init fails

/**
 * Analyze motion between two frames using grid-based analysis
 * Distinguishes between:
 * - Global motion: Camera movement, pans, scene cuts (whole frame changes uniformly)
 * - Local motion: Object movement (only parts of frame change)
 */
function analyzeMotion(
  currentFrame: ImageData,
  previousFrame: ImageData | null
): MotionResult {
  if (!previousFrame) {
    return { total: 0, global: 0, local: 0, isSceneCut: false };
  }

  const { width, height, data: curr } = currentFrame;
  const prev = previousFrame.data;

  // Divide frame into a 4x4 grid (16 regions)
  const gridSize = 4;
  const regionWidth = Math.floor(width / gridSize);
  const regionHeight = Math.floor(height / gridSize);
  const regionMotion: number[] = [];

  // Calculate motion for each region
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      let regionDiff = 0;
      let regionPixels = 0;

      const startX = gx * regionWidth;
      const startY = gy * regionHeight;
      const endX = Math.min(startX + regionWidth, width);
      const endY = Math.min(startY + regionHeight, height);

      // Sample every 2nd pixel in each region for performance
      for (let y = startY; y < endY; y += 2) {
        for (let x = startX; x < endX; x += 2) {
          const idx = (y * width + x) * 4;
          const currLum = curr[idx] * 0.299 + curr[idx + 1] * 0.587 + curr[idx + 2] * 0.114;
          const prevLum = prev[idx] * 0.299 + prev[idx + 1] * 0.587 + prev[idx + 2] * 0.114;
          regionDiff += Math.abs(currLum - prevLum);
          regionPixels++;
        }
      }

      // Normalize region motion to 0-1
      const normalizedRegion = regionPixels > 0 ? (regionDiff / regionPixels) / 255 : 0;
      regionMotion.push(Math.min(1, normalizedRegion * 5));
    }
  }

  // Calculate statistics across regions
  const avgMotion = regionMotion.reduce((a, b) => a + b, 0) / regionMotion.length;
  const motionVariance = regionMotion.reduce((acc, m) => acc + Math.pow(m - avgMotion, 2), 0) / regionMotion.length;
  const motionStdDev = Math.sqrt(motionVariance);

  // Determine motion type:
  // - Low variance + high motion = Global motion (camera/scene change)
  // - High variance = Local motion (objects moving)
  // Threshold: if std dev < 0.15 and avg motion > 0.1, it's mostly global

  const varianceThreshold = 0.15;
  const isUniform = motionStdDev < varianceThreshold;
  const sceneCutThreshold = 0.6;
  const isSceneCut = avgMotion > sceneCutThreshold;

  let globalMotion: number;
  let localMotion: number;

  if (isUniform) {
    // Uniform motion across frame = camera/global motion
    globalMotion = avgMotion;
    localMotion = 0;
  } else {
    // Non-uniform = mix of global and local
    // Global component is the minimum motion (background)
    const minRegionMotion = Math.min(...regionMotion);
    globalMotion = minRegionMotion;

    // Local component is the excess above the global baseline
    const maxRegionMotion = Math.max(...regionMotion);
    localMotion = maxRegionMotion - minRegionMotion;
  }

  // For scene cuts, mark as high global motion
  if (isSceneCut) {
    globalMotion = avgMotion;
    localMotion = 0;
  }

  return {
    total: avgMotion,
    global: Math.min(1, globalMotion),
    local: Math.min(1, localMotion),
    isSceneCut,
  };
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
 * Initialize GPU optical flow analyzer
 * @param forceRecreate - If true, destroys and recreates the analyzer
 */
async function initGPUAnalyzer(forceRecreate = false): Promise<boolean> {
  // If force recreate or no analyzer exists, destroy and create new
  if (forceRecreate && flowAnalyzer) {
    console.log('[ClipAnalyzer] Destroying existing GPU analyzer for fresh start');
    destroyOpticalFlowAnalyzer();
    flowAnalyzer = null;
  }

  if (flowAnalyzer) return true;

  try {
    const device = engine.getDevice();
    if (!device) {
      console.warn('[ClipAnalyzer] WebGPU device not available, falling back to CPU');
      useGPUAnalysis = false;
      return false;
    }

    flowAnalyzer = await getOpticalFlowAnalyzer(device);
    console.log('[ClipAnalyzer] GPU optical flow analyzer initialized');
    return true;
  } catch (error) {
    console.warn('[ClipAnalyzer] Failed to init GPU analyzer, falling back to CPU:', error);
    useGPUAnalysis = false;
    flowAnalyzer = null;
    return false;
  }
}

/**
 * Analyze motion using GPU optical flow
 */
async function analyzeMotionGPU(bitmap: ImageBitmap): Promise<MotionResult> {
  if (!flowAnalyzer) {
    return { total: 0, global: 0, local: 0, isSceneCut: false };
  }

  try {
    return await flowAnalyzer.analyzeFrame(bitmap);
  } catch (error) {
    console.warn('[ClipAnalyzer] GPU motion analysis failed:', error);
    return { total: 0, global: 0, local: 0, isSceneCut: false };
  }
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

  // Check for cached analysis first (from project folder, not browser cache)
  const mediaFileId = clip.source?.mediaFileId;
  const inPoint = clip.inPoint ?? 0;
  const outPoint = clip.outPoint ?? clip.duration;

  if (mediaFileId && projectFileService.isProjectOpen()) {
    try {
      const cachedAnalysis = await projectFileService.getAnalysis(mediaFileId, inPoint, outPoint);
      if (cachedAnalysis) {
        console.log('[ClipAnalyzer] Found cached analysis in project folder, loading...');

        const analysis: ClipAnalysis = {
          frames: cachedAnalysis.frames as FrameAnalysisData[],
          sampleInterval: cachedAnalysis.sampleInterval,
        };

        updateClipAnalysis(clipId, {
          status: 'ready',
          progress: 100,
          analysis,
        });

        triggerTimelineSave();
        isAnalyzing = false;
        currentClipId = null;
        return;
      }
    } catch (err) {
      console.warn('[ClipAnalyzer] Failed to check analysis cache:', err);
    }
  }

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

    // Try to initialize GPU optical flow analyzer
    // Force recreate analyzer to ensure fresh state (avoids stale GPU errors)
    const gpuAvailable = useGPUAnalysis && await initGPUAnalyzer(true);
    if (gpuAvailable) {
      console.log('[ClipAnalyzer] Using GPU optical flow analysis');
      resetOpticalFlowAnalyzer(); // Reset state for new clip
    } else {
      console.log('[ClipAnalyzer] Using CPU motion analysis (fallback)');
    }

    // Create canvas for frame extraction
    const canvas = document.createElement('canvas');
    // GPU uses 160x90, CPU uses 320x180
    canvas.width = gpuAvailable ? 160 : 320;
    canvas.height = gpuAvailable ? 90 : 180;
    const ctx = canvas.getContext('2d', { willReadFrequently: !gpuAvailable });

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

      // Extract frame
      const frame = await extractFrame(video, absoluteTime, canvas, ctx);

      // Analyze motion using GPU or CPU
      let motionResult: MotionResult;
      const analysisStart = performance.now();

      if (gpuAvailable) {
        // GPU path: create ImageBitmap and use optical flow
        const bitmap = await createImageBitmap(canvas);
        motionResult = await analyzeMotionGPU(bitmap);
        bitmap.close();
      } else {
        // CPU fallback: use grid-based analysis
        motionResult = analyzeMotion(frame, previousFrame);
      }

      const analysisTime = performance.now() - analysisStart;
      if (i === 0) {
        console.log(`[ClipAnalyzer] First frame analysis took ${analysisTime.toFixed(1)}ms (${gpuAvailable ? 'GPU' : 'CPU'})`);
      }

      const focus = analyzeSharpness(frame);
      const faceCount = detectFaceCount(frame);

      frames.push({
        // Store timestamp relative to the source file (for overlay rendering)
        timestamp: absoluteTime,
        motion: motionResult.total,
        globalMotion: motionResult.global,
        localMotion: motionResult.local,
        focus,
        brightness: 0.5, // TODO: Calculate actual brightness from frame
        faceCount,
        isSceneCut: motionResult.isSceneCut,
      });

      previousFrame = frame;

      // Update progress and analysis data in real-time
      const progress = Math.round(((i + 1) / totalSamples) * 100);

      // Update with current frames for real-time graph rendering
      const partialAnalysis: ClipAnalysis = {
        frames: [...frames],
        sampleInterval: SAMPLE_INTERVAL_MS,
      };
      updateClipAnalysis(clipId, { progress, analysis: partialAnalysis });

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

    // Save to project folder (not browser cache)
    if (mediaFileId && projectFileService.isProjectOpen()) {
      try {
        await projectFileService.saveAnalysis(
          mediaFileId,
          analysisStartTime,
          analysisEndTime,
          frames,
          SAMPLE_INTERVAL_MS
        );
        console.log('[ClipAnalyzer] Analysis saved to project folder');
      } catch (err) {
        console.warn('[ClipAnalyzer] Failed to save analysis to project folder:', err);
      }
    }

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
