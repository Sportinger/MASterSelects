// Multicam Analyzer Service
// Orchestrates CV analysis (motion, sharpness, faces) and audio analysis for multicam editing

import type { MultiCamSource, MultiCamAnalysis, CameraAnalysis, FrameAnalysis, DetectedFace } from '../stores/multicamStore';
import { useMediaStore } from '../stores/mediaStore';
import { audioAnalyzer } from './audioAnalyzer';

// Analysis sample interval in milliseconds
const SAMPLE_INTERVAL_MS = 500; // Sample every 500ms

/**
 * Analyze motion between two frames using simple pixel difference
 * Returns a value between 0 (no motion) and 1 (high motion)
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
    // Calculate luminance difference
    const currLum = curr[i] * 0.299 + curr[i + 1] * 0.587 + curr[i + 2] * 0.114;
    const prevLum = prev[i] * 0.299 + prev[i + 1] * 0.587 + prev[i + 2] * 0.114;
    diff += Math.abs(currLum - prevLum);
  }

  // Normalize to 0-1 range
  const pixelCount = curr.length / 16;
  const normalizedDiff = diff / (pixelCount * 255);

  // Apply some scaling to make values more meaningful
  return Math.min(1, normalizedDiff * 5);
}

/**
 * Analyze sharpness using Laplacian variance
 * Returns a value between 0 (blurry) and 1 (sharp)
 */
function analyzeSharpness(frame: ImageData): number {
  const { width, height, data } = frame;

  // Convert to grayscale and calculate Laplacian
  let variance = 0;
  let mean = 0;
  const values: number[] = [];

  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const idx = (y * width + x) * 4;

      // Get luminance
      const c = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;

      // Get neighbors
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

      // Laplacian
      const lap = 4 * c - t - b - l - r;
      values.push(lap);
      mean += lap;
    }
  }

  mean /= values.length;

  // Calculate variance
  for (const v of values) {
    variance += (v - mean) ** 2;
  }
  variance /= values.length;

  // Normalize - higher variance = sharper
  // Typical variance range is 0-5000 for sharp images
  return Math.min(1, Math.sqrt(variance) / 50);
}

/**
 * Simple face detection placeholder
 * In production, this would use TensorFlow.js with a face detection model
 */
async function detectFaces(_frame: ImageData): Promise<DetectedFace[]> {
  // TODO: Implement actual face detection with TensorFlow.js
  // For now, return empty array
  return [];
}

/**
 * Extract a frame from a video at a specific timestamp
 */
async function extractFrame(
  video: HTMLVideoElement,
  timestampMs: number,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): Promise<ImageData> {
  return new Promise((resolve, _reject) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);

      // Draw video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Get image data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      resolve(imageData);
    };

    video.addEventListener('seeked', onSeeked);
    video.currentTime = timestampMs / 1000;

    // Timeout fallback
    setTimeout(() => {
      video.removeEventListener('seeked', onSeeked);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      resolve(ctx.getImageData(0, 0, canvas.width, canvas.height));
    }, 1000);
  });
}

class MulticamAnalyzer {
  /**
   * Analyze a single camera
   */
  async analyzeCamera(
    camera: MultiCamSource,
    onProgress?: (progress: number) => void,
    checkCancelled?: () => boolean
  ): Promise<CameraAnalysis | null> {
    const mediaStore = useMediaStore.getState();
    const mediaFile = mediaStore.files.find(f => f.id === camera.mediaFileId);

    if (!mediaFile || !mediaFile.file) {
      console.warn('[MulticamAnalyzer] Media file not found:', camera.mediaFileId);
      return null;
    }

    // Create video element
    const video = document.createElement('video');
    video.src = URL.createObjectURL(mediaFile.file);
    video.muted = true;
    video.preload = 'auto';

    // Wait for video to load
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to load video'));
      setTimeout(() => reject(new Error('Video load timeout')), 30000);
    });

    // Create canvas for frame extraction
    const canvas = document.createElement('canvas');
    canvas.width = 320; // Analyze at lower resolution for speed
    canvas.height = 180;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    if (!ctx) {
      URL.revokeObjectURL(video.src);
      return null;
    }

    const duration = video.duration * 1000; // ms
    const totalSamples = Math.ceil(duration / SAMPLE_INTERVAL_MS);
    const frames: FrameAnalysis[] = [];
    let previousFrame: ImageData | null = null;

    // Get audio levels
    const audioCurve = await audioAnalyzer.analyzeLevels(camera.mediaFileId, SAMPLE_INTERVAL_MS);
    const audioLevelMap = new Map<number, number>();
    if (audioCurve) {
      for (const level of audioCurve.levels) {
        audioLevelMap.set(level.timestamp, level.level);
      }
    }

    // Analyze frames
    for (let i = 0; i < totalSamples; i++) {
      if (checkCancelled?.()) {
        URL.revokeObjectURL(video.src);
        return null;
      }

      const timestamp = i * SAMPLE_INTERVAL_MS;

      // Extract frame
      const frame = await extractFrame(video, timestamp, canvas, ctx);

      // Analyze motion
      const motion = analyzeMotion(frame, previousFrame);

      // Analyze sharpness
      const sharpness = analyzeSharpness(frame);

      // Detect faces (TODO: implement with TensorFlow.js)
      const faces = await detectFaces(frame);

      // Get audio level
      const audioLevel = audioLevelMap.get(timestamp) ?? 0;

      frames.push({
        timestamp,
        motion,
        sharpness,
        faces,
        audioLevel,
      });

      previousFrame = frame;

      // Update progress
      if (onProgress) {
        onProgress(Math.round(((i + 1) / totalSamples) * 100));
      }

      // Yield to UI every 10 frames
      if (i % 10 === 0) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    URL.revokeObjectURL(video.src);

    return {
      cameraId: camera.id,
      frames,
    };
  }

  /**
   * Analyze all cameras
   */
  async analyze(
    cameras: MultiCamSource[],
    onProgress?: (progress: number) => void,
    checkCancelled?: () => boolean
  ): Promise<MultiCamAnalysis> {
    console.log('[MulticamAnalyzer] Starting analysis for', cameras.length, 'cameras');

    const cameraAnalyses: CameraAnalysis[] = [];
    const totalCameras = cameras.length;

    for (let i = 0; i < totalCameras; i++) {
      if (checkCancelled?.()) {
        throw new Error('Analysis cancelled');
      }

      const camera = cameras[i];
      console.log('[MulticamAnalyzer] Analyzing camera:', camera.name);

      const analysis = await this.analyzeCamera(
        camera,
        (cameraProgress) => {
          // Combine camera progress with overall progress
          const overallProgress = Math.round(
            ((i * 100 + cameraProgress) / totalCameras)
          );
          onProgress?.(overallProgress);
        },
        checkCancelled
      );

      if (analysis) {
        cameraAnalyses.push(analysis);
      }
    }

    // Calculate project duration
    const projectDuration = Math.max(
      ...cameras.map(c => c.duration),
      0
    );

    // Aggregate audio levels from all cameras
    const audioLevels: { timestamp: number; level: number }[] = [];
    if (cameraAnalyses.length > 0) {
      const firstAnalysis = cameraAnalyses[0];
      for (const frame of firstAnalysis.frames) {
        // Average audio level across all cameras
        let totalLevel = 0;
        for (const analysis of cameraAnalyses) {
          const matchingFrame = analysis.frames.find(f => f.timestamp === frame.timestamp);
          if (matchingFrame) {
            totalLevel += matchingFrame.audioLevel;
          }
        }
        audioLevels.push({
          timestamp: frame.timestamp,
          level: totalLevel / cameraAnalyses.length,
        });
      }
    }

    console.log('[MulticamAnalyzer] Analysis complete');

    return {
      projectDuration,
      sampleInterval: SAMPLE_INTERVAL_MS,
      cameras: cameraAnalyses,
      audioLevels,
    };
  }
}

// Singleton instance
export const multicamAnalyzer = new MulticamAnalyzer();
