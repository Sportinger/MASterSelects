// Scene Describer Service
// Uses local Qwen3-VL Python server for native video understanding with temporal reasoning
// Server runs at localhost:5555, processes video files directly (no Ollama)

import { Logger } from './logger';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import type { SceneSegment, SceneDescriptionStatus } from '../types';

const log = Logger.create('SceneDescriber');

const SERVER_URL = 'http://localhost:5555';

// Cancellation state
let isDescribing = false;
let shouldCancel = false;

/**
 * Check if the Qwen3-VL Python server is available and model loaded
 */
export async function checkServerStatus(): Promise<{
  available: boolean;
  modelLoaded: boolean;
  error?: string;
}> {
  try {
    const response = await fetch(`${SERVER_URL}/api/status`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) return { available: false, modelLoaded: false, error: 'Server not responding' };

    const data = await response.json();
    return {
      available: data.available === true,
      modelLoaded: data.model_loaded === true,
      error: data.model_loaded ? undefined : 'Model not loaded on server',
    };
  } catch {
    return {
      available: false,
      modelLoaded: false,
      error: 'Qwen3-VL server not running. Start: cd tools/qwen3vl-server && venv\\Scripts\\python.exe server.py --preload',
    };
  }
}

/**
 * Get the absolute file path for a clip's video
 */
function getVideoPath(clip: { source?: { mediaFileId?: string; filePath?: string } | null; mediaFileId?: string; file?: File | null }): string | null {
  // 1. MediaFile absolutePath (set during drag&drop import)
  const mediaFileId = clip.source?.mediaFileId || clip.mediaFileId;
  if (mediaFileId) {
    const mediaFile = useMediaStore.getState().files.find(f => f.id === mediaFileId);
    if (mediaFile?.absolutePath) return mediaFile.absolutePath;
    if (mediaFile?.filePath && (mediaFile.filePath.includes('/') || mediaFile.filePath.includes('\\'))) {
      return mediaFile.filePath;
    }
  }

  // 2. clip.source.filePath
  const sourceFilePath = clip.source?.filePath;
  if (sourceFilePath && (sourceFilePath.includes('/') || sourceFilePath.includes('\\'))) {
    return sourceFilePath;
  }

  // 3. File object .path (set by drop handler: (file as any).path = filePath)
  const filePath = (clip.file as any)?.path;
  if (filePath && typeof filePath === 'string' && (filePath.includes('/') || filePath.includes('\\'))) {
    return filePath;
  }

  return null;
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
 * Describe a video clip using the Qwen3-VL Python server (native transformers, no Ollama)
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

  // Get file path for the Python server
  let videoPath = getVideoPath(clip);
  let tempUploadUsed = false;

  // If no local path, upload file to server's temp endpoint
  if (!videoPath && clip.file) {
    updateClipSceneDescription(clipId, {
      status: 'describing',
      progress: 5,
      message: 'Uploading video to AI server...',
    });

    try {
      const formData = new FormData();
      formData.append('video', clip.file);
      const uploadResp = await fetch(`${SERVER_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (uploadResp.ok) {
        const uploadData = await uploadResp.json();
        videoPath = uploadData.path;
        tempUploadUsed = true;
      }
    } catch (e) {
      log.warn('Upload fallback failed', e);
    }
  }

  if (!videoPath) {
    updateClipSceneDescription(clipId, {
      status: 'error',
      progress: 0,
      message: 'No file path available. Re-import the video by dragging from disk.',
    });
    return;
  }

  // Check server
  const status = await checkServerStatus();
  if (!status.available || !status.modelLoaded) {
    updateClipSceneDescription(clipId, {
      status: 'error',
      progress: 0,
      message: status.error || 'Server not available',
    });
    return;
  }

  isDescribing = true;
  shouldCancel = false;

  updateClipSceneDescription(clipId, {
    status: 'describing',
    progress: 10,
    message: 'Sending video to AI...',
  });

  try {
    if (shouldCancel) throw new Error('Cancelled');

    const inPoint = clip.inPoint ?? 0;
    const outPoint = clip.outPoint ?? clip.duration;
    const clipDuration = outPoint - inPoint;

    log.info(`Describing ${clip.name} (${clipDuration.toFixed(1)}s) via Qwen3-VL server`);

    updateClipSceneDescription(clipId, {
      progress: 20,
      message: 'AI analyzing video (this may take 1-2 minutes)...',
    });

    const response = await fetch(`${SERVER_URL}/api/describe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_path: videoPath,
        fps: 1.0,
        max_frames: 48,
        duration: clipDuration,
      }),
    });

    if (shouldCancel) throw new Error('Cancelled');

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      throw new Error(errData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    // Offset segments to source time (for playhead sync)
    const segments: SceneSegment[] = (data.segments || []).map((seg: { start: number; end: number; text: string }, i: number) => ({
      id: `scene-${i}`,
      text: seg.text,
      start: seg.start + inPoint,
      end: Math.min(seg.end + inPoint, outPoint),
    }));

    updateClipSceneDescription(clipId, {
      status: 'ready',
      progress: 100,
      segments,
      message: undefined,
    });

    // Propagate analysis status + coverage to MediaFile for badge display
    const mediaFileId = clip.source?.mediaFileId || clip.mediaFileId;
    if (mediaFileId) {
      try {
        const mediaState = useMediaStore.getState();
        const file = mediaState.files.find(f => f.id === mediaFileId);
        if (file) {
          const allClips = useTimelineStore.getState().clips;
          const ranges: [number, number][] = [];
          for (const c of allClips) {
            const mfId = c.source?.mediaFileId || c.mediaFileId;
            if (mfId === mediaFileId && (c.analysisStatus === 'ready' || c.sceneDescriptionStatus === 'ready')) {
              const i = c.inPoint ?? 0;
              const o = c.outPoint ?? (c.source?.naturalDuration ?? file.duration ?? 0);
              if (o > i) ranges.push([i, o]);
            }
          }
          const curIn = clip.inPoint ?? 0;
          const curOut = clip.outPoint ?? (clip.source?.naturalDuration ?? file.duration ?? 0);
          if (curOut > curIn) ranges.push([curIn, curOut]);

          let analysisCoverage = 0;
          if (file.duration && file.duration > 0 && ranges.length > 0) {
            const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
            const merged: [number, number][] = [sorted[0]];
            for (let i = 1; i < sorted.length; i++) {
              const last = merged[merged.length - 1];
              if (sorted[i][0] <= last[1]) last[1] = Math.max(last[1], sorted[i][1]);
              else merged.push([...sorted[i]]);
            }
            analysisCoverage = Math.min(1, merged.reduce((sum, [s, e]) => sum + (e - s), 0) / file.duration);
          }

          useMediaStore.setState({
            files: mediaState.files.map(f =>
              f.id === mediaFileId ? { ...f, analysisStatus: 'ready' as const, analysisCoverage } : f
            ),
          });
        }
      } catch { /* ignore */ }
    }

    // Clean up temp file if we uploaded
    if (tempUploadUsed && videoPath) {
      fetch(`${SERVER_URL}/api/cleanup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: videoPath }),
      }).catch(() => {});
    }

    log.info(`Scene description complete: ${segments.length} segments, ${data.elapsed_seconds}s server time`);

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
