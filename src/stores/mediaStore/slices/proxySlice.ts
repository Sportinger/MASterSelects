// Proxy generation slice

import type { MediaFile, MediaSliceCreator, ProxyStatus } from '../types';
import { PROXY_FPS } from '../constants';
import { projectFileService } from '../../../services/projectFileService';
import { useTimelineStore } from '../../timeline';
import { Logger } from '../../../services/logger';

const log = Logger.create('Proxy');

// Track active generations for cancellation
const activeProxyGenerations = new Map<string, { cancelled: boolean }>();

/** Check if a proxy is complete (>= 98% of expected frames) */
function isProxyComplete(file: MediaFile, frameCountOverride?: number): boolean {
  const frameCount = frameCountOverride ?? file.proxyFrameCount;
  if (!frameCount || !file.duration) return false;
  const expectedFrames = Math.ceil(file.duration * PROXY_FPS);
  return frameCount >= expectedFrames * 0.98;
}

export interface ProxyActions {
  proxyEnabled: boolean;
  setProxyEnabled: (enabled: boolean) => void;
  generateProxy: (mediaFileId: string) => Promise<void>;
  cancelProxyGeneration: (mediaFileId: string) => void;
  updateProxyProgress: (mediaFileId: string, progress: number) => void;
  setProxyStatus: (mediaFileId: string, status: ProxyStatus) => void;
  getNextFileNeedingProxy: () => MediaFile | undefined;
}

export const createProxySlice: MediaSliceCreator<ProxyActions> = (set, get) => ({
  proxyEnabled: false,

  setProxyEnabled: async (enabled: boolean) => {
    set({ proxyEnabled: enabled });

    if (enabled) {
      // Mute all video elements when enabling proxy mode
      const clips = useTimelineStore.getState().clips;
      clips.forEach(clip => {
        if (clip.source?.videoElement) {
          clip.source.videoElement.muted = true;
          if (!clip.source.videoElement.paused) {
            clip.source.videoElement.pause();
          }
        }
      });
      log.info('Mode enabled - muted all videos');
    }
  },

  updateProxyProgress: (mediaFileId: string, progress: number) => {
    set((state) => ({
      files: state.files.map((f) =>
        f.id === mediaFileId ? { ...f, proxyProgress: progress } : f
      ),
    }));
  },

  setProxyStatus: async (mediaFileId: string, status: ProxyStatus) => {
    const { proxyEnabled } = get();

    set((state) => ({
      files: state.files.map((f) =>
        f.id === mediaFileId ? { ...f, proxyStatus: status } : f
      ),
    }));

    // Mute video when proxy becomes ready
    if (status === 'ready' && proxyEnabled) {
      const clips = useTimelineStore.getState().clips;
      clips.forEach(clip => {
        if (clip.mediaFileId === mediaFileId && clip.source?.videoElement) {
          clip.source.videoElement.muted = true;
          if (!clip.source.videoElement.paused) {
            clip.source.videoElement.pause();
          }
        }
      });
    }
  },

  getNextFileNeedingProxy: () => {
    const { files, currentlyGeneratingProxyId } = get();
    return files.find(
      (f) =>
        f.type === 'video' &&
        f.file &&
        f.proxyStatus !== 'generating' &&
        f.id !== currentlyGeneratingProxyId &&
        (f.proxyStatus !== 'ready' || !isProxyComplete(f))
    );
  },

  generateProxy: async (mediaFileId: string) => {
    const { files, currentlyGeneratingProxyId } = get();

    if (currentlyGeneratingProxyId) {
      log.debug('Already generating, queuing:', mediaFileId);
      return;
    }

    const mediaFile = files.find((f) => f.id === mediaFileId);
    if (!mediaFile || mediaFile.type !== 'video' || !mediaFile.file) {
      log.warn('Invalid media file:', mediaFileId);
      return;
    }

    if (!projectFileService.isProjectOpen()) {
      log.error('No project open!');
      return;
    }

    log.info(`Starting generation for ${mediaFile.name}...`);

    // Check existing frames on disk
    const storageKey = mediaFile.fileHash || mediaFileId;
    const existingCount = await projectFileService.getProxyFrameCount(storageKey);
    if (existingCount > 0 && isProxyComplete(mediaFile, existingCount)) {
      log.debug('Already complete:', mediaFile.name);
      set((s) => ({
        files: s.files.map((f) =>
          f.id === mediaFileId
            ? { ...f, proxyStatus: 'ready' as ProxyStatus, proxyProgress: 100, proxyFrameCount: existingCount }
            : f
        ),
      }));
      return;
    }

    // Log resume info if partial proxy exists
    if (existingCount > 0) {
      const expectedFrames = Math.ceil((mediaFile.duration || 0) * PROXY_FPS);
      log.info(`Resuming proxy for ${mediaFile.name}: ${existingCount}/${expectedFrames} frames on disk`);
    }

    // Set up cancellation
    const controller = { cancelled: false };
    activeProxyGenerations.set(mediaFileId, controller);

    // Calculate initial progress for resume
    const initialProgress = existingCount > 0 && mediaFile.duration
      ? Math.min(99, Math.round((existingCount / Math.ceil(mediaFile.duration * PROXY_FPS)) * 100))
      : 0;

    // Inline setProxyStatus and updateProxyProgress
    set({ currentlyGeneratingProxyId: mediaFileId });
    set((state) => ({
      files: state.files.map((f) =>
        f.id === mediaFileId ? { ...f, proxyStatus: 'generating' as ProxyStatus, proxyProgress: initialProgress, proxyFps: PROXY_FPS } : f
      ),
    }));

    // Progress updater function
    const updateProgress = (progress: number) => {
      set((state) => ({
        files: state.files.map((f) =>
          f.id === mediaFileId ? { ...f, proxyProgress: progress } : f
        ),
      }));
    };

    try {
      // Generate video proxy
      const result = await generateVideoProxy(
        mediaFile,
        storageKey,
        controller,
        updateProgress
      );

      if (result && !controller.cancelled) {
        // Update status to 'ready' IMMEDIATELY after frames complete
        // Don't wait for audio extraction - it's optional and can happen in background
        set((s) => ({
          files: s.files.map((f) =>
            f.id === mediaFileId
              ? {
                  ...f,
                  proxyStatus: 'ready' as ProxyStatus,
                  proxyProgress: 100,
                  proxyFrameCount: result.frameCount,
                  proxyFps: result.fps,
                }
              : f
          ),
        }));

        log.info(`Complete: ${result.frameCount} frames for ${mediaFile.name}`);

        // Extract audio proxy in background (non-blocking)
        extractAudioProxy(mediaFile, storageKey).then(async () => {
          const hasAudioProxy = await projectFileService.hasProxyAudio(storageKey);
          if (hasAudioProxy) {
            set((s) => ({
              files: s.files.map((f) =>
                f.id === mediaFileId ? { ...f, hasProxyAudio: true } : f
              ),
            }));
            log.debug(`Audio proxy ready for ${mediaFile.name}`);
          }
        }).catch(() => {
          // Audio extraction errors are non-fatal
        });
      } else if (!controller.cancelled) {
        // Set error status inline
        set((state) => ({
          files: state.files.map((f) =>
            f.id === mediaFileId ? { ...f, proxyStatus: 'error' as ProxyStatus } : f
          ),
        }));
      }
    } catch (e) {
      log.error('Generation failed:', e);
      set((state) => ({
        files: state.files.map((f) =>
          f.id === mediaFileId ? { ...f, proxyStatus: 'error' as ProxyStatus } : f
        ),
      }));
    } finally {
      activeProxyGenerations.delete(mediaFileId);
      set({ currentlyGeneratingProxyId: null });
    }
  },

  cancelProxyGeneration: (mediaFileId: string) => {
    const controller = activeProxyGenerations.get(mediaFileId);
    if (controller) {
      controller.cancelled = true;
      log.info('Cancelled:', mediaFileId);
    }

    const { currentlyGeneratingProxyId, files } = get();
    if (currentlyGeneratingProxyId === mediaFileId) {
      const mediaFile = files.find((f) => f.id === mediaFileId);
      const hasExistingFrames = mediaFile?.proxyFrameCount && mediaFile.proxyFrameCount > 0;

      set((state) => ({
        currentlyGeneratingProxyId: null,
        files: state.files.map((f) =>
          f.id === mediaFileId
            ? {
                ...f,
                proxyStatus: (hasExistingFrames ? 'ready' : 'none') as ProxyStatus,
                proxyProgress: hasExistingFrames ? 100 : 0,
              }
            : f
        ),
      }));
    }
  },
});

async function generateVideoProxy(
  mediaFile: MediaFile,
  storageKey: string,
  controller: { cancelled: boolean },
  updateProgress: (progress: number) => void
): Promise<{ frameCount: number; fps: number } | null> {
  const { getProxyGenerator } = await import('../../../services/proxyGenerator');
  const generator = getProxyGenerator();

  // Fetch existing frame indices for resume
  const existingIndices = await projectFileService.getProxyFrameIndices(storageKey);

  const saveFrame = async (frame: { frameIndex: number; blob: Blob }) => {
    await projectFileService.saveProxyFrame(storageKey, frame.frameIndex, frame.blob);
  };

  return generator.generate(
    mediaFile.file!,
    mediaFile.id,
    updateProgress,
    () => controller.cancelled,
    saveFrame,
    existingIndices.size > 0 ? existingIndices : undefined
  );
}

async function extractAudioProxy(
  mediaFile: MediaFile,
  storageKey: string
): Promise<void> {
  try {
    log.debug('Extracting audio...');
    const { extractAudioFromVideo } = await import('../../../services/audioExtractor');

    const result = await extractAudioFromVideo(mediaFile.file!, () => {});
    if (result && result.blob && result.blob.size > 0) {
      await projectFileService.saveProxyAudio(storageKey, result.blob);
      log.debug(`Audio saved (${(result.blob.size / 1024).toFixed(1)}KB)`);
    }
  } catch (e) {
    log.warn('Audio extraction failed (non-fatal):', e);
  }
}
