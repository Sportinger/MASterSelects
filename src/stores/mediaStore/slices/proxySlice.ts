// Proxy generation slice

import type { MediaFile, MediaSliceCreator, ProxyStatus } from '../types';
import { PROXY_FPS } from '../constants';
import { projectFileService } from '../../../services/projectFileService';
import { useTimelineStore } from '../../timeline';
import { Logger } from '../../../services/logger';

const log = Logger.create('Proxy');

// Track active generations for cancellation
const activeProxyGenerations = new Map<string, { cancelled: boolean }>();

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
    log.info(`setProxyEnabled called: ${enabled}`);
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
        f.proxyStatus !== 'ready' &&
        f.proxyStatus !== 'generating' &&
        f.id !== currentlyGeneratingProxyId
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

    // Check if proxy video already exists
    const storageKey = mediaFile.fileHash || mediaFileId;
    const hasVideo = await projectFileService.hasProxyVideo(storageKey);
    if (hasVideo) {
      log.debug('Proxy video already exists:', mediaFile.name);

      // Load it and create blob URL
      const proxyFile = await projectFileService.getProxyVideo(storageKey);
      if (proxyFile) {
        const proxyVideoUrl = URL.createObjectURL(proxyFile);

        // Load proxy player
        const { loadProxyPlayer } = await import('../../../services/proxyPlayer');
        await loadProxyPlayer(mediaFileId, new Blob([await proxyFile.arrayBuffer()], { type: 'video/mp4' }));

        set((s) => ({
          files: s.files.map((f) =>
            f.id === mediaFileId
              ? { ...f, proxyStatus: 'ready' as ProxyStatus, proxyProgress: 100, proxyFps: PROXY_FPS, proxyVideoUrl }
              : f
          ),
        }));
        return;
      }

      // Fallback: check for legacy WebP frames
      const existingCount = await projectFileService.getProxyFrameCount(storageKey);
      if (existingCount > 0) {
        log.debug('Legacy WebP proxy found:', mediaFile.name);
        set((s) => ({
          files: s.files.map((f) =>
            f.id === mediaFileId
              ? { ...f, proxyStatus: 'ready' as ProxyStatus, proxyProgress: 100, proxyFrameCount: existingCount }
              : f
          ),
        }));
        return;
      }
    }

    // Set up cancellation
    const controller = { cancelled: false };
    activeProxyGenerations.set(mediaFileId, controller);

    set({ currentlyGeneratingProxyId: mediaFileId });
    set((state) => ({
      files: state.files.map((f) =>
        f.id === mediaFileId ? { ...f, proxyStatus: 'generating' as ProxyStatus, proxyProgress: 0, proxyFps: PROXY_FPS } : f
      ),
    }));

    const updateProgress = (progress: number) => {
      set((state) => ({
        files: state.files.map((f) =>
          f.id === mediaFileId ? { ...f, proxyProgress: progress } : f
        ),
      }));
    };

    try {
      const result = await generateVideoProxy(
        mediaFile,
        storageKey,
        controller,
        updateProgress
      );

      if (result && !controller.cancelled) {
        // Save proxy MP4 to project storage
        await projectFileService.saveProxyVideo(storageKey, result.blob);

        // Create blob URL and load proxy player
        const proxyVideoUrl = URL.createObjectURL(result.blob);
        const { loadProxyPlayer } = await import('../../../services/proxyPlayer');
        await loadProxyPlayer(mediaFileId, result.blob);

        set((s) => ({
          files: s.files.map((f) =>
            f.id === mediaFileId
              ? {
                  ...f,
                  proxyStatus: 'ready' as ProxyStatus,
                  proxyProgress: 100,
                  proxyFrameCount: result.frameCount,
                  proxyFps: result.fps,
                  proxyVideoUrl,
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

    const { currentlyGeneratingProxyId } = get();
    if (currentlyGeneratingProxyId === mediaFileId) {
      set((state) => ({
        currentlyGeneratingProxyId: null,
        files: state.files.map((f) =>
          f.id === mediaFileId
            ? { ...f, proxyStatus: 'none' as ProxyStatus, proxyProgress: 0 }
            : f
        ),
      }));
    }
  },
});

async function generateVideoProxy(
  mediaFile: MediaFile,
  _storageKey: string,
  controller: { cancelled: boolean },
  updateProgress: (progress: number) => void
): Promise<{ blob: Blob; frameCount: number; fps: number } | null> {
  const { getProxyGenerator } = await import('../../../services/proxyGenerator');
  const generator = getProxyGenerator();

  return generator.generate(
    mediaFile.file!,
    updateProgress,
    () => controller.cancelled,
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
