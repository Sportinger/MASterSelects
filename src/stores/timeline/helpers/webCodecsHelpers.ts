// WebCodecs initialization helper - eliminates 4x duplication across clip loading
// Handles WebCodecsPlayer setup and video decoder warm-up

import { WebCodecsPlayer } from '../../../engine/WebCodecsPlayer';
import { engine } from '../../../engine/WebGPUEngine';
import { flags } from '../../../engine/featureFlags';
import { Logger } from '../../../services/logger';

const log = Logger.create('WebCodecsHelpers');

async function waitForFullWebCodecsReady(
  webCodecsPlayer: WebCodecsPlayer,
  fileName: string,
  timeoutMs = 2000
): Promise<void> {
  if (webCodecsPlayer.ready) {
    return;
  }

  const startedAt = performance.now();

  await new Promise<void>((resolve) => {
    const poll = () => {
      if (webCodecsPlayer.ready) {
        resolve();
        return;
      }

      if (performance.now() - startedAt >= timeoutMs) {
        log.warn('WebCodecs ready wait timed out', { file: fileName, timeoutMs });
        resolve();
        return;
      }

      setTimeout(poll, 16);
    };

    poll();
  });
}

/**
 * Check if WebCodecs API is available in the browser.
 */
export function hasWebCodecsSupport(): boolean {
  return 'VideoDecoder' in window && 'VideoFrame' in window;
}

/**
 * Initialize WebCodecsPlayer for hardware-accelerated video decoding.
 * Returns null if WebCodecs is not available or initialization fails.
 *
 * When `flags.useFullWebCodecsPlayback` is active and a File is provided,
 * the player uses full mode (MP4Box + VideoDecoder API) instead of the
 * simple VideoFrame-wrapper around HTMLVideoElement.
 */
export async function initWebCodecsPlayer(
  video: HTMLVideoElement,
  fileName: string = 'video',
  file?: File
): Promise<WebCodecsPlayer | null> {
  if (!hasWebCodecsSupport()) {
    return null;
  }

  const useFullMode = flags.useFullWebCodecsPlayback && !!file;

  try {
    log.debug('Initializing WebCodecs', { file: fileName, fullMode: useFullMode });

    const webCodecsPlayer = new WebCodecsPlayer({
      loop: false,
      useSimpleMode: !useFullMode,
      onFrame: () => {
        engine.requestNewFrameRender();
      },
      onError: (error) => {
        log.warn('WebCodecs error', { error: error.message });
        engine.requestRender();
      },
    });

    if (useFullMode) {
      // Full mode: load file via MP4Box + VideoDecoder
      await webCodecsPlayer.loadFile(file);
      await waitForFullWebCodecsReady(webCodecsPlayer, fileName);
      // Still attach to video element for audio playback reference
      log.info('WebCodecs full mode ready', {
        file: fileName,
        ready: webCodecsPlayer.ready,
      });
    } else {
      // Simple mode: wrap HTMLVideoElement with VideoFrame
      webCodecsPlayer.attachToVideoElement(video);
      log.debug('WebCodecs simple mode ready', { file: fileName });
    }

    return webCodecsPlayer;
  } catch (err) {
    log.warn('WebCodecs init failed, using HTMLVideoElement', err);
    return null;
  }
}

/**
 * Warm up video decoder by forcing a frame decode.
 * This eliminates the "cold start" delay on first play.
 * Non-blocking - returns a promise that resolves when warm-up is complete.
 */
export function warmUpVideoDecoder(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    // Skip if video is already playing or has been decoded
    if (video.readyState >= 3) { // HAVE_FUTURE_DATA or HAVE_ENOUGH_DATA
      resolve();
      return;
    }

    // Use requestVideoFrameCallback if available (modern browsers)
    // This efficiently waits for the decoder to produce a frame
    if ('requestVideoFrameCallback' in video) {
      const warmUp = () => {
        video.currentTime = 0.001; // Seek to first frame (not exactly 0 to trigger decode)
        (video as any).requestVideoFrameCallback(() => {
          // Decoder has now processed at least one frame
          video.pause();
          resolve();
        });
        // Force decode by playing briefly
        video.play().catch(() => resolve());
      };

      if (video.readyState >= 1) { // HAVE_METADATA
        warmUp();
      } else {
        video.addEventListener('loadedmetadata', warmUp, { once: true });
      }
    } else {
      // Fallback: wait for canplay event which indicates decoder is ready
      // Cast needed because TypeScript narrows incorrectly after requestVideoFrameCallback check
      const videoEl = video as HTMLVideoElement;
      if (videoEl.readyState >= 2) { // HAVE_CURRENT_DATA
        resolve();
        return;
      }
      videoEl.addEventListener('canplay', () => resolve(), { once: true });
      // Trigger buffer by seeking
      videoEl.currentTime = 0.001;
    }

    // Timeout fallback (don't block forever)
    setTimeout(resolve, 500);
  });
}

/**
 * Create a video element with standard settings for timeline clips.
 */
export function createVideoElement(file: File): HTMLVideoElement {
  const video = document.createElement('video');
  video.src = URL.createObjectURL(file);
  video.preload = 'metadata';
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  return video;
}

/**
 * Create an audio element with standard settings for timeline clips.
 */
export function createAudioElement(file: File): HTMLAudioElement {
  const audio = document.createElement('audio');
  audio.src = URL.createObjectURL(file);
  audio.preload = 'auto';
  return audio;
}

/**
 * Wait for video metadata to load.
 * Includes timeout for large files where moov atom is at end of file (camera MOV/MP4).
 */
export function waitForVideoMetadata(video: HTMLVideoElement, timeout = 8000): Promise<void> {
  return new Promise((resolve) => {
    if (video.readyState >= 1) {
      resolve();
      return;
    }
    const timeoutId = setTimeout(() => {
      log.warn('Video metadata load timeout', { src: video.src?.substring(0, 50) });
      resolve();
    }, timeout);
    video.onloadedmetadata = () => {
      clearTimeout(timeoutId);
      resolve();
    };
    video.onerror = () => {
      clearTimeout(timeoutId);
      resolve();
    };
  });
}

/**
 * Wait for video to be ready for playback (canplaythrough).
 */
export function waitForVideoReady(video: HTMLVideoElement, timeout = 2000): Promise<void> {
  return new Promise((resolve) => {
    if (video.readyState >= 4) { // HAVE_ENOUGH_DATA
      resolve();
      return;
    }
    const timeoutId = setTimeout(resolve, timeout);
    const handler = () => {
      clearTimeout(timeoutId);
      resolve();
    };
    video.addEventListener('canplaythrough', handler, { once: true });
  });
}
