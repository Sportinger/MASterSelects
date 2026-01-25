// WebCodecs initialization helper - eliminates 4x duplication across clip loading
// Handles WebCodecsPlayer setup and video decoder warm-up

import { WebCodecsPlayer } from '../../../engine/WebCodecsPlayer';
import { Logger } from '../../../services/logger';

const log = Logger.create('WebCodecsHelpers');

/**
 * Check if WebCodecs API is available in the browser.
 */
export function hasWebCodecsSupport(): boolean {
  return 'VideoDecoder' in window && 'VideoFrame' in window;
}

/**
 * Initialize WebCodecsPlayer for hardware-accelerated video decoding.
 * Returns null if WebCodecs is not available or initialization fails.
 */
export async function initWebCodecsPlayer(
  video: HTMLVideoElement,
  fileName: string = 'video'
): Promise<WebCodecsPlayer | null> {
  if (!hasWebCodecsSupport()) {
    return null;
  }

  try {
    log.debug('Initializing WebCodecs', { file: fileName });

    const webCodecsPlayer = new WebCodecsPlayer({
      loop: false,
      useSimpleMode: true,
      onError: (error) => {
        log.warn('WebCodecs error', { error: error.message });
      },
    });

    webCodecsPlayer.attachToVideoElement(video);
    log.debug('WebCodecs ready', { file: fileName });

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
 */
export function waitForVideoMetadata(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    if (video.readyState >= 1) {
      resolve();
      return;
    }
    video.onloadedmetadata = () => resolve();
    video.onerror = () => resolve();
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
