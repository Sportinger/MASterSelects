import type { ScrubbingCache } from '../texture/ScrubbingCache';

function isFirefoxBrowser(): boolean {
  return typeof navigator !== 'undefined' && /Firefox\//.test(navigator.userAgent);
}

export function getCopiedHtmlVideoPreviewFrame(
  video: HTMLVideoElement,
  scrubbingCache: ScrubbingCache | null,
  targetTime?: number,
  lookupOwnerId?: string,
  captureOwnerId?: string
): { view: GPUTextureView; width: number; height: number; mediaTime?: number } | null {
  if (!isFirefoxBrowser() || !scrubbingCache) {
    return null;
  }

  if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
    return null;
  }

  const safeTargetTime = targetTime ?? video.currentTime;
  const previousFrame = scrubbingCache.getLastFrameNearTime(video, safeTargetTime, 0.35, lookupOwnerId);

  // Firefox can intermittently sample imported HTML video textures as black
  // during playback. Copying into a persistent texture is slower but stable.
  const captured = scrubbingCache.captureVideoFrame(video, captureOwnerId);
  if (captured) {
    return scrubbingCache.getLastFrameNearTime(video, safeTargetTime, 0.35, lookupOwnerId);
  }

  return previousFrame;
}
