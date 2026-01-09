// Video frame capture and tracking

export class VideoFrameManager {
  // Track video frame readiness - only import texture when new frame is available
  private videoFrameReady: Map<HTMLVideoElement, boolean> = new Map();
  private videoLastTime: Map<HTMLVideoElement, number> = new Map();
  private videoCallbackActive: Map<HTMLVideoElement, boolean> = new Map();

  // Track active video for requestVideoFrameCallback
  private activeVideo: HTMLVideoElement | null = null;
  private videoFrameCallbackId: number | null = null;

  constructor() {}

  // Register a video to track frame readiness
  registerVideo(video: HTMLVideoElement): void {
    // Already fully registered
    if (this.videoFrameReady.has(video) && this.videoCallbackActive.get(video)) {
      // Re-register callback if video is playing but callback isn't active
      if (!video.paused && !this.videoCallbackActive.get(video)) {
        this.startVideoFrameCallback(video);
      }
      return;
    }

    this.videoFrameReady.set(video, true); // First frame is ready
    this.videoLastTime.set(video, -1);
    this.videoCallbackActive.set(video, false);

    // Start frame callback if video is playing
    if (!video.paused) {
      this.startVideoFrameCallback(video);
    }

    // Listen for play event to restart callback
    video.addEventListener('play', () => {
      this.startVideoFrameCallback(video);
    });
  }

  private startVideoFrameCallback(video: HTMLVideoElement): void {
    if (!('requestVideoFrameCallback' in video)) return;
    if (this.videoCallbackActive.get(video)) return;

    this.videoCallbackActive.set(video, true);

    const onFrame = () => {
      this.videoFrameReady.set(video, true);
      if (!video.paused) {
        (video as any).requestVideoFrameCallback(onFrame);
      } else {
        this.videoCallbackActive.set(video, false);
      }
    };
    (video as any).requestVideoFrameCallback(onFrame);
  }

  // Check if a new frame is available (for non-rVFC browsers, check currentTime)
  hasNewFrame(video: HTMLVideoElement): boolean {
    const lastTime = this.videoLastTime.get(video) ?? -1;
    const currentTime = video.currentTime;

    // When video is paused, no new frames are being decoded
    // Return true only if time changed (e.g., user seeked) or first frame
    if (video.paused) {
      if (lastTime === -1 || Math.abs(currentTime - lastTime) > 0.001) {
        this.videoLastTime.set(video, currentTime);
        return true;
      }
      return false; // Same frame, use cache
    }

    // Video is playing - use requestVideoFrameCallback if available
    if ('requestVideoFrameCallback' in video) {
      const ready = this.videoFrameReady.get(video) ?? false;
      if (ready) {
        this.videoFrameReady.set(video, false);
        this.videoLastTime.set(video, currentTime);
        return true;
      }
      return false;
    }

    // Fallback: check if time changed (at least 1ms difference for ~1000fps max)
    if (Math.abs(currentTime - lastTime) > 0.001) {
      this.videoLastTime.set(video, currentTime);
      return true;
    }
    return false;
  }

  // Set the active video to sync rendering with its frame delivery
  setActiveVideo(video: HTMLVideoElement | null): void {
    // Clean up old callback
    if (this.activeVideo && this.videoFrameCallbackId !== null) {
      if ('cancelVideoFrameCallback' in this.activeVideo) {
        (this.activeVideo as any).cancelVideoFrameCallback(this.videoFrameCallbackId);
      }
      this.videoFrameCallbackId = null;
    }
    this.activeVideo = video;
  }

  // Cleanup resources for a video that's no longer used
  cleanupVideo(video: HTMLVideoElement): void {
    this.videoFrameReady.delete(video);
    this.videoLastTime.delete(video);
    this.videoCallbackActive.delete(video);
  }

  // Clear all video tracking
  clearAll(): void {
    this.videoFrameReady.clear();
    this.videoLastTime.clear();
    this.videoCallbackActive.clear();
    this.activeVideo = null;
    this.videoFrameCallbackId = null;
  }

  destroy(): void {
    this.clearAll();
  }
}
