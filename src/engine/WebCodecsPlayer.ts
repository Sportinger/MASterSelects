// WebCodecs-based video player for hardware-accelerated decoding
// Bypasses browser VAAPI issues by using WebCodecs API directly
// With timeout fallback for problematic MP4 files

import * as MP4BoxModule from 'mp4box';
const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

// MP4Box types
interface MP4ArrayBuffer extends ArrayBuffer {
  fileStart: number;
}

interface Sample {
  number: number;
  track_id: number;
  data: ArrayBuffer;
  size: number;
  cts: number;
  dts: number;
  duration: number;
  is_sync: boolean;
  timescale: number;
}

interface MP4VideoTrack {
  id: number;
  codec: string;
  duration: number;
  timescale: number;
  nb_samples: number;
  video: { width: number; height: number };
}

interface MP4File {
  onReady: (info: { videoTracks: MP4VideoTrack[] }) => void;
  onSamples: (trackId: number, ref: any, samples: Sample[]) => void;
  onError: (error: string) => void;
  appendBuffer: (buffer: MP4ArrayBuffer) => number;
  start: () => void;
  flush: () => void;
  setExtractionOptions: (trackId: number, user: any, options: { nbSamples: number }) => void;
}

export interface WebCodecsPlayerOptions {
  loop?: boolean;
  onFrame?: (frame: VideoFrame) => void;
  onReady?: (width: number, height: number) => void;
  onError?: (error: Error) => void;
  // Use simple VideoFrame extraction from HTMLVideoElement instead of MP4Box demuxing
  useSimpleMode?: boolean;
  // Use MediaStreamTrackProcessor for VideoFrame extraction (best performance)
  useStreamMode?: boolean;
}

export class WebCodecsPlayer {
  private mp4File: MP4File | null = null;
  private decoder: VideoDecoder | null = null;
  private currentFrame: VideoFrame | null = null;
  private samples: Sample[] = [];
  private sampleIndex = 0;
  private _isPlaying = false;
  private loop: boolean;
  private frameRate = 30;
  private frameInterval = 1000 / 30;
  private lastFrameTime = 0;
  private animationId: number | null = null;
  private videoTrack: MP4VideoTrack | null = null;
  private codecConfig: VideoDecoderConfig | null = null;

  // Simple mode (VideoFrame from HTMLVideoElement)
  private useSimpleMode = false;
  private videoElement: HTMLVideoElement | null = null;
  private videoFrameCallbackId: number | null = null;

  public width = 0;
  public height = 0;
  public ready = false;

  private onFrame?: (frame: VideoFrame) => void;
  private onReady?: (width: number, height: number) => void;
  private onError?: (error: Error) => void;

  // Stream mode (MediaStreamTrackProcessor)
  // Note: useStreamMode is set but only used as a flag for cleanup/destroy
  private _useStreamMode = false;
  private streamReader: ReadableStreamDefaultReader<VideoFrame> | null = null;
  private streamActive = false;

  constructor(options: WebCodecsPlayerOptions = {}) {
    this.loop = options.loop ?? true;
    this.onFrame = options.onFrame;
    this.onReady = options.onReady;
    this.onError = options.onError;
    this.useSimpleMode = options.useSimpleMode ?? false;
    this._useStreamMode = options.useStreamMode ?? false;
  }

  // Stream mode: Use captureStream + MediaStreamTrackProcessor for best performance
  // This gives us VideoFrames without blocking the main thread
  async attachWithStream(video: HTMLVideoElement): Promise<void> {
    if (!('MediaStreamTrackProcessor' in window)) {
      throw new Error('MediaStreamTrackProcessor not supported');
    }

    this._useStreamMode = true;
    this.isAttachedToExternal = true;
    this.videoElement = video;
    this.width = video.videoWidth;
    this.height = video.videoHeight;

    // Capture stream from video
    // Note: captureStream is not in the standard HTMLVideoElement type but exists in browsers
    const stream = (video as HTMLVideoElement & { captureStream(): MediaStream }).captureStream();
    const videoTrack = stream.getVideoTracks()[0];

    if (!videoTrack) {
      throw new Error('No video track in stream');
    }

    // Create processor to get VideoFrames
    const processor = new (window as any).MediaStreamTrackProcessor({ track: videoTrack });
    this.streamReader = processor.readable.getReader();

    this.ready = true;
    console.log(`[WebCodecs Stream] Attached: ${this.width}x${this.height}`);

    // Start reading frames
    this.startStreamCapture();

    this.onReady?.(this.width, this.height);
  }

  private async startStreamCapture(): Promise<void> {
    if (!this.streamReader || this.streamActive) return;

    this.streamActive = true;
    console.log('[WebCodecs Stream] Starting frame capture');

    try {
      while (this.streamActive) {
        const { value: frame, done } = await this.streamReader.read();

        if (done) {
          console.log('[WebCodecs Stream] Stream ended');
          break;
        }

        if (frame) {
          // Close previous frame
          if (this.currentFrame) {
            this.currentFrame.close();
          }
          this.currentFrame = frame;
          this.onFrame?.(frame);
        }
      }
    } catch (e) {
      console.warn('[WebCodecs Stream] Error reading frames:', e);
    }

    this.streamActive = false;
  }

  private stopStreamCapture(): void {
    this.streamActive = false;
    if (this.streamReader) {
      this.streamReader.cancel().catch(() => {});
      this.streamReader = null;
    }
  }

  async loadFile(file: File): Promise<void> {
    // Check VideoFrame support (needed for both modes)
    if (!('VideoFrame' in window)) {
      throw new Error('VideoFrame API not supported in this browser');
    }

    // Simple mode: use HTMLVideoElement + VideoFrame (no MP4Box parsing needed)
    if (this.useSimpleMode) {
      await this.loadFileSimple(file);
      return;
    }

    // Full mode: use MP4Box + VideoDecoder
    if (!('VideoDecoder' in window)) {
      throw new Error('WebCodecs VideoDecoder not supported in this browser');
    }

    const arrayBuffer = await file.arrayBuffer();
    await this.loadArrayBuffer(arrayBuffer);
  }

  // Track if we're attached to an external video (Timeline's video element)
  private isAttachedToExternal = false;
  private boundOnPlay: (() => void) | null = null;
  private boundOnPause: (() => void) | null = null;
  private boundOnSeeked: (() => void) | null = null;

  // Use an existing video element instead of creating one (for timeline integration)
  attachToVideoElement(video: HTMLVideoElement): void {
    if (!('VideoFrame' in window)) {
      throw new Error('VideoFrame API not supported');
    }

    this.useSimpleMode = true;
    this.isAttachedToExternal = true;
    this.videoElement = video;
    this.width = video.videoWidth;
    this.height = video.videoHeight;
    this.ready = true;

    console.log(`[WebCodecs Simple] Attached to existing video: ${this.width}x${this.height}`);

    // Listen to video element events - Timeline controls the video, we just capture frames
    this.boundOnPlay = () => {
      if (this._isPlaying) return; // Already playing
      console.log('[WebCodecs Simple] Video play event - starting frame capture');
      this._isPlaying = true;
      this.startSimpleFrameCapture();
    };
    this.boundOnPause = () => {
      if (!this._isPlaying) return; // Already paused
      console.log('[WebCodecs Simple] Video pause event');
      this._isPlaying = false;
      this.stopSimpleFrameCapture();
      // Capture the paused frame
      this.captureCurrentFrame();
    };
    this.boundOnSeeked = () => {
      // Only capture on seek if not playing (playing captures continuously)
      if (!this._isPlaying) {
        this.captureCurrentFrame();
      }
    };
    // No timeupdate listener - requestVideoFrameCallback is more efficient

    video.addEventListener('play', this.boundOnPlay);
    video.addEventListener('pause', this.boundOnPause);
    video.addEventListener('seeked', this.boundOnSeeked);

    // Capture initial frame
    if (video.readyState >= 2) {
      this.captureCurrentFrame();
    }

    // If video is already playing, start capture
    if (!video.paused) {
      this._isPlaying = true;
      this.startSimpleFrameCapture();
    }
  }

  // Simple mode: Create VideoFrames directly from HTMLVideoElement
  private async loadFileSimple(file: File): Promise<void> {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.muted = true;
      video.playsInline = true;
      video.preload = 'auto';
      video.loop = this.loop;

      const timeout = setTimeout(() => {
        reject(new Error('Video load timeout'));
      }, 10000);

      video.onloadedmetadata = () => {
        this.width = video.videoWidth;
        this.height = video.videoHeight;
        this.videoElement = video;

        // Estimate frame rate (assume 30fps if unknown)
        this.frameRate = 30;
        this.frameInterval = 1000 / this.frameRate;

        console.log(`[WebCodecs Simple] Video loaded: ${this.width}x${this.height}`);
      };

      video.oncanplay = () => {
        clearTimeout(timeout);
        this.ready = true;

        // Create initial frame
        this.captureCurrentFrame();

        console.log(`[WebCodecs Simple] READY: ${this.width}x${this.height}`);
        this.onReady?.(this.width, this.height);
        resolve();
      };

      video.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Failed to load video'));
      };

      video.load();
    });
  }

  // Capture current video frame as VideoFrame
  private captureCurrentFrame(): void {
    if (!this.videoElement || this.videoElement.readyState < 2) return;

    // Close previous frame
    if (this.currentFrame) {
      this.currentFrame.close();
    }

    // Create new VideoFrame from video element
    try {
      this.currentFrame = new VideoFrame(this.videoElement, {
        timestamp: this.videoElement.currentTime * 1_000_000,
      });
      this.onFrame?.(this.currentFrame);
    } catch (e) {
      // Ignore frame capture errors (can happen during seek)
    }
  }

  async loadArrayBuffer(buffer: ArrayBuffer): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`[WebCodecs] Parsing MP4 (${(buffer.byteLength / 1024 / 1024).toFixed(1)}MB)...`);

      // Timeout if mp4box fails to parse (e.g., due to unusual metadata boxes)
      // Increased to 10 seconds for larger files
      const timeout = setTimeout(() => {
        reject(new Error('MP4 parsing timeout - file may have unsupported metadata'));
      }, 10000);

      this.mp4File = MP4Box.createFile();
      const mp4File = this.mp4File;

      mp4File.onReady = (info) => {
        console.log(`[WebCodecs] MP4 onReady: ${info.videoTracks.length} video tracks`);
        // Don't clear timeout here - wait for onSamples to actually deliver frames
        const videoTrack = info.videoTracks[0];
        if (!videoTrack) {
          reject(new Error('No video track found in file'));
          return;
        }

        this.videoTrack = videoTrack;
        this.width = videoTrack.video.width;
        this.height = videoTrack.video.height;
        this.frameRate = videoTrack.nb_samples / (videoTrack.duration / videoTrack.timescale);
        this.frameInterval = 1000 / this.frameRate;

        // Build codec string
        const codec = this.getCodecString(videoTrack);

        this.codecConfig = {
          codec,
          codedWidth: videoTrack.video.width,
          codedHeight: videoTrack.video.height,
          hardwareAcceleration: 'prefer-hardware',
          optimizeForLatency: true,
        };

        // Check if codec is supported
        VideoDecoder.isConfigSupported(this.codecConfig).then((support) => {
          if (!support.supported) {
            reject(new Error(`Codec ${codec} not supported`));
            return;
          }

          console.log(`[WebCodecs] Codec ${codec} supported, config:`, support.config);
          this.initDecoder();

          // Set extraction options and start
          mp4File.setExtractionOptions(videoTrack.id, null, {
            nbSamples: Infinity,
          });
          mp4File.start();
        });
      };

      mp4File.onSamples = (_trackId, _ref, samples) => {
        this.samples.push(...samples);

        // Signal ready after first batch of samples
        if (!this.ready && this.samples.length > 0) {
          this.ready = true;
          clearTimeout(timeout);
          console.log(`[WebCodecs] READY: ${this.width}x${this.height} @ ${this.frameRate.toFixed(1)}fps, ${this.samples.length} samples`);

          // Decode first frame immediately so we have something to display
          this.decodeFirstFrame();

          this.onReady?.(this.width, this.height);
          resolve();
        }
      };

      mp4File.onError = (e) => {
        clearTimeout(timeout);
        const error = new Error(`MP4 parsing error: ${e}`);
        this.onError?.(error);
        reject(error);
      };

      // Feed the buffer to mp4box
      const mp4Buffer = buffer as MP4ArrayBuffer;
      mp4Buffer.fileStart = 0;
      try {
        const appendedBytes = mp4File.appendBuffer(mp4Buffer);
        console.log(`[WebCodecs] Appended ${appendedBytes} bytes to MP4Box`);
        mp4File.flush();
        console.log(`[WebCodecs] Flushed MP4Box, waiting for callbacks...`);
      } catch (e) {
        clearTimeout(timeout);
        reject(new Error(`MP4Box appendBuffer failed: ${e}`));
      }
    });
  }

  private getCodecString(track: MP4VideoTrack): string {
    const dominated = track.codec;

    // Handle common codecs
    if (dominated.startsWith('avc1') || dominated.startsWith('avc3')) {
      // H.264/AVC
      return dominated;
    } else if (dominated.startsWith('hvc1') || dominated.startsWith('hev1')) {
      // H.265/HEVC
      return dominated;
    } else if (dominated.startsWith('vp09')) {
      // VP9
      return dominated;
    } else if (dominated.startsWith('av01')) {
      // AV1
      return dominated;
    }

    return dominated;
  }

  private initDecoder(): void {
    if (!this.codecConfig) return;

    this.decoder = new VideoDecoder({
      output: (frame) => {
        // Close previous frame to prevent memory leak
        if (this.currentFrame) {
          this.currentFrame.close();
        }
        this.currentFrame = frame;
        this.onFrame?.(frame);
      },
      error: (e) => {
        console.error('VideoDecoder error:', e);
        this.onError?.(new Error(`Decoder error: ${e.message}`));
      },
    });

    this.decoder.configure(this.codecConfig);
  }

  play(): void {
    if (this._isPlaying || !this.ready) return;
    this._isPlaying = true;

    if (this.useSimpleMode && this.videoElement) {
      // If attached to external video, don't control it - just ensure frame capture is running
      // Timeline controls the video element, we get notified via events
      if (!this.isAttachedToExternal) {
        this.videoElement.play().catch(() => {});
      }
      this.startSimpleFrameCapture();
    } else {
      this.lastFrameTime = performance.now();
      this.scheduleNextFrame();
    }
  }

  pause(): void {
    this._isPlaying = false;

    if (this.useSimpleMode && this.videoElement) {
      // If attached to external video, don't control it - Timeline controls it
      if (!this.isAttachedToExternal) {
        this.videoElement.pause();
      }
      this.stopSimpleFrameCapture();
    } else {
      if (this.animationId !== null) {
        cancelAnimationFrame(this.animationId);
        this.animationId = null;
      }
    }
  }

  stop(): void {
    this.pause();

    if (this.useSimpleMode && this.videoElement) {
      // If attached to external video, don't control it
      if (!this.isAttachedToExternal) {
        this.videoElement.currentTime = 0;
      }
    } else {
      this.sampleIndex = 0;
    }

    if (this.currentFrame) {
      this.currentFrame.close();
      this.currentFrame = null;
    }
  }

  // Simple mode frame capture using requestVideoFrameCallback
  private startSimpleFrameCapture(): void {
    if (!this.videoElement || !('requestVideoFrameCallback' in this.videoElement)) {
      // Fallback to requestAnimationFrame
      this.startSimpleFrameCaptureRAF();
      return;
    }

    const captureFrame = () => {
      if (!this._isPlaying || !this.videoElement) return;

      this.captureCurrentFrame();

      this.videoFrameCallbackId = this.videoElement.requestVideoFrameCallback(captureFrame);
    };

    this.videoFrameCallbackId = this.videoElement.requestVideoFrameCallback(captureFrame);
  }

  private startSimpleFrameCaptureRAF(): void {
    const captureFrame = () => {
      if (!this._isPlaying) return;

      this.captureCurrentFrame();

      this.animationId = requestAnimationFrame(captureFrame);
    };

    this.animationId = requestAnimationFrame(captureFrame);
  }

  private stopSimpleFrameCapture(): void {
    if (this.videoFrameCallbackId !== null && this.videoElement && 'cancelVideoFrameCallback' in this.videoElement) {
      (this.videoElement as any).cancelVideoFrameCallback(this.videoFrameCallbackId);
      this.videoFrameCallbackId = null;
    }
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  private scheduleNextFrame(): void {
    if (!this._isPlaying) return;

    this.animationId = requestAnimationFrame((now) => {
      const elapsed = now - this.lastFrameTime;

      if (elapsed >= this.frameInterval) {
        this.decodeNextFrame();
        this.lastFrameTime = now - (elapsed % this.frameInterval);
      }

      this.scheduleNextFrame();
    });
  }

  private decodeFirstFrame(): void {
    if (!this.decoder || this.samples.length === 0) return;

    // Decode the first keyframe to have an initial frame available
    const firstSample = this.samples[0];
    if (!firstSample.is_sync) return; // First frame should be a keyframe

    const chunk = new EncodedVideoChunk({
      type: 'key',
      timestamp: (firstSample.cts * 1_000_000) / firstSample.timescale,
      duration: (firstSample.duration * 1_000_000) / firstSample.timescale,
      data: firstSample.data,
    });

    try {
      this.decoder.decode(chunk);
      this.sampleIndex = 1;
    } catch {
      // Ignore decode errors on first frame
    }
  }

  private decodeNextFrame(): void {
    if (!this.decoder || this.samples.length === 0) return;

    // Get next sample
    if (this.sampleIndex >= this.samples.length) {
      if (this.loop) {
        this.sampleIndex = 0;
        // Reset decoder for loop
        this.decoder.reset();
        this.decoder.configure(this.codecConfig!);
      } else {
        this.pause();
        return;
      }
    }

    const sample = this.samples[this.sampleIndex];
    this.sampleIndex++;

    // Create EncodedVideoChunk from sample
    const chunk = new EncodedVideoChunk({
      type: sample.is_sync ? 'key' : 'delta',
      timestamp: (sample.cts * 1_000_000) / sample.timescale, // Convert to microseconds
      duration: (sample.duration * 1_000_000) / sample.timescale,
      data: sample.data,
    });

    // Decode
    try {
      this.decoder.decode(chunk);
    } catch {
      // Silently skip decode errors - can happen during seek or loop
    }
  }

  // Get current frame for WebGPU rendering
  getCurrentFrame(): VideoFrame | null {
    return this.currentFrame;
  }

  // Check if there's a valid frame available
  hasFrame(): boolean {
    return this.currentFrame !== null;
  }

  seek(timeSeconds: number): void {
    // Simple mode: direct seek on video element
    if (this.useSimpleMode && this.videoElement) {
      this.videoElement.currentTime = timeSeconds;
      // Capture frame immediately and after seek completes
      this.captureCurrentFrame();

      // Also capture when seeked event fires
      const onSeeked = () => {
        this.captureCurrentFrame();
        this.videoElement?.removeEventListener('seeked', onSeeked);
      };
      this.videoElement.addEventListener('seeked', onSeeked);
      return;
    }

    // Full mode: decode from keyframe
    if (!this.videoTrack || this.samples.length === 0 || !this.decoder) return;

    const targetTime = timeSeconds * this.videoTrack.timescale;

    // Find the nearest keyframe before the target time
    let keyframeIndex = 0;
    let targetIndex = 0;
    for (let i = 0; i < this.samples.length; i++) {
      if (this.samples[i].cts > targetTime) break;
      targetIndex = i;
      if (this.samples[i].is_sync) {
        keyframeIndex = i;
      }
    }

    // Reset decoder
    this.decoder.reset();
    this.decoder.configure(this.codecConfig!);

    // Decode from keyframe up to target frame to get correct frame
    for (let i = keyframeIndex; i <= targetIndex; i++) {
      const sample = this.samples[i];
      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts * 1_000_000) / sample.timescale,
        duration: (sample.duration * 1_000_000) / sample.timescale,
        data: sample.data,
      });
      try {
        this.decoder.decode(chunk);
      } catch {
        // Skip decode errors
      }
    }

    this.sampleIndex = targetIndex + 1;
  }

  get duration(): number {
    if (this.useSimpleMode && this.videoElement) {
      return this.videoElement.duration || 0;
    }
    if (!this.videoTrack) return 0;
    return this.videoTrack.duration / this.videoTrack.timescale;
  }

  get isPlaying(): boolean {
    return this._isPlaying;
  }

  get currentTime(): number {
    if (this.useSimpleMode && this.videoElement) {
      return this.videoElement.currentTime;
    }
    if (!this.videoTrack || this.samples.length === 0 || this.sampleIndex === 0) return 0;
    const sample = this.samples[Math.min(this.sampleIndex - 1, this.samples.length - 1)];
    return sample.cts / sample.timescale;
  }

  destroy(): void {
    this.stop();

    // Stream mode cleanup
    this.stopStreamCapture();

    // Simple mode cleanup
    if (this.videoElement) {
      // Remove event listeners if attached to external video
      if (this.isAttachedToExternal) {
        if (this.boundOnPlay) this.videoElement.removeEventListener('play', this.boundOnPlay);
        if (this.boundOnPause) this.videoElement.removeEventListener('pause', this.boundOnPause);
        if (this.boundOnSeeked) this.videoElement.removeEventListener('seeked', this.boundOnSeeked);
        this.boundOnPlay = null;
        this.boundOnPause = null;
        this.boundOnSeeked = null;
        // Don't clear src or pause - Timeline owns the video element
      } else {
        this.videoElement.pause();
        this.videoElement.src = '';
      }
      this.videoElement = null;
    }

    this.isAttachedToExternal = false;
    this._useStreamMode = false;

    // Full mode cleanup
    if (this.decoder) {
      this.decoder.close();
      this.decoder = null;
    }

    if (this.currentFrame) {
      this.currentFrame.close();
      this.currentFrame = null;
    }

    this.mp4File = null;
    this.samples = [];
    this.ready = false;
  }
}
