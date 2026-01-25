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
  private streamReader: ReadableStreamDefaultReader<VideoFrame> | null = null;
  private streamActive = false;

  // Sequential export mode - avoids decoder reset on each frame
  private isInExportMode = false;
  private frameResolve: (() => void) | null = null; // For waiting on decoded frames
  private decoderInitialized = false; // Flag to track decoder ready state
  private pendingDecodeFirstFrame = false; // Flag to defer first frame decode
  private loadResolve: (() => void) | null = null; // For waiting on decoder init in loadArrayBuffer

  constructor(options: WebCodecsPlayerOptions = {}) {
    this.loop = options.loop ?? true;
    this.onFrame = options.onFrame;
    this.onReady = options.onReady;
    this.onError = options.onError;
    this.useSimpleMode = options.useSimpleMode ?? false;
  }

  // Stream mode: Use captureStream + MediaStreamTrackProcessor for best performance
  // This gives us VideoFrames without blocking the main thread
  async attachWithStream(video: HTMLVideoElement): Promise<void> {
    if (!('MediaStreamTrackProcessor' in window)) {
      throw new Error('MediaStreamTrackProcessor not supported');
    }

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
      const mp4File = this.mp4File!;

      mp4File.onReady = (info) => {
        console.log(`[WebCodecs] MP4 onReady: ${info.videoTracks.length} video tracks`);
        // Don't clear timeout here - wait for onSamples to actually deliver frames
        const videoTrack = info.videoTracks[0];
        if (!videoTrack) {
          clearTimeout(timeout);
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

        // Extract codec-specific description (avcC for H.264, hvcC for H.265, etc.)
        // This is REQUIRED for AVC/HEVC to work properly
        let description: ArrayBuffer | undefined;

        // Get the track structure from mp4File to access codec config boxes
        try {
          const trak = (mp4File as any).getTrackById(videoTrack.id);
          if (trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]) {
            const entry = trak.mdia.minf.stbl.stsd.entries[0];

            // Try to extract codec-specific configuration
            const configBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
            if (configBox) {
              const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
              configBox.write(stream);
              // The write() includes the box header (8 bytes: size + type), we need to skip it
              description = stream.buffer.slice(8);
              console.log(`[WebCodecs] Extracted codec description: ${description!.byteLength} bytes from ${entry.avcC ? 'avcC' : entry.hvcC ? 'hvcC' : entry.vpcC ? 'vpcC' : 'av1C'}`);
            } else {
              console.warn('[WebCodecs] No codec config box found in sample entry:', Object.keys(entry));
            }
          }
        } catch (e) {
          console.warn('[WebCodecs] Failed to extract codec description:', e);
        }

        this.codecConfig = {
          codec,
          codedWidth: videoTrack.video.width,
          codedHeight: videoTrack.video.height,
          hardwareAcceleration: 'prefer-hardware',
          optimizeForLatency: true,
          description,
        };

        // Set extraction options and start BEFORE codec check (to not miss samples)
        mp4File.setExtractionOptions(videoTrack.id, null, {
          nbSamples: Infinity,
        });
        mp4File.start();
        console.log(`[WebCodecs] Extraction started for track ${videoTrack.id}`);

        // Check if codec is supported (async, but extraction already started)
        VideoDecoder.isConfigSupported(this.codecConfig).then((support) => {
          if (!support.supported) {
            clearTimeout(timeout);
            reject(new Error(`Codec ${codec} not supported`));
            return;
          }

          console.log(`[WebCodecs] Codec ${codec} supported, config:`, support.config);
          this.initDecoder();
        });
      };

      mp4File.onSamples = (_trackId, _ref, samples) => {
        this.samples.push(...samples);

        // Signal ready after first batch of samples AND decoder is initialized
        if (!this.ready && this.samples.length > 0 && this.decoderInitialized) {
          this.ready = true;
          clearTimeout(timeout);
          console.log(`[WebCodecs] READY: ${this.width}x${this.height} @ ${this.frameRate.toFixed(1)}fps, ${this.samples.length} samples`);

          this.decodeFirstFrame();
          this.onReady?.(this.width, this.height);
          resolve();
        } else if (!this.ready && this.samples.length > 0) {
          // Samples received but decoder not ready yet - store resolve for initDecoder
          this.pendingDecodeFirstFrame = true;
          this.loadResolve = () => {
            clearTimeout(timeout);
            resolve();
          };
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

        // Resolve any pending frame wait (for sequential export)
        if (this.frameResolve) {
          this.frameResolve();
          this.frameResolve = null;
        }
      },
      error: (e) => {
        console.error('VideoDecoder error:', e);
        this.onError?.(new Error(`Decoder error: ${e.message}`));
      },
    });

    this.decoder.configure(this.codecConfig);
    this.decoderInitialized = true;

    // Handle any deferred first frame decode and resolve loadArrayBuffer promise
    if (this.pendingDecodeFirstFrame && this.samples.length > 0) {
      this.pendingDecodeFirstFrame = false;
      this.ready = true;
      console.log(`[WebCodecs] READY (deferred): ${this.width}x${this.height} @ ${this.frameRate.toFixed(1)}fps, ${this.samples.length} samples`);

      this.decodeFirstFrame();
      this.onReady?.(this.width, this.height);

      // Resolve the loadArrayBuffer promise
      if (this.loadResolve) {
        this.loadResolve();
        this.loadResolve = null;
      }
    }
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

  /**
   * Async seek that waits for the frame to be decoded
   * Use this for export where we need guaranteed frame accuracy
   */
  async seekAsync(timeSeconds: number): Promise<void> {
    // Simple mode: seek video element and wait for frame
    if (this.useSimpleMode && this.videoElement) {
      return new Promise<void>((resolve) => {
        const video = this.videoElement!;
        let resolved = false;

        const doResolve = () => {
          if (resolved) return;
          resolved = true;
          this.captureCurrentFrame();
          resolve();
        };

        // Longer timeout for export - we need accurate frames
        const timeout = setTimeout(() => {
          if (!resolved) {
            console.warn('[WebCodecs] seekAsync timeout at', timeSeconds, 'readyState:', video.readyState);
            doResolve();
          }
        }, 2000);

        // Wait for video to have enough data (readyState >= 2 means HAVE_CURRENT_DATA)
        const waitForReady = (callback: () => void) => {
          if (video.readyState >= 2 && !video.seeking) {
            callback();
            return;
          }
          // Poll until ready or timeout
          let retries = 0;
          const maxRetries = 60; // 60 * 16ms ≈ 1 second
          const checkReady = () => {
            retries++;
            if (video.readyState >= 2 && !video.seeking) {
              callback();
            } else if (retries < maxRetries) {
              requestAnimationFrame(checkReady);
            } else {
              // Give up waiting for readyState, proceed anyway
              console.warn('[WebCodecs] waitForReady gave up after', retries, 'retries, readyState:', video.readyState);
              callback();
            }
          };
          requestAnimationFrame(checkReady);
        };

        const waitForFrame = () => {
          // First ensure video has data, then wait for frame callback
          waitForReady(() => {
            // Use requestVideoFrameCallback if available for precise frame timing
            if ('requestVideoFrameCallback' in video) {
              (video as any).requestVideoFrameCallback(() => {
                clearTimeout(timeout);
                doResolve();
              });
              // Also set a shorter backup timeout since rvfc may not fire when paused
              setTimeout(() => {
                if (!resolved && video.readyState >= 2) {
                  clearTimeout(timeout);
                  doResolve();
                }
              }, 100);
            } else {
              // Fallback: wait two animation frames
              requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                  clearTimeout(timeout);
                  doResolve();
                });
              });
            }
          });
        };

        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          waitForFrame();
        };

        if (Math.abs(video.currentTime - timeSeconds) < 0.01 && !video.seeking) {
          // Already at position, just wait for frame
          waitForFrame();
          return;
        }

        video.addEventListener('seeked', onSeeked);
        video.currentTime = timeSeconds;
      });
    }

    // Full mode: decode and flush
    if (!this.videoTrack || this.samples.length === 0 || !this.decoder) {
      return;
    }

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

    // Decode from keyframe up to target frame
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

    // Flush to ensure all frames are decoded
    await this.decoder.flush();

    this.sampleIndex = targetIndex + 1;
  }

  /**
   * Prepare for sequential export - seeks to start time and enters export mode.
   * In export mode, subsequent calls to decodeNextFrameForExport() will decode
   * frames sequentially without resetting the decoder (much faster).
   */
  async prepareForSequentialExport(startTimeSeconds: number): Promise<void> {
    // Simple mode doesn't benefit from sequential decoding - browser handles it
    if (this.useSimpleMode) {
      this.isInExportMode = true;
      return;
    }

    // Full mode: seek to start time and enter export mode
    if (!this.videoTrack || this.samples.length === 0 || !this.decoder) {
      return;
    }

    const targetTime = startTimeSeconds * this.videoTrack.timescale;

    // Add half-frame tolerance to handle floating point precision issues
    const halfFrame = (this.samples[0]?.duration ?? 1) / 2;
    const targetTimeWithTolerance = targetTime + halfFrame;

    // Find the nearest keyframe before the target time
    let keyframeIndex = 0;
    let targetIndex = 0;
    for (let i = 0; i < this.samples.length; i++) {
      if (this.samples[i].cts > targetTimeWithTolerance) break;
      targetIndex = i;
      if (this.samples[i].is_sync) {
        keyframeIndex = i;
      }
    }

    // Reset decoder once at the start
    this.decoder.reset();
    this.decoder.configure(this.codecConfig!);

    // Decode from keyframe up to target frame
    // Don't use flush() as it resets decoder state and requires keyframe afterwards
    for (let i = keyframeIndex; i <= targetIndex; i++) {
      const sample = this.samples[i];
      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts * 1_000_000) / sample.timescale,
        duration: (sample.duration * 1_000_000) / sample.timescale,
        data: sample.data,
      });

      // For the last frame, wait for it to arrive in output callback
      if (i === targetIndex) {
        const framePromise = new Promise<void>((resolve) => {
          this.frameResolve = resolve;
          setTimeout(() => {
            if (this.frameResolve === resolve) {
              this.frameResolve = null;
              resolve();
            }
          }, 200); // Longer timeout for initial seek
        });

        try {
          this.decoder.decode(chunk);
          await framePromise;
        } catch {
          // Skip decode errors
        }
      } else {
        try {
          this.decoder.decode(chunk);
        } catch {
          // Skip decode errors
        }
      }
    }

    this.sampleIndex = targetIndex + 1;
    this.isInExportMode = true;

    console.log(`[WebCodecs] Export mode: started at sample ${targetIndex}, decoded ${targetIndex - keyframeIndex + 1} frames from keyframe`);
  }

  /**
   * Decode the next frame sequentially for export.
   * Much faster than seekAsync because it doesn't reset the decoder.
   * Must call prepareForSequentialExport() first.
   */
  async decodeNextFrameForExport(): Promise<void> {
    // Simple mode: capture current frame (browser already decoded it)
    if (this.useSimpleMode && this.videoElement) {
      this.captureCurrentFrame();
      return;
    }

    // Full mode: decode just the next sample
    if (!this.decoder || this.samples.length === 0 || !this.isInExportMode) {
      return;
    }

    // Check if we need to decode more frames
    if (this.sampleIndex >= this.samples.length) {
      console.warn('[WebCodecs] Export: reached end of samples');
      return;
    }

    const sample = this.samples[this.sampleIndex];
    this.sampleIndex++;

    const chunk = new EncodedVideoChunk({
      type: sample.is_sync ? 'key' : 'delta',
      timestamp: (sample.cts * 1_000_000) / sample.timescale,
      duration: (sample.duration * 1_000_000) / sample.timescale,
      data: sample.data,
    });

    // Create promise that resolves when frame arrives in output callback
    let timedOut = false;
    const framePromise = new Promise<void>((resolve) => {
      this.frameResolve = resolve;
      // Timeout fallback in case frame never arrives
      setTimeout(() => {
        if (this.frameResolve === resolve) {
          timedOut = true;
          this.frameResolve = null;
          resolve();
        }
      }, 200); // 200ms timeout for H.264/HEVC decoding
    });

    try {
      this.decoder.decode(chunk);
      // Wait for frame to arrive in callback
      await framePromise;

      // If timeout triggered, flush decoder to force frame delivery
      if (timedOut && this.decoder.decodeQueueSize > 0) {
        console.warn('[WebCodecs] Frame decode timeout - flushing decoder');
        await this.decoder.flush();
      }
    } catch (e) {
      console.warn('[WebCodecs] Export decode error:', e);
    }
  }

  /**
   * Seek to a specific time during export.
   * Use this for non-sequential access (e.g., reversed clips, speed changes).
   * Falls back to full seek but stays in export mode.
   */
  async seekDuringExport(timeSeconds: number): Promise<void> {
    if (this.useSimpleMode && this.videoElement) {
      // Simple mode: use normal seek
      await this.seekAsync(timeSeconds);
      return;
    }

    if (!this.videoTrack || this.samples.length === 0 || !this.decoder) {
      return;
    }

    const targetTime = timeSeconds * this.videoTrack.timescale;

    // Add half-frame tolerance to handle floating point precision issues
    // e.g., at 30fps: 1/30 = 0.0333... but floating point may give us slightly less
    const halfFrame = (this.samples[0]?.duration ?? 1) / 2;
    const targetTimeWithTolerance = targetTime + halfFrame;

    // Find target sample index - use tolerance-adjusted time
    let targetIndex = 0;
    for (let i = 0; i < this.samples.length; i++) {
      if (this.samples[i].cts > targetTimeWithTolerance) break;
      targetIndex = i;
    }

    // Check if we can continue sequentially (target is within a few samples)
    // Allow +/- 3 samples to handle frame rate conversions (e.g., 25fps video at 30fps export)
    if (this.isInExportMode && targetIndex >= this.sampleIndex - 1 && targetIndex <= this.sampleIndex + 3) {
      // Decode frames sequentially until we reach or pass the target
      // sampleIndex points to the NEXT frame to decode, so we need to decode while sampleIndex <= targetIndex
      const beforeIndex = this.sampleIndex;
      while (this.sampleIndex <= targetIndex) {
        await this.decodeNextFrameForExport();
      }
      // Debug: log first few seeks to verify frame-accurate export
      if (targetIndex < 5) {
        console.log(`[WebCodecs] Export seek: time=${timeSeconds.toFixed(3)}s → sample ${targetIndex} (decoded ${this.sampleIndex - beforeIndex} new)`);
      }
      return;
    }

    // DEBUG: Log when we need to reset decoder
    console.warn(`[WebCodecs] DECODER RESET: targetIndex=${targetIndex}, sampleIndex=${this.sampleIndex}, diff=${targetIndex - this.sampleIndex}`);

    // Need to jump - find keyframe and decode from there
    let keyframeIndex = 0;
    for (let i = 0; i <= targetIndex; i++) {
      if (this.samples[i].is_sync) {
        keyframeIndex = i;
      }
    }

    // Reset decoder for the jump
    this.decoder.reset();
    this.decoder.configure(this.codecConfig!);

    // Decode from keyframe up to target frame
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

    // Flush and wait for the last frame to arrive
    // Note: flush() ensures all decode operations complete
    await this.decoder.flush();
    this.sampleIndex = targetIndex + 1;
  }

  /**
   * Get the sample index for a given time (for checking if sequential decode is possible)
   */
  getSampleIndexForTime(timeSeconds: number): number {
    if (this.useSimpleMode || !this.videoTrack || this.samples.length === 0) {
      return -1;
    }

    const targetTime = timeSeconds * this.videoTrack.timescale;
    // Add half-frame tolerance to handle floating point precision issues
    const halfFrame = (this.samples[0]?.duration ?? 1) / 2;
    const targetTimeWithTolerance = targetTime + halfFrame;
    let targetIndex = 0;
    for (let i = 0; i < this.samples.length; i++) {
      if (this.samples[i].cts > targetTimeWithTolerance) break;
      targetIndex = i;
    }
    return targetIndex;
  }

  /**
   * Get current sample index (for sequential export tracking)
   */
  getCurrentSampleIndex(): number {
    return this.sampleIndex;
  }

  /**
   * Check if currently in export mode
   */
  isExportMode(): boolean {
    return this.isInExportMode;
  }

  /**
   * End sequential export mode
   */
  endSequentialExport(): void {
    this.isInExportMode = false;
    console.log('[WebCodecs] Export mode: ended');
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
