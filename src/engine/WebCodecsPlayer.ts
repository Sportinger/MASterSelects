// WebCodecs-based video player for hardware-accelerated decoding
// Bypasses browser VAAPI issues by using WebCodecs API directly
// Export mode delegated to WebCodecsExportMode

import { Logger } from '../services/logger';
import { wcPipelineMonitor } from '../services/wcPipelineMonitor';
import { vfPipelineMonitor } from '../services/vfPipelineMonitor';
const log = Logger.create('WebCodecsPlayer');

import * as MP4BoxModule from 'mp4box';
const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

import type { Sample, MP4VideoTrack, MP4ArrayBuffer, MP4File } from './webCodecsTypes';
import { WebCodecsExportMode } from './WebCodecsExportMode';
import type { ExportModePlayer } from './WebCodecsExportMode';

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

export class WebCodecsPlayer implements ExportModePlayer {
  private mp4File: MP4File | null = null;
  private decoder: VideoDecoder | null = null;
  private currentFrame: VideoFrame | null = null;
  private samples: Sample[] = [];
  private sampleIndex = 0;    // Display position (which sample is currently shown)
  private feedIndex = 0;       // Feed position (how far we've fed the decoder)
  private _isPlaying = false;
  private loop: boolean;
  private frameRate = 30;
  private frameInterval = 1000 / 30;
  private lastFrameTime = 0;
  private animationId: number | null = null;
  private videoTrack: MP4VideoTrack | null = null;
  private codecConfig: VideoDecoderConfig | null = null;
  private currentFrameTimestampUs: number | null = null;

  // Frame buffer: decoder outputs go here, display picks from here
  private frameBuffer: VideoFrame[] = [];
  private static readonly MAX_FRAME_BUFFER = 8;
  private static readonly FEED_LOOKAHEAD = 10;
  private static readonly FEED_QUEUE_TARGET = 5;
  private static readonly ADVANCE_SEEK_QUEUE_TARGET = 24;
  private static readonly ADVANCE_SEEK_FORWARD_TOLERANCE = 18;
  private static readonly ADVANCE_SEEK_BACKWARD_TOLERANCE = 2;
  private static readonly ADVANCE_SEEK_DIVERGENCE_TOLERANCE = 24;

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

  // Export mode (delegated to WebCodecsExportMode)
  private exportMode: WebCodecsExportMode;
  private frameResolve: (() => void) | null = null;
  private decoderInitialized = false;
  private pendingDecodeFirstFrame = false;
  private loadResolve: (() => void) | null = null;

  // CTS-sorted sample index for O(log n) time lookup (built lazily)
  private ctsSorted: { idx: number; cts: number }[] = [];
  private ctsSortedSampleCount = 0;

  // Seek target filtering: during a seek, intermediate GOP frames are decoded
  // before the target. This prevents the renderer from showing them.
  // Set by seek()/fastSeek(), cleared when target frame arrives in output callback.
  private seekTargetUs: number | null = null;
  private seekTargetToleranceUs = 0;
  private pendingAdvanceSeekTargetIdx: number | null = null;
  private trackedDecodeQueueSize = 0;

  // ExportModePlayer interface implementation
  getDecoder(): VideoDecoder | null { return this.decoder; }
  getSamples(): Sample[] { return this.samples; }
  getSampleIndex(): number { return this.sampleIndex; }
  setSampleIndex(index: number): void { this.sampleIndex = index; this.feedIndex = index; }
  getVideoTrackTimescale(): number | null { return this.videoTrack?.timescale ?? null; }
  getCodecConfig(): VideoDecoderConfig | null { return this.codecConfig; }
  getFrameRate(): number { return this.frameRate; }
  getCurrentFrame(): VideoFrame | null {
    // Keep showing last stable frame while seek traversal is in progress.
    // Intermediate traversal frames are dropped in decoder output callback.
    return this.currentFrame;
  }
  setCurrentFrame(frame: VideoFrame | null): void {
    this.currentFrame = frame;
    this.currentFrameTimestampUs = frame?.timestamp ?? null;
  }
  isSimpleMode(): boolean { return this.useSimpleMode; }
  isFullMode(): boolean { return !this.useSimpleMode && this.ready; }
  isSeeking(): boolean { return this.seekTargetUs !== null; }
  getPendingSeekTime(): number | null { return this.seekTargetUs !== null ? this.seekTargetUs / 1_000_000 : null; }
  getVideoElement(): HTMLVideoElement | null { return this.videoElement; }

  constructor(options: WebCodecsPlayerOptions = {}) {
    this.exportMode = new WebCodecsExportMode(this);
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
    log.info(`Stream attached: ${this.width}x${this.height}`);

    // Start reading frames
    this.startStreamCapture();

    this.onReady?.(this.width, this.height);
  }

  private async startStreamCapture(): Promise<void> {
    if (!this.streamReader || this.streamActive) return;

    this.streamActive = true;
    log.debug('Starting stream frame capture');

    try {
      while (this.streamActive) {
        const { value: frame, done } = await this.streamReader.read();

        if (done) {
          log.debug('Stream ended');
          break;
        }

        if (frame) {
          // Close previous frame
          if (this.currentFrame) {
            this.currentFrame.close();
          }
          this.currentFrame = frame;
          this.currentFrameTimestampUs = frame.timestamp;
          this.onFrame?.(frame);
        }
      }
    } catch (e) {
      log.warn('Error reading frames from stream', e);
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

    log.info(`Simple mode attached to existing video: ${this.width}x${this.height}`);

    // Listen to video element events - Timeline controls the video, we just capture frames
    this.boundOnPlay = () => {
      if (this._isPlaying) return; // Already playing
      log.debug('Video play event - starting frame capture');
      this._isPlaying = true;
      this.startSimpleFrameCapture();
    };
    this.boundOnPause = () => {
      if (!this._isPlaying) return; // Already paused
      log.debug('Video pause event');
      this._isPlaying = false;
      this.stopSimpleFrameCapture();
      // Capture the paused frame
      this.captureCurrentFrame();
    };
    this.boundOnSeeked = () => {
      vfPipelineMonitor.record('vf_seek_done', {
        currentTime: Math.round(video.currentTime * 1000) / 1000,
        readyState: video.readyState,
      });
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

        log.debug(`Video loaded: ${this.width}x${this.height}`);
      };

      video.oncanplay = () => {
        clearTimeout(timeout);
        this.ready = true;

        // Create initial frame
        this.captureCurrentFrame();

        log.info(`Simple mode READY: ${this.width}x${this.height}`);
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
      this.currentFrameTimestampUs = this.currentFrame.timestamp;
      vfPipelineMonitor.record('vf_capture', {
        videoTime: Math.round(this.videoElement.currentTime * 1000) / 1000,
        readyState: this.videoElement.readyState,
      });
      this.onFrame?.(this.currentFrame);
    } catch (e) {
      vfPipelineMonitor.record('vf_drop', {
        readyState: this.videoElement?.readyState ?? -1,
        reason: 'capture_error',
      });
    }
  }

  async loadArrayBuffer(buffer: ArrayBuffer): Promise<void> {
    const endLoad = log.time('loadArrayBuffer');
    return new Promise((resolve, reject) => {
      log.info(`Parsing MP4 (${(buffer.byteLength / 1024 / 1024).toFixed(1)}MB)...`);

      // Reduced timeout - we only wait for codec info now, not all samples
      const timeout = setTimeout(() => {
        reject(new Error('MP4 parsing timeout - file may have unsupported metadata'));
      }, 5000);

      this.mp4File = MP4Box.createFile();
      const mp4File = this.mp4File!;
      let resolved = false;

      mp4File.onReady = (info) => {
        log.info(`MP4 onReady: ${info.videoTracks.length} video tracks`);
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
              log.debug(`Extracted codec description: ${description!.byteLength} bytes from ${entry.avcC ? 'avcC' : entry.hvcC ? 'hvcC' : entry.vpcC ? 'vpcC' : 'av1C'}`);
            } else {
              log.warn('No codec config box found in sample entry', Object.keys(entry));
            }
          }
        } catch (e) {
          log.warn('Failed to extract codec description', e);
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
        log.debug(`Extraction started for track ${videoTrack.id}`);

        // Check if codec is supported (async, but extraction already started)
        VideoDecoder.isConfigSupported(this.codecConfig).then((support) => {
          if (!support.supported) {
            clearTimeout(timeout);
            reject(new Error(`Codec ${codec} not supported`));
            return;
          }

          log.debug(`Codec ${codec} supported`, support.config);
          this.initDecoder();

          // RESOLVE IMMEDIATELY after decoder is configured - don't wait for samples!
          // Samples will continue loading in background
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            endLoad();
            log.info(`Decoder configured: ${this.width}x${this.height} @ ${this.frameRate.toFixed(1)}fps (samples loading in background)`);
            resolve();
          }
        });
      };

      mp4File.onSamples = (_trackId, _ref, samples) => {
        this.samples.push(...samples);

        // Mark ready when we have samples and decoder (for playback mode)
        if (!this.ready && this.samples.length > 0 && this.decoderInitialized) {
          this.ready = true;
          log.info(`READY: ${this.samples.length} samples loaded so far`);

          this.decodeFirstFrame();
          this.onReady?.(this.width, this.height);
        } else if (!this.ready && this.samples.length > 0 && !this.decoderInitialized) {
          // Samples received but decoder not ready yet
          this.pendingDecodeFirstFrame = true;
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
        log.debug(`Appended ${appendedBytes} bytes to MP4Box`);
        mp4File.flush();
        log.debug('Flushed MP4Box, waiting for callbacks...');
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
        const queueSize = this.noteDecodeDequeued();
        wcPipelineMonitor.record('decode_output', {
          ts: frame.timestamp,
          queueSize,
        });

        // In export mode, buffer ALL frames via export mode handler
        if (this.exportMode.isInExportMode) {
          this.exportMode.handleDecoderOutput(frame);
        } else if (this._isPlaying) {
          // Playback mode: buffer frame for time-controlled presentation
          this.frameBuffer.push(frame);
          // Evict oldest if buffer overflows, but protect currentFrame
          while (this.frameBuffer.length > WebCodecsPlayer.MAX_FRAME_BUFFER) {
            const oldest = this.frameBuffer[0];
            if (oldest === this.currentFrame) break; // Don't close the displayed frame
            this.frameBuffer.shift()!.close();
          }
        } else {
          // Paused/seeking: keep currentFrame stable during seek traversal.
          // Only publish frame when the seek target is reached.
          if (this.seekTargetUs !== null) {
            const diff = Math.abs(frame.timestamp - this.seekTargetUs);
            if (diff <= this.seekTargetToleranceUs) {
              if (this.currentFrame && this.currentFrame !== frame) {
                this.currentFrame.close();
              }
              this.currentFrame = frame;
              this.currentFrameTimestampUs = frame.timestamp;
              wcPipelineMonitor.record('seek_publish', {
                targetUs: Math.round(this.seekTargetUs),
                frameUs: Math.round(frame.timestamp),
                diffUs: Math.round(diff),
              });
              this.seekTargetUs = null;
              this.seekTargetToleranceUs = 0;
              this.onFrame?.(frame);
            } else {
              // Drop intermediate GOP traversal frame to avoid visual jumps.
              wcPipelineMonitor.record('frame_drop', {
                reason: 'seek_intermediate',
                frameUs: Math.round(frame.timestamp),
                targetUs: Math.round(this.seekTargetUs),
              });
              frame.close();
            }
          } else {
            if (this.currentFrame && this.currentFrame !== frame) {
              this.currentFrame.close();
            }
            this.currentFrame = frame;
            this.currentFrameTimestampUs = frame.timestamp;
            this.onFrame?.(frame);
          }
        }
        // Resolve any pending frame wait
        if (this.frameResolve) {
          this.frameResolve();
          this.frameResolve = null;
        }
      },
      error: (e) => {
        log.error('VideoDecoder error', e);
        this.onError?.(new Error(`Decoder error: ${e.message}`));
      },
    });

    this.decoder.configure(this.codecConfig);
    this.resetDecodeQueueTracking();
    this.decoderInitialized = true;

    // Handle any deferred first frame decode and resolve loadArrayBuffer promise
    if (this.pendingDecodeFirstFrame && this.samples.length > 0) {
      this.pendingDecodeFirstFrame = false;
      this.ready = true;
      log.info(`READY (deferred): ${this.width}x${this.height} @ ${this.frameRate.toFixed(1)}fps, ${this.samples.length} samples`);

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

    if (!this.useSimpleMode) {
      wcPipelineMonitor.record('play');
    }

    if (this.useSimpleMode && this.videoElement) {
      vfPipelineMonitor.record('vf_play');
      // If attached to external video, don't control it - just ensure frame capture is running
      // Timeline controls the video element, we get notified via events
      if (!this.isAttachedToExternal) {
        this.videoElement.play().catch(() => {});
      }
      this.startSimpleFrameCapture();
    } else {
      // Sync feedIndex to sampleIndex on play start
      if (this.feedIndex < this.sampleIndex) {
        this.feedIndex = this.sampleIndex;
      }
      this.lastFrameTime = performance.now();
      this.lastRAFTime = 0;
      this.scheduleNextFrame();
    }
  }

  pause(): void {
    if (this._isPlaying && !this.useSimpleMode) {
      wcPipelineMonitor.record('pause', {
        buffered: this.frameBuffer.length,
        seeking: this.seekTargetUs !== null ? 'true' : 'false',
      });
    }
    if (this._isPlaying && this.useSimpleMode) {
      vfPipelineMonitor.record('vf_pause');
    }
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
      this.clearFrameBuffer();
      if (this.seekTargetUs !== null) {
        wcPipelineMonitor.record('seek_cancel', { reason: 'pause' });
      }
      this.seekTargetUs = null;
      this.seekTargetToleranceUs = 0;
      this.clearAdvanceSeekState();
      if (this.decoder && this.codecConfig && !this.exportMode.isInExportMode) {
        this.decoder.reset();
        this.decoder.configure(this.codecConfig);
        this.resetDecodeQueueTracking();
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
      this.feedIndex = 0;
    }

    this.clearFrameBuffer();
    this.seekTargetUs = null;
    this.seekTargetToleranceUs = 0;
    this.clearAdvanceSeekState();
    if (this.currentFrame) {
      this.currentFrame.close();
      this.currentFrame = null;
    }
    this.currentFrameTimestampUs = null;
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

  private lastRAFTime = 0;

  private scheduleNextFrame(): void {
    if (!this._isPlaying) return;

    this.animationId = requestAnimationFrame((now) => {
      // Track rAF gaps to detect main thread stalls
      if (this.lastRAFTime > 0) {
        const rafGap = now - this.lastRAFTime;
        if (rafGap > 100) {
          wcPipelineMonitor.record('rAF_gap', { gapMs: Math.round(rafGap) });
        }
      }
      this.lastRAFTime = now;

      // Keep decoder pipeline fed ahead of display
      this.pumpDecoder();

      const elapsed = now - this.lastFrameTime;

      if (elapsed >= this.frameInterval) {
        // Present next buffered frame at the video's natural frame rate
        this.presentBufferedFrame();
        this.lastFrameTime = now - (elapsed % this.frameInterval);
      }

      this.scheduleNextFrame();
    });
  }

  /** Feed samples to the decoder, staying ahead of the display position */
  private pumpDecoder(): void {
    if (!this.decoder || this.samples.length === 0) return;

    while (
      this.feedIndex < this.samples.length &&
      this.feedIndex - this.sampleIndex < WebCodecsPlayer.FEED_LOOKAHEAD &&
      this.getEffectiveDecodeQueueSize() < WebCodecsPlayer.FEED_QUEUE_TARGET
    ) {
      const sample = this.samples[this.feedIndex];
      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts * 1_000_000) / sample.timescale,
        duration: (sample.duration * 1_000_000) / sample.timescale,
        data: sample.data,
      });

      try {
        this.decoder.decode(chunk);
        const queueSize = this.decoder.decodeQueueSize;
        wcPipelineMonitor.record('decode_feed', {
          sampleIdx: this.feedIndex,
          type: sample.is_sync ? 'key' : 'delta',
          queueSize,
        });
        if (queueSize > 3) {
          wcPipelineMonitor.record('queue_pressure', { queueSize });
        }
      } catch {
        // Skip decode errors
      }

      this.feedIndex++;
    }

    // Handle loop
    if (this.feedIndex >= this.samples.length && this.loop) {
      this.feedIndex = 0;
      this.sampleIndex = 0;
      this.decoder.reset();
      this.decoder.configure(this.codecConfig!);
      this.resetDecodeQueueTracking();
      this.clearFrameBuffer();
    }
  }

  /** Present the oldest frame from the buffer */
  private presentBufferedFrame(): void {
    if (this.frameBuffer.length === 0) return;

    const frame = this.frameBuffer.shift()!;
    if (this.currentFrame) {
      this.currentFrame.close();
    }
    this.currentFrame = frame;
    this.currentFrameTimestampUs = frame.timestamp;
    this.sampleIndex++;
    this.onFrame?.(frame);
  }

  /** Close all buffered frames */
  private clearFrameBuffer(): void {
    for (const f of this.frameBuffer) {
      f.close();
    }
    this.frameBuffer.length = 0;
  }

  // === Render-loop-driven playback (replaces internal animation loop) ===

  /** Lazy-build CTS-sorted index for O(log n) sample lookup */
  private ensureCtsIndex(): void {
    if (this.ctsSortedSampleCount === this.samples.length) return;
    this.ctsSorted = this.samples.map((s, i) => ({ idx: i, cts: s.cts }));
    this.ctsSorted.sort((a, b) => a.cts - b.cts);
    this.ctsSortedSampleCount = this.samples.length;
  }

  /** Binary search for sample index whose CTS is closest to target */
  private findSampleNearCts(targetCts: number): number {
    this.ensureCtsIndex();
    const sorted = this.ctsSorted;
    if (sorted.length === 0) return 0;

    let lo = 0, hi = sorted.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid].cts < targetCts) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(sorted[lo - 1].cts - targetCts) < Math.abs(sorted[lo].cts - targetCts)) {
      return sorted[lo - 1].idx;
    }
    return sorted[lo].idx;
  }

  /** Find nearest keyframe at or before the given sample index (DTS order) */
  private findKeyframeBefore(sampleIndex: number): number {
    for (let i = Math.min(sampleIndex, this.samples.length - 1); i >= 0; i--) {
      if (this.samples[i].is_sync) return i;
    }
    return 0;
  }

  private getCurrentFrameSampleIndex(): number | null {
    if (this.currentFrameTimestampUs === null || !this.videoTrack) {
      return null;
    }
    const currentCts = (this.currentFrameTimestampUs * this.videoTrack.timescale) / 1_000_000;
    return this.findSampleNearCts(currentCts);
  }

  private clearAdvanceSeekState(): void {
    this.pendingAdvanceSeekTargetIdx = null;
  }

  private resetDecodeQueueTracking(): void {
    this.trackedDecodeQueueSize = 0;
  }

  private getEffectiveDecodeQueueSize(): number {
    return Math.max(this.decoder?.decodeQueueSize ?? 0, this.trackedDecodeQueueSize);
  }

  private noteDecodeQueued(): number {
    const reportedQueueSize = this.decoder?.decodeQueueSize ?? 0;
    this.trackedDecodeQueueSize = Math.max(
      reportedQueueSize,
      this.trackedDecodeQueueSize + 1
    );
    return this.getEffectiveDecodeQueueSize();
  }

  private noteDecodeDequeued(): number {
    const reportedQueueSize = this.decoder?.decodeQueueSize ?? 0;
    if (reportedQueueSize >= 0) {
      // Browser's VideoDecoder.decodeQueueSize is authoritative — trust it
      // to pull our tracked estimate back to reality and prevent inflation.
      this.trackedDecodeQueueSize = reportedQueueSize;
    } else {
      // Fallback: decrement tracked
      this.trackedDecodeQueueSize = Math.max(0, this.trackedDecodeQueueSize - 1);
    }
    return this.getEffectiveDecodeQueueSize();
  }

  private shouldContinueAdvanceSeek(
    targetIdx: number,
    decodeCoverageEnd: number
  ): boolean {
    const pendingTargetIdx = this.pendingAdvanceSeekTargetIdx;
    if (pendingTargetIdx === null) {
      return false;
    }

    if (
      targetIdx < pendingTargetIdx - WebCodecsPlayer.ADVANCE_SEEK_BACKWARD_TOLERANCE ||
      targetIdx > pendingTargetIdx + WebCodecsPlayer.ADVANCE_SEEK_DIVERGENCE_TOLERANCE
    ) {
      return false;
    }

    const desiredCoverageEnd =
      Math.max(targetIdx, pendingTargetIdx) + WebCodecsPlayer.FEED_LOOKAHEAD;

    return (
      this.getEffectiveDecodeQueueSize() > 0 ||
      decodeCoverageEnd < desiredCoverageEnd
    );
  }

  /** Compute seek acceptance tolerance in microseconds with VFR-aware neighbor spacing. */
  private computeSeekToleranceUs(targetIndex: number): number {
    const nominalFrameUs = 1_000_000 / Math.max(this.frameRate, 1);
    const target = this.samples[targetIndex];
    if (!target) return nominalFrameUs * 1.5;

    let neighborDeltaUs = Infinity;

    if (targetIndex > 0) {
      const prev = this.samples[targetIndex - 1];
      const prevDelta = Math.abs(target.cts - prev.cts) * 1_000_000 / target.timescale;
      if (prevDelta > 0) neighborDeltaUs = Math.min(neighborDeltaUs, prevDelta);
    }

    if (targetIndex < this.samples.length - 1) {
      const next = this.samples[targetIndex + 1];
      const nextDelta = Math.abs(next.cts - target.cts) * 1_000_000 / target.timescale;
      if (nextDelta > 0) neighborDeltaUs = Math.min(neighborDeltaUs, nextDelta);
    }

    const vfrAwareUs = Number.isFinite(neighborDeltaUs)
      ? neighborDeltaUs * 0.75
      : nominalFrameUs * 1.5;

    return Math.max(2_000, Math.min(200_000, Math.max(vfrAwareUs, nominalFrameUs)));
  }

  /**
   * Advance playback to the given source time.
   * Called by the render loop each frame instead of an internal animation loop.
   * Handles: decoder feeding, timestamp-based frame selection, position tracking.
   */
  advanceToTime(timeSeconds: number): void {
    if (this.useSimpleMode || !this.decoder || this.samples.length === 0 || !this.videoTrack) return;

    const startingPlayback = !this._isPlaying;
    const hadPendingPreciseSeek = this.seekTargetUs !== null;
    const hadPendingAdvanceSeek = this.pendingAdvanceSeekTargetIdx !== null;
    const hadQueuedDecodeBacklog =
      this.getEffectiveDecodeQueueSize() > 0 ||
      this.frameBuffer.length > 0;
    const shouldRestartPlaybackPipeline =
      startingPlayback &&
      (hadPendingPreciseSeek || hadPendingAdvanceSeek || hadQueuedDecodeBacklog);

    // Clear any pending seek target from paused seeking
    this.seekTargetUs = null;
    this.seekTargetToleranceUs = 0;

    // Auto-enter playing state so decoder output routes to frame buffer
    if (startingPlayback) {
      this._isPlaying = true;
      this.clearFrameBuffer();
      if (this.feedIndex < this.sampleIndex) {
        this.feedIndex = this.sampleIndex;
      }
    }

    // Cancel internal animation loop if running — we're externally driven
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    const targetCts = timeSeconds * this.videoTrack.timescale;
    const targetIdx = this.findSampleNearCts(targetCts);
    const targetUs = timeSeconds * 1_000_000;
    const frameDurationUs = 1_000_000 / this.frameRate;
    const currentFrameIdx = this.getCurrentFrameSampleIndex() ?? this.sampleIndex;
    const decodeCoverageEnd = Math.max(this.feedIndex, currentFrameIdx + this.frameBuffer.length);
    const backwardJump =
      targetIdx < currentFrameIdx - WebCodecsPlayer.ADVANCE_SEEK_BACKWARD_TOLERANCE;
    const forwardSeekThreshold = Math.max(
      WebCodecsPlayer.ADVANCE_SEEK_FORWARD_TOLERANCE,
      Math.ceil(this.frameRate * 0.35)
    );
    const forwardGap = targetIdx - decodeCoverageEnd;

    // Check if decoder needs repositioning:
    // - target is behind current position (backward jump)
    // - target is far ahead of what we've fed (gap/skip/clip start)
    // For playback restarts: skip the heavyweight decoder.reset() + configure()
    // when the decoder pipeline is already positioned at/near the target.
    // This avoids 50-100ms+ main-thread blocking from hardware decode resets.
    let restartNeedsReset = false;
    if (shouldRestartPlaybackPipeline) {
      const keyframeForTarget = this.findKeyframeBefore(targetIdx);
      // feedIndex is already at or slightly past the keyframe we'd reset to,
      // AND not too far ahead (within a small window of samples).
      // In that case the decoder is already producing the right frames.
      const feedDistFromKeyframe = this.feedIndex - keyframeForTarget;
      const isFeedPositionedCorrectly =
        feedDistFromKeyframe >= 0 && feedDistFromKeyframe <= 8;
      if (isFeedPositionedCorrectly) {
        wcPipelineMonitor.record('seek_skip', {
          reason: 'reset_already_positioned',
          feedIndex: this.feedIndex,
          keyframe: keyframeForTarget,
          targetIdx,
          feedDist: feedDistFromKeyframe,
        });
      } else {
        restartNeedsReset = true;
      }
    }
    let needsSeek =
      restartNeedsReset ||
      backwardJump ||
      forwardGap > forwardSeekThreshold;
    const keepPendingAdvanceSeekAlive =
      !shouldRestartPlaybackPipeline &&
      this.shouldContinueAdvanceSeek(
      targetIdx,
      decodeCoverageEnd
    );
    const advanceTargetIdx =
      keepPendingAdvanceSeekAlive && this.pendingAdvanceSeekTargetIdx !== null
        ? Math.max(targetIdx, this.pendingAdvanceSeekTargetIdx)
        : targetIdx;

    if (needsSeek && keepPendingAdvanceSeekAlive) {
      wcPipelineMonitor.record('seek_skip', {
        reason: 'advance_inflight',
        target: Math.round(timeSeconds * 1000) / 1000,
        targetIdx,
        pendingTargetIdx: this.pendingAdvanceSeekTargetIdx ?? -1,
        coverageEnd: decodeCoverageEnd,
        queueSize: this.getEffectiveDecodeQueueSize(),
      });
      needsSeek = false;
    }

    if (needsSeek) {
      const keyframe = this.findKeyframeBefore(targetIdx);
      this.decoder.reset();
      this.decoder.configure(this.codecConfig!);
      this.resetDecodeQueueTracking();
      this.clearFrameBuffer();
      this.feedIndex = keyframe;
      this.pendingAdvanceSeekTargetIdx = advanceTargetIdx;
      wcPipelineMonitor.record('advance_seek', {
        target: timeSeconds,
        keyframeDist: targetIdx - keyframe,
        forwardGap,
        currentFrameIdx,
        reason: shouldRestartPlaybackPipeline ? 'playback_restart' : 'advance',
      });
    } else if (keepPendingAdvanceSeekAlive) {
      this.pendingAdvanceSeekTargetIdx = advanceTargetIdx;
    }

    // Pump decoder: feed samples ahead of target position.
    // During seeks, bypass queue limit to push all GOP frames at once —
    // the decoder processes them off-main-thread in one burst.
    const keepAdvanceFeedActive = needsSeek || keepPendingAdvanceSeekAlive;
    const feedTarget = Math.min(
      (keepAdvanceFeedActive ? advanceTargetIdx : targetIdx) +
        WebCodecsPlayer.FEED_LOOKAHEAD,
      this.samples.length
    );
    const queueLimit = keepAdvanceFeedActive
      ? WebCodecsPlayer.ADVANCE_SEEK_QUEUE_TARGET
      : WebCodecsPlayer.FEED_QUEUE_TARGET;
    const queuePressureThreshold = Math.max(3, queueLimit - 2);
    let hitQueueCap = false;
    while (this.feedIndex < feedTarget) {
      if (this.getEffectiveDecodeQueueSize() >= queueLimit) {
        hitQueueCap = true;
        break;
      }
      const sample = this.samples[this.feedIndex];
      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts * 1_000_000) / sample.timescale,
        duration: (sample.duration * 1_000_000) / sample.timescale,
        data: sample.data,
      });
      try {
        this.decoder.decode(chunk);
        const queueSize = this.noteDecodeQueued();
        wcPipelineMonitor.record('decode_feed', {
          sampleIdx: this.feedIndex,
          type: sample.is_sync ? 'key' : 'delta',
          queueSize,
          mode: needsSeek
            ? 'advance_seek'
            : keepPendingAdvanceSeekAlive
              ? 'advance_pending'
              : 'advance',
        });
        if (queueSize >= queuePressureThreshold) {
          wcPipelineMonitor.record('queue_pressure', {
            queueSize,
            queueLimit,
            mode: needsSeek
              ? 'advance_seek'
              : keepPendingAdvanceSeekAlive
                ? 'advance_pending'
                : 'advance',
          });
        }
      } catch { /* skip decode errors */ }
      this.feedIndex++;
    }

    if (needsSeek && hitQueueCap) {
      wcPipelineMonitor.record('seek_skip', {
        reason: 'advance_queue_cap',
        target: Math.round(timeSeconds * 1000) / 1000,
        targetIdx,
        queueSize: this.getEffectiveDecodeQueueSize(),
        queueLimit,
      });
    }

    this.sampleIndex = targetIdx;

    // Pick the frame closest to target time from the decode buffer.
    // CRITICAL: Only accept frames within 1.5 frame-durations of the target.
    // This prevents showing intermediate GOP-traversal frames during seeks —
    // without this, the renderer would flash through keyframe → target visibly.
    if (this.frameBuffer.length > 0) {
      let bestIdx = -1;
      let bestDiff = Infinity;
      for (let i = 0; i < this.frameBuffer.length; i++) {
        const diff = Math.abs(this.frameBuffer[i].timestamp - targetUs);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIdx = i;
        }
      }

      const acceptable = bestIdx >= 0 && bestDiff < frameDurationUs * 1.5;

      if (acceptable) {
        // Accept this frame — close everything before it
        for (let i = 0; i < bestIdx; i++) {
          if (this.frameBuffer[i] !== this.currentFrame) {
            this.frameBuffer[i].close();
          }
        }
        const frame = this.frameBuffer[bestIdx];
        if (this.currentFrame && this.currentFrame !== frame) {
          this.currentFrame.close();
        }
        this.currentFrame = frame;
        this.currentFrameTimestampUs = frame.timestamp;
        this.onFrame?.(frame);
        this.frameBuffer.splice(0, bestIdx + 1);
        if (this.pendingAdvanceSeekTargetIdx !== null) {
          this.clearAdvanceSeekState();
        }
      } else {
        // No acceptable frame yet — clean up stale past frames but keep future ones.
        // The decoder is still producing frames; the right one will arrive soon.
        const expireThreshold = targetUs - frameDurationUs * 2;
        while (
          this.frameBuffer.length > 0 &&
          this.frameBuffer[0].timestamp < expireThreshold &&
          this.frameBuffer[0] !== this.currentFrame
        ) {
          this.frameBuffer.shift()!.close();
        }
      }
    }
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
      this.noteDecodeQueued();
      this.sampleIndex = 0;
      this.feedIndex = 1;
      this.clearAdvanceSeekState();
    } catch {
      // Ignore decode errors on first frame
    }
  }


  // Check if there's a valid frame available
  hasFrame(): boolean {
    return this.currentFrame !== null;
  }

  /** Debug info for stats overlay (null in simple mode) */
  getDebugInfo(): { codec: string; hwAccel: string; decodeQueueSize: number; samplesLoaded: number; sampleIndex: number } | null {
    if (this.useSimpleMode || !this.codecConfig) return null;
    return {
      codec: this.codecConfig.codec,
      hwAccel: (this.codecConfig.hardwareAcceleration as string) || 'no-preference',
      decodeQueueSize: this.getEffectiveDecodeQueueSize(),
      samplesLoaded: this.samples.length,
      sampleIndex: this.sampleIndex,
    };
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

    const seekStart = performance.now();
    const targetTime = timeSeconds * this.videoTrack.timescale;

    // Binary search for closest CTS match (O(log n) instead of O(n))
    const targetIndex = this.findSampleNearCts(targetTime);
    const keyframeIndex = this.findKeyframeBefore(targetIndex);
    const framesDecoded = targetIndex - keyframeIndex + 1;

    // Set seek target. Intermediate GOP traversal frames are dropped in output callback
    // so the renderer keeps showing the last stable frame until the target arrives.
    const targetSample = this.samples[targetIndex];
    this.seekTargetUs = (targetSample.cts * 1_000_000) / targetSample.timescale;
    this.seekTargetToleranceUs = this.computeSeekToleranceUs(targetIndex);
    this.clearAdvanceSeekState();

    wcPipelineMonitor.record('seek_start', {
      target: timeSeconds,
      keyframeDist: framesDecoded,
    });

    // Reset decoder
    this.decoder.reset();
    this.decoder.configure(this.codecConfig!);
    this.resetDecodeQueueTracking();

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
        this.noteDecodeQueued();
      } catch {
        // Skip decode errors
      }
    }

    this.sampleIndex = targetIndex;
    this.feedIndex = targetIndex + 1;
    this.clearFrameBuffer();

    wcPipelineMonitor.record('seek_end', {
      target: timeSeconds,
      framesDecoded,
      durationMs: Math.round(performance.now() - seekStart),
    });
  }

  /**
   * Fast seek: decode only the nearest keyframe (1 frame instead of N).
   * Use during fast scrubbing for instant feedback — shows nearest I-frame.
   */
  fastSeek(timeSeconds: number): void {
    if (this.useSimpleMode && this.videoElement) {
      const vid = this.videoElement;
      if (typeof (vid as any).fastSeek === 'function') {
        (vid as any).fastSeek(timeSeconds);
        vfPipelineMonitor.record('vf_seek_fast', {
          target: Math.round(timeSeconds * 1000) / 1000,
        });
      } else {
        vid.currentTime = timeSeconds;
        vfPipelineMonitor.record('vf_seek_precise', {
          target: Math.round(timeSeconds * 1000) / 1000,
        });
      }
      this.captureCurrentFrame();
      return;
    }

    if (!this.videoTrack || this.samples.length === 0 || !this.decoder) return;

    const targetTime = timeSeconds * this.videoTrack.timescale;

    // Find nearest keyframe: check before AND after target for closest match
    const targetIdx = this.findSampleNearCts(targetTime);
    const kfBefore = this.findKeyframeBefore(targetIdx);
    let bestKeyframe = kfBefore;
    // Check if a keyframe after target is closer
    for (let i = targetIdx + 1; i < this.samples.length; i++) {
      if (this.samples[i].is_sync) {
        if (Math.abs(this.samples[i].cts - targetTime) < Math.abs(this.samples[kfBefore].cts - targetTime)) {
          bestKeyframe = i;
        }
        break;
      }
    }

    // fastSeek shows keyframe directly — no GOP traversal, so no seekTargetUs needed
    this.seekTargetUs = null;
    this.seekTargetToleranceUs = 0;
    this.clearAdvanceSeekState();

    // Reset decoder and decode just the keyframe
    this.decoder.reset();
    this.decoder.configure(this.codecConfig!);
    this.resetDecodeQueueTracking();

    const sample = this.samples[bestKeyframe];
    const chunk = new EncodedVideoChunk({
      type: 'key',
      timestamp: (sample.cts * 1_000_000) / sample.timescale,
      duration: (sample.duration * 1_000_000) / sample.timescale,
      data: sample.data,
    });

    try {
      this.decoder.decode(chunk);
      this.noteDecodeQueued();
    } catch {
      // Skip decode errors
    }

    this.sampleIndex = bestKeyframe;
    this.feedIndex = bestKeyframe + 1;
    this.clearFrameBuffer();
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
            log.warn(`seekAsync timeout at ${timeSeconds}, readyState: ${video.readyState}`);
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
              log.warn(`waitForReady gave up after ${retries} retries, readyState: ${video.readyState}`);
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

    this.seekTargetUs = null;
    this.seekTargetToleranceUs = 0;
    this.clearAdvanceSeekState();

    const targetTime = timeSeconds * this.videoTrack.timescale;

    // Binary search for closest CTS match
    const targetIndex = this.findSampleNearCts(targetTime);
    const keyframeIndex = this.findKeyframeBefore(targetIndex);

    // Reset decoder
    this.decoder.reset();
    this.decoder.configure(this.codecConfig!);
    this.resetDecodeQueueTracking();

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
        this.noteDecodeQueued();
      } catch {
        // Skip decode errors
      }
    }

    // Flush to ensure all frames are decoded
    await this.decoder.flush();

    this.sampleIndex = targetIndex;
    this.feedIndex = targetIndex + 1;
    this.clearFrameBuffer();
  }

  // ==================== EXPORT MODE (delegated to WebCodecsExportMode) ====================

  async prepareForSequentialExport(startTimeSeconds: number): Promise<void> {
    return this.exportMode.prepareForSequentialExport(startTimeSeconds);
  }

  async seekDuringExport(timeSeconds: number): Promise<void> {
    return this.exportMode.seekDuringExport(timeSeconds);
  }

  getCurrentSampleIndex(): number {
    return this.sampleIndex;
  }

  isExportMode(): boolean {
    return this.exportMode.isInExportMode;
  }

  endSequentialExport(): void {
    this.exportMode.endSequentialExport();
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
    if (this.currentFrameTimestampUs !== null) {
      return this.currentFrameTimestampUs / 1_000_000;
    }
    if (!this.videoTrack || this.samples.length === 0) return 0;
    const sample = this.samples[Math.min(this.sampleIndex, this.samples.length - 1)];
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
      this.resetDecodeQueueTracking();
    }

    // Clean up export mode
    this.exportMode.destroy();

    if (this.currentFrame) {
      // Only close if not already closed in buffer cleanup
      try {
        this.currentFrame.close();
      } catch {
        // Already closed
      }
      this.currentFrame = null;
    }
    this.currentFrameTimestampUs = null;

    this.mp4File = null;
    this.samples = [];
    this.ready = false;
  }
}
