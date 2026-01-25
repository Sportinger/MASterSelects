// src/engine/ffmpeg/FFmpegBridge.ts
// Bridge to FFmpeg WASM for professional codec export

import type {
  FFmpegExportSettings,
  FFmpegProgress,
  FFmpegLogEntry,
  ProResProfile,
  FFmpegVideoCodec,
} from './types';
import { Logger } from '../../services/logger';

const log = Logger.create('FFmpegBridge');

// FFmpeg WASM core interface (direct core, not @ffmpeg/ffmpeg wrapper)
interface FFmpegCore {
  FS: {
    writeFile: (path: string, data: Uint8Array) => void;
    readFile: (path: string) => Uint8Array;
    unlink: (path: string) => void;
    readdir: (path: string) => string[];
    mkdir: (path: string) => void;
  };
  callMain: (args: string[]) => number;
  setLogger: (logger: (log: { type: string; message: string }) => void) => void;
  setProgress: (handler: (progress: { progress: number; time: number }) => void) => void;
  reset: () => void;
  ret: number;
}

type LoadState = 'idle' | 'loading' | 'ready' | 'error';

export class FFmpegBridge {
  private ffmpeg: FFmpegCore | null = null;
  private loadState: LoadState = 'idle';
  private loadPromise: Promise<void> | null = null;
  private logs: FFmpegLogEntry[] = [];
  private onProgress?: (progress: FFmpegProgress) => void;
  private onLog?: (log: FFmpegLogEntry) => void;
  private cancelled = false;
  private totalFrames = 0;
  private startTime = 0;

  /**
   * Check if FFmpeg WASM is supported in this browser
   * Now supports single-threaded mode, so only WebAssembly is required
   */
  static isSupported(): boolean {
    return typeof WebAssembly !== 'undefined';
  }

  /**
   * Check if multi-threaded mode is available (faster but needs SharedArrayBuffer)
   */
  static isMultiThreaded(): boolean {
    return typeof SharedArrayBuffer !== 'undefined';
  }

  /**
   * Load FFmpeg WASM module
   * Uses @ffmpeg/ffmpeg from npm (must be installed)
   */
  async load(): Promise<void> {
    if (this.loadState === 'ready') return;
    if (this.loadPromise) return this.loadPromise;

    this.loadState = 'loading';
    this.loadPromise = this.doLoad();

    try {
      await this.loadPromise;
      this.loadState = 'ready';
    } catch (error) {
      this.loadState = 'error';
      throw error;
    }
  }

  private async doLoad(): Promise<void> {
    log.info('Loading FFmpeg WASM...');
    const startTime = performance.now();

    try {
      // Load FFmpeg core directly (bypassing @ffmpeg/ffmpeg wrapper which has issues)
      const baseURL = `${window.location.origin}/ffmpeg`;

      log.debug('Fetching ffmpeg-core.js...');

      // Load via script tag since it's a UMD module, not ES module
      await new Promise<void>((resolve, reject) => {
        // Check if already loaded
        if ((window as unknown as Record<string, unknown>).createFFmpegCore) {
          resolve();
          return;
        }

        const script = document.createElement('script');
        script.src = `${baseURL}/ffmpeg-core.js`;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load ffmpeg-core.js'));
        document.head.appendChild(script);
      });

      // Get the global createFFmpegCore function
      const createFFmpegCore = (window as unknown as Record<string, unknown>).createFFmpegCore as (
        options: Record<string, unknown>
      ) => Promise<FFmpegCore>;

      if (!createFFmpegCore) {
        throw new Error('createFFmpegCore not found after script load');
      }

      log.debug('Fetching ffmpeg-core.wasm...');
      const wasmBinary = await fetch(`${baseURL}/ffmpeg-core.wasm`).then(r => r.arrayBuffer());

      log.debug('Initializing FFmpeg core...');
      const core = await createFFmpegCore({
        wasmBinary,
        locateFile: (path: string) => `${baseURL}/${path}`,
        // Capture stdout/stderr
        print: (message: string) => {
          log.debug('FFmpeg output', message);
          this.handleLog('info', message);
        },
        printErr: (message: string) => {
          if (!message.startsWith('Aborted')) {
            log.debug('FFmpeg stderr', message);
            this.handleLog('warning', message);
          }
        },
      }) as FFmpegCore;

      // Set up progress handler
      if (core.setProgress) {
        core.setProgress(({ progress, time }) => {
          if (this.onProgress && this.totalFrames > 0) {
            const elapsed = (performance.now() - this.startTime) / 1000;
            const speed = elapsed > 0 ? progress / elapsed : 0;
            const remaining = speed > 0 ? (1 - progress) / speed : 0;

            this.onProgress({
              frame: Math.floor(progress * this.totalFrames),
              fps: speed * 30,
              time,
              speed,
              bitrate: 0,
              size: 0,
              percent: progress * 100,
              eta: remaining,
            });
          }
        });
      }

      // Debug: expose to window for testing
      (window as unknown as Record<string, unknown>).ffmpegCore = core;

      this.ffmpeg = core;

      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
      log.info(`Loaded in ${loadTime}s`);
    } catch (error) {
      log.error('Failed to load', error);
      throw new Error(`Failed to load FFmpeg: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Check if FFmpeg is loaded and ready
   */
  isLoaded(): boolean {
    return this.loadState === 'ready' && this.ffmpeg !== null;
  }

  /**
   * Get current load state
   */
  getLoadState(): LoadState {
    return this.loadState;
  }

  private handleLog(type: 'info' | 'warning' | 'error', message: string): void {
    const entry: FFmpegLogEntry = {
      type,
      message,
      timestamp: Date.now(),
    };
    this.logs.push(entry);
    this.onLog?.(entry);

    // Parse progress from FFmpeg output
    // FFmpeg outputs: "frame=  123 fps= 45 q=2.0 size=   1234kB time=00:00:04.10 ..."
    const frameMatch = message.match(/frame=\s*(\d+)/);
    const fpsMatch = message.match(/fps=\s*([\d.]+)/);
    const sizeMatch = message.match(/size=\s*(\d+)/);
    const timeMatch = message.match(/time=(\d+):(\d+):([\d.]+)/);

    if (frameMatch && this.onProgress && this.totalFrames > 0) {
      const frame = parseInt(frameMatch[1]);
      const percent = (frame / this.totalFrames) * 100;
      const fps = fpsMatch ? parseFloat(fpsMatch[1]) : 0;
      const size = sizeMatch ? parseInt(sizeMatch[1]) * 1024 : 0;

      let time = frame / 30; // Default estimate
      if (timeMatch) {
        const hours = parseInt(timeMatch[1]);
        const minutes = parseInt(timeMatch[2]);
        const seconds = parseFloat(timeMatch[3]);
        time = hours * 3600 + minutes * 60 + seconds;
      }

      log.debug(`Progress: frame=${frame}/${this.totalFrames} (${percent.toFixed(1)}%)`);

      this.onProgress({
        frame,
        fps,
        time,
        speed: fps / 30, // Approximate speed multiplier
        bitrate: 0,
        size,
        percent,
        eta: fps > 0 ? (this.totalFrames - frame) / fps : 0,
      });
    }
  }

  /**
   * Encode frames to video using FFmpeg
   * @param frames - Array of raw RGBA frame data
   * @param settings - Export settings (resolution, codec, etc.)
   * @param onProgress - Progress callback
   * @param audioBuffer - Optional mixed audio buffer to include
   */
  async encode(
    frames: Uint8Array[],
    settings: FFmpegExportSettings,
    onProgress?: (progress: FFmpegProgress) => void,
    audioBuffer?: AudioBuffer | null
  ): Promise<Blob> {
    if (!this.ffmpeg) {
      await this.load();
    }
    if (!this.ffmpeg) {
      throw new Error('FFmpeg not loaded');
    }

    this.cancelled = false;
    this.onProgress = onProgress;
    this.logs = [];
    this.totalFrames = frames.length;
    this.startTime = performance.now();

    const fs = this.ffmpeg.FS;

    try {
      // Clean up any previous files first to reset FFmpeg state
      this.cleanup();

      // Create directories (they may already exist)
      try { fs.mkdir('/input'); } catch { /* exists */ }
      try { fs.mkdir('/output'); } catch { /* exists */ }

      // Concatenate all frames into a single raw file
      // This bypasses the pattern matching issue in WASM FFmpeg's image2 demuxer
      const frameSize = frames[0].byteLength;
      const totalSize = frameSize * frames.length;
      log.debug(`Writing ${frames.length} frames (${(totalSize / 1024 / 1024).toFixed(1)} MB)...`);
      log.debug(`Frame size: ${frameSize} bytes (expected: ${settings.width * settings.height * 4})`);

      // Debug: check if frames have unique content
      if (frames.length >= 2) {
        let sameCount = 0;
        for (let i = 0; i < Math.min(100, frameSize); i++) {
          if (frames[0][i] === frames[1][i]) sameCount++;
        }
        log.debug(`Frame 0 vs 1: ${sameCount}/100 bytes identical (should be <90 if different)`);
      }

      const allFrames = new Uint8Array(totalSize);
      for (let i = 0; i < frames.length; i++) {
        if (this.cancelled) throw new Error('Cancelled');
        allFrames.set(frames[i], i * frameSize);
      }

      fs.writeFile('/input/frames.raw', allFrames);
      log.debug(`Wrote /input/frames.raw: ${allFrames.byteLength} bytes`);

      // Write audio if provided
      let hasAudio = false;
      if (audioBuffer && audioBuffer.length > 0) {
        const audioData = this.audioBufferToPCM(audioBuffer);
        fs.writeFile('/input/audio.raw', new Uint8Array(audioData.buffer));
        hasAudio = true;
        log.debug(`Audio: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz`);
      }

      // Build FFmpeg arguments (using single file input)
      const args = this.buildArgs(settings, frames.length, hasAudio ? audioBuffer : null);
      log.debug('Running: ffmpeg', args.join(' '));

      // NOTE: Do NOT call reset() here - it clears the virtual filesystem!
      // The files we just wrote would be deleted.

      // Execute FFmpeg (callMain is synchronous but may take a while)
      log.info('Starting FFmpeg encode...');
      const encodeStart = performance.now();
      const exitCode = this.ffmpeg.callMain(args);
      const encodeTime = ((performance.now() - encodeStart) / 1000).toFixed(2);
      log.info(`FFmpeg finished in ${encodeTime}s with exit code ${exitCode}`);

      if (exitCode !== 0) {
        log.error('FFmpeg logs:', this.logs.map(l => l.message).join('\n'));
        throw new Error(`FFmpeg exited with code ${exitCode}`);
      }

      // Read output file
      const outputPath = `/output/output.${settings.container}`;
      const data = fs.readFile(outputPath);
      log.info(`Output file size: ${(data.byteLength / 1024 / 1024).toFixed(2)} MB`);

      // Create a copy of the data to ensure it's a standard ArrayBuffer
      const buffer = new ArrayBuffer(data.byteLength);
      new Uint8Array(buffer).set(data);
      return new Blob([buffer], { type: this.getMimeType(settings.container) });
    } finally {
      this.cleanup();
    }
  }

  /**
   * Build FFmpeg command line arguments
   */
  private buildArgs(settings: FFmpegExportSettings, frameCount?: number, audioBuffer?: AudioBuffer | null): string[] {
    // Calculate duration from frame count for rawvideo input
    const duration = frameCount && settings.fps ? frameCount / settings.fps : undefined;

    const args: string[] = [
      '-nostdin',                      // Don't read from stdin (prevents prompt in browser)
      '-y',                            // Overwrite output
      '-v', 'info',                    // Verbose output for progress parsing
      '-stats',                        // Force progress stats output
      '-f', 'rawvideo',                // Input format for video
      '-pix_fmt', 'rgba',              // Input pixel format (from canvas)
      '-s', `${settings.width}x${settings.height}`,
      '-r', String(settings.fps),
      '-i', '/input/frames.raw',       // Video input
    ];

    // Add audio input if available
    if (audioBuffer) {
      args.push(
        '-f', 'f32le',                 // Raw PCM float32 little-endian
        '-ar', String(audioBuffer.sampleRate),
        '-ac', String(audioBuffer.numberOfChannels),
        '-i', '/input/audio.raw'       // Audio input
      );
    }

    // Video codec settings
    args.push(...this.buildVideoArgs(settings));

    // Audio codec settings
    if (audioBuffer) {
      args.push(...this.buildAudioArgs(settings));
    } else {
      args.push('-an'); // No audio
    }

    // Explicitly set output frame count/duration (critical for rawvideo input)
    if (frameCount) {
      args.push('-frames:v', String(frameCount));
    }
    if (duration) {
      args.push('-t', String(duration.toFixed(6)));
    }

    // Output file
    args.push(`/output/output.${settings.container}`);

    return args;
  }

  /**
   * Build audio codec arguments based on container
   * NOTE: ASYNCIFY build only has aac, flac, alac, pcm_s16le, pcm_s24le, pcm_f32le, ac3
   */
  private buildAudioArgs(settings: FFmpegExportSettings): string[] {
    const args: string[] = [];

    // Choose audio codec based on container
    switch (settings.container) {
      case 'mov':
        // AAC for MOV (widely compatible)
        args.push('-c:a', 'aac');
        args.push('-b:a', '256k');
        break;

      case 'mkv':
        // FLAC for MKV (lossless, well supported)
        args.push('-c:a', 'flac');
        break;

      case 'avi':
        // PCM for AVI (libmp3lame not available)
        args.push('-c:a', 'pcm_s16le');
        break;

      case 'mxf':
        // PCM for MXF (professional standard)
        args.push('-c:a', 'pcm_s16le');
        break;

      default:
        args.push('-c:a', 'aac');
        args.push('-b:a', '256k');
    }

    return args;
  }

  /**
   * Convert AudioBuffer to interleaved PCM Float32 data
   */
  private audioBufferToPCM(audioBuffer: AudioBuffer): Float32Array {
    const channels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const interleaved = new Float32Array(length * channels);

    // Get channel data
    const channelData: Float32Array[] = [];
    for (let ch = 0; ch < channels; ch++) {
      channelData.push(audioBuffer.getChannelData(ch));
    }

    // Interleave samples
    for (let i = 0; i < length; i++) {
      for (let ch = 0; ch < channels; ch++) {
        interleaved[i * channels + ch] = channelData[ch][i];
      }
    }

    return interleaved;
  }

  /**
   * Build video codec-specific arguments
   * NOTE: ASYNCIFY build only has prores_ks, dnxhd, ffv1, utvideo, mjpeg
   */
  private buildVideoArgs(settings: FFmpegExportSettings): string[] {
    const args: string[] = [];

    switch (settings.codec) {
      case 'prores':
        args.push('-c:v', 'prores_ks');
        args.push('-profile:v', this.getProResProfileNumber(settings.proresProfile || 'hq'));
        args.push('-pix_fmt', settings.proresProfile?.includes('4444') ? 'yuva444p10le' : 'yuv422p10le');
        args.push('-vendor', 'apl0');
        break;

      case 'dnxhd':
        args.push('-c:v', 'dnxhd');
        args.push('-profile:v', settings.dnxhrProfile || 'dnxhr_hq');
        if (settings.dnxhrProfile === 'dnxhr_444') {
          args.push('-pix_fmt', 'yuv444p10le');
        } else if (settings.dnxhrProfile === 'dnxhr_hqx') {
          args.push('-pix_fmt', 'yuv422p10le');
        } else {
          args.push('-pix_fmt', 'yuv422p');
        }
        break;

      case 'ffv1':
        args.push('-c:v', 'ffv1');
        args.push('-level', '3');
        args.push('-coder', '1');
        args.push('-context', '1');
        args.push('-slicecrc', '1');
        args.push('-pix_fmt', settings.pixelFormat || 'yuv444p10le');
        break;

      case 'utvideo':
        args.push('-c:v', 'utvideo');
        args.push('-pix_fmt', 'rgba');
        break;

      case 'mjpeg':
        args.push('-c:v', 'mjpeg');
        args.push('-q:v', String(settings.quality ?? 2));
        args.push('-pix_fmt', 'yuvj422p');
        break;

      default:
        // Fallback to mjpeg as it's most widely compatible
        log.warn(`Unknown codec "${settings.codec}", falling back to mjpeg`);
        args.push('-c:v', 'mjpeg');
        args.push('-q:v', '2');
        args.push('-pix_fmt', 'yuvj422p');
    }

    return args;
  }

  /**
   * Get ProRes profile number for FFmpeg
   */
  private getProResProfileNumber(profile: ProResProfile): string {
    const profiles: Record<ProResProfile, string> = {
      proxy: '0',
      lt: '1',
      standard: '2',
      hq: '3',
      '4444': '4',
      '4444xq': '5',
    };
    return profiles[profile] || '3';
  }

  /**
   * Get MIME type for container format
   */
  private getMimeType(container: string): string {
    const types: Record<string, string> = {
      mov: 'video/quicktime',
      mp4: 'video/mp4',
      mkv: 'video/x-matroska',
      webm: 'video/webm',
      avi: 'video/x-msvideo',
      mxf: 'application/mxf',
    };
    return types[container] || 'video/mp4';
  }

  /**
   * Clean up virtual filesystem
   */
  private cleanup(): void {
    if (!this.ffmpeg) return;
    const fs = this.ffmpeg.FS;

    try {
      // Clean input files
      const inputFiles = fs.readdir('/input');
      for (const file of inputFiles) {
        if (file !== '.' && file !== '..') {
          try { fs.unlink(`/input/${file}`); } catch { /* ignore */ }
        }
      }

      // Clean output files
      const outputFiles = fs.readdir('/output');
      for (const file of outputFiles) {
        if (file !== '.' && file !== '..') {
          try { fs.unlink(`/output/${file}`); } catch { /* ignore */ }
        }
      }
    } catch (e) {
      log.warn('Cleanup error', e);
    }
  }

  /**
   * Cancel current export
   */
  cancel(): void {
    this.cancelled = true;
  }

  /**
   * Get accumulated logs
   */
  getLogs(): FFmpegLogEntry[] {
    return [...this.logs];
  }

  /**
   * Extract audio from a video file
   * Returns audio as AAC in M4A container for fast loading
   */
  async extractAudio(
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<Blob | null> {
    if (!this.ffmpeg) {
      throw new Error('FFmpeg not loaded. Call load() first.');
    }

    this.cancelled = false;
    this.logs = [];

    const fs = this.ffmpeg.FS;

    try {
      // Create directories
      try { fs.mkdir('/input'); } catch { /* exists */ }
      try { fs.mkdir('/output'); } catch { /* exists */ }

      // Write input file to virtual filesystem
      log.info(`Loading ${file.name} for audio extraction...`);
      onProgress?.(5);

      const inputData = new Uint8Array(await file.arrayBuffer());
      const inputPath = `/input/${file.name}`;
      fs.writeFile(inputPath, inputData);

      onProgress?.(20);

      // Build FFmpeg arguments for audio extraction
      // -vn: no video, -acodec aac: encode to AAC, -b:a 192k: 192kbps bitrate
      const args = [
        '-y',                    // Overwrite output
        '-i', inputPath,         // Input file
        '-vn',                   // No video
        '-acodec', 'aac',        // Encode to AAC
        '-b:a', '192k',          // 192 kbps bitrate
        '-ar', '48000',          // 48kHz sample rate
        '-ac', '2',              // Stereo
        '/output/audio.m4a',     // Output file
      ];

      log.debug('Extracting audio: ffmpeg', args.join(' '));
      onProgress?.(30);

      // Reset state before running
      if (this.ffmpeg.reset) {
        this.ffmpeg.reset();
      }

      // Execute FFmpeg
      const exitCode = this.ffmpeg.callMain(args);

      onProgress?.(90);

      if (exitCode !== 0) {
        log.warn('Audio extraction failed with code', exitCode);
        return null;
      }

      // Read output file
      const data = fs.readFile('/output/audio.m4a');

      // Create a copy to ensure it's a standard ArrayBuffer
      const buffer = new ArrayBuffer(data.byteLength);
      new Uint8Array(buffer).set(data);

      onProgress?.(100);

      log.info(`Audio extracted: ${(data.byteLength / 1024 / 1024).toFixed(2)} MB`);
      return new Blob([buffer], { type: 'audio/mp4' });
    } catch (error) {
      log.error('Audio extraction error', error);
      return null;
    } finally {
      this.cleanup();
    }
  }

  /**
   * Set log callback
   */
  setLogCallback(callback: (log: FFmpegLogEntry) => void): void {
    this.onLog = callback;
  }

  /**
   * Terminate FFmpeg instance (free resources)
   */
  terminate(): void {
    if (this.ffmpeg) {
      // Core doesn't have terminate, just clear reference
      this.ffmpeg = null;
      this.loadState = 'idle';
      this.loadPromise = null;
    }
  }

  /**
   * Check which codecs are available in the loaded FFmpeg build
   * NOTE: ASYNCIFY build includes only native FFmpeg encoders
   */
  async getAvailableCodecs(): Promise<FFmpegVideoCodec[]> {
    // ASYNCIFY build with native encoders only
    // External libs (libx264, libvpx, libsnappy) require pkg-config which fails in Emscripten
    return [
      'prores',    // Apple ProRes (prores_ks)
      'dnxhd',     // Avid DNxHR
      'ffv1',      // FFV1 lossless
      'utvideo',   // UTVideo lossless
      'mjpeg',     // Motion JPEG
    ];
  }
}

// Singleton instance with HMR support
let instance: FFmpegBridge | null = null;

// HMR handling
if (import.meta.hot) {
  import.meta.hot.accept();
  if (import.meta.hot.data?.ffmpegBridge) {
    instance = import.meta.hot.data.ffmpegBridge;
  }
  import.meta.hot.dispose((data) => {
    data.ffmpegBridge = instance;
  });
}

/**
 * Get singleton FFmpegBridge instance
 */
export function getFFmpegBridge(): FFmpegBridge {
  if (!instance) {
    instance = new FFmpegBridge();
  }
  return instance;
}
