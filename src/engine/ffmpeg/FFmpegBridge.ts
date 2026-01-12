// src/engine/ffmpeg/FFmpegBridge.ts
// Bridge to FFmpeg WASM for professional codec export

import type {
  FFmpegExportSettings,
  FFmpegProgress,
  FFmpegLogEntry,
  ProResProfile,
  FFmpegVideoCodec,
} from './types';

// FFmpeg WASM core interface (from @ffmpeg/ffmpeg)
interface FFmpegCore {
  FS: {
    writeFile: (path: string, data: Uint8Array) => void;
    readFile: (path: string) => Uint8Array;
    unlink: (path: string) => void;
    readdir: (path: string) => string[];
    mkdir: (path: string) => void;
  };
  exec: (args: string[]) => Promise<number>;
  on: (event: string, callback: (data: unknown) => void) => void;
  terminate: () => void;
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
    console.log('[FFmpegBridge] Loading FFmpeg WASM...');
    const startTime = performance.now();

    try {
      // Dynamic import of @ffmpeg/ffmpeg
      const { FFmpeg } = await import('@ffmpeg/ffmpeg');
      const { toBlobURL } = await import('@ffmpeg/util');

      const ffmpeg = new FFmpeg();

      // Set up logging
      ffmpeg.on('log', ({ message }: { message: string }) => {
        this.handleLog('info', message);
      });

      // Set up progress
      ffmpeg.on('progress', ({ progress, time }: { progress: number; time: number }) => {
        if (this.onProgress && this.totalFrames > 0) {
          const elapsed = (performance.now() - this.startTime) / 1000;
          const speed = elapsed > 0 ? progress / elapsed : 0;
          const remaining = speed > 0 ? (1 - progress) / speed : 0;

          this.onProgress({
            frame: Math.floor(progress * this.totalFrames),
            fps: speed * 30, // Approximate
            time,
            speed,
            bitrate: 0,
            size: 0,
            percent: progress * 100,
            eta: remaining,
          });
        }
      });

      // Load FFmpeg core from CDN
      // Use single-threaded version if SharedArrayBuffer not available
      const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
      const baseURL = hasSharedArrayBuffer
        ? 'https://unpkg.com/@ffmpeg/core-mt@0.12.6/dist/esm'  // Multi-threaded (faster)
        : 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm';     // Single-threaded (compatible)

      console.log(`[FFmpegBridge] Using ${hasSharedArrayBuffer ? 'multi-threaded' : 'single-threaded'} core`);

      const loadConfig: Record<string, unknown> = {
        coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
        wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
      };

      // Multi-threaded version needs worker
      if (hasSharedArrayBuffer) {
        loadConfig.workerURL = await toBlobURL(`${baseURL}/ffmpeg-core.worker.js`, 'text/javascript');
      }

      await ffmpeg.load(loadConfig);

      this.ffmpeg = ffmpeg as unknown as FFmpegCore;

      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
      console.log(`[FFmpegBridge] Loaded in ${loadTime}s`);
    } catch (error) {
      console.error('[FFmpegBridge] Failed to load:', error);
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

    // Parse progress from FFmpeg output if not getting progress events
    const frameMatch = message.match(/frame=\s*(\d+)/);
    if (frameMatch && this.onProgress && this.totalFrames > 0) {
      const frame = parseInt(frameMatch[1]);
      const percent = (frame / this.totalFrames) * 100;

      this.onProgress({
        frame,
        fps: 0,
        time: frame / 30,
        speed: 0,
        bitrate: 0,
        size: 0,
        percent,
        eta: 0,
      });
    }
  }

  /**
   * Encode frames to video using FFmpeg
   */
  async encode(
    frames: Uint8Array[],
    settings: FFmpegExportSettings,
    onProgress?: (progress: FFmpegProgress) => void
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
      // Create directories
      try { fs.mkdir('/input'); } catch { /* exists */ }
      try { fs.mkdir('/output'); } catch { /* exists */ }

      // Write frames to virtual filesystem
      console.log(`[FFmpegBridge] Writing ${frames.length} frames...`);
      for (let i = 0; i < frames.length; i++) {
        if (this.cancelled) throw new Error('Cancelled');
        const filename = `/input/frame_${String(i).padStart(6, '0')}.raw`;
        fs.writeFile(filename, frames[i]);

        // Report write progress
        if (onProgress && i % 10 === 0) {
          onProgress({
            frame: i,
            fps: 0,
            time: 0,
            speed: 0,
            bitrate: 0,
            size: 0,
            percent: (i / frames.length) * 10, // First 10% is writing
            eta: 0,
          });
        }
      }

      // Build FFmpeg arguments
      const args = this.buildArgs(settings);
      console.log('[FFmpegBridge] Running: ffmpeg', args.join(' '));

      // Execute FFmpeg
      const exitCode = await this.ffmpeg.exec(args);
      if (exitCode !== 0) {
        throw new Error(`FFmpeg exited with code ${exitCode}`);
      }

      // Read output file
      const outputPath = `/output/output.${settings.container}`;
      const data = fs.readFile(outputPath);

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
  private buildArgs(settings: FFmpegExportSettings): string[] {
    const args: string[] = [
      '-y',                            // Overwrite output
      '-f', 'rawvideo',                // Input format
      '-pix_fmt', 'rgba',              // Input pixel format (from canvas)
      '-s', `${settings.width}x${settings.height}`,
      '-r', String(settings.fps),
      '-i', '/input/frame_%06d.raw',   // Input pattern
    ];

    // Video codec settings
    args.push(...this.buildVideoArgs(settings));

    // Audio (none for now - frames only)
    args.push('-an');

    // Output file
    args.push(`/output/output.${settings.container}`);

    return args;
  }

  /**
   * Build video codec-specific arguments
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

      case 'hap':
        args.push('-c:v', 'hap');
        args.push('-format', settings.hapFormat || 'hap');
        args.push('-compressor', settings.hapCompressor || 'snappy');
        args.push('-chunks', String(settings.hapChunks || 4));
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

      case 'libx264':
        args.push('-c:v', 'libx264');
        args.push('-preset', 'medium');
        args.push('-pix_fmt', settings.pixelFormat || 'yuv420p');
        if (settings.quality !== undefined) {
          args.push('-crf', String(settings.quality));
        } else if (settings.bitrate) {
          args.push('-b:v', String(settings.bitrate));
        } else {
          args.push('-crf', '18');
        }
        break;

      case 'libx265':
        args.push('-c:v', 'libx265');
        args.push('-preset', 'medium');
        args.push('-pix_fmt', settings.pixelFormat || 'yuv420p');
        if (settings.quality !== undefined) {
          args.push('-crf', String(settings.quality));
        } else {
          args.push('-crf', '22');
        }
        break;

      case 'libvpx_vp9':
        args.push('-c:v', 'libvpx-vp9');
        args.push('-pix_fmt', settings.pixelFormat || 'yuv420p');
        if (settings.quality !== undefined) {
          args.push('-crf', String(settings.quality));
          args.push('-b:v', '0');
        } else if (settings.bitrate) {
          args.push('-b:v', String(settings.bitrate));
        }
        args.push('-row-mt', '1'); // Enable row-based multithreading
        break;

      case 'libsvtav1':
        args.push('-c:v', 'libsvtav1');
        args.push('-preset', '6');
        args.push('-pix_fmt', settings.pixelFormat || 'yuv420p');
        if (settings.quality !== undefined) {
          args.push('-crf', String(settings.quality));
        } else {
          args.push('-crf', '30');
        }
        break;

      default:
        // Fallback to codec name
        args.push('-c:v', settings.codec);
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
      console.warn('[FFmpegBridge] Cleanup error:', e);
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
      try {
        this.ffmpeg.terminate();
      } catch { /* ignore */ }
      this.ffmpeg = null;
      this.loadState = 'idle';
      this.loadPromise = null;
    }
  }

  /**
   * Check which codecs are available in the loaded FFmpeg build
   */
  async getAvailableCodecs(): Promise<FFmpegVideoCodec[]> {
    // The standard @ffmpeg/ffmpeg build includes:
    // - libx264 (H.264)
    // - libvpx (VP8/VP9)
    // Custom builds may add more
    const standardCodecs: FFmpegVideoCodec[] = [
      'libx264',
      'libvpx_vp9',
      'mjpeg',
    ];

    // Professional codecs require custom WASM build
    // These will fail with standard build but we list them for UI
    const professionalCodecs: FFmpegVideoCodec[] = [
      'prores',
      'hap',
      'dnxhd',
      'ffv1',
      'utvideo',
      'libx265',
      'libsvtav1',
    ];

    return [...standardCodecs, ...professionalCodecs];
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
