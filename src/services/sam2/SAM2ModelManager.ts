// SAM 2 Model Manager — handles download, OPFS caching, and ONNX session loading

import { Logger } from '../logger';
import type { SAM2ModelFile } from './types';

const log = Logger.create('SAM2ModelManager');

// Model file definitions
const MODEL_DIR = 'sam2-models';

// Models from webgpu-sam2 project CDN — ORT-optimized encoder + ONNX decoder
// These are proven to work in-browser with onnxruntime-web
// Encoder uses .ort format (ORT-optimized, same size, faster load)
// Decoder uses .onnx format (doesn't convert well to .ort)
const MODEL_FILES: SAM2ModelFile[] = [
  {
    name: 'sam2_hiera_small.encoder.ort',
    url: 'https://sam2-model-download.b-cdn.net/sam2_hiera_small.encoder.with_runtime_opt.ort',
    fallbackUrl: 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_small.encoder.ort',
    sizeBytes: 163_000_000, // ~163 MB
  },
  {
    name: 'sam2_hiera_small.decoder.onnx',
    url: 'https://sam2-model-download.b-cdn.net/sam2_hiera_small.decoder.onnx',
    fallbackUrl: 'https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_small.decoder.onnx',
    sizeBytes: 21_000_000, // ~21 MB
  },
];

const TOTAL_SIZE = MODEL_FILES.reduce((sum, f) => sum + f.sizeBytes, 0);

export class SAM2ModelManager {
  private opfsRoot: FileSystemDirectoryHandle | null = null;
  private modelDir: FileSystemDirectoryHandle | null = null;

  /** Check if models are already cached in OPFS */
  async isModelCached(): Promise<boolean> {
    try {
      const dir = await this.getModelDir();
      for (const file of MODEL_FILES) {
        try {
          const fileHandle = await dir.getFileHandle(file.name);
          const f = await fileHandle.getFile();
          // Check file has reasonable size (not empty/corrupt)
          if (f.size < file.sizeBytes * 0.9) {
            log.warn(`Cached file ${file.name} is too small (${f.size} bytes), re-download needed`);
            return false;
          }
        } catch {
          return false;
        }
      }
      return true;
    } catch {
      return false;
    }
  }

  /** Download models to OPFS with progress callback */
  async downloadModels(onProgress: (progress: number) => void): Promise<void> {
    // Clear any stale/incompatible cached models first
    await this.clearCache();
    const dir = await this.getModelDir();

    // Request persistent storage
    if (navigator.storage?.persist) {
      const persisted = await navigator.storage.persist();
      log.info(`Storage persistence: ${persisted ? 'granted' : 'denied'}`);
    }

    let totalDownloaded = 0;

    for (const file of MODEL_FILES) {
      log.info(`Downloading ${file.name} (~${Math.round(file.sizeBytes / 1024 / 1024)} MB)...`);

      const downloaded = await this.downloadFile(file, dir, (fileBytes) => {
        const overallProgress = ((totalDownloaded + fileBytes) / TOTAL_SIZE) * 100;
        onProgress(Math.min(overallProgress, 99.9));
      });

      totalDownloaded += downloaded;
    }

    onProgress(100);
    log.info('All SAM 2 models downloaded and cached');
  }

  /** Load model buffers from OPFS cache */
  async loadModelBuffers(): Promise<{ encoderBuffer: ArrayBuffer; decoderBuffer: ArrayBuffer }> {
    const dir = await this.getModelDir();

    const [encoderHandle, decoderHandle] = await Promise.all([
      dir.getFileHandle(MODEL_FILES[0].name),
      dir.getFileHandle(MODEL_FILES[1].name),
    ]);

    const [encoderFile, decoderFile] = await Promise.all([
      encoderHandle.getFile(),
      decoderHandle.getFile(),
    ]);

    const [encoderBuffer, decoderBuffer] = await Promise.all([
      encoderFile.arrayBuffer(),
      decoderFile.arrayBuffer(),
    ]);

    log.info(`Loaded encoder: ${Math.round(encoderBuffer.byteLength / 1024 / 1024)} MB, decoder: ${Math.round(decoderBuffer.byteLength / 1024 / 1024)} MB`);
    return { encoderBuffer, decoderBuffer };
  }

  /** Delete cached models */
  async clearCache(): Promise<void> {
    try {
      const root = await this.getOPFSRoot();
      await root.removeEntry(MODEL_DIR, { recursive: true });
      this.modelDir = null;
      log.info('SAM 2 model cache cleared');
    } catch (e) {
      log.warn('Failed to clear cache', e);
    }
  }

  // --- Private helpers ---

  private async getOPFSRoot(): Promise<FileSystemDirectoryHandle> {
    if (!this.opfsRoot) {
      this.opfsRoot = await navigator.storage.getDirectory();
    }
    return this.opfsRoot;
  }

  private async getModelDir(): Promise<FileSystemDirectoryHandle> {
    if (!this.modelDir) {
      const root = await this.getOPFSRoot();
      this.modelDir = await root.getDirectoryHandle(MODEL_DIR, { create: true });
    }
    return this.modelDir;
  }

  /** Stream-download a single file to OPFS, returning bytes downloaded */
  private async downloadFile(
    file: SAM2ModelFile,
    dir: FileSystemDirectoryHandle,
    onBytesProgress: (bytesDownloaded: number) => void
  ): Promise<number> {
    let response: Response;
    try {
      response = await fetch(file.url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    } catch (e) {
      log.warn(`Primary URL failed for ${file.name}, trying fallback...`, e);
      response = await fetch(file.fallbackUrl);
      if (!response.ok) {
        throw new Error(`Failed to download ${file.name}: HTTP ${response.status}`);
      }
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('ReadableStream not supported');

    const fileHandle = await dir.getFileHandle(file.name, { create: true });
    const writable = await fileHandle.createWritable();

    let downloaded = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        await writable.write(value);
        downloaded += value.byteLength;
        onBytesProgress(downloaded);
      }
    } finally {
      await writable.close();
    }

    log.info(`Downloaded ${file.name}: ${Math.round(downloaded / 1024 / 1024)} MB`);
    return downloaded;
  }
}

// --- HMR-safe singleton ---

let instance: SAM2ModelManager | null = null;

if (import.meta.hot) {
  import.meta.hot.accept();
  if (import.meta.hot.data?.sam2ModelManager) {
    instance = import.meta.hot.data.sam2ModelManager;
  }
  import.meta.hot.dispose((data) => {
    data.sam2ModelManager = instance;
  });
}

export function getSAM2ModelManager(): SAM2ModelManager {
  if (!instance) {
    instance = new SAM2ModelManager();
  }
  return instance;
}
