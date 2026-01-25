// PiAPI Service - Unified API for multiple AI video generation models
// Supports: Kling, Luma, Veo, Sora2, Wanx, Hailuo, SkyReels, Hunyuan, etc.
// Docs: https://piapi.ai/docs/overview

import { Logger } from './logger';

const log = Logger.create('PiAPI');

const BASE_URL = 'https://api.piapi.ai';
const MODELS_CACHE_KEY = 'piapi-video-models';
const MODELS_CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours

// Provider configuration type
export interface VideoProvider {
  id: string;
  name: string;
  description: string;
  versions: string[];
  supportedModes: string[];
  supportedDurations: number[];
  supportedAspectRatios: string[];
  supportsImageToVideo: boolean;
  supportsTextToVideo: boolean;
}

// Default available AI video providers/models
const DEFAULT_VIDEO_PROVIDERS: VideoProvider[] = [
  {
    id: 'kling',
    name: 'Kling AI',
    description: 'High quality, native audio (v2.6)',
    versions: ['2.6', '2.5', '2.1', '2.1-master', '1.6', '1.5'],
    supportedModes: ['std', 'pro'],
    supportedDurations: [5, 10],
    supportedAspectRatios: ['16:9', '9:16', '1:1'],
    supportsImageToVideo: true,
    supportsTextToVideo: true,
  },
  {
    id: 'luma',
    name: 'Luma Dream Machine',
    description: 'Cinematic quality',
    versions: ['1.5', '1.6', '2.0'],
    supportedModes: ['std'],
    supportedDurations: [5],
    supportedAspectRatios: ['16:9', '9:16', '1:1'],
    supportsImageToVideo: true,
    supportsTextToVideo: true,
  },
  {
    id: 'hailuo',
    name: 'Hailuo (MiniMax)',
    description: 'Fast generation',
    versions: ['1.0'],
    supportedModes: ['std'],
    supportedDurations: [5],
    supportedAspectRatios: ['16:9', '9:16', '1:1'],
    supportsImageToVideo: true,
    supportsTextToVideo: true,
  },
  {
    id: 'hunyuan',
    name: 'Hunyuan',
    description: 'Tencent model',
    versions: ['1.0'],
    supportedModes: ['std'],
    supportedDurations: [5],
    supportedAspectRatios: ['16:9', '9:16', '1:1'],
    supportsImageToVideo: true,
    supportsTextToVideo: true,
  },
  {
    id: 'wanx',
    name: 'Wanx (Wan)',
    description: 'Alibaba model',
    versions: ['2.6', '2.1', '1.3'],
    supportedModes: ['std'],
    supportedDurations: [5],
    supportedAspectRatios: ['16:9', '9:16', '1:1'],
    supportsImageToVideo: true,
    supportsTextToVideo: true,
  },
  {
    id: 'skyreels',
    name: 'SkyReels',
    description: 'AI video generation',
    versions: ['1.0'],
    supportedModes: ['std'],
    supportedDurations: [5],
    supportedAspectRatios: ['16:9'],
    supportsImageToVideo: true,
    supportsTextToVideo: true,
  },
];

// Mutable list that can be updated
let VIDEO_PROVIDERS: VideoProvider[] = [...DEFAULT_VIDEO_PROVIDERS];

// Load cached models from localStorage
function loadCachedModels(): VideoProvider[] | null {
  try {
    const cached = localStorage.getItem(MODELS_CACHE_KEY);
    if (cached) {
      const { models, timestamp } = JSON.parse(cached);
      if (Date.now() - timestamp < MODELS_CACHE_EXPIRY) {
        return models;
      }
    }
  } catch (e) {
    log.warn(' Failed to load cached models:', e);
  }
  return null;
}

// Save models to localStorage
function saveCachedModels(models: VideoProvider[]) {
  try {
    localStorage.setItem(MODELS_CACHE_KEY, JSON.stringify({
      models,
      timestamp: Date.now(),
    }));
  } catch (e) {
    log.warn(' Failed to cache models:', e);
  }
}

// Get current providers list
export function getVideoProviders(): VideoProvider[] {
  // Try to load from cache on first access
  const cached = loadCachedModels();
  if (cached && cached.length > 0) {
    VIDEO_PROVIDERS = cached;
  }
  return VIDEO_PROVIDERS;
}

// Update providers list (e.g., from manual refresh or API)
export function updateVideoProviders(providers: VideoProvider[]) {
  VIDEO_PROVIDERS = providers;
  saveCachedModels(providers);
}

// Reset to default providers
export function resetToDefaultProviders() {
  VIDEO_PROVIDERS = [...DEFAULT_VIDEO_PROVIDERS];
  localStorage.removeItem(MODELS_CACHE_KEY);
}

// Pricing per credit (approximate, based on PiAPI pay-as-you-go)
// Credits vary by model, mode, and duration
export const PRICING: Record<string, Record<string, Record<number, number>>> = {
  'kling': {
    'std': { 5: 0.14, 10: 0.28 },   // ~$0.14 per 5s std
    'pro': { 5: 0.28, 10: 0.56 },   // ~$0.28 per 5s pro
  },
  'luma': {
    'std': { 5: 0.20, 10: 0.40 },
  },
  'hailuo': {
    'std': { 5: 0.15, 10: 0.30 },
  },
  'hunyuan': {
    'std': { 5: 0.12, 10: 0.24 },
  },
  'wanx': {
    'std': { 5: 0.10, 10: 0.20 },
  },
  'skyreels': {
    'std': { 5: 0.15, 10: 0.30 },
  },
};

// Calculate cost for a generation
export function calculateCost(provider: string, mode: string, duration: number): number {
  const providerPricing = PRICING[provider];
  if (!providerPricing) return 0.20; // Default estimate
  const modePricing = providerPricing[mode];
  if (!modePricing) return 0.20;
  return modePricing[duration] ?? 0.20;
}

// Get provider info by ID
export function getProvider(providerId: string) {
  return VIDEO_PROVIDERS.find(p => p.id === providerId);
}

// Task status types
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface VideoTask {
  id: string;
  status: TaskStatus;
  progress?: number;
  videoUrl?: string;
  error?: string;
  createdAt: Date;
  completedAt?: Date;
}

export interface TextToVideoParams {
  provider: string;
  version: string;
  prompt: string;
  negativePrompt?: string;
  duration: number;
  aspectRatio: string;
  mode: string;
  cfgScale?: number;
}

export interface ImageToVideoParams {
  provider: string;
  version: string;
  prompt: string;
  negativePrompt?: string;
  startImageUrl?: string;  // Base64 or URL
  endImageUrl?: string;    // Base64 or URL (Kling only)
  duration: number;
  aspectRatio?: string;
  mode: string;
  cfgScale?: number;
}

export interface AccountInfo {
  accountName: string;
  accountId: string;
  credits: number;
  creditsUsd: number;
}

interface PiApiResponse {
  code: number;
  message: string;
  data: {
    task_id: string;
    status?: string;
    output?: {
      video_url?: string;
      works?: Array<{ video?: { url: string } }>;
    };
    error?: {
      message?: string;
    };
  };
}

class PiApiService {
  private apiKey: string = '';

  setApiKey(apiKey: string) {
    this.apiKey = apiKey;
  }

  hasApiKey(): boolean {
    return !!this.apiKey;
  }

  // Convert data URL to Blob
  private dataUrlToBlob(dataUrl: string): Blob {
    const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
    if (!match) {
      throw new Error('Invalid data URL');
    }
    const mimeType = match[1];
    const base64 = match[2];
    const binary = atob(base64);
    const array = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      array[i] = binary.charCodeAt(i);
    }
    return new Blob([array], { type: mimeType });
  }

  // Upload image using catbox.moe litterbox (temporary file host, supports CORS)
  private async uploadToLitterbox(dataUrl: string): Promise<string> {
    const blob = this.dataUrlToBlob(dataUrl);
    const formData = new FormData();
    formData.append('reqtype', 'fileupload');
    formData.append('time', '1h'); // File expires in 1 hour (enough for video generation)
    formData.append('fileToUpload', blob, 'image.jpg');

    log.debug(`Uploading image to litterbox.catbox.moe, size: ${Math.round(blob.size / 1024)} KB`);

    const response = await fetch('https://litterbox.catbox.moe/resources/internals/api.php', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Litterbox upload failed: ${response.status}`);
    }

    // Litterbox returns the URL as plain text
    const url = await response.text();
    log.debug(' Litterbox response:', url);

    if (!url.startsWith('http')) {
      throw new Error(url || 'Litterbox upload failed');
    }

    return url.trim();
  }

  // Main upload method - uses litterbox.catbox.moe
  private async uploadImage(dataUrl: string): Promise<string> {
    // Try litterbox.catbox.moe first (free, supports CORS, 1 hour expiry)
    try {
      return await this.uploadToLitterbox(dataUrl);
    } catch (err) {
      log.warn(' Litterbox upload failed:', err);
      throw new Error('Failed to upload image: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  }

  // Compress image to reduce size for inline base64 (max ~1MB target)
  private async compressImage(dataUrl: string, maxWidth = 1280, quality = 0.8): Promise<string> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        // Calculate new dimensions
        let width = img.width;
        let height = img.height;

        if (width > maxWidth) {
          height = Math.round(height * (maxWidth / width));
          width = maxWidth;
        }

        // Create canvas and draw resized image
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        ctx.drawImage(img, 0, 0, width, height);

        // Convert to JPEG with compression
        const compressed = canvas.toDataURL('image/jpeg', quality);
        const sizeKB = Math.round((compressed.length * 0.75) / 1024);
        log.debug(`Compressed image: ${img.width}x${img.height} -> ${width}x${height}, ~${sizeKB}KB`);

        resolve(compressed);
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = dataUrl;
    });
  }

  private async request<T = PiApiResponse>(
    endpoint: string,
    method: 'GET' | 'POST' = 'GET',
    body?: object
  ): Promise<T> {
    if (!this.hasApiKey()) {
      throw new Error('PiAPI key not set');
    }

    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.apiKey,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    const responseText = await response.text();
    let result: T;

    try {
      result = JSON.parse(responseText) as T;
    } catch {
      log.error(' Failed to parse response:', responseText);
      throw new Error(`PiAPI error: ${response.status} - Invalid JSON response`);
    }

    if (!response.ok) {
      log.error(' API error:', result);
      const errorMsg = (result as PiApiResponse).message || responseText;
      throw new Error(`PiAPI error: ${response.status} - ${errorMsg}`);
    }

    const apiResult = result as PiApiResponse;
    if (apiResult.code !== 200 && apiResult.code !== 0) {
      log.error(' API returned error code:', result);
      throw new Error(`PiAPI error: ${apiResult.message}`);
    }

    return result;
  }

  // Build request body based on provider
  private buildRequestBody(
    provider: string,
    version: string,
    _taskType: 'text_to_video' | 'image_to_video',
    params: {
      prompt: string;
      negativePrompt?: string;
      duration: number;
      aspectRatio?: string;
      mode: string;
      cfgScale?: number;
      imageUrl?: string;
      imageTailUrl?: string;
    }
  ): object {
    // Kling uses a specific format
    if (provider === 'kling') {
      const input: Record<string, unknown> = {
        prompt: params.prompt,
        duration: params.duration,
        aspect_ratio: params.aspectRatio || '16:9',
        mode: params.mode,
        version: version,
      };

      if (params.negativePrompt) {
        input.negative_prompt = params.negativePrompt;
      }
      if (params.cfgScale !== undefined) {
        input.cfg_scale = String(params.cfgScale);
      }
      if (params.imageUrl) {
        input.image_url = params.imageUrl;
      }
      if (params.imageTailUrl) {
        input.image_tail_url = params.imageTailUrl;
      }

      const body = {
        model: 'kling',
        task_type: 'video_generation',
        input,
        config: {
          service_mode: 'public',
        },
      };
      log.debug(' Request body:', JSON.stringify(body, null, 2));
      return body;
    }

    // Luma Dream Machine
    if (provider === 'luma') {
      const input: Record<string, unknown> = {
        prompt: params.prompt,
        aspect_ratio: params.aspectRatio || '16:9',
      };

      if (params.imageUrl) {
        input.image_url = params.imageUrl;
      }

      return {
        model: 'luma',
        task_type: 'video_generation',
        input,
        config: {
          service_mode: 'public',
        },
      };
    }

    // Hailuo (MiniMax)
    if (provider === 'hailuo') {
      return {
        model: 'hailuo',
        task_type: 'video_generation',
        input: {
          prompt: params.prompt,
          image_url: params.imageUrl,
        },
        config: {
          service_mode: 'public',
        },
      };
    }

    // Wanx (Alibaba Wan)
    if (provider === 'wanx') {
      return {
        model: 'wanx',
        task_type: 'video_generation',
        input: {
          prompt: params.prompt,
          version: version,
          image_url: params.imageUrl,
        },
        config: {
          service_mode: 'public',
        },
      };
    }

    // Generic fallback
    return {
      model: provider,
      task_type: 'video_generation',
      input: {
        prompt: params.prompt,
        image_url: params.imageUrl,
      },
      config: {
        service_mode: 'public',
      },
    };
  }

  async createTextToVideo(params: TextToVideoParams): Promise<string> {
    const body = this.buildRequestBody(
      params.provider,
      params.version,
      'text_to_video',
      {
        prompt: params.prompt,
        negativePrompt: params.negativePrompt,
        duration: params.duration,
        aspectRatio: params.aspectRatio,
        mode: params.mode,
        cfgScale: params.cfgScale,
      }
    );

    log.debug(' Creating text-to-video task:', {
      provider: params.provider,
      version: params.version,
      duration: params.duration,
    });

    const result = await this.request<PiApiResponse>('/api/v1/task', 'POST', body);
    return result.data.task_id;
  }

  async createImageToVideo(params: ImageToVideoParams): Promise<string> {
    // Upload images to get hosted URLs (PiAPI requires URLs, not base64)
    let imageUrl: string | undefined;
    let imageTailUrl: string | undefined;

    if (params.startImageUrl) {
      log.debug(' Compressing and uploading start image...');
      const compressed = await this.compressImage(params.startImageUrl);
      imageUrl = await this.uploadImage(compressed);
    }

    if (params.endImageUrl) {
      log.debug(' Compressing and uploading end image...');
      const compressed = await this.compressImage(params.endImageUrl);
      imageTailUrl = await this.uploadImage(compressed);
    }

    log.debug(' Creating image-to-video with uploaded URLs:', { imageUrl, imageTailUrl });

    const body = this.buildRequestBody(
      params.provider,
      params.version,
      'image_to_video',
      {
        prompt: params.prompt,
        negativePrompt: params.negativePrompt,
        duration: params.duration,
        aspectRatio: params.aspectRatio,
        mode: params.mode,
        cfgScale: params.cfgScale,
        imageUrl,
        imageTailUrl,
      }
    );

    log.debug(' Creating image-to-video task:', {
      provider: params.provider,
      version: params.version,
      duration: params.duration,
      hasImage: !!imageUrl,
      hasImageTail: !!imageTailUrl,
    });

    const result = await this.request<PiApiResponse>('/api/v1/task', 'POST', body);
    return result.data.task_id;
  }

  async getTaskStatus(taskId: string): Promise<VideoTask> {
    const result = await this.request<PiApiResponse>(`/api/v1/task/${taskId}`, 'GET');

    let status: TaskStatus = 'pending';
    const taskStatus = result.data.status?.toLowerCase() || '';

    if (taskStatus === 'completed' || taskStatus === 'success' || taskStatus === 'succeeded') {
      status = 'completed';
    } else if (taskStatus === 'processing' || taskStatus === 'running' || taskStatus === 'pending') {
      status = 'processing';
    } else if (taskStatus === 'failed' || taskStatus === 'error') {
      status = 'failed';
    }

    const task: VideoTask = {
      id: taskId,
      status,
      error: result.data.error?.message,
      createdAt: new Date(),
    };

    // Extract video URL from various response formats
    if (status === 'completed') {
      const output = result.data.output;
      if (output?.video_url) {
        task.videoUrl = output.video_url;
      } else if (output?.works?.[0]?.video?.url) {
        task.videoUrl = output.works[0].video.url;
      }
      task.completedAt = new Date();
    }

    return task;
  }

  async pollTaskUntilComplete(
    taskId: string,
    onProgress?: (task: VideoTask) => void,
    pollInterval = 5000,
    timeout = 600000 // 10 minutes
  ): Promise<VideoTask> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const task = await this.getTaskStatus(taskId);

      if (onProgress) {
        onProgress(task);
      }

      if (task.status === 'completed' || task.status === 'failed') {
        return task;
      }

      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error('Task timed out after 10 minutes');
  }

  // Get account info including balance
  async getAccountInfo(): Promise<AccountInfo> {
    if (!this.hasApiKey()) {
      throw new Error('PiAPI key not set');
    }

    const response = await fetch(`${BASE_URL}/account/info`, {
      method: 'GET',
      headers: {
        'x-api-key': this.apiKey,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      log.error(' Account info error:', errorText);
      throw new Error(`Failed to get account info: ${response.status}`);
    }

    const result = await response.json();
    log.debug(' Account info:', result);

    // Parse response - field names may vary
    return {
      accountName: result.data?.account_name || result.account_name || '',
      accountId: result.data?.account_id || result.account_id || '',
      credits: result.data?.credits || result.credits || 0,
      creditsUsd: result.data?.equivalent_in_usd || result.equivalent_in_usd || 0,
    };
  }
}

// Singleton instance
export const piApiService = new PiApiService();
