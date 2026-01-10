// Kling AI Video Generation Service
// Supports text-to-video and image-to-video generation via official Kling API
// Uses JWT authentication with Access Key (AK) and Secret Key (SK)

const BASE_URL = 'https://api.klingai.com';
const TOKEN_EXPIRATION = 1800; // 30 minutes in seconds

// Available models
export const KLING_MODELS = [
  { id: 'kling-v1', name: 'Kling v1.0' },
  { id: 'kling-v1-5', name: 'Kling v1.5' },
  { id: 'kling-v1-6', name: 'Kling v1.6' },
  { id: 'kling-v2-0', name: 'Kling v2.0' },
  { id: 'kling-v2-1', name: 'Kling v2.1' },
] as const;

// Duration options (in seconds)
export const KLING_DURATIONS = [
  { value: 5, label: '5 seconds' },
  { value: 10, label: '10 seconds' },
] as const;

// Aspect ratio options with dimensions
export const KLING_ASPECT_RATIOS = [
  { value: '16:9', label: '16:9 (Landscape)', width: 16, height: 9 },
  { value: '9:16', label: '9:16 (Portrait)', width: 9, height: 16 },
  { value: '1:1', label: '1:1 (Square)', width: 1, height: 1 },
] as const;

// Get aspect ratio dimensions
export function getAspectRatioDimensions(aspectRatio: string): { width: number; height: number } {
  const ar = KLING_ASPECT_RATIOS.find(a => a.value === aspectRatio);
  return ar ? { width: ar.width, height: ar.height } : { width: 16, height: 9 };
}

// Generation mode options
export const KLING_MODES = [
  { value: 'std', label: 'Standard', description: 'Faster generation, good quality' },
  { value: 'pro', label: 'Professional', description: 'Higher quality, slower generation' },
] as const;

// Camera control presets
export const KLING_CAMERA_CONTROLS = [
  { value: '', label: 'None' },
  { value: 'down_back', label: 'Down & Back' },
  { value: 'forward_up', label: 'Forward & Up' },
  { value: 'right_turn_forward', label: 'Right Turn Forward' },
  { value: 'left_turn_forward', label: 'Left Turn Forward' },
] as const;

// Task status types
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface KlingTask {
  id: string;
  status: TaskStatus;
  progress?: number;
  videoUrl?: string;
  thumbnailUrl?: string;
  error?: string;
  createdAt: Date;
  completedAt?: Date;
}

export interface TextToVideoParams {
  prompt: string;
  negativePrompt?: string;
  model: string;
  duration: number;
  aspectRatio: string;
  mode: string;
  cfgScale?: number;
  cameraControl?: string;
}

export interface ImageToVideoParams {
  prompt: string;
  negativePrompt?: string;
  startImageUrl?: string;
  endImageUrl?: string;
  model: string;
  duration: number;
  mode: string;
  cfgScale?: number;
}

interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

interface CreateTaskResponse {
  task_id: string;
}

interface TaskStatusResponse {
  task_id: string;
  task_status: string;
  task_status_msg?: string;
  task_result?: {
    videos?: Array<{
      url: string;
      duration: number;
    }>;
  };
}

// Base64URL encoding (no padding, URL-safe)
function base64UrlEncode(data: string): string {
  const base64 = btoa(data);
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// HMAC-SHA256 signing
async function hmacSha256(key: string, data: string): Promise<string> {
  const encoder = new TextEncoder();
  const keyData = encoder.encode(key);
  const dataToSign = encoder.encode(data);

  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    keyData,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );

  const signature = await crypto.subtle.sign('HMAC', cryptoKey, dataToSign);
  const signatureArray = new Uint8Array(signature);

  // Convert to base64url
  let binary = '';
  for (let i = 0; i < signatureArray.length; i++) {
    binary += String.fromCharCode(signatureArray[i]);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

class KlingService {
  private accessKey: string = '';
  private secretKey: string = '';
  private cachedToken: string | null = null;
  private tokenExpiry: number = 0;

  setCredentials(accessKey: string, secretKey: string) {
    this.accessKey = accessKey;
    this.secretKey = secretKey;
    // Clear cached token when credentials change
    this.cachedToken = null;
    this.tokenExpiry = 0;
  }

  hasCredentials(): boolean {
    return !!this.accessKey && !!this.secretKey;
  }

  // Generate JWT token from AK/SK
  private async generateToken(): Promise<string> {
    // Check if we have a valid cached token
    const now = Math.floor(Date.now() / 1000);
    if (this.cachedToken && this.tokenExpiry > now + 60) {
      // Token still valid for at least 60 more seconds
      return this.cachedToken;
    }

    // JWT Header
    const header = {
      alg: 'HS256',
      typ: 'JWT'
    };

    // JWT Payload
    const payload = {
      iss: this.accessKey,
      exp: now + TOKEN_EXPIRATION,
      nbf: now - 5 // Allow 5 seconds clock skew
    };

    // Encode header and payload
    const headerEncoded = base64UrlEncode(JSON.stringify(header));
    const payloadEncoded = base64UrlEncode(JSON.stringify(payload));

    // Create signature
    const dataToSign = `${headerEncoded}.${payloadEncoded}`;
    const signature = await hmacSha256(this.secretKey, dataToSign);

    // Combine to create JWT
    const token = `${dataToSign}.${signature}`;

    // Cache the token
    this.cachedToken = token;
    this.tokenExpiry = now + TOKEN_EXPIRATION;

    return token;
  }

  private async request<T>(
    endpoint: string,
    method: 'GET' | 'POST' = 'GET',
    body?: object
  ): Promise<T> {
    if (!this.hasCredentials()) {
      throw new Error('Kling API credentials not set');
    }

    const token = await this.generateToken();

    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Kling API error: ${response.status} - ${errorText}`);
    }

    const result = await response.json() as ApiResponse<T>;

    if (result.code !== 0) {
      throw new Error(`Kling API error: ${result.message}`);
    }

    return result.data;
  }

  async createTextToVideo(params: TextToVideoParams): Promise<string> {
    const body: Record<string, unknown> = {
      model_name: params.model,
      prompt: params.prompt,
      duration: String(params.duration),
      aspect_ratio: params.aspectRatio,
      mode: params.mode,
    };

    if (params.negativePrompt) {
      body.negative_prompt = params.negativePrompt;
    }
    if (params.cfgScale !== undefined) {
      body.cfg_scale = params.cfgScale;
    }
    if (params.cameraControl) {
      body.camera_control = { type: params.cameraControl };
    }

    const result = await this.request<CreateTaskResponse>(
      '/v1/videos/text2video',
      'POST',
      body
    );

    return result.task_id;
  }

  async createImageToVideo(params: ImageToVideoParams): Promise<string> {
    const body: Record<string, unknown> = {
      model_name: params.model,
      prompt: params.prompt,
      duration: String(params.duration),
      mode: params.mode,
    };

    if (params.startImageUrl) {
      body.image = params.startImageUrl;
    }
    if (params.endImageUrl) {
      body.image_tail = params.endImageUrl;
    }
    if (params.negativePrompt) {
      body.negative_prompt = params.negativePrompt;
    }
    if (params.cfgScale !== undefined) {
      body.cfg_scale = params.cfgScale;
    }

    const result = await this.request<CreateTaskResponse>(
      '/v1/videos/image2video',
      'POST',
      body
    );

    return result.task_id;
  }

  async getTaskStatus(taskId: string): Promise<KlingTask> {
    const result = await this.request<TaskStatusResponse>(
      `/v1/videos/${taskId}`,
      'GET'
    );

    let status: TaskStatus = 'pending';
    switch (result.task_status.toLowerCase()) {
      case 'completed':
      case 'succeed':
        status = 'completed';
        break;
      case 'processing':
      case 'running':
        status = 'processing';
        break;
      case 'failed':
      case 'error':
        status = 'failed';
        break;
      default:
        status = 'pending';
    }

    const task: KlingTask = {
      id: result.task_id,
      status,
      error: result.task_status_msg,
      createdAt: new Date(),
    };

    if (status === 'completed' && result.task_result?.videos?.[0]) {
      task.videoUrl = result.task_result.videos[0].url;
      task.completedAt = new Date();
    }

    return task;
  }

  async pollTaskUntilComplete(
    taskId: string,
    onProgress?: (task: KlingTask) => void,
    pollInterval = 5000,
    timeout = 600000 // 10 minutes
  ): Promise<KlingTask> {
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
}

// Singleton instance
export const klingService = new KlingService();
