export type MediaRuntimeKind = 'video' | 'audio' | 'image';

export type DecodeSessionPolicy =
  | 'interactive'
  | 'background'
  | 'export'
  | 'ram-preview';

export interface MediaSourceMetadata {
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
  codec?: string;
  audioCodec?: string;
  container?: string;
  hasAudio?: boolean;
}

export interface MediaSourceRuntimeDescriptor {
  sourceId?: string;
  mediaFileId?: string;
  kind: MediaRuntimeKind;
  file?: File;
  fileName?: string;
  fileSize?: number;
  fileLastModified?: number;
  fileHash?: string;
  filePath?: string;
}

export interface FrameRequest {
  sourceId: string;
  sessionKey: string;
  sourceTime: number;
  frameNumber?: number;
  playbackMode: DecodeSessionPolicy;
  allowCache?: boolean;
  tolerateStaleFrame?: boolean;
}

export type RuntimeFrame = VideoFrame | ImageBitmap | null;

export interface RuntimeFrameProvider {
  currentTime: number;
  isPlaying: boolean;
  isFullMode(): boolean;
  isSimpleMode(): boolean;
  getCurrentFrame(): VideoFrame | null;
  getFrameRate?(): number;
  getPendingSeekTime?(): number | null;
  isSeeking?(): boolean;
  isDecodePending?(): boolean;
  hasFrame?(): boolean;
  hasBufferedFutureFrame?(minFrameDelta?: number): boolean;
  getDebugInfo?(): {
    codec: string;
    hwAccel: string;
    decodeQueueSize: number;
    samplesLoaded: number;
    sampleIndex: number;
  } | null;
  advanceToTime?(timeSeconds: number): void;
  seek(timeSeconds: number): void;
  fastSeek?(timeSeconds: number): void;
  pause(): void;
  destroy?(): void;
}

export interface FrameHandle {
  sourceId: string;
  timestamp: number;
  frameNumber?: number;
  frame: RuntimeFrame;
  release(): void;
}

export interface DecodeSession {
  key: string;
  sourceId: string;
  ownerId?: string;
  policy: DecodeSessionPolicy;
  currentTime: number;
  createdAt: number;
  lastAccessedAt: number;
  frameProvider?: RuntimeFrameProvider | null;
  currentFrameTimestamp?: number | null;
  touch(time?: number): void;
  dispose(): void;
}

export interface MediaSourceRuntime {
  sourceId: string;
  descriptor: MediaSourceRuntimeDescriptor;
  metadata: MediaSourceMetadata;
  frameCache: Map<string, FrameHandle>;
  retain(ownerId: string): number;
  release(ownerId: string): number;
  ownerCount(): number;
  getSession(
    key: string,
    options?: {
      ownerId?: string;
      policy?: DecodeSessionPolicy;
    }
  ): DecodeSession;
  peekSession(key: string): DecodeSession | null;
  setSessionFrameProvider(
    key: string,
    provider: RuntimeFrameProvider | null,
    options?: {
      ownsProvider?: boolean;
    }
  ): DecodeSession | null;
  getSessionFrameProvider(key: string): RuntimeFrameProvider | null;
  updateSessionTime(key: string, time: number): DecodeSession | null;
  cacheFrame(
    request: Pick<FrameRequest, 'sourceTime' | 'frameNumber'>,
    frame: RuntimeFrame,
    options?: {
      timestamp?: number;
    }
  ): FrameHandle | null;
  getFrameSync(request: FrameRequest): FrameHandle | null;
  releaseSession(key: string): void;
  getFrame(request: FrameRequest): Promise<FrameHandle | null>;
  updateDescriptor(partial: Partial<MediaSourceRuntimeDescriptor>): void;
  updateMetadata(partial: Partial<MediaSourceMetadata>): void;
  dispose(): void;
}

export interface MediaRuntimeRegistry {
  resolveSourceId(descriptor: MediaSourceRuntimeDescriptor): string | null;
  retainRuntime(
    descriptor: MediaSourceRuntimeDescriptor,
    ownerId: string
  ): MediaSourceRuntime | null;
  getRuntime(sourceId: string): MediaSourceRuntime | null;
  getSession(sourceId: string, sessionKey: string): DecodeSession | null;
  setSessionFrameProvider(
    sourceId: string,
    sessionKey: string,
    provider: RuntimeFrameProvider | null,
    options?: {
      ownsProvider?: boolean;
    }
  ): DecodeSession | null;
  updateSessionTime(sourceId: string, sessionKey: string, time: number): DecodeSession | null;
  releaseSession(sourceId: string, sessionKey: string): void;
  releaseRuntime(sourceId: string, ownerId: string): void;
  releaseClip(clipId: string, sourceId?: string, sessionKey?: string): void;
  listRuntimes(): MediaSourceRuntime[];
  clear(): void;
}
