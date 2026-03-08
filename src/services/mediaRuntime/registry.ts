import type {
  DecodeSession,
  DecodeSessionPolicy,
  FrameHandle,
  FrameRequest,
  MediaRuntimeRegistry,
  MediaSourceMetadata,
  MediaSourceRuntime,
  MediaSourceRuntimeDescriptor,
  RuntimeFrame,
  RuntimeFrameProvider,
} from './types';

function mergeDefined<T extends object>(target: T, partial: Partial<T>): T {
  const next = { ...target } as T;
  for (const key of Object.keys(partial) as Array<keyof T>) {
    const value = partial[key];
    if (value !== undefined) {
      next[key] = value;
    }
  }
  return next;
}

function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').toLowerCase();
}

function getFrameCacheKey(request: Pick<FrameRequest, 'sourceTime' | 'frameNumber'>): string {
  if (request.frameNumber !== undefined) {
    return `frame:${request.frameNumber}`;
  }
  return `time:${Math.round(request.sourceTime * 1_000_000)}`;
}

const MAX_SOURCE_FRAME_CACHE_ENTRIES = 12;

function canCloneRuntimeFrame(frame: RuntimeFrame): frame is RuntimeFrame & {
  clone: () => RuntimeFrame;
} {
  return !!frame && typeof (frame as { clone?: unknown }).clone === 'function';
}

function cloneRuntimeFrame(frame: RuntimeFrame): RuntimeFrame {
  if (!canCloneRuntimeFrame(frame)) {
    return null;
  }
  return frame.clone();
}

class MediaRuntimeFrameHandle implements FrameHandle {
  sourceId: string;
  timestamp: number;
  frameNumber?: number;
  frame: RuntimeFrame;
  private ownsFrame: boolean;

  constructor(params: {
    sourceId: string;
    timestamp: number;
    frameNumber?: number;
    frame: RuntimeFrame;
    ownsFrame?: boolean;
  }) {
    this.sourceId = params.sourceId;
    this.timestamp = params.timestamp;
    this.frameNumber = params.frameNumber;
    this.frame = params.frame;
    this.ownsFrame = params.ownsFrame ?? false;
  }

  release(): void {
    if (!this.ownsFrame || !this.frame) {
      return;
    }
    const frame = this.frame as { close?: () => void };
    frame.close?.();
    this.frame = null;
  }
}

class BasicDecodeSession implements DecodeSession {
  key: string;
  sourceId: string;
  ownerId?: string;
  policy: DecodeSessionPolicy;
  currentTime: number;
  createdAt: number;
  lastAccessedAt: number;
  frameProvider: RuntimeFrameProvider | null = null;
  currentFrameTimestamp: number | null = null;
  ownsFrameProvider = false;

  constructor(params: {
    key: string;
    sourceId: string;
    ownerId?: string;
    policy: DecodeSessionPolicy;
  }) {
    const now = Date.now();
    this.key = params.key;
    this.sourceId = params.sourceId;
    this.ownerId = params.ownerId;
    this.policy = params.policy;
    this.currentTime = 0;
    this.createdAt = now;
    this.lastAccessedAt = now;
  }

  touch(time?: number): void {
    if (time !== undefined) {
      this.currentTime = time;
    }
    this.lastAccessedAt = Date.now();
  }

  dispose(): void {
    if (this.ownsFrameProvider) {
      this.frameProvider?.destroy?.();
    }
    this.frameProvider = null;
    this.currentFrameTimestamp = null;
    this.ownsFrameProvider = false;
  }
}

class BasicMediaSourceRuntime implements MediaSourceRuntime {
  sourceId: string;
  descriptor: MediaSourceRuntimeDescriptor;
  metadata: MediaSourceMetadata;
  frameCache = new Map<string, FrameHandle>();
  private owners = new Set<string>();
  private sessions = new Map<string, BasicDecodeSession>();

  constructor(sourceId: string, descriptor: MediaSourceRuntimeDescriptor) {
    this.sourceId = sourceId;
    this.descriptor = { ...descriptor, sourceId };
    this.metadata = {};
  }

  retain(ownerId: string): number {
    this.owners.add(ownerId);
    return this.owners.size;
  }

  release(ownerId: string): number {
    this.owners.delete(ownerId);
    return this.owners.size;
  }

  ownerCount(): number {
    return this.owners.size;
  }

  getSession(
    key: string,
    options?: {
      ownerId?: string;
      policy?: DecodeSessionPolicy;
    }
  ): DecodeSession {
    const existing = this.sessions.get(key);
    if (existing) {
      existing.touch();
      return existing;
    }

    const session = new BasicDecodeSession({
      key,
      sourceId: this.sourceId,
      ownerId: options?.ownerId,
      policy: options?.policy ?? 'interactive',
    });
    this.sessions.set(key, session);
    return session;
  }

  peekSession(key: string): DecodeSession | null {
    return this.sessions.get(key) ?? null;
  }

  setSessionFrameProvider(
    key: string,
    provider: RuntimeFrameProvider | null,
    options?: {
      ownsProvider?: boolean;
    }
  ): DecodeSession | null {
    const session = this.sessions.get(key);
    if (!session) {
      return null;
    }
    if (session.ownsFrameProvider && session.frameProvider && session.frameProvider !== provider) {
      session.frameProvider.destroy?.();
    }
    session.frameProvider = provider;
    session.ownsFrameProvider = options?.ownsProvider ?? false;
    if (!provider) {
      session.currentFrameTimestamp = null;
      session.ownsFrameProvider = false;
    }
    session.touch();
    return session;
  }

  getSessionFrameProvider(key: string): RuntimeFrameProvider | null {
    return this.sessions.get(key)?.frameProvider ?? null;
  }

  updateSessionTime(key: string, time: number): DecodeSession | null {
    const session = this.sessions.get(key);
    if (!session) {
      return null;
    }
    session.touch(time);
    return session;
  }

  cacheFrame(
    request: Pick<FrameRequest, 'sourceTime' | 'frameNumber'>,
    frame: RuntimeFrame,
    options?: {
      timestamp?: number;
    }
  ): FrameHandle | null {
    const clonedFrame = cloneRuntimeFrame(frame);
    if (!clonedFrame) {
      return null;
    }

    const key = getFrameCacheKey(request);
    const existing = this.frameCache.get(key);
    if (existing) {
      existing.release();
      this.frameCache.delete(key);
    }

    const cachedHandle = createFrameHandle({
      sourceId: this.sourceId,
      timestamp: options?.timestamp ?? request.sourceTime * 1_000_000,
      frameNumber: request.frameNumber,
      frame: clonedFrame,
      ownsFrame: true,
    });
    this.frameCache.set(key, cachedHandle);

    while (this.frameCache.size > MAX_SOURCE_FRAME_CACHE_ENTRIES) {
      const oldestKey = this.frameCache.keys().next().value;
      if (!oldestKey) {
        break;
      }
      const oldest = this.frameCache.get(oldestKey);
      oldest?.release();
      this.frameCache.delete(oldestKey);
    }

    return createFrameHandle({
      sourceId: this.sourceId,
      timestamp: cachedHandle.timestamp,
      frameNumber: cachedHandle.frameNumber,
      frame: cachedHandle.frame,
    });
  }

  private getCachedFrame(
    request: Pick<FrameRequest, 'sourceTime' | 'frameNumber'>
  ): FrameHandle | null {
    const key = getFrameCacheKey(request);
    const cachedHandle = this.frameCache.get(key);
    if (!cachedHandle) {
      return null;
    }

    // Refresh insertion order to approximate LRU behavior.
    this.frameCache.delete(key);
    this.frameCache.set(key, cachedHandle);

    return createFrameHandle({
      sourceId: this.sourceId,
      timestamp: cachedHandle.timestamp,
      frameNumber: cachedHandle.frameNumber,
      frame: cachedHandle.frame,
    });
  }

  private getSharedSessionTolerance(session: BasicDecodeSession): number {
    const frameRate = session.frameProvider?.getFrameRate?.() ?? 30;
    return Math.max(0.004, Math.min(0.03, 0.5 / Math.max(frameRate, 1)));
  }

  private getFrameFromSiblingSession(
    request: FrameRequest
  ): FrameHandle | null {
    for (const [key, session] of this.sessions) {
      if (key === request.sessionKey || !session.frameProvider) {
        continue;
      }

      const tolerance = this.getSharedSessionTolerance(session);
      if (Math.abs(session.currentTime - request.sourceTime) > tolerance) {
        continue;
      }

      const currentFrame = session.frameProvider.getCurrentFrame();
      if (!currentFrame) {
        continue;
      }

      const timestamp = currentFrame.timestamp ?? session.currentTime * 1_000_000;
      session.currentFrameTimestamp = timestamp;

      return (
        this.cacheFrame(request, currentFrame, { timestamp }) ??
        createFrameHandle({
          sourceId: this.sourceId,
          timestamp,
          frameNumber: request.frameNumber,
          frame: currentFrame,
        })
      );
    }

    return null;
  }

  getFrameSync(request: FrameRequest): FrameHandle | null {
    const session = this.getSession(request.sessionKey, {
      policy: request.playbackMode,
    });
    session.touch(request.sourceTime);

    const frameProvider = session.frameProvider;
    const currentFrame = frameProvider?.getCurrentFrame();
    if (currentFrame) {
      const timestamp = currentFrame.timestamp ?? request.sourceTime * 1_000_000;
      session.currentFrameTimestamp = timestamp;
      this.cacheFrame(request, currentFrame, { timestamp });
      return createFrameHandle({
        sourceId: this.sourceId,
        timestamp,
        frameNumber: request.frameNumber,
        frame: currentFrame,
      });
    }

    const siblingFrame = this.getFrameFromSiblingSession(request);
    if (siblingFrame) {
      return siblingFrame;
    }

    if (request.allowCache === false) {
      return null;
    }

    return this.getCachedFrame(request);
  }

  releaseSession(key: string): void {
    const session = this.sessions.get(key);
    if (!session) {
      return;
    }
    session.dispose();
    this.sessions.delete(key);
  }

  async getFrame(request: FrameRequest): Promise<FrameHandle | null> {
    return this.getFrameSync(request);
  }

  updateDescriptor(partial: Partial<MediaSourceRuntimeDescriptor>): void {
    this.descriptor = mergeDefined(this.descriptor, partial);
  }

  updateMetadata(partial: Partial<MediaSourceMetadata>): void {
    this.metadata = mergeDefined(this.metadata, partial);
  }

  dispose(): void {
    for (const session of this.sessions.values()) {
      session.dispose();
    }
    this.sessions.clear();

    for (const frame of this.frameCache.values()) {
      frame.release();
    }
    this.frameCache.clear();
    this.owners.clear();
  }
}

class DefaultMediaRuntimeRegistry implements MediaRuntimeRegistry {
  private runtimes = new Map<string, BasicMediaSourceRuntime>();

  resolveSourceId(descriptor: MediaSourceRuntimeDescriptor): string | null {
    if (descriptor.sourceId) {
      return descriptor.sourceId;
    }
    if (descriptor.mediaFileId) {
      return `media:${descriptor.mediaFileId}`;
    }
    if (descriptor.fileHash) {
      return `hash:${descriptor.fileHash}`;
    }
    if (descriptor.filePath) {
      return `path:${normalizePath(descriptor.filePath)}`;
    }
    if (descriptor.file) {
      return `file:${descriptor.kind}:${descriptor.file.name}:${descriptor.file.size}:${descriptor.file.lastModified}`;
    }
    if (descriptor.fileName) {
      return `name:${descriptor.kind}:${descriptor.fileName}`;
    }
    return null;
  }

  retainRuntime(
    descriptor: MediaSourceRuntimeDescriptor,
    ownerId: string
  ): MediaSourceRuntime | null {
    const sourceId = this.resolveSourceId(descriptor);
    if (!sourceId) {
      return null;
    }

    let runtime = this.runtimes.get(sourceId);
    if (!runtime) {
      runtime = new BasicMediaSourceRuntime(sourceId, descriptor);
      this.runtimes.set(sourceId, runtime);
    } else {
      runtime.updateDescriptor(descriptor);
    }

    runtime.retain(ownerId);
    return runtime;
  }

  getRuntime(sourceId: string): MediaSourceRuntime | null {
    return this.runtimes.get(sourceId) ?? null;
  }

  getSession(sourceId: string, sessionKey: string): DecodeSession | null {
    return this.runtimes.get(sourceId)?.peekSession(sessionKey) ?? null;
  }

  setSessionFrameProvider(
    sourceId: string,
    sessionKey: string,
    provider: RuntimeFrameProvider | null,
    options?: {
      ownsProvider?: boolean;
    }
  ): DecodeSession | null {
    return this.runtimes.get(sourceId)?.setSessionFrameProvider(sessionKey, provider, options) ?? null;
  }

  updateSessionTime(sourceId: string, sessionKey: string, time: number): DecodeSession | null {
    return this.runtimes.get(sourceId)?.updateSessionTime(sessionKey, time) ?? null;
  }

  releaseSession(sourceId: string, sessionKey: string): void {
    this.runtimes.get(sourceId)?.releaseSession(sessionKey);
  }

  releaseRuntime(sourceId: string, ownerId: string): void {
    const runtime = this.runtimes.get(sourceId);
    if (!runtime) {
      return;
    }

    const ownersLeft = runtime.release(ownerId);
    if (ownersLeft > 0) {
      return;
    }

    runtime.dispose();
    this.runtimes.delete(sourceId);
  }

  releaseClip(clipId: string, sourceId?: string, sessionKey?: string): void {
    if (!sourceId) {
      return;
    }

    const runtime = this.runtimes.get(sourceId);
    if (runtime && sessionKey) {
      runtime.releaseSession(sessionKey);
    }
    this.releaseRuntime(sourceId, clipId);
  }

  listRuntimes(): MediaSourceRuntime[] {
    return Array.from(this.runtimes.values());
  }

  clear(): void {
    for (const runtime of this.runtimes.values()) {
      runtime.dispose();
    }
    this.runtimes.clear();
  }
}

export function createFrameHandle(params: {
  sourceId: string;
  timestamp: number;
  frameNumber?: number;
  frame: RuntimeFrame;
  ownsFrame?: boolean;
}): FrameHandle {
  return new MediaRuntimeFrameHandle(params);
}

export const mediaRuntimeRegistry = new DefaultMediaRuntimeRegistry();
