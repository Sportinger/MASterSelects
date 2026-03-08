import type { LayerSource, TimelineClip } from '../../types';
import { engine } from '../../engine/WebGPUEngine';
import { WebCodecsPlayer } from '../../engine/WebCodecsPlayer';
import { useMediaStore } from '../../stores/mediaStore';
import { mediaRuntimeRegistry } from './registry';
import type {
  DecodeSession,
  DecodeSessionPolicy,
  FrameHandle,
  MediaSourceRuntime,
  RuntimeFrameProvider,
} from './types';

type RuntimeBackedSource = Pick<
  LayerSource,
  'runtimeSourceId' | 'runtimeSessionKey' | 'webCodecsPlayer'
> | Pick<
  NonNullable<TimelineClip['source']>,
  'runtimeSourceId' | 'runtimeSessionKey' | 'webCodecsPlayer'
>;

export interface RuntimePlaybackBinding {
  sourceId: string;
  sessionKey: string;
  session: DecodeSession;
  frameProvider: RuntimeFrameProvider | null;
}

const INTERACTIVE_PLAYBACK_SESSION_PREFIX = 'interactive-track:';
const INTERACTIVE_SCRUB_SESSION_PREFIX = 'interactive-scrub:';
const pendingRuntimeProviderLoads = new Map<string, Promise<RuntimeFrameProvider | null>>();

function buildPolicyRuntimeSessionKey(
  sourceId: string,
  policy: DecodeSessionPolicy,
  ownerId: string,
  sessionScope?: string
): string {
  if (sessionScope) {
    return `${policy}:${sessionScope}:${ownerId}:${sourceId}`;
  }
  return `${policy}:${ownerId}:${sourceId}`;
}

function shouldReplaceFrameProvider(
  currentProvider: RuntimeFrameProvider | null,
  sourcePlayer: RuntimeFrameProvider | undefined
): boolean {
  if (!sourcePlayer) {
    return false;
  }
  if (!currentProvider) {
    return true;
  }
  return !currentProvider.isFullMode() && sourcePlayer.isFullMode();
}

function isInteractiveScrubSessionKey(sessionKey: string | undefined): boolean {
  return !!sessionKey && sessionKey.startsWith(INTERACTIVE_SCRUB_SESSION_PREFIX);
}

function getPendingProviderLoadKey(sourceId: string, sessionKey: string): string {
  return `${sourceId}:${sessionKey}`;
}

function getRuntimeFile(runtime: MediaSourceRuntime): File | null {
  if (runtime.descriptor.file) {
    return runtime.descriptor.file;
  }

  if (runtime.descriptor.mediaFileId) {
    const mediaFile = useMediaStore.getState().files.find(
      (file) => file.id === runtime.descriptor.mediaFileId
    );
    if (mediaFile?.file) {
      return mediaFile.file;
    }
  }

  return null;
}

function hasRuntimeBinding(
  source: RuntimeBackedSource | null | undefined
): source is RuntimeBackedSource & {
  runtimeSourceId: string;
  runtimeSessionKey: string;
} {
  return !!source?.runtimeSourceId && !!source?.runtimeSessionKey;
}

export function resolveRuntimePlaybackBinding(
  source: RuntimeBackedSource | null | undefined,
  policy: DecodeSessionPolicy = 'interactive'
): RuntimePlaybackBinding | null {
  if (!hasRuntimeBinding(source)) {
    return null;
  }

  const runtime = mediaRuntimeRegistry.getRuntime(source.runtimeSourceId);
  if (!runtime) {
    return null;
  }

  const session =
    runtime.peekSession(source.runtimeSessionKey) ??
    runtime.getSession(source.runtimeSessionKey, { policy });
  let frameProvider = runtime.getSessionFrameProvider(source.runtimeSessionKey);

  const sourcePlayer = source.webCodecsPlayer ?? null;
  if (
    !isInteractiveScrubSessionKey(source.runtimeSessionKey) &&
    shouldReplaceFrameProvider(frameProvider, sourcePlayer ?? undefined)
  ) {
    runtime.setSessionFrameProvider(source.runtimeSessionKey, sourcePlayer);
    frameProvider = sourcePlayer;
  }

  return {
    sourceId: source.runtimeSourceId,
    sessionKey: source.runtimeSessionKey,
    session,
    frameProvider,
  };
}

export function updateRuntimePlaybackTime(
  source: RuntimeBackedSource | null | undefined,
  time: number,
  policy: DecodeSessionPolicy = 'interactive'
): RuntimePlaybackBinding | null {
  const binding = resolveRuntimePlaybackBinding(source, policy);
  if (!binding) {
    return null;
  }
  mediaRuntimeRegistry.updateSessionTime(binding.sourceId, binding.sessionKey, time);
  return binding;
}

export function getRuntimeFrameProvider(
  source: RuntimeBackedSource | null | undefined,
  policy: DecodeSessionPolicy = 'interactive'
): RuntimeFrameProvider | null {
  return resolveRuntimePlaybackBinding(source, policy)?.frameProvider ?? null;
}

export function isRuntimeFullWebCodecsSource(
  source: RuntimeBackedSource | null | undefined,
  policy: DecodeSessionPolicy = 'interactive'
): boolean {
  return !!getRuntimeFrameProvider(source, policy)?.isFullMode();
}

export function canUseSharedPreviewRuntimeSession(
  clip: Pick<TimelineClip, 'trackId'>,
  activeClips: Iterable<Pick<TimelineClip, 'trackId'>>
): boolean {
  if (!clip.trackId) {
    return false;
  }

  let activeOnTrack = 0;
  for (const activeClip of activeClips) {
    if (activeClip.trackId !== clip.trackId) {
      continue;
    }
    activeOnTrack += 1;
    if (activeOnTrack > 1) {
      return false;
    }
  }

  return activeOnTrack === 1;
}

export function getSharedPreviewRuntimeSessionKey(
  source: RuntimeBackedSource | null | undefined,
  trackId?: string,
  allowSharedSession = true,
  sessionScope?: string
): string | undefined {
  if (!source?.runtimeSourceId || !source.runtimeSessionKey) {
    return source?.runtimeSessionKey;
  }
  if (!allowSharedSession || !trackId) {
    return source.runtimeSessionKey;
  }

  const clipPlayer = source.webCodecsPlayer;
  const runtimeProvider = clipPlayer?.isFullMode()
    ? clipPlayer
    : getRuntimeFrameProvider(source);
  if (!runtimeProvider?.isFullMode()) {
    return source.runtimeSessionKey;
  }

  if (sessionScope) {
    return `${INTERACTIVE_PLAYBACK_SESSION_PREFIX}${sessionScope}:${trackId}:${source.runtimeSourceId}`;
  }
  return `${INTERACTIVE_PLAYBACK_SESSION_PREFIX}${trackId}:${source.runtimeSourceId}`;
}

export function getScrubRuntimeSessionKey(
  source: RuntimeBackedSource | null | undefined,
  trackId?: string,
  allowSharedSession = true,
  sessionScope?: string
): string | undefined {
  if (!source?.runtimeSourceId || !source.runtimeSessionKey) {
    return source?.runtimeSessionKey;
  }
  if (!allowSharedSession || !trackId) {
    return source.runtimeSessionKey;
  }

  const clipPlayer = source.webCodecsPlayer;
  const runtimeProvider = clipPlayer?.isFullMode()
    ? clipPlayer
    : getRuntimeFrameProvider(source);
  if (!runtimeProvider?.isFullMode()) {
    return source.runtimeSessionKey;
  }

  if (sessionScope) {
    return `${INTERACTIVE_SCRUB_SESSION_PREFIX}${sessionScope}:${trackId}:${source.runtimeSourceId}`;
  }
  return `${INTERACTIVE_SCRUB_SESSION_PREFIX}${trackId}:${source.runtimeSourceId}`;
}

export function getPreviewRuntimeSource<
  T extends RuntimeBackedSource | null | undefined,
>(
  source: T,
  trackId?: string,
  allowSharedSession = true,
  sessionScope?: string
): T {
  if (!source) {
    return source;
  }

  const sessionKey = getSharedPreviewRuntimeSessionKey(
    source,
    trackId,
    allowSharedSession,
    sessionScope
  );
  if (!sessionKey || sessionKey === source.runtimeSessionKey) {
    return source;
  }

  return {
    ...source,
    runtimeSessionKey: sessionKey,
  } as T;
}

export function getScrubRuntimeSource<
  T extends RuntimeBackedSource | null | undefined,
>(
  source: T,
  trackId?: string,
  allowSharedSession = true,
  sessionScope?: string
): T {
  if (!source) {
    return source;
  }

  const sessionKey = getScrubRuntimeSessionKey(
    source,
    trackId,
    allowSharedSession,
    sessionScope
  );
  if (!sessionKey || sessionKey === source.runtimeSessionKey) {
    return source;
  }

  return {
    ...source,
    runtimeSessionKey: sessionKey,
  } as T;
}

export function getPolicyRuntimeSessionKey(
  source: RuntimeBackedSource | null | undefined,
  policy: DecodeSessionPolicy,
  ownerId: string,
  sessionScope?: string
): string | undefined {
  if (!source?.runtimeSourceId || !source.runtimeSessionKey) {
    return source?.runtimeSessionKey;
  }
  return buildPolicyRuntimeSessionKey(
    source.runtimeSourceId,
    policy,
    ownerId,
    sessionScope
  );
}

export function getPolicyRuntimeSource<
  T extends RuntimeBackedSource | null | undefined,
>(
  source: T,
  policy: DecodeSessionPolicy,
  ownerId: string,
  sessionScope?: string
): T {
  if (!source) {
    return source;
  }

  const sessionKey = getPolicyRuntimeSessionKey(
    source,
    policy,
    ownerId,
    sessionScope
  );
  if (!sessionKey || sessionKey === source.runtimeSessionKey) {
    return source;
  }

  return {
    ...source,
    runtimeSessionKey: sessionKey,
  } as T;
}

export function setRuntimeFrameProvider(
  source: RuntimeBackedSource | null | undefined,
  provider: RuntimeFrameProvider | null,
  policy: DecodeSessionPolicy = 'interactive',
  options?: {
    ownsProvider?: boolean;
  }
): RuntimePlaybackBinding | null {
  if (!hasRuntimeBinding(source)) {
    return null;
  }

  const runtime = mediaRuntimeRegistry.getRuntime(source.runtimeSourceId);
  if (!runtime) {
    return null;
  }

  const session =
    runtime.peekSession(source.runtimeSessionKey) ??
    runtime.getSession(source.runtimeSessionKey, { policy });
  runtime.setSessionFrameProvider(source.runtimeSessionKey, provider, options);

  return {
    sourceId: source.runtimeSourceId,
    sessionKey: source.runtimeSessionKey,
    session,
    frameProvider: provider,
  };
}

export async function ensureRuntimeFrameProvider(
  source: RuntimeBackedSource | null | undefined,
  policy: DecodeSessionPolicy = 'interactive',
  sourceTime?: number
): Promise<RuntimeFrameProvider | null> {
  const binding = resolveRuntimePlaybackBinding(source, policy);
  if (!binding) {
    return null;
  }

  const runtime = mediaRuntimeRegistry.getRuntime(binding.sourceId);
  if (!runtime) {
    return null;
  }

  if (sourceTime !== undefined) {
    runtime.updateSessionTime(binding.sessionKey, sourceTime);
  }

  if (binding.frameProvider) {
    return binding.frameProvider;
  }

  if (
    policy !== 'interactive' ||
    !isInteractiveScrubSessionKey(binding.sessionKey) ||
    !source?.webCodecsPlayer?.isFullMode()
  ) {
    return null;
  }

  const file = getRuntimeFile(runtime);
  if (!file) {
    return null;
  }

  const loadKey = getPendingProviderLoadKey(binding.sourceId, binding.sessionKey);
  const pendingLoad = pendingRuntimeProviderLoads.get(loadKey);
  if (pendingLoad) {
    return pendingLoad;
  }

  const initialTime = sourceTime ?? binding.session.currentTime;
  const loadPromise = (async () => {
    const player = new WebCodecsPlayer({
      loop: false,
      useSimpleMode: false,
      onFrame: () => {
        engine.requestNewFrameRender();
      },
      onError: () => {
        engine.requestRender();
      },
    });

    try {
      await player.loadFile(file);
      runtime.setSessionFrameProvider(binding.sessionKey, player, {
        ownsProvider: true,
      });

      if (Number.isFinite(initialTime) && initialTime !== undefined) {
        player.seek(initialTime);
      }
      engine.requestRender();
      return player;
    } catch {
      player.destroy?.();
      return null;
    } finally {
      pendingRuntimeProviderLoads.delete(loadKey);
    }
  })();

  pendingRuntimeProviderLoads.set(loadKey, loadPromise);
  return loadPromise;
}

export function releaseRuntimePlaybackSession(
  source: RuntimeBackedSource | null | undefined
): void {
  if (!hasRuntimeBinding(source)) {
    return;
  }
  pendingRuntimeProviderLoads.delete(
    getPendingProviderLoadKey(source.runtimeSourceId, source.runtimeSessionKey)
  );
  mediaRuntimeRegistry.releaseSession(
    source.runtimeSourceId,
    source.runtimeSessionKey
  );
}

export function readRuntimeFrameForSource(
  source: RuntimeBackedSource | null | undefined,
  policy: DecodeSessionPolicy = 'interactive'
): {
  binding: RuntimePlaybackBinding;
  frameHandle: FrameHandle | null;
} | null {
  const binding = resolveRuntimePlaybackBinding(source, policy);
  if (!binding) {
    return null;
  }

  const runtime = mediaRuntimeRegistry.getRuntime(binding.sourceId);
  if (!runtime) {
    return null;
  }

  const frameHandle = runtime.getFrameSync({
    sourceId: binding.sourceId,
    sessionKey: binding.sessionKey,
    sourceTime: binding.session.currentTime,
    playbackMode: binding.session.policy,
    allowCache: true,
  });

  return {
    binding,
    frameHandle,
  };
}
