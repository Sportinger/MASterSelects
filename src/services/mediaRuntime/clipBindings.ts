import type { TimelineClip } from '../../types';
import { useMediaStore } from '../../stores/mediaStore';
import { mediaRuntimeRegistry } from './registry';
import type {
  DecodeSessionPolicy,
  MediaRuntimeKind,
  MediaSourceMetadata,
  MediaSourceRuntimeDescriptor,
} from './types';

type ClipSource = NonNullable<TimelineClip['source']>;

function getRuntimeKind(type: ClipSource['type']): MediaRuntimeKind | null {
  if (type === 'video' || type === 'audio' || type === 'image') {
    return type;
  }
  return null;
}

function buildRuntimeMetadata(
  source: ClipSource,
  mediaFileId?: string
): MediaSourceMetadata {
  const mediaFile = mediaFileId
    ? useMediaStore.getState().files.find((file) => file.id === mediaFileId)
    : undefined;

  return {
    duration: source.naturalDuration ?? mediaFile?.duration,
    width: mediaFile?.width,
    height: mediaFile?.height,
    fps: mediaFile?.fps,
    codec: mediaFile?.codec,
    audioCodec: mediaFile?.audioCodec,
    container: mediaFile?.container,
    hasAudio: mediaFile?.hasAudio,
  };
}

function buildRuntimeDescriptor(params: {
  source: ClipSource;
  file?: File;
  mediaFileId?: string;
  filePath?: string;
}): MediaSourceRuntimeDescriptor | null {
  const kind = getRuntimeKind(params.source.type);
  if (!kind) {
    return null;
  }

  const mediaFile = params.mediaFileId
    ? useMediaStore.getState().files.find((file) => file.id === params.mediaFileId)
    : undefined;

  const file =
    (params.file && params.file.size > 0 ? params.file : undefined) ??
    mediaFile?.file;
  const filePath =
    params.filePath ??
    params.source.filePath ??
    mediaFile?.absolutePath ??
    mediaFile?.filePath;

  return {
    kind,
    mediaFileId: params.mediaFileId,
    file,
    fileName: file?.name ?? mediaFile?.name,
    fileSize: file?.size ?? mediaFile?.fileSize,
    fileLastModified: file?.lastModified,
    fileHash: mediaFile?.fileHash,
    filePath,
  };
}

export function bindSourceRuntimeToClip(params: {
  clipId: string;
  source: TimelineClip['source'];
  file?: File;
  mediaFileId?: string;
  filePath?: string;
  sessionPolicy?: DecodeSessionPolicy;
  sessionOwnerId?: string;
}): TimelineClip['source'] {
  const source = params.source;
  if (!source) {
    return source;
  }

  const mediaFileId = params.mediaFileId ?? source.mediaFileId;
  const descriptor = buildRuntimeDescriptor({
    source,
    file: params.file,
    mediaFileId,
    filePath: params.filePath,
  });

  if (!descriptor) {
    return source;
  }

  const runtime = mediaRuntimeRegistry.retainRuntime(descriptor, params.clipId);
  if (!runtime) {
    return source;
  }

  runtime.updateMetadata(buildRuntimeMetadata(source, mediaFileId));

  const sessionKey = `${params.sessionPolicy ?? 'interactive'}:${params.sessionOwnerId ?? params.clipId}`;
  runtime.getSession(sessionKey, {
    ownerId: params.clipId,
    policy: params.sessionPolicy ?? 'interactive',
  });

  return {
    ...source,
    mediaFileId,
    filePath: descriptor.filePath ?? source.filePath,
    runtimeSourceId: runtime.sourceId,
    runtimeSessionKey: sessionKey,
  };
}

export function bindSourceRuntimeForOwner(params: {
  ownerId: string;
  source: TimelineClip['source'];
  file?: File;
  mediaFileId?: string;
  filePath?: string;
  sessionPolicy?: DecodeSessionPolicy;
  sessionOwnerId?: string;
}): TimelineClip['source'] {
  return bindSourceRuntimeToClip({
    clipId: params.ownerId,
    source: params.source,
    file: params.file,
    mediaFileId: params.mediaFileId,
    filePath: params.filePath,
    sessionPolicy: params.sessionPolicy,
    sessionOwnerId: params.sessionOwnerId,
  });
}

export function bindRuntimeToClip(
  clip: TimelineClip,
  options?: {
    source?: TimelineClip['source'];
    file?: File;
    mediaFileId?: string;
    filePath?: string;
    sessionPolicy?: DecodeSessionPolicy;
  }
): TimelineClip {
  const source = bindSourceRuntimeToClip({
    clipId: clip.id,
    source: options?.source ?? clip.source,
    file: options?.file ?? clip.file,
    mediaFileId: options?.mediaFileId ?? clip.mediaFileId ?? clip.source?.mediaFileId,
    filePath: options?.filePath ?? clip.source?.filePath,
    sessionPolicy: options?.sessionPolicy,
  });

  if (source === clip.source) {
    return clip;
  }

  return {
    ...clip,
    source,
  };
}

export function releaseClipSourceRuntime(
  clip: Pick<TimelineClip, 'id' | 'source'>
): void {
  mediaRuntimeRegistry.releaseClip(
    clip.id,
    clip.source?.runtimeSourceId,
    clip.source?.runtimeSessionKey
  );
}

export function releaseClipTreeRuntimeBindings(clip: TimelineClip): void {
  releaseClipSourceRuntime(clip);
  if (!clip.nestedClips) {
    return;
  }
  for (const nestedClip of clip.nestedClips) {
    releaseClipTreeRuntimeBindings(nestedClip);
  }
}
