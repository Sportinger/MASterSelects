import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useMediaStore } from '../../src/stores/mediaStore';
import {
  bindSourceRuntimeToClip,
  releaseClipTreeRuntimeBindings,
} from '../../src/services/mediaRuntime/clipBindings';
import { mediaRuntimeRegistry } from '../../src/services/mediaRuntime/registry';
import {
  canUseSharedPreviewRuntimeSession,
  ensureRuntimeFrameProvider,
  getPolicyRuntimeSource,
  getRuntimeFrameProvider,
  getPreviewRuntimeSource,
  getScrubRuntimeSource,
  releaseRuntimePlaybackSession,
  readRuntimeFrameForSource,
  setRuntimeFrameProvider,
  updateRuntimePlaybackTime,
} from '../../src/services/mediaRuntime/runtimePlayback';
import { WebCodecsPlayer } from '../../src/engine/WebCodecsPlayer';
import type { TimelineClip } from '../../src/types';

function makeTransform() {
  return {
    opacity: 1,
    blendMode: 'normal' as const,
    position: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1 },
    rotation: { x: 0, y: 0, z: 0 },
  };
}

function makeClip(
  id: string,
  file: File,
  source: TimelineClip['source'],
  nestedClips?: TimelineClip[]
): TimelineClip {
  return {
    id,
    trackId: 'track-1',
    name: id,
    file,
    startTime: 0,
    duration: 5,
    inPoint: 0,
    outPoint: 5,
    source,
    transform: makeTransform(),
    effects: [],
    isLoading: false,
    ...(nestedClips ? { nestedClips } : {}),
  };
}

function setMediaFiles(files: unknown[]): void {
  const mediaStoreMock = useMediaStore as unknown as {
    getState: ReturnType<typeof vi.fn>;
  };
  mediaStoreMock.getState.mockReturnValue({
    files,
    compositions: [],
    folders: [],
    selectedIds: [],
    expandedFolderIds: [],
    textItems: [],
    solidItems: [],
    activeCompositionId: null,
    outputResolution: { width: 1920, height: 1080 },
    addMediaFile: vi.fn(),
    updateComposition: vi.fn(),
    getActiveComposition: vi.fn().mockReturnValue({ width: 1920, height: 1080 }),
    getOrCreateTextFolder: vi.fn().mockReturnValue('text-folder-1'),
    createTextItem: vi.fn(),
    getOrCreateSolidFolder: vi.fn().mockReturnValue('solid-folder-1'),
    createSolidItem: vi.fn(),
  });
}

function makeCloneableVideoFrame(
  timestamp: number,
  cloneCloseSpies?: Array<ReturnType<typeof vi.fn>>
): VideoFrame {
  return {
    timestamp,
    displayWidth: 1920,
    displayHeight: 1080,
    codedWidth: 1920,
    codedHeight: 1080,
    close: vi.fn(),
    clone: vi.fn(() => {
      const closeSpy = vi.fn();
      cloneCloseSpies?.push(closeSpy);
      return {
        timestamp,
        displayWidth: 1920,
        displayHeight: 1080,
        codedWidth: 1920,
        codedHeight: 1080,
        close: closeSpy,
      } as VideoFrame;
    }),
  } as VideoFrame;
}

describe('media runtime bindings', () => {
  beforeEach(() => {
    mediaRuntimeRegistry.clear();
    setMediaFiles([]);
    vi.mocked(WebCodecsPlayer).mockReset();
  });

  it('uses mediaFileId as the canonical runtime identity', () => {
    const file = new File(['video'], 'demo.mp4', { type: 'video/mp4', lastModified: 1 });
    setMediaFiles([
      {
        id: 'media-1',
        file,
        name: 'demo.mp4',
        duration: 12,
        width: 1920,
        height: 1080,
        fps: 24,
      },
    ]);

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-1',
      source: {
        type: 'video',
        naturalDuration: 12,
        mediaFileId: 'media-1',
      },
      file,
      mediaFileId: 'media-1',
    });

    expect(source?.runtimeSourceId).toBe('media:media-1');
    expect(source?.runtimeSessionKey).toBe('interactive:clip-1');
    expect(mediaRuntimeRegistry.listRuntimes()).toHaveLength(1);
    expect(mediaRuntimeRegistry.getRuntime('media:media-1')?.ownerCount()).toBe(1);
    expect(mediaRuntimeRegistry.getRuntime('media:media-1')?.metadata.duration).toBe(12);
  });

  it('reuses a single runtime for multiple clips referencing the same media file', () => {
    const file = new File(['video'], 'shared.mp4', { type: 'video/mp4', lastModified: 2 });
    setMediaFiles([
      {
        id: 'media-shared',
        file,
        name: 'shared.mp4',
        duration: 20,
      },
    ]);

    bindSourceRuntimeToClip({
      clipId: 'clip-a',
      source: {
        type: 'video',
        naturalDuration: 20,
        mediaFileId: 'media-shared',
      },
      file,
      mediaFileId: 'media-shared',
    });

    bindSourceRuntimeToClip({
      clipId: 'clip-b',
      source: {
        type: 'video',
        naturalDuration: 20,
        mediaFileId: 'media-shared',
      },
      file,
      mediaFileId: 'media-shared',
    });

    expect(mediaRuntimeRegistry.listRuntimes()).toHaveLength(1);
    expect(mediaRuntimeRegistry.getRuntime('media:media-shared')?.ownerCount()).toBe(2);
  });

  it('releases runtimes recursively for nested clip trees', () => {
    const file = new File(['video'], 'nested.mp4', { type: 'video/mp4', lastModified: 3 });
    setMediaFiles([
      {
        id: 'media-nested',
        file,
        name: 'nested.mp4',
        duration: 8,
      },
    ]);

    const parentSource = bindSourceRuntimeToClip({
      clipId: 'clip-parent',
      source: {
        type: 'video',
        naturalDuration: 8,
        mediaFileId: 'media-nested',
      },
      file,
      mediaFileId: 'media-nested',
    });
    const nestedSource = bindSourceRuntimeToClip({
      clipId: 'clip-child',
      source: {
        type: 'video',
        naturalDuration: 8,
        mediaFileId: 'media-nested',
      },
      file,
      mediaFileId: 'media-nested',
    });

    const nestedClip = makeClip('clip-child', file, nestedSource);
    const parentClip = makeClip('clip-parent', file, parentSource, [nestedClip]);

    expect(mediaRuntimeRegistry.getRuntime('media:media-nested')?.ownerCount()).toBe(2);

    releaseClipTreeRuntimeBindings(parentClip);

    expect(mediaRuntimeRegistry.listRuntimes()).toHaveLength(0);
  });

  it('skips runtime creation for placeholder files without a stable source identity', () => {
    const placeholderFile = new File([], 'placeholder.wav', { type: 'audio/wav', lastModified: 4 });

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-placeholder',
      source: {
        type: 'audio',
        naturalDuration: 3,
      },
      file: placeholderFile,
    });

    expect(source?.runtimeSourceId).toBeUndefined();
    expect(source?.runtimeSessionKey).toBeUndefined();
    expect(mediaRuntimeRegistry.listRuntimes()).toHaveLength(0);
  });

  it('attaches a clip WebCodecs player to the runtime session on demand', () => {
    const file = new File(['video'], 'session.mp4', { type: 'video/mp4', lastModified: 5 });
    setMediaFiles([
      {
        id: 'media-session',
        file,
        name: 'session.mp4',
        duration: 6,
      },
    ]);

    const frame = {
      timestamp: 2_000_000,
      displayWidth: 1280,
      displayHeight: 720,
    } as VideoFrame;
    const player = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => frame,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-session',
      source: {
        type: 'video',
        naturalDuration: 6,
        mediaFileId: 'media-session',
        webCodecsPlayer: player as any,
      },
      file,
      mediaFileId: 'media-session',
    });

    const provider = getRuntimeFrameProvider(source);

    expect(provider).toBe(player);
    expect(
      mediaRuntimeRegistry.getSession('media:media-session', 'interactive:clip-session')?.frameProvider
    ).toBe(player);
  });

  it('reads the current frame through the runtime-backed preview binding', () => {
    const file = new File(['video'], 'frame.mp4', { type: 'video/mp4', lastModified: 6 });
    setMediaFiles([
      {
        id: 'media-frame',
        file,
        name: 'frame.mp4',
        duration: 9,
      },
    ]);

    const frame = {
      timestamp: 3_500_000,
      displayWidth: 1920,
      displayHeight: 1080,
    } as VideoFrame;
    const player = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => frame,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-frame',
      source: {
        type: 'video',
        naturalDuration: 9,
        mediaFileId: 'media-frame',
        webCodecsPlayer: player as any,
      },
      file,
      mediaFileId: 'media-frame',
    });

    updateRuntimePlaybackTime(source, 3.5);
    const runtimeFrame = readRuntimeFrameForSource(source);

    expect(runtimeFrame?.binding.session.currentTime).toBe(3.5);
    expect(runtimeFrame?.frameHandle?.frame).toBe(frame);
    expect(runtimeFrame?.frameHandle?.timestamp).toBe(3_500_000);
  });

  it('reuses one shared preview session for sequential same-source full WebCodecs clips on the same track', () => {
    const file = new File(['video'], 'sequence.mp4', { type: 'video/mp4', lastModified: 7 });
    setMediaFiles([
      {
        id: 'media-sequence',
        file,
        name: 'sequence.mp4',
        duration: 10,
      },
    ]);

    const playerA = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };
    const playerB = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const sourceA = bindSourceRuntimeToClip({
      clipId: 'clip-seq-a',
      source: {
        type: 'video',
        naturalDuration: 10,
        mediaFileId: 'media-sequence',
        webCodecsPlayer: playerA as any,
      },
      file,
      mediaFileId: 'media-sequence',
    });
    const sourceB = bindSourceRuntimeToClip({
      clipId: 'clip-seq-b',
      source: {
        type: 'video',
        naturalDuration: 10,
        mediaFileId: 'media-sequence',
        webCodecsPlayer: playerB as any,
      },
      file,
      mediaFileId: 'media-sequence',
    });

    expect(
      canUseSharedPreviewRuntimeSession(
        { trackId: 'track-1' },
        [{ trackId: 'track-1' }]
      )
    ).toBe(true);

    const previewSourceA = getPreviewRuntimeSource(sourceA, 'track-1', true);
    const previewSourceB = getPreviewRuntimeSource(sourceB, 'track-1', true);

    expect(previewSourceA?.runtimeSessionKey).toBe(
      'interactive-track:track-1:media:media-sequence'
    );
    expect(previewSourceB?.runtimeSessionKey).toBe(
      'interactive-track:track-1:media:media-sequence'
    );

    const providerA = getRuntimeFrameProvider(previewSourceA);
    const providerB = getRuntimeFrameProvider(previewSourceB);

    expect(providerA).toBe(playerA);
    expect(providerB).toBe(playerA);
    expect(
      mediaRuntimeRegistry.getSession(
        'media:media-sequence',
        'interactive-track:track-1:media:media-sequence'
      )?.frameProvider
    ).toBe(playerA);
  });

  it('scopes shared preview sessions by nested composition context', () => {
    const file = new File(['video'], 'nested-scope.mp4', { type: 'video/mp4', lastModified: 7_1 });
    setMediaFiles([
      {
        id: 'media-nested-scope',
        file,
        name: 'nested-scope.mp4',
        duration: 10,
      },
    ]);

    const playerA = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };
    const playerB = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const sourceA = bindSourceRuntimeToClip({
      clipId: 'clip-nested-scope-a',
      source: {
        type: 'video',
        naturalDuration: 10,
        mediaFileId: 'media-nested-scope',
        webCodecsPlayer: playerA as any,
      },
      file,
      mediaFileId: 'media-nested-scope',
    });
    const sourceB = bindSourceRuntimeToClip({
      clipId: 'clip-nested-scope-b',
      source: {
        type: 'video',
        naturalDuration: 10,
        mediaFileId: 'media-nested-scope',
        webCodecsPlayer: playerB as any,
      },
      file,
      mediaFileId: 'media-nested-scope',
    });

    const previewSourceA = getPreviewRuntimeSource(
      sourceA,
      'track-1',
      true,
      'composition:comp-a/nested:n1'
    );
    const previewSourceB = getPreviewRuntimeSource(
      sourceB,
      'track-1',
      true,
      'composition:comp-b/nested:n1'
    );

    expect(previewSourceA?.runtimeSessionKey).toBe(
      'interactive-track:composition:comp-a/nested:n1:track-1:media:media-nested-scope'
    );
    expect(previewSourceB?.runtimeSessionKey).toBe(
      'interactive-track:composition:comp-b/nested:n1:track-1:media:media-nested-scope'
    );
    expect(getRuntimeFrameProvider(previewSourceA)).toBe(playerA);
    expect(getRuntimeFrameProvider(previewSourceB)).toBe(playerB);
  });

  it('creates distinct policy-scoped runtime sessions for export and RAM preview', () => {
    const file = new File(['video'], 'policy-scope.mp4', { type: 'video/mp4', lastModified: 72 });
    setMediaFiles([
      {
        id: 'media-policy-scope',
        file,
        name: 'policy-scope.mp4',
        duration: 8,
      },
    ]);

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-policy-scope',
      source: {
        type: 'video',
        naturalDuration: 8,
        mediaFileId: 'media-policy-scope',
      },
      file,
      mediaFileId: 'media-policy-scope',
    });

    const exportSource = getPolicyRuntimeSource(
      source,
      'export',
      'clip-policy-scope'
    );
    const ramPreviewSource = getPolicyRuntimeSource(
      source,
      'ram-preview',
      'clip-policy-scope',
      'nested:comp-1'
    );

    expect(exportSource?.runtimeSessionKey).toBe(
      'export:clip-policy-scope:media:media-policy-scope'
    );
    expect(ramPreviewSource?.runtimeSessionKey).toBe(
      'ram-preview:nested:comp-1:clip-policy-scope:media:media-policy-scope'
    );
  });

  it('attaches and releases providers on explicit export runtime sessions', () => {
    const file = new File(['video'], 'export-session.mp4', { type: 'video/mp4', lastModified: 73 });
    setMediaFiles([
      {
        id: 'media-export-session',
        file,
        name: 'export-session.mp4',
        duration: 8,
      },
    ]);

    const player = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-export-session',
      source: {
        type: 'video',
        naturalDuration: 8,
        mediaFileId: 'media-export-session',
      },
      file,
      mediaFileId: 'media-export-session',
    });
    const exportSource = getPolicyRuntimeSource(
      source,
      'export',
      'clip-export-session'
    );

    setRuntimeFrameProvider(exportSource, player as any, 'export');

    expect(getRuntimeFrameProvider(exportSource, 'export')).toBe(player);

    releaseRuntimePlaybackSession(exportSource);

    expect(
      mediaRuntimeRegistry.getSession(
        'media:media-export-session',
        'export:clip-export-session:media:media-export-session'
      )
    ).toBeNull();
  });

  it('keeps preview and export runtime sessions isolated for the same source', () => {
    const file = new File(['video'], 'preview-export.mp4', { type: 'video/mp4', lastModified: 74 });
    setMediaFiles([
      {
        id: 'media-preview-export',
        file,
        name: 'preview-export.mp4',
        duration: 8,
      },
    ]);

    const previewPlayer = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };
    const exportPlayer = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-preview-export',
      source: {
        type: 'video',
        naturalDuration: 8,
        mediaFileId: 'media-preview-export',
        webCodecsPlayer: previewPlayer as any,
      },
      file,
      mediaFileId: 'media-preview-export',
    });
    const exportSource = getPolicyRuntimeSource(
      source,
      'export',
      'clip-preview-export'
    );

    setRuntimeFrameProvider(exportSource, exportPlayer as any, 'export');

    updateRuntimePlaybackTime(source, 1.25);
    updateRuntimePlaybackTime(exportSource, 4.5, 'export');

    expect(getRuntimeFrameProvider(source)).toBe(previewPlayer);
    expect(getRuntimeFrameProvider(exportSource, 'export')).toBe(exportPlayer);
    expect(
      mediaRuntimeRegistry.getSession(
        'media:media-preview-export',
        'interactive:clip-preview-export'
      )?.currentTime
    ).toBe(1.25);
    expect(
      mediaRuntimeRegistry.getSession(
        'media:media-preview-export',
        'export:clip-preview-export:media:media-preview-export'
      )?.currentTime
    ).toBe(4.5);

    releaseRuntimePlaybackSession(exportSource);

    expect(
      mediaRuntimeRegistry.getSession(
        'media:media-preview-export',
        'interactive:clip-preview-export'
      )
    ).not.toBeNull();
  });

  it('keeps clip-local runtime sessions when multiple clips overlap on the same track', () => {
    const file = new File(['video'], 'overlap.mp4', { type: 'video/mp4', lastModified: 8 });
    setMediaFiles([
      {
        id: 'media-overlap',
        file,
        name: 'overlap.mp4',
        duration: 10,
      },
    ]);

    const player = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-overlap',
      source: {
        type: 'video',
        naturalDuration: 10,
        mediaFileId: 'media-overlap',
        webCodecsPlayer: player as any,
      },
      file,
      mediaFileId: 'media-overlap',
    });

    expect(
      canUseSharedPreviewRuntimeSession(
        { trackId: 'track-1' },
        [{ trackId: 'track-1' }, { trackId: 'track-1' }]
      )
    ).toBe(false);

    const previewSource = getPreviewRuntimeSource(source, 'track-1', false);
    expect(previewSource?.runtimeSessionKey).toBe('interactive:clip-overlap');
  });

  it('creates a dedicated full WebCodecs scrub provider without replacing the playback provider', async () => {
    const file = new File(['video'], 'scrub-separate.mp4', { type: 'video/mp4', lastModified: 81 });
    setMediaFiles([
      {
        id: 'media-scrub-separate',
        file,
        name: 'scrub-separate.mp4',
        duration: 9,
      },
    ]);

    const previewPlayer = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const scrubPlayer = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      destroy: vi.fn(),
      loadFile: vi.fn().mockResolvedValue(undefined),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    vi.mocked(WebCodecsPlayer).mockImplementation(function MockWebCodecsPlayer() {
      return scrubPlayer as any;
    } as any);

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-scrub-separate',
      source: {
        type: 'video',
        naturalDuration: 9,
        mediaFileId: 'media-scrub-separate',
        webCodecsPlayer: previewPlayer as any,
      },
      file,
      mediaFileId: 'media-scrub-separate',
    });
    const scrubSource = getScrubRuntimeSource(source, 'track-1', true);

    const provider = await ensureRuntimeFrameProvider(scrubSource, 'interactive', 3.25);

    expect(scrubSource?.runtimeSessionKey).toBe(
      'interactive-scrub:track-1:media:media-scrub-separate'
    );
    expect(provider).toBe(scrubPlayer);
    expect(getRuntimeFrameProvider(source)).toBe(previewPlayer);
    expect(getRuntimeFrameProvider(scrubSource)).toBe(scrubPlayer);
    expect(scrubPlayer.loadFile).toHaveBeenCalledWith(file);
    expect(scrubPlayer.seek).toHaveBeenCalledWith(3.25);

    releaseRuntimePlaybackSession(scrubSource);

    expect(scrubPlayer.destroy).toHaveBeenCalledTimes(1);
    expect(
      mediaRuntimeRegistry.getSession(
        'media:media-scrub-separate',
        'interactive-scrub:track-1:media:media-scrub-separate'
      )
    ).toBeNull();
  });

  it('reuses a shared cached frame across simultaneous same-source sessions at the same source time', () => {
    const file = new File(['video'], 'simul.mp4', { type: 'video/mp4', lastModified: 9 });
    setMediaFiles([
      {
        id: 'media-simul',
        file,
        name: 'simul.mp4',
        duration: 12,
      },
    ]);

    const playerA = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => makeCloneableVideoFrame(2_000_000),
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };
    const playerB = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const sourceA = bindSourceRuntimeToClip({
      clipId: 'clip-simul-a',
      source: {
        type: 'video',
        naturalDuration: 12,
        mediaFileId: 'media-simul',
        webCodecsPlayer: playerA as any,
      },
      file,
      mediaFileId: 'media-simul',
    });
    const sourceB = bindSourceRuntimeToClip({
      clipId: 'clip-simul-b',
      source: {
        type: 'video',
        naturalDuration: 12,
        mediaFileId: 'media-simul',
        webCodecsPlayer: playerB as any,
      },
      file,
      mediaFileId: 'media-simul',
    });

    const previewSourceA = getPreviewRuntimeSource(sourceA, 'track-1', true);
    const previewSourceB = getPreviewRuntimeSource(sourceB, 'track-2', true);

    updateRuntimePlaybackTime(previewSourceA, 2);
    updateRuntimePlaybackTime(previewSourceB, 2);

    const runtimeFrame = readRuntimeFrameForSource(previewSourceB);

    expect(runtimeFrame?.frameHandle?.frame).not.toBeNull();
    expect(runtimeFrame?.frameHandle?.timestamp).toBe(2_000_000);
    expect(mediaRuntimeRegistry.getRuntime('media:media-simul')?.frameCache.size).toBe(1);
  });

  it('does not reuse a simultaneous session frame when the requested source times differ', () => {
    const file = new File(['video'], 'simul-drift.mp4', { type: 'video/mp4', lastModified: 10 });
    setMediaFiles([
      {
        id: 'media-simul-drift',
        file,
        name: 'simul-drift.mp4',
        duration: 12,
      },
    ]);

    const playerA = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => makeCloneableVideoFrame(2_000_000),
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };
    const playerB = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => null,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const sourceA = bindSourceRuntimeToClip({
      clipId: 'clip-simul-drift-a',
      source: {
        type: 'video',
        naturalDuration: 12,
        mediaFileId: 'media-simul-drift',
        webCodecsPlayer: playerA as any,
      },
      file,
      mediaFileId: 'media-simul-drift',
    });
    const sourceB = bindSourceRuntimeToClip({
      clipId: 'clip-simul-drift-b',
      source: {
        type: 'video',
        naturalDuration: 12,
        mediaFileId: 'media-simul-drift',
        webCodecsPlayer: playerB as any,
      },
      file,
      mediaFileId: 'media-simul-drift',
    });

    const previewSourceA = getPreviewRuntimeSource(sourceA, 'track-1', true);
    const previewSourceB = getPreviewRuntimeSource(sourceB, 'track-2', true);

    updateRuntimePlaybackTime(previewSourceA, 2);
    updateRuntimePlaybackTime(previewSourceB, 4);

    const runtimeFrame = readRuntimeFrameForSource(previewSourceB);

    expect(runtimeFrame?.frameHandle).toBeNull();
    expect(mediaRuntimeRegistry.getRuntime('media:media-simul-drift')?.frameCache.size).toBe(0);
  });

  it('bounds the shared frame cache and releases evicted cached frames', () => {
    const file = new File(['video'], 'cache.mp4', { type: 'video/mp4', lastModified: 11 });
    const cachedCloneCloseSpies: Array<ReturnType<typeof vi.fn>> = [];
    let currentFrame = makeCloneableVideoFrame(0, cachedCloneCloseSpies);

    setMediaFiles([
      {
        id: 'media-cache',
        file,
        name: 'cache.mp4',
        duration: 20,
      },
    ]);

    const player = {
      currentTime: 0,
      isPlaying: false,
      isFullMode: () => true,
      isSimpleMode: () => false,
      getCurrentFrame: () => currentFrame,
      seek: vi.fn(),
      pause: vi.fn(),
      getDebugInfo: vi.fn().mockReturnValue(null),
    };

    const source = bindSourceRuntimeToClip({
      clipId: 'clip-cache',
      source: {
        type: 'video',
        naturalDuration: 20,
        mediaFileId: 'media-cache',
        webCodecsPlayer: player as any,
      },
      file,
      mediaFileId: 'media-cache',
    });

    const previewSource = getPreviewRuntimeSource(source, 'track-1', true);

    for (let i = 0; i < 13; i++) {
      currentFrame = makeCloneableVideoFrame(i * 1_000_000, cachedCloneCloseSpies);
      updateRuntimePlaybackTime(previewSource, i);
      readRuntimeFrameForSource(previewSource);
    }

    const runtime = mediaRuntimeRegistry.getRuntime('media:media-cache');

    expect(runtime?.frameCache.size).toBeLessThanOrEqual(12);
    expect(cachedCloneCloseSpies[0]).toHaveBeenCalled();
  });
});
