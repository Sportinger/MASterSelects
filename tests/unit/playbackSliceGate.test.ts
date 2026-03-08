import { describe, expect, it, vi, beforeEach } from 'vitest';
import { createPlaybackSlice } from '../../src/stores/timeline/playbackSlice';

const getRuntimeFrameProvider = vi.fn();

vi.mock('../../src/stores/mediaStore', () => ({
  useMediaStore: {
    getState: () => ({
      activeCompositionId: null,
      updateComposition: vi.fn(),
    }),
  },
}));

vi.mock('../../src/services/mediaRuntime/runtimePlayback', () => ({
  getRuntimeFrameProvider: (...args: unknown[]) => getRuntimeFrameProvider(...args),
}));

describe('playbackSlice HTML readiness gate', () => {
  beforeEach(() => {
    getRuntimeFrameProvider.mockReset();
  });

  it('skips HTML readiness warmup for full WebCodecs clips', async () => {
    const htmlVideo = {
      readyState: 0,
      play: vi.fn().mockResolvedValue(undefined),
      pause: vi.fn(),
    } as unknown as HTMLVideoElement;

    const fullModeProvider = {
      isFullMode: () => true,
    };

    getRuntimeFrameProvider.mockReturnValue(fullModeProvider);

    const state: Record<string, any> = {
      clips: [
        {
          id: 'clip-1',
          startTime: 0,
          duration: 10,
          source: {
            videoElement: htmlVideo,
            webCodecsPlayer: fullModeProvider,
          },
        },
      ],
      playheadPosition: 1,
      duration: 60,
      isPlaying: false,
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      Object.assign(state, next);
    };
    const get = () => state as any;

    Object.assign(state, createPlaybackSlice(set as any, get as any));

    await state.play();

    expect(state.isPlaying).toBe(true);
    expect(htmlVideo.play).not.toHaveBeenCalled();
    expect(htmlVideo.pause).not.toHaveBeenCalled();
  });
});
