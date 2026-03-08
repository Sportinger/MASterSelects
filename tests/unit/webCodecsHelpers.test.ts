import { beforeEach, describe, expect, it, vi } from 'vitest';
import { WebCodecsPlayer } from '../../src/engine/WebCodecsPlayer';
import { engine } from '../../src/engine/WebGPUEngine';
import { flags } from '../../src/engine/featureFlags';
import { initWebCodecsPlayer } from '../../src/stores/timeline/helpers/webCodecsHelpers';

describe('initWebCodecsPlayer', () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.mocked(WebCodecsPlayer).mockReset();
    vi.mocked(engine.requestNewFrameRender).mockReset();
    vi.mocked(engine.requestRender).mockReset();
    flags.useFullWebCodecsPlayback = true;
    (window as any).VideoDecoder = vi.fn();
    (window as any).VideoFrame = vi.fn();
  });

  it('wakes the renderer when a normal full WebCodecs player emits a frame', async () => {
    const loadFile = vi.fn().mockResolvedValue(undefined);
    vi.mocked(WebCodecsPlayer).mockImplementation(function MockWebCodecsPlayer(options: any) {
      (this as any).loadFile = loadFile;
      (this as any).attachToVideoElement = vi.fn();
      (this as any).ready = true;
      (this as any).isFullMode = () => true;
      (this as any).__options = options;
      return this as any;
    } as any);

    const video = document.createElement('video');
    const file = new File(['video'], 'clip.mp4', { type: 'video/mp4' });
    const player = await initWebCodecsPlayer(video, file.name, file);

    expect(player).toBeTruthy();
    const options = vi.mocked(WebCodecsPlayer).mock.calls[0]?.[0] as any;
    expect(typeof options.onFrame).toBe('function');

    options.onFrame();

    expect(engine.requestNewFrameRender).toHaveBeenCalledTimes(1);
  });

  it('waits for full WebCodecs readiness before returning', async () => {
    vi.useFakeTimers();

    const loadFile = vi.fn().mockResolvedValue(undefined);
    let ready = false;
    vi.mocked(WebCodecsPlayer).mockImplementation(function MockWebCodecsPlayer() {
      (this as any).loadFile = loadFile;
      (this as any).attachToVideoElement = vi.fn();
      Object.defineProperty(this, 'ready', {
        configurable: true,
        get: () => ready,
        set: (value: boolean) => {
          ready = value;
        },
      });
      return this as any;
    } as any);

    const video = document.createElement('video');
    const file = new File(['video'], 'delayed-ready.mp4', { type: 'video/mp4' });

    let resolved = false;
    const playerPromise = initWebCodecsPlayer(video, file.name, file).then((player) => {
      resolved = true;
      return player;
    });

    await vi.runAllTicks();
    expect(loadFile).toHaveBeenCalledWith(file);
    expect(resolved).toBe(false);

    ready = true;
    await vi.advanceTimersByTimeAsync(32);

    await expect(playerPromise).resolves.toBeTruthy();
    expect(resolved).toBe(true);
  });
});
