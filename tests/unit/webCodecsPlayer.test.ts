import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('mp4box', () => ({ default: {} }));

type WebCodecsPlayerModule = typeof import('../../src/engine/WebCodecsPlayer');

class MockEncodedVideoChunk {
  constructor(public readonly init: Record<string, unknown>) {}
}

function makeSamples(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    cts: index,
    duration: 1,
    timescale: 30,
    is_sync: index === 0,
    data: new Uint8Array([index % 255]),
  }));
}

function makeDecoder() {
  const decoder = {
    state: 'configured',
    decodeQueueSize: 0,
    decode: vi.fn(() => {}),
    reset: vi.fn(() => {
      decoder.decodeQueueSize = 0;
    }),
    configure: vi.fn(),
    flush: vi.fn().mockResolvedValue(undefined),
    close: vi.fn(),
  };

  return decoder;
}

async function makePlayerHarness() {
  const module = await vi.importActual<WebCodecsPlayerModule>(
    '../../src/engine/WebCodecsPlayer'
  );
  const player = new module.WebCodecsPlayer() as unknown as Record<string, any>;
  const decoder = makeDecoder();

  player.useSimpleMode = false;
  player.ready = true;
  player.decoder = decoder;
  player.codecConfig = { codec: 'avc1.test' };
  player.videoTrack = { timescale: 30 };
  player.samples = makeSamples(120);
  player.frameRate = 30;
  player.frameBuffer = [];
  player.sampleIndex = 0;
  player.feedIndex = 0;
  player.currentFrame = null;
  player.currentFrameTimestampUs = null;
  player.pendingAdvanceSeekTargetIdx = null;
  player.trackedDecodeQueueSize = 0;
  player._isPlaying = false;

  return { player, decoder };
}

describe('WebCodecsPlayer advance playback', () => {
  beforeEach(() => {
    (globalThis as Record<string, unknown>).EncodedVideoChunk = MockEncodedVideoChunk;
  });

  it('caps advance feeding when decodeQueueSize lags behind decode() calls', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.advanceToTime(2);

    expect(decoder.reset).toHaveBeenCalledTimes(1);
    expect(decoder.decode).toHaveBeenCalledTimes(24);
    expect(player.feedIndex).toBe(24);
    expect(player.trackedDecodeQueueSize).toBe(24);
  });

  it('continues an in-flight advance seek without moving the pending resolve target forward', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.currentFrame = { timestamp: 0, close: vi.fn() };
    player.currentFrameTimestampUs = 0;

    player.advanceToTime(2);

    expect(decoder.reset).toHaveBeenCalledTimes(1);
    expect(player.pendingAdvanceSeekTargetIdx).toBe(60);
    expect(player.feedIndex).toBe(24);

    player.trackedDecodeQueueSize = 0;
    decoder.decodeQueueSize = 0;

    player.advanceToTime(2.1);

    expect(decoder.reset).toHaveBeenCalledTimes(1);
    expect(decoder.configure).toHaveBeenCalledTimes(1);
    expect(decoder.decode).toHaveBeenCalledTimes(48);
    expect(player.feedIndex).toBe(48);
    expect(player.pendingAdvanceSeekTargetIdx).toBe(60);
  });

  it('keeps an in-flight advance seek alive while playback moves forward within the timeout window', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.currentFrame = { timestamp: 0, close: vi.fn() };
    player.currentFrameTimestampUs = 0;

    player.advanceToTime(2);

    expect(decoder.reset).toHaveBeenCalledTimes(1);
    expect(player.pendingAdvanceSeekTargetIdx).toBe(60);

    player.trackedDecodeQueueSize = 0;
    decoder.decodeQueueSize = 0;
    player.pendingSeekStartedAtMs = performance.now() - 500;

    player.advanceToTime(3.2);

    expect(decoder.reset).toHaveBeenCalledTimes(1);
    expect(player.pendingAdvanceSeekTargetIdx).toBe(60);
  });

  it('reports the pending advance target time while playback warmup is in flight', async () => {
    const { player } = await makePlayerHarness();

    player.currentFrame = { timestamp: 0, close: vi.fn() };
    player.currentFrameTimestampUs = 0;

    player.advanceToTime(2);

    expect(player.getPendingSeekTime()).toBe(2);
  });

  it('caps paused precise seek feeding instead of queueing the whole GOP at once', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.seek(2);

    expect(decoder.reset).toHaveBeenCalledTimes(1);
    expect(decoder.decode).toHaveBeenCalledTimes(24);
    expect(player.sampleIndex).toBe(60);
    expect(player.feedIndex).toBe(24);
    expect(player.pendingSeekFeedEndIndex).toBe(60);
  });

  it('reuses the paused seek pipeline for nearby forward scrubs', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.sampleIndex = 60;
    player.feedIndex = 61;
    player.currentFrame = { timestamp: 2_000_000, close: vi.fn() };
    player.currentFrameTimestampUs = 2_000_000;
    player.trackedDecodeQueueSize = 0;
    decoder.decodeQueueSize = 0;

    player.seek(2.2);

    expect(decoder.reset).not.toHaveBeenCalled();
    expect(decoder.decode).toHaveBeenCalledTimes(6);
    expect(player.sampleIndex).toBe(66);
    expect(player.feedIndex).toBe(67);
    expect(player.pendingSeekFeedEndIndex).toBeNull();
  });

  it('extends an in-flight paused seek forward without resetting the decoder', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.sampleIndex = 90;
    player.feedIndex = 84;
    player.currentFrame = { timestamp: 2_000_000, close: vi.fn() };
    player.currentFrameTimestampUs = 2_000_000;
    player.seekTargetUs = 3_000_000;
    player.pendingSeekKind = 'seek';
    player.pendingSeekFeedEndIndex = 90;
    player.trackedDecodeQueueSize = 0;
    decoder.decodeQueueSize = 0;

    player.seek(3.2);

    expect(decoder.reset).not.toHaveBeenCalled();
    expect(decoder.decode).toHaveBeenCalledTimes(13);
    expect(player.sampleIndex).toBe(96);
    expect(player.feedIndex).toBe(97);
    expect(player.pendingSeekFeedEndIndex).toBeNull();
  });

  it('keeps buffered future frames hot when pausing playback', async () => {
    const { player, decoder } = await makePlayerHarness();
    const futureFrameA = { timestamp: 2_033_333, close: vi.fn() };
    const futureFrameB = { timestamp: 2_066_667, close: vi.fn() };

    player._isPlaying = true;
    player.sampleIndex = 60;
    player.feedIndex = 63;
    player.currentFrame = { timestamp: 2_000_000, close: vi.fn() };
    player.currentFrameTimestampUs = 2_000_000;
    player.frameBuffer = [futureFrameA, futureFrameB];
    player.decoder.state = 'configured';

    player.pause();

    expect(player.frameBuffer).toEqual([futureFrameA, futureFrameB]);
    expect(futureFrameA.close).not.toHaveBeenCalled();
    expect(futureFrameB.close).not.toHaveBeenCalled();
    expect(decoder.decode).not.toHaveBeenCalled();
    expect(player.hasBufferedFutureFrame()).toBe(true);
  });

  it('pre-rolls a couple of future frames when pausing without a hot future buffer', async () => {
    const { player, decoder } = await makePlayerHarness();

    player._isPlaying = true;
    player.sampleIndex = 60;
    player.feedIndex = 61;
    player.currentFrame = { timestamp: 2_000_000, close: vi.fn() };
    player.currentFrameTimestampUs = 2_000_000;
    player.decoder.state = 'configured';

    player.pause();

    expect(decoder.decode).toHaveBeenCalledTimes(2);
    expect(player.feedIndex).toBe(63);
    expect(player.hasBufferedFutureFrame()).toBe(false);
  });

  it('reuses a hot paused frame without resetting the decoder on resume', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.sampleIndex = 60;
    player.feedIndex = 61;
    player.currentFrame = { timestamp: 2_000_000, close: vi.fn() };
    player.currentFrameTimestampUs = 2_000_000;
    player.trackedDecodeQueueSize = 0;
    decoder.decodeQueueSize = 0;

    player.advanceToTime(2);

    expect(decoder.reset).not.toHaveBeenCalled();
    expect(decoder.decode).toHaveBeenCalledTimes(5);
    expect(player.feedIndex).toBe(66);
  });

  it('reuses a hot paused frame when only tracked seek backlog is stale', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.sampleIndex = 60;
    player.feedIndex = 61;
    player.currentFrame = { timestamp: 2_000_000, close: vi.fn() };
    player.currentFrameTimestampUs = 2_000_000;
    player.trackedDecodeQueueSize = 145;
    decoder.decodeQueueSize = 5;

    player.advanceToTime(2);

    expect(decoder.reset).not.toHaveBeenCalled();
    expect(decoder.decode).not.toHaveBeenCalled();
    expect(player.feedIndex).toBe(61);
    expect(player.pendingAdvanceSeekTargetIdx).toBeNull();
    expect(player.trackedDecodeQueueSize).toBe(5);
  });

  it('still restarts playback when the actual decoder queue is heavily backlogged', async () => {
    const { player, decoder } = await makePlayerHarness();

    player.sampleIndex = 60;
    player.feedIndex = 61;
    player.currentFrame = { timestamp: 2_000_000, close: vi.fn() };
    player.currentFrameTimestampUs = 2_000_000;
    player.trackedDecodeQueueSize = 145;
    decoder.decodeQueueSize = 48;

    player.advanceToTime(2);

    expect(decoder.reset).toHaveBeenCalledTimes(1);
    expect(decoder.decode).toHaveBeenCalledTimes(24);
    expect(player.feedIndex).toBe(24);
    expect(player.pendingAdvanceSeekTargetIdx).toBe(60);
    expect(player.trackedDecodeQueueSize).toBe(24);
  });
});
