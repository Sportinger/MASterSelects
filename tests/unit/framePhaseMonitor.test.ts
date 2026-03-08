import { beforeEach, describe, expect, it, vi } from 'vitest';
import { framePhaseMonitor } from '../../src/services/framePhaseMonitor';

describe('framePhaseMonitor', () => {
  beforeEach(() => {
    framePhaseMonitor.reset();
    vi.restoreAllMocks();
  });

  it('summarizes recent phase samples', () => {
    vi.spyOn(performance, 'now')
      .mockReturnValueOnce(1000)
      .mockReturnValueOnce(1100)
      .mockReturnValueOnce(1200)
      .mockReturnValue(1300);

    framePhaseMonitor.record({
      mode: 'live',
      statsMs: 1,
      buildMs: 4,
      renderMs: 0.5,
      syncVideoMs: 6,
      syncAudioMs: 1.5,
      cacheMs: 0.2,
      totalMs: 13.2,
    });
    framePhaseMonitor.record({
      mode: 'cached',
      statsMs: 0.2,
      buildMs: 0,
      renderMs: 0,
      syncVideoMs: 0,
      syncAudioMs: 0.8,
      cacheMs: 0,
      totalMs: 1,
    });
    framePhaseMonitor.record({
      mode: 'live',
      statsMs: 0.8,
      buildMs: 5,
      renderMs: 0.4,
      syncVideoMs: 9,
      syncAudioMs: 1.2,
      cacheMs: 0.3,
      totalMs: 16.7,
    });

    const summary = framePhaseMonitor.summary(500);

    expect(summary.samples).toBe(3);
    expect(summary.liveSamples).toBe(2);
    expect(summary.cachedSamples).toBe(1);
    expect(summary.avgSyncVideoMs).toBeCloseTo(5);
    expect(summary.maxSyncVideoMs).toBe(9);
    expect(summary.maxTotalMs).toBe(16.7);
    expect(summary.p95TotalMs).toBeGreaterThan(13);
  });
});
