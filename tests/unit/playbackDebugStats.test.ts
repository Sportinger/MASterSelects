import { describe, expect, it } from 'vitest';
import {
  buildPlaybackDebugStats,
  summarizeFrameCadence,
} from '../../src/services/playbackDebugStats';
import type { PipelineEvent } from '../../src/services/wcPipelineMonitor';
import type { VFPipelineEvent } from '../../src/services/vfPipelineMonitor';

describe('playback debug stats', () => {
  it('summarizes frame cadence from recent frame timestamps', () => {
    const cadence = summarizeFrameCadence([0, 33, 66, 99, 132]);

    expect(cadence.frameEvents).toBe(5);
    expect(cadence.cadenceFps).toBeCloseTo(30.3, 1);
    expect(cadence.avgFrameGapMs).toBe(33);
    expect(cadence.p95FrameGapMs).toBe(33);
    expect(cadence.maxFrameGapMs).toBe(33);
  });

  it('builds a unified WebCodecs playback snapshot with health data', () => {
    const wcTimeline: PipelineEvent[] = [
      { type: 'decode_feed', t: 0 },
      { type: 'decode_output', t: 20, detail: { queueSize: 1 } },
      { type: 'decode_feed', t: 33 },
      { type: 'decode_output', t: 70, detail: { queueSize: 2 } },
      { type: 'decoder_reset', t: 74, detail: { reason: 'advance_seek' } },
      { type: 'queue_pressure', t: 75, detail: { queueSize: 4 } },
      { type: 'advance_seek', t: 80 },
      { type: 'pending_seek_start', t: 81, detail: { kind: 'advance', targetUs: 80000 } },
      { type: 'seek_start', t: 85 },
      { type: 'seek_end', t: 125, detail: { durationMs: 40 } },
      { type: 'pending_seek_end', t: 132, detail: { kind: 'advance', durationMs: 51, reason: 'resolved' } },
      { type: 'drift_correct', t: 130 },
      { type: 'collector_hold', t: 160, detail: { reason: 'same_provider_pending' } },
      { type: 'collector_drop', t: 170, detail: { reason: 'pending_unstable' } },
      { type: 'stall', t: 200, detail: { gapMs: 130 } },
    ];

    const stats = buildPlaybackDebugStats({
      decoder: 'WebCodecs',
      now: 250,
      windowMs: 500,
      wcTimeline,
      healthVideos: [
        {
          clipId: 'clip-1',
          src: 'demo.mp4',
          currentTime: 1.25,
          readyState: 1,
          seeking: true,
          paused: false,
          played: 1,
          warmingUp: true,
          gpuReady: false,
        },
      ],
      healthAnomalies: [
        {
          type: 'GPU_SURFACE_COLD',
          timestamp: 220,
          recovered: false,
        },
      ],
    });

    expect(stats.pipeline).toBe('webcodecs');
    expect(stats.frameEvents).toBe(2);
    expect(stats.stalls).toBe(1);
    expect(stats.seeks).toBe(2);
    expect(stats.advanceSeeks).toBe(1);
    expect(stats.queuePressureEvents).toBe(1);
    expect(stats.avgDecodeLatencyMs).toBe(28.5);
    expect(stats.avgSeekLatencyMs).toBe(40);
    expect(stats.maxQueueDepth).toBe(4);
    expect(stats.decoderResets).toBe(1);
    expect(stats.pendingSeekResolves).toBe(1);
    expect(stats.avgPendingSeekMs).toBe(51);
    expect(stats.collectorHolds).toBe(1);
    expect(stats.collectorDrops).toBe(1);
    expect(stats.healthAnomalies).toBe(1);
    expect(stats.seekingVideos).toBe(1);
    expect(stats.playingVideos).toBe(1);
    expect(stats.warmingUpVideos).toBe(1);
    expect(stats.coldVideos).toBe(1);
    expect(stats.lastAnomalyType).toBe('GPU_SURFACE_COLD');
    expect(stats.status).toBe('bad');
  });

  it('does not mark idle cold webcodecs videos as bad without playback activity', () => {
    const stats = buildPlaybackDebugStats({
      decoder: 'WebCodecs',
      now: 250,
      windowMs: 500,
      wcTimeline: [],
      healthVideos: [
        {
          clipId: 'clip-idle',
          src: 'idle.mp4',
          currentTime: 4,
          readyState: 4,
          seeking: false,
          paused: true,
          played: 1,
          warmingUp: false,
          gpuReady: false,
        },
      ],
    });

    expect(stats.activeVideos).toBe(1);
    expect(stats.playingVideos).toBe(0);
    expect(stats.coldVideos).toBe(1);
    expect(stats.status).toBe('ok');
  });

  it('marks VF playback unhealthy when readyState drops and audio drift show up', () => {
    const vfTimeline: VFPipelineEvent[] = [
      { type: 'vf_capture', t: 0 },
      { type: 'vf_capture', t: 42 },
      { type: 'vf_seek_precise', t: 50 },
      { type: 'vf_seek_done', t: 82 },
      { type: 'vf_drift', t: 90 },
      { type: 'vf_readystate_drop', t: 95 },
      { type: 'audio_drift', t: 100, detail: { driftMs: 72 } },
    ];

    const stats = buildPlaybackDebugStats({
      decoder: 'HTMLVideo(VF)',
      now: 120,
      windowMs: 500,
      vfTimeline,
    });

    expect(stats.pipeline).toBe('vf');
    expect(stats.frameEvents).toBe(2);
    expect(stats.seeks).toBe(1);
    expect(stats.advanceSeeks).toBe(0);
    expect(stats.driftCorrections).toBe(1);
    expect(stats.readyStateDrops).toBe(1);
    expect(stats.avgSeekLatencyMs).toBe(32);
    expect(stats.avgAudioDriftMs).toBe(72);
    expect(stats.status).toBe('bad');
  });
});
