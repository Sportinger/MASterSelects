import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createMockClip, createMockTrack } from '../helpers/mockData';

const hoisted = vi.hoisted(() => ({
  timelineState: null as any,
  mediaState: {
    files: [],
    compositions: [],
    activeCompositionId: null,
    proxyEnabled: false,
  },
}));

vi.mock('../../src/stores/timeline', () => ({
  useTimelineStore: {
    getState: () => hoisted.timelineState,
  },
}));

vi.mock('../../src/stores/mediaStore', () => ({
  useMediaStore: {
    getState: () => hoisted.mediaState,
  },
}));

import { createFrameContext } from '../../src/services/layerBuilder/FrameContext';
import { playheadState } from '../../src/services/layerBuilder/PlayheadState';

function createTimelineState(overrides: Record<string, unknown> = {}) {
  return {
    clips: [],
    tracks: [createMockTrack({ id: 'video-1', type: 'video' })],
    isPlaying: false,
    isDraggingPlayhead: false,
    playheadPosition: 0,
    playbackSpeed: 1,
    getInterpolatedTransform: vi.fn(),
    getInterpolatedEffects: vi.fn(() => []),
    getInterpolatedSpeed: vi.fn(() => 1),
    getSourceTimeForClip: vi.fn((_clipId: string, localTime: number) => localTime),
    hasKeyframes: vi.fn(() => false),
    ...overrides,
  };
}

describe('FrameContext clipsAtTime', () => {
  beforeEach(() => {
    playheadState.isUsingInternalPosition = false;
    playheadState.position = 0;
    hoisted.timelineState = createTimelineState();
  });

  it('prefers the incoming clip at an exact cut boundary', () => {
    const clipA = createMockClip({
      id: 'clip-a',
      trackId: 'video-1',
      startTime: 0,
      duration: 10,
      inPoint: 0,
      outPoint: 10,
    });
    const clipB = createMockClip({
      id: 'clip-b',
      trackId: 'video-1',
      startTime: 10,
      duration: 10,
      inPoint: 10,
      outPoint: 20,
    });

    hoisted.timelineState = createTimelineState({
      clips: [clipA, clipB],
      playheadPosition: 10,
    });

    const ctx = createFrameContext();

    expect(ctx.clipsAtTime.map(clip => clip.id)).toEqual(['clip-b']);
  });

  it('keeps both clips active during a real overlap', () => {
    const outgoingClip = createMockClip({
      id: 'clip-out',
      trackId: 'video-1',
      startTime: 0,
      duration: 10.5,
      inPoint: 0,
      outPoint: 10.5,
    });
    const incomingClip = createMockClip({
      id: 'clip-in',
      trackId: 'video-1',
      startTime: 10,
      duration: 10,
      inPoint: 10,
      outPoint: 20,
    });

    hoisted.timelineState = createTimelineState({
      clips: [outgoingClip, incomingClip],
      playheadPosition: 10.25,
    });

    const ctx = createFrameContext();

    expect(ctx.clipsAtTime.map(clip => clip.id)).toEqual(['clip-out', 'clip-in']);
  });

  it('avoids an empty frame when the incoming clip starts a tiny fraction late', () => {
    const clipA = createMockClip({
      id: 'clip-a',
      trackId: 'video-1',
      startTime: 0,
      duration: 10,
      inPoint: 0,
      outPoint: 10,
    });
    const clipB = createMockClip({
      id: 'clip-b',
      trackId: 'video-1',
      startTime: 10.0000005,
      duration: 10,
      inPoint: 10,
      outPoint: 20,
    });

    hoisted.timelineState = createTimelineState({
      clips: [clipA, clipB],
      playheadPosition: 10,
    });

    const ctx = createFrameContext();

    expect(ctx.clipsAtTime.map(clip => clip.id)).toEqual(['clip-b']);
  });
});
