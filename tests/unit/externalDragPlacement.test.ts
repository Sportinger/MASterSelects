import { describe, expect, it } from 'vitest';

import {
  canPlaceOnTrack,
  findClosestNonOverlappingStartTime,
  findFirstTrackWithoutOverlap,
} from '../../src/components/timeline/utils/externalDragPlacement';
import { createMockClip, createMockTrack } from '../helpers/mockData';

describe('externalDragPlacement', () => {
  it('keeps the desired start time when the track slot is free', () => {
    const clips = [
      createMockClip({ trackId: 'video-1', startTime: 0, duration: 4 }),
      createMockClip({ trackId: 'video-1', startTime: 10, duration: 4 }),
    ];

    expect(canPlaceOnTrack('video-1', 5, 3, clips)).toBe(true);
    expect(findClosestNonOverlappingStartTime('video-1', 5, 3, clips)).toBe(5);
  });

  it('uses the real preview duration when resolving collisions', () => {
    const clips = [
      createMockClip({ trackId: 'video-1', startTime: 10, duration: 10 }),
    ];

    expect(findClosestNonOverlappingStartTime('video-1', 14, 2, clips)).toBe(8);
    expect(findClosestNonOverlappingStartTime('video-1', 14, 5, clips)).toBe(20);
  });

  it('finds the first compatible linked track without overlap', () => {
    const tracks = [
      createMockTrack({ id: 'video-1', type: 'video' }),
      createMockTrack({ id: 'audio-1', type: 'audio' }),
      createMockTrack({ id: 'audio-2', type: 'audio' }),
    ];
    const clips = [
      createMockClip({ trackId: 'audio-1', startTime: 3, duration: 5 }),
    ];

    expect(findFirstTrackWithoutOverlap('audio', 4, 3, tracks, clips)).toBe('audio-2');
    expect(findFirstTrackWithoutOverlap('video', 4, 3, tracks, clips)).toBe('video-1');
  });
});
