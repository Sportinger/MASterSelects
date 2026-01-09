// TimelineKeyframes component - Keyframe diamonds/handles

import { memo } from 'react';
import type { TimelineKeyframesProps } from './types';

function TimelineKeyframesComponent({
  trackId,
  property,
  clips,
  selectedKeyframeIds,
  getClipKeyframes,
  onSelectKeyframe,
  timeToPixel,
}: TimelineKeyframesProps) {
  // Get all clips on this track
  const trackClips = clips.filter((c) => c.trackId === trackId);

  // Collect all keyframes from all clips with their absolute positions
  const allKeyframes: Array<{
    kf: ReturnType<typeof getClipKeyframes>[0];
    clip: typeof clips[0];
    absTime: number;
  }> = [];

  trackClips.forEach((clip) => {
    const clipKeyframes = getClipKeyframes(clip.id).filter(
      (k) => k.property === property
    );
    clipKeyframes.forEach((kf) => {
      allKeyframes.push({
        kf,
        clip,
        absTime: clip.startTime + kf.time,
      });
    });
  });

  return (
    <>
      {allKeyframes.map(({ kf, absTime }) => {
        const xPos = timeToPixel(absTime);
        const isSelected = selectedKeyframeIds.has(kf.id);

        return (
          <div
            key={kf.id}
            className={`keyframe-diamond ${isSelected ? 'selected' : ''}`}
            style={{ left: `${xPos}px` }}
            onClick={(e) => {
              e.stopPropagation();
              onSelectKeyframe(kf.id, e.shiftKey);
            }}
            title={`${property}: ${kf.value.toFixed(3)} @ ${absTime.toFixed(2)}s (${kf.easing})`}
          />
        );
      })}
    </>
  );
}

export const TimelineKeyframes = memo(TimelineKeyframesComponent);
