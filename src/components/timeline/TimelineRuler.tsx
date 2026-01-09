// TimelineRuler component - Time ruler at the top of the timeline

import React, { memo } from 'react';
import type { TimelineRulerProps } from './types';

function TimelineRulerComponent({
  duration,
  zoom,
  scrollX,
  onRulerMouseDown,
  formatTime,
}: TimelineRulerProps) {
  // Time to pixel conversion
  const timeToPixel = (time: number) => time * zoom;

  const width = timeToPixel(duration);
  const markers: React.ReactElement[] = [];

  // Calculate marker interval based on zoom
  let interval = 1; // 1 second
  if (zoom < 20) interval = 5;
  if (zoom < 10) interval = 10;
  if (zoom > 100) interval = 0.5;

  for (let t = 0; t <= duration; t += interval) {
    const x = timeToPixel(t);
    const isMainMarker = t % (interval * 2) === 0 || interval >= 5;

    markers.push(
      <div
        key={t}
        className={`time-marker ${isMainMarker ? 'main' : 'sub'}`}
        style={{ left: x }}
      >
        {isMainMarker && <span className="time-label">{formatTime(t)}</span>}
      </div>
    );
  }

  return (
    <div
      className="time-ruler"
      style={{ width, transform: `translateX(-${scrollX}px)` }}
      onMouseDown={onRulerMouseDown}
    >
      {markers}
    </div>
  );
}

export const TimelineRuler = memo(TimelineRulerComponent);
