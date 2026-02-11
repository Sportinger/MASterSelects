// Timeline overlay elements (markers, work area, cache indicators, etc.)

import React from 'react';
import type { ClipDragState } from '../types';

interface TimelineOverlaysProps {
  // Time conversion
  timeToPixel: (time: number) => number;
  formatTime: (seconds: number) => string;

  // In/Out points
  inPoint: number | null;
  outPoint: number | null;
  duration: number;
  markerDrag: { type: 'in' | 'out' } | null;
  onMarkerMouseDown: (e: React.MouseEvent, type: 'in' | 'out') => void;

  // Clip drag
  clipDrag: ClipDragState | null;

  // RAM preview
  isRamPreviewing: boolean;
  ramPreviewProgress: number | null;
  playheadPosition: number;

  // Export
  isExporting: boolean;
  exportProgress: number | null;
  exportRange: { start: number; end: number } | null;

  // Cache
  getCachedRanges: () => { start: number; end: number }[];
  getProxyCachedRanges: () => { start: number; end: number }[];
}

export function TimelineOverlays({
  timeToPixel,
  formatTime,
  inPoint,
  outPoint,
  duration,
  markerDrag,
  onMarkerMouseDown,
  clipDrag,
  isRamPreviewing,
  ramPreviewProgress,
  playheadPosition,
  isExporting,
  exportProgress,
  exportRange,
  getCachedRanges,
  getProxyCachedRanges,
}: TimelineOverlaysProps) {
  return (
    <>
      {/* Snap line */}
      {clipDrag?.isSnapping && clipDrag.snappedTime !== null && (
        <div className="snap-line" style={{ left: timeToPixel(clipDrag.snappedTime) }} />
      )}

      {/* Work area overlays */}
      {(inPoint !== null || outPoint !== null) && (
        <>
          {inPoint !== null && inPoint > 0 && (
            <div
              className="work-area-overlay before"
              style={{
                left: 0,
                width: timeToPixel(inPoint),
              }}
            />
          )}
          {outPoint !== null && (
            <div
              className="work-area-overlay after"
              style={{
                left: timeToPixel(outPoint),
                width: timeToPixel(duration - outPoint),
              }}
            />
          )}
        </>
      )}

      {/* RAM preview progress */}
      {isRamPreviewing && ramPreviewProgress !== null && (
        <div
          className="ram-preview-progress-text"
          style={{
            left: timeToPixel(playheadPosition) + 10,
          }}
        >
          {Math.round(ramPreviewProgress)}%
        </div>
      )}

      {/* Export Progress Overlay */}
      {isExporting && exportRange && (
        <>
          {/* Progress bar - grows based on percentage (0-100%) */}
          <div
            className="timeline-export-overlay"
            style={{
              left: timeToPixel(exportRange.start),
              width: timeToPixel(
                (exportRange.end - exportRange.start) * ((exportProgress ?? 0) / 100)
              ),
            }}
          />
          {/* Percentage display - at end of progress bar */}
          <div
            className="timeline-export-text"
            style={{
              left:
                timeToPixel(
                  exportRange.start +
                    (exportRange.end - exportRange.start) * ((exportProgress ?? 0) / 100)
                ) - 10,
              transform: 'translateX(-100%)',
            }}
          >
            {Math.round(exportProgress ?? 0)}%
          </div>
        </>
      )}

      {/* Playback cache indicators (blue) */}
      {getCachedRanges().map((range, i) => (
        <div
          key={i}
          className="playback-cache-indicator"
          style={{
            left: timeToPixel(range.start),
            width: Math.max(2, timeToPixel(range.end - range.start)),
          }}
          title={`Cached: ${formatTime(range.start)} - ${formatTime(range.end)}`}
        />
      ))}

      {/* Proxy frame cache indicator (yellow) */}
      {getProxyCachedRanges().map((range, i) => (
        <div
          key={`proxy-${i}`}
          className="proxy-cache-indicator"
          style={{
            left: timeToPixel(range.start),
            width: Math.max(2, timeToPixel(range.end - range.start)),
          }}
          title={`Proxy cached: ${formatTime(range.start)} - ${formatTime(range.end)}`}
        />
      ))}

      {/* In marker */}
      {inPoint !== null && (
        <div
          className={`in-out-marker in-marker ${markerDrag?.type === 'in' ? 'dragging' : ''}`}
          style={{ left: timeToPixel(inPoint) }}
          title={`In: ${formatTime(inPoint)} (drag to move)`}
        >
          <div
            className="marker-flag"
            onMouseDown={(e) => onMarkerMouseDown(e, 'in')}
          >
            I
          </div>
          <div className="marker-line" />
        </div>
      )}

      {/* Out marker */}
      {outPoint !== null && (
        <div
          className={`in-out-marker out-marker ${markerDrag?.type === 'out' ? 'dragging' : ''}`}
          style={{ left: timeToPixel(outPoint) }}
          title={`Out: ${formatTime(outPoint)} (drag to move)`}
        >
          <div
            className="marker-flag"
            onMouseDown={(e) => onMarkerMouseDown(e, 'out')}
          >
            O
          </div>
          <div className="marker-line" />
        </div>
      )}
    </>
  );
}
