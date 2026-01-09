// TimelineTrack component - Individual track row

import React, { memo, useMemo, useRef, useEffect, useState } from 'react';
import type { TimelineTrackProps } from './types';
import type { AnimatableProperty, BezierHandle, Keyframe } from '../../types';
import { CurveEditor } from './CurveEditor';

// Render keyframe tracks for timeline area (right column) - flat list without folder structure
function TrackPropertyTracks({
  trackId,
  selectedClip,
  clipKeyframes,
  renderKeyframeDiamonds,
  expandedCurveProperties,
  selectedKeyframeIds,
  scrollX,
  onSelectKeyframe,
  onMoveKeyframe,
  onUpdateBezierHandle,
  timeToPixel,
  pixelToTime,
}: {
  trackId: string;
  selectedClip: { id: string; startTime: number; duration: number; effects?: Array<{ id: string; name: string; params: Record<string, unknown> }> } | null;
  clipKeyframes: Map<string, Array<{ id: string; clipId: string; time: number; property: AnimatableProperty; value: number; easing: string }>>;
  renderKeyframeDiamonds: (trackId: string, property: AnimatableProperty) => React.ReactNode;
  expandedCurveProperties: Map<string, Set<AnimatableProperty>>;
  selectedKeyframeIds: Set<string>;
  scrollX: number;
  onSelectKeyframe: (keyframeId: string, addToSelection: boolean) => void;
  onMoveKeyframe: (keyframeId: string, newTime: number) => void;
  onUpdateBezierHandle: (keyframeId: string, handle: 'in' | 'out', position: BezierHandle) => void;
  timeToPixel: (time: number) => number;
  pixelToTime: (pixel: number) => number;
}) {
  const clipId = selectedClip?.id;

  // Get keyframes for this clip - use clipKeyframes map to trigger re-render when keyframes change
  const keyframeProperties = useMemo(() => {
    if (!clipId) return new Set<string>();
    const props = new Set<string>();
    const keyframes = clipKeyframes.get(clipId) || [];
    keyframes.forEach((kf) => props.add(kf.property));
    return props;
  }, [clipId, clipKeyframes]);

  // Track container ref for getting width
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(1000);

  // Measure container width
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth);
      }
    };
    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // If no clip is selected in this track or no keyframes, show nothing
  if (!selectedClip || keyframeProperties.size === 0) {
    return <div className="track-property-tracks" ref={containerRef} />;
  }

  // Convert Set to sorted array for consistent ordering (matching the labels)
  const sortedProperties = Array.from(keyframeProperties).sort((a, b) => {
    const order = ['opacity', 'position.x', 'position.y', 'position.z', 'scale.x', 'scale.y', 'rotation.x', 'rotation.y', 'rotation.z'];
    const aIdx = order.indexOf(a);
    const bIdx = order.indexOf(b);
    if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
    if (aIdx !== -1) return -1;
    if (bIdx !== -1) return 1;
    return a.localeCompare(b);
  });

  // Get expanded curve properties for this track
  const trackCurveProps = expandedCurveProperties.get(trackId);

  // Get all keyframes for this clip
  const allKeyframes = clipKeyframes.get(selectedClip.id) || [];

  return (
    <div className="track-property-tracks" ref={containerRef}>
      {sortedProperties.map((prop) => {
        const isCurveExpanded = trackCurveProps?.has(prop as AnimatableProperty) ?? false;
        const propKeyframes = allKeyframes.filter(kf => kf.property === prop);

        return (
          <div key={prop} className={`keyframe-track-row flat ${isCurveExpanded ? 'curve-expanded' : ''}`}>
            <div className="keyframe-track">
              <div className="keyframe-track-line" />
              {renderKeyframeDiamonds(trackId, prop as AnimatableProperty)}
            </div>
            {isCurveExpanded && (
              <CurveEditor
                trackId={trackId}
                clipId={selectedClip.id}
                property={prop as AnimatableProperty}
                keyframes={propKeyframes as Keyframe[]}
                clipStartTime={selectedClip.startTime}
                clipDuration={selectedClip.duration}
                width={containerWidth}
                scrollX={scrollX}
                selectedKeyframeIds={selectedKeyframeIds}
                onSelectKeyframe={onSelectKeyframe}
                onMoveKeyframe={(id, newTime, _newValue) => {
                  // CurveEditor passes both time and value, but we need separate calls
                  onMoveKeyframe(id, newTime);
                  // Value changes happen through bezier handle updates
                }}
                onUpdateBezierHandle={onUpdateBezierHandle}
                timeToPixel={timeToPixel}
                pixelToTime={pixelToTime}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

function TimelineTrackComponent({
  track,
  clips,
  isDimmed,
  isExpanded,
  dynamicHeight,
  isDragTarget,
  isExternalDragTarget,
  selectedClipIds,
  selectedKeyframeIds,
  clipDrag,
  externalDrag,
  onDrop,
  onDragOver,
  onDragEnter,
  onDragLeave,
  renderClip,
  clipKeyframes,
  renderKeyframeDiamonds,
  timeToPixel,
  pixelToTime,
  scrollX,
  expandedCurveProperties,
  onSelectKeyframe,
  onMoveKeyframe,
  onUpdateBezierHandle,
}: TimelineTrackProps) {
  // Get clips belonging to this track
  const trackClips = clips.filter((c) => c.trackId === track.id);
  const selectedTrackClip = trackClips.find((c) => selectedClipIds.has(c.id));

  return (
    <div
      className={`track-lane ${track.type} ${isDimmed ? 'dimmed' : ''} ${
        isExpanded ? 'expanded' : ''
      } ${isDragTarget ? 'drag-target' : ''} ${
        isExternalDragTarget ? 'external-drag-target' : ''
      }`}
      style={{ height: dynamicHeight }}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
    >
      {/* Clip row - the normal clip area */}
      <div className="track-clip-row" style={{ height: track.height }}>
        {/* Render clips belonging to this track */}
        {trackClips.map((clip) => renderClip(clip, track.id))}
        {/* Render clip being dragged TO this track */}
        {clipDrag &&
          clipDrag.currentTrackId === track.id &&
          clipDrag.originalTrackId !== track.id &&
          clips
            .filter((c) => c.id === clipDrag.clipId)
            .map((clip) => renderClip(clip, track.id))}
        {/* External file drag preview - video clip */}
        {externalDrag && externalDrag.trackId === track.id && (
          <div
            className="timeline-clip-preview"
            style={{
              left: timeToPixel(externalDrag.startTime),
              width: timeToPixel(externalDrag.duration ?? 5),
            }}
          >
            <div className="clip-content">
              <span className="clip-name">Drop to add clip</span>
            </div>
          </div>
        )}
        {/* External file drag preview - linked audio clip */}
        {externalDrag &&
          externalDrag.isVideo &&
          externalDrag.audioTrackId === track.id && (
            <div
              className="timeline-clip-preview audio"
              style={{
                left: timeToPixel(externalDrag.startTime),
                width: timeToPixel(externalDrag.duration ?? 5),
              }}
            >
              <div className="clip-content">
                <span className="clip-name">Audio</span>
              </div>
            </div>
          )}
      </div>
      {/* Property rows - only shown when track is expanded */}
      {track.type === 'video' && isExpanded && (
        <TrackPropertyTracks
          trackId={track.id}
          selectedClip={selectedTrackClip || null}
          clipKeyframes={clipKeyframes}
          renderKeyframeDiamonds={renderKeyframeDiamonds}
          expandedCurveProperties={expandedCurveProperties}
          selectedKeyframeIds={selectedKeyframeIds}
          scrollX={scrollX}
          onSelectKeyframe={onSelectKeyframe}
          onMoveKeyframe={onMoveKeyframe}
          onUpdateBezierHandle={onUpdateBezierHandle}
          timeToPixel={timeToPixel}
          pixelToTime={pixelToTime}
        />
      )}
    </div>
  );
}

export const TimelineTrack = memo(TimelineTrackComponent);
