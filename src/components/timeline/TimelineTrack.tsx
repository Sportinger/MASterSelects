// TimelineTrack component - Individual track row

import React, { memo } from 'react';
import type { TimelineTrackProps } from './types';
import type { AnimatableProperty } from '../../types';

// Render keyframe tracks for timeline area (right column) - only show properties with keyframes
function TrackPropertyTracks({
  trackId,
  selectedClip,
  isTrackPropertyGroupExpanded,
  hasPropertyKeyframes,
  renderKeyframeDiamonds,
}: {
  trackId: string;
  selectedClip: { id: string; effects?: Array<{ id: string; name: string; params: Record<string, unknown> }> } | null;
  isTrackPropertyGroupExpanded: (trackId: string, group: string) => boolean;
  hasPropertyKeyframes: (clipId: string, property?: AnimatableProperty) => boolean;
  renderKeyframeDiamonds: (trackId: string, property: AnimatableProperty) => React.ReactNode;
}) {
  // If no clip is selected in this track, show nothing
  if (!selectedClip) {
    return <div className="track-property-tracks" />;
  }

  const clipId = selectedClip.id;

  // Check which property groups have keyframes
  const hasOpacityKeyframes = hasPropertyKeyframes(clipId, 'opacity');
  const hasPositionKeyframes =
    hasPropertyKeyframes(clipId, 'position.x') ||
    hasPropertyKeyframes(clipId, 'position.y') ||
    hasPropertyKeyframes(clipId, 'position.z');
  const hasScaleKeyframes =
    hasPropertyKeyframes(clipId, 'scale.x') ||
    hasPropertyKeyframes(clipId, 'scale.y');
  const hasRotationKeyframes =
    hasPropertyKeyframes(clipId, 'rotation.x') ||
    hasPropertyKeyframes(clipId, 'rotation.y') ||
    hasPropertyKeyframes(clipId, 'rotation.z');

  // Check for effect keyframes - which effects have at least one keyframed parameter
  const effectsWithKeyframes =
    selectedClip.effects?.filter((effect) => {
      const numericParams = Object.keys(effect.params).filter(
        (k) => typeof effect.params[k] === 'number'
      );
      return numericParams.some((paramName) =>
        hasPropertyKeyframes(
          clipId,
          `effect.${effect.id}.${paramName}` as AnimatableProperty
        )
      );
    }) || [];

  // If no keyframes at all, show nothing
  if (
    !hasOpacityKeyframes &&
    !hasPositionKeyframes &&
    !hasScaleKeyframes &&
    !hasRotationKeyframes &&
    effectsWithKeyframes.length === 0
  ) {
    return <div className="track-property-tracks" />;
  }

  return (
    <div className="track-property-tracks">
      {/* Opacity - only show if has keyframes */}
      {hasOpacityKeyframes && (
        <div className="keyframe-track-group">
          <div className="keyframe-track-row group-header">
            <div className="keyframe-track">
              <div className="keyframe-track-line" />
              {renderKeyframeDiamonds(trackId, 'opacity')}
            </div>
          </div>
        </div>
      )}

      {/* Position group tracks - only show if has keyframes */}
      {hasPositionKeyframes && (
        <div className="keyframe-track-group">
          <div className="keyframe-track-row group-header" />
          {isTrackPropertyGroupExpanded(trackId, 'position') && (
            <>
              {hasPropertyKeyframes(clipId, 'position.x') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'position.x')}
                  </div>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'position.y') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'position.y')}
                  </div>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'position.z') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'position.z')}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Scale group tracks - only show if has keyframes */}
      {hasScaleKeyframes && (
        <div className="keyframe-track-group">
          <div className="keyframe-track-row group-header" />
          {isTrackPropertyGroupExpanded(trackId, 'scale') && (
            <>
              {hasPropertyKeyframes(clipId, 'scale.x') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'scale.x')}
                  </div>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'scale.y') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'scale.y')}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Rotation group tracks - only show if has keyframes */}
      {hasRotationKeyframes && (
        <div className="keyframe-track-group">
          <div className="keyframe-track-row group-header" />
          {isTrackPropertyGroupExpanded(trackId, 'rotation') && (
            <>
              {hasPropertyKeyframes(clipId, 'rotation.x') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'rotation.x')}
                  </div>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'rotation.y') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'rotation.y')}
                  </div>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'rotation.z') && (
                <div className="keyframe-track-row">
                  <div className="keyframe-track">
                    <div className="keyframe-track-line" />
                    {renderKeyframeDiamonds(trackId, 'rotation.z')}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Effects group tracks - only show effects that have keyframes */}
      {effectsWithKeyframes.length > 0 && (
        <div className="keyframe-track-group">
          <div className="keyframe-track-row group-header" />
          {isTrackPropertyGroupExpanded(trackId, 'effects') &&
            effectsWithKeyframes.map((effect) => {
              // Only show params with keyframes
              const paramsWithKeyframes = Object.keys(effect.params)
                .filter((k) => typeof effect.params[k] === 'number')
                .filter((paramName) =>
                  hasPropertyKeyframes(
                    clipId,
                    `effect.${effect.id}.${paramName}` as AnimatableProperty
                  )
                );

              return (
                <div key={effect.id} className="keyframe-track-group nested">
                  <div className="keyframe-track-row group-header" />
                  {isTrackPropertyGroupExpanded(trackId, `effect.${effect.id}`) && (
                    <>
                      {paramsWithKeyframes.map((paramName) => (
                        <div key={paramName} className="keyframe-track-row">
                          <div className="keyframe-track">
                            <div className="keyframe-track-line" />
                            {renderKeyframeDiamonds(
                              trackId,
                              `effect.${effect.id}.${paramName}` as AnimatableProperty
                            )}
                          </div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              );
            })}
        </div>
      )}
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
  selectedClipId,
  clipDrag,
  externalDrag,
  onDrop,
  onDragOver,
  onDragEnter,
  onDragLeave,
  renderClip,
  isTrackPropertyGroupExpanded,
  hasPropertyKeyframes,
  renderKeyframeDiamonds,
  timeToPixel,
}: TimelineTrackProps) {
  // Get clips belonging to this track
  const trackClips = clips.filter((c) => c.trackId === track.id);
  const selectedTrackClip = trackClips.find((c) => c.id === selectedClipId);

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
          isTrackPropertyGroupExpanded={isTrackPropertyGroupExpanded}
          hasPropertyKeyframes={hasPropertyKeyframes}
          renderKeyframeDiamonds={renderKeyframeDiamonds}
        />
      )}
    </div>
  );
}

export const TimelineTrack = memo(TimelineTrackComponent);
