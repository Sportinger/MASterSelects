// TimelineHeader component - Track headers (left side)

import { memo } from 'react';
import type { TimelineHeaderProps } from './types';
import type { AnimatableProperty } from '../../types';

// Render property labels for track header (left column) - only show properties with keyframes
function TrackPropertyLabels({
  trackId,
  selectedClip,
  isTrackPropertyGroupExpanded,
  toggleTrackPropertyGroupExpanded,
  hasPropertyKeyframes,
}: {
  trackId: string;
  selectedClip: { id: string; effects?: Array<{ id: string; name: string; params: Record<string, unknown> }> } | null;
  isTrackPropertyGroupExpanded: (trackId: string, group: string) => boolean;
  toggleTrackPropertyGroupExpanded: (trackId: string, group: string) => void;
  hasPropertyKeyframes: (clipId: string, property?: AnimatableProperty) => boolean;
}) {
  // If no clip is selected in this track, show nothing
  if (!selectedClip) {
    return <div className="track-property-labels" />;
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
    return <div className="track-property-labels" />;
  }

  return (
    <div className="track-property-labels">
      {/* Opacity - only show if has keyframes */}
      {hasOpacityKeyframes && (
        <div className="property-label-group">
          <div className="property-group-header" style={{ cursor: 'default' }}>
            <span
              className="property-group-arrow"
              style={{ visibility: 'hidden' }}
            >
              {'\u25B6'}
            </span>
            <span>Opacity</span>
          </div>
        </div>
      )}

      {/* Position group - only show if has keyframes */}
      {hasPositionKeyframes && (
        <div className="property-label-group">
          <div
            className="property-group-header"
            onClick={(e) => {
              e.stopPropagation();
              toggleTrackPropertyGroupExpanded(trackId, 'position');
            }}
          >
            <span
              className={`property-group-arrow ${
                isTrackPropertyGroupExpanded(trackId, 'position')
                  ? 'expanded'
                  : ''
              }`}
            >
              {'\u25B6'}
            </span>
            <span>Position</span>
          </div>
          {isTrackPropertyGroupExpanded(trackId, 'position') && (
            <>
              {hasPropertyKeyframes(clipId, 'position.x') && (
                <div className="property-label-row sub">
                  <span className="property-label">X</span>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'position.y') && (
                <div className="property-label-row sub">
                  <span className="property-label">Y</span>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'position.z') && (
                <div className="property-label-row sub">
                  <span className="property-label">Z</span>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Scale group - only show if has keyframes */}
      {hasScaleKeyframes && (
        <div className="property-label-group">
          <div
            className="property-group-header"
            onClick={(e) => {
              e.stopPropagation();
              toggleTrackPropertyGroupExpanded(trackId, 'scale');
            }}
          >
            <span
              className={`property-group-arrow ${
                isTrackPropertyGroupExpanded(trackId, 'scale') ? 'expanded' : ''
              }`}
            >
              {'\u25B6'}
            </span>
            <span>Scale</span>
          </div>
          {isTrackPropertyGroupExpanded(trackId, 'scale') && (
            <>
              {hasPropertyKeyframes(clipId, 'scale.x') && (
                <div className="property-label-row sub">
                  <span className="property-label">X</span>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'scale.y') && (
                <div className="property-label-row sub">
                  <span className="property-label">Y</span>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Rotation group - only show if has keyframes */}
      {hasRotationKeyframes && (
        <div className="property-label-group">
          <div
            className="property-group-header"
            onClick={(e) => {
              e.stopPropagation();
              toggleTrackPropertyGroupExpanded(trackId, 'rotation');
            }}
          >
            <span
              className={`property-group-arrow ${
                isTrackPropertyGroupExpanded(trackId, 'rotation')
                  ? 'expanded'
                  : ''
              }`}
            >
              {'\u25B6'}
            </span>
            <span>Rotation</span>
          </div>
          {isTrackPropertyGroupExpanded(trackId, 'rotation') && (
            <>
              {hasPropertyKeyframes(clipId, 'rotation.x') && (
                <div className="property-label-row sub">
                  <span className="property-label">X</span>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'rotation.y') && (
                <div className="property-label-row sub">
                  <span className="property-label">Y</span>
                </div>
              )}
              {hasPropertyKeyframes(clipId, 'rotation.z') && (
                <div className="property-label-row sub">
                  <span className="property-label">Z</span>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Effects group - only show effects that have keyframes */}
      {effectsWithKeyframes.length > 0 && (
        <div className="property-label-group">
          <div
            className="property-group-header"
            onClick={(e) => {
              e.stopPropagation();
              toggleTrackPropertyGroupExpanded(trackId, 'effects');
            }}
          >
            <span
              className={`property-group-arrow ${
                isTrackPropertyGroupExpanded(trackId, 'effects')
                  ? 'expanded'
                  : ''
              }`}
            >
              {'\u25B6'}
            </span>
            <span>Effects</span>
          </div>
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
                <div key={effect.id} className="property-label-group nested">
                  <div
                    className="property-group-header sub"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleTrackPropertyGroupExpanded(
                        trackId,
                        `effect.${effect.id}`
                      );
                    }}
                  >
                    <span
                      className={`property-group-arrow ${
                        isTrackPropertyGroupExpanded(
                          trackId,
                          `effect.${effect.id}`
                        )
                          ? 'expanded'
                          : ''
                      }`}
                    >
                      {'\u25B6'}
                    </span>
                    <span>{effect.name}</span>
                  </div>
                  {isTrackPropertyGroupExpanded(
                    trackId,
                    `effect.${effect.id}`
                  ) && (
                    <>
                      {paramsWithKeyframes.map((paramName) => (
                        <div key={paramName} className="property-label-row sub">
                          <span className="property-label">{paramName}</span>
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

function TimelineHeaderComponent({
  track,
  isDimmed,
  isExpanded,
  dynamicHeight,
  hasKeyframes,
  selectedClipId,
  clips,
  onToggleExpand,
  onToggleSolo,
  onToggleMuted,
  onToggleVisible,
  onWheel,
  isTrackPropertyGroupExpanded,
  toggleTrackPropertyGroupExpanded,
  hasPropertyKeyframes,
}: TimelineHeaderProps) {
  // Get the selected clip in this track
  const trackClips = clips.filter((c) => c.trackId === track.id);
  const selectedTrackClip = trackClips.find((c) => c.id === selectedClipId);

  return (
    <div
      className={`track-header ${track.type} ${isDimmed ? 'dimmed' : ''} ${
        isExpanded ? 'expanded' : ''
      }`}
      style={{ height: dynamicHeight }}
      onWheel={onWheel}
    >
      <div className="track-header-top" style={{ height: track.height }}>
        <div className="track-header-main">
          {/* Only video tracks get expand arrow */}
          {track.type === 'video' && (
            <span
              className={`track-expand-arrow ${isExpanded ? 'expanded' : ''} ${
                hasKeyframes ? 'has-keyframes' : ''
              }`}
              onClick={(e) => {
                e.stopPropagation();
                onToggleExpand();
              }}
              title={isExpanded ? 'Collapse properties' : 'Expand properties'}
            >
              {'\u25B6'}
            </span>
          )}
          <span className="track-name">{track.name}</span>
        </div>
        <div className="track-controls">
          <button
            className={`btn-icon ${track.solo ? 'solo-active' : ''}`}
            onClick={onToggleSolo}
            title={track.solo ? 'Solo On' : 'Solo Off'}
          >
            S
          </button>
          {track.type === 'audio' && (
            <button
              className={`btn-icon ${track.muted ? 'muted' : ''}`}
              onClick={onToggleMuted}
              title={track.muted ? 'Unmute' : 'Mute'}
            >
              {track.muted ? '\uD83D\uDD07' : '\uD83D\uDD0A'}
            </button>
          )}
          {track.type === 'video' && (
            <button
              className={`btn-icon ${!track.visible ? 'hidden' : ''}`}
              onClick={onToggleVisible}
              title={track.visible ? 'Hide' : 'Show'}
            >
              {track.visible ? '\uD83D\uDC41' : '\uD83D\uDC41\u200D\uD83D\uDDE8'}
            </button>
          )}
        </div>
      </div>
      {/* Property labels - shown when track is expanded */}
      {track.type === 'video' && isExpanded && (
        <TrackPropertyLabels
          trackId={track.id}
          selectedClip={selectedTrackClip || null}
          isTrackPropertyGroupExpanded={isTrackPropertyGroupExpanded}
          toggleTrackPropertyGroupExpanded={toggleTrackPropertyGroupExpanded}
          hasPropertyKeyframes={hasPropertyKeyframes}
        />
      )}
    </div>
  );
}

export const TimelineHeader = memo(TimelineHeaderComponent);
