// TimelineControls component - Playback controls and toolbar

import { memo } from 'react';
import type { TimelineControlsProps } from './types';

function TimelineControlsComponent({
  isPlaying,
  loopPlayback,
  playheadPosition,
  duration,
  zoom,
  inPoint,
  outPoint,
  ramPreviewEnabled,
  proxyEnabled,
  currentlyGeneratingProxyId,
  mediaFilesWithProxy,
  onPlay,
  onPause,
  onStop,
  onToggleLoop,
  onSetZoom,
  onSetInPoint,
  onSetOutPoint,
  onClearInOut,
  onToggleRamPreview,
  onToggleProxy,
  onAddVideoTrack,
  onAddAudioTrack,
  formatTime,
}: TimelineControlsProps) {
  return (
    <div className="timeline-toolbar">
      <div className="timeline-controls">
        <button className="btn btn-sm" onClick={onStop} title="Stop">
          {'\u23F9'}
        </button>
        <button
          className={`btn btn-sm ${isPlaying ? 'btn-active' : ''}`}
          onClick={isPlaying ? onPause : onPlay}
        >
          {isPlaying ? '\u23F8' : '\u25B6'}
        </button>
        <button
          className={`btn btn-sm ${loopPlayback ? 'btn-active' : ''}`}
          onClick={onToggleLoop}
          title={loopPlayback ? 'Loop On (L)' : 'Loop Off (L)'}
        >
          {'\uD83D\uDD01'}
        </button>
      </div>
      <div className="timeline-time">
        {formatTime(playheadPosition)} / {formatTime(duration)}
      </div>
      <div className="timeline-zoom">
        <button className="btn btn-sm" onClick={() => onSetZoom(zoom - 10)}>
          {'\u2212'}
        </button>
        <span>{Math.round(zoom)}px/s</span>
        <button className="btn btn-sm" onClick={() => onSetZoom(zoom + 10)}>
          +
        </button>
      </div>
      <div className="timeline-inout-controls">
        <button
          className={`btn btn-sm ${inPoint !== null ? 'btn-active' : ''}`}
          onClick={onSetInPoint}
          title="Set In point (I)"
        >
          I
        </button>
        <button
          className={`btn btn-sm ${outPoint !== null ? 'btn-active' : ''}`}
          onClick={onSetOutPoint}
          title="Set Out point (O)"
        >
          O
        </button>
        {(inPoint !== null || outPoint !== null) && (
          <button
            className="btn btn-sm"
            onClick={onClearInOut}
            title="Clear In/Out (X)"
          >
            X
          </button>
        )}
      </div>
      <div className="timeline-ram-preview">
        <button
          className={`btn btn-sm ${ramPreviewEnabled ? 'btn-active' : ''}`}
          onClick={onToggleRamPreview}
          title={
            ramPreviewEnabled
              ? 'RAM Preview ON - Auto-caches frames for instant scrubbing. Click to disable and clear cache.'
              : 'RAM Preview OFF - Click to enable auto-caching for instant scrubbing'
          }
        >
          RAM {ramPreviewEnabled ? 'ON' : 'OFF'}
        </button>
        <button
          className={`btn btn-sm ${proxyEnabled ? 'btn-active' : ''}`}
          onClick={onToggleProxy}
          title={
            proxyEnabled
              ? 'Proxy enabled - using optimized frames'
              : 'Proxy disabled - using original video'
          }
        >
          {currentlyGeneratingProxyId ? (
            <>
              {'\u23F3'} Generating...
            </>
          ) : (
            <>
              {proxyEnabled ? '\uD83C\uDFAC' : '\uD83C\uDFA5'} Proxy{' '}
              {proxyEnabled ? 'On' : 'Off'}
              {mediaFilesWithProxy > 0 && (
                <span className="proxy-count">({mediaFilesWithProxy})</span>
              )}
            </>
          )}
        </button>
      </div>
      <div className="timeline-tracks-controls">
        <button className="btn btn-sm" onClick={onAddVideoTrack}>
          + Video Track
        </button>
        <button className="btn btn-sm" onClick={onAddAudioTrack}>
          + Audio Track
        </button>
      </div>
    </div>
  );
}

export const TimelineControls = memo(TimelineControlsComponent);
