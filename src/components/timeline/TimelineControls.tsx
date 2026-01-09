// TimelineControls component - Playback controls and toolbar

import { memo, useState, useRef, useEffect } from 'react';
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
  showTranscriptMarkers,
  thumbnailsEnabled,
  waveformsEnabled,
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
  onToggleTranscriptMarkers,
  onToggleThumbnails,
  onToggleWaveforms,
  onAddVideoTrack,
  onAddAudioTrack,
  onSetDuration,
  formatTime,
  parseTime,
}: TimelineControlsProps) {
  const [isEditingDuration, setIsEditingDuration] = useState(false);
  const [durationInputValue, setDurationInputValue] = useState('');
  const durationInputRef = useRef<HTMLInputElement>(null);

  // Focus input when editing starts
  useEffect(() => {
    if (isEditingDuration && durationInputRef.current) {
      durationInputRef.current.focus();
      durationInputRef.current.select();
    }
  }, [isEditingDuration]);

  const handleDurationClick = () => {
    setDurationInputValue(formatTime(duration));
    setIsEditingDuration(true);
  };

  const handleDurationSubmit = () => {
    const newDuration = parseTime(durationInputValue);
    if (newDuration !== null && newDuration > 0) {
      onSetDuration(newDuration);
    }
    setIsEditingDuration(false);
  };

  const handleDurationKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleDurationSubmit();
    } else if (e.key === 'Escape') {
      setIsEditingDuration(false);
    }
  };

  const handleDurationBlur = () => {
    handleDurationSubmit();
  };
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
        {formatTime(playheadPosition)} /{' '}
        {isEditingDuration ? (
          <input
            ref={durationInputRef}
            type="text"
            className="duration-input"
            value={durationInputValue}
            onChange={(e) => setDurationInputValue(e.target.value)}
            onKeyDown={handleDurationKeyDown}
            onBlur={handleDurationBlur}
          />
        ) : (
          <span
            className="duration-display"
            onClick={handleDurationClick}
            title="Click to edit composition duration"
          >
            {formatTime(duration)}
          </span>
        )}
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
        <button
          className={`btn btn-sm ${showTranscriptMarkers ? 'btn-active' : ''}`}
          onClick={onToggleTranscriptMarkers}
          title={
            showTranscriptMarkers
              ? 'Transcript markers visible - click to hide'
              : 'Transcript markers hidden - click to show'
          }
        >
          T {showTranscriptMarkers ? 'On' : 'Off'}
        </button>
        <button
          className={`btn btn-sm ${thumbnailsEnabled ? 'btn-active' : ''}`}
          onClick={onToggleThumbnails}
          title={
            thumbnailsEnabled
              ? 'Thumbnails enabled - generating thumbnails for clips'
              : 'Thumbnails disabled - improves performance for long videos'
          }
        >
          Thumb {thumbnailsEnabled ? 'On' : 'Off'}
        </button>
        <button
          className={`btn btn-sm ${waveformsEnabled ? 'btn-active' : ''}`}
          onClick={onToggleWaveforms}
          title={
            waveformsEnabled
              ? 'Waveforms enabled - generating audio waveforms'
              : 'Waveforms disabled - improves performance for long audio'
          }
        >
          Wave {waveformsEnabled ? 'On' : 'Off'}
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
