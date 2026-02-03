// TimelineControls component - Playback controls and toolbar

import { memo, useState, useRef, useEffect } from 'react';
import type { TimelineControlsProps } from './types';

function TimelineControlsComponent({
  isPlaying,
  loopPlayback,
  playheadPosition,
  duration,
  zoom,
  snappingEnabled,
  inPoint,
  outPoint,
  ramPreviewEnabled,
  proxyEnabled,
  currentlyGeneratingProxyId,
  mediaFilesWithProxy,
  showTranscriptMarkers,
  thumbnailsEnabled,
  waveformsEnabled,
  toolMode,
  onPlay,
  onPause,
  onStop,
  onToggleLoop,
  onSetZoom,
  onToggleSnapping,
  onSetInPoint,
  onSetOutPoint,
  onClearInOut,
  onToggleRamPreview,
  onToggleProxy,
  onStartProxyCachePreload,
  onCancelProxyCachePreload,
  isProxyCaching,
  proxyCacheProgress,
  onToggleTranscriptMarkers,
  onToggleThumbnails,
  onToggleWaveforms,
  onToggleCutTool,
  onAddVideoTrack,
  onAddAudioTrack,
  onAddTextClip,
  onSetDuration,
  onFitToWindow,
  formatTime,
  parseTime,
}: TimelineControlsProps) {
  const [isEditingDuration, setIsEditingDuration] = useState(false);
  const [durationInputValue, setDurationInputValue] = useState('');
  const [viewDropdownOpen, setViewDropdownOpen] = useState(false);
  const durationInputRef = useRef<HTMLInputElement>(null);
  const viewDropdownRef = useRef<HTMLDivElement>(null);

  // Focus input when editing starts
  useEffect(() => {
    if (isEditingDuration && durationInputRef.current) {
      durationInputRef.current.focus();
      durationInputRef.current.select();
    }
  }, [isEditingDuration]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (viewDropdownRef.current && !viewDropdownRef.current.contains(e.target as Node)) {
        setViewDropdownOpen(false);
      }
    };
    if (viewDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [viewDropdownOpen]);

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
        <button
          className={`btn btn-sm btn-icon ${snappingEnabled ? 'btn-active' : ''}`}
          onClick={onToggleSnapping}
          title={snappingEnabled ? 'Snapping enabled - clips snap to edges' : 'Snapping disabled - free positioning'}
        >
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L12 6M12 18L12 22M2 12L6 12M18 12L22 12" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>
        <button
          className={`btn btn-sm btn-icon ${toolMode === 'cut' ? 'btn-active' : ''}`}
          onClick={onToggleCutTool}
          title={toolMode === 'cut' ? 'Cut Tool active (C) - click clips to split' : 'Cut Tool (C) - click to activate'}
        >
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="6" cy="6" r="3" />
            <circle cx="6" cy="18" r="3" />
            <line x1="20" y1="4" x2="8.12" y2="15.88" />
            <line x1="14.47" y1="14.48" x2="20" y2="20" />
            <line x1="8.12" y1="8.12" x2="12" y2="12" />
          </svg>
        </button>
        <button className="btn btn-sm" onClick={() => onSetZoom(zoom - 10)} title="Zoom out">
          {'\u2212'}
        </button>
        <button className="btn btn-sm" onClick={() => onSetZoom(zoom + 10)} title="Zoom in">
          +
        </button>
        <button className="btn btn-sm" onClick={onFitToWindow} title="Fit composition to window">
          Fit
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
          className={`btn btn-sm ${isProxyCaching ? 'btn-active' : ''}`}
          onClick={isProxyCaching ? onCancelProxyCachePreload : onStartProxyCachePreload}
          title={
            isProxyCaching
              ? `Caching proxy frames... ${proxyCacheProgress ?? 0}% - Click to cancel`
              : 'Cache all proxy frames for instant scrubbing'
          }
          disabled={!proxyEnabled || mediaFilesWithProxy === 0}
        >
          {isProxyCaching ? `Cache ${proxyCacheProgress ?? 0}%` : 'Cache'}
        </button>
        <div className="view-dropdown" ref={viewDropdownRef}>
          <button
            className={`btn btn-sm ${viewDropdownOpen ? 'btn-active' : ''}`}
            onClick={() => setViewDropdownOpen(!viewDropdownOpen)}
            title="View options"
          >
            View
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginLeft: 4 }}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
          {viewDropdownOpen && (
            <div className="view-dropdown-menu">
              <label
                className="view-dropdown-item"
                onClick={(e) => { e.preventDefault(); onToggleProxy(); }}
              >
                <input type="checkbox" checked={proxyEnabled} readOnly />
                <span>
                  Proxy
                  {currentlyGeneratingProxyId && ' (Generating...)'}
                  {!currentlyGeneratingProxyId && mediaFilesWithProxy > 0 && ` (${mediaFilesWithProxy})`}
                </span>
              </label>
              <label
                className="view-dropdown-item"
                onClick={(e) => { e.preventDefault(); onToggleThumbnails(); }}
              >
                <input type="checkbox" checked={thumbnailsEnabled} readOnly />
                <span>Thumbnails</span>
              </label>
              <label
                className="view-dropdown-item"
                onClick={(e) => { e.preventDefault(); onToggleWaveforms(); }}
              >
                <input type="checkbox" checked={waveformsEnabled} readOnly />
                <span>Waveforms</span>
              </label>
              <label
                className="view-dropdown-item"
                onClick={(e) => { e.preventDefault(); onToggleTranscriptMarkers(); }}
              >
                <input type="checkbox" checked={showTranscriptMarkers} readOnly />
                <span>Transcript Markers</span>
              </label>
            </div>
          )}
        </div>
      </div>
      <div className="timeline-tracks-controls">
        <button className="btn btn-sm" onClick={onAddVideoTrack}>
          + Video Track
        </button>
        <button className="btn btn-sm" onClick={onAddAudioTrack}>
          + Audio Track
        </button>
        <button className="btn btn-sm" onClick={onAddTextClip} title="Add text clip at playhead">
          + Text
        </button>
      </div>
    </div>
  );
}

export const TimelineControls = memo(TimelineControlsComponent);
