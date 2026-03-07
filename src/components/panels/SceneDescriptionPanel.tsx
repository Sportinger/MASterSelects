// Scene Description Panel - AI-powered video content description
// Shows scene descriptions with real-time highlighting during playback

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timeline';
import type { SceneSegment } from '../../types';
import './SceneDescriptionPanel.css';

// =============================================================================
// Sub-components
// =============================================================================

interface SegmentProps {
  segment: SceneSegment;
  isActive: boolean;
  isHighlighted: boolean;
  onClick: (time: number) => void;
}

function Segment({ segment, isActive, isHighlighted, onClick }: SegmentProps) {
  return (
    <div
      className={`scene-segment ${isActive ? 'active' : ''} ${isHighlighted ? 'highlighted' : ''}`}
      onClick={() => onClick(segment.start)}
    >
      <div className="scene-segment-header">
        <span className="scene-segment-time">
          {formatTime(segment.start)} - {formatTime(segment.end)}
        </span>
      </div>
      <div className="scene-segment-text">
        {segment.text}
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// =============================================================================
// Main Component
// =============================================================================

export function SceneDescriptionPanel() {
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  const {
    clips,
    selectedClipIds,
    playheadPosition,
    setPlayheadPosition,
  } = useTimelineStore(useShallow(s => ({
    clips: s.clips,
    selectedClipIds: s.selectedClipIds,
    playheadPosition: s.playheadPosition,
    setPlayheadPosition: s.setPlayheadPosition,
  })));

  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;

  const selectedClip = useMemo(() => {
    if (selectedClipId) {
      return clips.find(c => c.id === selectedClipId);
    }
    return clips.find(c => c.sceneDescriptions && c.sceneDescriptions.length > 0);
  }, [clips, selectedClipId]);

  const segments = selectedClip?.sceneDescriptions ?? [];
  const descStatus = selectedClip?.sceneDescriptionStatus ?? 'none';
  const descProgress = selectedClip?.sceneDescriptionProgress ?? 0;
  const descMessage = selectedClip?.sceneDescriptionMessage;

  // Calculate clip-local time for segment matching
  const clipLocalTime = useMemo(() => {
    if (!selectedClip) return -1;
    return playheadPosition - selectedClip.startTime + selectedClip.inPoint;
  }, [selectedClip, playheadPosition]);

  // Find current active segment
  const activeSegmentId = useMemo(() => {
    if (clipLocalTime < 0 || segments.length === 0) return null;

    for (const seg of segments) {
      if (clipLocalTime >= seg.start && clipLocalTime < seg.end) {
        return seg.id;
      }
    }
    return null;
  }, [segments, clipLocalTime]);

  // Filter by search
  const filteredSegments = useMemo(() => {
    if (!searchQuery.trim()) return segments;
    const query = searchQuery.toLowerCase();
    return segments.filter(s => s.text.toLowerCase().includes(query));
  }, [segments, searchQuery]);

  // Click to seek
  const handleSegmentClick = useCallback((sourceTime: number) => {
    if (!selectedClip) return;
    const timelinePosition = selectedClip.startTime + (sourceTime - selectedClip.inPoint);
    setPlayheadPosition(Math.max(0, timelinePosition));
  }, [selectedClip, setPlayheadPosition]);

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeSegmentId && containerRef.current) {
      const activeElement = containerRef.current.querySelector('.scene-segment.active');
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [activeSegmentId]);

  // Handle describe button
  const handleDescribe = useCallback(async () => {
    if (!selectedClipId) return;
    const { describeClip } = await import('../../services/sceneDescriber');
    await describeClip(selectedClipId);
  }, [selectedClipId]);

  // Handle cancel
  const handleCancel = useCallback(async () => {
    const { cancelDescription } = await import('../../services/sceneDescriber');
    cancelDescription();
  }, []);

  // Handle clear
  const handleClear = useCallback(async () => {
    if (!selectedClipId) return;
    const { clearSceneDescriptions } = await import('../../services/sceneDescriber');
    clearSceneDescriptions(selectedClipId);
  }, [selectedClipId]);

  // Empty state
  if (!selectedClip) {
    return (
      <div className="scene-description-panel">
        <div className="scene-description-header">
          <h2>AI Scene Description</h2>
        </div>
        <div className="scene-description-empty">
          <p>Select a video clip to generate AI scene descriptions</p>
        </div>
      </div>
    );
  }

  const isVideo = selectedClip.source?.type === 'video' || selectedClip.file?.type.startsWith('video/');
  if (!isVideo) {
    return (
      <div className="scene-description-panel">
        <div className="scene-description-header">
          <h2>AI Scene Description</h2>
        </div>
        <div className="scene-description-empty">
          <p>Scene description is only available for video clips</p>
        </div>
      </div>
    );
  }

  return (
    <div className="scene-description-panel">
      {/* Header */}
      <div className="scene-description-header">
        <h2>AI Scene Description</h2>
      </div>

      {/* Search */}
      {segments.length > 0 && (
        <div className="scene-description-search">
          <input
            type="text"
            placeholder="Search descriptions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button className="search-clear" onClick={() => setSearchQuery('')}>x</button>
          )}
        </div>
      )}

      {/* Clip info */}
      <div className="scene-description-clip-info">
        <span className="clip-name" title={selectedClip.name}>
          {selectedClip.name}
        </span>
        {descStatus === 'describing' && (
          <span className="scene-description-status describing" title={descMessage}>
            {descProgress}%
          </span>
        )}
        {descStatus === 'ready' && (
          <span className="scene-description-status ready">
            {segments.length} scenes
          </span>
        )}
        {descStatus === 'error' && (
          <span className="scene-description-status error" title={descMessage}>
            Error
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="scene-description-actions">
        {descStatus !== 'ready' && descStatus !== 'describing' && (
          <button className="btn-describe" onClick={handleDescribe}>
            AI Describe
          </button>
        )}
        {descStatus === 'describing' && (
          <button className="btn-describe btn-cancel" onClick={handleCancel}>
            Cancel
          </button>
        )}
        {descStatus === 'ready' && (
          <div className="scene-description-btn-row">
            <button className="btn-describe btn-secondary" onClick={handleDescribe}>
              Re-describe
            </button>
            <button className="btn-describe btn-danger" onClick={handleClear}>
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Progress bar */}
      {descStatus === 'describing' && (
        <div className="scene-description-progress">
          <div
            className="scene-description-progress-bar"
            style={{ width: `${descProgress}%` }}
          />
        </div>
      )}

      {/* Status message */}
      {descStatus === 'describing' && descMessage && (
        <div className="scene-description-message">{descMessage}</div>
      )}

      {/* Error message */}
      {descStatus === 'error' && descMessage && (
        <div className="scene-description-error">{descMessage}</div>
      )}

      {/* Content */}
      <div className="scene-description-content" ref={containerRef}>
        {filteredSegments.length === 0 ? (
          <div className="scene-description-empty">
            {descStatus === 'describing' ? (
              <p>Analyzing video... Descriptions will appear here.</p>
            ) : segments.length === 0 && descStatus !== 'error' ? (
              <p>Click "AI Describe" to generate scene descriptions using local AI (Ollama).</p>
            ) : searchQuery ? (
              <p>No results for "{searchQuery}"</p>
            ) : null}
          </div>
        ) : (
          filteredSegments.map((segment) => (
            <Segment
              key={segment.id}
              segment={segment}
              isActive={segment.id === activeSegmentId}
              isHighlighted={!!searchQuery && segment.text.toLowerCase().includes(searchQuery.toLowerCase())}
              onClick={handleSegmentClick}
            />
          ))
        )}
      </div>

      {/* Footer */}
      <div className="scene-description-footer">
        <span className="scene-description-hint">
          Click segment to seek. Powered by Ollama ({descStatus === 'ready' ? 'local AI' : 'qwen3.5:9b'}).
        </span>
      </div>
    </div>
  );
}
