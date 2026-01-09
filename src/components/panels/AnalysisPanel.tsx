// Analysis Panel - Focus, Motion, and Face detection for clips

import { useCallback, useMemo } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { FrameAnalysisData } from '../../types';
import './AnalysisPanel.css';

export function AnalysisPanel() {
  const { clips, selectedClipIds, playheadPosition } = useTimelineStore();

  // Get first selected clip ID
  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;

  // Get selected clip
  const selectedClip = useMemo(() => {
    if (selectedClipId) {
      return clips.find(c => c.id === selectedClipId);
    }
    return clips.find(c => c.source?.type === 'video');
  }, [clips, selectedClipId]);

  const analysis = selectedClip?.analysis;
  const analysisStatus = selectedClip?.analysisStatus ?? 'none';
  const analysisProgress = selectedClip?.analysisProgress ?? 0;

  // Calculate summary stats
  const stats = useMemo(() => {
    if (!analysis?.frames.length) return null;

    const frames = analysis.frames;
    const avgFocus = frames.reduce((sum, f) => sum + f.focus, 0) / frames.length;
    const avgMotion = frames.reduce((sum, f) => sum + f.motion, 0) / frames.length;
    const maxFocus = Math.max(...frames.map(f => f.focus));
    const maxMotion = Math.max(...frames.map(f => f.motion));
    const totalFaces = frames.reduce((sum, f) => sum + f.faceCount, 0);

    // Find best focus segment
    let bestFocusStart = 0;
    let bestFocusScore = 0;
    for (let i = 0; i < frames.length; i++) {
      if (frames[i].focus > bestFocusScore) {
        bestFocusScore = frames[i].focus;
        bestFocusStart = frames[i].timestamp;
      }
    }

    return {
      avgFocus: Math.round(avgFocus * 100),
      avgMotion: Math.round(avgMotion * 100),
      maxFocus: Math.round(maxFocus * 100),
      maxMotion: Math.round(maxMotion * 100),
      totalFaces,
      bestFocusTime: bestFocusStart,
      frameCount: frames.length,
    };
  }, [analysis]);

  // Calculate real-time values at current playhead position
  const currentValues = useMemo((): FrameAnalysisData | null => {
    if (!analysis?.frames.length || !selectedClip) return null;

    // Check if playhead is within this clip's time range
    const clipStart = selectedClip.startTime;
    const clipEnd = clipStart + (selectedClip.outPoint - selectedClip.inPoint);

    if (playheadPosition < clipStart || playheadPosition > clipEnd) {
      return null;
    }

    // Calculate the source time at the playhead
    const timeInClip = playheadPosition - clipStart;
    const sourceTime = selectedClip.inPoint + timeInClip;

    // Find the closest frame in the analysis
    let closestFrame = analysis.frames[0];
    let closestDistance = Math.abs(closestFrame.timestamp - sourceTime);

    for (const frame of analysis.frames) {
      const distance = Math.abs(frame.timestamp - sourceTime);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestFrame = frame;
      }
    }

    return closestFrame;
  }, [analysis, selectedClip, playheadPosition]);

  // Handle analyze button click
  const handleAnalyze = useCallback(async () => {
    if (!selectedClipId) return;

    const { analyzeClip } = await import('../../services/clipAnalyzer');
    await analyzeClip(selectedClipId);
  }, [selectedClipId]);

  // Handle cancel analysis
  const handleCancel = useCallback(async () => {
    const { cancelAnalysis } = await import('../../services/clipAnalyzer');
    cancelAnalysis();
  }, []);

  // Handle clear analysis
  const handleClear = useCallback(async () => {
    if (!selectedClipId) return;

    const { clearClipAnalysis } = await import('../../services/clipAnalyzer');
    clearClipAnalysis(selectedClipId);
  }, [selectedClipId]);

  // Render empty state
  if (!selectedClip) {
    return (
      <div className="analysis-panel">
        <div className="analysis-header">
          <h2>Analysis</h2>
        </div>
        <div className="analysis-empty">
          <p>Select a video clip to analyze</p>
        </div>
      </div>
    );
  }

  // Check if it's a video clip
  const isVideo = selectedClip.source?.type === 'video' || selectedClip.file?.type.startsWith('video/');

  if (!isVideo) {
    return (
      <div className="analysis-panel">
        <div className="analysis-header">
          <h2>Analysis</h2>
        </div>
        <div className="analysis-empty">
          <p>Analysis only available for video clips</p>
        </div>
      </div>
    );
  }

  return (
    <div className="analysis-panel">
      {/* Header */}
      <div className="analysis-header">
        <h2>Analysis</h2>
      </div>

      {/* Clip info */}
      <div className="analysis-clip-info">
        <span className="clip-name" title={selectedClip.name}>
          {selectedClip.name}
        </span>
        {analysisStatus === 'analyzing' && (
          <span className="analysis-status analyzing">
            Analyzing... {analysisProgress}%
          </span>
        )}
        {analysisStatus === 'ready' && (
          <span className="analysis-status ready">
            {stats?.frameCount} frames
          </span>
        )}
        {analysisStatus === 'error' && (
          <span className="analysis-status error">
            Error
          </span>
        )}
      </div>

      {/* Real-time values at playhead */}
      {(analysisStatus === 'ready' || analysisStatus === 'analyzing') && currentValues && (
        <div className="analysis-realtime">
          <h3>Current Values</h3>
          <div className="realtime-grid">
            <div className="realtime-item focus">
              <span className="realtime-label">Focus</span>
              <div className="realtime-bar-container">
                <div
                  className="realtime-bar"
                  style={{ width: `${Math.round(currentValues.focus * 100)}%` }}
                />
              </div>
              <span className="realtime-value">{Math.round(currentValues.focus * 100)}%</span>
            </div>
            <div className="realtime-item motion">
              <span className="realtime-label">Global Motion</span>
              <div className="realtime-bar-container">
                <div
                  className="realtime-bar"
                  style={{ width: `${Math.round((currentValues.globalMotion ?? currentValues.motion) * 100)}%` }}
                />
              </div>
              <span className="realtime-value">{Math.round((currentValues.globalMotion ?? currentValues.motion) * 100)}%</span>
            </div>
            <div className="realtime-item local-motion">
              <span className="realtime-label">Local Motion</span>
              <div className="realtime-bar-container">
                <div
                  className="realtime-bar"
                  style={{ width: `${Math.round((currentValues.localMotion ?? 0) * 100)}%` }}
                />
              </div>
              <span className="realtime-value">{Math.round((currentValues.localMotion ?? 0) * 100)}%</span>
            </div>
            {currentValues.faceCount > 0 && (
              <div className="realtime-item faces">
                <span className="realtime-label">Faces</span>
                <span className="realtime-value face-count">{currentValues.faceCount}</span>
              </div>
            )}
            {currentValues.isSceneCut && (
              <div className="realtime-item scene-cut">
                <span className="scene-cut-badge">Scene Cut</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Playhead outside clip indicator */}
      {(analysisStatus === 'ready' || analysisStatus === 'analyzing') && analysis?.frames.length && !currentValues && (
        <div className="analysis-realtime analysis-outside">
          <h3>Current Values</h3>
          <div className="realtime-placeholder">
            <span>Move playhead over clip to see values</span>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="analysis-actions">
        {analysisStatus !== 'ready' && analysisStatus !== 'analyzing' && (
          <button
            className="btn-analyze"
            onClick={handleAnalyze}
          >
            Analyze Clip
          </button>
        )}
        {analysisStatus === 'ready' && (
          <div className="analysis-btn-row">
            <button className="btn-analyze btn-secondary" onClick={handleAnalyze}>
              Re-analyze
            </button>
            <button className="btn-analyze btn-danger" onClick={handleClear}>
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Progress bar and cancel button */}
      {analysisStatus === 'analyzing' && (
        <div className="analysis-progress-section">
          <div className="analysis-progress">
            <div
              className="analysis-progress-bar"
              style={{ width: `${analysisProgress}%` }}
            />
          </div>
          <button className="btn-analyze btn-cancel" onClick={handleCancel}>
            Cancel
          </button>
        </div>
      )}

      {/* Stats */}
      {analysisStatus === 'ready' && stats && (
        <div className="analysis-stats">
          <div className="stat-section">
            <h3>Focus (Sharpness)</h3>
            <div className="stat-row">
              <span className="stat-label">Average</span>
              <div className="stat-bar-container">
                <div className="stat-bar focus" style={{ width: `${stats.avgFocus}%` }} />
              </div>
              <span className="stat-value">{stats.avgFocus}%</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Peak</span>
              <div className="stat-bar-container">
                <div className="stat-bar focus" style={{ width: `${stats.maxFocus}%` }} />
              </div>
              <span className="stat-value">{stats.maxFocus}%</span>
            </div>
          </div>

          <div className="stat-section">
            <h3>Motion</h3>
            <div className="stat-row">
              <span className="stat-label">Average</span>
              <div className="stat-bar-container">
                <div className="stat-bar motion" style={{ width: `${stats.avgMotion}%` }} />
              </div>
              <span className="stat-value">{stats.avgMotion}%</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Peak</span>
              <div className="stat-bar-container">
                <div className="stat-bar motion" style={{ width: `${stats.maxMotion}%` }} />
              </div>
              <span className="stat-value">{stats.maxMotion}%</span>
            </div>
          </div>

          <div className="stat-section">
            <h3>Faces</h3>
            <div className="stat-row">
              <span className="stat-label">Detected</span>
              <span className="stat-value">{stats.totalFaces} total</span>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      {analysisStatus === 'ready' && (
        <div className="analysis-legend">
          <h3>Clip Overlay</h3>
          <div className="legend-items">
            <div className="legend-item">
              <span className="legend-color focus" />
              <span>Focus (green)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color motion" />
              <span>Motion (blue)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color face" />
              <span>Face (yellow)</span>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="analysis-footer">
        <span className="analysis-hint">
          Analysis samples every 500ms
        </span>
      </div>
    </div>
  );
}
