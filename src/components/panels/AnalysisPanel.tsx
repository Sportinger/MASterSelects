// Analysis Panel - Focus, Motion, and Face detection for clips

import { useCallback, useMemo } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import './AnalysisPanel.css';

export function AnalysisPanel() {
  const { clips, selectedClipIds } = useTimelineStore();

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

  // Handle analyze button click
  const handleAnalyze = useCallback(async () => {
    if (!selectedClipId) return;

    const { analyzeClip } = await import('../../services/clipAnalyzer');
    await analyzeClip(selectedClipId);
  }, [selectedClipId]);

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

      {/* Actions */}
      <div className="analysis-actions">
        {analysisStatus !== 'ready' && analysisStatus !== 'analyzing' && (
          <button
            className="btn-analyze"
            onClick={handleAnalyze}
            disabled={analysisStatus === 'analyzing'}
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

      {/* Progress bar */}
      {analysisStatus === 'analyzing' && (
        <div className="analysis-progress">
          <div
            className="analysis-progress-bar"
            style={{ width: `${analysisProgress}%` }}
          />
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
