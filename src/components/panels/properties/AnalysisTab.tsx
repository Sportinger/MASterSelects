// Analysis Tab - View clip analysis data (focus, motion, faces)
import { useMemo, useCallback } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import type { FrameAnalysisData } from '../../../types';

interface AnalysisTabProps {
  clipId: string;
  analysis: { frames: FrameAnalysisData[] } | undefined;
  analysisStatus: 'none' | 'analyzing' | 'ready' | 'error';
  analysisProgress: number;
  clipStartTime: number;
  inPoint: number;
  outPoint: number;
}

export function AnalysisTab({ clipId, analysis, analysisStatus, analysisProgress, clipStartTime, inPoint, outPoint }: AnalysisTabProps) {
  // Reactive data - subscribe to specific value only
  const playheadPosition = useTimelineStore(state => state.playheadPosition);

  // Calculate current values at playhead
  const currentValues = useMemo((): FrameAnalysisData | null => {
    if (!analysis?.frames.length) return null;

    const clipEnd = clipStartTime + (outPoint - inPoint);
    if (playheadPosition < clipStartTime || playheadPosition > clipEnd) return null;

    const timeInClip = playheadPosition - clipStartTime;
    const sourceTime = inPoint + timeInClip;

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
  }, [analysis, clipStartTime, inPoint, outPoint, playheadPosition]);

  // Stats summary
  const stats = useMemo(() => {
    if (!analysis?.frames.length) return null;
    const frames = analysis.frames;
    return {
      avgFocus: Math.round(frames.reduce((s, f) => s + f.focus, 0) / frames.length * 100),
      avgMotion: Math.round(frames.reduce((s, f) => s + f.motion, 0) / frames.length * 100),
      maxFocus: Math.round(Math.max(...frames.map(f => f.focus)) * 100),
      maxMotion: Math.round(Math.max(...frames.map(f => f.motion)) * 100),
      totalFaces: frames.reduce((s, f) => s + f.faceCount, 0),
      frameCount: frames.length,
    };
  }, [analysis]);

  const handleAnalyze = useCallback(async () => {
    const { analyzeClip } = await import('../../../services/clipAnalyzer');
    await analyzeClip(clipId);
  }, [clipId]);

  const handleCancel = useCallback(async () => {
    const { cancelAnalysis } = await import('../../../services/clipAnalyzer');
    cancelAnalysis();
  }, []);

  const handleClear = useCallback(async () => {
    const { clearClipAnalysis } = await import('../../../services/clipAnalyzer');
    clearClipAnalysis(clipId);
  }, [clipId]);

  return (
    <div className="properties-tab-content analysis-tab">
      {/* Actions */}
      <div className="properties-section">
        <div className="analysis-tab-actions">
          {analysisStatus !== 'ready' && analysisStatus !== 'analyzing' && (
            <button className="btn btn-sm" onClick={handleAnalyze}>Analyze Clip</button>
          )}
          {analysisStatus === 'analyzing' && (
            <button className="btn btn-sm btn-danger" onClick={handleCancel}>Cancel</button>
          )}
          {analysisStatus === 'ready' && (
            <>
              <button className="btn btn-sm" onClick={handleAnalyze}>Re-analyze</button>
              <button className="btn btn-sm btn-danger" onClick={handleClear}>Clear</button>
            </>
          )}
        </div>
      </div>

      {/* Progress */}
      {analysisStatus === 'analyzing' && (
        <div className="properties-section">
          <div className="analysis-progress-bar">
            <div className="analysis-progress-fill" style={{ width: `${analysisProgress}%` }} />
          </div>
          <span className="analysis-progress-text">{analysisProgress}%</span>
        </div>
      )}

      {/* Current values at playhead */}
      {currentValues && (
        <div className="properties-section">
          <h4>Current Frame</h4>
          <div className="analysis-realtime-grid">
            <div className="analysis-metric">
              <span className="metric-label">Focus</span>
              <div className="metric-bar"><div className="metric-fill focus" style={{ width: `${Math.round(currentValues.focus * 100)}%` }} /></div>
              <span className="metric-value">{Math.round(currentValues.focus * 100)}%</span>
            </div>
            <div className="analysis-metric">
              <span className="metric-label">Motion</span>
              <div className="metric-bar"><div className="metric-fill motion" style={{ width: `${Math.round(currentValues.motion * 100)}%` }} /></div>
              <span className="metric-value">{Math.round(currentValues.motion * 100)}%</span>
            </div>
            {currentValues.faceCount > 0 && (
              <div className="analysis-metric">
                <span className="metric-label">Faces</span>
                <span className="metric-value">{currentValues.faceCount}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stats summary */}
      {stats && (
        <div className="properties-section">
          <h4>Summary ({stats.frameCount} frames)</h4>
          <div className="analysis-stats-grid">
            <div className="stat-row"><span>Avg Focus:</span><span>{stats.avgFocus}%</span></div>
            <div className="stat-row"><span>Peak Focus:</span><span>{stats.maxFocus}%</span></div>
            <div className="stat-row"><span>Avg Motion:</span><span>{stats.avgMotion}%</span></div>
            <div className="stat-row"><span>Peak Motion:</span><span>{stats.maxMotion}%</span></div>
            <div className="stat-row"><span>Total Faces:</span><span>{stats.totalFaces}</span></div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {analysisStatus !== 'ready' && analysisStatus !== 'analyzing' && (
        <div className="analysis-empty-state">
          Click "Analyze Clip" to detect focus, motion, and faces.
        </div>
      )}
    </div>
  );
}
