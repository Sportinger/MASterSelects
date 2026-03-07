// Analysis Tab - View clip analysis data (focus, motion, faces) + AI scene descriptions
import { useMemo, useCallback } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import type { FrameAnalysisData, SceneSegment, SceneDescriptionStatus } from '../../../types';

interface AnalysisTabProps {
  clipId: string;
  analysis: { frames: FrameAnalysisData[] } | undefined;
  analysisStatus: 'none' | 'analyzing' | 'ready' | 'error';
  analysisProgress: number;
  clipStartTime: number;
  inPoint: number;
  outPoint: number;
  sceneDescriptions?: SceneSegment[];
  sceneDescriptionStatus?: SceneDescriptionStatus;
  sceneDescriptionProgress?: number;
  sceneDescriptionMessage?: string;
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export function AnalysisTab({ clipId, analysis, analysisStatus, analysisProgress, clipStartTime, inPoint, outPoint, sceneDescriptions, sceneDescriptionStatus, sceneDescriptionProgress, sceneDescriptionMessage }: AnalysisTabProps) {
  const descStatus = sceneDescriptionStatus ?? 'none';
  const descProgress = sceneDescriptionProgress ?? 0;
  const segments = sceneDescriptions ?? [];

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

  // AI scene description handlers
  const handleDescribe = useCallback(async () => {
    const { describeClip } = await import('../../../services/sceneDescriber');
    await describeClip(clipId);
  }, [clipId]);

  const handleCancelDescribe = useCallback(async () => {
    const { cancelDescription } = await import('../../../services/sceneDescriber');
    cancelDescription();
  }, []);

  const handleClearDescriptions = useCallback(async () => {
    const { clearSceneDescriptions } = await import('../../../services/sceneDescriber');
    clearSceneDescriptions(clipId);
  }, [clipId]);

  // Find active scene segment at playhead
  const activeSegment = useMemo(() => {
    if (segments.length === 0) return null;
    const clipEnd = clipStartTime + (outPoint - inPoint);
    if (playheadPosition < clipStartTime || playheadPosition > clipEnd) return null;
    const sourceTime = inPoint + (playheadPosition - clipStartTime);
    return segments.find(s => sourceTime >= s.start && sourceTime < s.end) ?? null;
  }, [segments, clipStartTime, inPoint, outPoint, playheadPosition]);

  const handleSeekToSegment = useCallback((sourceTime: number) => {
    const timelinePosition = clipStartTime + (sourceTime - inPoint);
    useTimelineStore.getState().setPlayheadPosition(Math.max(0, timelinePosition));
  }, [clipStartTime, inPoint]);

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

      {/* AI Scene Description Section */}
      <div className="properties-section" style={{ borderTop: '1px solid var(--border-color)', marginTop: '8px', paddingTop: '8px' }}>
        <h4>AI Scene Description</h4>
        <div className="analysis-tab-actions">
          {descStatus !== 'ready' && descStatus !== 'describing' && (
            <button className="btn btn-sm" onClick={handleDescribe}>AI Describe</button>
          )}
          {descStatus === 'describing' && (
            <button className="btn btn-sm btn-danger" onClick={handleCancelDescribe}>Cancel</button>
          )}
          {descStatus === 'ready' && (
            <>
              <button className="btn btn-sm" onClick={handleDescribe}>Re-describe</button>
              <button className="btn btn-sm btn-danger" onClick={handleClearDescriptions}>Clear</button>
            </>
          )}
        </div>

        {/* Progress */}
        {descStatus === 'describing' && (
          <div style={{ marginTop: '6px' }}>
            <div className="analysis-progress-bar">
              <div className="analysis-progress-fill" style={{ width: `${descProgress}%` }} />
            </div>
            <span className="analysis-progress-text">
              {sceneDescriptionMessage || `${descProgress}%`}
            </span>
          </div>
        )}

        {/* Error */}
        {descStatus === 'error' && sceneDescriptionMessage && (
          <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--danger-light)' }}>
            {sceneDescriptionMessage}
          </div>
        )}

        {/* Current scene at playhead */}
        {activeSegment && (
          <div style={{ marginTop: '6px', padding: '6px 8px', background: 'var(--accent-subtle)', borderRadius: '4px', borderLeft: '3px solid var(--accent)' }}>
            <div style={{ fontSize: '10px', color: 'var(--accent)', fontFamily: 'var(--font-mono)', marginBottom: '2px' }}>
              {formatTimestamp(activeSegment.start)} - {formatTimestamp(activeSegment.end)}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-primary)' }}>
              {activeSegment.text}
            </div>
          </div>
        )}

        {/* Scene list */}
        {segments.length > 0 && (
          <div style={{ marginTop: '6px', maxHeight: '200px', overflowY: 'auto' }}>
            {segments.map(seg => (
              <div
                key={seg.id}
                onClick={() => handleSeekToSegment(seg.start)}
                style={{
                  padding: '4px 6px',
                  cursor: 'pointer',
                  borderLeft: seg.id === activeSegment?.id ? '3px solid var(--accent)' : '3px solid transparent',
                  background: seg.id === activeSegment?.id ? 'var(--accent-subtle)' : 'transparent',
                  fontSize: '11px',
                  lineHeight: '1.4',
                  transition: 'background 0.15s',
                }}
              >
                <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '10px', marginRight: '6px' }}>
                  {formatTimestamp(seg.start)}
                </span>
                <span style={{ color: seg.id === activeSegment?.id ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                  {seg.text}
                </span>
              </div>
            ))}
          </div>
        )}

        {descStatus === 'none' && segments.length === 0 && (
          <div className="analysis-empty-state" style={{ marginTop: '4px', fontSize: '11px' }}>
            Uses local Ollama AI to describe video content.
          </div>
        )}
      </div>
    </div>
  );
}
