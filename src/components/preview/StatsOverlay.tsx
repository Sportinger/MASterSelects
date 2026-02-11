// Detailed stats overlay component — extracted from Preview.tsx

import { useMemo } from 'react';
import type { EngineStats } from '../../types';

interface StatsOverlayProps {
  stats: EngineStats;
  resolution: { width: number; height: number };
  expanded: boolean;
  onToggle: () => void;
}

export function StatsOverlay({ stats, resolution, expanded, onToggle }: StatsOverlayProps) {
  const fpsColor = stats.fps >= 55 ? '#4f4' : stats.fps >= 30 ? '#ff4' : '#f44';
  const dropColor = stats.drops.lastSecond > 0 ? '#f44' : '#4f4';
  const decoderColor = stats.decoder === 'NativeHelper' ? '#4af' : stats.decoder === 'WebCodecs' ? '#4f4' : stats.decoder === 'ParallelDecode' ? '#a4f' : stats.decoder.startsWith('HTMLVideo') ? '#fa4' : '#888';
  // Render time color: green < 10ms, yellow < 16.67ms (60fps target), red >= 16.67ms
  const renderTime = stats.timing.total;
  const renderTimeColor = renderTime < 10 ? '#4f4' : renderTime < 16.67 ? '#ff4' : '#f44';

  // Determine bottleneck
  const bottleneck = useMemo(() => {
    const { timing } = stats;
    if (timing.total < 10) return null;
    if (timing.importTexture > timing.renderPass && timing.importTexture > timing.submit) {
      return 'Video Import';
    }
    if (timing.renderPass > timing.submit) {
      return 'GPU Render';
    }
    return 'GPU Submit';
  }, [stats.timing]);

  if (!expanded) {
    return (
      <div
        className="preview-stats preview-stats-compact"
        onClick={onToggle}
        title="Click for detailed stats"
      >
        <span style={{ color: fpsColor, fontWeight: 'bold' }}>{stats.fps}</span>
        <span style={{ opacity: 0.7 }}> FPS</span>
        {!stats.isIdle && renderTime > 0 && (
          <span style={{ color: renderTimeColor, marginLeft: 6, fontSize: 10 }}>
            {renderTime.toFixed(1)}ms
          </span>
        )}
        {stats.isIdle && (
          <span style={{ color: '#888', marginLeft: 6, fontSize: 9 }}>[IDLE]</span>
        )}
        {stats.decoder !== 'none' && !stats.isIdle && (
          <span style={{ color: decoderColor, marginLeft: 6, fontSize: 9 }}>[{stats.decoder === 'WebCodecs' ? 'WC' : stats.decoder === 'NativeHelper' ? 'NH' : stats.decoder === 'ParallelDecode' ? 'PD' : 'HTML'}]</span>
        )}
        {stats.drops.lastSecond > 0 && (
          <span style={{ color: '#f44', marginLeft: 6 }}>▼{stats.drops.lastSecond}</span>
        )}
        {stats.audio?.status && stats.audio.status !== 'silent' && (
          <span style={{
            marginLeft: 6,
            color: stats.audio.status === 'sync' ? '#4f4'
              : stats.audio.status === 'drift' ? '#ff4'
              : '#f44'
          }}>
            🔊{stats.audio.status === 'drift' ? `(${stats.audio.drift}ms)` : ''}
          </span>
        )}
        <span style={{ opacity: 0.5, marginLeft: 8 }}>
          {resolution.width}×{resolution.height}
        </span>
      </div>
    );
  }

  return (
    <div className="preview-stats preview-stats-expanded" onClick={onToggle}>
      <div className="stats-header">
        <span style={{ color: fpsColor, fontWeight: 'bold', fontSize: 18 }}>{stats.fps}</span>
        <span style={{ opacity: 0.7 }}> / {stats.targetFps} FPS</span>
        {stats.isIdle && (
          <span style={{ color: '#888', marginLeft: 8, fontSize: 11 }}>[IDLE]</span>
        )}
        <span style={{ opacity: 0.5, marginLeft: 8, fontSize: 11 }}>
          {resolution.width}×{resolution.height}
        </span>
      </div>

      <div className="stats-section">
        <div className="stats-row">
          <span>Frame Gap</span>
          <span style={{ color: stats.timing.rafGap > 20 ? '#ff4' : '#aaa' }}>
            {stats.timing.rafGap.toFixed(1)}ms
          </span>
        </div>
        <div className="stats-row">
          <span>Render Total</span>
          <span style={{ color: stats.timing.total > 12 ? '#ff4' : '#aaa' }}>
            {stats.timing.total.toFixed(2)}ms
          </span>
        </div>
      </div>

      <div className="stats-section">
        <div className="stats-label">Pipeline Breakdown</div>
        <div className="stats-bar-container">
          <div
            className="stats-bar stats-bar-import"
            style={{ width: `${Math.min(100, (stats.timing.importTexture / 16.67) * 100)}%` }}
            title={`Import: ${stats.timing.importTexture.toFixed(2)}ms`}
          />
          <div
            className="stats-bar stats-bar-render"
            style={{ width: `${Math.min(100, (stats.timing.renderPass / 16.67) * 100)}%` }}
            title={`Render: ${stats.timing.renderPass.toFixed(2)}ms`}
          />
          <div
            className="stats-bar stats-bar-submit"
            style={{ width: `${Math.min(100, (stats.timing.submit / 16.67) * 100)}%` }}
            title={`Submit: ${stats.timing.submit.toFixed(2)}ms`}
          />
        </div>
        <div className="stats-row" style={{ fontSize: 10, opacity: 0.6 }}>
          <span>Import: {stats.timing.importTexture.toFixed(2)}ms</span>
          <span>Render: {stats.timing.renderPass.toFixed(2)}ms</span>
          <span>Submit: {stats.timing.submit.toFixed(2)}ms</span>
        </div>
      </div>

      <div className="stats-section">
        <div className="stats-row">
          <span>Engine</span>
          <span style={{ color: stats.isIdle ? '#888' : '#4f4' }}>
            {stats.isIdle ? '● Idle (saving power)' : '● Active'}
          </span>
        </div>
        <div className="stats-row">
          <span>Layers</span>
          <span>{stats.layerCount}</span>
        </div>
        <div className="stats-row">
          <span>Decoder</span>
          <span style={{ color: decoderColor }}>{stats.decoder}</span>
        </div>
        <div className="stats-row">
          <span style={{ color: dropColor }}>Drops (last sec)</span>
          <span style={{ color: dropColor }}>{stats.drops.lastSecond}</span>
        </div>
        <div className="stats-row">
          <span>Drops (total)</span>
          <span>{stats.drops.count}</span>
        </div>
        {stats.drops.reason !== 'none' && (
          <div className="stats-row">
            <span>Last Drop Reason</span>
            <span style={{ color: '#f44' }}>{stats.drops.reason.replace('_', ' ')}</span>
          </div>
        )}
        {bottleneck && (
          <div className="stats-row">
            <span>Bottleneck</span>
            <span style={{ color: '#ff4' }}>{bottleneck}</span>
          </div>
        )}
      </div>

      {/* Audio Status Section */}
      {stats.audio && (
        <div className="stats-section">
          <div className="stats-label">Audio</div>
          <div className="stats-row">
            <span>Status</span>
            <span style={{
              color: stats.audio.status === 'sync' ? '#4f4'
                : stats.audio.status === 'drift' ? '#ff4'
                : stats.audio.status === 'error' ? '#f44'
                : '#888'
            }}>
              {stats.audio.status === 'sync' ? '● Sync'
                : stats.audio.status === 'drift' ? '◐ Drift'
                : stats.audio.status === 'error' ? '✕ Error'
                : '○ Silent'}
            </span>
          </div>
          {stats.audio.playing > 0 && (
            <div className="stats-row">
              <span>Playing</span>
              <span>{stats.audio.playing} track{stats.audio.playing !== 1 ? 's' : ''}</span>
            </div>
          )}
          {stats.audio.drift > 0 && (
            <div className="stats-row">
              <span>Drift</span>
              <span style={{ color: stats.audio.drift > 100 ? '#f44' : stats.audio.drift > 50 ? '#ff4' : '#aaa' }}>
                {stats.audio.drift}ms
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
