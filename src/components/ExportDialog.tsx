// Export Dialog for frame-by-frame video export
// After Effects-style precise rendering

import { useState, useEffect, useCallback } from 'react';
import { FrameExporter, downloadBlob } from '../engine/FrameExporter';
import type { ExportProgress } from '../engine/FrameExporter';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';

interface ExportDialogProps {
  onClose: () => void;
}

export function ExportDialog({ onClose }: ExportDialogProps) {
  const { duration } = useTimelineStore();
  const { getActiveComposition } = useMediaStore();
  const composition = getActiveComposition();

  // Export settings
  const [width, setWidth] = useState(composition?.width ?? 1920);
  const [height, setHeight] = useState(composition?.height ?? 1080);
  const [fps, setFps] = useState(composition?.frameRate ?? 30);
  const [bitrate, setBitrate] = useState(15_000_000);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(duration);
  const [filename, setFilename] = useState('export');

  // Export state
  const [isExporting, setIsExporting] = useState(false);
  const [progress, setProgress] = useState<ExportProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exporter, setExporter] = useState<FrameExporter | null>(null);

  // Check WebCodecs support
  const [isSupported, setIsSupported] = useState(true);
  useEffect(() => {
    setIsSupported(FrameExporter.isSupported());
  }, []);

  // Update recommended bitrate when resolution changes
  useEffect(() => {
    setBitrate(FrameExporter.getRecommendedBitrate(width, height, fps));
  }, [width, height, fps]);

  // Handle resolution preset change
  const handleResolutionChange = useCallback((value: string) => {
    const [w, h] = value.split('x').map(Number);
    setWidth(w);
    setHeight(h);
  }, []);

  // Handle export
  const handleExport = useCallback(async () => {
    if (isExporting) return;

    setIsExporting(true);
    setError(null);
    setProgress(null);

    const exp = new FrameExporter({
      width,
      height,
      fps,
      codec: 'h264',
      bitrate,
      startTime,
      endTime,
    });
    setExporter(exp);

    try {
      const blob = await exp.export((p) => {
        setProgress(p);
      });

      if (blob) {
        downloadBlob(blob, `${filename}.mp4`);
        onClose();
      }
    } catch (e) {
      console.error('[ExportDialog] Export failed:', e);
      setError(e instanceof Error ? e.message : 'Export failed');
    } finally {
      setIsExporting(false);
      setExporter(null);
    }
  }, [width, height, fps, bitrate, startTime, endTime, filename, isExporting, onClose]);

  // Handle cancel
  const handleCancel = useCallback(() => {
    if (exporter) {
      exporter.cancel();
    }
    setIsExporting(false);
    setExporter(null);
  }, [exporter]);

  // Format time as MM:SS
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  // Format file size estimate
  const estimatedSize = () => {
    const durationSec = endTime - startTime;
    const bytes = (bitrate / 8) * durationSec;
    if (bytes > 1024 * 1024 * 1024) {
      return `~${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
    return `~${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  };

  if (!isSupported) {
    return (
      <div className="export-dialog-overlay" onClick={onClose}>
        <div className="export-dialog" onClick={(e) => e.stopPropagation()}>
          <h2>Export Video</h2>
          <div className="export-error">
            WebCodecs is not supported in this browser.
            Please use Chrome 94+ or Safari 16.4+.
          </div>
          <div className="export-actions">
            <button onClick={onClose}>Close</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="export-dialog-overlay" onClick={onClose}>
      <div className="export-dialog" onClick={(e) => e.stopPropagation()}>
        <h2>Export Video</h2>

        {!isExporting ? (
          <>
            <div className="export-form">
              {/* Filename */}
              <div className="export-row">
                <label>Filename</label>
                <div className="export-input-group">
                  <input
                    type="text"
                    value={filename}
                    onChange={(e) => setFilename(e.target.value)}
                    placeholder="export"
                  />
                  <span className="export-extension">.mp4</span>
                </div>
              </div>

              {/* Resolution */}
              <div className="export-row">
                <label>Resolution</label>
                <select
                  value={`${width}x${height}`}
                  onChange={(e) => handleResolutionChange(e.target.value)}
                >
                  {FrameExporter.getPresetResolutions().map(({ label, width: w, height: h }) => (
                    <option key={`${w}x${h}`} value={`${w}x${h}`}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Frame Rate */}
              <div className="export-row">
                <label>Frame Rate</label>
                <select
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                >
                  {FrameExporter.getPresetFrameRates().map(({ label, fps: f }) => (
                    <option key={f} value={f}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Bitrate */}
              <div className="export-row">
                <label>Quality</label>
                <select
                  value={bitrate}
                  onChange={(e) => setBitrate(Number(e.target.value))}
                >
                  <option value={5_000_000}>Low (5 Mbps)</option>
                  <option value={10_000_000}>Medium (10 Mbps)</option>
                  <option value={15_000_000}>High (15 Mbps)</option>
                  <option value={25_000_000}>Very High (25 Mbps)</option>
                  <option value={35_000_000}>Maximum (35 Mbps)</option>
                </select>
              </div>

              {/* Time Range */}
              <div className="export-row">
                <label>Time Range</label>
                <div className="export-time-range">
                  <input
                    type="number"
                    value={startTime.toFixed(2)}
                    onChange={(e) => setStartTime(Math.max(0, Number(e.target.value)))}
                    step="0.1"
                    min="0"
                    max={endTime}
                  />
                  <span>to</span>
                  <input
                    type="number"
                    value={endTime.toFixed(2)}
                    onChange={(e) => setEndTime(Math.min(duration, Number(e.target.value)))}
                    step="0.1"
                    min={startTime}
                    max={duration}
                  />
                  <span>sec</span>
                </div>
              </div>

              {/* Summary */}
              <div className="export-summary">
                <div>Duration: {formatTime(endTime - startTime)}</div>
                <div>Total Frames: {Math.ceil((endTime - startTime) * fps)}</div>
                <div>Estimated Size: {estimatedSize()}</div>
              </div>
            </div>

            {error && <div className="export-error">{error}</div>}

            <div className="export-actions">
              <button className="export-cancel" onClick={onClose}>
                Cancel
              </button>
              <button className="export-start" onClick={handleExport}>
                Export
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="export-progress">
              <div className="export-progress-bar">
                <div
                  className="export-progress-fill"
                  style={{ width: `${progress?.percent ?? 0}%` }}
                />
              </div>
              <div className="export-progress-info">
                <span>
                  Frame {progress?.currentFrame ?? 0} / {progress?.totalFrames ?? 0}
                </span>
                <span>{(progress?.percent ?? 0).toFixed(1)}%</span>
              </div>
              {progress && progress.estimatedTimeRemaining > 0 && (
                <div className="export-eta">
                  ETA: {formatTime(progress.estimatedTimeRemaining)}
                </div>
              )}
            </div>

            <div className="export-actions">
              <button className="export-cancel" onClick={handleCancel}>
                Cancel Export
              </button>
            </div>
          </>
        )}
      </div>

      <style>{`
        .export-dialog-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 10000;
        }

        .export-dialog {
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 24px;
          min-width: 400px;
          max-width: 500px;
        }

        .export-dialog h2 {
          margin: 0 0 20px 0;
          font-size: 18px;
          color: var(--text-primary);
        }

        .export-form {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .export-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .export-row label {
          width: 100px;
          color: var(--text-secondary);
          font-size: 13px;
        }

        .export-row select,
        .export-row input[type="text"],
        .export-row input[type="number"] {
          flex: 1;
          padding: 8px 12px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          color: var(--text-primary);
          font-size: 14px;
        }

        .export-row select:focus,
        .export-row input:focus {
          outline: none;
          border-color: var(--accent);
        }

        .export-input-group {
          flex: 1;
          display: flex;
          align-items: center;
        }

        .export-input-group input {
          border-radius: 4px 0 0 4px;
        }

        .export-extension {
          padding: 8px 12px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-color);
          border-left: none;
          border-radius: 0 4px 4px 0;
          color: var(--text-secondary);
        }

        .export-time-range {
          flex: 1;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .export-time-range input {
          width: 80px;
          flex: none;
        }

        .export-time-range span {
          color: var(--text-secondary);
          font-size: 13px;
        }

        .export-summary {
          margin-top: 8px;
          padding: 12px;
          background: var(--bg-tertiary);
          border-radius: 4px;
          display: flex;
          flex-direction: column;
          gap: 4px;
          font-size: 13px;
          color: var(--text-secondary);
        }

        .export-error {
          margin-top: 16px;
          padding: 12px;
          background: rgba(255, 68, 68, 0.1);
          border: 1px solid var(--danger);
          border-radius: 4px;
          color: var(--danger);
          font-size: 13px;
        }

        .export-actions {
          display: flex;
          justify-content: flex-end;
          gap: 12px;
          margin-top: 24px;
        }

        .export-actions button {
          padding: 10px 20px;
          border-radius: 4px;
          border: none;
          font-size: 14px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .export-cancel {
          background: var(--bg-tertiary);
          color: var(--text-primary);
        }

        .export-cancel:hover {
          background: var(--bg-hover);
        }

        .export-start {
          background: var(--accent);
          color: var(--bg-primary);
          font-weight: 600;
        }

        .export-start:hover {
          background: var(--accent-hover);
        }

        .export-progress {
          margin: 20px 0;
        }

        .export-progress-bar {
          height: 8px;
          background: var(--bg-tertiary);
          border-radius: 4px;
          overflow: hidden;
        }

        .export-progress-fill {
          height: 100%;
          background: var(--accent);
          transition: width 0.1s ease-out;
        }

        .export-progress-info {
          display: flex;
          justify-content: space-between;
          margin-top: 8px;
          font-size: 13px;
          color: var(--text-secondary);
        }

        .export-eta {
          text-align: center;
          margin-top: 8px;
          font-size: 14px;
          color: var(--text-primary);
        }
      `}</style>
    </div>
  );
}
