// Export Panel - embedded panel for frame-by-frame video export

import { useState, useEffect, useCallback } from 'react';
import { FrameExporter, downloadBlob } from '../engine/FrameExporter';
import type { ExportProgress } from '../engine/FrameExporter';
import { useTimelineStore } from '../stores/timelineStore';
import { useMediaStore } from '../stores/mediaStore';

export function ExportPanel() {
  const { duration, inPoint, outPoint } = useTimelineStore();
  const { getActiveComposition } = useMediaStore();
  const composition = getActiveComposition();

  // Export settings
  const [width, setWidth] = useState(composition?.width ?? 1920);
  const [height, setHeight] = useState(composition?.height ?? 1080);
  const [fps, setFps] = useState(composition?.frameRate ?? 30);
  const [bitrate, setBitrate] = useState(15_000_000);
  const [useInOut, setUseInOut] = useState(true);
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

  // Compute actual start/end based on In/Out markers
  const startTime = useInOut && inPoint !== null ? inPoint : 0;
  const endTime = useInOut && outPoint !== null ? outPoint : duration;

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
      }
    } catch (e) {
      console.error('[ExportPanel] Export failed:', e);
      setError(e instanceof Error ? e.message : 'Export failed');
    } finally {
      setIsExporting(false);
      setExporter(null);
    }
  }, [width, height, fps, bitrate, startTime, endTime, filename, isExporting]);

  // Handle cancel
  const handleCancel = useCallback(() => {
    if (exporter) {
      exporter.cancel();
    }
    setIsExporting(false);
    setExporter(null);
  }, [exporter]);

  // Format time as MM:SS.ff
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(2);
    return `${m}:${s.padStart(5, '0')}`;
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
      <div className="export-panel">
        <div className="panel-header">
          <h3>Export</h3>
        </div>
        <div className="export-error">
          WebCodecs is not supported in this browser.
          Please use Chrome 94+ or Safari 16.4+.
        </div>
      </div>
    );
  }

  return (
    <div className="export-panel">
      <div className="panel-header">
        <h3>Export</h3>
      </div>

      {!isExporting ? (
        <div className="export-form">
          {/* Filename */}
          <div className="control-row">
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
          <div className="control-row">
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
          <div className="control-row">
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

          {/* Quality */}
          <div className="control-row">
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

          {/* Use In/Out markers */}
          <div className="control-row">
            <label>
              <input
                type="checkbox"
                checked={useInOut}
                onChange={(e) => setUseInOut(e.target.checked)}
              />
              Use In/Out Markers
            </label>
          </div>

          {/* Time Range Display */}
          <div className="export-summary">
            <div>Range: {formatTime(startTime)} - {formatTime(endTime)}</div>
            <div>Duration: {formatTime(endTime - startTime)}</div>
            <div>Frames: {Math.ceil((endTime - startTime) * fps)}</div>
            <div>Est. Size: {estimatedSize()}</div>
          </div>

          {error && <div className="export-error">{error}</div>}

          <button
            className="btn export-start-btn"
            onClick={handleExport}
            disabled={endTime <= startTime}
          >
            Export Video
          </button>
        </div>
      ) : (
        <div className="export-progress-container">
          <div className="export-progress-bar">
            <div
              className="export-progress-fill"
              style={{ width: `${progress?.percent ?? 0}%` }}
            />
          </div>
          <div className="export-progress-info">
            <span>Frame {progress?.currentFrame ?? 0} / {progress?.totalFrames ?? 0}</span>
            <span>{(progress?.percent ?? 0).toFixed(1)}%</span>
          </div>
          {progress && progress.estimatedTimeRemaining > 0 && (
            <div className="export-eta">
              ETA: {formatTime(progress.estimatedTimeRemaining)}
            </div>
          )}
          <button className="btn export-cancel-btn" onClick={handleCancel}>
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
