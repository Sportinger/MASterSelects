// Export Dialog for frame-by-frame video export
// After Effects-style precise rendering

import { useState, useEffect, useCallback } from 'react';
import { FrameExporter, downloadBlob } from '../../engine/FrameExporter';
import type { ExportProgress, VideoCodec, ContainerFormat } from '../../engine/FrameExporter';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';

interface ExportDialogProps {
  onClose: () => void;
}

export function ExportDialog({ onClose }: ExportDialogProps) {
  const { duration, startExport, setExportProgress, endExport } = useTimelineStore();
  const { getActiveComposition } = useMediaStore();
  const composition = getActiveComposition();

  // Export settings
  const [width, setWidth] = useState(composition?.width ?? 1920);
  const [height, setHeight] = useState(composition?.height ?? 1080);
  const [fps, setFps] = useState(composition?.frameRate ?? 30);
  const [container, setContainer] = useState<ContainerFormat>('mp4');
  const [codec, setCodec] = useState<VideoCodec>('h264');
  const [bitrate, setBitrate] = useState(15_000_000);
  const [useCustomBitrate, setUseCustomBitrate] = useState(false);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(duration);
  const [filename, setFilename] = useState('export');
  const [codecSupport, setCodecSupport] = useState<Record<VideoCodec, boolean>>({
    h264: true, h265: false, vp9: true, av1: false
  });

  // Audio settings
  const [includeAudio, setIncludeAudio] = useState(true);
  const [audioSampleRate, setAudioSampleRate] = useState<44100 | 48000>(48000);
  const [audioBitrate, setAudioBitrate] = useState(256000);
  const [normalizeAudio, setNormalizeAudio] = useState(false);

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

  // Check codec support when resolution changes
  useEffect(() => {
    const checkSupport = async () => {
      const support: Record<VideoCodec, boolean> = {
        h264: await FrameExporter.checkCodecSupport('h264', width, height),
        h265: await FrameExporter.checkCodecSupport('h265', width, height),
        vp9: await FrameExporter.checkCodecSupport('vp9', width, height),
        av1: await FrameExporter.checkCodecSupport('av1', width, height),
      };
      setCodecSupport(support);
    };
    checkSupport();
  }, [width, height]);

  // Update recommended bitrate when resolution changes (only if not using custom)
  useEffect(() => {
    if (!useCustomBitrate) {
      setBitrate(FrameExporter.getRecommendedBitrate(width, height, fps));
    }
  }, [width, height, fps, useCustomBitrate]);

  // Handle container change - reset codec if incompatible
  useEffect(() => {
    const availableCodecs = FrameExporter.getVideoCodecs(container);
    if (!availableCodecs.find(c => c.id === codec)) {
      setCodec(availableCodecs[0].id);
    }
  }, [container, codec]);

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

    const containerInfo = FrameExporter.getContainerFormats().find(c => c.id === container);
    const extension = containerInfo?.extension ?? '.mp4';

    const exp = new FrameExporter({
      width,
      height,
      fps,
      codec,
      container,
      bitrate,
      startTime,
      endTime,
      // Audio settings
      includeAudio,
      audioSampleRate,
      audioBitrate,
      normalizeAudio,
    });
    setExporter(exp);

    // Start export progress in timeline
    startExport(startTime, endTime);

    try {
      const blob = await exp.export((p) => {
        setProgress(p);
        // Update timeline export progress
        setExportProgress(p.percent, p.currentTime);
      });

      if (blob) {
        downloadBlob(blob, `${filename}${extension}`);
        onClose();
      }
    } catch (e) {
      console.error('[ExportDialog] Export failed:', e);
      setError(e instanceof Error ? e.message : 'Export failed');
    } finally {
      setIsExporting(false);
      setExporter(null);
      // End export progress in timeline
      endExport();
    }
  }, [width, height, fps, codec, container, bitrate, startTime, endTime, filename, isExporting, onClose, includeAudio, audioSampleRate, audioBitrate, normalizeAudio, startExport, setExportProgress, endExport]);

  // Handle cancel
  const handleCancel = useCallback(() => {
    if (exporter) {
      exporter.cancel();
    }
    setIsExporting(false);
    setExporter(null);
    // End export progress in timeline
    endExport();
  }, [exporter, endExport]);

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
                  <span className="export-extension">
                    {FrameExporter.getContainerFormats().find(c => c.id === container)?.extension ?? '.mp4'}
                  </span>
                </div>
              </div>

              {/* Container Format */}
              <div className="export-row">
                <label>Container</label>
                <select
                  value={container}
                  onChange={(e) => setContainer(e.target.value as ContainerFormat)}
                >
                  {FrameExporter.getContainerFormats().map(({ id, label }) => (
                    <option key={id} value={id}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Video Codec */}
              <div className="export-row">
                <label>Codec</label>
                <select
                  value={codec}
                  onChange={(e) => setCodec(e.target.value as VideoCodec)}
                >
                  {FrameExporter.getVideoCodecs(container).map(({ id, label, description }) => (
                    <option
                      key={id}
                      value={id}
                      disabled={!codecSupport[id]}
                      title={description}
                    >
                      {label} {!codecSupport[id] ? '(not supported)' : ''}
                    </option>
                  ))}
                </select>
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
                <div className="export-bitrate-group">
                  {!useCustomBitrate ? (
                    <select
                      value={bitrate}
                      onChange={(e) => setBitrate(Number(e.target.value))}
                    >
                      <option value={5_000_000}>Low (5 Mbps)</option>
                      <option value={10_000_000}>Medium (10 Mbps)</option>
                      <option value={15_000_000}>High (15 Mbps)</option>
                      <option value={25_000_000}>Very High (25 Mbps)</option>
                      <option value={35_000_000}>Maximum (35 Mbps)</option>
                      <option value={50_000_000}>Ultra (50 Mbps)</option>
                    </select>
                  ) : (
                    <div className="export-custom-bitrate">
                      <input
                        type="range"
                        min={FrameExporter.getBitrateRange().min}
                        max={FrameExporter.getBitrateRange().max}
                        step={FrameExporter.getBitrateRange().step}
                        value={bitrate}
                        onChange={(e) => setBitrate(Number(e.target.value))}
                      />
                      <span className="bitrate-value">{FrameExporter.formatBitrate(bitrate)}</span>
                    </div>
                  )}
                  <button
                    type="button"
                    className="bitrate-toggle"
                    onClick={() => setUseCustomBitrate(!useCustomBitrate)}
                    title={useCustomBitrate ? 'Use presets' : 'Custom bitrate'}
                  >
                    {useCustomBitrate ? 'Presets' : 'Custom'}
                  </button>
                </div>
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

              {/* Audio Section */}
              <div className="export-section-header">Audio</div>

              {/* Include Audio */}
              <div className="export-row">
                <label>Include Audio</label>
                <div className="export-checkbox-group">
                  <input
                    type="checkbox"
                    id="includeAudio"
                    checked={includeAudio}
                    onChange={(e) => setIncludeAudio(e.target.checked)}
                  />
                  <label htmlFor="includeAudio" className="checkbox-label">
                    Export audio tracks (AAC)
                  </label>
                </div>
              </div>

              {includeAudio && (
                <>
                  {/* Audio Sample Rate */}
                  <div className="export-row">
                    <label>Sample Rate</label>
                    <select
                      value={audioSampleRate}
                      onChange={(e) => setAudioSampleRate(Number(e.target.value) as 44100 | 48000)}
                    >
                      <option value={48000}>48 kHz (Video Standard)</option>
                      <option value={44100}>44.1 kHz (CD Quality)</option>
                    </select>
                  </div>

                  {/* Audio Bitrate */}
                  <div className="export-row">
                    <label>Audio Quality</label>
                    <select
                      value={audioBitrate}
                      onChange={(e) => setAudioBitrate(Number(e.target.value))}
                    >
                      <option value={128000}>128 kbps (Good)</option>
                      <option value={192000}>192 kbps (Better)</option>
                      <option value={256000}>256 kbps (High Quality)</option>
                      <option value={320000}>320 kbps (Maximum)</option>
                    </select>
                  </div>

                  {/* Normalize Audio */}
                  <div className="export-row">
                    <label>Normalize</label>
                    <div className="export-checkbox-group">
                      <input
                        type="checkbox"
                        id="normalizeAudio"
                        checked={normalizeAudio}
                        onChange={(e) => setNormalizeAudio(e.target.checked)}
                      />
                      <label htmlFor="normalizeAudio" className="checkbox-label">
                        Peak normalize (prevent clipping)
                      </label>
                    </div>
                  </div>
                </>
              )}

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
              {/* Phase indicator */}
              <div className="export-phase">
                {progress?.phase === 'video' && 'Encoding video frames...'}
                {progress?.phase === 'audio' && (
                  <>
                    Processing audio: {progress.audioPhase}
                    {progress.audioPhase && ` (${progress.audioPercent}%)`}
                  </>
                )}
                {progress?.phase === 'muxing' && 'Finalizing...'}
              </div>

              <div className="export-progress-bar">
                <div
                  className="export-progress-fill"
                  style={{ width: `${progress?.percent ?? 0}%` }}
                />
              </div>
              <div className="export-progress-info">
                {progress?.phase === 'video' ? (
                  <span>
                    Frame {progress?.currentFrame ?? 0} / {progress?.totalFrames ?? 0}
                  </span>
                ) : (
                  <span>Audio processing</span>
                )}
                <span>{(progress?.percent ?? 0).toFixed(1)}%</span>
              </div>
              {progress && progress.phase === 'video' && progress.estimatedTimeRemaining > 0 && (
                <div className="export-eta">
                  ETA: {formatTime(progress.estimatedTimeRemaining)}
                </div>
              )}

              {/* Warning for slow clips */}
              {progress?.slowClipsCount && progress.slowClipsCount > 0 && (
                <div className="export-warning">
                  {progress.slowClipsMessage}
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
        }

        .export-bitrate-group {
          flex: 1;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .export-bitrate-group select {
          flex: 1;
        }

        .export-custom-bitrate {
          flex: 1;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .export-custom-bitrate input[type="range"] {
          flex: 1;
          height: 4px;
          background: var(--bg-tertiary);
          border-radius: 2px;
          cursor: pointer;
          accent-color: var(--accent);
        }

        .bitrate-value {
          min-width: 70px;
          text-align: right;
          color: var(--text-primary);
          font-size: 13px;
          font-family: monospace;
        }

        .bitrate-toggle {
          padding: 6px 10px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          color: var(--text-secondary);
          font-size: 12px;
          cursor: pointer;
          transition: all 0.15s;
          white-space: nowrap;
        }

        .bitrate-toggle:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
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

        .export-section-header {
          margin-top: 16px;
          margin-bottom: 4px;
          font-size: 13px;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .export-checkbox-group {
          flex: 1;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .export-checkbox-group input[type="checkbox"] {
          width: 16px;
          height: 16px;
          accent-color: var(--accent);
          cursor: pointer;
        }

        .export-checkbox-group .checkbox-label {
          width: auto;
          color: var(--text-primary);
          font-size: 13px;
          cursor: pointer;
        }

        .export-phase {
          margin-bottom: 12px;
          font-size: 14px;
          color: var(--text-primary);
          font-weight: 500;
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

        .export-warning {
          margin-top: 12px;
          padding: 10px 12px;
          background: rgba(255, 170, 0, 0.1);
          border: 1px solid #ffaa00;
          border-radius: 4px;
          color: #ffaa00;
          font-size: 12px;
        }
      `}</style>
    </div>
  );
}
