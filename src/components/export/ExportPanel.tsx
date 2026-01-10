// Export Panel - embedded panel for frame-by-frame video export

import { useState, useEffect, useCallback } from 'react';
import { FrameExporter, downloadBlob } from '../../engine/FrameExporter';
import type { ExportProgress } from '../../engine/FrameExporter';
import { AudioExportPipeline } from '../../engine/audio';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { engine } from '../../engine/WebGPUEngine';

export function ExportPanel() {
  const { duration, inPoint, outPoint, playheadPosition } = useTimelineStore();
  const { getActiveComposition } = useMediaStore();
  const composition = getActiveComposition();

  // Export settings
  const [width, setWidth] = useState(composition?.width ?? 1920);
  const [height, setHeight] = useState(composition?.height ?? 1080);
  const [customWidth, setCustomWidth] = useState(composition?.width ?? 1920);
  const [customHeight, setCustomHeight] = useState(composition?.height ?? 1080);
  const [useCustomResolution, setUseCustomResolution] = useState(false);
  const [fps, setFps] = useState(composition?.frameRate ?? 30);
  const [bitrate, setBitrate] = useState(15_000_000);
  const [useInOut, setUseInOut] = useState(true);
  const [filename, setFilename] = useState('export');

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

    const actualWidth = useCustomResolution ? customWidth : width;
    const actualHeight = useCustomResolution ? customHeight : height;

    const exp = new FrameExporter({
      width: actualWidth,
      height: actualHeight,
      fps,
      codec: 'h264',
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
  }, [width, height, customWidth, customHeight, useCustomResolution, fps, bitrate, startTime, endTime, filename, isExporting, includeAudio, audioSampleRate, audioBitrate, normalizeAudio]);

  // Handle cancel
  const handleCancel = useCallback(() => {
    if (exporter) {
      exporter.cancel();
    }
    setIsExporting(false);
    setExporter(null);
  }, [exporter]);

  // Handle audio-only export
  const handleExportAudioOnly = useCallback(async () => {
    if (isExporting) return;

    setIsExporting(true);
    setError(null);
    setProgress({
      phase: 'audio',
      currentFrame: 0,
      totalFrames: 0,
      percent: 0,
      estimatedTimeRemaining: 0,
      currentTime: startTime,
      audioPhase: 'extracting',
      audioPercent: 0,
    });

    const audioPipeline = new AudioExportPipeline({
      sampleRate: audioSampleRate,
      bitrate: audioBitrate,
      normalize: normalizeAudio,
    });

    try {
      const audioResult = await audioPipeline.exportAudio(
        startTime,
        endTime,
        (audioProgress) => {
          setProgress({
            phase: 'audio',
            currentFrame: 0,
            totalFrames: 0,
            percent: audioProgress.percent,
            estimatedTimeRemaining: 0,
            currentTime: endTime,
            audioPhase: audioProgress.phase,
            audioPercent: audioProgress.percent,
          });
        }
      );

      if (audioResult && audioResult.chunks.length > 0) {
        // Convert AAC chunks to a downloadable file
        // For standalone audio, we need to create an audio container
        const audioBlobs: Blob[] = [];
        for (const chunk of audioResult.chunks) {
          const buffer = new ArrayBuffer(chunk.byteLength);
          chunk.copyTo(buffer);
          audioBlobs.push(new Blob([buffer]));
        }
        const audioBlob = new Blob(audioBlobs, { type: 'audio/aac' });
        downloadBlob(audioBlob, `${filename}.aac`);
      } else {
        setError('No audio clips found in the selected range');
      }
    } catch (e) {
      console.error('[ExportPanel] Audio export failed:', e);
      setError(e instanceof Error ? e.message : 'Audio export failed');
    } finally {
      setIsExporting(false);
    }
  }, [startTime, endTime, filename, isExporting, audioSampleRate, audioBitrate, normalizeAudio]);

  // Handle render current frame
  const handleRenderFrame = useCallback(async () => {
    try {
      // Read pixels from the engine's composited frame
      const pixels = await engine.readPixels();
      if (!pixels) {
        setError('Failed to read frame from GPU');
        return;
      }

      // Get the engine's output dimensions (this is what was actually rendered)
      const { width: engineWidth, height: engineHeight } = engine.getOutputDimensions();

      // Create ImageData from the pixels
      const imageData = new ImageData(pixels, engineWidth, engineHeight);

      // Create a canvas to draw the image
      const canvas = document.createElement('canvas');
      canvas.width = engineWidth;
      canvas.height = engineHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        setError('Failed to create canvas context');
        return;
      }

      // Draw the image data
      ctx.putImageData(imageData, 0, 0);

      // If custom resolution is different, scale to target size
      const actualWidth = useCustomResolution ? customWidth : width;
      const actualHeight = useCustomResolution ? customHeight : height;

      if (actualWidth !== engineWidth || actualHeight !== engineHeight) {
        // Create a scaled canvas
        const scaledCanvas = document.createElement('canvas');
        scaledCanvas.width = actualWidth;
        scaledCanvas.height = actualHeight;
        const scaledCtx = scaledCanvas.getContext('2d');
        if (scaledCtx) {
          scaledCtx.drawImage(canvas, 0, 0, actualWidth, actualHeight);
          scaledCanvas.toBlob((blob) => {
            if (blob) {
              const frameName = `${filename}_frame_${Math.floor(playheadPosition * 1000)}.png`;
              downloadBlob(blob, frameName);
            }
          }, 'image/png');
        }
      } else {
        // Export at native resolution
        canvas.toBlob((blob) => {
          if (blob) {
            const frameName = `${filename}_frame_${Math.floor(playheadPosition * 1000)}.png`;
            downloadBlob(blob, frameName);
          }
        }, 'image/png');
      }
    } catch (e) {
      console.error('[ExportPanel] Frame render failed:', e);
      setError(e instanceof Error ? e.message : 'Frame render failed');
    }
  }, [width, height, customWidth, customHeight, useCustomResolution, filename, playheadPosition]);

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
      {/* Action Buttons - Always visible at top */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', padding: '12px 12px 0' }}>
        <button
          className="btn"
          onClick={handleRenderFrame}
          style={{ flex: 1 }}
          disabled={isExporting}
        >
          Render Frame
        </button>
        <button
          className="btn export-start-btn"
          onClick={handleExport}
          disabled={isExporting || endTime <= startTime}
          style={{ flex: 1 }}
        >
          Export Video
        </button>
        <button
          className="btn"
          onClick={handleExportAudioOnly}
          disabled={isExporting || endTime <= startTime || !includeAudio}
          style={{ flex: 1 }}
          title={!includeAudio ? 'Enable "Include Audio" first' : ''}
        >
          Export Audio
        </button>
      </div>

      {!isExporting ? (
        <div className="export-form">
          {/* Export Settings */}
          <div className="export-section">
            <div className="export-section-header">Video</div>

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
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <select
                value={useCustomResolution ? 'custom' : `${width}x${height}`}
                onChange={(e) => {
                  if (e.target.value === 'custom') {
                    setUseCustomResolution(true);
                  } else {
                    setUseCustomResolution(false);
                    handleResolutionChange(e.target.value);
                  }
                }}
                disabled={useCustomResolution}
                style={{ flex: 1 }}
              >
                {FrameExporter.getPresetResolutions().map(({ label, width: w, height: h }) => (
                  <option key={`${w}x${h}`} value={`${w}x${h}`}>
                    {label}
                  </option>
                ))}
                <option value="custom">Custom...</option>
              </select>
              <label style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <input
                  type="checkbox"
                  checked={useCustomResolution}
                  onChange={(e) => setUseCustomResolution(e.target.checked)}
                />
                Custom
              </label>
            </div>
          </div>

          {/* Custom Resolution Inputs */}
          {useCustomResolution && (
            <div className="control-row">
              <label>Custom Size</label>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <input
                  type="number"
                  value={customWidth}
                  onChange={(e) => setCustomWidth(Math.max(1, parseInt(e.target.value) || 1920))}
                  placeholder="Width"
                  min="1"
                  max="7680"
                  style={{ flex: 1 }}
                />
                <span>×</span>
                <input
                  type="number"
                  value={customHeight}
                  onChange={(e) => setCustomHeight(Math.max(1, parseInt(e.target.value) || 1080))}
                  placeholder="Height"
                  min="1"
                  max="4320"
                  style={{ flex: 1 }}
                />
              </div>
            </div>
          )}

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
          </div>

          {/* Audio Settings */}
          <div className="export-section">
            <div className="export-section-header">Audio</div>

            {/* Include Audio */}
            <div className="control-row">
              <label>
                <input
                  type="checkbox"
                  checked={includeAudio}
                  onChange={(e) => setIncludeAudio(e.target.checked)}
                />
                Include Audio (AAC)
              </label>
            </div>

            {includeAudio && (
              <>
                {/* Sample Rate */}
                <div className="control-row">
                  <label>Sample Rate</label>
                  <select
                    value={audioSampleRate}
                    onChange={(e) => setAudioSampleRate(Number(e.target.value) as 44100 | 48000)}
                  >
                    <option value={48000}>48 kHz (Video)</option>
                    <option value={44100}>44.1 kHz (CD)</option>
                  </select>
                </div>

                {/* Audio Quality */}
                <div className="control-row">
                  <label>Audio Quality</label>
                  <select
                    value={audioBitrate}
                    onChange={(e) => setAudioBitrate(Number(e.target.value))}
                  >
                    <option value={128000}>128 kbps</option>
                    <option value={192000}>192 kbps</option>
                    <option value={256000}>256 kbps (High)</option>
                    <option value={320000}>320 kbps (Max)</option>
                  </select>
                </div>

                {/* Normalize */}
                <div className="control-row">
                  <label>
                    <input
                      type="checkbox"
                      checked={normalizeAudio}
                      onChange={(e) => setNormalizeAudio(e.target.checked)}
                    />
                    Normalize (prevent clipping)
                  </label>
                </div>
              </>
            )}
          </div>

          {/* Range Settings */}
          <div className="export-section">
            <div className="export-section-header">Range</div>

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
          </div>

          {error && <div className="export-error">{error}</div>}
        </div>
      ) : (
        <div className="export-progress-container">
          {/* Phase indicator */}
          <div style={{ marginBottom: '12px', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>
            {progress?.phase === 'video' && 'Encoding video frames...'}
            {progress?.phase === 'audio' && (
              <>Processing audio: {progress.audioPhase} ({progress.audioPercent}%)</>
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
              <span>Frame {progress?.currentFrame ?? 0} / {progress?.totalFrames ?? 0}</span>
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
          <button className="btn export-cancel-btn" onClick={handleCancel}>
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
