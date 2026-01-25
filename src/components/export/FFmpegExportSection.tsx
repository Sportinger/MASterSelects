// src/components/export/FFmpegExportSection.tsx
// FFmpeg WASM export section with professional codec support

import { useState, useCallback } from 'react';
import { Logger } from '../../services/logger';

const log = Logger.create('FFmpegExportSection');
import {
  getFFmpegBridge,
  FFmpegBridge,
  PRORES_PROFILES,
  DNXHR_PROFILES,
  CONTAINER_FORMATS,
  PLATFORM_PRESETS,
  getCodecInfo,
} from '../../engine/ffmpeg';
import { CodecSelector } from './CodecSelector';
import type {
  FFmpegExportSettings,
  FFmpegProgress,
  FFmpegVideoCodec,
  FFmpegContainer,
  ProResProfile,
  DnxhrProfile,
} from '../../engine/ffmpeg';

interface FFmpegExportSectionProps {
  width: number;
  height: number;
  fps: number;
  startTime: number;
  endTime: number;
  filename: string;
  onRenderFrames: (settings: FFmpegExportSettings) => Promise<Uint8Array[]>;
}

export function FFmpegExportSection({
  width,
  height,
  fps,
  startTime,
  endTime,
  filename,
  onRenderFrames,
}: FFmpegExportSectionProps) {
  // FFmpeg loading state
  const [isLoading, setIsLoading] = useState(false);
  const [isFFmpegReady, setIsFFmpegReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Export settings
  const [codec, setCodec] = useState<FFmpegVideoCodec>('prores');
  const [container, setContainer] = useState<FFmpegContainer>('mov');
  const [preset, setPreset] = useState<string>('');

  // Codec-specific settings
  const [proresProfile, setProresProfile] = useState<ProResProfile>('hq');
  const [dnxhrProfile, setDnxhrProfile] = useState<DnxhrProfile>('dnxhr_hq');
  // HAP not available in this FFmpeg build (requires libsnappy)
  // const [hapFormat, setHapFormat] = useState<HapFormat>('hap_q');
  // const [hapChunks, setHapChunks] = useState(4);

  // Quality settings
  const [useQuality, setUseQuality] = useState(true);
  const [quality, setQuality] = useState(18);
  const [bitrate, setBitrate] = useState(20_000_000);

  // Export state
  const [isExporting, setIsExporting] = useState(false);
  const [progress, setProgress] = useState<FFmpegProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<'idle' | 'rendering' | 'encoding'>('idle');

  // Check if FFmpeg is supported
  const isSupported = FFmpegBridge.isSupported();
  const isMultiThreaded = FFmpegBridge.isMultiThreaded();

  // Load FFmpeg on demand
  const loadFFmpeg = useCallback(async () => {
    if (isFFmpegReady) return;

    setIsLoading(true);
    setLoadError(null);

    try {
      const ffmpeg = getFFmpegBridge();
      await ffmpeg.load();
      setIsFFmpegReady(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load FFmpeg';
      setLoadError(msg);
      log.error('Load error', e);
    } finally {
      setIsLoading(false);
    }
  }, [isFFmpegReady]);

  // Apply platform preset
  const applyPreset = useCallback((presetId: string) => {
    const presetConfig = PLATFORM_PRESETS[presetId];
    if (!presetConfig) {
      setPreset('');
      return;
    }

    setCodec(presetConfig.codec);
    setContainer(presetConfig.container);

    if (presetConfig.quality !== undefined) {
      setUseQuality(true);
      setQuality(presetConfig.quality);
    }
    if (presetConfig.bitrate !== undefined) {
      setUseQuality(false);
      setBitrate(presetConfig.bitrate);
    }
    if (presetConfig.proresProfile) {
      setProresProfile(presetConfig.proresProfile);
    }
    if (presetConfig.dnxhrProfile) {
      setDnxhrProfile(presetConfig.dnxhrProfile);
    }
    // HAP not available
    // if (presetConfig.hapFormat) {
    //   setHapFormat(presetConfig.hapFormat);
    // }

    setPreset(presetId);
  }, []);

  // Handle container change - update codec if incompatible
  const handleContainerChange = useCallback((newContainer: FFmpegContainer) => {
    setContainer(newContainer);
    setPreset('');

    // Check if current codec is compatible
    const codecInfo = getCodecInfo(codec);
    if (codecInfo && !codecInfo.containers.includes(newContainer)) {
      // Switch to a compatible codec
      if (newContainer === 'mxf') {
        setCodec('dnxhd');
      } else {
        // Default to ProRes for other containers
        setCodec('prores');
      }
    }
  }, [codec]);

  // Handle codec change
  const handleCodecChange = useCallback((newCodec: FFmpegVideoCodec) => {
    setCodec(newCodec);
    setPreset('');

    // Update container if needed
    const codecInfo = getCodecInfo(newCodec);
    if (codecInfo && !codecInfo.containers.includes(container)) {
      setContainer(codecInfo.containers[0]);
    }
  }, [container]);

  // Start export
  const handleExport = useCallback(async () => {
    if (isExporting) return;

    // Ensure FFmpeg is loaded
    if (!isFFmpegReady) {
      await loadFFmpeg();
      if (!getFFmpegBridge().isLoaded()) {
        setError('FFmpeg not loaded');
        return;
      }
    }

    setIsExporting(true);
    setError(null);
    setProgress(null);
    setPhase('rendering');

    try {
      const settings: FFmpegExportSettings = {
        codec,
        container,
        width,
        height,
        fps,
        startTime,
        endTime,
        quality: useQuality ? quality : undefined,
        bitrate: !useQuality ? bitrate : undefined,
        proresProfile: codec === 'prores' ? proresProfile : undefined,
        dnxhrProfile: codec === 'dnxhd' ? dnxhrProfile : undefined,
        // HAP not available in this build
        hapFormat: undefined,
        hapChunks: undefined,
      };

      // Render frames
      log.info('Rendering frames...');
      const frames = await onRenderFrames(settings);

      if (frames.length === 0) {
        throw new Error('No frames rendered');
      }

      // Encode with FFmpeg
      setPhase('encoding');
      log.info(`Encoding ${frames.length} frames...`);

      const ffmpeg = getFFmpegBridge();
      const blob = await ffmpeg.encode(frames, settings, (p) => {
        setProgress(p);
      });

      // Download the file
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.${container}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      log.info('Export complete');
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Export failed';
      setError(msg);
      log.error('Export error', e);
    } finally {
      setIsExporting(false);
      setPhase('idle');
    }
  }, [
    codec, container, width, height, fps, startTime, endTime,
    quality, bitrate, useQuality, proresProfile, dnxhrProfile,
    isFFmpegReady, loadFFmpeg, onRenderFrames, filename,
  ]);

  // Cancel export
  const handleCancel = useCallback(() => {
    const ffmpeg = getFFmpegBridge();
    ffmpeg.cancel();
    setIsExporting(false);
    setPhase('idle');
  }, []);

  // Check if codec needs quality control
  const showQualityControl = ['libx264', 'libx265', 'libvpx_vp9', 'libsvtav1'].includes(codec);

  // Get codec info for display
  const codecInfo = getCodecInfo(codec);

  if (!isSupported) {
    return (
      <div className="export-section">
        <div className="export-section-header">
          FFmpeg Export (Not Supported)
        </div>
        <div className="export-error" style={{ margin: '8px 0' }}>
          FFmpeg WASM requires WebAssembly support.
        </div>
      </div>
    );
  }

  return (
    <div className="export-section">
      <div className="export-section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>
          FFmpeg Export
          {!isMultiThreaded && (
            <span style={{ fontSize: '10px', color: 'var(--text-secondary)', marginLeft: '6px' }} title="Single-threaded mode (slower but compatible)">
              (ST)
            </span>
          )}
        </span>
        {!isFFmpegReady && (
          <button
            onClick={loadFFmpeg}
            disabled={isLoading}
            className="btn-small"
            style={{ fontSize: '11px', padding: '2px 8px' }}
          >
            {isLoading ? 'Loading...' : 'Load FFmpeg'}
          </button>
        )}
        {isFFmpegReady && (
          <span style={{ fontSize: '11px', color: 'var(--success, #4caf50)' }}>Ready</span>
        )}
      </div>

      {loadError && (
        <div className="export-error" style={{ margin: '8px 0', fontSize: '12px' }}>
          {loadError}
        </div>
      )}

      {/* Preset Selector */}
      <div className="control-row">
        <label>Preset</label>
        <select value={preset} onChange={(e) => applyPreset(e.target.value)}>
          <option value="">Custom</option>
          <optgroup label="Social Media">
            <option value="youtube">YouTube</option>
            <option value="youtube_hdr">YouTube HDR</option>
            <option value="vimeo">Vimeo</option>
            <option value="instagram">Instagram</option>
            <option value="tiktok">TikTok</option>
            <option value="twitter">Twitter/X</option>
          </optgroup>
          <optgroup label="Professional">
            <option value="premiere">Adobe Premiere</option>
            <option value="finalcut">Final Cut Pro</option>
            <option value="davinci">DaVinci Resolve</option>
            <option value="avid">Avid Media Composer</option>
          </optgroup>
          <optgroup label="Special">
            <option value="vj">VJ / Media Server</option>
            <option value="vj_alpha">VJ with Alpha</option>
            <option value="archive">Archive (Lossless)</option>
            <option value="web_transparent">Web with Alpha</option>
          </optgroup>
        </select>
      </div>

      {/* Container Format */}
      <div className="control-row">
        <label>Container</label>
        <select
          value={container}
          onChange={(e) => handleContainerChange(e.target.value as FFmpegContainer)}
        >
          {CONTAINER_FORMATS.map(({ id, name }) => (
            <option key={id} value={id}>{name}</option>
          ))}
        </select>
      </div>

      {/* Video Codec */}
      <div className="control-row">
        <label>Codec</label>
        <CodecSelector
          container={container}
          value={codec}
          onChange={handleCodecChange}
        />
      </div>

      {/* Codec description */}
      {codecInfo && (
        <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '-4px', marginBottom: '8px', paddingLeft: '4px' }}>
          {codecInfo.description}
          {codecInfo.supportsAlpha && ' • Alpha'}
          {codecInfo.supports10bit && ' • 10-bit'}
        </div>
      )}

      {/* ProRes Profile */}
      {codec === 'prores' && (
        <div className="control-row">
          <label>Profile</label>
          <select
            value={proresProfile}
            onChange={(e) => setProresProfile(e.target.value as ProResProfile)}
          >
            {PRORES_PROFILES.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} - {p.description}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* DNxHR Profile */}
      {codec === 'dnxhd' && (
        <div className="control-row">
          <label>Profile</label>
          <select
            value={dnxhrProfile}
            onChange={(e) => setDnxhrProfile(e.target.value as DnxhrProfile)}
          >
            {DNXHR_PROFILES.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} - {p.description}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Quality/Bitrate Control */}
      {showQualityControl && (
        <>
          <div className="control-row">
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <input
                type="checkbox"
                checked={useQuality}
                onChange={(e) => setUseQuality(e.target.checked)}
              />
              Quality (CRF)
            </label>
          </div>
          {useQuality ? (
            <div className="control-row">
              <label>CRF</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
                <input
                  type="range"
                  min={0}
                  max={51}
                  value={quality}
                  onChange={(e) => setQuality(parseInt(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ minWidth: '30px', textAlign: 'right' }}>{quality}</span>
              </div>
            </div>
          ) : (
            <div className="control-row">
              <label>Bitrate</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
                <input
                  type="range"
                  min={1_000_000}
                  max={100_000_000}
                  step={500_000}
                  value={bitrate}
                  onChange={(e) => setBitrate(parseInt(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ minWidth: '60px', textAlign: 'right' }}>
                  {(bitrate / 1_000_000).toFixed(1)} Mbps
                </span>
              </div>
            </div>
          )}
        </>
      )}

      {/* Export Button */}
      {!isExporting ? (
        <button
          className="btn export-start-btn"
          onClick={handleExport}
          disabled={isLoading || endTime <= startTime}
          style={{ marginTop: '12px', width: '100%' }}
        >
          Export with FFmpeg
        </button>
      ) : (
        <div style={{ marginTop: '12px' }}>
          {/* Phase indicator */}
          <div style={{ marginBottom: '8px', fontSize: '13px', fontWeight: 500 }}>
            {phase === 'rendering' && 'Rendering frames...'}
            {phase === 'encoding' && 'Encoding video...'}
          </div>

          {/* Progress bar */}
          <div className="export-progress-bar">
            <div
              className="export-progress-fill"
              style={{ width: `${progress?.percent ?? 0}%` }}
            />
          </div>

          {/* Progress info */}
          <div className="export-progress-info">
            <span>
              {progress ? `Frame ${progress.frame}` : 'Starting...'}
            </span>
            <span>{(progress?.percent ?? 0).toFixed(1)}%</span>
          </div>

          {/* Cancel button */}
          <button
            className="btn export-cancel-btn"
            onClick={handleCancel}
            style={{ marginTop: '8px', width: '100%' }}
          >
            Cancel
          </button>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="export-error" style={{ marginTop: '8px' }}>
          {error}
        </div>
      )}
    </div>
  );
}
