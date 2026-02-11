// Export Panel - embedded panel for frame-by-frame video export

import { useCallback } from 'react';
import { Logger } from '../../services/logger';
import { downloadFCPXML } from '../../services/export/fcpxmlExport';

const log = Logger.create('ExportPanel');
import { FrameExporter, downloadBlob } from '../../engine/export';
import type { VideoCodec, ContainerFormat } from '../../engine/export';
import { AudioExportPipeline } from '../../engine/audio';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { engine } from '../../engine/WebGPUEngine';
import {
  getFFmpegBridge,
  PRORES_PROFILES,
  DNXHR_PROFILES,
  CONTAINER_FORMATS,
  getCodecInfo,
} from '../../engine/ffmpeg';
import { CodecSelector } from './CodecSelector';
import type {
  FFmpegExportSettings,
  FFmpegProgress,
  FFmpegContainer,
  ProResProfile,
  DnxhrProfile,
} from '../../engine/ffmpeg';
import { seekAllClipsToTime, buildLayersAtTime } from './exportHelpers';
import { useExportState, type EncoderType } from './useExportState';

export function ExportPanel() {
  const { duration, inPoint, outPoint, playheadPosition, startExport, setExportProgress, endExport } = useTimelineStore();
  const { getActiveComposition } = useMediaStore();
  const composition = getActiveComposition();

  // All export state, effects, and simple handlers extracted to hook
  const {
    encoder, setEncoder,
    width, height,
    customWidth, setCustomWidth, customHeight, setCustomHeight,
    useCustomResolution, setUseCustomResolution,
    fps, setFps, customFps, setCustomFps, useCustomFps, setUseCustomFps,
    useInOut, setUseInOut, filename, setFilename,
    bitrate, setBitrate, containerFormat, setContainerFormat,
    videoCodec, setVideoCodec, codecSupport, rateControl, setRateControl,
    ffmpegCodec, ffmpegContainer, ffmpegPreset,
    proresProfile, setProresProfile, dnxhrProfile, setDnxhrProfile,
    ffmpegQuality, setFfmpegQuality, ffmpegBitrate, ffmpegRateControl,
    isFFmpegLoading, isFFmpegReady, ffmpegLoadError,
    includeAudio, setIncludeAudio, audioSampleRate, setAudioSampleRate,
    audioBitrate, setAudioBitrate, normalizeAudio, setNormalizeAudio,
    isExporting, setIsExporting, progress, setProgress,
    ffmpegProgress, setFfmpegProgress, exportPhase, setExportPhase,
    error, setError, exporter, setExporter,
    isSupported, isAudioSupported, audioCodec,
    isFFmpegSupported, isFFmpegMultiThreaded,
    handleResolutionChange, loadFFmpeg, applyFFmpegPreset,
    handleFFmpegContainerChange, handleFFmpegCodecChange,
  } = useExportState(composition);

  // Compute actual start/end based on In/Out markers
  const startTime = useInOut && inPoint !== null ? inPoint : 0;
  const endTime = useInOut && outPoint !== null ? outPoint : duration;

  // Handle export (WebCodecs)
  const handleExport = useCallback(async () => {
    if (isExporting) return;

    setIsExporting(true);
    setError(null);
    setProgress(null);

    const actualWidth = useCustomResolution ? customWidth : width;
    const actualHeight = useCustomResolution ? customHeight : height;

    // Get file extension from container format
    const fileExtension = containerFormat === 'webm' ? 'webm' : 'mp4';

    const exportFps = useCustomFps ? customFps : fps;

    const exp = new FrameExporter({
      width: actualWidth,
      height: actualHeight,
      fps: exportFps,
      codec: videoCodec,
      container: containerFormat,
      bitrate,
      startTime,
      endTime,
      // Audio settings
      includeAudio,
      audioSampleRate,
      audioBitrate,
      normalizeAudio,
      // Export mode: webcodecs = fast, htmlvideo = precise
      exportMode: encoder === 'webcodecs' ? 'fast' : 'precise',
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
        downloadBlob(blob, `${filename}.${fileExtension}`);
      }
    } catch (e) {
      log.error('Export failed', e);
      setError(e instanceof Error ? e.message : 'Export failed');
    } finally {
      setIsExporting(false);
      setExporter(null);
      // End export progress in timeline
      endExport();
    }
  }, [width, height, customWidth, customHeight, useCustomResolution, fps, customFps, useCustomFps, bitrate, startTime, endTime, filename, isExporting, includeAudio, audioSampleRate, audioBitrate, normalizeAudio, containerFormat, videoCodec]);

  // Handle cancel
  const handleCancel = useCallback(() => {
    if (encoder === 'webcodecs' || encoder === 'htmlvideo') {
      if (exporter) {
        exporter.cancel();
      }
      setExporter(null);
    } else {
      const ffmpeg = getFFmpegBridge();
      ffmpeg.cancel();
    }
    setIsExporting(false);
    setExportPhase('idle');
    // End export progress in timeline
    endExport();
  }, [exporter, encoder, endExport]);

  // Handle FFmpeg export
  const handleFFmpegExport = useCallback(async () => {
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
    setFfmpegProgress(null);
    setExportPhase('rendering');

    const actualWidth = useCustomResolution ? customWidth : width;
    const actualHeight = useCustomResolution ? customHeight : height;
    const exportFps = useCustomFps ? customFps : fps;

    // Start export progress in timeline
    startExport(startTime, endTime);

    try {
      const settings: FFmpegExportSettings = {
        codec: ffmpegCodec,
        container: ffmpegContainer,
        width: actualWidth,
        height: actualHeight,
        fps: exportFps,
        startTime,
        endTime,
        quality: ffmpegCodec === 'mjpeg' ? ffmpegQuality : undefined,
        bitrate: undefined, // Bitrate not used for available codecs
        proresProfile: ffmpegCodec === 'prores' ? proresProfile : undefined,
        dnxhrProfile: ffmpegCodec === 'dnxhd' ? dnxhrProfile : undefined,
        // HAP not available in ASYNCIFY build
      };

      // Render frames
      log.info('Rendering frames for FFmpeg...');
      const frames: Uint8Array[] = [];
      const totalFrames = Math.ceil((endTime - startTime) * exportFps);
      const frameDuration = 1 / exportFps;

      log.info(`Total frames: ${totalFrames}, duration: ${frameDuration.toFixed(4)}s per frame`);

      // Set engine to export mode and correct resolution
      engine.setExporting(true);
      engine.setResolution(actualWidth, actualHeight);

      const frameStartTime = performance.now();

      for (let i = 0; i < totalFrames; i++) {
        const time = startTime + i * frameDuration;

        if (i === 0) log.debug('Frame 0: Starting seek...');

        // Seek all video clips to the exact frame time
        await seekAllClipsToTime(time);

        if (i === 0) log.debug('Frame 0: Seek complete, waiting for decode...');

        // Small delay to ensure video frame is decoded (browser needs time after seek)
        await new Promise(resolve => setTimeout(resolve, 16));

        if (i === 0) log.debug('Frame 0: Building layers...');

        // Build layers at this time and render
        const layers = buildLayersAtTime(time);

        if (i === 0) log.debug(`Frame 0: Got ${layers.length} layers`);

        if (layers.length === 0) {
          log.warn(`No layers at time ${time.toFixed(3)}`);
        }

        engine.render(layers);

        if (i === 0) log.debug('Frame 0: Render complete, reading pixels...');

        // Read pixels
        const pixels = await engine.readPixels();

        if (i === 0) log.debug(`Frame 0: Got pixels: ${pixels ? pixels.length : 'null'}`);
        if (pixels) {
          // Make a COPY of pixels (not a view) to ensure each frame is unique
          const frameCopy = new Uint8Array(pixels.length);
          frameCopy.set(pixels);
          frames.push(frameCopy);

          // Debug: check first few pixels to see if content changes
          if (i < 3 || i % 30 === 0) {
            const sample = [pixels[0], pixels[1], pixels[2], pixels[3], pixels[1000], pixels[2000]];
            log.debug(`Frame ${i} sample pixels: [${sample.join(', ')}]`);
          }
        }

        // Log progress every 30 frames or on first frame
        if (i === 0 || i % 30 === 0) {
          const elapsed = (performance.now() - frameStartTime) / 1000;
          const renderFps = (i + 1) / elapsed;
          log.debug(`Frame ${i + 1}/${totalFrames} at ${time.toFixed(3)}s, ${renderFps.toFixed(1)} fps, ${layers.length} layers`);
        }

        // Update progress during rendering (0-30% - frame capture is fast, encoding is slow)
        const percent = ((i + 1) / totalFrames) * 30;
        setFfmpegProgress({
          percent,
          frame: i + 1,
          fps: 0,
          time: time,
          speed: 0,
          bitrate: 0,
          size: 0,
          eta: 0,
        });
        // Update timeline export progress
        setExportProgress(percent, time);
      }

      // Reset export mode
      engine.setExporting(false);

      if (frames.length === 0) {
        throw new Error('No frames rendered');
      }

      // Extract audio from timeline (if enabled)
      let audioBuffer: AudioBuffer | null = null;

      if (includeAudio) {
        setExportPhase('audio');
        log.info('Extracting audio for FFmpeg...');

        try {
          const audioPipeline = new AudioExportPipeline({
            sampleRate: audioSampleRate,
            bitrate: audioBitrate,
            normalize: normalizeAudio,
          });

          audioBuffer = await audioPipeline.exportRawAudio(
            startTime,
            endTime,
            (audioProgress) => {
              // Audio extraction: 30-40%
              const percent = 30 + (audioProgress.percent * 0.1);
              setFfmpegProgress({
                percent,
                frame: frames.length,
                fps: 0,
                time: endTime,
                speed: 0,
                bitrate: 0,
                size: 0,
                eta: 0,
              });
              setExportProgress(percent, endTime);
            }
          );

          if (audioBuffer && audioBuffer.length > 0) {
            log.info(`Audio extracted: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.numberOfChannels}ch, ${audioBuffer.length} samples`);
          } else {
            log.warn('Audio extraction returned empty or null buffer');
          }
        } catch (audioError) {
          log.warn('Audio extraction failed, continuing without audio', audioError);
        }
      }

      // Encode with FFmpeg - this is the slow part (40-100%)
      setExportPhase('encoding');
      log.info(`Encoding ${frames.length} frames with FFmpeg...`);

      // Show 40% while encoding starts
      setFfmpegProgress({
        percent: 40,
        frame: frames.length,
        fps: 0,
        time: endTime,
        speed: 0,
        bitrate: 0,
        size: 0,
        eta: 0,
      });
      setExportProgress(40, endTime);

      // Allow React to render "Encoding..." before FFmpeg blocks the main thread
      await new Promise(resolve => setTimeout(resolve, 50));

      const ffmpeg = getFFmpegBridge();

      const blob = await ffmpeg.encode(frames, settings, (p: FFmpegProgress) => {
        // Encoding is 40-100% (60% of the progress bar - this is the slow part)
        const totalPercent = 40 + (p.percent / 100) * 60;
        setFfmpegProgress({
          ...p,
          percent: totalPercent,
        });
        setExportProgress(totalPercent, endTime);

        // Force UI update during encoding (callMain blocks main thread)
        // Note: This may not work due to WASM blocking, but worth trying
      }, audioBuffer);

      // Download the file
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.${ffmpegContainer}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      log.info('FFmpeg export complete');
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Export failed';
      setError(msg);
      log.error('FFmpeg export error', e);
    } finally {
      // Always reset export mode
      engine.setExporting(false);
      setIsExporting(false);
      setExportPhase('idle');
      // End export progress in timeline
      endExport();
    }
  }, [
    isExporting, isFFmpegReady, loadFFmpeg, useCustomResolution, customWidth, customHeight,
    width, height, fps, customFps, useCustomFps, startTime, endTime, ffmpegCodec, ffmpegContainer,
    ffmpegQuality, proresProfile, dnxhrProfile, filename,
    includeAudio, audioSampleRate, audioBitrate, normalizeAudio,
    startExport, setExportProgress, endExport,
  ]);

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
        // Convert audio chunks to a downloadable file
        // Use the codec from the result to determine mime type and extension
        const audioBlobs: Blob[] = [];
        for (const chunk of audioResult.chunks) {
          const buffer = new ArrayBuffer(chunk.byteLength);
          chunk.copyTo(buffer);
          audioBlobs.push(new Blob([buffer]));
        }
        const mimeType = audioResult.codec === 'opus' ? 'audio/ogg' : 'audio/aac';
        const extension = audioResult.codec === 'opus' ? 'ogg' : 'aac';
        const audioBlob = new Blob(audioBlobs, { type: mimeType });
        downloadBlob(audioBlob, `${filename}.${extension}`);
      } else {
        setError('No audio clips found in the selected range');
      }
    } catch (e) {
      log.error('Audio export failed', e);
      setError(e instanceof Error ? e.message : 'Audio export failed');
    } finally {
      setIsExporting(false);
    }
  }, [startTime, endTime, filename, isExporting, audioSampleRate, audioBitrate, normalizeAudio]);

  // Handle FCPXML export
  const handleExportFCPXML = useCallback(() => {
    const { clips, tracks, duration: timelineDuration } = useTimelineStore.getState();
    const activeComp = getActiveComposition();

    downloadFCPXML(clips, tracks, timelineDuration, {
      projectName: activeComp?.name || filename || 'MasterSelects Export',
      frameRate: activeComp?.frameRate || fps,
      width: activeComp?.width || width,
      height: activeComp?.height || height,
      includeAudio,
    });
  }, [filename, fps, width, height, includeAudio, getActiveComposition]);

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
      const imageData = new ImageData(new Uint8ClampedArray(pixels), engineWidth, engineHeight);

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
      log.error('Frame render failed', e);
      setError(e instanceof Error ? e.message : 'Frame render failed');
    }
  }, [width, height, customWidth, customHeight, useCustomResolution, filename, playheadPosition]);

  // Format time as MM:SS.ff
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(2);
    return `${m}:${s.padStart(5, '0')}`;
  };

  // Get actual FPS value
  const actualFps = useCustomFps ? customFps : fps;

  // Format file size estimate - works for both encoders
  const estimatedSize = () => {
    const durationSec = endTime - startTime;
    if (durationSec <= 0) return '—';

    let estimatedBitrate: number;

    if (encoder === 'webcodecs' || encoder === 'htmlvideo') {
      estimatedBitrate = bitrate;
    } else {
      // FFmpeg estimation
      if (ffmpegRateControl === 'crf') {
        // CRF: estimate based on quality and resolution
        const pixels = (useCustomResolution ? customWidth * customHeight : width * height);
        const qualityFactor = Math.pow(2, (51 - ffmpegQuality) / 6); // CRF scale
        estimatedBitrate = (pixels * actualFps * qualityFactor) / 10000;
        estimatedBitrate = Math.min(estimatedBitrate, 100_000_000); // Cap at 100 Mbps
      } else {
        estimatedBitrate = ffmpegBitrate;
      }
    }

    // Add ~10% for audio if included
    if (includeAudio && (encoder === 'webcodecs' || encoder === 'htmlvideo')) {
      estimatedBitrate += audioBitrate;
    }

    const bytes = (estimatedBitrate / 8) * durationSec;
    if (bytes > 1024 * 1024 * 1024) {
      return `~${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
    return `~${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  };

  // Check if current encoder is available
  const webCodecsAvailable = isSupported;
  const ffmpegAvailable = isFFmpegSupported;

  // Get codec info for FFmpeg display
  const ffmpegCodecInfo = getCodecInfo(ffmpegCodec);
  // Only MJPEG has quality slider (q:v), professional codecs use profiles
  const showFFmpegQualityControl = ffmpegCodec === 'mjpeg';

  // If neither encoder is supported, show error
  if (!webCodecsAvailable && !ffmpegAvailable) {
    return (
      <div className="export-panel">
        <div className="panel-header">
          <h3>Export</h3>
        </div>
        <div className="export-error">
          No video encoder available. WebCodecs requires Chrome 94+ or Safari 16.4+.
          FFmpeg WASM requires WebAssembly support.
        </div>
      </div>
    );
  }

  return (
    <div className="export-panel">
      {/* Action Buttons - Always visible at top */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', padding: '12px 12px 0' }}>
        <button
          className="btn"
          onClick={handleRenderFrame}
          style={{ flex: 1 }}
          disabled={isExporting}
        >
          Frame
        </button>
        <button
          className="btn export-start-btn"
          onClick={(encoder === 'webcodecs' || encoder === 'htmlvideo') ? handleExport : handleFFmpegExport}
          disabled={isExporting || endTime <= startTime || (encoder === 'ffmpeg' && isFFmpegLoading)}
          style={{ flex: 1 }}
        >
          Export Video
        </button>
        <button
          className="btn"
          onClick={handleExportAudioOnly}
          disabled={isExporting || endTime <= startTime || !isAudioSupported}
          style={{ flex: 1 }}
          title={!isAudioSupported ? 'Audio encoding not supported in this browser' : `Export as ${audioCodec?.toUpperCase() || 'audio'}`}
        >
          Export Audio
        </button>
        <button
          className="btn"
          onClick={handleExportFCPXML}
          disabled={isExporting}
          style={{ flex: 1 }}
          title="Export timeline as Final Cut Pro XML (compatible with Resolve, Premiere)"
        >
          XML
        </button>
      </div>

      {!isExporting ? (
        <div className="export-form">
          {/* Encoder Selection */}
          <div className="export-section">
            <div className="export-section-header">Encoder</div>
            <div className="control-row">
              <label>Method</label>
              <select
                value={encoder}
                onChange={(e) => setEncoder(e.target.value as EncoderType)}
              >
                {webCodecsAvailable && (
                  <option value="webcodecs">⚡ WebCodecs (Fast)</option>
                )}
                {webCodecsAvailable && (
                  <option value="htmlvideo">🎯 HTMLVideo (Precise)</option>
                )}
                {ffmpegAvailable && (
                  <option value="ffmpeg">
                    FFmpeg (CPU){!isFFmpegMultiThreaded ? ' - ST' : ''}
                  </option>
                )}
              </select>
            </div>

            {/* FFmpeg Load Button / Status */}
            {encoder === 'ffmpeg' && (
              <div className="control-row" style={{ justifyContent: 'flex-end' }}>
                {!isFFmpegReady ? (
                  <button
                    onClick={loadFFmpeg}
                    disabled={isFFmpegLoading}
                    className="btn-small"
                    style={{ fontSize: '11px', padding: '4px 12px' }}
                  >
                    {isFFmpegLoading ? 'Loading...' : 'Load FFmpeg'}
                  </button>
                ) : (
                  <span style={{ fontSize: '11px', color: 'var(--success, #4caf50)' }}>
                    FFmpeg Ready
                  </span>
                )}
              </div>
            )}

            {ffmpegLoadError && encoder === 'ffmpeg' && (
              <div className="export-error" style={{ margin: '4px 0', fontSize: '11px' }}>
                {ffmpegLoadError}
              </div>
            )}
          </div>

          {/* Video Settings */}
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
                <select
                  className="export-extension-select"
                  value={(encoder === 'webcodecs' || encoder === 'htmlvideo') ? containerFormat : ffmpegContainer}
                  onChange={(e) => {
                    if (encoder === 'webcodecs' || encoder === 'htmlvideo') {
                      setContainerFormat(e.target.value as ContainerFormat);
                    } else {
                      handleFFmpegContainerChange(e.target.value as FFmpegContainer);
                    }
                  }}
                  title="Click to change container format"
                >
                  {(encoder === 'webcodecs' || encoder === 'htmlvideo') ? (
                    FrameExporter.getContainerFormats().map(({ id }) => (
                      <option key={id} value={id}>.{id}</option>
                    ))
                  ) : (
                    CONTAINER_FORMATS.map(({ id }) => (
                      <option key={id} value={id}>.{id}</option>
                    ))
                  )}
                </select>
              </div>
            </div>

            {/* FFmpeg Preset - only for FFmpeg */}
            {encoder === 'ffmpeg' && (
              <div className="control-row">
                <label>Preset</label>
                <select value={ffmpegPreset} onChange={(e) => applyFFmpegPreset(e.target.value)}>
                  <option value="">Custom</option>
                  <optgroup label="Professional NLEs">
                    <option value="premiere">Adobe Premiere</option>
                    <option value="finalcut">Final Cut Pro</option>
                    <option value="davinci">DaVinci Resolve</option>
                    <option value="avid">Avid Media Composer</option>
                  </optgroup>
                  <optgroup label="ProRes Quality">
                    <option value="prores_proxy">ProRes Proxy</option>
                    <option value="prores_lt">ProRes LT</option>
                    <option value="prores_hq">ProRes HQ</option>
                    <option value="prores_4444">ProRes 4444 (Alpha)</option>
                  </optgroup>
                  <optgroup label="Lossless / Archive">
                    <option value="archive">Archive (FFV1)</option>
                    <option value="utvideo_alpha">UTVideo (Alpha)</option>
                  </optgroup>
                  <optgroup label="Quick Export">
                    <option value="mjpeg_preview">MJPEG Preview</option>
                  </optgroup>
                </select>
              </div>
            )}

            {/* Video Codec */}
            <div className="control-row">
              <label>Codec</label>
              {(encoder === 'webcodecs' || encoder === 'htmlvideo') ? (
                <select
                  value={videoCodec}
                  onChange={(e) => setVideoCodec(e.target.value as VideoCodec)}
                >
                  {FrameExporter.getVideoCodecs(containerFormat).map(({ id, label }) => (
                    <option key={id} value={id} disabled={!codecSupport[id]}>
                      {label} {!codecSupport[id] ? '(not supported)' : ''}
                    </option>
                  ))}
                </select>
              ) : (
                <CodecSelector
                  container={ffmpegContainer}
                  value={ffmpegCodec}
                  onChange={handleFFmpegCodecChange}
                />
              )}
            </div>

            {/* FFmpeg Codec description */}
            {encoder === 'ffmpeg' && ffmpegCodecInfo && (
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '-4px', marginBottom: '8px', paddingLeft: '4px' }}>
                {ffmpegCodecInfo.description}
                {ffmpegCodecInfo.supportsAlpha && ' • Alpha'}
                {ffmpegCodecInfo.supports10bit && ' • 10-bit'}
              </div>
            )}

            {/* FFmpeg ProRes Profile */}
            {encoder === 'ffmpeg' && ffmpegCodec === 'prores' && (
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

            {/* FFmpeg DNxHR Profile */}
            {encoder === 'ffmpeg' && ffmpegCodec === 'dnxhd' && (
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

            {/* HAP codec removed - requires snappy which doesn't build with ASYNCIFY */}

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
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                {!useCustomFps ? (
                  <select
                    value={fps}
                    onChange={(e) => setFps(Number(e.target.value))}
                    style={{ flex: 1 }}
                  >
                    <option value={23.976}>23.976 fps (Film)</option>
                    <option value={24}>24 fps (Cinema)</option>
                    <option value={25}>25 fps (PAL)</option>
                    <option value={29.97}>29.97 fps (NTSC)</option>
                    <option value={30}>30 fps</option>
                    <option value={48}>48 fps (HFR)</option>
                    <option value={50}>50 fps (PAL)</option>
                    <option value={59.94}>59.94 fps (NTSC)</option>
                    <option value={60}>60 fps</option>
                    <option value={120}>120 fps</option>
                  </select>
                ) : (
                  <input
                    type="number"
                    value={customFps}
                    onChange={(e) => setCustomFps(Math.max(1, Math.min(240, parseFloat(e.target.value) || 30)))}
                    min={1}
                    max={240}
                    step={0.001}
                    style={{ flex: 1 }}
                  />
                )}
                <label style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', whiteSpace: 'nowrap' }}>
                  <input
                    type="checkbox"
                    checked={useCustomFps}
                    onChange={(e) => setUseCustomFps(e.target.checked)}
                  />
                  Custom
                </label>
              </div>
            </div>

            {/* Quality - different controls for each encoder */}
            {(encoder === 'webcodecs' || encoder === 'htmlvideo') ? (
              <>
                {/* Rate Control */}
                <div className="control-row">
                  <label>Rate Control</label>
                  <select
                    value={rateControl}
                    onChange={(e) => setRateControl(e.target.value as 'vbr' | 'cbr')}
                  >
                    <option value="vbr">VBR (Variable Bitrate)</option>
                    <option value="cbr">CBR (Constant Bitrate)</option>
                  </select>
                </div>

                {/* Bitrate */}
                <div className="control-row">
                  <label>{rateControl === 'cbr' ? 'Bitrate' : 'Target Bitrate'}</label>
                  <select
                    value={bitrate}
                    onChange={(e) => setBitrate(Number(e.target.value))}
                  >
                    <option value={5_000_000}>5 Mbps (Low)</option>
                    <option value={10_000_000}>10 Mbps (Medium)</option>
                    <option value={15_000_000}>15 Mbps (High)</option>
                    <option value={20_000_000}>20 Mbps</option>
                    <option value={25_000_000}>25 Mbps (Very High)</option>
                    <option value={35_000_000}>35 Mbps</option>
                    <option value={50_000_000}>50 Mbps (Max)</option>
                  </select>
                </div>

                {/* Bitrate Slider */}
                <div className="control-row">
                  <label></label>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flex: 1 }}>
                    <input
                      type="range"
                      min={1_000_000}
                      max={100_000_000}
                      step={500_000}
                      value={bitrate}
                      onChange={(e) => setBitrate(Number(e.target.value))}
                      style={{ flex: 1 }}
                    />
                    <span style={{ minWidth: '70px', fontSize: '12px', textAlign: 'right' }}>
                      {(bitrate / 1_000_000).toFixed(1)} Mbps
                    </span>
                  </div>
                </div>
              </>
            ) : showFFmpegQualityControl && (
              /* MJPEG Quality Control - lower values = higher quality */
              <div className="control-row">
                <label>Quality</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
                  <input
                    type="range"
                    min={1}
                    max={31}
                    value={ffmpegQuality}
                    onChange={(e) => setFfmpegQuality(parseInt(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <span style={{ minWidth: '60px', textAlign: 'right', fontSize: '12px' }}>
                    {ffmpegQuality} {ffmpegQuality <= 5 ? '(High)' : ffmpegQuality <= 10 ? '(Good)' : ffmpegQuality <= 20 ? '(Med)' : '(Low)'}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Audio Settings */}
          <div className="export-section">
            <div className="export-section-header">Audio</div>

            <div className="control-row">
              <label>
                <input
                  type="checkbox"
                  checked={includeAudio}
                  onChange={(e) => setIncludeAudio(e.target.checked)}
                  disabled={(encoder === 'webcodecs' || encoder === 'htmlvideo') && !isAudioSupported}
                />
                Include Audio
              </label>
              {(encoder === 'webcodecs' || encoder === 'htmlvideo') && !isAudioSupported && (
                <span style={{ color: 'var(--warning)', fontSize: '11px', marginLeft: '8px' }}>
                  Not supported
                </span>
              )}
            </div>

            {includeAudio && (
              <>
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

                {encoder === 'ffmpeg' && (
                  <div className="control-row">
                    <label>Audio Codec</label>
                    <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {ffmpegContainer === 'mov' ? 'AAC' :
                       ffmpegContainer === 'mkv' ? 'FLAC' :
                       ffmpegContainer === 'avi' ? 'PCM' :
                       ffmpegContainer === 'mxf' ? 'PCM' : 'AAC'} (auto)
                    </span>
                  </div>
                )}

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

            <div className="export-summary">
              <div>Range: {formatTime(startTime)} - {formatTime(endTime)}</div>
              <div>Duration: {formatTime(endTime - startTime)}</div>
              <div>Frames: {Math.ceil((endTime - startTime) * actualFps)}</div>
              <div>Est. Size: {estimatedSize()}</div>
            </div>
          </div>

          {error && <div className="export-error">{error}</div>}
        </div>
      ) : (
        <div className="export-progress-container">
          {/* Phase indicator */}
          <div style={{ marginBottom: '12px', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>
            {(encoder === 'webcodecs' || encoder === 'htmlvideo') ? (
              <>
                {progress?.phase === 'video' && 'Encoding video frames...'}
                {progress?.phase === 'audio' && (
                  <>Processing audio: {progress.audioPhase} ({progress.audioPercent}%)</>
                )}
                {progress?.phase === 'muxing' && 'Finalizing...'}
              </>
            ) : (
              <>
                {exportPhase === 'rendering' && 'Rendering frames...'}
                {exportPhase === 'audio' && 'Processing audio...'}
                {exportPhase === 'encoding' && 'Encoding video (please wait)...'}
              </>
            )}
          </div>

          <div className="export-progress-bar">
            <div
              className="export-progress-fill"
              style={{
                width: `${(encoder === 'webcodecs' || encoder === 'htmlvideo')
                  ? (progress?.percent ?? 0)
                  : (ffmpegProgress?.percent ?? 0)}%`
              }}
            />
          </div>
          <div className="export-progress-info">
            {(encoder === 'webcodecs' || encoder === 'htmlvideo') ? (
              <>
                {progress?.phase === 'video' ? (
                  <span>Frame {progress?.currentFrame ?? 0} / {progress?.totalFrames ?? 0}</span>
                ) : (
                  <span>Audio processing</span>
                )}
                <span>{(progress?.percent ?? 0).toFixed(1)}%</span>
              </>
            ) : (
              <>
                <span>Frame {ffmpegProgress?.frame ?? 0}</span>
                <span>{(ffmpegProgress?.percent ?? 0).toFixed(1)}%</span>
              </>
            )}
          </div>
          {(encoder === 'webcodecs' || encoder === 'htmlvideo') && progress && progress.phase === 'video' && progress.estimatedTimeRemaining > 0 && (
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
