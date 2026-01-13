// Export Panel - embedded panel for frame-by-frame video export

import { useState, useEffect, useCallback } from 'react';
import { FrameExporter, downloadBlob } from '../../engine/FrameExporter';
import type { ExportProgress, VideoCodec, ContainerFormat } from '../../engine/FrameExporter';
import { AudioExportPipeline, AudioEncoderWrapper, type AudioCodec } from '../../engine/audio';
import { useTimelineStore } from '../../stores/timeline';
import { useMediaStore } from '../../stores/mediaStore';
import { engine } from '../../engine/WebGPUEngine';
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
import type { Layer, LayerSource, TimelineClip, TimelineTrack } from '../../types';

type EncoderType = 'webcodecs' | 'ffmpeg';

// Helper: Seek all video clips to exact time for frame-accurate export
async function seekAllClipsToTime(time: number): Promise<void> {
  const { clips, tracks, getSourceTimeForClip, getInterpolatedSpeed } = useTimelineStore.getState();
  const seekPromises: Promise<void>[] = [];

  // Get clips at this time
  const clipsAtTime = clips.filter(
    c => time >= c.startTime && time < c.startTime + c.duration
  );

  console.log(`[seekAllClipsToTime] time=${time.toFixed(3)}, total clips=${clips.length}, clips at time=${clipsAtTime.length}`);

  for (const clip of clipsAtTime) {
    const track = tracks.find(t => t.id === clip.trackId);
    if (!track?.visible) continue;

    // Handle nested composition clips
    if (clip.isComposition && clip.nestedClips) {
      const clipLocalTime = time - clip.startTime;
      const nestedTime = clipLocalTime + (clip.inPoint || 0);

      for (const nestedClip of clip.nestedClips) {
        if (nestedTime >= nestedClip.startTime && nestedTime < nestedClip.startTime + nestedClip.duration) {
          if (nestedClip.source?.videoElement) {
            const nestedLocalTime = nestedTime - nestedClip.startTime;
            const nestedClipTime = nestedClip.reversed
              ? nestedClip.outPoint - nestedLocalTime
              : nestedLocalTime + nestedClip.inPoint;

            // Always seek the HTMLVideoElement since that's what we use for texture rendering
            seekPromises.push(seekVideo(nestedClip.source.videoElement, nestedClipTime));
          }
        }
      }
      continue;
    }

    // Handle regular video clips
    if (clip.source?.type === 'video' && clip.source.videoElement) {
      const clipLocalTime = time - clip.startTime;
      let clipTime: number;

      try {
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
      } catch {
        clipTime = clip.reversed
          ? clip.outPoint - clipLocalTime
          : clipLocalTime + clip.inPoint;
        clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, clipTime));
      }

      // Always seek the HTMLVideoElement since that's what we use for texture rendering
      seekPromises.push(seekVideo(clip.source.videoElement, clipTime));
    }
  }

  console.log(`[seekAllClipsToTime] Waiting for ${seekPromises.length} seek promises...`);
  await Promise.all(seekPromises);
  console.log(`[seekAllClipsToTime] All seeks complete`);
}

// Helper: Seek HTMLVideoElement to exact time
function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const targetTime = Math.max(0, Math.min(time, video.duration || 0));

    // If already at target time, just wait for frame
    if (Math.abs(video.currentTime - targetTime) < 0.01 && !video.seeking && video.readyState >= 3) {
      requestAnimationFrame(() => resolve());
      return;
    }

    // Set timeout in case seek never completes
    const timeout = setTimeout(() => {
      video.removeEventListener('seeked', onSeeked);
      resolve();
    }, 500);

    const onSeeked = () => {
      clearTimeout(timeout);
      video.removeEventListener('seeked', onSeeked);
      // Wait one frame for the video texture to update
      requestAnimationFrame(() => resolve());
    };

    video.addEventListener('seeked', onSeeked);
    video.currentTime = targetTime;
  });
}

// Helper: Build render layers at specific time
function buildLayersAtTime(time: number): Layer[] {
  const { clips, tracks, getInterpolatedTransform, getInterpolatedEffects } = useTimelineStore.getState();
  const layers: Layer[] = [];

  const videoTracks = tracks.filter(t => t.type === 'video');
  const anyVideoSolo = videoTracks.some(t => t.solo);

  const isTrackVisible = (track: TimelineTrack) => {
    if (!track.visible) return false;
    if (anyVideoSolo) return track.solo;
    return true;
  };

  const clipsAtTime = clips.filter(
    c => time >= c.startTime && time < c.startTime + c.duration
  );

  // Sort by track order - lower index tracks should be first in array (render on top)
  // WebGPU renders layers[0] on top, layers[end] on bottom
  const sortedTracks = [...videoTracks].sort((a, b) => {
    const aIndex = tracks.indexOf(a);
    const bIndex = tracks.indexOf(b);
    return aIndex - bIndex; // Lower index (top track) first = renders on top
  });

  for (const track of sortedTracks) {
    if (!isTrackVisible(track)) continue;

    const trackClips = clipsAtTime.filter(c => c.trackId === track.id);

    for (const clip of trackClips) {
      const layer = buildLayerFromClip(clip, time, getInterpolatedTransform, getInterpolatedEffects);
      if (layer) {
        layers.push(layer);
      }
    }
  }

  return layers;
}

// Helper: Build a single layer from a clip
function buildLayerFromClip(
  clip: TimelineClip,
  time: number,
  getInterpolatedTransform: (clipId: string, localTime: number) => any,
  getInterpolatedEffects: (clipId: string, localTime: number) => any
): Layer | null {
  const clipLocalTime = time - clip.startTime;
  const transform = getInterpolatedTransform(clip.id, clipLocalTime);
  const effects = getInterpolatedEffects(clip.id, clipLocalTime);

  // Handle nested compositions
  if (clip.isComposition && clip.nestedClips && clip.nestedTracks) {
    // For nested compositions, we need to get the rendered frame from nested clips
    // This is simplified - full implementation would recurse
    return null;
  }

  // Handle video/image clips
  if (clip.source?.videoElement || clip.source?.imageElement) {
    // For export, we need to use HTMLVideoElement (not WebCodecsPlayer)
    // because we control seeking via video.currentTime
    // WebCodecsPlayer has its own playback and doesn't follow currentTime
    const exportSource: LayerSource = {
      type: clip.source.videoElement ? 'video' : 'image',
      videoElement: clip.source.videoElement,
      imageElement: clip.source.imageElement,
      // webCodecsPlayer explicitly omitted - force HTMLVideoElement path during export
    };

    return {
      id: clip.id,
      name: clip.name || clip.id,
      source: exportSource,
      visible: true,
      opacity: transform.opacity ?? 1,
      blendMode: transform.blendMode ?? 'normal',
      // Compositor expects position/scale/rotation in this format
      position: {
        x: transform.x ?? 0,
        y: transform.y ?? 0,
        z: 0,
      },
      scale: {
        x: transform.scaleX ?? 1,
        y: transform.scaleY ?? 1,
      },
      rotation: transform.rotation ?? 0,
      effects: effects || [],
    };
  }

  return null;
}

export function ExportPanel() {
  const { duration, inPoint, outPoint, playheadPosition, startExport, setExportProgress, endExport } = useTimelineStore();
  const { getActiveComposition } = useMediaStore();
  const composition = getActiveComposition();

  // Encoder selection
  const [encoder, setEncoder] = useState<EncoderType>('webcodecs');

  // Shared settings
  const [width, setWidth] = useState(composition?.width ?? 1920);
  const [height, setHeight] = useState(composition?.height ?? 1080);
  const [customWidth, setCustomWidth] = useState(composition?.width ?? 1920);
  const [customHeight, setCustomHeight] = useState(composition?.height ?? 1080);
  const [useCustomResolution, setUseCustomResolution] = useState(false);
  const [fps, setFps] = useState(composition?.frameRate ?? 30);
  const [customFps, setCustomFps] = useState(30);
  const [useCustomFps, setUseCustomFps] = useState(false);
  const [useInOut, setUseInOut] = useState(true);
  const [filename, setFilename] = useState('export');

  // WebCodecs settings
  const [bitrate, setBitrate] = useState(15_000_000);
  const [containerFormat, setContainerFormat] = useState<ContainerFormat>('mp4');
  const [videoCodec, setVideoCodec] = useState<VideoCodec>('h264');
  const [codecSupport, setCodecSupport] = useState<Record<VideoCodec, boolean>>({
    h264: true, h265: false, vp9: false, av1: false
  });
  const [rateControl, setRateControl] = useState<'vbr' | 'cbr'>('vbr');

  // FFmpeg settings (default to ProRes which is most universally useful)
  const [ffmpegCodec, setFfmpegCodec] = useState<FFmpegVideoCodec>('prores');
  const [ffmpegContainer, setFfmpegContainer] = useState<FFmpegContainer>('mov');
  const [ffmpegPreset, setFfmpegPreset] = useState<string>('');
  const [proresProfile, setProresProfile] = useState<ProResProfile>('hq');
  const [dnxhrProfile, setDnxhrProfile] = useState<DnxhrProfile>('dnxhr_hq');
  // HAP not available in this FFmpeg build
  const [ffmpegQuality, setFfmpegQuality] = useState(18);
  const [ffmpegBitrate, setFfmpegBitrate] = useState(20_000_000);
  const [ffmpegRateControl, setFfmpegRateControl] = useState<'crf' | 'cbr' | 'vbr'>('crf');

  // FFmpeg loading state
  const [isFFmpegLoading, setIsFFmpegLoading] = useState(false);
  const [isFFmpegReady, setIsFFmpegReady] = useState(false);
  const [ffmpegLoadError, setFfmpegLoadError] = useState<string | null>(null);

  // Audio settings
  const [includeAudio, setIncludeAudio] = useState(true);
  const [audioSampleRate, setAudioSampleRate] = useState<44100 | 48000>(48000);
  const [audioBitrate, setAudioBitrate] = useState(256000);
  const [normalizeAudio, setNormalizeAudio] = useState(false);

  // Export state
  const [isExporting, setIsExporting] = useState(false);
  const [progress, setProgress] = useState<ExportProgress | null>(null);
  const [ffmpegProgress, setFfmpegProgress] = useState<FFmpegProgress | null>(null);
  const [exportPhase, setExportPhase] = useState<'idle' | 'rendering' | 'audio' | 'encoding'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [exporter, setExporter] = useState<FrameExporter | null>(null);

  // Check WebCodecs support
  const [isSupported, setIsSupported] = useState(true);
  const [isAudioSupported, setIsAudioSupported] = useState(true);
  const [audioCodec, setAudioCodec] = useState<AudioCodec | null>(null);

  // Check FFmpeg support
  const isFFmpegSupported = FFmpegBridge.isSupported();
  const isFFmpegMultiThreaded = FFmpegBridge.isMultiThreaded();

  useEffect(() => {
    setIsSupported(FrameExporter.isSupported());
    // Check audio encoder support and detect codec
    AudioEncoderWrapper.detectSupportedCodec().then(result => {
      if (result) {
        setIsAudioSupported(true);
        setAudioCodec(result.codec);
        console.log(`[ExportPanel] Audio codec detected: ${result.codec.toUpperCase()}`);
      } else {
        setIsAudioSupported(false);
        setIncludeAudio(false);
        console.warn('[ExportPanel] No audio encoding supported in this browser');
      }
    });
  }, []);

  // Check codec support when resolution changes
  useEffect(() => {
    const checkSupport = async () => {
      const actualWidth = useCustomResolution ? customWidth : width;
      const actualHeight = useCustomResolution ? customHeight : height;

      const support: Record<VideoCodec, boolean> = {
        h264: await FrameExporter.checkCodecSupport('h264', actualWidth, actualHeight),
        h265: await FrameExporter.checkCodecSupport('h265', actualWidth, actualHeight),
        vp9: await FrameExporter.checkCodecSupport('vp9', actualWidth, actualHeight),
        av1: await FrameExporter.checkCodecSupport('av1', actualWidth, actualHeight),
      };
      setCodecSupport(support);

      // If current codec is not supported, select first supported one
      const availableCodecs = FrameExporter.getVideoCodecs(containerFormat);
      if (!support[videoCodec]) {
        const firstSupported = availableCodecs.find(c => support[c.id]);
        if (firstSupported) {
          setVideoCodec(firstSupported.id);
        }
      }
    };
    checkSupport();
  }, [width, height, customWidth, customHeight, useCustomResolution, containerFormat, videoCodec]);

  // Update video codec when container changes
  useEffect(() => {
    const availableCodecs = FrameExporter.getVideoCodecs(containerFormat);
    if (!availableCodecs.find(c => c.id === videoCodec)) {
      setVideoCodec(availableCodecs[0].id);
    }
  }, [containerFormat, videoCodec]);

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

  // Load FFmpeg on demand
  const loadFFmpeg = useCallback(async () => {
    if (isFFmpegReady) return;

    setIsFFmpegLoading(true);
    setFfmpegLoadError(null);

    try {
      const ffmpeg = getFFmpegBridge();
      await ffmpeg.load();
      setIsFFmpegReady(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load FFmpeg';
      setFfmpegLoadError(msg);
      console.error('[ExportPanel] FFmpeg load error:', e);
    } finally {
      setIsFFmpegLoading(false);
    }
  }, [isFFmpegReady]);

  // Apply FFmpeg platform preset
  const applyFFmpegPreset = useCallback((presetId: string) => {
    const presetConfig = PLATFORM_PRESETS[presetId];
    if (!presetConfig) {
      setFfmpegPreset('');
      return;
    }

    setFfmpegCodec(presetConfig.codec);
    setFfmpegContainer(presetConfig.container);

    if (presetConfig.quality !== undefined) {
      setFfmpegRateControl('crf');
      setFfmpegQuality(presetConfig.quality);
    }
    if (presetConfig.bitrate !== undefined) {
      setFfmpegRateControl('vbr');
      setFfmpegBitrate(presetConfig.bitrate);
    }
    if (presetConfig.proresProfile) {
      setProresProfile(presetConfig.proresProfile);
    }
    if (presetConfig.dnxhrProfile) {
      setDnxhrProfile(presetConfig.dnxhrProfile);
    }
    // HAP not available in this build

    setFfmpegPreset(presetId);
  }, []);

  // Handle FFmpeg container change
  // NOTE: ASYNCIFY build only supports mov, mkv, avi, mxf
  const handleFFmpegContainerChange = useCallback((newContainer: FFmpegContainer) => {
    setFfmpegContainer(newContainer);
    setFfmpegPreset('');

    const codecInfo = getCodecInfo(ffmpegCodec);
    if (codecInfo && !codecInfo.containers.includes(newContainer)) {
      // Select appropriate codec for container
      if (newContainer === 'mxf') {
        setFfmpegCodec('dnxhd');
      } else if (newContainer === 'mov') {
        setFfmpegCodec('prores');
      } else if (newContainer === 'mkv' || newContainer === 'avi') {
        setFfmpegCodec('mjpeg');
      }
    }
  }, [ffmpegCodec]);

  // Handle FFmpeg codec change
  const handleFFmpegCodecChange = useCallback((newCodec: FFmpegVideoCodec) => {
    setFfmpegCodec(newCodec);
    setFfmpegPreset('');

    const codecInfo = getCodecInfo(newCodec);
    if (codecInfo && !codecInfo.containers.includes(ffmpegContainer)) {
      setFfmpegContainer(codecInfo.containers[0]);
    }
  }, [ffmpegContainer]);

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
      console.error('[ExportPanel] Export failed:', e);
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
    if (encoder === 'webcodecs') {
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
      console.log('[ExportPanel] Rendering frames for FFmpeg...');
      const frames: Uint8Array[] = [];
      const totalFrames = Math.ceil((endTime - startTime) * exportFps);
      const frameDuration = 1 / exportFps;

      console.log(`[ExportPanel] Total frames: ${totalFrames}, duration: ${frameDuration.toFixed(4)}s per frame`);

      // Set engine to export mode and correct resolution
      engine.setExporting(true);
      engine.setResolution(actualWidth, actualHeight);

      const frameStartTime = performance.now();

      for (let i = 0; i < totalFrames; i++) {
        const time = startTime + i * frameDuration;

        if (i === 0) console.log('[ExportPanel] Frame 0: Starting seek...');

        // Seek all video clips to the exact frame time
        await seekAllClipsToTime(time);

        if (i === 0) console.log('[ExportPanel] Frame 0: Seek complete, waiting for decode...');

        // Small delay to ensure video frame is decoded (browser needs time after seek)
        await new Promise(resolve => setTimeout(resolve, 16));

        if (i === 0) console.log('[ExportPanel] Frame 0: Building layers...');

        // Build layers at this time and render
        const layers = buildLayersAtTime(time);

        if (i === 0) console.log(`[ExportPanel] Frame 0: Got ${layers.length} layers`);

        if (layers.length === 0) {
          console.warn(`[ExportPanel] No layers at time ${time.toFixed(3)}`);
        }

        engine.render(layers);

        if (i === 0) console.log('[ExportPanel] Frame 0: Render complete, reading pixels...');

        // Read pixels
        const pixels = await engine.readPixels();

        if (i === 0) console.log(`[ExportPanel] Frame 0: Got pixels: ${pixels ? pixels.length : 'null'}`);
        if (pixels) {
          // Make a COPY of pixels (not a view) to ensure each frame is unique
          const frameCopy = new Uint8Array(pixels.length);
          frameCopy.set(pixels);
          frames.push(frameCopy);

          // Debug: check first few pixels to see if content changes
          if (i < 3 || i % 30 === 0) {
            const sample = [pixels[0], pixels[1], pixels[2], pixels[3], pixels[1000], pixels[2000]];
            console.log(`[ExportPanel] Frame ${i} sample pixels: [${sample.join(', ')}]`);
          }
        }

        // Log progress every 30 frames or on first frame
        if (i === 0 || i % 30 === 0) {
          const elapsed = (performance.now() - frameStartTime) / 1000;
          const renderFps = (i + 1) / elapsed;
          console.log(`[ExportPanel] Frame ${i + 1}/${totalFrames} at ${time.toFixed(3)}s, ${renderFps.toFixed(1)} fps, ${layers.length} layers`);
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
        console.log('[ExportPanel] Extracting audio for FFmpeg...');

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
            console.log(`[ExportPanel] Audio extracted: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.numberOfChannels}ch, ${audioBuffer.length} samples`);
          } else {
            console.warn('[ExportPanel] Audio extraction returned empty or null buffer');
          }
        } catch (audioError) {
          console.warn('[ExportPanel] Audio extraction failed, continuing without audio:', audioError);
        }
      }

      // Encode with FFmpeg - this is the slow part (40-100%)
      setExportPhase('encoding');
      console.log(`[ExportPanel] Encoding ${frames.length} frames with FFmpeg...`);

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

      console.log('[ExportPanel] FFmpeg export complete');
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Export failed';
      setError(msg);
      console.error('[ExportPanel] FFmpeg export error:', e);
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

  // Get actual FPS value
  const actualFps = useCustomFps ? customFps : fps;

  // Format file size estimate - works for both encoders
  const estimatedSize = () => {
    const durationSec = endTime - startTime;
    if (durationSec <= 0) return '—';

    let estimatedBitrate: number;

    if (encoder === 'webcodecs') {
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
    if (includeAudio && encoder === 'webcodecs') {
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
          onClick={encoder === 'webcodecs' ? handleExport : handleFFmpegExport}
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
                  <option value="webcodecs">WebCodecs (GPU)</option>
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
                  value={encoder === 'webcodecs' ? containerFormat : ffmpegContainer}
                  onChange={(e) => {
                    if (encoder === 'webcodecs') {
                      setContainerFormat(e.target.value as ContainerFormat);
                    } else {
                      handleFFmpegContainerChange(e.target.value as FFmpegContainer);
                    }
                  }}
                  title="Click to change container format"
                >
                  {encoder === 'webcodecs' ? (
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
              {encoder === 'webcodecs' ? (
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
            {encoder === 'webcodecs' ? (
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
                  disabled={encoder === 'webcodecs' && !isAudioSupported}
                />
                Include Audio
              </label>
              {encoder === 'webcodecs' && !isAudioSupported && (
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
            {encoder === 'webcodecs' ? (
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
                width: `${encoder === 'webcodecs'
                  ? (progress?.percent ?? 0)
                  : (ffmpegProgress?.percent ?? 0)}%`
              }}
            />
          </div>
          <div className="export-progress-info">
            {encoder === 'webcodecs' ? (
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
          {encoder === 'webcodecs' && progress && progress.phase === 'video' && progress.estimatedTimeRemaining > 0 && (
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
