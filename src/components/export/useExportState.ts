// Export state management hook - encapsulates all ExportPanel state and initialization effects

import { useState, useEffect, useCallback } from 'react';
import { Logger } from '../../services/logger';
import { FrameExporter } from '../../engine/export';
import type { ExportProgress, VideoCodec, ContainerFormat } from '../../engine/export';
import { AudioEncoderWrapper, type AudioCodec } from '../../engine/audio';
import {
  getFFmpegBridge,
  FFmpegBridge,
  PLATFORM_PRESETS,
  getCodecInfo,
} from '../../engine/ffmpeg';
import type {
  FFmpegProgress,
  FFmpegVideoCodec,
  FFmpegContainer,
  ProResProfile,
  DnxhrProfile,
} from '../../engine/ffmpeg';
import type { Composition } from '../../stores/mediaStore';

const log = Logger.create('ExportState');

export type EncoderType = 'webcodecs' | 'htmlvideo' | 'ffmpeg';

export function useExportState(composition: Composition | undefined) {
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

  // --- Effects ---

  // Detect WebCodecs and audio support on mount
  useEffect(() => {
    setIsSupported(FrameExporter.isSupported());
    AudioEncoderWrapper.detectSupportedCodec().then(result => {
      if (result) {
        setIsAudioSupported(true);
        setAudioCodec(result.codec);
        log.info(`Audio codec detected: ${result.codec.toUpperCase()}`);
      } else {
        setIsAudioSupported(false);
        setIncludeAudio(false);
        log.warn('No audio encoding supported in this browser');
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

  // Update recommended bitrate when resolution changes
  useEffect(() => {
    setBitrate(FrameExporter.getRecommendedBitrate(width, height, fps));
  }, [width, height, fps]);

  // --- Handlers ---

  const handleResolutionChange = useCallback((value: string) => {
    const [w, h] = value.split('x').map(Number);
    setWidth(w);
    setHeight(h);
  }, []);

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
      log.error('FFmpeg load error', e);
    } finally {
      setIsFFmpegLoading(false);
    }
  }, [isFFmpegReady]);

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

    setFfmpegPreset(presetId);
  }, []);

  const handleFFmpegContainerChange = useCallback((newContainer: FFmpegContainer) => {
    setFfmpegContainer(newContainer);
    setFfmpegPreset('');

    const codecInfo = getCodecInfo(ffmpegCodec);
    if (codecInfo && !codecInfo.containers.includes(newContainer)) {
      if (newContainer === 'mxf') {
        setFfmpegCodec('dnxhd');
      } else if (newContainer === 'mov') {
        setFfmpegCodec('prores');
      } else if (newContainer === 'mkv' || newContainer === 'avi') {
        setFfmpegCodec('mjpeg');
      }
    }
  }, [ffmpegCodec]);

  const handleFFmpegCodecChange = useCallback((newCodec: FFmpegVideoCodec) => {
    setFfmpegCodec(newCodec);
    setFfmpegPreset('');

    const codecInfo = getCodecInfo(newCodec);
    if (codecInfo && !codecInfo.containers.includes(ffmpegContainer)) {
      setFfmpegContainer(codecInfo.containers[0]);
    }
  }, [ffmpegContainer]);

  return {
    // Encoder
    encoder, setEncoder,
    // Resolution
    width, setWidth, height, setHeight,
    customWidth, setCustomWidth, customHeight, setCustomHeight,
    useCustomResolution, setUseCustomResolution,
    // Frame rate
    fps, setFps, customFps, setCustomFps, useCustomFps, setUseCustomFps,
    // Range
    useInOut, setUseInOut,
    // Filename
    filename, setFilename,
    // WebCodecs
    bitrate, setBitrate, containerFormat, setContainerFormat,
    videoCodec, setVideoCodec, codecSupport, rateControl, setRateControl,
    // FFmpeg
    ffmpegCodec, ffmpegContainer, ffmpegPreset,
    proresProfile, setProresProfile, dnxhrProfile, setDnxhrProfile,
    ffmpegQuality, setFfmpegQuality, ffmpegBitrate, setFfmpegBitrate,
    ffmpegRateControl, setFfmpegRateControl,
    // FFmpeg loading
    isFFmpegLoading, isFFmpegReady, ffmpegLoadError,
    // Audio
    includeAudio, setIncludeAudio,
    audioSampleRate, setAudioSampleRate,
    audioBitrate, setAudioBitrate,
    normalizeAudio, setNormalizeAudio,
    // Export state
    isExporting, setIsExporting,
    progress, setProgress,
    ffmpegProgress, setFfmpegProgress,
    exportPhase, setExportPhase,
    error, setError,
    exporter, setExporter,
    // Support detection
    isSupported, isAudioSupported, audioCodec,
    isFFmpegSupported, isFFmpegMultiThreaded,
    // Handlers
    handleResolutionChange, loadFFmpeg, applyFFmpegPreset,
    handleFFmpegContainerChange, handleFFmpegCodecChange,
  };
}
