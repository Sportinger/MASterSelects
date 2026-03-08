// Source Monitor - displays raw media files (video/image) in the Preview panel

import { useCallback, useEffect, useRef, useState } from 'react';
import { WebCodecsPlayer } from '../../engine/WebCodecsPlayer';
import type { MediaFile } from '../../stores/mediaStore';

interface SourceMonitorProps {
  file: MediaFile;
  onClose: () => void;
}

type SourceBackend = 'webcodecs' | 'html';

export function SourceMonitor({ file, onClose }: SourceMonitorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const webCodecsPlayerRef = useRef<WebCodecsPlayer | null>(null);

  const isVideo = file.type === 'video';
  const fps = file.fps || 30;
  const webCodecsAvailable = isVideo && !!file.file;

  const [backend, setBackend] = useState<SourceBackend>(
    webCodecsAvailable ? 'webcodecs' : 'html'
  );
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(file.duration || 0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isScrubbing, setIsScrubbing] = useState(false);

  const destroyWebCodecsPlayer = useCallback(() => {
    webCodecsPlayerRef.current?.destroy();
    webCodecsPlayerRef.current = null;
    setBackendReady(false);
    setIsPlaying(false);
  }, []);

  const drawWebCodecsFrame = useCallback((frame: VideoFrame | null) => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) {
      return;
    }

    if (canvas.width !== frame.displayWidth || canvas.height !== frame.displayHeight) {
      canvas.width = frame.displayWidth;
      canvas.height = frame.displayHeight;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(frame as unknown as CanvasImageSource, 0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => {
    setBackend(webCodecsAvailable ? 'webcodecs' : 'html');
    setBackendError(null);
    setBackendReady(false);
    setCurrentTime(0);
    setDuration(file.duration || 0);
    setIsPlaying(false);
  }, [file.id, file.duration, webCodecsAvailable]);

  // HTML video event listeners
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isVideo || backend !== 'html') return;

    const onTimeUpdate = () => {
      if (!isScrubbing) {
        setCurrentTime(video.currentTime);
      }
    };
    const onLoadedMetadata = () => {
      setDuration(video.duration);
      setBackendReady(true);
      if (currentTime > 0.01) {
        video.currentTime = Math.min(currentTime, video.duration || currentTime);
      }
    };
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => setIsPlaying(false);
    const onError = () => setBackendError('HTML preview failed to load');

    video.addEventListener('timeupdate', onTimeUpdate);
    video.addEventListener('loadedmetadata', onLoadedMetadata);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('ended', onEnded);
    video.addEventListener('error', onError);

    setBackendReady(video.readyState >= 1);
    if (video.readyState >= 1) {
      setDuration(video.duration || file.duration || 0);
    }

    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate);
      video.removeEventListener('loadedmetadata', onLoadedMetadata);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('ended', onEnded);
      video.removeEventListener('error', onError);
    };
  }, [backend, file.duration, isScrubbing, isVideo]);

  // WebCodecs source monitor
  useEffect(() => {
    if (!isVideo || backend !== 'webcodecs') {
      destroyWebCodecsPlayer();
      return;
    }

    if (!file.file) {
      setBackend('html');
      return;
    }

    let disposed = false;
    setBackendReady(false);
    setBackendError(null);

    const player = new WebCodecsPlayer({
      loop: false,
      onFrame: (frame) => {
        if (disposed) {
          return;
        }
        drawWebCodecsFrame(frame);
        setCurrentTime(player.currentTime);
        setDuration(player.duration || file.duration || 0);
        setIsPlaying(player.isPlaying);
      },
      onReady: () => {
        if (disposed) {
          return;
        }
        setBackendReady(true);
        setDuration(player.duration || file.duration || 0);
      },
      onError: (error) => {
        if (disposed) {
          return;
        }
        setBackendError(error.message || 'WebCodecs preview failed');
        setBackend('html');
      },
    });
    webCodecsPlayerRef.current = player;

    void (async () => {
      try {
        await player.loadFile(file.file!);
        if (disposed) {
          return;
        }
        setBackendReady(true);
        setDuration(player.duration || file.duration || 0);

        if (currentTime > 0.01) {
          player.seek(Math.min(currentTime, player.duration || currentTime));
        } else if (player.getCurrentFrame()) {
          drawWebCodecsFrame(player.getCurrentFrame());
        }
      } catch (error) {
        if (disposed) {
          return;
        }
        setBackendError(
          error instanceof Error ? error.message : 'WebCodecs preview failed'
        );
        setBackend('html');
      }
    })();

    return () => {
      disposed = true;
      player.destroy();
      if (webCodecsPlayerRef.current === player) {
        webCodecsPlayerRef.current = null;
      }
    };
  }, [backend, destroyWebCodecsPlayer, drawWebCodecsFrame, file.duration, file.file, isVideo]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      destroyWebCodecsPlayer();
      const video = videoRef.current;
      if (video) {
        video.pause();
        video.src = '';
        video.load();
      }
    };
  }, [destroyWebCodecsPlayer]);

  const togglePlayback = useCallback(() => {
    if (!isVideo) {
      return;
    }

    if (backend === 'webcodecs' && webCodecsPlayerRef.current) {
      const player = webCodecsPlayerRef.current;
      if (player.isPlaying) {
        player.pause();
        setIsPlaying(false);
      } else {
        player.play();
        setIsPlaying(true);
      }
      return;
    }

    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }, [backend, isVideo]);

  // Keyboard handler: Space = play/pause, Escape = close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const active = document.activeElement;
      const isInput = active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement ||
        active?.getAttribute('contenteditable') === 'true';
      if (isInput) return;

      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === ' ' && isVideo) {
        e.preventDefault();
        togglePlayback();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isVideo, onClose, togglePlayback]);

  const seekSourceMonitor = useCallback((time: number, precise: boolean) => {
    const clampedTime = Math.max(0, Math.min(duration || time, time));
    setCurrentTime(clampedTime);

    if (backend === 'webcodecs' && webCodecsPlayerRef.current) {
      const player = webCodecsPlayerRef.current;
      player.pause();
      setIsPlaying(false);
      if (precise) {
        player.seek(clampedTime);
      } else {
        player.fastSeek(clampedTime);
      }
      return;
    }

    const video = videoRef.current;
    if (!video) {
      return;
    }
    video.currentTime = clampedTime;
  }, [backend, duration]);

  // Frame step
  const stepFrame = useCallback((direction: 1 | -1) => {
    const frameDuration = 1 / fps;
    seekSourceMonitor(currentTime + direction * frameDuration, true);
  }, [currentTime, fps, seekSourceMonitor]);

  // Go to start / end
  const goToStart = useCallback(() => {
    seekSourceMonitor(0, true);
  }, [seekSourceMonitor]);

  const goToEnd = useCallback(() => {
    seekSourceMonitor(duration, true);
  }, [duration, seekSourceMonitor]);

  // Scrub bar interaction
  const seekToPosition = useCallback((clientX: number, precise: boolean) => {
    const bar = scrubRef.current;
    if (!bar) return;

    const rect = bar.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    seekSourceMonitor(fraction * duration, precise);
  }, [duration, seekSourceMonitor]);

  const handleScrubMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsScrubbing(true);
    seekToPosition(e.clientX, false);

    const handleMouseMove = (moveEvent: MouseEvent) => {
      seekToPosition(moveEvent.clientX, false);
    };

    const handleMouseUp = (upEvent: MouseEvent) => {
      setIsScrubbing(false);
      seekToPosition(upEvent.clientX, true);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [seekToPosition]);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="source-monitor">
      <div className="source-monitor-media">
        {isVideo ? (
          <>
            <video
              ref={videoRef}
              src={file.url}
              className="source-monitor-video"
              onClick={togglePlayback}
              playsInline
              style={{ display: backend === 'html' ? 'block' : 'none' }}
            />
            <canvas
              ref={canvasRef}
              className="source-monitor-canvas"
              onClick={togglePlayback}
              style={{ display: backend === 'webcodecs' ? 'block' : 'none' }}
            />
          </>
        ) : (
          <img
            src={file.url}
            alt={file.name}
            className="source-monitor-image"
          />
        )}
      </div>

      {isVideo && (
        <div className="source-monitor-toolbar">
          <div className="source-monitor-transport">
            <button className="btn btn-sm" onClick={goToStart} title="Go to start">
              Start
            </button>
            <button className="btn btn-sm" onClick={() => stepFrame(-1)} title="Previous frame">
              Prev
            </button>
            <button
              className={`btn btn-sm ${isPlaying ? 'btn-active' : ''}`}
              onClick={togglePlayback}
              title={isPlaying ? 'Pause [Space]' : 'Play [Space]'}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button className="btn btn-sm" onClick={() => stepFrame(1)} title="Next frame">
              Next
            </button>
            <button className="btn btn-sm" onClick={goToEnd} title="Go to end">
              End
            </button>
          </div>

          <div className="source-monitor-backend">
            <button
              className={`btn btn-sm ${backend === 'webcodecs' ? 'btn-active' : ''}`}
              onClick={() => setBackend('webcodecs')}
              disabled={!webCodecsAvailable}
              title={webCodecsAvailable ? 'Use WebCodecs source preview' : 'WebCodecs requires a local file handle'}
            >
              WebCodecs
            </button>
            <button
              className={`btn btn-sm ${backend === 'html' ? 'btn-active' : ''}`}
              onClick={() => setBackend('html')}
              title="Use HTML video source preview"
            >
              HTML
            </button>
          </div>

          <div className="source-monitor-timecode">
            <span className="timeline-time">{formatTimecode(currentTime, fps)}</span>
            <span className="source-monitor-time-sep">/</span>
            <span className="timeline-time">{formatTimecode(duration, fps)}</span>
          </div>

          <div className="source-monitor-scrub" onMouseDown={handleScrubMouseDown} ref={scrubRef}>
            <div className="source-monitor-scrub-track">
              <div className="source-monitor-scrub-fill" style={{ width: `${progress}%` }} />
              <div className="source-monitor-scrub-handle" style={{ left: `${progress}%` }} />
            </div>
          </div>

          <div className="source-monitor-status">
            {backendReady ? backend : 'loading'}
            {backendError ? ` - ${backendError}` : ''}
          </div>
        </div>
      )}
    </div>
  );
}

function formatTimecode(seconds: number, fps: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const f = Math.floor((seconds % 1) * fps);
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}:${f.toString().padStart(2, '0')}`;
  }
  return `${m}:${s.toString().padStart(2, '0')}:${f.toString().padStart(2, '0')}`;
}
