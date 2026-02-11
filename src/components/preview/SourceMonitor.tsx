// Source Monitor - displays raw media files (video/image) in the Preview panel

import { useEffect, useRef, useState, useCallback } from 'react';
import type { MediaFile } from '../../stores/mediaStore';

interface SourceMonitorProps {
  file: MediaFile;
  onClose: () => void;
}

export function SourceMonitor({ file, onClose }: SourceMonitorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(file.duration || 0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isScrubbing, setIsScrubbing] = useState(false);

  const isVideo = file.type === 'video';

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
        const video = videoRef.current;
        if (!video) return;
        if (video.paused) {
          video.play();
        } else {
          video.pause();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose, isVideo]);

  // Video event listeners
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isVideo) return;

    const onTimeUpdate = () => {
      if (!isScrubbing) {
        setCurrentTime(video.currentTime);
      }
    };
    const onLoadedMetadata = () => {
      setDuration(video.duration);
    };
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => setIsPlaying(false);

    video.addEventListener('timeupdate', onTimeUpdate);
    video.addEventListener('loadedmetadata', onLoadedMetadata);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('ended', onEnded);

    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate);
      video.removeEventListener('loadedmetadata', onLoadedMetadata);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('ended', onEnded);
    };
  }, [isVideo, isScrubbing]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      const video = videoRef.current;
      if (video) {
        video.pause();
        video.src = '';
        video.load();
      }
    };
  }, []);

  // Toggle play/pause on video click
  const handleVideoClick = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
    } else {
      video.pause();
    }
  }, []);

  // Scrub bar interaction
  const seekToPosition = useCallback((clientX: number) => {
    const video = videoRef.current;
    const bar = scrubRef.current;
    if (!video || !bar) return;

    const rect = bar.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const time = fraction * duration;
    video.currentTime = time;
    setCurrentTime(time);
  }, [duration]);

  const handleScrubMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsScrubbing(true);
    seekToPosition(e.clientX);

    const handleMouseMove = (moveEvent: MouseEvent) => {
      seekToPosition(moveEvent.clientX);
    };

    const handleMouseUp = () => {
      setIsScrubbing(false);
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
          <video
            ref={videoRef}
            src={file.url}
            className="source-monitor-video"
            onClick={handleVideoClick}
            playsInline
          />
        ) : (
          <img
            src={file.url}
            alt={file.name}
            className="source-monitor-image"
          />
        )}
      </div>

      {isVideo && (
        <div className="source-monitor-scrub" onMouseDown={handleScrubMouseDown} ref={scrubRef}>
          <span className="source-monitor-time">{formatTime(currentTime)}</span>
          <div className="source-monitor-scrub-track">
            <div className="source-monitor-scrub-fill" style={{ width: `${progress}%` }} />
            <div className="source-monitor-scrub-handle" style={{ left: `${progress}%` }} />
          </div>
          <span className="source-monitor-time">{formatTime(duration)}</span>
          <button
            className="source-monitor-play-btn"
            onClick={(e) => { e.stopPropagation(); handleVideoClick(); }}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
        </div>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const f = Math.floor((seconds % 1) * 100);
  return `${m}:${s.toString().padStart(2, '0')}.${f.toString().padStart(2, '0')}`;
}
