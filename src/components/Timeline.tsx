// Timeline component - Video editing timeline with tracks and clips

import { useRef, useState, useCallback, useEffect } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import { useMixerStore } from '../stores/mixerStore';
import { engine } from '../engine/WebGPUEngine';

export function Timeline() {
  const {
    tracks,
    clips,
    playheadPosition,
    duration,
    zoom,
    scrollX,
    isPlaying,
    selectedClipId,
    addTrack,
    addClip,
    moveClip,
    trimClip,
    removeClip,
    selectClip,
    setPlayheadPosition,
    setZoom,
    setScrollX,
    play,
    pause,
    stop,
    getSnappedPosition,
    findNonOverlappingPosition,
  } = useTimelineStore();


  const timelineRef = useRef<HTMLDivElement>(null);
  const trackLanesRef = useRef<HTMLDivElement>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);

  // Premiere-style clip dragging state
  const [clipDrag, setClipDrag] = useState<{
    clipId: string;
    originalStartTime: number;
    originalTrackId: string;
    grabOffsetX: number;  // Where on the clip we grabbed (in pixels)
    currentX: number;     // Current mouse X position
    currentTrackId: string;
    snappedTime: number | null;  // Snapped position (if snapping)
    isSnapping: boolean;         // Whether currently snapping
  } | null>(null);

  // Clip trimming state
  const [clipTrim, setClipTrim] = useState<{
    clipId: string;
    edge: 'left' | 'right';
    originalStartTime: number;
    originalDuration: number;
    originalInPoint: number;
    originalOutPoint: number;
    startX: number;
    currentX: number;
    altKey: boolean;  // If true, don't trim linked clip
  } | null>(null);

  // External file drag preview state
  const [externalDrag, setExternalDrag] = useState<{
    trackId: string;
    startTime: number;
    x: number;
    y: number;
    audioTrackId?: string;  // Preview for linked audio clip
    isVideo?: boolean;      // Is the dragged file a video?
  } | null>(null);
  const dragCounterRef = useRef(0); // Track drag enter/leave balance

  // Space key to toggle play/pause (global, works regardless of focus)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle space if not typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        if (isPlaying) {
          pause();
        } else {
          play();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, play, pause]);

  // Track last seek time to throttle during scrubbing
  const lastSeekRef = useRef<{ [clipId: string]: number }>({});
  const pendingSeekRef = useRef<{ [clipId: string]: number }>({});
  const lastCacheTimeRef = useRef<number>(0);

  // Apply pending seeks when scrubbing stops
  useEffect(() => {
    if (isDraggingPlayhead) return;

    // Apply any pending seeks
    Object.entries(pendingSeekRef.current).forEach(([clipId, seekTime]) => {
      const clip = clips.find(c => c.id === clipId);
      if (clip?.source?.videoElement) {
        clip.source.videoElement.currentTime = seekTime;
      }
    });
    pendingSeekRef.current = {};
  }, [isDraggingPlayhead, clips]);

  // Sync timeline playback with Preview - update mixer layers based on clips at playhead
  // IMPORTANT: This uses batched updates to prevent flickering from race conditions
  useEffect(() => {
    const clipsAtTime = getClipsAtTime(playheadPosition);
    const currentLayers = useMixerStore.getState().layers;

    // Get video tracks sorted by index (for layer order)
    const videoTracks = tracks.filter(t => t.type === 'video');

    // Build new layers array atomically - don't setState inside loop!
    const newLayers = [...currentLayers];
    let layersChanged = false;

    // Process each video layer and collect updates
    videoTracks.forEach((track, layerIndex) => {
      const clip = clipsAtTime.find(c => c.trackId === track.id);
      const layer = currentLayers[layerIndex];

      if (clip?.source?.videoElement) {
        // Seek video to correct position within clip
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const video = clip.source.videoElement;
        const timeDiff = Math.abs(video.currentTime - clipTime);

        // Only seek if difference is significant
        if (timeDiff > 0.05) {
          const now = performance.now();
          const lastSeek = lastSeekRef.current[clip.id] || 0;

          // Throttle seeks during scrubbing - wait for previous seek to settle
          if (isDraggingPlayhead && now - lastSeek < 80) {
            // Store pending seek, will be applied when scrubbing stops
            pendingSeekRef.current[clip.id] = clipTime;
          } else {
            // Use fastSeek during scrubbing for smoother experience (seeks to nearest keyframe)
            if (isDraggingPlayhead && 'fastSeek' in video) {
              video.fastSeek(clipTime);
            } else {
              video.currentTime = clipTime;
            }
            lastSeekRef.current[clip.id] = now;
            delete pendingSeekRef.current[clip.id];
          }
        }

        if (isPlaying && video.paused) {
          video.play().catch(() => {});
        } else if (!isPlaying && !video.paused) {
          video.pause();
        }

        // Cache frame for smooth scrubbing (only during normal playback, throttled)
        // Don't cache while scrubbing - the video is seeking and frames aren't reliable
        const now = performance.now();
        if (!isDraggingPlayhead && !video.seeking && video.readyState >= 2 && now - lastCacheTimeRef.current > 100) {
          engine.cacheFrameAtTime(video, clipTime);
          lastCacheTimeRef.current = now;
        }

        // Check if layer needs update
        const transform = clip.transform;
        const needsUpdate = !layer ||
          layer.source?.videoElement !== video ||
          layer.opacity !== transform.opacity ||
          layer.blendMode !== transform.blendMode ||
          layer.position.x !== transform.position.x ||
          layer.position.y !== transform.position.y ||
          layer.scale.x !== transform.scale.x ||
          layer.scale.y !== transform.scale.y ||
          layer.rotation !== (transform.rotation.z * Math.PI / 180);

        if (needsUpdate) {
          newLayers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: track.visible,
            opacity: transform.opacity,
            blendMode: transform.blendMode,
            source: {
              type: 'video',
              videoElement: video,
            },
            effects: [],
            position: { x: transform.position.x, y: transform.position.y },
            scale: { x: transform.scale.x, y: transform.scale.y },
            rotation: transform.rotation.z * Math.PI / 180,
          };
          layersChanged = true;
        }
      } else if (clip?.source?.imageElement) {
        // Handle image clips
        const img = clip.source.imageElement;
        const transform = clip.transform;
        const needsUpdate = !layer ||
          layer.source?.imageElement !== img ||
          layer.opacity !== transform.opacity ||
          layer.blendMode !== transform.blendMode ||
          layer.position.x !== transform.position.x ||
          layer.position.y !== transform.position.y ||
          layer.scale.x !== transform.scale.x ||
          layer.scale.y !== transform.scale.y ||
          layer.rotation !== (transform.rotation.z * Math.PI / 180);

        if (needsUpdate) {
          newLayers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: track.visible,
            opacity: transform.opacity,
            blendMode: transform.blendMode,
            source: {
              type: 'image',
              imageElement: img,
            },
            effects: [],
            position: { x: transform.position.x, y: transform.position.y },
            scale: { x: transform.scale.x, y: transform.scale.y },
            rotation: transform.rotation.z * Math.PI / 180,
          };
          layersChanged = true;
        }
      } else {
        // No clip at this position - clear the layer
        if (layer?.source) {
          newLayers[layerIndex] = undefined as any;
          layersChanged = true;
        }
      }
    });

    // Single atomic update for all layer changes - prevents flickering!
    if (layersChanged) {
      useMixerStore.setState({ layers: newLayers });
    }

    // Handle audio tracks - sync audio elements with playhead
    const audioTracks = tracks.filter(t => t.type === 'audio');
    audioTracks.forEach((track) => {
      const clip = clipsAtTime.find(c => c.trackId === track.id);

      if (clip?.source?.audioElement) {
        const audio = clip.source.audioElement;
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const timeDiff = Math.abs(audio.currentTime - clipTime);

        // Sync audio position if out of sync (but not while scrubbing to avoid choppy audio)
        if (timeDiff > 0.1 && !isDraggingPlayhead) {
          audio.currentTime = clipTime;
        }

        // Handle mute state
        audio.muted = track.muted;

        // Play/pause audio based on timeline state (pause while scrubbing)
        const shouldPlay = isPlaying && !track.muted && !isDraggingPlayhead;
        if (shouldPlay && audio.paused) {
          audio.play().catch(() => {});
        } else if (!shouldPlay && !audio.paused) {
          audio.pause();
        }
      }
    });

    // Also pause audio from clips that are no longer at playhead
    clips.forEach(clip => {
      if (clip.source?.audioElement) {
        const isAtPlayhead = clipsAtTime.some(c => c.id === clip.id);
        if (!isAtPlayhead && !clip.source.audioElement.paused) {
          clip.source.audioElement.pause();
        }
      }
    });
  }, [playheadPosition, clips, tracks, isPlaying, isDraggingPlayhead]);

  // Get clips at time helper
  const getClipsAtTime = useCallback((time: number) => {
    return clips.filter(c => time >= c.startTime && time < c.startTime + c.duration);
  }, [clips]);

  // Playback loop
  useEffect(() => {
    if (!isPlaying) return;

    let lastTime = performance.now();
    let animationId: number;

    const tick = (now: number) => {
      const delta = (now - lastTime) / 1000;
      lastTime = now;

      const { playheadPosition, duration } = useTimelineStore.getState();
      let newPosition = playheadPosition + delta;

      if (newPosition >= duration) {
        newPosition = 0; // Loop
      }

      setPlayheadPosition(newPosition);
      animationId = requestAnimationFrame(tick);
    };

    animationId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, setPlayheadPosition]);

  // Time to pixel conversion
  const timeToPixel = (time: number) => time * zoom;
  const pixelToTime = (pixel: number) => pixel / zoom;

  // Format time as MM:SS.ms
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  
  
  // Handle shift+mousewheel on track header to resize height
  // Handle alt+mousewheel on track header to resize all tracks of same type
  const handleTrackHeaderWheel = (e: React.WheelEvent, trackId: string) => {
    const track = tracks.find(t => t.id === trackId);
    if (!track) return;

    if (e.altKey) {
      // Alt + mousewheel: scale all tracks of same type (video or audio)
      e.preventDefault();
      e.stopPropagation();
      const delta = e.deltaY > 0 ? -10 : 10;
      useTimelineStore.getState().scaleTracksOfType(track.type, delta);
    } else if (e.shiftKey) {
      // Shift + mousewheel: scale individual track
      e.preventDefault();
      e.stopPropagation();
      const delta = e.deltaY > 0 ? -10 : 10;
      useTimelineStore.getState().setTrackHeight(trackId, track.height + delta);
    }
  };

  // Render waveform for audio clips
  const renderWaveform = (waveform: number[], width: number, height: number) => {
    if (!waveform || waveform.length === 0) return null;
    
    const barWidth = Math.max(1, width / waveform.length);
    const bars = waveform.map((value, i) => {
      const barHeight = Math.max(2, value * (height - 8));
      return (
        <div
          key={i}
          className="waveform-bar"
          style={{
            left: i * barWidth,
            height: barHeight,
            width: Math.max(1, barWidth - 1),
          }}
        />
      );
    });
    
    return (
      <div className="waveform-container" style={{ width, height }}>
        {bars}
      </div>
    );
  };

  // Handle time ruler mousedown - jump playhead immediately and enter drag mode
  const handleRulerMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();

    // Jump playhead immediately
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left + scrollX;
    const time = pixelToTime(x);
    setPlayheadPosition(Math.max(0, Math.min(time, duration)));

    // Enter drag mode immediately
    setIsDraggingPlayhead(true);
  };

  // Handle playhead drag
  const handlePlayheadMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDraggingPlayhead(true);
  };

  useEffect(() => {
    if (!isDraggingPlayhead) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!timelineRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const time = pixelToTime(x);
      setPlayheadPosition(Math.max(0, Math.min(time, duration)));
    };

    const handleMouseUp = () => {
      setIsDraggingPlayhead(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingPlayhead, scrollX, duration, setPlayheadPosition, pixelToTime]);

  // Premiere-style clip drag - mouse down on clip
  const handleClipMouseDown = (e: React.MouseEvent, clipId: string) => {
    if (e.button !== 0) return; // Only left click
    e.stopPropagation();
    e.preventDefault(); // Prevent text selection

    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    const clipElement = e.currentTarget as HTMLElement;
    const clipRect = clipElement.getBoundingClientRect();
    const grabOffsetX = e.clientX - clipRect.left;

    selectClip(clipId);
    setClipDrag({
      clipId,
      originalStartTime: clip.startTime,
      originalTrackId: clip.trackId,
      grabOffsetX,
      currentX: e.clientX,
      currentTrackId: clip.trackId,
      snappedTime: null,
      isSnapping: false,
    });
  };

  // Handle clip dragging
  useEffect(() => {
    if (!clipDrag) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!trackLanesRef.current || !timelineRef.current) return;

      // Find which track the mouse is over
      const lanesRect = trackLanesRef.current.getBoundingClientRect();
      const mouseY = e.clientY - lanesRect.top;

      // Calculate cumulative track positions
      let currentY = 24; // Time ruler height
      let newTrackId = clipDrag.currentTrackId;

      for (const track of tracks) {
        if (mouseY >= currentY && mouseY < currentY + track.height) {
          newTrackId = track.id;
          break;
        }
        currentY += track.height;
      }

      // Calculate current drag position in time
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX - clipDrag.grabOffsetX;
      const rawTime = Math.max(0, pixelToTime(x));

      // Check for snapping
      const { startTime: snappedTime, snapped } = getSnappedPosition(clipDrag.clipId, rawTime, newTrackId);

      setClipDrag(prev => prev ? {
        ...prev,
        currentX: e.clientX,
        currentTrackId: newTrackId,
        snappedTime: snapped ? snappedTime : null,
        isSnapping: snapped,
      } : null);
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (!clipDrag || !timelineRef.current) return;

      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX - clipDrag.grabOffsetX;
      const newStartTime = Math.max(0, pixelToTime(x));

      // Move clip to new position and track (store handles snapping and collision)
      moveClip(clipDrag.clipId, newStartTime, clipDrag.currentTrackId);
      setClipDrag(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [clipDrag, tracks, scrollX, moveClip, pixelToTime, getSnappedPosition]);

  // Handle trim start (mousedown on trim handle)
  const handleTrimStart = (e: React.MouseEvent, clipId: string, edge: 'left' | 'right') => {
    e.stopPropagation();
    e.preventDefault();

    const clip = clips.find(c => c.id === clipId);
    if (!clip) return;

    selectClip(clipId);
    setClipTrim({
      clipId,
      edge,
      originalStartTime: clip.startTime,
      originalDuration: clip.duration,
      originalInPoint: clip.inPoint,
      originalOutPoint: clip.outPoint,
      startX: e.clientX,
      currentX: e.clientX,
      altKey: e.altKey,
    });
  };

  // Handle trim dragging
  useEffect(() => {
    if (!clipTrim) return;

    const handleMouseMove = (e: MouseEvent) => {
      setClipTrim(prev => prev ? { ...prev, currentX: e.clientX, altKey: e.altKey } : null);
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (!clipTrim) return;

      const clip = clips.find(c => c.id === clipTrim.clipId);
      if (!clip) {
        setClipTrim(null);
        return;
      }

      // Calculate the delta in time
      const deltaX = clipTrim.currentX - clipTrim.startX;
      const deltaTime = pixelToTime(deltaX);

      // Get source duration limit
      const maxDuration = clip.source?.naturalDuration || clip.duration;

      let newStartTime = clipTrim.originalStartTime;
      let newInPoint = clipTrim.originalInPoint;
      let newOutPoint = clipTrim.originalOutPoint;

      if (clipTrim.edge === 'left') {
        // Trimming left edge - changes startTime and inPoint
        const maxTrim = clipTrim.originalDuration - 0.1; // Keep at least 0.1s
        const minTrim = -clipTrim.originalInPoint; // Can't go before source start
        const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));

        newStartTime = clipTrim.originalStartTime + clampedDelta;
        newInPoint = clipTrim.originalInPoint + clampedDelta;
      } else {
        // Trimming right edge - changes outPoint only
        const maxExtend = maxDuration - clipTrim.originalOutPoint; // Can't go past source end
        const minTrim = -(clipTrim.originalDuration - 0.1); // Keep at least 0.1s
        const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));

        newOutPoint = clipTrim.originalOutPoint + clampedDelta;
      }

      // Apply trim to this clip
      trimClip(clip.id, newInPoint, newOutPoint);
      if (clipTrim.edge === 'left') {
        // When Alt is held, skip moving linked clips
        moveClip(clip.id, Math.max(0, newStartTime), clip.trackId, clipTrim.altKey);
      }

      // Also trim linked clip unless Alt was held
      if (!clipTrim.altKey && clip.linkedClipId) {
        const linkedClip = clips.find(c => c.id === clip.linkedClipId);
        if (linkedClip) {
          const linkedMaxDuration = linkedClip.source?.naturalDuration || linkedClip.duration;

          if (clipTrim.edge === 'left') {
            const linkedNewInPoint = Math.max(0, Math.min(linkedMaxDuration - 0.1, newInPoint));
            trimClip(linkedClip.id, linkedNewInPoint, linkedClip.outPoint);
            // skipLinked=true since we're manually handling the linked clip
            moveClip(linkedClip.id, Math.max(0, newStartTime), linkedClip.trackId, true);
          } else {
            const linkedNewOutPoint = Math.max(0.1, Math.min(linkedMaxDuration, newOutPoint));
            trimClip(linkedClip.id, linkedClip.inPoint, linkedNewOutPoint);
          }
        }
      }

      setClipTrim(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [clipTrim, clips, pixelToTime, trimClip, moveClip]);

  // Handle external file drag enter on track
  const handleTrackDragEnter = (e: React.DragEvent, trackId: string) => {
    e.preventDefault();
    dragCounterRef.current++;

    // Only show preview for files
    if (e.dataTransfer.types.includes('Files')) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const startTime = pixelToTime(x);
      setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY });
    }
  };

  // Handle external file drag over track
  const handleTrackDragOver = (e: React.DragEvent, trackId: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';

    // Update preview position
    if (e.dataTransfer.types.includes('Files') && timelineRef.current) {
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const startTime = pixelToTime(x);

      // Check if dragging to a video track - if so, find available audio track for preview
      const targetTrack = tracks.find(t => t.id === trackId);
      const isVideoTrack = targetTrack?.type === 'video';

      // For video tracks, find an available audio track for the linked audio clip preview
      // Use 5 seconds as estimated duration for preview
      let audioTrackId: string | undefined;
      if (isVideoTrack) {
        const estimatedDuration = 5;
        const audioTracks = tracks.filter(t => t.type === 'audio');
        const endTime = startTime + estimatedDuration;

        // Find first available audio track
        for (const aTrack of audioTracks) {
          const trackClips = clips.filter(c => c.trackId === aTrack.id);
          const hasOverlap = trackClips.some(clip => {
            const clipEnd = clip.startTime + clip.duration;
            return !(endTime <= clip.startTime || startTime >= clipEnd);
          });
          if (!hasOverlap) {
            audioTrackId = aTrack.id;
            break;
          }
        }
        // If no available track, show on a "new track" area (we'll indicate this)
        if (!audioTrackId) {
          audioTrackId = '__new_audio_track__';
        }
      }

      setExternalDrag({
        trackId,
        startTime,
        x: e.clientX,
        y: e.clientY,
        audioTrackId,
        isVideo: isVideoTrack
      });
    }
  };

  // Handle external file drag leave
  const handleTrackDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current--;

    if (dragCounterRef.current === 0) {
      setExternalDrag(null);
    }
  };

  // Handle external file drop on track
  const handleTrackDrop = (e: React.DragEvent, trackId: string) => {
    e.preventDefault();
    dragCounterRef.current = 0;
    setExternalDrag(null);

    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/') || file.type.startsWith('audio/') || file.type.startsWith('image/')) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);
        addClip(trackId, file, Math.max(0, startTime));
      }
    }
  };

  // Zoom with mouse wheel (Ctrl or Alt)
  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.altKey) {
      // Ctrl + mousewheel OR Alt + mousewheel: zoom timeline view
      e.preventDefault();
      const delta = e.deltaY > 0 ? -5 : 5;
      setZoom(zoom + delta);
    } else {
      setScrollX(scrollX + e.deltaX);
    }
  };

  // Render time ruler
  const renderTimeRuler = () => {
    const width = timeToPixel(duration);
    const markers: React.ReactElement[] = [];

    // Calculate marker interval based on zoom
    let interval = 1; // 1 second
    if (zoom < 20) interval = 5;
    if (zoom < 10) interval = 10;
    if (zoom > 100) interval = 0.5;

    for (let t = 0; t <= duration; t += interval) {
      const x = timeToPixel(t);
      const isMainMarker = t % (interval * 2) === 0 || interval >= 5;

      markers.push(
        <div
          key={t}
          className={`time-marker ${isMainMarker ? 'main' : 'sub'}`}
          style={{ left: x }}
        >
          {isMainMarker && <span className="time-label">{formatTime(t)}</span>}
        </div>
      );
    }

    return (
      <div
        className="time-ruler"
        style={{ width }}
        onMouseDown={handleRulerMouseDown}
      >
        {markers}
      </div>
    );
  };

  // Render a clip - now with Premiere-style direct dragging and trimming
  const renderClip = (clip: typeof clips[0], trackId: string) => {
    const isSelected = selectedClipId === clip.id;
    const isDragging = clipDrag?.clipId === clip.id;
    const isTrimming = clipTrim?.clipId === clip.id;
    const thumbnails = clip.thumbnails || [];

    // Check if this clip is linked to the dragging/trimming clip
    const draggedClip = clipDrag ? clips.find(c => c.id === clipDrag.clipId) : null;
    const trimmedClip = clipTrim ? clips.find(c => c.id === clipTrim.clipId) : null;
    const isLinkedToDragging = clipDrag && draggedClip && (
      clip.linkedClipId === clipDrag.clipId ||
      draggedClip.linkedClipId === clip.id
    );
    const isLinkedToTrimming = clipTrim && !clipTrim.altKey && trimmedClip && (
      clip.linkedClipId === clipTrim.clipId ||
      trimmedClip.linkedClipId === clip.id
    );

    // Calculate live trim values
    let displayStartTime = clip.startTime;
    let displayDuration = clip.duration;

    if (isTrimming && clipTrim) {
      const deltaX = clipTrim.currentX - clipTrim.startX;
      const deltaTime = pixelToTime(deltaX);
      const maxDuration = clip.source?.naturalDuration || clip.duration;

      if (clipTrim.edge === 'left') {
        const maxTrim = clipTrim.originalDuration - 0.1;
        const minTrim = -clipTrim.originalInPoint;
        const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));
        displayStartTime = clipTrim.originalStartTime + clampedDelta;
        displayDuration = clipTrim.originalDuration - clampedDelta;
      } else {
        const maxExtend = maxDuration - clipTrim.originalOutPoint;
        const minTrim = -(clipTrim.originalDuration - 0.1);
        const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
        displayDuration = clipTrim.originalDuration + clampedDelta;
      }
    } else if (isLinkedToTrimming && clipTrim && trimmedClip) {
      // Apply same trim to linked clip visually
      const deltaX = clipTrim.currentX - clipTrim.startX;
      const deltaTime = pixelToTime(deltaX);
      const maxDuration = clip.source?.naturalDuration || clip.duration;

      if (clipTrim.edge === 'left') {
        const maxTrim = clip.duration - 0.1;
        const minTrim = -clip.inPoint;
        const clampedDelta = Math.max(minTrim, Math.min(maxTrim, deltaTime));
        displayStartTime = clip.startTime + clampedDelta;
        displayDuration = clip.duration - clampedDelta;
      } else {
        const maxExtend = maxDuration - clip.outPoint;
        const minTrim = -(clip.duration - 0.1);
        const clampedDelta = Math.max(minTrim, Math.min(maxExtend, deltaTime));
        displayDuration = clip.duration + clampedDelta;
      }
    }

    const width = timeToPixel(displayDuration);

    // Calculate position - if dragging, use snapped position if available
    let left = timeToPixel(displayStartTime);
    if (isDragging && clipDrag && timelineRef.current) {
      // Use snapped time if snapping, otherwise raw position
      if (clipDrag.isSnapping && clipDrag.snappedTime !== null) {
        left = timeToPixel(clipDrag.snappedTime);
      } else {
        const rect = timelineRef.current.getBoundingClientRect();
        const x = clipDrag.currentX - rect.left + scrollX - clipDrag.grabOffsetX;
        left = Math.max(0, x);
      }
    } else if (isLinkedToDragging && clipDrag && timelineRef.current && draggedClip) {
      // Move linked clip in sync - use snapped position if available
      let newDragTime: number;
      if (clipDrag.isSnapping && clipDrag.snappedTime !== null) {
        newDragTime = clipDrag.snappedTime;
      } else {
        const rect = timelineRef.current.getBoundingClientRect();
        const dragX = clipDrag.currentX - rect.left + scrollX - clipDrag.grabOffsetX;
        newDragTime = pixelToTime(Math.max(0, dragX));
      }
      const timeDelta = newDragTime - draggedClip.startTime;
      left = timeToPixel(Math.max(0, clip.startTime + timeDelta));
    }

    // Calculate how many thumbnails to show based on clip width
    const thumbWidth = 71;
    const visibleThumbs = Math.max(1, Math.ceil(width / thumbWidth));

    // Track filtering
    if (isDragging && clipDrag && clipDrag.currentTrackId !== trackId) {
      return null;
    }
    if (!isDragging && !isLinkedToDragging && clip.trackId !== trackId) {
      return null;
    }
    if (clip.trackId !== trackId && !isDragging) {
      return null;
    }

    const clipClass = [
      'timeline-clip',
      isSelected ? 'selected' : '',
      isDragging ? 'dragging' : '',
      isLinkedToDragging ? 'linked-dragging' : '',
      isTrimming ? 'trimming' : '',
      isLinkedToTrimming ? 'linked-trimming' : '',
      clip.source?.type || 'video'
    ].filter(Boolean).join(' ');

    return (
      <div
        key={clip.id}
        className={clipClass}
        style={{ left, width }}
        onMouseDown={(e) => handleClipMouseDown(e, clip.id)}
      >
        {/* Audio waveform */}
        {clip.source?.type === 'audio' && clip.waveform && clip.waveform.length > 0 && (
          <div className="clip-waveform">
            {renderWaveform(clip.waveform, width, Math.max(20, (tracks.find(t => t.id === trackId)?.height || 40) - 12))}
          </div>
        )}
        {/* Thumbnail filmstrip */}
        {thumbnails.length > 0 && clip.source?.type !== 'audio' && (
          <div className="clip-thumbnails">
            {Array.from({ length: visibleThumbs }).map((_, i) => {
              const thumbIndex = Math.floor((i / visibleThumbs) * thumbnails.length);
              const thumb = thumbnails[Math.min(thumbIndex, thumbnails.length - 1)];
              return (
                <img
                  key={i}
                  src={thumb}
                  alt=""
                  className="clip-thumb"
                  draggable={false}
                />
              );
            })}
          </div>
        )}
        <div className="clip-content">
          <span className="clip-name">{clip.name}</span>
          <span className="clip-duration">{formatTime(displayDuration)}</span>
        </div>
        {/* Trim handles */}
        <div
          className="trim-handle left"
          onMouseDown={(e) => handleTrimStart(e, clip.id, 'left')}
        />
        <div
          className="trim-handle right"
          onMouseDown={(e) => handleTrimStart(e, clip.id, 'right')}
        />
      </div>
    );
  };

  return (
    <div className={`timeline-container ${clipDrag || clipTrim ? 'is-dragging' : ''}`}>
      {/* Timeline toolbar */}
      <div className="timeline-toolbar">
        <div className="timeline-controls">
          <button className="btn btn-sm" onClick={stop} title="Stop">⏹</button>
          <button className={`btn btn-sm ${isPlaying ? 'btn-active' : ''}`} onClick={isPlaying ? pause : play}>
            {isPlaying ? '⏸' : '▶'}
          </button>
        </div>
        <div className="timeline-time">
          {formatTime(playheadPosition)} / {formatTime(duration)}
        </div>
        <div className="timeline-zoom">
          <button className="btn btn-sm" onClick={() => setZoom(zoom - 10)}>−</button>
          <span>{Math.round(zoom)}px/s</span>
          <button className="btn btn-sm" onClick={() => setZoom(zoom + 10)}>+</button>
        </div>
        <div className="timeline-tracks-controls">
          <button className="btn btn-sm" onClick={() => addTrack('video')}>+ Video Track</button>
          <button className="btn btn-sm" onClick={() => addTrack('audio')}>+ Audio Track</button>
        </div>
      </div>

      {/* Timeline body */}
      <div className="timeline-body">
        {/* Track headers */}
        <div className="track-headers">
          <div className="ruler-header">Time</div>
          {tracks.map(track => (
            <div
              key={track.id}
              className={`track-header ${track.type}`}
              style={{ height: track.height }}
              onWheel={(e) => handleTrackHeaderWheel(e, track.id)}
            >
              <span className="track-name">{track.name}</span>
              <div className="track-controls">
                <button
                  className={`btn-icon ${track.muted ? 'muted' : ''}`}
                  onClick={() => useTimelineStore.getState().setTrackMuted(track.id, !track.muted)}
                  title={track.muted ? 'Unmute' : 'Mute'}
                >
                  {track.muted ? '🔇' : '🔊'}
                </button>
                {track.type === 'video' && (
                  <button
                    className={`btn-icon ${!track.visible ? 'hidden' : ''}`}
                    onClick={() => useTimelineStore.getState().setTrackVisible(track.id, !track.visible)}
                    title={track.visible ? 'Hide' : 'Show'}
                  >
                    {track.visible ? '👁' : '👁‍🗨'}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Timeline tracks */}
        <div
          ref={(el) => {
            (timelineRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
            (trackLanesRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
          }}
          className={`timeline-tracks ${clipDrag ? 'dragging-clip' : ''}`}
          onWheel={handleWheel}
          style={{ transform: `translateX(-${scrollX}px)` }}
        >
          {/* Time ruler */}
          {renderTimeRuler()}

          {/* Track lanes */}
          {tracks.map(track => (
            <div
              key={track.id}
              className={`track-lane ${track.type} ${clipDrag?.currentTrackId === track.id ? 'drag-target' : ''} ${externalDrag?.trackId === track.id || externalDrag?.audioTrackId === track.id ? 'external-drag-target' : ''}`}
              style={{ height: track.height }}
              onDrop={(e) => handleTrackDrop(e, track.id)}
              onDragOver={(e) => handleTrackDragOver(e, track.id)}
              onDragEnter={(e) => handleTrackDragEnter(e, track.id)}
              onDragLeave={handleTrackDragLeave}
            >
              {/* Render clips belonging to this track */}
              {clips
                .filter(c => c.trackId === track.id)
                .map(clip => renderClip(clip, track.id))}
              {/* Render clip being dragged TO this track */}
              {clipDrag && clipDrag.currentTrackId === track.id && clipDrag.originalTrackId !== track.id && (
                clips
                  .filter(c => c.id === clipDrag.clipId)
                  .map(clip => renderClip(clip, track.id))
              )}
              {/* External file drag preview - video clip */}
              {externalDrag && externalDrag.trackId === track.id && (
                <div
                  className="timeline-clip-preview"
                  style={{
                    left: timeToPixel(externalDrag.startTime),
                    width: timeToPixel(5), // Default 5 seconds preview
                  }}
                >
                  <div className="clip-content">
                    <span className="clip-name">Drop to add clip</span>
                  </div>
                </div>
              )}
              {/* External file drag preview - linked audio clip */}
              {externalDrag && externalDrag.isVideo && externalDrag.audioTrackId === track.id && (
                <div
                  className="timeline-clip-preview audio"
                  style={{
                    left: timeToPixel(externalDrag.startTime),
                    width: timeToPixel(5), // Default 5 seconds preview
                  }}
                >
                  <div className="clip-content">
                    <span className="clip-name">Audio</span>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Preview for new audio track that will be created */}
          {externalDrag && externalDrag.isVideo && externalDrag.audioTrackId === '__new_audio_track__' && (
            <div
              className="track-lane audio new-track-preview"
              style={{ height: 40 }}
            >
              <div
                className="timeline-clip-preview audio"
                style={{
                  left: timeToPixel(externalDrag.startTime),
                  width: timeToPixel(5),
                }}
              >
                <div className="clip-content">
                  <span className="clip-name">+ New Audio Track</span>
                </div>
              </div>
            </div>
          )}

          {/* Snap indicator line - shows when clip is snapping to another */}
          {clipDrag?.isSnapping && clipDrag.snappedTime !== null && (
            <div
              className="snap-line"
              style={{ left: timeToPixel(clipDrag.snappedTime) }}
            />
          )}

          {/* Playhead */}
          <div
            className="playhead"
            style={{ left: timeToPixel(playheadPosition) }}
            onMouseDown={handlePlayheadMouseDown}
          >
            <div className="playhead-head" />
            <div className="playhead-line" />
          </div>
        </div>
      </div>
    </div>
  );
}
