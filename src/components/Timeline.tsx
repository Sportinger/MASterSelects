// Timeline component - Video editing timeline with tracks and clips

import { useRef, useState, useCallback, useEffect } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import { useMixerStore } from '../stores/mixerStore';
import { useMediaStore } from '../stores/mediaStore';
import { engine } from '../engine/WebGPUEngine';
import { proxyFrameCache } from '../services/proxyFrameCache';

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
    inPoint,
    outPoint,
    addTrack,
    addClip,
    addCompClip,
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
    setInPoint,
    setOutPoint,
    setInPointAtPlayhead,
    setOutPointAtPlayhead,
    clearInOut,
    getSnappedPosition,
    findNonOverlappingPosition,
    loopPlayback,
    toggleLoopPlayback,
    ramPreviewProgress,
    ramPreviewRange,
    isRamPreviewing,
    startRamPreview,
    cancelRamPreview,
    clearRamPreview,
    getCachedRanges,
    isDraggingPlayhead,
    setDraggingPlayhead,
    ramPreviewEnabled,
    toggleRamPreviewEnabled,
  } = useTimelineStore();

  const { activeCompositionId, getActiveComposition, proxyEnabled, setProxyEnabled, files: mediaFiles, currentlyGeneratingProxyId } = useMediaStore();
  const activeComposition = getActiveComposition();

  const timelineRef = useRef<HTMLDivElement>(null);
  const trackLanesRef = useRef<HTMLDivElement>(null);

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

  // In/Out marker drag state
  const [markerDrag, setMarkerDrag] = useState<{
    type: 'in' | 'out';
    startX: number;
    originalTime: number;
  } | null>(null);

  // External file drag preview state
  const [externalDrag, setExternalDrag] = useState<{
    trackId: string;
    startTime: number;
    x: number;
    y: number;
    audioTrackId?: string;  // Preview for linked audio clip
    isVideo?: boolean;      // Is the dragged file a video?
    duration?: number;      // Actual duration of dragged file
  } | null>(null);
  const dragCounterRef = useRef(0); // Track drag enter/leave balance
  const dragDurationCacheRef = useRef<{ url: string; duration: number } | null>(null); // Cache duration during drag

  // Keyboard shortcuts (global, works regardless of focus)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Space: toggle play/pause
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        if (isPlaying) {
          pause();
        } else {
          play();
        }
        return;
      }

      // I: set In point at playhead
      if (e.key === 'i' || e.key === 'I') {
        e.preventDefault();
        setInPointAtPlayhead();
        return;
      }

      // O: set Out point at playhead
      if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        setOutPointAtPlayhead();
        return;
      }

      // X: clear In/Out points
      if (e.key === 'x' || e.key === 'X') {
        e.preventDefault();
        clearInOut();
        return;
      }

      // L: toggle loop playback
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        toggleLoopPlayback();
        return;
      }

      // Delete/Backspace: remove selected clip from timeline
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        if (selectedClipId) {
          removeClip(selectedClipId);
        }
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, play, pause, setInPointAtPlayhead, setOutPointAtPlayhead, clearInOut, toggleLoopPlayback, selectedClipId, removeClip]);

  // Auto-start RAM Preview after 2 seconds of idle (like After Effects)
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cancel RAM preview when user starts playing or scrubbing (keep cached frames)
  useEffect(() => {
    if ((isPlaying || isDraggingPlayhead) && isRamPreviewing) {
      cancelRamPreview();
    }
  }, [isPlaying, isDraggingPlayhead, isRamPreviewing, cancelRamPreview]);

  // Check for idle state and auto-start RAM preview
  useEffect(() => {
    // Clear any existing timer
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }

    // Don't auto-render if:
    // - RAM Preview is disabled
    // - Currently playing
    // - Currently RAM previewing
    // - Scrubbing
    // - No clips on timeline
    if (!ramPreviewEnabled || isPlaying || isRamPreviewing || isDraggingPlayhead || clips.length === 0) {
      return;
    }

    // Determine render range: use In/Out if set, otherwise use clips extent
    const renderStart = inPoint ?? 0;
    const renderEnd = outPoint ?? Math.max(...clips.map(c => c.startTime + c.duration));

    // Skip if range is too small
    if (renderEnd - renderStart < 0.1) {
      return;
    }

    // Check if we already have RAM preview for this range
    if (ramPreviewRange &&
        ramPreviewRange.start <= renderStart &&
        ramPreviewRange.end >= renderEnd) {
      return;
    }

    // Start timer to auto-render after 2 seconds of idle
    idleTimerRef.current = setTimeout(() => {
      // Double-check conditions before starting
      const state = useTimelineStore.getState();
      if (state.ramPreviewEnabled && !state.isPlaying && !state.isRamPreviewing) {
        startRamPreview();
      }
    }, 2000);

    return () => {
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
        idleTimerRef.current = null;
      }
    };
  }, [ramPreviewEnabled, isPlaying, isRamPreviewing, isDraggingPlayhead, inPoint, outPoint, ramPreviewRange, clips, startRamPreview]);

  // Auto-generate proxies after 3 seconds of idle
  const proxyIdleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { getNextFileNeedingProxy, generateProxy } = useMediaStore();

  useEffect(() => {
    // Clear any existing timer
    if (proxyIdleTimerRef.current) {
      clearTimeout(proxyIdleTimerRef.current);
      proxyIdleTimerRef.current = null;
    }

    // Don't auto-generate if:
    // - Proxy is disabled
    // - Currently playing
    // - Currently generating a proxy
    // - Scrubbing
    if (!proxyEnabled || isPlaying || currentlyGeneratingProxyId || isDraggingPlayhead) {
      return;
    }

    // Start timer to auto-generate after 3 seconds of idle
    proxyIdleTimerRef.current = setTimeout(() => {
      const mediaStore = useMediaStore.getState();
      if (mediaStore.proxyEnabled && !mediaStore.currentlyGeneratingProxyId) {
        const nextFile = mediaStore.getNextFileNeedingProxy();
        if (nextFile) {
          console.log('[Proxy] Auto-starting proxy generation for:', nextFile.name);
          mediaStore.generateProxy(nextFile.id);
        }
      }
    }, 3000);

    return () => {
      if (proxyIdleTimerRef.current) {
        clearTimeout(proxyIdleTimerRef.current);
        proxyIdleTimerRef.current = null;
      }
    };
  }, [proxyEnabled, isPlaying, currentlyGeneratingProxyId, isDraggingPlayhead, clips, getNextFileNeedingProxy, generateProxy]);

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

  // Track which clips are currently active (to detect clip changes)
  const activeClipIdsRef = useRef<string>('');

  // Track current proxy frames for each clip (for smooth proxy playback)
  const proxyFramesRef = useRef<Map<string, { frameIndex: number; image: HTMLImageElement }>>(new Map());
  const proxyLoadingRef = useRef<Set<string>>(new Set());

  // Sync timeline playback with Preview - update mixer layers based on clips at playhead
  // IMPORTANT: This uses batched updates to prevent flickering from race conditions
  useEffect(() => {
    // CRITICAL: Skip video seeking during RAM Preview generation
    if (isRamPreviewing) {
      return;
    }

    // Try to use cached RAM Preview frame first (instant playback)
    if (ramPreviewRange &&
        playheadPosition >= ramPreviewRange.start &&
        playheadPosition <= ramPreviewRange.end) {
      if (engine.renderCachedFrame(playheadPosition)) {
        return;
      }
    }

    const clipsAtTime = getClipsAtTime(playheadPosition);

    // Get current active clip IDs to detect clip boundary crossings
    const currentActiveIds = clipsAtTime
      .filter(c => c.source?.type === 'video' || c.source?.type === 'image' || c.isComposition)
      .map(c => c.id)
      .sort()
      .join(',');

    // During playback: skip layer sync if same clips are active (fast path)
    // But MUST re-sync when clips change (clip boundaries crossed)
    // IMPORTANT: Don't skip if proxy is enabled - we need to update frames each tick
    const mediaStoreState = useMediaStore.getState();
    const hasActiveProxies = mediaStoreState.proxyEnabled &&
      clipsAtTime.some(clip => {
        const mediaFile = mediaStoreState.files.find(f =>
          f.id === clip.source?.mediaFileId || f.name === clip.name
        );
        return mediaFile?.proxyStatus === 'ready';
      });

    if (isPlaying && !hasActiveProxies && activeClipIdsRef.current.startsWith('playing:')) {
      const prevActiveIds = activeClipIdsRef.current.slice(8); // Remove 'playing:' prefix
      if (prevActiveIds === currentActiveIds) {
        // Same clips active - skip layer sync, just ensure videos are playing
        clipsAtTime.forEach(clip => {
          if (clip.source?.videoElement?.paused) {
            clip.source.videoElement.play().catch(() => {});
          }
        });
        return;
      }
      // Clips changed - need to re-sync layers, continue below
    }

    // Update active clip tracking
    if (isPlaying) {
      clipsAtTime.forEach(clip => {
        if (clip.source?.videoElement?.paused) {
          clip.source.videoElement.play().catch(() => {});
        }
      });
      activeClipIdsRef.current = 'playing:' + currentActiveIds;
    } else {
      activeClipIdsRef.current = '';
    }

    const currentLayers = useMixerStore.getState().layers;

    // Get video tracks sorted by index (for layer order)
    const videoTracks = tracks.filter(t => t.type === 'video');

    // Check if any video track has solo enabled
    const anyVideoSolo = videoTracks.some(t => t.solo);

    // Helper to determine effective visibility for a video track
    const isVideoTrackVisible = (track: typeof videoTracks[0]) => {
      if (!track.visible) return false;
      if (anyVideoSolo) return track.solo;
      return true;
    };

    // Build new layers array atomically - don't setState inside loop!
    const newLayers = [...currentLayers];
    let layersChanged = false;

    // Helper to get video element from a clip (handles nested compositions)
    const getVideoFromClip = (clip: typeof clips[0], clipTime: number): { video: HTMLVideoElement | null; transform: typeof clip.transform } => {
      // Handle nested composition clips
      if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        // Find active clip in nested composition at the current nested time
        const nestedTime = clipTime; // clipTime is already relative to the comp clip
        const nestedVideoTracks = (clip.nestedTracks || []).filter(t => t.type === 'video' && t.visible);

        // Find the topmost visible video clip at nestedTime
        for (const nestedTrack of nestedVideoTracks) {
          const nestedClip = clip.nestedClips.find(nc =>
            nc.trackId === nestedTrack.id &&
            nestedTime >= nc.startTime &&
            nestedTime < nc.startTime + nc.duration &&
            nc.source?.videoElement
          );

          if (nestedClip?.source?.videoElement) {
            const nestedClipTime = nestedTime - nestedClip.startTime + nestedClip.inPoint;
            const video = nestedClip.source.videoElement;

            // Seek nested video
            const timeDiff = Math.abs(video.currentTime - nestedClipTime);
            if (timeDiff > 0.05) {
              video.currentTime = nestedClipTime;
            }

            // Control playback
            if (isPlaying && video.paused) {
              video.play().catch(() => {});
            } else if (!isPlaying && !video.paused) {
              video.pause();
            }

            return { video, transform: clip.transform };
          }
        }
        return { video: null, transform: clip.transform };
      }

      // Regular video clip
      return { video: clip.source?.videoElement || null, transform: clip.transform };
    };

    // Process each video layer and collect updates
    videoTracks.forEach((track, layerIndex) => {
      const clip = clipsAtTime.find(c => c.trackId === track.id);
      const layer = currentLayers[layerIndex];

      // Handle composition clips with nested content
      if (clip?.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const { video, transform } = getVideoFromClip(clip, clipTime);

        if (video) {
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
              visible: isVideoTrackVisible(track),
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
        } else {
          // Nested comp has no video at this time - clear layer
          if (layer?.source) {
            newLayers[layerIndex] = undefined as any;
            layersChanged = true;
          }
        }
      } else if (clip?.source?.videoElement) {
        // Seek video to correct position within clip
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const video = clip.source.videoElement;
        const webCodecsPlayer = clip.source.webCodecsPlayer;
        const timeDiff = Math.abs(video.currentTime - clipTime);

        // Check if we should use proxy frames for this clip
        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find(f => f.name === clip.name || clip.source?.mediaFileId === f.id);
        // Use proxy if enabled AND (ready OR generating with some frames available)
        const hasProxyFrames = mediaFile?.proxyStatus === 'ready' ||
          (mediaFile?.proxyStatus === 'generating' && (mediaFile?.proxyProgress || 0) > 0);
        const useProxy = mediaStore.proxyEnabled && hasProxyFrames && mediaFile?.proxyFps;
        const proxyFps = mediaFile?.proxyFps || 30;

        // Debug logging
        if (mediaStore.proxyEnabled && mediaFile) {
          console.log('[Proxy] Clip:', clip.name, 'mediaFileId:', mediaFile.id, 'status:', mediaFile?.proxyStatus, 'fps:', mediaFile?.proxyFps, 'useProxy:', useProxy);
        }

        if (useProxy && mediaFile) {
          // Use proxy frames for DISPLAY but let video play for audio sync
          const frameIndex = Math.floor(clipTime * proxyFps);
          const cacheKey = `${mediaFile.id}_${clip.id}`;
          const cached = proxyFramesRef.current.get(cacheKey);

          // Let video play for audio sync - we just won't display its frames
          // Mute video since audio comes from linked audio track
          if (!video.muted) {
            video.muted = true;
          }
          if (isPlaying && video.paused) {
            video.play().catch(() => {});
          } else if (!isPlaying && !video.paused) {
            video.pause();
          }

          // Sync video position when scrubbing (not playing)
          if (!isPlaying) {
            const timeDiff = Math.abs(video.currentTime - clipTime);
            if (timeDiff > 0.1) {
              video.currentTime = clipTime;
            }
          }

          // Load proxy frame - use synchronous path if already cached in proxyFrameCache
          const loadKey = `${mediaFile.id}_${frameIndex}`;

          // Try to get frame from proxyFrameCache (might be sync if already cached)
          // Pass fps to trigger preloading of upcoming frames
          const cachedInService = proxyFrameCache.getCachedFrame(mediaFile.id, frameIndex, proxyFps);

          console.log('[Proxy] Frame lookup:', frameIndex, 'cachedInService:', !!cachedInService, 'cachedLocal:', !!cached, 'cachedLocalFrame:', cached?.frameIndex);

          if (cachedInService) {
            // Frame is already in the service cache - use it immediately
            console.log('[Proxy] Using cached frame:', frameIndex, 'image:', cachedInService.src?.slice(0, 50));
            proxyFramesRef.current.set(cacheKey, { frameIndex, image: cachedInService });

            const transform = clip.transform;
            newLayers[layerIndex] = {
              id: `timeline_layer_${layerIndex}`,
              name: clip.name,
              visible: isVideoTrackVisible(track),
              opacity: transform.opacity,
              blendMode: transform.blendMode,
              source: {
                type: 'image',
                imageElement: cachedInService,
              },
              effects: [],
              position: { x: transform.position.x, y: transform.position.y },
              scale: { x: transform.scale.x, y: transform.scale.y },
              rotation: transform.rotation.z * Math.PI / 180,
            };
            layersChanged = true;
          } else if (!cached || cached.frameIndex !== frameIndex) {
            // Need to load from IndexedDB
            if (!proxyLoadingRef.current.has(loadKey)) {
              proxyLoadingRef.current.add(loadKey);

              // Capture values for closure
              const capturedLayerIndex = layerIndex;
              const capturedTransform = clip.transform;
              const capturedTrackVisible = isVideoTrackVisible(track);
              const capturedClipName = clip.name;

              proxyFrameCache.getFrame(mediaFile.id, clipTime, proxyFps).then(image => {
                proxyLoadingRef.current.delete(loadKey);
                if (image) {
                  proxyFramesRef.current.set(cacheKey, { frameIndex, image });

                  // Directly update the layer with the new frame
                  const currentLayers = useMixerStore.getState().layers;
                  const updatedLayers = [...currentLayers];
                  updatedLayers[capturedLayerIndex] = {
                    id: `timeline_layer_${capturedLayerIndex}`,
                    name: capturedClipName,
                    visible: capturedTrackVisible,
                    opacity: capturedTransform.opacity,
                    blendMode: capturedTransform.blendMode,
                    source: {
                      type: 'image',
                      imageElement: image,
                    },
                    effects: [],
                    position: { x: capturedTransform.position.x, y: capturedTransform.position.y },
                    scale: { x: capturedTransform.scale.x, y: capturedTransform.scale.y },
                    rotation: capturedTransform.rotation.z * Math.PI / 180,
                  };
                  useMixerStore.setState({ layers: updatedLayers });
                }
              });
            }

            // Use previous frame while loading (if available)
            if (cached?.image) {
              const transform = clip.transform;
              newLayers[layerIndex] = {
                id: `timeline_layer_${layerIndex}`,
                name: clip.name,
                visible: isVideoTrackVisible(track),
                opacity: transform.opacity,
                blendMode: transform.blendMode,
                source: {
                  type: 'image',
                  imageElement: cached.image,
                },
                effects: [],
                position: { x: transform.position.x, y: transform.position.y },
                scale: { x: transform.scale.x, y: transform.scale.y },
                rotation: transform.rotation.z * Math.PI / 180,
              };
              layersChanged = true;
            }
          } else if (cached?.image) {
            // Same frame as before, just use it
            const transform = clip.transform;
            const needsUpdate = !layer ||
              layer.source?.imageElement !== cached.image ||
              layer.source?.type !== 'image';

            if (needsUpdate) {
              newLayers[layerIndex] = {
                id: `timeline_layer_${layerIndex}`,
                name: clip.name,
                visible: isVideoTrackVisible(track),
                opacity: transform.opacity,
                blendMode: transform.blendMode,
                source: {
                  type: 'image',
                  imageElement: cached.image,
                },
                effects: [],
                position: { x: transform.position.x, y: transform.position.y },
                scale: { x: transform.scale.x, y: transform.scale.y },
                rotation: transform.rotation.z * Math.PI / 180,
              };
              layersChanged = true;
            }
          }
          // If proxy frame not loaded yet, keep showing the last frame (or black)
          // Do NOT fall back to video - proxy mode means proxy only
        } else {
          // Original video playback (no proxy)
          // WebCodecsPlayer (if available) listens to video element events automatically
          // We just need to seek it when scrubbing
          if (webCodecsPlayer) {
            const wcTimeDiff = Math.abs(webCodecsPlayer.currentTime - clipTime);
            if (wcTimeDiff > 0.05) {
              webCodecsPlayer.seek(clipTime);
            }
          }

          // Control HTMLVideoElement playback
          if (isPlaying && video.paused) {
            video.play().catch(() => {});
          } else if (!isPlaying && !video.paused) {
            video.pause();
          }

          // During playback: DON'T seek - let video play naturally, we sync FROM it
          // Only seek when scrubbing or paused
          if (!isPlaying) {
            const seekThreshold = isDraggingPlayhead ? 0.1 : 0.05;

            if (timeDiff > seekThreshold) {
              const now = performance.now();
              const lastSeek = lastSeekRef.current[clip.id] || 0;

              // Throttle seeks during scrubbing
              if (isDraggingPlayhead && now - lastSeek < 80) {
                pendingSeekRef.current[clip.id] = clipTime;
              } else {
                if (isDraggingPlayhead && 'fastSeek' in video) {
                  video.fastSeek(clipTime);
                } else {
                  video.currentTime = clipTime;
                }
                lastSeekRef.current[clip.id] = now;
                delete pendingSeekRef.current[clip.id];
              }
            }
          }

          // Check if layer needs update
          const transform = clip.transform;
          const needsUpdate = !layer ||
            layer.source?.videoElement !== video ||
            layer.source?.webCodecsPlayer !== webCodecsPlayer ||
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
              visible: isVideoTrackVisible(track),
              opacity: transform.opacity,
              blendMode: transform.blendMode,
              source: {
                type: 'video',
                videoElement: video,
                webCodecsPlayer: webCodecsPlayer,
              },
              effects: [],
              position: { x: transform.position.x, y: transform.position.y },
              scale: { x: transform.scale.x, y: transform.scale.y },
              rotation: transform.rotation.z * Math.PI / 180,
            };
            layersChanged = true;
          }
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
            visible: isVideoTrackVisible(track),
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
      console.log('[LayerSync] Updating layers, count:', newLayers.length, 'types:', newLayers.map(l => l?.source?.type).join(','));
      useMixerStore.setState({ layers: newLayers });
    }

    // Handle audio tracks - sync audio elements with playhead
    const audioTracks = tracks.filter(t => t.type === 'audio');

    // Check if any audio track has solo enabled
    const anyAudioSolo = audioTracks.some(t => t.solo);

    // Helper to determine if an audio track should be muted (considering solo)
    const isAudioTrackMuted = (track: typeof audioTracks[0]) => {
      if (track.muted) return true;
      if (anyAudioSolo) return !track.solo;
      return false;
    };

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

        // Handle mute state (including solo logic)
        const effectivelyMuted = isAudioTrackMuted(track);
        audio.muted = effectivelyMuted;

        // Play/pause audio based on timeline state (pause while scrubbing)
        const shouldPlay = isPlaying && !effectivelyMuted && !isDraggingPlayhead;
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
  }, [playheadPosition, clips, tracks, isPlaying, isDraggingPlayhead, ramPreviewRange, isRamPreviewing]);

  // Get clips at time helper
  const getClipsAtTime = useCallback((time: number) => {
    return clips.filter(c => time >= c.startTime && time < c.startTime + c.duration);
  }, [clips]);

  // Playback loop - minimal work, just update UI playhead periodically
  // Video plays independently via video.play(), we just read its time for UI
  useEffect(() => {
    if (!isPlaying) return;

    let intervalId: ReturnType<typeof setInterval>;
    let lastVideoTime = 0;

    // Find the active video clip
    const getActiveVideoClip = () => {
      const state = useTimelineStore.getState();
      const pos = state.playheadPosition;
      for (const clip of state.clips) {
        if (clip.source?.videoElement && pos >= clip.startTime && pos < clip.startTime + clip.duration) {
          return clip;
        }
      }
      return null;
    };

    // Update playhead from video time - run at low frequency (15fps) to minimize React work
    const updatePlayhead = () => {
      const state = useTimelineStore.getState();
      const { duration, inPoint, outPoint, loopPlayback, pause } = state;
      const effectiveEnd = outPoint !== null ? outPoint : duration;
      const effectiveStart = inPoint !== null ? inPoint : 0;

      const activeClip = getActiveVideoClip();
      let newPosition: number;

      if (activeClip?.source?.videoElement) {
        const video = activeClip.source.videoElement;
        newPosition = activeClip.startTime + video.currentTime - activeClip.inPoint;
        lastVideoTime = video.currentTime;
      } else {
        // Fallback - increment by interval time
        newPosition = state.playheadPosition + 0.066; // ~15fps
      }

      // Handle loop/end
      if (newPosition >= effectiveEnd) {
        if (loopPlayback) {
          newPosition = effectiveStart;
          const clip = getActiveVideoClip();
          if (clip?.source?.videoElement) {
            clip.source.videoElement.currentTime = clip.inPoint;
          }
        } else {
          newPosition = effectiveEnd;
          pause();
          setPlayheadPosition(newPosition);
          return;
        }
      }

      setPlayheadPosition(newPosition);
    };

    // Use setInterval instead of RAF - completely decoupled from render loop
    // 15fps is enough for smooth playhead UI movement
    intervalId = setInterval(updatePlayhead, 66);

    return () => clearInterval(intervalId);
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

    // IMMEDIATELY cancel RAM preview before any seeking to prevent race condition
    if (isRamPreviewing) {
      cancelRamPreview();
    }

    // Jump playhead immediately
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left + scrollX;
    const time = pixelToTime(x);
    setPlayheadPosition(Math.max(0, Math.min(time, duration)));

    // Enter drag mode immediately
    setDraggingPlayhead(true);
  };

  // Handle playhead drag
  const handlePlayheadMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    // IMMEDIATELY cancel RAM preview before any seeking to prevent race condition
    if (isRamPreviewing) {
      cancelRamPreview();
    }
    setDraggingPlayhead(true);
  };

  // Handle In/Out marker drag
  const handleMarkerMouseDown = (e: React.MouseEvent, type: 'in' | 'out') => {
    e.stopPropagation();
    e.preventDefault();
    const originalTime = type === 'in' ? inPoint : outPoint;
    if (originalTime === null) return;

    setMarkerDrag({
      type,
      startX: e.clientX,
      originalTime,
    });
  };

  // Handle marker dragging
  useEffect(() => {
    if (!markerDrag) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!timelineRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const time = Math.max(0, Math.min(pixelToTime(x), duration));

      if (markerDrag.type === 'in') {
        // In point can't exceed out point
        const maxTime = outPoint !== null ? outPoint : duration;
        setInPoint(Math.min(time, maxTime));
      } else {
        // Out point can't precede in point
        const minTime = inPoint !== null ? inPoint : 0;
        setOutPoint(Math.max(time, minTime));
      }
    };

    const handleMouseUp = () => {
      setMarkerDrag(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [markerDrag, scrollX, duration, inPoint, outPoint, setInPoint, setOutPoint]);

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
      setDraggingPlayhead(false);
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

  // Quick duration check for dragged video files
  const getVideoDurationQuick = async (file: File): Promise<number | null> => {
    if (!file.type.startsWith('video/')) return null;

    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.preload = 'metadata';

      const cleanup = () => {
        URL.revokeObjectURL(video.src);
        video.remove();
      };

      video.onloadedmetadata = () => {
        const duration = video.duration;
        cleanup();
        resolve(isFinite(duration) ? duration : null);
      };

      video.onerror = () => {
        cleanup();
        resolve(null);
      };

      // Timeout after 500ms - we want this to be fast
      setTimeout(() => {
        cleanup();
        resolve(null);
      }, 500);

      video.src = URL.createObjectURL(file);
    });
  };

  // Handle external file drag enter on track
  const handleTrackDragEnter = (e: React.DragEvent, trackId: string) => {
    e.preventDefault();
    dragCounterRef.current++;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left + scrollX;
    const startTime = pixelToTime(x);

    // Check if dragging composition from MediaPanel
    if (e.dataTransfer.types.includes('application/x-composition-id')) {
      setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
      return;
    }

    // Check if dragging media file from MediaPanel
    if (e.dataTransfer.types.includes('application/x-media-file-id')) {
      // We can't access the data during dragenter, but we can show a preview
      // The actual mediaFileId will be read on drop
      setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration: 5, isVideo: true });
      return;
    }

    // Show preview for external files
    if (e.dataTransfer.types.includes('Files')) {
      // Check cache first for duration - find the file item (not always at index 0)
      let duration: number | undefined;
      const items = e.dataTransfer.items;
      if (items && items.length > 0) {
        // Find the file item in the list
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          if (item.kind === 'file') {
            const file = item.getAsFile();
            if (file && file.type.startsWith('video/')) {
              const cacheKey = `${file.name}_${file.size}`;
              if (dragDurationCacheRef.current?.url === cacheKey) {
                // Use cached duration
                duration = dragDurationCacheRef.current.duration;
              } else {
                // Load duration in background, update state when ready
                getVideoDurationQuick(file).then(dur => {
                  if (dur) {
                    dragDurationCacheRef.current = { url: cacheKey, duration: dur };
                    // Update the externalDrag state with the loaded duration
                    setExternalDrag(prev => prev ? { ...prev, duration: dur } : null);
                  }
                });
              }
              break; // Found the file, stop searching
            }
          }
        }
      }

      setExternalDrag({ trackId, startTime, x: e.clientX, y: e.clientY, duration });
    }
  };

  // Handle external file drag over track
  const handleTrackDragOver = (e: React.DragEvent, trackId: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';

    // Check if dragging from MediaPanel or external files
    const isCompDrag = e.dataTransfer.types.includes('application/x-composition-id');
    const isMediaPanelDrag = e.dataTransfer.types.includes('application/x-media-file-id');
    const isFileDrag = e.dataTransfer.types.includes('Files');

    // Update preview position
    if ((isCompDrag || isMediaPanelDrag || isFileDrag) && timelineRef.current) {
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + scrollX;
      const startTime = pixelToTime(x);

      // Check if dragging to a video track - if so, find available audio track for preview
      const targetTrack = tracks.find(t => t.id === trackId);
      const isVideoTrack = targetTrack?.type === 'video';

      // Use cached duration from state, ref cache, or fallback to 5 seconds
      const previewDuration = externalDrag?.duration ?? dragDurationCacheRef.current?.duration ?? 5;

      // For video tracks, find an available audio track for the linked audio clip preview
      let audioTrackId: string | undefined;
      if (isVideoTrack) {
        const audioTracks = tracks.filter(t => t.type === 'audio');
        const endTime = startTime + previewDuration;

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

      setExternalDrag(prev => ({
        trackId,
        startTime,
        x: e.clientX,
        y: e.clientY,
        audioTrackId,
        isVideo: isVideoTrack,
        duration: prev?.duration ?? dragDurationCacheRef.current?.duration  // Keep cached duration
      }));
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

    // Get cached duration before clearing state
    const cachedDuration = externalDrag?.duration ?? dragDurationCacheRef.current?.duration;

    dragCounterRef.current = 0;
    setExternalDrag(null);

    // Check for composition from MediaPanel
    const compositionId = e.dataTransfer.getData('application/x-composition-id');
    if (compositionId) {
      const mediaStore = useMediaStore.getState();
      const comp = mediaStore.compositions.find(c => c.id === compositionId);
      if (comp) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);
        // Add composition as a clip (uses composition duration)
        addCompClip(trackId, comp, Math.max(0, startTime));
        return;
      }
    }

    // Check for media file from MediaPanel
    const mediaFileId = e.dataTransfer.getData('application/x-media-file-id');
    if (mediaFileId) {
      // Look up file from media store
      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
      if (mediaFile?.file) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);
        // Pass mediaFileId for proxy frame lookup
        addClip(trackId, mediaFile.file, Math.max(0, startTime), mediaFile.duration, mediaFileId);
        return;
      }
    }

    // Check for external file drop (from file explorer)
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/') || file.type.startsWith('audio/') || file.type.startsWith('image/')) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left + scrollX;
        const startTime = pixelToTime(x);
        addClip(trackId, file, Math.max(0, startTime), cachedDuration);
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

    // Get proxy status for this clip's media file
    const mediaStore = useMediaStore.getState();
    // Look up by mediaFileId first (most reliable), then by name
    const mediaFile = mediaStore.files.find(f =>
      f.id === clip.source?.mediaFileId ||
      f.name === clip.name ||
      f.name === clip.name.replace(' (Audio)', '')
    );
    const proxyProgress = mediaFile?.proxyProgress || 0;
    const proxyStatus = mediaFile?.proxyStatus;
    const isGeneratingProxy = proxyStatus === 'generating';
    const hasProxy = proxyStatus === 'ready';

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
      clip.source?.type || 'video',
      clip.isLoading ? 'loading' : '',
      hasProxy ? 'has-proxy' : '',
      isGeneratingProxy ? 'generating-proxy' : ''
    ].filter(Boolean).join(' ');

    return (
      <div
        key={clip.id}
        className={clipClass}
        style={{ left, width }}
        onMouseDown={(e) => handleClipMouseDown(e, clip.id)}
      >
        {/* Proxy progress bar */}
        {isGeneratingProxy && (
          <div className="clip-proxy-progress">
            <div
              className="clip-proxy-progress-bar"
              style={{ width: `${proxyProgress}%` }}
            />
          </div>
        )}
        {/* Proxy ready indicator */}
        {hasProxy && mediaStore.proxyEnabled && (
          <div className="clip-proxy-badge" title="Proxy ready">P</div>
        )}
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
          {clip.isLoading && <div className="clip-loading-spinner" />}
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

  // No active composition - show empty state
  if (!activeComposition) {
    return (
      <div className="timeline-container timeline-empty">
        <div className="timeline-empty-message">
          <p>No composition open</p>
          <p className="hint">Double-click a composition in the Media panel to open it</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`timeline-container ${clipDrag || clipTrim ? 'is-dragging' : ''}`}>
      {/* Timeline toolbar */}
      <div className="timeline-toolbar">
        <div className="timeline-controls">
          <button className="btn btn-sm" onClick={stop} title="Stop">⏹</button>
          <button className={`btn btn-sm ${isPlaying ? 'btn-active' : ''}`} onClick={isPlaying ? pause : play}>
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button
            className={`btn btn-sm ${loopPlayback ? 'btn-active' : ''}`}
            onClick={toggleLoopPlayback}
            title={loopPlayback ? 'Loop On (L)' : 'Loop Off (L)'}
          >
            🔁
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
        <div className="timeline-inout-controls">
          <button
            className={`btn btn-sm ${inPoint !== null ? 'btn-active' : ''}`}
            onClick={setInPointAtPlayhead}
            title="Set In point (I)"
          >
            I
          </button>
          <button
            className={`btn btn-sm ${outPoint !== null ? 'btn-active' : ''}`}
            onClick={setOutPointAtPlayhead}
            title="Set Out point (O)"
          >
            O
          </button>
          {(inPoint !== null || outPoint !== null) && (
            <button
              className="btn btn-sm"
              onClick={clearInOut}
              title="Clear In/Out (X)"
            >
              X
            </button>
          )}
        </div>
        <div className="timeline-ram-preview">
          <button
            className={`btn btn-sm ${ramPreviewEnabled ? 'btn-active' : ''}`}
            onClick={toggleRamPreviewEnabled}
            title={ramPreviewEnabled
              ? "RAM Preview ON - Auto-caches frames for instant scrubbing. Click to disable and clear cache."
              : "RAM Preview OFF - Click to enable auto-caching for instant scrubbing"}
          >
            RAM {ramPreviewEnabled ? 'ON' : 'OFF'}
          </button>
          <button
            className={`btn btn-sm ${proxyEnabled ? 'btn-active' : ''}`}
            onClick={() => setProxyEnabled(!proxyEnabled)}
            title={proxyEnabled ? 'Proxy enabled - using optimized frames' : 'Proxy disabled - using original video'}
          >
            {currentlyGeneratingProxyId ? (
              <>⏳ Generating...</>
            ) : (
              <>
                {proxyEnabled ? '🎬' : '🎥'} Proxy {proxyEnabled ? 'On' : 'Off'}
                {mediaFiles.filter(f => f.proxyStatus === 'ready').length > 0 && (
                  <span className="proxy-count">
                    ({mediaFiles.filter(f => f.proxyStatus === 'ready').length})
                  </span>
                )}
              </>
            )}
          </button>
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
          {(() => {
            // Check if any track of each type has solo enabled
            const anyVideoSolo = tracks.some(t => t.type === 'video' && t.solo);
            const anyAudioSolo = tracks.some(t => t.type === 'audio' && t.solo);

            return tracks.map(track => {
              // Determine if this track should be dimmed (another track of same type is solo'd, but not this one)
              const isDimmed = (track.type === 'video' && anyVideoSolo && !track.solo) ||
                               (track.type === 'audio' && anyAudioSolo && !track.solo);

              return (
                <div
                  key={track.id}
                  className={`track-header ${track.type} ${isDimmed ? 'dimmed' : ''}`}
                  style={{ height: track.height }}
                  onWheel={(e) => handleTrackHeaderWheel(e, track.id)}
                >
                  <span className="track-name">{track.name}</span>
                  <div className="track-controls">
                    <button
                      className={`btn-icon ${track.solo ? 'solo-active' : ''}`}
                      onClick={() => useTimelineStore.getState().setTrackSolo(track.id, !track.solo)}
                      title={track.solo ? 'Solo On' : 'Solo Off'}
                    >
                      S
                    </button>
                    {track.type === 'audio' && (
                      <button
                        className={`btn-icon ${track.muted ? 'muted' : ''}`}
                        onClick={() => useTimelineStore.getState().setTrackMuted(track.id, !track.muted)}
                        title={track.muted ? 'Unmute' : 'Mute'}
                      >
                        {track.muted ? '🔇' : '🔊'}
                      </button>
                    )}
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
              );
            });
          })()}
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
          {(() => {
            const anyVideoSolo = tracks.some(t => t.type === 'video' && t.solo);
            const anyAudioSolo = tracks.some(t => t.type === 'audio' && t.solo);

            return tracks.map(track => {
              const isDimmed = (track.type === 'video' && anyVideoSolo && !track.solo) ||
                               (track.type === 'audio' && anyAudioSolo && !track.solo);
              return (
            <div
              key={track.id}
              className={`track-lane ${track.type} ${isDimmed ? 'dimmed' : ''} ${clipDrag?.currentTrackId === track.id ? 'drag-target' : ''} ${externalDrag?.trackId === track.id || externalDrag?.audioTrackId === track.id ? 'external-drag-target' : ''}`}
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
                    width: timeToPixel(externalDrag.duration ?? 5),
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
                    width: timeToPixel(externalDrag.duration ?? 5),
                  }}
                >
                  <div className="clip-content">
                    <span className="clip-name">Audio</span>
                  </div>
                </div>
              )}
            </div>
              );
            });
          })()}

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
                  width: timeToPixel(externalDrag.duration ?? 5),
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

          {/* In/Out work area - grey out regions outside */}
          {(inPoint !== null || outPoint !== null) && (
            <>
              {/* Grey overlay before In point */}
              {inPoint !== null && inPoint > 0 && (
                <div
                  className="work-area-overlay before"
                  style={{
                    left: 0,
                    width: timeToPixel(inPoint),
                  }}
                />
              )}
              {/* Grey overlay after Out point */}
              {outPoint !== null && (
                <div
                  className="work-area-overlay after"
                  style={{
                    left: timeToPixel(outPoint),
                    width: timeToPixel(duration - outPoint),
                  }}
                />
              )}
            </>
          )}

          {/* RAM Preview progress text (during rendering) */}
          {isRamPreviewing && ramPreviewProgress !== null && (
            <div
              className="ram-preview-progress-text"
              style={{
                left: timeToPixel(playheadPosition) + 10,
              }}
            >
              {Math.round(ramPreviewProgress)}%
            </div>
          )}

          {/* Cached frames indicator (green = frames in RAM) */}
          {getCachedRanges().map((range, i) => (
            <div
              key={i}
              className="playback-cache-indicator"
              style={{
                left: timeToPixel(range.start),
                width: Math.max(2, timeToPixel(range.end - range.start)),
              }}
              title={`Cached: ${formatTime(range.start)} - ${formatTime(range.end)}`}
            />
          ))}

          {/* In point marker */}
          {inPoint !== null && (
            <div
              className={`in-out-marker in-marker ${markerDrag?.type === 'in' ? 'dragging' : ''}`}
              style={{ left: timeToPixel(inPoint) }}
              title={`In: ${formatTime(inPoint)} (drag to move)`}
            >
              <div
                className="marker-flag"
                onMouseDown={(e) => handleMarkerMouseDown(e, 'in')}
              >I</div>
              <div className="marker-line" />
            </div>
          )}

          {/* Out point marker */}
          {outPoint !== null && (
            <div
              className={`in-out-marker out-marker ${markerDrag?.type === 'out' ? 'dragging' : ''}`}
              style={{ left: timeToPixel(outPoint) }}
              title={`Out: ${formatTime(outPoint)} (drag to move)`}
            >
              <div
                className="marker-flag"
                onMouseDown={(e) => handleMarkerMouseDown(e, 'out')}
              >O</div>
              <div className="marker-line" />
            </div>
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
