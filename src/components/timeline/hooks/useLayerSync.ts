// useLayerSync - Syncs timeline clips to mixer layers for rendering
// Extracted from Timeline.tsx for better maintainability

import { useEffect, useRef, useCallback } from 'react';
import type { TimelineClip, TimelineTrack, Layer, Effect, NestedCompositionData, AnimatableProperty, BlendMode } from '../../../types';
import type { ClipDragState } from '../types';
import { useMixerStore } from '../../../stores/mixerStore';
import { useMediaStore } from '../../../stores/mediaStore';
import { engine } from '../../../engine/WebGPUEngine';
import { proxyFrameCache } from '../../../services/proxyFrameCache';
import { audioManager, audioStatusTracker } from '../../../services/audioManager';
import { Logger } from '../../../services/logger';

const log = Logger.create('useLayerSync');

interface UseLayerSyncProps {
  // Refs
  timelineRef: React.RefObject<HTMLDivElement | null>;

  // State
  playheadPosition: number;
  clips: TimelineClip[];
  tracks: TimelineTrack[];
  isPlaying: boolean;
  isDraggingPlayhead: boolean;
  ramPreviewRange: { start: number; end: number } | null;
  isRamPreviewing: boolean;
  clipKeyframes: Map<string, Array<{ id: string; clipId: string; time: number; property: AnimatableProperty; value: number; easing: string }>>;
  clipDrag: ClipDragState | null;
  zoom: number;
  scrollX: number;

  // Derived state
  clipMap: Map<string, TimelineClip>;
  videoTracks: TimelineTrack[];
  audioTracks: TimelineTrack[];

  // Helper functions
  getClipsAtTime: (time: number) => TimelineClip[];
  getInterpolatedTransform: (clipId: string, localTime: number) => {
    position: { x: number; y: number; z: number };
    scale: { x: number; y: number };
    rotation: { x: number; y: number; z: number };
    opacity: number;
    blendMode: BlendMode;
  };
  getInterpolatedEffects: (clipId: string, localTime: number) => Effect[];
  getInterpolatedSpeed: (clipId: string, localTime: number) => number;
  getSourceTimeForClip: (clipId: string, localTime: number) => number;
  isVideoTrackVisible: (track: TimelineTrack) => boolean;
  isAudioTrackMuted: (track: TimelineTrack) => boolean;
}

// Throttle layer sync during playback to reduce CPU load
const LAYER_SYNC_INTERVAL = 100; // Only do full layer sync every 100ms during playback

export function useLayerSync({
  timelineRef,
  playheadPosition,
  clips,
  tracks,
  isPlaying,
  isDraggingPlayhead,
  ramPreviewRange,
  isRamPreviewing,
  clipKeyframes,
  clipDrag,
  zoom,
  scrollX,
  clipMap,
  videoTracks,
  audioTracks,
  getClipsAtTime,
  getInterpolatedTransform,
  getInterpolatedEffects,
  getInterpolatedSpeed,
  getSourceTimeForClip,
  isVideoTrackVisible,
  isAudioTrackMuted,
}: UseLayerSyncProps): void {
  // Track last seek time to throttle during scrubbing
  const lastSeekRef = useRef<{ [clipId: string]: number }>({});
  const pendingSeekRef = useRef<{ [clipId: string]: number }>({});

  // Native decoder throttling
  const nativeDecoderLastFrameRef = useRef<{ [clipId: string]: number }>({});
  const nativeDecoderPendingRef = useRef<{ [clipId: string]: boolean }>({});

  // Track which clips are currently active (to detect clip changes)
  const activeClipIdsRef = useRef<string>('');

  // Throttle layer sync during playback
  const lastLayerSyncRef = useRef<number>(0);

  // Track current proxy frames for each clip (for smooth proxy playback)
  const proxyFramesRef = useRef<
    Map<string, { frameIndex: number; image: HTMLImageElement }>
  >(new Map());
  const proxyLoadingRef = useRef<Set<string>>(new Set());

  // Apply pending seeks when scrubbing stops
  useEffect(() => {
    if (isDraggingPlayhead) return;

    Object.entries(pendingSeekRef.current).forEach(([clipId, seekTime]) => {
      const clip = clipMap.get(clipId);
      if (clip?.source?.videoElement) {
        clip.source.videoElement.currentTime = seekTime;
      }
    });
    pendingSeekRef.current = {};
  }, [isDraggingPlayhead, clipMap]);

  // Helper: Check if effects have changed
  const effectsChanged = useCallback(
    (layerEffects: Effect[] | undefined, clipEffects: Effect[] | undefined): boolean => {
      const le = layerEffects || [];
      const ce = clipEffects || [];
      if (le.length !== ce.length) return true;
      for (let i = 0; i < le.length; i++) {
        if (le[i].id !== ce[i].id || le[i].enabled !== ce[i].enabled) return true;
        const lp = le[i].params;
        const cp = ce[i].params;
        const lKeys = Object.keys(lp);
        const cKeys = Object.keys(cp);
        if (lKeys.length !== cKeys.length) return true;
        for (const key of lKeys) {
          if (lp[key] !== cp[key]) return true;
        }
      }
      return false;
    },
    []
  );

  // Build Layer objects from nested clips for pre-rendering
  const buildNestedLayers = useCallback(
    (clip: TimelineClip, clipTime: number): Layer[] => {
      if (!clip.nestedClips || !clip.nestedTracks) return [];

      const nestedVideoTracks = clip.nestedTracks.filter(
        (t) => t.type === 'video' && t.visible
      );

      const layers: Layer[] = [];

      for (let i = nestedVideoTracks.length - 1; i >= 0; i--) {
        const nestedTrack = nestedVideoTracks[i];
        const nestedClip = clip.nestedClips.find(
          (nc) =>
            nc.trackId === nestedTrack.id &&
            clipTime >= nc.startTime &&
            clipTime < nc.startTime + nc.duration
        );

        if (!nestedClip) continue;

        const nestedLocalTime = clipTime - nestedClip.startTime;
        const nestedClipTime = nestedClip.reversed
          ? nestedClip.outPoint - nestedLocalTime
          : nestedLocalTime + nestedClip.inPoint;

        // Update video currentTime
        if (nestedClip.source?.videoElement) {
          const video = nestedClip.source.videoElement;
          const timeDiff = Math.abs(video.currentTime - nestedClipTime);
          if (timeDiff > 0.05) {
            video.currentTime = nestedClipTime;
          }
          if (isPlaying && video.paused) {
            video.play().catch(() => {});
          } else if (!isPlaying && !video.paused) {
            video.pause();
          }
        }

        // Build layer from nested clip
        const transform = nestedClip.transform || {
          position: { x: 0, y: 0, z: 0 },
          scale: { x: 1, y: 1 },
          rotation: { x: 0, y: 0, z: 0 },
          anchor: { x: 0.5, y: 0.5 },
          opacity: 1,
          blendMode: 'normal' as const,
        };

        if (nestedClip.source?.videoElement) {
          layers.push({
            id: `nested-layer-${nestedClip.id}`,
            name: nestedClip.name,
            visible: true,
            opacity: transform.opacity ?? 1,
            blendMode: transform.blendMode || 'normal',
            source: {
              type: 'video',
              videoElement: nestedClip.source.videoElement,
              webCodecsPlayer: nestedClip.source.webCodecsPlayer,
            },
            effects: nestedClip.effects || [],
            position: {
              x: transform.position?.x || 0,
              y: transform.position?.y || 0,
              z: transform.position?.z || 0,
            },
            scale: {
              x: transform.scale?.x ?? 1,
              y: transform.scale?.y ?? 1,
            },
            rotation: {
              x: ((transform.rotation?.x || 0) * Math.PI) / 180,
              y: ((transform.rotation?.y || 0) * Math.PI) / 180,
              z: ((transform.rotation?.z || 0) * Math.PI) / 180,
            },
          });
        } else if (nestedClip.source?.imageElement) {
          layers.push({
            id: `nested-layer-${nestedClip.id}`,
            name: nestedClip.name,
            visible: true,
            opacity: transform.opacity ?? 1,
            blendMode: transform.blendMode || 'normal',
            source: {
              type: 'image',
              imageElement: nestedClip.source.imageElement,
            },
            effects: nestedClip.effects || [],
            position: {
              x: transform.position?.x || 0,
              y: transform.position?.y || 0,
              z: transform.position?.z || 0,
            },
            scale: {
              x: transform.scale?.x ?? 1,
              y: transform.scale?.y ?? 1,
            },
            rotation: {
              x: ((transform.rotation?.x || 0) * Math.PI) / 180,
              y: ((transform.rotation?.y || 0) * Math.PI) / 180,
              z: ((transform.rotation?.z || 0) * Math.PI) / 180,
            },
          });
        }
      }

      return layers;
    },
    [isPlaying]
  );

  // Main layer sync effect
  // PERFORMANCE: During playback, layerBuilder handles all sync in the RAF loop
  // This effect only runs when paused (for scrubbing, editing, etc.)
  useEffect(() => {
    // Skip all work during playback - layerBuilder handles video/audio sync in RAF
    if (isPlaying) {
      return;
    }

    if (isRamPreviewing) {
      return;
    }

    // Try to use RAM Preview cache for instant scrubbing
    // This provides instant access to pre-rendered frames
    if (ramPreviewRange) {
      const inRange = playheadPosition >= ramPreviewRange.start && playheadPosition <= ramPreviewRange.end;
      if (inRange) {
        if (engine.renderCachedFrame(playheadPosition)) {
          return; // Cache hit - instant render, no video seek needed
        }
        // Cache miss within range - will fall through to regular render
      }
    } else {
      // No RAM preview range, but still try the cache in case frames were cached during playback
      if (engine.renderCachedFrame(playheadPosition)) {
        return;
      }
    }

    let clipsAtTime = getClipsAtTime(playheadPosition);

    if (clipDrag) {
      const draggedClipId = clipDrag.clipId;
      const rawPixelX = clipDrag.currentX
        ? clipDrag.currentX -
          (timelineRef.current?.getBoundingClientRect().left || 0) +
          scrollX -
          clipDrag.grabOffsetX
        : 0;
      const tempStartTime =
        clipDrag.snappedTime ??
        (clipDrag.currentX ? Math.max(0, rawPixelX / zoom) : null);

      if (tempStartTime !== null) {
        const modifiedClips = clips.map((c) => {
          if (c.id === draggedClipId) {
            return { ...c, startTime: tempStartTime, trackId: clipDrag.currentTrackId };
          }
          return c;
        });
        clipsAtTime = modifiedClips.filter(
          (c) =>
            playheadPosition >= c.startTime &&
            playheadPosition < c.startTime + c.duration
        );
      }
    }

    const currentActiveIds = clipsAtTime
      .filter(
        (c) =>
          c.source?.type === 'video' ||
          c.source?.type === 'image' ||
          c.source?.type === 'text' ||
          c.isComposition
      )
      .map((c) => c.id)
      .sort()
      .join(',');

    // During playback, throttle the heavy layer sync work to reduce CPU load
    const now = performance.now();
    const timeSinceLastSync = now - lastLayerSyncRef.current;
    const clipsChanged = activeClipIdsRef.current !== ('playing:' + currentActiveIds);

    if (isPlaying) {
      // Always ensure videos are playing (this is cheap)
      clipsAtTime.forEach((clip) => {
        if (clip.source?.videoElement?.paused) {
          clip.source.videoElement.play().catch(() => {});
        }
      });
      activeClipIdsRef.current = 'playing:' + currentActiveIds;

      // Skip heavy layer sync if within throttle interval AND clips haven't changed
      if (timeSinceLastSync < LAYER_SYNC_INTERVAL && !clipsChanged) {
        return;
      }
      lastLayerSyncRef.current = now;
    } else {
      activeClipIdsRef.current = '';
    }

    const currentLayers = useMixerStore.getState().layers;
    const newLayers = [...currentLayers];
    let layersChanged = false;

    videoTracks.forEach((track, layerIndex) => {
      const clip = clipsAtTime.find((c) => c.trackId === track.id);
      const layer = currentLayers[layerIndex];

      if (clip?.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;

        // Build all nested layers for pre-rendering
        const nestedLayers = buildNestedLayers(clip, clipTime);

        // Get parent clip transform
        const interpolatedTransform = getInterpolatedTransform(clip.id, clipTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, clipTime);

        // Get composition dimensions from mediaStore
        const mediaStore = useMediaStore.getState();
        const composition = mediaStore.compositions.find(c => c.id === clip.compositionId);
        const compWidth = composition?.width || 1920;
        const compHeight = composition?.height || 1080;

        if (nestedLayers.length > 0) {
          const trackVisible = isVideoTrackVisible(track);

          // Build nestedComposition data for pre-rendering
          const nestedCompData: NestedCompositionData = {
            compositionId: clip.compositionId || clip.id,
            layers: nestedLayers,
            width: compWidth,
            height: compHeight,
          };

          // Always update nested composition layers (they change with time)
          newLayers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: trackVisible,
            opacity: interpolatedTransform.opacity,
            blendMode: interpolatedTransform.blendMode,
            source: {
              type: 'video', // Will be pre-rendered to texture
              nestedComposition: nestedCompData,
            },
            effects: interpolatedEffects,
            position: { x: interpolatedTransform.position.x, y: interpolatedTransform.position.y, z: interpolatedTransform.position.z },
            scale: { x: interpolatedTransform.scale.x, y: interpolatedTransform.scale.y },
            rotation: {
              x: (interpolatedTransform.rotation.x * Math.PI) / 180,
              y: (interpolatedTransform.rotation.y * Math.PI) / 180,
              z: (interpolatedTransform.rotation.z * Math.PI) / 180,
            },
          };
          layersChanged = true;
        } else {
          if (layer?.source) {
            newLayers[layerIndex] = undefined as unknown as (typeof newLayers)[0];
            layersChanged = true;
          }
        }
      } else if (clip?.source?.nativeDecoder) {
        // Native Helper decoder for ProRes/DNxHD (turbo mode)
        const nativeDecoder = clip.source.nativeDecoder;
        const clipLocalTime = playheadPosition - clip.startTime;
        const keyframeLocalTime = clipLocalTime;

        // Calculate source time using speed integration (handles keyframes)
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

        // Calculate target frame and throttle seeks
        const fps = nativeDecoder.fps || 25;
        const targetFrame = Math.round(clipTime * fps);
        const lastFrame = nativeDecoderLastFrameRef.current[clip.id] ?? -1;
        const isPending = nativeDecoderPendingRef.current[clip.id] || false;

        // Only seek if frame changed and no pending seek
        if (targetFrame !== lastFrame && !isPending) {
          nativeDecoderLastFrameRef.current[clip.id] = targetFrame;
          nativeDecoderPendingRef.current[clip.id] = true;

          // Use fast scrub (scaled down) during playhead drag for smoother scrubbing
          nativeDecoder.seekToFrame(targetFrame, isDraggingPlayhead)
            .then(() => {
              nativeDecoderPendingRef.current[clip.id] = false;
            })
            .catch((err) => {
              nativeDecoderPendingRef.current[clip.id] = false;
              log.warn('NativeDecoder seek failed:', err);
            });
        }

        const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
        const nativeInterpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);
        const trackVisible = isVideoTrackVisible(track);

        const needsUpdate =
          !layer ||
          layer.visible !== trackVisible ||
          layer.source?.nativeDecoder !== nativeDecoder ||
          layer.opacity !== transform.opacity ||
          layer.blendMode !== transform.blendMode ||
          layer.position.x !== transform.position.x ||
          layer.position.y !== transform.position.y ||
          layer.position.z !== transform.position.z ||
          layer.scale.x !== transform.scale.x ||
          layer.scale.y !== transform.scale.y ||
          (layer.rotation as { z?: number })?.z !== (transform.rotation.z * Math.PI) / 180 ||
          (layer.rotation as { x?: number })?.x !== (transform.rotation.x * Math.PI) / 180 ||
          (layer.rotation as { y?: number })?.y !== (transform.rotation.y * Math.PI) / 180 ||
          effectsChanged(layer.effects, nativeInterpolatedEffects);

        if (needsUpdate) {
          newLayers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: trackVisible,
            opacity: transform.opacity,
            blendMode: transform.blendMode,
            source: {
              type: 'video',
              nativeDecoder: nativeDecoder,
            },
            effects: nativeInterpolatedEffects,
            position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
            scale: { x: transform.scale.x, y: transform.scale.y },
            rotation: {
              x: (transform.rotation.x * Math.PI) / 180,
              y: (transform.rotation.y * Math.PI) / 180,
              z: (transform.rotation.z * Math.PI) / 180,
            },
          };
          layersChanged = true;
        }
      } else if (clip?.source?.videoElement) {
        const clipLocalTime = playheadPosition - clip.startTime;
        // Keyframe interpolation uses timeline-local time
        const keyframeLocalTime = clipLocalTime;
        // Calculate source time using speed integration (handles keyframes)
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        // Determine start point based on INITIAL speed (speed at t=0), not clip.speed
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
        const video = clip.source.videoElement;
        const webCodecsPlayer = clip.source.webCodecsPlayer;
        const timeDiff = Math.abs(video.currentTime - clipTime);

        const mediaStore = useMediaStore.getState();
        const mediaFile = mediaStore.files.find(
          (f) => f.name === clip.name || clip.mediaFileId === f.id
        );
        const proxyFps = mediaFile?.proxyFps || 30;

        const frameIndex = Math.floor(clipTime * proxyFps);
        let useProxy = false;

        if (mediaStore.proxyEnabled && mediaFile?.proxyFps) {
          if (mediaFile.proxyStatus === 'ready') {
            useProxy = true;
          } else if (
            mediaFile.proxyStatus === 'generating' &&
            (mediaFile.proxyProgress || 0) > 0
          ) {
            const totalFrames = Math.ceil(
              (mediaFile.duration || 10) * proxyFps
            );
            const maxGeneratedFrame = Math.floor(
              totalFrames * ((mediaFile.proxyProgress || 0) / 100)
            );
            useProxy = frameIndex < maxGeneratedFrame;
          }
        }

        if (useProxy && mediaFile) {
          const cacheKey = `${mediaFile.id}_${clip.id}`;
          const cached = proxyFramesRef.current.get(cacheKey);

          if (!video.muted) {
            video.muted = true;
          }
          if (isPlaying && video.paused) {
            video.play().catch(() => {});
          } else if (!isPlaying && !video.paused) {
            video.pause();
          }

          if (!isPlaying) {
            const timeDiff2 = Math.abs(video.currentTime - clipTime);
            if (timeDiff2 > 0.1) {
              video.currentTime = clipTime;
            }
          }

          const loadKey = `${mediaFile.id}_${frameIndex}`;
          const cachedInService = proxyFrameCache.getCachedFrame(
            mediaFile.id,
            frameIndex,
            proxyFps
          );
          const interpolatedEffectsForProxy = getInterpolatedEffects(
            clip.id,
            keyframeLocalTime
          );

          if (cachedInService) {
            proxyFramesRef.current.set(cacheKey, {
              frameIndex,
              image: cachedInService,
            });

            const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
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
              effects: interpolatedEffectsForProxy,
              position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
              scale: { x: transform.scale.x, y: transform.scale.y },
              rotation: {
                x: (transform.rotation.x * Math.PI) / 180,
                y: (transform.rotation.y * Math.PI) / 180,
                z: (transform.rotation.z * Math.PI) / 180,
              },
            };
            layersChanged = true;
          } else if (!cached || cached.frameIndex !== frameIndex) {
            if (!proxyLoadingRef.current.has(loadKey)) {
              proxyLoadingRef.current.add(loadKey);

              const capturedLayerIndex = layerIndex;
              const capturedTransform = getInterpolatedTransform(
                clip.id,
                keyframeLocalTime
              );
              const capturedTrackVisible = isVideoTrackVisible(track);
              const capturedClipName = clip.name;
              const capturedEffects = interpolatedEffectsForProxy;

              proxyFrameCache
                .getFrame(mediaFile.id, clipTime, proxyFps)
                .then((image) => {
                  proxyLoadingRef.current.delete(loadKey);
                  if (image) {
                    proxyFramesRef.current.set(cacheKey, { frameIndex, image });

                    const currentLayers2 = useMixerStore.getState().layers;
                    const updatedLayers = [...currentLayers2];
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
                      effects: capturedEffects,
                      position: {
                        x: capturedTransform.position.x,
                        y: capturedTransform.position.y,
                        z: capturedTransform.position.z,
                      },
                      scale: {
                        x: capturedTransform.scale.x,
                        y: capturedTransform.scale.y,
                      },
                      rotation: {
                        x: (capturedTransform.rotation.x * Math.PI) / 180,
                        y: (capturedTransform.rotation.y * Math.PI) / 180,
                        z: (capturedTransform.rotation.z * Math.PI) / 180,
                      },
                    };
                    useMixerStore.setState({ layers: updatedLayers });
                  }
                });
            }

            if (cached?.image) {
              const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
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
                effects: interpolatedEffectsForProxy,
                position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
                scale: { x: transform.scale.x, y: transform.scale.y },
                rotation: {
                  x: (transform.rotation.x * Math.PI) / 180,
                  y: (transform.rotation.y * Math.PI) / 180,
                  z: (transform.rotation.z * Math.PI) / 180,
                },
              };
              layersChanged = true;
            }
          } else if (cached?.image) {
            const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
            const trackVisible = isVideoTrackVisible(track);
            const needsUpdate =
              !layer ||
              layer.visible !== trackVisible ||
              layer.source?.imageElement !== cached.image ||
              layer.source?.type !== 'image' ||
              effectsChanged(layer.effects, interpolatedEffectsForProxy);

            if (needsUpdate) {
              newLayers[layerIndex] = {
                id: `timeline_layer_${layerIndex}`,
                name: clip.name,
                visible: trackVisible,
                opacity: transform.opacity,
                blendMode: transform.blendMode,
                source: {
                  type: 'image',
                  imageElement: cached.image,
                },
                effects: interpolatedEffectsForProxy,
                position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
                scale: { x: transform.scale.x, y: transform.scale.y },
                rotation: {
                  x: (transform.rotation.x * Math.PI) / 180,
                  y: (transform.rotation.y * Math.PI) / 180,
                  z: (transform.rotation.z * Math.PI) / 180,
                },
              };
              layersChanged = true;
            }
          }
        } else {
          if (webCodecsPlayer) {
            const wcTimeDiff = Math.abs(webCodecsPlayer.currentTime - clipTime);
            if (wcTimeDiff > 0.05) {
              webCodecsPlayer.seek(clipTime);
            }
          }

          if (clip.reversed) {
            if (!video.paused) {
              video.pause();
            }
            const seekThreshold = isDraggingPlayhead ? 0.1 : 0.03;
            if (timeDiff > seekThreshold) {
              const now = performance.now();
              const lastSeek = lastSeekRef.current[clip.id] || 0;
              if (now - lastSeek > 33) {
                video.currentTime = clipTime;
                lastSeekRef.current[clip.id] = now;
              }
            }
          } else {
            if (isPlaying && video.paused) {
              video.play().catch(() => {});
            } else if (!isPlaying && !video.paused) {
              video.pause();
            }

            if (!isPlaying) {
              const seekThreshold = isDraggingPlayhead ? 0.1 : 0.05;

              if (timeDiff > seekThreshold) {
                const now = performance.now();
                const lastSeek = lastSeekRef.current[clip.id] || 0;

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
          }

          const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
          const videoInterpolatedEffects = getInterpolatedEffects(
            clip.id,
            keyframeLocalTime
          );
          const trackVisible = isVideoTrackVisible(track);
          const needsUpdate =
            !layer ||
            layer.visible !== trackVisible ||
            layer.source?.videoElement !== video ||
            layer.source?.webCodecsPlayer !== webCodecsPlayer ||
            layer.opacity !== transform.opacity ||
            layer.blendMode !== transform.blendMode ||
            layer.position.x !== transform.position.x ||
            layer.position.y !== transform.position.y ||
            layer.position.z !== transform.position.z ||
            layer.scale.x !== transform.scale.x ||
            layer.scale.y !== transform.scale.y ||
            (layer.rotation as { z?: number })?.z !==
              (transform.rotation.z * Math.PI) / 180 ||
            (layer.rotation as { x?: number })?.x !==
              (transform.rotation.x * Math.PI) / 180 ||
            (layer.rotation as { y?: number })?.y !==
              (transform.rotation.y * Math.PI) / 180 ||
            effectsChanged(layer.effects, videoInterpolatedEffects);

          if (needsUpdate) {
            newLayers[layerIndex] = {
              id: `timeline_layer_${layerIndex}`,
              name: clip.name,
              visible: trackVisible,
              opacity: transform.opacity,
              blendMode: transform.blendMode,
              source: {
                type: 'video',
                videoElement: video,
                webCodecsPlayer: webCodecsPlayer,
              },
              effects: videoInterpolatedEffects,
              position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
              scale: { x: transform.scale.x, y: transform.scale.y },
              rotation: {
                x: (transform.rotation.x * Math.PI) / 180,
                y: (transform.rotation.y * Math.PI) / 180,
                z: (transform.rotation.z * Math.PI) / 180,
              },
            };
            layersChanged = true;
          }
        }
      } else if (clip?.source?.imageElement) {
        const img = clip.source.imageElement;
        const imageClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, imageClipLocalTime);
        const imageInterpolatedEffects = getInterpolatedEffects(
          clip.id,
          imageClipLocalTime
        );
        const trackVisible = isVideoTrackVisible(track);
        const needsUpdate =
          !layer ||
          layer.visible !== trackVisible ||
          layer.source?.imageElement !== img ||
          layer.opacity !== transform.opacity ||
          layer.blendMode !== transform.blendMode ||
          layer.position.x !== transform.position.x ||
          layer.position.y !== transform.position.y ||
          layer.position.z !== transform.position.z ||
          layer.scale.x !== transform.scale.x ||
          layer.scale.y !== transform.scale.y ||
          (layer.rotation as { z?: number })?.z !==
            (transform.rotation.z * Math.PI) / 180 ||
          (layer.rotation as { x?: number })?.x !==
            (transform.rotation.x * Math.PI) / 180 ||
          (layer.rotation as { y?: number })?.y !==
            (transform.rotation.y * Math.PI) / 180 ||
          effectsChanged(layer.effects, imageInterpolatedEffects);

        if (needsUpdate) {
          newLayers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: trackVisible,
            opacity: transform.opacity,
            blendMode: transform.blendMode,
            source: {
              type: 'image',
              imageElement: img,
            },
            effects: imageInterpolatedEffects,
            position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
            scale: { x: transform.scale.x, y: transform.scale.y },
            rotation: {
              x: (transform.rotation.x * Math.PI) / 180,
              y: (transform.rotation.y * Math.PI) / 180,
              z: (transform.rotation.z * Math.PI) / 180,
            },
          };
          layersChanged = true;
        }
      } else if (clip?.source?.textCanvas) {
        // Text clip handling
        const textCanvas = clip.source.textCanvas;
        const textClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, textClipLocalTime);
        const textInterpolatedEffects = getInterpolatedEffects(
          clip.id,
          textClipLocalTime
        );
        const trackVisible = isVideoTrackVisible(track);
        const needsUpdate =
          !layer ||
          layer.visible !== trackVisible ||
          layer.source?.textCanvas !== textCanvas ||
          layer.opacity !== transform.opacity ||
          layer.blendMode !== transform.blendMode ||
          layer.position.x !== transform.position.x ||
          layer.position.y !== transform.position.y ||
          layer.position.z !== transform.position.z ||
          layer.scale.x !== transform.scale.x ||
          layer.scale.y !== transform.scale.y ||
          (layer.rotation as { z?: number })?.z !==
            (transform.rotation.z * Math.PI) / 180 ||
          (layer.rotation as { x?: number })?.x !==
            (transform.rotation.x * Math.PI) / 180 ||
          (layer.rotation as { y?: number })?.y !==
            (transform.rotation.y * Math.PI) / 180 ||
          effectsChanged(layer.effects, textInterpolatedEffects);

        if (needsUpdate) {
          newLayers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: trackVisible,
            opacity: transform.opacity,
            blendMode: transform.blendMode,
            source: {
              type: 'text',
              textCanvas: textCanvas,
            },
            effects: textInterpolatedEffects,
            position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
            scale: { x: transform.scale.x, y: transform.scale.y },
            rotation: {
              x: (transform.rotation.x * Math.PI) / 180,
              y: (transform.rotation.y * Math.PI) / 180,
              z: (transform.rotation.z * Math.PI) / 180,
            },
          };
          layersChanged = true;
        }
      } else {
        if (layer?.source) {
          newLayers[layerIndex] = undefined as unknown as (typeof newLayers)[0];
          layersChanged = true;
        }
      }
    });

    if (layersChanged) {
      useMixerStore.setState({ layers: newLayers });
      // Trigger one-shot render when paused (render loop not running)
      // This ensures text edits, property changes, etc. update the preview immediately
      engine.render(newLayers);
    }

    // Audio sync with status tracking
    let audioPlayingCount = 0;
    let maxAudioDrift = 0;
    let hasAudioError = false;

    // Resume audio context if needed (browser autoplay policy)
    if (isPlaying && !isDraggingPlayhead) {
      audioManager.resume().catch(() => {});
    }

    audioTracks.forEach((track) => {
      const clip = clipsAtTime.find((c) => c.trackId === track.id);

      if (clip?.source?.audioElement) {
        const audio = clip.source.audioElement;
        const clipLocalTime = playheadPosition - clip.startTime;

        // Get current speed for this clip (accounts for keyframes)
        const currentSpeed = getInterpolatedSpeed(clip.id, clipLocalTime);
        const absSpeed = Math.abs(currentSpeed);

        // Calculate source time using speed integration
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

        const timeDiff = audio.currentTime - clipTime;

        // Track drift for stats
        if (Math.abs(timeDiff) > maxAudioDrift) {
          maxAudioDrift = Math.abs(timeDiff);
        }

        const effectivelyMuted = isAudioTrackMuted(track);
        audio.muted = effectivelyMuted;

        // Set playback rate for speed effect (use absolute value, negative speed not supported for audio)
        const targetRate = absSpeed > 0.1 ? absSpeed : 1;
        if (Math.abs(audio.playbackRate - targetRate) > 0.01) {
          audio.playbackRate = Math.max(0.25, Math.min(4, targetRate));
        }

        // Set preservesPitch based on clip setting (default true)
        const shouldPreservePitch = clip.preservesPitch !== false;
        if ((audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch !== shouldPreservePitch) {
          (audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch = shouldPreservePitch;
        }

        const shouldPlay = isPlaying && !effectivelyMuted && !isDraggingPlayhead && absSpeed > 0.1;

        if (shouldPlay) {
          // Only sync audio on significant drift (>200ms) to avoid constant glitches
          if (Math.abs(timeDiff) > 0.2) {
            audio.currentTime = clipTime;
          }

          // Ensure audio is playing - sync on start
          if (audio.paused) {
            audio.currentTime = clipTime;
            audio.play().catch((err) => {
              log.warn('Audio failed to play:', err.message);
              hasAudioError = true;
            });
          }

          if (!audio.paused && !effectivelyMuted) {
            audioPlayingCount++;
          }
        } else {
          if (!audio.paused) {
            audio.pause();
          }
        }
      }
    });

    // Pause audio clips that are no longer at playhead
    clips.forEach((clip) => {
      if (clip.source?.audioElement) {
        const isAtPlayhead = clipsAtTime.some((c) => c.id === clip.id);
        if (!isAtPlayhead && !clip.source.audioElement.paused) {
          clip.source.audioElement.pause();
        }
      }
      // Also pause nested composition mixdown audio
      if (clip.mixdownAudio) {
        const isAtPlayhead = clipsAtTime.some((c) => c.id === clip.id);
        if (!isAtPlayhead && !clip.mixdownAudio.paused) {
          clip.mixdownAudio.pause();
        }
      }
    });

    // Play nested composition mixdown audio for clips at playhead
    clipsAtTime.forEach((clip) => {
      if (clip.isComposition && clip.mixdownAudio && clip.hasMixdownAudio) {
        const audio = clip.mixdownAudio;
        const clipLocalTime = playheadPosition - clip.startTime;

        // Get current speed for this clip (accounts for keyframes)
        const currentSpeed = getInterpolatedSpeed(clip.id, clipLocalTime);
        const absSpeed = Math.abs(currentSpeed);

        // Calculate source time using speed integration
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

        // Find the track this clip is on
        const track = videoTracks.find(t => t.id === clip.trackId);
        const effectivelyMuted = track ? !isVideoTrackVisible(track) : false;
        audio.muted = effectivelyMuted;

        // Set playback rate for speed effect
        const targetRate = absSpeed > 0.1 ? absSpeed : 1;
        if (Math.abs(audio.playbackRate - targetRate) > 0.01) {
          audio.playbackRate = Math.max(0.25, Math.min(4, targetRate));
        }

        // Set preservesPitch based on clip setting
        const shouldPreservePitch = clip.preservesPitch !== false;
        if ((audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch !== shouldPreservePitch) {
          (audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch = shouldPreservePitch;
        }

        const timeDiff = audio.currentTime - clipTime;

        // Track drift for stats
        if (Math.abs(timeDiff) > maxAudioDrift) {
          maxAudioDrift = Math.abs(timeDiff);
        }

        const shouldPlay = isPlaying && !effectivelyMuted && !isDraggingPlayhead && absSpeed > 0.1;

        if (shouldPlay) {
          // Only sync audio on significant drift to avoid pops
          if (Math.abs(timeDiff) > 0.2) {
            audio.currentTime = clipTime;
          }

          // Ensure audio is playing
          if (audio.paused) {
            audio.currentTime = clipTime;
            audio.play().catch((err) => {
              log.warn('Nested Comp Audio failed to play:', err.message);
            });
          }

          if (!audio.paused && !effectivelyMuted) {
            audioPlayingCount++;
          }
        } else {
          if (!audio.paused) {
            audio.pause();
          }
        }
      }
    });

    // Update audio status for stats display
    audioStatusTracker.updateStatus(audioPlayingCount, maxAudioDrift, hasAudioError);
  }, [
    playheadPosition,
    clips,
    tracks,
    isPlaying,
    isDraggingPlayhead,
    ramPreviewRange,
    isRamPreviewing,
    clipKeyframes,
    clipDrag,
    zoom,
    scrollX,
    getClipsAtTime,
    getInterpolatedTransform,
    getInterpolatedEffects,
    getInterpolatedSpeed,
    getSourceTimeForClip,
    videoTracks,
    audioTracks,
    isVideoTrackVisible,
    isAudioTrackMuted,
    buildNestedLayers,
    effectsChanged,
    timelineRef,
    clipMap,
  ]);
}
