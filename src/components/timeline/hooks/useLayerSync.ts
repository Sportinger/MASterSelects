// useLayerSync - Syncs timeline clips to mixer layers for rendering
// Extracted from Timeline.tsx for better maintainability

import { useEffect, useRef, useCallback } from 'react';
import type { TimelineClip, TimelineTrack, Layer, Effect, NestedCompositionData, AnimatableProperty, BlendMode, ClipTransform } from '../../../types';
import type { ClipDragState } from '../types';
import { useTimelineStore } from '../../../stores/timeline';
import { useMediaStore } from '../../../stores/mediaStore';
import { engine } from '../../../engine/WebGPUEngine';
import { getProxyPlayer } from '../../../services/proxyPlayer';
import { audioManager, audioStatusTracker } from '../../../services/audioManager';
import { Logger } from '../../../services/logger';
import { getInterpolatedClipTransform } from '../../../utils/keyframeInterpolation';
import { DEFAULT_TRANSFORM } from '../../../stores/timeline/constants';

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

  // RAF debounce: batch rapid scrubbing into one sync per animation frame
  const pendingRafRef = useRef<number | null>(null);

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

      // Iterate forwards to maintain correct layer order (track 0 = bottom, track N = top)
      for (let i = 0; i < nestedVideoTracks.length; i++) {
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

        // Update video currentTime and WebCodecsPlayer
        if (nestedClip.source?.videoElement) {
          const video = nestedClip.source.videoElement;
          const webCodecsPlayer = nestedClip.source.webCodecsPlayer;
          const timeDiff = Math.abs(video.currentTime - nestedClipTime);
          if (timeDiff > 0.05) {
            video.currentTime = nestedClipTime;
          }
          // Seek WebCodecsPlayer for nested clips (critical for preview during playback)
          if (webCodecsPlayer) {
            const wcTimeDiff = Math.abs(webCodecsPlayer.currentTime - nestedClipTime);
            if (wcTimeDiff > 0.05) {
              webCodecsPlayer.seek(nestedClipTime);
            }
          }
          if (isPlaying && video.paused) {
            video.play().catch(() => {});
          } else if (!isPlaying && !video.paused) {
            video.pause();
          }
        }

        // Get keyframes for the nested clip from store
        const { clipKeyframes: storeClipKeyframes } = useTimelineStore.getState();
        const keyframes = storeClipKeyframes.get(nestedClip.id) || [];

        // Build base transform from the nested clip's static transform
        const baseTransform: ClipTransform = {
          opacity: nestedClip.transform?.opacity ?? DEFAULT_TRANSFORM.opacity,
          blendMode: nestedClip.transform?.blendMode ?? DEFAULT_TRANSFORM.blendMode,
          position: {
            x: nestedClip.transform?.position?.x ?? DEFAULT_TRANSFORM.position.x,
            y: nestedClip.transform?.position?.y ?? DEFAULT_TRANSFORM.position.y,
            z: nestedClip.transform?.position?.z ?? DEFAULT_TRANSFORM.position.z,
          },
          scale: {
            x: nestedClip.transform?.scale?.x ?? DEFAULT_TRANSFORM.scale.x,
            y: nestedClip.transform?.scale?.y ?? DEFAULT_TRANSFORM.scale.y,
          },
          rotation: {
            x: nestedClip.transform?.rotation?.x ?? DEFAULT_TRANSFORM.rotation.x,
            y: nestedClip.transform?.rotation?.y ?? DEFAULT_TRANSFORM.rotation.y,
            z: nestedClip.transform?.rotation?.z ?? DEFAULT_TRANSFORM.rotation.z,
          },
        };

        // Interpolate transform using keyframes (supports opacity fades, position animations, etc.)
        const transform = keyframes.length > 0
          ? getInterpolatedClipTransform(keyframes, nestedLocalTime, baseTransform)
          : baseTransform;

        // Interpolate effect parameters if there are effect keyframes
        const effectKeyframes = keyframes.filter(k => k.property.startsWith('effect.'));
        let effects = nestedClip.effects || [];
        if (effectKeyframes.length > 0 && effects.length > 0) {
          effects = effects.map(effect => {
            const newParams = { ...effect.params };
            Object.keys(effect.params).forEach(paramName => {
              if (typeof effect.params[paramName] !== 'number') return;
              const propertyKey = `effect.${effect.id}.${paramName}`;
              const paramKeyframes = effectKeyframes.filter(k => k.property === propertyKey);
              if (paramKeyframes.length > 0) {
                // Simple linear interpolation for effect params
                const sorted = [...paramKeyframes].sort((a, b) => a.time - b.time);
                if (nestedLocalTime <= sorted[0].time) {
                  newParams[paramName] = sorted[0].value;
                } else if (nestedLocalTime >= sorted[sorted.length - 1].time) {
                  newParams[paramName] = sorted[sorted.length - 1].value;
                } else {
                  for (let i = 0; i < sorted.length - 1; i++) {
                    if (nestedLocalTime >= sorted[i].time && nestedLocalTime <= sorted[i + 1].time) {
                      const t = (nestedLocalTime - sorted[i].time) / (sorted[i + 1].time - sorted[i].time);
                      newParams[paramName] = sorted[i].value + t * (sorted[i + 1].value - sorted[i].value);
                      break;
                    }
                  }
                }
              }
            });
            return { ...effect, params: newParams };
          });
        }

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
            effects,
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
            effects,
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

  // Cleanup pending RAF on unmount
  useEffect(() => {
    return () => {
      if (pendingRafRef.current !== null) {
        cancelAnimationFrame(pendingRafRef.current);
        pendingRafRef.current = null;
      }
    };
  }, []);

  // Main layer sync effect
  // PERFORMANCE: During playback, layerBuilder handles all sync in the RAF loop
  // This effect only runs when paused (for scrubbing, editing, etc.)
  // RAF-debounced: rapid scrubbing batches into max 1 sync per animation frame
  useEffect(() => {
    // Cancel any pending RAF sync from previous render
    if (pendingRafRef.current !== null) {
      cancelAnimationFrame(pendingRafRef.current);
      pendingRafRef.current = null;
    }

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
        const hit = engine.renderCachedFrame(playheadPosition);
        if (hit) {
          return; // Cache hit - instant render, no video seek needed
        }
        // Cache miss within range - will fall through to regular render
      }
    } else {
      // No RAM preview range, but still try the cache in case frames were cached during playback
      const hit = engine.renderCachedFrame(playheadPosition);
      if (hit) {
        return;
      }
    }

    // Debounce heavy layer sync via requestAnimationFrame.
    // During rapid scrubbing, multiple playheadPosition changes within a single frame
    // are batched — only the last scheduled sync actually executes.
    pendingRafRef.current = requestAnimationFrame(() => {
    pendingRafRef.current = null;

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

    const currentLayers = useTimelineStore.getState().layers;
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
              type: 'image', // Nested comps are pre-rendered to texture
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
        let useProxy = false;

        if (mediaStore.proxyEnabled && mediaFile?.proxyFps && mediaFile.proxyStatus === 'ready') {
          const player = getProxyPlayer(mediaFile.id);
          if (player && player.ready) {
            useProxy = true;
            const frame = player.seekToTime(clipTime);

            if (!video.muted) video.muted = true;
            if (!video.paused) video.pause();

            if (frame) {
              const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
              const interpolatedEffectsForProxy = getInterpolatedEffects(clip.id, keyframeLocalTime);
              newLayers[layerIndex] = {
                id: `timeline_layer_${layerIndex}`,
                name: clip.name,
                visible: isVideoTrackVisible(track),
                opacity: transform.opacity,
                blendMode: transform.blendMode,
                source: {
                  type: 'video',
                  videoFrame: frame,
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
        }

        if (!useProxy) {
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
        // Text/Solid clip handling (both use canvas)
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
              type: clip.source!.type === 'solid' ? 'solid' : 'text',
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
      useTimelineStore.setState({ layers: newLayers });
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

    }); // end requestAnimationFrame
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
