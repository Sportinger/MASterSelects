// LayerBuilder - Calculates render layers on-demand without React state overhead
// Called directly from the render loop for maximum performance

import type { TimelineClip, TimelineTrack, Layer, Effect, NestedCompositionData } from '../types';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { proxyFrameCache } from './proxyFrameCache';
import { audioManager, audioStatusTracker } from './audioManager';

class LayerBuilderService {
  private lastSeekRef: { [clipId: string]: number } = {};
  private proxyFramesRef: Map<string, { frameIndex: number; image: HTMLImageElement }> = new Map();
  private proxyLoadingRef: Set<string> = new Set();

  /**
   * Build layers for the current frame - called directly from render loop
   * Gets all data from stores directly, no React overhead
   */
  buildLayersFromStore(): Layer[] {
    const timelineState = useTimelineStore.getState();
    const {
      playheadPosition,
      clips,
      tracks,
      isPlaying,
      isDraggingPlayhead,
      getInterpolatedTransform,
      getInterpolatedEffects,
      getInterpolatedSpeed,
      getSourceTimeForClip,
    } = timelineState;

    const videoTracks = tracks.filter(t => t.type === 'video' && t.visible !== false);
    const anyVideoSolo = videoTracks.some(t => t.solo);

    // Helper to check track visibility
    const isVideoTrackVisible = (track: TimelineTrack) => {
      if (!track.visible) return false;
      if (anyVideoSolo) return track.solo;
      return true;
    };

    // Get clips at current playhead
    const clipsAtTime = clips.filter(
      c => playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
    );

    const layers: Layer[] = [];

    videoTracks.forEach((track, layerIndex) => {
      const clip = clipsAtTime.find(c => c.trackId === track.id);
      const trackVisible = isVideoTrackVisible(track);

      if (!trackVisible) {
        return; // Skip invisible tracks
      }

      if (clip?.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        // Handle nested composition
        const clipTime = playheadPosition - clip.startTime + clip.inPoint;
        const nestedLayers = this.buildNestedLayers(clip, clipTime, isPlaying);
        const interpolatedTransform = getInterpolatedTransform(clip.id, clipTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, clipTime);

        // Get composition dimensions
        const mediaStore = useMediaStore.getState();
        const composition = mediaStore.compositions.find(c => c.id === clip.compositionId);
        const compWidth = composition?.width || 1920;
        const compHeight = composition?.height || 1080;

        if (nestedLayers.length > 0) {
          const nestedCompData: NestedCompositionData = {
            compositionId: clip.compositionId || clip.id,
            layers: nestedLayers,
            width: compWidth,
            height: compHeight,
          };

          layers[layerIndex] = {
            id: `timeline_layer_${layerIndex}`,
            name: clip.name,
            visible: true,
            opacity: interpolatedTransform.opacity,
            blendMode: interpolatedTransform.blendMode,
            source: {
              type: 'video',
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
        }
      } else if (clip?.source?.videoElement) {
        // Handle video clip
        const layer = this.buildVideoLayer(
          clip,
          layerIndex,
          playheadPosition,
          isPlaying,
          isDraggingPlayhead,
          getInterpolatedTransform,
          getInterpolatedEffects,
          getInterpolatedSpeed,
          getSourceTimeForClip
        );
        if (layer) {
          layers[layerIndex] = layer;
        }
      } else if (clip?.source?.imageElement) {
        // Handle image clip
        const imageClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, imageClipLocalTime);
        const imageInterpolatedEffects = getInterpolatedEffects(clip.id, imageClipLocalTime);

        layers[layerIndex] = {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'image',
            imageElement: clip.source.imageElement,
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
      } else if (clip?.source?.textCanvas) {
        // Handle text clip
        const textClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, textClipLocalTime);
        const textInterpolatedEffects = getInterpolatedEffects(clip.id, textClipLocalTime);

        layers[layerIndex] = {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'text',
            textCanvas: clip.source.textCanvas,
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
      }
    });

    return layers;
  }

  /**
   * Sync video elements to current playhead - call this from render loop
   * Handles video.currentTime updates and play/pause state
   */
  syncVideoElements(): void {
    const { playheadPosition, clips, tracks, isPlaying, isDraggingPlayhead, getInterpolatedSpeed, getSourceTimeForClip } = useTimelineStore.getState();

    const clipsAtTime = clips.filter(
      c => playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
    );

    clipsAtTime.forEach(clip => {
      if (clip.source?.videoElement) {
        const video = clip.source.videoElement;
        const clipLocalTime = playheadPosition - clip.startTime;
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
        const timeDiff = Math.abs(video.currentTime - clipTime);

        if (clip.reversed) {
          if (!video.paused) video.pause();
          const seekThreshold = isDraggingPlayhead ? 0.1 : 0.03;
          if (timeDiff > seekThreshold) {
            const now = performance.now();
            const lastSeek = this.lastSeekRef[clip.id] || 0;
            if (now - lastSeek > 33) {
              video.currentTime = clipTime;
              this.lastSeekRef[clip.id] = now;
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
              const lastSeek = this.lastSeekRef[clip.id] || 0;
              if (now - lastSeek > (isDraggingPlayhead ? 80 : 33)) {
                if (isDraggingPlayhead && 'fastSeek' in video) {
                  video.fastSeek(clipTime);
                } else {
                  video.currentTime = clipTime;
                }
                this.lastSeekRef[clip.id] = now;
              }
            }
          }
        }
      }
    });

    // Pause videos not at playhead
    clips.forEach(clip => {
      if (clip.source?.videoElement) {
        const isAtPlayhead = clipsAtTime.some(c => c.id === clip.id);
        if (!isAtPlayhead && !clip.source.videoElement.paused) {
          clip.source.videoElement.pause();
        }
      }
    });
  }

  /**
   * Sync audio elements to current playhead - call this from render loop
   * Handles audio play/pause/seek and drift tracking
   */
  syncAudioElements(): void {
    const timelineState = useTimelineStore.getState();
    const {
      playheadPosition,
      clips,
      tracks,
      isPlaying,
      isDraggingPlayhead,
      getInterpolatedSpeed,
      getSourceTimeForClip,
    } = timelineState;

    const audioTracks = tracks.filter(t => t.type === 'audio');
    const videoTracks = tracks.filter(t => t.type === 'video');
    const anyAudioSolo = audioTracks.some(t => t.solo);
    const anyVideoSolo = videoTracks.some(t => t.solo);

    // Helper to check track mute state
    const isAudioTrackMuted = (track: TimelineTrack) => {
      if (track.muted) return true;
      if (anyAudioSolo && !track.solo) return true;
      return false;
    };

    const isVideoTrackVisible = (track: TimelineTrack) => {
      if (!track.visible) return false;
      if (anyVideoSolo) return track.solo;
      return true;
    };

    const clipsAtTime = clips.filter(
      c => playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
    );

    let audioPlayingCount = 0;
    let maxAudioDrift = 0;
    let hasAudioError = false;

    // Resume audio context if needed (browser autoplay policy)
    if (isPlaying && !isDraggingPlayhead) {
      audioManager.resume().catch(() => {});
    }

    // Sync audio track clips
    audioTracks.forEach(track => {
      const clip = clipsAtTime.find(c => c.trackId === track.id);

      if (clip?.source?.audioElement) {
        const audio = clip.source.audioElement;
        const clipLocalTime = playheadPosition - clip.startTime;
        const currentSpeed = getInterpolatedSpeed(clip.id, clipLocalTime);
        const absSpeed = Math.abs(currentSpeed);
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
        const timeDiff = audio.currentTime - clipTime;

        if (Math.abs(timeDiff) > maxAudioDrift) {
          maxAudioDrift = Math.abs(timeDiff);
        }

        const effectivelyMuted = isAudioTrackMuted(track);
        audio.muted = effectivelyMuted;

        // Set playback rate
        const targetRate = absSpeed > 0.1 ? absSpeed : 1;
        if (Math.abs(audio.playbackRate - targetRate) > 0.01) {
          audio.playbackRate = Math.max(0.25, Math.min(4, targetRate));
        }

        // Set preservesPitch
        const shouldPreservePitch = clip.preservesPitch !== false;
        if ((audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch !== shouldPreservePitch) {
          (audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch = shouldPreservePitch;
        }

        const shouldPlay = isPlaying && !effectivelyMuted && !isDraggingPlayhead && absSpeed > 0.1;

        if (shouldPlay) {
          // Only sync audio on significant drift (>100ms) to reduce pops - was 200ms
          if (Math.abs(timeDiff) > 0.1) {
            audio.currentTime = clipTime;
          }

          if (audio.paused) {
            audio.currentTime = clipTime;
            audio.play().catch(err => {
              console.warn('[Audio] Failed to play:', err.message);
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

    // Pause audio clips not at playhead
    clips.forEach(clip => {
      if (clip.source?.audioElement) {
        const isAtPlayhead = clipsAtTime.some(c => c.id === clip.id);
        if (!isAtPlayhead && !clip.source.audioElement.paused) {
          clip.source.audioElement.pause();
        }
      }
      if (clip.mixdownAudio) {
        const isAtPlayhead = clipsAtTime.some(c => c.id === clip.id);
        if (!isAtPlayhead && !clip.mixdownAudio.paused) {
          clip.mixdownAudio.pause();
        }
      }
    });

    // Sync nested composition mixdown audio
    clipsAtTime.forEach(clip => {
      if (clip.isComposition && clip.mixdownAudio && clip.hasMixdownAudio) {
        const audio = clip.mixdownAudio;
        const clipLocalTime = playheadPosition - clip.startTime;
        const currentSpeed = getInterpolatedSpeed(clip.id, clipLocalTime);
        const absSpeed = Math.abs(currentSpeed);
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

        const track = videoTracks.find(t => t.id === clip.trackId);
        const effectivelyMuted = track ? !isVideoTrackVisible(track) : false;
        audio.muted = effectivelyMuted;

        const targetRate = absSpeed > 0.1 ? absSpeed : 1;
        if (Math.abs(audio.playbackRate - targetRate) > 0.01) {
          audio.playbackRate = Math.max(0.25, Math.min(4, targetRate));
        }

        const shouldPreservePitch = clip.preservesPitch !== false;
        if ((audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch !== shouldPreservePitch) {
          (audio as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch = shouldPreservePitch;
        }

        const timeDiff = audio.currentTime - clipTime;
        if (Math.abs(timeDiff) > maxAudioDrift) {
          maxAudioDrift = Math.abs(timeDiff);
        }

        const shouldPlay = isPlaying && !effectivelyMuted && !isDraggingPlayhead && absSpeed > 0.1;

        if (shouldPlay) {
          if (Math.abs(timeDiff) > 0.1) {
            audio.currentTime = clipTime;
          }

          if (audio.paused) {
            audio.currentTime = clipTime;
            audio.play().catch(err => {
              console.warn('[Nested Comp Audio] Failed to play:', err.message);
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
  }

  /**
   * Build video layer with video seeking and proxy handling
   */
  private buildVideoLayer(
    clip: TimelineClip,
    layerIndex: number,
    playheadPosition: number,
    isPlaying: boolean,
    isDraggingPlayhead: boolean,
    getInterpolatedTransform: (clipId: string, localTime: number) => ReturnType<typeof useTimelineStore.getState>['getInterpolatedTransform'] extends (clipId: string, localTime: number) => infer R ? R : never,
    getInterpolatedEffects: (clipId: string, localTime: number) => Effect[],
    getInterpolatedSpeed: (clipId: string, localTime: number) => number,
    getSourceTimeForClip: (clipId: string, localTime: number) => number
  ): Layer | null {
    const clipLocalTime = playheadPosition - clip.startTime;
    const keyframeLocalTime = clipLocalTime;
    const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
    const initialSpeed = getInterpolatedSpeed(clip.id, 0);
    const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
    const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
    const video = clip.source!.videoElement!;
    const webCodecsPlayer = clip.source?.webCodecsPlayer;

    // Check for proxy usage
    const mediaStore = useMediaStore.getState();
    const mediaFile = mediaStore.files.find(
      f => f.name === clip.name || clip.source?.mediaFileId === f.id
    );
    const proxyFps = mediaFile?.proxyFps || 30;
    const frameIndex = Math.floor(clipTime * proxyFps);
    let useProxy = false;

    if (mediaStore.proxyEnabled && mediaFile?.proxyFps) {
      if (mediaFile.proxyStatus === 'ready') {
        useProxy = true;
      } else if (mediaFile.proxyStatus === 'generating' && (mediaFile.proxyProgress || 0) > 0) {
        const totalFrames = Math.ceil((mediaFile.duration || 10) * proxyFps);
        const maxGeneratedFrame = Math.floor(totalFrames * ((mediaFile.proxyProgress || 0) / 100));
        useProxy = frameIndex < maxGeneratedFrame;
      }
    }

    if (useProxy && mediaFile) {
      // Proxy playback - use cached proxy frames
      const cacheKey = `${mediaFile.id}_${clip.id}`;
      const cachedInService = proxyFrameCache.getCachedFrame(mediaFile.id, frameIndex, proxyFps);

      if (cachedInService) {
        this.proxyFramesRef.set(cacheKey, { frameIndex, image: cachedInService });
        const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

        return {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'image',
            imageElement: cachedInService,
          },
          effects: interpolatedEffects,
          position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
          scale: { x: transform.scale.x, y: transform.scale.y },
          rotation: {
            x: (transform.rotation.x * Math.PI) / 180,
            y: (transform.rotation.y * Math.PI) / 180,
            z: (transform.rotation.z * Math.PI) / 180,
          },
        };
      }

      // Use cached proxy frame if available while loading new one
      const cached = this.proxyFramesRef.get(cacheKey);
      if (cached?.image) {
        const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

        return {
          id: `timeline_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
          opacity: transform.opacity,
          blendMode: transform.blendMode,
          source: {
            type: 'image',
            imageElement: cached.image,
          },
          effects: interpolatedEffects,
          position: { x: transform.position.x, y: transform.position.y, z: transform.position.z },
          scale: { x: transform.scale.x, y: transform.scale.y },
          rotation: {
            x: (transform.rotation.x * Math.PI) / 180,
            y: (transform.rotation.y * Math.PI) / 180,
            z: (transform.rotation.z * Math.PI) / 180,
          },
        };
      }
    }

    // Direct video playback
    const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
    const videoInterpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

    return {
      id: `timeline_layer_${layerIndex}`,
      name: clip.name,
      visible: true,
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
  }

  /**
   * Build nested composition layers
   */
  private buildNestedLayers(clip: TimelineClip, clipTime: number, isPlaying: boolean): Layer[] {
    if (!clip.nestedClips || !clip.nestedTracks) return [];

    const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
    const layers: Layer[] = [];

    for (let i = nestedVideoTracks.length - 1; i >= 0; i--) {
      const nestedTrack = nestedVideoTracks[i];
      const nestedClip = clip.nestedClips.find(
        nc =>
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
  }
}

// Singleton instance
export const layerBuilder = new LayerBuilderService();
