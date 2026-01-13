// LayerBuilder - Calculates render layers on-demand without React state overhead
// Called directly from the render loop for maximum performance

import type { TimelineClip, TimelineTrack, Layer, Effect, NestedCompositionData } from '../types';
import { useTimelineStore } from '../stores/timeline';
import { useMediaStore } from '../stores/mediaStore';
import { proxyFrameCache } from './proxyFrameCache';
import { audioManager, audioStatusTracker } from './audioManager';

// High-frequency playhead position - updated every frame by playback loop
// This avoids store updates which trigger subscriber cascades
export const playheadState = {
  position: 0,
  isUsingInternalPosition: false, // true during playback, false when paused
  playbackJustStarted: false, // true for first few frames after playback starts
};

class LayerBuilderService {
  private lastSeekRef: { [clipId: string]: number } = {};
  private proxyFramesRef: Map<string, { frameIndex: number; image: HTMLImageElement }> = new Map();

  // Native decoder throttling - avoid overwhelming the decoder with seeks
  private nativeDecoderLastSeekTime: Map<string, number> = new Map();
  private nativeDecoderLastSeekFrame: Map<string, number> = new Map();
  private nativeDecoderPendingSeek: Map<string, boolean> = new Map();
  private readonly NATIVE_SEEK_THROTTLE_MS = 16; // ~60fps max seek rate

  // Audio sync throttling - don't sync every frame to avoid glitches
  private lastAudioSyncTime = 0;
  private readonly AUDIO_SYNC_INTERVAL = 50; // Check audio sync every 50ms (balance between drift and glitches)

  // Track playback start for initial sync
  private playbackStartFrames = 0;

  // Debug frame counter for throttled logging
  private debugFrameCount = 0;

  // Lookahead preloading throttle
  private lastLookaheadTime = 0;
  private readonly LOOKAHEAD_INTERVAL = 100; // Check for upcoming nested comps every 100ms
  private readonly LOOKAHEAD_SECONDS = 3.0; // Look 3 seconds ahead for more preload time

  /**
   * Build layers for the current frame - called directly from render loop
   * Gets all data from stores directly, no React overhead
   */
  buildLayersFromStore(): Layer[] {
    const timelineState = useTimelineStore.getState();
    const {
      clips,
      tracks,
      isPlaying,
      isDraggingPlayhead,
      getInterpolatedTransform,
      getInterpolatedEffects,
      getInterpolatedSpeed,
      getSourceTimeForClip,
    } = timelineState;

    // Get active composition ID for unique layer IDs across compositions
    const activeCompId = useMediaStore.getState().activeCompositionId || 'default';

    // Use high-frequency playhead position during playback to avoid store read latency
    const playheadPosition = playheadState.isUsingInternalPosition
      ? playheadState.position
      : timelineState.playheadPosition;

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

          const nestedCompLayer: Layer = {
            id: `${activeCompId}_layer_${layerIndex}`,
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

          // Add mask properties if nested comp clip has masks
          if (clip.masks && clip.masks.length > 0) {
            nestedCompLayer.maskClipId = clip.id;
            nestedCompLayer.maskInvert = clip.masks.some(m => m.inverted);
          }

          layers[layerIndex] = nestedCompLayer;
        }
      } else if (clip?.source?.nativeDecoder) {
        // Handle Native Helper decoded clip (ProRes/DNxHD turbo mode)
        const nativeDecoder = clip.source.nativeDecoder;
        const clipLocalTime = playheadPosition - clip.startTime;
        const keyframeLocalTime = clipLocalTime;

        // Note: source time/clipTime calculation happens in syncVideoElements for native decoder

        const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
        const nativeInterpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

        const nativeLayer: Layer = {
          id: `${activeCompId}_layer_${layerIndex}`,
          name: clip.name,
          visible: true,
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

        // Add mask properties if native clip has masks
        if (clip.masks && clip.masks.length > 0) {
          nativeLayer.maskClipId = clip.id;
          nativeLayer.maskInvert = clip.masks.some(m => m.inverted);
        }

        layers[layerIndex] = nativeLayer;
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
          getSourceTimeForClip,
          activeCompId
        );
        if (layer) {
          layers[layerIndex] = layer;
        }
      } else if (clip?.source?.imageElement) {
        // Handle image clip
        const imageClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, imageClipLocalTime);
        const imageInterpolatedEffects = getInterpolatedEffects(clip.id, imageClipLocalTime);

        const imageLayer: Layer = {
          id: `${activeCompId}_layer_${layerIndex}`,
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

        // Add mask properties if image clip has masks
        if (clip.masks && clip.masks.length > 0) {
          imageLayer.maskClipId = clip.id;
          imageLayer.maskInvert = clip.masks.some(m => m.inverted);
        }

        layers[layerIndex] = imageLayer;
      } else if (clip?.source?.textCanvas) {
        // Handle text clip
        const textClipLocalTime = playheadPosition - clip.startTime;
        const transform = getInterpolatedTransform(clip.id, textClipLocalTime);
        const textInterpolatedEffects = getInterpolatedEffects(clip.id, textClipLocalTime);

        const textLayer: Layer = {
          id: `${activeCompId}_layer_${layerIndex}`,
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

        // Add mask properties if text clip has masks
        if (clip.masks && clip.masks.length > 0) {
          textLayer.maskClipId = clip.id;
          textLayer.maskInvert = clip.masks.some(m => m.inverted);
        }

        layers[layerIndex] = textLayer;
      }
    });

    // Preload proxy frames for upcoming nested compositions during playback
    if (isPlaying) {
      this.preloadUpcomingNestedCompFrames(clips, playheadPosition);
    }

    return layers;
  }

  /**
   * Preload proxy frames for nested compositions that will be active soon
   * Called during playback to ensure smooth entry into nested comps
   */
  private preloadUpcomingNestedCompFrames(clips: TimelineClip[], playheadPosition: number): void {
    const now = performance.now();

    // Throttle to avoid checking every frame
    if (now - this.lastLookaheadTime < this.LOOKAHEAD_INTERVAL) {
      return;
    }
    this.lastLookaheadTime = now;

    const mediaStore = useMediaStore.getState();
    const lookaheadEnd = playheadPosition + this.LOOKAHEAD_SECONDS;

    // Pre-seek WebCodecs decoders for non-proxy mode
    // This primes the decoder so the first frame is ready when needed
    if (!mediaStore.proxyEnabled) {
      this.primeUpcomingNestedCompVideos(clips, playheadPosition, lookaheadEnd);
      return;
    }

    // Find nested composition clips that will be active within lookahead window
    const upcomingNestedComps = clips.filter(clip =>
      clip.isComposition &&
      clip.nestedClips &&
      clip.nestedClips.length > 0 &&
      clip.startTime > playheadPosition && // Not yet active
      clip.startTime < lookaheadEnd // But will be soon
    );

    for (const nestedCompClip of upcomingNestedComps) {
      if (!nestedCompClip.nestedClips) continue;

      // Calculate the time within the nested comp when it starts
      const nestedStartTime = nestedCompClip.inPoint || 0;

      // Preload frames for each video clip in the nested composition
      for (const nestedClip of nestedCompClip.nestedClips) {
        if (!nestedClip.source?.videoElement) continue;

        // Check if this nested clip is active at the nested comp's start time
        if (nestedStartTime < nestedClip.startTime ||
            nestedStartTime >= nestedClip.startTime + nestedClip.duration) {
          continue;
        }

        // Find media file for this nested clip
        const nestedMediaFile = mediaStore.files.find(f =>
          f.id === nestedClip.source?.mediaFileId ||
          f.name === nestedClip.file?.name ||
          f.name === nestedClip.name
        );

        if (!nestedMediaFile?.proxyFps) continue;
        if (nestedMediaFile.proxyStatus !== 'ready' &&
            nestedMediaFile.proxyStatus !== 'generating') continue;

        // Calculate which frame to preload
        const nestedLocalTime = nestedStartTime - nestedClip.startTime;
        const nestedClipTime = nestedClip.reversed
          ? nestedClip.outPoint - nestedLocalTime
          : nestedLocalTime + nestedClip.inPoint;

        const proxyFps = nestedMediaFile.proxyFps;
        const frameIndex = Math.floor(nestedClipTime * proxyFps);

        // Trigger preloading by calling getCachedFrame (it preloads even if miss)
        // Preload 60 frames (2 seconds at 30fps) for smooth playback start
        const framesToPreload = Math.min(60, Math.ceil(proxyFps * 2));
        for (let i = 0; i < framesToPreload; i++) {
          proxyFrameCache.getCachedFrame(nestedMediaFile.id, frameIndex + i, proxyFps);
        }
      }
    }
  }

  // Track which nested comp videos have been primed to avoid repeated seeks
  private primedNestedVideos: Set<string> = new Set();

  /**
   * Pre-seek WebCodecs decoders for upcoming nested compositions (non-proxy mode)
   * This ensures the first frame is decoded and ready when the playhead enters
   */
  private primeUpcomingNestedCompVideos(
    clips: TimelineClip[],
    playheadPosition: number,
    lookaheadEnd: number
  ): void {
    // Find nested composition clips that will be active within lookahead window
    const upcomingNestedComps = clips.filter(clip =>
      clip.isComposition &&
      clip.nestedClips &&
      clip.nestedClips.length > 0 &&
      clip.startTime > playheadPosition && // Not yet active
      clip.startTime < lookaheadEnd // But will be soon
    );

    for (const nestedCompClip of upcomingNestedComps) {
      if (!nestedCompClip.nestedClips) continue;

      // Calculate the time within the nested comp when it starts
      const nestedStartTime = nestedCompClip.inPoint || 0;

      // Pre-seek each video clip in the nested composition
      for (const nestedClip of nestedCompClip.nestedClips) {
        if (!nestedClip.source?.videoElement) continue;

        // Check if this nested clip is active at the nested comp's start time
        if (nestedStartTime < nestedClip.startTime ||
            nestedStartTime >= nestedClip.startTime + nestedClip.duration) {
          continue;
        }

        // Create a unique key for this clip to track if already primed
        const primeKey = `${nestedCompClip.id}_${nestedClip.id}`;
        if (this.primedNestedVideos.has(primeKey)) {
          continue; // Already primed
        }

        // Calculate the target seek time
        const nestedLocalTime = nestedStartTime - nestedClip.startTime;
        const nestedClipTime = nestedClip.reversed
          ? nestedClip.outPoint - nestedLocalTime
          : nestedLocalTime + nestedClip.inPoint;

        // Prime using WebCodecs async seek if available (fire and forget)
        if (nestedClip.source.webCodecsPlayer) {
          console.log(`[LayerBuilder] Priming nested video "${nestedClip.name}" at ${nestedClipTime.toFixed(2)}s`);
          nestedClip.source.webCodecsPlayer.seekAsync(nestedClipTime).then(() => {
            this.primedNestedVideos.add(primeKey);
          });
        } else {
          // Fallback: just seek the video element
          const video = nestedClip.source.videoElement;
          video.currentTime = nestedClipTime;
          this.primedNestedVideos.add(primeKey);
        }
      }
    }

    // Clean up old primed entries when playhead moves past them
    // Reset when playhead moves backwards significantly (e.g., loop or scrub)
    if (this.primedNestedVideos.size > 100) {
      this.primedNestedVideos.clear();
    }
  }

  /**
   * Sync video elements to current playhead - call this from render loop
   * Handles video.currentTime updates and play/pause state
   */
  syncVideoElements(): void {
    const timelineState = useTimelineStore.getState();
    const { clips, isPlaying, isDraggingPlayhead, getInterpolatedSpeed, getSourceTimeForClip } = timelineState;
    const playheadPosition = playheadState.isUsingInternalPosition
      ? playheadState.position
      : timelineState.playheadPosition;

    // Check if proxy mode is enabled
    const mediaStore = useMediaStore.getState();
    const proxyEnabled = mediaStore.proxyEnabled;

    const clipsAtTime = clips.filter(
      c => playheadPosition >= c.startTime && playheadPosition < c.startTime + c.duration
    );

    clipsAtTime.forEach(clip => {
      // Handle Native Helper decoder (ProRes/DNxHD turbo mode)
      if (clip.source?.nativeDecoder) {
        const nativeDecoder = clip.source.nativeDecoder;
        const clipLocalTime = playheadPosition - clip.startTime;
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

        // Calculate target frame number
        const fps = nativeDecoder.fps || 25;
        const targetFrame = Math.round(clipTime * fps);

        // Throttle seeks to avoid overwhelming the decoder
        const now = performance.now();
        const lastSeekTime = this.nativeDecoderLastSeekTime.get(clip.id) || 0;
        const lastSeekFrame = this.nativeDecoderLastSeekFrame.get(clip.id) ?? -1;
        const isPending = this.nativeDecoderPendingSeek.get(clip.id) || false;

        // Skip if same frame or throttled (unless it's been too long)
        const timeSinceLastSeek = now - lastSeekTime;
        const shouldSeek = !isPending &&
          (targetFrame !== lastSeekFrame || timeSinceLastSeek > 100);

        if (shouldSeek && timeSinceLastSeek >= this.NATIVE_SEEK_THROTTLE_MS) {
          this.nativeDecoderLastSeekTime.set(clip.id, now);
          this.nativeDecoderLastSeekFrame.set(clip.id, targetFrame);
          this.nativeDecoderPendingSeek.set(clip.id, true);

          // Use fast scrub (scaled down) during playhead drag for smoother scrubbing
          nativeDecoder.seekToFrame(targetFrame, isDraggingPlayhead)
            .then(() => {
              this.nativeDecoderPendingSeek.set(clip.id, false);
            })
            .catch((err) => {
              this.nativeDecoderPendingSeek.set(clip.id, false);
              console.warn('[NativeDecoder] Seek failed:', err);
            });
        }
        return; // Skip other decoders for this clip
      }

      if (clip.source?.videoElement) {
        const video = clip.source.videoElement;
        const clipLocalTime = playheadPosition - clip.startTime;
        const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
        const initialSpeed = getInterpolatedSpeed(clip.id, 0);
        const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
        const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
        const timeDiff = Math.abs(video.currentTime - clipTime);

        // Check if this clip should use proxy mode
        const mediaFile = mediaStore.files.find(
          f => f.name === clip.name || clip.source?.mediaFileId === f.id
        );
        const useProxy = proxyEnabled && mediaFile?.proxyFps &&
          (mediaFile.proxyStatus === 'ready' || mediaFile.proxyStatus === 'generating');

        // In proxy mode: pause video and let audio proxy handle audio
        // We use cached WebP frames instead of video frames
        if (useProxy) {
          if (!video.paused) {
            video.pause();
          }
          // Mute video to prevent any audio leakage
          if (!video.muted) {
            video.muted = true;
          }
          return; // Skip normal video sync
        }

        // Non-proxy mode: normal video playback
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

  // Track audio proxies currently playing to manage their lifecycle
  private activeAudioProxies: Map<string, HTMLAudioElement> = new Map();

  // Audio scrubbing state
  private lastScrubPosition = -1;
  private scrubAudioTimeout: ReturnType<typeof setTimeout> | null = null;
  private readonly SCRUB_AUDIO_DURATION = 80; // ms of audio to play when scrubbing

  /**
   * Play a short audio snippet for scrubbing feedback
   */
  private playScrubAudio(audio: HTMLAudioElement, time: number): void {
    // Seek to position
    audio.currentTime = time;
    audio.volume = 0.7; // Slightly lower volume for scrubbing

    // Play short snippet
    audio.play().catch(() => {});

    // Stop after short duration
    if (this.scrubAudioTimeout) {
      clearTimeout(this.scrubAudioTimeout);
    }
    this.scrubAudioTimeout = setTimeout(() => {
      audio.pause();
      this.scrubAudioTimeout = null;
    }, this.SCRUB_AUDIO_DURATION);
  }

  /**
   * Sync audio elements to current playhead - call this from render loop
   * Handles audio play/pause/seek and drift tracking
   * THROTTLED to avoid audio glitches from constant seeking
   */
  syncAudioElements(): void {
    const now = performance.now();

    // Handle playback start - force immediate sync for first 10 frames
    if (playheadState.playbackJustStarted) {
      this.playbackStartFrames++;
      if (this.playbackStartFrames > 10) {
        playheadState.playbackJustStarted = false;
        this.playbackStartFrames = 0;
      }
      // Don't throttle during startup - we need to sync immediately
    } else {
      // Throttle audio sync to avoid glitches - only run every AUDIO_SYNC_INTERVAL ms
      if (now - this.lastAudioSyncTime < this.AUDIO_SYNC_INTERVAL) {
        return;
      }
    }
    this.lastAudioSyncTime = now;

    const timelineState = useTimelineStore.getState();
    const {
      clips,
      tracks,
      isPlaying,
      isDraggingPlayhead,
      getInterpolatedSpeed,
      getSourceTimeForClip,
    } = timelineState;

    // Use high-frequency playhead position during playback
    const playheadPosition = playheadState.isUsingInternalPosition
      ? playheadState.position
      : timelineState.playheadPosition;

    // Force hard sync on playback start
    const forceHardSync = playheadState.playbackJustStarted;

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

        // Audio scrubbing for audio track clips
        if (isDraggingPlayhead && !effectivelyMuted) {
          const positionChanged = Math.abs(playheadPosition - this.lastScrubPosition) > 0.02;
          if (positionChanged) {
            this.lastScrubPosition = playheadPosition;
            audio.playbackRate = 1;
            this.playScrubAudio(audio, clipTime);
          }
        } else if (shouldPlay) {
          // Gradual drift correction using playback rate adjustment
          const absDrift = Math.abs(timeDiff);

          let newRate = targetRate;

          if (forceHardSync || absDrift > 0.25) {
            // Force sync on playback start OR large drift (>250ms): hard seek
            audio.currentTime = clipTime;
            newRate = targetRate;
          } else if (absDrift > 0.05) {
            // Drift 50-250ms: gentle playback rate adjustment (max 3%)
            const correctionFactor = Math.min(0.03, absDrift * 0.1);
            const correction = timeDiff > 0 ? -correctionFactor : correctionFactor;
            newRate = targetRate * (1 + correction);
          }

          // Only change rate if difference is significant (avoid constant micro-adjustments)
          if (Math.abs(audio.playbackRate - newRate) > 0.005) {
            audio.playbackRate = Math.max(0.25, Math.min(4, newRate));
          }

          // Reset volume after scrubbing
          if (audio.volume !== 1) {
            audio.volume = 1;
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

    // Sync audio proxies for video clips in proxy mode
    // This provides faster audio sync than video element buffering
    const mediaStore = useMediaStore.getState();
    const activeVideoClipIds = new Set<string>();

    videoTracks.forEach(track => {
      const clip = clipsAtTime.find(c => c.trackId === track.id);

      if (clip?.source?.videoElement && !clip.isComposition) {
        // Check if proxy mode is enabled for this clip
        const mediaFile = mediaStore.files.find(
          f => f.name === clip.name || clip.source?.mediaFileId === f.id
        );

        const shouldUseAudioProxy = mediaStore.proxyEnabled &&
          mediaFile?.hasProxyAudio &&
          (mediaFile.proxyStatus === 'ready' || mediaFile.proxyStatus === 'generating');

        if (shouldUseAudioProxy && mediaFile) {
          activeVideoClipIds.add(clip.id);

          // Mute video element audio when using audio proxy
          const video = clip.source.videoElement;
          if (!video.muted) {
            video.muted = true;
          }

          // Get or load audio proxy
          const audioProxy = proxyFrameCache.getCachedAudioProxy(mediaFile.id);

          if (audioProxy) {
            // Track active proxy
            this.activeAudioProxies.set(clip.id, audioProxy);

            const clipLocalTime = playheadPosition - clip.startTime;
            const currentSpeed = getInterpolatedSpeed(clip.id, clipLocalTime);
            const absSpeed = Math.abs(currentSpeed);
            const sourceTime = getSourceTimeForClip(clip.id, clipLocalTime);
            const initialSpeed = getInterpolatedSpeed(clip.id, 0);
            const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
            const clipTime = Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));

            const trackObj = videoTracks.find(t => t.id === clip.trackId);
            const effectivelyMuted = trackObj ? !isVideoTrackVisible(trackObj) : false;
            audioProxy.muted = effectivelyMuted;

            const targetRate = absSpeed > 0.1 ? absSpeed : 1;

            // Set preservesPitch
            const shouldPreservePitch = clip.preservesPitch !== false;
            if ((audioProxy as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch !== shouldPreservePitch) {
              (audioProxy as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch = shouldPreservePitch;
            }

            const timeDiff = audioProxy.currentTime - clipTime;
            if (Math.abs(timeDiff) > maxAudioDrift) {
              maxAudioDrift = Math.abs(timeDiff);
            }

            const shouldPlay = isPlaying && !effectivelyMuted && !isDraggingPlayhead && absSpeed > 0.1;

            // Audio scrubbing - use instant Web Audio API scrubbing
            if (isDraggingPlayhead && !effectivelyMuted) {
              // Trigger every ~5ms of playhead movement for continuous sound
              const positionChanged = Math.abs(playheadPosition - this.lastScrubPosition) > 0.005;
              if (positionChanged) {
                this.lastScrubPosition = playheadPosition;
                // Use instant AudioBuffer scrubbing - overlapping snippets
                proxyFrameCache.playScrubAudio(mediaFile.id, clipTime, 0.12);
              }
            } else if (shouldPlay) {
              const absDrift = Math.abs(timeDiff);
              let newRate = targetRate;

              if (forceHardSync || absDrift > 0.25) {
                // Force sync on playback start OR large drift (>250ms): hard seek
                audioProxy.currentTime = clipTime;
                newRate = targetRate;
              } else if (absDrift > 0.05) {
                // Drift 50-250ms: gentle playback rate adjustment (max 3%)
                const correctionFactor = Math.min(0.03, absDrift * 0.1);
                const correction = timeDiff > 0 ? -correctionFactor : correctionFactor;
                newRate = targetRate * (1 + correction);
              }

              if (Math.abs(audioProxy.playbackRate - newRate) > 0.005) {
                audioProxy.playbackRate = Math.max(0.25, Math.min(4, newRate));
              }

              // Reset volume after scrubbing
              if (audioProxy.volume !== 1) {
                audioProxy.volume = 1;
              }

              if (audioProxy.paused) {
                audioProxy.currentTime = clipTime;
                audioProxy.play().catch(err => {
                  console.warn('[Audio Proxy] Failed to play:', err.message);
                  hasAudioError = true;
                });
              }

              if (!audioProxy.paused && !effectivelyMuted) {
                audioPlayingCount++;
              }
            } else {
              if (!audioProxy.paused) {
                audioProxy.pause();
              }
              // Reset scrub position when not dragging
              if (!isDraggingPlayhead) {
                this.lastScrubPosition = -1;
              }
            }
          } else {
            // Audio proxy not loaded yet - trigger preload
            proxyFrameCache.preloadAudioProxy(mediaFile.id);
            // Also preload AudioBuffer for instant scrubbing
            proxyFrameCache.getAudioBuffer(mediaFile.id);
          }
        }
      }
    });

    // Pause audio proxies for clips no longer at playhead
    for (const [clipId, audioProxy] of this.activeAudioProxies) {
      if (!activeVideoClipIds.has(clipId) && !audioProxy.paused) {
        audioProxy.pause();
        this.activeAudioProxies.delete(clipId);
      }
    }

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
          // Gradual drift correction using playback rate adjustment
          const absDrift = Math.abs(timeDiff);

          let newRate = targetRate;

          if (forceHardSync || absDrift > 0.25) {
            // Force sync on playback start OR large drift (>250ms): hard seek
            audio.currentTime = clipTime;
            newRate = targetRate;
          } else if (absDrift > 0.05) {
            // Drift 50-250ms: gentle playback rate adjustment (max 3%)
            const correctionFactor = Math.min(0.03, absDrift * 0.1);
            const correction = timeDiff > 0 ? -correctionFactor : correctionFactor;
            newRate = targetRate * (1 + correction);
          }

          // Only change rate if difference is significant
          if (Math.abs(audio.playbackRate - newRate) > 0.005) {
            audio.playbackRate = Math.max(0.25, Math.min(4, newRate));
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
    _isPlaying: boolean,
    _isDraggingPlayhead: boolean,
    getInterpolatedTransform: (clipId: string, localTime: number) => ReturnType<typeof useTimelineStore.getState>['getInterpolatedTransform'] extends (clipId: string, localTime: number) => infer R ? R : never,
    getInterpolatedEffects: (clipId: string, localTime: number) => Effect[],
    getInterpolatedSpeed: (clipId: string, localTime: number) => number,
    getSourceTimeForClip: (clipId: string, localTime: number) => number,
    activeCompId: string
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

        const proxyLayer: Layer = {
          id: `${activeCompId}_layer_${layerIndex}`,
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

        // Add mask properties if clip has masks
        if (clip.masks && clip.masks.length > 0) {
          proxyLayer.maskClipId = clip.id;
          proxyLayer.maskInvert = clip.masks.some(m => m.inverted);
        }

        return proxyLayer;
      }

      // Use cached proxy frame if available while loading new one
      const cached = this.proxyFramesRef.get(cacheKey);
      if (cached?.image) {
        const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
        const interpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

        const cachedProxyLayer: Layer = {
          id: `${activeCompId}_layer_${layerIndex}`,
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

        // Add mask properties if clip has masks
        if (clip.masks && clip.masks.length > 0) {
          cachedProxyLayer.maskClipId = clip.id;
          cachedProxyLayer.maskInvert = clip.masks.some(m => m.inverted);
        }

        return cachedProxyLayer;
      }
    }

    // Direct video playback
    const transform = getInterpolatedTransform(clip.id, keyframeLocalTime);
    const videoInterpolatedEffects = getInterpolatedEffects(clip.id, keyframeLocalTime);

    const layer: Layer = {
      id: `${activeCompId}_layer_${layerIndex}`,
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

    // Add mask properties if clip has masks
    if (clip.masks && clip.masks.length > 0) {
      // Store clip ID on layer for mask texture lookup (consistent across systems)
      layer.maskClipId = clip.id;

      // Add mask properties to layer (invert is handled in shader)
      layer.maskInvert = clip.masks.some(m => m.inverted);
    }

    return layer;
  }

  /**
   * Build nested composition layers
   */
  private buildNestedLayers(clip: TimelineClip, clipTime: number, isPlaying: boolean): Layer[] {
    if (!clip.nestedClips || !clip.nestedTracks) return [];

    const nestedVideoTracks = clip.nestedTracks.filter(t => t.type === 'video' && t.visible);
    const layers: Layer[] = [];

    // Debug: Log nested composition state
    if (this.debugFrameCount++ % 60 === 0) {
      console.log(`[NestedComp] clipTime=${clipTime.toFixed(2)}, tracks=${nestedVideoTracks.length}, clips=${clip.nestedClips.length}`);
      clip.nestedClips.forEach(nc => {
        const hasVideo = !!nc.source?.videoElement;
        const readyState = nc.source?.videoElement?.readyState ?? -1;
        console.log(`  - ${nc.name}: hasVideo=${hasVideo}, readyState=${readyState}, startTime=${nc.startTime}, duration=${nc.duration}`);
      });
    }

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

      // Check if we should use proxy mode BEFORE touching video elements
      const mediaStore = useMediaStore.getState();
      const nestedMediaFile = mediaStore.files.find(f =>
        f.id === nestedClip.source?.mediaFileId ||
        f.name === nestedClip.file?.name ||
        f.name === nestedClip.name
      );
      const shouldUseProxy = mediaStore.proxyEnabled &&
        nestedMediaFile?.proxyFps &&
        (nestedMediaFile.proxyStatus === 'ready' || nestedMediaFile.proxyStatus === 'generating');

      // Only sync video elements if NOT in proxy mode
      // In proxy mode, we don't need the video at all - touching it causes conflicts
      if (nestedClip.source?.videoElement && !shouldUseProxy) {
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
      } else if (nestedClip.source?.videoElement && shouldUseProxy) {
        // In proxy mode: ensure video is paused to avoid resource conflicts
        const video = nestedClip.source.videoElement;
        if (!video.paused) {
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

      // Build base layer properties
      const baseLayer = {
        id: `nested-layer-${nestedClip.id}`,
        name: nestedClip.name,
        visible: true,
        opacity: transform.opacity ?? 1,
        blendMode: transform.blendMode || 'normal',
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
        // Add mask properties if nested clip has masks
        ...(nestedClip.masks && nestedClip.masks.length > 0 ? {
          maskClipId: nestedClip.id,
          maskInvert: nestedClip.masks.some(m => m.inverted),
        } : {}),
      };

      if (nestedClip.source?.videoElement) {
        // Reuse mediaStore and nestedMediaFile from earlier proxy check
        const proxyFps = nestedMediaFile?.proxyFps || 30;
        const frameIndex = Math.floor(nestedClipTime * proxyFps);

        // Determine if we can use proxy for this specific frame
        let useProxy = shouldUseProxy;
        if (useProxy && nestedMediaFile?.proxyStatus === 'generating') {
          // For generating proxies, check if this frame is ready
          const totalFrames = Math.ceil((nestedMediaFile.duration || 10) * proxyFps);
          const maxGeneratedFrame = Math.floor(totalFrames * ((nestedMediaFile.proxyProgress || 0) / 100));
          useProxy = frameIndex < maxGeneratedFrame;
        }

        if (useProxy && nestedMediaFile) {
          // Try to get proxy frame
          const cacheKey = `nested_${nestedMediaFile.id}_${nestedClip.id}`;
          const cachedInService = proxyFrameCache.getCachedFrame(nestedMediaFile.id, frameIndex, proxyFps);

          if (cachedInService) {
            this.proxyFramesRef.set(cacheKey, { frameIndex, image: cachedInService });
            const proxyLayer: Layer = {
              ...baseLayer,
              source: {
                type: 'image',
                imageElement: cachedInService,
              },
            } as Layer;
            layers.push(proxyLayer);
            continue; // Skip direct video playback
          }

          // During playback: skip to video fallback if exact frame not cached
          // During scrubbing: use nearest cached frame for smooth scrubbing
          if (!isPlaying) {
            // Try nearest cached frame for scrubbing fallback
            const nearestFrame = proxyFrameCache.getNearestCachedFrame(nestedMediaFile.id, frameIndex);
            if (nearestFrame) {
              this.proxyFramesRef.set(cacheKey, { frameIndex, image: nearestFrame });
              const proxyLayer: Layer = {
                ...baseLayer,
                source: {
                  type: 'image',
                  imageElement: nearestFrame,
                },
              } as Layer;
              layers.push(proxyLayer);
              continue; // Skip direct video playback
            }

            // Use cached proxy frame if available while loading new one
            const cached = this.proxyFramesRef.get(cacheKey);
            if (cached?.image) {
              const proxyLayer: Layer = {
                ...baseLayer,
                source: {
                  type: 'image',
                  imageElement: cached.image,
                },
              } as Layer;
              layers.push(proxyLayer);
              continue; // Skip direct video playback
            }
          }

          // Frame not in cache - use last rendered frame to prevent flicker
          // This shows a brief "stale frame" instead of flashing/disappearing
          const cached = this.proxyFramesRef.get(cacheKey);
          if (cached?.image) {
            const proxyLayer: Layer = {
              ...baseLayer,
              source: {
                type: 'image',
                imageElement: cached.image,
              },
            } as Layer;
            layers.push(proxyLayer);
            continue;
          }

          // No cached frame at all - skip during playback to avoid video conflicts
          if (isPlaying) {
            continue;
          }

          // During scrubbing: check if video is ready for fallback
          const video = nestedClip.source.videoElement;
          if (video.seeking || video.readyState < 2) {
            continue;
          }
        }

        // Fall back to direct video playback (only when NOT in proxy mode)
        if (!shouldUseProxy) {
          layers.push({
            ...baseLayer,
            source: {
              type: 'video',
              videoElement: nestedClip.source.videoElement,
              webCodecsPlayer: nestedClip.source.webCodecsPlayer,
            },
          } as Layer);
        }
      } else if (nestedClip.source?.imageElement) {
        layers.push({
          ...baseLayer,
          source: {
            type: 'image',
            imageElement: nestedClip.source.imageElement,
          },
        } as Layer);
      }
    }

    return layers;
  }
}

// Singleton instance
export const layerBuilder = new LayerBuilderService();
