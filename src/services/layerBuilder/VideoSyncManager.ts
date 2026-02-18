// VideoSyncManager - Handles video element synchronization with playhead
// Extracted from LayerBuilderService for separation of concerns

import type { TimelineClip } from '../../types';
import type { FrameContext, NativeDecoderState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip } from './FrameContext';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { engine } from '../../engine/WebGPUEngine';
import { Logger } from '../logger';
import { playbackMetrics } from '../playbackMetrics';

const log = Logger.create('VideoSync');

export class VideoSyncManager {
  // Native decoder state
  private nativeDecoderState = new Map<string, NativeDecoderState>();

  // Video sync throttling
  private lastVideoSyncFrame = -1;
  private lastVideoSyncPlaying = false;
  private lastSeekRef: Record<string, number> = {};

  // Videos currently being warmed up (brief play to activate GPU surface)
  // After page reload, video GPU surfaces are empty — all sync rendering APIs
  // (importExternalTexture, canvas.drawImage, copyExternalImageToTexture) return black.
  // The ONLY way to populate the GPU surface is video.play().
  // We do this lazily on first scrub attempt, not during restore, because
  // the render loop's syncClipVideo would immediately pause the warmup video.
  private warmingUpVideos = new WeakSet<HTMLVideoElement>();
  // Cooldown for failed warmup attempts (avoids spamming play() every frame)
  private warmupRetryCooldown = new WeakMap<HTMLVideoElement, number>();

  // Track which videos are being force-decoded to avoid duplicate calls
  private forceDecodeInProgress = new Set<string>();

  // Hybrid seek state
  private rvfcHandles: Record<string, number> = {};
  private preciseSeekTimers: Record<string, ReturnType<typeof setTimeout>> = {};
  private latestSeekTargets: Record<string, number> = {};

  // Sorted clips cache for efficient preload (early-break optimization)
  private sortedClipsByStart: TimelineClip[] = [];
  private lastClipsRef: TimelineClip[] = [];

  // Continuous playback: track which clip was active per track to detect transitions
  // and hand off video elements for contiguous same-source clips (like DaVinci's approach)
  private trackPlaybackState = new Map<string, {
    clipId: string;
    sourceMediaFileId: string;
    outPoint: number;
    endTime: number;
  }>();
  private wcPlayerModeByClipId = new Map<string, 'realtime' | 'seek'>();
  private wcLastSeekWarnByClipId = new Map<string, number>();
  private wcLastSeekTargetByClipId = new Map<string, number>();
  private wcLastRealtimeResyncByClipId = new Map<string, number>();
  private wcLastRealtimeTimeByClipId = new Map<string, number>();
  private wcLastRealtimeTickByClipId = new Map<string, number>();
  private wcRealtimeStartupUntilByClipId = new Map<string, number>();

  /**
   * Pre-render step: finalize prerolled clips that are now active.
   * Seeks prerolled videos to correct position and unmutes BEFORE render.
   * Does NOT pause — the video must stay playing so the GPU surface remains
   * active (videoGpuReady) and importExternalTexture produces valid frames.
   * The scrubbing cache from preroll phase 1 provides the correct frame
   * while the seek completes.
   *
   * Must be called BEFORE engine.render() in the render callback.
   */
  finalizePrerolls(): void {
    if (this.prerollingClips.size === 0) return;

    const ctx = createFrameContext();
    if (!ctx.isPlaying) return;

    for (const clipId of this.prerollingClips) {
      // Find if this clip is now at the playhead
      const clip = ctx.clipsAtTime.find(c => c.id === clipId);
      if (!clip?.source?.videoElement) continue;

      const video = clip.source.videoElement;
      const timeInfo = getClipTimeInfo(ctx, clip);

      // Unmute (preroll was muted) — keep playing for GPU surface
      // Only unmute if no linked audio clip (linked audio handles audio separately)
      if (!clip.linkedClipId) {
        video.muted = false;
      }

      // Seek to correct position (video was ~0.5s ahead from preroll)
      const timeDiff = Math.abs(video.currentTime - timeInfo.clipTime);
      if (timeDiff > 0.05) {
        video.currentTime = timeInfo.clipTime;
      }

      this.prerollingClips.delete(clipId);
    }
  }

  /**
   * Continuous playback optimization for contiguous same-source clips.
   *
   * When clip B follows clip A on the same track from the same source file,
   * and they're contiguous in both timeline and source time (i.e., a simple cut),
   * we swap video elements: clip B inherits clip A's playing video element.
   * The video keeps playing through the cut point — zero seek, zero stutter.
   *
   * This mimics how DaVinci Resolve handles cuts: one decoder per source file
   * that plays through cut points without interruption.
   *
   * Must be called BEFORE buildLayersFromStore() so the Layer gets the correct video.
   */
  prepareContinuousPlayback(): void {
    const ctx = createFrameContext();

    if (!ctx.isPlaying) {
      this.trackPlaybackState.clear();
      return;
    }

    for (const clip of ctx.clipsAtTime) {
      if (!clip.source?.videoElement || clip.source.type !== 'video') continue;
      if (clip.reversed) continue; // Don't optimize reversed clips

      const prev = this.trackPlaybackState.get(clip.trackId);

      if (prev && prev.clipId !== clip.id) {
        // Transition detected — find previous clip for source comparison
        const prevClip = ctx.clips.find(c => c.id === prev.clipId);

        if (prevClip?.source?.videoElement) {
          // Check same source — multiple strategies (File ref, mediaFileId, file name+size)
          const isSameFileRef = !!(clip.file && prevClip.file && clip.file === prevClip.file);
          const isSameMediaId = !!(
            (clip.source.mediaFileId && prevClip.source?.mediaFileId &&
              clip.source.mediaFileId === prevClip.source.mediaFileId) ||
            (clip.mediaFileId && prevClip.mediaFileId &&
              clip.mediaFileId === prevClip.mediaFileId)
          );
          // Fallback: compare file name + size (works after project reload when File ref is lost)
          const isSameFileName = !!(
            clip.file && prevClip.file &&
            clip.file.name === prevClip.file.name &&
            clip.file.size === prevClip.file.size
          );
          const isSameSource = isSameFileRef || isSameMediaId || isSameFileName;

          const isContiguousTimeline = Math.abs(prev.endTime - clip.startTime) < 0.016; // ~1 frame tolerance
          const isContiguousSource = Math.abs(prev.outPoint - clip.inPoint) < 0.02; // ~1 frame

          if (isSameSource && isContiguousTimeline && isContiguousSource) {
            const playingVideo = prevClip.source.videoElement;
            const idleVideo = clip.source.videoElement;

            if (playingVideo === idleVideo) {
              // Shared element (split clips) — video plays through the cut point
              // No swap needed, just cancel preroll
              this.prerollingClips.delete(clip.id);
            } else if (!playingVideo.paused && playingVideo.readyState >= 2) {
              // Different elements — swap them
              const playingWC = prevClip.source.webCodecsPlayer;
              const idleWC = clip.source.webCodecsPlayer;
              clip.source.videoElement = playingVideo;
              clip.source.webCodecsPlayer = playingWC;
              prevClip.source.videoElement = idleVideo;
              prevClip.source.webCodecsPlayer = idleWC;

              // Cancel preroll — video is already playing at the correct position
              this.prerollingClips.delete(clip.id);
            }
          }
        }
      }

      // Update tracking for current active clip
      const sourceId = clip.source.mediaFileId || clip.mediaFileId || '';
      this.trackPlaybackState.set(clip.trackId, {
        clipId: clip.id,
        sourceMediaFileId: sourceId,
        outPoint: clip.outPoint,
        endTime: clip.startTime + clip.duration,
      });
    }
  }

  /**
   * Sync video elements to current playhead
   */
  syncVideoElements(): void {
    const ctx = createFrameContext();

    const hasUnstartedWebCodecsAtPlayhead = ctx.isPlaying && (
      ctx.clipsAtTime.some(
        c => !!c.source?.webCodecsPlayer && !c.source.videoElement && !c.source.webCodecsPlayer.isPlaying
      ) ||
      ctx.clipsAtTime.some(c =>
        !!c.isComposition &&
        !!c.nestedClips?.some(
          nc => !!nc.source?.webCodecsPlayer && !nc.source.videoElement && !nc.source.webCodecsPlayer.isPlaying
        )
      )
    );

    // Skip if same frame during playback
    if (ctx.isPlaying && !ctx.isDraggingPlayhead &&
        ctx.frameNumber === this.lastVideoSyncFrame &&
        ctx.isPlaying === this.lastVideoSyncPlaying &&
        !hasUnstartedWebCodecsAtPlayhead) {
      return;
    }
    this.lastVideoSyncFrame = ctx.frameNumber;
    this.lastVideoSyncPlaying = ctx.isPlaying;

    // Sync each clip at playhead
    for (const clip of ctx.clipsAtTime) {
      this.syncClipVideo(clip, ctx);

      // Sync nested composition videos
      if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        this.syncNestedCompVideos(clip, ctx);
      }
    }

    // Pause videos/players not at playhead (but skip clips being prerolled or sharing elements)
    for (const clip of ctx.clips) {
      const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
        ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;

      // Pause WebCodecs players not at playhead
      if (clip.source?.webCodecsPlayer && !clip.source.videoElement && !isAtPlayhead) {
        // Stop decoder loop immediately when clip is no longer active.
        if (clip.source.webCodecsPlayer.isPlaying) {
          clip.source.webCodecsPlayer.pause();
        }
      }

      if (clip.source?.videoElement) {
        const isPrerolling = this.prerollingClips.has(clip.id);
        if (!isAtPlayhead && !isPrerolling && !clip.source.videoElement.paused) {
          // Don't pause if an active clip shares this video element (split clips)
          const video = clip.source.videoElement;
          const sharedByActiveClip = Array.from(ctx.clipsByTrackId.values()).some(
            active => active.source?.videoElement === video
          );
          if (!sharedByActiveClip) {
            video.pause();
          }
        }
      }

      // Pause nested comp videos not at playhead
      if (clip.isComposition && clip.nestedClips) {
        const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
          ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;
        if (!isAtPlayhead) {
          for (const nestedClip of clip.nestedClips) {
            const isNestedPrerolling = this.prerollingClips.has(nestedClip.id);
            if (nestedClip.source?.videoElement && !nestedClip.source.videoElement.paused && !isNestedPrerolling) {
              nestedClip.source.videoElement.pause();
            }
          }
        }
      }
    }

    // Preload upcoming clips (seek video to start position before they become active)
    if (ctx.isPlaying) {
      this.preloadUpcomingClips(ctx);
    }

    // Sync background layer video elements
    layerPlaybackManager.syncVideoElements(ctx.playheadPosition, ctx.isPlaying);
  }

  // Track which clips are being pre-rolled to avoid redundant play() calls
  private prerollingClips = new Set<string>();

  /**
   * Preload video elements for clips about to become active.
   * Two-phase strategy:
   *   - 2s+ out: seek to start position and pause (buffering)
   *   - <0.5s out: start playing muted from smart position (preroll)
   *
   * Smart preroll: starts from (inPoint - timeUntilStart) so the video
   * naturally arrives at inPoint when the clip becomes active — no seek needed
   * at transition. This eliminates stutter with many short clips.
   *
   * Uses sorted clips with early break for O(k) iteration instead of O(n).
   */
  private preloadUpcomingClips(ctx: FrameContext): void {
    const lookaheadSec = 2;
    const prerollSec = 0.5; // Start playing muted when clip is <0.5s away
    const lookaheadEnd = ctx.playheadPosition + lookaheadSec;

    // Re-sort clips by startTime when the clips array reference changes
    // (Zustand returns same array reference if nothing changed)
    if (ctx.clips !== this.lastClipsRef) {
      this.sortedClipsByStart = [...ctx.clips].sort((a, b) => a.startTime - b.startTime);
      this.lastClipsRef = ctx.clips;
    }

    // Iterate sorted clips with early break — only visits clips in the lookahead window
    for (const clip of this.sortedClipsByStart) {
      if (clip.startTime <= ctx.playheadPosition) continue; // Skip past clips
      if (clip.startTime > lookaheadEnd) break; // Sorted: all remaining are past lookahead

      const timeUntilStart = clip.startTime - ctx.playheadPosition;

      // Skip preload for clips that share a video element with the active clip,
      // or will receive continuous playback handoff (same source + contiguous).
      // The video element will just keep playing through the cut point.
      if (clip.source?.videoElement && !clip.reversed) {
        const activeClip = ctx.clipsByTrackId.get(clip.trackId);
        if (activeClip?.source?.videoElement) {
          // Shared element (split clips) — always skip preload
          if (clip.source.videoElement === activeClip.source.videoElement) {
            continue;
          }
          const isSameFile = clip.file && activeClip.file && clip.file === activeClip.file;
          const isSameMediaId = (
            (clip.source.mediaFileId && activeClip.source?.mediaFileId &&
              clip.source.mediaFileId === activeClip.source.mediaFileId) ||
            (clip.mediaFileId && activeClip.mediaFileId &&
              clip.mediaFileId === activeClip.mediaFileId)
          );
          if (isSameFile || isSameMediaId) {
            const isContiguousTimeline = Math.abs(
              (activeClip.startTime + activeClip.duration) - clip.startTime
            ) < 0.001;
            const isContiguousSource = Math.abs(activeClip.outPoint - clip.inPoint) < 0.02;
            if (isContiguousTimeline && isContiguousSource) {
              continue; // Skip — video will play through from active clip
            }
          }
        }
      }

      if (clip.source?.videoElement) {
        this.preloadVideoElement(clip.id, clip.source.videoElement, clip, timeUntilStart, prerollSec);
      } else if (clip.source?.webCodecsPlayer && timeUntilStart <= prerollSec) {
        // WebCodecs: pre-seek to inPoint (instant, no warmup needed)
        const inPoint = clip.reversed ? clip.outPoint : clip.inPoint;
        clip.source.webCodecsPlayer.seek(inPoint);
      }

      if (clip.source?.nativeDecoder) {
        const targetTime = clip.reversed ? clip.outPoint : clip.inPoint;
        const fps = clip.source.nativeDecoder.fps || 25;
        const targetFrame = Math.round(targetTime * fps);
        clip.source.nativeDecoder.seekToFrame(targetFrame, false).catch(() => {});
      }

      // Preload nested comp video elements
      if (clip.isComposition && clip.nestedClips) {
        for (const nestedClip of clip.nestedClips) {
          if (!nestedClip.source?.videoElement) continue;
          this.preloadVideoElement(nestedClip.id, nestedClip.source.videoElement, nestedClip, timeUntilStart, prerollSec);
        }
      }
    }

    // Clean up preroll tracking for clips that are now active
    for (const clipId of this.prerollingClips) {
      const clip = ctx.clips.find(c => c.id === clipId);
      if (clip && ctx.playheadPosition >= clip.startTime) {
        this.prerollingClips.delete(clipId);
      }
    }
  }

  /**
   * Preload a single video element.
   * Phase 1 (>0.5s): seek to inPoint + pause (buffer ahead)
   * Phase 2 (<0.5s): play muted from smart position (decoder warm-up)
   *
   * Smart preroll: Instead of starting from inPoint (which causes a backward
   * seek at transition because the video drifts to inPoint + prerollDuration),
   * we start from (inPoint - timeUntilStart). This way the video naturally
   * arrives at inPoint when the clip becomes active — zero seek needed.
   */
  private preloadVideoElement(
    clipId: string,
    video: HTMLVideoElement,
    clip: TimelineClip,
    timeUntilStart: number,
    prerollSec: number
  ): void {
    const inPoint = clip.reversed ? clip.outPoint : clip.inPoint;

    if (timeUntilStart <= prerollSec) {
      // Phase 2: Preroll — play muted so decoder is warm
      // Smart position: start from (inPoint - timeUntilStart) so by the time
      // the clip becomes active, the video naturally arrives at inPoint.
      // For clips where inPoint < timeUntilStart (e.g. inPoint=0), we clamp to 0
      // and accept a small drift that finalizePrerolls() will correct.
      if (!this.prerollingClips.has(clipId)) {
        const prerollStart = Math.max(0, inPoint - timeUntilStart);
        const timeDiff = Math.abs(video.currentTime - prerollStart);
        if (timeDiff > 0.1 && !video.seeking) {
          video.currentTime = prerollStart;
        }
        video.muted = true;
        // Add to prerollingClips BEFORE play() to prevent the pause loop
        // from killing the preroll during the async .then() gap
        this.prerollingClips.add(clipId);
        video.play().then(() => {
          // Mark GPU-ready: during preroll the video plays muted but
          // tryHTMLVideo never runs for it (not at playhead).
          // Without this, the first render of the clip returns null → black frame.
          engine.markVideoGpuReady(video);
        }).catch(() => {
          this.prerollingClips.delete(clipId);
        });
      }
    } else {
      // Phase 1: Seek to inPoint and pause (buffer ahead of time)
      const timeDiff = Math.abs(video.currentTime - inPoint);
      if (timeDiff > 0.1 && !video.seeking) {
        video.currentTime = inPoint;
        // Cache frame at inPoint after seek completes — this frame will be
        // used by tryHTMLVideo when the clip transitions from preroll to active,
        // preventing a wrong-frame flash on the first render
        video.addEventListener('seeked', () => {
          if (video.readyState >= 2) {
            engine.ensureVideoFrameCached(video);
            engine.cacheFrameAtTime(video, video.currentTime);
          }
        }, { once: true });
      }
      if (!video.paused) {
        video.pause();
      }
    }
  }

  /**
   * Sync nested composition video elements
   * Uses same logic as regular clips: play during playback, seek when paused
   * Also ensures videos have decoded frames (readyState >= 2) for rendering
   */
  private syncNestedCompVideos(compClip: TimelineClip, ctx: FrameContext): void {
    if (!compClip.nestedClips || !compClip.nestedTracks) return;

    // Calculate time within the composition
    const compLocalTime = ctx.playheadPosition - compClip.startTime;
    const compTime = compLocalTime + compClip.inPoint;

    for (const nestedClip of compClip.nestedClips) {
      if (!nestedClip.source?.videoElement) continue;

      // Check if nested clip is active at current comp time
      const isActive = compTime >= nestedClip.startTime && compTime < nestedClip.startTime + nestedClip.duration;

      if (!isActive) {
        // Pause if not active
        if (!nestedClip.source.videoElement.paused) {
          nestedClip.source.videoElement.pause();
        }
        continue;
      }

      // Calculate time within the nested clip
      const nestedLocalTime = compTime - nestedClip.startTime;
      const nestedClipTime = nestedClip.reversed
        ? nestedClip.outPoint - nestedLocalTime
        : nestedLocalTime + nestedClip.inPoint;

      const video = nestedClip.source.videoElement;
      const webCodecsPlayer = nestedClip.source.webCodecsPlayer;
      const timeDiff = Math.abs(video.currentTime - nestedClipTime);

      // Pre-capture: ensure scrubbing cache has a frame before seeking
      if (!video.seeking && video.readyState >= 2) {
        engine.ensureVideoFrameCached(video);
      }

      // During playback: let video play naturally (like regular clips)
      if (ctx.isPlaying) {
        if (video.paused) {
          video.play().catch(() => {});
        }
        // Only seek if significantly out of sync (>0.5s)
        if (timeDiff > 0.5) {
          video.currentTime = nestedClipTime;
        }
      } else {
        // When paused: pause video and seek to exact time
        if (!video.paused) video.pause();

        // Force first-frame decode for videos that haven't played yet (e.g. after reload)
        if (video.played.length === 0 && !video.seeking && !this.forceDecodeInProgress.has(nestedClip.id)) {
          this.forceVideoFrameDecode(nestedClip.id, video);
        }

        const seekThreshold = ctx.isDraggingPlayhead ? 0.1 : 0.05;
        if (timeDiff > seekThreshold) {
          this.throttledSeek(nestedClip.id, video, nestedClipTime, ctx);
          video.addEventListener('seeked', () => engine.requestRender(), { once: true });
        }

        // If video readyState < 2 (no frame data), force decode via play/pause
        // This can happen after seeking to unbuffered regions
        if (video.readyState < 2 && !video.seeking) {
          this.forceVideoFrameDecode(nestedClip.id, video);
        }
      }

      // Sync WebCodecsPlayer only when not playing (it handles its own playback)
      if (webCodecsPlayer && !ctx.isPlaying) {
        const wcTimeDiff = Math.abs(webCodecsPlayer.currentTime - nestedClipTime);
        if (wcTimeDiff > 0.05) {
          webCodecsPlayer.seek(nestedClipTime);
        }
      }
    }
  }

  /**
   * Force video to decode current frame by briefly playing
   * Used when video has never played (after reload) or readyState drops below 2
   */
  private forceVideoFrameDecode(clipId: string, video: HTMLVideoElement): void {
    if (this.forceDecodeInProgress.has(clipId)) return;
    this.forceDecodeInProgress.add(clipId);

    const currentTime = video.currentTime;
    video.muted = true; // Prevent autoplay restrictions
    video.play()
      .then(() => {
        video.pause();
        video.currentTime = currentTime;
        this.forceDecodeInProgress.delete(clipId);
        engine.requestRender();
      })
      .catch(() => {
        // Fallback: tiny seek to trigger decode
        video.currentTime = currentTime + 0.001;
        this.forceDecodeInProgress.delete(clipId);
        engine.requestRender();
      });
  }

  /**
   * Sync a single clip's video element
   */
  private syncClipVideo(clip: TimelineClip, ctx: FrameContext): void {
    // Handle native decoder
    if (clip.source?.nativeDecoder) {
      this.syncNativeDecoder(clip, ctx);
      return;
    }

    // WebCodecs Full Mode — no HTMLVideoElement needed
    if (clip.source?.webCodecsPlayer && !clip.source.videoElement) {
      this.syncWebCodecsPlayer(clip, ctx);
      return;
    }

    if (!clip.source?.videoElement) return;

    const video = clip.source.videoElement;
    const timeInfo = getClipTimeInfo(ctx, clip);
    const mediaFile = getMediaFileForClip(ctx, clip);

    // Check proxy mode based on playback policy.
    // Stability rule: if "prefer original for playback" is enabled, keep one source
    // path (original) for both play and pause/scrub to avoid source-hop jitter.
    const preferNormalizedSource = !ctx.preferOriginalForPlay && (
      ctx.isPlaying || ctx.preferNormalizedForScrub
    );
    const useProxy = ctx.proxyEnabled && preferNormalizedSource && mediaFile?.proxyFps &&
      (mediaFile.proxyStatus === 'ready' || mediaFile.proxyStatus === 'generating');

    if (useProxy) {
      // In proxy mode: pause video
      if (!video.paused) video.pause();
      if (!video.muted) video.muted = true;
      return;
    }

    // Skip sync during GPU surface warmup — the video is playing briefly
    // to activate Chrome's GPU decoder. Don't pause or seek it.
    if (this.warmingUpVideos.has(video)) return;

    // Warmup: after page reload, video GPU surfaces are empty.
    // importExternalTexture, canvas.drawImage, etc. all return black.
    // The ONLY fix is video.play() to activate the GPU compositor.
    // We do this here (not during restore) because restore-time warmup
    // gets immediately killed by this very function's "pause if not playing" logic.
    const hasSrc = !!(video.src || video.currentSrc);
    const warmupCooldown = this.warmupRetryCooldown.get(video);
    const cooldownOk = !warmupCooldown || performance.now() - warmupCooldown > 2000;
    if (!ctx.isPlaying && !video.seeking && hasSrc && cooldownOk &&
        video.played.length === 0 && !this.warmingUpVideos.has(video)) {
      this.warmingUpVideos.add(video);
      const targetTime = timeInfo.clipTime;
      video.play().then(() => {
        // Wait for actual frame presentation via requestVideoFrameCallback
        const rvfc = (video as any).requestVideoFrameCallback;
        if (typeof rvfc === 'function') {
          rvfc.call(video, () => {
            // Frame is now presented to GPU — capture it
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = targetTime;
            this.warmingUpVideos.delete(video);
            engine.requestRender();
          });
        } else {
          // Fallback: wait 100ms for frame presentation
          setTimeout(() => {
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = targetTime;
            this.warmingUpVideos.delete(video);
            engine.requestRender();
          }, 100);
        }
      }).catch(() => {
        this.warmingUpVideos.delete(video);
        this.warmupRetryCooldown.set(video, performance.now());
      });
      return; // Skip normal sync — warmup is handling video state
    }

    // Normal video sync
    const timeDiff = Math.abs(video.currentTime - timeInfo.clipTime);

    // Pre-capture: ensure scrubbing cache has a frame BEFORE seeking
    if (!video.seeking && video.readyState >= 2) {
      engine.ensureVideoFrameCached(video);
    }

    // Reverse playback: either clip is reversed OR timeline playbackSpeed is negative
    // H.264 can't play backwards, so we seek frame-by-frame
    const isReversePlayback = clip.reversed || ctx.playbackSpeed < 0;

    if (isReversePlayback) {
      // For reverse: pause video and seek to each frame
      if (!video.paused) video.pause();
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.02;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
    } else if (ctx.playbackSpeed !== 1) {
      // Non-standard forward speed (2x, 4x, etc.): seek frame-by-frame for accuracy
      if (!video.paused) video.pause();
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.03;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
    } else {
      // Normal 1x forward playback: let video play naturally

      // Clean up preroll state FIRST — need to know if this clip just
      // transitioned from preroll to active for position correction below
      const wasPrerolling = this.prerollingClips.has(clip.id);
      if (wasPrerolling) {
        this.prerollingClips.delete(clip.id);
        // Only unmute if no linked audio clip (linked audio handles audio separately)
        if (!clip.linkedClipId) {
          video.muted = false;
        }
      }

      if (ctx.isPlaying && video.paused) {
        // Video is paused but should be playing — seek to correct position
        // (preroll drift or stale position from previous pause)
        if (timeDiff > 0.05) {
          video.currentTime = timeInfo.clipTime;
        }
        video.play().catch(() => {});
      } else if (ctx.isPlaying && !video.paused) {
        // Video is already playing — correct drift
        if (wasPrerolling) {
          // Just transitioned from preroll: video was playing from inPoint
          // for ~0.5s, so it's significantly ahead. Always correct.
          if (timeDiff > 0.05) {
            video.currentTime = timeInfo.clipTime;
          }
        } else if (timeDiff > 0.15) {
          // Ongoing playback: correct if drift > 0.15s (~4-5 frames at 30fps)
          video.currentTime = timeInfo.clipTime;
        }
      } else if (!ctx.isPlaying && !video.paused) {
        // Stopping playback: pause and seek to exact playhead position
        // to prevent "jump back one frame" visual artifact
        video.pause();
        if (timeDiff > 0.02) {
          video.currentTime = timeInfo.clipTime;
        }
      }

      if (!ctx.isPlaying) {
        // 0.04s ≈ slightly more than 1 frame at 30fps.
        // Previous 0.1s threshold skipped up to 3 frames during slow scrubbing.
        const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.04;
        if (timeDiff > seekThreshold) {
          this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
        }

        // Force decode if readyState dropped after seek
        if (video.readyState < 2 && !video.seeking) {
          this.forceVideoFrameDecode(clip.id, video);
        }
      }
    }
  }

  /**
   * Sync a clip using WebCodecs Full Mode (no HTMLVideoElement).
   * Much simpler than the HTMLVideoElement path — no warmup, no readyState,
   * no GPU surface issues, no fastSeek hybrid. Just seek() and the frame is there.
   */
  private syncWebCodecsPlayer(clip: TimelineClip, ctx: FrameContext): void {
    const player = clip.source!.webCodecsPlayer!;
    const timeInfo = getClipTimeInfo(ctx, clip);

    if (!player.ready) return;

    const drift = Math.abs(player.currentTime - timeInfo.clipTime);
    playbackMetrics.recordDecodeQueueDepth(player.getDecodeQueueDepth());
    const canRealtimePlay =
      ctx.isPlaying &&
      !ctx.isDraggingPlayhead &&
      ctx.playbackSpeed > 0 &&
      Math.abs(ctx.playbackSpeed - 1) < 0.001 &&
      !clip.reversed;

    if (canRealtimePlay) {
      if (this.wcPlayerModeByClipId.get(clip.id) !== 'realtime') {
        this.wcPlayerModeByClipId.set(clip.id, 'realtime');
        log.info('WC mode -> realtime', {
          clipId: clip.id,
          clipName: clip.name,
          playhead: Number(ctx.playheadPosition.toFixed(3)),
          driftMs: Math.round(drift * 1000),
        });
      }
      // Real-time 1x forward playback: run decoder continuously.
      if (!player.isPlaying) {
        // Use playFrom() for atomic seek + play to avoid the race condition
        // where play() cancels a preceding seek() via seekScheduler.reset().
        // playFrom() resets the decoder, decodes from the nearest keyframe to
        // clipTime, then starts the realtime loop from the correct position.
        player.playFrom(timeInfo.clipTime);
        this.lastSeekRef[clip.id] = ctx.now;
        this.wcLastSeekTargetByClipId.set(clip.id, timeInfo.clipTime);
        this.wcLastRealtimeTimeByClipId.set(clip.id, player.currentTime);
        this.wcLastRealtimeTickByClipId.set(clip.id, ctx.now);
        this.wcRealtimeStartupUntilByClipId.set(clip.id, ctx.now + 2000);
      } else if (drift > 5.0) {
        // Emergency-only resync with cooldown.
        // Without throttling this can issue seek storms and freeze preview.
        const lastResync = this.wcLastRealtimeResyncByClipId.get(clip.id) ?? 0;
        if (ctx.now - lastResync > 1500) {
          this.wcLastRealtimeResyncByClipId.set(clip.id, ctx.now);
          player.seek(timeInfo.clipTime);
        }
      } else {
        // Early playback: allow sparse corrective seeks while decoder locks in.
        const startupUntil = this.wcRealtimeStartupUntilByClipId.get(clip.id) ?? 0;
        if (ctx.now < startupUntil && drift > 0.6) {
          const lastSeek = this.lastSeekRef[clip.id] || 0;
          if (ctx.now - lastSeek > 400) {
            player.seek(timeInfo.clipTime);
            this.lastSeekRef[clip.id] = ctx.now;
            this.wcLastSeekTargetByClipId.set(clip.id, timeInfo.clipTime);
          }
        }

        // Decoder stall recovery: if WC time is not moving, kick play loop.
        const current = player.currentTime;
        const prevTime = this.wcLastRealtimeTimeByClipId.get(clip.id) ?? current;
        const prevTick = this.wcLastRealtimeTickByClipId.get(clip.id) ?? ctx.now;
        const advanced = Math.abs(current - prevTime) > 0.001;
        if (advanced) {
          this.wcLastRealtimeTimeByClipId.set(clip.id, current);
          this.wcLastRealtimeTickByClipId.set(clip.id, ctx.now);
        } else if (ctx.now - prevTick > 700) {
          const lastResync = this.wcLastRealtimeResyncByClipId.get(clip.id) ?? 0;
          if (ctx.now - lastResync > 1200) {
            this.wcLastRealtimeResyncByClipId.set(clip.id, ctx.now);
            player.pause();
            player.play();
          }
        }
      }
    } else {
      if (this.wcPlayerModeByClipId.get(clip.id) !== 'seek') {
        this.wcPlayerModeByClipId.set(clip.id, 'seek');
        log.info('WC mode -> seek', {
          clipId: clip.id,
          clipName: clip.name,
          isPlaying: ctx.isPlaying,
          isDraggingPlayhead: ctx.isDraggingPlayhead,
          playbackSpeed: ctx.playbackSpeed,
          playhead: Number(ctx.playheadPosition.toFixed(3)),
        });
      }
      // Paused/scrub/reverse/variable speed: frame-accurate seek mode.
      if (player.isPlaying) {
        player.pause();
      }

      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.02;
      if (drift > seekThreshold) {
        const lastSeek = this.lastSeekRef[clip.id] || 0;
        const throttle = ctx.isDraggingPlayhead ? 50 : 33;
        const lastTarget = this.wcLastSeekTargetByClipId.get(clip.id);
        const targetChanged = lastTarget === undefined || Math.abs(lastTarget - timeInfo.clipTime) > 0.01;
        const watchdogDue = !ctx.isDraggingPlayhead && drift > 0.25 && ctx.now - lastSeek > 3000;
        if (ctx.now - lastSeek > throttle && (ctx.isDraggingPlayhead || targetChanged || watchdogDue)) {
          player.seek(timeInfo.clipTime);
          this.lastSeekRef[clip.id] = ctx.now;
          this.wcLastSeekTargetByClipId.set(clip.id, timeInfo.clipTime);
          engine.requestRender();
          const prevWarn = this.wcLastSeekWarnByClipId.get(clip.id) ?? 0;
          if (!ctx.isDraggingPlayhead && drift > 0.2 && ctx.now - prevWarn > 1000) {
            this.wcLastSeekWarnByClipId.set(clip.id, ctx.now);
            log.warn('Large WC seek correction while paused', {
              clipId: clip.id,
              clipName: clip.name,
              driftMs: Math.round(drift * 1000),
              targetTime: Number(timeInfo.clipTime.toFixed(3)),
              currentTime: Number(player.currentTime.toFixed(3)),
            });
          }
        }
      }
    }

  }

  /**
   * Hybrid seeking strategy for smooth scrubbing on all codec types:
   *
   * During drag (fast scrubbing):
   *   Phase 1: fastSeek → instant keyframe feedback (<10ms, shows nearest I-frame)
   *   Phase 2: deferred precise seek → exact frame when scrubbing pauses (debounced 120ms)
   *
   * This solves the long-GOP problem: YouTube/phone videos with 5-7s keyframe distance
   * previously showed a stale cached frame for 100-300ms per seek (currentTime decodes
   * from keyframe to target). Now the user sees the nearest keyframe immediately, then
   * the exact frame fills in when they pause.
   *
   * When not dragging (single click / arrow keys): precise seek via currentTime.
   *
   * RVFC (requestVideoFrameCallback) triggers re-render when the decoded frame is
   * actually presented to the compositor — more accurate than the 'seeked' event.
   */
  private throttledSeek(clipId: string, video: HTMLVideoElement, time: number, ctx: FrameContext): void {
    const lastSeek = this.lastSeekRef[clipId] || 0;
    const threshold = ctx.isDraggingPlayhead ? 50 : 33;
    if (ctx.now - lastSeek > threshold) {
      if (ctx.isDraggingPlayhead && 'fastSeek' in video) {
        // Phase 1: Instant keyframe feedback via fastSeek.
        // For all-intra codecs this IS the exact frame. For long-GOP codecs
        // this shows the nearest keyframe — better than a stale cached frame.
        video.fastSeek(time);

        // Phase 2: Schedule deferred precise seek for exact frame.
        // Debounced: resets on each new scrub position, only fires when
        // the user pauses or slows their scrubbing.
        this.latestSeekTargets[clipId] = time;
        clearTimeout(this.preciseSeekTimers[clipId]);
        this.preciseSeekTimers[clipId] = setTimeout(() => {
          const target = this.latestSeekTargets[clipId];
          // Only do precise seek if the fastSeek landed far from the target
          // (i.e., this is a long-GOP video where fastSeek shows a different frame)
          if (target !== undefined && Math.abs(video.currentTime - target) > 0.01) {
            video.currentTime = target;
            // Register RVFC for when the precise frame arrives
            this.registerRVFC(clipId, video);
          }
        }, 120);
      } else {
        // Not dragging: precise seek immediately (click, arrow keys, etc.)
        video.currentTime = time;
        clearTimeout(this.preciseSeekTimers[clipId]);
      }
      this.lastSeekRef[clipId] = ctx.now;

      // Register RVFC to trigger re-render when the decoded frame is presented.
      this.registerRVFC(clipId, video);
    }
  }

  private registerRVFC(clipId: string, video: HTMLVideoElement): void {
    const rvfc = (video as any).requestVideoFrameCallback;
    if (typeof rvfc === 'function') {
      const prevHandle = this.rvfcHandles[clipId];
      if (prevHandle !== undefined) {
        (video as any).cancelVideoFrameCallback(prevHandle);
      }
      this.rvfcHandles[clipId] = rvfc.call(video, () => {
        delete this.rvfcHandles[clipId];
        // Bypass the scrub rate limiter — a fresh decoded frame should be displayed immediately
        engine.requestNewFrameRender();
      });
    }
  }

  /**
   * Sync native decoder
   */
  private syncNativeDecoder(clip: TimelineClip, ctx: FrameContext): void {
    const nativeDecoder = clip.source!.nativeDecoder!;
    const timeInfo = getClipTimeInfo(ctx, clip);

    const fps = nativeDecoder.fps || 25;
    const targetFrame = Math.round(timeInfo.clipTime * fps);

    // Get or create state
    let state = this.nativeDecoderState.get(clip.id);
    if (!state) {
      state = { lastSeekTime: 0, lastSeekFrame: -1, isPending: false };
      this.nativeDecoderState.set(clip.id, state);
    }

    const timeSinceLastSeek = ctx.now - state.lastSeekTime;
    const shouldSeek = !state.isPending &&
      (targetFrame !== state.lastSeekFrame || timeSinceLastSeek > 100);

    if (shouldSeek && timeSinceLastSeek >= LAYER_BUILDER_CONSTANTS.NATIVE_SEEK_THROTTLE_MS) {
      state.lastSeekTime = ctx.now;
      state.lastSeekFrame = targetFrame;
      state.isPending = true;

      nativeDecoder.seekToFrame(targetFrame, ctx.isDraggingPlayhead)
        .then(() => { state!.isPending = false; })
        .catch((err: unknown) => {
          state!.isPending = false;
          log.warn('Native decoder seek failed', { clipId: clip.id, targetFrame, err });
        });
    }
  }
}
