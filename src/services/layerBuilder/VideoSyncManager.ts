// VideoSyncManager - Handles video element synchronization with playhead
// Extracted from LayerBuilderService for separation of concerns

import type { TimelineClip } from '../../types';
import type { FrameContext, NativeDecoderState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip } from './FrameContext';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { engine } from '../../engine/WebGPUEngine';

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
      video.muted = false;

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

      const sourceId = clip.source.mediaFileId || clip.mediaFileId || '';
      const prev = this.trackPlaybackState.get(clip.trackId);

      if (prev && prev.clipId !== clip.id && sourceId !== '') {
        // Transition detected — check if same source + contiguous
        const isSameSource = sourceId === prev.sourceMediaFileId;
        const isContiguousTimeline = Math.abs(prev.endTime - clip.startTime) < 0.001;
        const isContiguousSource = Math.abs(prev.outPoint - clip.inPoint) < 0.02; // ~1 frame tolerance

        if (isSameSource && isContiguousTimeline && isContiguousSource) {
          // Find previous clip to swap video elements
          const prevClip = ctx.clips.find(c => c.id === prev.clipId);
          if (prevClip?.source?.videoElement) {
            // SWAP: give the playing video to the new clip.
            // The playing video is already at the correct position (outPoint of prev ≈ inPoint of clip).
            // The new clip's idle video goes to the previous clip (will be paused by sync loop).
            const playingVideo = prevClip.source.videoElement;
            const playingWC = prevClip.source.webCodecsPlayer;
            const idleVideo = clip.source.videoElement;
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

      // Update tracking for current active clip
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

    // Skip if same frame during playback
    if (ctx.isPlaying && !ctx.isDraggingPlayhead &&
        ctx.frameNumber === this.lastVideoSyncFrame &&
        ctx.isPlaying === this.lastVideoSyncPlaying) {
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

    // Pause videos not at playhead (but skip clips being prerolled)
    for (const clip of ctx.clips) {
      if (clip.source?.videoElement) {
        const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
          ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;
        const isPrerolling = this.prerollingClips.has(clip.id);
        if (!isAtPlayhead && !isPrerolling && !clip.source.videoElement.paused) {
          clip.source.videoElement.pause();
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

      // Skip preload for clips that will receive continuous playback handoff:
      // same source + contiguous with the currently active clip on the same track.
      // The video element will just keep playing through the cut point.
      if (clip.source?.videoElement && !clip.reversed) {
        const activeClip = ctx.clipsByTrackId.get(clip.trackId);
        if (activeClip?.source?.videoElement) {
          const activeSourceId = activeClip.source.mediaFileId || activeClip.mediaFileId || '';
          const upcomingSourceId = clip.source.mediaFileId || clip.mediaFileId || '';
          if (activeSourceId !== '' && activeSourceId === upcomingSourceId) {
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

    if (!clip.source?.videoElement) return;

    const video = clip.source.videoElement;
    const timeInfo = getClipTimeInfo(ctx, clip);
    const mediaFile = getMediaFileForClip(ctx, clip);

    // Check proxy mode
    const useProxy = ctx.proxyEnabled && mediaFile?.proxyFps &&
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
        video.muted = false; // Preroll muted the video — restore for audio playback
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
        .catch((err: unknown) => { state!.isPending = false; console.warn('[NH] seek failed frame', targetFrame, err); });
    }
  }
}
