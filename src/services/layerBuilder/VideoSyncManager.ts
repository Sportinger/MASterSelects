// VideoSyncManager - Handles video element synchronization with playhead
// Extracted from LayerBuilderService for separation of concerns

import type { TimelineClip } from '../../types';
import type { FrameContext, NativeDecoderState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip } from './FrameContext';
import { playheadState } from './PlayheadState';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { engine } from '../../engine/WebGPUEngine';
import { useTimelineStore } from '../../stores/timeline';
import { MAX_NESTING_DEPTH } from '../../stores/timeline/constants';
import { wcPipelineMonitor } from '../wcPipelineMonitor';
import { vfPipelineMonitor } from '../vfPipelineMonitor';

export class VideoSyncManager {
  // Native decoder state
  private nativeDecoderState = new Map<string, NativeDecoderState>();

  // Video sync throttling
  private lastVideoSyncFrame = -1;
  private lastVideoSyncPlaying = false;
  private lastSeekRef: Record<string, number> = {};

  // Track per-clip playing state to detect playing→paused transitions
  private clipWasPlaying = new Set<string>();

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

  // WebCodecs precise seek debounce
  private wcPreciseSeekTimers: Record<string, ReturnType<typeof setTimeout>> = {};

  // Seamless cut transition: track last video element per track.
  // When same-source clips are sequential (split clips), the outgoing clip's
  // video element keeps playing through the cut — no pause/play gap.
  private lastTrackState = new Map<string, {
    clipId: string;
    videoSrc: string;
    videoElement: HTMLVideoElement;
    outPoint: number;
  }>();
  private activeHandoffs = new Map<string, HTMLVideoElement>();
  private handoffElements = new Set<HTMLVideoElement>();

  /**
   * Clamp seek time to valid range, preventing EOF decoder stalls.
   * H.264 B-frame decoders stall when seeking to exactly video.duration
   * because they wait for reference frames that don't exist.
   */
  private safeSeekTime(video: HTMLVideoElement, time: number): number {
    const dur = video.duration;
    if (!isFinite(dur) || dur <= 0) return Math.max(0, time);
    return Math.max(0, Math.min(time, dur - 0.001));
  }

  /**
   * Detect same-source sequential clips and set up handoffs.
   * Called from both LayerBuilderService.buildLayers() (for rendering)
   * and syncVideoElements() (for sync + pause prevention).
   */
  computeHandoffs(ctx: FrameContext): void {
    this.activeHandoffs.clear();
    this.handoffElements.clear();

    if (!ctx.isPlaying || ctx.isDraggingPlayhead) return;

    for (const clip of ctx.clipsAtTime) {
      if (!clip.source?.videoElement || !clip.trackId) continue;

      const prev = this.lastTrackState.get(clip.trackId);
      if (!prev || prev.clipId === clip.id) continue;

      const clipSrc = clip.source.videoElement.src || clip.source.videoElement.currentSrc;
      if (!clipSrc || prev.videoSrc !== clipSrc) continue;

      // Continuous cut: clip's inPoint matches previous clip's outPoint
      if (Math.abs(clip.inPoint - prev.outPoint) > 0.1) continue;

      // Previous element should be near the clip's inPoint (playing through)
      if (Math.abs(prev.videoElement.currentTime - clip.inPoint) > 0.5) continue;

      this.activeHandoffs.set(clip.id, prev.videoElement);
      this.handoffElements.add(prev.videoElement);
    }
  }

  /**
   * Get handoff video element for a clip (if same-source sequential transition).
   * Returns null if no handoff is active.
   */
  getHandoffVideoElement(clipId: string): HTMLVideoElement | null {
    return this.activeHandoffs.get(clipId) ?? null;
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

    // Compute handoffs for seamless cut transitions
    this.computeHandoffs(ctx);

    // Sync each clip at playhead
    for (const clip of ctx.clipsAtTime) {
      this.syncClipVideo(clip, ctx);

      // Sync nested composition videos
      if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
        this.syncNestedCompVideos(clip, ctx);
      }
    }

    // Pause videos not at playhead (but don't pause videos during GPU warmup)
    for (const clip of ctx.clips) {
      if (clip.source?.videoElement) {
        const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
          ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;
        if (!isAtPlayhead && !clip.source.videoElement.paused &&
            !this.warmingUpVideos.has(clip.source.videoElement) &&
            !this.handoffElements.has(clip.source.videoElement)) {
          clip.source.videoElement.pause();
        }
      }

      // Pause nested comp videos not at playhead
      if (clip.isComposition && clip.nestedClips) {
        const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
          ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;
        if (!isAtPlayhead) {
          for (const nestedClip of clip.nestedClips) {
            if (nestedClip.source?.videoElement && !nestedClip.source.videoElement.paused) {
              nestedClip.source.videoElement.pause();
            }
          }
        }
      }
    }

    // Proactive GPU warmup: look ahead 0.5s and warm up video elements
    // for clips that are about to enter the frame. Without this, split clips
    // stutter at cut boundaries because each HTMLVideoElement needs GPU activation.
    if (ctx.isPlaying) {
      this.warmupUpcomingClips(ctx);
    }

    // Update track state for seamless cut transition detection
    this.updateLastTrackState(ctx);

    // Sync background layer video elements
    layerPlaybackManager.syncVideoElements(ctx.playheadPosition, ctx.isPlaying);
  }

  /**
   * Sync nested composition video elements
   * Uses same logic as regular clips: play during playback, seek when paused
   * Also ensures videos have decoded frames (readyState >= 2) for rendering
   */
  private syncNestedCompVideos(compClip: TimelineClip, ctx: FrameContext, depth: number = 0): void {
    if (!compClip.nestedClips || !compClip.nestedTracks) return;
    if (depth >= MAX_NESTING_DEPTH) return;

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
          video.currentTime = this.safeSeekTime(video, nestedClipTime);
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

    // Recursively sync sub-nested composition videos (Level 3+)
    for (const nestedClip of compClip.nestedClips) {
      if (nestedClip.isComposition && nestedClip.nestedClips && nestedClip.nestedClips.length > 0) {
        // Create a virtual context with adjusted time for the sub-composition
        const compLocalTime = ctx.playheadPosition - compClip.startTime;
        const compTime = compLocalTime + compClip.inPoint;
        const isActive = compTime >= nestedClip.startTime && compTime < nestedClip.startTime + nestedClip.duration;
        if (isActive) {
          const subCtx = {
            ...ctx,
            playheadPosition: compTime - nestedClip.startTime + nestedClip.inPoint,
          };
          // Temporarily adjust the nested clip's startTime context for recursive call
          const virtualCompClip = {
            ...nestedClip,
            startTime: 0, // Already offset-adjusted above
          };
          this.syncNestedCompVideos(virtualCompClip, { ...subCtx, playheadPosition: compTime }, depth + 1);
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

    // Full-mode WebCodecs: player handles its own decode loop,
    // HTMLVideoElement is only used for audio. Sync both.
    if (clip.source.webCodecsPlayer?.isFullMode()) {
      this.syncFullWebCodecs(clip, ctx);
      return;
    }

    // Use handoff element if available (seamless cut transition)
    const video = this.activeHandoffs.get(clip.id) ?? clip.source.videoElement;
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
      vfPipelineMonitor.record('vf_gpu_cold', { clipId: clip.id });
      const targetTime = timeInfo.clipTime;
      video.play().then(() => {
        // Wait for actual frame presentation via requestVideoFrameCallback
        const rvfc = (video as any).requestVideoFrameCallback;
        if (typeof rvfc === 'function') {
          rvfc.call(video, () => {
            // Frame is now presented to GPU — capture it
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = this.safeSeekTime(video, targetTime);
            this.warmingUpVideos.delete(video);
            vfPipelineMonitor.record('vf_gpu_ready', { clipId: clip.id });
            engine.requestRender();
          });
        } else {
          // Fallback: wait 100ms for frame presentation
          setTimeout(() => {
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = this.safeSeekTime(video, targetTime);
            this.warmingUpVideos.delete(video);
            vfPipelineMonitor.record('vf_gpu_ready', { clipId: clip.id, fallback: 'true' });
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

    // Reverse playback: clip is reversed, timeline playbackSpeed is negative, or clip speed is negative
    // H.264 can't play backwards, so we seek frame-by-frame
    const isReversePlayback = clip.reversed || ctx.playbackSpeed < 0 || timeInfo.speed < 0;

    // Clip-level speed (separate from transport playbackSpeed)
    const clipAbsSpeed = timeInfo.absSpeed;
    const needsClipSpeedAdjust = clipAbsSpeed > 0.01 && Math.abs(clipAbsSpeed - 1) > 0.01;

    // Speed keyframes: variable speed uses playbackRate + relaxed drift correction.
    // The per-frame drift between instantaneous playbackRate and the integrated curve
    // is tiny (<100ms over 10s for typical ramps). Frequent seeking causes decoder
    // stalls (SEEK_STUCK), so we use a high threshold and let playbackRate handle it.
    const hasSpeedKeyframes = ctx.hasKeyframes(clip.id, 'speed');

    if (isReversePlayback) {
      // For reverse: pause video and seek to each frame
      if (!video.paused) video.pause();
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.02;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
    } else if (ctx.playbackSpeed !== 1) {
      // Non-standard forward transport speed (2x, 4x, etc.): seek frame-by-frame
      if (!video.paused) video.pause();
      this.clipWasPlaying.delete(clip.id);
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.03;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
    } else {
      // Normal forward playback (transport speed = 1x)
      // Apply clip-level speed via video.playbackRate (e.g. 2x, 0.5x, or speed keyframes)
      if (needsClipSpeedAdjust || hasSpeedKeyframes) {
        const targetRate = Math.max(0.0625, Math.min(16, clipAbsSpeed));
        if (Math.abs(video.playbackRate - targetRate) > 0.01) {
          video.playbackRate = targetRate;
        }
        // Set preservesPitch based on clip setting (default true)
        const shouldPreservePitch = clip.preservesPitch !== false;
        if ((video as any).preservesPitch !== shouldPreservePitch) {
          (video as any).preservesPitch = shouldPreservePitch;
        }
      } else if (video.playbackRate !== 1) {
        // Reset playbackRate when clip speed returns to 1x
        video.playbackRate = 1;
      }

      if (ctx.isPlaying) {
        this.clipWasPlaying.add(clip.id);
        if (video.paused) {
          // Only seek before play if video is significantly out of sync.
          // After a clean pause the video is already at the correct frame —
          // an unnecessary seek forces the decoder to re-decode from the last
          // keyframe which causes a visible backward jitter on long-GOP codecs.
          if (timeDiff > 0.05) {
            video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
          }
          video.play().catch(() => {});
          vfPipelineMonitor.record('vf_play', { clipId: clip.id });
        }
        // Drift correction during playback.
        // For speed-keyframed clips: playbackRate follows the curve closely, drift is
        // small. Use a high threshold to avoid decoder stalls (SEEK_STUCK) from frequent seeks.
        const driftThreshold = hasSpeedKeyframes ? 1.5 : 0.3;
        if (timeDiff > driftThreshold) {
          vfPipelineMonitor.record('vf_drift', {
            clipId: clip.id,
            driftMs: Math.round(timeDiff * 1000),
            target: Math.round(timeInfo.clipTime * 1000) / 1000,
          });
          video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
        }
      } else {
        // Playing → Paused transition: snap playhead to where the video actually is
        // instead of seeking the video back to the playhead (which causes visible frame jump).
        // The video was playing freely with up to 0.3s drift — we keep its current frame
        // and move the playhead to match.
        const justStopped = this.clipWasPlaying.has(clip.id);
        if (justStopped) {
          this.clipWasPlaying.delete(clip.id);
          // If handoff was active, the actual playing element differs from clip's own
          const prevTrack = this.lastTrackState.get(clip.trackId);
          const actualVideo = (prevTrack && prevTrack.videoElement !== video)
            ? prevTrack.videoElement : video;
          if (!actualVideo.paused) {
            actualVideo.pause();
            vfPipelineMonitor.record('vf_pause', { clipId: clip.id });
          }
          // Convert actualVideo.currentTime back to timeline position
          const effectiveSpeed = timeInfo.absSpeed > 0.01 ? timeInfo.absSpeed : 1;
          const videoClipTime = actualVideo.currentTime;
          const newPlayheadPos = clip.reversed
            ? clip.startTime + (clip.outPoint - videoClipTime) / effectiveSpeed
            : clip.startTime + (videoClipTime - clip.inPoint) / effectiveSpeed;
          const currentPlayhead = playheadState.isUsingInternalPosition
            ? playheadState.position
            : ctx.playheadPosition;
          const videoAdvanced = Math.abs(newPlayheadPos - currentPlayhead) > 0.01;
          if (videoAdvanced) {
            playheadState.position = newPlayheadPos;
            useTimelineStore.setState({ playheadPosition: newPlayheadPos });
          }
          // If handoff was active, seek clip's own element so it's ready for scrubbing
          if (actualVideo !== video) {
            video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
          }
          return;
        }

        if (!video.paused) {
          video.pause();
        }

        // 0.04s ≈ slightly more than 1 frame at 30fps.
        // Previous 0.1s threshold skipped up to 3 frames during slow scrubbing.
        const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.04;
        if (timeDiff > seekThreshold) {
          this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
        }

        // Force decode if readyState dropped after seek
        if (video.readyState < 2 && !video.seeking) {
          vfPipelineMonitor.record('vf_readystate_drop', {
            clipId: clip.id,
            readyState: video.readyState,
          });
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
        video.fastSeek(this.safeSeekTime(video, time));
        vfPipelineMonitor.record('vf_seek_fast', {
          clipId,
          target: Math.round(time * 1000) / 1000,
        });

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
            video.currentTime = this.safeSeekTime(video, target);
            vfPipelineMonitor.record('vf_seek_precise', {
              clipId,
              target: Math.round(target * 1000) / 1000,
              deferred: 'true',
            });
            // Register RVFC for when the precise frame arrives
            this.registerRVFC(clipId, video);
          }
        }, 120);
      } else {
        // Not dragging: precise seek immediately (click, arrow keys, etc.)
        video.currentTime = this.safeSeekTime(video, time);
        vfPipelineMonitor.record('vf_seek_precise', {
          clipId,
          target: Math.round(time * 1000) / 1000,
        });
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
        vfPipelineMonitor.record('vf_seek_done', { clipId });
        // Bypass the scrub rate limiter — a fresh decoded frame should be displayed immediately
        engine.requestNewFrameRender();
      });
    }
  }

  // --- Proactive GPU Warmup ---

  // Videos whose GPU surface has been confirmed active via RVFC
  private gpuWarmedUp = new WeakSet<HTMLVideoElement>();
  private static readonly LOOKAHEAD_TIME = 0.5; // seconds

  /**
   * Warm up video elements for clips that will become active within LOOKAHEAD_TIME.
   * Each split clip has its own HTMLVideoElement with a cold GPU surface.
   * Without proactive warmup, crossing a cut boundary causes a black frame
   * while the GPU decoder activates (~100-500ms stutter).
   *
   * Note: useVideoPreload.ts also does lookahead (2s) and calls play()/pause(50ms),
   * but 50ms doesn't guarantee GPU surface activation. This method uses
   * requestVideoFrameCallback to confirm actual frame presentation.
   */
  private warmupUpcomingClips(ctx: FrameContext): void {
    const lookaheadEnd = ctx.playheadPosition + VideoSyncManager.LOOKAHEAD_TIME;

    for (const clip of ctx.clips) {
      if (!clip.source?.videoElement) continue;

      const video = clip.source.videoElement;
      const clipStart = clip.startTime;

      // Is this clip about to become active? (starts within lookahead window, not yet active)
      if (clipStart <= ctx.playheadPosition || clipStart > lookaheadEnd) continue;

      // Skip if GPU already confirmed warm, or warmup in progress
      if (this.gpuWarmedUp.has(video) || this.warmingUpVideos.has(video)) continue;

      // Skip if no source loaded
      if (!video.src && !video.currentSrc) continue;

      // Cooldown check
      const warmupCooldown = this.warmupRetryCooldown.get(video);
      if (warmupCooldown && performance.now() - warmupCooldown < 2000) continue;

      // Start proactive warmup: briefly play to activate GPU surface
      this.warmingUpVideos.add(video);
      const clipTime = clip.inPoint;

      video.muted = true; // Prevent audio blip during warmup
      video.play().then(() => {
        const rvfc = (video as any).requestVideoFrameCallback;
        if (typeof rvfc === 'function') {
          rvfc.call(video, () => {
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = this.safeSeekTime(video, clipTime);
            this.warmingUpVideos.delete(video);
            this.gpuWarmedUp.add(video);
            vfPipelineMonitor.record('vf_gpu_ready', { clipId: clip.id, proactive: 'true' });
          });
        } else {
          setTimeout(() => {
            engine.ensureVideoFrameCached(video);
            video.pause();
            video.currentTime = this.safeSeekTime(video, clipTime);
            this.warmingUpVideos.delete(video);
            this.gpuWarmedUp.add(video);
            vfPipelineMonitor.record('vf_gpu_ready', { clipId: clip.id, proactive: 'true', fallback: 'true' });
          }, 50);
        }
      }).catch(() => {
        this.warmingUpVideos.delete(video);
        this.warmupRetryCooldown.set(video, performance.now());
      });
    }
  }

  /**
   * Update per-track state after syncing (for cut transition detection next frame)
   */
  private updateLastTrackState(ctx: FrameContext): void {
    for (const clip of ctx.clipsAtTime) {
      if (!clip.source?.videoElement || !clip.trackId) continue;

      // Use the actual playing element (handoff or clip's own)
      const handoffElement = this.activeHandoffs.get(clip.id);
      const video = handoffElement ?? clip.source.videoElement;
      const videoSrc = video.src || video.currentSrc;
      if (!videoSrc) continue;

      this.lastTrackState.set(clip.trackId, {
        clipId: clip.id,
        videoSrc,
        videoElement: video,
        outPoint: clip.outPoint,
      });
    }
  }

  // --- Health Monitor Accessors ---

  getActiveRvfcClipIds(): string[] {
    return Object.keys(this.rvfcHandles);
  }

  getActivePreciseSeekClipIds(): string[] {
    return Object.keys(this.preciseSeekTimers);
  }

  getForceDecodeClipIds(): string[] {
    return [...this.forceDecodeInProgress];
  }

  isVideoWarmingUp(video: HTMLVideoElement): boolean {
    return this.warmingUpVideos.has(video);
  }

  cancelRvfcHandle(clipId: string, video?: HTMLVideoElement): void {
    const handle = this.rvfcHandles[clipId];
    if (handle !== undefined) {
      if (video) (video as any).cancelVideoFrameCallback?.(handle);
      delete this.rvfcHandles[clipId];
    }
  }

  clearWarmupState(video: HTMLVideoElement): void {
    this.warmingUpVideos.delete(video);
  }

  /**
   * Sync full-mode WebCodecs player.
   * The WebCodecsPlayer handles its own frame decoding via MP4Box + VideoDecoder.
   * The HTMLVideoElement is kept for audio playback.
   */
  private syncFullWebCodecs(clip: TimelineClip, ctx: FrameContext): void {
    const video = clip.source!.videoElement!;
    const wcp = clip.source!.webCodecsPlayer!;
    const timeInfo = getClipTimeInfo(ctx, clip);

    if (ctx.isPlaying) {
      // Start WebCodecs decode loop and video (for audio)
      if (!wcp.isPlaying) wcp.play();
      if (video.paused) video.play().catch(() => {});

      // Drift correction: if WebCodecs time drifts from expected, seek
      const wcDrift = Math.abs(wcp.currentTime - timeInfo.clipTime);
      if (wcDrift > 0.3) {
        wcPipelineMonitor.record('drift_correct', {
          drift: Math.round(wcDrift * 1000) / 1000,
          target: Math.round(timeInfo.clipTime * 1000) / 1000,
        });
        wcp.seek(timeInfo.clipTime);
      }
      // Keep audio in sync
      const audioDrift = Math.abs(video.currentTime - timeInfo.clipTime);
      if (audioDrift > 0.3) {
        video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
      }
    } else {
      // Paused: stop decode loop, seek to frame
      if (wcp.isPlaying) wcp.pause();
      if (!video.paused) video.pause();

      const wcTimeDiff = Math.abs(wcp.currentTime - timeInfo.clipTime);
      if (wcTimeDiff > 0.05) {
        if (ctx.isDraggingPlayhead) {
          // Fast scrubbing: keyframe-only for instant feedback
          wcp.fastSeek(timeInfo.clipTime);
          // Debounced precise seek when scrubbing pauses
          this.schedulePreciseWcSeek(clip.id, wcp, timeInfo.clipTime);
        } else {
          // Click/arrow: precise seek immediately
          wcp.seek(timeInfo.clipTime);
        }
      }
      // Keep audio element at same position
      const timeDiff = Math.abs(video.currentTime - timeInfo.clipTime);
      if (timeDiff > 0.05) {
        video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
      }
    }
  }

  /**
   * Schedule a debounced precise WebCodecs seek.
   * During fast scrubbing, fastSeek shows keyframes instantly.
   * When scrubbing pauses (120ms), do a full decode for the exact frame.
   */
  private schedulePreciseWcSeek(clipId: string, wcp: { seek: (t: number) => void; currentTime: number }, time: number): void {
    clearTimeout(this.wcPreciseSeekTimers[clipId]);
    this.wcPreciseSeekTimers[clipId] = setTimeout(() => {
      // Only seek if still at a different position
      if (Math.abs(wcp.currentTime - time) > 0.01) {
        wcp.seek(time);
        engine.requestRender();
      }
    }, 120);
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
