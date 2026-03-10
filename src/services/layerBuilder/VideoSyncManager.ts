// VideoSyncManager - Handles video element synchronization with playhead
// Extracted from LayerBuilderService for separation of concerns

import type { TimelineClip } from '../../types';
import type { FrameContext, NativeDecoderState } from './types';
import { LAYER_BUILDER_CONSTANTS } from './types';
import { createFrameContext, getClipTimeInfo, getMediaFileForClip } from './FrameContext';
import { playheadState } from './PlayheadState';
import { layerPlaybackManager } from '../layerPlaybackManager';
import { engine } from '../../engine/WebGPUEngine';
import { flags } from '../../engine/featureFlags';
import { useTimelineStore } from '../../stores/timeline';
import { MAX_NESTING_DEPTH } from '../../stores/timeline/constants';
import {
  canUseSharedPreviewRuntimeSession,
  ensureRuntimeFrameProvider,
  getPreviewRuntimeSource,
  getRuntimeFrameProvider,
  getScrubRuntimeSource,
  updateRuntimePlaybackTime,
} from '../mediaRuntime/runtimePlayback';
import { scrubSettleState } from '../scrubSettleState';
import { vfPipelineMonitor } from '../vfPipelineMonitor';
import { Logger } from '../logger';

const log = Logger.create('CutTransition');

export class VideoSyncManager {
  // Native decoder state
  private nativeDecoderState = new Map<string, NativeDecoderState>();

  // Video sync throttling
  private lastVideoSyncFrame = -1;
  private lastVideoSyncPlaying = false;
  private lastVideoSyncClipsRef: TimelineClip[] | null = null;
  private lastSeekRef: Record<string, number> = {};

  // Track per-clip playing state to detect playing→paused transitions
  private clipWasPlaying = new Set<string>();

  // Track per-clip dragging state to detect scrub-stop transitions
  private clipWasDragging = new Set<string>();

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
  private pendingSeekTargets: Record<string, number> = {};
  private pendingSeekStartedAt: Record<string, number> = {};
  private queuedSeekTargets: Record<string, number> = {};
  private seekedFlushArmed = new Set<string>();

  // WebCodecs precise seek debounce
  private wcPreciseSeekTimers: Record<string, ReturnType<typeof setTimeout>> = {};
  private latestWcPreciseSeekTargets: Record<string, number> = {};
  private lastWcFastSeekTarget: Record<string, number> = {};
  private lastWcFastSeekAt: Record<string, number> = {};
  private lastWcPreciseSeekAt: Record<string, number> = {};

  // (lastWcFastSeekTarget removed — replaced by wcp.isDecodePending() check)

  // Seamless cut transition: track last video element per track.
  // When same-source clips are sequential (split clips), the outgoing clip's
  // video element keeps playing through the cut — no pause/play gap.
  private lastTrackState = new Map<string, {
    clipId: string;
    fileId: string;       // mediaFileId — same for all split clips from same source
    file: File;           // File object — same reference for split clips
    videoElement: HTMLVideoElement;
    outPoint: number;
  }>();
  private activeHandoffs = new Map<string, HTMLVideoElement>();
  private handoffElements = new Set<HTMLVideoElement>();
  private static readonly PAUSED_PRECISE_SEEK_THRESHOLD = 0.015;

  /**
   * Reset all per-clip state. Called during composition switch to prevent
   * stale references to destroyed video elements / WebCodecsPlayers.
   */
  reset(): void {
    this.lastTrackState.clear();
    this.activeHandoffs.clear();
    this.handoffElements.clear();
    this.lastSeekRef = {};
    this.clipWasPlaying.clear();
    this.clipWasDragging.clear();
    this.forceDecodeInProgress.clear();
    this.rvfcHandles = {};
    this.latestSeekTargets = {};
    this.pendingSeekTargets = {};
    this.pendingSeekStartedAt = {};
    this.queuedSeekTargets = {};
    this.seekedFlushArmed.clear();
    this.lastWcFastSeekTarget = {};
    this.lastWcFastSeekAt = {};
    this.lastVideoSyncFrame = -1;
    this.lastVideoSyncPlaying = false;
    this.lastVideoSyncClipsRef = null;
    scrubSettleState.clear();
    // Clear debounce timers
    for (const id of Object.values(this.preciseSeekTimers)) clearTimeout(id);
    this.preciseSeekTimers = {};
    for (const id of Object.values(this.wcPreciseSeekTimers)) clearTimeout(id);
    this.wcPreciseSeekTimers = {};
    this.latestWcPreciseSeekTargets = {};
    this.lastWcPreciseSeekAt = {};
  }

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

  private maybeRecoverScrubSettle(
    clipId: string,
    video: HTMLVideoElement,
    targetTime: number
  ): void {
    const settle = scrubSettleState.get(clipId);
    if (!settle) {
      return;
    }

    if (Math.abs(settle.targetTime - targetTime) > 0.05) {
      scrubSettleState.begin(clipId, targetTime, VideoSyncManager.SCRUB_SETTLE_TIMEOUT_MS);
      return;
    }

    const lastPresentedTime = engine.getLastPresentedVideoTime(video);
    if (typeof lastPresentedTime === 'number' && Math.abs(lastPresentedTime - targetTime) <= 0.12) {
      scrubSettleState.resolve(clipId);
      return;
    }

    if (video.seeking || !scrubSettleState.isDue(clipId)) {
      return;
    }

    if (settle.stage === 'settle') {
      this.beginOrQueueSettleSeek(clipId, video, targetTime, { retry: 'true' });
      engine.requestNewFrameRender();
      scrubSettleState.markRetry(clipId, targetTime, VideoSyncManager.SCRUB_SETTLE_TIMEOUT_MS);
      return;
    }

    if (settle.stage === 'retry') {
      vfPipelineMonitor.record('vf_settle_seek', {
        clipId,
        target: Math.round(targetTime * 1000) / 1000,
        recovery: 'warmup',
      });
      this.startTargetedWarmup(clipId, video, targetTime, {
        proactive: false,
        requestRender: true,
      });
      scrubSettleState.markWarmup(clipId, targetTime, VideoSyncManager.SCRUB_SETTLE_WARMUP_MS);
      return;
    }

    if (settle.stage === 'warmup' && video.readyState >= 2 && !video.seeking) {
      scrubSettleState.resolve(clipId);
    }
  }

  private getClipStartTime(ctx: FrameContext, clip: TimelineClip): number {
    const initialSpeed = ctx.getInterpolatedSpeed(clip.id, 0);
    const startPoint = initialSpeed >= 0 ? clip.inPoint : clip.outPoint;
    let sourceTime = 0;
    try {
      sourceTime = ctx.getSourceTimeForClip(clip.id, 0);
    } catch {
      sourceTime = 0;
    }
    return Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
  }

  private getWarmupClipTime(ctx: FrameContext, clip: TimelineClip): number {
    if (!ctx.isDraggingPlayhead) {
      return this.getClipStartTime(ctx, clip);
    }

    const clipEnd = clip.startTime + clip.duration;
    const sampleTimelineTime = Math.max(
      clip.startTime,
      Math.min(Math.max(ctx.playheadPosition, clip.startTime), clipEnd - 1 / 120)
    );
    const clipLocalTime = Math.max(0, sampleTimelineTime - clip.startTime);
    const speed = ctx.getInterpolatedSpeed(clip.id, clipLocalTime);
    const startPoint = speed >= 0 ? clip.inPoint : clip.outPoint;

    let sourceTime = 0;
    try {
      sourceTime = ctx.getSourceTimeForClip(clip.id, clipLocalTime);
    } catch {
      sourceTime = 0;
    }

    return Math.max(clip.inPoint, Math.min(clip.outPoint, startPoint + sourceTime));
  }

  private beginOrQueueSettleSeek(
    clipId: string,
    video: HTMLVideoElement,
    targetTime: number,
    detail?: Record<string, string>,
    reason?: 'manual-seek' | 'scrub-stop' | 'playback-stop'
  ): void {
    scrubSettleState.begin(clipId, targetTime, VideoSyncManager.SCRUB_SETTLE_TIMEOUT_MS, reason);

    const pendingTarget = this.pendingSeekTargets[clipId];
    const hasNearPendingTarget =
      typeof pendingTarget === 'number' &&
      Math.abs(pendingTarget - targetTime) <= 0.08;

    if (video.seeking || this.rvfcHandles[clipId] !== undefined || hasNearPendingTarget) {
      this.queuedSeekTargets[clipId] = targetTime;
      this.armSeekedFlush(clipId, video);
      vfPipelineMonitor.record('vf_settle_seek', {
        clipId,
        target: Math.round(targetTime * 1000) / 1000,
        queued: 'true',
        ...detail,
      });
      return;
    }

    this.pendingSeekTargets[clipId] = targetTime;
    this.pendingSeekStartedAt[clipId] = performance.now();
    video.currentTime = this.safeSeekTime(video, targetTime);
    this.armSeekedFlush(clipId, video);
    vfPipelineMonitor.record('vf_settle_seek', {
      clipId,
      target: Math.round(targetTime * 1000) / 1000,
      ...detail,
    });
    this.registerRVFC(clipId, video);
  }

  private sharesActiveTrackDecoder(ctx: FrameContext, clip: TimelineClip): boolean {
    if (!clip.trackId) {
      return false;
    }

    const activeClip = ctx.clipsByTrackId.get(clip.trackId);
    if (!activeClip?.source || activeClip.id === clip.id) {
      return false;
    }

    if (clip.source?.runtimeSourceId && activeClip.source.runtimeSourceId) {
      return clip.source.runtimeSourceId === activeClip.source.runtimeSourceId;
    }

    return !!(
      clip.source?.webCodecsPlayer &&
      activeClip.source.webCodecsPlayer &&
      clip.source.webCodecsPlayer === activeClip.source.webCodecsPlayer
    );
  }

  private getActiveClipsAtTime(ctx: FrameContext, time: number): TimelineClip[] {
    return ctx.clips.filter((clip) => time >= clip.startTime && time < clip.startTime + clip.duration);
  }

  private prewarmUpcomingWebCodecsClip(
    ctx: FrameContext,
    clip: TimelineClip,
    clipTime: number
  ): void {
    if (!clip.source || this.sharesActiveTrackDecoder(ctx, clip)) {
      return;
    }

    const futureActiveClips = this.getActiveClipsAtTime(
      ctx,
      Math.min(clip.startTime + 0.001, clip.startTime + Math.max(clip.duration * 0.25, 0.001))
    );
    const previewSource = getPreviewRuntimeSource(
      clip.source,
      clip.trackId,
      canUseSharedPreviewRuntimeSession(clip, futureActiveClips)
    );

    updateRuntimePlaybackTime(previewSource, clipTime);

    const runtimeProvider = getRuntimeFrameProvider(previewSource);
    const frameProvider =
      runtimeProvider?.isFullMode()
        ? runtimeProvider
        : clip.source.webCodecsPlayer?.isFullMode()
          ? clip.source.webCodecsPlayer
          : null;

    if (!frameProvider) {
      return;
    }

    const pendingTarget = frameProvider.getPendingSeekTime?.();
    const effectiveTime = pendingTarget ?? frameProvider.currentTime;
    const hasFrame = frameProvider.hasFrame?.() ?? true;
    if (pendingTarget != null && Math.abs(pendingTarget - clipTime) <= 0.05) {
      return;
    }
    if (hasFrame && Math.abs(effectiveTime - clipTime) <= 0.05) {
      return;
    }

    frameProvider.seek(clipTime);
  }

  private getPlaybackRuntimeSourceForClip(ctx: FrameContext, clip: TimelineClip) {
    return getPreviewRuntimeSource(
      clip.source,
      clip.trackId,
      canUseSharedPreviewRuntimeSession(clip, ctx.clipsAtTime)
    );
  }

  private getScrubRuntimeSourceForClip(ctx: FrameContext, clip: TimelineClip) {
    return getScrubRuntimeSource(
      clip.source,
      clip.trackId,
      canUseSharedPreviewRuntimeSession(clip, ctx.clipsAtTime)
    );
  }

  private providerHasFrame(
    provider:
      | {
        hasFrame?: () => boolean;
        getCurrentFrame?: () => unknown;
      }
      | null
      | undefined
  ): boolean {
    if (!provider) {
      return false;
    }

    return (provider.hasFrame?.() ?? false) || !!provider.getCurrentFrame?.();
  }

  private getPausedWebCodecsProvider(
    source: TimelineClip['source'],
    runtimeProvider: ReturnType<typeof getRuntimeFrameProvider>,
    targetTime: number,
    options?: { preferFreshRuntime?: boolean }
  ) {
    const preferFreshRuntime = options?.preferFreshRuntime === true;
    const freshFrameTolerance = 0.12;
    const providerDistance = (
      provider:
        | {
          currentTime: number;
          getPendingSeekTime?: () => number | null | undefined;
          hasFrame?: () => boolean;
          getCurrentFrame?: () => unknown;
          isFullMode?: () => boolean;
        }
        | null
        | undefined
    ): number => {
      if (!provider?.isFullMode?.()) {
        return Number.POSITIVE_INFINITY;
      }
      if (!this.providerHasFrame(provider)) {
        return Number.POSITIVE_INFINITY;
      }
      const effectiveTime = provider.getPendingSeekTime?.() ?? provider.currentTime;
      return Number.isFinite(effectiveTime)
        ? Math.abs(effectiveTime - targetTime)
        : Number.POSITIVE_INFINITY;
    };

    const clipPlayer = source?.webCodecsPlayer?.isFullMode()
      ? source.webCodecsPlayer
      : null;
    const runtimeIsFullMode = !!runtimeProvider?.isFullMode();
    const runtimeHasFrame = this.providerHasFrame(runtimeProvider);
    const runtimeEffectiveTime = runtimeProvider?.getPendingSeekTime?.() ?? runtimeProvider?.currentTime;

    if (
      runtimeIsFullMode &&
      runtimeHasFrame &&
      runtimeEffectiveTime !== undefined &&
      Math.abs(runtimeEffectiveTime - targetTime) <= 0.05
    ) {
      return runtimeProvider;
    }

    const clipHasFrame = this.providerHasFrame(clipPlayer);
    const runtimeDistance = providerDistance(runtimeProvider);
    const clipDistance = providerDistance(clipPlayer);

    if (!clipPlayer) {
      return runtimeHasFrame && runtimeIsFullMode ? runtimeProvider : null;
    }

    if (preferFreshRuntime && runtimeIsFullMode) {
      const runtimeIsFresh = runtimeHasFrame && runtimeDistance <= freshFrameTolerance;
      const clipIsFresh = clipHasFrame && clipDistance <= freshFrameTolerance;

      if (runtimeIsFresh && runtimeDistance <= clipDistance) {
        return runtimeProvider;
      }
      if (clipIsFresh) {
        return clipPlayer;
      }
      return runtimeProvider;
    }

    if (runtimeHasFrame && runtimeDistance < clipDistance) {
      return runtimeProvider;
    }

    if (clipHasFrame) {
      return clipPlayer;
    }

    if (runtimeHasFrame && runtimeIsFullMode) {
      return runtimeProvider;
    }

    return clipPlayer;
  }

  private hasPendingDuplicateSeek(
    clipId: string,
    video: HTMLVideoElement,
    targetTime: number
  ): boolean {
    const pendingTarget = this.pendingSeekTargets[clipId];
    if (pendingTarget === undefined || Math.abs(pendingTarget - targetTime) > 0.01) {
      return false;
    }

    return (
      video.seeking ||
      this.rvfcHandles[clipId] !== undefined ||
      this.preciseSeekTimers[clipId] !== undefined
    );
  }

  private shouldRetargetPendingSeek(
    clipId: string,
    nextTargetTime: number,
    now: number,
    isDragging: boolean,
    allowInFlightRetarget: boolean,
    displayedDriftSeconds: number = 0
  ): boolean {
    const pendingTarget = this.pendingSeekTargets[clipId];
    if (pendingTarget === undefined) {
      return false;
    }

    const pendingAge = now - (this.pendingSeekStartedAt[clipId] ?? now);
    const targetDrift = Math.abs(pendingTarget - nextTargetTime);
    if (isDragging && !allowInFlightRetarget) {
      if (displayedDriftSeconds >= 1.2) {
        return pendingAge >= 65 && targetDrift >= 0.12;
      }
      if (displayedDriftSeconds >= 0.5) {
        return pendingAge >= 95 && targetDrift >= 0.16;
      }
      return pendingAge >= 170 && targetDrift >= 0.28;
    }

    return pendingAge >= (isDragging ? 90 : 120) && targetDrift >= (isDragging ? 0.12 : 0.2);
  }

  private flushQueuedSeekTarget(
    clipId: string,
    video: HTMLVideoElement,
    source: 'seeked' | 'rvfc'
  ): void {
    const queuedTarget = this.queuedSeekTargets[clipId];
    if (queuedTarget === undefined) {
      return;
    }

    delete this.queuedSeekTargets[clipId];
    if (Math.abs(video.currentTime - queuedTarget) <= 0.01 && !video.seeking) {
      delete this.pendingSeekTargets[clipId];
      delete this.pendingSeekStartedAt[clipId];
      return;
    }

    const isDragging = useTimelineStore.getState().isDraggingPlayhead;
    const supportsFastSeek = typeof (video as HTMLVideoElement & {
      fastSeek?: (time: number) => void;
    }).fastSeek === 'function';
    const presentedTime = engine.getLastPresentedVideoTime(video);
    const effectiveTime = typeof presentedTime === 'number' ? presentedTime : video.currentTime;
    const targetDrift = Math.abs(effectiveTime - queuedTarget);
    const settle = scrubSettleState.get(clipId);

    if (isDragging && !supportsFastSeek && source === 'rvfc') {
      if (targetDrift <= 0.04) {
        this.latestSeekTargets[clipId] = queuedTarget;
        this.lastSeekRef[clipId] = performance.now();
        engine.requestNewFrameRender();
        return;
      }

      if (targetDrift <= VideoSyncManager.SCRUB_DRAG_RVFC_FOLLOW_THRESHOLD) {
        this.pendingSeekTargets[clipId] = queuedTarget;
        this.pendingSeekStartedAt[clipId] = performance.now();
        video.currentTime = this.safeSeekTime(video, queuedTarget);
        this.armSeekedFlush(clipId, video);
        vfPipelineMonitor.record('vf_seek_precise', {
          clipId,
          target: Math.round(queuedTarget * 1000) / 1000,
          coalesced: source,
          followup: 'drag-rvfc',
        });
        this.registerRVFC(clipId, video);
        engine.requestNewFrameRender();
        return;
      }

      this.latestSeekTargets[clipId] = queuedTarget;
      this.lastSeekRef[clipId] = performance.now();
      vfPipelineMonitor.record('vf_settle_seek', {
        clipId,
        target: Math.round(queuedTarget * 1000) / 1000,
        deferred: 'drag-rvfc',
        driftMs: Math.round(targetDrift * 1000),
      });
      engine.requestNewFrameRender();
      return;
    }

    if (!isDragging && source === 'rvfc') {
      if (targetDrift <= 0.08) {
        scrubSettleState.resolve(clipId);
        vfPipelineMonitor.record('vf_settle_seek', {
          clipId,
          target: Math.round(queuedTarget * 1000) / 1000,
          satisfied: 'rvfc',
          driftMs: Math.round(targetDrift * 1000),
        });
        engine.requestNewFrameRender();
        return;
      }

      if (settle?.stage === 'settle' && targetDrift <= 0.35) {
        scrubSettleState.begin(
          clipId,
          queuedTarget,
          VideoSyncManager.SCRUB_SETTLE_RVFC_DEFER_MS
        );
        vfPipelineMonitor.record('vf_settle_seek', {
          clipId,
          target: Math.round(queuedTarget * 1000) / 1000,
          deferred: 'rvfc',
          driftMs: Math.round(targetDrift * 1000),
        });
        engine.requestNewFrameRender();
        return;
      }
    }

    this.pendingSeekTargets[clipId] = queuedTarget;
    this.pendingSeekStartedAt[clipId] = performance.now();
    if (isDragging && supportsFastSeek) {
      this.latestSeekTargets[clipId] = queuedTarget;
      video.fastSeek(this.safeSeekTime(video, queuedTarget));
      this.armSeekedFlush(clipId, video);
      vfPipelineMonitor.record('vf_seek_fast', {
        clipId,
        target: Math.round(queuedTarget * 1000) / 1000,
        coalesced: source,
      });

      clearTimeout(this.preciseSeekTimers[clipId]);
      this.preciseSeekTimers[clipId] = setTimeout(() => {
        const target = this.latestSeekTargets[clipId];
        if (target !== undefined && Math.abs(video.currentTime - target) > 0.01) {
          this.pendingSeekTargets[clipId] = target;
          this.pendingSeekStartedAt[clipId] = performance.now();
          video.currentTime = this.safeSeekTime(video, target);
          this.armSeekedFlush(clipId, video);
          vfPipelineMonitor.record('vf_seek_precise', {
            clipId,
            target: Math.round(target * 1000) / 1000,
            deferred: 'true',
            coalesced: source,
          });
          this.registerRVFC(clipId, video);
        }
      }, 90);
    } else {
      video.currentTime = this.safeSeekTime(video, queuedTarget);
      this.armSeekedFlush(clipId, video);
      vfPipelineMonitor.record('vf_seek_precise', {
        clipId,
        target: Math.round(queuedTarget * 1000) / 1000,
        coalesced: source,
      });
      this.registerRVFC(clipId, video);
    }

    engine.requestNewFrameRender();
  }

  private armSeekedFlush(clipId: string, video: HTMLVideoElement): void {
    if (this.seekedFlushArmed.has(clipId)) {
      return;
    }

    this.seekedFlushArmed.add(clipId);
    video.addEventListener('seeked', () => {
      this.seekedFlushArmed.delete(clipId);
      this.flushQueuedSeekTarget(clipId, video, 'seeked');
    }, { once: true });
  }

  private shouldSeekPausedWebCodecsProvider(
    provider:
      | {
        currentTime: number;
        getPendingSeekTime?: () => number | null | undefined;
        isDecodePending?: () => boolean;
        hasFrame?: () => boolean;
        getCurrentFrame?: () => unknown;
      }
      | null
      | undefined,
    targetTime: number
  ): boolean {
    if (!provider) {
      return false;
    }

    const pendingSeek = provider.getPendingSeekTime?.();
    if (pendingSeek != null && Math.abs(pendingSeek - targetTime) <= 0.05) {
      return false;
    }

    if (provider.isDecodePending?.()) {
      return false;
    }

    const effectivePos = pendingSeek ?? provider.currentTime;
    return !this.providerHasFrame(provider) || Math.abs(effectivePos - targetTime) > 0.05;
  }

  private shouldFastSeekPausedWebCodecsProvider(
    provider:
      | {
        currentTime: number;
        getPendingSeekTime?: () => number | null | undefined;
        isDecodePending?: () => boolean;
        hasFrame?: () => boolean;
        getCurrentFrame?: () => unknown;
      }
      | null
      | undefined,
    providerKey: string,
    targetTime: number
  ): boolean {
    if (!provider) {
      return false;
    }

    const decodeBusy = provider.isDecodePending?.() ?? false;
    const hasFrame = this.providerHasFrame(provider);
    const effectivePos = provider.getPendingSeekTime?.() ?? provider.currentTime;
    const posDiff = Math.abs(effectivePos - targetTime);
    const lastFastSeekTarget = this.lastWcFastSeekTarget[providerKey];
    const targetMovedSinceFastSeek =
      lastFastSeekTarget === undefined ||
      Math.abs(lastFastSeekTarget - targetTime) > 0.01;
    const staleBusySeek =
      decodeBusy &&
      targetMovedSinceFastSeek &&
      performance.now() - (this.lastWcFastSeekAt[providerKey] ?? 0) > 80;

    return (
      (!decodeBusy || staleBusySeek) &&
      (!hasFrame || posDiff > 0.05 || targetMovedSinceFastSeek)
    );
  }

  private shouldUseSequentialScrubSeek(
    provider:
      | {
        currentTime: number;
        getPendingSeekTime?: () => number | null | undefined;
        isDecodePending?: () => boolean;
        hasFrame?: () => boolean;
        getCurrentFrame?: () => unknown;
      }
      | null
      | undefined,
    targetTime: number
  ): boolean {
    if (!provider || !this.providerHasFrame(provider)) {
      return false;
    }

    if (provider.isDecodePending?.()) {
      return false;
    }

    const effectivePos = provider.getPendingSeekTime?.() ?? provider.currentTime;
    const delta = targetTime - effectivePos;

    return delta > 0.01 && delta <= 0.35;
  }

  private clearFastSeekTracking(providerKey: string): void {
    delete this.lastWcFastSeekTarget[providerKey];
    delete this.lastWcFastSeekAt[providerKey];
  }

  private isPlaybackProviderReadyForAudioStart(
    provider:
      | {
        currentTime: number;
        getPendingSeekTime?: () => number | null | undefined;
        hasFrame?: () => boolean;
        hasBufferedFutureFrame?: (minFrameDelta?: number) => boolean;
        getCurrentFrame?: () => unknown;
        isAdvanceSeekPending?: () => boolean;
      }
      | null
      | undefined,
    targetTime: number
  ): boolean {
    if (!provider) {
      return false;
    }

    if (!this.providerHasFrame(provider)) {
      return false;
    }

    // Wait for advance seek to resolve — the decoder was just reset and
    // currentFrame is stale from the previous session, not yet replaced.
    if (provider.isAdvanceSeekPending?.()) {
      return false;
    }

    const effectiveTime = provider.getPendingSeekTime?.() ?? provider.currentTime;
    return (
      Math.abs(effectiveTime - targetTime) <= 0.05 &&
      (provider.hasBufferedFutureFrame?.(0.5) ?? true)
    );
  }

  private syncPausedWebCodecsProvider(
    provider:
      | {
        currentTime: number;
        seek: (time: number) => void;
        fastSeek?: (time: number) => void;
        isPlaying?: boolean;
        pause?: () => void;
        getPendingSeekTime?: () => number | null | undefined;
        isDecodePending?: () => boolean;
        hasFrame?: () => boolean;
        getCurrentFrame?: () => unknown;
      }
      | null
      | undefined,
    providerKey: string,
    targetTime: number,
    isDragging: boolean,
    schedulePreciseSeek = false,
    allowSequentialDuringDrag = true
  ): void {
    if (!provider) {
      return;
    }

    if (provider.isPlaying) {
      provider.pause?.();
    }

    if (isDragging) {
      if (allowSequentialDuringDrag && this.shouldUseSequentialScrubSeek(provider, targetTime)) {
        this.clearFastSeekTracking(providerKey);
        provider.seek(targetTime);
        return;
      }

      if (this.shouldFastSeekPausedWebCodecsProvider(provider, providerKey, targetTime)) {
        provider.fastSeek?.(targetTime);
        this.lastWcFastSeekTarget[providerKey] = targetTime;
        this.lastWcFastSeekAt[providerKey] = performance.now();
        if (schedulePreciseSeek) {
          this.schedulePreciseWcSeek(providerKey, provider, targetTime);
        }
      }
      return;
    }

    this.clearFastSeekTracking(providerKey);
    if (this.shouldSeekPausedWebCodecsProvider(provider, targetTime)) {
      provider.seek(targetTime);
    }
  }

  /**
   * Detect same-source sequential clips and set up handoffs.
   * Called from both LayerBuilderService.buildLayers() (for rendering)
   * and syncVideoElements() (for sync + pause prevention).
   */
  computeHandoffs(ctx: FrameContext): void {
    if (ctx.isDraggingPlayhead) {
      this.activeHandoffs.clear();
      this.handoffElements.clear();
      return;
    }

    if (!ctx.isPlaying) {
      for (const clipId of [...this.activeHandoffs.keys()]) {
        const settle = scrubSettleState.get(clipId);
        const keepHandoff =
          settle?.reason === 'playback-stop' &&
          scrubSettleState.isPending(clipId);
        if (!keepHandoff) {
          this.activeHandoffs.delete(clipId);
        }
      }

      this.handoffElements.clear();
      for (const handoff of this.activeHandoffs.values()) {
        this.handoffElements.add(handoff);
      }
      return;
    }

    this.activeHandoffs.clear();
    this.handoffElements.clear();

    for (const clip of ctx.clipsAtTime) {
      if (!clip.source?.videoElement || !clip.trackId) continue;

      const prev = this.lastTrackState.get(clip.trackId);
      if (!prev) continue;

      if (prev.clipId === clip.id) {
        // Same clip as last frame — persist handoff if we were using one.
        if (prev.videoElement !== clip.source.videoElement) {
          this.activeHandoffs.set(clip.id, prev.videoElement);
          this.handoffElements.add(prev.videoElement);
        }
        continue;
      }

      // Different clip — detect same-source sequential cut for new handoff.
      const clipFileId = clip.source.mediaFileId || clip.mediaFileId;
      const sameSource = clipFileId
        ? clipFileId === prev.fileId
        : clip.file === prev.file;

      if (!sameSource) {
        log.debug('Handoff SKIP: different source', {
          track: clip.trackId,
          prevClip: prev.clipId.slice(-6),
          newClip: clip.id.slice(-6),
          prevFileId: prev.fileId?.slice(-6),
          newFileId: clipFileId?.slice(-6),
        });
        continue;
      }

      const inOutGap = Math.abs(clip.inPoint - prev.outPoint);
      const isContinuousCut = inOutGap <= 0.1;
      if (!isContinuousCut) {
        log.debug('Handoff SKIP: non-continuous cut', {
          track: clip.trackId,
          inPoint: clip.inPoint.toFixed(3),
          prevOutPoint: prev.outPoint.toFixed(3),
          gap: inOutGap.toFixed(3),
        });
        continue;
      }

      const elemDrift = Math.abs(prev.videoElement.currentTime - clip.inPoint);
      // Continuous cut: previous element should already be near the new inPoint
      if (elemDrift > 0.5) {
        log.debug('Handoff SKIP: element too far from inPoint', {
          track: clip.trackId,
          elementTime: prev.videoElement.currentTime.toFixed(3),
          inPoint: clip.inPoint.toFixed(3),
          drift: elemDrift.toFixed(3),
        });
        continue;
      }

      log.info('Handoff START', {
        track: clip.trackId,
        prevClip: prev.clipId.slice(-6),
        newClip: clip.id.slice(-6),
        elementTime: prev.videoElement.currentTime.toFixed(3),
        inPoint: clip.inPoint.toFixed(3),
        drift: elemDrift.toFixed(3),
      });
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
        ctx.isPlaying === this.lastVideoSyncPlaying &&
        ctx.clips === this.lastVideoSyncClipsRef) {
      return;
    }
    this.lastVideoSyncFrame = ctx.frameNumber;
    this.lastVideoSyncPlaying = ctx.isPlaying;
    this.lastVideoSyncClipsRef = ctx.clips;

    // Compute handoffs for seamless cut transitions
    this.computeHandoffs(ctx);

    // Proactively warm upcoming clips before sync so boundary crossings during
    // playback or drag scrubbing are less likely to hit a cold decoder surface.
    if (ctx.isPlaying || ctx.isDraggingPlayhead) {
      this.warmupUpcomingClips(ctx);
    }
    if (ctx.isPlaying) {
      this.preBufferUpcomingVideoAudio(ctx);
      this.preBufferUpcomingNestedCompVideos(ctx);
    }

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
        if (!ctx.isPlaying && !isAtPlayhead) {
          this.clipWasPlaying.delete(clip.id);
        }
        // NOTE: Do NOT pause WebCodecsPlayer here. Split clips share the same
        // player instance, so clip1 exiting and clip2 entering use the same decoder.
        // Pausing here would reset the decoder right after clip2's advanceToTime()
        // just set it up — causing a permanent freeze. The player's advanceToTime()
        // handles all state transitions (seek, restart) automatically.
        // syncFullWebCodecs() pauses the player when ctx.isPlaying is false.
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
        scrubSettleState.resolve(nestedClip.id);
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
        scrubSettleState.resolve(nestedClip.id);
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
        if (ctx.isDraggingPlayhead) {
          scrubSettleState.resolve(nestedClip.id);
        }

        // Force first-frame decode for videos that haven't played yet (e.g. after reload)
        if (video.played.length === 0 && !video.seeking && !this.forceDecodeInProgress.has(nestedClip.id)) {
          this.forceVideoFrameDecode(nestedClip.id, video);
        }

        const seekThreshold = ctx.isDraggingPlayhead
          ? 0.1
          : VideoSyncManager.PAUSED_PRECISE_SEEK_THRESHOLD;
        if (timeDiff > seekThreshold) {
          if (!ctx.isDraggingPlayhead) {
            scrubSettleState.begin(
              nestedClip.id,
              nestedClipTime,
              VideoSyncManager.SCRUB_SETTLE_TIMEOUT_MS,
              'manual-seek'
            );
          }
          this.throttledSeek(nestedClip.id, video, nestedClipTime, ctx);
          video.addEventListener('seeked', () => engine.requestRender(), { once: true });
        }

        // If video readyState < 2 (no frame data), force decode via play/pause
        // This can happen after seeking to unbuffered regions
        if (video.readyState < 2 && !video.seeking) {
          this.forceVideoFrameDecode(nestedClip.id, video);
        }

        if (!ctx.isDraggingPlayhead) {
          this.maybeRecoverScrubSettle(nestedClip.id, video, nestedClipTime);
        }
      }

      // Sync a dedicated scrub provider when paused so playback decode state stays untouched.
      if (webCodecsPlayer?.isFullMode() && !ctx.isPlaying) {
        const scrubRuntimeSource = getScrubRuntimeSource(
          nestedClip.source,
          nestedClip.trackId,
          true
        );
        updateRuntimePlaybackTime(scrubRuntimeSource, nestedClipTime);
        void ensureRuntimeFrameProvider(scrubRuntimeSource, 'interactive', nestedClipTime);

        const scrubProvider = getRuntimeFrameProvider(scrubRuntimeSource);
        const pausedProvider = this.getPausedWebCodecsProvider(
          nestedClip.source,
          scrubProvider,
          nestedClipTime
        ) ?? webCodecsPlayer;
        if (pausedProvider?.isFullMode()) {
          if (this.shouldSeekPausedWebCodecsProvider(pausedProvider, nestedClipTime)) {
            pausedProvider.seek(nestedClipTime);
          }
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

  private startTargetedWarmup(
    clipId: string,
    video: HTMLVideoElement,
    targetTime: number,
    options?: { proactive?: boolean; requestRender?: boolean }
  ): void {
    const safeTargetTime = this.safeSeekTime(video, targetTime);
    const proactive = options?.proactive === true;
    const shouldRequestRender = options?.requestRender !== false;

    this.warmingUpVideos.add(video);
    video.muted = true;

    if (video.preload !== 'auto') {
      video.preload = 'auto';
    }

    try {
      if (Math.abs(video.currentTime - safeTargetTime) > 0.01) {
        video.currentTime = safeTargetTime;
      }
    } catch {
      // Ignore if metadata is not fully ready for seeking yet.
    }

    const finishWarmup = (fallback = false) => {
      const presentedTime = video.currentTime;
      engine.markVideoFramePresented(video, presentedTime, clipId);
      if (!engine.captureVideoFrameAtTime(video, presentedTime, clipId)) {
        engine.ensureVideoFrameCached(video, clipId);
      }
      engine.markVideoGpuReady(video);
      scrubSettleState.resolve(clipId);
      video.pause();
      this.warmingUpVideos.delete(video);
      this.gpuWarmedUp.add(video);
      vfPipelineMonitor.record('vf_gpu_ready', {
        clipId,
        ...(proactive ? { proactive: 'true' } : {}),
        ...(fallback ? { fallback: 'true' } : {}),
      });
      if (shouldRequestRender) {
        engine.requestRender();
      }
    };

    video.play().then(() => {
      const rvfc = (video as any).requestVideoFrameCallback;
      if (typeof rvfc === 'function') {
        rvfc.call(video, () => {
          finishWarmup(false);
        });
      } else {
        setTimeout(() => {
          finishWarmup(true);
        }, 100);
      }
    }).catch(() => {
      this.warmingUpVideos.delete(video);
      this.warmupRetryCooldown.set(video, performance.now());
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

    // Full-mode WebCodecs owns preview sync whenever it's enabled.
    // Drag scrubbing uses the dedicated scrub session inside syncFullWebCodecs.
    const useFullWebCodecsPreview =
      flags.useFullWebCodecsPlayback &&
      clip.source?.webCodecsPlayer?.isFullMode();

    if (useFullWebCodecsPreview) {
      this.syncFullWebCodecs(clip, ctx);
      return;
    }

    if (!clip.source?.videoElement) return;

    // Keep using the handoff element only during playback or while the clip's
    // own element is still settling onto the pause/scrub target.
    const handoffVideo = this.activeHandoffs.get(clip.id);
    const settle = scrubSettleState.get(clip.id);
    const useHandoffVideo = !!handoffVideo && (
      ctx.isPlaying ||
      (settle?.reason === 'playback-stop' && scrubSettleState.isPending(clip.id))
    );
    const video = useHandoffVideo ? handoffVideo : clip.source.videoElement;
    const timeInfo = getClipTimeInfo(ctx, clip);
    const mediaFile = getMediaFileForClip(ctx, clip);

    // Check proxy mode
    const useProxy = ctx.proxyEnabled && mediaFile?.proxyFps &&
      (mediaFile.proxyStatus === 'ready' || mediaFile.proxyStatus === 'generating');

    if (useProxy) {
      // In proxy mode: pause video
      if (!video.paused) video.pause();
      if (!video.muted) video.muted = true;
      scrubSettleState.resolve(clip.id);
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
      vfPipelineMonitor.record('vf_gpu_cold', { clipId: clip.id });
      this.startTargetedWarmup(clip.id, video, timeInfo.clipTime, {
        proactive: false,
        requestRender: true,
      });
      if (false) {
            // Frame is now presented to GPU — capture it
        }
      return; // Skip normal sync — warmup is handling video state
    }

    // Normal video sync
    const timeDiff = Math.abs(video.currentTime - timeInfo.clipTime);

    // Pre-capture: ensure scrubbing cache has a frame BEFORE seeking
    if (!video.seeking && video.readyState >= 2) {
      engine.ensureVideoFrameCached(video);
    }

    if (ctx.isPlaying || ctx.isDraggingPlayhead) {
      scrubSettleState.resolve(clip.id);
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
      if (ctx.isDraggingPlayhead) {
        this.clipWasDragging.add(clip.id);
      } else if (this.clipWasDragging.has(clip.id)) {
        // Settle-seek on scrub stop for reverse playback
        this.clipWasDragging.delete(clip.id);
        clearTimeout(this.preciseSeekTimers[clip.id]);
        delete this.preciseSeekTimers[clip.id];
        if (timeDiff > 0.001) {
          this.beginOrQueueSettleSeek(clip.id, video, timeInfo.clipTime, undefined, 'scrub-stop');
          video.addEventListener('seeked', () => engine.requestNewFrameRender(), { once: true });
        } else {
          scrubSettleState.resolve(clip.id);
        }
        return;
      }
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.02;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
      if (!ctx.isDraggingPlayhead) {
        this.maybeRecoverScrubSettle(clip.id, video, timeInfo.clipTime);
      }
    } else if (ctx.playbackSpeed !== 1) {
      // Non-standard forward transport speed (2x, 4x, etc.): seek frame-by-frame
      if (!video.paused) video.pause();
      this.clipWasPlaying.delete(clip.id);
      if (ctx.isDraggingPlayhead) {
        this.clipWasDragging.add(clip.id);
      } else if (this.clipWasDragging.has(clip.id)) {
        this.clipWasDragging.delete(clip.id);
        clearTimeout(this.preciseSeekTimers[clip.id]);
        delete this.preciseSeekTimers[clip.id];
        if (timeDiff > 0.001) {
          this.beginOrQueueSettleSeek(clip.id, video, timeInfo.clipTime, undefined, 'scrub-stop');
          video.addEventListener('seeked', () => engine.requestNewFrameRender(), { once: true });
        } else {
          scrubSettleState.resolve(clip.id);
        }
        return;
      }
      const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.03;
      if (timeDiff > seekThreshold) {
        this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
      }
      if (!ctx.isDraggingPlayhead) {
        this.maybeRecoverScrubSettle(clip.id, video, timeInfo.clipTime);
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
          scrubSettleState.resolve(clip.id);
          const clipVideo = clip.source.videoElement;
          // If handoff was active, the actual playing element differs from clip's own
          const prevTrack = this.lastTrackState.get(clip.trackId);
          const actualVideo = (prevTrack && prevTrack.videoElement !== video)
            ? prevTrack.videoElement : video;
          if (!actualVideo.paused) {
            actualVideo.pause();
            vfPipelineMonitor.record('vf_pause', { clipId: clip.id });
          }
          const pauseTargetTime = actualVideo.currentTime;
          engine.markVideoFramePresented(actualVideo, pauseTargetTime, clip.id);
          if (!engine.captureVideoFrameAtTime(actualVideo, pauseTargetTime, clip.id)) {
            engine.ensureVideoFrameCached(actualVideo, clip.id);
          }
          // Convert actualVideo.currentTime back to timeline position
          const effectiveSpeed = timeInfo.absSpeed > 0.01 ? timeInfo.absSpeed : 1;
          const videoClipTime = pauseTargetTime;
          const newPlayheadPos = clip.reversed
            ? clip.startTime + (clip.outPoint - videoClipTime) / effectiveSpeed
            : clip.startTime + (videoClipTime - clip.inPoint) / effectiveSpeed;
          const currentPlayhead = playheadState.isUsingInternalPosition
            ? playheadState.position
            : ctx.playheadPosition;
          const videoAdvanced = Math.abs(newPlayheadPos - currentPlayhead) > 0.01;
          const shouldSnapPlayheadToStopFrame =
            Math.abs(newPlayheadPos - currentPlayhead) <= VideoSyncManager.PLAYBACK_STOP_SNAP_MAX_DELTA;
          if (videoAdvanced && shouldSnapPlayheadToStopFrame) {
            playheadState.position = newPlayheadPos;
            useTimelineStore.setState({ playheadPosition: newPlayheadPos });
          }
          // If playback used a handoff element, transition back to the clip's own
          // element immediately so pause/step/scrub can't keep showing the old
          // playing element's frame.
          const handoffReleased = clipVideo !== actualVideo;
          if (handoffReleased) {
            this.activeHandoffs.set(clip.id, actualVideo);
            this.handoffElements.add(actualVideo);
            const ownVideoTimeDiff = Math.abs(clipVideo.currentTime - pauseTargetTime);
            if (ownVideoTimeDiff > 0.001 || clipVideo.readyState < 2) {
              this.beginOrQueueSettleSeek(
                clip.id,
                clipVideo,
                pauseTargetTime,
                { handoffRelease: 'true' },
                'playback-stop'
              );
            } else {
              engine.markVideoFramePresented(clipVideo, pauseTargetTime, clip.id);
              if (!engine.captureVideoFrameAtTime(clipVideo, pauseTargetTime, clip.id)) {
                engine.ensureVideoFrameCached(clipVideo, clip.id);
              }
              scrubSettleState.resolve(clip.id);
              this.activeHandoffs.delete(clip.id);
              this.handoffElements.delete(actualVideo);
            }
            engine.requestNewFrameRender();
            return;
          }
          engine.requestNewFrameRender();
          return;
        }

        if (!video.paused) {
          video.pause();
        }

        // Detect scrub-stop transition: user just released the playhead.
        // Force a precise seek to the exact position + RVFC so the correct
        // frame is displayed once the seek completes.
        const justStoppedDragging = this.clipWasDragging.has(clip.id) && !ctx.isDraggingPlayhead;
        if (justStoppedDragging) {
          this.clipWasDragging.delete(clip.id);
          // Cancel any pending deferred precise seek from the drag phase
          clearTimeout(this.preciseSeekTimers[clip.id]);
          delete this.preciseSeekTimers[clip.id];
          // Always do a precise seek to the exact playhead position,
          // bypassing throttle — this is the definitive "settle" seek.
          if (timeDiff > 0.001) {
            this.beginOrQueueSettleSeek(clip.id, video, timeInfo.clipTime, undefined, 'scrub-stop');
          } else {
            scrubSettleState.resolve(clip.id);
          }
          // Also register a seeked listener as fallback — RVFC may not fire
          // if the video frame doesn't change (same keyframe).
          video.addEventListener('seeked', () => {
            engine.requestNewFrameRender();
          }, { once: true });
        } else {
          // 0.04s ≈ slightly more than 1 frame at 30fps.
          // Previous 0.1s threshold skipped up to 3 frames during slow scrubbing.
          const seekThreshold = ctx.isDraggingPlayhead
            ? 0.04
            : VideoSyncManager.PAUSED_PRECISE_SEEK_THRESHOLD;
          if (timeDiff > seekThreshold) {
            this.throttledSeek(clip.id, video, timeInfo.clipTime, ctx);
          }
        }

        // Track dragging state for next frame
        if (ctx.isDraggingPlayhead) {
          this.clipWasDragging.add(clip.id);
        }

        // Force decode if readyState dropped after seek
        if (video.readyState < 2 && !video.seeking) {
          vfPipelineMonitor.record('vf_readystate_drop', {
            clipId: clip.id,
            readyState: video.readyState,
          });
          this.forceVideoFrameDecode(clip.id, video);
        }
        if (!ctx.isDraggingPlayhead) {
          this.maybeRecoverScrubSettle(clip.id, video, timeInfo.clipTime);
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
    const presentedTime = engine.getLastPresentedVideoTime(video);
    const effectiveDisplayedTime =
      typeof presentedTime === 'number' ? presentedTime : video.currentTime;
    const displayedDriftSeconds = Math.abs(effectiveDisplayedTime - time);

    if (this.hasPendingDuplicateSeek(clipId, video, time)) {
      if (ctx.isDraggingPlayhead) {
        this.latestSeekTargets[clipId] = time;
      }
      return;
    }

    if ((video.seeking || this.rvfcHandles[clipId] !== undefined) && this.pendingSeekTargets[clipId] !== undefined) {
      const allowInFlightRetarget = ctx.isDraggingPlayhead && 'fastSeek' in video;
      if (ctx.isDraggingPlayhead && !allowInFlightRetarget) {
        this.queuedSeekTargets[clipId] = time;
        this.latestSeekTargets[clipId] = time;
        this.armSeekedFlush(clipId, video);
        return;
      }
      if (this.shouldRetargetPendingSeek(
        clipId,
        time,
        ctx.now,
        ctx.isDraggingPlayhead,
        allowInFlightRetarget,
        displayedDriftSeconds
      )) {
        this.pendingSeekTargets[clipId] = time;
        this.pendingSeekStartedAt[clipId] = ctx.now;
        if (ctx.isDraggingPlayhead) {
          this.latestSeekTargets[clipId] = time;
        }

        if (ctx.isDraggingPlayhead && 'fastSeek' in video) {
          video.fastSeek(this.safeSeekTime(video, time));
          this.armSeekedFlush(clipId, video);
          vfPipelineMonitor.record('vf_seek_fast', {
            clipId,
            target: Math.round(time * 1000) / 1000,
            retarget: 'true',
          });

          clearTimeout(this.preciseSeekTimers[clipId]);
          this.preciseSeekTimers[clipId] = setTimeout(() => {
            const target = this.latestSeekTargets[clipId];
            if (target !== undefined && Math.abs(video.currentTime - target) > 0.01) {
              this.pendingSeekTargets[clipId] = target;
              this.pendingSeekStartedAt[clipId] = performance.now();
              video.currentTime = this.safeSeekTime(video, target);
              this.armSeekedFlush(clipId, video);
              vfPipelineMonitor.record('vf_seek_precise', {
                clipId,
                target: Math.round(target * 1000) / 1000,
                deferred: 'true',
                retarget: 'true',
              });
              this.registerRVFC(clipId, video);
            }
          }, 90);
        } else {
          video.currentTime = this.safeSeekTime(video, time);
          this.armSeekedFlush(clipId, video);
          vfPipelineMonitor.record('vf_seek_precise', {
            clipId,
            target: Math.round(time * 1000) / 1000,
            retarget: 'true',
          });
          this.registerRVFC(clipId, video);
        }

        this.lastSeekRef[clipId] = ctx.now;
        return;
      }

      this.queuedSeekTargets[clipId] = time;
      if (ctx.isDraggingPlayhead) {
        this.latestSeekTargets[clipId] = time;
      }
      this.armSeekedFlush(clipId, video);
      return;
    }

    const lastSeek = this.lastSeekRef[clipId] || 0;
    const dragDrift = Math.abs(effectiveDisplayedTime - time);
    const threshold = ctx.isDraggingPlayhead
      ? dragDrift >= 1
        ? 16
        : dragDrift >= 0.35
          ? 28
          : 50
      : 33;
    if (ctx.now - lastSeek > threshold) {
      if (ctx.isDraggingPlayhead && 'fastSeek' in video) {
        // Phase 1: Instant keyframe feedback via fastSeek.
        // For all-intra codecs this IS the exact frame. For long-GOP codecs
        // this shows the nearest keyframe — better than a stale cached frame.
        this.pendingSeekTargets[clipId] = time;
        this.pendingSeekStartedAt[clipId] = ctx.now;
        video.fastSeek(this.safeSeekTime(video, time));
        this.armSeekedFlush(clipId, video);
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
            this.pendingSeekTargets[clipId] = target;
            this.pendingSeekStartedAt[clipId] = performance.now();
            video.currentTime = this.safeSeekTime(video, target);
            this.armSeekedFlush(clipId, video);
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
        scrubSettleState.begin(clipId, time, VideoSyncManager.SCRUB_SETTLE_TIMEOUT_MS, 'manual-seek');
        this.pendingSeekTargets[clipId] = time;
        this.pendingSeekStartedAt[clipId] = ctx.now;
        video.currentTime = this.safeSeekTime(video, time);
        this.armSeekedFlush(clipId, video);
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
        const presentedTime = video.currentTime;
        delete this.rvfcHandles[clipId];
        delete this.pendingSeekTargets[clipId];
        delete this.pendingSeekStartedAt[clipId];
        engine.markVideoFramePresented(video, presentedTime, clipId);
        engine.captureVideoFrameAtTime(video, presentedTime, clipId);
        scrubSettleState.resolve(clipId);
        vfPipelineMonitor.record('vf_seek_done', { clipId });
        this.flushQueuedSeekTarget(clipId, video, 'rvfc');
        // Bypass the scrub rate limiter — a fresh decoded frame should be displayed immediately
        engine.requestNewFrameRender();
      });
    }
  }

  // --- Proactive GPU Warmup ---

  // Videos whose GPU surface has been confirmed active via RVFC
  private gpuWarmedUp = new WeakSet<HTMLVideoElement>();
  private static readonly LOOKAHEAD_TIME = 1.5; // seconds (increased from 0.5 for reliable GPU warmup)
  private static readonly SCRUB_WARMUP_LOOKAHEAD = 0.9;
  private static readonly SCRUB_WARMUP_LOOKBEHIND = 0.25;
  private static readonly PLAYBACK_STOP_SNAP_MAX_DELTA = 0.5;
  private static readonly SCRUB_SETTLE_TIMEOUT_MS = 220;
  private static readonly SCRUB_SETTLE_RVFC_DEFER_MS = 90;
  private static readonly SCRUB_DRAG_RVFC_FOLLOW_THRESHOLD = 0.16;
  private static readonly SCRUB_SETTLE_WARMUP_MS = 350;

  /**
   * Warm up video elements for clips that will become active within LOOKAHEAD_TIME.
   * Each split clip has its own HTMLVideoElement with a cold GPU surface.
   * Without proactive warmup, crossing a cut boundary causes a black frame
   * while the GPU decoder activates (~100-500ms stutter).
   *
   * This is the single HTML-video warmup path for upcoming clips.
   * It uses requestVideoFrameCallback to confirm actual frame presentation
   * instead of relying on blind pre-seeks.
   */
  private warmupUpcomingClips(ctx: FrameContext): void {
    const windowStart = ctx.isDraggingPlayhead
      ? Math.max(0, ctx.playheadPosition - VideoSyncManager.SCRUB_WARMUP_LOOKBEHIND)
      : ctx.playheadPosition;
    const windowEnd = ctx.playheadPosition + (
      ctx.isDraggingPlayhead
        ? VideoSyncManager.SCRUB_WARMUP_LOOKAHEAD
        : VideoSyncManager.LOOKAHEAD_TIME
    );

    for (const clip of ctx.clips) {
      const clipStart = clip.startTime;
      const clipEnd = clip.startTime + clip.duration;
      const clipTime = this.getWarmupClipTime(ctx, clip);
      const isCurrentlyActive = clipStart <= ctx.playheadPosition && clipEnd > ctx.playheadPosition;

      if (ctx.isDraggingPlayhead) {
        // While dragging, warm clips near the playhead on both sides so boundary
        // crossings do not cold-start the GPU surface.
        if (isCurrentlyActive || clipEnd <= windowStart || clipStart > windowEnd) continue;
      } else {
        // During playback, only warm clips that start soon ahead of the playhead.
        if (clipStart <= ctx.playheadPosition || clipStart > windowEnd) continue;
      }

      if (flags.useFullWebCodecsPlayback) {
        this.prewarmUpcomingWebCodecsClip(ctx, clip, clipTime);
      }

      if (!clip.source?.videoElement) continue;

      const video = clip.source.videoElement;

      // Skip if GPU already confirmed warm, or warmup in progress
      if (this.gpuWarmedUp.has(video) || this.warmingUpVideos.has(video)) continue;

      // Skip if no source loaded
      if (!video.src && !video.currentSrc) continue;

      // Cooldown check
      const warmupCooldown = this.warmupRetryCooldown.get(video);
      if (warmupCooldown && performance.now() - warmupCooldown < 2000) continue;

      this.startTargetedWarmup(clip.id, video, clipTime, {
        proactive: true,
        requestRender: false,
      });
    }
  }

  /**
   * Pre-buffer video elements (for audio) for upcoming clips.
   * In full WebCodecs mode, the HTMLVideoElement provides audio.
   * Without pre-buffering, the element is cold at cut points → audio gap.
   */
  private preBufferUpcomingVideoAudio(ctx: FrameContext): void {
    if (!ctx.isPlaying || ctx.isDraggingPlayhead) return;

    const lookaheadEnd = ctx.playheadPosition + VideoSyncManager.LOOKAHEAD_TIME;

    for (const clip of ctx.clips) {
      if (!clip.source?.videoElement) continue;

      const video = clip.source.videoElement;
      const clipStart = clip.startTime;
      const targetTime = this.getClipStartTime(ctx, clip);

      // Is this clip about to become active? (starts within lookahead, not yet active)
      if (clipStart <= ctx.playheadPosition || clipStart > lookaheadEnd) continue;

      // Skip videos already warmed up or warming up (handled by warmupUpcomingClips)
      if (this.gpuWarmedUp.has(video) || this.warmingUpVideos.has(video)) continue;

      if (video.preload !== 'auto') {
        video.preload = 'auto';
      }

      // Pre-seek the video element to inPoint so audio data is buffered
      if (Math.abs(video.currentTime - targetTime) > 0.5) {
        video.currentTime = this.safeSeekTime(video, targetTime);
      }
    }
  }

  /**
   * Pre-buffer the nested clip that will be visible when an upcoming composition clip starts.
   * This preserves the old nested preload behavior, but keeps it in the same module as the
   * main HTML video warmup/prebuffer logic instead of a separate timeline hook.
   */
  private preBufferUpcomingNestedCompVideos(ctx: FrameContext): void {
    if (!ctx.isPlaying || ctx.isDraggingPlayhead) return;

    const lookaheadEnd = ctx.playheadPosition + VideoSyncManager.LOOKAHEAD_TIME;

    for (const compClip of ctx.clips) {
      const clipStart = compClip.startTime;
      if (
        !compClip.isComposition ||
        !compClip.nestedClips ||
        compClip.nestedClips.length === 0 ||
        clipStart <= ctx.playheadPosition ||
        clipStart > lookaheadEnd
      ) {
        continue;
      }

      const compStartTime = compClip.inPoint;
      for (const nestedClip of compClip.nestedClips) {
        const video = nestedClip.source?.videoElement;
        if (!video) continue;

        const nestedClipEnd = nestedClip.startTime + nestedClip.duration;
        if (compStartTime < nestedClip.startTime || compStartTime >= nestedClipEnd) {
          continue;
        }

        const nestedLocalTime = compStartTime - nestedClip.startTime;
        const targetTime = nestedClip.reversed
          ? nestedClip.outPoint - nestedLocalTime
          : nestedLocalTime + nestedClip.inPoint;

        if (this.warmingUpVideos.has(video) || this.gpuWarmedUp.has(video) || video.seeking) {
          continue;
        }

        if (video.preload !== 'auto') {
          video.preload = 'auto';
        }

        if (Math.abs(video.currentTime - targetTime) > 0.1) {
          video.currentTime = this.safeSeekTime(video, targetTime);
        }
      }
    }
  }

  /**
   * Update per-track state after syncing (for cut transition detection next frame)
   */
  private updateLastTrackState(ctx: FrameContext): void {
    for (const clip of ctx.clipsAtTime) {
      if (!clip.source?.videoElement || !clip.trackId) continue;

      const handoffElement = this.activeHandoffs.get(clip.id);
      // Only propagate handoff elements during real playback. While paused,
      // storing the bridged element here can poison later manual seeks/steps
      // with a stale frame from an earlier clip.
      const video =
        ctx.isPlaying && handoffElement
          ? handoffElement
          : clip.source.videoElement;

      const fileId = clip.source.mediaFileId || clip.mediaFileId || '';

      this.lastTrackState.set(clip.trackId, {
        clipId: clip.id,
        fileId,
        file: clip.file,
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
    const video = clip.source!.videoElement; // May be undefined (fast-attach before canplaythrough)
    const timeInfo = getClipTimeInfo(ctx, clip);
    const playbackRuntimeSource = this.getPlaybackRuntimeSourceForClip(ctx, clip);
    const scrubRuntimeSource = this.getScrubRuntimeSourceForClip(ctx, clip);

    // Use handoff video element for audio continuity at cut points.
    // In full WebCodecs mode, the HTMLVideoElement is only used for audio.
    // At a same-source cut, the previous clip's element is already playing
    // at the right position — reuse it instead of cold-starting clip B's element.
    const handoffVideo = this.activeHandoffs.get(clip.id);
    const audioVideo = handoffVideo ?? video;

    if (ctx.isPlaying) {
      updateRuntimePlaybackTime(playbackRuntimeSource, timeInfo.clipTime);
      const playbackProvider =
        getRuntimeFrameProvider(playbackRuntimeSource) ??
        clip.source!.webCodecsPlayer!;
      if (!playbackProvider?.isFullMode()) {
        return;
      }

      // Playback takes over — no scrub tracking needed

      // Render-loop-driven: advance decoder to clip time each frame.
      // No internal animation loop — advanceToTime handles decode feeding + frame selection.
      playbackProvider.advanceToTime?.(timeInfo.clipTime);

      // Keep video element in sync for audio (if available)
      // Use handoff element for seamless audio across cuts
      if (audioVideo) {
        const playbackReadyForAudio = this.isPlaybackProviderReadyForAudioStart(
          playbackProvider,
          timeInfo.clipTime
        );
        if (audioVideo.paused && playbackReadyForAudio) {
          log.info('Audio element PLAY', {
            clip: clip.id.slice(-6),
            isHandoff: !!handoffVideo,
            time: audioVideo.currentTime.toFixed(3),
            target: timeInfo.clipTime.toFixed(3),
          });
          audioVideo.play().catch(() => {});
        }
        const audioDrift = Math.abs(audioVideo.currentTime - timeInfo.clipTime);
        if (audioDrift > 0.3) {
          log.warn('Audio drift SEEK', {
            clip: clip.id.slice(-6),
            isHandoff: !!handoffVideo,
            elementTime: audioVideo.currentTime.toFixed(3),
            target: timeInfo.clipTime.toFixed(3),
            drift: audioDrift.toFixed(3),
          });
          audioVideo.currentTime = this.safeSeekTime(audioVideo, timeInfo.clipTime);
        }
      }
    } else {
      // Detect scrub-stop transition for WebCodecs path
      const justStoppedDraggingWc = this.clipWasDragging.has(clip.id) && !ctx.isDraggingPlayhead;
      if (ctx.isDraggingPlayhead) {
        this.clipWasDragging.add(clip.id);
      } else if (justStoppedDraggingWc) {
        this.clipWasDragging.delete(clip.id);
        // Cancel pending deferred precise seeks from drag phase
        clearTimeout(this.wcPreciseSeekTimers[`${clip.id}:scrub`]);
        delete this.wcPreciseSeekTimers[`${clip.id}:scrub`];
        clearTimeout(this.wcPreciseSeekTimers[`${clip.id}:fallback`]);
        delete this.wcPreciseSeekTimers[`${clip.id}:fallback`];
      }

      const useDedicatedScrubProvider = ctx.isDraggingPlayhead;
      const pausedRuntimeSource = useDedicatedScrubProvider
        ? scrubRuntimeSource
        : playbackRuntimeSource;

      updateRuntimePlaybackTime(pausedRuntimeSource, timeInfo.clipTime);
      if (useDedicatedScrubProvider) {
        void ensureRuntimeFrameProvider(scrubRuntimeSource, 'interactive', timeInfo.clipTime);
      }

      const pausedRuntimeProvider = getRuntimeFrameProvider(pausedRuntimeSource);
      const dedicatedScrubProvider =
        useDedicatedScrubProvider && pausedRuntimeProvider?.isFullMode()
          ? pausedRuntimeProvider
          : null;
      const pausedProvider = this.getPausedWebCodecsProvider(
        clip.source,
        pausedRuntimeProvider,
        timeInfo.clipTime,
        { preferFreshRuntime: useDedicatedScrubProvider }
      );
      const fallbackProvider =
        dedicatedScrubProvider && pausedProvider && pausedProvider !== dedicatedScrubProvider
          ? pausedProvider
          : null;
      const scrubProviderReady = this.isPlaybackProviderReadyForAudioStart(
        dedicatedScrubProvider,
        timeInfo.clipTime
      );

      // Paused: stop decode loop, seek to frame
      if (dedicatedScrubProvider?.isPlaying) dedicatedScrubProvider.pause();
      if (fallbackProvider?.isPlaying) fallbackProvider.pause();
      if (pausedProvider?.isPlaying) pausedProvider.pause();
      if (video && !video.paused) video.pause();

      if (!pausedProvider?.isFullMode()) {
        return;
      }

      // On scrub-stop: force a precise seek on the playback provider
      // to ensure the exact frame is decoded (not a cached keyframe from fastSeek)
      if (justStoppedDraggingWc) {
        const wcTimeDiff = Math.abs(pausedProvider.currentTime - timeInfo.clipTime);
        if (wcTimeDiff > 0.001) {
          pausedProvider.seek(timeInfo.clipTime);
          engine.requestRender();
          vfPipelineMonitor.record('vf_wc_settle_seek', {
            clipId: clip.id,
            target: Math.round(timeInfo.clipTime * 1000) / 1000,
          });
        }
      }

      // Determine if a new seek is needed.
      // Key constraint: fastSeek calls decoder.reset() which cancels any pending decode.
      // At 120fps rAF (~8ms), the hardware decoder needs ~10-20ms to produce a frame.
      // If we call fastSeek every frame, the decoder is constantly reset before it can
      // output anything → preview freezes. Only issue a new fastSeek when:
      //   1. No decode is currently in progress (decoder finished), AND
      //   2. The actual frame position differs significantly from target
      if (dedicatedScrubProvider) {
        this.syncPausedWebCodecsProvider(
          dedicatedScrubProvider,
          `${clip.id}:scrub`,
          timeInfo.clipTime,
          ctx.isDraggingPlayhead,
          true,
          true
        );
      }

      if (!dedicatedScrubProvider) {
        this.syncPausedWebCodecsProvider(
          pausedProvider,
          `${clip.id}:fallback`,
          timeInfo.clipTime,
          ctx.isDraggingPlayhead,
          true,
          true
        );
      } else if (fallbackProvider && !scrubProviderReady) {
        this.syncPausedWebCodecsProvider(
          fallbackProvider,
          `${clip.id}:fallback`,
          timeInfo.clipTime,
          ctx.isDraggingPlayhead,
          false,
          false
        );
      } else {
        this.clearFastSeekTracking(`${clip.id}:fallback`);
        clearTimeout(this.wcPreciseSeekTimers[`${clip.id}:fallback`]);
      }

      // Keep audio element at same position — but NOT during scrubbing.
      // video.currentTime triggers the browser's internal decoder (heavy for long-GOP).
      // During scrubbing, audio feedback is handled by proxyFrameCache.playScrubAudio()
      // via Web Audio API, so the video element seek is wasted work.
      if (video && !ctx.isDraggingPlayhead) {
        const timeDiff = Math.abs(video.currentTime - timeInfo.clipTime);
        if (timeDiff > 0.05) {
          video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
        }
      }
    }
  }

  /**
   * Schedule a debounced precise WebCodecs seek.
   * During fast scrubbing, fastSeek shows keyframes instantly.
   * When scrubbing pauses (120ms), do a full decode for the exact frame.
   */
  private schedulePreciseWcSeek(clipId: string, wcp: { seek: (t: number) => void; currentTime: number }, time: number): void {
    this.latestWcPreciseSeekTargets[clipId] = time;
    if (this.wcPreciseSeekTimers[clipId]) {
      clearTimeout(this.wcPreciseSeekTimers[clipId]);
    }

    this.wcPreciseSeekTimers[clipId] = setTimeout(() => {
      delete this.wcPreciseSeekTimers[clipId];
      const targetTime = this.latestWcPreciseSeekTargets[clipId] ?? time;
      // Only seek if still at a different position
      if (Math.abs(wcp.currentTime - targetTime) > 0.01) {
        wcp.seek(targetTime);
        this.lastWcPreciseSeekAt[clipId] = performance.now();
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
