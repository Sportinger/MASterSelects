# Agent 3: Playback-Engine & Audio/Video-Synchronisation in MasterSelects

## 1. Architecture Overview

MasterSelects runs two parallel `requestAnimationFrame` loops during playback:

1. **Playback Loop** (`usePlaybackLoop.ts`) -- advances the playhead position every frame
2. **Render Loop** (`RenderLoop.ts`) -- drives the GPU render pipeline every frame

Both loops run at the browser's native refresh rate (typically 60Hz on most displays, up to 120Hz or 144Hz on high-refresh monitors). The render loop applies frame rate limiting to avoid unnecessary GPU work, while the playback loop runs unconstrained to keep the playhead position as accurate as possible.

### Call Chain Per Frame (during playback)

```
usePlaybackLoop RAF tick
  -> reads masterAudioElement.currentTime (or system clock fallback)
  -> writes playheadState.position (mutable object, no store dispatch)
  -> throttled: every 33ms writes to Zustand store for UI subscribers

RenderLoop RAF tick (separate chain)
  -> idle detection / frame rate limiting
  -> onRender callback (defined in useEngine.ts):
       1. layerBuilder.syncVideoElements()     -- seek/play HTMLVideoElements
       2. layerBuilder.buildLayersFromStore()   -- collect Layer[] for compositor
       3. engine.render(layers)                 -- WebGPU compositing
       4. layerBuilder.syncAudioElements()      -- audio drift correction
```

Key insight: the playhead position is NOT stored in the Zustand store during playback. It lives in a plain mutable object (`playheadState`) to avoid triggering React re-renders at 60fps. The store is updated at a throttled 30fps for UI elements like the timeline cursor.

**Source:** `src/services/layerBuilder/PlayheadState.ts`

```typescript
export const playheadState: PlayheadStateData = {
  position: 0,
  isUsingInternalPosition: false,
  playbackJustStarted: false,
  masterAudioElement: null,
  masterClipStartTime: 0,
  masterClipInPoint: 0,
  masterClipSpeed: 1,
  hasMasterAudio: false,
};
```

---

## 2. requestAnimationFrame vs. setInterval

MasterSelects uses `requestAnimationFrame` exclusively for both playback timing and rendering. `setInterval` / `setTimeout` is only used for:

- Watchdog timer (detect frozen render loops) -- `setInterval` every 2s
- PlaybackHealthMonitor polling -- `requestIdleCallback` with 500ms timeout fallback
- Scrub audio snippet cutoff -- `setTimeout` for brief audio preview
- Deferred precise seeks -- `setTimeout` debouncing (120ms)

The reason is fundamental: `setInterval` has 4ms minimum granularity, can drift under load, and does not synchronize with the display refresh. `requestAnimationFrame` fires exactly once per vsync, giving the tightest possible coupling between position calculation and frame presentation.

**Source:** `src/engine/render/RenderLoop.ts` lines 72-157 -- the entire render loop is a single `requestAnimationFrame` chain.

---

## 3. Audio Master Clock: Who Leads?

MasterSelects implements an **Audio Master Clock** pattern. During normal 1x forward playback, the audio element runs freely and the playhead follows it:

```typescript
// src/components/timeline/hooks/usePlaybackLoop.ts, lines 57-75
if (playheadState.hasMasterAudio && playheadState.masterAudioElement && playbackSpeed === 1) {
  const audio = playheadState.masterAudioElement;
  if (!audio.paused && audio.readyState >= 2) {
    const audioTime = audio.currentTime;
    const speed = playheadState.masterClipSpeed || 1;
    newPosition =
      playheadState.masterClipStartTime +
      (audioTime - playheadState.masterClipInPoint) / speed;
  } else {
    // Fallback to system clock
    const deltaTime = (currentTime - lastTime) / 1000;
    const cappedDelta = Math.min(deltaTime, 0.1);
    newPosition = playheadState.position + cappedDelta * playbackSpeed;
  }
}
```

### Why Audio is Master

Audio is perceptually unforgiving. A single dropped video frame is barely noticeable, but a 20ms audio glitch (pop, click, gap) is immediately audible. By letting the audio element run at its native rate and deriving the playhead from `audio.currentTime`, MasterSelects guarantees glitch-free audio. Video then corrects itself to match.

### Fallback: System Clock

When there is no audio (video-only timeline, or muted), the playback loop falls back to `performance.now()` delta timing:

```typescript
const deltaTime = (currentTime - lastTime) / 1000;
const cappedDelta = Math.min(deltaTime, 0.1); // Cap to 100ms to handle tab switches
newPosition = playheadState.position + cappedDelta * playbackSpeed;
```

The 100ms cap prevents the playhead from jumping forward when the tab is backgrounded and RAF pauses.

### Master Election

The master audio element is elected per frame by `AudioSyncHandler.handlePlayback()`. The first non-paused, non-muted audio element that plays successfully becomes master via `setMasterAudio()`. This is reset on loop boundaries and speed changes.

**Source:** `src/services/layerBuilder/AudioSyncHandler.ts` lines 146-149

---

## 4. Video Drift Correction

During playback, `HTMLVideoElement.play()` runs the video at its native decode rate. The video clock and the audio-derived playhead will inevitably drift apart. MasterSelects uses a **threshold-based correction** strategy:

```typescript
// src/services/layerBuilder/VideoSyncManager.ts, lines 1667-1678
// Normal clips: correct if drift > 300ms
// Speed-keyframed clips: correct if drift > 1500ms (avoids SEEK_STUCK)
const driftThreshold = hasSpeedKeyframes ? 1.5 : 0.3;
if (timeDiff > driftThreshold) {
  video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
}
```

The 300ms threshold is a deliberate trade-off: seeking causes a visible frame stutter (the decoder must re-decode from the nearest keyframe), so small drifts are tolerated. For clips with speed keyframes, the threshold is relaxed to 1.5s because `playbackRate` already tracks the speed curve closely and frequent seeks cause decoder stalls.

### Audio Drift Correction (AudioSyncHandler)

Audio elements get a stricter treatment. Drift > 300ms triggers a hard `currentTime` reset. Drift between 50-300ms is logged but tolerated:

```typescript
// src/services/layerBuilder/AudioSyncHandler.ts, lines 152-167
const timeDiff = element.currentTime - clipTime;
if (Math.abs(timeDiff) > 0.3) {
  element.currentTime = clipTime; // Hard correction
} else if (Math.abs(timeDiff) > 0.05) {
  // Log but tolerate
}
```

---

## 5. HTMLVideoElement.playbackRate -- Limits and Artifacts

MasterSelects clamps `playbackRate` to the browser-supported range:

- **Audio elements:** `Math.max(0.25, Math.min(4, targetRate))` (AudioSyncHandler line 105)
- **Video elements:** `Math.max(0.0625, Math.min(16, targetRate))` (VideoSyncManager line 1640)

The wider range for video (0.0625x to 16x) works because video can be seeked frame-by-frame when playback is too fast for real-time decode. For speeds beyond 1x transport speed, MasterSelects pauses the video and seeks frame-by-frame rather than using `playbackRate`:

```typescript
// VideoSyncManager.ts, line 1611
} else if (ctx.playbackSpeed !== 1) {
  // Non-standard forward transport speed (2x, 4x, etc.): seek frame-by-frame
  if (!video.paused) video.pause();
```

**Pitch preservation** is configurable per clip via `clip.preservesPitch`. This maps to the `HTMLMediaElement.preservesPitch` property. When enabled (default), the browser applies time-stretching to maintain pitch at non-1x rates. When disabled, audio plays at natural pitch (chipmunk/slow-mo effect).

---

## 6. Frame-Dropping Strategy

### RAF Gap Detection

Frame drops are detected by measuring the gap between consecutive `requestAnimationFrame` callbacks:

```typescript
// src/engine/stats/PerformanceStats.ts, lines 54-67
recordRafGap(gap: number, isScrubbing = false): void {
  const targetTime = isScrubbing ? 33 : this.TARGET_FRAME_TIME; // 16.67ms
  const dropThreshold = targetTime * 2; // ~33ms for 60fps
  if (gap > dropThreshold) {
    const missedFrames = Math.max(1, Math.round(gap / targetTime) - 1);
    this.detailedStats.dropsTotal += missedFrames;
    this.detailedStats.dropsThisSecond += missedFrames;
    this.detailedStats.lastDropReason = 'slow_raf';
  }
}
```

During scrubbing the baseline is 33ms (30fps target) so intentional rate-limiting does not inflate drop counts.

### Render Timing Classification

The `PerformanceStats.recordRenderTiming()` method classifies slow renders by bottleneck:

- `slow_import`: texture import took >50% of frame budget (video decode / GPU upload stall)
- `slow_render`: compositing pass took too long
- `slow_raf`: RAF callback was late (browser/OS scheduling)

### Graceful Degradation

MasterSelects does not have an explicit frame-skip mechanism. Instead, it relies on:

1. **Scrubbing cache (RAM preview):** During playback, previously-rendered frames are cached. On scrub, cached frames are served instantly without GPU work.
2. **Effects skip during scrubbing:** `isDraggingPlayhead` skips expensive per-layer effects (line 179 of RenderDispatcher).
3. **Rate limiting:** Playback is capped at 60fps regardless of display refresh rate (see section 8).

---

## 7. PlaybackHealthMonitor

The `PlaybackHealthMonitor` is a background polling service that runs via `requestIdleCallback` every ~500ms. It detects 8 anomaly types and auto-recovers:

| Anomaly | Detection | Recovery |
|---------|-----------|----------|
| `FRAME_STALL` | `video.currentTime` unchanged for 1.5s | Nudge `currentTime + 0.001` |
| `WARMUP_STUCK` | Video in warmup state >3s | Clear warmup state |
| `RVFC_ORPHANED` | RVFC handle for deleted clip | Cancel handle |
| `SEEK_STUCK` | `video.seeking === true` for >2s | Re-set `currentTime` |
| `READYSTATE_DROP` | `readyState < 2` during playback | Log only |
| `GPU_SURFACE_COLD` | Playing video not in `videoGpuReady` set | Reset GPU ready state |
| `RENDER_STALL` | No render for >3s while playing | Force `requestRender()` |
| `HIGH_DROP_RATE` | >10 drops/second | Log only |

### Escalation Protocol

If a single clip triggers 3+ anomalies within 12 seconds, the monitor escalates to `recoverClipPlaybackState()` which performs a full clip recovery: clear all tracking state, reset GPU readiness, and re-sync the video element to the current playhead position.

Each anomaly type has a 5-second cooldown to prevent log spam. The monitor uses a ring buffer (max 200 events) accessible via `__PLAYBACK_HEALTH__` on the console.

**Source:** `src/services/playbackHealthMonitor.ts`

---

## 8. Idle Detection and FPS Limiting

### Idle Detection (RenderLoop)

The render loop enters idle mode after 1 second of no activity:

```typescript
// RenderLoop.ts, lines 45-48
private readonly IDLE_TIMEOUT = 1000;
private readonly VIDEO_FRAME_TIME = 16.67;  // ~60fps
private readonly SCRUB_FRAME_TIME = 33;     // ~30fps scrubbing
```

In idle mode, the RAF loop keeps running (to maintain the callback chain) but skips the render callback entirely. Activity is tracked via `requestRender()`, which any code path can call -- playhead changes, scrubbing, property edits, etc.

### Idle Suppression After Reload

After a page reload, video GPU surfaces are empty (all rendering APIs return black until `video.play()` activates the GPU compositor). The render loop suppresses idle detection (`idleSuppressed = true`) until the user presses play for the first time, keeping the loop running so warmup can complete.

### Frame Rate Limiting

During playback, frames are limited to ~60fps (16.67ms minimum between renders), even on 120Hz/144Hz displays:

```typescript
// RenderLoop.ts, lines 111-130
if (this.hasActiveVideo || this.isPlaying) {
  if (this.isPlaying) {
    if (timeSinceLastRender < this.VIDEO_FRAME_TIME) {
      this.animationId = requestAnimationFrame(loop);
      return; // Skip this frame
    }
  } else if (this.isScrubbing) {
    // 30fps baseline, BUT render immediately if RVFC signaled new frame
    if (!this.newFrameReady && timeSinceLastRender < this.SCRUB_FRAME_TIME) {
      this.animationId = requestAnimationFrame(loop);
      return;
    }
    this.newFrameReady = false;
  }
  this.lastRenderTime = timestamp;
}
```

The comment explains an important subtlety: without the `isPlaying` check for the 60fps limiter, a 30fps video on a 120Hz display would cause `hasActiveVideo` to oscillate (75% cache-hits where no external texture is needed, so the flag goes false), accidentally disabling the limiter and rendering at 120fps.

### Watchdog

A `setInterval` watchdog runs every 2 seconds. If no render has occurred for 3+ seconds while the engine should be active, it forces a wake-up. If the RAF loop itself has died (`animationId === null` while `isRunning === true`), it restarts the entire loop.

---

## 9. Audio Sync: Web Audio API + HTMLVideoElement

MasterSelects has two audio paths:

### Path 1: Simple Volume (HTMLMediaElement.volume)

When no EQ effects are applied, audio stays on the simple path: `element.volume = targetVolume` (clamped 0-1). This avoids Web Audio API overhead entirely.

### Path 2: Web Audio Routing (AudioRoutingManager)

When EQ effects are active, the `AudioRoutingManager` creates a Web Audio graph:

```
HTMLMediaElement -> MediaElementAudioSourceNode -> GainNode -> BiquadFilters -> AudioContext.destination
```

This is managed per-element and lazily created. The AudioContext is created once and shared. The `resume()` call handles browser autoplay policy:

```typescript
// AudioSyncHandler.ts, lines 256-260
export async function resumeAudioContextIfNeeded(isPlaying, isDraggingPlayhead) {
  if (isPlaying && !isDraggingPlayhead) {
    await audioManager.resume().catch(() => {});
  }
}
```

### Scrub Audio Preview

During drag scrubbing, a short audio snippet is played at the current position for auditory feedback:

```typescript
// AudioSyncHandler.ts, lines 74-86
private playScrubAudio(element, time): void {
  element.currentTime = time;
  element.volume = 0.8;
  element.play().catch(() => {});
  if (!this.scrubAudioTimeout) {
    this.scrubAudioTimeout = setTimeout(() => {
      element.pause();
      this.scrubAudioTimeout = null;
    }, LAYER_BUILDER_CONSTANTS.SCRUB_AUDIO_DURATION);
  }
}
```

---

## 10. Timeline Playhead Sync: Store State vs. Actual Position

This is one of the most subtle aspects of the architecture. There are three "playhead positions":

1. **`playheadState.position`** -- mutable object, updated every RAF tick (~60fps), used by the render pipeline
2. **`useTimelineStore.playheadPosition`** -- Zustand store, updated every 33ms (~30fps), used by React UI
3. **`video.currentTime`** -- per-video element, updated by the browser's media decoder

During playback, (1) leads and is derived from audio master clock. (2) lags by up to 33ms. (3) can drift up to 300ms from (1) before correction.

The `FrameContext` always reads the authoritative position:

```typescript
// FrameContext.ts, line 93
const playheadPosition = getPlayheadPosition(storePlayheadPosition);
```

Where `getPlayheadPosition()` returns `playheadState.position` during playback and `storePlayheadPosition` when paused:

```typescript
export function getPlayheadPosition(storePosition: number): number {
  return playheadState.isUsingInternalPosition
    ? sanitizePlayheadPosition(playheadState.position, 0)
    : sanitizePlayheadPosition(storePosition, playheadState.position);
}
```

### Play-to-Pause Transition

When playback stops, the video's actual position is snapped to the playhead (not the other way around). The `VideoSyncManager` detects the playing-to-paused transition, captures the video's current frame, and optionally snaps the playhead to the video's actual position if the drift is within 0.5s:

```typescript
// VideoSyncManager.ts, lines 1570-1574
const shouldSnapPlayheadToStopFrame =
  Math.abs(newPlayheadPos - currentPlayhead) <= VideoSyncManager.PLAYBACK_STOP_SNAP_MAX_DELTA;
if (videoAdvanced && shouldSnapPlayheadToStopFrame) {
  playheadState.position = newPlayheadPos;
  useTimelineStore.setState({ playheadPosition: newPlayheadPos });
}
```

---

## 11. requestVideoFrameCallback (RVFC)

RVFC is a key browser API that fires when a video frame is actually composited to the screen. MasterSelects uses it extensively for precise scrubbing:

```typescript
// VideoSyncManager.ts, lines 1865-1886
private registerRVFC(clipId: string, video: HTMLVideoElement): void {
  const rvfc = (video as any).requestVideoFrameCallback;
  if (typeof rvfc === 'function') {
    const prevHandle = this.rvfcHandles[clipId];
    if (prevHandle !== undefined) {
      (video as any).cancelVideoFrameCallback(prevHandle);
    }
    this.rvfcHandles[clipId] = rvfc.call(video, () => {
      delete this.rvfcHandles[clipId];
      delete this.pendingSeekTargets[clipId];
      engine.markVideoFramePresented(video, video.currentTime, clipId);
      engine.captureVideoFrameAtTime(video, video.currentTime, clipId);
      scrubSettleState.resolve(clipId);
      engine.requestNewFrameRender(); // Bypass scrub rate limiter
    });
  }
}
```

RVFC serves two purposes:
1. **Accurate frame presentation tracking:** Unlike the `seeked` event (which fires when the seek completes but before the frame is composited), RVFC fires when the decoded frame is actually on screen.
2. **Scrub rate limiter bypass:** When RVFC fires, `requestNewFrameRender()` sets `newFrameReady = true` in the RenderLoop, bypassing the 30fps scrub rate limiter for immediate display.

---

## 12. Hybrid Seeking Strategy

Scrubbing through long-GOP video (YouTube clips with 5-7 second keyframe intervals) is a known pain point. MasterSelects implements a two-phase hybrid seek:

**Phase 1 (immediate):** `video.fastSeek(time)` -- shows the nearest keyframe instantly (<10ms).

**Phase 2 (deferred, 120ms debounce):** `video.currentTime = time` -- precise frame decode when scrubbing pauses.

```typescript
// VideoSyncManager.ts, lines 1823-1839 (comment block)
// During drag (fast scrubbing):
//   Phase 1: fastSeek -> instant keyframe feedback
//   Phase 2: deferred precise seek -> exact frame when scrubbing pauses
//
// When not dragging (single click / arrow keys): precise seek via currentTime
```

The 120ms debounce prevents decoder saturation. If the user keeps dragging, only `fastSeek` calls fire. When they pause, the precise seek delivers the exact frame.

---

## 13. Stale Closure Protection

The playback loop explicitly avoids stale closures by reading state fresh each frame:

```typescript
// usePlaybackLoop.ts, line 42
const state = useTimelineStore.getState(); // Fresh read every tick
```

The render callback in `useEngine.ts` similarly reads from stores inside the callback, never from closure-captured values. The `try/catch` in the playback loop (line 164) ensures a thrown error never breaks the RAF chain, which would desync audio.

The `FrameContext` pattern further centralizes this: all store reads happen once per frame in `createFrameContext()`, and the resulting object is passed through the entire sync and render pipeline.

---

## 14. Seamless Cut Transitions (Handoff)

When sequential clips on the same track come from the same source file (split clips), MasterSelects performs a "handoff": the outgoing clip's video element keeps playing through the cut boundary without pause/play. The `VideoSyncManager.computeHandoffs()` method detects same-source sequential clips and maintains a `lastTrackState` map to track which video element was playing on each track.

This avoids the audible gap and visual glitch that would occur if the old clip paused and the new clip started playing from scratch.

---

## 15. Summary: Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Audio is master clock | Audio glitches are more noticeable than video drops |
| Mutable playhead object, not Zustand | 60fps updates would thrash React subscribers |
| 60fps render cap even on 120Hz displays | Video content is typically 24-60fps; double GPU work for zero benefit |
| 300ms video drift tolerance | Seeking causes decoder stall; small drift is invisible |
| RVFC over seeked event | Accurate frame presentation timing |
| fastSeek + deferred precise seek | Responsive scrubbing on long-GOP codecs |
| requestIdleCallback for health monitor | Non-critical monitoring should not steal frame budget |
| Watchdog via setInterval | Must survive render loop death |
| Idle suppression after reload | GPU surfaces need render loop for warmup |
| Frame caching only when paused | GPU readback (mapAsync) stalls the pipeline during playback |
