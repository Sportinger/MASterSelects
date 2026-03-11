# Agent 7: Full Codebase Exploration — HTMLVideoElement in a Browser NLE

## Overview

This document provides an independent, cross-cutting analysis of how MasterSelects handles
HTMLVideoElement and related Web APIs throughout its entire rendering, playback, sync, caching,
and export pipeline. The analysis is based on a systematic read of every engine, service, and
hook file involved in video processing.

---

## 1. Decode & Seeking

### 1.1 The Decoder Priority Chain

The codebase implements a four-tier decode hierarchy, visible in `LayerCollector.collectLayerData()`:

1. **NativeHelper** (Rust FFmpeg via WebSocket) — returns `ImageBitmap`, uploaded as `texture_2d`
2. **ParallelDecode** (WebCodecs `VideoDecoder` per clip) — returns `VideoFrame`, imported as `texture_external`
3. **WebCodecsPlayer** (MP4Box demux + WebCodecs decode) — returns `VideoFrame`, imported as `texture_external`
4. **HTMLVideoElement** — zero-copy `importExternalTexture` or copied fallback via `ScrubbingCache`

This ordering is significant. The code falls through from top to bottom, preferring the most
deterministic decode path. The HTMLVideoElement path is the last resort for preview but remains
the default for most users (the `useFullWebCodecsPlayback` feature flag is `false`).

**File:** `src/engine/featureFlags.ts` — `useFullWebCodecsPlayback: false`

### 1.2 HTMLVideoElement.currentTime Precision

The codebase treats `video.currentTime` as inherently imprecise. Evidence:

- **VideoSyncManager.safeSeekTime()** clamps to `duration - 0.001` to avoid EOF decoder stalls
  where H.264 B-frame decoders wait for reference frames that do not exist.
- **Tolerance constants** throughout: `0.01` for "close enough" seeks, `0.05` for paused
  settle detection, `0.12` for "fresh presented frame", `0.35` for scrub cache lookups.
- **lastPresentedTime tracking** via `ScrubbingCache.markFramePresented()` provides a separate
  "what did we actually show" signal distinct from `video.currentTime`.

The fundamental problem: `video.currentTime` reports the *intended* position but not necessarily
the *displayed* frame. After `video.currentTime = X`, the actual decoded frame may be the
nearest preceding I-frame, not frame X. The codebase addresses this with the
"presented time" concept (see 1.4).

### 1.3 The Hybrid Seek Strategy

`VideoSyncManager` implements a sophisticated multi-phase seek pipeline:

**During scrubbing (dragging playhead):**
1. `fastSeek()` is used when available (Safari/Firefox) for low-latency seeking
2. A debounced `preciseSeekTimer` fires 90ms later with a `currentTime` assignment
3. `requestVideoFrameCallback` (RVFC) is registered to detect when the new frame lands
4. Queued seek targets accumulate in `queuedSeekTargets` while a seek is in-flight
5. The `armSeekedFlush()` mechanism listens for `seeked` events to dispatch the next queued target

**File:** `src/services/layerBuilder/VideoSyncManager.ts`, lines 450-486 (beginOrQueueSettleSeek)

**Surprising finding:** The system tracks *three* different time references per clip simultaneously:
- `latestSeekTargets[clipId]` — the most recent user-requested position
- `pendingSeekTargets[clipId]` — the position currently being sought
- `queuedSeekTargets[clipId]` — the next position to seek after current completes

This triple-buffer approach prevents seeks from being lost during rapid scrubbing while ensuring
the most recent target always wins.

### 1.4 requestVideoFrameCallback (RVFC) Usage

RVFC is used in two distinct contexts:

1. **VideoSyncManager.registerRVFC()** — After setting `video.currentTime`, RVFC is registered
   to detect when the browser has actually decoded and presented the new frame. This signal
   triggers `flushQueuedSeekTarget()` which decides whether to issue another seek or resolve.

2. **VideoSeeker (export)** — `seekVideo()` uses RVFC as the primary "frame ready" signal
   during frame-by-frame export, with a 500ms timeout fallback for slow codecs like AV1.

3. **RenderLoop.requestNewFrameRender()** — When RVFC fires, the render loop's scrub rate
   limiter is bypassed via `newFrameReady = true`, ensuring the freshly decoded frame is
   displayed immediately rather than waiting for the next 30fps scrub tick.

### 1.5 The GPU Surface Warmup Problem

After a page reload, `importExternalTexture()` returns a valid `GPUExternalTexture` but the
actual pixel data is black because Chrome defers video frame decoding until `play()` is called.

The codebase solves this with a "warmup" mechanism:

- `VideoSyncManager.warmingUpVideos` (WeakSet) tracks videos currently being warmed up
- `startTargetedWarmup()` calls `video.play()`, waits for RVFC, then pauses and captures
- `LayerCollector.videoGpuReady` (WeakSet) tracks which videos have had their GPU surface activated
- `RenderLoop.idleSuppressed` keeps the engine rendering during warmup (prevents idle timeout)
- `ScrubbingCache.captureVideoFrameViaImageBitmap()` uses `createImageBitmap()` as a forced
  decode path — the *only* API that forces Chrome to actually decode a frame synchronously

**File:** `src/engine/texture/ScrubbingCache.ts`, lines 276-323

### 1.6 Race Conditions in Seeking

The codebase handles several known race conditions:

- **Seek overlap:** `beginOrQueueSettleSeek()` checks `video.seeking` and `rvfcHandles[clipId]`
  before issuing a new seek. If either is active, the target is queued instead.
- **Stale pending seeks:** `shouldRetargetPendingSeek()` considers seek age (170ms for idle,
  90ms for drag) and target drift before interrupting an in-flight seek.
- **Pending seek hangs:** `maybeRecoverDraggingPendingSeek()` detects when `video.seeking`
  has been true for too long with `readyState < 2`, indicating a stuck seek.
- **Owner mismatch:** When a video element is shared across clips (common with split clips),
  `lastPresentedOwner` tracking prevents displaying a frame captured for clip A while clip B
  is active.

---

## 2. Multi-Track & Compositing

### 2.1 Layer Collection Architecture

The render pipeline processes layers in reverse order (bottom-up compositing):

```
RenderDispatcher.render(layers)
  -> LayerCollector.collect(layers) — imports textures
  -> NestedCompRenderer.preRender() — handles compositions-in-compositions
  -> Compositor.composite(layerData) — ping-pong GPU compositing
  -> OutputPipeline.renderToCanvas() — final output
```

Each layer goes through a texture acquisition phase (`collectLayerData`) that returns one of:
- `isVideo: true, externalTexture: GPUExternalTexture` — zero-copy video path
- `isVideo: false, textureView: GPUTextureView` — copied/cached texture path

The Compositor then uses different shader pipelines depending on the texture type:
- `getExternalCompositePipeline()` — for `texture_external` (video)
- `getCompositePipeline()` — for `texture_2d<f32>` (images, cached frames)

**File:** `src/engine/render/Compositor.ts`

### 2.2 GPU Texture Type Strategy

| Source | GPU Type | Binding | Performance |
|--------|----------|---------|-------------|
| HTMLVideoElement (live) | `texture_external` | `importExternalTexture` | Zero-copy, single-frame lifetime |
| VideoFrame (WebCodecs) | `texture_external` | `importExternalTexture` | Zero-copy, must guard `closed` state |
| HTMLVideoElement (cached) | `texture_2d<f32>` | `copyExternalImageToTexture` | GPU copy, persistent |
| ImageBitmap (NativeHelper) | `texture_2d<f32>` | `copyExternalImageToTexture` | GPU copy, reusable texture |
| HTMLImageElement | `texture_2d<f32>` | `copyExternalImageToTexture` | Copied once, cached |
| HTMLCanvasElement (text) | `texture_2d<f32>` | `copyExternalImageToTexture` | Cached by reference |

**Critical detail:** `texture_external` has a single-frame lifetime. It is valid only for the
current `GPUCommandEncoder` and becomes invalid after `device.queue.submit()`. This is why the
`CompositorPipeline.beginFrame()` clears frame-scoped caches at the start of each render.

**Surprising finding:** The `TextureManager.importVideoTexture()` explicitly guards against
closed `VideoFrame` objects (line 224): a closed frame passed to `importExternalTexture`
crashes the GPU process with `STATUS_BREAKPOINT`.

### 2.3 Ping-Pong Compositing

The compositing pipeline uses a classic ping-pong buffer approach with dedicated render targets:

- `pingTexture/pongTexture` — main composition buffers
- `independentPingTexture/independentPongTexture` — separate buffers for multi-composition preview
- `effectTempTexture/effectTempTexture2` — temporary buffers for effect processing

Each layer is composited by reading from one buffer and writing to the other, then swapping.
The final result is in `readView`, which is passed to `OutputPipeline`.

### 2.4 Nested Compositions

`NestedCompRenderer` handles compositions-within-compositions with:

- **Texture pooling:** Ping-pong buffers are pooled by dimensions (`acquireTexturePair`)
- **Frame caching:** Quantized time-based caching skips re-renders for static nested comps
- **Depth limiting:** `MAX_NESTING_DEPTH` prevents infinite recursion
- **Efficient copy:** Final result uses `copyTextureToTexture` (GPU-to-GPU) rather than readback

The nested renderer duplicates much of `LayerCollector`'s video handling logic (scrubbing cache,
fallback chain, owner tracking). This is a notable code duplication — the same fallback cascade
(scrub-cache -> seeking-cache -> emergency-hold -> live-import -> final-cache) appears in three
places: `LayerCollector`, `NestedCompRenderer`, and `RenderDispatcher.renderToPreviewCanvas`.

**File:** `src/engine/render/NestedCompRenderer.ts`, lines 380-769

### 2.5 Multi-Preview Rendering

`RenderDispatcher.renderToPreviewCanvas()` supports rendering different layers to different
target canvases simultaneously. This enables multi-composition preview where each canvas shows
different content. It uses the independent ping/pong buffers to avoid interfering with the main
composition.

---

## 3. Playback & Synchronisation

### 3.1 Render Loop Architecture

`RenderLoop` is a `requestAnimationFrame`-based loop with several sophistications:

- **Idle detection:** After 1000ms of no activity, the loop continues running but skips render
  calls. This saves GPU power when the user is not interacting.
- **Idle suppression:** After page reload, idle detection is disabled until the first `play()`,
  allowing video GPU warmup to complete.
- **Rate limiting:** During playback, renders are capped at ~60fps (16.67ms). During scrubbing,
  renders are capped at ~30fps (33ms) unless RVFC signals a new frame (`newFrameReady`).
- **Watchdog:** A 2-second interval timer detects stalled render loops and restarts them if
  the RAF loop died.

**File:** `src/engine/render/RenderLoop.ts`

**Surprising finding:** The rate limiter during playback applies even when `hasActiveVideo` is
false. This handles 120Hz displays where a 30fps video causes `hasActiveVideo` to oscillate
(75% cache-hit frames are isVideo=false), which without this fix caused rendering at 120fps.

### 3.2 Audio/Video Synchronization

The audio sync strategy uses video as the master clock:

1. **AudioSyncHandler.handlePlayback()** sets `element.playbackRate` to match the clip speed
2. Audio drift detection: if `|element.currentTime - clipTime| > 0.3`, force resync
3. Smaller drifts (>0.05) are logged but tolerated
4. `setMasterAudio()` designates one audio element as the master timing reference

**File:** `src/services/layerBuilder/AudioSyncHandler.ts`

The `playbackRate` is clamped to `[0.25, 4.0]` — the browser's supported range. Pitch
preservation is controlled per-clip via `element.preservesPitch`.

### 3.3 Playhead State vs Actual Video Position

The system maintains careful separation between:

1. **Playhead position** — the timeline time the user sees (`useTimelineStore.playheadPosition`)
2. **Target video time** — computed from playhead via `getSourceTimeForClip()` accounting for
   speed keyframes, in/out points, and reversed playback
3. **video.currentTime** — what the browser's decoder is at
4. **lastPresentedTime** — what was actually displayed (tracked via RVFC + ScrubbingCache)

The `layer.source.mediaTime` field carries the computed target time through to the renderer.
When available, it overrides `video.currentTime` for cache lookups:
```typescript
private getTargetVideoTime(layer: Layer, video: HTMLVideoElement): number {
  return layer.source?.mediaTime ?? video.currentTime;
}
```

### 3.4 Frame Dropping & Health Monitoring

`PlaybackHealthMonitor` polls every 500ms (via `requestIdleCallback`) and detects:

| Anomaly | Threshold | Recovery |
|---------|-----------|----------|
| FRAME_STALL | currentTime unchanged for 1.5s | Nudge `currentTime += 0.001` or play/pause cycle |
| WARMUP_STUCK | warmup > 3s | Clear warmup state |
| SEEK_STUCK | `video.seeking` for > 2s | Re-assign `video.currentTime` |
| READYSTATE_DROP | `readyState < 2` during playback | Log only |
| GPU_SURFACE_COLD | Playing video not in `videoGpuReady` | Reset GPU ready state |
| RENDER_STALL | No render for > 3s while playing | Force render request |
| HIGH_DROP_RATE | > 10 drops/second | Log only |

**Escalation:** If the same clip triggers 3 anomalies within 12 seconds, `escalateClipRecovery()`
performs a full playback state recovery (pause, reset GPU, re-seek, resume).

**File:** `src/services/playbackHealthMonitor.ts`

### 3.5 The Scrub-Settle State Machine

After scrubbing stops, there is a "settle" phase to ensure the correct frame is displayed:

1. **settle** — initial post-scrub state, issues a precise seek
2. **retry** — if the first seek did not produce the correct frame, try again
3. **warmup** — if retry failed, do a play/pause warmup cycle

This is tracked by `scrubSettleState` and monitored by `VideoSyncManager.maybeRecoverScrubSettle()`.
The `RenderLoop.setIsScrubbing()` ensures at least one more render cycle runs after scrub stops
so the settle-seek can fire.

---

## 4. Memory & Performance

### 4.1 Texture Memory Budget

The ScrubbingCache has explicit memory budgets:

| Cache | Max Entries | Estimated VRAM |
|-------|-------------|----------------|
| Scrubbing frame cache | 300 frames | ~2.4 GB at 1080p |
| RAM preview composite | 900 frames, 512 MB CPU | 512 MB RAM |
| GPU frame cache | 60 frames | ~500 MB at 1080p |
| Last-frame cache | 1 per video | ~8 MB per 1080p video |

**File:** `src/engine/texture/ScrubbingCache.ts`

### 4.2 The "Don't Destroy" Pattern

A pervasive pattern throughout the codebase: **GPU textures are not explicitly destroyed; they
are left to GC.**

Examples:
- `ScrubbingCache.clearScrubbingCache()`: "Don't destroy textures - let GC handle to avoid GPU conflicts"
- `TextureManager.clearCaches()`: "Don't destroy textures - let GC handle... WebGPU will automatically
  release GPU resources when JS objects are GC'd AND the GPU is done using them"
- `NestedCompRenderer.cleanupTexture()`: "Just remove from map - don't destroy, let GC handle"

The exception is `TextureManager.dynamicTextures` (NativeDecoder textures), which ARE explicitly
destroyed because they are rapidly re-created per frame and would leak without explicit cleanup.

This pattern trades potential temporary VRAM overuse for safety against use-after-destroy bugs,
which would crash the GPU process. It is a pragmatic choice given the complexity of GPU resource
lifetimes in a multi-path rendering pipeline.

### 4.3 VideoFrame.close() Safety

The `TextureManager.importVideoTexture()` guards against closed VideoFrames:
```typescript
if ((source as any).closed || source.codedWidth === 0 || source.codedHeight === 0) {
  return null;
}
```

In the export pipeline (`FrameExporter`), `videoFrame.close()` is called immediately after
`encoder.encodeVideoFrame()`, ensuring frames do not accumulate in memory during long exports.

The `WebCodecsPlayer` manages a frame buffer of max 8 frames (`MAX_FRAME_BUFFER`), closing
old frames as new ones arrive. The `ParallelDecodeManager` keeps a larger buffer (300 frames)
but tracks timestamps for eviction.

### 4.4 Caching Strategy During Playback vs Scrubbing

The codebase makes an important distinction:

**During playback:** Frame caching into `ScrubbingCache` is SKIPPED. Comment from LayerCollector:
> "With 4+ videos, the GPU copies (copyExternalImageToTexture per video at 20fps) waste
> significant bandwidth that's needed for rendering + effects."

**During scrubbing/paused:** Frames are aggressively cached via `captureVideoFrame()` and
`cacheFrameAtTime()`, with rate limiting (`lastCaptureTime`, minimum 50ms between captures).

This is a well-considered tradeoff: playback performance is prioritized over cache building,
since cached frames are only needed for scrubbing.

### 4.5 LRU Eviction via Map Insertion Order

All caches use JavaScript `Map` insertion order for O(1) LRU operations. When accessing a cached
entry, it is deleted and re-inserted (moved to end). Eviction removes the first entry (oldest).
This avoids the overhead of a dedicated LRU data structure.

---

## 5. Export & Encoding

### 5.1 Export Pipeline Architecture

The export pipeline (`FrameExporter`) operates in two phases:

**Phase 1: Video** — Frame-by-frame rendering:
1. `seekAllClipsToTime()` — position all videos at the target time
2. `waitForAllVideosReady()` — ensure frames are decoded
3. `buildLayersAtTime()` — construct layer data
4. `engine.render(layers)` — GPU compositing
5. Capture: zero-copy (`OffscreenCanvas -> VideoFrame`) or fallback (`readPixels`)
6. `encoder.encodeVideoFrame(frame)` — WebCodecs encoding

**Phase 2: Audio** — Full audio mixdown via `AudioExportPipeline`

### 5.2 Export Decode Modes

Three decode strategies are available during export:

1. **Parallel decode** (`ParallelDecodeManager`): Dedicated `VideoDecoder` per clip, pre-decoding
   frames in a 60-frame buffer ahead of render position. Fastest for multi-clip timelines.

2. **Sequential WebCodecs** (`WebCodecsExportMode`): Single decoder per clip, seeking sequentially
   through the file. Used for single-clip or simple timelines.

3. **HTMLVideoElement seeking** (`seekVideo`): Falls back to `video.currentTime` assignment
   with `requestVideoFrameCallback` for frame-accurate waiting. Slowest but most compatible.

### 5.3 The seekVideo Promise Pattern

`VideoSeeker.seekVideo()` wraps the async seek operation in a Promise with a 500ms timeout:

```typescript
function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(), 500);
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      waitForFrame();  // Uses RVFC or readyState polling
    };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = targetTime;
  });
}
```

The 500ms timeout is noted as being for "AV1 and other slow-decoding codecs." The RVFC fallback
path uses `setTimeout` (not `requestAnimationFrame`) since export runs off-screen where rAF
may be throttled.

### 5.4 WebCodecs Video Encoding

`VideoEncoderWrapper` supports four codecs across two containers:

| Codec | Container | Codec String |
|-------|-----------|-------------|
| H.264 | MP4 | `avc1.4d0028` (Main Profile, Level 4.0) |
| H.265 | MP4 | `hvc1.1.6.L93.B0` (Main Profile, Level 3.1) |
| VP9 | MP4/WebM | `vp09.00.10.08` (Profile 0, 8-bit) |
| AV1 | MP4/WebM | `av01.0.04M.08` (Main Profile, 8-bit) |

The encoder uses `latencyMode: 'quality'` and `bitrateMode: 'variable'` for best output quality.
Keyframes are inserted every `fps` frames (1 per second by default).

### 5.5 Audio Export

Audio is processed separately via `AudioExportPipeline`:
- `AudioMixer` combines all tracks using `OfflineAudioContext`
- Track mute/solo state is respected
- Normalization is optional with configurable headroom
- Output is encoded as AAC (MP4) or Opus (WebM) via `AudioEncoderWrapper`

### 5.6 Zero-Copy Export Path

The preferred export path creates a `VideoFrame` directly from an `OffscreenCanvas` that
the engine renders to. This avoids a GPU readback and is significantly faster than `readPixels`.
The fallback path reads pixels from the GPU into CPU memory, creates a `VideoFrame` from the
raw RGBA buffer, then encodes it.

---

## 6. Browser Compatibility

### 6.1 The Firefox Texture Sampling Bug

`htmlVideoPreviewFallback.ts` contains a Firefox-specific workaround:

```typescript
function getCopiedHtmlVideoPreviewFrame(video, scrubbingCache, ...): ... {
  if (!isFirefoxBrowser() || !scrubbingCache) return null;
  // Firefox can intermittently sample imported HTML video textures as black
  // during playback. Copying into a persistent texture is slower but stable.
  const captured = scrubbingCache.captureVideoFrame(video, captureOwnerId);
  ...
}
```

On Firefox, `importExternalTexture` from an HTMLVideoElement intermittently returns black frames.
The workaround copies every frame to a persistent `texture_2d` via `copyExternalImageToTexture`,
sacrificing zero-copy performance for visual correctness. This function is called in all three
layer collection code paths (LayerCollector, NestedCompRenderer, RenderDispatcher).

### 6.2 Feature Detection

The codebase performs runtime feature detection for:

- **WebCodecs:** `'VideoEncoder' in window && 'VideoFrame' in window` (FrameExporter.isSupported)
- **RVFC:** `typeof (video as any).requestVideoFrameCallback === 'function'`
- **fastSeek:** `typeof fastSeek === 'function'` (Safari/Firefox only, Chrome lacks it)
- **MediaStreamTrackProcessor:** Checked in WebCodecsPlayer stream mode
- **requestIdleCallback:** `typeof requestIdleCallback !== 'undefined'` (PlaybackHealthMonitor)
- **WebGPU:** Full adapter/device probing in WebGPUContext

### 6.3 The Three-Decoder Fallback Chain

For preview playback, the system tries decoders in this order:

1. **WebCodecs full mode** — if `useFullWebCodecsPlayback` flag is on (currently off by default)
2. **HTMLVideoElement** — default path, with all the scrubbing/caching infrastructure
3. **Scrubbing cache** — when video is not ready (seeking, loading), show last known good frame

For export, the chain is:
1. **Parallel WebCodecs decode** — preferred, bypasses HTMLVideoElement entirely
2. **Sequential WebCodecs** — single decoder per clip
3. **HTMLVideoElement seeking** — fallback

### 6.4 The VAAPI Workaround

The WebCodecsPlayer header comment mentions: "Bypasses browser VAAPI issues by using WebCodecs
API directly." On Linux, Chrome's VAAPI video acceleration can produce corrupt frames or fail
entirely. WebCodecs provides a more reliable decode path because it uses a separate code path
in Chrome's GPU process.

### 6.5 Autoplay Policy

`AudioSyncHandler.resumeAudioContextIfNeeded()` handles browser autoplay restrictions by
explicitly calling `audioManager.resume()` when playback starts. The `play()` calls throughout
the codebase always use `.catch()` to handle autoplay rejections gracefully.

### 6.6 Codec Support Detection

`codecHelpers.checkCodecSupport()` uses `VideoEncoder.isConfigSupported()` to probe for
hardware encoder support before export begins. The system falls back gracefully: WebM containers
get VP9 if the requested codec is not supported, MP4 containers get H.264.

---

## 7. Cross-Cutting Observations

### 7.1 Code Duplication in Layer Collection

The video fallback cascade (scrub-cache -> seeking-cache -> drag-hold -> emergency-hold ->
live-import -> copied-preview -> final-cache) is duplicated in three files:

1. `LayerCollector.tryHTMLVideo()` (~370 lines)
2. `NestedCompRenderer.collectNestedLayerData()` (~390 lines)
3. `RenderDispatcher.renderToPreviewCanvas()` (~200 lines)

Each has slightly different conditions for `allowLiveVideoImport`, `allowConfirmedFrameCaching`,
etc. This is a maintenance risk — a fix in one place must be replicated in the other two.

### 7.2 The Presented-Time Tracking System

A sophisticated "presented time" system tracks what frame was *actually* displayed vs what
was *requested*. This involves:

- `ScrubbingCache.lastPresentedFrameTimes` (WeakMap)
- `ScrubbingCache.lastPresentedFrameOwners` (WeakMap)
- `engine.markVideoFramePresented()` / `engine.getLastPresentedVideoTime()`
- Owner mismatch detection to prevent cross-clip frame contamination

This system exists because `video.currentTime` is not a reliable indicator of what frame is
being rendered — there can be a multi-frame delay between setting `currentTime` and the browser
actually decoding and presenting that frame.

### 7.3 The vfPipelineMonitor Telemetry

Nearly every decision point in the video pipeline emits structured telemetry via
`vfPipelineMonitor.record()`. Events include:

- `vf_scrub_path` — which cache/import path was taken for each layer during scrub
- `vf_settle_seek` — settle seek decisions and outcomes
- `vf_seek_precise` / `vf_seek_fast` — seek type and target
- `vf_preview_frame` — what was actually displayed, with drift metrics
- `audio_drift` / `audio_drift_correct` — audio sync events

This telemetry is accessible via the AI tools bridge (`/masterselects getPlaybackTrace`),
enabling runtime debugging without browser devtools.

### 7.4 The HMR Singleton Pattern

All singletons (engine, health monitor, etc.) survive hot module replacement via:
```typescript
if (import.meta.hot?.data?.myService) {
  instance = import.meta.hot.data.myService;
}
import.meta.hot?.dispose((data) => {
  data.myService = instance;
});
```

This is critical for development — without it, each HMR update would create a new engine
instance, orphaning GPU resources and video elements.

### 7.5 Seamless Cut Transitions

`VideoSyncManager.lastTrackState` tracks the last video element per track. When sequential
clips on the same track share the same source file (split clips), the outgoing clip's video
element keeps playing through the cut with no pause/play gap. This is a professional-grade
feature that avoids the audible pop and visual glitch of stopping and restarting a video
decoder at cut points.

### 7.6 The RenderLoop Rate Limiter Insight

The render loop has a nuanced rate limiting strategy that deserves emphasis:

- **Playback:** 60fps cap (16.67ms) regardless of `hasActiveVideo`, because a 120Hz display
  would otherwise render at 120fps with cache-hit frames for zero visual benefit
- **Scrubbing:** 30fps baseline, but RVFC bypasses it immediately when a new frame is ready
- **First frame:** `lastRenderTime` is reset to 0 when scrubbing starts or playback resumes,
  ensuring the first frame renders immediately

### 7.7 Proactive Preloading

`VideoSyncManager.preloadPausedJumpNeighborhood()` pre-seeks videos for clips near the
playhead even when paused. It looks 0.35s behind and 1.5s ahead, preloading up to 3 clips.
This means clicking to a new timeline position shows the correct frame faster because the
video was already seeking to that position.

### 7.8 Scrub Grace Period

When scrubbing stops, `LayerCollector` keeps the HTMLVideo scrub preview path active for
150ms (`SCRUB_GRACE_MS`). This gives the settle-seek time to complete before switching to
the WebCodecs path, preventing a brief flash of an incorrect WebCodecs frame.

---

## 8. Summary of Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| HTMLVideoElement as default decoder | Maximum compatibility, browser handles codec negotiation |
| WebCodecs as opt-in/export path | Better frame accuracy but requires MP4Box demuxing |
| Zero-copy via `importExternalTexture` | Performance-critical; avoids GPU readback |
| Firefox copied-frame workaround | Stability over performance for minority browser |
| Don't destroy GPU textures | Safety against use-after-destroy GPU crashes |
| Scrubbing cache at 30fps quantization | Balance between cache hit rate and memory usage |
| Skip frame caching during playback | GPU bandwidth for rendering > cache building |
| Triple-buffered seek targets | Never lose a seek request during rapid scrubbing |
| 500ms seek timeout for export | Support slow codecs (AV1) without hanging |
| RVFC as presented-frame signal | More reliable than `seeked` event for frame readiness |
| Idle suppression after reload | Allow GPU warmup to complete before idling |
| 3-stage settle state machine | Ensure correct frame after scrub with escalating recovery |

---

## 9. Key File Reference

| Concern | Primary File |
|---------|-------------|
| Render loop | `src/engine/render/RenderLoop.ts` |
| Layer collection & decode priority | `src/engine/render/LayerCollector.ts` |
| GPU compositing | `src/engine/render/Compositor.ts` |
| Nested composition rendering | `src/engine/render/NestedCompRenderer.ts` |
| Render orchestration | `src/engine/render/RenderDispatcher.ts` |
| Video seeking & sync | `src/services/layerBuilder/VideoSyncManager.ts` |
| Audio sync | `src/services/layerBuilder/AudioSyncHandler.ts` |
| Texture management | `src/engine/texture/TextureManager.ts` |
| Scrubbing & composite cache | `src/engine/texture/ScrubbingCache.ts` |
| Playback health monitoring | `src/services/playbackHealthMonitor.ts` |
| Firefox video fallback | `src/engine/render/htmlVideoPreviewFallback.ts` |
| Frame export pipeline | `src/engine/export/FrameExporter.ts` |
| WebCodecs encoder | `src/engine/export/VideoEncoderWrapper.ts` |
| Export video seeking | `src/engine/export/VideoSeeker.ts` |
| WebCodecs player (decode) | `src/engine/WebCodecsPlayer.ts` |
| Parallel export decode | `src/engine/ParallelDecodeManager.ts` |
| Native FFmpeg decoder | `src/services/nativeHelper/NativeDecoder.ts` |
| Runtime playback registry | `src/services/mediaRuntime/runtimePlayback.ts` |
| Feature flags | `src/engine/featureFlags.ts` |
| Video frame tracking | `src/engine/video/VideoFrameManager.ts` |
| Codec helpers | `src/engine/export/codecHelpers.ts` |
