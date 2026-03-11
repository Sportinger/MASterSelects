# HTMLVideoElement in einer Browser-NLE: Decoding, Seeking & Frame-Accuracy

## 1. Executive Summary

HTMLVideoElement was designed for media playback, not frame-accurate editing. Browser NLEs must fight against its abstractions at every level: seeking lands on keyframes instead of arbitrary frames, `currentTime` precision varies by codec and browser, the GPU surface goes cold after page reload, and race conditions between concurrent seeks cause decoder stalls. MasterSelects addresses these limitations through a layered architecture: a hybrid seeking strategy (fastSeek + deferred precise seek), requestVideoFrameCallback-based frame confirmation, a multi-tier GPU texture cache, warmup sequences to activate cold GPU surfaces, and an optional full WebCodecs pipeline that bypasses HTMLVideoElement entirely for frame-accurate decode.

---

## 2. HTMLVideoElement.currentTime: Precision & Limitations

### 2.1 What currentTime Actually Reports

`currentTime` is a `double` representing seconds, but its precision is **not** frame-accurate by specification. The browser reports the **presentation timestamp of the most recently rendered frame**, which depends on:

- The codec's GOP (Group of Pictures) structure
- The browser's internal decoder state
- Whether the video has actually decoded a frame at the requested position

Setting `video.currentTime = X` initiates a seek, but the value you read back after the seek completes may differ from X. The browser decodes from the nearest preceding keyframe and presents the frame whose PTS is closest to X, but this is implementation-dependent.

### 2.2 How MasterSelects Handles Precision

MasterSelects uses a tolerance-based comparison throughout, never relying on exact equality:

```typescript
// From VideoSyncManager.ts — typical precision thresholds
private static readonly PAUSED_PRECISE_SEEK_THRESHOLD = 0.015; // 15ms
// Drift correction during playback
const driftThreshold = hasSpeedKeyframes ? 1.5 : 0.3; // 300ms normal, 1.5s with speed keyframes
// Scrub threshold
const seekThreshold = ctx.isDraggingPlayhead ? 0.04 : 0.02; // 40ms drag, 20ms precise
```

The `safeSeekTime` method in `VideoSyncManager` clamps seeks away from the exact end of the video, preventing H.264 B-frame decoder stalls:

```typescript
// VideoSyncManager.ts:131-135
private safeSeekTime(video: HTMLVideoElement, time: number): number {
  const dur = video.duration;
  if (!isFinite(dur) || dur <= 0) return Math.max(0, time);
  return Math.max(0, Math.min(time, dur - 0.001));
}
```

This 1ms clamp prevents a real decoder issue: H.264 B-frame decoders stall when seeking to exactly `video.duration` because they wait for reference frames that don't exist.

---

## 3. GOP Structure and Why Seeking is Keyframe-Limited

### 3.1 The I-Frame Problem

Video codecs compress frames into Groups of Pictures. Only **I-frames (keyframes)** contain full image data; P-frames and B-frames store differences relative to other frames. To decode frame N, the decoder must start at the preceding I-frame and decode every frame in sequence up to N.

**Implications for NLE seeking:**
- Seeking to frame 150 in a 250-frame GOP requires decoding ~150 frames
- YouTube/phone videos typically have 5-7 second keyframe intervals
- This means a single seek can require decoding 150-210 frames before showing the target
- During fast scrubbing, this latency (100-300ms per seek) makes the timeline feel sluggish

### 3.2 MasterSelects' Hybrid Seeking Strategy

MasterSelects solves this with a two-phase approach documented in `VideoSyncManager.throttledSeek()`:

```
Phase 1: fastSeek() -> instant keyframe feedback (<10ms, shows nearest I-frame)
Phase 2: deferred precise seek via currentTime (debounced 120ms, exact frame)
```

**Phase 1 — fastSeek (keyframe preview):**
```typescript
// VideoSyncManager.ts:1807-1812
// Phase 1: Instant keyframe feedback via fastSeek.
// For all-intra codecs this IS the exact frame. For long-GOP codecs
// this shows the nearest keyframe — better than a stale cached frame.
fastSeek(this.safeSeekTime(video, time));
```

`video.fastSeek()` is a non-standard API (available in Firefox, Chrome 130+) that seeks to the nearest keyframe without decoding intermediate frames. For all-intra codecs (ProRes, DNxHR), this gives the exact frame. For long-GOP codecs (H.264, H.265), it shows the nearest I-frame.

**Phase 2 — Precise seek (debounced):**
```typescript
// VideoSyncManager.ts:1824-1841
this.latestSeekTargets[clipId] = time;
clearTimeout(this.preciseSeekTimers[clipId]);
this.preciseSeekTimers[clipId] = setTimeout(() => {
  const target = this.latestSeekTargets[clipId];
  // Only do precise seek if the fastSeek landed far from the target
  if (target !== undefined && Math.abs(video.currentTime - target) > 0.01) {
    video.currentTime = this.safeSeekTime(video, target);
    this.registerRVFC(clipId, video);
  }
}, 120);
```

The 120ms debounce ensures precise seeks only fire when the user pauses scrubbing. During fast scrubbing, only keyframe previews are shown, avoiding the decoder thrashing that would occur from queuing hundreds of precise seeks.

### 3.3 Browser Differences in fastSeek

`VideoSyncManager.getFastSeek()` probes for fastSeek support at runtime:

```typescript
// VideoSyncManager.ts:137-142
private getFastSeek(video: HTMLVideoElement): ((time: number) => void) | null {
  const fastSeek = (video as HTMLVideoElement & {
    fastSeek?: (time: number) => void;
  }).fastSeek;
  return typeof fastSeek === 'function' ? fastSeek.bind(video) : null;
}
```

When fastSeek is unavailable (older Chrome, some mobile browsers), the code falls back to rate-limited `currentTime` seeks with higher throttle intervals:

```typescript
// VideoSyncManager.ts:1792-1804 — adaptive throttle based on fastSeek support
const threshold = ctx.isDraggingPlayhead
  ? supportsFastSeek
    ? dragDrift >= 1 ? 16 : dragDrift >= 0.35 ? 28 : 50    // ms between seeks
    : dragDrift >= 1 ? 60 : dragDrift >= 0.35 ? 85 : 110   // slower without fastSeek
  : 33;
```

---

## 4. Seeking Events: seeked, seeking, timeupdate

### 4.1 Event Sequence

When `video.currentTime` is set:
1. `seeking` fires immediately (video.seeking becomes true)
2. Browser decodes from nearest keyframe to target
3. `seeked` fires when decode completes (video.seeking becomes false)
4. `timeupdate` fires on next presentation

### 4.2 Why seeked Is Insufficient

The `seeked` event signals that the browser has finished seeking, but it does NOT guarantee:
- The decoded frame is actually composited to the GPU surface
- The frame is available for WebGPU `importExternalTexture()`
- The frame matches the exact requested time

MasterSelects uses `requestVideoFrameCallback` (RVFC) instead of `seeked` as the definitive signal:

```typescript
// VideoSyncManager.ts:1865-1886
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
      engine.markVideoFramePresented(video, presentedTime, clipId);
      engine.captureVideoFrameAtTime(video, presentedTime, clipId);
      engine.requestNewFrameRender();
    });
  }
}
```

RVFC fires when the frame is **actually presented to the compositor** -- more accurate than `seeked` which fires when the decoder finishes but before the frame reaches the GPU.

### 4.3 The seeked Event as Fallback

MasterSelects still listens to `seeked` as a fallback for queued seek coalescing:

```typescript
// VideoSyncManager.ts:723-733
private armSeekedFlush(clipId: string, video: HTMLVideoElement): void {
  if (this.seekedFlushArmed.has(clipId)) return;
  this.seekedFlushArmed.add(clipId);
  video.addEventListener('seeked', () => {
    this.seekedFlushArmed.delete(clipId);
    this.flushQueuedSeekTarget(clipId, video, 'seeked');
  }, { once: true });
}
```

This arms a one-shot `seeked` listener that flushes any queued seek target. When the RVFC fires first (normal case), the RVFC handler also flushes the queue. The `seeked` handler serves as insurance for browsers or situations where RVFC doesn't fire (e.g., seeking to the same keyframe).

---

## 5. readyState and Frame Availability

### 5.1 readyState Values

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | HAVE_NOTHING | No data available |
| 1 | HAVE_METADATA | Duration, dimensions known |
| 2 | HAVE_CURRENT_DATA | Current frame available |
| 3 | HAVE_FUTURE_DATA | Next frame also available |
| 4 | HAVE_ENOUGH_DATA | Enough data for uninterrupted playback |

### 5.2 MasterSelects' readyState Strategy

The codebase consistently checks `readyState >= 2` as the minimum for rendering:

```typescript
// LayerCollector.ts:458 — gate for HTML video import
if (video.readyState >= 2) {
  // ... import video texture
}

// VideoSeeker.ts:135 — export frame readiness
if (!video.seeking && video.readyState >= 3) {
  // Frame is ready for export
}
```

For warmup/decoder initialization, the thresholds differ:

```typescript
// webCodecsHelpers.ts:118 — readyState >= 3 means decoder has produced a frame
if (video.readyState >= 3) { // HAVE_FUTURE_DATA
  resolve();
  return;
}

// webCodecsHelpers.ts:211-214 — canplaythrough for export readiness
export function waitForVideoReady(video: HTMLVideoElement, timeout = 2000): Promise<void> {
  return new Promise((resolve) => {
    if (video.readyState >= 4) { // HAVE_ENOUGH_DATA
      resolve();
      return;
    }
    video.addEventListener('canplaythrough', handler, { once: true });
  });
}
```

### 5.3 readyState Drop Recovery

The `PlaybackHealthMonitor` detects `readyState` drops during playback:

```typescript
// playbackHealthMonitor.ts:230-234
if (isPlaying) {
  for (const clip of htmlHealthVideoClips) {
    const video = clip.source!.videoElement!;
    if (video.readyState < 2 && !video.seeking) {
      this.recordAnomaly('READYSTATE_DROP', clip.id, `readyState=${video.readyState}`);
    }
  }
}
```

When `readyState` drops below 2 outside of a seek, `VideoSyncManager` forces a decode via play/pause:

```typescript
// VideoSyncManager.ts:1257-1276
private forceVideoFrameDecode(clipId: string, video: HTMLVideoElement): void {
  video.muted = true;
  video.play()
    .then(() => {
      video.pause();
      video.currentTime = currentTime;
      engine.requestRender();
    })
    .catch(() => {
      video.currentTime = currentTime + 0.001; // tiny seek to trigger decode
      engine.requestRender();
    });
}
```

---

## 6. Race Conditions beim Seeking

### 6.1 Multiple Concurrent Seeks

The biggest challenge in an NLE timeline is rapid scrubbing: the user drags the playhead, generating dozens of seek requests per second. Without coordination:

- Each `video.currentTime = X` cancels the previous seek
- The browser's internal decoder state becomes unpredictable
- `seeked` events fire out of order or not at all
- The GPU surface shows stale or wrong frames

### 6.2 MasterSelects' Seek Queue Architecture

`VideoSyncManager` maintains three layers of seek state per clip:

```typescript
// VideoSyncManager.ts:62-66
private latestSeekTargets: Record<string, number> = {};     // Most recent desired position
private pendingSeekTargets: Record<string, number> = {};     // Currently active seek
private pendingSeekStartedAt: Record<string, number> = {};   // When the active seek started
private queuedSeekTargets: Record<string, number> = {};      // Next seek to issue after current completes
private seekedFlushArmed = new Set<string>();                 // Whether seeked listener is active
```

**Seek coalescing logic:**
1. If a seek is already pending (`pendingSeekTargets[clipId]` exists), new seeks go into `queuedSeekTargets`
2. When the current seek completes (via RVFC or seeked), the queued target is flushed
3. If the queued target matches where the video already is (within 10ms), no new seek is issued
4. Retargeting (replacing the in-flight seek) only happens after age + drift thresholds are exceeded

```typescript
// VideoSyncManager.ts:555-581
private shouldRetargetPendingSeek(...): boolean {
  const pendingAge = now - (this.pendingSeekStartedAt[clipId] ?? now);
  const targetDrift = Math.abs(pendingTarget - nextTargetTime);
  if (isDragging && !allowInFlightRetarget) {
    if (displayedDriftSeconds >= 1.2) return pendingAge >= 65 && targetDrift >= 0.12;
    if (displayedDriftSeconds >= 0.5) return pendingAge >= 95 && targetDrift >= 0.16;
    return pendingAge >= 170 && targetDrift >= 0.28;
  }
  return pendingAge >= (isDragging ? 90 : 120) && targetDrift >= (isDragging ? 0.12 : 0.2);
}
```

### 6.3 Hung Seek Recovery

The health monitor detects stuck seeks (video.seeking true for > 2 seconds):

```typescript
// playbackHealthMonitor.ts:209-227
// 4. SEEK_STUCK
for (const clip of htmlHealthVideoClips) {
  const video = clip.source!.videoElement!;
  if (video.seeking) {
    const seekStart = this.seekStartTimes.get(clip.id);
    if (seekStart && now - seekStart > SEEK_STUCK_MS) {
      this.recoverSeekStuck(video);
    }
  }
}
```

Recovery is a re-seek to the same position (`video.currentTime = video.currentTime`), which forces the browser to restart its internal seek pipeline. If a clip accumulates 3+ anomalies within 12 seconds, the monitor escalates to a full clip recovery (GPU surface reset + targeted warmup).

---

## 7. GPU Surface Cold Start Problem

### 7.1 The Problem

After a page reload, `HTMLVideoElement` instances that were restored from project state have a cold GPU surface. All synchronous rendering APIs return black frames:
- `importExternalTexture(video)` returns a valid GPUExternalTexture, but the pixel data is black
- `copyExternalImageToTexture({ source: video })` copies black pixels
- `new VideoFrame(video)` creates a VideoFrame with black content
- `canvas.drawImage(video)` draws black

The **only** API that forces actual frame decode is `createImageBitmap(video)` (async) or `video.play()`.

### 7.2 MasterSelects' Warmup System

The `VideoSyncManager` implements a lazy warmup triggered on first render attempt:

```typescript
// VideoSyncManager.ts:1397-1416
// Warmup: after page reload, video GPU surfaces are empty.
// importExternalTexture, canvas.drawImage, etc. all return black.
// The ONLY fix is video.play() to activate the GPU compositor.
if (!ctx.isPlaying && !video.seeking && hasSrc && cooldownOk &&
    video.played.length === 0 && !this.warmingUpVideos.has(video)) {
  this.startTargetedWarmup(clip.id, video, timeInfo.clipTime, {
    proactive: false,
    requestRender: true,
  });
  return;
}
```

The warmup sequence:
1. Seek to target time
2. `video.play()` to activate the GPU decoder
3. Wait for RVFC to confirm frame presentation
4. Capture the presented frame to the scrubbing cache
5. Mark the video as GPU-ready
6. Pause

The `LayerCollector` tracks GPU readiness with a WeakSet:

```typescript
// LayerCollector.ts:384,596-600
private videoGpuReady = new WeakSet<HTMLVideoElement>();

// Only mark as ready when video is actually playing (not just seeked)
if (!video.paused && !video.seeking) {
  this.videoGpuReady.add(video);
}
```

### 7.3 Proactive Warmup for Cut Boundaries

Split clips each have their own HTMLVideoElement. When playback crosses a cut boundary, the new clip's video has a cold GPU surface. MasterSelects warms upcoming clips proactively:

```typescript
// VideoSyncManager.ts:1892-1893
private static readonly LOOKAHEAD_TIME = 1.5; // seconds ahead during playback
private static readonly SCRUB_WARMUP_LOOKAHEAD = 0.9; // seconds during scrubbing
```

---

## 8. The Multi-Tier Texture Cache

### 8.1 Cache Architecture

`ScrubbingCache` provides three cache tiers to avoid black frames during seeks:

| Tier | Key | Purpose | Max Size |
|------|-----|---------|----------|
| Per-time scrubbing cache | `videoSrc:quantizedTime` | Instant frame at exact position during re-visits | 300 frames (~10s @ 30fps) |
| Last-frame cache | `WeakMap<HTMLVideoElement>` | Most recent frame per video, shown during seeks | 1 per video |
| Composite cache (RAM Preview) | `quantizedTime` | Pre-rendered composite frames | 900 frames (30s @ 30fps) |

### 8.2 Time Quantization

Cache keys quantize time to 30fps frame boundaries for better hit rates:

```typescript
// ScrubbingCache.ts:56-58
private quantizeToFrame(time: number): string {
  return (Math.round(time * this.SCRUB_CACHE_FPS) / this.SCRUB_CACHE_FPS).toFixed(3);
}
```

Two scrub positions within the same frame (e.g., 1.5001s and 1.5009s) map to the same key.

### 8.3 LayerCollector's Frame Selection Priority

When rendering a video layer, `LayerCollector.tryHTMLVideo()` selects the frame source in this priority:

1. **Per-time scrubbing cache** (exact match or nearest within 6-12 frames) during seeking
2. **Last-frame cache** (within 0.35s tolerance) during seeking
3. **Live importExternalTexture** when video is ready and not seeking
4. **Copied preview frame** (Firefox workaround for black texture imports)
5. **Final cache fallback** if live import fails

---

## 9. WebCodecs as the Escape Hatch

### 9.1 Why WebCodecs

HTMLVideoElement's fundamental limitation is that the application cannot control the decoder. WebCodecs (`VideoDecoder` + `EncodedVideoChunk`) gives direct access to:
- Frame-level decode control
- Keyframe identification via MP4Box demuxing
- Exact timestamp matching (microsecond precision)
- Decoder state management (reset, configure, flush)

### 9.2 MasterSelects' WebCodecs Integration

The `WebCodecsPlayer` class operates in two modes:

**Simple mode:** Wraps HTMLVideoElement with `new VideoFrame(video)` for zero-copy GPU texture import. The timeline still controls the video element; WebCodecsPlayer just captures frames.

**Full mode:** Uses MP4Box to demux the file, feeds `EncodedVideoChunk`s to `VideoDecoder`, manages a frame buffer, and provides frame-accurate seeking by decoding from the nearest keyframe:

```typescript
// WebCodecsPlayer.ts:1644-1703 (seek in full mode)
// Binary search for closest CTS match (O(log n))
const targetIndex = this.findSampleNearCts(targetTime);
const keyframeIndex = this.findKeyframeBefore(targetIndex);
// Reset decoder and decode from keyframe to target
this.decoder.reset();
this.decoder.configure(this.codecConfig!);
this.sampleIndex = targetIndex;
this.feedIndex = keyframeIndex;
this.feedPendingSeekSamples('seek');
```

The decoder output callback filters intermediate GOP frames, only publishing the target:

```typescript
// WebCodecsPlayer.ts:560-586
if (this.seekTargetUs !== null) {
  const diff = Math.abs(frame.timestamp - this.seekTargetUs);
  if (diff <= this.seekTargetToleranceUs) {
    this.currentFrame = frame; // Publish target frame
    this.seekTargetUs = null;
  } else {
    frame.close(); // Drop intermediate GOP traversal frame
  }
}
```

### 9.3 Current Status

Full WebCodecs playback is behind a feature flag (currently disabled by default):

```typescript
// featureFlags.ts:8
useFullWebCodecsPlayback: false,  // Preview runs HTML-only for now
```

When enabled, the system uses dedicated scrub and playback runtime sessions per clip, with the `VideoSyncManager` coordinating between HTML video seeking (for audio reference) and WebCodecs decoding (for frame-accurate video preview).

---

## 10. Browser-Specific Issues

### 10.1 Firefox: Black Texture Imports

Firefox intermittently samples imported HTML video textures as black during playback. MasterSelects works around this with an explicit copy path:

```typescript
// htmlVideoPreviewFallback.ts:25-28
// Firefox can intermittently sample imported HTML video textures as black
// during playback. Copying into a persistent texture is slower but stable.
const captured = scrubbingCache.captureVideoFrame(video, captureOwnerId);
```

### 10.2 Chrome: createImageBitmap as Forced Decode

After page reload, Chrome's GPU decoder surface is empty. `ScrubbingCache.captureVideoFrameViaImageBitmap()` uses the only API that forces actual decode:

```typescript
// ScrubbingCache.ts:280-289
// createImageBitmap is the ONLY browser API that forces actual frame decode
// After page reload, all sync APIs return black/empty data because
// Chrome defers frame decoding.
const bitmap = await createImageBitmap(video);
```

### 10.3 Mobile Considerations

Mobile browsers have additional constraints:
- Hardware decoder pools are smaller (often 1-2 simultaneous decoders)
- `fastSeek()` support varies
- autoplay restrictions require user gesture before `video.play()`
- Background tab throttling affects `requestVideoFrameCallback` delivery

MasterSelects handles autoplay restrictions by muting videos before warmup (`video.muted = true`).

---

## 11. Key Takeaways for Browser NLE Development

1. **Never trust `currentTime` precision** -- always use tolerance-based comparison (15-40ms thresholds depending on context)
2. **`seeked` is necessary but not sufficient** -- use `requestVideoFrameCallback` to confirm frame presentation
3. **GOP structure makes seeking inherently slow** -- use fastSeek for immediate keyframe feedback, debounced precise seek for exact frame
4. **The GPU surface goes cold on reload** -- only `video.play()` activates it; lazy warmup with RVFC confirmation
5. **Cache aggressively** -- per-time frame cache, last-frame fallback, and nearest-frame search prevent black flashes
6. **Serialize seeks** -- maintain a queue with coalescing to prevent decoder thrashing
7. **Monitor and recover** -- detect stuck seeks, frame stalls, readyState drops, and GPU surface issues; auto-recover with escalation
8. **WebCodecs is the long-term solution** -- provides frame-level decode control, but requires significant infrastructure (MP4Box demuxing, frame buffer management, seek target filtering)
