# Consensus Report A: Architecture & Design Patterns

## Meta

- **Focus:** Architecture decisions, design patterns, and structural recommendations for MasterSelects as a browser-based NLE
- **Sources:** Agent 1 (Decode/Seeking), Agent 2 (Multi-Track Compositing), Agent 3 (Playback Sync), Agent 4 (Memory/Performance), Agent 5 (Export/Encoding), Agent 6 (Browser Compat/Fallbacks), Agent 7 (Full Codebase Exploration)
- **Agreement levels:** KONSENS (6-7 agents agree), MEHRHEIT (4-5 agents), EINZELMEINUNG (1-3 agents)

---

## 1. KONSENS: The Multi-Tier Decode Fallback Chain Is the Core Architectural Strength

**Agents 1, 2, 6, 7** describe the four-tier decode hierarchy in detail; **Agents 3, 4, 5** reference it indirectly. All seven agents treat this as the defining architectural pattern:

```
NativeHelper (Rust/FFmpeg) -> WebCodecs (VideoDecoder) -> HTMLVideoElement -> ScrubbingCache
```

**Why this matters:** Every agent recognizes that no single browser API can serve all needs of an NLE. The layered fallback chain means MasterSelects degrades gracefully rather than failing. Agent 6 maps this to 10+ concrete fallback scenarios (preview, export, scrub, metadata, audio detection) and concludes: *"Die Architektur ist bemerkenswert robust: Jeder Pfad hat mindestens einen Fallback, und das System degradiert graceful statt zu crashen."*

**Design Principle:** Build for graceful degradation with ranked decoder paths. Never depend on a single API.

---

## 2. KONSENS: HTMLVideoElement Is Necessary but Insufficient

All 7 agents agree that `HTMLVideoElement` is the right *default* for preview playback due to maximum compatibility, but all also document its fundamental limitations:

| Limitation | Cited by |
|-----------|----------|
| `currentTime` is not frame-accurate | Agents 1, 3, 7 |
| Seeking is keyframe-limited (GOP structure) | Agents 1, 3, 6 |
| GPU surface goes cold after reload | Agents 1, 2, 3, 6, 7 |
| `seeked` event does not confirm frame presentation | Agents 1, 3, 7 |
| Browser controls decode scheduling (no app control) | Agents 1, 2, 5 |
| Concurrent decoder limits per origin | Agents 2, 4, 6 |

**KONSENS conclusion:** HTMLVideoElement must be augmented with:
1. Tolerance-based time comparison (never exact equality)
2. RVFC for frame presentation confirmation
3. A cache layer to cover seeking gaps
4. WebCodecs as the long-term "escape hatch"

Agent 1 states it most clearly: *"HTMLVideoElement was designed for media playback, not frame-accurate editing. Browser NLEs must fight against its abstractions at every level."*

---

## 3. KONSENS: requestVideoFrameCallback (RVFC) Over `seeked` Events

**Agents 1, 3, 7** provide detailed analysis; **Agents 2, 5, 6** reference RVFC usage. All agree:

- `seeked` fires when the decoder finishes, NOT when the frame is composited to the GPU surface
- RVFC fires when the frame is actually *presented*, making it the authoritative signal
- RVFC also enables rate-limiter bypass (Agent 3, 7): when a new frame is ready, the scrub rate limiter is bypassed for immediate display

**No disagreement** on this point. RVFC is universally recognized as the correct signal for frame-readiness in a browser NLE.

---

## 4. KONSENS: Audio Master Clock Pattern

**Agents 3, 7** analyze this in depth. **Agent 2** references it. The pattern:

- During 1x forward playback, the audio element runs freely
- The playhead derives its position from `audio.currentTime`
- Video corrects drift relative to the audio-derived playhead

Agent 3 explains the rationale: *"Audio is perceptually unforgiving. A single dropped video frame is barely noticeable, but a 20ms audio glitch is immediately audible."*

**System-clock fallback** when no audio is present uses `performance.now()` delta timing with a 100ms cap to handle tab backgrounding.

---

## 5. KONSENS: The Hybrid Seeking Strategy (fastSeek + Debounced Precise Seek)

**Agents 1, 3, 7** document this in full detail. **Agents 2, 6** reference it. All agree on the two-phase approach:

- **Phase 1:** `fastSeek()` for instant keyframe preview (< 10ms)
- **Phase 2:** Debounced `currentTime` assignment (90-120ms) for exact frame

Agent 1 provides the key insight: *"The 120ms debounce ensures precise seeks only fire when the user pauses scrubbing. During fast scrubbing, only keyframe previews are shown, avoiding the decoder thrashing that would occur from queuing hundreds of precise seeks."*

**Adaptive throttling** based on `fastSeek` availability (Agent 1): without fastSeek support, throttle intervals increase from 16-50ms to 60-110ms.

---

## 6. KONSENS: GPU Surface Warmup After Page Reload

**Agents 1, 2, 3, 6, 7** all identify this as a critical problem and describe the same solution:

1. After reload, `importExternalTexture` returns valid but black textures
2. The ONLY fix is `video.play()` to activate the GPU compositor
3. `videoGpuReady` WeakSet tracks which videos are warmed
4. Proactive warmup for upcoming clips (lookahead 1.5s playback, 0.9s scrubbing)
5. `createImageBitmap()` as forced-decode fallback (Agent 1, 7)

Agent 7 adds: *"RenderLoop.idleSuppressed keeps the engine rendering during warmup (prevents idle timeout)"* -- a subtle but important detail that only Agent 7 caught in full.

---

## 7. KONSENS: Multi-Tier Caching Is Essential

**All 7 agents** reference the caching architecture. Agents 1, 2, 4, 7 provide the most detail:

| Tier | Purpose | Budget | Cited by |
|------|---------|--------|----------|
| Per-time scrubbing cache | Instant frame at exact position | 300 frames, ~2.4 GB VRAM | All |
| Last-frame cache | Most recent frame per video | 1 per video | Agents 1, 4, 7 |
| RAM preview composite | Pre-rendered composites | 900 frames, 512 MB | Agents 4, 7 |
| GPU frame cache | Avoid CPU-to-GPU re-upload | 60 frames, ~500 MB | Agents 2, 4, 7 |

**Key architectural decision (KONSENS):** Caching is SKIPPED during playback to preserve GPU bandwidth. Only during scrubbing/paused states are frames cached. Agents 2, 4, 7 all cite this tradeoff explicitly.

---

## 8. KONSENS: The "Don't Destroy GPU Textures" Pattern

**Agents 4, 7** analyze this in depth. **Agent 2** references it:

GPU textures are NOT explicitly destroyed (except NativeDecoder dynamic textures). Instead, Map references are cleared and GC handles deallocation. The rationale (Agent 7): *"This pattern trades potential temporary VRAM overuse for safety against use-after-destroy bugs, which would crash the GPU process."*

**The exception** (Agents 4, 7): `dynamicTextures` from NativeDecoder ARE explicitly destroyed because MasterSelects owns the full lifecycle.

---

## 9. KONSENS: Dual Shader Pipeline for texture_external vs texture_2d

**Agents 2, 6, 7** describe this architecture in detail. Because WebGPU's `texture_external` and `texture_2d<f32>` are different WGSL types, two parallel composite pipelines must be maintained:

- `getCompositePipeline()` for images, text, cached frames
- `getExternalCompositePipeline()` for live video frames

Agent 2 notes that bind group caching works for static layers but must be invalidated every frame for video layers because the external texture changes.

---

## 10. KONSENS: Seek Queue with Triple-Buffered Targets

**Agents 1, 3, 7** all document the triple-buffer seek architecture:

- `latestSeekTargets` -- most recent desired position
- `pendingSeekTargets` -- currently active seek
- `queuedSeekTargets` -- next seek after current completes

Agent 7 calls this a "triple-buffer approach" that "prevents seeks from being lost during rapid scrubbing while ensuring the most recent target always wins." Agent 1 adds the retargeting logic: in-flight seeks are only interrupted after age + drift thresholds are exceeded.

---

## 11. MEHRHEIT: PlaybackHealthMonitor as Self-Healing Architecture

**Agents 1, 3, 4, 7** describe the monitor in detail. **Agent 2** references it. The 7 anomaly types and escalation protocol are consistently documented:

- 8 anomaly types (FRAME_STALL, SEEK_STUCK, WARMUP_STUCK, etc.)
- 3 anomalies within 12 seconds -> escalated recovery
- Per-clip cooldowns prevent recovery loops
- `requestIdleCallback`-based polling to avoid stealing frame budget

**Design Principle (MEHRHEIT):** Self-healing systems are essential for browser NLEs because the underlying APIs (HTMLVideoElement, WebGPU) have non-deterministic failure modes that cannot be prevented, only detected and recovered from.

---

## 12. MEHRHEIT: Mutable Playhead Object (Not Zustand Store)

**Agents 3, 7** analyze this in depth. The playhead position during playback lives in a plain mutable object (`playheadState`), NOT in the Zustand store:

- Updated every RAF tick (~60fps) for the render pipeline
- Zustand store throttled to ~30fps for React UI subscribers
- Prevents 60fps React re-renders across the entire component tree

Agent 3: *"60fps updates would thrash React subscribers."* This separation of render-critical state from UI state is a crucial architectural decision.

---

## 13. MEHRHEIT: Two-Tier Export Architecture

**Agents 5, 6, 7** describe the export pipeline. **Agent 2** references it:

| Path | Technology | Use Case |
|------|-----------|----------|
| WebCodecs | Browser-native VideoEncoder + JS muxers | Delivery (H.264, H.265, VP9, AV1) |
| FFmpeg WASM | Emscripten build | Professional (ProRes, DNxHR, FFV1) |

Both share the same frame-by-frame WebGPU rendering pipeline. The zero-copy export path (`OffscreenCanvas -> VideoFrame -> VideoEncoder`) avoids CPU roundtrips entirely.

---

## 14. MEHRHEIT: Seamless Cut Transitions via Video Element Handoff

**Agents 2, 3, 4, 7** describe the handoff system: when sequential clips on the same track share the same source file (split clips), the outgoing video element keeps playing through the cut boundary. Agent 7: *"This is a professional-grade feature that avoids the audible pop and visual glitch of stopping and restarting a video decoder at cut points."*

---

## 15. EINZELMEINUNG: Code Duplication in Layer Collection (Agent 7)

Only **Agent 7** identifies this structural weakness in detail: the video fallback cascade is duplicated in three files:

1. `LayerCollector.tryHTMLVideo()` (~370 lines)
2. `NestedCompRenderer.collectNestedLayerData()` (~390 lines)
3. `RenderDispatcher.renderToPreviewCanvas()` (~200 lines)

Agent 7: *"Each has slightly different conditions for `allowLiveVideoImport`, `allowConfirmedFrameCaching`, etc. This is a maintenance risk -- a fix in one place must be replicated in the other two."*

**Recommendation:** Extract a shared `VideoFrameAcquisition` service.

---

## 16. EINZELMEINUNG: No visibilitychange Handling (Agent 4)

Only **Agent 4** notes that MasterSelects does not handle `document.visibilitychange` events. Background tab throttling can cause WebGPU device loss during export. Agent 4 identifies this as a gap, though the export error message does warn users to keep the tab in focus.

---

## 17. EINZELMEINUNG: VRAM Budget Not Dynamic (Agents 4, 7)

**Agents 4 and 7** note that the scrubbing cache budget (300 frames = ~2.4 GB VRAM at 1080p) is static. On integrated GPUs with 4 GB total VRAM, this can cause device loss. No dynamic adjustment based on available VRAM or device capabilities exists.

---

## Top 10 Design Principles for HTMLVideoElement in a Browser NLE

Synthesized from all 7 agents, ranked by consensus strength:

### 1. Never Trust `currentTime` Precision (KONSENS)
Use tolerance-based comparison with context-dependent thresholds (15ms paused, 40ms dragging, 300ms playback drift). Agents 1, 3, 7.

### 2. Use RVFC as the Authoritative Frame Signal (KONSENS)
`requestVideoFrameCallback` confirms actual frame presentation. `seeked` is necessary but not sufficient. Agents 1, 3, 5, 7.

### 3. Build a Multi-Tier Fallback Chain (KONSENS)
No single decode API works everywhere. Build NativeHelper -> WebCodecs -> HTMLVideoElement -> Cache fallbacks. Agents 1, 2, 5, 6, 7.

### 4. Cache Aggressively During Scrub, Skip During Playback (KONSENS)
GPU bandwidth is precious during playback. Cache frames only when paused/scrubbing. Serve cached frames to avoid black flashes. Agents 1, 2, 4, 7.

### 5. Serialize and Coalesce Seeks (KONSENS)
Maintain a seek queue with pending/queued buffers. Rapid scrubbing generates hundreds of seeks -- only the latest matters. Agents 1, 3, 7.

### 6. Warm the GPU Surface Proactively (KONSENS)
After page reload, `video.play()` is the only way to activate the GPU surface. Warm upcoming clips 1-2 seconds ahead. Agents 1, 2, 3, 6, 7.

### 7. Audio Leads, Video Follows (KONSENS)
Use audio as the master clock during playback. Video drift is corrected with threshold-based seeks. Audio glitches are less tolerable than dropped frames. Agents 3, 7.

### 8. Separate Render-Critical State from UI State (MEHRHEIT)
Keep the playhead in a mutable object for the render pipeline, throttle Zustand store updates for React. Agents 3, 7.

### 9. Monitor and Self-Heal (MEHRHEIT)
Detect anomalies (stuck seeks, frame stalls, cold GPU surfaces) and auto-recover with escalation. Browser APIs have non-deterministic failure modes. Agents 1, 3, 4, 7.

### 10. Feature-Detect, Don't Browser-Sniff (MEHRHEIT)
Check for `fastSeek`, RVFC, WebCodecs, WebGPU at runtime. Only use user-agent sniffing for known browser bugs (Firefox black texture workaround). Agents 1, 6, 7.

---

## Identified Architectural Weaknesses

### High Priority (MEHRHEIT agreement)

1. **Static VRAM Budget** (Agents 4, 7): The scrubbing cache allocates up to 2.4 GB VRAM without checking available GPU memory. On integrated GPUs this can cause device loss. **Recommendation:** Query `GPUAdapterInfo` and scale budgets accordingly.

2. **No Video Element Pooling** (Agents 4, 7): Each clip creates its own `HTMLVideoElement`. Timelines with 50+ clips can hit browser decoder limits. **Recommendation:** Implement a video element pool with LRU eviction for off-screen clips.

3. **FFmpeg WASM Memory Bottleneck** (Agents 5, 7): All raw frames are concatenated in memory before encoding (~14.9 GB for 60s 1080p). **Recommendation:** Stream frames incrementally to the FFmpeg virtual filesystem.

### Medium Priority (EINZELMEINUNG but well-reasoned)

4. **Layer Collection Code Duplication** (Agent 7): The video frame acquisition cascade is duplicated in 3 files (~960 lines total). **Recommendation:** Extract a shared `VideoFrameAcquisition` or `FrameResolver` service.

5. **No visibilitychange Handling** (Agent 4): Background tab throttling can cause export failures and device loss. **Recommendation:** Detect background state, warn users, optionally pause export.

6. **No Mobile-Specific Resource Limits** (Agent 4): iOS Safari has ~1 GB memory limit. Cache budgets are not adjusted for mobile. **Recommendation:** Detect mobile context and reduce cache budgets (e.g., 100 scrub frames instead of 300).

7. **GC-Reliant Texture Cleanup** (Agents 4, 7): Most GPU textures rely on GC. Under memory pressure, VRAM can accumulate before GC cycles. **Recommendation:** Consider explicit `destroy()` for caches with deterministic eviction paths where GPU reference lifetimes are known.

---

## Contradictions Between Agents

### Minor: fastSeek Availability

- Agent 1 states fastSeek is available in "Firefox, Chrome 130+"
- Agent 7 states fastSeek is "Safari/Firefox only, Chrome lacks it"
- Agent 6 does not specify browser support

**Resolution:** Chrome added `fastSeek` support around version 130 (late 2024). Both statements were likely accurate at different knowledge cutoffs. The codebase correctly feature-detects rather than browser-sniffs.

### Minor: Audio Drift Threshold

- Agent 3 states audio drift > 300ms triggers a hard reset, 50-300ms is tolerated
- Agent 7 states audio drift > 0.3 triggers resync, > 0.05 is logged

These describe the same behavior in different terms (0.3s = 300ms, 0.05s = 50ms). No actual contradiction.

### Minor: WebCodecs as Default

- Agent 7 states HTMLVideoElement is the "last resort" in the decode hierarchy
- Agents 1, 6 note that `useFullWebCodecsPlayback` is `false`, making HTMLVideoElement the *default* for most users

**Resolution:** Both are correct. HTMLVideoElement is the default because the WebCodecs flag is off, but architecturally it is the lowest-priority path. The fallback chain runs top-to-bottom, and since NativeHelper and WebCodecs are typically unavailable, HTMLVideoElement wins by elimination.

---

## Summary: What Makes This Architecture Work

The consensus across all 7 agents is that MasterSelects succeeds as a browser NLE not through any single API or technique, but through the *layering* of defensive strategies:

1. **Multiple decode paths** that fall through gracefully
2. **Multiple cache tiers** that prevent visual artifacts
3. **Multiple time references** (requested, pending, presented) that handle async decode reality
4. **Multiple recovery mechanisms** that self-heal from non-deterministic failures
5. **Multiple render modes** (playback/scrub/idle/export) with distinct optimization strategies

The architecture's defining characteristic is *defense in depth*. Every subsystem assumes its dependencies will misbehave and has fallbacks for when they do. This is the essential architectural pattern for any application that builds professional-grade functionality on top of browser APIs not designed for that purpose.
