# Playback Pipeline Consensus Analysis

**Date:** 2026-03-07
**Method:** 6 parallel AI agents (3x Codex CLI / GPT-5.4, 3x Claude Explore) independently analyzed the same codebase
**Total raw findings:** 75 across all agents, deduplicated to 22 distinct issues

---

## UNANIMOUS (6/6 Agents)

These issues were independently identified by every single agent. Highest confidence.

### U1. No Last-Good-Frame Fallback — NULL = BLACK FRAME
- **Severity:** CRITICAL
- **Files:** `LayerCollector.ts:149-188`, `WebCodecsPlayer.ts:107-110`
- **Root Cause:** In full WebCodecs mode, if `getCurrentFrame()` returns null (during decoder reset, seek gap, clip transition), `LayerCollector` returns no layer data. HTMLVideo fallback is disabled (line 188). The compositor renders black. Every transient decode gap becomes a visible black flash.
- **Fix:** Keep a per-layer/source last-successful GPU texture or cloned VideoFrame. Reuse it until a new frame imports successfully. Do NOT re-enable HTML video fallback.
- **Affects:** Bug #1 (cuts), Bug #4 (scrubbing)

### U2. Shared WebCodecsPlayer Across Split Clips — Dual Seek Conflict
- **Severity:** CRITICAL
- **Files:** `clipSlice.ts:518-528`, `VideoSyncManager.ts:207-215`
- **Root Cause:** `splitClip()` shares one `WebCodecsPlayer` instance across both halves. One decoder cannot serve two active clip times. At cut boundaries, both clips call `advanceToTime()` in the same RAF tick with different target times. After `moveClip()`, split siblings can overlap or be on different tracks — the "never overlap" assumption is unenforced.
- **Fix:** Share parsed MP4/sample data, not the live decoder state. Fork a new decoder session when a second clip referencing the same player becomes concurrently active.
- **Affects:** Bug #1 (cuts), Bug #2 (repositioned cuts), Bug #3 (degradation)

### U3. advanceToTime() Async Decode Gap — Frames Not Ready When Renderer Reads
- **Severity:** CRITICAL
- **Files:** `WebCodecsPlayer.ts:982-1213`, `useEngine.ts:414-420`
- **Root Cause:** `advanceToTime()` feeds samples to `decoder.decode()` which is async — frames arrive via the output callback later. But `buildLayersFromStore()` → `LayerCollector.collect()` calls `getCurrentFrame()` in the same RAF tick, reading the stale/null frame before the decoder has produced output. The render-loop reorder fix (syncVideo before buildLayers) only helps if the frame was decoded in a previous tick.
- **Fix:** Wire `WebCodecsPlayer.onFrame` to `engine.requestNewFrameRender()` so the render loop runs again when the target frame actually arrives. For scrubbing, expose seek completion as a callback/promise.
- **Affects:** Bug #1 (cuts), Bug #4 (scrubbing)

### U4. clipsAtTime EPSILON Causes Both Clips Active at Cut Boundary
- **Severity:** HIGH
- **Files:** `FrameContext.ts:136-146`
- **Root Cause:** `clipsAtTime` uses `playheadPosition < c.startTime + c.duration + EPSILON` (1e-6). At exact cut points, adjacent clips are simultaneously active. Video sync runs both clips, but audio picks "last clip per track" while video renders `trackClips[0]` (outgoing clip). Audio and video disagree for 1 frame.
- **Fix:** Resolve a single boundary winner per track for non-transition cuts. Use the same rule in video sync, audio sync, and layer building.
- **Affects:** Bug #1 (cuts)

### U5. Pre-Decode System Structurally Unreliable
- **Severity:** HIGH
- **Files:** `WebCodecsPlayer.ts:1626-1717`, `VideoSyncManager.ts:808-833`
- **Root Cause:** Multiple compounding issues found by all agents:
  - Single slot (`preDecodedFrame`) — later lookahead overwrites earlier upcoming cuts
  - Stores last output frame, not closest-to-target (fails with B-frame reordering)
  - `preDecodeUpcomingCuts()` only triggers when `posDiff > 0.5` — misses smaller discontinuities
  - `flush()` has no timeout — can deadlock `preDecodeInProgress` forever
  - Error callback doesn't `.close()` the temp decoder — leaks hardware decoder slots
  - Async completion never awaited — frame often not ready when cut arrives
- **Fix:** Store predecoded frames keyed by target time, use target-timestamp filtering (not last output), add flush timeout + error cleanup, trigger on any edit discontinuity.
- **Affects:** Bug #2 (repositioned cuts), Bug #3 (degradation)

### U6. Decoder Stall Detection — False Positives + Incomplete Coverage
- **Severity:** HIGH
- **Files:** `WebCodecsPlayer.ts:1215-1250`
- **Root Cause:** Multiple issues found by all agents:
  - 800ms threshold false-triggers on legitimate long-GOP seeks (5-10s GOPs common in phone/YouTube video)
  - Only runs inside `advanceToTime()` — paused `seek()`/`fastSeek()` stalls are never detected
  - `decoderRecreateCount` resets only in `play()`, but full-mode playback uses `advanceToTime()` — recovery stops after 3 recreates for the session lifetime
  - `lastDecodeFeedTime`/`lastDecodeOutputTime` not reset after recreate — new decoder immediately detected as "stalled"
- **Fix:** Reset stall timestamps per decoder generation, add grace period after recreate, track frames-pending instead of just elapsed time, add seek-completion watchdog for paused paths.
- **Affects:** Bug #3 (degradation), Bug #1 (cuts — false recovery extends freeze)

---

## MAJORITY (4-5/6 Agents)

### M1. Paused Seek Completion Never Wakes Render Loop
- **Severity:** CRITICAL
- **Agents:** 5/6 (Codex1, Codex2, Codex3, Explore2, Explore3)
- **Files:** `WebCodecsPlayer.ts:551`, `webCodecsHelpers.ts:39`, `VideoSyncManager.ts:988`
- **Root Cause:** Full-mode WCPs are created without `onFrame`. When paused `seek()`/`fastSeek()` completes async in the decoder output callback, `currentFrame` updates but no `requestNewFrameRender()` fires. The render loop already finished its RAF tick.
- **Fix:** Wire `onFrame` to `engine.requestNewFrameRender()` for full-mode players.
- **Affects:** Bug #4 (scrubbing)

### M2. Temp Pre-Decode Decoders Leak Hardware Slots
- **Severity:** CRITICAL
- **Agents:** 4/6 (Codex1, Codex2, Codex3, Explore1)
- **Files:** `WebCodecsPlayer.ts:1647-1687`
- **Root Cause:** `preDecodeAtTime()` creates a temporary `VideoDecoder`. The `error` callback flips `preDecodeInProgress = false` but never calls `tempDecoder.close()`. No timeout on `flush()`. Over extended use, leaked temp decoders exhaust Chrome's hardware decoder pool.
- **Fix:** Close temp decoder in error path, add hard timeout around flush, clear predecode state on failure.
- **Affects:** Bug #3 (degradation)

### M3. clearFrameBuffer() Protection Incomplete
- **Severity:** HIGH
- **Agents:** 4/6 (Explore1, Explore2, Explore3, Codex3)
- **Files:** `WebCodecsPlayer.ts:838-845`, `WebCodecsPlayer.ts:1190-1195`
- **Root Cause:** The recent fix skips closing frames that match `currentFrame` in the buffer. But after `splice()` removes `currentFrame` from the buffer (line 1197), subsequent `clearFrameBuffer()` calls can't find it to protect it. Also, the async decoder output callback can close a frame concurrently while `currentFrame` still references it.
- **Fix:** Track currentFrame separately from buffer. Use a "previous frame" pattern — defer closing old frame until next frame arrives.
- **Affects:** Bug #1 (cuts), Bug #3 (degradation)

### M4. WCP Cached Forever — Never Destroyed
- **Severity:** CRITICAL
- **Agents:** 4/6 (Codex1, Codex2, Codex3, Explore — implied)
- **Files:** `serializationUtils.ts:21-41`, `clipSlice.ts:361`
- **Root Cause:** `globalWcpCache` stores live `WebCodecsPlayer` instances. `removeClip` never destroys them. `clearTimeline` explicitly keeps them. This retains decoders, sample arrays, and `currentFrame` objects across the entire session. Direct cause of "only Chrome restart fixes it."
- **Fix:** Add refcounted ownership by `mediaFileId`. Destroy WCP when last clip referencing it is removed. Split cached demux data from live decoder instances.
- **Affects:** Bug #3 (degradation)

### M5. Fast Scrubbing Drops Latest Seek Target
- **Severity:** HIGH
- **Agents:** 4/6 (Codex1, Codex2, Codex3, Explore2)
- **Files:** `VideoSyncManager.ts:952-988`
- **Root Cause:** During drag scrubbing, if `isDecodePending()` is true, the code skips issuing a new `fastSeek()` and doesn't store the latest target. The debounced precise seek fires with an old position. WCP seek timers are keyed by clipId, not decoder instance — shared players get conflicting seeks.
- **Fix:** Always store `latestWcSeekTarget`, key seeks by player/source, cancel on play/clip exit.
- **Affects:** Bug #4 (scrubbing)

### M6. First Clip Activation Forces Unnecessary Decoder Reset
- **Severity:** HIGH
- **Agents:** 4/6 (Codex1, Codex3, Explore1, Explore2)
- **Files:** `WebCodecsPlayer.ts:982-1053`
- **Root Cause:** "Skip reset if already positioned" optimization requires `frameBuffer.length > 0`. Cold/inactive players have only `currentFrame` from `decodeFirstFrame()`, not buffered frames. First activation always does `decoder.reset()+configure()+re-feed`, even when already at the correct keyframe.
- **Fix:** Treat valid `currentFrame` near target + correct `feedIndex` as sufficient to skip reset.
- **Affects:** Bug #1 (cuts)

---

## MINORITY (2-3/6 Agents)

### m1. advanceToTime() Frame Acceptance Window Too Strict
- **Severity:** HIGH | **Agents:** 3/6 (Explore1, Explore2, Explore3)
- **Files:** `WebCodecsPlayer.ts:1181-1213`
- **Root Cause:** 1.5 frame tolerance for accepting buffered frames. During repositioned cut seeks, decoder produces GOP traversal frames that may not fall within this window. No "best effort" fallback.
- **Affects:** Bug #2

### m2. Health Monitor Observes HTMLVideo, Not WebCodecs Decoder
- **Severity:** HIGH | **Agents:** 3/6 (Codex1, Codex3, Explore — implied)
- **Files:** `playbackHealthMonitor.ts:127-189`
- **Root Cause:** Monitor uses `video.currentTime`, `video.readyState` from HTML element. In full mode, audio element can keep playing while WebCodecs decoder is frozen.
- **Affects:** Bug #3

### m3. Off-Playhead Full-Mode Players Never Leave "Playing" State
- **Severity:** HIGH | **Agents:** 2/6 (Codex3, Explore3)
- **Files:** `WebCodecsPlayer.ts:997`, `VideoSyncManager.ts:227`
- **Root Cause:** `advanceToTime()` auto-sets `_isPlaying = true`. Inactive WCPs are never paused, keeping decoders/frames alive.
- **Affects:** Bug #3

### m4. Decode Errors Silently Swallowed
- **Severity:** MEDIUM | **Agents:** 2/6 (Codex3, Explore2)
- **Files:** `WebCodecsPlayer.ts:798-805, 1130-1150`
- **Root Cause:** All decode paths catch `decoder.decode()` failures but keep advancing feed/seek state. Target frame becomes unreachable.
- **Affects:** Bug #1, Bug #3

### m5. Layer Cache Doesn't Invalidate on Paused Playhead Jump
- **Severity:** HIGH | **Agents:** 1/6 (Codex2)
- **Files:** `LayerCache.ts:49`, `LayerBuilderService.ts:51`
- **Root Cause:** Paused click across a cut can keep rendering previous clip's layer selection.
- **Affects:** Bug #4

### m6. Pause + Seek = Double Reset/Configure Churn
- **Severity:** MEDIUM | **Agents:** 2/6 (Codex2, Explore3)
- **Files:** `VideoSyncManager.ts:937`, `WebCodecsPlayer.ts:646`
- **Root Cause:** `pause()` does `reset()+configure()`, then immediate `seek()` does another. Multiplied decoder resets accelerate degradation.
- **Affects:** Bug #3

### m7. Handoff Element Stale / Not Warmed
- **Severity:** MEDIUM | **Agents:** 3/6 (Explore1, Explore3, Codex2)
- **Files:** `VideoSyncManager.ts:844-855, 903-904`
- **Root Cause:** `computeHandoffs()` clears all handoffs every frame. Orphaned elements left playing. New handoffs not pre-seeked.
- **Affects:** Bug #1

### m8. Pipeline Diagnostics Global, Not Per-Decoder
- **Severity:** LOW | **Agents:** 2/6 (Codex1, Codex3)
- **Files:** `wcPipelineMonitor.ts:36-78`
- **Root Cause:** One player's output resets stall timer for another. Noisy for multi-clip debugging.
- **Affects:** Bug #3

---

## Top 5 Fixes by Impact (Consensus-Ranked)

| Priority | Fix | Consensus | Effort | Impact |
|----------|-----|-----------|--------|--------|
| **1** | Add last-good-frame hold in LayerCollector | 6/6 | Low | Eliminates ALL black frames at cuts/scrubs |
| **2** | Fix clipsAtTime boundary — single winner per track | 6/6 | Low | Eliminates dual-clip conflict at cuts |
| **3** | Wire decoder output to `requestNewFrameRender()` | 5/6 | Low | Fixes scrubbing race completely |
| **4** | Close temp decoders on error + add flush timeout | 4/6 | Low | Stops decoder pool exhaustion |
| **5** | Add WCP refcount + destroy on last clip removal | 4/6 | Medium | Fixes long-session degradation |

### Secondary Fixes (High Impact, More Effort)
| Priority | Fix | Consensus |
|----------|-----|-----------|
| 6 | Don't share live decoder across split clips | 6/6 |
| 7 | Reset stall timestamps per decoder generation | 6/6 |
| 8 | Pre-decode burst (not single frame) with target filtering | 6/6 |
| 9 | Skip unnecessary reset on cold clip activation | 4/6 |
| 10 | Store latest WC seek target during fast scrubbing | 4/6 |

---

## Bug-to-Fix Matrix

| Bug | Primary Causes (Unanimous) | Secondary Causes |
|-----|---------------------------|------------------|
| **#1 Video Glitch at Cuts** | U1 (no fallback), U2 (shared decoder), U4 (epsilon overlap) | M3 (clearFrameBuffer incomplete), M6 (unnecessary reset), m7 (handoff stale) |
| **#2 Repositioned Cuts Stutter** | U2 (shared decoder), U5 (pre-decode unreliable) | m1 (acceptance window strict) |
| **#3 Decoder Degradation** | U2 (shared decoder), U5 (temp decoder leak), U6 (stall detection) | M2 (temp decoder leak), M4 (WCP never destroyed), m3 (never paused), m6 (reset churn) |
| **#4 Scrubbing Race** | U1 (no fallback), U3 (async gap) | M1 (no render wake), M5 (latest target dropped), m5 (layer cache) |
