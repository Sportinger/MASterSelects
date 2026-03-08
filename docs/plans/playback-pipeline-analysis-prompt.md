# Playback Pipeline Analysis — Agent Prompt

Copy this entire prompt into a new Claude Code session.

---

## TASK

You are debugging the MasterSelects video editor's playback pipeline. Your job is to launch 6 parallel analysis agents (3 Codex CLI + 3 internal Explore agents), wait for all results, then compile and rank findings by consensus.

## STEP 1: Launch all 6 agents with the ANALYSIS PROMPT below

Launch these in a SINGLE message (all parallel):

**3 Codex agents** (each via Bash, run_in_background=true):
```
codex exec --dangerously-bypass-approvals-and-sandbox "ANALYSIS_PROMPT_HERE"
```

**3 internal agents** (each via Agent tool, subagent_type=Explore, run_in_background=true):
Use the same ANALYSIS_PROMPT_HERE as the prompt parameter.

## STEP 2: Wait for all 6 to complete, then compile results

Once all agents finish, create a ranked table:
1. Group identical/similar findings across all agents
2. Rank by consensus count (how many of the 6 agents found the same issue)
3. For each finding: severity, file:line, root cause, proposed fix
4. Separate "unanimous" (5-6 agents), "majority" (3-4 agents), "minority" (1-2 agents)

---

## ANALYSIS PROMPT (use this verbatim for all 6 agents)

```
You are analyzing the MasterSelects video editor playback pipeline. This is a browser-based video editor (React + Zustand + WebGPU) using WebCodecs API (VideoDecoder + MP4Box) for hardware-accelerated video decoding.

PROBLEM STATEMENT — ACTIVE BUGS:
1. VIDEO GLITCH AT CUTS: When playhead crosses a cut point between two clips, video preview freezes, shows black frame, or stutters for 0.5-3 seconds. Audio is now seamless (handoff system works), but video still glitches.
2. REPOSITIONED CUTS STUTTER: When clip B plays a different part of the same source video as clip A (not continuous), the transition causes a visible freeze/stutter (~100-200ms) because the decoder must seek to a new position.
3. DECODER STATE DEGRADATION: After extended use, the WebCodecs VideoDecoder gets "stuck" — stops producing frames. Only a Chrome restart fixes it. Suggests VideoFrame leaks or hardware decoder pool exhaustion.
4. SCRUBBING RACE CONDITION: Clicking to a new playhead position doesn't always update the video preview. The async decode completes after the renderer already read getCurrentFrame().

ARCHITECTURE (read these files):
- src/engine/WebCodecsPlayer.ts — Full-mode hardware decoder (MP4Box demux + VideoDecoder). Split clips share the SAME instance (one decoder per source file). Key methods: advanceToTime(), seek(), fastSeek(), pause(), stop(), clearFrameBuffer(), getCurrentFrame(), initDecoder(), preDecodeAtTime()
- src/engine/render/LayerCollector.ts — Reads getCurrentFrame() from WebCodecsPlayer, imports as GPU texture. Has disabled HTMLVideo fallback for full WebCodecs mode (line ~188). If getCurrentFrame() returns null → BLACK FRAME with no fallback.
- src/services/layerBuilder/VideoSyncManager.ts — Orchestrates video sync per frame. Calls advanceToTime() during playback, seek()/fastSeek() during scrubbing. Handles cut transition handoffs, GPU warmup, audio pre-buffering, WebCodecs pre-decode lookahead.
- src/services/layerBuilder/FrameContext.ts — Single store read with lazy cached computations. clipsAtTime uses half-open interval with EPSILON tolerance.
- src/hooks/useEngine.ts — Render loop: syncVideoElements() → buildLayersFromStore() → engine.render(layers) → syncAudioElements()
- src/stores/timeline/clipSlice.ts — splitClip: split clips share WebCodecsPlayer instance
- src/services/layerBuilder/AudioSyncHandler.ts — Unified audio sync
- src/services/layerBuilder/AudioTrackSyncManager.ts — Audio track handoffs
- src/services/playbackHealthMonitor.ts — Anomaly detection + recovery (softReset, forceDecodeAll)
- src/services/wcPipelineMonitor.ts — Decode pipeline event ring buffer

KEY TECHNICAL DETAILS:
- advanceToTime() is called every frame during playback. It feeds samples to the decoder and picks the best frame from the buffer. The decoder output callback is ASYNC — frames arrive later.
- When advanceToTime() needs to seek (backward jump, large forward gap, playback restart): it calls decoder.reset() + clearFrameBuffer() + re-feeds from keyframe. During the decode gap, getCurrentFrame() returns the old frame (or null).
- clearFrameBuffer() now skips closing currentFrame (recently fixed). But if currentFrame was NOT in the buffer, the fix has no effect.
- Render loop order was recently fixed: syncVideoElements() now runs BEFORE buildLayersFromStore() so LayerCollector reads the freshly decoded frame.
- Pre-decode system exists: preDecodeAtTime() uses a temporary VideoDecoder to pre-decode frames for upcoming repositioned cuts. consumePreDecodedFrame() uses them at the cut point.
- Decoder stall detection exists: if decoder produces no output for 800ms while being fed, it's destroyed and recreated (max 3x per session).
- LayerCollector line ~188: HTMLVideo fallback is DISABLED for full WebCodecs mode. Enabling it previously made things WORSE (two decoders fighting over the same video element).

RECENT FIXES APPLIED (may or may not be working correctly):
1. clearFrameBuffer() preserves currentFrame — prevents displayed frame destruction during decoder resets
2. Render loop reorder — syncVideoElements before buildLayersFromStore
3. Decoder stall auto-recovery — destroy + recreate VideoDecoder after 800ms stall
4. Pre-decode for repositioned cuts — temporary decoder pre-decodes first frame during 1.5s lookahead
5. Audio handoff persistence — handoff video element reused for entire clip duration
6. AudioSyncHandler skip unnecessary seek on resume — only seek if drift > 0.1s

YOUR TASK:
Read ALL the files listed above thoroughly. For each of the 4 problems, identify:

1. EXACT root cause with file:line references
2. Whether the recent fixes actually address the issue or have bugs themselves
3. Race conditions in the async decode pipeline
4. Frame lifecycle issues (premature close, null references, leaked VideoFrames)
5. Timing issues between video sync, layer building, and rendering
6. Missing error handling or edge cases

Also look for:
- VideoFrame objects that are never .close()d (GPU memory leaks → decoder exhaustion)
- Paths where currentFrame could become a closed/invalid VideoFrame
- Whether the pre-decode system (preDecodeAtTime/consumePreDecodedFrame) works correctly
- Whether the decoder stall detection has false positives or misses real stalls
- Edge cases in clipsAtTime boundary detection (floating-point, epsilon)
- Whether shared WebCodecsPlayer across split clips causes conflicts when clips overlap or transition

Output format — provide a NUMBERED LIST of ALL findings:
For each finding:
- NUMBER and SHORT TITLE
- SEVERITY: CRITICAL / HIGH / MEDIUM / LOW
- FILE:LINE reference
- ROOT CAUSE: exact description of the bug
- PROPOSED FIX: specific code change
- AFFECTS: which of the 4 problems this relates to

Be thorough. Read every line of the key methods. Don't just describe symptoms — trace the exact code path that causes each bug. Maximum detail.
```

---

## NOTES
- Codex agents may timeout on large codebases. If one fails, its findings are simply excluded from the ranking.
- The 6-agent approach gives strong consensus signal: if 5/6 agents find the same bug, it's almost certainly real.
- After ranking, implement fixes starting from the highest-consensus CRITICAL findings.
