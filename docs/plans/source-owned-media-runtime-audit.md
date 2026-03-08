# Source-Owned Media Runtime Audit

## Purpose

This document records the current ownership model for media playback and decode state.

The main question is:

`What currently belongs to clip instances that should instead belong to a shared source runtime?`

## Short Answer

Today, video runtime state is mostly clip-owned.

That includes:

- `HTMLVideoElement`
- `HTMLAudioElement`
- `WebCodecsPlayer`
- native decoder references
- cleanup lifecycle
- playback sync assumptions
- layer-building assumptions

The codebase already contains one partial counterexample:

- export parallel decode via `ParallelDecodeManager`

That path is closer to source-oriented ownership than preview playback is.

## Key Findings

### 1. Ownership Is Embedded In Core Types

`TimelineClip.source` stores live runtime objects directly:

- `videoElement`
- `audioElement`
- `webCodecsPlayer`
- `nativeDecoder`

Evidence:

- `src/types/index.ts:292`
- `src/types/index.ts:293`
- `src/types/index.ts:295`

`LayerSource` also carries those runtime objects directly into render layers:

- `src/types/index.ts:70`
- `src/types/index.ts:75`

Implication:

- runtime ownership is not a side cache or service
- it is part of the clip data model itself

### 2. Clip Creation Paths Instantiate Runtime Objects Per Clip

#### Regular video add

The normal add-video path creates a new `HTMLVideoElement`, stores it on the clip, then attaches a `WebCodecsPlayer` back onto the same clip:

- `src/stores/timeline/clip/addVideoClip.ts:188`
- `src/stores/timeline/clip/addVideoClip.ts:223`
- `src/stores/timeline/clip/addVideoClip.ts:286`

#### Split clips

Split clips explicitly clone media elements for derived parts:

- `src/stores/timeline/clipSlice.ts:518`
- `src/stores/timeline/clipSlice.ts:519`
- `src/stores/timeline/clipSlice.ts:523`
- `src/stores/timeline/clipSlice.ts:526`

The AI batch split path does the same:

- `src/services/aiTools/handlers/clips.ts:44`
- `src/services/aiTools/handlers/clips.ts:45`
- `src/services/aiTools/handlers/clips.ts:52`

Implication:

- split clips are separate runtime owners, not separate views into a shared source

#### Clipboard paste

Pasted clips create a new `HTMLVideoElement` and then attach a new `WebCodecsPlayer`:

- `src/stores/timeline/clipboardSlice.ts:328`
- `src/stores/timeline/clipboardSlice.ts:345`

Implication:

- clipboard is another clip-owned constructor path that would bypass a shared runtime if left unchanged

#### Nested composition loading

Nested clip loading also creates clip-local video elements and players:

- `src/stores/timeline/clip/addCompClip.ts:370`
- `src/stores/timeline/clip/addCompClip.ts:373`
- `src/stores/timeline/clip/addCompClip.ts:626`
- `src/stores/timeline/clip/addCompClip.ts:631`

Implication:

- nested comps currently duplicate source runtime state across nested clip instances

#### Restore / deserialization

Restore attaches media elements and WebCodecs players back onto individual clips:

- `src/stores/timeline/serializationUtils.ts:402`
- `src/stores/timeline/serializationUtils.ts:543`
- `src/stores/timeline/serializationUtils.ts:557`
- `src/stores/timeline/serializationUtils.ts:924`
- `src/stores/timeline/serializationUtils.ts:942`

Implication:

- restored projects rebuild clip-owned runtime objects instead of reconstructing shared source runtimes

### 3. Preview Sync Operates On Clip-Owned Runtime State

`VideoSyncManager` is built around direct access to `clip.source.videoElement` and `clip.source.webCodecsPlayer`.

Examples:

- same-source handoff stores previous `videoElement` per track:
  - `src/services/layerBuilder/VideoSyncManager.ts:64`
  - `src/services/layerBuilder/VideoSyncManager.ts:87`
  - `src/services/layerBuilder/VideoSyncManager.ts:115`
- inactive clips are paused directly via clip-owned elements:
  - `src/services/layerBuilder/VideoSyncManager.ts:158`
  - `src/services/layerBuilder/VideoSyncManager.ts:164`
- nested clips are also synced through their own `videoElement` / `webCodecsPlayer`:
  - `src/services/layerBuilder/VideoSyncManager.ts:210`
  - `src/services/layerBuilder/VideoSyncManager.ts:229`
  - `src/services/layerBuilder/VideoSyncManager.ts:230`
  - `src/services/layerBuilder/VideoSyncManager.ts:270`
  - `src/services/layerBuilder/VideoSyncManager.ts:273`
- normal preview sync branches on clip-local full WebCodecs:
  - `src/services/layerBuilder/VideoSyncManager.ts:340`
  - `src/services/layerBuilder/VideoSyncManager.ts:346`
- full WebCodecs sync directly consumes clip-owned element and player:
  - `src/services/layerBuilder/VideoSyncManager.ts:762`
  - `src/services/layerBuilder/VideoSyncManager.ts:763`

Implication:

- preview playback assumes one live runtime owner per clip

### 4. Same-Source Cut Behavior Is Implemented As A Handoff Workaround

The current seamless-cut strategy reuses a previous clip's `HTMLVideoElement` across a cut:

- `src/services/layerBuilder/VideoSyncManager.ts:93`
- `src/services/layerBuilder/VideoSyncManager.ts:113`
- `src/services/layerBuilder/VideoSyncManager.ts:115`

`LayerBuilderService` then swaps the layer source to the handoff element and explicitly drops the clip's own `WebCodecsPlayer` during that handoff:

- `src/services/layerBuilder/LayerBuilderService.ts:370`
- `src/services/layerBuilder/LayerBuilderService.ts:372`

Implication:

- same-source continuity is currently a special-case patch over clip-owned playback
- it is not a shared runtime design
- it only helps continuous cuts, not arbitrary repeated source reuse

### 5. GPU Warmup Logic Exists Because Each Clip Owns Its Own Surface

`VideoSyncManager` explicitly contains proactive GPU warmup for upcoming clips because split clips have separate cold `HTMLVideoElement` surfaces:

- `src/services/layerBuilder/VideoSyncManager.ts:186`
- `src/services/layerBuilder/VideoSyncManager.ts:648`
- `src/services/layerBuilder/VideoSyncManager.ts:654`

Implication:

- the codebase already acknowledges the architectural cost of per-clip media elements

### 6. Layer Construction Passes Clip-Owned Runtime Objects Into Rendering

Regular preview layers:

- `src/services/layerBuilder/LayerBuilderService.ts:370`
- `src/services/layerBuilder/LayerBuilderService.ts:372`

Nested preview layers:

- `src/services/layerBuilder/LayerBuilderService.ts:756`
- `src/services/layerBuilder/LayerBuilderService.ts:761`
- `src/services/layerBuilder/LayerBuilderService.ts:762`

Implication:

- render layers are assembled from clip-owned runtime state, not from a shared frame request API

### 7. RAM Preview Also Assumes Clip-Owned Players

RAM preview seeks clip-owned video elements and clip-owned WebCodecs players directly:

- `src/services/ramPreviewEngine.ts:216`
- `src/services/ramPreviewEngine.ts:217`
- `src/services/ramPreviewEngine.ts:230`
- `src/services/ramPreviewEngine.ts:243`

Nested RAM preview does the same:

- `src/services/ramPreviewEngine.ts:277`
- `src/services/ramPreviewEngine.ts:279`
- `src/services/ramPreviewEngine.ts:283`
- `src/services/ramPreviewEngine.ts:293`

Implication:

- RAM preview will need either runtime-backed sessions or an explicit export-like session model

### 8. Export Is Mixed: Still Clip-Aware, But Already Has A Better Precedent

#### Good precedent: parallel decode

Export already has a non-clip-local decode path through `ParallelDecodeManager`:

- `src/engine/export/ClipPreparation.ts:204`
- `src/engine/export/ClipPreparation.ts:266`
- `src/engine/export/ClipPreparation.ts:272`
- `src/engine/export/VideoSeeker.ts:23`
- `src/engine/export/ExportLayerBuilder.ts:191`

This is an important precedent because it is closer to source-oriented decode management.

#### Still clip-aware in several places

Sequential export still uses clip-specific `webCodecsPlayer` state or clip-owned `videoElement` state:

- `src/engine/export/VideoSeeker.ts:66`
- `src/engine/export/VideoSeeker.ts:85`
- `src/engine/export/VideoSeeker.ts:90`
- `src/engine/export/ExportLayerBuilder.ts:187`
- `src/engine/export/ExportLayerBuilder.ts:212`
- `src/engine/export/ExportLayerBuilder.ts:234`

Clip preparation also falls back to clip-owned `videoElement.src` for file loading:

- `src/engine/export/ClipPreparation.ts:326`
- `src/engine/export/ClipPreparation.ts:429`

Implication:

- export is not cleanly source-owned either
- but the parallel decode path is a strong migration model to reuse

### 9. Composition Renderer Has Its Own Clip-Keyed Source Cache

`CompositionRenderer` stores `clipSources` keyed by clip id, not by media source identity:

- `src/services/compositionRenderer.ts:17`
- `src/services/compositionRenderer.ts:111`
- `src/services/compositionRenderer.ts:139`
- `src/services/compositionRenderer.ts:144`
- `src/services/compositionRenderer.ts:227`
- `src/services/compositionRenderer.ts:238`

At evaluation time it seeks and emits layers from those clip-keyed sources:

- `src/services/compositionRenderer.ts:383`
- `src/services/compositionRenderer.ts:385`
- `src/services/compositionRenderer.ts:405`
- `src/services/compositionRenderer.ts:406`

Implication:

- there is a second clip-scoped runtime model outside the main timeline preview path
- this will need alignment with the shared source runtime eventually

### 10. Background Layer Playback Repeats The Pattern

Background layer playback stores and syncs clip-owned video/audio elements:

- `src/services/layerPlaybackManager.ts:276`
- `src/services/layerPlaybackManager.ts:281`
- `src/services/layerPlaybackManager.ts:282`
- `src/services/layerPlaybackManager.ts:314`
- `src/services/layerPlaybackManager.ts:348`

Implication:

- background layers are another independent consumer that will need a session model

### 11. Cleanup Lifecycle Is Per Clip

Clip removal cleans up clip-owned media directly:

- `src/stores/timeline/clipSlice.ts:345`
- `src/stores/timeline/clipSlice.ts:361`
- `src/stores/timeline/clipSlice.ts:368`

Timeline clear destroys clip-owned media directly:

- `src/stores/timeline/serializationUtils.ts:1015`
- `src/stores/timeline/serializationUtils.ts:1023`
- `src/stores/timeline/serializationUtils.ts:1031`

Background-layer deactivation also tears down per-clip elements:

- `src/services/layerPlaybackManager.ts:130`
- `src/services/layerPlaybackManager.ts:135`

Composition renderer cleanup does the same for its clip-keyed source cache:

- `src/services/compositionRenderer.ts:649`
- `src/services/compositionRenderer.ts:650`

Implication:

- lifecycle management is distributed and clip-oriented
- introducing a shared runtime requires centralized retain/release ownership

## Architectural Conclusion

The current model is:

- clip-owned in preview
- clip-owned in nested playback
- clip-owned in RAM preview
- partly clip-owned in export
- clip-keyed in composition renderer

This is incompatible with the target system where:

- one source may appear many times
- one source may appear at multiple times simultaneously
- nested comps should not duplicate source runtime state
- frame reuse should happen by source identity and source time, not by clip id

## Best Existing Migration Seam

The best existing seam is export parallel decode:

- `ParallelDecodeManager`
- frame lookup by requested time
- no direct dependence on one `HTMLVideoElement` per visible clip

That does not solve preview on its own, but it demonstrates that the codebase already has a more source-oriented decode model available.

## Immediate Refactor Targets

Phase 2 should design and introduce:

1. `MediaSourceRuntime`
   - keyed by `mediaFileId` or stable file identity
   - owns source metadata, demux state, codec config, shared frame cache

2. `DecodeSession`
   - one temporal cursor against a source runtime
   - allocated per active simultaneous time cursor, not per clip by default

3. `FrameRequest` API
   - callers request a source frame by source identity and time
   - render paths stop reading clip-owned `webCodecsPlayer` directly

4. centralized lifecycle
   - retain/release source runtimes
   - session allocation policy
   - frame lifetime policy

## Open Questions For Phase 2

1. Which identity should be canonical for source runtime lookup?
   - `mediaFileId`
   - file handle identity
   - file hash
   - fallback blob/file identity

2. Should interactive preview sessions and export sessions share the same runtime object but use different session policies?

3. Should `HTMLVideoElement` remain only for:
   - audio playback
   - bootstrap fallback
   - non-WebCodecs browsers

4. Should composition renderer migrate onto the same runtime registry immediately, or after main preview cutover?

## Recommended Next Step

Phase 2: define the concrete code interfaces for:

- `MediaSourceRuntime`
- `MediaRuntimeRegistry`
- `DecodeSession`
- `FrameRequest`
- `FrameHandle`

Then wire the registry into clip load/restore paths without changing playback behavior yet.
