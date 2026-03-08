# Source-Owned Media Runtime Next Steps

This document continues the work described in the initial plan:

- [Source-Owned Media Runtime Plan](./source-owned-media-runtime-plan.md)
- [Source-Owned Media Runtime Audit](./source-owned-media-runtime-audit.md)

It is not a replacement for those documents. It is the execution-focused follow-up for the next implementation passes.

## Current Status

### Phase 1 completed

The ownership audit is written down in [source-owned-media-runtime-audit.md](./source-owned-media-runtime-audit.md).

### Phase 2 completed

Shared runtime scaffolding exists in code:

- runtime types
- runtime registry
- clip-to-runtime binding utilities
- retain / release wiring in add, split, paste, restore, and cleanup paths

This establishes source identity and session identity without removing the old clip-owned playback path yet.

### Phase 3 partially completed

The main full-WebCodecs preview path now resolves through runtime sessions before falling back to clip-local player access.

What is done:

- runtime session/provider plumbing exists
- preview layer collection can read frames from runtime-backed sessions
- runtime ids/session keys now flow into preview layer sources

What is not done yet:

- same-source cut handoff logic still exists in the codebase
- simultaneous same-source frame reuse is not implemented as a true shared frame cache policy yet
- nested comp sync is not fully runtime-owned yet
- RAM preview and export are not migrated

## What To Do Next

### 1. Phase 4: Remove Same-Source Cut Handoff Hacks

This is the next highest-value step from the initial plan.

Goal:

- sequential same-source cuts should work because runtime sessions stay warm
- preview should stop depending on clip-to-clip `HTMLVideoElement` handoff

Primary files:

- `src/services/layerBuilder/VideoSyncManager.ts`
- `src/services/layerBuilder/LayerBuilderService.ts`

Concrete work:

- remove or bypass same-source handoff behavior for runtime-backed full-WebCodecs clips
- remove runtime-path dependence on previous clip `videoElement`
- remove runtime-path GPU warmup heuristics that only exist because clip surfaces are separate
- verify reordered split clips behave the same as original-order split clips

Deliverable:

- full-WebCodecs sequential same-source playback works through runtime/session continuity only

### 2. Phase 5: Support Simultaneous Same-Source Usage

After handoff removal is stable, the runtime has to support multiple active consumers of one source.

Goal:

- reuse the same decoded frame when consumers want the same source time
- allocate multiple sessions only when consumers need different temporal cursors

Primary files:

- `src/services/mediaRuntime/types.ts`
- `src/services/mediaRuntime/registry.ts`
- `src/services/mediaRuntime/runtimePlayback.ts`
- `src/engine/render/LayerCollector.ts`

Concrete work:

- define cache key policy by `sourceId` + source time or frame number
- define when a session can be shared vs when a second session must be created
- attach frame lifetime rules so cached `VideoFrame` objects are released safely
- validate repeated same source with different effects and different source times simultaneously

Deliverable:

- explicit session allocation policy and shared frame cache policy

### 3. Phase 6: Finish Nested Comp Migration

Nested comps still carry clip-local assumptions in parts of sync and evaluation.

Goal:

- nested comps should request source frames through the same runtime model as top-level preview

Primary files:

- `src/services/layerBuilder/VideoSyncManager.ts`
- `src/services/layerBuilder/LayerBuilderService.ts`
- `src/engine/render/NestedCompRenderer.ts`
- `src/services/compositionRenderer.ts`
- `src/stores/timeline/clip/addCompClip.ts`

Concrete work:

- remove remaining nested-preview reliance on clip-local player ownership
- stop duplicating source runtime state across nested clip instances where possible
- align `CompositionRenderer` clip-keyed source caches with the runtime registry

Deliverable:

- nested comp frame evaluation is runtime-backed end to end

### 4. Phase 7: Migrate RAM Preview And Export

Preview should be stable first. Then move the offline and semi-offline paths.

Goal:

- make RAM preview and export explicit runtime/session consumers instead of special clip-local branches

Primary files:

- `src/services/ramPreviewEngine.ts`
- `src/engine/export/VideoSeeker.ts`
- `src/engine/export/ClipPreparation.ts`
- `src/engine/export/ExportLayerBuilder.ts`
- `src/engine/ParallelDecodeManager.ts`

Concrete work:

- decide whether preview, RAM preview, and export share one runtime with different session policies
- reuse `ParallelDecodeManager` ideas where that reduces duplication
- remove export fallbacks that depend on clip-local `videoElement` or `webCodecsPlayer`

Deliverable:

- explicit interactive vs export session ownership model

### 5. Phase 8: Remove Obsolete Clip-Owned Assumptions

Only do this after runtime-backed playback paths are proven.

Goal:

- reduce clip source objects to lightweight source references plus fallback media objects only where still needed

Concrete work:

- remove `clip.source.webCodecsPlayer` as a core ownership concept
- narrow direct `HTMLVideoElement` usage to fallback or audio-specific cases
- simplify cleanup logic around clip removal and timeline clear
- simplify layer-building and sync branches that only exist for clip-local ownership

Deliverable:

- source-owned media runtime becomes the primary architecture, not an overlay

## Immediate Implementation Order

1. Finish Phase 4 in preview.
2. Re-test same-source split clips in original and random order.
3. Implement cache/session policy for simultaneous same-source use.
4. Finish nested comp migration onto the runtime.
5. Move RAM preview and export onto explicit runtime sessions.
6. Remove old clip-owned assumptions after the runtime path is proven.

## Validation Before Moving Past Preview

These cases should pass before RAM preview or export migration becomes the main focus:

- paused seek
- scrub seek
- refresh bootstrap without black preview
- same-source split clips in original order
- same-source split clips in random order
- repeated same source with different effects
- repeated same source at different times simultaneously
- nested comps using the same source multiple times

## Working Rule

Keep using the initial plan as the architectural source of truth:

- [Source-Owned Media Runtime Plan](./source-owned-media-runtime-plan.md)

Use this document as the short execution queue for the next implementation passes.
