# Source-Owned Media Runtime Plan

## Goal

Move video playback and decode state from clip-owned objects to a shared media runtime model that can support:

- many layers from the same source
- nested comps
- the same source reused multiple times at different times simultaneously
- stable same-source cuts without cold-start flicker
- a path toward After Effects-style reuse instead of NLE-style clip-local playback objects

## Why This Change Is Needed

The current model stores playback state on each clip instance:

- `clip.source.videoElement`
- `clip.source.webCodecsPlayer`

That creates several structural problems:

- split clips clone media elements and decoder state
- same-source reordered cuts hit cold decoders / cold GPU surfaces
- "handoff" logic is a workaround for clip-local ownership
- repeated use of one source at different times does not share demuxed data or decoded frames
- nested comps compose on top of clip-local playback state instead of shared source state

This is workable for a basic timeline, but it is the wrong abstraction for a compositing-heavy system.

## Non-Goals

This plan does not aim for exactly one decoder per source in all cases.

That would be too restrictive because the same source may need multiple temporal cursors at once. The target is:

- one shared runtime per media source
- zero or more decode sessions per source
- shared frame cache per source

## Target Architecture

### 1. Media Source Runtime

Introduce a shared runtime object keyed by `mediaFileId` or stable file identity.

Responsibilities:

- own media metadata
- own demuxed sample tables / codec config
- own source-level caches
- manage decode sessions
- expose frame request APIs

Example shape:

```ts
interface MediaSourceRuntime {
  sourceId: string;
  file: File;
  kind: 'video' | 'audio' | 'image';
  metadata: SourceMetadata;
  frameCache: DecodedFrameCache;
  getSession(key: SessionKey): DecodeSession;
  releaseSession(key: SessionKey): void;
  getFrame(request: FrameRequest): Promise<FrameHandle | null>;
}
```

### 2. Clip Instance

Clips remain timeline instances only.

Responsibilities:

- `assetId` / source reference
- `startTime`
- `inPoint`
- `outPoint`
- transform / effects / masks / speed / reverse

Clips should not own:

- decoder instances
- demux state
- source-level playback state

### 3. Decode Session

A decode session is a temporal cursor against one source runtime.

Needed because the same source may appear:

- twice on screen at different times
- inside nested comps while also visible in the parent comp
- in preview and export concurrently

Responsibilities:

- seek / advance / pause state for one cursor
- feed / flush decoder
- hold short-lived decode-local buffers
- publish frames into the shared cache

### 4. Shared Frame Cache

Decoded frames should be reusable across clip instances when they reference the same source time.

Key ideas:

- cache by source identity + source time / frame number
- multiple layers may consume one decoded frame with different effects
- reuse should happen before decode-session duplication

## Migration Strategy

### Phase 1. Audit Current Ownership

Map every place that assumes media state lives on the clip.

Primary areas:

- timeline clip creation / split / restore
- `clip.source.videoElement`
- `clip.source.webCodecsPlayer`
- playback sync
- layer building
- nested comps
- export / RAM preview
- cleanup / destroy lifecycle

Deliverable:

- ownership inventory with exact call sites

### Phase 2. Introduce Shared Runtime Registry

Add a runtime registry keyed by `mediaFileId` or stable file identity.

Responsibilities:

- create runtime lazily
- return existing runtime for repeated source use
- reference counting or explicit retain/release
- cleanup when last consumer is gone

Important constraint:

- keep old clip-owned path alive temporarily so the migration can be staged

Deliverable:

- registry and runtime skeleton with no playback cutover yet

### Phase 3. Move Full WebCodecs Preview to Runtime Sessions

First migration target is full WebCodecs preview playback.

Why first:

- current flicker problem lives here
- HTML/simple mode can remain as a fallback during migration
- this gives the biggest structural win without touching export immediately

Deliverable:

- preview path requests frames from runtime-backed sessions instead of clip-owned players

### Phase 4. Remove Same-Source Cut Handoff Hacks

Replace clip-to-clip handoff logic with runtime/session reuse for sequential same-source playback.

Expected result:

- no clip-boundary cold decoder behavior
- reordered split clips should no longer depend on warmup luck

Deliverable:

- sequential same-source cuts work through shared runtime instead of handoff video reuse

### Phase 5. Support Simultaneous Same-Source Usage

Extend the runtime path so one source can appear:

- multiple times in the same comp
- at different source times simultaneously
- inside nested comps and parent comps at once

Decision rule:

- if multiple consumers want the same source frame, reuse cache
- if multiple consumers need different temporal cursors, allocate multiple decode sessions

Deliverable:

- session allocation policy and shared frame cache policy

### Phase 6. Migrate Nested Comps

Nested comps should stop relying on clip-local media playback objects.

Deliverable:

- nested comp frame evaluation requests source frames through the shared runtime

### Phase 7. Adapt Export and RAM Preview

Do this after preview playback is stable.

Possible outcome:

- preview sessions use low-latency interactive policies
- export uses a separate sequential session policy

Deliverable:

- explicit export/session ownership model

### Phase 8. Remove Obsolete Clip-Owned Assumptions

Once runtime-backed playback is proven:

- remove `clip.source.webCodecsPlayer` as a core ownership concept
- reduce reliance on per-clip `HTMLVideoElement` for video playback
- simplify sync and render code
- narrow clip source objects to lightweight media references

Deliverable:

- simplified, source-owned preview architecture

## Validation Checklist

We should not treat this as done until these cases pass:

- paused seek
- scrub seek
- refresh bootstrap without black preview
- same-source split clips in original order
- same-source split clips in random order
- repeated same source with different effects
- repeated same source at different times simultaneously
- nested comps using the same source multiple times
- export correctness
- RAM preview correctness

## Immediate Implementation Order

1. Audit ownership points and write down exact dependencies.
2. Define `MediaSourceRuntime`, `DecodeSession`, and frame request interfaces in code.
3. Build the shared registry.
4. Cut over full WebCodecs preview to runtime-backed sessions.
5. Re-test same-source cut playback before touching export.

## Key Risks

- hidden assumptions in cleanup paths
- export code depending on clip-local player semantics
- nested comp code passing through clip-local media objects
- session explosion if simultaneous-use policy is naive
- frame cache leaks if `VideoFrame` lifetime is not tightly managed

## Guiding Rule

Effects and transforms belong to clip/layer instances.

Demux, decode, source metadata, and reusable decoded frames belong to the shared media runtime.
