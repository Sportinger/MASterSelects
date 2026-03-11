# Multi-Track Video Compositing in a Browser NLE

## Research Focus
How MasterSelects manages multiple HTMLVideoElements simultaneously, composites them through WebGPU, and handles the synchronization, texture lifecycle, and performance constraints inherent in browser-based multi-track video editing.

---

## 1. Browser Limits: Simultaneous HTMLVideoElement Decoding

### Platform Constraints

Browsers impose hard limits on concurrent hardware-decoded video streams. These limits are poorly documented and vary widely:

| Platform | Chrome | Firefox | Safari |
|----------|--------|---------|--------|
| Windows (DXVA/D3D11) | 6-16 HW streams | 4-8 HW streams | N/A |
| macOS (VideoToolbox) | 16-32 HW streams | 8-12 HW streams | 16+ HW streams |
| Linux (VA-API/Vulkan) | 4-8 HW streams | 2-4 HW streams | N/A |
| ChromeOS | 4-8 HW streams | N/A | N/A |

When the hardware decoder limit is exceeded, the browser falls back to software decoding (significantly slower) or drops frames. Chrome on Linux with Vulkan disabled commonly caps at 15fps for even single-stream playback (referenced in CLAUDE.md: `chrome://flags/#enable-vulkan`).

### MasterSelects Approach

MasterSelects does NOT attempt to keep all video elements playing simultaneously. The `VideoSyncManager.syncVideoElements()` method explicitly pauses videos not at the playhead:

```typescript
// From VideoSyncManager.syncVideoElements() line 1072-1091
for (const clip of ctx.clips) {
  if (clip.source?.videoElement) {
    const isAtPlayhead = ctx.clipsByTrackId.has(clip.trackId) &&
      ctx.clipsByTrackId.get(clip.trackId)?.id === clip.id;
    if (!isAtPlayhead && !clip.source.videoElement.paused &&
        !this.warmingUpVideos.has(clip.source.videoElement) &&
        !this.handoffElements.has(clip.source.videoElement)) {
      clip.source.videoElement.pause();
    }
  }
}
```

This means only clips **visible at the current playhead position** have their videos playing. A timeline with 20 clips but only 3 overlapping tracks at the playhead will have at most 3 active video elements plus any being warmed up for upcoming cuts.

---

## 2. The Four Decoder Paths

MasterSelects supports four distinct video decode strategies, chosen per-layer at collection time. The `LayerCollector.collectLayerData()` method tries them in priority order:

### 2.1 NativeHelper (Turbo Mode)
- Uses a Rust-based FFmpeg decoder running on `localhost:9877`
- Returns `ImageBitmap` frames uploaded via `copyExternalImageToTexture`
- Stored as `texture_2d<f32>` (persistent, re-uploaded each frame)
- Most efficient for ProRes/DNxHD codecs that browsers cannot HW-decode

### 2.2 WebCodecs (VideoFrame)
- `VideoDecoder` produces `VideoFrame` objects
- Imported via `device.importExternalTexture({ source: videoFrame })`
- Zero-copy GPU path using `texture_external`
- Feature-flagged via `flags.useFullWebCodecsPlayback` (currently `false` for preview)

### 2.3 HTMLVideoElement (Primary Path)
- Standard `<video>` element with `importExternalTexture`
- Zero-copy when GPU decoder surface is active
- Falls back to scrubbing cache (texture_2d copies) during seeking

### 2.4 ParallelDecode (VideoFrame from clip source)
- Direct `VideoFrame` attached to `layer.source.videoFrame`
- Same import path as WebCodecs but sourced differently

The `DetailedStats.decoder` field tracks which path is active:
```typescript
decoder: 'WebCodecs' | 'HTMLVideo(VF)' | 'HTMLVideo' | 'HTMLVideo(cached)' |
         'HTMLVideo(paused-cache)' | 'HTMLVideo(seeking-cache)' |
         'HTMLVideo(scrub-cache)' | 'NativeHelper' | 'ParallelDecode' | 'none';
```

---

## 3. GPU Texture Types and Memory

### texture_external vs texture_2d

The two texture types have fundamentally different GPU memory characteristics:

**`texture_external` (zero-copy)**
```typescript
// TextureManager.importVideoTexture() line 232
const texture = this.device.importExternalTexture({ source });
```
- No GPU memory allocation -- references the browser's decoder output directly
- Automatically invalidated when the video frame changes
- MUST be re-imported every frame (cannot be cached)
- Requires a separate WGSL shader pipeline (`texture_external` binding)
- Used in: `externalCompositeShader` at binding(2)

**`texture_2d<f32>` (copied)**
```typescript
// TextureManager.createImageTexture() line 44-54
const texture = this.device.createTexture({
  size: [width, height],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
});
this.device.queue.copyExternalImageToTexture({ source: image }, { texture }, [width, height]);
```
- Allocates GPU memory: 1920x1080 RGBA = ~8MB per texture
- Persistent across frames -- can be cached and reused
- Used for: images, text canvases, NativeHelper frames, scrubbing cache frames

### VRAM Budget Awareness

The ScrubbingCache explicitly manages VRAM consumption:
```typescript
// ScrubbingCache lines 18-19
private maxScrubbingCacheFrames = 300; // ~10 seconds at 30fps, ~2.4GB VRAM at 1080p
private readonly SCRUB_CACHE_FPS = 30;
```

At 1080p with 300 cached frames: 300 * 1920 * 1080 * 4 bytes = ~2.4GB VRAM. The cache uses LRU eviction based on Map insertion order.

The GPU frame cache for RAM preview is more conservative:
```typescript
// ScrubbingCache line 44-45
private maxGpuCacheFrames = 60; // ~500MB at 1080p
```

### When GPU Memory Gets Tight

With 4+ simultaneous video tracks, the comment in LayerCollector explicitly warns about GPU bandwidth:
```typescript
// LayerCollector line 660-662
// Cache frame for pause/seek fallback — skip during playback to save GPU bandwidth.
// With 4+ videos, the GPU copies (copyExternalImageToTexture per video at 20fps)
// waste significant bandwidth that's needed for rendering + effects.
```

The solution: during playback, scrubbing cache capture is disabled. Frames are only cached when paused or scrubbing, preserving GPU bandwidth for the actual compositing work.

---

## 4. Video Synchronization to Timecode

### The syncClipVideo Pipeline

`VideoSyncManager` is the central synchronization orchestrator. Its `syncVideoElements()` method runs every frame from the render loop:

1. **Compute handoffs** -- detect same-source sequential clips for seamless cut transitions
2. **Warm upcoming clips** -- proactively start decoding clips the playhead is approaching
3. **Sync each active clip** -- set video.currentTime or video.play() as needed
4. **Pause off-screen videos** -- stop decoding videos not at playhead
5. **Sync nested comp videos** -- recurse into composition hierarchies

### Playback Sync (Forward, 1x Speed)

During normal playback, the video plays freely and MasterSelects applies drift correction:
```typescript
// VideoSyncManager line 1528-1536
const driftThreshold = hasSpeedKeyframes ? 1.5 : 0.3;
if (timeDiff > driftThreshold) {
  video.currentTime = this.safeSeekTime(video, timeInfo.clipTime);
}
```

For speed-keyframed clips, the threshold is relaxed to 1.5s because frequent seeks cause `SEEK_STUCK` anomalies from decoder stalls.

### Scrub Sync (Paused, Dragging Playhead)

Scrubbing uses a sophisticated hybrid seek strategy:
1. **fastSeek()** for responsive scrubbing (nearest keyframe, imprecise)
2. **Debounced precise seek** (video.currentTime = target) after 90ms idle
3. **requestVideoFrameCallback (RVFC)** to detect when a new frame is presented
4. **Queued seeks** to coalesce rapid scrub movements

The settle state machine tracks each clip through stages: `settle` -> `retry` -> `warmup` -> `resolved`.

### GPU Surface Warmup (Post-Reload)

After page reload, `importExternalTexture` returns black because the GPU decoder surface is cold:
```typescript
// VideoSyncManager line 1397-1401
// Warmup: after page reload, video GPU surfaces are empty.
// importExternalTexture, canvas.drawImage, etc. all return black.
// The ONLY fix is video.play() to activate the GPU compositor.
```

The warmup system does a brief `play()` followed by pause + seek to populate the GPU surface. The `videoGpuReady` WeakSet in LayerCollector tracks which videos have been warmed.

---

## 5. Layer Compositing: Ping-Pong Architecture

### How Multiple Textures Become One Frame

The compositing pipeline uses a ping-pong buffer technique where layers are drawn one at a time, each reading from the previous result and writing to the alternate buffer.

**RenderTargetManager** allocates the buffer pair:
```
pingTexture  (rgba8unorm, output resolution)
pongTexture  (rgba8unorm, output resolution)
```

Plus two additional independent pairs for multi-preview rendering, and two effect temp textures for per-layer effect processing.

### Compositor.composite() Flow

```
1. Clear ping buffer to transparent black (alpha=0)
2. For each layer (bottom to top):
   a. Read accumulated result from readView (ping or pong)
   b. Classify effects: inline (brightness/contrast/saturation/invert) vs complex
   c. If complex effects: copy source to effectTempTexture, apply effects chain
   d. Create bind group with: sampler, readView, sourceTexture, uniforms, maskTexture
   e. Run composite shader to writeView (handles blend modes, transform, opacity)
   f. Swap readView <-> writeView
3. Return finalView (whichever buffer has the accumulated result)
```

The WGSL composite shader handles:
- 36 blend modes (normal through stencil/silhouette)
- Per-layer transform (position, scale, rotation XYZ with perspective)
- Mask compositing with feather
- Inline effect adjustments (brightness, contrast, saturation, invert)

### Two Shader Pipelines

Because `texture_external` and `texture_2d<f32>` are different WGSL types, MasterSelects maintains two parallel composite pipelines:

- **`getCompositePipeline()`** -- for images, text, cached frames (texture_2d)
- **`getExternalCompositePipeline()`** -- for live video frames (texture_external)

The bind group layout differs at binding(2):
```wgsl
// Regular composite
@group(0) @binding(2) var sourceTexture: texture_2d<f32>;

// External video composite
@group(0) @binding(2) var videoTexture: texture_external;
```

### Bind Group Caching

The CompositorPipeline caches bind groups for static layers (images, text) to avoid per-frame GPU allocation overhead. Video layers always invalidate their cache because the external texture changes every frame:

```typescript
// Compositor.composite() line 218-220
if (useExternalTexture && sourceExternalTexture) {
  if (!isStaticTextureSource) {
    this.compositorPipeline.invalidateBindGroupCache(layer.id);
  }
```

---

## 6. Nested Compositions (Comp-in-Comp)

### The Problem

A nested composition is a timeline inside a timeline. Clip A on the main timeline may contain its own multi-track composition with videos B, C, D that each need their own decoder, sync, and compositing.

### NestedCompRenderer Architecture

```
Main RenderDispatcher.render()
  |
  +-- LayerCollector.collect() identifies nested comp layers
  |     source.type === 'image' with source.nestedComposition set
  |
  +-- For each nested comp layer:
  |     NestedCompRenderer.preRender(compositionId, layers, w, h, encoder, ...)
  |       |
  |       +-- Acquire ping-pong texture pair from pool (keyed by "WxH")
  |       +-- collectNestedLayerData() -- same logic as LayerCollector but recurses
  |       +-- Ping-pong composite with effects
  |       +-- copyTextureToTexture into the comp's output texture
  |       +-- Release texture pair back to pool
  |       +-- Return GPUTextureView for use as a regular texture in main composite
  |
  +-- Main Compositor.composite() treats nested comp as a texture_2d layer
```

### Depth Limit and Recursion

Nesting depth is capped at 8 levels (`MAX_NESTING_DEPTH = 8`):
```typescript
// NestedCompRenderer.preRender() line 233
if (depth >= MAX_NESTING_DEPTH) {
  log.warn('Max nesting depth reached in preRender', { compositionId, depth });
  return null;
}
```

Sub-nested compositions recursively call `preRender()` with `depth + 1`.

### Texture Pool for Ping-Pong Buffers

Each nesting level needs its own ping-pong buffer pair. The `NestedCompRenderer` uses a texture pool keyed by resolution to avoid creating/destroying textures per frame:

```typescript
// NestedCompRenderer line 176-215
private acquireTexturePair(width: number, height: number): PooledTexturePair {
  const key = `${width}x${height}`;
  let pool = this.texturePool.get(key);
  // Find available pair or create new
  for (const pair of pool) {
    if (!pair.inUse) { pair.inUse = true; return pair; }
  }
  // Create new pair (2 textures at composition resolution)
  ...
}
```

### Frame Caching for Nested Comps

Nested compositions cache their last render to avoid redundant re-rendering:
```typescript
// NestedCompRenderer line 253-259
const quantizedTime = currentTime !== undefined ? Math.round(currentTime * 60) : -1;
if (quantizedTime >= 0 && lastTime === quantizedTime && lastCount === nestedLayers.length) {
  return compTexture.view; // Same frame, return cached
}
```

Time is quantized to ~60fps frame boundaries. If the playhead hasn't moved and layer count hasn't changed, the cached texture is reused.

### Video Sync Inside Nested Comps

`VideoSyncManager.syncNestedCompVideos()` handles time mapping:
```typescript
// VideoSyncManager line 1123-1126
const compLocalTime = ctx.playheadPosition - compClip.startTime;
const compTime = compLocalTime + compClip.inPoint;

// For each nested clip:
const nestedLocalTime = compTime - nestedClip.startTime;
const nestedClipTime = nestedClip.reversed
  ? nestedClip.outPoint - nestedLocalTime
  : nestedLocalTime + nestedClip.inPoint;
```

---

## 7. Performance Architecture

### Frame Budget

At 60fps, each frame has ~16.67ms. The render loop allocates this across stages:

| Stage | Typical Time | Tracked By |
|-------|-------------|------------|
| Layer collection + texture import | 0.5-3ms | `importTexture` |
| Compositing (all layers) | 1-5ms | `renderPass` |
| GPU submit | 0.1-0.5ms | `submit` |
| Total render | 2-8ms | `total` |

The `RenderLoop` applies frame rate limiting:
```typescript
private readonly VIDEO_FRAME_TIME = 16.67;  // ~60fps target
private readonly SCRUB_FRAME_TIME = 33;     // ~30fps during scrubbing
```

During scrubbing, the render rate is halved to avoid wasted renders while video seeks are in-flight.

### Idle Detection

The render loop enters idle mode after 1 second of no activity, stopping renders entirely. Any user interaction or state change calls `requestRender()` to wake it:
```typescript
private readonly IDLE_TIMEOUT = 1000; // 1s before idle
```

### Playback Health Monitoring

The `PlaybackHealthMonitor` polls every 500ms for anomalies:
- **FRAME_STALL**: video.currentTime unchanged for 1.5s during playback
- **WARMUP_STUCK**: GPU surface warmup taking > 3s
- **SEEK_STUCK**: video.seeking === true for > 2s
- **READYSTATE_DROP**: video.readyState < 2 during playback
- **GPU_SURFACE_COLD**: playing video not in videoGpuReady
- **RENDER_STALL**: no render for > 3s while playing
- **HIGH_DROP_RATE**: > 10 dropped frames/second

### DOM Overhead Mitigation

Each HTMLVideoElement creates a DOM node with an associated media pipeline (demuxer, decoder, audio sink). MasterSelects minimizes overhead by:

1. **Pausing off-screen videos** -- only clips at playhead are active
2. **Handoff reuse** -- same-source sequential clips (splits) reuse the same video element across cut boundaries instead of creating a new one
3. **WebCodecs path** -- bypasses HTMLVideoElement entirely, using `VideoDecoder` API directly (currently feature-flagged)
4. **NativeHelper** -- for ProRes/DNxHD, uses a separate Rust process, no DOM elements needed

---

## 8. The Full Render Path (End-to-End)

Tracing a single frame from playhead to pixels:

```
1. RenderLoop.loop() fires (requestAnimationFrame)
     |
2. RenderLoop calls RenderDispatcher.render(layers)
   where layers = LayerBuilderService.buildLayersFromStore()
     |
3. LayerBuilderService:
   a. createFrameContext() -- single store read for playhead, clips, tracks
   b. For each video track with clip at playhead:
      - Compute clipTime from playhead position, inPoint, speed
      - Resolve video source (handoff, proxy, or clip's own videoElement)
      - Build Layer object with source references
   c. mergeBackgroundLayers() -- interleave with background compositions
     |
4. RenderDispatcher.render(layers):
   a. compositorPipeline.beginFrame() -- clear external texture cache
   b. LayerCollector.collect(layers, deps):
      - For each layer (reverse order = bottom-up):
        - Video: try NativeHelper -> VideoFrame -> WebCodecs -> HTMLVideo
        - Image: cache texture, return textureView
        - Text: upload canvas to GPU, return textureView
        - Nested Comp: return placeholder (textureView set later)
      - Returns LayerRenderData[]
     |
5. Pre-render nested compositions:
   For each layer with nestedComposition:
     NestedCompRenderer.preRender() -> GPUTextureView
     Set data.textureView = view
     |
6. Compositor.composite(layerData, encoder, state):
   For each layer:
     - Select external or regular pipeline
     - Create bind group (sampler + readView + source + uniforms + mask)
     - Run composite render pass to writeView
     - Apply complex effects if any
     - Swap ping/pong
   Returns { finalView, layerCount }
     |
7. OutputPipeline.renderToCanvas(encoder, previewContext, bindGroup):
   - Renders finalView to the preview canvas
   - Also renders to export canvas if exporting
   - Also renders to all activeComp render targets
     |
8. device.queue.submit([commandBuffers])
   - Single batch submit for all command buffers (nested comp + main)
```

---

## 9. Key Insights and Trade-offs

### What Works Well
- **Zero-copy video import** via `importExternalTexture` avoids the single largest bottleneck (GPU upload)
- **Aggressive caching hierarchy** (scrubbing cache, per-time cache, last-frame cache, emergency hold) ensures no black flashes during scrubbing
- **Texture pool for nested comps** prevents GPU allocation churn
- **Selective pause/play** keeps decoder count within browser limits
- **Dual pipeline architecture** (external vs regular) cleanly separates the two texture type worlds

### Current Limitations
- **Full WebCodecs playback is feature-flagged off** (`useFullWebCodecsPlayback = false`), meaning preview relies on HTMLVideoElement with its inherent seek latency
- **No shared decoder across compositions** -- each nested comp video needs its own HTMLVideoElement or WebCodecs instance
- **VRAM pressure with many cached frames** -- 300 scrub cache frames at 1080p = 2.4GB, which is the entire VRAM on many integrated GPUs
- **Linux performance** is significantly worse due to Vulkan/VA-API inconsistencies
- **Post-reload warmup dance** -- the need for play()/pause() to activate GPU surfaces is a Chrome-specific quirk that adds complexity

### Future Optimization Vectors
- Enable `useFullWebCodecsPlayback` to bypass HTMLVideoElement latency
- Implement `useDecoderPool` for shared decoders across clips using the same source file
- Implement `useRenderGraph` for automatic resource scheduling and barrier insertion
- Reduce scrubbing cache size on low-VRAM devices based on GPU info from `engineStore`
