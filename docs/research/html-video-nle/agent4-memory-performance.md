# Agent 4: Memory Management, Performance & Resource Limits

## 1. Browser Memory Limits for a Web NLE

Web browsers impose hard limits on memory consumption. Chrome's V8 heap is capped at ~4 GB on 64-bit systems, but GPU/VRAM is separate and managed by the browser process. Key constraints:

- **JS Heap**: ~4 GB max. Each `ImageData` for a 1080p frame is ~8 MB (1920x1080x4 bytes). Caching 900 frames (30 seconds at 30fps) costs ~7.2 GB if unconstrained.
- **GPU/VRAM**: Shared with the browser compositor. Chrome will kill a tab that consumes too much VRAM, but there is no explicit quota — the OS OOM killer acts as the last resort.
- **ArrayBuffer / Blob storage**: Each `File` kept in JS memory holds its full data. A 1 GB video import means 1 GB of JS-reachable memory *plus* the blob URL reference.

MasterSelects mitigates these by setting explicit budget caps in its caching layers (see Sections 4-6).

---

## 2. VideoFrame Lifecycle — Why `.close()` Is Critical

`VideoFrame` (from WebCodecs) holds a reference to GPU-side decoded pixel data. Unlike regular JS objects, **the garbage collector does not release the GPU memory** — you must call `.close()` explicitly. Failing to close frames causes unbounded VRAM growth and eventual STATUS_BREAKPOINT crashes.

### MasterSelects' Approach

The `WebCodecsPlayer` (`src/engine/WebCodecsPlayer.ts`) is meticulous about closing frames at every lifecycle boundary:

**Frame buffer overflow eviction** (line 552-556):
```typescript
while (this.frameBuffer.length > WebCodecsPlayer.MAX_FRAME_BUFFER) {
  const oldest = this.frameBuffer[0];
  if (oldest === this.currentFrame) break; // Don't close the displayed frame
  this.frameBuffer.shift()!.close();
}
```

**Seek intermediate frames** are closed immediately (line 585):
```typescript
// Drop intermediate GOP traversal frame to avoid visual jumps.
frame.close();
```

**Frame presentation** closes the previous frame before replacing (line 867-868):
```typescript
if (this.currentFrame) {
  this.currentFrame.close();
}
this.currentFrame = frame;
```

**Destroy path** wraps close in try/catch since frames may already be closed (line 2009-2016):
```typescript
if (this.currentFrame) {
  try { this.currentFrame.close(); } catch { /* Already closed */ }
  this.currentFrame = null;
}
```

The `TextureManager.importVideoTexture()` also guards against closed frames before passing them to WebGPU (`src/engine/texture/TextureManager.ts`, line 222-228):
```typescript
if ((source as any).closed || source.codedWidth === 0 || source.codedHeight === 0) {
  return null; // Guard against closed VideoFrames — crashes the GPU process
}
```

**Buffer size**: `MAX_FRAME_BUFFER = 8` limits the number of decoded-but-not-yet-displayed frames. At 1080p each `VideoFrame` is ~8 MB VRAM, so the buffer caps at ~64 MB.

The `ParallelDecodeManager` (`src/engine/ParallelDecodeManager.ts`) for multi-clip exports follows the same discipline, closing frames at every eviction, replacement, and cleanup point (lines 148, 156, 295, 333, 590, 597, 622, 741, 1027, 1124).

---

## 3. HTMLVideoElement Pooling vs. Creation

MasterSelects creates a **new HTMLVideoElement per clip** during deserialization (`src/stores/timeline/serializationUtils.ts`, line 990):
```typescript
const video = document.createElement('video');
video.src = fileUrl;
video.muted = true;
video.playsInline = true;
video.preload = 'auto';
video.crossOrigin = 'anonymous';
```

There is **no element pool**. Instead, the project uses these strategies to manage video element count:

1. **WebCodecsPlayer cache** (`globalWcpCache`): Full-mode WebCodecs players are cached by `mediaFileId` across composition switches, avoiding re-reading entire video files (which was "the #1 cause of OOM crashes" per the code comment at line 1173).

2. **Cleanup on composition switch** (`serializationUtils.ts`, line 1174-1199): When switching compositions, all video elements are paused, their blob URLs revoked, `src` cleared, and `.load()` called to release internal decoder resources. WebCodecsPlayers are only paused, not destroyed.

3. **Seamless cut transitions** (`VideoSyncManager`, `lastTrackState`): When sequential clips on the same track share the same source file (split clips), the video element from the outgoing clip keeps playing through the cut — avoiding a pause/play gap and the creation of a redundant video element.

Each browser has a per-origin limit on active media decoders (typically 16-32 in Chrome). MasterSelects relies on the WebCodecs path for playback-active clips and keeps HTMLVideoElements mainly as fallback/thumbnail sources.

---

## 4. Texture Caching — TextureManager

The `TextureManager` (`src/engine/texture/TextureManager.ts`) maintains several caches:

| Cache | Key | Purpose | Cleanup Strategy |
|-------|-----|---------|-----------------|
| `imageTextures` | `HTMLImageElement` ref | Images uploaded once, cached forever | GC-based (Map cleared, textures not destroyed) |
| `canvasTextures` | `HTMLCanvasElement` ref | Text clip canvases, invalidated by new canvas ref | GC-based |
| `dynamicTextures` | Layer ID string | NativeDecoder frames, re-uploaded every frame | **Explicit `.destroy()`** on eviction |
| `cachedImageViews` | `GPUTexture` ref | Avoid creating views every frame | GC-based |

The critical design decision: **image and canvas textures are NOT explicitly destroyed** — the comment at line 274 explains:
```typescript
// Don't destroy textures - let GC handle to avoid GPU conflicts
// WebGPU will automatically release GPU resources when JS objects are GC'd
// AND the GPU is done using them
```

This is deliberate: calling `.destroy()` on a texture while the GPU is still referencing it (e.g., in a pending command buffer) causes validation errors. By clearing the Map reference and letting GC handle it, MasterSelects avoids a race condition between JS cleanup and GPU command submission.

**Dynamic textures** (NativeDecoder frames) are the exception — they ARE explicitly destroyed because MasterSelects owns the entire lifecycle and can guarantee no pending GPU references:
```typescript
removeDynamicTexture(key: string): void {
  const entry = this.dynamicTextures.get(key);
  if (entry) {
    entry.texture.destroy();
    this.dynamicTextures.delete(key);
  }
}
```

---

## 5. ScrubbingCache — Multi-Tier Frame Caching

The `ScrubbingCache` (`src/engine/texture/ScrubbingCache.ts`) implements three tiers of caching with explicit memory budgets:

### Tier 1: Scrubbing Frame Cache (GPU textures)
- **Key**: `"videoSrc:quantizedFrameTime"` (quantized to 30fps boundaries)
- **Max entries**: 300 frames (~10 seconds at 30fps)
- **Memory estimate**: ~2.4 GB VRAM at 1080p (300 * 1920 * 1080 * 4 bytes)
- **Eviction**: LRU via Map insertion order, oldest deleted first
- **Nearest-frame fallback**: `getNearestCachedFrame()` searches +/- 6 frames to avoid black flashes during seeks

### Tier 2: RAM Preview Composite Cache (CPU ImageData)
- **Key**: Quantized time (30fps)
- **Max entries**: 900 frames (30 seconds at 30fps)
- **Memory cap**: 512 MB explicit (`maxCompositeCacheBytes`)
- **Eviction**: Dual constraint — frame count OR byte limit, whichever triggers first
- **Storage**: `ImageData` (CPU-side) for memory efficiency vs GPU textures

### Tier 3: GPU Frame Cache (composited textures)
- **Key**: Quantized time
- **Max entries**: 60 frames
- **Memory estimate**: ~500 MB at 1080p
- **Purpose**: Avoids CPU-to-GPU re-upload of composite cache hits during RAM preview playback

### Last-Frame Cache
A separate per-video-element cache keeps the **last successfully rendered frame** visible during seeks. Uses texture reuse when dimensions match (line 230-248) to avoid creating new textures on every seek. Also supports `captureVideoFrameViaImageBitmap()` — the only browser API that forces Chrome to actually decode a frame after page reload (line 280-289).

---

## 6. CacheManager Orchestration

The `CacheManager` (`src/engine/managers/CacheManager.ts`) wraps `ScrubbingCache` and adds lifecycle management:

- **`initialize(device)`**: Creates `ScrubbingCache` bound to the GPU device
- **`handleDeviceLost()`**: Nulls out the scrubbing cache entirely — all GPU textures are invalid after device loss
- **`clearAll()`**: Full cache flush, used on composition switch via `engine.clearCaches()`

This clean separation means device loss recovery is straightforward: null everything, re-initialize from the restored device.

---

## 7. Garbage Collection Impact on Playback

GC pauses are a real concern for 60fps playback. MasterSelects addresses this through:

1. **Pre-allocated ring buffers**: `PerformanceStats` uses `Float32Array(60)` for frame timing, avoiding per-frame allocations (`src/engine/stats/PerformanceStats.ts`, line 31).

2. **Object reuse over creation**: The `LayerCollector` reuses the `layerRenderData` array each frame (`this.layerRenderData.length = 0` instead of allocating new).

3. **WeakMap/WeakSet for DOM references**: `ScrubbingCache` uses `WeakMap<HTMLVideoElement, ...>` for owner tracking (line 26-31), allowing GC to reclaim entries when video elements are removed.

4. **Map-based LRU instead of sort**: All caches use Map insertion order for O(1) LRU operations, avoiding sorting which would create temporary arrays.

5. **Frame drop detection**: The `RenderLoop` monitors RAF gaps and counts dropped frames. A gap exceeding `2 * targetFrameTime` is recorded as a drop, which could be GC-related.

---

## 8. Tab Throttling and Background Behavior

MasterSelects does **not** currently handle `visibilitychange` events — a grep for `visibilitychange`, `document.hidden`, and `page hide` returned zero results.

However, the system is implicitly resilient:

- **RenderLoop idle detection** (`src/engine/render/RenderLoop.ts`): The loop enters idle mode after 1 second of inactivity, reducing GPU work to zero.
- **RAF-based loop**: `requestAnimationFrame` is automatically throttled by the browser to ~1fps in background tabs.
- **Export warning**: `FrameExporter` (`src/engine/export/FrameExporter.ts`, line 127) detects device loss during export with the message: "Try keeping the browser tab in focus."

**Risk**: If a user starts export and switches tabs, Chrome may throttle the RAF loop, slow down `setTimeout`/`setInterval`, and even lose the WebGPU device. This is documented via the export error message but not programmatically prevented.

---

## 9. WebGPU Device Lost — Recovery Mechanism

The `WebGPUContext` (`src/engine/core/WebGPUContext.ts`) implements automatic device recovery:

1. **Detection**: `device.lost.then(callback)` fires on device loss
2. **Notification**: All registered `deviceLostCallbacks` are called
3. **Cleanup** (`WebGPUEngine.handleDeviceLost()`): RenderLoop stops, all GPU resources are nulled (textures, pipelines, render targets, caches)
4. **Recovery**: After 100ms delay, `WebGPUContext.initialize()` is re-called
5. **Restoration** (`handleDeviceRestored()`): All pipelines re-created, canvases re-configured, render loop restarted
6. **Retry limit**: Max 3 recovery attempts before giving up with "Please reload the page"

The Vulkan-specific delays during initialization (50ms after device creation, 100ms after pipeline creation, 50ms before large texture allocation) indicate real-world stability issues on Linux/Vulkan backends where the GPU memory manager needs settling time.

---

## 10. Object URL Lifecycle

### BlobUrlManager — Centralized Leak Prevention

The `BlobUrlManager` (`src/stores/timeline/helpers/blobUrlManager.ts`) is a singleton that tracks all blob URLs by clip ID and type:

```typescript
class BlobUrlManager {
  private urls = new Map<string, Map<UrlType, ManagedUrl>>();
  private totalCreated = 0;
  private totalRevoked = 0;
```

Key operations:
- **`create(clipId, file, type)`**: Auto-revokes any existing URL of the same type before creating a new one
- **`revokeAll(clipId)`**: Called when a clip is removed from the timeline
- **`transfer(from, to, type)`**: Moves ownership without revoking (used during clip splitting)
- **`share(from, to, type)`**: Both clips reference the same URL (split clips sharing source)
- **`getStats()`**: Returns active/created/revoked counts for debugging

### File Management Cleanup

`fileManageSlice.ts` (line 25-26) revokes URLs when a file is removed from the media pool:
```typescript
if (file?.url) URL.revokeObjectURL(file.url);
if (file?.thumbnailUrl?.startsWith('blob:')) URL.revokeObjectURL(file.thumbnailUrl);
```

### Composition Switch Cleanup

`serializationUtils.ts` (line 1178) revokes video blob URLs during composition transitions:
```typescript
try { if (video.src) URL.revokeObjectURL(video.src); } catch {}
video.removeAttribute('src');
video.load(); // Forces browser to release internal decoder resources
```

The `.load()` call after clearing `src` is critical — without it, the browser keeps the video decoder alive even though no source is attached.

---

## 11. Proxy/Offline Strategies for Large Files

### ProxyGenerator — WebCodecs Frame Extraction

The `ProxyGenerator` (`src/services/proxyGenerator.ts`) decodes source video using hardware `VideoDecoder` and saves individual JPEG frames:

- **Target resolution**: Max 1280px wide (`PROXY_MAX_WIDTH`)
- **Frame rate**: 30fps (`PROXY_FPS`)
- **JPEG quality**: 0.82
- **Parallel encoding**: Pool of 8 `OffscreenCanvas` instances for concurrent `convertToBlob`
- **Batch decoding**: 30 samples at a time before yielding to avoid blocking

VideoFrames are closed immediately after canvas draw (line 427-430):
```typescript
if (existing) existing.close();
// ... draw to canvas ...
frame.close();
```

### ProxyFrameCache — In-Memory LRU

The `ProxyFrameCache` (`src/services/proxyFrameCache.ts`) loads JPEG frames from project folder storage:

- **Max cache**: 900 `HTMLImageElement` entries (30 seconds at 30fps)
- **Preload ahead**: 60 frames (2 seconds) during playback
- **Preload behind**: 30 frames (1 second) for reverse scrubbing
- **Scrub mode**: 90 frames bidirectional preload
- **Parallel loads**: 16 concurrent frame loads
- **Direction-aware**: Prioritizes frames in the scrub direction for visual continuity

Storage is OPFS-based (Origin Private File System) via the project folder service, not IndexedDB, for better performance with large numbers of small files.

---

## 12. SnapshotManager — Structural Sharing for Undo/Redo

The `SnapshotManager` (`src/engine/structuralSharing/SnapshotManager.ts`) minimizes memory pressure from the undo/redo history:

- **Reference sharing**: Unchanged clips share object references with the previous snapshot instead of being cloned
- **Change detection**: Uses Zustand's immutable update pattern — if `prevClipRefs.get(clip.id) !== clip`, the clip changed
- **DOM ref exclusion**: Video/audio elements are NOT included in snapshots (stored separately in DomRefRegistry)

This means a 50-clip timeline where only 1 clip changed creates a snapshot that allocates memory for ~1 clip, not 50.

---

## 13. PlaybackHealthMonitor — Anomaly Detection

The `PlaybackHealthMonitor` (`src/services/playbackHealthMonitor.ts`) polls every 500ms and detects:

| Anomaly | Threshold | Meaning |
|---------|-----------|---------|
| `FRAME_STALL` | 1.5s unchanged `currentTime` | Video decoder frozen |
| `WARMUP_STUCK` | 3s in warmup state | GPU surface won't activate |
| `SEEK_STUCK` | 2s with `video.seeking === true` | Browser seek deadlock |
| `READYSTATE_DROP` | `readyState < 2` while playing | Decoder underrun |
| `GPU_SURFACE_COLD` | Playing video not GPU-ready | Zero-copy texture import will fail |
| `RENDER_STALL` | 3s without render | RenderLoop frozen |
| `HIGH_DROP_RATE` | >10 drops/second | Sustained frame drops |

Escalation: If a clip triggers 3 anomalies within 12 seconds, the monitor escalates recovery (with a 15-second cooldown per clip to prevent recovery loops).

---

## 14. Summary of Memory Budgets

| Resource | Budget | Location |
|----------|--------|----------|
| Scrubbing cache GPU textures | 300 frames (~2.4 GB VRAM at 1080p) | `ScrubbingCache.maxScrubbingCacheFrames` |
| RAM Preview composite cache | 900 frames / 512 MB | `ScrubbingCache.maxCompositeCacheFrames/Bytes` |
| GPU frame cache | 60 frames (~500 MB) | `ScrubbingCache.maxGpuCacheFrames` |
| WebCodecs frame buffer | 8 frames (~64 MB) | `WebCodecsPlayer.MAX_FRAME_BUFFER` |
| Proxy frame image cache | 900 entries | `ProxyFrameCache.MAX_CACHE_SIZE` |
| Proxy preload parallelism | 16 concurrent loads | `ProxyFrameCache.PARALLEL_LOAD_COUNT` |
| Device recovery attempts | 3 max | `WebGPUContext.MAX_RECOVERY_ATTEMPTS` |
| Canvas encoding pool | 8 OffscreenCanvases | `ProxyGenerator.CANVAS_POOL_SIZE` |

---

## 15. Identified Gaps and Risks

1. **No `visibilitychange` handling**: Background tab throttling can cause device loss during export with no automatic mitigation beyond an error message.

2. **GC-reliant texture cleanup**: Most GPU textures rely on GC for deallocation. Under heavy memory pressure, GC may not run frequently enough, and VRAM could accumulate before JS GC cycles.

3. **No video element pooling**: Each clip creates its own `HTMLVideoElement`. For timelines with 50+ clips, this could hit browser decoder limits. The WebCodecs path mitigates this for playback but not for thumbnail generation.

4. **Scrubbing cache VRAM budget**: 300 frames at 1080p is ~2.4 GB VRAM. On GPUs with 4 GB total VRAM (common integrated GPUs), this could cause device loss. There is no dynamic budget adjustment based on available VRAM.

5. **Blob URL leak potential**: While `BlobUrlManager` tracks URLs per clip, some code paths (e.g., `projectSlice.ts`, `importPipeline.ts`) create blob URLs outside the manager. These require manual `revokeObjectURL` calls that could be missed.

6. **No mobile-specific resource limits**: iOS Safari has aggressive memory limits (~1 GB for web content) and autoplay restrictions. The codebase has a `components/mobile/` directory but no evidence of reduced cache budgets or mobile-specific resource management.
