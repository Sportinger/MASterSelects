# Video Export & Encoding in a Browser NLE

## Research Agent #5 -- MasterSelects Export Pipeline Analysis

---

## 1. Architecture Overview: Timeline to MP4

MasterSelects implements a **two-tier export architecture** with distinct paths for different use cases:

| Path | Technology | Codecs | Use Case |
|------|-----------|--------|----------|
| **WebCodecs** | Browser-native VideoEncoder + mp4-muxer/webm-muxer | H.264, H.265, VP9, AV1 | Delivery (YouTube, social, web) |
| **FFmpeg WASM** | ffmpeg-core.js (ASYNCIFY build) | ProRes, DNxHR, FFV1, UTVideo, MJPEG | Professional intermediate / archival |

Both paths share the same **frame-by-frame rendering pipeline**: the WebGPU engine renders each frame offline, then passes the pixel data to the chosen encoder. This is a non-realtime export -- no `MediaRecorder` or canvas capture stream is involved.

**Key files:**
- `src/engine/export/FrameExporter.ts` -- Main orchestrator
- `src/engine/export/VideoEncoderWrapper.ts` -- WebCodecs encoder + muxer
- `src/engine/ffmpeg/FFmpegBridge.ts` -- FFmpeg WASM bridge
- `src/engine/audio/AudioExportPipeline.ts` -- Audio export orchestrator
- `src/engine/managers/ExportCanvasManager.ts` -- Zero-copy frame capture

---

## 2. Frame Capture: GPU Canvas to VideoFrame

The most critical performance question in browser video export is: how do you get rendered pixels from the GPU into the encoder?

MasterSelects supports two paths, selected automatically:

### 2.1 Zero-Copy Path (Preferred)

An `OffscreenCanvas` with a WebGPU context is created at export resolution. After the engine renders a frame into it, a `VideoFrame` is constructed directly from the canvas:

```typescript
// ExportCanvasManager.ts
async createVideoFrameFromExport(device: GPUDevice, timestamp: number, duration: number) {
  // CRITICAL: Wait for GPU to finish rendering before capturing frame
  await device.queue.onSubmittedWorkDone();

  const frame = new VideoFrame(this.exportCanvas, {
    timestamp,
    duration,
    alpha: 'discard',
  });
  return frame;
}
```

This avoids any CPU-side pixel read. The GPU texture backing the canvas is handed directly to the `VideoEncoder`, which can submit it to the hardware encoder without a roundtrip through system memory. The `device.queue.onSubmittedWorkDone()` call is essential -- without it, the frame capture races against the GPU render pass.

### 2.2 ReadPixels Fallback

When zero-copy is unavailable, the engine reads pixels back from the GPU via `readPixels()` (a `GPUBuffer` mapped back to CPU), then constructs a `VideoFrame` from raw RGBA bytes:

```typescript
// VideoEncoderWrapper.ts
const frame = new VideoFrame(pixels.buffer, {
  format: 'RGBA',
  codedWidth: this.settings.width,
  codedHeight: this.settings.height,
  timestamp: timestampMicros,
  duration: durationMicros,
});
```

This path is significantly slower because it involves a GPU-to-CPU transfer for every frame, plus the allocation of a `Uint8ClampedArray` buffer.

---

## 3. WebCodecs VideoEncoder Configuration

The `VideoEncoderWrapper` configures the browser's native hardware-accelerated encoder:

```typescript
// VideoEncoderWrapper.ts
this.encoder = new VideoEncoder({
  output: (chunk, meta) => {
    this.muxer.addVideoChunk(chunk, meta);
  },
  error: (e) => log.error('Encode error:', e),
});

await this.encoder.configure({
  codec: codecString,
  width: this.settings.width,
  height: this.settings.height,
  bitrate: this.settings.bitrate,
  framerate: this.settings.fps,
  latencyMode: 'quality',     // Prioritize quality over low latency
  bitrateMode: 'variable',    // VBR for better quality distribution
});
```

**Key design decisions:**
- `latencyMode: 'quality'` -- tells the encoder it can buffer frames for better compression (no real-time constraint)
- `bitrateMode: 'variable'` -- VBR allows the encoder to allocate more bits to complex scenes
- Keyframe interval defaults to 1 per second (`keyframeInterval = fps`), forced via `encode(frame, { keyFrame: true })`

### 3.1 Codec Strings

MasterSelects maps friendly codec names to precise WebCodecs codec strings:

```typescript
// codecHelpers.ts
'h264' -> 'avc1.4d0028'        // Main Profile, Level 4.0
'h265' -> 'hvc1.1.6.L93.B0'    // Main Profile, Level 3.1
'vp9'  -> 'vp09.00.10.08'      // Profile 0, Level 1.0, 8-bit
'av1'  -> 'av01.0.04M.08'      // Main Profile, Level 3.0, 8-bit
```

The H.264 profile uses Main at Level 4.0 for VLC compatibility. Before using a codec, `VideoEncoder.isConfigSupported()` is called to verify the browser can actually encode at the requested resolution/codec combination.

### 3.2 Hardware vs Software Encoding

MasterSelects does not explicitly control whether encoding happens in hardware or software -- that is the browser's decision. In practice:
- **H.264**: Nearly always hardware-accelerated (NVENC, Intel QSV, AMD VCE)
- **H.265**: Hardware on Windows (NVENC), software on Linux
- **VP9**: Software in most browsers
- **AV1**: Hardware on recent GPUs (Intel Arc, NVIDIA RTX 40+), software otherwise

The `checkCodecSupport()` function probes each codec at the target resolution to detect what is available.

---

## 4. Muxing: mp4-muxer and webm-muxer

MasterSelects uses two JavaScript muxer libraries (no FFmpeg needed for delivery formats):

| Container | Library | Audio | Video |
|-----------|---------|-------|-------|
| MP4 | `mp4-muxer` | AAC, Opus | H.264, H.265, VP9, AV1 |
| WebM | `webm-muxer` | Opus | VP9, AV1 |

```typescript
// VideoEncoderWrapper.ts -- MP4 path
this.muxer = new Mp4Muxer({
  target: new Mp4Target(),
  video: { codec: mp4VideoCodec, width, height },
  audio: { codec: this.audioCodec, sampleRate, numberOfChannels: 2 },
  fastStart: 'in-memory',   // moov atom at beginning for streaming
});
```

The `fastStart: 'in-memory'` option is significant: it places the MP4 moov atom at the beginning of the file, which is necessary for progressive playback. Without it, players must download the entire file before they can start playing.

**Container-codec validation** is enforced: WebM only accepts VP9/AV1 video and Opus audio. If the user selects H.264 with a WebM container, the system falls back to VP9.

After all video and audio chunks are added, `muxer.finalize()` writes the container structure, and the result is extracted as an `ArrayBuffer` from the `ArrayBufferTarget`, then wrapped in a `Blob`.

---

## 5. Audio Export Pipeline

Audio export is a **separate phase** that runs after video encoding. The pipeline is orchestrated by `AudioExportPipeline` and follows a clear 6-step process:

### 5.1 Pipeline Stages

```
1. Extract    -- Decode audio from source files (Web Audio API decodeAudioData)
2. Process    -- Apply speed changes / time-stretch (SoundTouchJS)
3. Effects    -- Apply EQ, volume, audio effects (AudioEffectRenderer)
4. Mix        -- Sum all tracks with correct timing (OfflineAudioContext)
5. Encode     -- Compress to AAC or Opus (WebCodecs AudioEncoder)
6. Mux        -- Add encoded chunks to the video container
```

### 5.2 Audio Extraction

Audio is decoded using the Web Audio API's `decodeAudioData`, which handles all common formats. For clips with custom in/out points, the buffer is trimmed:

```typescript
// AudioExtractor.ts
const arrayBuffer = await file.arrayBuffer();
const audioBuffer = await context.decodeAudioData(arrayBuffer);
const trimmedBuffer = this.trimBuffer(buffer, clip.inPoint, clip.outPoint);
```

A cache of up to 5 decoded buffers prevents re-decoding the same source file multiple times.

### 5.3 Time-Stretching

Speed changes (including keyframed speed ramps) are processed using SoundTouchJS. This handles the pitch preservation question: if `preservesPitch` is true (default), the audio tempo changes but pitch remains constant.

### 5.4 Mixing with OfflineAudioContext

The mixer uses `OfflineAudioContext` to sum all audio clips at their correct timeline positions:

```typescript
// AudioMixer.ts
const offlineContext = new OfflineAudioContext(numberOfChannels, totalSamples, sampleRate);

for (const track of activeTracks) {
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(gainNode);
  gainNode.connect(context.destination);
  source.start(startTime);  // Position on timeline
}

const mixedBuffer = await offlineContext.startRendering();
```

This handles overlapping clips, mute/solo states, and per-clip volume. Optional peak normalization prevents clipping when many tracks are summed.

### 5.5 Audio Encoding

The mixed buffer is encoded using `WebCodecs AudioEncoder`:

```typescript
// AudioEncoder.ts
this.encoder = new AudioEncoder({
  output: (chunk, meta) => this.handleChunk(chunk, meta),
  error: (e) => this.handleError(e),
});

this.encoder.configure({
  codec: 'mp4a.40.2',  // AAC-LC (or 'opus' as fallback)
  sampleRate: 48000,
  numberOfChannels: 2,
  bitrate: 256000,
});
```

Codec selection is automatic: AAC is preferred for MP4 containers, Opus for WebM. On Linux where AAC support may be missing, the system falls back to Opus. Audio is processed in 1024-sample frames (standard AAC frame size) and yields to the event loop every 100 frames.

---

## 6. FFmpeg WASM: Professional Codecs

For professional workflows requiring ProRes, DNxHR, or lossless codecs, MasterSelects loads FFmpeg as a WASM module. This is the ASYNCIFY (single-threaded) build with native FFmpeg encoders only.

### 6.1 Available Codecs

| Codec | Library | Use Case |
|-------|---------|----------|
| ProRes (prores_ks) | Native | Apple editing workflows |
| DNxHR (dnxhd) | Native | Avid / broadcast |
| FFV1 | Native | Lossless archival |
| UTVideo | Native | Fast lossless with alpha |
| MJPEG | Native | Simple intermediate |

Notably absent: H.264 (libx264), VP9 (libvpx), and Opus (libopus) -- these require pkg-config which fails in the Emscripten build environment. For H.264 delivery, the WebCodecs path is used instead.

### 6.2 FFmpeg Frame Input

FFmpeg receives frames as a single raw RGBA file concatenated in memory:

```typescript
// FFmpegBridge.ts
const allFrames = new Uint8Array(totalSize);
for (let i = 0; i < frames.length; i++) {
  allFrames.set(frames[i], i * frameSize);
}
fs.writeFile('/input/frames.raw', allFrames);
```

This approach has a **major memory implication**: all frames must fit in RAM simultaneously. For a 10-second 1080p export at 30fps, that is `1920 * 1080 * 4 * 300 = ~2.4 GB` of raw pixel data. This makes FFmpeg export impractical for long sequences at high resolutions.

The FFmpeg arguments use `-f rawvideo -pix_fmt rgba` as input format, with explicit `-frames:v` and `-t` to control output length.

### 6.3 Audio with FFmpeg

When audio is included, the `AudioExportPipeline.exportRawAudio()` method produces a mixed `AudioBuffer` (not encoded), which is converted to interleaved PCM float32 and written as raw audio:

```typescript
// FFmpegBridge.ts
private audioBufferToPCM(audioBuffer: AudioBuffer): Float32Array {
  const interleaved = new Float32Array(length * channels);
  for (let i = 0; i < length; i++) {
    for (let ch = 0; ch < channels; ch++) {
      interleaved[i * channels + ch] = channelData[ch][i];
    }
  }
  return interleaved;
}
```

FFmpeg then encodes this with the chosen audio codec (AAC for MOV, FLAC for MKV, PCM for AVI/MXF).

---

## 7. Non-Realtime Rendering: Frame-by-Frame Export

The export loop in `FrameExporter.export()` is purely sequential and non-realtime:

```
for each frame (0..totalFrames):
  1. Calculate timeline time for this frame
  2. Seek all video clips to correct source time
  3. Wait for video frames to be ready
  4. Build layer list (with transforms, effects, blend modes)
  5. Render via WebGPU engine
  6. Capture frame (zero-copy or readPixels)
  7. Encode frame via VideoEncoder
  8. Report progress
```

### 7.1 Two Decoding Modes

MasterSelects offers two modes for sourcing video frames during export:

**FAST mode** (default): Uses WebCodecs `VideoDecoder` with MP4Box.js parsing. The source file is loaded as an `ArrayBuffer`, parsed with MP4Box to extract samples, and decoded sequentially. For multi-clip timelines, a `ParallelDecodeManager` runs separate `VideoDecoder` instances per clip with prefetched frame buffers.

**PRECISE mode**: Falls back to `HTMLVideoElement.currentTime` seeking. More frame-accurate but significantly slower due to seek latency and browser decode overhead. Uses `requestVideoFrameCallback` when available for frame-accurate waiting.

### 7.2 Parallel Decoding

When 2+ video clips are in the export range, the `ParallelDecodeManager` initializes separate `VideoDecoder` instances per clip and prefetches frames ahead of the render position:

```typescript
// ParallelDecodeManager prefetches frames for the current timeline time
await parallelDecoder.prefetchFramesForTime(time);
parallelDecoder.advanceToTime(time);
const videoFrame = parallelDecoder.getFrameForClip(clipId, time);
```

Each clip's source file is parsed with MP4Box.js to extract codec-specific sample data, which is then fed to a dedicated `VideoDecoder`. The system uses the main timeline time to calculate the correct source time for each clip, accounting for speed, inPoint, outPoint, and reversed playback.

### 7.3 FrameContext Optimization

A per-frame `FrameContext` object caches all state lookups to avoid repeated `getState()` calls:

```typescript
// FrameExporter.ts
private createFrameContext(time: number, fps: number, frameTolerance: number): FrameContext {
  const state = useTimelineStore.getState();
  const clipsAtTime = state.getClipsAtTime(time);
  const trackMap = new Map(state.tracks.map(t => [t.id, t]));
  const clipsByTrack = new Map(clipsAtTime.map(c => [c.trackId, c]));
  return { time, fps, frameTolerance, clipsAtTime, trackMap, clipsByTrack, ... };
}
```

This reduces per-frame overhead from multiple O(n) lookups to a single `getState()` call plus O(1) Map lookups.

---

## 8. Codec Support Matrix

### WebCodecs Path (browser-native)

| Codec | MP4 | WebM | Hardware Accel | Notes |
|-------|-----|------|---------------|-------|
| H.264 | Yes | No | Nearly always | Most compatible, fastest encode |
| H.265 | Yes | No | Windows (NVENC) | Limited browser support |
| VP9 | Yes | Yes | Rare | Good quality, slow encode |
| AV1 | Yes | Yes | Recent GPUs only | Best compression, slowest |

### FFmpeg WASM Path

| Codec | MOV | MKV | AVI | MXF | Notes |
|-------|-----|-----|-----|-----|-------|
| ProRes | Yes | -- | -- | -- | 6 quality profiles (Proxy to 4444 XQ) |
| DNxHR | Yes | -- | -- | Yes | 5 profiles (LB to 444) |
| FFV1 | -- | Yes | Yes | -- | Lossless, 10-bit capable |
| UTVideo | Yes | Yes | Yes | -- | Fast lossless, alpha support |
| MJPEG | Yes | Yes | Yes | -- | Simple, widely compatible |

---

## 9. Memory and Streaming Considerations

### WebCodecs Path: Streaming by Design

The WebCodecs path is inherently streaming: each frame is encoded and immediately passed to the muxer. Only the muxer's internal buffer accumulates data. The `ArrayBufferTarget` used by both mp4-muxer and webm-muxer does hold the entire output in RAM, but the output is typically 10-100x smaller than raw frames.

For a 60-second 1080p H.264 export at 15 Mbps, the output is approximately 112 MB -- well within browser memory limits.

### FFmpeg Path: All-in-RAM Bottleneck

The FFmpeg WASM path concatenates ALL raw frames into a single `Uint8Array` before encoding. This is the primary scaling limitation:

```
Memory = width * height * 4 (RGBA) * totalFrames
1080p @ 30fps for 60 seconds = 1920 * 1080 * 4 * 1800 = ~14.9 GB
```

This makes the FFmpeg path unsuitable for long exports. A streaming approach (writing frames incrementally to the virtual filesystem) would require restructuring the FFmpeg bridge to interleave frame writing with encoding.

### Mitigation Strategies in the Codebase

- The `queueMicrotask` yield every 30 frames prevents the encoder from starving the event loop
- `VideoFrame.close()` is called immediately after encoding to release GPU memory
- The audio pipeline uses `OfflineAudioContext` which processes in a single pass without accumulating intermediate buffers
- Export canvas is cleaned up immediately after export completes

---

## 10. Export Progress and Error Handling

### Progress Reporting

Progress is reported through two phases with weighted percentages:

```typescript
// Video phase: 0-95% (or 0-100% without audio)
const videoWeight = includeAudio ? 0.95 : 1.0;
const videoPercent = ((frame + 1) / totalFrames) * 100 * videoWeight;

// Audio phase: 95-100%
percent: 95 + (audioProgress.percent * 0.05);
```

The `ExportProgress` type includes ETA calculation based on a rolling average of the last 30 frame times.

### Error Handling

Several error conditions are explicitly handled:
- **WebGPU device loss**: Checked after each render and frame capture. Throws a user-facing error suggesting keeping the browser tab in focus.
- **Codec not supported**: Detected before export via `VideoEncoder.isConfigSupported()` and falls back to a supported codec.
- **Missing video frames**: Individual frame failures log a warning and skip (continue), preventing a single bad frame from aborting the entire export.
- **Audio extraction failure**: Creates a silent buffer as fallback, so the export completes even if one clip's audio is unreadable.
- **Export cancellation**: Checked at multiple points in the loop (before seek, after encode), with proper cleanup of all resources.

---

## 11. Nested Composition Export

MasterSelects supports compositions-within-compositions. During export, the `ExportLayerBuilder` detects nested composition clips and recursively builds their layer trees:

```typescript
// ExportLayerBuilder.ts
if (clip.isComposition && clip.nestedClips && clip.nestedClips.length > 0) {
  const nestedLayers = buildNestedLayersForExport(clip, nestedTime, ...);
  // These layers are rendered into a separate texture by NestedCompRenderer
  layers.push({
    ...baseLayerProps,
    source: { type: 'image', nestedComposition: nestedCompData },
  });
}
```

The nested composition's internal timeline is evaluated at the correct nested time (accounting for the parent clip's position and inPoint). Both regular and nested video clips can use the parallel decoder when in FAST mode. For audio, nested compositions with pre-mixed audio (`mixdownBuffer`) are handled specially -- the pre-rendered audio buffer is used directly without re-extracting from individual clips.

---

## 12. Summary: What Makes This Work in a Browser

MasterSelects demonstrates that professional-grade video export is achievable entirely in the browser by combining:

1. **WebGPU** for GPU-accelerated rendering at arbitrary resolutions
2. **WebCodecs VideoEncoder** for hardware-accelerated H.264/H.265/VP9/AV1 encoding
3. **mp4-muxer / webm-muxer** for pure-JS container writing (no native code needed for delivery formats)
4. **FFmpeg WASM** as an escape hatch for professional codecs (ProRes, DNxHR)
5. **Zero-copy frame transfer** from OffscreenCanvas to VideoEncoder, avoiding CPU roundtrips
6. **OfflineAudioContext** for efficient multi-track audio mixing
7. **ParallelDecodeManager** with per-clip VideoDecoder instances for multi-clip export performance

The main limitations are:
- **FFmpeg memory usage** for long sequences (all frames in RAM)
- **No 10-bit or HDR** in the WebCodecs path (browser limitation)
- **Encoding speed** depends entirely on browser/hardware support
- **Tab must remain in focus** to prevent WebGPU device loss
