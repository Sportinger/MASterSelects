# WebCodecs in NLE Web Applications: Research Report

> **Date:** 2026-03-08
> **Scope:** WebCodecs API usage in browser-based Non-Linear Editors (NLEs), with analysis of MasterSelects' current implementation

---

## Table of Contents

1. [WebCodecs API Overview](#1-webcodecs-api-overview)
2. [WebCodecs in Video Editing Web Apps](#2-webcodecs-in-video-editing-web-apps)
3. [Real-World Examples](#3-real-world-examples)
4. [WebCodecs + WebGPU Pipeline](#4-webcodecs--webgpu-pipeline)
5. [Limitations and Challenges](#5-limitations-and-challenges)
6. [Best Practices](#6-best-practices)
7. [Alternatives and Fallbacks](#7-alternatives-and-fallbacks)
8. [Latest Developments (2025-2026)](#8-latest-developments-2025-2026)
9. [MasterSelects: Current Implementation Analysis](#9-masterselects-current-implementation-analysis)
10. [Recommendations for MasterSelects](#10-recommendations-for-masterselects)

---

## 1. WebCodecs API Overview

The [WebCodecs API](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API) is a low-level browser API that provides direct, high-performance access to the browser's built-in media encoders and decoders. Unlike higher-level abstractions (`HTMLVideoElement`, Media Source Extensions), WebCodecs exposes individual frames and encoded chunks to JavaScript.

**Key difference from FFmpeg.wasm:** WebCodecs is NOT running inside a WebAssembly sandbox. It is a native JavaScript API implemented in the browser engine with privileged access to system-level codec APIs: VA-API (Linux), VideoToolbox (macOS), Media Foundation (Windows). Hardware acceleration is available for both encoding and decoding.

### Key Classes

| Class | Purpose |
|-------|---------|
| `VideoDecoder` | Decodes `EncodedVideoChunk` into `VideoFrame` objects |
| `VideoEncoder` | Encodes raw `VideoFrame` into `EncodedVideoChunk` objects |
| `AudioDecoder` | Decodes encoded audio data into `AudioData` objects |
| `AudioEncoder` | Encodes raw `AudioData` into encoded audio chunks |
| `VideoFrame` | Single decoded video frame; holds GPU memory, must be manually `.close()`d |
| `EncodedVideoChunk` | Wrapper for compressed video data (timestamp, type, duration) |
| `AudioData` | Wrapper for raw decoded audio samples |

### Browser Support (2025-2026)

| Browser | Status |
|---------|--------|
| Chrome/Chromium (Edge, Opera) | Full support since Chrome 94 (2021). Chrome 138 added rotation/flip metadata |
| Firefox Desktop | Support shipped H1 2024. HEVC pass-through under discussion |
| Firefox Mobile | Still in progress |
| Safari 16.4-17.x | Partial (VideoDecoder/VideoEncoder only) |
| Safari 26+ (late 2025) | **Full support** including AudioEncoder and AudioDecoder. Fixed WebGPU black-screen bugs |

The W3C specification targets **Candidate Recommendation Q1 2026** and **full W3C Recommendation Q4 2026**.

### What WebCodecs Does NOT Include

No built-in API for demuxing (parsing containers like MP4/WebM) or muxing (writing containers). External libraries required:
- **MP4Box.js** for MP4/ISOBMFF containers
- **jswebm** for WebM containers
- **Remotion's media-parser** for higher-level TypeScript abstraction
- **mp4-muxer / webm-muxer** for writing containers

---

## 2. WebCodecs in Video Editing Web Apps

### Frame-Accurate Video Decoding

This is the primary reason NLE web applications adopt WebCodecs. With `HTMLVideoElement`, seeking is imprecise -- the browser controls frame selection internally. WebCodecs decodes each frame individually, giving the application full control over which frames are produced and when.

The decode path bypasses the browser's black-box I-frame/B-frame/P-frame reconstruction logic. The application manages GOP (Group of Pictures) navigation directly -- essential for thumbnail generation, timeline scrubbing, and frame-by-frame stepping.

### Real-Time Preview and Playback

WebCodecs-decoded `VideoFrame` objects can be:
1. Rendered to `<canvas>` via `drawImage()`
2. Imported as GPU textures (`GPUExternalTexture` in WebGPU)
3. Processed through compositing pipelines (multiple layers, effects, overlays)

### Video Encoding and Export

`VideoEncoder` accepts raw `VideoFrame` objects and produces `EncodedVideoChunk` objects. Muxing must be handled externally.

Key encoding considerations:
- Hardware acceleration available, but software/CPU encoding produces **better compression quality** at the same bitrate
- `encodeQueueSize` enables back-pressure management
- Device capabilities vary -- older mobile devices may not support encoding above 1080p
- GPU encoding produces larger files or worse quality vs CPU encoding

### Performance vs HTMLVideoElement

| Aspect | HTMLVideoElement | WebCodecs |
|--------|-----------------|-----------|
| Frame access | None (black box) | Full per-frame control |
| Seeking precision | Approximate | Frame-accurate |
| Concurrent streams | Limited by DOM overhead | Many decoders possible |
| GPU integration | Limited | Direct via GPUExternalTexture |
| Hardware acceleration | Yes (internal) | Yes (exposed to JS) |
| Color space | Handled internally | YUV frames exposed |

---

## 3. Real-World Examples

### Commercial Products

**Clipchamp (Microsoft):** The most documented implementation. Wraps WebCodecs as codec plugins inside their FFmpeg WASM port. `VideoEncoder` API lifecycle runs from JavaScript while FFmpeg WASM handles container muxing. Presented at the W3C/SMPTE Joint Workshop on Professional Media Production.

**Kapwing:** Uses WebCodecs for thumbnail generation in Web Workers (avoiding main thread blocking). Has moved key decoding operations to WebCodecs. Detailed in a web.dev case study.

**CapCut (ByteDance):** Uses WebCodecs for browser-based editing with performance approaching desktop software.

### Open-Source Projects

| Project | Description |
|---------|-------------|
| **Diffusion Studio Core** | TypeScript video engine with timeline, keyframing, effects, transitions, masking, HW-accelerated rendering |
| **OpenVideo** | WebCodecs + PixiJS rendering engine with compositor |
| **Omniclip** | Full NLE web app using WebCodecs, supports up to 4K |
| **Browser Studio** | "Fastest WebCodecs renderer," WebGPU + WebGL + WebCodecs |
| **OpenReel Video** | React + TypeScript + WebCodecs + WebGPU, open-source CapCut alternative |
| **Framecrafter** | NLE aiming for Final Cut Pro-level export performance in browser |
| **Remotion** | Programmatic video creation with deep WebCodecs + media-parser integration |

---

## 4. WebCodecs + WebGPU Pipeline

### Architecture

```
Demuxer (MP4Box.js) → EncodedVideoChunk → VideoDecoder → VideoFrame
    → GPUExternalTexture → Shader Processing → Canvas (preview)
                                             → VideoEncoder (export)
```

### Pipeline Steps

1. **Demux:** Parse container (MP4, WebM) to extract `EncodedVideoChunk` objects
2. **Decode:** `VideoDecoder` produces `VideoFrame` objects
3. **Import:** `device.importExternalTexture({ source: videoFrame })` → zero-copy `GPUExternalTexture`
4. **Bind:** Bind external texture to render pipeline via bind groups
5. **Process:** Run vertex + fragment shaders for effects
6. **Composite:** Ping-pong rendering to combine multiple layers
7. **Output:** Render to canvas for preview, or read pixels for `VideoEncoder` during export

### WGSL Shader Usage

```wgsl
@group(0) @binding(0) var videoTexture: texture_external;
@group(0) @binding(1) var videoSampler: sampler;

@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSampleBaseClampToEdge(videoTexture, videoSampler, uv);
}
```

Note: `texture_external` uses `textureSampleBaseClampToEdge` (not `textureSample`). Images use `texture_2d<f32>` with standard sampling.

### Key Benefits

- **Zero-copy:** VideoFrame data stays on GPU. No CPU-GPU roundtrip.
- **Native YUV support:** WebGPU handles YUV-to-RGB in shader via external textures.
- **Parallel processing:** Multiple layers composited in a single GPU pass.

### Critical Caveats

- `GPUExternalTexture` is only valid for the **current microtask**. Must be re-imported every frame.
- **Dual-GPU systems** (integrated + dedicated) can cause extra CPU-GPU-GPU copies, hurting performance significantly ([w3c/webcodecs#873](https://github.com/w3c/webcodecs/issues/873)).
- GPU encoding produces larger files / worse quality vs CPU encoding at same bitrate.

---

## 5. Limitations and Challenges

### Seeking / Random Access (The Biggest NLE Challenge)

This is the most significant obstacle for NLE applications:

- After `flush()` or `configure()`, the decoder **requires** the next input to be a keyframe (IDR frame). No way around this.
- **Frame-stepping** (arrow key navigation) requires decoding from nearest previous keyframe. With GOP size 250, you must decode up to 250 frames to display one.
- **No way to flush output queue without invalidating decode pipeline** ([w3c/webcodecs#698](https://github.com/w3c/webcodecs/issues/698)).
- H.264 Recovery Point SEI frames **not recognized as keyframes** ([w3c/webcodecs#650](https://github.com/w3c/webcodecs/issues/650)).

**Workarounds:**
1. Pre-parse container → build keyframe index for fast GOP lookup
2. Cache decoded frames within each GOP for instant frame-stepping
3. `decoder.flush()` → re-feed from nearest preceding keyframe to target
4. Dual decoders: one for playback, another for seeking/scrubbing
5. Feed dummy IDR frame before real stream, drop first output
6. Encode every frame as keyframe (much larger files)

### Memory Management with VideoFrame

**Most critical operational concern.** `VideoFrame` holds GPU/CPU memory that is NOT automatically garbage collected in time.

- Every frame **must** be explicitly closed via `frame.close()`
- `frame.clone()` references same buffer (cheap, but both must be closed)
- `frame.copyTo()` creates expensive CPU-side copy (10-20ms desktop, 40-60ms mobile for 1080p)
- Accessing a closed VideoFrame **crashes the GPU process** (STATUS_BREAKPOINT on Linux/Vulkan)

### Codec Support Variations

Always check at runtime:

```typescript
const support = await VideoDecoder.isConfigSupported({
  codec: 'avc1.42E01E', // H.264 Baseline
  codedWidth: 1920,
  codedHeight: 1080,
});
```

VP9: ~85%+ on modern browsers. H.264: most universal fallback. HEVC: hardware-dependent, inconsistent.

### Threading

- VideoFrame is `Transferable` → can move between Workers via `postMessage`
- Decoder/encoder callbacks fire on creating thread → high-frequency callbacks on main thread cause jank
- Best practice: decoders/encoders in dedicated Web Workers + `OffscreenCanvas` for rendering
- SharedArrayBuffer enables parallel processing but requires cross-origin isolation headers

---

## 6. Best Practices

### Frame Queue Management

A full pipeline has ~6 queues: demux, video decode, audio decode, processing, video encode, audio encode. Each must be balanced.

```typescript
// Back-pressure: drop frame if encoder congested
function encodeFrame(frame: VideoFrame) {
  if (encoder.encodeQueueSize > 2) {
    frame.close(); // Drop frame
    return;
  }
  encoder.encode(frame, { keyFrame: false });
  frame.close();
}
```

### Seeking Strategies for NLEs

1. **Keyframe index:** Pre-parse container → map timestamps to keyframe positions
2. **GOP cache:** Cache decoded frames per GOP → instant frame-stepping within range
3. **Flush + re-feed:** On seek, `flush()` → feed from nearest keyframe through target
4. **Dual decoders:** One for playback, one for scrubbing (avoids pipeline disruption)
5. **Prefetch:** Pre-decode window of future frames (e.g., next 30) for smooth playback

### AudioData Handling

- Same `close()` requirement as VideoFrame
- Synchronize audio/video by **timestamp**, not frame/sample count
- Resample to common sample rate before mixing
- Use `AudioContext` (Web Audio API) for playback; WebCodecs for decode/encode only

### Web Worker Architecture (Recommended)

```
Main Thread:
  - UI rendering (React/Canvas)
  - Timeline state management
  - User interaction

Decode Worker:
  - VideoDecoder + AudioDecoder
  - Container parsing (MP4Box.js)
  - Transfers VideoFrame to render thread

Render Worker (OffscreenCanvas):
  - WebGPU compositing
  - Effect processing
  - Preview output

Encode Worker:
  - VideoEncoder + AudioEncoder
  - Container muxing
  - File output
```

---

## 7. Alternatives and Fallbacks

### FFmpeg.wasm vs WebCodecs

| Aspect | FFmpeg.wasm | WebCodecs |
|--------|------------|-----------|
| **Speed (1080p H.264)** | ~25fps, CPU-bound | Up to 70x faster (HW-accelerated) |
| **GPU** | None (WASM sandbox) | OS-level HW acceleration |
| **Memory** | Entire file in memory, 2GB limit | Streaming/chunked, no practical limit |
| **Codec coverage** | Nearly all codecs/formats | Only browser-supported codecs |
| **Browser support** | All browsers via WASM | Chrome full, Safari/Firefox catching up |
| **Muxing/demuxing** | Built-in (full FFmpeg) | External library required |
| **API complexity** | CLI-like interface | Low-level, significant plumbing |
| **Best for** | Complex filters, obscure formats | Real-time preview, HW export |

### Hybrid Approach (Recommended by Industry)

- **FFmpeg.wasm for:** audio extraction, waveform generation, thumbnails, remuxing, subtitle processing, obscure formats (ProRes, DNxHR, HAP)
- **WebCodecs for:** real-time timeline preview, HW-accelerated encode/decode, canvas-based compositing, scrubbing

Clipchamp pioneered this: FFmpeg WASM handles containers while WebCodecs codec stubs handle actual encode/decode with hardware acceleration.

### HTMLVideoElement Fallback

For browsers without WebCodecs:
- Use `HTMLVideoElement` for playback with `video.currentTime` seeking
- `MediaRecorder` for basic encoding
- Reduced precision (no frame-accurate seeking)
- Limited concurrent streams

### Newer Libraries Worth Watching

- **Remotion media-parser:** High-level TypeScript abstraction over WebCodecs with built-in queue management
- **MediaBunny:** Consolidates reading/writing/conversion with WebCodecs
- **Diffusion Studio Core:** Full compositing engine built on WebCodecs

---

## 8. Latest Developments (2025-2026)

### Spec Progress
- W3C targets **Candidate Recommendation Q1 2026**, **full Recommendation Q4 2026**
- Editors: Paul Adenot (Mozilla), Eugene Zemtsov (Google)

### New Features Shipped

1. **VideoFrame Rotation & Flip (Chrome 138):** `rotation` and `flip` properties. Solves iPhone 90-degree rotation issue.
2. **Safari 26 Full Support (Late 2025):** AudioEncoder + AudioDecoder added. WebGPU black-screen bug fixed. WebCodecs now fully cross-browser on desktop.

### Active Proposals

- **MSE for WebCodecs (MSE4WC):** Let `HTMLMediaElement` and MSE buffer WebCodecs structures directly (containerless). Bridges WebCodecs frame access with MSE buffering/synchronization.
- **VideoFrame Metadata Registry:** Formal W3C registry for metadata fields on VideoFrame.
- **Codec Registry:** Standardized codec strings to avoid collisions.

### In Progress

- **Firefox Mobile WebCodecs:** Active development
- **Firefox HEVC/H.265:** Under discussion for pass-through decoding
- **WebTransport + WebCodecs:** Designed together for low-latency media streaming
- **Media over QUIC (MoQ):** Uses WebCodecs decoder configs as standard format

### NOT Coming (Yet)

- **Custom codecs** (own WASM codec through WebCodecs): out of scope
- **Built-in demuxing/muxing API:** still requires external libraries
- **Content protection / DRM integration:** open issue, not actively implemented

---

## 9. MasterSelects: Current Implementation Analysis

### Architecture Overview

MasterSelects is a **GPU-first browser video compositor** built with WebGPU. It uses a sophisticated **tiered video decoding strategy** that adapts to project complexity:

```
Tier 1: NativeDecoder (Rust helper for ProRes/DNxHD)
Tier 2: WebCodecs VideoDecoder (full mode: MP4Box + decode)
Tier 3: WebCodecs Simple Mode (HTMLVideoElement + VideoFrame wrapper)
Tier 4: HTMLVideoElement (fallback)
```

### Key Implementation Files

| Functionality | File |
|---|---|
| WebCodecs Playback | `src/engine/WebCodecsPlayer.ts` (66KB) |
| Export Sequential Decode | `src/engine/WebCodecsExportMode.ts` |
| Multi-Clip Parallel Decode | `src/engine/ParallelDecodeManager.ts` |
| Video Encoding | `src/engine/export/VideoEncoderWrapper.ts` |
| Audio Encoding | `src/engine/audio/AudioEncoder.ts` |
| Audio Extraction | `src/engine/audio/AudioExtractor.ts` |
| Audio Export Pipeline | `src/engine/audio/AudioExportPipeline.ts` |
| GPU Texture Import | `src/engine/texture/TextureManager.ts` |
| Frame Collection | `src/engine/render/LayerCollector.ts` |
| Export Orchestration | `src/engine/export/FrameExporter.ts` |
| Codec Helpers | `src/stores/timeline/helpers/codecHelpers.ts` |
| WebCodecs Helpers | `src/stores/timeline/helpers/webCodecsHelpers.ts` |
| FFmpeg Bridge | `src/engine/ffmpeg/FFmpegBridge.ts` |
| Pipeline Monitor | `src/services/wcPipelineMonitor.ts` |

### Three Playback Modes in WebCodecsPlayer

**A) Simple Mode** -- HTMLVideoElement + VideoFrame extraction
- Wraps existing HTMLVideoElement
- Creates VideoFrames from video element on demand
- Uses `requestVideoFrameCallback()` for frame delivery
- Fallback: `canplaythrough` event

**B) Stream Mode** -- MediaStreamTrackProcessor (optimal for playback)
- `HTMLVideoElement.captureStream()` → `MediaStreamTrackProcessor` → VideoFrame
- Zero-copy path

**C) Full Mode** -- MP4Box + VideoDecoder (fastest for export)
- Parses MP4 structure with mp4box library
- Extracts `EncodedVideoChunk` objects
- Hardware-accelerated decode
- Manages frame buffer with CTS (Composition Timestamp) tracking
- Supports keyframe-based seeking and recovery

### Export Modes

| Mode | Decoder | Speed | Codec Support | Use Case |
|---|---|---|---|---|
| Fast (WebCodecs) | VideoDecoder + MP4Box | ~60fps | H.264, H.265, VP9, AV1 | Simple timelines |
| Precise (HTMLVideo) | Browser video element | ~20fps | All browser codecs | Complex comps |
| FFmpeg WASM | FFmpeg (WASM) | ~10fps | ProRes, DNxHR, HAP | Professional codecs |
| FCP XML | None (metadata only) | Instant | XML interchange | Round-trip to NLE |

### GPU Texture Pipeline

```
HTMLVideoElement/VideoFrame → importExternalTexture() → texture_external (zero-copy)
ImageBitmap (NativeDecoder) → copyExternalImageToTexture() → texture_2d (GPU copy)
HTMLCanvasElement (text)     → copyExternalImageToTexture() → texture_2d (GPU copy)
```

### VideoEncoder Configuration

- Supports: H.264 (AVC), H.265 (HEVC), VP9, AV1
- Muxing: `mp4-muxer` (MP4), `webm-muxer` (WebM)
- `fastStart='in-memory'` for playable-before-download MP4s
- Zero-copy encode path via `encodeVideoFrame(frame: VideoFrame)`

### Audio Pipeline

```
AudioExtractor (decode files → AudioBuffer)
  → TimeStretchProcessor (tempo/pitch)
  → AudioEffectRenderer (EQ + volume)
  → AudioMixer (multi-track → stereo)
  → AudioEncoderWrapper (AAC-LC or Opus)
  → Muxer (video + audio)
```

### Parallel Decode for Multi-Clip Export

`ParallelDecodeManager`: One VideoDecoder per unique clip with 60-frame buffer each. Batch decode with smart flush timing.

### Current Limitations

- WebCodecs VideoDecoder runs on **main thread** (not in Workers)
- One VideoDecoder per clip instance (not per unique file)
- Nested compositions can create many decoders (N clips x M nesting levels)
- Linux/Vulkan may have lower HW acceleration (fallback to software)

---

## 10. Recommendations for MasterSelects

Based on the web research and codebase analysis, here are the key findings:

### What MasterSelects Already Does Well

1. **Tiered decoding strategy** -- The 4-tier fallback (Native → Full WebCodecs → Simple → HTMLVideoElement) is industry-best-practice
2. **Zero-copy GPU pipeline** -- Using `texture_external` via `importExternalTexture()` for VideoFrame is the optimal approach
3. **Hybrid FFmpeg + WebCodecs** -- Matches the Clipchamp-pioneered pattern recommended by the industry
4. **Comprehensive codec support** -- H.264, H.265, VP9, AV1 via WebCodecs + ProRes, DNxHR, HAP via FFmpeg WASM
5. **Pipeline monitoring** -- The `wcPipelineMonitor` and `vfPipelineMonitor` ring buffers are excellent for debugging
6. **Frame-accurate export** -- `WebCodecsExportMode` with CTS-indexed frame buffer and binary search is solid

### Potential Improvement Areas

1. **Worker-based decoding** -- Moving VideoDecoder to Web Workers would free the main thread. The existing `SharedDecoderArchitecture.md` design doc already covers this. Would help with UI responsiveness during timeline scrubbing.

2. **Dual-decoder strategy for scrubbing** -- Industry best practice is one decoder for continuous playback, another for scrub/seek. Would integrate well with existing `ParallelDecodeManager`.

3. **Safari 26 full audio pipeline** -- With Safari 26 completing WebCodecs support (AudioEncoder + AudioDecoder), the audio export pipeline could potentially use WebCodecs directly instead of Web Audio API for cross-browser consistency.

4. **VideoFrame rotation metadata** -- Chrome 138 added `rotation` and `flip` to VideoFrame. Could improve handling of iPhone-recorded videos without manual rotation transforms.

5. **MSE4WC integration** -- When the MSE for WebCodecs proposal matures, it could simplify the playback buffering architecture.

---

## Sources

### Official Documentation
- [MDN WebCodecs API](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API)
- [W3C WebCodecs Specification](https://www.w3.org/TR/webcodecs/)
- [Chrome WebCodecs Best Practices](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs)
- [Can I Use: WebCodecs](https://caniuse.com/webcodecs)

### W3C Issues
- [w3c/webcodecs#396 - Frame-stepping / random access seeking](https://github.com/w3c/webcodecs/issues/396)
- [w3c/webcodecs#698 - Flushing without invalidating pipeline](https://github.com/w3c/webcodecs/issues/698)
- [w3c/webcodecs#650 - Recovery Point SEI as keyframe](https://github.com/w3c/webcodecs/issues/650)
- [w3c/webcodecs#873 - Dual-GPU performance](https://github.com/w3c/webcodecs/issues/873)

### Industry Examples & Articles
- [Clipchamp W3C/SMPTE Workshop Talk](https://www.w3.org/2021/03/media-production-workshop/talks/soeren-balko-clipchamp-webcodecs.html)
- [Kapwing web.dev Case Study](https://web.dev/case-studies/kapwing)
- [Remotion: Clearing Up WebCodecs Misconceptions](https://www.remotion.dev/docs/webcodecs/misconceptions)
- [Remotion Media Parser + WebCodecs](https://www.remotion.dev/docs/media-parser/webcodecs)
- [Rendley: Rendering Videos with WebCodecs](https://dev.to/rendley/rendering-videos-in-the-browser-using-webcodecs-api-328n)
- [webrtcHacks: Video Frame Processing](https://webrtchacks.com/video-frame-processing-on-the-web-webassembly-webgpu-webgl-webcodecs-webnn-and-webtransport/)
- [Transloadit: Real-time Video Filters](https://transloadit.com/devtips/real-time-video-filters-in-browsers-with-ffmpeg-and-webcodecs/)
- [Revideo: Faster Rendering](https://re.video/blog/faster-rendering)
- [DojoClip: FFmpeg in Browser Deep Dive](https://dojoclip.com/en/blogs/render-lab/ffmpeg-webassembly-browser-video-editor)
- [Dayverse: FFmpeg.wasm Alternatives](https://dayverse.id/en/articles/best-ffmpeg-wasm-alternatives-client-side)
- [Dayverse: Why FFmpeg.wasm Cannot Use GPU](https://dayverse.id/en/articles/why-ffmpeg-wasm-fails-leverage-gpu-acceleration)

### Browser Updates
- [WebKit Safari 26 Blog](https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/)
- [web.dev: New to Web Platform June 2025](https://web.dev/blog/web-platform-06-2025)
- [Chrome WebGPU 116 Blog](https://developer.chrome.com/blog/new-in-webgpu-116/)

### Proposals
- [MSE for WebCodecs Explainer](https://github.com/wolenetz/mse-for-webcodecs/blob/main/explainer.md)
- [W3C Media Working Group Charter](https://w3c.github.io/charter-media-wg/)
