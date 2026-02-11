<div align="center">

# MasterSelects

### Browser-based Video Compositor

[![Version](https://img.shields.io/badge/version-1.2.4-blue.svg)](https://github.com/Sportinger/MASterSelects/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![WebGPU](https://img.shields.io/badge/WebGPU-Powered-990000?style=flat-square&logo=webgpu&logoColor=white)](#)
[![React 19](https://img.shields.io/badge/React_19-61DAFB?style=flat-square&logo=react&logoColor=black)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)](#)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white)](#)
[![Rust](https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white)](#native-helper)

<table>
<tr>
<td align="center"><b>31</b><br><sub>GPU Effects</sub></td>
<td align="center"><b>37</b><br><sub>Blend Modes</sub></td>
<td align="center"><b>2,200+</b><br><sub>Lines WGSL</sub></td>
<td align="center"><b>33</b><br><sub>AI Tools</sub></td>
<td align="center"><b>13</b><br><sub>Dependencies</sub></td>
</tr>
</table>

![MASterSelects Screenshot](docs/images/screenshot-main.png)

</div>

---

## What Makes This Different

Most browser-based video editors share a pattern: Canvas 2D compositing, heavyweight dependency trees, and CPU-bound rendering that falls apart at scale. This project takes a fundamentally different approach.

**GPU-first architecture.** Preview, scrubbing, and export all run through the same **WebGPU ping-pong compositor**. Video textures are imported as `texture_external` (**zero-copy**, no CPU roundtrip). **37 blend modes**, 3D rotation, and inline color effects all execute in a **single WGSL composite shader** per layer. No THREE.js, no GSAP, no Canvas 2D fallback in the hot path.

**Zero-copy export pipeline.** Frames are captured as `new VideoFrame(offscreenCanvas)` directly from the GPU canvas. **No `readPixels()`**, no `getImageData()`, no staging buffers in the default path. The GPU renders, **WebCodecs encodes**. That's it.

**3-tier scrubbing cache.** **300 GPU textures in VRAM** for instant scrub (Tier 1), per-video last-frame cache for seek transitions (Tier 2), and a **900-frame RAM Preview** with CPU/GPU promotion (Tier 3). When the cache is warm, **scrubbing doesn't decode at all**.

**13 production dependencies.** React, Zustand, FFmpeg WASM, mp4box, mp4/webm muxers, HuggingFace Transformers, ONNX Runtime, SoundTouch, WebGPU types. **Everything else is custom-built from scratch**: the entire WebGPU compositor, all 31 effect shaders, the keyframe animation system, the export engine, the audio mixer, the text renderer, the mask engine, the video scope renderers, the dock/panel system, the timeline UI. Zero runtime abstraction layers between your timeline and the GPU.

**Nested composition rendering.** Compositions within compositions, each with their own resolution. Rendered to **pooled GPU textures** with frame-level caching, composited in the parent's ping-pong pass, all in a **single `device.queue.submit()`**.

**On-device AI.** SAM2 (Segment Anything Model 2) runs entirely in-browser via ONNX Runtime. Click to select objects in the preview, propagate masks across frames. No server, no API key, no upload. ~220MB model loaded on demand.

---

## Why I Built This

No Adobe subscription, no patience for cracks, and every free online editor felt like garbage. I needed something that actually works - fast, in the browser, with the power of After Effects, Premiere, and a bit of Ableton mixed in.

**The vision:** A tool where AI can control *everything*. 33 editing tools accessible via GPT function calling, plus a multi-output system for live performances (been doing video art for 16 years, so yeah, that matters to me).

Built with Claude as my pair-programmer. Every feature gets debugged, refactored, and beaten into shape until it does what I need. ~100k lines of TypeScript, ~2,200 lines of WGSL, and a Rust native helper for the stuff browsers can't do.

---

## What It Does

| Feature | Description |
|---|---|
| [**Multi-track Timeline**](docs/Features/Timeline.md) | Cut, copy, paste, multi-select, JKL shuttle, nested compositions |
| [**31 GPU Effects**](docs/Features/Effects.md) | Color correction, blur, distort, stylize, keying - all real-time |
| [**Video Scopes**](docs/Features/UI-Panels.md#video-scopes-panels) | GPU-accelerated Histogram, Vectorscope, Waveform monitor |
| [**Keyframe Animation**](docs/Features/Keyframes.md) | Bezier curves, copy/paste, tick marks, 5 easing modes |
| [**Vector Masks**](docs/Features/Masks.md) | Pen tool, edge dragging, feathering, multiple masks per clip |
| [**SAM2 Segmentation**](docs/Features/AI-Integration.md) | AI object selection in preview - click to mask, propagate across frames |
| [**Transitions**](docs/Features/UI-Panels.md#transitions-panel) | Crossfade transitions with GPU-accelerated rendering |
| [**AI Integration**](docs/Features/AI-Integration.md) | 33 tools controllable via GPT-4/GPT-5 function calling |
| [**4 Export Modes**](docs/Features/Export.md) | WebCodecs Fast/Precise, FFmpeg ProRes/DNxHR, FCP XML |
| [**Live EQ & Audio**](docs/Features/Audio.md) | 10-band parametric EQ with real-time Web Audio preview |
| [**Download Panel**](docs/Features/YouTube.md) | YouTube, TikTok, Instagram, Twitter/X via Native Helper |
| [**Text & Solids**](docs/Features/Text-Clips.md) | 50 Google Fonts, stroke, shadow, solid color clips |
| [**Proxy System**](docs/Features/Proxy-System.md) | GPU-accelerated proxies with resume and cache indicator |
| [**Output Manager**](docs/Features/Preview.md) | Multi-window outputs, source routing, corner pin warping, slice masks |
| [**Slot Grid**](docs/Features/UI-Panels.md) | Resolume-style 4x12 grid with multi-layer live playback |
| [**Preview & Playback**](docs/Features/Preview.md) | RAM Preview, transform handles, multiple render targets |
| [**Project Storage**](docs/Features/Project-Persistence.md) | Local folders, raw media auto-copy, autosave, backups |
| [**Interactive Tutorial**](docs/Features/UI-Panels.md) | Guided onboarding with animated Clippy mascot |

<details>
<summary><b>See Keyframe Editor</b></summary>
<br>
<img src="docs/images/screenshot-curves.png" alt="Bezier Curve Editor" width="400">
</details>

---

## Quick Start

```bash
npm install
npm run dev     # http://localhost:5173
```

**Requirements:** Chrome 113+ with WebGPU support. Dedicated GPU recommended.

> **Linux:** Enable Vulkan for smooth 60fps: `chrome://flags/#enable-vulkan`

---

## Native Helper

Optional cross-platform Rust binary for features browsers can't do natively.

```bash
cd tools/native-helper
cargo run --release    # WebSocket :9876, HTTP :9877
```

| Capability | Details |
|---|---|
| **Decode** | H.264, ProRes, DNxHD + LRU frame cache |
| **Encode** | ProRes, DNxHR, H.264, H.265, VP9, FFV1, UTVideo, MJPEG |
| **Download** | yt-dlp integration (YouTube, TikTok, Instagram, Twitter/X, 100+ sites) |

**Platforms:** Windows, Linux, macOS. Requires Rust + FFmpeg. See [Native Helper docs](tools/native-helper/README.md) for platform-specific setup.

---

## Known Issues

This is alpha software. Features get added fast, things break.

- Video downloads require Native Helper with yt-dlp installed
- Audio waveforms may not display for some video formats
- Very long videos (>2 hours) may cause performance issues

If something breaks, refresh. If it's still broken, [open an issue](https://github.com/Sportinger/MASterSelects/issues).

---

## Tech Stack

- **Frontend:** React 19, TypeScript, Zustand, Vite 7.2
- **Rendering:** WebGPU + 2,200 lines of WGSL shaders
- **Video:** WebCodecs for decode/encode, FFmpeg WASM for ProRes/DNxHR/HAP
- **Audio:** Web Audio API with 10-band live EQ, audio master clock, varispeed
- **AI:** OpenAI GPT-4/GPT-5 function calling, SAM2 via ONNX Runtime, PiAPI video generation
- **Native:** Rust binary for FFmpeg decode/encode + yt-dlp downloads
- **Storage:** File System Access API, local project folders with raw media

---

## Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `J` / `K` / `L` | Reverse / Pause / Forward (shuttle) |
| `C` | Cut at playhead |
| `I` / `O` | Set in/out points |
| `Ctrl+C/V` | Copy/Paste clips or keyframes |
| `Shift+Click` | Multi-select clips |
| `Tab` | Toggle edit mode |
| `Ctrl+Z/Y` | Undo/Redo |
| `Ctrl+S` | Save project |

[All 77 shortcuts](docs/Features/Keyboard-Shortcuts.md)

---

## Documentation

Detailed docs for each feature: **[docs/Features/](docs/Features/README.md)**

---

## Development

```bash
npm run dev              # Dev server with HMR
npm run dev:changelog    # Dev server with changelog dialog
npm run build            # Production build
npm run lint             # ESLint
```

<details>
<summary><b>Project Structure</b></summary>

```
src/
├── components/          # React UI
│   ├── timeline/        # Timeline editor (hooks/, components/)
│   ├── panels/          # Properties, Media, AI, Download, Export, Scopes
│   ├── preview/         # Canvas + overlays + transform handles
│   ├── outputManager/   # Multi-window output with slices
│   ├── dock/            # Panel/tab system
│   └── common/          # Dialogs, tutorial, shared components
├── stores/              # Zustand state management
│   ├── timeline/        # Slices: track, clip, keyframe, mask, playback
│   └── mediaStore/      # Slices: import, folder, proxy, composition
├── engine/              # WebGPU rendering pipeline
│   ├── core/            # WebGPUContext, RenderTargetManager
│   ├── render/          # Compositor, RenderLoop, LayerCollector
│   ├── export/          # FrameExporter, VideoEncoder, AudioEncoder
│   ├── audio/           # AudioMixer, TimeStretch
│   └── ffmpeg/          # FFmpegBridge
├── effects/             # 31 GPU effects (color/, blur/, distort/, stylize/, keying/)
├── services/            # Audio, AI, Project, NativeHelper, Logger
├── shaders/             # WGSL (composite, effects, output, scopes)
└── workers/             # SAM2 inference, clip analysis
```

```
tools/
└── native-helper/       # Rust binary (FFmpeg + yt-dlp bridge)
    └── src/             # WebSocket server, decode/encode sessions
```

</details>

---

<div align="center">

**MIT License** · Built by a video artist who got tired of waiting for Adobe to load

</div>
