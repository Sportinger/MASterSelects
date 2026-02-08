<div align="center">

# MasterSelects

### Browser-based Video Compositor

[![Version](https://img.shields.io/badge/version-1.2.2-blue.svg)](https://github.com/Sportinger/MASterSelects/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-Powered-orange.svg)](#)

![MASterSelects Screenshot](docs/images/screenshot-main.png)

</div>

---

## Why I Built This

No Adobe subscription, no patience for cracks, and every free online editor felt like garbage. I needed something that actually works - fast, in the browser, with the power of After Effects, Premiere, and a bit of Ableton mixed in.

**The vision:** A tool where AI can control *everything*. 50+ editing tools accessible via GPT. Plus a live video output for VJ performances (been doing video art for 16 years, so yeah, that matters to me).

**The reality:** ~75 hours of coding in, and I'm mass-producing features faster than I can stabilize them. Things break. A lot. But when it works, it *works*.

Built with Claude as my pair-programmer. I'm not mass-prompting generic code - every feature gets debugged, refactored, and beaten into shape until it does what I need.

---

## What It Does

| Feature | Description |
|---|---|
| [**Multi-track Timeline**](docs/Features/Timeline.md) | Cut, copy, paste, multi-select, JKL shuttle, nested compositions |
| [**30+ GPU Effects**](docs/Features/Effects.md) | Color correction, blur, distort, keying - all real-time |
| [**Video Scopes**](docs/Features/UI-Panels.md#video-scopes-panels) | GPU-accelerated Histogram, Vectorscope, Waveform monitor |
| [**Keyframe Animation**](docs/Features/Keyframes.md) | Bezier curves, copy/paste, tick marks, 5 easing modes |
| [**Vector Masks**](docs/Features/Masks.md) | Pen tool, edge dragging, feathering, multiple masks per clip |
| [**Transitions**](docs/Features/UI-Panels.md#transitions-panel) | Crossfade transitions with GPU-accelerated rendering |
| [**AI Integration**](docs/Features/AI-Integration.md) | 50+ tools controllable via GPT-4/GPT-5 |
| [**4 Export Modes**](docs/Features/Export.md) | WebCodecs Fast/Precise, FFmpeg ProRes/DNxHR, FCP XML |
| [**Live EQ & Audio**](docs/Features/Audio.md) | 10-band parametric EQ with real-time Web Audio preview |
| [**YouTube Download**](docs/Features/YouTube.md) | Search, download, and edit directly |
| [**Text & Solids**](docs/Features/Text-Clips.md) | 50 Google Fonts, stroke, shadow, solid color clips |
| [**Proxy System**](docs/Features/Proxy-System.md) | GPU-accelerated proxies with resume and cache indicator |
| [**Preview & Playback**](docs/Features/Preview.md) | RAM Preview, transform handles, multiple outputs |
| [**Project Storage**](docs/Features/Project-Persistence.md) | Local folders, Raw media auto-copy, autosave, backups |

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

## Known Issues

This is alpha software. Features get added fast, things break.

- YouTube download requires Native Helper with yt-dlp installed
- Audio waveforms may not display for some video formats
- Very long videos (>2 hours) may cause performance issues

If something breaks, refresh. If it's still broken, [open an issue](https://github.com/Sportinger/MASterSelects/issues).

---

## Tech Stack

- **Frontend:** React 19, TypeScript, Zustand, Vite 7.2
- **Rendering:** WebGPU + WGSL shaders (2,000+ lines)
- **Video:** WebCodecs for decode/encode, FFmpeg WASM for ProRes/DNxHR/HAP
- **Audio:** Web Audio API with 10-band live EQ, audio master clock, varispeed
- **AI:** OpenAI GPT-4/GPT-5 function calling, PiAPI video generation
- **Storage:** File System Access API, local project folders with Raw media

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
| `Ctrl+Z` | Undo |
| `Ctrl+S` | Save project |

[All shortcuts →](docs/Features/Keyboard-Shortcuts.md)

---

## Documentation

Detailed docs for each feature: **[docs/Features/](docs/Features/README.md)**

---

## Development

```bash
npm run dev      # Dev server with HMR
npm run build    # Production build
npm run lint     # ESLint
```

<details>
<summary><b>Project Structure</b></summary>

```
src/
├── components/     # React UI (timeline, panels, preview)
├── stores/         # Zustand state management
├── engine/         # WebGPU rendering pipeline
├── effects/        # 30+ GPU effect shaders
├── shaders/        # WGSL shader code
└── services/       # Audio, AI, project persistence
```

</details>

---

<div align="center">

**MIT License** • Built by a video artist who got tired of waiting for Adobe to load

</div>
