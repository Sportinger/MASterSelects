<div align="center">

# MASterSelects

### Browser-based Video Compositor

[![Version](https://img.shields.io/badge/version-1.1.5-blue.svg)](https://github.com/Sportinger/MASterSelects/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-Powered-orange.svg)](#)

![MASterSelects Screenshot](docs/images/screenshot-main.png)

</div>

---

## Why I Built This

No Adobe subscription, no patience for cracks, and every free online editor felt like garbage. I needed something that actually works - fast, in the browser, with the power of After Effects, Premiere, and a bit of Ableton mixed in.

**The vision:** A tool where AI can control *everything*. 50+ editing tools accessible via GPT. Plus a live video output for VJ performances (been doing video art for 16 years, so yeah, that matters to me).

**The reality:** 3 weeks in, ~50 hours of coding, and I'm mass-producing features faster than I can stabilize them. Things break. A lot. But when it works, it *works*.

Built with Claude as my pair-programmer. I'm not mass-prompting generic code - every feature gets debugged, refactored, and beaten into shape until it does what I need.

---

## What It Does

| | |
|---|---|
| **Multi-track Timeline** | Cut, copy, paste, nested compositions |
| **30+ GPU Effects** | Color correction, blur, distort, keying - all real-time |
| **Keyframe Animation** | Bezier curves, 5 easing modes |
| **Vector Masks** | Pen tool, feathering, multiple masks per clip |
| **AI Integration** | 50+ tools controllable via GPT-4/GPT-5 |
| **Live Output** | Separate window for live video performances |
| **YouTube Download** | Search and grab videos directly |
| **Text & Typography** | 50 Google Fonts, stroke, shadow |

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

- Some effects still bleed through layers occasionally
- Export can be flaky with complex compositions
- RAM Preview needs more RAM than it should
- Nested compositions are powerful but buggy

If something breaks, refresh. If it's still broken, [open an issue](https://github.com/Sportinger/MASterSelects/issues).

---

## Tech Stack

- **Frontend:** React 19, TypeScript, Zustand
- **Rendering:** WebGPU + WGSL shaders (the hard part)
- **Video:** WebCodecs for decode/encode, FFmpeg WASM for ProRes
- **Audio:** Web Audio API with 10-band EQ
- **AI:** OpenAI API integration with custom tool handlers

---

## Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `C` | Cut at playhead |
| `I` / `O` | Set in/out points |
| `Ctrl+C/V` | Copy/Paste clips |
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
