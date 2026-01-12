# MASterSelects

![Version](https://img.shields.io/badge/version-1.0.6-blue.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)
![Platform](https://img.shields.io/badge/platform-Browser-lightgrey.svg)
![WebGPU](https://img.shields.io/badge/WebGPU-Powered-green.svg)
![React](https://img.shields.io/badge/React-19.2-61DAFB.svg?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6.svg?logo=typescript)
![Vite](https://img.shields.io/badge/Vite-7.2-646CFF.svg?logo=vite)

**Professional WebGPU Video Compositor & Timeline Editor**

A browser-based video editing application with After Effects-style compositing, AI-powered workflows, and real-time GPU rendering. No plugins or installations required.

## Features

| Feature | Description | Docs |
|---------|-------------|------|
| **Multi-track Timeline** | Video and audio tracks with nested compositions | [Timeline](docs/Features/Timeline.md) |
| **Keyframe Animation** | Bezier curve editor with 5 easing modes | [Keyframes](docs/Features/Keyframes.md) |
| **37 Blend Modes** | All After Effects blend modes | [Effects](docs/Features/Effects.md) |
| **30+ GPU Effects** | Color, blur, distort, stylize, keying effects | [Effects](docs/Features/Effects.md) |
| **Vector Masks** | Rectangle, ellipse, pen tool with GPU feathering | [Masks](docs/Features/Masks.md) |
| **AI Integration** | 50+ editing tools via GPT-4 | [AI Integration](docs/Features/AI-Integration.md) |
| **10-Band EQ** | Parametric equalizer with keyframe support | [Audio](docs/Features/Audio.md) |
| **Multicam Sync** | Audio-based cross-correlation synchronization | [Audio](docs/Features/Audio.md) |
| **4 Transcription Providers** | Local Whisper, OpenAI, AssemblyAI, Deepgram | [AI Integration](docs/Features/AI-Integration.md) |
| **RAM Preview** | Cached playback at 30fps with idle mode | [Preview](docs/Features/Preview.md) |
| **Video Export** | H.264/VP9 WebCodecs + FFmpeg WASM (ProRes, DNxHR) | [Export](docs/Features/Export.md) |
| **Local Project Storage** | Project folder with autosave and backups | [Project Persistence](docs/Features/Project-Persistence.md) |
| **Smart Media Relink** | Auto-reconnect and relink missing media files | [Media Panel](docs/Features/Media-Panel.md) |

**[Full Documentation](docs/Features/README.md)** | **[Keyboard Shortcuts](docs/Features/Keyboard-Shortcuts.md)**

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| React 19 + TypeScript | UI framework |
| Vite 7.2 | Build tooling & HMR |
| Zustand | State management |
| WebGPU + WGSL | GPU rendering (2,000+ lines of shaders) |
| WebCodecs | Hardware video decoding & encoding |
| FFmpeg WASM | Professional codecs (ProRes, DNxHR, HAP) |
| Web Audio API | 10-band EQ, audio sync |
| File System Access API | Local project storage |
| OpenAI API | AI editing tools |

---

## Quick Start

```bash
npm install
npm run dev     # http://localhost:5173
```

### Requirements

- **Browser**: Chrome 113+ or Edge 113+ (WebGPU required)
- **GPU**: Dedicated GPU recommended
- **RAM**: 8GB minimum, 16GB recommended

### Linux Users

Enable Vulkan for 60fps performance:
```
chrome://flags/#enable-vulkan → Enabled
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI Layer (React)                         │
│  Timeline │ Preview │ Media │ Properties │ AI Chat │ Export     │
├─────────────────────────────────────────────────────────────────┤
│                      State Layer (Zustand)                       │
│  timelineStore │ mediaStore │ multicamStore │ settingsStore     │
├─────────────────────────────────────────────────────────────────┤
│                     Engine Layer (WebGPU)                        │
│  Compositor │ Effects │ Masks │ Textures │ Scrubbing Cache      │
├─────────────────────────────────────────────────────────────────┤
│                      Services Layer                              │
│  Audio │ AI Tools │ Whisper │ Project DB │ Proxy Generator      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Documentation

### Core Features
- [Timeline](docs/Features/Timeline.md) - Multi-track editing, clips, snapping, compositions
- [Keyframes](docs/Features/Keyframes.md) - Animation system with curve editor
- [Effects](docs/Features/Effects.md) - GPU effects and 37 blend modes
- [Masks](docs/Features/Masks.md) - Shape masks with GPU feathering

### Media & Audio
- [Media Panel](docs/Features/Media-Panel.md) - Import, folders, proxy generation
- [Audio](docs/Features/Audio.md) - 10-band EQ, waveforms, multicam sync
- [Preview](docs/Features/Preview.md) - RAM Preview, scrubbing, multiple outputs

### AI & Export
- [AI Integration](docs/Features/AI-Integration.md) - 50+ tools, transcription
- [Export](docs/Features/Export.md) - H.264/VP9 encoding

### System
- [UI & Panels](docs/Features/UI-Panels.md) - Dockable panels, layouts
- [GPU Engine](docs/Features/GPU-Engine.md) - WebGPU rendering details
- [Project Persistence](docs/Features/Project-Persistence.md) - Auto-save, IndexedDB
- [Keyboard Shortcuts](docs/Features/Keyboard-Shortcuts.md) - Complete reference

---

## Panel System

8 dockable panels with drag-and-drop arrangement:

| Panel | Purpose |
|-------|---------|
| **Preview** | Composition output canvas |
| **Timeline** | Multi-track editor |
| **Media** | Media browser and folders |
| **Properties** | Transform, Effects, Masks, Volume (unified) |
| **Export** | Render settings and progress |
| **Multicam** | Camera sync and EDL |
| **AI Chat** | GPT-powered editing assistant |
| **Slots** | Layer slot management |

---

## Key Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `C` | Split clip at playhead |
| `Delete` | Delete selected |
| `I` / `O` | Set In/Out points |
| `Ctrl+Z` | Undo |
| `Ctrl+S` | Save project |

[Full shortcut reference](docs/Features/Keyboard-Shortcuts.md)

---

## Development

```bash
npm run dev      # Development server
npm run build    # Production build
npm run lint     # ESLint check
npm run preview  # Preview production build
```

### Project Structure

```
src/
├── components/     # React components
│   ├── timeline/   # Timeline editor
│   ├── panels/     # Dock panels (Properties, Media, etc.)
│   ├── preview/    # Preview canvas
│   └── dock/       # Panel system
├── stores/         # Zustand state management
├── engine/         # WebGPU rendering
├── shaders/        # WGSL shaders
├── services/       # Audio, AI, persistence
└── hooks/          # React hooks
```

See [CLAUDE.md](CLAUDE.md) for detailed structure and patterns.

---

## License

Proprietary
