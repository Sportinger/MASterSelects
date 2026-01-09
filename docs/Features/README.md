# MASterSelects Feature Manual

A comprehensive guide to all features in MASterSelects - a professional WebGPU-powered video compositor and timeline editor.

## Quick Navigation

| Category | Description |
|----------|-------------|
| [Timeline](./Timeline.md) | Multi-track editing, clips, compositions |
| [Keyframes](./Keyframes.md) | Animation system, interpolation, easing |
| [Preview & Playback](./Preview.md) | RAM Preview, scrubbing, multiple previews |
| [Effects](./Effects.md) | GPU effects, blend modes, transforms |
| [Masks](./Masks.md) | Shape masks, feathering, editing |
| [AI Integration](./AI-Integration.md) | AI Editor, transcription, analysis |
| [Media Panel](./Media-Panel.md) | Import, folders, compositions |
| [Audio](./Audio.md) | Waveforms, multicam sync, audio tracks |
| [Export](./Export.md) | Rendering, formats, frame export |
| [UI & Panels](./UI-Panels.md) | Dock system, layouts, menus |
| [GPU Engine](./GPU-Engine.md) | WebGPU rendering, performance |
| [Project Persistence](./Project-Persistence.md) | Auto-save, IndexedDB storage |
| [Proxy System](./Proxy-System.md) | GPU proxy generation for large files |
| [Keyboard Shortcuts](./Keyboard-Shortcuts.md) | Complete shortcut reference |

---

## Feature Overview

### Core Capabilities

- **WebGPU Rendering** - Hardware-accelerated compositing with 60fps performance
- **Multi-track Timeline** - Video and audio tracks with nested compositions
- **Keyframe Animation** - Full property animation with multiple easing modes
- **AI-Powered Editing** - GPT integration for intelligent timeline manipulation
- **Professional Effects** - 37 blend modes, GPU effects, masks
- **RAM Preview** - After Effects-style cached playback

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        UI Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │ Timeline │ │ Preview  │ │ Media    │ │ Effects/Props    ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      State Layer (Zustand)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │ Timeline │ │ Mixer    │ │ Media    │ │ History          ││
│  │ Store    │ │ Store    │ │ Store    │ │ Store            ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                     Engine Layer (WebGPU)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │Compositor│ │ Effects  │ │ Texture  │ │ Frame            ││
│  │ Pipeline │ │ Pipeline │ │ Manager  │ │ Exporter         ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Version History

This application has evolved through **362 commits** from initial release to current state. Key milestones:

| Phase | Features Added |
|-------|----------------|
| Foundation | WebGPU engine, basic timeline, video playback |
| Timeline v2 | Multi-track, compositions, clip management |
| Animation | Keyframe system, property animation |
| Effects | Blend modes, GPU effects, masks |
| AI | GPT integration, smart editing tools |
| Polish | RAM Preview, multicam sync, export |

---

## Getting Started

See individual feature pages for detailed usage:

1. **[Timeline](./Timeline.md)** - Start here for basic editing
2. **[Media Panel](./Media-Panel.md)** - Import and organize media
3. **[Keyframes](./Keyframes.md)** - Add animation
4. **[Effects](./Effects.md)** - Apply visual effects
5. **[Export](./Export.md)** - Render your project

---

*Last updated: Based on commit d63e381*
