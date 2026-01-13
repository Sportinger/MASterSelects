# MASterSelects Documentation

**Professional WebGPU Video Compositor & Timeline Editor**

Version 1.0.6 | January 2026

---

## Overview

MASterSelects is a browser-based professional video editing application built on cutting-edge WebGPU technology. It delivers After Effects-style compositing, multi-track timeline editing, AI-powered workflows, and real-time GPU rendering—all running entirely in the browser with no plugins or installations required.

### Key Highlights

| Capability | Description |
|------------|-------------|
| **WebGPU Rendering** | Hardware-accelerated compositing with zero-copy video textures at 60fps |
| **Multi-track Timeline** | Professional NLE with video/audio tracks, nested compositions, and multicam |
| **Keyframe Animation** | Full property animation with bezier curve editor and 5 easing modes |
| **AI Integration** | 50+ intelligent editing tools via OpenAI function calling |
| **30+ GPU Effects** | Color, blur, distort, stylize, keying effects with quality controls |
| **Professional Audio** | 10-band parametric EQ with keyframe automation |
| **Multicam Support** | Audio-based cross-correlation synchronization |
| **Video Export** | H.264/VP9 WebCodecs + FFmpeg WASM (ProRes, DNxHR, HAP) |
| **Native Helper** | Optional 10x faster ProRes/DNxHD decode with hardware accel |
| **Local Storage** | Project folder with autosave, backups, and smart media relinking |

---

## Technology Stack

```
Frontend          React 19 + TypeScript + Vite 7.2
State Management  Zustand with slice architecture
GPU Rendering     WebGPU + WGSL shaders (2,000+ lines)
Video Decoding    WebCodecs API with hardware acceleration
Video Encoding    WebCodecs + FFmpeg WASM (ProRes, DNxHR, HAP)
Audio Processing  Web Audio API with AnalyserNode
AI Services       OpenAI GPT-4 function calling
Persistence       File System Access API + local project folders
UI Framework      Custom dockable panel system
```

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Timeline](./Timeline.md) | Multi-track editing, clips, snapping, compositions, multicam |
| [Keyframes](./Keyframes.md) | Animation system, curve editor, bezier interpolation |
| [Preview & Playback](./Preview.md) | RAM Preview, scrubbing, multiple outputs, edit mode |
| [Effects](./Effects.md) | GPU effects, 37 blend modes, transforms |
| [Masks](./Masks.md) | Shape masks, pen tool, GPU feathering |
| [AI Integration](./AI-Integration.md) | 50+ AI tools, transcription, intelligent editing |
| [Media Panel](./Media-Panel.md) | Import, folder organization, compositions |
| [Audio](./Audio.md) | 10-band EQ, waveforms, multicam sync |
| [Text Clips](./Text-Clips.md) | Typography, 50 Google Fonts, stroke, shadow |
| [Export](./Export.md) | H.264/VP9 encoding, frame-by-frame rendering |
| [UI & Panels](./UI-Panels.md) | Dockable panels, layouts, menus, MIDI control |
| [GPU Engine](./GPU-Engine.md) | WebGPU architecture, optical flow, texture management |
| [Project Persistence](./Project-Persistence.md) | Auto-save, IndexedDB, file handles |
| [Proxy System](./Proxy-System.md) | GPU-accelerated proxy generation |
| [Native Helper](./Native-Helper.md) | Optional Turbo Mode for ProRes/DNxHD |
| [Keyboard Shortcuts](./Keyboard-Shortcuts.md) | Complete shortcut reference |

---

## Feature Catalog

### Timeline & Editing

| Feature | Status | Details |
|---------|--------|---------|
| Multi-track Timeline | ✅ | Unlimited video and audio tracks |
| Clip Operations | ✅ | Move, trim, split, delete, reverse |
| Magnetic Snapping | ✅ | 0.1s snap distance with edge alignment |
| Snap Toggle | ✅ | Toolbar button to enable/disable snapping |
| Overlap Resistance | ✅ | 100px resistance with auto-trim |
| Marquee Selection | ✅ | Rectangle selection with Shift modifier |
| Linked Audio | ✅ | Video-audio linking with Alt+drag override |
| Nested Compositions | ✅ | Double-click to edit, recursive rendering |
| Composition Tabs | ✅ | Multiple open compositions with tab navigation |
| Track Controls | ✅ | Visibility, mute, solo, expand |
| Playback Looping | ✅ | In/Out points with loop mode |

### Text Clips

| Feature | Status | Details |
|---------|--------|---------|
| Text Overlays | ✅ | Add text clips to timeline with "+ Text" button |
| Google Fonts | ✅ | 50 popular fonts dynamically loaded |
| Typography | ✅ | Font size, weight (100-900), style (normal/italic) |
| Alignment | ✅ | Horizontal (L/C/R) and vertical (T/M/B) alignment |
| Spacing | ✅ | Line height and letter spacing controls |
| Stroke (Outline) | ✅ | Configurable color and width |
| Drop Shadow | ✅ | Color, offset X/Y, blur radius |
| GPU Rendering | ✅ | Canvas2D → GPU texture for full compositing |
| Animations | ✅ | All transforms and effects work with text |
| Serialization | ✅ | Text clips saved/restored with projects |

### Keyframe Animation

| Feature | Status | Details |
|---------|--------|---------|
| Transform Animation | ✅ | Position (X,Y,Z), Scale (X,Y), Rotation (X,Y,Z), Opacity |
| Effect Animation | ✅ | All numeric effect parameters keyframeable |
| Curve Editor | ✅ | SVG-based with bezier handle manipulation |
| Linear Easing | ✅ | Constant rate interpolation |
| Ease In | ✅ | Slow start acceleration |
| Ease Out | ✅ | Slow end deceleration |
| Ease In-Out | ✅ | Smooth acceleration and deceleration |
| Custom Bezier | ✅ | User-defined bezier curves with handles |
| Recording Mode | ✅ | Auto-keyframe on value change |
| Keyframe Selection | ✅ | Multi-select with Delete support |

### GPU Effects & Compositing

| Category | Effects | Count |
|----------|---------|-------|
| **Color** | Brightness, Contrast, Saturation, Vibrance, Hue Shift, Temperature, Exposure, Levels, Invert | 9 |
| **Blur** | Box Blur, Gaussian Blur, Motion Blur, Radial Blur, Zoom Blur | 5 |
| **Distort** | Pixelate, Kaleidoscope, Mirror, RGB Split, Twirl, Wave, Bulge | 7 |
| **Stylize** | Vignette, Grain, Glow, Posterize, Edge Detect, Scanlines, Threshold, Sharpen | 8 |
| **Keying** | Chroma Key | 1 |

**Effect Controls:**
- ✅ Bypass toggle for A/B comparison
- ✅ Draggable values with precision modifiers (Shift/Ctrl)
- ✅ Quality parameters for blur/glow effects
- ✅ Auto performance protection (resets if too slow)

### Blend Modes (37 Total)

| Category | Modes |
|----------|-------|
| **Normal** | Normal, Dissolve |
| **Darken** | Darken, Multiply, Color Burn, Linear Burn, Darker Color |
| **Lighten** | Lighten, Screen, Color Dodge, Linear Dodge (Add), Lighter Color |
| **Contrast** | Overlay, Soft Light, Hard Light, Vivid Light, Linear Light, Pin Light, Hard Mix |
| **Inversion** | Difference, Exclusion, Subtract, Divide |
| **Component** | Hue, Saturation, Color, Luminosity |
| **Stencil** | Stencil Alpha, Stencil Luma, Silhouette Alpha, Silhouette Luma |

### Masks

| Feature | Status | Details |
|---------|--------|---------|
| Rectangle Tool | ✅ | Click-drag creation |
| Ellipse Tool | ✅ | Click-drag creation |
| Pen Tool | ✅ | Bezier path drawing |
| Add Mode | ✅ | Combine mask regions |
| Subtract Mode | ✅ | Remove from mask |
| Intersect Mode | ✅ | Overlap only |
| GPU Feathering | ✅ | 3-tier blur (17/33/61 taps) |
| Mask Expansion | ✅ | Grow/shrink mask boundary |
| Vertex Editing | ✅ | Select, move, delete points |
| Mask Inversion | ✅ | Invert mask selection |

### AI Integration

| Feature | Status | Details |
|---------|--------|---------|
| GPT-4 Chat | ✅ | Natural language editing commands |
| 50+ AI Tools | ✅ | Clip, track, keyframe, effect operations |
| Local Whisper | ✅ | Browser-based transcription |
| OpenAI Whisper API | ✅ | Cloud transcription service |
| AssemblyAI | ✅ | Professional transcription |
| Deepgram | ✅ | Fast transcription service |
| Multicam EDL | ✅ | AI-generated edit decision lists |
| Context Awareness | ✅ | AI knows timeline state |

### YouTube Integration

| Feature | Status | Details |
|---------|--------|---------|
| YouTube Search | ✅ | Search videos via Invidious or YouTube Data API |
| Video Thumbnails | ✅ | Display thumbnails, titles, channels, duration |
| Drag to Timeline | ✅ | Drag videos to timeline to download |
| Cobalt Download | ✅ | Download videos via Cobalt API |
| Pending Clip State | ✅ | Dashed preview with download progress |
| Dual API Support | ✅ | Invidious (no key) or YouTube Data API (optional) |

### Audio

| Feature | Status | Details |
|---------|--------|---------|
| 10-Band Parametric EQ | ✅ | 31Hz to 16kHz frequency bands |
| EQ Gain Range | ✅ | -12dB to +12dB per band |
| EQ Keyframes | ✅ | Animate EQ parameters over time |
| Waveform Display | ✅ | 50 samples/second resolution |
| Volume Control | ✅ | Per-clip volume adjustment |
| Multicam Audio Sync | ✅ | Cross-correlation algorithm |
| Audio Track Mute | ✅ | Per-track mute control |
| Audio Solo | ✅ | Isolate audio tracks |

### Preview & Playback

| Feature | Status | Details |
|---------|--------|---------|
| Real-time Preview | ✅ | 60fps GPU rendering |
| Idle Mode | ✅ | Auto-pause GPU when nothing changes |
| Preview Quality | ✅ | Full/Half/Quarter resolution for performance |
| RAM Preview | ✅ | 30fps cached playback, 900 frame limit |
| Multiple Outputs | ✅ | Open multiple preview windows |
| Edit Mode | ✅ | Direct manipulation in preview |
| Scrubbing Cache | ✅ | 3-tier caching system |
| Statistics Overlay | ✅ | FPS, timing, idle status indicators |
| Resolution Presets | ✅ | 480p to 4K preview |
| Pause on Drag | ✅ | Playback pauses when dragging playhead |

### Export

| Feature | Status | Details |
|---------|--------|---------|
| H.264 Export | ✅ | MP4 container |
| VP9 Export | ✅ | WebM container |
| Resolution Presets | ✅ | 480p, 720p, 1080p, 4K |
| Frame Rate Options | ✅ | 24, 25, 30, 60 fps |
| Quality Presets | ✅ | 5-35 Mbps bitrate |
| In/Out Export | ✅ | Export marked region |
| Single Frame Export | ✅ | PNG frame capture |
| Progress Tracking | ✅ | Frame count and ETA |

### Project & Media Management

| Feature | Status | Details |
|---------|--------|---------|
| Local Project Folder | ✅ | All data stored in user-selected folder |
| Auto-Save | ✅ | Configurable interval (1-10 min) |
| Backup System | ✅ | Keeps last 20 backups automatically |
| Welcome Overlay | ✅ | Project folder selection on launch |
| Save As | ✅ | Export project to new location |
| Smart Media Relink | ✅ | Auto-find moved/renamed files |
| Reload All | ✅ | Restore file permissions after restart |
| Media Import | ✅ | MP4, WebM, MOV, WAV, MP3, PNG, JPG |
| Proxy Generation | ✅ | GPU-accelerated, Windows/Linux/Mac |
| Hash Deduplication | ✅ | Same files share proxies/thumbnails |

### Native Helper (Turbo Mode)

| Feature | Status | Details |
|---------|--------|---------|
| ProRes Decoding | ✅ | All profiles at native speed |
| DNxHD/DNxHR Decoding | ✅ | All profiles at native speed |
| Hardware Acceleration | ✅ | VAAPI (Intel/AMD), NVDEC (NVIDIA) |
| Frame Cache | ✅ | LRU cache up to 2GB |
| Background Prefetch | ✅ | Frames loaded ahead of playhead |
| Native Encoding | ✅ | 10x faster ProRes/DNxHD export |
| Auto-Detection | ✅ | Toolbar shows "⚡ Turbo" when connected |
| Download Link | ✅ | Click indicator for helper download |

### User Interface

| Feature | Status | Details |
|---------|--------|---------|
| Dockable Panels | ✅ | Drag, resize, tab grouping |
| 9 Panel Types | ✅ | Preview, Timeline, Media, Properties, Export, Multicam, AI Chat, AI Video, YouTube |
| Unified Properties Panel | ✅ | Transform, Effects, Masks, Volume tabs |
| Menu Bar | ✅ | File, Edit, View, Output, Window |
| Context Menus | ✅ | Right-click operations |
| MIDI Control | ✅ | Web MIDI API integration |
| Keyboard Shortcuts | ✅ | Comprehensive hotkey support |
| Hold-to-Drag Tabs | ✅ | 500ms hold to reorder |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              UI Layer                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │  Timeline   │ │   Preview   │ │   Media     │ │  Effects/Props    │  │
│  │   (React)   │ │  (Canvas)   │ │   Panel     │ │  AI Chat Panel    │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                         State Layer (Zustand)                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │  Timeline   │ │   Mixer     │ │   Media     │ │    Multicam       │  │
│  │   Store     │ │   Store     │ │   Store     │ │     Store         │  │
│  │  (7 slices) │ │             │ │             │ │                   │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                        Engine Layer (WebGPU)                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │ Compositor  │ │  Effects    │ │  Texture    │ │     Frame         │  │
│  │  Pipeline   │ │  Pipeline   │ │  Manager    │ │    Exporter       │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────────┘  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                        │
│  │   Mask      │ │  Scrubbing  │ │  Optical    │                        │
│  │  Manager    │ │   Cache     │ │    Flow     │                        │
│  └─────────────┘ └─────────────┘ └─────────────┘                        │
├─────────────────────────────────────────────────────────────────────────┤
│                          Services Layer                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │   Audio     │ │  Whisper    │ │  Project    │ │    AI Tools       │  │
│  │  Manager    │ │  Service    │ │     DB      │ │   (OpenAI)        │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────────┘  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                        │
│  │   Proxy     │ │ FileSystem  │ │   Audio     │                        │
│  │ Generator   │ │  Service    │ │    Sync     │                        │
│  └─────────────┘ └─────────────┘ └─────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### WGSL Shader Breakdown

| File | Lines | Purpose |
|------|-------|---------|
| `composite.wgsl` | 743 | Layer compositing, 37 blend modes |
| `opticalflow.wgsl` | 326 | Motion analysis, scene detection |
| `effects.wgsl` | 243 | GPU effect implementations |
| `output.wgsl` | 40 | Final output passthrough |
| **Total** | **1,352** | |

### Zustand Store Architecture

```
timelineStore/
├── trackSlice.ts      # Track CRUD operations
├── clipSlice.ts       # Clip operations, transforms
├── playbackSlice.ts   # Play/pause, seeking, time
├── keyframeSlice.ts   # Keyframe CRUD, interpolation
├── selectionSlice.ts  # Clip/keyframe selection
├── maskSlice.ts       # Mask shapes and vertices
└── compositionSlice.ts # Composition management
```

---

## Browser Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Browser** | Chrome 113+ / Edge 113+ | Chrome 120+ / Edge 120+ |
| **WebGPU** | Required | Required |
| **WebCodecs** | Required for export | Required |
| **File System Access** | Optional | Recommended |
| **Web MIDI** | Optional | For MIDI control |
| **GPU** | Integrated | Dedicated GPU |
| **RAM** | 8GB | 16GB+ |

### Enabling WebGPU on Linux

```
chrome://flags/#enable-vulkan → Enabled
chrome://flags/#enable-unsafe-webgpu → Enabled (if needed)
```

---

## Quick Start Guide

### 1. Import Media

Open the **Media Panel** and use one of these methods:
- Click **Add** → **Import Media** and select files
- Drag and drop files directly into the panel

Supported formats: MP4, WebM, MOV, WAV, MP3, AAC, PNG, JPG, GIF, WebP

### 2. Create a Composition

- Click **Add** → **Composition**
- Configure resolution (up to 7680×4320) and frame rate
- The composition opens in the Timeline

### 3. Add Clips to Timeline

- Drag media from the Media Panel to a timeline track
- Video/images go on video tracks, audio on audio tracks
- Clips show duration preview while dragging

### 4. Edit and Animate

- **Move clips**: Drag horizontally on the timeline
- **Trim clips**: Drag the left/right edges
- **Split clips**: Press `C` at the playhead
- **Animate**: Expand track, add keyframes to properties

### 5. Apply Effects

- Select a clip
- Open the **Effects Panel**
- Choose a blend mode or add GPU effects
- Adjust parameters with sliders

### 6. Preview and Export

- Press `Space` to play/pause
- Enable **RAM Preview** for smooth cached playback
- Open **Export Panel** when ready
- Select codec, resolution, and quality
- Click **Export** to render

---

## Keyboard Reference

| Category | Key | Action |
|----------|-----|--------|
| **Playback** | `Space` | Play/Pause |
| | `L` | Toggle loop |
| | `←` / `→` | Step frame |
| | `Home` | Go to start |
| **Editing** | `C` | Split at playhead |
| | `Delete` | Delete selected |
| | `Ctrl+Z` | Undo |
| | `Ctrl+Shift+Z` | Redo |
| **Timeline** | `I` | Set In point |
| | `O` | Set Out point |
| | `X` | Clear In/Out |
| | `-` / `=` | Zoom out/in |
| **Project** | `Ctrl+S` | Save project |
| | `Ctrl+N` | New project |

See [Keyboard Shortcuts](./Keyboard-Shortcuts.md) for complete reference.

---

## Performance Optimization

### For Best Performance

1. **Enable Vulkan on Linux**: `chrome://flags/#enable-vulkan`
2. **Use dedicated GPU**: Check `chrome://gpu` for WebGPU status
3. **Generate proxies**: Right-click large videos → Generate Proxy
4. **Enable RAM Preview**: Caches 900 frames at 30fps
5. **Close unused panels**: Reduces React re-renders

### Troubleshooting

| Issue | Solution |
|-------|----------|
| 15fps on Linux | Enable Vulkan in chrome://flags |
| Black preview | Check video readyState, refresh page |
| Export fails | Verify WebCodecs support in browser |
| Slow scrubbing | Generate proxies for large files |
| Memory issues | Clear RAM Preview cache, reduce preview resolution |

---

## Source Code Reference

| Area | Location |
|------|----------|
| Timeline Components | `src/components/timeline/` |
| Panel Components | `src/components/panels/` |
| Preview System | `src/components/preview/` |
| GPU Engine | `src/engine/` |
| WGSL Shaders | `src/shaders/` |
| State Management | `src/stores/` |
| Services | `src/services/` |
| React Hooks | `src/hooks/` |

---

## Not Yet Implemented

The following features are planned but not currently available:

- Cloud storage integration
- Asset library across projects
- Batch import settings
- Multi-pass encoding
- Background export queue

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.6 | Jan 2026 | Windows GPU proxy fix, streaming decode, FFmpeg WASM |
| 1.0.5 | Jan 2026 | 30+ modular GPU effects, effect bypass/quality controls |
| 1.0.4 | Jan 2026 | Local project storage, autosave, backup system |
| 1.0.3 | Jan 2026 | Smart media relink, reload all, welcome overlay |
| 1.0.2 | Jan 2026 | Media panel columns, idle mode, version display |
| 1.0.1 | Jan 2026 | Audio export, nested composition fixes |
| 1.0.0 | Jan 2026 | Initial release with full WebGPU pipeline |

---

## License

MASterSelects is proprietary software.

---

*Documentation updated January 2026*
