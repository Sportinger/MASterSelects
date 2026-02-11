# MASterSelects Documentation

**Professional WebGPU Video Compositor & Timeline Editor**

Version 1.2.4 | February 2026

---

## Overview

MASterSelects is a browser-based professional video editing application built on cutting-edge WebGPU technology. It delivers After Effects-style compositing, multi-track timeline editing, AI-powered workflows, and real-time GPU rendering—all running entirely in the browser with no plugins or installations required.

### Key Highlights

| Capability | Description |
|------------|-------------|
| **WebGPU Rendering** | Hardware-accelerated compositing with zero-copy video textures at 60fps |
| **Multi-track Timeline** | Professional NLE with video/audio tracks, nested compositions, and multicam |
| **Keyframe Animation** | Full property animation with bezier curve editor and 5 easing modes |
| **AI Integration** | 33 intelligent editing tools via OpenAI function calling (GPT-4/GPT-5) |
| **AI Video Generation** | PiAPI integration for AI-powered video creation |
| **Download Panel** | Download videos from YouTube, TikTok, Instagram, Twitter/X and more |
| **30+ GPU Effects** | Modular color, blur, distort, stylize, keying effects with quality controls |
| **Video Scopes** | GPU-accelerated Histogram, Vectorscope, Waveform monitor (DaVinci-style) |
| **Text Clips** | Typography with 50 Google Fonts, stroke, shadow effects |
| **Solid Color Clips** | Solid color layers with color picker and comp dimensions |
| **Professional Audio** | 10-band parametric EQ with live Web Audio, audio master clock, varispeed |
| **Multicam Support** | Audio-based cross-correlation synchronization |
| **Transitions** | Crossfade transitions with GPU-accelerated rendering |
| **4 Export Modes** | WebCodecs Fast, HTMLVideo Precise, FFmpeg WASM, FCP XML interchange |
| **Parallel Decoding** | Multi-clip parallel decode for faster exports |
| **Output Manager** | Source routing, slice management, corner pin warping, multi-window control |
| **Slot Grid** | Resolume-style 4x12 grid with multi-layer playback and column activation |
| **Native Helper** | Optional 10x faster ProRes/DNxHD decode with hardware accel |
| **Local Storage** | Project folder with Raw media, autosave, backups, smart relinking |
| **Mobile Support** | Responsive UI with touch gestures |

---

## Technology Stack

```
Frontend          React 19 + TypeScript + Vite 7.2
State Management  Zustand with modular slice architecture
GPU Rendering     WebGPU + WGSL shaders (2,400+ lines)
GPU Effects       30+ modular effects with individual WGSL shaders
Video Decoding    WebCodecs API with hardware acceleration + parallel decode
Video Encoding    WebCodecs (Fast/Precise) + FFmpeg WASM (ProRes, DNxHR, HAP)
Audio Processing  Web Audio API, audio master clock, varispeed scrubbing
AI Services       OpenAI GPT-4/GPT-5 function calling, PiAPI video generation
Persistence       File System Access API + local project folders with Raw media
Native Helper     Rust + FFmpeg + yt-dlp (unified cross-platform)
UI Framework      Custom dockable panel system with mobile support
```

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Timeline](./Timeline.md) | Multi-track editing, clips, snapping, compositions, multicam |
| [Keyframes](./Keyframes.md) | Animation system, curve editor, bezier interpolation |
| [Preview & Playback](./Preview.md) | RAM Preview, scrubbing, multiple outputs, edit mode |
| [Output Manager](./Preview.md#output-manager) | Source routing, slices, corner pin warping, mask layers |
| [Effects](./Effects.md) | 30+ modular GPU effects, 37 blend modes, transforms |
| [Masks](./Masks.md) | Shape masks, pen tool, GPU feathering |
| [AI Integration](./AI-Integration.md) | 33 AI tools, transcription, AI video generation |
| [Media Panel](./Media-Panel.md) | Import, folder organization, columns, compositions |
| [Audio](./Audio.md) | 10-band EQ, audio master clock, varispeed scrubbing |
| [Text Clips](./Text-Clips.md) | Typography, 50 Google Fonts, stroke, shadow |
| [Export](./Export.md) | WebCodecs Fast/Precise, FFmpeg, parallel decoding |
| [UI & Panels](./UI-Panels.md) | Dockable panels, layouts, menus, mobile support |
| [GPU Engine](./GPU-Engine.md) | WebGPU architecture, modular render pipeline |
| [Project Persistence](./Project-Persistence.md) | Local folders, Raw media, autosave, backups |
| [Proxy System](./Proxy-System.md) | GPU-accelerated proxy generation |
| [Download Panel](./YouTube.md) | YouTube, TikTok, Instagram, Twitter/X downloads |
| [Native Helper](./Native-Helper.md) | Turbo Mode for ProRes/DNxHD, YouTube downloads |
| [Keyboard Shortcuts](./Keyboard-Shortcuts.md) | Complete shortcut reference |
| [Debugging](./Debugging.md) | Logger service, module filtering, AI-agent inspection |

---

## Feature Catalog

### Timeline & Editing

| Feature | Status | Details |
|---------|--------|---------|
| Multi-track Timeline | ✅ | Unlimited video and audio tracks |
| Clip Operations | ✅ | Move, trim, split, delete, reverse |
| Cut Tool | ✅ | `C` to split clips with snapping (Alt to disable) |
| Copy/Paste | ✅ | `Ctrl+C/V` with effects, keyframes, masks preserved |
| Multi-Select Movement | ✅ | Shift+Click to select multiple clips, drag as group with boundary collision |
| Bezier Fade Curves | ✅ | Visual opacity fades with real-time preview |
| Magnetic Snapping | ✅ | 0.1s snap distance with edge alignment |
| Snap Toggle | ✅ | Toolbar button to enable/disable snapping |
| Overlap Resistance | ✅ | 100px horizontal + 100px vertical cross-track resistance |
| Linked Audio | ✅ | Video-audio linking with Alt+drag override |
| Linked Clip Selection | ✅ | Click linked video/audio to select both, Shift+Click for independent |
| Nested Compositions | ✅ | Orange outline, boundary markers, recursive rendering |
| Composition Tabs | ✅ | Multiple open compositions with tab navigation |
| Clip Entrance Animation | ✅ | Smooth animation when switching compositions |
| Track Controls | ✅ | Visibility, mute, solo, expand |
| Smooth Track Height | ✅ | Continuous scrolling resize, minimum 20px for compact view |
| Exponential Zoom | ✅ | Alt+Scroll with 8% per step, consistent at all zoom levels |
| Vertical Scroll Snapping | ✅ | Scroll snaps to track boundaries, one layer per step |
| Video/Audio Separator | ✅ | Green divider line between video and audio tracks |
| Playback Looping | ✅ | In/Out points with loop mode |
| JKL Playback | ✅ | Industry-standard J/K/L shortcuts for playback control |
| Solid Color Clips | ✅ | Solid layers with color picker and comp dimensions |
| Transitions | ✅ | Crossfade transitions with GPU-accelerated rendering |
| Marker Drag-to-Create | ✅ | Drag M button to create markers with ghost preview |

### Output Manager

| Feature | Status | Details |
|---------|--------|---------|
| RenderTarget System | ✅ | Unified rendering to multiple independent outputs |
| Output Manager | ✅ | Source routing, slice management, multi-window control |
| Corner Pin Warping | ✅ | Slice system with 4-corner warping for projection mapping |
| Mask Layers | ✅ | Per-slice mask layers with invert and drag-drop reorder |
| Output Persistence | ✅ | Auto-save per project, window geometry preservation |

### Slot Grid

| Feature | Status | Details |
|---------|--------|---------|
| Slot Grid | ✅ | Resolume-style 4x12 grid with click-to-play activation |
| Multi-Layer Playback | ✅ | 4 independent layers (A-D) with wall-clock time |
| Column Activation | ✅ | Click column header to activate entire column |

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
| Keyframe Copy/Paste | ✅ | Ctrl+C/V copies selected keyframes, paste at playhead |
| Keyframe Tick Marks | ✅ | Amber diamond markers on clip bars show keyframe positions |
| Multi-Select Movement | ✅ | Select multiple keyframes and move together by same time delta |
| Curve Editor Auto-Scale | ✅ | Y-axis auto-scales to fit curve, Shift+wheel to resize height |

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
- ✅ **Inline effects:** Brightness, contrast, saturation, invert run inside composite shader (no extra render passes)

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
| 33 AI Tools | ✅ | Clip, track, keyframe, effect operations |
| Local Whisper | ✅ | Browser-based transcription |
| OpenAI Whisper API | ✅ | Cloud transcription service |
| AssemblyAI | ✅ | Professional transcription |
| Deepgram | ✅ | Fast transcription service |
| Multicam EDL | ✅ | AI-generated edit decision lists |
| Context Awareness | ✅ | AI knows timeline state |
| SAM2 Segmentation | ✅ | Click-to-segment with WebGPU ONNX inference |

### Download Panel

| Feature | Status | Details |
|---------|--------|---------|
| YouTube Search | ✅ | Search videos via Invidious or YouTube Data API |
| Video Thumbnails | ✅ | Display thumbnails, titles, channels, duration |
| Quality Selection | ✅ | Choose video quality before download |
| Native Helper Download | ✅ | Fast downloads via yt-dlp integration |
| Cobalt Fallback | ✅ | Download via Cobalt API if Native Helper unavailable |
| Add to Timeline | ✅ | Download and add directly to timeline |
| Project Storage | ✅ | Downloads saved to project YT/ folder |
| H.264 Preference | ✅ | Prefers H.264 over AV1/VP9 for compatibility |
| Dual API Support | ✅ | Invidious (no key) or YouTube Data API (optional) |
| Multi-Platform Download | ✅ | TikTok, Instagram, Twitter/X, Facebook, Reddit, Vimeo, Twitch |
| Platform Subfolders | ✅ | Downloads organized by platform in project folder |

### Audio

| Feature | Status | Details |
|---------|--------|---------|
| 10-Band Parametric EQ | ✅ | 31Hz to 16kHz frequency bands |
| EQ Gain Range | ✅ | -12dB to +12dB per band |
| EQ Keyframes | ✅ | Animate EQ parameters over time |
| Audio Master Clock | ✅ | Playhead follows audio for perfect sync |
| Varispeed Scrubbing | ✅ | Continuous playback with speed adjustment |
| Speed Property | ✅ | Keyframeable clip playback speed |
| Waveform Display | ✅ | 50 samples/second resolution |
| Volume Control | ✅ | Per-clip volume adjustment |
| Multicam Audio Sync | ✅ | Cross-correlation algorithm |
| Audio Track Mute | ✅ | Per-track mute control |
| Audio Solo | ✅ | Isolate audio tracks |
| Live EQ via Web Audio | ✅ | Real-time equalization using Web Audio API, hear changes instantly |
| Audio Tab for Video Clips | ✅ | Video clips have dedicated Audio tab with volume and keyframes |
| Composition Audio | ✅ | Nested composition audio mixdown |

### Preview & Playback

| Feature | Status | Details |
|---------|--------|---------|
| Real-time Preview | ✅ | 60fps GPU rendering |
| Idle Mode | ✅ | Auto-pause GPU when nothing changes |
| Preview Quality | ✅ | Full/Half/Quarter resolution for performance |
| Layer Caching | ✅ | Better performance when paused or scrubbing |
| Auto Frame Caching | ✅ | Cache frames during playback for instant scrubbing |
| RAM Preview | ✅ | Cached playback with 900 frame limit |
| Multiple Outputs | ✅ | Open multiple preview windows |
| Per-Preview Grid | ✅ | Individual transparency grid toggle |
| Edit Mode | ✅ | Direct manipulation in preview (Tab toggle) with corner/edge transform handles |
| Transform Handles | ✅ | Corner and edge handles for scaling, Shift for aspect ratio lock |
| Proxy Cache Indicator | ✅ | Yellow indicator on ruler shows cached proxy frames |
| Video Warmup Button | ✅ | Cache button for preloading proxy frames before playback |
| Scrubbing Cache | ✅ | 3-tier caching system (LRU 300 + last frame + composite) |
| Statistics Overlay | ✅ | FPS, timing, idle status, GPU vendor |
| Resolution Presets | ✅ | 480p to 4K preview |
| Pause on Drag | ✅ | Playback pauses when dragging playhead |
| GPU Recovery | ✅ | Automatic WebGPU device recovery |
| Numpad Blend Cycling | ✅ | Numpad +/- to cycle through blend modes |

### Export

| Feature | Status | Details |
|---------|--------|---------|
| Export System V2 | ✅ | Shared decoder pool with intelligent frame caching |
| Export Planner | ✅ | Smart decode scheduling for optimized performance |
| WebCodecs Fast Mode | ✅ | Sequential decoding with MP4Box parsing |
| HTMLVideo Precise Mode | ✅ | Frame-accurate seeking for complex timelines |
| FFmpeg WASM Export | ✅ | ProRes, DNxHR, HAP codecs |
| Parallel Decoding | ✅ | Multi-clip parallel decode for faster exports |
| H.264/VP9 Export | ✅ | MP4/WebM containers |
| Resolution Presets | ✅ | 480p, 720p, 1080p, 4K |
| Frame Rate Options | ✅ | 24, 25, 30, 60 fps |
| Quality Presets | ✅ | 5-35 Mbps bitrate |
| AAC/Opus Audio | ✅ | Auto-detect browser codec support |
| In/Out Export | ✅ | Export marked region |
| Single Frame Export | ✅ | PNG frame capture |
| Progress Overlay | ✅ | Timeline progress bar with cancel |
| Auto Fallback | ✅ | Falls back to Precise mode if Fast fails |
| FCP XML Export | ✅ | Export timeline to Final Cut Pro XML for interchange with Premiere/Resolve |

### Project & Media Management

| Feature | Status | Details |
|---------|--------|---------|
| Local Project Folder | ✅ | All data stored in user-selected folder |
| Auto-Copy to Raw | ✅ | Imported media automatically copied to Raw/ folder |
| Auto-Relink from Raw | ✅ | Auto-restore missing files from Raw folder on load |
| Auto-Save | ✅ | Configurable interval (1-10 min) |
| Backup System | ✅ | Keeps last 20 backups automatically |
| UI State Persistence | ✅ | Dock layout + timeline view state saved per project |
| Welcome Overlay | ✅ | Project folder selection on launch |
| Save As | ✅ | Export project to new location |
| Smart Media Relink | ✅ | Auto-find moved/renamed files |
| Reload All | ✅ | Restore file permissions after restart |
| Media Import | ✅ | MP4, WebM, MOV, WAV, MP3, PNG, JPG |
| Proxy Generation | ✅ | GPU-accelerated, Windows/Linux/Mac |
| Hash Deduplication | ✅ | Same files share proxies/thumbnails |
| IndexedDB Error Dialog | ✅ | Clear error when browser storage is corrupted |

### Native Helper (Turbo Mode)

| Feature | Status | Details |
|---------|--------|---------|
| Unified Cross-Platform Build | ✅ | FFmpeg decode/encode + yt-dlp downloads on all platforms |
| ProRes Decoding | ✅ | All profiles at native speed |
| DNxHD/DNxHR Decoding | ✅ | All profiles at native speed |
| Hardware Acceleration | ✅ | VAAPI (Intel/AMD), NVDEC (NVIDIA) |
| YouTube Downloads | ✅ | yt-dlp integration with quality selection |
| Frame Cache | ✅ | LRU cache up to 2GB |
| Background Prefetch | ✅ | Frames loaded ahead of playhead |
| Native Encoding | ✅ | 10x faster ProRes/DNxHD export |
| Auto-Detection | ✅ | Toolbar shows "⚡ Turbo" when connected |
| Download Link | ✅ | Click indicator for helper download |

### User Interface

| Feature | Status | Details |
|---------|--------|---------|
| Dockable Panels | ✅ | Drag, resize, tab grouping |
| 14 Panel Types | ✅ | Preview, Timeline, Media, Properties, Export, Multicam, AI Chat, AI Video, YouTube, Transitions, Histogram, Vectorscope, Waveform, Slots |
| Video Scopes | ✅ | GPU-accelerated Histogram, Vectorscope, Waveform monitor with RGB/R/G/B/Luma modes |
| Transitions Panel | ✅ | Modular panel with drag-drop support for applying transitions |
| Unified Properties Panel | ✅ | Transform, Effects, Masks, Audio, Transcript, Analysis |
| AE-Style Settings Dialog | ✅ | Sidebar navigation with categorized settings, draggable |
| Menu Bar | ✅ | File, Edit, View, Output, Audio, Info, Window |
| Context Menus | ✅ | Right-click operations (viewport-bounded) |
| WYSIWYG Thumbnails | ✅ | Thumbnails show effects applied to clips |
| WebGPU Thumbnail Renderer | ✅ | GPU-accelerated thumbnails for nested comps |
| MIDI Control | ✅ | Web MIDI API integration |
| Keyboard Shortcuts | ✅ | Comprehensive hotkey support |
| Hold-to-Drag Tabs | ✅ | 500ms hold to reorder |
| Mobile Support | ✅ | Responsive layout with touch gestures |
| Desktop Mode Toggle | ✅ | Option to view full UI on mobile |
| What's New Dialog | ✅ | Time-grouped changelog on refresh |
| Welcome Overlay | ✅ | Project folder selection on launch |
| Tutorial System | ✅ | Spotlight-based panel intro with Clippy mascot |
| Welcome Screen | ✅ | Program selection (Premiere, Resolve, FCP, AE, Beginner) |

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
│  │  Timeline   │ │   Dock      │ │   Media     │ │    Multicam       │  │
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
| `composite.wgsl` | 618 | Layer compositing, 37 blend modes |
| `opticalflow.wgsl` | 326 | Motion analysis, scene detection |
| `effects.wgsl` | 243 | GPU effect implementations |
| `output.wgsl` | 71 | Final output passthrough |
| `slice.wgsl` | 33 | Output slice rendering |
| `common.wgsl` | 154 | Shared effect utilities |
| 30 effect shaders | ~954 | Individual GPU effect shaders |
| **Total** | **~2,400** | |

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
| | `J` / `K` / `L` | Reverse / Pause / Forward (JKL shuttle) |
| | `←` / `→` | Step frame |
| | `Home` / `End` | Go to start / end |
| **Editing** | `C` | Split at playhead |
| | `Delete` | Delete selected |
| | `Ctrl+C` / `Ctrl+V` | Copy/Paste clips or keyframes |
| | `Ctrl+Z` | Undo |
| | `Ctrl+Shift+Z` | Redo |
| **Timeline** | `I` | Set In point |
| | `O` | Set Out point |
| | `X` | Clear In/Out |
| | `Tab` | Toggle edit mode |
| **Selection** | `Shift+Click` | Multi-select clips |
| | `Shift++` / `Shift+-` | Cycle blend modes |
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

## Test Coverage

Overview of unit test coverage across feature areas. Run all tests with `npx vitest run`.

| Feature Doc | Test Files | Tests | Notes |
|-------------|-----------|-------|-------|
| [Timeline](./Timeline.md) | 5 | 101 | Clips, tracks, selection, playback, markers |
| [Keyframes](./Keyframes.md) | 2 | 66 | Keyframe CRUD, easing, bezier interpolation |
| [Preview](./Preview.md) | — | — | GPU-dependent, not unit testable |
| [Export](./Export.md) | 1 | 42 | FCP XML, time calculations, settings |
| [Audio](./Audio.md) | 2 | 49 | Volume, EQ, timing, speed, audio sync |
| [Effects](./Effects.md) | 2 | 42 | Registry, parameters, categories, type helpers |
| [GPU Engine](./GPU-Engine.md) | 1 | 13 | Transform math |
| [Masks](./Masks.md) | 1 | 22 | Mask CRUD, modes, vertices |
| [AI Integration](./AI-Integration.md) | 1 | 60 | Tool definitions, schemas |
| [Text Clips](./Text-Clips.md) | 1 | 50 | Covered by clipSlice tests |
| [Media Panel](./Media-Panel.md) | 2 | 60 | Files, folders, solids, compositions |
| [Proxy System](./Proxy-System.md) | — | — | Hardware-dependent |
| [Download Panel](./YouTube.md) | — | — | Requires network/native helper |
| [Project Persistence](./Project-Persistence.md) | 2 | 38 | Serialization, undo/redo |
| [Native Helper](./Native-Helper.md) | — | — | Rust binary, tested separately |
| [Keyboard Shortcuts](./Keyboard-Shortcuts.md) | 1 | 16 | Playback shortcuts |
| [UI Panels](./UI-Panels.md) | — | — | React component-level UI |
| [Multicam AI](./Multicam-AI.md) | 1 | 6 | Audio sync cross-correlation |

**Total: ~565 tests across 22 test files**

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
| 1.2.4 | Feb 2026 | Slot Grid (Resolume-style multi-layer composition), Output Manager (source routing, corner pin warping, mask layers, persistence), Download Panel (multi-platform: TikTok, Instagram, Twitter/X), unified native helper, render loop watchdog |
| 1.2.3 | Feb 2026 | Tutorial system (Clippy mascot, welcome screen, panel intro, timeline deep-dive), SAM2 AI segmentation, composition resolution drives render pipeline, Vitest test suite (182 tests) |
| 1.2.2 | Feb 2026 | Video Scopes (Histogram, Vectorscope, Waveform), keyframe copy/paste, keyframe tick marks, curve editor auto-scale, mask edge dragging, exponential zoom, cross-track resistance, vertical scroll snapping |
| 1.2.1 | Feb 2026 | Linked clip selection, proxy resume from disk, proxy rewrite with parallel JPEG encoding, instant media import, split deep-clone fix |
| 1.2.0 | Feb 2026 | Solid color clips, AE visual redesign, inline GPU effects (brightness/contrast/saturation/invert in composite shader), AE-style media panel, lazy-load panels |
| 1.1.9 | Feb 2026 | React performance optimization, store subscription cleanup, PropertiesPanel code splitting, draggable settings dialog |
| 1.1.8 | Feb 2026 | Transitions system, JKL playback shortcuts, multi-select clip/keyframe movement, FCP XML export, transform handles in edit mode, proxy cache indicator, settings dialog redesign |
| 1.1.7 | Feb 2026 | Live EQ via Web Audio, audio tab for video clips, export crash fixes, texture lifecycle management |
| 1.1.6 | Feb 2026 | Nested comp boundary markers, fade curve bezier display, ESLint cleanup |
| 1.1.5 | Jan 2026 | WYSIWYG thumbnails, copy/paste clips, content-aware thumbnail sampling |
| 1.1.4 | Jan 2026 | WebGPU thumbnail renderer for nested compositions |
| 1.1.3 | Jan 2026 | Export System V2 with shared decoder pool, export planner |
| 1.1.2 | Jan 2026 | Nested comp export fixes, Windows build notice |
| 1.1.1 | Jan 2026 | Cut tool, bezier fade curves, auto frame caching, UI state persistence, marker drag-to-create, major refactoring (WebGPUEngine, Timeline, FrameExporter, ClipSlice), auto-copy to Raw folder |
| 1.1.0 | Jan 2026 | Export mode selection, audio master clock, varispeed scrubbing, parallel decoding |
| 1.0.9 | Jan 2026 | Layer caching, WebCodecs export optimization, case-insensitive file detection |
| 1.0.8 | Jan 2026 | Native Helper YouTube download, NativeDecoder integration, FFmpeg audio export |
| 1.0.7 | Jan 2026 | Mobile UI, desktop mode toggle, FFmpeg direct loading |
| 1.0.6 | Jan 2026 | Windows GPU proxy fix, streaming decode, FFmpeg WASM |
| 1.0.5 | Jan 2026 | 30+ modular GPU effects, effect bypass/quality controls |
| 1.0.4 | Jan 2026 | Local project storage, autosave, backup system |
| 1.0.3 | Jan 2026 | Smart media relink, reload all, welcome overlay |
| 1.0.2 | Jan 2026 | Media panel columns, idle mode, version display |
| 1.0.1 | Jan 2026 | Audio export, nested composition fixes |
| 1.0.0 | Jan 2026 | Initial release with full WebGPU pipeline |

---

## License

MIT - see [LICENSE](../../LICENSE)

---

*Documentation updated February 2026*
