# UI & Panels

[← Back to Index](./README.md)

Dockable panel system with After Effects-style menu bar and unified Properties panel.

---

## Table of Contents

- [Menu Bar](#menu-bar)
- [Panel System](#panel-system)
- [Available Panels](#available-panels)
- [Properties Panel](#properties-panel)
- [Dock Layouts](#dock-layouts)
- [MIDI Control](#midi-control)

---

## Menu Bar

### Structure
| Menu | Contents |
|------|----------|
| **File** | New, Save, Open Recent |
| **Edit** | Copy, Paste, Settings |
| **View** | Panels, Resolution, Layout |
| **Output** | New Output Window, Active Outputs |
| **Window** | MIDI Control |

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New Project |
| `Ctrl+S` | Save Project |
| `Ctrl+O` | Open (shows menu) |

### Project Name
- Displayed at left of menu bar
- Click to edit/rename
- Updates on save

---

## Panel System

### Dockable Behavior
All panels can be:
- Dragged to rearrange
- Grouped in tabs
- Resized
- Closed/opened

### Tab Controls
| Action | Method |
|--------|--------|
| Switch tab | Click |
| Cycle tabs | Middle mouse scroll |
| Drag tab | Hold 500ms + drag |

### Hold-to-Drag
```
1. Click and hold tab for 500ms
2. Glow animation indicates ready
3. Drag to new position
4. Drop to place
```

### Tab Slot Indicators
Resolume-style visual feedback:
- Shows valid drop locations
- Highlights target slot

---

## Available Panels

MASterSelects has 9 dockable panel types:

| Panel | Purpose |
|-------|---------|
| **Preview** | Composition output canvas |
| **Timeline** | Multi-track editor |
| **Media** | Media browser and folders |
| **Properties** | Unified clip editing (Transform, Effects, Masks, Volume) |
| **Export** | Render settings and progress |
| **Multicam** | Camera sync and EDL |
| **AI Chat** | GPT-powered editing assistant |
| **AI Video** | AI video generation (Kling) |
| **Slots** | Layer slot management |

### Preview Panel
- Canvas for composition output
- Composition selector dropdown
- Edit mode toggle for direct manipulation
- Multiple preview panels supported
- Statistics overlay option

### Timeline Panel
- Multi-track video/audio editor
- Composition tabs for switching
- Playback controls toolbar
- Snap toggle button
- Ruler with time display
- Track headers with controls

### Media Panel
- Media browser with thumbnails
- Folder organization tree
- Composition list
- Add dropdown (Import, Composition, Folder)
- Drag-to-timeline support

### Properties Panel
See [Properties Panel](#properties-panel) section below for details.

### Export Panel
- Codec selection (H.264, VP9)
- Resolution presets
- Frame rate options
- Quality/bitrate settings
- Progress indicator with ETA
- Single frame export

### Multicam Panel
- Camera clip management
- Audio-based sync controls
- EDL generation
- Group linking controls

### AI Chat Panel
- Chat interface with GPT-4
- Model/provider selector
- Context-aware editing commands
- 50+ available tools

### AI Video Panel
- Text-to-video generation
- Image-to-video animation
- Kling AI integration
- Model/duration/aspect ratio selection
- CFG scale and camera controls
- Generation queue with status

### Slots Panel
- Layer slot grid
- Visibility toggles
- Effect status indicators

---

## Properties Panel

The unified Properties panel consolidates clip editing into a single tabbed interface. It automatically adapts based on the selected clip type.

### Video Clip Tabs

| Tab | Contents |
|-----|----------|
| **Transform** | Position, Scale, Rotation, Opacity, Blend Mode |
| **Effects** | GPU effects list with parameters |
| **Masks** | Mask shapes with mode and feather controls |

### Audio Clip Tabs

| Tab | Contents |
|-----|----------|
| **Volume** | Volume slider + 10-band parametric EQ |
| **Effects** | Audio effects (future expansion) |

### Transform Tab Features
- **Position**: X, Y, Z (depth) sliders
- **Scale**: X, Y with link toggle
- **Rotation**: X, Y, Z (3D rotation)
- **Opacity**: 0-100% slider
- **Blend Mode**: Dropdown with 37 modes grouped by category
- Keyframe toggles on each property

### Volume Tab Features
- **Volume Slider**: -60dB to +12dB
- **10-Band EQ**: 31Hz to 16kHz
- **Per-Band Gain**: -12dB to +12dB
- Keyframe toggles for animation
- EQ automatically added on first use

### Effects Tab Features
- **Add Effect**: Dropdown with available effects
- **Effect List**: Expandable sections
- **Parameter Sliders**: With keyframe toggles
- **Remove Button**: Per-effect deletion

### Masks Tab Features
- **Add Mask**: Rectangle, Ellipse, Pen tool
- **Mask List**: With expand/collapse
- **Mode Selector**: Add, Subtract, Intersect
- **Feather Slider**: 0-100px GPU blur
- **Expansion**: -100 to +100px
- **Invert Toggle**: Flip mask selection
- **Vertex Selection**: Edit mask points

### Tab Title Display
- Shows selected clip name in tab title
- Example: "Properties - Interview_01.mp4"
- Updates automatically on clip selection

---

## Dock Layouts

### Default Layout (3-column)
```
┌─────────────────────────────────────────┐
│              Menu Bar                    │
├───────────┬─────────────────┬───────────┤
│  Media    │                 │Properties │
│  Panel    │    Preview      │  Panel    │
│           │                 │           │
│           │                 │           │
│           │                 │           │
├───────────┴─────────────────┴───────────┤
│              Timeline                    │
└─────────────────────────────────────────┘
```

### Layout Persistence
- Auto-saved to localStorage
- Survives page refresh
- Multiple preview panels preserved
- Auto-cleanup of invalid panel types

### Layout Actions
| Action | Location |
|--------|----------|
| Save as Default | View menu |
| Reset Layout | View menu |

### Panel Visibility
View menu → Panels:
- Checkbox for each panel type
- Toggle panels on/off

---

## MIDI Control

### Enabling MIDI
Window menu → MIDI Control

### Requirements
- Browser Web MIDI API support
- MIDI device connected
- Permission granted

### Status Display
```
✓ MIDI Control (N devices)
```

### Device Discovery
- Automatic device detection
- Shows device count when enabled

---

## Resolution Settings

### Output Resolution
View menu → Resolution:
| Preset | Dimensions |
|--------|------------|
| 1080p | 1920×1080 |
| 720p | 1280×720 |
| 4K | 3840×2160 |
| 16:10 | 1920×1200 |
| 4:3 | 1024×768 |

### Preview Quality
View menu → Preview Quality:
| Option | Render Size | Performance |
|--------|-------------|-------------|
| **Full (100%)** | 1920×1080 | Best quality |
| **Half (50%)** | 960×540 | 4× faster, 75% less memory |
| **Quarter (25%)** | 480×270 | 16× faster, 94% less memory |

Preview Quality scales the internal render resolution while maintaining the output aspect ratio. Lower quality settings significantly reduce GPU workload and memory usage—ideal for complex compositions or slower hardware.

**Memory Savings at Half Resolution:**
- Ping-pong buffers: 75% reduction
- RAM Preview cache: 75% reduction (7.2GB → 1.8GB)
- Scrubbing cache: 75% reduction

### Setting Resolution
```typescript
setResolution(width, height)
setPreviewQuality(quality) // 1, 0.5, or 0.25
```

---

## Settings Dialog

### Opening
Edit menu → Settings

### Contents
- API key management
- Transcription provider selection
- Language selection
- Thumbnail/waveform toggles

### Storage
Settings persisted in localStorage.

---

## Status Indicator

### WebGPU Status
Top-right of toolbar:
```
● WebGPU Ready  (green)
○ Loading...    (gray)
```

---

## Context Menus

### Behavior
- Right-click to open
- Stay within viewport bounds
- Solid backgrounds
- Close on outside click

### Common Options
- Rename
- Delete
- Settings
- Context-specific actions

---

## Related Features

- [Timeline](./Timeline.md) - Timeline panel details
- [Preview](./Preview.md) - Preview panel details
- [Media Panel](./Media-Panel.md) - Media browser
- [Effects](./Effects.md) - Effect parameters
- [Audio](./Audio.md) - Volume and EQ details
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Source: `src/components/panels/PropertiesPanel.tsx`, `src/components/dock/`, `src/stores/dockStore.ts`, `src/types/dock.ts`*
