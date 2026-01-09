# Preview & Playback

[← Back to Index](./README.md)

Preview system with RAM caching, multiple panels, and real-time GPU rendering.

---

## Table of Contents

- [Preview Panel](#preview-panel)
- [Playback Controls](#playback-controls)
- [RAM Preview](#ram-preview)
- [Multiple Previews](#multiple-previews)
- [Scrubbing](#scrubbing)
- [Edit Mode](#edit-mode)

---

## Preview Panel

The Preview panel displays the rendered composition output.

### Features
- **Real-time GPU rendering** via WebGPU
- **Aspect ratio preserved** automatically
- **Close button** to hide panel
- **Composition selector** to view different compositions

### Resolution
- Renders at composition resolution
- Scales to fit panel size
- Full resolution available in export

---

## Playback Controls

Located in timeline toolbar:

| Control | Function |
|---------|----------|
| **Play/Pause** | `Space` - Toggle playback |
| **Stop** | Stop and return to start |
| **Loop** | Toggle loop playback |
| **Frame Back** | `←` - Previous frame |
| **Frame Forward** | `→` - Next frame |
| **Reverse** | Play backwards |

### Loop Playback
- Loops between In/Out points if set
- Otherwise loops full composition
- Toggle with loop button

### Reverse Playback
- Frame-by-frame backward playback
- Keyframe mirroring for proper animation
- Smooth reverse video playback

### In/Out Points
- Set with `I` (In) and `O` (Out) keys
- Defines playback and export range
- Visual markers on ruler

---

## RAM Preview

After Effects-style cached preview for smooth playback.

### How It Works
1. Toggle RAM Preview button
2. System caches frames to GPU textures
3. Playback uses cached frames (instant)
4. Cache invalidated on changes

### RAM Preview Behavior
- **Renders outward** from current playhead position
- **Progress indicator** shows caching status
- **Auto-start** works without In/Out points
- **Skips empty areas** for efficiency

### Cache Invalidation
Cache clears automatically when:
- Clip transforms change
- Track visibility changes
- Effects modified
- Timeline structure changes

### Performance
- Uses GPU texture cache
- Instant scrubbing after caching
- Reuses already-cached frames
- Video paused during RAM Preview generation

---

## Multiple Previews

Open multiple preview panels to view different compositions.

### Adding Preview Panels
1. View menu → Add Preview Panel
2. Or duplicate existing panel

### Composition Selection
Each preview can show different composition:
1. Use composition dropdown in panel
2. Select composition to display
3. Panel renders independently

### Independent Rendering
- Each panel has own canvas
- Separate ping-pong buffers
- Can show parent and nested comp simultaneously

### Layout
- Panels appear side-by-side
- Drag to rearrange in dock
- Save layout for persistence

---

## Scrubbing

### Timeline Scrubbing
- Drag playhead for real-time preview
- GPU renders each frame on demand
- Smooth with frame caching enabled

### Performance Optimization
- Frame cache stores recent frames
- Skips caching during active drag
- Prevents glitchy frames during scrub

### Value Scrubbing
In properties panel:
- Left-click drag on values to scrub
- Real-time preview updates
- Tooltip shows current value

---

## Edit Mode

Toggle edit mode to interact with layer transforms directly in preview.

### Bounding Boxes
- Shows bounding box for selected layer
- Handles at corners and edges
- Visual feedback for transforms

### Drag Operations

| Drag | Action |
|------|--------|
| Center | Move layer position |
| Corner | Scale layer |
| Edge | Scale single axis |
| Outside | Rotate layer |

### Real-time Updates
- Preview updates during drag
- Position synced with properties panel
- Changes reflected in timeline

---

## Frame-by-Frame Navigation

| Key | Action |
|-----|--------|
| `←` | Previous frame |
| `→` | Next frame |
| `Home` | Go to start |
| `End` | Go to end |

---

## Playback Sync

### Video Sync
- Videos seek to playhead position
- Uses `requestVideoFrameCallback` for smooth playback
- Falls back to standard playback if needed

### Nested Composition Sync
- Parent timeline controls nested playback
- Nested composition syncs to parent time
- Recursive rendering for deep nesting

---

## Related Features

- [Timeline](./Timeline.md) - Main editing interface
- [Export](./Export.md) - Render to file
- [GPU Engine](./GPU-Engine.md) - Rendering details
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Commits: 3d3b4fb through d63e381*
