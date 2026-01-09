# Timeline

[← Back to Index](./README.md)

The Timeline is the core editing interface, providing multi-track video and audio editing with composition support.

---

## Table of Contents

- [Track Types](#track-types)
- [Clips](#clips)
- [Compositions](#compositions)
- [Navigation & Zoom](#navigation--zoom)
- [Selection](#selection)
- [Track Controls](#track-controls)

---

## Track Types

### Video Tracks
- Support video files, images, and nested compositions
- Stack from top to bottom (top track = front layer)
- Auto-created when dropping media

### Audio Tracks
- Audio-only tracks at bottom of timeline
- Linked audio follows video clip movement
- Waveform visualization (see [Audio](./Audio.md))

### Track Creation
- **Drop media** on empty area → creates new track
- **New tracks**: Video tracks added at top, audio at bottom
- **Type enforcement**: Audio files only go to audio tracks

---

## Clips

### Adding Clips
1. Drag from [Media Panel](./Media-Panel.md) to timeline
2. Drop shows dashed preview with actual duration
3. Clips load thumbnails in background

### Clip Operations

| Action | Method |
|--------|--------|
| Move | Drag clip horizontally |
| Trim | Drag clip edges |
| Split | Press `C` at playhead |
| Delete | Select + `Delete` key |
| Duplicate | Copy/paste (planned) |

### Clip Snapping
- **Magnetic snapping** to other clip edges
- **Overlap resistance** prevents accidental overlap (2.0s buffer)
- **Linked audio** maintains sync during drag

### Clip Properties
Select a clip to edit in [Clip Properties Panel](./UI-Panels.md#clip-properties):
- Position (X, Y, Z)
- Scale (X, Y)
- Rotation (X, Y, Z) - full 3D rotation
- Opacity

### Split Clips
Press `C` to split all clips at playhead:
- Works on video and audio clips
- Linked clips split together
- Creates two independent clips

---

## Compositions

Compositions are containers for timeline content, similar to After Effects.

### Creating Compositions
1. Click `+` in Media Panel
2. Set name and duration
3. Composition appears in Media Panel

### Nested Compositions
- Drag composition from Media Panel to Timeline
- Double-click to enter and edit
- Changes reflect in parent composition

### Composition Tabs
- Tabs appear in timeline toolbar
- Click to switch between compositions
- Drag to reorder tabs
- Each composition has independent timeline data

### Composition Settings
- **Duration**: Editable via settings dialog
- **Resolution**: Set in composition settings
- Accessed from Media Panel context menu

---

## Navigation & Zoom

### Playhead
- Click timeline ruler to position
- Drag for scrubbing
- Snaps to keyframes when `Shift` held
- Snaps to clip start/end points

### Zooming

| Action | Effect |
|--------|--------|
| `Alt + Scroll` | Zoom in/out (centered on playhead) |
| `Fit` button | Fit composition to view |
| Zoom slider | Manual zoom control |

### Scrolling

| Action | Effect |
|--------|--------|
| Scroll wheel | Vertical scroll |
| `Shift + Scroll` | Horizontal scroll |
| Middle mouse | Pan timeline |

### Timeline Limits
- Zoom out limited to composition duration
- Prevents scrolling beyond content

---

## Selection

### Single Selection
- Click clip to select
- Click empty area to deselect

### Marquee Selection
- Click and drag on empty timeline area
- Rectangle selects all clips it touches
- Live visual feedback during drag

### Keyframe Selection
See [Keyframes](./Keyframes.md#selection)

---

## Track Controls

Each track has controls in the header:

| Control | Function |
|---------|----------|
| **Eye** | Toggle track visibility |
| **Mute** | Mute track audio |
| **Solo** | Solo this track (dim others) |
| **Name** | Double-click to edit |
| **Expand** | Show keyframe lanes |

### Track Visibility
- Eye toggle hides layer from render
- Muted tracks play no audio
- Solo shows only that track (others dimmed)

### Track Names
- Double-click track name to edit
- Names shown in AI context for smart editing

---

## Related Features

- [Keyframes](./Keyframes.md) - Animate clip properties
- [Preview](./Preview.md) - Playback and RAM Preview
- [Audio](./Audio.md) - Audio tracks and waveforms
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md) - Timeline shortcuts

---

*Commits: Initial through d63e381*
