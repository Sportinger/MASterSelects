# Masks

[← Back to Index](./README.md)

Shape-based masking system with GPU-accelerated feathering.

---

## Table of Contents

- [Overview](#overview)
- [Creating Masks](#creating-masks)
- [Mask Modes](#mask-modes)
- [Editing Masks](#editing-masks)
- [Feathering](#feathering)
- [Mask Controls](#mask-controls)

---

## Overview

Masks define visible regions of a clip using vector shapes. Multiple masks can be combined for complex cutouts.

### Features
- **Vector shapes** - Resolution independent
- **Multiple masks** per clip
- **GPU feathering** - Smooth, fast blur
- **Add/Subtract modes** - Combine masks

---

## Creating Masks

### Shape Tools
Draw shapes directly in preview:

1. Select clip
2. Choose shape tool
3. Click and drag to draw

### Available Shapes
- **Rectangle** - Click-drag for corners
- **Ellipse** - Click-drag for bounds
- **Polygon** - Click points, double-click to close
- **Bezier** - Click for points, drag for handles

### Drag-to-Draw
- Click starting point
- Drag to define shape
- Release to complete
- Shape becomes editable

---

## Mask Modes

### Add Mode
- Reveals area inside mask
- Multiple Add masks combine (union)
- Default mode for new masks

### Subtract Mode
- Hides area inside mask
- Cuts holes in existing masks
- Works as first mask (shows everything except)

### Combining Modes
```
Mask 1 (Add): Large rectangle
Mask 2 (Subtract): Small circle
Result: Rectangle with circular hole
```

---

## Editing Masks

### Selection
- Click mask path to select
- Selected mask shows handles

### Moving Masks
- Drag mask center to move
- Position updates in real-time
- `Shift` + drag constrains to axis

### Vertex Editing
- Click vertex to select
- Drag to move vertex
- Bezier handles available

### Bezier Handles
- `Shift` + drag vertex for handle mode
- Handles scale along their line
- Smooth curves between points

### Hide Mask UI
- Toggle eye icon to hide mask editing UI
- Mask still applies, just hidden overlay

---

## Feathering

GPU-accelerated blur for soft mask edges.

### Feather Control
- Slider from 1-100
- Higher values = softer edges
- Real-time preview

### Feather Quality
Quality slider affects blur smoothness:
- Low: Faster, slight banding
- High: Smoother, more processing

### GPU Implementation
- Feather processed entirely on GPU
- No CPU texture upload needed
- Smooth slider interaction

---

## Mask Controls

### Control Panel
When clip with mask selected:
- **Mode** dropdown (Add/Subtract)
- **Feather** slider
- **Quality** slider
- **Delete** mask button

### Per-Mask Settings
Each mask has independent:
- Shape vertices
- Mode (Add/Subtract)
- Feather amount

---

## Coordinate Systems

### Mask Coordinates
- Masks defined in normalized coordinates
- Transform with layer position/scale
- Applied in output frame space

### Aspect Ratio
- SVG overlay preserves aspect ratio
- Coordinates match video dimensions
- No Y-axis distortion

---

## Technical Details

### Mask Rendering
1. Mask shape rendered to texture
2. Feather blur applied (GPU)
3. Result used as alpha mask
4. Applied during composition

### Texture Generation
- Uses engine render resolution
- Vertices transformed before generation
- Sampled in output frame space

---

## Related Features

- [Effects](./Effects.md) - Visual effects
- [Preview](./Preview.md) - Edit mode
- [GPU Engine](./GPU-Engine.md) - Rendering
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Commits: eeac396 through d63e381*
