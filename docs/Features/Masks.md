# Masks

[← Back to Index](./README.md)

Vector mask system with GPU-accelerated feathering.

---

## Table of Contents

- [Shape Tools](#shape-tools)
- [Mask Modes](#mask-modes)
- [Vertex Editing](#vertex-editing)
- [Feathering](#feathering)
- [Mask Properties](#mask-properties)

---

## Shape Tools

### Available Shapes

| Shape | Creation | Notes |
|-------|----------|-------|
| **Rectangle** | Click-drag corners | Instant shape |
| **Ellipse** | Click-drag bounds | Bezier approximation (k=0.5523) |
| **Freehand** | Click vertices | Pen tool, double-click to close |
| **Bezier Path** | Click + drag handles | Full cubic bezier |

### Drawing Modes
```typescript
type MaskEditMode =
  | 'none'
  | 'drawing'
  | 'editing'
  | 'drawingRect'
  | 'drawingEllipse'
  | 'drawingPen'
```

### Creating Masks
1. Select clip
2. Choose shape tool from panel
3. Draw in preview
4. Shape becomes editable

---

## Mask Modes

### Add Mode (Default)
- Reveals area inside mask
- Multiple Add masks combine (union)
- First mask shows only inside

### Subtract Mode
- Hides area inside mask
- Cuts holes in existing masks
- Works as first mask (shows everything except)

### Intersect Mode
- Shows only overlapping areas
- Requires existing mask

### Combining Example
```
Mask 1 (Add):      Large rectangle
Mask 2 (Subtract): Small circle
Result:            Rectangle with hole
```

---

## Vertex Editing

### Selection
- Click vertex to select
- Selected shows as cyan square
- Multiple selection supported

### Moving
- Drag vertex to reposition
- Real-time preview update

### Edge Dragging
- Drag a line segment between two mask vertices to move both at once
- Works alongside vertex and whole-mask dragging
- Intuitive for reshaping mask boundaries

### Bezier Handles
```typescript
interface BezierHandle {
  x: number;  // Time offset
  y: number;  // Value offset
}
```

- **In-handle**: Controls incoming curve
- **Out-handle**: Controls outgoing curve
- Drag each independently
- `Shift + drag`: Scale both handles proportionally

### Deleting Vertices
- Select vertex
- Press `Delete` or `Backspace`

---

## Feathering

### GPU Blur Implementation
3-tier quality system:

| Quality | Taps | Range |
|---------|------|-------|
| Low | 17-tap | 1-33 |
| Medium | 33-tap | 34-66 |
| High | 61-tap | 67-100 |

### Blur Algorithm
```
Multi-ring sampling at radii:
0.2r, 0.4r, 0.6r, 0.8r, 1.0r, 1.2r, 1.4r
Weighted averaging for smooth edges
```

### Controls
- **Feather slider**: 0-50 pixels
- **Quality slider**: 1-100 (affects tap count)
- Real-time preview

---

## Mask Properties

### Per-Mask Settings
```typescript
interface ClipMask {
  id: string;
  mode: 'add' | 'subtract' | 'intersect';
  opacity: number;      // 0-1
  feather: number;      // 0-50 pixels
  featherQuality: number; // 1-100
  inverted: boolean;
  position: { x: number; y: number };
  vertices: MaskVertex[];
  closed: boolean;
}
```

### Mask Operations
```typescript
addMask(clipId)
removeMask(clipId, maskId)
updateMask(clipId, maskId, updates)
reorderMasks(clipId, fromIndex, toIndex)
```

### Vertex Operations
```typescript
addVertex(clipId, maskId, vertex, index?)
removeVertex(clipId, maskId, vertexId)
updateVertex(clipId, maskId, vertexId, updates)
closeMask(clipId, maskId)
```

---

## Visual Feedback

### Overlay Colors
- **Blue dashed**: Unclosed path
- **Cyan solid**: Closed path
- **Red**: First vertex (in drawing mode)
- **Orange**: Bezier handle lines
- **Cyan squares**: Vertex points

### Cursor States
- **Crosshair**: Drawing modes
- **Move**: Over mask fill
- **Pointer**: Over handles

### Instructions
On-screen text shows current mode instructions.

---

## Rendering Pipeline

### Performance Optimizations
- Skip history snapshots during mask dragging for smooth interaction
- Vertex and handle updates throttled to 60fps (16ms interval) during drag operations
- Whole-mask dragging also throttled to ~60fps to prevent excessive store updates
- Mask texture regeneration skipped entirely during active drag for smooth interaction
- GPU texture updates throttled at 30fps instead of every frame
- Targeted cache invalidation (only affected layers)

### CPU Generation
```typescript
// maskRenderer.ts
generateMaskTexture(masks, width, height)
- Uses OffscreenCanvas with Canvas2D
- Bezier paths via ctx.bezierCurveTo()
- Composite operations for modes
- Returns ImageData (RGBA)
```

### GPU Application
```wgsl
// composite.wgsl
- Receives mask as texture_2d<f32>
- Applies feather blur
- Handles inversion
- R channel = mask value
```

### Coordinate System
- Normalized coordinates (0-1)
- Transforms with layer position/scale
- Applied in output frame space
- SVG overlay preserves aspect ratio

---

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| [`maskSlice.test.ts`](../../tests/stores/timeline/maskSlice.test.ts) | 22 | Mask CRUD, modes, vertices |

Run tests: `npx vitest run`

---

## Not Implemented

- Animated mask paths
- Mask tracking
- Rotobezier (auto-smooth)
- Mask interpolation between shapes

---

## Related Features

- [Effects](./Effects.md) - Visual effects
- [Preview](./Preview.md) - Edit mode
- [GPU Engine](./GPU-Engine.md) - Rendering
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Source: `src/stores/timeline/maskSlice.ts`, `src/components/preview/MaskOverlay.tsx`, `src/utils/maskRenderer.ts`*
