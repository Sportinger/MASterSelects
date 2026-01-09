# Keyframes

[← Back to Index](./README.md)

The keyframe animation system enables property animation over time with multiple interpolation modes.

---

## Table of Contents

- [Overview](#overview)
- [Creating Keyframes](#creating-keyframes)
- [Editing Keyframes](#editing-keyframes)
- [Interpolation & Easing](#interpolation--easing)
- [Selection](#selection)
- [Keyframe Display](#keyframe-display)

---

## Overview

Keyframes define property values at specific points in time. The system interpolates values between keyframes for smooth animation.

### Animatable Properties

| Category | Properties |
|----------|------------|
| **Position** | X, Y, Z (depth) |
| **Scale** | X, Y |
| **Rotation** | X, Y, Z (3D rotation) |
| **Opacity** | 0-100% |
| **Effects** | All effect parameters |

---

## Creating Keyframes

### Method 1: Property Row Controls
1. Expand track to show properties
2. Click diamond icon next to property name
3. Keyframe added at current playhead position

### Method 2: Value Change
1. Move playhead to desired time
2. Change property value in Clip Properties panel
3. Keyframe auto-created if animation enabled

### Method 3: Context Menu
1. Right-click on property row
2. Select "Add Keyframe"

---

## Editing Keyframes

### Moving Keyframes
- **Drag** keyframe diamond horizontally to change time
- **Live preview** updates as you drag
- Movement synced with clip if clip is also being dragged

### Changing Values
1. Position playhead on keyframe
2. Adjust property value in properties panel
3. Keyframe value updates

### Deleting Keyframes
- Select keyframe(s)
- Press `Delete` key
- Or right-click → Delete

### Reset to Default
- **Right-click** on property value → resets to default
- Works on any property slider or value

---

## Interpolation & Easing

### Easing Modes
Access via right-click context menu on keyframe:

| Mode | Icon | Behavior |
|------|------|----------|
| **Linear** | Diamond | Constant rate of change |
| **Ease In** | ◀ pointed | Slow start, fast end |
| **Ease Out** | ▶ pointed | Fast start, slow end |
| **Ease In-Out** | ◆ pointed both | Slow start and end |
| **Hold** | Square | No interpolation (step) |

### Visual Differentiation
- Each easing mode shows unique diamond shape
- Easy to identify animation style at a glance

### Setting Easing
1. Right-click keyframe
2. Select easing mode from context menu
3. Visual indicator updates immediately

---

## Selection

### Single Selection
- Click keyframe diamond

### Marquee Selection
- Drag rectangle in keyframe area
- Selects all keyframes within bounds
- Auto-expands video tracks to show keyframes
- Live feedback during drag

### Multi-Selection Behavior
- Selected keyframes highlighted
- `Delete` removes all selected
- Drag moves all selected together

### Selection Priority
When keyframes are selected:
- `Delete` key removes keyframes (not clips)
- Clip selection preserved during keyframe selection

---

## Keyframe Display

### Expanded Tracks
Click expand arrow on track to show:
- Property groups (Position, Scale, Rotation, Opacity)
- Individual property lanes with keyframe diamonds
- Only properties with keyframes are shown

### Flattened Display
- All keyframes visible in single expanded area
- Clean, organized layout
- Track height adjusts automatically

### Property Groups
```
▼ Position
  ├─ X ────────◆──────◆────────
  ├─ Y ────◆────────────◆──────
  └─ Z ──────────◆─────────────
▼ Scale
  ├─ X ────◆──────────◆────────
  └─ Y ────◆──────────◆────────
▼ Rotation
  └─ Z ────────◆───────────────
▼ Opacity
  └─ ──────◆──────────◆────────
```

### Effects Keyframes
- Effect parameters appear as additional groups
- Same keyframe controls as transform properties
- See [Effects](./Effects.md#keyframes)

---

## Playhead Snapping

When `Shift` is held:
- Playhead snaps to nearest keyframe
- Also snaps to clip start/end points
- Helps precise positioning

---

## Related Features

- [Timeline](./Timeline.md) - Main editing interface
- [Effects](./Effects.md) - Effect parameter keyframes
- [Preview](./Preview.md) - See animated results
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Commits: dca5e09 through d63e381*
