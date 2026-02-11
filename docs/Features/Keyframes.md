# Keyframes

[← Back to Index](./README.md)

The keyframe animation system enables property animation over time with bezier curve editing.

---

## Table of Contents

- [Animatable Properties](#animatable-properties)
- [Creating Keyframes](#creating-keyframes)
- [Editing Keyframes](#editing-keyframes)
- [Easing Modes](#easing-modes)
- [Curve Editor](#curve-editor)
- [Recording Mode](#recording-mode)

---

## Animatable Properties

### Transform Properties (9 total)
| Property | Range | Default |
|----------|-------|---------|
| `opacity` | 0-1 | 1 |
| `position.x` | -∞ to +∞ | 0 |
| `position.y` | -∞ to +∞ | 0 |
| `position.z` | -∞ to +∞ | 0 (depth) |
| `scale.x` | 0 to ∞ | 1 |
| `scale.y` | 0 to ∞ | 1 |
| `rotation.x` | degrees | 0 |
| `rotation.y` | degrees | 0 |
| `rotation.z` | degrees | 0 |

### Effect Properties
Any numeric effect parameter can be keyframed:
```
effect.{effectId}.{paramName}
```
Example: `effect.effect_123.shift` for hue shift animation

---

## Creating Keyframes

### Method 1: Property Row Controls
1. Expand track to show properties
2. Click diamond icon (◇) next to property
3. Keyframe added at current playhead

### Method 2: Value Change with Recording
1. Enable recording mode (toggle button)
2. Move playhead to desired time
3. Change property value
4. Keyframe auto-created

### Keyframe Data Structure
```typescript
interface Keyframe {
  id: string;           // kf_{timestamp}_{random}
  clipId: string;       // Reference to clip
  time: number;         // Relative to clip start (seconds)
  property: string;     // e.g., 'opacity', 'position.x'
  value: number;        // Interpolated value
  easing: EasingType;   // Interpolation mode
  handleIn?: BezierHandle;   // Custom in-tangent
  handleOut?: BezierHandle;  // Custom out-tangent
}
```

---

## Editing Keyframes

### Moving Keyframes
- **Drag** keyframe diamond horizontally
- **Shift+drag** for fine control (10x slower)
- Clamped to clip duration [0, clipDuration]
- Live preview updates during drag

### Changing Values
1. Position playhead on keyframe
2. Adjust value in Clip Properties panel
3. Keyframe value updates automatically

### Deleting Keyframes
- Select keyframe(s)
- Press `Delete` key
- Or right-click → Delete

### Copy/Paste Keyframes
- **Copy:** `Ctrl+C` with keyframes selected copies only keyframes (not clips)
- **Paste:** `Ctrl+V` pastes keyframes at playhead position on the selected clip
- Pasted keyframes maintain relative timing between each other

### Multi-Select Movement
- Select multiple keyframes with `Shift+Click`
- Drag any selected keyframe to move all by the same time delta
- All selected keyframes move together maintaining relative spacing

### Keyframe Toggle Off
- Toggling keyframes off for a property saves the current value
- All keyframes for that property are deleted cleanly
- Property reverts to a static value

### Tick Marks on Clips
- Small amber diamond markers at the bottom of clip bars
- Show keyframe positions without needing to expand tracks
- Visible at all zoom levels for quick keyframe overview

### Batch Operations
```typescript
addKeyframe(clipId, property, value, time?, easing)
removeKeyframe(keyframeId)
updateKeyframe(keyframeId, updates)
moveKeyframe(keyframeId, newTime)
deleteSelectedKeyframes()
```

---

## Easing Modes

### Available Modes (5 total)

| Mode | Bezier Points | Behavior |
|------|---------------|----------|
| `linear` | [0,0] → [1,1] | Constant rate |
| `ease-in` | [0.42,0] → [1,1] | Slow start |
| `ease-out` | [0,0] → [0.58,1] | Slow end |
| `ease-in-out` | [0.42,0] → [0.58,1] | Smooth both |
| `bezier` | Custom handles | User-defined |

### Visual Indicators
Each easing mode shows unique diamond shape:
- Linear: ◇ regular diamond
- Ease In: ◀ left-pointed
- Ease Out: ▶ right-pointed
- Ease In-Out: ◆ filled
- Bezier: custom shape

### Setting Easing
1. Right-click keyframe
2. Select easing from context menu
3. Or modify bezier handles in curve editor

---

## Curve Editor

### Opening the Curve Editor
1. Expand track to show properties
2. Click curve icon next to property
3. Editor appears below property row

### Features
- **SVG-based** with grid background
- **Bezier curves** drawn between keyframes
- **Value range** auto-computed with padding
- **Auto-scale Y-axis** fits curve tightly to visible range
- **Shift+wheel** to resize curve editor height
- **Single editor open** — only one curve editor at a time to prevent UI clutter

### Keyframe Manipulation
| Action | Effect |
|--------|--------|
| Click+drag point | Move time and value |
| Shift+drag | Constrain to horizontal or vertical |
| Click empty | Deselect all |

### Bezier Handle Editing
- In-handle: controls incoming curve (x ≤ 0)
- Out-handle: controls outgoing curve (x ≥ 0)
- Shift+drag handle: constrain to horizontal

### Grid System
- Horizontal: time axis (from timeline scroll)
- Vertical: value axis (auto-scaled)
- Major/minor grid lines with labels

---

## Recording Mode

### Enabling Recording
```typescript
toggleKeyframeRecording(clipId, property)
```
- Format: `{clipId}:{property}` in Set
- Visual indicator when active

### Behavior When Recording
- Property changes create/update keyframes at playhead
- Existing keyframe at time → updates value
- No keyframe at time → creates new one

### Without Recording
- Property changes update static clip values
- No keyframes created automatically

---

## Interpolation Algorithm

### Between Keyframes
1. Calculate normalized time `t` between keyframes
2. Apply easing function to get eased time
3. Linear interpolate value: `v1 + (v2 - v1) * easedT`

### Bezier Interpolation
Uses cubic Bezier with Newton-Raphson solver:
- 10 iterations
- Epsilon: 0.0001
- Solves for X to get eased time

### Edge Cases
- No keyframes → returns default value
- Single keyframe → returns its value
- Before first → returns first value
- After last → returns last value

---

## Constants

```typescript
PROPERTY_ROW_HEIGHT = 18px
CURVE_EDITOR_HEIGHT = 250px
BEZIER_HANDLE_SIZE = 8px
KEYFRAME_TOLERANCE = 0.01s (10ms)
```

---

## Track Expansion

### Expanded Track Shows
- Property groups (Position, Scale, Rotation, Opacity)
- Individual property lanes with diamonds
- Only properties with keyframes displayed

### Height Calculation
```
baseHeight
+ (propertyCount × PROPERTY_ROW_HEIGHT)
+ (expandedCurves × CURVE_EDITOR_HEIGHT)
```

---

## Related Features

- [Timeline](./Timeline.md) - Main editing interface
- [Effects](./Effects.md) - Effect parameter keyframes
- [Preview](./Preview.md) - See animated results
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| [`keyframeSlice.test.ts`](../../tests/stores/timeline/keyframeSlice.test.ts) | 14 | Keyframe CRUD operations |
| [`keyframeInterpolation.test.ts`](../../tests/unit/keyframeInterpolation.test.ts) | 52 | Easing, bezier, interpolation |

Run tests: `npx vitest run`

---

*Source: `src/stores/timeline/keyframeSlice.ts`, `src/utils/keyframeInterpolation.ts`, `src/components/timeline/CurveEditor.tsx`*
