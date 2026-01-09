# Effects

[← Back to Index](./README.md)

GPU-accelerated visual effects including blend modes, transforms, and custom shaders.

---

## Table of Contents

- [Transform Effects](#transform-effects)
- [Blend Modes](#blend-modes)
- [GPU Effects](#gpu-effects)
- [Effect Keyframes](#effect-keyframes)
- [Effects Panel](#effects-panel)

---

## Transform Effects

Every clip has built-in transform properties:

### Position
| Property | Range | Default |
|----------|-------|---------|
| X | -∞ to +∞ | 0 |
| Y | -∞ to +∞ | 0 |
| Z | -∞ to +∞ | 0 (depth) |

Position Z enables 3D layer positioning - layers with higher Z appear closer.

### Scale
| Property | Range | Default |
|----------|-------|---------|
| X | 0% to ∞ | 100% |
| Y | 0% to ∞ | 100% |

### Rotation
Full 3D rotation with perspective:
| Property | Range | Default |
|----------|-------|---------|
| X | -180° to 180° | 0° |
| Y | -180° to 180° | 0° |
| Z | -180° to 180° | 0° |

### Opacity
| Property | Range | Default |
|----------|-------|---------|
| Opacity | 0% to 100% | 100% |

### Reset to Default
- **Right-click** any property value to reset to default
- Works on sliders and input fields

---

## Blend Modes

37 After Effects-style blend modes available:

### Normal Modes
- **Normal** - Standard compositing
- **Dissolve** - Random pixel dissolve

### Darken Modes
- Darken
- Multiply
- Color Burn
- Linear Burn
- Darker Color

### Lighten Modes
- Lighten
- Screen
- Color Dodge
- Linear Dodge (Add)
- Lighter Color

### Contrast Modes
- Overlay
- Soft Light
- Hard Light
- Vivid Light
- Linear Light
- Pin Light
- Hard Mix

### Inversion Modes
- Difference
- Exclusion
- Subtract
- Divide

### Component Modes
- Hue
- Saturation
- Color
- Luminosity

### Changing Blend Mode
1. Select clip
2. Open Clip Properties panel
3. Choose blend mode from dropdown

### Blend Mode Cycling
- `Shift` + `+` cycles forward through modes
- `Shift` + `-` cycles backward
- Quick way to preview different modes

---

## GPU Effects

Custom shader-based effects rendered on GPU.

### Available Effects

Effects are defined in WGSL shaders and controlled via parameters.

### Adding Effects
1. Select clip
2. Open Effects Panel
3. Click effect to add

### Effect Parameters
Each effect has adjustable parameters:
- Sliders for numeric values
- Real-time preview updates
- Right-click to reset to default

### Effect Order
Effects process in order listed:
1. First effect applied to original
2. Each subsequent effect applied to result
3. Drag to reorder (if supported)

---

## Effect Keyframes

Animate any effect parameter over time.

### Adding Effect Keyframes
1. Expand track to show effect properties
2. Move playhead to desired time
3. Click diamond icon next to parameter
4. Keyframe added at current time

### Interpolation
- Same easing modes as transform keyframes
- Linear, Ease In, Ease Out, Ease In-Out, Hold
- Right-click keyframe for easing menu

### Timeline Display
- Effect keyframes appear below transform properties
- Grouped by effect name
- Only parameters with keyframes shown

### Example: Animated Blur
```
Time 0s: Blur = 0
Time 1s: Blur = 50  (keyframe with Ease In)
Time 2s: Blur = 0   (keyframe with Ease Out)
```
Creates smooth blur in/out animation.

---

## Effects Panel

### Panel Location
- Accessible from View menu
- Or dock panel tabs

### Panel Contents
- List of available effects
- Click to add to selected clip
- Shows current effects on clip

### Modifying Effects
- Adjust sliders for parameters
- Changes apply in real-time
- Works during playback

---

## Effect Development

### Adding Custom Effects
1. Create shader in `src/shaders/effects.wgsl`
2. Add params type in `src/stores/timeline/utils.ts`
3. Add UI controls in `EffectsPanel.tsx`

### Shader Structure
```wgsl
// Effect shader receives:
// - Input texture
// - Effect parameters as uniforms
// - Outputs processed texture
```

---

## Related Features

- [Masks](./Masks.md) - Shape-based masking
- [Keyframes](./Keyframes.md) - Animation system
- [GPU Engine](./GPU-Engine.md) - Rendering pipeline
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

*Commits: 388428a through d63e381*
