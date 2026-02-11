# Effects

[← Back to Index](./README.md)

GPU-accelerated visual effects with 37 blend modes and 30 shader effects.

---

## Table of Contents

- [Transform Properties](#transform-properties)
- [Blend Modes](#blend-modes)
- [GPU Effects](#gpu-effects)
- [Effect Keyframes](#effect-keyframes)

---

## Transform Properties

Every clip has built-in transform properties (all keyframeable):

### Position
| Property | Range | Default |
|----------|-------|---------|
| X | -∞ to +∞ | 0 |
| Y | -∞ to +∞ | 0 |
| Z | -∞ to +∞ | 0 (depth) |

Position Z enables 3D layer positioning with perspective.

### Scale
| Property | Range | Default |
|----------|-------|---------|
| X | 0% to ∞ | 100% |
| Y | 0% to ∞ | 100% |

### Rotation (3D)
Full 3D rotation with configurable perspective:
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
- **Right-click** any property value → resets to default

---

## Blend Modes

### 37 Blend Modes (All Implemented)

#### Normal (3 modes)
| Mode | Description |
|------|-------------|
| `normal` | Standard alpha blending |
| `dissolve` | Random pixel dithering based on opacity |
| `dancing-dissolve` | Animated dithering with time variation |

#### Darken (6 modes)
| Mode | Description |
|------|-------------|
| `darken` | Minimum of base/blend |
| `multiply` | Multiply RGB channels |
| `color-burn` | Color burn intensity |
| `classic-color-burn` | Alternative formula |
| `linear-burn` | `max(base + blend - 1, 0)` |
| `darker-color` | Picks darker by luminosity |

#### Lighten (7 modes)
| Mode | Description |
|------|-------------|
| `add` | Clamped addition (linear dodge) |
| `lighten` | Maximum of base/blend |
| `screen` | `1 - (1-base)*(1-blend)` |
| `color-dodge` | Color dodge intensity |
| `classic-color-dodge` | Alternative formula |
| `linear-dodge` | Same as add |
| `lighter-color` | Picks lighter by luminosity |

#### Contrast (7 modes)
| Mode | Description |
|------|-------------|
| `overlay` | Conditional multiply/screen |
| `soft-light` | Softer version of overlay |
| `hard-light` | Based on blend value |
| `linear-light` | `clamp(base + 2*blend - 1, 0, 1)` |
| `vivid-light` | Conditional dodge/burn |
| `pin-light` | Min/max blend |
| `hard-mix` | Binary threshold (0 or 1) |

#### Inversion (5 modes)
| Mode | Description |
|------|-------------|
| `difference` | Absolute difference |
| `classic-difference` | Same as difference |
| `exclusion` | `base + blend - 2*base*blend` |
| `subtract` | `max(base - blend, 0)` |
| `divide` | `base / max(blend, 0.001)` |

#### Component (4 modes)
| Mode | Description |
|------|-------------|
| `hue` | Blend hue with base sat/lum |
| `saturation` | Blend saturation with base hue/lum |
| `color` | Blend hue/sat with base lum |
| `luminosity` | Blend lum with base hue/sat |

#### Stencil (5 modes)
| Mode | Description |
|------|-------------|
| `stencil-alpha` | Layer alpha as opacity |
| `stencil-luma` | Layer luminosity as opacity |
| `silhouette-alpha` | Inverted layer alpha |
| `silhouette-luma` | Inverted layer luminosity |
| `alpha-add` | Additive alpha blending |

### Blend Mode Cycling
- `Shift` + `+` cycles forward through modes
- `Shift` + `-` cycles backward

---

## GPU Effects

### 30+ Modular Effects

Effects are organized by category in `src/effects/`:

#### Color Correction
| Effect | Parameters |
|--------|------------|
| Brightness | amount (-1 to +1) |
| Contrast | amount (0-3) |
| Saturation | amount (0-3) |
| Vibrance | amount (0-2) |
| Hue Shift | shift (0-1) |
| Temperature | amount (-1 to +1) |
| Exposure | amount (-3 to +3) |
| Levels | inputBlack, inputWhite, gamma, outputBlack, outputWhite |
| Invert | (no params) |

#### Blur Effects
| Effect | Parameters |
|--------|------------|
| Box Blur | radius (0-20) |
| Gaussian Blur | radius (0-50), **quality** (1-3) |
| Motion Blur | amount, angle, **quality** (1-3) |
| Radial Blur | amount, centerX, centerY, **quality** (1-3) |
| Zoom Blur | amount, centerX, centerY, **quality** (1-3) |

#### Distort Effects
| Effect | Parameters |
|--------|------------|
| Pixelate | size (1-64) |
| Kaleidoscope | segments, rotation |
| Mirror | horizontal, vertical |
| RGB Split | amount, angle |
| Twirl | amount, radius, centerX, centerY |
| Wave | amplitude, frequency, speed |
| Bulge | amount, radius, centerX, centerY |

#### Stylize Effects
| Effect | Parameters |
|--------|------------|
| Vignette | amount, size, softness, roundness |
| Grain | amount, size |
| Glow | amount, threshold, radius, softness, **quality** (1-3) |
| Posterize | levels |
| Edge Detect | threshold |
| Scanlines | count, intensity |
| Threshold | level |
| Sharpen | amount, radius |

#### Keying
| Effect | Parameters |
|--------|------------|
| Chroma Key | keyColor, tolerance, softness |

### Effect Controls

#### Bypass Toggle
- Click the **checkmark icon** left of the effect name to toggle effect on/off
- Bypassed effects show as semi-transparent
- Useful for A/B comparisons without removing effects

#### Draggable Values
- **Drag** on any numeric value to adjust it
- **Shift+Drag** for 10x slower precision
- **Ctrl+Drag** for 100x slower precision
- **Right-click** on any value to reset to default

#### Quality Section
Multi-sample effects (blur, glow) have a collapsible **Quality** section with direct control:

| Effect | Quality Parameters |
|--------|-------------------|
| Gaussian Blur | `samples` (1-64, default 5) |
| Zoom Blur | `samples` (4-256, default 16) |
| Motion Blur | `samples` (4-128, default 24) |
| Radial Blur | `samples` (4-256, default 32) |
| Glow | `rings` (1-32, default 4), `samplesPerRing` (4-64, default 16) |

- Click "Quality" header to expand/collapse
- **No upper limit** when dragging values (can go beyond slider max)
- "Reset" button restores defaults
- Warning shown about potential slowdowns

#### Performance Protection
The app monitors render times and **automatically resets quality parameters** to defaults when:
- Frame time exceeds 100ms (below 10fps)
- 5 consecutive slow frames detected

This prevents the app from becoming unresponsive when quality values are set too high.

### Adding Effects
1. Select clip
2. Open Properties Panel → Effects tab
3. Choose effect from dropdown (grouped by category)
4. Adjust parameters with sliders or drag on values

### Inline Effects (Composite Shader)
The following effects run directly inside the composite shader with **no extra render passes**:
- **Brightness**
- **Contrast**
- **Saturation**
- **Invert**

These are optimized for zero overhead since they're applied during compositing.

### Effect Order
All other effects process top-to-bottom (ping-pong rendering).

---

## Effect Keyframes

### Keyframing Effect Parameters
Each numeric effect parameter can be animated:

```typescript
// Property format
`effect.{effectId}.{paramName}`

// Example: animate hue shift
addKeyframe(clipId, 'effect.effect_xyz.shift', 0, 0);    // Start
addKeyframe(clipId, 'effect.effect_xyz.shift', 1, 2);    // End at 2s
```

### Interpolated Effects
```typescript
getInterpolatedEffects(clipId, time)
```
Returns effect parameters with interpolated values at specific time.

### Timeline Display
- Effect keyframes appear below transform properties
- Grouped by effect name
- Only parameters with keyframes shown

---

## GPU Pipeline

### Effects Pipeline Architecture
```
Input Texture
    ↓
Effect 1 → Ping Buffer
    ↓
Effect 2 → Pong Buffer
    ↓
...
    ↓
Final Output
```

### Per-Effect Resources
- Shader module (from `effects.wgsl`)
- Bind group layout
- Render pipeline
- Uniform buffer (16-32 bytes)

---

## Related Features

- [Masks](./Masks.md) - Shape-based masking
- [Keyframes](./Keyframes.md) - Animation system
- [GPU Engine](./GPU-Engine.md) - Rendering pipeline
- [Keyboard Shortcuts](./Keyboard-Shortcuts.md)

---

## Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| [`effectsRegistry.test.ts`](../../tests/unit/effectsRegistry.test.ts) | 32 | Registry, parameters, categories |
| [`typeHelpers.test.ts`](../../tests/unit/typeHelpers.test.ts) | 10 | Effect property parsing |

Run tests: `npx vitest run`

---

*Source: `src/shaders/effects.wgsl` (243 lines), `src/shaders/composite.wgsl` (743 lines), `src/components/panels/EffectsPanel.tsx`*
