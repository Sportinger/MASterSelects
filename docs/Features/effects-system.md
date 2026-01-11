# Modular Effects System

The effects system has been refactored to a modular plugin architecture that supports 100+ effects with automatic registration.

## Architecture

```
src/effects/
├── index.ts                    # Registry & auto-discovery
├── types.ts                    # EffectDefinition interface
├── EffectsPipeline.ts          # GPU pipeline orchestrator
├── EffectControls.tsx          # Generic UI renderer
├── _shared/
│   └── common.wgsl             # Shared vertex shader, color helpers
│
├── color/                      # Color correction effects
├── blur/                       # Blur effects
├── distort/                    # Distortion effects
├── stylize/                    # Stylize effects
├── generate/                   # Generator effects
├── keying/                     # Keying effects
├── time/                       # Time-based effects
└── transition/                 # Transition effects
```

## Available Effects (30+)

### Color Correction
| Effect | Description |
|--------|-------------|
| Hue Shift | Rotate hue on color wheel |
| Brightness | Adjust brightness |
| Contrast | Adjust contrast |
| Saturation | Adjust color saturation |
| Levels | Input/output levels with gamma |
| Invert | Invert colors |
| Vibrance | Smart saturation (preserves skin tones) |
| Temperature | Warm/cool color temperature |
| Exposure | EV stops, offset, gamma |

### Blur Effects
| Effect | Description |
|--------|-------------|
| Gaussian Blur | Smooth blur |
| Box Blur | Fast box filter blur |
| Radial Blur | Blur from center point |
| Zoom Blur | Zoom effect blur |
| Motion Blur | Directional blur |

### Distort Effects
| Effect | Description |
|--------|-------------|
| Pixelate | Mosaic pixelation |
| Kaleidoscope | Symmetric patterns |
| Mirror | Horizontal/vertical mirror |
| RGB Split | Chromatic aberration |
| Twirl | Spiral distortion |
| Wave | Sine wave distortion |
| Bulge/Pinch | Bulge or pinch effect |

### Stylize Effects
| Effect | Description |
|--------|-------------|
| Vignette | Corner darkening |
| Film Grain | Animated noise grain |
| Sharpen | Edge sharpening |
| Posterize | Reduce color levels |
| Glow | Bloom/glow effect |
| Edge Detect | Sobel edge detection |
| Scanlines | CRT scanline overlay |
| Threshold | B/W threshold |

### Keying Effects
| Effect | Description |
|--------|-------------|
| Chroma Key | Green/blue screen removal |

## Adding New Effects

Each effect is a self-contained module with:

1. **shader.wgsl** - WGSL shader code
2. **index.ts** - Effect definition with metadata

### Example: Creating a New Effect

```typescript
// src/effects/stylize/my-effect/index.ts
import shader from './shader.wgsl?raw';
import type { EffectDefinition } from '../../types';

export const myEffect: EffectDefinition = {
  id: 'my-effect',
  name: 'My Effect',
  category: 'stylize',

  shader,
  entryPoint: 'myEffectFragment',
  uniformSize: 16,

  params: {
    amount: {
      type: 'number',
      label: 'Amount',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      animatable: true,
    },
  },

  packUniforms: (params, width, height) => {
    return new Float32Array([
      params.amount as number || 0.5,
      width,
      height,
      0, // padding
    ]);
  },
};
```

```wgsl
// src/effects/stylize/my-effect/shader.wgsl
struct MyEffectParams {
  amount: f32,
  width: f32,
  height: f32,
  _pad: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: MyEffectParams;

@fragment
fn myEffectFragment(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTex, texSampler, input.uv);
  // Your effect logic here
  return color;
}
```

### Register the Effect

Add export to category index:

```typescript
// src/effects/stylize/index.ts
export { myEffect } from './my-effect';
```

The effect is automatically registered and appears in the UI.

## Effect Definition Interface

```typescript
interface EffectDefinition {
  id: string;                    // Unique identifier
  name: string;                  // Display name
  category: EffectCategory;      // Category for grouping

  shader: string;                // WGSL code
  entryPoint: string;            // Fragment shader function
  uniformSize: number;           // Bytes (16-aligned)

  params: Record<string, EffectParam>;

  packUniforms: (
    params: Record<string, number | boolean | string>,
    width: number,
    height: number
  ) => Float32Array | null;

  passes?: number;               // Multi-pass effects
  customControls?: React.ComponentType;  // Custom UI
}
```

## Parameter Types

| Type | Description | UI Control |
|------|-------------|------------|
| `number` | Numeric value | Slider |
| `boolean` | On/off toggle | Checkbox |
| `select` | Option list | Dropdown |
| `color` | Color picker | Color input |
| `point` | 2D position | XY controls |

## Shared Shader Utilities

The `_shared/common.wgsl` file provides:

- **Vertex shader** - Fullscreen quad
- **Color conversions** - RGB↔HSV, RGB↔HSL, RGB↔YCbCr
- **Utilities** - luminance(), gaussian(), noise2d()
- **Constants** - PI, TAU, E

## GPU Pipeline

Effects are applied using ping-pong rendering:

1. Input texture → Effect 1 → Ping buffer
2. Ping buffer → Effect 2 → Pong buffer
3. Pong buffer → Effect 3 → Ping buffer
4. ...continue chain...
5. Final buffer → Output

All parameters support keyframe animation.
