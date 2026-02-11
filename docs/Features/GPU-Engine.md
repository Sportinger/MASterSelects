# GPU Engine

[← Back to Index](./README.md)

WebGPU-powered rendering with zero-copy textures and optical flow analysis.

---

## Table of Contents

- [Architecture](#architecture)
- [Texture Management](#texture-management)
- [Compositing Pipeline](#compositing-pipeline)
- [Shader Capabilities](#shader-capabilities)
- [Optical Flow](#optical-flow)
- [Performance](#performance)

---

## Architecture

### Engine Structure
```
WebGPUEngine (Facade)
├── WebGPUContext       # GPU initialization
├── CompositorPipeline  # Layer compositing
├── EffectsPipeline     # Effect processing
├── OutputPipeline      # Final output
├── TextureManager      # Texture handling
├── MaskTextureManager  # Mask generation
├── ScrubbingCache      # Frame caching
└── FrameExporter       # Video export
```

### Initialization
```typescript
// WebGPUContext.ts
- GPU adapter: powerPreference: 'high-performance'
- Device limits: maxTextureDimension2D: 4096
- Canvas: bgra8unorm format, premultiplied alpha
- Sampler: linear filtering, clamp-to-edge
```

### HMR Singleton
```typescript
// Survives hot module reload
if (hot?.data?.engine) {
  engineInstance = hot.data.engine;
} else {
  engineInstance = new WebGPUEngine();
  hot.data.engine = engineInstance;
}
```

---

## Texture Management

### Texture Types
| Source | Type | Copy |
|--------|------|------|
| HTMLVideoElement | `GPUExternalTexture` | Zero-copy |
| VideoFrame (WebCodecs) | `GPUExternalTexture` | Zero-copy |
| HTMLImageElement | `texture_2d<f32>` | Copy once |

### Video Textures
```typescript
// Zero-copy import
device.importExternalTexture({ source: video })

// Requirements
- readyState >= 2 (HAVE_CURRENT_DATA)
- Not seeking
- Fallback to cached frame on failure
```

### Image Textures
```typescript
copyExternalImageToTexture(image, texture)
- LRU cache with eviction
- View caching
- Uses naturalWidth/naturalHeight
```

---

## Compositing Pipeline

### Two Render Pipelines
1. **Standard Composite** - Image textures (`texture_2d<f32>`)
2. **External Composite** - Video textures (`texture_external`)

### Layer Transforms (GPU)
```wgsl
// Uniform structure (80 bytes)
position: vec3<f32>     // X, Y, Z depth
scale: vec2<f32>        // X, Y
rotation: vec3<f32>     // X, Y, Z (radians)
opacity: f32
blendMode: u32          // 0-36
aspectRatio: f32
perspectiveDistance: f32
```

### 3D Rotation
- Full X, Y, Z rotation
- Configurable perspective distance
- Z-depth affects apparent scale

### Ping-Pong Rendering
```
Layer 1 → Composite onto Ping
Layer 2 → Composite onto Pong
Layer 3 → Composite onto Ping
...
Final → Output to Canvas
```

---

## Shader Capabilities

### Total WGSL Code: ~2,400 lines

| File | Lines | Purpose |
|------|-------|---------|
| `composite.wgsl` | 618 | Blending + 37 modes |
| `effects.wgsl` | 243 | Inline GPU effects |
| `opticalflow.wgsl` | 326 | Motion analysis |
| `output.wgsl` | 71 | Passthrough |
| `slice.wgsl` | 33 | Output slice rendering |
| `common.wgsl` | 154 | Shared effect utilities |
| 30 effect shaders | ~954 | Individual effect shaders |

### Blend Mode Implementation
All 37 modes in switch statement:
- HSL/RGB conversion helpers
- Luminosity calculations (BT.601)
- Stencil/silhouette operations

### Effect Shaders
Per-effect entry points with uniform buffers.

---

## Optical Flow

### GPU Motion Detection
```wgsl
// opticalflow.wgsl compute shaders
1. Grayscale conversion (BT.601)
2. Gaussian blur 5x5 (sigma=1.0)
3. Pyramid downsampling (3 levels)
4. Spatial gradients (Ix, Iy)
5. Temporal gradient (It)
6. Lucas-Kanade solver
7. Statistics aggregation
```

### Analysis Resolution
160×90 pixels (fast, sufficient for statistics)

### Motion Metrics
```typescript
interface MotionStats {
  meanMagnitude: number;    // 0-1 normalized
  directionCoherence: number; // Global vs local
  coverageRatio: number;    // Motion density
}
```

### Thresholds
| Detection | Value |
|-----------|-------|
| Motion | 0.5 magnitude |
| Scene cut | 8.0 magnitude + 0.7 coverage |
| Global coherence | 0.6 |

---

## Performance

### Frame Rate Targets
- **Preview**: 60fps target
- **During video playback**: 30fps limit
- **Frame drop detection**: 1.5x target time

### Idle Mode (Disabled)
Idle mode was originally designed to pause the render loop after 1 second of inactivity to save GPU resources. However, it has been **disabled** to ensure scrubbing always works reliably after page reload. The engine now renders continuously.

### Render Loop Watchdog
A watchdog monitors the render loop for crashes and hangs:
- Detects when the render loop stops unexpectedly
- Automatically recovers from GPU-related crashes
- Prevents permanent preview freezes

### Statistics Tracking
```typescript
interface RenderStats {
  fps: number;
  frameGap: number;
  layerCount: number;
  importTime: number;
  renderTime: number;
  submitTime: number;
  dropsThisSecond: number;
  dropsTotal: number;
  isIdle: boolean;      // Engine in power-saving mode
}
```

### Bottleneck Identification
- **Video Import** - Texture upload slow
- **GPU Render** - Compositing slow
- **GPU Submit** - Command submission slow

---

## Video Decoding

### 3 Decoding Modes
```typescript
// WebCodecsPlayer.ts
1. Simple Mode - Direct VideoFrame extraction
2. MP4Box Mode - Demux + WebCodecs decode
3. Stream Mode - MediaStreamTrackProcessor
```

### Fallback Chain
```
WebCodecs → HTMLVideoElement
```

---

## Export Pipeline

### FrameExporter
```typescript
// Frame-by-frame export
1. Seek all clips to time
2. Build layer composition
3. Render via engine.render()
4. Read pixels (staging buffer)
5. Encode VideoFrame
6. Mux to container
```

### Codec Support
| Codec | Container | ID |
|-------|-----------|-----|
| H.264 | MP4 | avc1.640028 |
| VP9 | WebM | vp09.00.10.08 |

### Settings
- Resolution: 480p to 4K
- Frame rate: 24-60 fps
- Bitrate: 5-35 Mbps

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| 15fps on Linux | Enable Vulkan: `chrome://flags/#enable-vulkan` |
| "Device mismatch" | HMR broke singleton - refresh page |
| Black canvas | Check `readyState >= 2` |
| WebCodecs fails | Falls back to HTMLVideoElement |

### GPU Status
```
chrome://gpu
```

---

## Related Features

- [Preview](./Preview.md) - Rendering output
- [Effects](./Effects.md) - Effect pipeline
- [Export](./Export.md) - Export rendering
- [Masks](./Masks.md) - Mask rendering

---

*Source: `src/engine/WebGPUEngine.ts`, `src/engine/pipeline/`, `src/shaders/`*
