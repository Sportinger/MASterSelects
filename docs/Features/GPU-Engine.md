# GPU Engine

[← Back to Index](./README.md)

WebGPU-powered rendering engine for hardware-accelerated compositing.

---

## Table of Contents

- [Architecture](#architecture)
- [Render Pipeline](#render-pipeline)
- [Texture Management](#texture-management)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Architecture

### Engine Structure
```
WebGPUEngine (Facade)
├── WebGPUContext      # GPU initialization
├── CompositorPipeline # Layer compositing
├── EffectsPipeline    # Effect processing
├── OutputPipeline     # Final output
├── TextureManager     # Texture handling
├── MaskTextureManager # Mask generation
└── ScrubbingCache     # Frame caching
```

### Singleton Pattern
Engine survives hot module reload (HMR):
```typescript
if (hot?.data?.engine) {
  engineInstance = hot.data.engine;
} else {
  engineInstance = new WebGPUEngine();
  hot.data.engine = engineInstance;
}
```
Prevents "Device mismatch" errors during development.

---

## Render Pipeline

### Render Loop
```
useEngine hook
    └─► engine.start(callback)
            └─► requestAnimationFrame loop
                    └─► engine.render(layers)
                            ├─► Import external textures
                            ├─► Ping-pong composite each layer
                            └─► Output to canvas(es)
```

### Compositing
1. Layers processed front-to-back
2. Each layer blended onto composite
3. Ping-pong buffers for intermediate results
4. Final output to canvas

### Texture Types
| Source | Texture Type |
|--------|-------------|
| Video (HTMLVideoElement) | `texture_external` (zero-copy) |
| Video (WebCodecs VideoFrame) | `texture_external` (zero-copy) |
| Image | `texture_2d<f32>` (copied once) |

---

## Texture Management

### External Textures
- Video frames imported as external textures
- Zero-copy GPU operation
- Re-imported each frame

### Image Textures
- Copied to GPU once
- Bind groups recreated each frame
- Uses `naturalWidth`/`naturalHeight`

### Mask Textures
- Generated from vector shapes
- Feather blur applied on GPU
- Used as alpha channel

### Frame Cache
- GPU texture cache for RAM Preview
- Instant scrubbing after caching
- Cache key includes layer state

---

## Performance

### Target Performance
- 60fps real-time preview
- Efficient GPU utilization
- Minimal CPU overhead

### Optimizations

#### Layer Updates
- Check if layer `needsUpdate`
- Skip unchanged layers
- Includes position.z in update check

#### Video Playback
- `requestVideoFrameCallback` for sync
- Early return fast path
- Rate limited to 30fps during playback

#### Memory Management
- Frame cache limits
- Texture cleanup
- Leak prevention in render loop

### Profiling
Automatic profile output:
```
[PROFILE] FPS=60 | gap=16ms | layers=3 | render=2.50ms
```

### Slow Frame Detection
```
[RAF] Very slow frame: rafDelay=150ms
```

---

## Shaders

### Shader Files
Located in `src/shaders/`:
- `composite.wgsl` - Layer compositing
- `effects.wgsl` - Visual effects
- `output.wgsl` - Final output

### Adding Effects
1. Add shader code in `effects.wgsl`
2. Add params in `utils.ts:getDefaultEffectParams()`
3. Add UI in `EffectsPanel.tsx`

### Shader Inputs
- Input textures
- Transform uniforms
- Effect parameters
- Blend mode

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| 15fps on Linux | Enable Vulkan: `chrome://flags/#enable-vulkan` |
| "Device mismatch" | HMR broke singleton - refresh page |
| Black canvas | Check `readyState >= 2` before texture import |
| WebCodecs fails | Falls back to HTMLVideoElement automatically |

### GPU Status
Check in Chrome:
```
chrome://gpu
```

### Video Ready State
Always wait for `canplaythrough`:
```typescript
video.addEventListener('canplaythrough', () => {
  // Video ready for texture creation
}, { once: true });
```

### Stale Closures
Use fresh state in callbacks:
```typescript
// WRONG - stale closure
const { layers } = get();
video.onload = () => set({ layers: layers.map(...) });

// CORRECT - fresh state
video.onload = () => {
  const current = get().layers;
  set({ layers: current.map(...) });
};
```

---

## Multiple Canvases

### Independent Rendering
Multiple preview panels supported:
- Separate canvas registration
- Independent ping-pong buffers
- Proper cleanup on unmount

### Canvas Management
- Register canvas with engine
- Each canvas renders composition
- Cleanup on panel close

---

## WebCodecs Integration

### Video Decoding
When available, uses WebCodecs:
- Hardware-accelerated decode
- Direct GPU upload
- Falls back to HTMLVideoElement

### Proxy Frames
For large files:
- Proxy frames generated
- Used during preview
- Full-res for export

---

## Related Features

- [Preview](./Preview.md) - Rendering output
- [Effects](./Effects.md) - Effect pipeline
- [Export](./Export.md) - Export rendering
- [Masks](./Masks.md) - Mask rendering

---

*Commits: d6e130c through d63e381*
