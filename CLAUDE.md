# CLAUDE.md

## Workflow

**Always commit and push after completing changes:**
```bash
git add . && git commit -m "description" && git push
```

**Documentation: FEATURES.md pflegen!**
- Bei jedem Commit mit neuen/geänderten Features: `FEATURES.md` aktualisieren
- Falls ein bestehendes Feature auffällt, das noch nicht dokumentiert ist: hinzufügen
- Das Feature-Handbuch dient als vollständige Referenz aller App-Funktionen

## Quick Start

```bash
npm install && npm run dev   # http://localhost:5173
npm run build                 # Production build
npm run lint                  # ESLint check
```

## Project Structure

```
src/
├── components/
│   ├── timeline/           # Timeline editor components
│   │   ├── Timeline.tsx    # Main orchestrator
│   │   ├── TimelineRuler.tsx
│   │   ├── TimelineTrack.tsx
│   │   ├── TimelineClip.tsx
│   │   ├── TimelineKeyframes.tsx
│   │   ├── TimelineControls.tsx
│   │   └── TimelineHeader.tsx
│   ├── panels/             # Dock panel contents
│   │   ├── MediaPanel.tsx
│   │   ├── LayerPanel.tsx
│   │   ├── EffectsPanel.tsx
│   │   └── ClipPropertiesPanel.tsx
│   ├── preview/            # Preview & overlay
│   │   ├── Preview.tsx
│   │   └── MaskOverlay.tsx
│   ├── export/             # Export functionality
│   │   ├── ExportDialog.tsx
│   │   └── ExportPanel.tsx
│   ├── common/             # Shared components
│   │   └── Toolbar.tsx
│   └── dock/               # Dockable panel system
│
├── stores/
│   ├── timeline/           # Timeline state (slices)
│   │   ├── index.ts        # Main store export
│   │   ├── types.ts        # Timeline types
│   │   ├── constants.ts    # Default values
│   │   ├── utils.ts        # Helper functions
│   │   ├── trackSlice.ts
│   │   ├── clipSlice.ts
│   │   ├── playbackSlice.ts
│   │   ├── keyframeSlice.ts
│   │   └── selectionSlice.ts
│   ├── mediaStore.ts       # Media & video state
│   ├── mixerStore.ts       # Layer, effect, grid state
│   ├── dockStore.ts        # UI layout state
│   └── historyStore.ts     # Undo/redo
│
├── engine/
│   ├── core/               # GPU initialization
│   │   ├── WebGPUContext.ts
│   │   └── types.ts
│   ├── pipeline/           # Render pipelines
│   │   ├── CompositorPipeline.ts
│   │   ├── EffectsPipeline.ts
│   │   └── OutputPipeline.ts
│   ├── texture/            # Texture management
│   │   ├── TextureManager.ts
│   │   ├── MaskTextureManager.ts
│   │   └── ScrubbingCache.ts
│   ├── video/              # Video handling
│   │   ├── VideoFrameManager.ts
│   │   └── WebCodecsPlayer.ts
│   ├── export/
│   │   └── FrameExporter.ts
│   └── WebGPUEngine.ts     # Facade (orchestrates modules)
│
├── shaders/                # WGSL shaders
│   ├── composite.wgsl
│   ├── effects.wgsl
│   └── output.wgsl
│
├── hooks/
│   ├── useEngine.ts
│   ├── useGlobalHistory.ts
│   └── useMIDI.ts
│
└── services/
    ├── proxyGenerator.ts
    ├── projectDB.ts
    ├── fileSystemService.ts
    └── proxyFrameCache.ts
```

## Critical Patterns

### HMR Singleton

Engine must survive hot reloads to prevent "Device mismatch" errors:

```typescript
const hot = import.meta.hot;
if (hot?.data?.engine) {
  engineInstance = hot.data.engine;
} else {
  engineInstance = new WebGPUEngine();
  hot.data.engine = engineInstance;
}
```

### Stale Closure Fix

Always use `get().layers` inside async callbacks:

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

### Video Ready State

Wait for `canplaythrough`, not `loadeddata`:

```typescript
video.addEventListener('canplaythrough', () => {
  // Video ready for texture creation
}, { once: true });
```

### Zustand Slice Pattern

Store slices follow this pattern:

```typescript
export const createSlice: SliceCreator<Actions> = (set, get) => ({
  actionName: (params) => {
    const state = get();
    // ... logic
    set({ /* updates */ });
  },
});
```

## Texture Types

| Source | Texture Type |
|--------|-------------|
| Video (HTMLVideoElement) | `texture_external` (zero-copy) |
| Video (WebCodecs VideoFrame) | `texture_external` (zero-copy) |
| Image | `texture_2d<f32>` (copied once) |

## Common Issues

| Problem | Solution |
|---------|----------|
| 15fps on Linux | Enable Vulkan: `chrome://flags/#enable-vulkan` |
| "Device mismatch" | HMR broke singleton - refresh page |
| Black canvas | Check `readyState >= 2` before texture import |
| WebCodecs fails | Falls back to HTMLVideoElement automatically |

## Render Loop

```
useEngine hook
    └─► engine.start(callback)
            └─► requestAnimationFrame loop
                    └─► engine.render(layers)
                            ├─► Import external textures
                            ├─► Ping-pong composite each layer
                            └─► Output to canvas(es)
```

## Adding Effects

1. Add shader in `src/shaders/effects.wgsl`
2. Add params type and defaults in `src/stores/timeline/utils.ts:getDefaultEffectParams()`
3. Add UI controls in `src/components/panels/EffectsPanel.tsx`

## Debugging

```typescript
// Profile output (automatic, every 1s)
// [PROFILE] FPS=60 | gap=16ms | layers=3 | render=2.50ms

// Check GPU status
chrome://gpu

// Slow frame warnings (automatic)
// [RAF] Very slow frame: rafDelay=150ms
```
