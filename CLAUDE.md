# CLAUDE.md

## Workflow

**Branch-Strategie (Cloudflare Pages):**
- `staging` Branch: Entwicklung (kein Auto-Deploy)
- `master` Branch: Production (Cloudflare baut nur hier)

```bash
# Normal entwickeln auf staging:
git add . && git commit -m "description" && git push origin staging

# Wenn bereit für Production - VERSION ERHÖHEN!
# 1. Version in src/version.ts erhöhen (z.B. 1.0.5 -> 1.0.6)
# 2. Commit & Push auf staging
# 3. Dann merge zu master:
git checkout master && git merge staging && git push origin master
git checkout staging
```

**WICHTIG: Version nur bei MERGE zu master erhöhen!**
- Datei: `src/version.ts`
- Format: `MAJOR.MINOR.PATCH`
- VOR dem Merge zu master: PATCH um 1 erhöhen
- Version wird oben rechts neben "WebGPU Ready" angezeigt

**Documentation: docs/Features/ pflegen!**
- Bei jedem Commit mit neuen/geänderten Features: `docs/Features/` aktualisieren
- Falls ein bestehendes Feature auffällt, das noch nicht dokumentiert ist: hinzufügen
- Das Feature-Handbuch dient als vollständige Referenz aller App-Funktionen
- Main index: `docs/Features/README.md`

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
│   │   ├── TimelineHeader.tsx
│   │   ├── CurveEditor.tsx     # Bezier keyframe editor
│   │   ├── MulticamDialog.tsx  # Multicam setup dialog
│   │   ├── PickWhip.tsx        # Expression linking
│   │   └── ParentChildLink.tsx # Layer parenting
│   ├── panels/             # Dock panel contents
│   │   ├── PropertiesPanel.tsx  # Unified: Transform, Effects, Masks, Volume
│   │   ├── MediaPanel.tsx
│   │   ├── AIChatPanel.tsx
│   │   ├── MultiCamPanel.tsx
│   │   ├── TranscriptPanel.tsx
│   │   ├── AnalysisPanel.tsx
│   │   ├── LayerPanel.tsx
│   │   ├── EffectsPanel.tsx     # Legacy, use PropertiesPanel
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
│   │   ├── selectionSlice.ts
│   │   └── maskSlice.ts    # Mask shapes and vertices
│   ├── mediaStore.ts       # Media & video state
│   ├── mixerStore.ts       # Layer, effect, grid state
│   ├── multicamStore.ts    # Multicam sources and sync
│   ├── settingsStore.ts    # App settings and preferences
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
│   ├── composite.wgsl      # Layer compositing, 37 blend modes
│   ├── effects.wgsl        # GPU effects
│   ├── opticalflow.wgsl    # Motion analysis
│   └── output.wgsl         # Final output
│
├── hooks/
│   ├── useEngine.ts
│   ├── useGlobalHistory.ts
│   ├── useMIDI.ts
│   ├── useClipPanelSync.ts     # Auto-switch panels on clip select
│   └── useContextMenuPosition.ts
│
└── services/
    ├── projectDB.ts            # IndexedDB persistence
    ├── fileSystemService.ts    # File System Access API
    ├── proxyGenerator.ts       # GPU proxy generation
    ├── proxyFrameCache.ts      # Proxy frame caching
    ├── audioManager.ts         # Web Audio API, 10-band EQ
    ├── audioSync.ts            # Cross-correlation sync
    ├── audioAnalyzer.ts        # Audio level analysis
    ├── aiTools.ts              # 50+ AI editing tools
    ├── claudeService.ts        # Claude API integration
    ├── whisperService.ts       # Transcription providers
    ├── clipTranscriber.ts      # Clip transcription
    ├── transcriptSync.ts       # Transcript synchronization
    ├── clipAnalyzer.ts         # Clip analysis
    ├── multicamAnalyzer.ts     # Multicam analysis
    ├── compositionRenderer.ts  # Nested composition rendering
    └── apiKeyManager.ts        # API key storage
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
3. Add UI controls in `src/components/panels/PropertiesPanel.tsx` (Effects tab)

## Debugging

```typescript
// Profile output (automatic, every 1s)
// [PROFILE] FPS=60 | gap=16ms | layers=3 | render=2.50ms

// Check GPU status
chrome://gpu

// Slow frame warnings (automatic)
// [RAF] Very slow frame: rafDelay=150ms
```

## Panel System

8 dockable panel types:
- **Preview** - Composition canvas
- **Timeline** - Multi-track editor
- **Media** - Media browser
- **Properties** - Unified clip editing (Transform, Effects, Masks, Volume)
- **Export** - Render settings
- **Multicam** - Camera sync
- **AI Chat** - GPT assistant
- **Slots** - Layer management

## Key Features

- **WebGPU** rendering with 60fps compositing
- **37 blend modes** (all After Effects modes)
- **9 GPU effects** with keyframe animation
- **10-band EQ** with keyframe support
- **Multicam sync** via audio cross-correlation
- **AI tools** (50+) via OpenAI function calling
- **4 transcription providers** (Local Whisper, OpenAI, AssemblyAI, Deepgram)
- **H.264/VP9 export** via WebCodecs
