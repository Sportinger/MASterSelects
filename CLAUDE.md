# CLAUDE.md

## Workflow

**Branch-Strategie (Cloudflare Pages):**
- `staging` Branch: Entwicklung (kein Auto-Deploy)
- `master` Branch: Production (Cloudflare baut nur hier)

**WICHTIG - Commit-Regeln:**
- **IMMER auf `staging` committen** - niemals direkt auf master
- **Nach JEDER Änderung committen und pushen** - nicht mehrere Änderungen sammeln
- **NUR PR/Merge zu master wenn User es explizit verlangt!**
- Nicht selbstständig zu master mergen

```bash
# Nach JEDER Änderung sofort committen und pushen:
git add . && git commit -m "description" && git push origin staging

# NUR wenn User "merge zu master" oder "PR zu master" sagt:
# 1. Version in src/version.ts erhöhen (z.B. 1.0.5 -> 1.0.6)
# 2. CHANGELOG in src/version.ts aktualisieren (neuer Eintrag mit allen Änderungen)
# 3. KNOWN_ISSUES in src/version.ts prüfen und aktualisieren
# 4. Commit & Push auf staging
# 5. PR erstellen und mergen:
gh pr create --base master --head staging --title "..." --body "..."
gh pr merge --merge
# 6. Staging mit master synchronisieren:
git fetch origin && git merge origin/master && git push origin staging
```

**Version nur bei MERGE zu master erhöhen!**
- Datei: `src/version.ts`
- Format: `MAJOR.MINOR.PATCH`
- VOR dem Merge zu master: PATCH um 1 erhöhen
- Version wird oben rechts neben "WebGPU Ready" angezeigt

**What's New Dialog aktualisieren bei MERGE zu master!**
- Datei: `src/version.ts`
- `CHANGELOG` Array: Neuen Eintrag am ANFANG hinzufügen mit:
  - `version`: Die neue Version (z.B. '1.0.8')
  - `date`: Aktuelles Datum (z.B. '2026-01-15')
  - `changes`: Array mit allen Änderungen seit letztem Release
    - `type`: 'new' (grün), 'fix' (blau), oder 'improve' (orange)
    - `description`: Kurze Beschreibung der Änderung
- `KNOWN_ISSUES` Array: Aktuelle Bugs/Einschränkungen pflegen
  - Behobene Issues entfernen
  - Neue bekannte Probleme hinzufügen

```typescript
// Beispiel für neuen CHANGELOG Eintrag:
export const CHANGELOG: ChangelogEntry[] = [
  {
    version: '1.0.8',
    date: '2026-01-15',
    changes: [
      { type: 'new', description: 'Neue Feature Beschreibung' },
      { type: 'fix', description: 'Behobener Bug' },
      { type: 'improve', description: 'Verbesserung' },
    ],
  },
  // ... ältere Einträge
];

export const KNOWN_ISSUES: string[] = [
  'Aktuelles Problem 1',
  'Aktuelles Problem 2',
];
```

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

## Native Helpers

Native helpers for video decoding and YouTube downloads. Located in `tools/helpers/`.

**Platform-specific:**
- `tools/helpers/win/` - Windows: YouTube download only (no FFmpeg needed)
- `tools/helpers/linux/` - Linux: Full FFmpeg decoder
- `tools/helpers/mac/` - macOS: Full FFmpeg decoder

```bash
# Windows (YouTube downloads):
cd tools/helpers/win && cargo run --release

# Linux (FFmpeg decoding):
cd tools/helpers/linux && cargo run --release

# Linux with FFmpeg 8.0+ (Arch Linux - use ffmpeg4.4):
cd tools/helpers/linux && \
FFMPEG_INCLUDE_DIR=/usr/include/ffmpeg4.4 \
FFMPEG_LIB_DIR=/usr/lib/ffmpeg4.4 \
PKG_CONFIG_PATH=/usr/lib/ffmpeg4.4/pkgconfig \
cargo run --release

# macOS (FFmpeg decoding):
cd tools/helpers/mac && cargo run --release
```

**Electron Helper (YouTube tray app):**
```bash
cd tools/native-helper && npm install && npm start
```

**Ports:**
- WebSocket: `ws://127.0.0.1:9876`
- HTTP File Server: `http://127.0.0.1:9877`

## Project Structure

```
src/
├── components/
│   ├── timeline/           # Timeline editor components
│   │   ├── Timeline.tsx    # Main orchestrator (1323 LOC after refactor)
│   │   ├── hooks/          # Extracted hooks
│   │   │   ├── useTimelineKeyboard.ts
│   │   │   ├── useTimelineZoom.ts
│   │   │   ├── usePlayheadDrag.ts
│   │   │   ├── useMarqueeSelection.ts
│   │   │   ├── useClipTrim.ts
│   │   │   ├── useClipDrag.ts
│   │   │   └── useLayerSync.ts
│   │   ├── components/     # Sub-components
│   │   │   └── TimelineContextMenu.tsx
│   │   ├── utils/          # Timeline utilities
│   │   ├── TimelineRuler.tsx
│   │   ├── TimelineTrack.tsx
│   │   ├── TimelineClip.tsx
│   │   ├── TimelineKeyframes.tsx
│   │   ├── TimelineControls.tsx
│   │   ├── TimelineHeader.tsx
│   │   ├── TimelineNavigator.tsx  # Bottom scrollbar with zoom handles
│   │   ├── CurveEditor.tsx        # Bezier keyframe editor
│   │   ├── MulticamDialog.tsx
│   │   ├── PickWhip.tsx           # Layer parenting UI
│   │   └── ParentChildLink.tsx    # Physics-based cable animation
│   ├── panels/             # Dock panel contents
│   │   ├── PropertiesPanel.tsx    # Unified: Transform, Effects, Masks, Volume, Transcript, Analysis
│   │   ├── MediaPanel.tsx         # Columns, folders, drag-drop
│   │   ├── AIChatPanel.tsx        # GPT-4/5 function calling
│   │   ├── AIVideoPanel.tsx       # PiAPI integration
│   │   ├── YouTubePanel.tsx       # Search & download
│   │   ├── MultiCamPanel.tsx
│   │   └── LayerPanel.tsx
│   ├── preview/            # Preview & overlay
│   │   ├── Preview.tsx     # Quality dropdown, transparency grid
│   │   └── MaskOverlay.tsx
│   ├── export/             # Export functionality
│   │   ├── ExportDialog.tsx
│   │   └── ExportPanel.tsx  # Fast/Precise/FFmpeg modes
│   ├── mobile/             # Mobile-specific components
│   │   └── MobileWarningOverlay.tsx
│   ├── common/             # Shared components
│   │   └── Toolbar.tsx     # Menu bar
│   └── dock/               # Dockable panel system
│
├── stores/
│   ├── timeline/           # Timeline state (slices)
│   │   ├── index.ts
│   │   ├── types.ts
│   │   ├── constants.ts
│   │   ├── utils.ts
│   │   ├── trackSlice.ts
│   │   ├── clipSlice.ts
│   │   ├── playbackSlice.ts
│   │   ├── keyframeSlice.ts
│   │   ├── selectionSlice.ts
│   │   ├── maskSlice.ts
│   │   ├── clip/           # Modular clip operations
│   │   │   ├── addVideoClip.ts
│   │   │   ├── addAudioClip.ts
│   │   │   ├── addImageClip.ts
│   │   │   ├── addCompClip.ts
│   │   │   └── completeDownload.ts
│   │   └── helpers/        # Utility functions
│   │       ├── mediaTypeHelpers.ts
│   │       ├── thumbnailHelpers.ts
│   │       ├── waveformHelpers.ts
│   │       ├── audioTrackHelpers.ts
│   │       ├── webCodecsHelpers.ts
│   │       ├── clipStateHelpers.ts
│   │       ├── idGenerator.ts
│   │       └── blobUrlManager.ts
│   ├── mediaStore/         # Media state (modular slices)
│   │   ├── index.ts
│   │   ├── types.ts
│   │   ├── constants.ts
│   │   ├── init.ts
│   │   ├── slices/
│   │   │   ├── fileImportSlice.ts
│   │   │   ├── fileManageSlice.ts
│   │   │   ├── folderSlice.ts
│   │   │   ├── selectionSlice.ts
│   │   │   ├── compositionSlice.ts
│   │   │   ├── projectSlice.ts
│   │   │   └── proxySlice.ts
│   │   └── helpers/
│   │       ├── fileHashHelpers.ts
│   │       ├── mediaInfoHelpers.ts
│   │       ├── thumbnailHelpers.ts
│   │       └── importPipeline.ts
│   ├── mixerStore.ts       # Layer, effect, grid state
│   ├── multicamStore.ts    # Multicam sources and sync
│   ├── youtubeStore.ts     # YouTube search and downloads
│   ├── settingsStore.ts    # App settings and preferences
│   ├── dockStore.ts        # UI layout state
│   └── historyStore.ts     # Undo/redo
│
├── engine/
│   ├── WebGPUEngine.ts     # Facade (orchestrates modules)
│   ├── WebCodecsPlayer.ts  # Video decoding
│   ├── ParallelDecodeManager.ts  # Multi-clip parallel decode for export
│   ├── core/               # GPU initialization
│   │   ├── WebGPUContext.ts
│   │   ├── RenderTargetManager.ts
│   │   └── types.ts
│   ├── render/             # Rendering
│   │   ├── RenderLoop.ts
│   │   ├── Compositor.ts
│   │   ├── LayerCollector.ts
│   │   └── NestedCompRenderer.ts
│   ├── pipeline/           # GPU pipelines
│   │   ├── CompositorPipeline.ts
│   │   ├── EffectsPipeline.ts
│   │   └── OutputPipeline.ts
│   ├── texture/            # Texture management
│   │   ├── TextureManager.ts
│   │   ├── MaskTextureManager.ts
│   │   └── ScrubbingCache.ts
│   ├── video/              # Video handling
│   │   └── VideoFrameManager.ts
│   ├── export/             # Export system (modular)
│   │   ├── FrameExporter.ts
│   │   ├── ClipPreparation.ts
│   │   ├── ExportLayerBuilder.ts
│   │   ├── VideoEncoderWrapper.ts
│   │   ├── VideoSeeker.ts
│   │   ├── codecHelpers.ts
│   │   └── types.ts
│   ├── audio/              # Audio processing
│   │   ├── AudioEncoder.ts
│   │   ├── AudioExportPipeline.ts
│   │   ├── AudioExtractor.ts
│   │   ├── AudioMixer.ts
│   │   ├── AudioEffectRenderer.ts
│   │   └── TimeStretchProcessor.ts
│   ├── ffmpeg/             # FFmpeg WASM
│   │   ├── FFmpegBridge.ts
│   │   ├── codecs.ts
│   │   └── types.ts
│   ├── proxy/              # Proxy generation
│   │   └── ProxyResizePipeline.ts
│   ├── analysis/           # Video analysis
│   │   └── OpticalFlowAnalyzer.ts
│   ├── stats/              # Performance monitoring
│   │   └── PerformanceStats.ts
│   └── managers/           # Output management
│       └── OutputWindowManager.ts
│
├── effects/                # Modular GPU effects (30+)
│   ├── _shared/            # Shared effect utilities
│   ├── color/              # Color effects
│   │   ├── brightness/
│   │   ├── contrast/
│   │   ├── saturation/
│   │   ├── vibrance/
│   │   ├── hue-shift/
│   │   ├── temperature/
│   │   ├── exposure/
│   │   ├── levels/
│   │   └── invert/
│   ├── blur/               # Blur effects
│   │   ├── box/
│   │   ├── gaussian/
│   │   ├── motion/
│   │   ├── radial/
│   │   └── zoom/
│   ├── distort/            # Distortion effects
│   │   ├── pixelate/
│   │   ├── kaleidoscope/
│   │   ├── mirror/
│   │   ├── rgb-split/
│   │   ├── twirl/
│   │   ├── wave/
│   │   └── bulge/
│   ├── stylize/            # Stylize effects
│   │   ├── vignette/
│   │   ├── grain/
│   │   ├── glow/
│   │   ├── posterize/
│   │   ├── edge-detect/
│   │   ├── scanlines/
│   │   ├── threshold/
│   │   └── sharpen/
│   └── keying/             # Keying effects
│       └── chroma-key/
│
├── shaders/                # WGSL shaders
│   ├── composite.wgsl      # Layer compositing, 37 blend modes
│   ├── effects.wgsl        # Legacy GPU effects
│   ├── opticalflow.wgsl    # Motion analysis
│   └── output.wgsl         # Final output
│
├── hooks/
│   ├── useEngine.ts
│   ├── useGlobalHistory.ts
│   ├── useMIDI.ts
│   ├── useIsMobile.ts          # Mobile detection
│   ├── useClipPanelSync.ts
│   └── useContextMenuPosition.ts
│
├── services/
│   ├── project/                # Project file service (modular)
│   │   ├── ProjectFileService.ts
│   │   ├── core/
│   │   │   ├── ProjectCoreService.ts
│   │   │   ├── FileStorageService.ts
│   │   │   └── constants.ts
│   │   ├── domains/
│   │   │   ├── AnalysisService.ts
│   │   │   ├── CacheService.ts
│   │   │   ├── ProxyStorageService.ts
│   │   │   ├── RawMediaService.ts
│   │   │   └── TranscriptService.ts
│   │   └── types/
│   │       ├── project.types.ts
│   │       ├── media.types.ts
│   │       ├── composition.types.ts
│   │       ├── timeline.types.ts
│   │       └── folder.types.ts
│   ├── nativeHelper/           # Native Helper client
│   │   ├── NativeHelperClient.ts
│   │   ├── NativeDecoder.ts
│   │   └── protocol.ts
│   ├── projectDB.ts
│   ├── projectSync.ts
│   ├── fileSystemService.ts
│   ├── proxyGenerator.ts
│   ├── proxyFrameCache.ts
│   ├── audioManager.ts
│   ├── audioSync.ts
│   ├── audioExtractor.ts
│   ├── audioAnalyzer.ts
│   ├── compositionAudioMixer.ts
│   ├── aiTools.ts
│   ├── claudeService.ts
│   ├── whisperService.ts
│   ├── clipTranscriber.ts
│   ├── transcriptSync.ts
│   ├── clipAnalyzer.ts
│   ├── multicamAnalyzer.ts
│   ├── compositionRenderer.ts
│   ├── previewRenderManager.ts
│   ├── layerBuilder.ts
│   ├── textRenderer.ts
│   ├── googleFontsService.ts
│   ├── youtubeDownloader.ts
│   ├── klingService.ts
│   ├── piApiService.ts
│   ├── performanceMonitor.ts
│   └── apiKeyManager.ts
│
├── workers/                # Web Workers
│   └── whisper.worker.ts
│
└── types/                  # TypeScript types

tools/
├── helpers/                # Native Helper (Rust)
│   ├── win/                # Windows: YouTube only (yt-dlp)
│   ├── linux/              # Linux: Full FFmpeg decoder
│   └── mac/                # macOS: Full FFmpeg decoder
└── native-helper/          # Electron YouTube tray app
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

## Debugging & Logging

### Logger Service (`src/services/logger.ts`)

Professional logging with levels, module filtering, and AI-agent inspection.

```typescript
// Usage in code:
import { Logger } from '@/services/logger';
const log = Logger.create('ModuleName');

log.debug('Details here', { data });   // Only shows if DEBUG enabled for module
log.info('Important event');            // Always shows (unless level > INFO)
log.warn('Warning message', data);      // Orange in console
log.error('Error occurred', error);     // Red, always shows, captures stack

// Timing helper:
const done = log.time('Operation');
// ... work ...
done(); // Logs "Operation completed in 42.50ms"
```

### Console Commands (Runtime)

```javascript
// Enable debug logs for specific modules:
Logger.enable('WebGPU,FFmpeg')     // Only these modules
Logger.enable('*')                  // All modules
Logger.disable()                    // Turn off debug logs

// Set minimum log level:
Logger.setLevel('DEBUG')            // Show all
Logger.setLevel('WARN')             // Only warnings and errors

// Inspect logs (for AI agents):
Logger.getBuffer()                  // Get all buffered logs
Logger.getBuffer('ERROR')           // Only errors
Logger.search('device')             // Search by keyword
Logger.errors()                     // Recent errors only

// Status & diagnostics:
Logger.status()                     // Show current config
Logger.modules()                    // List registered modules
Logger.dump(50)                     // Pretty print last 50 entries
Logger.summary()                    // Quick summary for AI agents
Logger.export()                     // Export as JSON
```

### Performance Monitoring

```typescript
// Profile output (automatic, every 1s)
// [PROFILE] FPS=60 | gap=16ms | layers=3 | render=2.50ms

// Check GPU status
chrome://gpu

// Slow frame warnings (automatic)
// [RAF] Very slow frame: rafDelay=150ms
```

### Migrating console.log to Logger

```typescript
// OLD:
console.log('[ModuleName] Something happened:', data);
console.error('[ModuleName] Failed:', error);

// NEW:
import { Logger } from '@/services/logger';
const log = Logger.create('ModuleName');
log.info('Something happened', data);
log.error('Failed', error);
```

## Panel System

10 dockable panel types:
- **Preview** - Composition canvas with quality/transparency controls
- **Timeline** - Multi-track editor with navigator
- **Media** - Media browser with columns/folders
- **Properties** - Unified: Transform, Effects, Masks, Volume, Transcript, Analysis
- **Export** - Fast/Precise/FFmpeg export modes
- **Multicam** - Audio-based camera sync
- **AI Chat** - GPT-4/5 editing assistant
- **AI Video** - PiAPI video generation
- **YouTube** - Search and download
- **Slots** - Layer management

## Key Features

- **WebGPU** rendering with 60fps compositing
- **37 blend modes** (all After Effects modes)
- **30+ GPU effects** (color, blur, distort, stylize, keying) with modular architecture
- **Text clips** with 50 Google Fonts, stroke, shadow
- **10-band EQ** with keyframe support
- **Audio master clock** - playhead follows audio like Premiere/Resolve
- **Varispeed audio scrubbing** for all video clips
- **Multicam sync** via audio cross-correlation
- **AI tools** (50+) via OpenAI function calling (GPT-4/GPT-5)
- **AI Video** generation via PiAPI
- **YouTube integration** - search, download, edit
- **4 transcription providers** (Local Whisper, OpenAI, AssemblyAI, Deepgram)
- **3 export modes**: WebCodecs Fast, HTMLVideo Precise, FFmpeg
- **Parallel decoding** for faster multi-clip exports
- **Native Helper** (Rust) for hardware-accelerated ProRes/DNxHD
- **Local project storage** with Raw folder, autosave, backups
- **Mobile support** with responsive UI and touch gestures
