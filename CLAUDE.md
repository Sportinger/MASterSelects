# CLAUDE.md

Anweisungen für AI-Assistenten (Claude, GPT, etc.) bei der Arbeit an diesem Projekt.

---

## 0. AI Debug Tools (kein Browser-Plugin nötig!)

Die `/masterselects` Skill stellt 4 Debug-Tools bereit, die über den HTTP Bridge laufen (`POST http://localhost:5173/api/ai-tools`). Voraussetzung: Dev-Server läuft + App in Browser-Tab geöffnet.

| Tool | Parameters | Beschreibung |
|------|-----------|-------------|
| `getStats` | _(none)_ | Engine-Snapshot: FPS, Timing, Decoder, Drops, Audio, GPU |
| `getStatsHistory` | `samples?`, `intervalMs?` | N Snapshots über Zeit sammeln mit min/max/avg Summary |
| `getLogs` | `limit?`, `level?`, `module?`, `search?` | Browser-Logs filtern nach Level (DEBUG/INFO/WARN/ERROR), Modul, Suchtext |
| `getPlaybackTrace` | `windowMs?`, `limit?` | WebCodecs/VF Pipeline-Events + Health-State für Playback-Debugging |

**Nutzung:** `/masterselects getLogs module=PlaybackHealth level=WARN` oder `/masterselects getPlaybackTrace windowMs=10000`

---

## 1. Workflow (WICHTIG!)

### Branch-Regeln
| Branch | Zweck |
|--------|-------|
| `staging` | Entwicklung - hierhin committen |
| `master` | Production - nur via PR |

### Commit-Regeln
```bash
# VOR jedem Commit: Build prüfen!
npm run build

# Nach JEDER Änderung sofort:
git add . && git commit -m "description" && git push origin staging
```

**IMMER vor Commit:**
- `npm run build` ausführen
- Alle Errors beheben (Warnings sind OK)
- Erst dann committen

**NIEMALS:**
- Direkt auf `master` committen
- Selbstständig zu `master` mergen
- Mehrere Änderungen sammeln
- Committen ohne vorherigen Build-Check

### Merge zu Master (nur wenn User es verlangt!)
```bash
# 1. Version erhöhen in src/version.ts
# 2. CHANGELOG aktualisieren in src/version.ts
# 3. Commit & Push
# 4. PR erstellen und mergen:
gh pr create --base master --head staging --title "..." --body "..."
gh pr merge --merge
# 5. Staging synchronisieren:
git fetch origin && git merge origin/master && git push origin staging
```

### Version & Changelog
- **Datei:** `src/version.ts`
- **Version:** Nur bei Merge zu master erhöhen (PATCH +1)
- **CHANGELOG:** Neuen Eintrag am Anfang mit `version`, `date`, `changes[]`
- **KNOWN_ISSUES:** Aktuelle Bugs pflegen

### Dokumentation
Bei Feature-Änderungen: `docs/Features/` aktualisieren

---

## 2. Quick Reference

```bash
npm install && npm run dev   # http://localhost:5173 (ohne Changelog)
npm run dev:changelog        # Dev-Server MIT Changelog-Dialog
npm run build                # Production build (tsc + vite, Changelog immer aktiv)
npm run build:deploy         # Production build ohne tsc (nur vite)
npm run lint                 # ESLint check
npm run test                 # Vitest einmal ausführen
npm run test:watch           # Vitest im Watch-Modus
npm run test:unit            # Nur Unit-Tests (tests/unit/)
npm run preview              # Built output lokal serven
```

### Dev-Server Regeln
- **IMMER `npm run dev` verwenden** (ohne Changelog)
- `npm run dev:changelog` nur wenn User Changelog sehen will
- Production builds zeigen Changelog automatisch

### Native Helper (optional, cross-platform)
```bash
# All platforms (FFmpeg decode/encode + yt-dlp downloads):
cd tools/native-helper && cargo run --release

# Windows: requires FFMPEG_DIR + LIBCLANG_PATH env vars (see tools/native-helper/README.md)
```
Ports: WebSocket `9876`, HTTP `9877`

---

## 3. Architektur (Kurzübersicht)

```
src/
├── components/          # React UI
│   ├── timeline/        # Timeline-Editor (hooks/, components/, utils/)
│   ├── panels/          # Properties, Media, AI, Scopes, Transitions, SAM2, Transcript, etc.
│   ├── preview/         # Canvas + Overlays (Mask, SAM2, MultiPreview)
│   ├── dock/            # Panel-System (DockContainer, Tabs, Split, Float)
│   ├── common/          # Shared UI: Toolbar, Settings, Dialogs, Overlays
│   ├── export/          # Export Dialog + Panel
│   ├── outputManager/   # Output Window / Slice management
│   └── mobile/          # Mobile-optimized UI
├── stores/              # Zustand State
│   ├── timeline/        # Slices: track, clip, keyframe, mask, playback, selection, transition, ...
│   ├── mediaStore/      # Slices: fileImport, fileManage, folder, proxy, composition, slot, ...
│   ├── historyStore.ts  # Snapshot-based Undo/Redo
│   ├── engineStore.ts   # Engine ready state, GPU info
│   ├── settingsStore.ts # User preferences
│   ├── dockStore.ts     # Panel layout state
│   ├── sliceStore.ts    # Slice/region management
│   ├── renderTargetStore.ts # Output targets
│   ├── sam2Store.ts     # SAM2 segmentation state
│   ├── multicamStore.ts # Multicam editing state
│   └── youtubeStore.ts  # YouTube download state
├── engine/              # WebGPU Rendering
│   ├── core/            # WebGPUContext, RenderTargetManager
│   ├── render/          # RenderLoop, RenderDispatcher, LayerCollector, Compositor, NestedCompRenderer
│   ├── pipeline/        # CompositorPipeline, EffectsPipeline, OutputPipeline, SlicePipeline
│   ├── texture/         # TextureManager, MaskTextureManager, ScrubbingCache
│   ├── managers/        # CacheManager, ExportCanvasManager, OutputWindowManager
│   ├── export/          # FrameExporter, VideoEncoderWrapper, AudioEncoder
│   ├── audio/           # AudioMixer, TimeStretchProcessor, AudioExportPipeline
│   ├── video/           # VideoFrameManager
│   ├── ffmpeg/          # FFmpegBridge
│   ├── analysis/        # Scopes (Histogram, Waveform, Vectorscope, OpticalFlow)
│   ├── stats/           # PerformanceStats
│   └── structuralSharing/ # SnapshotManager for render optimization
├── effects/             # ~30 GPU Effects (color/, blur/, distort/, stylize/, keying/)
│   └── EffectsPipeline.ts # Effect orchestration
├── transitions/         # GPU Transitions (crossfade, etc.)
├── services/            # Business logic
│   ├── layerBuilder/    # LayerBuilderService, VideoSyncManager, AudioSyncHandler
│   ├── mediaRuntime/    # Clip bindings, runtime playback registry
│   ├── project/         # ProjectCoreService, save/load, file service
│   ├── nativeHelper/    # Native FFmpeg decoder client
│   ├── sam2/            # SAM2 segmentation service
│   ├── aiTools/         # AI tool bridge (Claude integration)
│   ├── export/          # FCPXML export
│   └── (standalone)     # logger, audioManager, thumbnailRenderer, whisperService, etc.
├── hooks/               # React hooks: useEngine, useGlobalHistory, useMIDI, useTheme, ...
├── utils/               # Helpers: keyframeInterpolation, maskRenderer, fileLoader, etc.
├── types/               # TypeScript types, mp4box.d.ts
├── workers/             # Web Workers (transcription)
└── shaders/             # WGSL: composite, effects, output, slice, opticalflow
```

**Detaillierte Struktur:** siehe `README.md` oder `docs/Features/`

---

## 4. Critical Patterns (MUST READ)

### HMR Singleton
Singletons (Engine, FFmpegBridge, SAM2Service) müssen Hot Reloads überleben:
```typescript
let instance: MyService | null = null;

if (import.meta.hot) {
  import.meta.hot.accept();
  if (import.meta.hot.data?.myService) {
    instance = import.meta.hot.data.myService;
  }
  import.meta.hot.dispose((data) => {
    data.myService = instance;
  });
}
```

### Stale Closure Fix
Immer `get()` in async Callbacks:
```typescript
// FALSCH
const { layers } = get();
video.onload = () => set({ layers: layers.map(...) });

// RICHTIG
video.onload = () => {
  const current = get().layers;
  set({ layers: current.map(...) });
};
```

### Video Ready State
Warten auf `canplaythrough`, nicht `loadeddata`:
```typescript
video.addEventListener('canplaythrough', () => {
  // Jetzt ist Video bereit
}, { once: true });
```

### Zustand Slice Pattern
```typescript
export const createSlice: SliceCreator<Actions> = (set, get) => ({
  actionName: (params) => {
    const state = get();
    set({ /* updates */ });
  },
});
```

---

## 5. Debugging & Logging

### Logger verwenden
```typescript
import { Logger } from '@/services/logger';
const log = Logger.create('ModuleName');

log.debug('Verbose', { data });  // Nur wenn DEBUG aktiv
log.info('Event');               // Immer sichtbar
log.warn('Warning', data);       // Orange
log.error('Fehler', error);      // Rot + Stack Trace
```

### Console Commands
```javascript
Logger.enable('WebGPU,FFmpeg')  // Module aktivieren
Logger.enable('*')              // Alle aktivieren
Logger.disable()                // Nur Errors

Logger.setLevel('DEBUG')        // Alle Level
Logger.setLevel('WARN')         // Nur Warn+Error

Logger.search('device')         // Logs durchsuchen
Logger.errors()                 // Nur Fehler
Logger.dump(50)                 // Letzte 50 ausgeben
Logger.summary()                // Übersicht für AI
```

### Common Issues

| Problem | Lösung |
|---------|--------|
| 15fps auf Linux | `chrome://flags/#enable-vulkan` aktivieren |
| "Device mismatch" | HMR kaputt -> Seite neu laden |
| Schwarzes Canvas | `readyState >= 2` prüfen |
| WebCodecs Fehler | Fällt automatisch auf HTMLVideoElement zurück |

---

## 6. Wichtige Dateien

| Bereich | Datei |
|---------|-------|
| Version/Changelog | `src/version.ts` |
| Engine Entry | `src/engine/WebGPUEngine.ts` |
| Render Dispatcher | `src/engine/render/RenderDispatcher.ts` |
| Timeline Store | `src/stores/timeline/index.ts` |
| Media Store | `src/stores/mediaStore/index.ts` |
| History (Undo/Redo) | `src/stores/historyStore.ts` |
| Effects Registry | `src/effects/index.ts` |
| Effects Pipeline | `src/effects/EffectsPipeline.ts` |
| Layer Builder | `src/services/layerBuilder/LayerBuilderService.ts` |
| Video Sync | `src/services/layerBuilder/VideoSyncManager.ts` |
| Logger | `src/services/logger.ts` |
| Project Storage | `src/services/project/core/ProjectCoreService.ts` |
| Engine Hook | `src/hooks/useEngine.ts` |

### Neuen Effect hinzufügen
1. Shader in `src/effects/[category]/[name]/shader.wgsl`
2. Index in `src/effects/[category]/[name]/index.ts`
3. Export in `src/effects/[category]/index.ts`
4. UI in `src/components/panels/PropertiesPanel.tsx`

---

## 7. Texture Types

| Source | GPU Type |
|--------|----------|
| Video (HTMLVideoElement) | `texture_external` via `importExternalTexture` (zero-copy) |
| Video (VideoFrame) | `texture_external` via `importExternalTexture` (zero-copy) |
| Image (HTMLImageElement) | `texture_2d<f32>` via `copyExternalImageToTexture` (copied once, cached) |
| Canvas (Text Clips) | `texture_2d<f32>` via `copyExternalImageToTexture` (cached by reference) |
| Native Decoder Frames | `texture_2d<f32>` dynamic textures (re-uploaded per frame) |

---

## 8. Render Pipeline

```
useEngine hook (src/hooks/useEngine.ts)
  └── engine.initialize() -> WebGPUContext + all pipelines
        └── RenderLoop.start()
              └── requestAnimationFrame loop (idle detection + fps limiting)
                    └── RenderDispatcher.render(layers)
                          ├── LayerCollector: Import textures (external/cached/scrubbing)
                          ├── Compositor: Ping-pong compositing + effects per layer
                          ├── NestedCompRenderer: Handle compositions-in-compositions
                          ├── OutputPipeline: Output to preview canvas
                          └── SlicePipeline: Output to slice/render target canvases
```

---

*Ausführliche Dokumentation: `docs/Features/README.md`*

---

## 9. React/Next.js Best Practices (Vercel Engineering)

> Vollständige Dokumentation: [REACT-BEST-PRACTICES.md](./docs/REACT-BEST-PRACTICES.md) | [GitHub Source](https://github.com/vercel-labs/agent-skills/tree/main/skills/react-best-practices)

### Prioritäten nach Impact
1. **CRITICAL:** Eliminating Waterfalls, Bundle Size
2. **HIGH:** Server-Side Performance
3. **MEDIUM:** Client-Side Data, Re-renders, Rendering
4. **LOW:** JavaScript Micro-Optimizations, Advanced Patterns

---

### CRITICAL: Eliminating Waterfalls

**Waterfalls sind der #1 Performance-Killer!**

#### Promise.all() für unabhängige Operations
```typescript
// FALSCH: 3 sequentielle Round-Trips
const user = await fetchUser()
const posts = await fetchPosts()
const comments = await fetchComments()

// RICHTIG: 1 paralleler Round-Trip
const [user, posts, comments] = await Promise.all([
  fetchUser(),
  fetchPosts(),
  fetchComments()
])
```

#### Defer Await Until Needed
```typescript
// FALSCH: blockiert beide Branches
async function handleRequest(userId: string, skipProcessing: boolean) {
  const userData = await fetchUserData(userId)
  if (skipProcessing) return { skipped: true }
  return processUserData(userData)
}

// RICHTIG: fetch nur wenn nötig
async function handleRequest(userId: string, skipProcessing: boolean) {
  if (skipProcessing) return { skipped: true }
  const userData = await fetchUserData(userId)
  return processUserData(userData)
}
```

#### Strategic Suspense Boundaries
```tsx
// FALSCH: Ganzes Layout wartet auf Daten
async function Page() {
  const data = await fetchData() // Blockiert alles
  return <div><Sidebar /><DataDisplay data={data} /><Footer /></div>
}

// RICHTIG: Layout sofort, Daten streamen
function Page() {
  return (
    <div>
      <Sidebar />
      <Suspense fallback={<Skeleton />}>
        <DataDisplay />
      </Suspense>
      <Footer />
    </div>
  )
}
```

---

### CRITICAL: Bundle Size Optimization

#### Avoid Barrel File Imports (200-800ms Import-Cost!)
```tsx
// FALSCH: Lädt 1,583 Module
import { Check, X, Menu } from 'lucide-react'

// RICHTIG: Lädt nur 3 Module
import Check from 'lucide-react/dist/esm/icons/check'
import X from 'lucide-react/dist/esm/icons/x'
import Menu from 'lucide-react/dist/esm/icons/menu'

// ALTERNATIVE (Next.js 13.5+):
// next.config.js
module.exports = {
  experimental: {
    optimizePackageImports: ['lucide-react', '@mui/material']
  }
}
```

#### Dynamic Imports für Heavy Components
```tsx
// FALSCH: Monaco im Main Bundle (~300KB)
import { MonacoEditor } from './monaco-editor'

// RICHTIG: Monaco on-demand
import dynamic from 'next/dynamic'
const MonacoEditor = dynamic(
  () => import('./monaco-editor').then(m => m.MonacoEditor),
  { ssr: false }
)
```

---

### HIGH: Server-Side Performance

#### React.cache() für Request-Deduplication
```typescript
import { cache } from 'react'

export const getCurrentUser = cache(async () => {
  const session = await auth()
  if (!session?.user?.id) return null
  return await db.user.findUnique({ where: { id: session.user.id } })
})
// Mehrere Calls -> nur 1 Query pro Request
```

#### Minimize Serialization at RSC Boundaries
```tsx
// FALSCH: Serialisiert alle 50 Felder
<Profile user={user} />

// RICHTIG: Nur 1 Feld
<Profile name={user.name} />
```

---

### MEDIUM: Re-render Optimization

#### Functional setState (verhindert Stale Closures!)
```typescript
// FALSCH: Braucht items als Dependency
const addItems = useCallback((newItems) => {
  setItems([...items, ...newItems])
}, [items])  // Wird bei jeder Änderung neu erstellt

// RICHTIG: Stable Callback, kein Stale Closure
const addItems = useCallback((newItems) => {
  setItems(curr => [...curr, ...newItems])
}, [])  // Keine Dependencies nötig
```

#### Lazy State Initialization
```typescript
// FALSCH: Läuft bei JEDEM Render
const [index, setIndex] = useState(buildSearchIndex(items))

// RICHTIG: Läuft nur einmal
const [index, setIndex] = useState(() => buildSearchIndex(items))
```

#### toSorted() statt sort() (verhindert State-Mutation!)
```typescript
// FALSCH: Mutiert das Original-Array
const sorted = users.sort((a, b) => a.name.localeCompare(b.name))

// RICHTIG: Erstellt neues Array
const sorted = users.toSorted((a, b) => a.name.localeCompare(b.name))
```

---

### Projekt-spezifische Ergänzungen

Diese Best Practices ergänzen unsere bestehenden Critical Patterns:
- **Stale Closure Fix** (S4) -> Functional setState nutzen
- **Zustand Slices** -> `get()` in Callbacks statt State-Capture
- **WebGPU Engine** -> Heavy Components mit Dynamic Import laden
