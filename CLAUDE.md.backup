# CLAUDE.md

Anweisungen für AI-Assistenten (Claude, GPT, etc.) bei der Arbeit an diesem Projekt.

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
npm run build                # Production build (Changelog immer aktiv)
npm run lint                 # ESLint check
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
│   ├── timeline/        # Timeline-Editor (hooks/, components/)
│   ├── panels/          # Properties, Media, AI, YouTube, Export, Scopes, Transitions
│   ├── preview/         # Canvas + Overlays
│   └── dock/            # Panel-System
├── stores/              # Zustand State
│   ├── timeline/        # Slices: track, clip, keyframe, mask, playback
│   └── mediaStore/      # Slices: import, folder, proxy, composition
├── engine/              # WebGPU Rendering
│   ├── core/            # WebGPUContext, RenderTargetManager
│   ├── render/          # Compositor, RenderLoop, LayerCollector
│   ├── export/          # FrameExporter, VideoEncoder, AudioEncoder
│   ├── audio/           # AudioMixer, TimeStretch
│   └── ffmpeg/          # FFmpegBridge
├── effects/             # 30+ GPU Effects (color/, blur/, distort/, stylize/, keying/)
├── services/            # Audio, AI, Project, NativeHelper
└── shaders/             # WGSL (composite, effects, output)
```

**Detaillierte Struktur:** siehe `README.md` oder `docs/Features/`

---

## 4. Critical Patterns (MUST READ)

### HMR Singleton
Engine muss Hot Reloads überleben:
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
| "Device mismatch" | HMR kaputt → Seite neu laden |
| Schwarzes Canvas | `readyState >= 2` prüfen |
| WebCodecs Fehler | Fällt automatisch auf HTMLVideoElement zurück |

---

## 6. Wichtige Dateien

| Bereich | Datei |
|---------|-------|
| Version/Changelog | `src/version.ts` |
| Engine Entry | `src/engine/WebGPUEngine.ts` |
| Timeline Store | `src/stores/timeline/index.ts` |
| Media Store | `src/stores/mediaStore/index.ts` |
| Effects Registry | `src/effects/index.ts` |
| Logger | `src/services/logger.ts` |
| Project Storage | `src/services/project/core/ProjectCoreService.ts` |

### Neuen Effect hinzufügen
1. Shader in `src/effects/[category]/[name]/shader.wgsl`
2. Index in `src/effects/[category]/[name]/index.ts`
3. Export in `src/effects/[category]/index.ts`
4. UI in `src/components/panels/PropertiesPanel.tsx`

---

## 7. Texture Types

| Source | GPU Type |
|--------|----------|
| Video (HTMLVideoElement) | `texture_external` (zero-copy) |
| Video (VideoFrame) | `texture_external` (zero-copy) |
| Image | `texture_2d<f32>` (copied once) |

---

## 8. Render Pipeline

```
useEngine hook
  └─► engine.start(callback)
        └─► requestAnimationFrame loop
              └─► engine.render(layers)
                    ├─► Import textures
                    ├─► Composite layers (ping-pong)
                    └─► Output to canvas
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
// ❌ FALSCH: 3 sequentielle Round-Trips
const user = await fetchUser()
const posts = await fetchPosts()
const comments = await fetchComments()

// ✅ RICHTIG: 1 paralleler Round-Trip
const [user, posts, comments] = await Promise.all([
  fetchUser(),
  fetchPosts(),
  fetchComments()
])
```

#### Defer Await Until Needed
```typescript
// ❌ FALSCH: blockiert beide Branches
async function handleRequest(userId: string, skipProcessing: boolean) {
  const userData = await fetchUserData(userId)
  if (skipProcessing) return { skipped: true }
  return processUserData(userData)
}

// ✅ RICHTIG: fetch nur wenn nötig
async function handleRequest(userId: string, skipProcessing: boolean) {
  if (skipProcessing) return { skipped: true }
  const userData = await fetchUserData(userId)
  return processUserData(userData)
}
```

#### Strategic Suspense Boundaries
```tsx
// ❌ FALSCH: Ganzes Layout wartet auf Daten
async function Page() {
  const data = await fetchData() // Blockiert alles
  return <div><Sidebar /><DataDisplay data={data} /><Footer /></div>
}

// ✅ RICHTIG: Layout sofort, Daten streamen
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
// ❌ FALSCH: Lädt 1,583 Module
import { Check, X, Menu } from 'lucide-react'

// ✅ RICHTIG: Lädt nur 3 Module
import Check from 'lucide-react/dist/esm/icons/check'
import X from 'lucide-react/dist/esm/icons/x'
import Menu from 'lucide-react/dist/esm/icons/menu'

// ✅ ALTERNATIVE (Next.js 13.5+):
// next.config.js
module.exports = {
  experimental: {
    optimizePackageImports: ['lucide-react', '@mui/material']
  }
}
```

#### Dynamic Imports für Heavy Components
```tsx
// ❌ FALSCH: Monaco im Main Bundle (~300KB)
import { MonacoEditor } from './monaco-editor'

// ✅ RICHTIG: Monaco on-demand
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
// Mehrere Calls → nur 1 Query pro Request
```

#### Minimize Serialization at RSC Boundaries
```tsx
// ❌ FALSCH: Serialisiert alle 50 Felder
<Profile user={user} />

// ✅ RICHTIG: Nur 1 Feld
<Profile name={user.name} />
```

---

### MEDIUM: Re-render Optimization

#### Functional setState (verhindert Stale Closures!)
```typescript
// ❌ FALSCH: Braucht items als Dependency
const addItems = useCallback((newItems) => {
  setItems([...items, ...newItems])
}, [items])  // Wird bei jeder Änderung neu erstellt

// ✅ RICHTIG: Stable Callback, kein Stale Closure
const addItems = useCallback((newItems) => {
  setItems(curr => [...curr, ...newItems])
}, [])  // Keine Dependencies nötig
```

#### Lazy State Initialization
```typescript
// ❌ FALSCH: Läuft bei JEDEM Render
const [index, setIndex] = useState(buildSearchIndex(items))

// ✅ RICHTIG: Läuft nur einmal
const [index, setIndex] = useState(() => buildSearchIndex(items))
```

#### toSorted() statt sort() (verhindert State-Mutation!)
```typescript
// ❌ FALSCH: Mutiert das Original-Array
const sorted = users.sort((a, b) => a.name.localeCompare(b.name))

// ✅ RICHTIG: Erstellt neues Array
const sorted = users.toSorted((a, b) => a.name.localeCompare(b.name))
```

---

### Projekt-spezifische Ergänzungen

Diese Best Practices ergänzen unsere bestehenden Critical Patterns:
- **Stale Closure Fix** (§4) → Functional setState nutzen
- **Zustand Slices** → `get()` in Callbacks statt State-Capture
- **WebGPU Engine** → Heavy Components mit Dynamic Import laden
