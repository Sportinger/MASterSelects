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
# Nach JEDER Änderung sofort:
git add . && git commit -m "description" && git push origin staging
```

**NIEMALS:**
- Direkt auf `master` committen
- Selbstständig zu `master` mergen
- Mehrere Änderungen sammeln

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
npm install && npm run dev   # http://localhost:5173
npm run build                # Production build
npm run lint                 # ESLint check
```

### Native Helper (optional)
```bash
# Windows (YouTube):
cd tools/helpers/win && cargo run --release

# Linux/Mac (FFmpeg decode):
cd tools/helpers/linux && cargo run --release  # oder /mac
```
Ports: WebSocket `9876`, HTTP `9877`

---

## 3. Architektur (Kurzübersicht)

```
src/
├── components/          # React UI
│   ├── timeline/        # Timeline-Editor (hooks/, components/)
│   ├── panels/          # Properties, Media, AI, YouTube, Export
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
├── effects/             # 30+ GPU Effects (color/, blur/, distort/, stylize/)
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
