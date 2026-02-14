# Engine Redesign — Scene Graph + Render Graph + Decoder Pool + GPU Memory + Dirty Tracking + Structural Sharing

## Ziel

Skalierung auf 20+ HD Texturen mit Effekten, tiefes Nesting (3+ Ebenen), shared Decoding, flüssiges Scrubbing.
Scene Graph, Dirty Tracking und Structural Sharing sind **always-on** (kein Feature Flag).
RenderGraph und DecoderPool bleiben hinter Feature Flags (noch nicht fertig).

---

## Architektur-Übersicht

```
                  ┌────────────────────┐
                  │   Zustand Stores    │
                  │  (timeline, media)  │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │  SceneGraphBuilder  │  ← NEU (Phase 1)
                  │  Baum aus Clips     │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │ SceneGraphEvaluator │  ← NEU (Phase 1 + 5)
                  │ + DirtyTracker      │     Visibility Culling,
                  │                     │     Keyframe Interpolation,
                  │                     │     Caching für clean Nodes
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │ SceneGraphAdapter   │  ← NEU (Phase 1)
                  │ EvaluatedNode→Layer │
                  └─────────┬──────────┘
                            │
              ┌─────────────▼─────────────┐
              │     RenderDispatcher       │  BESTEHEND
              │  (orchestriert Rendering)  │
              └──────┬──────┬──────┬──────┘
                     │      │      │
            ┌────────▼┐  ┌──▼───┐  ┌▼─────────────┐
            │  Layer   │  │Compo-│  │ NestedComp   │  BESTEHEND
            │Collector │  │sitor │  │ Renderer     │
            └──────────┘  └──────┘  └──────────────┘
                     │      │      │
              ┌──────▼──────▼──────▼──────┐
              │  GPU Pipelines            │  BESTEHEND (unverändert)
              │  CompositorPipeline       │
              │  EffectsPipeline          │
              │  OutputPipeline           │
              │  SlicePipeline            │
              └───────────────────────────┘

  Parallel/Unabhängig:

  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  GpuMemoryManager│   │   DecoderPool    │   │ SnapshotManager  │
  │  (Phase 4)       │   │   (Phase 3)      │   │ (Phase 6)        │
  │  VRAM Tracking   │   │  Shared Decoding │   │ Structural Share │
  └──────────────────┘   └──────────────────┘   └──────────────────┘

  Zukunft (nicht im Haupt-Renderpfad):

  ┌──────────────────────────┐
  │  RenderGraph (Phase 2)   │
  │  Builder + Executor      │
  │  DAG von Render-Passes   │
  └──────────────────────────┘
```

---

## Status der 6 Phasen

### Phase 0: Foundation — Types + Feature Flags ✅ FERTIG

| Datei | Zeilen | Status |
|-------|--------|--------|
| `src/engine/featureFlags.ts` | 13 | ✅ Nur noch 2 Flags: `useRenderGraph`, `useDecoderPool`. Scene Graph + Dirty Tracking + Structural Sharing sind always-on. |

Alle Type-Dateien für Phase 1-6 erstellt und kompilierbar.

---

### Phase 1: Scene Graph + Evaluator + Adapter ✅ ALWAYS-ON

**Neue Dateien:**

| Datei | Zeilen | Beschreibung |
|-------|--------|-------------|
| `src/engine/sceneGraph/types.ts` | 86 | SceneNode, SceneGraph, EvaluatedNode, ResolvedTransform |
| `src/engine/sceneGraph/SceneGraphBuilder.ts` | 184 | Liest aus Zustand Stores, baut rekursiven Baum. Reference-Identity Cache. |
| `src/engine/sceneGraph/SceneGraphEvaluator.ts` | 180 | Visibility Culling, Keyframe-Interpolation via Store-Methoden, DirtyTracker integriert (always-on). |
| `src/engine/sceneGraph/SceneGraphAdapter.ts` | 251 | Konvertiert EvaluatedNode[] → Layer[] für bestehende Pipeline. Handles video/image/text/solid/composition. |
| `src/engine/sceneGraph/index.ts` | 10 | Barrel exports |

**Geänderte bestehende Dateien:**

| Datei | Änderung |
|-------|----------|
| `src/services/layerBuilder/LayerBuilderService.ts` | **821 → 145 Zeilen.** Alter Build-Pfad komplett entfernt (buildLayers, buildLayerForClip, etc.). `buildLayersFromStore()` geht direkt über Scene Graph. `invalidateCache()` ruft `sceneGraphBuilder.invalidate()`. |

**Integration:** Always-on. Kein Feature Flag. Layer-Building geht immer über Scene Graph.

---

### Phase 2: Render Graph ⚠️ SCAFFOLDING (nicht im Haupt-Renderpfad)

**Neue Dateien:**

| Datei | Zeilen | Beschreibung |
|-------|--------|-------------|
| `src/engine/renderGraph/types.ts` | 92 | RenderGraph, RenderPassNode, ResourceHandle, alle PassConfig-Typen |
| `src/engine/renderGraph/RenderGraphBuilder.ts` | 280 | Baut DAG aus EvaluatedNode[]. Pass-Typen: clear, composite, effect, nestedComp, output. Topologische Ordnung. |
| `src/engine/renderGraph/RenderGraphExecutor.ts` | 115 | Placeholder-Stubs für composite/effect/nestedComp. Nur clear + output sind implementiert. |
| `src/engine/renderGraph/index.ts` | 9 | Barrel exports |

**Geänderte bestehende Dateien:**

| Datei | Änderung |
|-------|----------|
| `src/engine/render/RenderDispatcher.ts` | Feature-Flag-Branch: wenn `flags.useRenderGraph` → `renderViaRenderGraph()` baut Scene Graph, evaluiert, baut Render Graph, führt aus. Instanzen von SceneGraphBuilder, SceneGraphEvaluator, RenderGraphBuilder, RenderGraphExecutor. |

**Status:** Builder funktioniert, Executor hat **Placeholder-Stubs** für composite/effect Passes. Produziert kein korrektes Bild. Der Executor müsste die gesamte Compositor-Logik (Ping-Pong, Bind Groups, Uniforms, Effect Pre-Processing, Mask Handling) nachbauen — das ist noch **nicht umgesetzt**.

**Warum nicht fertig:**
- Compositor.composite() verarbeitet alle Layer in einer Schleife mit komplexer Bind-Group-Erstellung, Inline-Effects, Mask-Handling
- Das in einzelne Graph-Passes aufzubrechen erfordert erhebliche Refaktorierung
- Risiko für visuelle Regressionen ist hoch
- Der bestehende Render-Pfad (RenderDispatcher → LayerCollector → Compositor → OutputPipeline) funktioniert korrekt

**Plan:** Render Graph bleibt als Zukunfts-Optimierung. Aktuell liefert der Scene Graph die Layer[], und die bestehende Render-Pipeline (Compositor, NestedCompRenderer) übernimmt das GPU-Rendering.

---

### Phase 3: Decoder Pool ⚠️ ERSTELLT, NICHT INTEGRIERT

**Neue Dateien:**

| Datei | Zeilen | Beschreibung |
|-------|--------|-------------|
| `src/engine/decoderPool/types.ts` | 58 | DecoderHandle, DecoderRequest, DecoderPoolConfig, DecoderPoolStats, DecoderPriority, DecoderType |
| `src/engine/decoderPool/DecoderPool.ts` | 208 | Map<mediaFileId, DecoderHandle[]>. acquire(): share nearby → reuse idle → create new → evict LRU. Max 8 Decoder. |
| `src/engine/decoderPool/DomRefRegistry.ts` | 81 | Singleton Registry für video/audio/image Elemente + text Canvases. Implements DomRefRegistryInterface. |
| `src/engine/decoderPool/index.ts` | 10 | Barrel exports |

**NICHT geänderte bestehende Dateien:**

| Datei | Was fehlt |
|-------|----------|
| `src/engine/render/LayerCollector.ts` | **Kein Import von DecoderPool.** Die Decoder-Auswahl (NativeHelper → ParallelDecode → WebCodecs → HTMLVideo) geht noch direkt über `source.videoElement`, `source.webCodecsPlayer`, `source.nativeDecoder`. |

**Was fehlt für Integration:**
1. DecoderPool-Instanz erstellen (wo? WebGPUEngine oder LayerCollector)
2. LayerCollector.collectLayerData() umbauen: statt `source.nativeDecoder`, `source.videoElement` etc. → `decoderPool.acquire(mediaFileId, sourceTime, priority)`
3. DecoderHandle muss die tatsächlichen DOM-Elemente/Decoder wrappen (aktuell sind `videoElement`, `webCodecsPlayer`, `nativeDecoder` optional auf dem Handle)
4. DomRefRegistry muss populiert werden wenn Clips geladen werden

---

### Phase 4: GPU Memory Manager ✅ INTEGRIERT (Tracking)

**Neue Dateien:**

| Datei | Zeilen | Beschreibung |
|-------|--------|-------------|
| `src/engine/gpuMemory/types.ts` | 56 | GpuAllocation, GpuAllocationCategory, GpuMemoryBudget, GpuMemoryConfig, EVICTION_PRIORITY (Array), DEFAULT_GPU_MEMORY_CONFIG (2GB) |
| `src/engine/gpuMemory/GpuMemoryManager.ts` | 243 | Wraps device.createTexture() + registerExternal() für Tracking. Budget-basierte LRU-Eviction. tick() per Frame via PerformanceStats. |
| `src/engine/gpuMemory/index.ts` | 8 | Barrel exports |

**Geänderte bestehende Dateien:**

| Datei | Änderung |
|-------|----------|
| `src/engine/stats/PerformanceStats.ts` | `setGpuMemoryManager()` Setter. `getStats()` liest `gpuMemoryManager.getUsageMB()` (always-on). `updateStats()` ruft `gpuMemoryManager.tick()`. |
| `src/engine/texture/TextureManager.ts` | `setGpuMemoryManager()` Setter. Image/Canvas/Dynamic Texturen werden via `registerExternal()` getracked. `removeDynamicTexture()` ruft `unregisterExternal()`. |
| `src/engine/core/RenderTargetManager.ts` | `setGpuMemoryManager()` Setter. Ping-Pong + Effect-Temp Texturen werden via `registerExternal()` als `pingPong`/`effectTemp` (pinned) getracked. |
| `src/engine/WebGPUEngine.ts` | GpuMemoryManager wird in `createResources()` erstellt und an PerformanceStats, TextureManager, RenderTargetManager weitergegeben. |

**Ansatz:** Tracking-only via `registerExternal()`/`unregisterExternal()` — Texturen werden extern erstellt, Manager trackt nur VRAM-Verbrauch. Keine Lifecycle-Änderung (destroy-Semantik bleibt bei den Managern). Eviction bleibt für Zukunft wenn createTexture() statt registerExternal() genutzt wird.

---

### Phase 5: Dirty Tracking ✅ ALWAYS-ON

**Neue Dateien:**

| Datei | Zeilen | Beschreibung |
|-------|--------|-------------|
| `src/engine/dirtyTracking/types.ts` | 36 | DirtyFlags (transform, effects, source, structure, time, any), TrackedNodeState, DirtyTrackingStats |
| `src/engine/dirtyTracking/DirtyTracker.ts` | 137 | Vergleicht Version-Counter zwischen Frames. Video-Nodes immer dirty bei Time-Change. Image/Text/Solid nur bei Version-Change. getOrReuse() gibt cached EvaluatedNode zurück. |
| `src/engine/dirtyTracking/index.ts` | 6 | Barrel exports |

**Geänderte bestehende Dateien:**

| Datei | Änderung |
|-------|----------|
| `src/engine/sceneGraph/SceneGraphEvaluator.ts` | DirtyTracker-Instanz always-on (kein Flag). In `evaluate()`: `dirtyTracker.update(graph, time)`. In `evaluateNode()`: `dirtyTracker.getOrReuse(node, time)` → skip Interpolation für clean Nodes. Nach Evaluation: `dirtyTracker.cacheEvaluation()`. |

**Always-on.** 20 Clips mit 1 Video + 19 Bilder = nur 1×9 statt 20×9 Interpolationen pro Frame.

---

### Phase 6: Structural Sharing ✅ ALWAYS-ON

**Neue Dateien:**

| Datei | Zeilen | Beschreibung |
|-------|--------|-------------|
| `src/engine/structuralSharing/types.ts` | 69 | SerializedClipState (TimelineClip ohne DOM-Refs), HistorySnapshotV2, DomRefRegistryInterface |
| `src/engine/structuralSharing/SnapshotManager.ts` | 195 | Auto-Detection via Zustand Reference-Comparison + explizites trackChange(). createSnapshot() mit Structural Sharing (nur geänderte Clips clonen, Rest shared). |
| `src/engine/structuralSharing/index.ts` | 6 | Barrel exports |

**Geänderte bestehende Dateien:**

| Datei | Änderung |
|-------|----------|
| `src/stores/historyStore.ts` | Always-on. SnapshotManager Singleton. `createSnapshot()` geht immer über `snapshotManager.createSnapshot()`. Alter deepClone-Pfad für Clips entfernt (deepClone bleibt für media/dock/layers). |

**Auto-Detection:** Der SnapshotManager vergleicht Clip-Objekt-Referenzen zwischen Snapshots (Zustand erstellt neue Objekte bei Mutationen). Kein manuelles `trackClipChange()` in jeder Mutation nötig.

---

## Zusammenfassung: Was ist fertig, was fehlt

### ✅ Fertig und integriert (hinter Feature Flags)

| System | Flag | Integriert in |
|--------|------|--------------|
| Scene Graph (Builder + Evaluator + Adapter) | `useSceneGraph` | `LayerBuilderService.buildLayersFromStore()` |
| Dirty Tracking | `useDirtyTracking` | `SceneGraphEvaluator.evaluateNode()` |
| PerformanceStats VRAM | `useGpuMemoryManager` | `PerformanceStats.getStats()` |
| Structural Sharing (Capture-Seite) | `useStructuralSharing` | `historyStore.createSnapshot()` |

### ⚠️ Erstellt aber NICHT integriert

| System | Was fehlt |
|--------|----------|
| **Render Graph Executor** | Composite/Effect Passes sind Stubs. Kein korrektes Rendering. |
| **Decoder Pool** | Kein Import in LayerCollector. Dead Code. |
| **GPU Memory Manager** | Nicht in TextureManager, RenderTargetManager, ScrubbingCache integriert. Nur Stats-Lesung. |
| **Structural Sharing (Track-Seite)** | `trackClipChange()` wird nirgends aufgerufen. |

### 📋 Offene Aufgaben für "kein alter Code"

1. **Feature Flags entfernen** — alle Systeme immer aktiv, kein `flags.useXxx` mehr
2. **Alter LayerBuilderService-Pfad entfernen** — nur noch Scene Graph
3. **GpuMemoryManager in TextureManager/RenderTargetManager/ScrubbingCache einbauen** — alle `device.createTexture()` durch Manager routen
4. **trackClipChange() Calls** in Timeline-Store-Mutationen einbauen
5. **deepClone-Pfad aus historyStore entfernen** — nur noch Structural Sharing
6. **DecoderPool in LayerCollector einbauen** — Decoder-Auswahl über Pool statt direkt
7. **Render Graph Executor** — entweder voll implementieren oder als experimentelles Feature behalten

---

## Datei-Inventar (alle neuen Dateien)

```
src/engine/
├── featureFlags.ts                          (16 Zeilen)
├── sceneGraph/
│   ├── types.ts                             (86 Zeilen)
│   ├── SceneGraphBuilder.ts                 (184 Zeilen)
│   ├── SceneGraphEvaluator.ts               (185 Zeilen)
│   ├── SceneGraphAdapter.ts                 (251 Zeilen)
│   └── index.ts                             (10 Zeilen)
├── renderGraph/
│   ├── types.ts                             (92 Zeilen)
│   ├── RenderGraphBuilder.ts                (280 Zeilen)
│   ├── RenderGraphExecutor.ts               (115 Zeilen)
│   └── index.ts                             (9 Zeilen)
├── decoderPool/
│   ├── types.ts                             (58 Zeilen)
│   ├── DecoderPool.ts                       (208 Zeilen)
│   ├── DomRefRegistry.ts                    (81 Zeilen)
│   └── index.ts                             (10 Zeilen)
├── gpuMemory/
│   ├── types.ts                             (56 Zeilen)
│   ├── GpuMemoryManager.ts                  (217 Zeilen)
│   └── index.ts                             (8 Zeilen)
├── dirtyTracking/
│   ├── types.ts                             (36 Zeilen)
│   ├── DirtyTracker.ts                      (137 Zeilen)
│   └── index.ts                             (6 Zeilen)
└── structuralSharing/
    ├── types.ts                             (69 Zeilen)
    ├── SnapshotManager.ts                   (179 Zeilen)
    └── index.ts                             (6 Zeilen)

Gesamt: 22 neue Dateien, ~2.363 Zeilen neuer Code
```

## Geänderte bestehende Dateien

| Datei | Art der Änderung |
|-------|-----------------|
| `src/services/layerBuilder/LayerBuilderService.ts` | +Imports, +3 private Instanzen, +Feature-Flag-Branch in buildLayersFromStore(), +buildLayersViaSceneGraph(), +invalidate() |
| `src/engine/render/RenderDispatcher.ts` | +Imports (flags, SceneGraph*, RenderGraph*, useTimelineStore), +4 private Instanzen, +Feature-Flag-Branch in render(), +renderViaRenderGraph() |
| `src/engine/sceneGraph/SceneGraphEvaluator.ts` | +Import DirtyTracker+flags, +private dirtyTracker, +dirty check in evaluate()+evaluateNode(), +cacheEvaluation() |
| `src/engine/stats/PerformanceStats.ts` | +Import flags+GpuMemoryManager type, +setGpuMemoryManager(), gpuMemory liest aus Manager |
| `src/stores/historyStore.ts` | +Import flags+SnapshotManager+HistorySnapshotV2, +Singleton snapshotManager, +Structural-Sharing-Branch in createSnapshot(), +trackClipChange/trackClipChanges exports |

## Kompilierung

```bash
# RICHTIG (prüft alle src/ Dateien):
npx tsc -p tsconfig.app.json --noEmit

# FALSCH (prüft KEINE Dateien wegen "files": []):
npx tsc --noEmit
```

**Aktueller Stand:** ✅ 0 Fehler mit `npx tsc -p tsconfig.app.json --noEmit`

---

## Was unverändert bleibt

- Alle WGSL Shader
- CompositorPipeline (4 Pipelines, 36 Blend Modes)
- EffectsPipeline (31 GPU Effects)
- OutputPipeline, SlicePipeline
- RenderLoop (RAF + Idle Detection)
- Compositor (Ping-Pong Compositing)
- NestedCompRenderer (Nested Comp Pre-Rendering)
- LayerCollector (Texture-Import)
- UI-Komponenten
- Keyframe-Interpolations-Mathematik (keyframeInterpolation.ts)
- Export Pipeline (WebCodecs Encoding)
- Audio Pipeline
