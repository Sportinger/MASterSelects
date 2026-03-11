# Consensus Report C: Verbesserungsvorschlaege & Roadmap fuer MasterSelects

## Quellen

Dieser Bericht konsolidiert die Verbesserungsvorschlaege aller 7 Research-Agents:

| Agent | Fokus |
|-------|-------|
| Agent 1 | Decoding, Seeking & Frame-Accuracy |
| Agent 2 | Multi-Track Compositing |
| Agent 3 | Playback-Engine & Audio/Video-Synchronisation |
| Agent 4 | Memory Management, Performance & Resource Limits |
| Agent 5 | Export & Encoding Pipeline |
| Agent 6 | Browser-Kompatibilitaet & Fallback-Strategien |
| Agent 7 | Full Codebase Cross-Cutting Analysis |

---

## 1. Konsens-Analyse: Empfehlungen nach Haeufigkeit

Die folgende Tabelle zeigt, wie viele Agents dieselbe Verbesserung identifiziert oder empfohlen haben. Hoehere Konsens-Staerke bedeutet hoeheres Vertrauen in die Prioritaet.

### Konsens-Staerke 7/7 (Alle Agents)
- **WebCodecs Full Playback aktivieren** -- Alle Agents identifizieren `useFullWebCodecsPlayback: false` als zentrale Limitation. HTMLVideoElement bleibt der Default-Decoder fuer Preview, obwohl die WebCodecs-Infrastruktur fertig implementiert ist. (A1 S9.3, A2 S9, A3 S12, A5 S7.1, A6 S2.4, A7 S1.1)

### Konsens-Staerke 5/7
- **Dynamische VRAM-Budget-Anpassung** -- ScrubbingCache mit 300 Frames bei 1080p = ~2.4 GB VRAM ist zu viel fuer integrierte GPUs (4 GB total). Keine Anpassung basierend auf verfuegbarem VRAM. (A2 S3, A4 S5, A4 S15.4, A7 S4.1, A2 S9)
- **Code-Duplikation in Layer Collection eliminieren** -- Die Video-Fallback-Kaskade existiert dreifach in LayerCollector, NestedCompRenderer und RenderDispatcher. (A2 S6, A7 S7.1, A7 S2.4, A6 S7.1)

### Konsens-Staerke 4/7
- **FFmpeg WASM Streaming-Export** -- Alle Frames muessen gleichzeitig in RAM gehalten werden. 60s @ 1080p30 = ~14.9 GB. Streaming-Ansatz wuerde dies auf wenige Frames reduzieren. (A4 S9, A5 S6.2, A5 S9, A7 S5.1)
- **`visibilitychange` Handler implementieren** -- Kein Handling fuer Background-Tabs. Export kann durch Tab-Wechsel fehlschlagen. (A4 S8, A4 S15.1, A3 S8)
- **Video-Element-Pooling** -- Jeder Clip erzeugt ein neues HTMLVideoElement. Bei 50+ Clips koennen Browser-Decoder-Limits erreicht werden. (A2 S1, A4 S3, A4 S15.3, A7 S2.4)
- **Decoder-Pool (`useDecoderPool`)** -- Shared Decoder fuer Clips mit demselben Source-File wuerde Decoder-Slots sparen. (A2 S9, A4 S3, A6 S14)

### Konsens-Staerke 3/7
- **Mobile-spezifische Resource-Limits** -- iOS Safari hat ~1 GB Memory-Limit. Keine reduzierten Cache-Budgets fuer mobile Geraete. (A4 S15.6, A1 S10.3, A6 S9)
- **GC-abhaengige Texture-Cleanup verbessern** -- GPU-Texturen werden nicht explizit zerstoert, sondern auf GC verlassen. Unter Speicherdruck kann VRAM akkumulieren. (A4 S15.2, A7 S4.2, A2 S3)
- **Render-Graph (`useRenderGraph`)** -- Automatische Resource-Scheduling und Barrier-Insertion fuer den Render-Pipeline. (A2 S9, A7 S2.1)
- **10-Bit / HDR Support** -- WebCodecs-Path unterstuetzt nur 8-Bit. (A5 S12, A6 S3.2)

### Konsens-Staerke 2/7
- **WebM/MKV Demuxing** -- MP4Box.js unterstuetzt nur ISO BMFF Container. WebM/MKV laufen nur ueber HTMLVideoElement. (A6 S4.1, A7 S6.6)
- **Blob-URL Leak Prevention erweitern** -- Einige Code-Pfade erzeugen Blob-URLs ausserhalb des BlobUrlManager. (A4 S10, A4 S15.5)
- **Linux Vulkan/VAAPI Robustheit** -- 15fps ohne Vulkan, VAAPI kann korrupte Frames produzieren. (A6 S8, A7 S6.4)
- **VFR-aware FPS Detection** -- FPS wird aus Container-Metadaten gelesen (Durchschnitt), nicht aus tatsaechlichen Sample-Timings. (A6 S5.2, A1 S2.1)

---

## 2. Priorisierte Roadmap

### Phase 1: Quick Wins (1-2 Wochen, hoher Impact)

#### 1.1 Dynamische VRAM-Budget-Anpassung
**Konsens:** 5/7 | **Impact:** Hoch | **Effort:** Niedrig | **Risiko:** Niedrig

**Problem:** ScrubbingCache allokiert bis zu 2.4 GB VRAM unabhaengig von der GPU. Integrierte GPUs (Intel/AMD mit 4 GB shared) laufen in Device-Loss.

**Loesung:**
- `engineStore` enthaelt bereits GPU-Info (`gpu.limits`, `gpu.adapterInfo`)
- ScrubbingCache-Limits basierend auf geschaetztem VRAM dynamisch setzen
- Heuristik: Integrierte GPU = 150 Frames max, Dedizierte GPU = 300 Frames, Low-End = 80 Frames
- GPU frame cache proportional anpassen (30/60/20)

**Betroffene Dateien:**
- `src/engine/texture/ScrubbingCache.ts` (Zeile 18-19: maxScrubbingCacheFrames, Zeile 44-45: maxGpuCacheFrames)
- `src/stores/engineStore.ts` (GPU-Info lesen)

**Erwarteter Benefit:** Eliminiert Device-Loss auf Low-VRAM-Systemen. Keine Regression auf High-End-Hardware.

---

#### 1.2 `visibilitychange` Handler fuer Export-Schutz
**Konsens:** 4/7 | **Impact:** Hoch | **Effort:** Niedrig | **Risiko:** Niedrig

**Problem:** Tab-Wechsel waehrend Export fuehrt zu RAF-Throttling, WebGPU-Device-Loss und fehlgeschlagenem Export. Einzige Massnahme: Error-Message "Try keeping the browser tab in focus."

**Loesung:**
- `document.addEventListener('visibilitychange', ...)` in `FrameExporter`
- Bei `document.hidden === true` waehrend Export: Export pausieren, User-Warning anzeigen
- Bei Rueckkehr: Export fortsetzen (GPU-Device pruefen, ggf. re-initialisieren)
- Optional: `navigator.wakeLock` API fuer Tab-im-Vordergrund-Garantie

**Betroffene Dateien:**
- `src/engine/export/FrameExporter.ts` (Export-Loop)
- `src/engine/render/RenderLoop.ts` (Idle-Detection)

**Erwarteter Benefit:** Robustere Exports, weniger User-Frustration bei versehentlichem Tab-Wechsel.

---

#### 1.3 Blob-URL Lifecycle Audit
**Konsens:** 2/7 | **Impact:** Mittel | **Effort:** Niedrig | **Risiko:** Niedrig

**Problem:** BlobUrlManager trackt URLs per Clip, aber `projectSlice.ts` und `importPipeline.ts` erzeugen URLs ausserhalb des Managers.

**Loesung:**
- Grep nach `URL.createObjectURL` ausserhalb von `BlobUrlManager`
- Alle Aufrufe ueber den Manager leiten oder explizite Revoke-Calls sicherstellen
- Devtools-Helper: `BlobUrlManager.getStats()` in Health-Monitor integrieren

**Betroffene Dateien:**
- `src/stores/timeline/helpers/blobUrlManager.ts`
- `src/stores/mediaStore/slices/fileManageSlice.ts`
- `src/services/project/` (verschiedene Import-Pfade)

**Erwarteter Benefit:** Eliminiert Memory-Leaks bei Langzeit-Sessions.

---

#### 1.4 Mobile Cache-Budget-Reduktion
**Konsens:** 3/7 | **Impact:** Mittel | **Effort:** Niedrig | **Risiko:** Niedrig

**Problem:** iOS Safari hat ~1 GB Memory-Limit, aber Cache-Budgets sind auf Desktop ausgelegt (2.4 GB VRAM, 512 MB RAM-Preview).

**Loesung:**
- Plattform-Erkennung (`navigator.maxTouchPoints > 0` + `navigator.userAgent`)
- Reduzierte Budgets: maxScrubbingCacheFrames=60, maxCompositeCacheFrames=200, maxCompositeCacheBytes=128MB
- Frueheres LRU-Eviction auf Mobile

**Betroffene Dateien:**
- `src/engine/texture/ScrubbingCache.ts`
- `src/components/mobile/` (bestehende Mobile-UI)

**Erwarteter Benefit:** Verhindert OOM-Crashes auf iPad/iPhone, ermoeglicht funktionale mobile NLE-Experience.

---

### Phase 2: Medium-Term (1-2 Monate, wichtige Verbesserungen)

#### 2.1 WebCodecs Full Playback aktivieren (Feature Flag)
**Konsens:** 7/7 | **Impact:** Sehr hoch | **Effort:** Mittel | **Risiko:** Mittel

**Problem:** `useFullWebCodecsPlayback: false` bedeutet, dass Preview auf HTMLVideoElement angewiesen ist mit all seinen Seeking-Limitationen (GOP-gebunden, ~300ms Seek-Latenz, readyState-Probleme).

**Loesung:**
- Schrittweise Aktivierung: Erst fuer einzelne Clips, dann fuer alle
- A/B-Test ueber Feature-Flag mit Telemetrie-Vergleich (RVFC-Timing, Drop-Rate, Seek-Latenz)
- Fallback: Automatischer Downgrade auf HTMLVideo wenn WebCodecs-Decoder fehlt
- Kritischer Test: Multi-Track Playback mit 3+ simultanen WebCodecs-Decodern

**Betroffene Dateien:**
- `src/engine/featureFlags.ts` (Flag aktivieren)
- `src/engine/WebCodecsPlayer.ts` (Stabilisierung fuer dauerhaften Einsatz)
- `src/services/layerBuilder/VideoSyncManager.ts` (Sync-Logik fuer WebCodecs-Modus)
- `src/engine/render/LayerCollector.ts` (Decoder-Prioritaet anpassen)

**Erwarteter Benefit:** Frame-genaues Seeking (<1ms statt 100-300ms), eliminiert GOP-Probleme, bessere Scrub-Performance. Laut Agent 1: "WebCodecs is the long-term solution."

**Risiken:**
- Mehr Decoder-Instanzen = hoehere GPU-Last
- MP4Box.js muss alle Container korrekt parsen (nur ISO BMFF)
- Memory-Management fuer VideoFrame.close() muss lueckenlos sein

---

#### 2.2 Video-Element-Pool & Decoder-Pool
**Konsens:** 4/7 (Element) + 4/7 (Decoder) | **Impact:** Hoch | **Effort:** Mittel | **Risiko:** Mittel

**Problem:** Jeder Clip erzeugt sein eigenes HTMLVideoElement. Bei 50+ Clips werden Browser-Decoder-Limits erreicht (Chrome: 6-16 HW-Streams).

**Loesung:**
- **Video-Element-Pool:** Pool von 8-12 HTMLVideoElements, dynamisch Clips zugewiesen
- **Decoder-Pool (`useDecoderPool`):** Shared WebCodecs VideoDecoder fuer Clips mit identischer Source-Datei
- Vorrang fuer sichtbare Clips (am Playhead), Off-Screen-Clips geben Element zurueck
- Handoff-Logik beibehalten fuer Split-Clips auf demselben Track

**Betroffene Dateien:**
- `src/stores/timeline/serializationUtils.ts` (Zeile 990: Video-Element-Erstellung)
- `src/services/layerBuilder/VideoSyncManager.ts` (Pool-Management)
- `src/engine/featureFlags.ts` (`useDecoderPool` Flag)

**Erwarteter Benefit:** Skaliert auf 100+ Clips ohne Decoder-Limit-Probleme. Reduziert DOM-Overhead.

**Risiken:**
- Pool-Miss bei schnellem Scrubbing durch viele Clips
- Warmup-Overhead beim Element-Recycling (GPU-Surface muss neu aktiviert werden)

---

#### 2.3 Code-Deduplizierung der Layer-Collection-Fallback-Kaskade
**Konsens:** 5/7 | **Impact:** Mittel | **Effort:** Mittel | **Risiko:** Mittel

**Problem:** Die Video-Fallback-Kaskade (scrub-cache -> seeking-cache -> drag-hold -> emergency-hold -> live-import -> copied-preview -> final-cache) ist dreifach dupliziert: LayerCollector (~370 Zeilen), NestedCompRenderer (~390 Zeilen), RenderDispatcher (~200 Zeilen). Fix in einem File muss in zwei anderen repliziert werden.

**Loesung:**
- Gemeinsame `VideoFrameResolver`-Klasse extrahieren
- Parameter fuer kontextspezifische Bedingungen (`allowLiveVideoImport`, `allowConfirmedFrameCaching`, `isNested`)
- Alle drei Consumer nutzen dieselbe Resolver-Instanz

**Betroffene Dateien:**
- `src/engine/render/LayerCollector.ts` (tryHTMLVideo Methode)
- `src/engine/render/NestedCompRenderer.ts` (collectNestedLayerData)
- `src/engine/render/RenderDispatcher.ts` (renderToPreviewCanvas)
- Neu: `src/engine/render/VideoFrameResolver.ts`

**Erwarteter Benefit:** Einfachere Wartung, konsistentes Verhalten ueber alle Render-Pfade, weniger Regression-Risiko bei Bug-Fixes.

**Risiken:**
- Abstraktion koennte kontextspezifische Optimierungen erschweren
- Gruendliches Testing noetig (alle drei Pfade muessen identisch funktionieren)

---

#### 2.4 FFmpeg WASM Streaming-Export
**Konsens:** 4/7 | **Impact:** Hoch | **Effort:** Mittel-Hoch | **Risiko:** Mittel

**Problem:** FFmpeg-Path laedt alle Frames als einzelnes `Uint8Array` in RAM. 60s @ 1080p30 = 14.9 GB -- unpraktisch fuer laengere Exporte.

**Loesung:**
- Streaming-Ansatz: Frames inkrementell in FFmpeg-Virtual-Filesystem schreiben
- Pipe-basiert: Named Pipe oder sequentielle Datei-Schreibvorgaenge waehrend Encoding
- Alternative: FFmpeg WASM Multi-Thread Build (wenn Emscripten Shared Memory erlaubt)
- Frame-Buffer von max. 30 Frames statt aller Frames

**Betroffene Dateien:**
- `src/engine/ffmpeg/FFmpegBridge.ts` (Frame-Input-Logik, Zeile ~allFrames)
- `src/engine/export/FrameExporter.ts` (Export-Loop)

**Erwarteter Benefit:** ProRes/DNxHR-Export fuer laengere Projekte (30min+) moeglich. Reduziert Peak-Memory von GB auf MB.

**Risiken:**
- FFmpeg WASM ASYNCIFY-Build hat Limitierungen bei Pipe-I/O
- Timing-Koordination zwischen Frame-Rendering und FFmpeg-Encoding komplex

---

#### 2.5 GC-sichere Texture-Cleanup-Strategie
**Konsens:** 3/7 | **Impact:** Mittel | **Effort:** Mittel | **Risiko:** Mittel

**Problem:** GPU-Texturen werden absichtlich nicht explizit zerstoert ("Don't destroy -- let GC handle"), um Use-after-destroy Crashes zu vermeiden. Unter Speicherdruck kann VRAM akkumulieren.

**Loesung:**
- Deferred-Destroy-Queue: Texturen nach 2 Frames delay zerstoeren (garantiert keine GPU-Referenz mehr)
- `FinalizationRegistry` fuer automatisches Cleanup wenn JS-Objekte GC'd werden
- Explizites Destroy fuer bekannte Szenarien: Composition-Switch, Project-Close

**Betroffene Dateien:**
- `src/engine/texture/TextureManager.ts` (clearCaches Methode)
- `src/engine/texture/ScrubbingCache.ts` (clearScrubbingCache)
- `src/engine/render/NestedCompRenderer.ts` (cleanupTexture)

**Erwarteter Benefit:** Reduzierter VRAM-Footprint in Langzeit-Sessions, weniger Device-Loss.

**Risiken:**
- Zu fruehes Destroy verursacht GPU-Crashes (STATUS_BREAKPOINT)
- FinalizationRegistry hat keine garantierte Timing

---

### Phase 3: Long-Term (3-6 Monate, architektonische Aenderungen)

#### 3.1 Render-Graph-Architektur
**Konsens:** 3/7 | **Impact:** Sehr hoch | **Effort:** Sehr hoch | **Risiko:** Hoch

**Problem:** Die aktuelle Render-Pipeline ist imperativ: LayerCollector -> Compositor -> OutputPipeline mit manueller Resource-Verwaltung. Keine automatische Barrier-Insertion, keine Parallelisierung von unabhaengigen Render-Passes.

**Loesung:**
- Deklarativer Render-Graph: Nodes (Passes) und Edges (Resourcen-Abhaengigkeiten)
- Automatische Texture-Lifetime-Tracking und Aliasing
- GPU-Command-Buffer-Batching basierend auf Graph-Analyse
- Aktivierung ueber `useRenderGraph` Feature-Flag

**Betroffene Dateien:**
- Neues Modul: `src/engine/renderGraph/`
- `src/engine/render/RenderDispatcher.ts` (Refactoring)
- `src/engine/render/Compositor.ts` (Integration)
- `src/engine/render/NestedCompRenderer.ts` (Integration)

**Erwarteter Benefit:** Optimierte GPU-Auslastung, automatische Resource-Verwaltung, sauberere Architektur. Eliminiert die Ping-Pong-Verwaltung als manuelle Logik.

**Risiken:**
- Massiver Refactor der gesamten Render-Pipeline
- Render-Graph muss alle Edge-Cases (Nested Comps, Multi-Preview, Export) abdecken
- Performance-Regression moeglich wenn Graph-Overhead die Optimierungen uebersteigt

---

#### 3.2 WebM/MKV Demuxing-Support
**Konsens:** 2/7 | **Impact:** Mittel | **Effort:** Hoch | **Risiko:** Mittel

**Problem:** MP4Box.js unterstuetzt nur ISO BMFF Container (MP4, MOV, M4V). WebM/MKV-Dateien laufen nur ueber HTMLVideoElement, kein WebCodecs-Decoding moeglich.

**Loesung:**
- Integration eines EBML/Matroska-Demuxers (z.B. `ebml-parser` oder `matroska-demuxer`)
- Oder: `jsmkvdemuxer` fuer VP8/VP9/AV1-Streams in WebM-Containern
- Unified Demuxer-Interface: `DemuxerInterface.getSamples(trackId)` fuer beide Container-Typen

**Betroffene Dateien:**
- `src/engine/WebCodecsPlayer.ts` (MP4Box-spezifischer Code abstrahieren)
- `src/stores/timeline/helpers/audioDetection.ts` (EBML-Parser bereits vorhanden)
- Neu: `src/engine/demuxer/` Modul

**Erwarteter Benefit:** WebCodecs-Decoding fuer WebM/MKV-Dateien, konsistentes Verhalten ueber alle Container.

---

#### 3.3 10-Bit / HDR Pipeline
**Konsens:** 3/7 | **Impact:** Mittel | **Effort:** Sehr hoch | **Risiko:** Hoch

**Problem:** Gesamte Pipeline ist auf 8-Bit RGBA (`rgba8unorm`) ausgelegt. Kein HDR-Tone-Mapping, keine 10-Bit-Texturformate.

**Loesung:**
- `rgba16float` Texturen fuer interne Render-Targets
- HDR-aware Compositor-Shader (PQ/HLG Transfer-Funktionen)
- `VideoEncoder` mit 10-Bit Profilen (AV1 Main 10, HEVC Main 10)
- Tone-Mapping fuer SDR-Preview von HDR-Content

**Betroffene Dateien:**
- `src/engine/core/RenderTargetManager.ts` (Textur-Formate)
- `src/shaders/composite/` (HDR-Shader)
- `src/engine/export/codecHelpers.ts` (10-Bit Codec-Strings)
- `src/engine/export/VideoEncoderWrapper.ts`

**Erwarteter Benefit:** Professionelle HDR-Workflows, zukunftssicher fuer iPhone/Kamera-Content.

**Risiken:**
- Browser-Support fuer 10-Bit Encoding ist limitiert
- Doppelter VRAM-Verbrauch fuer 16-Bit Texturen
- Alle Effects muessen HDR-kompatibel werden

---

#### 3.4 Shared Worker / OffscreenCanvas Export
**Konsens:** 2/7 | **Impact:** Hoch | **Effort:** Hoch | **Risiko:** Hoch

**Problem:** Export laeuft im Main-Thread und ist anfaellig fuer Tab-Throttling, UI-Blockierung, und Device-Loss bei Tab-Wechsel.

**Loesung:**
- WebGPU in Shared Worker (wenn Browser-Support ausreicht)
- OffscreenCanvas fuer Export-Rendering in Worker-Thread
- Main-Thread bleibt frei fuer UI-Interaktion waehrend Export

**Betroffene Dateien:**
- `src/engine/export/FrameExporter.ts`
- `src/engine/managers/ExportCanvasManager.ts`
- Neu: `src/workers/exportWorker.ts`

**Erwarteter Benefit:** Export blockiert nicht die UI, Tab-Wechsel hat keinen Einfluss, bessere UX.

**Risiken:**
- WebGPU in Workers hat noch limitierten Browser-Support
- GPU-Device-Sharing zwischen Main-Thread und Worker ist komplex
- Zustand-Store-Zugriff aus Worker erfordert Message-Passing

---

## 3. Risiko-Bewertung der kritischsten Aenderungen

### Hoechstes Risiko: WebCodecs Full Playback (Phase 2.1)
- **Was koennte breaken:** Multi-Track Playback mit 4+ Videos. Jeder Clip benoetigt einen eigenen VideoDecoder. Hardware-Decoder-Limits (4-8 auf Linux, 6-16 auf Windows) koennten schneller erreicht werden als mit HTMLVideoElement.
- **VideoFrame.close() Luecken:** Ein einziges nicht geschlossenes VideoFrame fuehrt zu unbegrenztem VRAM-Wachstum. Die Close-Disziplin muss lueckenlos sein.
- **Container-Kompatibilitaet:** MP4Box.js muss alle User-Dateien korrekt parsen. Korrupte oder ungewoehnliche Container fuehren zu schwarzen Frames statt Fallback.
- **Mitigation:** Feature-Flag mit automatischem Fallback, Telemetrie-Vergleich, progressiver Rollout.

### Hohes Risiko: Render-Graph (Phase 3.1)
- **Was koennte breaken:** Gesamte Render-Pipeline. Jeder Edge-Case (Nested Comps, Multi-Preview, Export, Effects-Pipeline) muss korrekt im Graph abgebildet werden.
- **Mitigation:** Parallel-Implementierung neben bestehendem System, Feature-Flag, umfangreiche Visual-Regression-Tests.

### Mittleres Risiko: FFmpeg Streaming-Export (Phase 2.4)
- **Was koennte breaken:** ProRes/DNxHR-Export. FFmpeg WASM ASYNCIFY-Build hat Limitierungen bei inkrementellem I/O.
- **Mitigation:** Bestehenden Batch-Export als Fallback beibehalten, neuen Streaming-Pfad nur fuer laengere Exporte nutzen.

### Mittleres Risiko: Code-Deduplizierung (Phase 2.3)
- **Was koennte breaken:** Subtile Verhaltensunterschiede zwischen den drei Render-Pfaden. LayerCollector, NestedCompRenderer und RenderDispatcher haben leicht unterschiedliche Bedingungen fuer Cache-Zugriff und Live-Import.
- **Mitigation:** Gruendliche Diff-Analyse der drei Implementierungen, umfangreiche A/B-Tests.

---

## 4. Zusammenfassung: Die Top-10 nach Impact/Effort-Ratio

| # | Verbesserung | Phase | Konsens | Impact | Effort | Ratio |
|---|-------------|-------|---------|--------|--------|-------|
| 1 | Dynamische VRAM-Budgets | 1.1 | 5/7 | Hoch | Niedrig | Sehr gut |
| 2 | `visibilitychange` Handler | 1.2 | 4/7 | Hoch | Niedrig | Sehr gut |
| 3 | Mobile Cache-Budgets | 1.4 | 3/7 | Mittel | Niedrig | Gut |
| 4 | Blob-URL Audit | 1.3 | 2/7 | Mittel | Niedrig | Gut |
| 5 | WebCodecs Full Playback | 2.1 | 7/7 | Sehr hoch | Mittel | Gut |
| 6 | Layer-Collection Deduplizierung | 2.3 | 5/7 | Mittel | Mittel | Mittel |
| 7 | Video/Decoder-Pool | 2.2 | 4/7 | Hoch | Mittel | Mittel |
| 8 | FFmpeg Streaming-Export | 2.4 | 4/7 | Hoch | Mittel-Hoch | Mittel |
| 9 | GC-sichere Texture-Cleanup | 2.5 | 3/7 | Mittel | Mittel | Mittel |
| 10 | Render-Graph | 3.1 | 3/7 | Sehr hoch | Sehr hoch | Langfristig |

---

## 5. Sofort-Empfehlung

Die vier Quick Wins aus Phase 1 (VRAM-Budgets, `visibilitychange`, Blob-URL-Audit, Mobile-Budgets) haben gemeinsam:
- Geringstes Regressions-Risiko
- Keine architektonischen Aenderungen
- Direkt messbare Verbesserung (weniger Crashes, weniger Memory-Leaks)
- Koennen parallel implementiert werden

Danach sollte WebCodecs Full Playback (Phase 2.1) als strategische Hauptverbesserung priorisiert werden, da **alle 7 Agents** es als die wichtigste langfristige Verbesserung identifiziert haben. Agent 1 fasst es praegnant zusammen: "WebCodecs is the long-term solution -- provides frame-level decode control, but requires significant infrastructure." Diese Infrastruktur ist in MasterSelects bereits implementiert (WebCodecsPlayer, MP4Box-Integration, Frame-Buffer-Management) -- sie muss nur stabilisiert und fuer den Default-Betrieb freigeschaltet werden.

---

## 6. Agent-uebergreifende Beobachtungen

### Architektonische Staerken (alle Agents einig)
- Die Multi-Tier-Caching-Architektur (Scrubbing, Last-Frame, Composite, GPU-Cache) ist ausgezeichnet konzipiert
- Die Fallback-Kette (NativeHelper -> WebCodecs -> HTMLVideo -> Cache) degradiert graceful
- Zero-copy GPU-Pfade (`importExternalTexture`) sind korrekt implementiert
- PlaybackHealthMonitor mit Eskalations-Protokoll ist ein professionelles Feature
- Triple-buffered Seek-Targets verhindern Seek-Verlust bei schnellem Scrubbing

### Architektonische Schwaechen (mehrere Agents uebereinstimmend)
- HTMLVideoElement als Default-Decoder ist das groesste einzelne Performance-Bottleneck
- VRAM-Budgets sind statisch und nicht an Hardware angepasst
- Code-Duplikation in der Layer-Collection erhoet Wartungsaufwand
- FFmpeg-Export skaliert nicht fuer laengere Projekte
- Kein Background-Tab-Handling gefaehrdet laufende Exporte

### Was NICHT geaendert werden sollte
- Die Audio-Master-Clock-Strategie (Agent 3): Audio als Timing-Referenz ist korrekt
- Die HMR-Singleton-Pattern (Agent 7): Essentiell fuer Development-Workflow
- Die Seamless-Cut-Transitions (Agent 3, 7): Professionelle Qualitaet
- Die RVFC-basierte Frame-Confirmation (Agent 1, 3, 7): Zuverlaessiger als `seeked` Event
- Die Hybrid-Seek-Strategie (Agent 1, 3, 6): fastSeek + debounced precise seek ist optimal
