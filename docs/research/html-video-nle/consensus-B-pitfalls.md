# Consensus Report B: Fallstricke, Bugs & Loesungen

## HTMLVideoElement in einer Browser-NLE -- Die 20 gefaehrlichsten Pitfalls

**Methodik:** Analyse von 7 unabhaengigen Research-Agents. Jeder Fallstrick wird mit Severity bewertet und es wird angegeben, welche Agents ihn identifiziert haben. Konsens = 3+ Agents nennen das gleiche Problem. Starker Konsens = 5+ Agents.

---

## Tier 1: CRITICAL -- Anwendung crasht oder ist unbenutzbar

### Pitfall #1: GPU Surface Cold Start nach Page Reload
**Severity:** CRITICAL
**Konsens:** Starker Konsens (Agents 1, 2, 3, 6, 7)

**Problem:** Nach einem Page-Reload sind die GPU-Surfaces von HTMLVideoElements leer. `importExternalTexture()`, `canvas.drawImage()`, `copyExternalImageToTexture()` und sogar `new VideoFrame(video)` liefern schwarze Frames -- obwohl `readyState >= 2` meldet, dass Daten verfuegbar sind.

**Warum:** Chrome deferred die Video-Frame-Dekodierung bis `play()` aufgerufen wird. Die GPU-Decoder-Surface wird erst durch tatsaechliche Playback-Aktivitaet aktiviert. Das ist kein Bug, sondern ein Optimierungs-Design in Chromium.

**MasterSelects-Loesung:**
- `VideoSyncManager.startTargetedWarmup()`: Ruft `video.play()` auf, wartet auf RVFC-Bestaetigung, pausiert dann und cached den Frame
- `LayerCollector.videoGpuReady` (WeakSet): Trackt welche Videos eine aktive GPU-Surface haben
- `ScrubbingCache.captureVideoFrameViaImageBitmap()`: Verwendet `createImageBitmap()` als einzige API die Chrome zum Decodieren zwingt
- `RenderLoop.idleSuppressed`: Verhindert Idle-Detection waehrend Warmup

**Best Practice:** Immer ein play/pause-Warmup nach Reload durchfuehren. `createImageBitmap(video)` als erzwungenen Decode-Pfad nutzen. Nie annehmen, dass `readyState >= 2` GPU-Readiness bedeutet.

---

### Pitfall #2: VideoFrame.close() vergessen fuehrt zu VRAM-Leak und STATUS_BREAKPOINT
**Severity:** CRITICAL
**Konsens:** Agents 4, 6, 7

**Problem:** `VideoFrame`-Objekte (WebCodecs) halten GPU-seitige Pixeldaten. Der Garbage Collector gibt diesen VRAM NICHT frei -- man muss `.close()` explizit aufrufen. Vergisst man es, waechst der VRAM unbegrenzt und Chrome crasht mit STATUS_BREAKPOINT.

**Warum:** VideoFrame ist eine Ressource mit externalem Speicher (GPU/Decoder-Buffer). Im Gegensatz zu normalen JS-Objekten reicht GC nicht aus, weil die GPU-Referenz explizit freigegeben werden muss.

**MasterSelects-Loesung:**
- `WebCodecsPlayer`: Schliesst Frames an jedem Lifecycle-Punkt: Buffer-Overflow-Eviction, Seek-Intermediate-Frames, Frame-Presentation, Destroy-Path
- `TextureManager.importVideoTexture()`: Guard gegen geschlossene Frames (`source.closed || source.codedWidth === 0`) -- ein geschlossenes Frame an `importExternalTexture` uebergeben crasht den GPU-Prozess
- `FrameExporter`: `videoFrame.close()` sofort nach Encoding
- `MAX_FRAME_BUFFER = 8`: Limitiert decoded-but-not-displayed Frames auf ~64MB

**Best Practice:** VideoFrame.close() in JEDEM Code-Pfad aufrufen (Success, Error, Timeout). Defensive Guards vor GPU-API-Aufrufen. Frame-Buffer mit harten Limits.

---

### Pitfall #3: Geschlossenes VideoFrame an importExternalTexture crasht GPU-Prozess
**Severity:** CRITICAL
**Konsens:** Agents 4, 6, 7

**Problem:** Ein `VideoFrame` das bereits `.close()` aufgerufen hat, an `device.importExternalTexture()` zu uebergeben, crasht den GPU-Prozess mit STATUS_BREAKPOINT. Es gibt keine Exception -- der Prozess stirbt einfach.

**Warum:** WebGPU validiert die Source, aber ein geschlossenes VideoFrame hat ungueltige interne Handles. Die Validierung greift nicht zuverlaessig vor dem nativen GPU-Aufruf.

**MasterSelects-Loesung:**
```typescript
if ((source as any).closed || source.codedWidth === 0 || source.codedHeight === 0) {
  return null;
}
```

**Best Practice:** Immer `.closed`-Property pruefen UND `codedWidth/codedHeight > 0` validieren bevor ein VideoFrame an jede GPU-API uebergeben wird.

---

### Pitfall #4: WebGPU Device Loss bei Tab-Wechsel waehrend Export
**Severity:** CRITICAL
**Konsens:** Agents 4, 5, 7

**Problem:** Wenn der User waehrend eines Exports den Browser-Tab wechselt, kann Chrome den WebGPU-Device verlieren. RAF wird auf ~1fps gedrosselt, und der GPU-Prozess kann die Ressourcen freigeben. Der Export schlaegt fehl.

**Warum:** Chrome drosselt Background-Tabs aggressiv. RAF-basierte Loops laufen nur noch mit ~1fps, setTimeout/setInterval werden verzoegert, und der GPU kann als nicht mehr genutzt eingestuft werden.

**MasterSelects-Loesung:**
- `WebGPUContext`: Device-Loss-Recovery mit bis zu 3 Versuchen (100ms Delay, Re-Initialize)
- `FrameExporter`: Erkennt Device-Loss und zeigt "Try keeping the browser tab in focus"
- ABER: Kein `visibilitychange`-Handler (Agent 4 bestaetigt: 0 Treffer im Code)

**Noch nicht geloest:** Es gibt keine programmatische Prevention. Kein `visibilitychange`-Event-Handler der den User warnt oder den Export pausiert. Agent 4 identifiziert dies als offenen Gap.

**Best Practice:** `visibilitychange`-Event nutzen um Export zu pausieren. Alternativ: Web Workers + OffscreenCanvas fuer GPU-Arbeit die nicht an Tab-Visibility gebunden ist.

---

### Pitfall #5: FFmpeg WASM All-in-RAM Bottleneck
**Severity:** CRITICAL
**Konsens:** Agents 5, 7

**Problem:** Der FFmpeg-WASM-Pfad laedt ALLE Raw-Frames in einen einzigen `Uint8Array` bevor das Encoding beginnt. Fuer 60 Sekunden 1080p @ 30fps: `1920 * 1080 * 4 * 1800 = ~14.9 GB`. Das uebersteigt den Browser-Heap.

**Warum:** Die FFmpegBridge schreibt alle Frames als einen zusammenhaengenden Raw-RGBA-Block ins virtuelle Filesystem. Es gibt kein Streaming.

**MasterSelects-Loesung:** Keine vollstaendige Loesung. Der FFmpeg-Pfad ist praktisch nur fuer kurze Sequenzen nutzbar.

**Noch nicht geloest:** Streaming-Approach (inkrementelles Frame-Writing) fehlt. Fuer lange ProRes/DNxHR-Exports ein fundamentales Limit.

**Best Practice:** FFmpeg-WASM mit inkrementellem Frame-Writing (Pipe oder Frame-fuer-Frame Filesystem-Writes). Alternativ: WebCodecs-Pfad fuer alle Codecs wo moeglich.

---

## Tier 2: HIGH -- Signifikante User-Experience-Probleme

### Pitfall #6: Race Conditions bei Rapid Scrubbing (Seek Thrashing)
**Severity:** HIGH
**Konsens:** Starker Konsens (Agents 1, 2, 3, 6, 7)

**Problem:** Schnelles Scrubben generiert Dutzende Seek-Requests pro Sekunde. Jedes `video.currentTime = X` bricht den vorherigen Seek ab. Der Browser-Decoder wird in einen unvorhersagbaren Zustand versetzt, `seeked`-Events feuern nicht oder out-of-order, die GPU-Surface zeigt falsche Frames.

**Warum:** HTMLVideoElement wurde fuer lineares Playback designed, nicht fuer Random Access. Jeder Seek muss vom naechsten Keyframe dekodieren, und ueberlappende Seeks korrumpieren den internen State.

**MasterSelects-Loesung:**
- Triple-Buffer Seek-Tracking: `latestSeekTargets`, `pendingSeekTargets`, `queuedSeekTargets`
- Seek-Coalescing: Neue Seeks waehrend laufendem Seek werden gequeued, nicht sofort ausgefuehrt
- `shouldRetargetPendingSeek()`: Retargeting erst nach Age + Drift Thresholds
- Seek-Flush via RVFC + seeked-Event als Fallback

**Best Practice:** Seeks serialisieren mit Queue und Coalescing. Nie mehr als einen Seek gleichzeitig pro Video-Element. fastSeek fuer Preview, debounced precise Seek fuer finales Frame.

---

### Pitfall #7: GOP-Structure macht Seeking inherent langsam
**Severity:** HIGH
**Konsens:** Starker Konsens (Agents 1, 3, 6, 7)

**Problem:** Video-Codecs (H.264, H.265) organisieren Frames in GOPs. Nur I-Frames enthalten vollstaendige Bilddaten. Um Frame 150 in einem 250-Frame GOP zu dekodieren, muessen ~150 Frames sequentiell dekodiert werden. YouTube/Handy-Videos haben 5-7 Sekunden Keyframe-Intervalle, was 150-210 Frames Decode-Arbeit pro Seek bedeutet.

**Warum:** Das ist fundamentales Codec-Design. I-Frames sind gross, P/B-Frames sind klein. Mehr Kompression = weniger Keyframes = langsameres Seeking.

**MasterSelects-Loesung:**
- Phase 1: `fastSeek()` fuer sofortiges Keyframe-Preview (<10ms)
- Phase 2: Debounced precise Seek (120ms) fuer exaktes Frame wenn Scrubbing pausiert
- Adaptive Throttle basierend auf fastSeek-Support und Drift-Groesse
- WebCodecs Full Mode (Feature-flagged): Binary Search nach naechstem Keyframe + sequentielles Decode

**Best Practice:** Hybrid-Seeking (fastSeek + debounced precise). Fuer professionelle Workflows: All-Intra Codecs (ProRes, DNxHR) verwenden die kein GOP-Problem haben.

---

### Pitfall #8: Firefox importExternalTexture Black Frames
**Severity:** HIGH
**Konsens:** Agents 1, 6, 7

**Problem:** Firefox sampelt `importExternalTexture()` von HTMLVideoElements intermittierend als schwarz waehrend Playback. Das Zero-Copy-Rendering ist auf Firefox unzuverlaessig.

**Warum:** Firefox-interner Bug in der WebGPU-Implementierung. Die Video-Texture-Referenz wird nicht zuverlaessig an den GPU-Sampler weitergegeben.

**MasterSelects-Loesung:**
- `htmlVideoPreviewFallback.ts`: Browser-Detection via UserAgent
- Auf Firefox wird jedes Frame via `copyExternalImageToTexture` in eine persistente `texture_2d` kopiert
- Performance-Cost: Kein Zero-Copy mehr, aber stabil

**Best Practice:** Firefox-Detection + Fallback auf kopierten Texture-Pfad. Performance-Regression akzeptieren fuer Korrektheit.

---

### Pitfall #9: Scrubbing Cache VRAM-Budget sprengt integrierte GPUs
**Severity:** HIGH
**Konsens:** Agents 2, 4, 7

**Problem:** Der ScrubbingCache haelt bis zu 300 Frames bei 1080p -- das sind ~2.4 GB VRAM. Integrierte GPUs haben oft nur 2-4 GB geteilten VRAM. Das kann Device Loss ausloesen.

**Warum:** Feste Cache-Budgets die nicht an die tatsaechlich verfuegbare VRAM angepasst werden.

**MasterSelects-Loesung:** Feste Limits (`maxScrubbingCacheFrames = 300`), LRU-Eviction.

**Noch nicht geloest:** Keine dynamische Budget-Anpassung basierend auf GPU-Info. Agent 4 und 7 identifizieren dies als offenen Gap. `engineStore` hat GPU-Info, aber sie wird nicht fuer Cache-Sizing genutzt.

**Best Practice:** GPU-VRAM zur Laufzeit abfragen (via `adapter.requestAdapterInfo()`) und Cache-Budget dynamisch anpassen. Aggressive Limits auf integrierten GPUs.

---

### Pitfall #10: currentTime-Praezision ist codec- und browser-abhaengig
**Severity:** HIGH
**Konsens:** Starker Konsens (Agents 1, 3, 7)

**Problem:** `video.currentTime` gibt NICHT den exakt angezeigten Frame wieder. Es reportet den Presentation Timestamp des zuletzt dekodierten Frames, was von GOP-Struktur, Browser-Implementation und Decoder-State abhaengt. Exakte Gleichheit (`===`) ist nie zuverlaessig.

**Warum:** HTMLVideoElement abstrahiert die Decoder-Ebene. Der Spec definiert keine Frame-Genauigkeit fuer currentTime.

**MasterSelects-Loesung:**
- Toleranz-basierte Vergleiche: 15ms (paused), 20ms (precise), 40ms (drag), 300ms (playback drift)
- `lastPresentedTime` Tracking via ScrubbingCache als separate "was haben wir wirklich gezeigt" Signal
- RVFC als definitive Frame-Presentation-Bestaetigung

**Best Practice:** NIE `currentTime` mit `===` vergleichen. Immer Toleranzen verwenden. RVFC als Frame-Ready-Signal nutzen statt `seeked`-Event.

---

### Pitfall #11: Browser Hardware Decoder Limits
**Severity:** HIGH
**Konsens:** Agents 2, 4, 7

**Problem:** Browser haben harte Limits fuer gleichzeitige HW-dekodierte Video-Streams. Chrome Windows: 6-16, Firefox: 4-8, Linux: 2-4. Wird das Limit ueberschritten, faellt der Browser auf Software-Decoding zurueck (deutlich langsamer) oder droppt Frames.

**Warum:** Hardware-Video-Decoder (NVDEC, Intel QSV, DXVA) haben physische Limits an gleichzeitigen Sessions.

**MasterSelects-Loesung:**
- Nur Clips am Playhead haben aktive Video-Elemente (`.play()`)
- Off-Screen Videos werden pausiert (`video.pause()`)
- Handoff-System fuer Same-Source Split-Clips: Teilen sich ein Video-Element
- Kein Video-Element-Pooling (Agent 4: "There is no element pool")

**Noch nicht geloest:** Kein explizites Pooling. Bei 50+ Clips in einer Timeline koennen Browser-Decoder-Limits erreicht werden (Agent 4 Risiko-Assessment).

**Best Practice:** Decoder-Pool implementieren. Video-Elemente wiederverwenden statt pro Clip neu erstellen. Maximal aktive Decoder tracken.

---

### Pitfall #12: Stuck Seeks (video.seeking bleibt true)
**Severity:** HIGH
**Konsens:** Agents 1, 3, 4, 7

**Problem:** In seltenen Faellen bleibt `video.seeking === true` haengen (>2 Sekunden). Der Browser-Decoder ist in einem Deadlock-Zustand. Das Video zeigt ein eingefrorenes oder falsches Frame.

**Warum:** Race Conditions im Browser-internen Seek-Pipeline, besonders bei schnellen aufeinanderfolgenden Seeks. Auch bei H.264 B-Frame-Decodern die am Ende des Videos auf nicht-existente Referenzframes warten.

**MasterSelects-Loesung:**
- `PlaybackHealthMonitor`: Erkennt `video.seeking === true` fuer > 2s
- Recovery: Re-Seek (`video.currentTime = video.currentTime`)
- `safeSeekTime()`: Clampt auf `duration - 0.001` um B-Frame EOF-Stalls zu vermeiden
- Escalation nach 3 Anomalien in 12s: Full Clip Recovery (GPU Reset + Re-Seek)

**Best Practice:** Stuck-Seek-Detection mit Timeout. Immer von exaktem `duration`-Ende wegclampen. Play/Pause-Cycle als letzte Recovery-Option.

---

## Tier 3: MEDIUM -- Spuerbare Qualitaetseinbussen

### Pitfall #13: seeked-Event ist nicht ausreichend fuer Frame-Readiness
**Severity:** MEDIUM
**Konsens:** Agents 1, 3, 7

**Problem:** Das `seeked`-Event signalisiert nur, dass der Browser den Seek beendet hat -- NICHT dass das Frame auf der GPU-Surface composited ist. Zwischen `seeked` und tatsaechlicher GPU-Verfuegbarkeit koennen mehrere Millisekunden liegen.

**Warum:** Die Browser-Pipeline hat mehrere Stufen: Decode -> Composite -> GPU Surface. `seeked` feuert nach Decode, nicht nach GPU-Presentation.

**MasterSelects-Loesung:**
- `requestVideoFrameCallback` (RVFC) als primaeres Frame-Ready-Signal
- `seeked` nur als Fallback fuer Queue-Flushing
- `armSeekedFlush()`: One-shot seeked-Listener als Insurance

**Best Practice:** RVFC > seeked > timeupdate als Signal-Hierarchie. seeked fuer Seek-Queue-Management, RVFC fuer Frame-Presentation-Bestaetigung.

---

### Pitfall #14: Variable Frame Rate (VFR) Videos
**Severity:** MEDIUM
**Konsens:** Agents 6, 7

**Problem:** VFR-Videos (Screen Recordings, Handy-Videos) haben keine konstante Frame-Duration. Die aus dem Container gelesene FPS ist ein Durchschnitt, nicht die tatsaechliche Rate. Feste Seek-Toleranzen basierend auf nomineller FPS koennen Frames verfehlen.

**Warum:** VFR ist Standard bei Bildschirmaufnahmen und Smartphone-Kameras. Die tatsaechliche Frame-Duration variiert stark.

**MasterSelects-Loesung:**
- `WebCodecsPlayer.computeSeekToleranceUs()`: Misst den tatsaechlichen Abstand zwischen benachbarten Samples statt feste Toleranz
- Adaptive Toleranz: `Math.max(2000us, Math.min(200000us, vfrAwareUs))`

**Best Practice:** Seek-Toleranz dynamisch aus tatsaechlichen Sample-Timings berechnen. Nie nur auf Container-FPS vertrauen.

---

### Pitfall #15: Linux Performance (15fps ohne Vulkan)
**Severity:** MEDIUM
**Konsens:** Agents 2, 6, 7

**Problem:** Chrome auf Linux ohne Vulkan erreicht nur ~15fps. VAAPI-Video-Acceleration kann korrupte Frames produzieren oder komplett fehlschlagen.

**Warum:** Chromium's GPU-Backend auf Linux ohne Vulkan ist deutlich langsamer. VAAPI-Integration ist fehleranfaellig.

**MasterSelects-Loesung:**
- `LinuxVulkanWarning.tsx`: Explizite Warnung an den User
- WebCodecsPlayer-Header: "Bypasses browser VAAPI issues by using WebCodecs API directly"
- Vulkan-spezifische Delays bei GPU-Initialisierung (50ms/100ms/50ms Settling-Time)

**Best Practice:** Linux-User auf `chrome://flags/#enable-vulkan` hinweisen. WebCodecs als VAAPI-Bypass anbieten.

---

### Pitfall #16: GC-basierte GPU-Texture-Cleanup
**Severity:** MEDIUM
**Konsens:** Agents 4, 7

**Problem:** Die meisten GPU-Textures werden NICHT explizit mit `.destroy()` freigegeben, sondern dem GC ueberlassen. Unter hohem Speicherdruck kann GC nicht schnell genug laufen, und VRAM akkumuliert bevor JS-GC-Zyklen greifen.

**Warum:** Explizites `.destroy()` waehrend die GPU die Texture noch referenziert (z.B. in einem pending Command Buffer) verursacht Validation Errors oder Crashes. Der sichere Ansatz ist GC-basiert.

**MasterSelects-Loesung:**
- Bewusste Design-Entscheidung: "Don't destroy textures - let GC handle to avoid GPU conflicts"
- Ausnahme: `dynamicTextures` (NativeDecoder) werden explizit destroyed weil der Lifecycle vollstaendig kontrolliert wird

**Noch nicht geloest:** Unter extremem VRAM-Druck kann der GC zu langsam sein. Kein Mechanismus um VRAM-Pressure zu messen und proaktiv aufzuraeumen.

**Best Practice:** Wo moeglich explizites Lifecycle-Management. Wo nicht moeglich: GC-basiert aber mit VRAM-Budget-Monitoring. `performance.measureUserAgentSpecificMemory()` als Indikator nutzen.

---

### Pitfall #17: Autoplay-Restriktionen auf Mobile
**Severity:** MEDIUM
**Konsens:** Agents 1, 6, 7

**Problem:** Mobile Browser (und teilweise Desktop) blockieren `video.play()` ohne User-Geste. Die Warmup-Sequenz (play/pause) schlaegt fehl wenn kein User-Gesture-Context vorhanden ist.

**Warum:** Browser-Policy gegen ungewollte Audio/Video-Wiedergabe. Ohne User-Interaktion wird `play()` rejected.

**MasterSelects-Loesung:**
- `video.muted = true` vor Warmup (erfuellt die meisten Autoplay-Policies)
- `video.playsInline = true`
- `.play().catch(() => {})` fuer graceful Handling

**Best Practice:** Videos immer muted fuer Warmup. `playsInline` setzen. `play()`-Promise immer catchen. AudioContext erst nach User-Geste resumieren.

---

### Pitfall #18: Blob URL Leaks
**Severity:** MEDIUM
**Konsens:** Agents 4, 7

**Problem:** Jede Blob-URL (`URL.createObjectURL(file)`) haelt die gesamte Datei im Speicher. Nicht revoked Blob-URLs fuer 1GB+ Video-Dateien fuehren zu massivem Speicherverbrauch.

**Warum:** Blob-URLs halten eine Referenz auf den Blob. Der Blob wird erst freigegeben wenn die URL revoked wird UND keine anderen Referenzen existieren.

**MasterSelects-Loesung:**
- `BlobUrlManager`: Zentrales Tracking aller Blob-URLs per Clip-ID
- Auto-Revoke bei Clip-Entfernung (`revokeAll(clipId)`)
- Transfer/Share-Operationen fuer Split-Clips

**Noch nicht geloest:** Agent 4 identifiziert Code-Pfade (`projectSlice.ts`, `importPipeline.ts`) die Blob-URLs ausserhalb des Managers erstellen. Manuelles `revokeObjectURL` kann vergessen werden.

**Best Practice:** Alle Blob-URL-Erstellung durch einen zentralen Manager leiten. Nie `createObjectURL` direkt aufrufen.

---

### Pitfall #19: readyState Drop waehrend Playback
**Severity:** MEDIUM
**Konsens:** Agents 1, 3, 4, 7

**Problem:** `video.readyState` kann waehrend Playback unter 2 fallen (HAVE_CURRENT_DATA). Das bedeutet kein dekodierbares Frame verfuegbar. `importExternalTexture` liefert dann schwarze oder korrupte Frames.

**Warum:** Decoder-Underrun durch CPU-Last, langsame IO, oder Browser-interne Ressourcen-Konflikte.

**MasterSelects-Loesung:**
- `PlaybackHealthMonitor`: Erkennt `readyState < 2` waehrend Playback
- `forceVideoFrameDecode()`: Play/Pause-Cycle oder Tiny-Seek (`currentTime + 0.001`) um Decode zu erzwingen
- `LayerCollector`: Gate `readyState >= 2` vor jedem Texture-Import

**Best Practice:** Immer `readyState >= 2` pruefen vor Texture-Import. Recovery via Play/Pause oder Micro-Seek.

---

### Pitfall #20: Code-Duplikation der Video-Fallback-Cascade
**Severity:** MEDIUM
**Konsens:** Agent 7 (solo, aber architekturell signifikant)

**Problem:** Die Video-Fallback-Kaskade (scrub-cache -> seeking-cache -> drag-hold -> emergency-hold -> live-import -> copied-preview -> final-cache) ist in drei Dateien dupliziert: `LayerCollector` (~370 Zeilen), `NestedCompRenderer` (~390 Zeilen), `RenderDispatcher` (~200 Zeilen). Jede hat leicht unterschiedliche Bedingungen.

**Warum:** Organisches Wachstum. Nested Compositions und Multi-Preview brauchen aehnliche aber nicht identische Logik.

**MasterSelects-Loesung:** Derzeit nicht geloest -- die Duplikation besteht.

**Noch nicht geloest:** Ein Fix in einer Stelle muss in den anderen beiden repliziert werden. Bugs koennen in einer Kopie gefixt werden aber in den anderen bestehen bleiben.

**Best Practice:** Gemeinsame Basis-Klasse oder Utility-Modul fuer die Fallback-Kaskade extrahieren. Unterschiede ueber Configuration/Callbacks abbilden.

---

## Browser-spezifische Fallen (Zusammenfassung)

| Browser | Pitfall | Agents | Status |
|---------|---------|--------|--------|
| **Chrome** | GPU Surface Cold Start nach Reload | 1,2,3,6,7 | Geloest (Warmup) |
| **Chrome** | Device Loss bei Tab-Wechsel (Export) | 4,5,7 | Teilweise (Warning, kein Prevention) |
| **Chrome** | `createImageBitmap` als einziger Force-Decode | 1,7 | Geloest |
| **Firefox** | importExternalTexture Black Frames | 1,6,7 | Geloest (Copied Texture Fallback) |
| **Firefox** | WebCodecs weniger ausgereift (seit v130) | 6 | Feature-Detection vorhanden |
| **Safari** | WebGPU erst ab v17 | 6 | navigator.gpu Check |
| **Safari** | ProRes HW-Decode exklusiv | 6 | NativeHelper als Cross-Platform-Loesung |
| **Linux** | 15fps ohne Vulkan | 2,6,7 | Warnung, kein Auto-Fix |
| **Linux** | VAAPI korrupte Frames | 7 | WebCodecs als Bypass |
| **Linux** | Kein AAC Encoder (Patent) | 5,6 | Opus Fallback |
| **Mobile** | Autoplay Restrictions | 1,6,7 | Muted Warmup |
| **Mobile** | Kleinere HW-Decoder-Pools (1-2) | 1,2 | Selektives Pause/Play |
| **Mobile** | iOS 1GB Memory Limit | 4 | Nicht adressiert |

---

## Race Conditions & Timing-Probleme (Zusammenfassung)

| Race Condition | Agents | Loesung |
|---------------|--------|---------|
| Ueberlappende Seeks bei Rapid Scrubbing | 1,3,7 | Triple-Buffer + Queue + Coalescing |
| RVFC + seeked Wettlauf | 1,3,7 | RVFC primaer, seeked als Fallback |
| Seek waehrend laufendem Seek | 1,7 | `shouldRetargetPendingSeek()` mit Age/Drift Thresholds |
| Video-Element Sharing bei Split-Clips | 7 | `lastPresentedOwner` Tracking |
| GPU Submit vs Frame Capture Race | 5,7 | `device.queue.onSubmittedWorkDone()` |
| HMR Singleton Orphaning | 7 | `import.meta.hot` Dispose/Restore Pattern |
| Play-to-Pause Playhead Snap | 3 | Snap innerhalb 0.5s Toleranz |
| Stale Closures in Async Callbacks | 3,7 | `getState()` in Callback, nicht Closure-Capture |

---

## Memory Leaks & Resource-Erschoepfung (Zusammenfassung)

| Ressource | Risiko | Agents | Status |
|-----------|--------|--------|--------|
| VideoFrame VRAM | Unbegrenztes Wachstum ohne .close() | 4,6,7 | Geloest (explizites close) |
| ScrubbingCache VRAM | 2.4GB bei 1080p, kein dynamisches Budget | 2,4,7 | Offen |
| Blob URLs | Leak bei Code-Pfaden ausserhalb BlobUrlManager | 4 | Teilweise geloest |
| FFmpeg WASM RAM | Alle Frames gleichzeitig im Speicher | 5 | Offen |
| GPU Textures (GC-basiert) | Verzoegertes Cleanup unter Druck | 4,7 | Design-Trade-off |
| HTMLVideoElement Count | Kein Pooling, pro Clip neu | 4,7 | Offen |
| Composition Switch | Alte Decoder nicht immer freigegeben | 4 | Geloest (video.load()) |
| Mobile iOS | 1GB Gesamtlimit, keine angepassten Budgets | 4 | Offen |

---

## Offene Probleme in MasterSelects (Agent-uebergreifender Konsens)

Die folgenden Probleme werden von mehreren Agents als ungeloest identifiziert:

1. **Kein `visibilitychange`-Handler** (Agents 4, 7): Tab-Wechsel waehrend Export hat keine programmatische Mitigation.

2. **Kein dynamisches VRAM-Budget** (Agents 2, 4, 7): ScrubbingCache und GPU-Frame-Cache haben feste Limits die auf integrierten GPUs zu gross sind.

3. **Kein Video-Element-Pooling** (Agents 4, 7): Jeder Clip erstellt ein neues HTMLVideoElement. Bei grossen Timelines problematisch.

4. **FFmpeg WASM Memory** (Agent 5): All-in-RAM Approach macht lange Exporte in professionellen Codecs unmoeglich.

5. **Keine Mobile-spezifischen Limits** (Agent 4): iOS Safari hat ~1GB Limit, aber Cache-Budgets werden nicht angepasst.

6. **Code-Duplikation in Layer Collection** (Agent 7): Drei Kopien der Fallback-Kaskade sind ein Wartungsrisiko.

7. **WebCodecs Preview noch feature-flagged** (Agents 1, 2, 6, 7): `useFullWebCodecsPlayback = false` bedeutet Preview bleibt auf HTMLVideoElement mit allen inherenten Limitierungen.

---

## Konsens-Widersprueche zwischen Agents

**fastSeek Verfuegbarkeit:** Agent 1 sagt "Chrome 130+", Agent 3 sagt "Safari/Firefox", Agent 7 sagt "Safari/Firefox only, Chrome lacks it". Die Realitaet: `fastSeek` ist in Safari und Firefox nativ vorhanden. Chrome hatte es laengere Zeit nicht, hat es aber in neueren Versionen ergaenzt. MasterSelects prueft zur Laufzeit.

**Audio Master Clock:** Agent 3 beschreibt "Audio is Master Clock" ausfuehrlich. Agent 7 beschreibt es als "video as the master clock" (Section 3.2). Tatsaechlich ist die Architektur hybrid: Bei Vorhandensein von Audio ist Audio der Master (Agent 3 korrekt), bei Video-Only Timelines wird System-Clock genutzt.

**Decoder-Prioritaet:** Agents 1 und 6 beschreiben eine 3-stufige Kette, Agents 2 und 7 beschreiben 4 Stufen (mit ParallelDecode als eigenstaendiger Pfad). Korrekt sind 4 Stufen im LayerCollector, wobei ParallelDecode und WebCodecsPlayer verschiedene Varianten des WebCodecs-Pfads sind.

---

## Fazit

Die gefaehrlichsten Fallstricke beim Einsatz von HTMLVideoElement in einer Browser-NLE konzentrieren sich auf drei Kernbereiche:

1. **GPU-Surface-Lifecycle** (Cold Start, Device Loss, Texture Cleanup): Die Bruecke zwischen HTMLVideoElement und WebGPU ist fragil. Jeder Lifecycle-Uebergang (Reload, Tab-Switch, GC-Pressure) kann schwarze Frames oder Crashes verursachen.

2. **Seeking-Praezision und Timing** (GOP-Latenz, Race Conditions, Stuck Seeks): HTMLVideoElement wurde nicht fuer Random Access designed. Jede NLE muss eine eigene Seek-Queue, Coalescing-Logik und Recovery-Mechanismen implementieren.

3. **Memory-Management** (VideoFrame-Leaks, VRAM-Budgets, Blob-URLs): Browser-APIs geben Ressourcen nicht automatisch frei. Explizites Lifecycle-Management ist Pflicht, und feste Budgets muessen an die Hardware angepasst werden.

MasterSelects loest die meisten dieser Probleme durch eine beeindruckend tiefe Architektur (Triple-Buffer Seeks, Multi-Tier Caching, Health Monitoring mit Escalation). Die verbleibenden offenen Punkte (dynamisches VRAM-Budget, visibilitychange, FFmpeg Streaming, Mobile Limits) sind klar identifiziert und adressierbar.
