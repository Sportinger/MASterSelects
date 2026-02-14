# CLAUDE.md — MasterSelects Native Engine

Anweisungen für AI-Assistenten bei der Arbeit an der nativen Rust-Engine.

> **Vollständiger Architekturplan:** [`docs/plans/native-engine-plan.md`](./docs/plans/native-engine-plan.md)
> **Build-Checkliste:** [`docs/plans/native-engine-checklist.md`](./docs/plans/native-engine-checklist.md)

---

## 1. Projekt-Übersicht

**Was wir bauen:** Eine vollständig native Desktop-Video-Editing-App in Rust. Dual-GPU-Backend (CUDA für NVIDIA, Vulkan Compute für AMD/Intel/Mac), egui UI, kein Tauri, kein FFmpeg — eigener Demuxer, eigene NVDEC/NVENC FFI Bindings.

**Kernentscheidungen:**
- **Kein Tauri** — Alles nativ in Rust, egui für die komplette UI
- **Kein FFmpeg** — Eigener MP4/MKV Demuxer, eigene NVDEC/NVENC FFI Bindings, Symphonia nur für Audio
- **Dual GPU Backend** — `trait GpuBackend` abstrahiert CUDA und Vulkan Compute
- **wgpu nur für egui** — Video-Processing läuft über CUDA/Vulkan direkt, nicht über wgpu
- **30+ GPU Effects** — Jeder Effekt als `.cu` (CUDA) UND `.comp` (Vulkan Compute Shader)
- **Web-App-kompatibles Projektformat** — Liest/schreibt dieselben JSON-Projektdateien

### Crate-Struktur (12 Crates + App)

```
crates/
├── common/           # Typen, Traits, Errors, Config                    (~1200 LOC)
├── gpu-hal/          # GPU Abstraction + CUDA/Vulkan Backends            (~4000 LOC)
├── demux/            # Eigener MP4 + MKV Container Parser                (~2500 LOC)
├── decoder/          # HW Decode Management (NVDEC/Vulkan Video)         (~1500 LOC)
├── timeline-eval/    # Timeline @ Time T → Layer-Beschreibungen          (~1500 LOC)
├── compositor/       # GPU Compositing (Blend, Transform, Mask)          (~2000 LOC)
├── effects/          # 30+ GPU Effects (CUDA + Vulkan)                   (~5000 LOC)
├── encoder/          # HW Encode (NVENC/Vulkan Video Encode)             (~2000 LOC)
├── mux/              # Container Muxing (Muxide)                         (~800 LOC)
├── audio/            # Symphonia Decode + CPAL Output + Mixer            (~2500 LOC)
├── project/          # Project Save/Load (Web-kompatibel)                (~1200 LOC)
└── app-state/        # App State + Undo/Redo                             (~1500 LOC)

native-ui/            # egui App (bestehend, wird erweitert)              (~4000 LOC)
kernels/              # GPU Kernels: cuda/*.cu + vulkan/*.comp            (~6400 LOC)
```

**Dependency Graph:** `common` → `gpu-hal` → dann 6 parallele Agent-Gruppen (siehe Plan §6)

### Aktueller Stand

| Komponente | Pfad | Status |
|------------|------|--------|
| Native Helper (Legacy) | `tools/native-helper/` | Fertig (FFmpeg-basiert, wird langfristig ersetzt) |
| egui UI Mockup | `native-ui/` | Basis-UI vorhanden, wird zur echten App erweitert |
| Engine Crates | `crates/` | In Entwicklung — Phase 0 (Proof of Concept) |
| Web-App | `src/` | Production (TypeScript/React/WebGPU) |

---

## 2. Workflow (WICHTIG!)

### Branch-Regeln
| Branch | Zweck |
|--------|-------|
| `staging` | Entwicklung — hierhin committen |
| `master` | Production — nur via PR |

### Build & Commit

```bash
# VOR jedem Commit: Build prüfen!
# Für Native Engine:
cd native-ui && cargo build --release
# oder für einzelne Crates:
cd crates/common && cargo build

# Für Web-App:
npm run build

# Nach JEDEM Feature/Milestone sofort committen + pushen:
git add . && git commit -m "[native] description" && git push origin staging
```

### Commit-Prefix Konvention

| Prefix | Bereich | Beispiel |
|--------|---------|---------|
| `[native]` | Engine Crates, native-ui, Kernels | `[native] Add MP4 box parser for demux crate` |
| `[web]` | Web-App (src/, TypeScript/React) | `[web] Fix timeline clip rendering` |
| `[helper]` | Legacy native-helper | `[helper] Update FFmpeg bindings` |
| (kein Prefix) | Docs, Config, CI, übergreifend | `Update CLAUDE.md with native engine plan` |

### Commit-Häufigkeit

**Nach JEDEM abgeschlossenen Feature-Schritt committen + pushen:**
- Neues Modul/Datei angelegt und kompiliert? → Commit
- Trait oder Typ definiert? → Commit
- Funktion implementiert + Test geschrieben? → Commit
- Bug gefixt? → Commit
- **NICHT** stundenlang Code sammeln und dann einen Riesen-Commit machen

**IMMER vor Commit:**
- `cargo build` (bzw. `npm run build` für Web) ausführen
- Alle Errors beheben (Warnings sind OK, aber `cargo clippy` Warnings beachten)
- `cargo clippy --all-targets -- -D warnings` — keine Clippy-Warnings erlaubt
- Erst dann committen

**NIEMALS:**
- Direkt auf `master` committen
- Selbstständig zu `master` mergen
- Mehrere Änderungen sammeln ohne zu committen
- Committen ohne vorherigen Build-Check

### Merge zu Master (nur wenn User es verlangt!)
```bash
# 1. Version erhöhen in src/version.ts + Cargo.toml
# 2. CHANGELOG aktualisieren
# 3. Commit & Push
# 4. PR erstellen und mergen:
gh pr create --base master --head staging --title "..." --body "..."
gh pr merge --merge
# 5. Staging synchronisieren:
git fetch origin && git merge origin/master && git push origin staging
```

---

## 3. Quick Reference

```bash
# Native Helper
cd tools/native-helper && cargo run --release    # WebSocket :9876, HTTP :9877
cd tools/native-helper && cargo test             # Tests ausführen
cd tools/native-helper && cargo clippy            # Linting

# Native UI
cd native-ui && cargo run --release               # egui App starten
cd native-ui && cargo test
cd native-ui && cargo clippy

# Code Quality (alle Crates)
cargo fmt --all                                   # Formatierung
cargo clippy --all-targets --all-features -- -D warnings  # Linting
cargo test --all                                  # Alle Tests

# Web-App
npm install && npm run dev                        # http://localhost:5173
npm run build                                     # Production build
```

### Windows Build-Voraussetzungen
```bash
# native-helper braucht:
# - FFMPEG_DIR env var (Pfad zu FFmpeg-Installation)
# - LIBCLANG_PATH env var (für ffmpeg-next bindgen)
# Siehe tools/native-helper/README.md
```

---

## 4. Architektur

> Detail-Architektur, GPU Abstraction Layer, Threading-Modell, Dual-Kernel-Strategie:
> Siehe [`docs/plans/native-engine-plan.md`](./docs/plans/native-engine-plan.md) §1-§8

### Render Pipeline (Ziel)
```
Demuxer (eigener MP4/MKV Parser)
  └─► NAL Units (Annex-B)
        └─► HW Decoder (NVDEC / Vulkan Video)
              └─► GPU Frame (NV12, auf Device)
                    └─► NV12→RGBA Kernel
                          └─► Compositor (Blend, Transform, Mask)
                                └─► Effects Engine (30+ Effekte)
                                      └─► Preview: GPU → Staging → wgpu → egui
                                      └─► Export:  GPU → HW Encoder → Muxer → Datei
```

### Key Dependencies (Engine)
- `cudarc` — CUDA Runtime (Kernel Launch, Memory, Streams)
- `ash` + `gpu-allocator` — Vulkan Compute
- Eigene FFI — NVDEC (nvcuvid.h), NVENC (nvEncodeAPI.h)
- `symphonia` — Audio Decode (AAC, MP3, FLAC, WAV, Opus)
- `cpal` — Audio Output
- `crossbeam` — Thread-Kommunikation (Channels)
- `eframe` / `egui` — UI
- `muxide` — MP4 Container Muxing

### Threading-Modell
```
Thread 1: Main (egui Event Loop)          — NICHT blockieren!
Thread 2: Render (GPU Context Owner)      — Timeline Eval + Compositing + Effects
Thread 3..N: Decode Pool                  — 1 Thread pro aktive Video-Datei
Thread N+1: Audio (Realtime-Priority)     — CPAL Output + Mixer + Master Clock
Thread N+2: Export (on-demand)            — Eigene Render-Loop + HW Encode
```

---

## 5. Rust Critical Patterns (MUST READ)

### Error Handling
```rust
// ✅ RICHTIG: anyhow für Application-Code
use anyhow::{Context, Result};

fn load_config() -> Result<Config> {
    let data = std::fs::read_to_string("config.toml")
        .context("Failed to read config file")?;
    toml::from_str(&data).context("Failed to parse config")
}

// ✅ RICHTIG: thiserror für Library-Code
#[derive(Debug, thiserror::Error)]
enum DecodeError {
    #[error("FFmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}
```

**NIEMALS:**
- `.unwrap()` in Production-Code (nur in Tests erlaubt)
- `.expect()` nur wenn Panic wirklich gewollt ist (mit klarer Message)
- Errors still verschlucken — immer propagieren oder loggen

### Ownership & Borrowing
```rust
// ❌ FALSCH: Unnötiges Cloning
fn process(data: Vec<u8>) -> Vec<u8> {
    let copy = data.clone();  // Warum?
    transform(copy)
}

// ✅ RICHTIG: Ownership nutzen
fn process(data: Vec<u8>) -> Vec<u8> {
    transform(data)  // Move statt Clone
}

// ✅ RICHTIG: Borrow wenn kein Ownership nötig
fn analyze(data: &[u8]) -> Analysis {
    // Nur lesen, kein Ownership nötig
}
```

### Async/Tokio Patterns
```rust
// ✅ RICHTIG: Tokio-Tasks für CPU-bound Arbeit
let result = tokio::task::spawn_blocking(move || {
    // CPU-intensive Arbeit hier (z.B. FFmpeg Decoding)
    decode_frame(&data)
}).await?;

// ✅ RICHTIG: Parallel mit tokio::join!
let (frames, audio) = tokio::join!(
    decode_video(path),
    decode_audio(path)
);

// ❌ FALSCH: Sequentiell wenn parallel möglich
let frames = decode_video(path).await?;
let audio = decode_audio(path).await?;  // Wartet unnötig
```

### String Handling
```rust
// ✅ RICHTIG: &str für Parameter, String für Ownership
fn greet(name: &str) -> String {
    format!("Hello, {name}!")
}

// ❌ FALSCH: String als Parameter wenn &str reicht
fn greet(name: String) -> String {
    format!("Hello, {name}!")
}
```

### Stale Reference Prevention (wie Stale Closure in JS)
```rust
// ❌ FALSCH: Arc<Mutex> Daten vorher lesen und dann async nutzen
let data = state.lock().data.clone();
tokio::spawn(async move {
    // data könnte veraltet sein!
    process(data).await;
});

// ✅ RICHTIG: Daten erst im Task lesen
let state = state.clone();
tokio::spawn(async move {
    let data = state.lock().data.clone();
    process(data).await;
});
```

---

## 6. Memory Safety & Performance

### Zero-Copy wo möglich
```rust
// ✅ RICHTIG: Slice statt Vec kopieren
fn process_frame(frame: &[u8]) -> Result<()> {
    // Arbeite mit Referenz, kein Kopieren
}

// ✅ RICHTIG: Cow für optionales Ownership
use std::borrow::Cow;
fn normalize(input: &str) -> Cow<'_, str> {
    if input.contains('\0') {
        Cow::Owned(input.replace('\0', ""))
    } else {
        Cow::Borrowed(input)  // Kein Alloc wenn nicht nötig
    }
}
```

### Release-Profil (bereits konfiguriert)
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
```

### Unsafe Code
- **NIEMALS** `unsafe` ohne klaren Kommentar warum es sicher ist
- FFI-Grenzen (NVDEC/NVENC, CUDA, Vulkan) sind die einzige Ausnahme
- Immer `// SAFETY: ...` Kommentar bei `unsafe` Blöcken
- RAII für alle GPU-Ressourcen (Drop-Implementierung Pflicht)

---

## 7. Logging & Debugging

### Tracing verwenden (native-helper)
```rust
use tracing::{debug, info, warn, error, instrument};

#[instrument(skip(data))]
fn process_frame(id: u32, data: &[u8]) -> Result<()> {
    info!(frame_id = id, size = data.len(), "Processing frame");
    debug!("Frame header: {:?}", &data[..16]);
    // ...
}
```

### Tracing initialisieren
```rust
tracing_subscriber::fmt()
    .with_env_filter("masterselects_helper=debug,warn")
    .init();
```

### Common Issues

| Problem | Lösung |
|---------|--------|
| FFmpeg Linking Fehler | `FFMPEG_DIR` und `LIBCLANG_PATH` prüfen |
| Deadlock in async | Kein `Mutex::lock()` über `.await` Grenzen — `tokio::sync::Mutex` nutzen |
| Segfault in FFmpeg | Frame-Lifetime prüfen, FFmpeg-Objekte nicht über Thread-Grenzen |
| egui flackert | `request_repaint()` nach State-Änderung |
| Langsamer Debug-Build | `cargo build --release` für Performance-Tests |

---

## 8. Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Unit Tests — .unwrap() ist hier OK
        let result = parse_header(b"RIFF").unwrap();
        assert_eq!(result.format, Format::Riff);
    }

    #[tokio::test]
    async fn test_async_operation() {
        let result = fetch_data("test").await.unwrap();
        assert!(!result.is_empty());
    }
}

// Integration Tests in tests/ Verzeichnis
// tests/integration_test.rs
```

### Test-Befehle
```bash
cargo test                          # Alle Unit Tests
cargo test --test integration       # Integration Tests
cargo test -- --nocapture           # Mit stdout Output
cargo test specific_test_name       # Einzelner Test
```

---

## 9. Dependency Management

### Crate-Auswahl Prinzipien
1. **Bevorzuge** bekannte, gut-maintained Crates (tokio, serde, anyhow)
2. **Prüfe** letzte Veröffentlichung, Maintenance-Status, Downloads
3. **Minimiere** Feature-Flags — nur aktivieren was gebraucht wird
4. **Auditiere** regelmäßig: `cargo audit`

### Nützliche Cargo-Plugins
```bash
cargo install cargo-audit           # Vulnerability Scanner
cargo install cargo-watch           # Auto-Rebuild bei Änderungen
cargo install cargo-expand          # Macro-Expansion anzeigen
cargo install cargo-flamegraph      # Performance Profiling
```

---

## 10. Code Quality Toolchain

```bash
# Formatierung (muss vor Commit laufen)
cargo fmt --all

# Linting (keine Warnings erlaubt)
cargo clippy --all-targets --all-features -- -D warnings

# Dependency Audit
cargo audit

# Code-Qualität Checkliste:
# ✅ cargo fmt
# ✅ cargo clippy (0 warnings)
# ✅ cargo test (0 failures)
# ✅ cargo build --release (0 errors)
```

---

## 11. Wichtige Dateien

| Bereich | Datei |
|---------|-------|
| **Architekturplan** | `docs/plans/native-engine-plan.md` |
| Native UI Entry | `native-ui/src/main.rs` |
| Native UI Config | `native-ui/Cargo.toml` |
| Workspace Root | `Cargo.toml` (workspace) |
| GPU Traits | `crates/common/src/traits.rs` |
| CUDA Backend | `crates/gpu-hal/src/cuda/mod.rs` |
| Vulkan Backend | `crates/gpu-hal/src/vulkan/mod.rs` |
| Helper (Legacy) | `tools/native-helper/src/main.rs` |
| Web Version | `src/version.ts` |

---

## 12. Agent-Arbeitsregeln

1. **Trait-First:** Implementiere gegen Traits in `crates/common/`, nicht gegen konkrete Typen
2. **Keine Cross-Crate-Änderungen:** Wenn du `common/` ändern musst, vorher abstimmen
3. **Tests zuerst:** Jedes Modul braucht `#[cfg(test)] mod tests`
4. **Feature-gated GPU Code:** CUDA hinter `#[cfg(feature = "cuda")]`, Vulkan hinter `#[cfg(feature = "vulkan")]`
5. **Dual-Kernel-Regel:** Kein Effekt ist fertig ohne `.cu` UND `.comp` Variante
6. **Kein FFmpeg:** Keine Abhängigkeit auf ffmpeg-sys, ffmpeg-next, oder ähnliche Crates. **Niemals.**
7. **SAFETY-Kommentare:** Jeder `unsafe` Block braucht `// SAFETY:` Erklärung
8. **Error Propagation:** `thiserror` in Crates, `anyhow` nur in `native-ui/`
9. **Commit-Regel:** `cargo fmt && cargo clippy && cargo test` vor jedem Commit

---

## 13. Phasen-Übersicht

| Phase | Fokus | Status |
|-------|-------|--------|
| **Phase 0** | Proof-of-Concept: NVDEC → CUDA → egui + Vulkan → egui | **Aktuell** |
| **Phase 1** | Foundation: Demuxer, Decoder, Timeline, Audio, Project | Geplant |
| **Phase 2** | Compositing + 30+ Effects (CUDA + Vulkan Kernels) | Geplant |
| **Phase 3** | Export (HW Encode + Mux) + Audio Complete + Undo/Redo | Geplant |
| **Phase 4** | UI Integration: alle Panels mit echten Daten verbinden | Geplant |
| **Phase 5** | Post-v1: AI Engine (ONNX Runtime) | Zukunft |

> Details zu jeder Phase: [`docs/plans/native-engine-plan.md`](./docs/plans/native-engine-plan.md) §10

---

## 14. Web-App (Referenz)

Die Web-App unter `src/` nutzt TypeScript/React/WebGPU. Für Web-spezifische Patterns siehe `CLAUDE.md.backup`.

```bash
npm install && npm run dev   # http://localhost:5173
npm run build                # Production build
npm run lint                 # ESLint
```

---

*Backup der vorherigen Web-App CLAUDE.md: `CLAUDE.md.backup`*
