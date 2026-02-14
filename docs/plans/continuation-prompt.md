# Native Engine — Continuation Prompt

> Kopiere diesen Prompt in eine neue Claude Code Session, um mit bis zu 10 parallelen Agents weiterzubauen.

---

## PROMPT START

Du arbeitest an einer nativen Rust Video-Editing-Engine (MasterSelects). Das Projekt hat eine Web-App (TypeScript/React in `src/`) und eine native Engine (Rust/egui in `crates/`, `native-ui/`, `kernels/`). Wir arbeiten NUR an der nativen Engine.

### Kontext

Lies diese Dateien für den vollständigen Kontext:
- `docs/plans/native-engine-plan.md` — Architekturplan
- `docs/plans/native-engine-checklist.md` — Build-Checkliste mit Status
- `CLAUDE.md` — Projektregeln und Conventions

### Aktueller Stand (Phase 0 — fast fertig)

**Was existiert und funktioniert:**

| Crate | Status | Tests |
|-------|--------|-------|
| `crates/common` | ✅ Fertig — Typen, Traits (GpuBackend, HwDecoder), Errors | 12 |
| `crates/demux` | ✅ Fertig — MP4 Parser, NAL Extraction, Audio Tracks (AAC/Opus) | 44 |
| `crates/decoder` | ✅ Fertig — NVDEC FFI, Decoder Session, Surface Lifecycle, Seek | 17 |
| `crates/gpu-hal` (CUDA) | ✅ Fertig — CudaBackend impl GpuBackend, KernelManager, raw cuLaunchKernel | 50 |
| `crates/gpu-hal` (Vulkan) | ✅ Fertig — VulkanBackend impl GpuBackend, Context, Pipeline, Memory, Shader | 31 |
| `kernels/` | ✅ 4 Kernel-Paare — nv12_to_rgba, composite, transform, blend (.cu + .comp) | — |
| `native-ui` | ✅ Engine Orchestrator mit 2-Thread Decode Pipeline, State Machine, Preview Bridge | 21 |
| Build System | ✅ PTX/SPIR-V Compilation in build.rs, Kernel Embedding | 2 |

**Was FEHLT für Phase 0 Abschluss (Checkliste 0.8):**
- ⬚ End-to-End Integration: MP4 → Demux → NVDEC → NV12→RGBA Kernel → egui Preview
- ⬚ Vulkan-Pfad: gleiche Pipeline mit Vulkan Compute statt CUDA
- ⬚ Beide Backends rendern dasselbe Bild
- ⬚ Performance < 6ms/Frame @ 1080p

**Was FEHLT für Phase 1 (Foundation):**
- ⬚ `crates/timeline-eval/` — Timeline @ Time T → Vec<LayerDesc> (Clips, Tracks, Keyframes, Transitions)
- ⬚ `crates/audio/` — Symphonia Decode + CPAL Output + Mixer + A/V Sync
- ⬚ `crates/project/` — Web-App-kompatibles JSON Projekt Save/Load
- ⬚ `crates/app-state/` — Undo/Redo, Selection, Playback Controller
- ⬚ MKV Demuxer in `crates/demux/` (EBML Parser)
- ⬚ Decoder Pool mit Prefetch in `crates/decoder/`
- ⬚ 30+ GPU Effects in `kernels/` (nur 4 Basis-Kernels existieren)

### Workspace-Struktur

```
Cargo.toml (workspace: native-ui, crates/common, crates/gpu-hal, crates/demux, crates/decoder)
crates/
  common/     — ms-common (Typen, Traits, Errors)
  gpu-hal/    — ms-gpu-hal (CUDA + Vulkan Backends)
  demux/      — ms-demux (MP4 Parser, NAL, Audio)
  decoder/    — ms-decoder (NVDEC Wrapper)
kernels/
  cuda/       — .cu Kernels (nv12_to_rgba, composite, transform, blend)
  vulkan/     — .comp Shaders (gleiche wie CUDA)
native-ui/    — masterselects-native (egui App)
```

### Aufgabe

Starte bis zu **10 parallele Agents** die an verschiedenen Teilen der Engine arbeiten. Jeder Agent bekommt klare File-Ownership um Konflikte zu vermeiden.

**Empfohlene Agent-Aufteilung (wähle was am sinnvollsten ist):**

**Priorität 1 — End-to-End Integration (Phase 0.8):**
1. **Integration Agent** — `native-ui/src/engine.rs` verdrahten: Demux → Decoder → NV12→RGBA Kernel → PreviewBridge. Ersetze synthetische Frames durch echte Decode-Pipeline.

**Priorität 2 — Neue Crates (Phase 1):**
2. **Timeline Eval** — `crates/timeline-eval/` neu erstellen: Timeline-Modell, evaluate(time) → Vec<LayerDesc>, Keyframe-Interpolation
3. **Audio Crate** — `crates/audio/` neu erstellen: Symphonia Decode, CPAL Output, Mixer, A/V Sync
4. **Project Crate** — `crates/project/` neu erstellen: Web-App JSON Format lesen/schreiben
5. **App State** — `crates/app-state/` neu erstellen: Undo/Redo (Snapshot-basiert), Selection, Playback

**Priorität 3 — GPU Effects (Phase 2 Vorarbeit):**
6. **Color Effects** — `kernels/cuda/effects/` + `kernels/vulkan/effects/`: brightness, contrast, saturation, hue_rotate, curves (.cu + .comp)
7. **Blur Effects** — gaussian_blur, directional_blur, radial_blur (.cu + .comp)
8. **Stylize Effects** — glow, sharpen, noise/grain, vignette (.cu + .comp)

**Priorität 4 — Erweiterungen:**
9. **MKV Demuxer** — `crates/demux/src/mkv/`: EBML Parser, Cluster/Block → NAL
10. **Compositor Crate** — `crates/compositor/` neu erstellen: Multi-Layer Compositing mit bestehenden Kernels

### Regeln für Agents

1. **Neue Crates** müssen in `Cargo.toml` (workspace members) registriert werden
2. **Trait-First**: Gegen Traits in `crates/common/` implementieren
3. **thiserror** in Library-Crates, **anyhow** nur in native-ui
4. **Kein FFmpeg** — Niemals
5. **Tests**: Jedes Modul braucht `#[cfg(test)] mod tests`
6. **SAFETY-Kommentare** an jedem `unsafe` Block
7. **Commit-Prefix**: `[native]` für Engine-Code
8. **Feature-gated**: CUDA hinter `#[cfg(feature = "cuda")]`, Vulkan hinter `#[cfg(feature = "vulkan")]`
9. **Dual-Kernel-Regel**: Kein Effekt ohne `.cu` UND `.comp` Variante
10. **Build prüfen**: `cargo build` bzw. `cargo check` vor Abschluss

### Build-Befehle

```bash
# Einzelne Crates
cargo check -p ms-common
cargo check -p ms-gpu-hal --features cuda
cargo check -p ms-gpu-hal --features vulkan
cargo check -p ms-demux
cargo check -p ms-decoder
cargo check -p masterselects-native

# Tests
cargo test -p ms-common
cargo test -p ms-demux
cargo test -p ms-decoder
cargo test -p ms-gpu-hal --features cuda
cargo test -p ms-gpu-hal --features vulkan

# Alles
cargo build --workspace
cargo test --workspace
```

### Nach Abschluss

1. Alle Änderungen committen mit `[native]` Prefix
2. Auf `staging` Branch pushen (NICHT master)
3. Checklist in `docs/plans/native-engine-checklist.md` aktualisieren

## PROMPT ENDE
