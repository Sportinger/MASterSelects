# Native Engine — Build-Checkliste

> **Stand:** 2026-02-14 | **Aktuelle Phase:** Phase 3 (Export) + Phase 4 (UI Integration)
>
> Legende: ✅ Fertig | 🔨 In Arbeit | ⬚ Offen

---

## Phase 0: Workspace & Foundation

### 0.1 Workspace Setup
- ✅ Root `Cargo.toml` (Workspace mit allen Crates — 13 Members)
- ✅ `crates/` Verzeichnis erstellen
- ✅ `kernels/` Verzeichnis erstellen (cuda/ + vulkan/)

### 0.2 `crates/common/` — Typen & Traits (~1200 LOC)
- ✅ `Cargo.toml` + Crate-Struktur
- ✅ Core-Types: `FrameNumber`, `TimeCode`, `Resolution`
- ✅ Enums: `PixelFormat`, `VideoCodec`, `AudioCodec`, `BlendMode`
- ✅ `VideoPacket` Struct (NAL data, PTS, DTS, keyframe flag)
- ✅ `GpuFrame` Struct (device pointer, format, dimensions)
- ✅ `LayerDesc` Struct (source, transform, opacity, blend, effects, mask)
- ✅ `Transform2D` Struct (position, scale, rotation, anchor)
- ✅ `trait GpuBackend` (alloc, dispatch, transfer, decoder/encoder creation)
- ✅ `trait HwDecoder` (decode, flush)
- ✅ `trait HwEncoder` (encode, flush)
- ✅ `trait GpuBuffer`, `trait GpuTexture`, `trait GpuStream`
- ✅ Error-Types mit `thiserror`
- ✅ Config-Structs (`DecoderConfig`, `EncoderConfig`, `RenderConfig`)
- ✅ `cargo test` + `cargo clippy` bestanden (12 Tests)

### 0.3 `crates/gpu-hal/` — GPU Abstraction (~4000 LOC)
- ✅ `Cargo.toml` + Feature-Flags (`cuda`, `vulkan`)
- ✅ CUDA Backend (`cuda/mod.rs`)
  - ✅ `cudarc`-basierte Device-Initialisierung
  - ✅ Buffer-Allokation (device + pinned host)
  - ✅ Kernel-Dispatch (PTX laden + launch)
  - ✅ Stream-Management
  - ✅ Device-Info (Name, VRAM)
- ✅ Vulkan Backend (`vulkan/mod.rs`)
  - ✅ `ash` + `gpu-allocator` Setup
  - ✅ Compute Pipeline erstellen
  - ✅ Buffer-Allokation
  - ✅ Shader-Dispatch (SPIR-V laden + dispatch)
  - ✅ Queue/Fence-Management
- 🔨 `gpu_to_wgpu()` — PreviewBridge (GPU Texture → wgpu Texture) — CUDA interop stub exists
- ✅ Unit Tests für beide Backends (50 CUDA + 31 Vulkan Tests)
- ✅ `cargo test` + `cargo clippy` bestanden

---

## Phase 0: Proof of Concept — Video in egui

### 0.4 Minimaler MP4 Demuxer
- ✅ `crates/demux/Cargo.toml` + Crate-Struktur
- ✅ MP4 Box-Parser: `ftyp`, `moov`, `trak`, `stbl`, `mdat`
- ✅ Sample Table auslesen (stts, stsc, stsz, stco/co64, stss)
- ✅ H.264 SPS/PPS aus `avcC` Box extrahieren
- ✅ NAL-Extraktion: AVCC → Annex-B Konvertierung
- ✅ Iterator-API: `fn next_video_packet() -> Option<VideoPacket>`
- ✅ Unit Tests (121 Tests inkl. MKV)
- ✅ `cargo test` + `cargo clippy` bestanden

### 0.5 HW Decoder (NVDEC)
- ✅ `crates/decoder/Cargo.toml` + Crate-Struktur
- ✅ NVDEC FFI Bindings (`nvcuvid.h` → Rust `unsafe`)
  - ✅ `cuvidCreateDecoder`
  - ✅ `cuvidDecodePicture`
  - ✅ `cuvidMapVideoFrame` / `cuvidUnmapVideoFrame`
  - ✅ `cuvidDestroyDecoder`
- ✅ Safe Wrapper: `NvDecoder` implementiert `HwDecoder` Trait
- ✅ Frame-Output als NV12 auf GPU
- ✅ RAII: Drop-Implementierung für Decoder-Ressourcen
- ✅ Integration Test: MP4 → Demux → Decode → NV12 Frame (26 tests in decoder/tests/)

### 0.6 NV12→RGBA Kernel
- ✅ `kernels/cuda/nv12_to_rgba.cu` — CUDA Kernel (BT.709)
- ✅ `kernels/vulkan/nv12_to_rgba.comp` — Vulkan Compute Shader
- ✅ `kernels/cuda/composite.cu` — Alpha-Blend Kernel
- ✅ `kernels/vulkan/composite.comp` — Alpha-Blend Shader
- ✅ `kernels/cuda/transform.cu` — Transform Kernel
- ✅ `kernels/vulkan/transform.comp` — Transform Shader
- ✅ `kernels/cuda/blend.cu` — Blend-Mode Kernel
- ✅ `kernels/vulkan/blend.comp` — Blend-Mode Shader
- ✅ PTX kompilieren (CUDA) — verifiziert
- ⬚ SPIR-V kompilieren (Vulkan) — braucht glslc
- ⬚ Integration in gpu-hal Dispatch

### 0.7 Preview Bridge (GPU → egui)
- ✅ GPU RGBA Buffer → CPU Staging Transfer (PreviewBridge)
- ✅ `egui::TextureHandle` aus ColorImage erstellen
- ✅ Preview Panel: Animiertes Test Pattern statt schwarzes Rect
- ✅ Live Stats: FPS, Frame Time, Resolution im Preview Panel
- ✅ EngineOrchestrator mit Decode-Thread-Pipeline
- ✅ Echte MP4+MKV-Metadaten via ms-demux (Resolution, FPS, Duration, Codec)
- ✅ Real Demux → Packet-Extraktion im Decode-Thread (synthetische Pixel, echtes Timing)
- ✅ Frame-Timing: Decode @ richtigem FPS (NVDEC pipeline mit frame pacing)

### 0.8 End-to-End PoC Test
- ✅ MP4-Datei öffnen → Demux → echte Pakete → synthetische Frames → egui Display
- ✅ MKV-Datei öffnen → Demux → echte Pakete → synthetische Frames → egui Display
- ✅ NVDEC Decode → NV12→RGBA → egui Display (GPU kernel + CPU fallback)
- ⬚ Vulkan-Pfad: MP4 → Decode → Vulkan Compute → egui Display (stubs exist)
- ⬚ Beide Backends rendern dasselbe Bild korrekt
- ⬚ Performance: < 6ms pro Frame @ 1080p

---

## Phase 1: Foundation — ✅ ABGESCHLOSSEN

### 1.1 `crates/demux/` — Vollständiger Demuxer (~2500 LOC)
- ✅ MKV Container Parser (EBML) — `src/mkv/` mit ebml.rs, elements.rs, cluster.rs, mod.rs
- ✅ H.264/H.265/VP9/AV1 Codec-Support (MKV codec_id Mapping)
- ✅ Audio-Track Extraktion (MKV + MP4)
- ✅ Seeking (Random Access via Keyframes/Cues)
- ✅ Mehrspur-Support (Video + Audio)
- ✅ 121 Tests bestanden

### 1.2 `crates/decoder/` — Decoder Pool (~3000 LOC) — ✅ FERTIG (102 Tests)
- ✅ Decoder-Pool (1 Decoder pro aktive Videodatei, LRU-Eviction) — `pool.rs`
- ✅ Prefetch-Queue (vorausdekodieren, Ring-Buffer) — `prefetch.rs`
- ✅ Thumbnail-Generierung (Cache mit Eviction) — `thumbnail.rs`
- ✅ Software NV12→RGBA CPU Decoder — `software.rs`
- ✅ Vulkan Video Decode Backend (stub) — `vulkan_video/`
- ✅ Integration Tests (26 tests) — `tests/nvdec_integration.rs`

### 1.3 `crates/timeline-eval/` (~1500 LOC) — ✅ FERTIG (47 Tests)
- ✅ Timeline-Modell (Tracks, Clips, Keyframes) — `types.rs`
- ✅ `evaluate(time: TimeCode) -> Vec<LayerDesc>` — `evaluator.rs`
- ✅ Keyframe-Interpolation (linear, bezier, hold) — `keyframe.rs`
- ✅ Transition-Evaluation (CrossDissolve, Fade, Wipe, Slide) — `transition.rs`
- ✅ Composition-Unterstützung (verschachtelt, max depth 16) — `nested.rs`
- ✅ Marker-System (in Timeline types)

### 1.4 `crates/audio/` (~2500 LOC) — ✅ FERTIG (66 Tests)
- ✅ Symphonia Decode (AAC, MP3, FLAC, WAV, Opus) — `decoder.rs`
- ✅ CPAL Audio Output (Lock-free Ring Buffer) — `output.rs`
- ✅ Audio Mixer (Tracks, Volume, Constant-Power Pan, Soft-Clip) — `mixer.rs`
- ✅ A/V Sync (Audio als Master Clock, AtomicU64-basiert) — `sync.rs`
- ✅ Waveform-Daten Generierung — `waveform.rs`
- ✅ Sample-Rate Conversion — `resampler.rs`
- ✅ LUFS/Peak/RMS Metering — `meter.rs`

### 1.5 `crates/project/` (~1200 LOC) — ✅ FERTIG (64 Tests)
- ✅ Web-App-kompatibles JSON Format lesen — `load.rs`
- ✅ Projekt speichern/laden (atomarer Schreibvorgang) — `save.rs`
- ✅ Format-Migration (Versionen) — `migrate.rs`
- ✅ Recent Projects Liste — `recent.rs`
- ✅ Auto-Save Timer — `autosave.rs`
- ✅ Vollständiges Datenmodell (camelCase, Web-kompatibel) — `types.rs`

### 1.6 `crates/app-state/` (~1500 LOC) — ✅ FERTIG (79 Tests)
- ✅ Zentraler App-State (Timeline, Media, Selection) — `state.rs`
- ✅ Undo/Redo (Snapshot-basiert, Batch-Gruppierung) — `history.rs` + `snapshot.rs`
- ✅ Playback-Controller (Play, Pause, Seek, Scrub, In/Out, Loop, Rate) — `playback.rs`
- ✅ Selection-Management (Clips, Tracks, Keyframes, Multi-Select) — `selection.rs`

---

## Phase 2: Compositing & Effects — ✅ ABGESCHLOSSEN

### 2.1 `crates/compositor/` (~2000 LOC) — ✅ FERTIG (28 Tests)
- ✅ Multi-Layer GPU Compositing — `compositor.rs`
- ✅ Blend-Modes (Normal, Multiply, Screen, Overlay, Add, etc.) — `blend.rs`
- ✅ Transform-Pipeline (Position, Scale, Rotation, Anchor) — `transform.rs`
- ✅ Mask-Compositing (Rect, Ellipse, Path, Feather) — `mask.rs`
- ✅ Transitions (Dissolve, Wipe, Slide) — `transition.rs`
- ✅ Ping-Pong Render Pipeline — `pipeline.rs`
- ✅ Color-Space Utilities — `color.rs`

### 2.2 `crates/effects/` (~5000 LOC) — ✅ FERTIG (74 Tests)
- ✅ Effect Trait + Registry System — `traits.rs` + `registry.rs`
- ✅ Parameter Validation + Defaults — `params.rs`
- ✅ **Color (6):** Brightness/Contrast, HSL, Curves, Exposure, White Balance, Color Wheels
- ✅ **Blur (4):** Gaussian (separable), Directional, Radial, Zoom
- ✅ **Keying (2):** Chroma Key, Luma Key
- ✅ **Stylize (4):** Glow (multi-pass), Sharpen, Noise/Grain, Vignette
- ✅ Jeder Effekt: `.cu` (CUDA) + `.comp` (Vulkan) Variante
- ✅ Effect-Parameter-System (animierbar via Keyframes)

### 2.3 GPU Kernels — ✅ FERTIG (36 Dateien)
- ✅ 16 CUDA Effect Kernels (`kernels/cuda/effects/`) — 20 Entry Points
- ✅ 20 Vulkan Effect Shaders (`kernels/vulkan/effects/`)
- ✅ Basis-Kernels: nv12_to_rgba, composite, transform, blend (CUDA + Vulkan)

---

## Phase 3: Export — 🔨 IN ARBEIT

### 3.1 `crates/encoder/` (~3500 LOC) — ✅ FERTIG (36 Tests)
- ✅ NVENC FFI Bindings (`nvEncodeAPI.h` → Rust `unsafe`) — `nvenc/ffi.rs`
  - ✅ Session öffnen/schließen
  - ✅ Parameter-Konfiguration (Preset, Profile, Bitrate)
  - ✅ Input-Buffer Registration (CUDA Device Ptr)
  - ✅ Output-Buffer (Bitstream) Lock/Unlock
  - ✅ Function Pointer Table via `libloading`
  - ✅ Alle GUIDs (H.264/H.265, Presets, Profiles)
- ✅ Safe Wrapper: `NvEncoder` — `nvenc/mod.rs`
  - ✅ Implementiert `HwEncoder` Trait
  - ✅ RAII Drop für Session-Cleanup
- ✅ Parameter-Builder (EncoderConfig → NVENC Params) — `nvenc/params.rs`
  - ✅ VideoCodec → NVENC GUID Mapping
  - ✅ EncoderPreset → NVENC Preset GUID
  - ✅ EncoderProfile → NVENC Profile GUID
  - ✅ Bitrate-Control (CBR, VBR, CQP) → RC Mode
- ✅ Input/Output Buffer-Pool — `nvenc/buffer.rs`
- ✅ Encoder-Session (Frame-Counting, PTS, Keyframe-Interval) — `session.rs`
- ✅ Export-Pipeline Skeleton — `export.rs`
  - ✅ ExportConfig mit Validierung
  - ✅ ExportPipeline mit Background-Thread
  - ✅ Progress-Reporting (Crossbeam Channel)
  - ✅ Export-Abbruch (AtomicBool Cancel-Flag)
- ✅ Vulkan Video Encode Backend (stub) — `crates/encoder/src/vulkan_video/`

### 3.2 `crates/mux/` (~3000 LOC) — ✅ FERTIG (67 Tests)
- ✅ MP4 Box Writer (eigene Implementierung, kein FFmpeg) — `atoms.rs` + `mp4.rs`
  - ✅ ftyp, moov, mvhd, trak, tkhd, mdhd, hdlr, minf, stbl
  - ✅ stts, stsc, stsz, stco/co64, stss (Sync Sample)
  - ✅ avcC Box (H.264 SPS/PPS)
  - ✅ hvcC Box (H.265 VPS/SPS/PPS)
  - ✅ Audio stsd (mp4a/Opus)
- ✅ High-Level Muxer API — `muxer.rs`
  - ✅ `Mp4Muxer::new()` → `add_video_track()` → `write_video_sample()` → `finalize()`
  - ✅ Audio Track Support
  - ✅ Multi-Track (Video + Audio)
  - ✅ Progressive mdat + moov-at-end
  - ✅ Timescale-Konvertierung (90000 Video, SampleRate Audio)

### 3.3 Export-Workflow in native-ui — ✅ FERTIG (20 Tests)
- ✅ `ExportPipeline` Struct mit Background-Thread — `native-ui/src/export.rs`
- ✅ `ExportConfig` (Pfad, Resolution, FPS, Codec, Bitrate)
- ✅ `ExportProgress` mit State-Machine (Idle → Preparing → Rendering → Encoding → Finalizing → Complete)
- ✅ Progress-Reporting (Frames, ETA, Encoding-FPS)
- ✅ Export-Abbruch
- 🔨 Integration mit echtem Encoder + Muxer (Skeleton steht, braucht GPU)

---

## Phase 4: UI Integration — 🔨 IN ARBEIT

### 4.1 native-ui Kern-Integration — ✅ FERTIG
- ✅ `AppState` (ms-app-state) in MasterSelectsApp eingebunden — `app.rs`
- ✅ `HistoryManager` für Undo/Redo
- ✅ `ProjectFile` (ms-project) für Projekt-Management
- ✅ `AutoSaver` + `RecentProjects`
- ✅ `EffectRegistry` (ms-effects) registriert
- ✅ Keyboard-Shortcuts (Ctrl+N/O/S/Z/Y, Space)
- ✅ Status-Bar mit Meldungen (3s Fade)

### 4.2 Toolbar — ✅ FERTIG
- ✅ File-Menü (New, Open, Save, Save As, Import, Export) — `toolbar.rs`
- ✅ Edit-Menü (Undo, Redo)
- ✅ Keyboard-Shortcut Labels in Menüs
- ✅ `ToolbarAction` enum für Action-Dispatching
- ✅ Engine-State Anzeige (GPU, Status, FPS)

### 4.3 Engine MKV-Support — ✅ FERTIG
- ✅ MKV/WebM Demuxing in EngineOrchestrator — `engine.rs`
- ✅ `probe_file_info()` unterstützt MKV/WebM
- ✅ `try_open_demuxer()` erstellt MkvDemuxer

### 4.4 Verbleibende UI-Integration
- ⬚ Media Panel: Echte Dateien laden via rfd + ms-project
- ⬚ Timeline: Echte Tracks/Clips aus AppState
- ⬚ Properties: Transform/Effects/Masks an Engine binden
- ⬚ Preview Panel: Live GPU-Rendering via Compositor
- ⬚ Playback: Play/Pause/Seek mit echtem Audio (ms-audio)
- ⬚ Undo/Redo: History-Snapshots bei jeder Aktion

---

## Bestehende Komponenten

### native-ui (egui App)
- ✅ Toolbar mit File/Edit-Menüs, Shortcuts, GPU-Status
- ✅ Media Panel (Tabs, Ordner-Baum, Spalten)
- ✅ Preview Panel (16:9 Canvas, Quality-Selector, Stats)
- ✅ Properties Panel (Transform, Effects, Masks, Export, Waveform, Histogram)
- ✅ Timeline (Composition-Tabs, Transport, Tracks, Clips, Ruler, Playhead)
- ✅ Dark Theme (komplettes Styling)
- ✅ EngineOrchestrator mit MP4+MKV-Demuxing (Decode-Thread)
- ✅ ExportPipeline mit Background-Thread + Progress
- ✅ AppState + History + Project-Management integriert
- ✅ Baut erfolgreich (`cargo build --release`)

### tools/native-helper (Legacy, FFmpeg-basiert)
- ✅ WebSocket-Server (Port 9876)
- ✅ HTTP-File-Server (Port 9877)
- ✅ FFmpeg Video-Decoder + HW-Accel-Detection
- ✅ Encoder (NVENC/VideoToolbox)
- ✅ Download-Manager (yt-dlp)
- ✅ LRU Frame Cache
- ✅ Windows System Tray

---

## Test-Statistik

| Crate | Tests |
|-------|-------|
| ms-common | 12 |
| ms-gpu-hal | 50 + 31 |
| ms-demux | 121 |
| ms-decoder | 76 |
| ms-timeline-eval | 47 |
| ms-audio | 66 |
| ms-project | 64 |
| ms-app-state | 79 |
| ms-compositor | 28 |
| ms-effects | 74 |
| ms-encoder | 36 |
| ms-mux | 67 |
| masterselects-native | 47 |
| **Gesamt** | **798** |

*Zuletzt aktualisiert: 2026-02-14*
