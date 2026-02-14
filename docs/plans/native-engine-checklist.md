# Native Engine — Build-Checkliste

> **Stand:** 2026-02-14 | **Aktuelle Phase:** Phase 0 (Proof of Concept)
>
> Legende: ✅ Fertig | 🔨 In Arbeit | ⬚ Offen

---

## Phase 0: Workspace & Foundation

### 0.1 Workspace Setup
- ✅ Root `Cargo.toml` (Workspace mit allen Crates)
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
- ✅ Unit Tests für beide Backends (19 Tests)
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
- ✅ Unit Tests (31 Tests)
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
- ⬚ Integration Test: MP4 → Demux → Decode → NV12 Frame

### 0.6 NV12→RGBA Kernel
- ✅ `kernels/cuda/nv12_to_rgba.cu` — CUDA Kernel (BT.709)
- ✅ `kernels/vulkan/nv12_to_rgba.comp` — Vulkan Compute Shader
- ✅ `kernels/cuda/composite.cu` — Alpha-Blend Kernel
- ✅ `kernels/vulkan/composite.comp` — Alpha-Blend Shader
- ✅ PTX kompilieren (CUDA) — verifiziert
- ⬚ SPIR-V kompilieren (Vulkan) — braucht glslc
- ⬚ Integration in gpu-hal Dispatch

### 0.7 Preview Bridge (GPU → egui)
- ✅ GPU RGBA Buffer → CPU Staging Transfer (PreviewBridge)
- ✅ `egui::TextureHandle` aus ColorImage erstellen
- ✅ Preview Panel: Animiertes Test Pattern statt schwarzes Rect
- ✅ Live Stats: FPS, Frame Time, Resolution im Preview Panel
- ✅ EngineOrchestrator Stub mit Test Pattern Generator
- ⬚ Frame-Timing: Decode @ richtigem FPS (braucht echten Decoder)

### 0.8 End-to-End PoC Test
- ⬚ MP4-Datei öffnen → Demux → NVDEC Decode → NV12→RGBA → egui Display
- ⬚ Vulkan-Pfad: MP4 → Decode → Vulkan Compute → egui Display
- ⬚ Beide Backends rendern dasselbe Bild korrekt
- ⬚ Performance: < 6ms pro Frame @ 1080p

---

## Phase 1: Foundation (nach Phase 0)

### 1.1 `crates/demux/` — Vollständiger Demuxer (~2500 LOC)
- ⬚ MKV Container Parser (EBML)
- ⬚ H.265/VP9/AV1 Codec-Support
- ⬚ Audio-Track Extraktion
- ⬚ Seeking (Random Access via Keyframes)
- ⬚ Mehrspur-Support (Video + Audio + Subtitle)

### 1.2 `crates/decoder/` — Decoder Pool (~1500 LOC)
- ⬚ Decoder-Pool (1 Decoder pro aktive Videodatei)
- ⬚ Prefetch-Queue (vorausdekodieren)
- ⬚ Thumbnail-Generierung
- ⬚ Vulkan Video Decode Backend

### 1.3 `crates/timeline-eval/` (~1500 LOC)
- ⬚ Timeline-Modell (Tracks, Clips, Keyframes)
- ⬚ `evaluate(time: TimeCode) -> Vec<LayerDesc>`
- ⬚ Keyframe-Interpolation (linear, bezier, hold)
- ⬚ Composition-Unterstützung (verschachtelt)
- ⬚ Marker-System

### 1.4 `crates/audio/` (~2500 LOC)
- ⬚ Symphonia Decode (AAC, MP3, FLAC, WAV, Opus)
- ⬚ CPAL Audio Output (Realtime-Priority Thread)
- ⬚ Audio Mixer (Tracks, Volume, Pan)
- ⬚ A/V Sync (Audio als Master Clock)
- ⬚ Waveform-Daten Generierung

### 1.5 `crates/project/` (~1200 LOC)
- ⬚ Web-App-kompatibles JSON Format lesen
- ⬚ Projekt speichern/laden
- ⬚ Media-Referenz-Auflösung (Dateipfade)
- ⬚ Format-Migration (Versionen)

### 1.6 `crates/app-state/` (~1500 LOC)
- ⬚ Zentraler App-State (Timeline, Media, Selection)
- ⬚ Undo/Redo (Snapshot-basiert)
- ⬚ Playback-Controller (Play, Pause, Seek)
- ⬚ Selection-Management

---

## Phase 2: Compositing & Effects

### 2.1 `crates/compositor/` (~2000 LOC)
- ⬚ Multi-Layer GPU Compositing
- ⬚ Blend-Modes (Normal, Multiply, Screen, Overlay, Add, etc.)
- ⬚ Transform-Pipeline (Position, Scale, Rotation, Anchor)
- ⬚ Mask-Compositing (Alpha, Luminance, Feather)
- ⬚ Transitions (Dissolve, Wipe, Slide, etc.)

### 2.2 `crates/effects/` (~5000 LOC) — 30+ GPU Effects
- ⬚ **Color:** Brightness, Contrast, Saturation, Hue Rotate, Color Balance, Curves, Levels, LUT
- ⬚ **Blur:** Gaussian, Box, Directional, Radial, Zoom, Lens
- ⬚ **Distort:** Displacement, Turbulence, Spherize, Bulge, Ripple, Wave
- ⬚ **Keying:** Chroma Key, Luma Key, Color Range
- ⬚ **Stylize:** Glow, Sharpen, Emboss, Find Edges, Posterize, Noise, Grain, Vignette
- ⬚ Jeder Effekt: `.cu` (CUDA) + `.comp` (Vulkan) Variante
- ⬚ Effect-Parameter-System (animierbar via Keyframes)

---

## Phase 3: Export & Audio Complete

### 3.1 `crates/encoder/` (~2000 LOC)
- ⬚ NVENC FFI Bindings (nvEncodeAPI.h → Rust)
- ⬚ H.264/H.265 Hardware-Encoding
- ⬚ Vulkan Video Encode
- ⬚ Bitrate-Control (CBR, VBR, CQP)
- ⬚ Export-Pipeline (Timeline → Render → Encode)

### 3.2 `crates/mux/` (~800 LOC)
- ⬚ MP4 Container Muxing (via Muxide)
- ⬚ Audio + Video Interleaving
- ⬚ Metadata (Duration, Codec-Info)

### 3.3 Vollständiger Export-Workflow
- ⬚ Timeline → Frame-by-Frame Render → Encode → Mux → Datei
- ⬚ Progress-Reporting
- ⬚ Export-Abbruch

---

## Phase 4: UI Integration

### 4.1 native-ui mit Engine verbinden
- ⬚ Media Panel: Echte Dateien laden + anzeigen
- ⬚ Preview Panel: Live GPU-Rendering
- ⬚ Timeline: Echte Tracks/Clips/Playhead
- ⬚ Properties: Transform/Effects/Masks an Engine binden
- ⬚ Export: Echte Export-Pipeline triggern
- ⬚ Playback: Play/Pause/Seek funktioniert mit Audio
- ⬚ Undo/Redo: History-System aktiv

---

## Bestehende Komponenten

### native-ui (egui Mockup)
- ✅ Toolbar (Menüs, Projektname, GPU-Status)
- ✅ Media Panel (Tabs, Ordner-Baum, Spalten)
- ✅ Preview Panel (16:9 Canvas, Quality-Selector, Stats)
- ✅ Properties Panel (Transform, Effects, Masks, Export, Waveform, Histogram)
- ✅ Timeline (Composition-Tabs, Transport, Tracks, Clips, Ruler, Playhead)
- ✅ Dark Theme (komplettes Styling)
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

*Zuletzt aktualisiert: 2026-02-14*
