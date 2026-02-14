# Native Engine Plan — MasterSelects

> **Ziel:** Rendering-Engine von Browser-WebGPU in einen nativen Rust-Backend verlagern.
> GPU-to-GPU Zero-Copy Pipeline, AI-Effekte via ONNX, Desktop-App via Tauri.
> React UI bleibt 1:1 erhalten.

---

## 1. Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│  Tauri App                                                      │
│                                                                 │
│  ┌──────────────────────┐    ┌────────────────────────────────┐ │
│  │  Webview (React UI)  │    │  Rust Backend                  │ │
│  │                      │    │                                │ │
│  │  Timeline            │    │  ┌──────────┐  ┌───────────┐  │ │
│  │  Panels              │◄──►│  │ Decoder  │  │ AI Engine │  │ │
│  │  Controls            │IPC │  │ (FFmpeg)  │  │ (ONNX RT) │  │ │
│  │  Dock                │    │  └────┬─────┘  └─────┬─────┘  │ │
│  │  Stores (Zustand)    │    │       │ GPU          │ GPU     │ │
│  │                      │    │  ┌────▼──────────────▼─────┐  │ │
│  │                      │    │  │  Timeline Evaluator      │  │ │
│  │                      │    │  │  (Keyframes, Transitions) │  │ │
│  │                      │    │  └────────────┬────────────┘  │ │
│  │                      │    │  ┌────────────▼────────────┐  │ │
│  │                      │    │  │   Engine Core (wgpu)    │  │ │
│  │                      │    │  │   Compositing + Effekte │  │ │
│  └──────────────────────┘    │  └────────────┬────────────┘  │ │
│                              │               │               │ │
│  ┌──────────────────────┐    │  ┌────────────▼────────────┐  │ │
│  │  Native Preview      │◄───│──│   wgpu Surface (direct) │  │ │
│  │  (wgpu Window/Overlay)│    │  └────────────┬────────────┘  │ │
│  └──────────────────────┘    │               │ GPU           │ │
│                              │  ┌────────────▼────────────┐  │ │
│                              │  │   Encoder (GPU Export)  │  │ │
│                              │  └─────────────────────────┘  │ │
│                              └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Wichtige Architektur-Entscheidung:** Der Preview wird **nicht** durch den Webview gerendert. Stattdessen rendert wgpu direkt auf eine native Surface (eigenes Fenster oder Overlay über dem Webview). Das eliminiert den GPU→CPU Readback beim Preview komplett — der #1 Bottleneck. Alle professionellen NLEs (DaVinci Resolve, Premiere) machen das so. Der Webview ist nur für UI (Timeline, Panels, Controls).

---

## 2. Design-Prinzipien

### Code-Organisation
- **Ein Konzept pro Datei** — Dateiname beschreibt in 1-2 Worten was drin ist
- **Sweet Spot: 200-500 LOC** — Über 600 LOC → splitten. Unter 50 LOC → zusammenlegen
- **Kein künstliches Splitten** — 3 Dateien die sich ständig gegenseitig importieren sind schlechter als 1 Datei
- **Flache Hierarchien** — Max 2 Verzeichnis-Ebenen innerhalb eines Crates

### Rust-Patterns
- **Traits für Abstraktion** — Plattform-spezifischer Code hinter Traits verstecken
- **Error Handling** — `thiserror` für Library-Crates, `anyhow` nur im Tauri-Layer
- **No unsafe ohne Kommentar** — Jeder `unsafe` Block braucht ein `// SAFETY:` Kommentar
- **Builder Pattern** für komplexe Konfiguration
- **Zero-Cost Abstractions** — Generics statt `dyn Trait` im Hot Path

### Testing
- **Unit Tests** in jedem Modul (`#[cfg(test)] mod tests`)
- **Integration Tests** pro Crate (`tests/`)
- **GPU Tests** mit `wgpu` im Software-Renderer (CI-tauglich)
- **Benchmark Tests** mit `criterion` für Performance-kritische Pfade

---

## 3. GPU Pipeline (Zero-Copy)

### Decode → Render → Export (alles GPU, kein CPU-Roundtrip)

```
Video File (H.264/H.265/AV1)
  ↓ NVDEC / VAAPI / VideoToolbox (Hardware Decode)
  ↓ Frame bleibt auf GPU
GPU Texture (CUDA Surface / D3D11 Texture / VAAPI Surface)
  ↓ Import via gpu-interop Crate (plattform-abstrakt)
  ↓ Zero-Copy
wgpu Texture
  ↓ Timeline Evaluator: aktive Clips + interpolierte Params
  ↓ Compositing (WGSL Shader, 1:1 aus bestehender Engine)
  ↓ Color Management (Input → Working → Display Colorspace)
  ↓ Effekte (Color, Blur, Keying, etc.)
  ↓ AI Effekte (ONNX → Tensor, zurück zu wgpu Texture)
  ↓
Fertiges Frame
  ├──► Display: wgpu Surface direkt (KEIN Readback, kein CPU-Transfer)
  └──► Export:  GPU → Hardware Encoder (NVENC/VAAPI/VT) → Datei
```

### ProRes / DNxHR (kein Hardware-Decoder verfügbar)

```
ProRes File
  ↓ FFmpeg Software Decode → CPU RAM (unvermeidbar)
  ↓ Upload → GPU Texture
  ↓ Ab hier: selbe GPU Pipeline wie oben
```

### Performance-Erwartung (5 Layers, 1080p) — realistische Schätzung

| | Aktuell (Browser) | Native Engine (realistisch) | Native (Best Case) | Faktor |
|--|-------------------|----------------------------|--------------------|----|
| Decode | CPU ~8ms/Layer | GPU ~2-4ms/Layer | ~1ms/Layer | 2-4× |
| Transfer | WebSocket ~3ms/Layer | Zero-Copy ~0.2ms (wenn Interop klappt) | ~0.2ms | 15× |
| Transfer (Fallback) | — | CPU-Copy ~1-2ms/Layer | — | 2-3× |
| Compositing | Browser WebGPU ~1ms | Native wgpu ~0.7ms | ~0.5ms | 1.4× |
| **Total (5 Layers)** | **~63ms (15fps)** | **~15-25ms (40-60fps)** | ~8ms | **3-4×** |
| **Total (5× 4K)** | **~250ms (4fps)** | **~40-80ms (12-25fps)** | ~20ms | **3-6×** |

> **Anmerkung:** Die "Best Case" Spalte setzt perfekte Zero-Copy GPU Interop voraus. Realistisch werden wir auf manchen Plattformen/Treibern auf CPU-Copy zurückfallen. Ein 3-5× Speedup ist realistisch, 8-12× wäre außergewöhnlich. Zum Vergleich: DaVinci Resolve (20 Jahre Entwicklung) schafft ~60fps mit 5 Layers 4K.

---

## 4. Hardware Backends pro Plattform

| | Windows | Linux | macOS |
|--|---------|-------|-------|
| **Decode** | DXVA2 / D3D11VA | VAAPI | VideoToolbox |
| **GPU API** | wgpu → D3D12/Vulkan | wgpu → Vulkan | wgpu → Metal |
| **AI Runtime** | CUDA / DirectML | CUDA / Vulkan | CoreML / Metal |
| **Encode** | NVENC / QSV / AMF | NVENC / VAAPI | VideoToolbox |
| **GPU→wgpu Import** | DXGI Shared Handle | DMA-BUF / VK Ext. Memory | Metal Shared Texture |
| **Audio Output** | CPAL → WASAPI | CPAL → PulseAudio/ALSA | CPAL → CoreAudio |

---

## 5. AI Effekte (ONNX Runtime)

### Cross-Platform via ONNX

```rust
// Ein Modell, alle Plattformen:
let session = ort::Session::builder()?
    .with_execution_providers([
        CUDAExecutionProvider::default().build(),      // NVIDIA
        CoreMLExecutionProvider::default().build(),     // Apple
        DirectMLExecutionProvider::default().build(),   // Windows AMD/Intel
    ])?
    .commit_from_file("models/super_resolution.onnx")?;
```

### AI Features (priorisiert)

**Phase 5a — Kernfeatures (zuerst):**

| Feature | Modell | Nutzen |
|---------|--------|--------|
| **Super Resolution** | Real-ESRGAN / ESPCN | 720p → 4K Upscaling |
| **Background Removal** | SAM / MODNet | Automatisches Keying |
| **Speech-to-Text** | Whisper | Auto-Untertitel |

**Phase 5b — Erweitert (danach):**

| Feature | Modell | Nutzen |
|---------|--------|--------|
| **Frame Interpolation** | RIFE | Slow-Mo (24fps → 120fps) |
| **Noise Reduction** | NAFNet | AI Denoising |
| **Object Tracking** | ByteTrack / CoTracker | Motion Tracking |

**Phase 7 — Zukunft:**

| Feature | Modell | Nutzen |
|---------|--------|--------|
| **Style Transfer** | AdaIN | Look/Style anwenden |
| **Auto Color Match** | 3D-LUT Generation | Shots angleichen |
| **Object Removal** | ProPainter / E2FGVI | Inpainting in Video |

### AI Pipeline Integration

```
wgpu Texture (Input Frame)
  ↓ GPU → Tensor
  ↓ (CUDA: Zero-Copy möglich, DirectML/CoreML: kurzer CPU-Roundtrip wahrscheinlich)
ONNX Runtime Inference
  ↓ Tensor → GPU Texture
wgpu Texture (Processed Frame)
  ↓ Weiter im Compositing
```

> **Achtung:** Die Zero-Copy GPU Tensor Bridge (wgpu ↔ ONNX) ist auf CUDA machbar, aber auf DirectML/CoreML wahrscheinlich nicht ohne CPU-Zwischenschritt. Das muss als Fallback eingeplant werden.

---

## 6. Crate-Architektur

### Übersicht: 12 fokussierte Crates

```
crates/
├── common/          # Universelle Typen, Error Handling        (~1000 LOC)
├── gpu-context/     # wgpu Device, Queue, Capabilities         (~800 LOC)
├── gpu-interop/     # Plattform GPU Import Abstraction          (~3000 LOC)
├── frame-pool/      # GPU Texture Pool, FrameProvider Trait     (~1200 LOC)
├── decoder/         # FFmpeg Wrapper + HW Accel                 (~3500 LOC)
├── timeline-eval/   # Timeline @ Time T → aktive Layers         (~1500 LOC)
├── compositor/      # Layer Compositing, Blend, Transform       (~2000 LOC)
├── effects/         # Effekt-System + alle Effekte              (~3500 LOC)
├── ai/              # ONNX Runtime + AI Features                (~3000 LOC)
├── encoder/         # GPU Export Pipeline                       (~2500 LOC)
├── audio/           # Audio Decode, Mix, Timestretch, Output    (~2500 LOC)
└── display/         # Native Preview Surface Management         (~800 LOC)
                                                          Total: ~25.300 LOC
```

### Dependency Graph (kein Circular!)

```
common ←──────────────────────────────────────────────────────┐
  ↑                                                            │
gpu-context ←──────────────────────────────────────────┐       │
  ↑                                                     │       │
gpu-interop ←─────────────────────────────────────┐     │       │
  ↑                                                │     │       │
frame-pool ←───────────────────────────────┐       │     │       │
  ↑                                         │       │     │       │
decoder ───────────────────────────────────┤       │     │       │
  ↑                                         │       │     │       │
timeline-eval ←──────────────────────┐      │       │     │       │
  ↑                                   │      │       │     │       │
compositor ──────────────────────────┤      │       │     │       │
  ↑                                   │      │       │     │       │
effects ─────────────────────────────┤      │       │     │       │
  ↑                                   │      │       │     │       │
ai ──────────────────────────────────┘      │       │     │       │
  ↑                                          │       │     │       │
display ─────────────────────────────────────┤       │     │       │
  ↑                                          │       │     │       │
encoder ─────────────────────────────────────┘       │     │       │
  ↑                                                   │     │       │
audio ────────────────────────────────────────────────┘     │       │
  ↑                                                          │       │
src-tauri (orchestriert alles) ──────────────────────────────┘───────┘
```

---

### 6.1 `crates/common/` — Shared Foundation (~1000 LOC)

```
common/src/
├── lib.rs              # Re-exports
├── types.rs            # TimeCode, FrameNumber, Resolution, Rational
├── color.rs            # ColorSpace, PixelFormat, TransferFunction (nur Enums)
├── layer.rs            # LayerDesc: Source, Transform, Opacity — schlankes Struct
├── error.rs            # Zentrales Error Enum (thiserror)
└── config.rs           # Engine-Konfiguration, Feature Flags
```

**Verantwortung:** Typen die überall gebraucht werden. Keine Logik, keine Dependencies außer serde. **Keine Timeline-Typen** (die leben in `timeline-eval/`).

`LayerDesc` ist das Interface zwischen timeline-eval und compositor:
```rust
pub struct LayerDesc {
    pub source_id: String,
    pub transform: Transform2D,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub effects: Vec<EffectInstance>,
    pub mask: Option<MaskDesc>,
}
```

---

### 6.2 `crates/gpu-context/` — GPU Lifecycle (~800 LOC)

```
gpu-context/src/
├── lib.rs              # Re-exports
├── device.rs           # wgpu Device + Queue erstellen, Capabilities abfragen
├── surface.rs          # Output Surface Management (für native Preview)
├── limits.rs           # GPU Limits, Feature Detection, Fallbacks
└── diagnostics.rs      # GPU Info, Memory Usage, Debug Labels
```

**Verantwortung:** Einen `GpuContext` struct bereitstellen der Device + Queue + Capabilities kapselt.

```rust
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub capabilities: GpuCapabilities,
}
```

---

### 6.3 `crates/gpu-interop/` — Platform GPU Import (~3000 LOC)

```
gpu-interop/src/
├── lib.rs              # Trait GpuImporter + factory
├── trait.rs            # pub trait GpuImporter { fn import(...) -> wgpu::Texture }
├── cpu_fallback.rs     # CPU-Copy Fallback (alle Plattformen)
│
├── win/               # Windows-spezifisch
│   ├── mod.rs
│   ├── dxgi.rs         # DXGI Shared Handle → wgpu (create_texture_from_hal)
│   ├── d3d11.rs        # D3D11 Texture Interop
│   └── quirks.rs       # Driver-spezifische Workarounds
│
├── linux/             # Linux-spezifisch
│   ├── mod.rs
│   ├── dmabuf.rs       # DMA-BUF → Vulkan External Memory → wgpu
│   ├── vaapi.rs        # VAAPI Surface → DMA-BUF
│   └── quirks.rs       # AMD vs NVIDIA vs Intel Unterschiede
│
└── macos/             # macOS-spezifisch
    ├── mod.rs
    └── metal.rs        # Metal Shared Texture → wgpu
```

**Verantwortung:** Plattform-spezifischen GPU-Import hinter einem einheitlichen Trait verstecken. **Immer mit CPU-Copy Fallback** falls Zero-Copy auf dem jeweiligen Treiber nicht funktioniert.

```rust
pub trait GpuImporter: Send + Sync {
    /// Versucht Zero-Copy Import, fällt auf CPU-Copy zurück
    fn import_texture(
        &self,
        ctx: &GpuContext,
        handle: PlatformHandle,
        desc: &TextureDesc,
    ) -> Result<ImportResult>;
}

pub struct ImportResult {
    pub frame: GpuFrame,
    pub was_zero_copy: bool,  // Für Diagnostik
}

pub fn create_importer() -> Box<dyn GpuImporter> {
    #[cfg(target_os = "windows")]  DxgiImporter::new()
    #[cfg(target_os = "linux")]    DmaBufImporter::new()
    #[cfg(target_os = "macos")]    MetalImporter::new()
}
```

> **Anmerkung:** Das ist das höchste technische Risiko des Projekts. `wgpu::create_texture_from_hal()` ist unsafe, schlecht dokumentiert, und die HAL-API ist nicht stabil zwischen wgpu-Versionen. Jede Plattform hat eigene Driver Quirks (besonders AMD auf Linux). Die LOC-Schätzung enthält Platz für Workarounds und robustes Error-Handling.

---

### 6.4 `crates/frame-pool/` — GPU Memory Management (~1200 LOC)

```
frame-pool/src/
├── lib.rs              # Re-exports
├── pool.rs             # TexturePool: Alloc, Recycle, Evict (LRU)
├── gpu_frame.rs        # GpuFrame struct (Texture + View + Metadata)
├── provider.rs         # FrameProvider Trait — abstrahiert Decode→Import→Pool
├── ring_buffer.rs      # Frame Ring Buffer für Decode-Ahead
├── readback.rs         # GPU → CPU Readback (für Export-Fallback + Thumbnails)
└── stats.rs            # Pool-Statistiken, Memory Tracking
```

**Verantwortung:** GPU Texturen verwalten + `FrameProvider` Trait bereitstellen. Der Compositor weiß nichts über Decoder oder Import — er fragt den FrameProvider nach Frames.

```rust
pub struct GpuFrame {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub colorspace: ColorSpace,  // Input Colorspace (z.B. Rec.709, Log)
    pub pool_key: PoolKey,
}

/// Abstrahiert die gesamte Decode → Import → Pool Kette
pub trait FrameProvider: Send + Sync {
    fn get_frame(&self, source_id: &str, time: f64) -> Result<GpuFrame>;
    fn prefetch(&self, source_id: &str, time_range: Range<f64>);
    fn release(&self, frame: GpuFrame);
}
```

Der Compositor wird dadurch testbar (Mock-FrameProvider) und entkoppelt:
```rust
// Compositor weiß nichts über FFmpeg, Decoder, Import
compositor.render(layers, &frame_provider)?;
```

Die `FrameProvider`-Implementierung lebt in `src-tauri/` und verdrahtet Decoder + Importer + Pool.

---

### 6.5 `crates/decoder/` — Video Decode (~3500 LOC)

```
decoder/src/
├── lib.rs              # Re-exports + DecoderManager
├── manager.rs          # Decoder Lifecycle: Open, Seek, Decode, Close
├── ffmpeg/
│   ├── mod.rs
│   ├── context.rs      # AVFormatContext + AVCodecContext Wrapper (safe Rust)
│   ├── hwaccel.rs      # Hardware Accel Setup (NVDEC/VAAPI/VT)
│   ├── frame.rs        # AVFrame → PlatformHandle Konvertierung
│   ├── seek.rs         # Keyframe-basiertes Seeking (Long-GOP handling)
│   └── probe.rs        # File Probing: Codecs, Streams, Metadata, Color
├── pool.rs             # Decoder Pool: File → Decoder Mapping, Sharing
├── prefetch.rs         # Look-Ahead Prefetching Strategie
├── fallback.rs         # Software-Decode Fallback (ProRes, DNxHR)
├── thumbnail.rs        # Thumbnail-Generierung (für Timeline UI)
└── errors.rs           # Decoder-spezifische Fehlertypen
```

**Verantwortung:** Video-Dateien öffnen, Frames decodieren, GPU-Frames ausgeben.

```rust
pub trait Decoder: Send {
    fn decode(&mut self, frame: FrameNumber) -> Result<PlatformHandle>;
    fn seek(&mut self, frame: FrameNumber) -> Result<()>;
    fn probe(&self) -> &MediaInfo;
    fn colorspace(&self) -> ColorSpace;
}
```

> **Seeking ist komplex:** Long-GOP Codecs (H.264, H.265) haben nur alle 1-5 Sekunden ein Keyframe. Zum Seeking muss man zum vorherigen Keyframe springen und dann alle Frames bis zum Ziel decodieren. Das `seek.rs` Modul kapselt diese Logik inklusive Seek-Cache.

Wichtig: Der Decoder gibt `PlatformHandle` aus (CUDA ptr, D3D11 texture, VAAPI surface) — **nicht** wgpu Textures. Die Konvertierung macht `gpu-interop`.

---

### 6.6 `crates/timeline-eval/` — Timeline Evaluation (~1500 LOC)

```
timeline-eval/src/
├── lib.rs              # Re-exports
├── evaluator.rs        # Timeline @ Time T → Vec<LayerDesc>
├── types.rs            # Clip, Track, Marker, Transition (Rust-seitige Timeline-Typen)
├── keyframe.rs         # Keyframe Interpolation (Linear, Bezier, Hold, Step)
├── transition.rs       # Transition-Berechnung (Cross-Dissolve, Wipe, etc.)
├── nested.rs           # Nested Composition Evaluation
└── serde_bridge.rs     # Deserialization von React Timeline-State (JSON → Rust types)
```

**Verantwortung:** Beantwortet die Frage "Was wird bei Time T gerendert?" — berechnet aktive Clips, interpoliert Keyframe-Werte, löst Transitions auf.

```rust
pub struct TimelineEvaluator {
    timeline: Timeline,
}

impl TimelineEvaluator {
    /// React schickt die Timeline als JSON, Rust deserialisiert und cached
    pub fn update_timeline(&mut self, json: &str) -> Result<()>;

    /// Hauptmethode: Time → fertige Layer-Liste für Compositor
    pub fn evaluate(&self, time: f64) -> Vec<LayerDesc>;
}
```

**Warum eigenes Crate?** Ohne das müsste entweder:
- React bei jedem Frame die Evaluation machen und per IPC schicken (60fps × kompletter Layer-State = zu viel IPC-Overhead)
- Oder der Compositor Timeline-Logik kennen (verletzt Single Responsibility)

Datenfluss:
```
React: "Render at time 4.5s" (1× IPC, nur eine Zahl)
  ↓
timeline-eval: "Bei 4.5s: Clip A 100%, Clip B opacity 73% (interpoliert), Transition 40%"
  ↓
compositor: "Hier sind 3 LayerDescs → compositen"
```

---

### 6.7 `crates/compositor/` — Layer Compositing (~2000 LOC)

```
compositor/src/
├── lib.rs              # Re-exports
├── compositor.rs       # Hauptlogik: LayerDescs + FrameProvider → fertiges Frame
├── layer.rs            # Layer Processing: Source holen, Transform anwenden
├── blend.rs            # Blend Modes (Normal, Multiply, Screen, etc.)
├── transform.rs        # Position, Scale, Rotation, Anchor Point
├── mask.rs             # Masken: Rect, Ellipse, Path, Feather
├── transition.rs       # GPU-seitige Transition-Shader
├── color.rs            # Color Management: Input → Working → Output Colorspace
├── pipeline.rs         # wgpu Render/Compute Pipelines cachen
└── nested.rs           # Nested Composition Rendering
```

**Verantwortung:** Mehrere Layers zu einem Frame compositen. **Keine Timeline-Logik**, keine Decoder-Logik.

**Lineare Pipeline (Phase 4):** Layer-für-Layer compositen (Ping-Pong wie aktuelle Browser-Engine). Einfach, bewährt.

**Color Management** lebt als Modul im Compositor (nicht als eigenes Crate):
```rust
// compositor/color.rs
pub struct ColorPipeline {
    input_to_working: ColorTransform,   // z.B. S-Log3 → Linear
    working_to_display: ColorTransform, // z.B. Linear → sRGB/Rec.709
}
```

Unterstützte Colorspaces (Phase 4): sRGB, Rec.709, Linear.
Erweitert (Phase 7): Rec.2020, S-Log3, V-Log, ACES, HDR/PQ.

---

### 6.8 `crates/effects/` — Effekt-System (~3500 LOC)

```
effects/src/
├── lib.rs              # Re-exports + Effect Registry
├── registry.rs         # Alle Effekte registrieren, by-name lookup
├── trait.rs            # pub trait Effect: Params, Shader, Pipeline
├── params.rs           # Effekt-Parameter Typen (Float, Color, Curve, etc.)
├── shader.rs           # WGSL Shader laden, Pipeline erstellen
├── uniforms.rs         # Parameter → GPU Uniform Buffer Mapping
│
├── color/
│   ├── mod.rs          # Re-exports
│   ├── brightness.rs   # Brightness/Contrast
│   ├── hsl.rs          # Hue/Saturation/Lightness
│   ├── curves.rs       # RGB Curves
│   ├── lut.rs          # 3D LUT Apply
│   ├── white_balance.rs
│   ├── exposure.rs
│   └── color_wheels.rs # Lift/Gamma/Gain
│
├── blur/
│   ├── mod.rs
│   ├── gaussian.rs
│   ├── directional.rs
│   ├── radial.rs
│   └── zoom.rs
│
├── distort/
│   ├── mod.rs
│   ├── lens.rs         # Lens Distortion
│   ├── warp.rs
│   └── turbulence.rs
│
├── keying/
│   ├── mod.rs
│   ├── chroma_key.rs
│   └── luma_key.rs
│
├── stylize/
│   ├── mod.rs
│   ├── glow.rs
│   ├── sharpen.rs
│   ├── noise.rs
│   └── vignette.rs
│
└── generate/
    ├── mod.rs
    ├── solid.rs        # Solid Color Layer
    ├── gradient.rs
    └── text.rs         # Text Rendering
```

**Verantwortung:** Jeder Effekt ist eine Datei mit:
- Parameter-Definition (serializable)
- WGSL Shader Path
- Pipeline Setup + Uniform Binding

```rust
pub trait Effect: Send + Sync {
    fn name(&self) -> &str;
    fn params(&self) -> &[ParamDef];
    fn create_pipeline(&self, ctx: &GpuContext) -> Result<wgpu::RenderPipeline>;
    fn encode_pass(&self, params: &ParamValues, encoder: &mut wgpu::CommandEncoder);
}
```

Neuen Effekt hinzufügen = **1 Datei + 1 Zeile in registry.rs**. Kein anderer Code muss angefasst werden.

> **LOC-Hinweis:** WGSL Shader werden 1:1 übernommen, aber das Rust-seitige Binding (Pipeline-Erstellung, Bind Group Layout, Uniform-Buffer-Setup, Parameter-Validierung) ist pro Effekt ~80-120 LOC, nicht ~60.

---

### 6.9 `crates/ai/` — AI Effekte (~3000 LOC)

```
ai/src/
├── lib.rs              # Re-exports
├── runtime.rs          # ONNX Session Management, Provider Selection
├── bridge.rs           # wgpu Texture ↔ ONNX Tensor Konvertierung
├── bridge_cuda.rs      # CUDA-spezifisch: Zero-Copy GPU Tensor
├── bridge_cpu.rs       # Fallback: GPU → CPU → Tensor → CPU → GPU
├── model_cache.rs      # Modell Download, Cache, Versioning
│
├── effects/
│   ├── mod.rs          # Alle AI-Effekte als Effect trait
│   ├── super_res.rs    # Real-ESRGAN / ESPCN
│   ├── roto.rs         # Background Removal (SAM/MODNet)
│   ├── denoise.rs      # NAFNet
│   ├── interpolation.rs # RIFE Frame Interpolation
│   ├── tracking.rs     # Object Tracking (CoTracker)
│   ├── inpainting.rs   # Object Removal (ProPainter)
│   └── style.rs        # Style Transfer
│
└── whisper/
    ├── mod.rs          # Speech-to-Text (CPU-basiert, eigener Thread)
    └── subtitles.rs    # Whisper Output → Subtitle Format
```

**Verantwortung:** AI-Effekte implementieren dasselbe `Effect` Trait wie normale Effekte. Für die UI gibt es keinen Unterschied.

> **Wichtig:** Die GPU Tensor Bridge (`bridge.rs`) hat zwei Codepfade:
> - **CUDA (NVIDIA):** Potentiell Zero-Copy via CUDA-Interop. Komplex aber machbar.
> - **DirectML/CoreML (alle anderen):** CPU-Roundtrip wahrscheinlich unvermeidbar. ONNX Runtime gibt keinen direkten GPU-Pointer aus den man in wgpu importieren könnte.
>
> Phase 5 startet mit dem CPU-Roundtrip-Pfad (funktioniert überall) und optimiert CUDA Zero-Copy danach.

---

### 6.10 `crates/encoder/` — GPU Export (~2500 LOC)

```
encoder/src/
├── lib.rs              # Re-exports
├── pipeline.rs         # Export Pipeline: Timeline durchlaufen → Encode → Mux
├── gpu_encoder.rs      # wgpu Texture → Hardware Encoder (Zero-Copy)
├── hwaccel/
│   ├── mod.rs          # Trait HwEncoder
│   ├── nvenc.rs        # NVIDIA NVENC
│   ├── vaapi.rs        # Linux VAAPI
│   ├── videotoolbox.rs # macOS VideoToolbox
│   ├── qsv.rs          # Intel QuickSync
│   └── amf.rs          # AMD AMF
├── sw_encoder.rs       # Software Fallback (ProRes, DNxHR Encode via FFmpeg)
├── muxer.rs            # Container Muxing (MP4, MOV, MKV, WebM)
├── audio_encode.rs     # Audio Re-Encode (AAC, FLAC, PCM)
├── presets.rs          # Export Presets (YouTube, Instagram, Master, etc.)
└── progress.rs         # Export Progress Tracking + Cancellation
```

---

### 6.11 `crates/audio/` — Audio Processing (~2500 LOC)

```
audio/src/
├── lib.rs              # Re-exports
├── decoder.rs          # Audio Decode (via FFmpeg)
├── mixer.rs            # Multi-Track Audio Mixing (sample-genaue Sync)
├── output.rs           # CPAL Audio Output (Realtime Playback)
├── sync.rs             # Audio-Video Synchronisation
├── timestretch.rs      # Time Stretching (Rubberband Integration)
├── waveform.rs         # Waveform-Daten generieren (für Timeline UI)
├── meter.rs            # Lautstärke-Metering (LUFS, Peak, RMS)
├── resampler.rs        # Sample Rate Conversion
└── effects/
    ├── mod.rs
    ├── gain.rs         # Volume, Pan
    ├── eq.rs           # Equalizer
    └── compressor.rs   # Dynamic Compressor
```

**Audio-Architektur-Entscheidung: CPAL (nativ)**

Gründe gegen Web Audio API:
- Web Audio hat ~20ms Latenz durch den Webview
- Audio-Processing (Mix, Timestretch) soll in Rust laufen — Ergebnis zurück in den Webview schicken wäre absurd
- CPAL ist battle-tested (WASAPI/CoreAudio/PulseAudio) und ~200 LOC Setup

Audio läuft auf einem **eigenen dedizierten Thread** (Realtime-Priority).

---

### 6.12 `crates/display/` — Native Preview (~800 LOC)

```
display/src/
├── lib.rs              # Re-exports
├── surface.rs          # wgpu Surface erstellen (natives Fenster oder Overlay)
├── present.rs          # Frame → Surface präsentieren (VSync, Frame-Pacing)
├── resize.rs           # Surface Resize bei Viewport-Änderung
└── overlay.rs          # Overlay-Management (Webview ↔ Native Window Positioning)
```

**Verantwortung:** Fertiges Frame direkt auf dem Bildschirm anzeigen — ohne GPU→CPU Readback.

```
Ansatz: Natives Overlay-Fenster
  - Tauri erstellt ein rahmenloses Fenster
  - Positioniert über dem Preview-Bereich des Webviews
  - wgpu rendert direkt auf die Surface dieses Fensters
  - React steuert Position/Größe via IPC

Fallback: GPU Readback → Webview Canvas
  - Für Plattformen wo Overlay nicht funktioniert
  - Shared Memory → OffscreenCanvas → drawImage()
```

---

## 7. Threading-Modell

### Thread-Zuordnung

```
┌─────────────────────────────────────────────────────────┐
│ Thread 1: Tauri Main Thread                              │
│   - IPC Command Handling (Tauri invoke)                   │
│   - Event Emission (Rust → React)                        │
│   - NICHT blockieren! Nur dispatchen.                    │
│                                                          │
│ Thread 2: Render Thread (dediziert)                       │
│   - wgpu Command Submission (single-threaded!)           │
│   - Timeline Evaluation                                  │
│   - Compositing + Effects                                │
│   - Display Surface Present                              │
│   - Ownership: GpuContext, Compositor, FrameProvider     │
│                                                          │
│ Thread 3..N: Decode Pool (1 Thread pro aktive Datei)     │
│   - FFmpeg Decode (blockierend)                          │
│   - Prefetch Ring Buffer füllen                          │
│   - Thumbnail-Generierung                                │
│                                                          │
│ Thread N+1: Audio Thread (Realtime-Priority)             │
│   - CPAL Output Callback                                 │
│   - Audio Mixing                                         │
│   - Metering                                             │
│                                                          │
│ Thread N+2: AI Thread (on-demand)                        │
│   - ONNX Inference (kann 50-500ms dauern)                │
│   - Eigener Thread damit Render Thread nicht blockiert   │
│                                                          │
│ Thread N+3: Export Thread (on-demand)                    │
│   - Eigene Render-Loop (nicht an Display gebunden)       │
│   - Encode + Mux                                         │
│   - Unabhängig von Preview-Framerate                     │
└─────────────────────────────────────────────────────────┘
```

### Kommunikation zwischen Threads

```rust
// src-tauri/src/channels.rs

// Main → Render: "Rendere Frame bei Time T"
type RenderRequest = crossbeam::channel::Sender<RenderCommand>;

// Render → Main: "Frame fertig" (für Event-Emission)
type RenderResult = crossbeam::channel::Sender<FrameComplete>;

// Decoder → Frame Pool: "Decoded Frame bereit"
type DecodedFrame = crossbeam::channel::Sender<(SourceId, PlatformHandle)>;

// Main → Export: "Starte/Stoppe Export"
type ExportControl = crossbeam::channel::Sender<ExportCommand>;
```

**Wichtig:** GPU-Arbeit (`wgpu`) ist single-threaded. Alle GPU Command Submissions müssen vom Render Thread kommen. Andere Threads stellen Requests in eine Queue, der Render Thread verarbeitet sie.

---

## 8. Repo-Struktur (Gesamt)

```
masterselects/
│
├── src/                          # React UI (bleibt 1:1)
│   ├── components/
│   │   ├── timeline/
│   │   ├── panels/
│   │   ├── preview/              # Steuert Position des nativen Preview-Fensters
│   │   └── dock/
│   ├── stores/                   # Zustand State (Source of Truth für Media + Project)
│   ├── engine/                   # Browser-Engine (Fallback für Web-Version)
│   └── shaders/                  # WGSL Shader (Browser-Version)
│
├── src-tauri/                    # Tauri Shell + Orchestrator
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
│       ├── main.rs               # Tauri Entry Point
│       ├── orchestrator.rs       # Thread-Spawning, Lifecycle-Management
│       ├── channels.rs           # Message Types zwischen Threads
│       ├── frame_provider.rs     # FrameProvider Impl (Decoder + Importer + Pool)
│       ├── ipc/
│       │   ├── mod.rs            # IPC Command Registry
│       │   ├── decode.rs         # probe_file, open_file, generate_thumbnail
│       │   ├── render.rs         # render_at_time, update_timeline, set_viewport
│       │   ├── export.rs         # start_export, cancel_export
│       │   ├── ai.rs             # apply_ai_effect, download_model
│       │   └── project.rs        # save_project, load_project
│       └── state.rs              # App-weiter State (Arc<Mutex<...>>)
│
├── crates/
│   ├── common/                   # Universelle Typen               (~1000 LOC)
│   ├── gpu-context/              # wgpu Device + Queue              (~800 LOC)
│   ├── gpu-interop/              # DXGI / DMA-BUF / Metal           (~3000 LOC)
│   ├── frame-pool/               # GPU Texture Pool + FrameProvider (~1200 LOC)
│   ├── decoder/                  # FFmpeg + HW Decode               (~3500 LOC)
│   ├── timeline-eval/            # Timeline Evaluation              (~1500 LOC)
│   ├── compositor/               # Layer Compositing + Color Mgmt   (~2000 LOC)
│   ├── effects/                  # 30+ GPU Effekte                  (~3500 LOC)
│   ├── ai/                       # ONNX Runtime + AI Features       (~3000 LOC)
│   ├── encoder/                  # GPU Export Pipeline              (~2500 LOC)
│   ├── audio/                    # Audio Processing + CPAL Output   (~2500 LOC)
│   └── display/                  # Native Preview Surface           (~800 LOC)
│                                                              Total: ~25.300 LOC
│
├── shaders/                      # WGSL Shader (geteilt)
│   ├── composite.wgsl
│   ├── output.wgsl
│   ├── blend.wgsl
│   ├── color_transform.wgsl      # Color Management Shader
│   └── effects/
│       ├── brightness.wgsl
│       ├── gaussian_blur.wgsl
│       ├── chroma_key.wgsl
│       └── ...
│
├── models/                       # AI Modelle (.onnx) — .gitignore'd
│   └── .gitkeep                  # Werden on-demand heruntergeladen
│
├── Cargo.toml                    # Workspace Root
├── package.json
└── plan-native-engine.md
```

### Cargo Workspace

```toml
# /Cargo.toml
[workspace]
resolver = "2"
members = [
    "src-tauri",
    "crates/*",
]

[workspace.dependencies]
# GPU
wgpu = "24"
naga = "24"              # WGSL Parser/Validator

# AI
ort = "2"                # ONNX Runtime

# Video
ffmpeg-next = "7"

# Platform
ash = "0.38"             # Vulkan Raw Bindings
windows = "0.58"         # Win32/DXGI
metal = "0.29"           # macOS Metal

# Audio
cpal = "0.15"            # Cross-Platform Audio Output
rubberband-sys = "0.2"   # Time Stretching

# Async + Threading
tokio = { version = "1", features = ["full"] }
crossbeam = "0.8"        # Lock-free Channels

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Error Handling
thiserror = "2"
anyhow = "1"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# App
tauri = "2"

# Testing
criterion = "0.5"        # Benchmarks
```

---

## 9. IPC Design (Tauri ↔ React)

### Commands (React → Rust)

```typescript
import { invoke } from '@tauri-apps/api/core';

// Decode
const info = await invoke('probe_file', { path: '/video.mov' });
await invoke('open_file', { fileId, path });
await invoke('generate_thumbnail', { fileId, time: 5.0, width: 160 });

// Timeline (React schickt State-Updates, Rust cached)
await invoke('update_timeline', { timelineJson });
await invoke('render_at_time', { time: 4.5 });
await invoke('set_viewport', { width: 1920, height: 1080 });

// Preview Window Positioning
await invoke('set_preview_rect', { x, y, width, height });

// Effects
await invoke('update_effect', { clipId, effectId, params });

// AI
await invoke('apply_ai_effect', { type: 'super_res', clipId, params });
await invoke('download_model', { modelId: 'real-esrgan-x4' });

// Export
await invoke('start_export', { output, preset: 'youtube_4k' });
await invoke('cancel_export', { exportId });

// Project
await invoke('save_project', { path, projectJson });
const project = await invoke('load_project', { path });
```

### Events (Rust → React)

```rust
app.emit("render:complete", &RenderStats)?;      // Frame gerendert (FPS, GPU time)
app.emit("export:progress", &ExportProgress)?;    // Export %
app.emit("ai:progress", &AiProgress)?;            // AI Processing Status
app.emit("decode:ready", &FileInfo)?;              // File probed + ready
app.emit("audio:meters", &AudioLevels)?;           // Realtime Audio Levels
app.emit("gpu:stats", &GpuStats)?;                // VRAM Usage, Temp, Driver
app.emit("thumbnail:ready", &ThumbnailData)?;      // Thumbnail generiert
```

### Display Pipeline

```
Primär: Native wgpu Surface (KEIN Readback)
  - Tauri erstellt rahmenloses Child-Window über Preview-Bereich
  - wgpu rendert direkt auf Surface (VSync)
  - React steuert Position via invoke('set_preview_rect', ...)
  - React Overlays (Guides, Safe Areas) bleiben im Webview über dem Preview

Fallback: Shared Memory → Webview Canvas
  - Für Plattformen wo Overlay-Fenster nicht zuverlässig funktioniert
  - GPU Readback → Mapped Buffer → SharedArrayBuffer → OffscreenCanvas
  - ~2-3ms Latenz pro Frame
```

---

## 10. Was bleibt, was ändert sich

### Bleibt 1:1 (React/TypeScript)

- Alle UI Components (Timeline, Panels, Dock, Controls)
- Zustand Stores (Timeline, Media, History) — **React bleibt Source of Truth**
- Keyboard Shortcuts, Drag & Drop
- Project Save/Load (JSON) — React serialisiert, Rust speichert
- Panel Layout System
- Undo/Redo History
- Media Management (Import, Ordner, Proxy-Trigger)

### Wird ersetzt (Rust übernimmt)

| Vorher (Browser) | Nachher (Rust Crate) |
|-------------------|---------------------|
| `WebGPUEngine.ts` | `compositor/` |
| `TextureManager.ts` | `frame-pool/` + `gpu-context/` |
| `RenderDispatcher.ts` | `compositor/compositor.rs` |
| `LayerCollector.ts` | `timeline-eval/evaluator.rs` |
| `NativeDecoder.ts` | `decoder/` |
| `NativeHelperClient.ts` | Tauri IPC (kein WebSocket) |
| `FrameExporter.ts` | `encoder/pipeline.rs` |
| `VideoEncoderWrapper.ts` | `encoder/gpu_encoder.rs` |
| `AudioEncoder.ts` | `encoder/audio_encode.rs` |
| `AudioMixer.ts` | `audio/mixer.rs` |
| Browser Canvas Preview | `display/` (native wgpu Surface) |
| Browser WGSL Shaders | `shaders/` (geteilt, 1:1) |
| Web Audio API | `audio/output.rs` (CPAL) |

### Wird optional / Fallback

- Browser WebGPU Engine → Web-Version ohne Tauri
- WebSocket Native Helper → Legacy
- JPEG Frame Transfer → nicht mehr nötig

---

## 11. Migrationsplan (Phasen)

### Phase 0: GPU Interop Proof-of-Concept (2-3 Wochen) ⚠️ KRITISCH

**Ziel:** Beweisen dass Zero-Copy GPU Import funktioniert. **Gate:** Wenn das nicht klappt, muss die Architektur überdacht werden.

- [ ] Minimaler Rust-Binary (kein Tauri nötig)
- [ ] FFmpeg öffnet H.264 Datei mit DXVA2 Hardware Decode
- [ ] Decoded Frame (D3D11 Texture) → DXGI Shared Handle
- [ ] `wgpu::create_texture_from_hal()` → wgpu Texture
- [ ] wgpu Texture → native Window Surface rendern
- [ ] Performance messen: Frames/Sekunde, Latenz
- [ ] CPU-Copy Fallback implementieren + vergleichen
- [ ] Dokumentation: was funktioniert, was nicht, welche Driver

**Go/No-Go Entscheidung:**
- Zero-Copy funktioniert → weiter wie geplant
- Zero-Copy funktioniert nicht zuverlässig → CPU-Copy wird Standard-Pfad, Zero-Copy wird auf Phase 7 verschoben
- Nichts funktioniert → Architektur überdenken (ggf. wgpu HAL bypassen, direkter Vulkan/D3D12)

**Ergebnis:** Validierte GPU-Pipeline, realistische Performance-Zahlen.

---

### Phase 1: Tauri Shell (2-3 Wochen)

**Ziel:** Desktop-App, alles funktioniert wie bisher.

- [ ] Tauri 2 Setup (`src-tauri/`)
- [ ] React App in Webview laden
- [ ] Native Helper als Sidecar (Übergangsphase)
- [ ] WebSocket-Kommunikation bleibt erstmal
- [ ] Build + Installer (Windows first)
- [ ] Auto-Updates via Tauri Updater
- [ ] **Audio-Entscheidung validieren:** CPAL Test-Output (Sinus-Ton abspielen)

**Ergebnis:** Installierbare Desktop-App, selbe Funktionalität.

---

### Phase 2: Foundation Crates (4-6 Wochen)

**Ziel:** Basis-Layer die alle anderen Crates brauchen. GPU-Pipeline steht.

- [ ] `crates/common/` — Types, Errors, Config, LayerDesc
- [ ] `crates/gpu-context/` — wgpu Device Setup
- [ ] `crates/frame-pool/` — Texture Allokation + Recycling + FrameProvider Trait
- [ ] `crates/gpu-interop/` — DXGI Import (Windows, basierend auf Phase 0 PoC)
- [ ] `crates/gpu-interop/` — CPU-Copy Fallback
- [ ] `crates/display/` — Native Preview Surface (wgpu → Window)
- [ ] Unit Tests für alle Foundation Crates
- [ ] CI Pipeline: `cargo test`, `cargo clippy`, `cargo fmt`
- [ ] Integration: PoC-Code aus Phase 0 in saubere Crate-Struktur überführen

**Ergebnis:** Stabile Basis, GPU initialisiert, Preview-Fenster zeigt Test-Frame an.

---

### Phase 3: Decoder + Timeline Evaluation (6-10 Wochen)

**Ziel:** Hardware-Decode, Timeline-Auswertung, Frames auf Screen.

- [ ] `crates/decoder/` — FFmpeg + HW Accel
- [ ] Windows: DXVA2/D3D11VA → DXGI Handle → wgpu Texture
- [ ] Software Fallback für ProRes/DNxHR
- [ ] Keyframe-basiertes Seeking (Long-GOP)
- [ ] Decoder Pool (File-Sharing, max N gleichzeitige Decoder)
- [ ] Prefetch Ring Buffer (GPU-seitig)
- [ ] Thumbnail-Generierung
- [ ] `crates/timeline-eval/` — Timeline JSON → aktive LayerDescs
- [ ] Keyframe Interpolation (Linear, Bezier, Hold)
- [ ] Transition Evaluation
- [ ] `src-tauri/frame_provider.rs` — FrameProvider Implementierung
- [ ] `src-tauri/orchestrator.rs` — Render Thread + Decode Threads
- [ ] Tauri IPC Commands: `probe_file`, `open_file`, `render_at_time`, `update_timeline`
- [ ] Integration Test: Datei öffnen → Timeline evaluieren → GPU Frame → Preview
- [ ] **Crash Recovery:** FFmpeg/GPU Driver Crashes abfangen (Sidecar-Isolation?)

**Ergebnis:** Video-Dateien decodieren, Timeline auswerten, einzelne Frames im nativen Preview anzeigen.

---

### Phase 4: Render Engine (8-12 Wochen)

**Ziel:** Vollständiges Compositing + Effekte in wgpu.

- [ ] `crates/compositor/` — Layer Compositing (lineare Ping-Pong Pipeline)
- [ ] Blend Modes, Transform, Opacity
- [ ] Masken (Rect, Ellipse, Path, Feather)
- [ ] Transitions (Cross-Dissolve, Wipe, etc.)
- [ ] `compositor/color.rs` — Basis Color Management (sRGB, Rec.709, Linear)
- [ ] Nested Composition Rendering
- [ ] `crates/effects/` — Alle 30+ Effekte portieren (WGSL 1:1, Rust Bindings neu)
- [ ] Tauri IPC: `render_at_time` mit vollem Compositing
- [ ] Browser-Engine als Fallback behalten
- [ ] Benchmark: 5 Layers 1080p, Vergleich mit Browser-Engine
- [ ] **Project Format:** Migration-Layer für bestehende Projekte

**Ergebnis:** Full GPU Rendering. Preview im nativen Fenster. Browser-Engine als Fallback.

---

### Phase 5: AI Engine (8-12 Wochen)

**Ziel:** AI-Effekte, GPU-native. Fokus auf 3 Kernfeatures.

**Phase 5a: Infrastruktur + Super Resolution (4-6 Wochen)**
- [ ] `crates/ai/runtime.rs` — ONNX Runtime Setup
- [ ] Execution Provider: CUDA / DirectML / CoreML
- [ ] `crates/ai/bridge_cpu.rs` — CPU-Roundtrip Fallback (funktioniert überall)
- [ ] Modell Download + Cache System
- [ ] Super Resolution (Real-ESRGAN) als erster Effekt
- [ ] UI: AI-Effekte im Effects Panel

**Phase 5b: Weitere Effekte (4-6 Wochen)**
- [ ] Background Removal (SAM/MODNet)
- [ ] Speech-to-Text (Whisper, CPU-basiert)
- [ ] `crates/ai/bridge_cuda.rs` — CUDA Zero-Copy Optimierung (NVIDIA only)

**Ergebnis:** AI Features auf allen Plattformen (CPU-Fallback), optimiert auf NVIDIA.

---

### Phase 6: Export + Audio (6-8 Wochen)

**Ziel:** GPU Export, Audio Processing, vollständige Pipeline.

- [ ] `crates/encoder/` — Hardware Encode (NVENC first)
- [ ] GPU Texture → Encoder (Zero-Copy wo möglich)
- [ ] Software Fallback (ProRes, DNxHR)
- [ ] Container Muxing, Audio Encode
- [ ] Export Presets
- [ ] Export Progress + Cancellation
- [ ] `crates/audio/` — Decode, Mix, Resampling
- [ ] `audio/output.rs` — CPAL Realtime Output
- [ ] `audio/sync.rs` — Audio-Video Synchronisation
- [ ] Timestretch (Rubberband Integration)
- [ ] Audio Metering (LUFS, Peak, RMS)
- [ ] Waveform Generation (für Timeline UI)

**Ergebnis:** Complete Pipeline — Decode → Timeline Eval → Render → AI → Export, alles GPU. Audio nativ.

---

### Phase 7: Polish + Advanced (ongoing)

- [ ] Linux: VAAPI + DMA-BUF Support in `gpu-interop`
- [ ] macOS: VideoToolbox + Metal Support
- [ ] Render Graph (topologische Sortierung, Pass-Merging) — Ersatz für lineare Pipeline
- [ ] Erweiterte Colorspaces (Rec.2020, S-Log3, V-Log, ACES, HDR/PQ)
- [ ] Multi-GPU Support
- [ ] Vulkan Video (FFmpeg bypassen)
- [ ] Plugin System
- [ ] Proxy-freies 4K/8K Editing
- [ ] AI: Frame Interpolation, Object Tracking, Inpainting, Style Transfer
- [ ] AI: CUDA Zero-Copy Tensor Bridge verfeinern
- [ ] Audio: EQ, Compressor, Reverb
- [ ] File Watcher (externe Änderungen an Media-Dateien erkennen)

---

## 12. Dependencies (Rust Crates)

| Crate | Version | Zweck |
|-------|---------|-------|
| `tauri` | 2.x | Desktop Shell, IPC |
| `wgpu` | 24.x | GPU Rendering (Vulkan/D3D12/Metal) |
| `naga` | 24.x | WGSL Parser/Validator |
| `ort` | 2.x | ONNX Runtime (AI Inference) |
| `ffmpeg-next` | 7.x | Video Decode/Encode |
| `ash` | 0.38 | Vulkan Raw Bindings (GPU Interop) |
| `windows` | 0.58 | Win32/DXGI APIs |
| `metal` | 0.29 | Metal APIs (macOS) |
| `cpal` | 0.15 | Cross-Platform Audio Output |
| `rubberband-sys` | 0.2 | Audio Time Stretching |
| `tokio` | 1.x | Async Runtime |
| `crossbeam` | 0.8 | Lock-free Channels (Thread-Kommunikation) |
| `serde` | 1.x | Serialization (IPC, Project Files) |
| `thiserror` | 2.x | Error Types (Library Crates) |
| `anyhow` | 1.x | Error Handling (Tauri Layer) |
| `tracing` | 0.1 | Structured Logging |
| `tracing-subscriber` | 0.3 | Log Output |
| `criterion` | 0.5 | Benchmarks |

---

## 13. Risiken & Mitigations

### Risiken (nach Schwere sortiert)

| # | Risiko | Impact | Wahrscheinl. | Mitigation |
|---|--------|--------|-------------|------------|
| 1 | **GPU Interop (DXGI/DMA-BUF)** — `wgpu::create_texture_from_hal()` ist unsafe, schlecht dokumentiert, HAL-API instabil zwischen wgpu-Versionen | Hoch | Hoch | **Phase 0 PoC.** CPU-Copy Fallback immer bereit. wgpu Issues aktiv tracken. |
| 2 | **wgpu HAL Breaking Changes** — Minor-Version-Upgrades können Interop-Code brechen | Hoch | Mittel | wgpu Version pinnen, nur bewusst upgraden. HAL-Code isolieren in `gpu-interop`. |
| 3 | **ONNX GPU ↔ wgpu Bridge** — Zero-Copy zwischen ONNX Runtime und wgpu ist auf nicht-CUDA Plattformen wahrscheinlich unmöglich | Mittel | Hoch | CPU-Roundtrip als Standard. CUDA Zero-Copy als Optimierung. |
| 4 | **FFmpeg Cross-Platform Linking** — Static vs Dynamic, GPL/LGPL Probleme, Windows Build-Toolchain | Mittel | Mittel | Static Linking bevorzugen. vcpkg für Windows. Lizenzen vorab klären. |
| 5 | **Native Crash Recovery** — SIGSEGV in FFmpeg oder GPU Driver tötet den Prozess (im Browser wäre das isoliert) | Mittel | Mittel | Decoder in Sidecar-Prozess isolieren? Oder Crash-Reporter + Auto-Restart. |
| 6 | **Native Preview Overlay** — Rahmloses Fenster über Webview positionieren ist plattform-spezifisch, Window-Management Quirks | Mittel | Mittel | Tauri Window-APIs testen. Fallback: Readback → Webview Canvas. |
| 7 | **macOS Metal + wgpu** — Apple-spezifische Bugs, MoltenVK Limitierungen | Mittel | Mittel | macOS zuletzt (Phase 7). Metal-Backend direkt statt MoltenVK. |
| 8 | **Driver Quirks (AMD Linux)** — DMA-BUF Interop funktioniert unterschiedlich auf AMD vs NVIDIA vs Intel | Mittel | Hoch | `gpu-interop/linux/quirks.rs`. Pro-Treiber testen. CPU-Copy Fallback. |
| 9 | **Audio-Video Sync** — Sample-genaue Synchronisation zwischen CPAL Output und GPU Render ist nicht-trivial | Niedrig | Mittel | Audio Clock als Master. Render Thread synchronisiert auf Audio-Timestamps. |

### Entschiedene Fragen

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| **Audio Output** | CPAL (nativ) | Web Audio hat ~20ms Latenz, Audio-Processing soll in Rust laufen |
| **Preview Display** | Native wgpu Surface | Eliminiert GPU→CPU Readback. Fallback: Shared Memory → Webview |
| **AI Modelle** | On-demand Download | Bundlen würde Installer ~500MB+ vergrößern |
| **Media Management** | React bleibt Source of Truth | Kein Rust `media-store` — verhindert zwei Sources of Truth |
| **Render Graph** | Phase 7 (nicht Phase 4) | Lineare Pipeline reicht für aktuelle Anforderungen, weniger Komplexität |
| **Timeline-Typen** | In `timeline-eval/` integriert | Kein eigenes Crate — zu wenig Code für eigene Compile-Unit |

### Offene Fragen

1. **Web-Version behalten?** Browser-Engine als Fallback für Web-Version ohne Tauri?
2. **Lizenzen:** FFmpeg GPL/LGPL Kompatibilität mit proprietärer App prüfen
3. **Shader Sharing:** WGSL Dateien symlinken oder zur Build-Zeit via `include_str!()` einbetten?
4. **Proxy Workflow:** Mit GPU-Power noch nötig? (Wahrscheinlich ja für 8K+ Material)
5. **Crash Isolation:** Decoder in Sidecar-Prozess oder in-process mit Crash-Handler?

---

## 14. Timeline (realistisch)

```
Phase 0: GPU Interop PoC ........... 2-3 Wochen    ← GATE
Phase 1: Tauri Shell ............... 2-3 Wochen
Phase 2: Foundation Crates ......... 4-6 Wochen
Phase 3: Decoder + Timeline Eval ... 6-10 Wochen
Phase 4: Render Engine ............. 8-12 Wochen
Phase 5: AI Engine ................. 8-12 Wochen    ← teilweise parallel mit Phase 4
Phase 6: Export + Audio ............ 6-8 Wochen     ← teilweise parallel mit Phase 5
Phase 7: Polish + Advanced ......... ongoing
                                     ───────────
Gesamt bis Feature-Complete:         ~14-18 Monate (Windows)
Cross-Platform (Linux + macOS):      +3-6 Monate
```

### Parallelisierung

```
Monat:  1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16
        ├───┤
        Ph0 (PoC)
            ├───┤
            Ph1 (Tauri)
                ├───────┤
                Ph2 (Foundation)
                        ├───────────────┤
                        Ph3 (Decoder + Timeline)
                                        ├───────────────────┤
                                        Ph4 (Render Engine)
                                                ├───────────────────┤
                                                Ph5 (AI Engine)
                                                        ├───────────────┤
                                                        Ph6 (Export+Audio)
                                                                        ├──── Ph7 ongoing
```

> **Anmerkung:** Diese Timeline geht von ~1 Vollzeit-Entwickler aus. Mit 2 Entwicklern (Frontend + Rust) könnte Phase 4+5 stärker parallelisiert werden (~10-12 Monate). Die Browser-Version funktioniert während der gesamten Migration weiter.

---

## 15. LOC-Verteilung (realistische Schätzung)

```
Rust Engine (crates/):
  common .............. ~1000 LOC
  gpu-context ......... ~800 LOC
  gpu-interop ......... ~3000 LOC  (3 Plattformen × ~800 + Fallback + Quirks)
  frame-pool .......... ~1200 LOC  (Pool + FrameProvider + Readback)
  decoder ............. ~3500 LOC  (FFmpeg + HW Accel + Seeking + Prefetch + Thumbnails)
  timeline-eval ....... ~1500 LOC  (Evaluator + Types + Keyframes + Transitions)
  compositor .......... ~2000 LOC  (Compositing + Color + Masks + Nested)
  effects ............. ~3500 LOC  (30 Effekte × ~100 LOC Rust + Registry + Uniforms)
  ai .................. ~3000 LOC  (Runtime + Bridge + 3 Effekte + Model Cache)
  encoder ............. ~2500 LOC  (HW Encode + SW Fallback + Mux + Presets)
  audio ............... ~2500 LOC  (Decode + Mix + CPAL + Timestretch + Waveform + Meter)
  display ............. ~800 LOC   (Native Surface + Present + Overlay)
  ─────────────────────────────
  Subtotal:           ~25.300 LOC

Tauri Shell (src-tauri/):
  IPC + State ......... ~1000 LOC
  Orchestrator ........ ~600 LOC   (Threads, Channels)
  FrameProvider Impl .. ~400 LOC
  ─────────────────────────────
  Subtotal:           ~2.000 LOC

WGSL Shaders:
  Composite + Effects .. ~1500 LOC  (großteils übernommen)

React UI:
  Anpassungen .......... ~500 LOC  (Preview Positioning, IPC-Bridge, Tauri Commands)

                        ═══════════
TOTAL Neu:              ~29.300 LOC Rust + Shader + React-Anpassungen
```

Größte Einzeldatei: ~500 LOC. Kein File über 600 LOC.
