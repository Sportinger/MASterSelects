# Native Engine Plan — MasterSelects (CUDA + Vulkan + egui)

> **Ziel:** Vollständig native Desktop-App in Rust. Dual-GPU-Backend (CUDA + Vulkan Compute), egui UI, kein Tauri, kein FFmpeg, eigener Demuxer, eigene NVDEC/NVENC FFI Bindings.
> Aufgebaut auf bestehendem egui-Mockup in `native-ui/`.
> Plan ist für parallele AI-Agent-Entwicklung optimiert — unabhängige Crates mit stabilen Interfaces.

---

## 1. Architektur-Übersicht

```
┌──────────────────────────────────────────────────────────────────────┐
│  Native App (egui + Dual GPU Backend)                                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  egui UI (eframe → wgpu Backend)                                │  │
│  │  Timeline, Panels, Toolbar, Preview                             │  │
│  │  Bestehender Code aus native-ui/                                │  │
│  └──────────────────────┬─────────────────────────────────────────┘  │
│                          │ GPU Texture → wgpu Texture                 │
│  ┌──────────────────────▼─────────────────────────────────────────┐  │
│  │  Engine Core (trait GpuBackend)                                  │  │
│  │                                                                  │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │  GPU Backend Abstraction (gpu-hal/)                      │    │  │
│  │  │                                                          │    │  │
│  │  │  ┌──────────────────┐    ┌──────────────────────────┐   │    │  │
│  │  │  │  CUDA Backend    │    │  Vulkan Compute Backend  │   │    │  │
│  │  │  │  ├ NVDEC Decode  │    │  ├ Vulkan Video Decode   │   │    │  │
│  │  │  │  ├ CUDA Kernels  │    │  ├ GLSL Compute Shaders  │   │    │  │
│  │  │  │  ├ NVENC Encode  │    │  ├ Vulkan Video Encode   │   │    │  │
│  │  │  │  └ cudarc + FFI  │    │  └ ash + gpu-allocator   │   │    │  │
│  │  │  └──────────────────┘    └──────────────────────────────┘   │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  │                                                                  │  │
│  │  ┌──────────┐  ┌─────────────┐  ┌───────────────────┐          │  │
│  │  │ Demuxer  │  │ Compositor  │  │ Effects Engine    │          │  │
│  │  │ (eigen)  │  │ (gpu-hal)   │  │ (gpu-hal)         │          │  │
│  │  └──────────┘  └─────────────┘  └───────────────────┘          │  │
│  │                                                                  │  │
│  │  Preview: GPU → Staging Buffer → wgpu Texture → egui           │  │
│  │  Export:  GPU → Encoder → Muxer → Datei                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Audio: Symphonia (Decode) + CPAL (Output) + Mixer              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

**Kernentscheidungen:**
- **Kein Tauri** — Alles nativ in Rust, egui für komplette UI
- **Kein FFmpeg** — Eigener Demuxer, eigene FFI Bindings (NVDEC/NVENC), Symphonia nur für Audio, Muxide für Container Output
- **Dual GPU Backend** — CUDA für NVIDIA, Vulkan Compute für AMD/Intel/Linux/Mac (via MoltenVK)
- **GPU Abstraction Layer** — `trait GpuBackend` abstrahiert CUDA und Vulkan, Effects in `.cu` UND `.comp`
- **wgpu nur für egui** — egui's eigenes Rendering, nicht für Video-Processing
- **Eigene FFI Bindings** — Keine Abhängigkeit von nvidia-video-codec-sdk Crate, direkte C-Bindings
- **Eigener Demuxer** — Volle Kontrolle über NAL Unit Extraction für NVDEC
- **AI nach v1** — ONNX Runtime + AI-Effekte sind nicht im v1 Scope
- **Projektformat-Kompatibilität** — Liest/schreibt dieselben JSON-Projektdateien wie die Web-App

---

## 2. Design-Prinzipien

### Code-Organisation
- **Ein Konzept pro Datei** — Dateiname = Inhalt in 1-2 Worten
- **Sweet Spot: 200-500 LOC** — Über 600 LOC → splitten
- **Flache Hierarchien** — Max 2 Verzeichnis-Ebenen innerhalb eines Crates
- **`pub(crate)`** für interne APIs, `pub` nur für echte öffentliche Schnittstellen
- **Multi-Agent-freundlich** — Stabile Trait-Interfaces in `common/` definiert, Crates unabhängig entwickelbar

### Rust-Patterns
- **Error Handling** — `thiserror` in Library-Crates, `anyhow` nur im App-Layer
- **No unsafe ohne Kommentar** — Jeder `unsafe` Block braucht `// SAFETY:` Kommentar
- **Builder Pattern** für komplexe Konfiguration (Encoder, Decoder, Effects)
- **Zero-Cost Abstractions** — Generics statt `dyn Trait` im Hot Path
- **Newtype Pattern** — Typensicherheit für FrameNumber, TimeCode, ByteOffset
- **Functional State Updates** — Kein Stale-State in Callbacks

### Ownership & Memory
- **RAII** für alle GPU-Ressourcen (CUDA Memory, Vulkan Buffers, Texturen, Decoder-Sessions)
- **Channels statt Shared State** — `crossbeam::channel` für Thread-Kommunikation
- **Pinned/Staging Memory** für CPU↔GPU Transfers
- **Zero-Copy** innerhalb der GPU Pipeline — Daten bleiben auf GPU Device
- **Keine unnötigen Clones** — Borrow statt Copy, `Cow<T>` wo nötig

### Concurrency
- **Rayon** für Daten-Parallelismus (Batch-Operationen)
- **crossbeam** für Message-Passing zwischen Threads
- **Dedizierte Threads** für Render, Decode, Audio
- **CUDA Streams / Vulkan Queues** für asynchrone GPU-Operationen

### Performance
- **Cache-Friendly Data** — Struct-of-Arrays wo sinnvoll
- **`#[inline(always)]`** für Hot-Path Funktionen (Pixel-Sampling, Interpolation)
- **Stack-Allokation** für kleine temporäre Buffers
- **Keine Heap-Allokation in Tight Loops**

### Testing
- **Unit Tests** in jedem Modul (`#[cfg(test)] mod tests`)
- **Integration Tests** pro Crate (`tests/`)
- **Property-Based Tests** mit `proptest` für Codec-Roundtrips
- **Benchmarks** mit `criterion` für Performance-kritische Pfade
- **GPU Tests** mit echtem GPU Device (kein Mock)
- **Miri** für Unsafe-Code Validierung

### Code Quality
- `cargo fmt --all` vor jedem Commit
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo deny check` für Dependency-Audit
- `cargo test --all-features` muss grün sein

---

## 3. GPU Abstraction Layer (Kern-Innovation)

### Das zentrale Trait

```rust
// gpu-hal/src/traits.rs

/// Abstrahiert CUDA und Vulkan Compute
pub trait GpuBackend: Send + Sync {
    type Buffer: GpuBuffer;
    type Texture: GpuTexture;
    type Stream: GpuStream;
    type Decoder: HwDecoder;
    type Encoder: HwEncoder;

    // Device Management
    fn device_name(&self) -> &str;
    fn vram_total(&self) -> u64;
    fn vram_used(&self) -> u64;

    // Memory
    fn alloc_buffer(&self, size: usize) -> Result<Self::Buffer>;
    fn alloc_texture(&self, width: u32, height: u32, format: PixelFormat) -> Result<Self::Texture>;
    fn staging_buffer(&self, size: usize) -> Result<StagingBuffer>;

    // Streams / Command Queues
    fn create_stream(&self) -> Result<Self::Stream>;
    fn synchronize(&self, stream: &Self::Stream) -> Result<()>;

    // Kernel / Shader Dispatch
    fn dispatch_kernel(
        &self,
        kernel: &KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        args: &KernelArgs,
        stream: &Self::Stream,
    ) -> Result<()>;

    // Transfer
    fn copy_to_host(&self, src: &Self::Buffer, dst: &mut [u8], stream: &Self::Stream) -> Result<()>;
    fn copy_to_device(&self, src: &[u8], dst: &Self::Buffer, stream: &Self::Stream) -> Result<()>;
    fn copy_buffer(&self, src: &Self::Buffer, dst: &Self::Buffer, stream: &Self::Stream) -> Result<()>;

    // Hardware Decode/Encode
    fn create_decoder(&self, codec: VideoCodec, config: &DecoderConfig) -> Result<Self::Decoder>;
    fn create_encoder(&self, codec: VideoCodec, config: &EncoderConfig) -> Result<Self::Encoder>;

    // Display Bridge
    fn gpu_to_wgpu(&self, src: &Self::Texture, wgpu_tex: &wgpu::Texture, queue: &wgpu::Queue) -> Result<()>;
}

pub trait HwDecoder: Send {
    fn decode(&mut self, packet: &NalPacket) -> Result<GpuFrame>;
    fn flush(&mut self) -> Result<Vec<GpuFrame>>;
}

pub trait HwEncoder: Send {
    fn encode(&mut self, frame: &GpuFrame) -> Result<EncodedPacket>;
    fn flush(&mut self) -> Result<Vec<EncodedPacket>>;
}
```

### Dual Kernel Strategie

Jeder GPU-Effekt existiert in zwei Varianten:

```
kernels/
├── cuda/                    # CUDA Kernels (.cu → .ptx via nvcc)
│   ├── composite.cu
│   ├── blend.cu
│   ├── transform.cu
│   ├── nv12_to_rgba.cu
│   ├── color_convert.cu
│   ├── mask.cu
│   ├── transition.cu
│   └── effects/
│       ├── brightness.cu
│       ├── gaussian_blur.cu
│       ├── chroma_key.cu
│       └── ... (30+ Effekte)
│
└── vulkan/                  # Vulkan Compute Shaders (.comp → .spv via glslc)
    ├── composite.comp
    ├── blend.comp
    ├── transform.comp
    ├── nv12_to_rgba.comp
    ├── color_convert.comp
    ├── mask.comp
    ├── transition.comp
    └── effects/
        ├── brightness.comp
        ├── gaussian_blur.comp
        ├── chroma_key.comp
        └── ... (30+ Effekte)
```

### Build Pipeline

```rust
// build.rs
fn main() {
    // CUDA: .cu → .ptx via nvcc (nur wenn Feature "cuda" aktiv)
    #[cfg(feature = "cuda")]
    {
        for kernel in glob("kernels/cuda/**/*.cu") {
            compile_cuda_to_ptx(&kernel);
        }
    }

    // Vulkan: .comp → .spv via glslc (immer)
    for shader in glob("kernels/vulkan/**/*.comp") {
        compile_glsl_to_spirv(&shader);
    }
}
```

### Backend-Auswahl zur Laufzeit

```rust
pub fn select_backend() -> Box<dyn GpuBackend> {
    // 1. Versuche CUDA (NVIDIA)
    if let Ok(cuda) = CudaBackend::new() {
        log::info!("Using CUDA backend: {}", cuda.device_name());
        return Box::new(cuda);
    }

    // 2. Fallback: Vulkan Compute (AMD, Intel, NVIDIA ohne CUDA)
    if let Ok(vulkan) = VulkanBackend::new() {
        log::info!("Using Vulkan backend: {}", vulkan.device_name());
        return Box::new(vulkan);
    }

    panic!("No GPU backend available. CUDA or Vulkan required.");
}
```

### Performance-Erwartung (5 Layers, 1080p)

| | Browser (aktuell) | CUDA | Vulkan Compute |
|--|-------------------|------|----------------|
| Decode | CPU ~8ms/Layer | NVDEC ~1-2ms | Vulkan Video ~2-3ms |
| Transfer | WebSocket ~3ms | Zero-Copy 0ms | Zero-Copy 0ms |
| Compositing | WebGPU ~1ms | CUDA ~0.5ms | Vulkan ~0.7ms |
| Display | — | Pinned ~0.5ms | Staging ~0.6ms |
| **Total (5 Layers)** | **~63ms (15fps)** | **~8-12ms (80-120fps)** | **~12-18ms (55-80fps)** |

---

## 4. Eigener Demuxer (kein FFmpeg, kein Symphonia für Video)

### Warum eigener Demuxer?

- **NVDEC braucht NAL Units** — Symphonia gibt Audio-Pakete, keine Video NAL Units
- **Volle Kontrolle** — AVCC↔Annex-B Konvertierung, SPS/PPS Extraktion, Keyframe-Index
- **Vulkan Video** — Braucht ebenfalls spezifisch formatierte NAL Units
- **Kein FFmpeg** — Konsequent kein FFmpeg in der gesamten Pipeline

### Unterstützte Container

| Container | Parser | Codec-Support |
|-----------|--------|---------------|
| **MP4/MOV** (ISO BMFF) | Eigener Box-Parser | H.264, H.265, AV1 |
| **MKV/WebM** (Matroska/EBML) | Eigener EBML-Parser | H.264, H.265, VP9, AV1 |

### Architektur

```
demux/src/
├── lib.rs           # Re-exports
├── probe.rs         # File Probing: Codecs, Streams, Duration, Metadata
├── traits.rs        # pub trait Demuxer: open, read_packet, seek
├── packet.rs        # VideoPacket (NAL Units), AudioPacket
├── nal.rs           # NAL Unit Parser: AVCC → Annex-B, SPS/PPS, Slice Types
│
├── mp4/
│   ├── mod.rs       # MP4 Demuxer Implementation
│   ├── boxes.rs     # ISO BMFF Box Parser (moov, trak, stbl, mdat)
│   ├── sample.rs    # Sample Table → Byte Offsets + Sizes
│   └── avc_config.rs # AVCDecoderConfigurationRecord Parser
│
├── mkv/
│   ├── mod.rs       # MKV/WebM Demuxer Implementation
│   ├── ebml.rs      # EBML Element Parser
│   ├── elements.rs  # Matroska Element IDs + Parsing
│   └── cluster.rs   # Cluster/Block → NAL Extraction
│
└── index.rs         # Keyframe Index für Random Access / Seeking
```

### NAL Unit Extraction

```rust
/// Video-Paket mit NAL Units, bereit für NVDEC/Vulkan Video
pub struct VideoPacket {
    pub data: Vec<u8>,           // NAL Units (Annex-B Format: 0x00000001 + NALU)
    pub pts: TimeCode,
    pub dts: TimeCode,
    pub is_keyframe: bool,
    pub codec: VideoCodec,
}

/// AVCC → Annex-B Konvertierung (MP4 speichert AVCC, NVDEC will Annex-B)
pub fn avcc_to_annexb(avcc_data: &[u8], length_size: u8) -> Vec<u8> {
    // Ersetzt 4-Byte Length Prefix durch 0x00000001 Start Code
    // Prepended SPS/PPS bei Keyframes
}
```

---

## 5. Eigene NVDEC/NVENC FFI Bindings

### Warum eigene Bindings?

- `nvidia-video-codec-sdk` Crate ist v0.4, von einer Person maintained, 2K Downloads total
- Direkte C-Bindings gegen die offiziellen NVIDIA Video Codec SDK Headers
- Volle Kontrolle über API-Surface, kein Risiko durch fremde Breaking Changes
- Nur die Funktionen binden, die wir brauchen

### Struktur

```
gpu-hal/src/cuda/
├── mod.rs           # CUDA Backend Implementation
├── context.rs       # cudarc-basierter CUDA Context
├── memory.rs        # DeviceBuffer, PinnedBuffer (via cudarc)
├── kernel.rs        # PTX laden + Kernel Launch (via cudarc)
├── stream.rs        # CUDA Stream Management (via cudarc)
│
├── nvdec/
│   ├── mod.rs       # NVDEC Decoder Implementation
│   ├── ffi.rs       # Eigene FFI Bindings: nvcuvid.h → Rust
│   ├── session.rs   # Decoder Session Management (RAII)
│   ├── parser.rs    # CUVID Video Parser (Callback-basiert)
│   └── surface.rs   # Decoded Surface → CUDA DeviceBuffer
│
├── nvenc/
│   ├── mod.rs       # NVENC Encoder Implementation
│   ├── ffi.rs       # Eigene FFI Bindings: nvEncodeAPI.h → Rust
│   ├── session.rs   # Encoder Session Management (RAII)
│   └── config.rs    # Encode Presets + Rate Control
│
└── interop.rs       # CUDA → wgpu Transfer (Pinned Memory Bridge)
```

### FFI Binding Beispiel

```rust
// gpu-hal/src/cuda/nvdec/ffi.rs

// Bindgen oder manuell gegen nvcuvid.h
extern "C" {
    pub fn cuvidCreateDecoder(
        decoder: *mut CUvideodecoder,
        params: *const CUVIDDECODECREATEINFO,
    ) -> CUresult;

    pub fn cuvidDecodePicture(
        decoder: CUvideodecoder,
        params: *const CUVIDPICPARAMS,
    ) -> CUresult;

    pub fn cuvidMapVideoFrame64(
        decoder: CUvideodecoder,
        pic_idx: c_int,
        dev_ptr: *mut CUdeviceptr,
        pitch: *mut c_uint,
        params: *mut CUVIDPROCPARAMS,
    ) -> CUresult;

    pub fn cuvidDestroyDecoder(decoder: CUvideodecoder) -> CUresult;
}

/// RAII Wrapper
pub struct NvDecSession {
    handle: CUvideodecoder,
}

impl Drop for NvDecSession {
    fn drop(&mut self) {
        // SAFETY: handle wurde in new() erstellt und ist gültig bis Drop
        unsafe { cuvidDestroyDecoder(self.handle); }
    }
}
```

---

## 6. Crate-Architektur (für Multi-Agent-Entwicklung)

### Übersicht: 12 Crates (3 Parallelisierungs-Gruppen)

```
crates/
├── common/           # [ZUERST] Typen, Traits, Errors, Config          (~1200 LOC)
├── gpu-hal/          # [ZUERST] GPU Abstraction + CUDA/Vulkan Backends  (~4000 LOC)
├── demux/            # [PARALLEL A] Eigener Container Parser            (~2500 LOC)
├── decoder/          # [PARALLEL A] HW Decode Management                (~1500 LOC)
├── timeline-eval/    # [PARALLEL B] Timeline @ Time T → Layers          (~1500 LOC)
├── compositor/       # [PARALLEL B] GPU Compositing                     (~2000 LOC)
├── effects/          # [PARALLEL C] 30+ GPU Effects (CUDA + Vulkan)     (~5000 LOC)
├── encoder/          # [PARALLEL D] HW Encode + Muxing                  (~2000 LOC)
├── audio/            # [PARALLEL E] Audio Decode + Mix + Output          (~2500 LOC)
├── project/          # [PARALLEL F] Project Save/Load (Web-kompatibel)   (~1200 LOC)
├── mux/              # [PARALLEL D] Container Muxing (Muxide Wrapper)    (~800 LOC)
└── app-state/        # [NACH B] App State + Undo/Redo                    (~1500 LOC)
                                                              Total: ~25.700 LOC
```

### App (native-ui/)
```
native-ui/
├── Cargo.toml
├── build.rs            # PTX + SPIR-V Kernel Compilation
├── src/
│   ├── main.rs         # Entry Point (bestehend, erweitern)
│   ├── app.rs          # App Container (bestehend, erweitern)
│   ├── theme.rs        # Dark Theme (bestehend, 1:1)
│   ├── toolbar.rs      # Menu Bar (bestehend, erweitern)
│   ├── media_panel.rs  # Media Browser (bestehend, erweitern)
│   ├── preview_panel.rs # Preview + GPU Display (bestehend, erweitern)
│   ├── properties_panel.rs # Properties (bestehend, erweitern)
│   ├── timeline.rs     # Timeline Editor (bestehend, erweitern)
│   ├── engine.rs       # Engine Orchestrator (NEU)
│   ├── state.rs        # App State Bridge (NEU)
│   └── bridge.rs       # GPU→wgpu Texture Bridge (NEU)
├── kernels/
│   ├── cuda/           # .cu Dateien
│   └── vulkan/         # .comp Dateien
└── shaders/            # WGSL (nur für egui-Custom-Rendering)
    └── preview_output.wgsl
```

### Dependency Graph (zeigt Parallelisierungsmöglichkeiten)

```
                    common
                      ↑
                   gpu-hal
                      ↑
       ┌──────────────┼──────────────────────────────┐
       │              │              │                │
     demux      timeline-eval    effects           audio
       ↑              ↑              ↑                ↑
    decoder      compositor          │                │
       ↑              ↑              │                │
       │         app-state           │                │
       │              ↑              │                │
    encoder           │              │                │
       ↑              │              │                │
      mux             │              │                │
       ↑              │              │                │
       └──────────────┴──────────────┴────────────────┘
                      ↑
              native-ui (App)

    ════════════════════════════════════════════════════
    Parallelisierungsgruppen (nach common + gpu-hal):

    Agent A: demux → decoder → encoder → mux
    Agent B: timeline-eval → compositor → app-state
    Agent C: effects (alle 30+, CUDA + Vulkan)
    Agent D: audio
    Agent E: project
    Agent F: native-ui (UI Integration)
```

---

### 6.1 `crates/common/` — Shared Foundation (~1200 LOC)

**Muss ZUERST fertig sein — definiert alle Interfaces.**

```
common/src/
├── lib.rs           # Re-exports
├── types.rs         # FrameNumber, TimeCode, Resolution, Rational (Newtypes!)
├── color.rs         # ColorSpace, PixelFormat, TransferFunction
├── layer.rs         # LayerDesc: Source, Transform, Opacity, Effects, Mask
├── effect.rs        # EffectId, EffectInstance, ParamDef, ParamValue
├── blend.rs         # BlendMode Enum (Normal, Multiply, Screen, etc.)
├── codec.rs         # VideoCodec, AudioCodec, ContainerFormat Enums
├── error.rs         # Zentrales Error Enum (thiserror)
├── config.rs        # EngineConfig, GpuPreference
├── project.rs       # ProjectFile, ClipRef, TrackRef — Web-App-kompatible Typen
└── kernel.rs        # KernelId, KernelArgs — GPU Kernel Abstraktion
```

```rust
// Newtype Pattern für Typensicherheit
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FrameNumber(pub u64);

#[derive(Copy, Clone, Debug)]
pub struct TimeCode(pub f64);  // Sekunden

#[derive(Copy, Clone, Debug)]
pub struct Resolution { pub width: u32, pub height: u32 }

// GPU Backend Preference
pub enum GpuPreference {
    Auto,           // CUDA wenn verfügbar, sonst Vulkan
    ForceCuda,
    ForceVulkan,
}

// LayerDesc — Interface zwischen timeline-eval und compositor
pub struct LayerDesc {
    pub source_id: String,
    pub transform: Transform2D,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub effects: Vec<EffectInstance>,
    pub mask: Option<MaskDesc>,
}

// Web-App-kompatibles Projektformat
pub struct ProjectFile {
    pub version: String,
    pub timeline: TimelineData,
    pub media: Vec<MediaRef>,
    // Identisch zur Web-App JSON-Struktur
}
```

---

### 6.2 `crates/gpu-hal/` — GPU Hardware Abstraction (~4000 LOC)

**Muss als ZWEITES fertig sein — alle GPU-Crates hängen davon ab.**

```
gpu-hal/src/
├── lib.rs           # Re-exports + Backend Selection
├── traits.rs        # GpuBackend, GpuBuffer, GpuTexture, HwDecoder, HwEncoder Traits
├── types.rs         # GpuFrame, StagingBuffer, KernelId, GpuCapabilities
├── select.rs        # Runtime Backend Selection (CUDA vs Vulkan)
│
├── cuda/
│   ├── mod.rs       # CudaBackend: impl GpuBackend
│   ├── context.rs   # cudarc-basierter CUDA Context
│   ├── memory.rs    # DeviceBuffer, PinnedBuffer, TexturePool (RAII)
│   ├── kernel.rs    # PTX Module laden, Kernel Dispatch
│   ├── stream.rs    # CUDA Stream Management
│   ├── interop.rs   # CUDA → wgpu Transfer (Pinned Memory)
│   ├── diagnostics.rs # GPU Info, VRAM, Temperature
│   │
│   ├── nvdec/
│   │   ├── mod.rs       # NvDecoder: impl HwDecoder
│   │   ├── ffi.rs       # Eigene FFI: nvcuvid.h Bindings
│   │   ├── session.rs   # RAII Decoder Session
│   │   ├── parser.rs    # CUVID Video Parser
│   │   └── surface.rs   # NV12 Surface → CUDA Buffer
│   │
│   └── nvenc/
│       ├── mod.rs       # NvEncoder: impl HwEncoder
│       ├── ffi.rs       # Eigene FFI: nvEncodeAPI.h Bindings
│       ├── session.rs   # RAII Encoder Session
│       └── config.rs    # Presets + Rate Control
│
└── vulkan/
    ├── mod.rs       # VulkanBackend: impl GpuBackend
    ├── context.rs   # ash-basierter Vulkan Context + Device Selection
    ├── memory.rs    # gpu-allocator Buffers + Images
    ├── pipeline.rs  # Compute Pipeline Management
    ├── shader.rs    # SPIR-V Module laden, Descriptor Sets
    ├── queue.rs     # Queue Management (Compute + Transfer)
    ├── interop.rs   # Vulkan → wgpu Transfer (External Memory / Staging)
    ├── diagnostics.rs # GPU Info via VkPhysicalDeviceProperties
    │
    ├── video_decode/
    │   ├── mod.rs       # VulkanDecoder: impl HwDecoder
    │   ├── session.rs   # VK_KHR_video_decode Session
    │   └── h264.rs      # H.264 Decode Profile Setup
    │
    └── video_encode/
        ├── mod.rs       # VulkanEncoder: impl HwEncoder
        ├── session.rs   # VK_KHR_video_encode Session
        └── h264.rs      # H.264 Encode Profile Setup
```

---

### 6.3 `crates/demux/` — Eigener Container Parser (~2500 LOC)

**Agent A — unabhängig nach gpu-hal.**

```
demux/src/
├── lib.rs           # Re-exports
├── traits.rs        # pub trait Demuxer
├── probe.rs         # File Probing: Codecs, Streams, Duration, Metadata
├── packet.rs        # VideoPacket (NAL Units), AudioPacket
├── nal.rs           # NAL Unit Parser: AVCC → Annex-B, SPS/PPS, Slice Headers
├── index.rs         # Keyframe Index für Random Access
│
├── mp4/
│   ├── mod.rs       # Mp4Demuxer: impl Demuxer
│   ├── boxes.rs     # ISO BMFF Box Parser (ftyp, moov, trak, stbl, mdat, etc.)
│   ├── sample.rs    # Sample Table (stts, stss, stsc, stsz, stco/co64)
│   └── avc.rs       # AVCDecoderConfigurationRecord + HEVCDecoderConfigurationRecord
│
└── mkv/
    ├── mod.rs       # MkvDemuxer: impl Demuxer
    ├── ebml.rs      # EBML Variable-Size Integer + Element Parser
    ├── elements.rs  # Matroska Element IDs + Semantik
    └── cluster.rs   # Cluster/SimpleBlock → NAL Extraction
```

---

### 6.4 `crates/decoder/` — HW Decode Management (~1500 LOC)

**Agent A — hängt von demux + gpu-hal ab.**

```
decoder/src/
├── lib.rs           # Re-exports
├── manager.rs       # Decoder Pool: Source → Decoder Mapping
├── seek.rs          # Keyframe-basiertes Seeking (Long-GOP aware)
├── prefetch.rs      # Look-Ahead Ring Buffer (auf GPU Device)
├── thumbnail.rs     # Thumbnail-Generierung (Decode + Scale)
└── colorspace.rs    # NV12/P010 → RGBA GPU Kernel Wrapper (via gpu-hal)
```

---

### 6.5 `crates/timeline-eval/` — Timeline Evaluation (~1500 LOC)

**Agent B — unabhängig nach common.**

```
timeline-eval/src/
├── lib.rs           # Re-exports
├── evaluator.rs     # Timeline @ Time T → Vec<LayerDesc>
├── types.rs         # Clip, Track, Marker, Transition (Rust-native)
├── keyframe.rs      # Keyframe Interpolation (Linear, Bezier, Hold)
├── transition.rs    # Transition Evaluation
├── nested.rs        # Nested Composition Evaluation
└── serde.rs         # JSON Serialization (Web-App-kompatibel)
```

---

### 6.6 `crates/compositor/` — GPU Compositing (~2000 LOC)

**Agent B — hängt von gpu-hal + timeline-eval ab.**

```
compositor/src/
├── lib.rs           # Re-exports
├── compositor.rs    # LayerDescs + GpuFrames → fertiges Frame
├── blend.rs         # Blend Kernel Dispatch (per Backend)
├── transform.rs     # Transform Kernel (Position, Scale, Rotation, Bilinear/Bicubic)
├── mask.rs          # Mask Kernel (Rect, Ellipse, Path, Feather)
├── transition.rs    # Transition Kernel (Cross-Dissolve, Wipe)
├── color.rs         # Color Management (sRGB ↔ Linear ↔ Rec.709)
├── pipeline.rs      # Multi-Pass Pipeline (Ping-Pong Buffers)
└── nested.rs        # Nested Composition Rendering
```

---

### 6.7 `crates/effects/` — GPU Effects Engine (~5000 LOC)

**Agent C — größtes Crate, unabhängig nach gpu-hal. Hoch parallelisierbar (jeder Effekt ist eigenständig).**

```
effects/src/
├── lib.rs           # Re-exports + Effect Registry
├── registry.rs      # Alle Effekte registrieren, by-name lookup
├── traits.rs        # pub trait Effect: apply(backend, input, output, params)
├── params.rs        # Effekt-Parameter Typen (Float, Int, Color, Enum)
├── uniforms.rs      # Parameter → GPU Constant Memory / Push Constants
│
├── color/
│   ├── mod.rs
│   ├── brightness.rs    # Brightness/Contrast
│   ├── hsl.rs           # Hue/Saturation/Lightness
│   ├── curves.rs        # RGB Curves
│   ├── lut.rs           # 3D LUT Apply
│   ├── white_balance.rs
│   ├── exposure.rs
│   └── color_wheels.rs  # Lift/Gamma/Gain
│
├── blur/
│   ├── mod.rs
│   ├── gaussian.rs      # Separable Gaussian Blur
│   ├── directional.rs   # Directional/Motion Blur
│   ├── radial.rs        # Radial Blur
│   └── zoom.rs          # Zoom Blur
│
├── distort/
│   ├── mod.rs
│   ├── lens.rs          # Lens Distortion
│   ├── warp.rs          # Warp/Displacement
│   └── turbulence.rs    # Turbulence/Noise Displacement
│
├── keying/
│   ├── mod.rs
│   ├── chroma_key.rs    # Chroma Keying (Green/Blue Screen)
│   └── luma_key.rs      # Luma Keying
│
└── stylize/
    ├── mod.rs
    ├── glow.rs          # Glow/Bloom
    ├── sharpen.rs       # Unsharp Mask
    ├── noise.rs         # Film Grain
    └── vignette.rs      # Vignette
```

```rust
/// Effekt-Trait — Backend-agnostisch
pub trait Effect: Send + Sync {
    fn name(&self) -> &str;
    fn category(&self) -> EffectCategory;
    fn params(&self) -> &[ParamDef];

    /// Kernel IDs für diesen Effekt (je nach Backend CUDA oder Vulkan)
    fn kernel_ids(&self) -> &[KernelId];

    /// Effekt anwenden — nutzt gpu-hal für Backend-agnostischen Dispatch
    fn apply(
        &self,
        backend: &dyn GpuBackend,
        input: &GpuFrame,
        output: &mut GpuFrame,
        params: &ParamValues,
        stream: &dyn GpuStream,
    ) -> Result<()>;
}
```

Neuen Effekt hinzufügen = **1 .rs Datei + 1 .cu Kernel + 1 .comp Shader + 1 Zeile in registry.rs**.

---

### 6.8 `crates/encoder/` — HW Encode (~2000 LOC)

**Agent D — hängt von gpu-hal ab.**

```
encoder/src/
├── lib.rs           # Re-exports
├── pipeline.rs      # Export Pipeline: Timeline → Encode → Mux
├── presets.rs       # Export Presets (YouTube 1080p, YouTube 4K, Master, etc.)
├── progress.rs      # Progress Tracking + Cancellation
└── config.rs        # EncoderConfig Builder
```

---

### 6.9 `crates/mux/` — Container Muxing (~800 LOC)

**Agent D — Muxide Wrapper.**

```
mux/src/
├── lib.rs           # Re-exports
├── mp4.rs           # MP4 Muxing via Muxide
├── traits.rs        # pub trait Muxer
└── interleave.rs    # Audio/Video Interleaving
```

---

### 6.10 `crates/audio/` — Audio Processing (~2500 LOC)

**Agent E — komplett unabhängig nach common.**

```
audio/src/
├── lib.rs           # Re-exports
├── decoder.rs       # Symphonia Audio Decode (AAC, MP3, FLAC, WAV, Opus)
├── mixer.rs         # Multi-Track Audio Mixing
├── output.rs        # CPAL Realtime Output
├── sync.rs          # Audio-Video Synchronisation (Audio = Master Clock)
├── timestretch.rs   # Time Stretching (Rubberband oder eigene Implementierung)
├── waveform.rs      # Waveform-Daten für Timeline UI
├── meter.rs         # LUFS, Peak, RMS Metering
└── resampler.rs     # Sample Rate Conversion
```

---

### 6.11 `crates/project/` — Project Management (~1200 LOC)

**Agent F — hängt nur von common ab.**

```
project/src/
├── lib.rs           # Re-exports
├── save.rs          # Project → JSON Serialization (Web-App-kompatibel)
├── load.rs          # JSON → Project Deserialization
├── migrate.rs       # Version Migration (Web-Format ↔ Native-Format)
├── types.rs         # ProjectFile, MediaReference, etc.
├── recent.rs        # Recent Projects Liste
└── autosave.rs      # Periodische Auto-Save
```

### Web-App Kompatibilität

```rust
/// Liest Web-App Projektdateien direkt
pub fn load_web_project(path: &Path) -> Result<ProjectFile> {
    let json = std::fs::read_to_string(path)?;
    let web_format: serde_json::Value = serde_json::from_str(&json)?;

    // Gleiche Struktur wie web-app/src/services/project/
    // clipKeyframes: Record<string, Keyframe[]>
    // tracks, clips, markers, layers
    // media: files, compositions, folders

    serde_json::from_value(web_format).context("failed to parse project file")
}
```

---

### 6.12 `crates/app-state/` — App State + Undo/Redo (~1500 LOC)

**Hängt von timeline-eval ab.**

```
app-state/src/
├── lib.rs           # Re-exports
├── state.rs         # AppState: Timeline + Media + UI + Engine Status
├── history.rs       # Snapshot-basiertes Undo/Redo (wie historyStore.ts)
├── snapshot.rs      # AppSnapshot: serialisierbarer State-Snapshot
├── selection.rs     # Clip/Track Selection State
└── playback.rs      # Playback State + Scrub Control
```

---

## 7. Threading-Modell

```
┌─────────────────────────────────────────────────────────┐
│ Thread 1: Main Thread (egui Event Loop)                  │
│   - egui Rendering (eframe → wgpu)                      │
│   - Input Handling                                       │
│   - NICHT blockieren!                                    │
│                                                          │
│ Thread 2: Render Thread (dediziert)                       │
│   - GPU Context Owner (Primary)                          │
│   - Timeline Evaluation                                  │
│   - Compositing + Effects                                │
│   - GPU → Staging Buffer → Signal an Main Thread         │
│                                                          │
│ Thread 3..N: Decode Pool (1 Thread pro aktive Datei)     │
│   - HW Decode (eigener GPU Stream/Queue pro Decoder)     │
│   - Prefetch Ring Buffer füllen                          │
│   - Thumbnail-Generierung                                │
│                                                          │
│ Thread N+1: Audio Thread (Realtime-Priority)             │
│   - CPAL Output Callback                                 │
│   - Audio Mixing + Metering                              │
│   - Master Clock für A/V Sync                            │
│                                                          │
│ Thread N+2: Export Thread (on-demand)                    │
│   - Eigene Render-Loop + HW Encode                       │
│   - Unabhängig von Preview                               │
└─────────────────────────────────────────────────────────┘
```

### Thread-Kommunikation (crossbeam Channels)

```rust
// Main → Render: "Rendere Frame bei Time T"
type RenderRequest = crossbeam::Sender<RenderCommand>;

// Render → Main: "Frame fertig, Textur aktualisiert"
type FrameReady = crossbeam::Sender<FrameComplete>;

// Decoder → Render: "Decoded Frame bereit" (GPU Device Ptr)
type DecodedFrame = crossbeam::Sender<(SourceId, GpuFrame)>;

// Main → Export: "Starte/Stoppe Export"
type ExportControl = crossbeam::Sender<ExportCommand>;
```

---

## 8. GPU → egui Display Pipeline

### Preview Rendering (Backend-agnostisch)

```rust
// bridge.rs — GPU Frame → egui Texture

pub struct PreviewBridge {
    staging: StagingBuffer,            // Backend-spezifischer Staging Buffer
    wgpu_texture: wgpu::Texture,      // egui's wgpu Instance
    egui_texture_id: egui::TextureId,  // egui Handle
}

impl PreviewBridge {
    /// Backend-agnostisch: GPU Frame → wgpu Texture
    pub fn update(
        &mut self,
        backend: &dyn GpuBackend,
        gpu_frame: &GpuFrame,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        // 1. GPU → Staging Buffer (~0.3-0.6ms für 1080p)
        backend.gpu_to_wgpu(&gpu_frame.texture, &self.wgpu_texture, queue)?;

        // 2. egui zeigt Texture via egui::Image
        Ok(())
    }
}
```

---

## 9. Abhängigkeiten

### Cargo Workspace

```toml
# /Cargo.toml
[workspace]
resolver = "2"
members = [
    "native-ui",
    "crates/*",
]

[workspace.dependencies]
# GPU — CUDA
cudarc = "0.16"

# GPU — Vulkan
ash = "0.38"
gpu-allocator = "0.27"

# Container (nur Audio)
symphonia = { version = "0.5", features = ["mp3", "aac", "flac", "wav", "ogg", "isomp4"] }

# Muxing
muxide = "0.1"

# UI
eframe = "0.31"
egui = "0.31"
egui_extras = "0.31"

# Audio Output
cpal = "0.15"

# Threading
crossbeam = "0.8"
rayon = "1.10"
parking_lot = "0.12"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Error Handling
thiserror = "2"
anyhow = "1"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Testing
criterion = "0.5"
proptest = "1.4"

# Build
glob = "0.3"
shaderc = "0.8"  # GLSL → SPIR-V Compilation

# Byte Parsing (für Demuxer)
byteorder = "1"
```

### Nicht unterstützt (bewusst):
- **ProRes** — Kein Rust-Decoder, kein NVDEC/Vulkan-Support. Konvertierung vorher nötig.
- **DNxHR** — Ditto.
- **FFmpeg** — Kein FFmpeg. Niemals.

### Feature Flags

```toml
# gpu-hal/Cargo.toml
[features]
default = ["cuda", "vulkan"]
cuda = ["cudarc"]          # NVIDIA CUDA Backend
vulkan = ["ash", "gpu-allocator"]  # Vulkan Compute Backend
```

---

## 10. Migrationsplan (Phasen + Agent-Zuordnung)

### Phase 0: Dual-Backend Proof-of-Concept (3-4 Wochen) — GATE

**Ziel:** NVDEC Decode → CUDA Frame → egui Preview + Vulkan Compute Shader → egui Preview.

**Sequenziell (1 Agent):**
- [ ] `crates/common/` — Alle Typen + Traits definieren
- [ ] `crates/gpu-hal/` — GpuBackend Trait + CUDA impl (Basis)
- [ ] Eigene NVDEC FFI Bindings (nvcuvid.h → Rust)
- [ ] Eigener MP4 Parser (Minimal: ftyp, moov, trak, stbl, mdat)
- [ ] NAL Unit Extraction (AVCC → Annex-B)
- [ ] NVDEC decodiert H.264 → CUDA Surface (NV12)
- [ ] CUDA Kernel: NV12 → RGBA
- [ ] CUDA → Pinned Memory → wgpu Texture → egui Image
- [ ] In bestehendem native-ui Preview-Panel anzeigen

**Parallel dazu (2. Agent):**
- [ ] `crates/gpu-hal/vulkan/` — VulkanBackend Basis-Setup
- [ ] Vulkan Compute Pipeline: einfacher Passthrough-Shader
- [ ] Vulkan → Staging → wgpu Texture → egui

**Ergebnis:** Video wird via CUDA UND Vulkan im egui Preview angezeigt. Dual-Backend validiert.

---

### Phase 1: Foundation Complete (4-6 Wochen)

**6 Agents parallel:**

| Agent | Crate(s) | Aufgabe |
|-------|----------|---------|
| **A** | `demux/` | Vollständiger MP4 + MKV Parser, NAL Extraction, Seeking |
| **B** | `decoder/` | Decoder Pool, Prefetch, Thumbnails, Seek Management |
| **C** | `timeline-eval/` | Timeline Evaluation, Keyframe Interpolation, Nested Comps |
| **D** | `audio/` | Symphonia Decode, CPAL Output, Basic Mixer |
| **E** | `project/` | Web-App-kompatibles Save/Load, Recent Projects |
| **F** | `gpu-hal/` erweitern | Vulkan Video Decode, NVENC Bindings, Texture Pool |

---

### Phase 2: Compositing + Effects (6-8 Wochen)

**4 Agents parallel:**

| Agent | Crate(s) | Aufgabe |
|-------|----------|---------|
| **A** | `compositor/` | Blend, Transform, Mask, Transition — CUDA + Vulkan Kernels |
| **B** | `effects/color/` | Brightness, HSL, Curves, LUT, WhiteBalance, Exposure, ColorWheels |
| **C** | `effects/blur/` + `effects/distort/` | Gaussian, Directional, Radial, Zoom, Lens, Warp, Turbulence |
| **D** | `effects/keying/` + `effects/stylize/` | Chroma Key, Luma Key, Glow, Sharpen, Noise, Vignette |

Jeder Effect-Agent schreibt `.cu` UND `.comp` für jeden Effekt.

---

### Phase 3: Export + Audio Complete (4-6 Wochen)

**3 Agents parallel:**

| Agent | Crate(s) | Aufgabe |
|-------|----------|---------|
| **A** | `encoder/` + `mux/` | NVENC/Vulkan Encode + Muxide MP4 Output + Presets |
| **B** | `audio/` erweitern | Multi-Track Mixing, A/V Sync, Time Stretch, Waveform, Metering |
| **C** | `app-state/` | App State, Undo/Redo, Selection, Playback Control |

---

### Phase 4: UI Integration + Polish (4-6 Wochen)

**3 Agents parallel:**

| Agent | Crate(s) | Aufgabe |
|-------|----------|---------|
| **A** | `native-ui/` | Engine Orchestrator, alle Panels mit echten Daten verbinden |
| **B** | `native-ui/` | Timeline UI: Drag&Drop, Clip Editing, Keyframe Curves |
| **C** | `native-ui/` | Media Import, Export Dialog, Keyboard Shortcuts |

---

### Phase 5 (Post-v1): AI Engine

- [ ] `crates/ai/` — ONNX Runtime + CUDA/Vulkan EP
- [ ] Super Resolution, Background Removal, Frame Interpolation
- [ ] Model Download + Cache

---

## 11. Rust Best Practices (Projekt-spezifisch)

### CRITICAL: Error Handling

```rust
// Library Crates: thiserror für präzise Fehlertypen
#[derive(Error, Debug)]
pub enum DecoderError {
    #[error("HW decoder init failed: {0}")]
    HwDecoderInit(String),

    #[error("unsupported codec: {codec}")]
    UnsupportedCodec { codec: String },

    #[error("decode failed at frame {frame}: {source}")]
    DecodeFailed { frame: u64, #[source] source: Box<dyn std::error::Error + Send + Sync> },
}

// App Layer: anyhow für Kontext
fn open_video(path: &Path) -> anyhow::Result<()> {
    let demux = Mp4Demuxer::open(path)
        .context("failed to open container")?;
    let decoder = backend.create_decoder(demux.video_codec(), &config)
        .context("failed to initialize decoder")?;
    Ok(())
}
```

### CRITICAL: RAII für GPU-Ressourcen

```rust
// ✅ RICHTIG: Automatische Cleanup
pub struct NvDecSession {
    handle: CUvideodecoder,
}

impl Drop for NvDecSession {
    fn drop(&mut self) {
        // SAFETY: handle ist gültig, wurde im Konstruktor erstellt
        unsafe { cuvidDestroyDecoder(self.handle); }
    }
}
```

### HIGH: Backend-agnostischer Code

```rust
// ✅ RICHTIG: Compositor weiß nicht ob CUDA oder Vulkan
pub struct Compositor<B: GpuBackend> {
    backend: Arc<B>,
    temp_buffers: Vec<B::Buffer>,
}

impl<B: GpuBackend> Compositor<B> {
    pub fn render(
        &mut self,
        layers: &[LayerDesc],
        frames: &HashMap<String, GpuFrame>,
        output: &mut GpuFrame,
        stream: &B::Stream,
    ) -> Result<()> {
        for layer in layers {
            self.backend.dispatch_kernel(
                &KernelId::AlphaBlend,
                grid, block, &args, stream,
            )?;
        }
        Ok(())
    }
}
```

### HIGH: Zero-Copy GPU Pipeline

```rust
// ❌ FALSCH: Unnötiger CPU Roundtrip
let gpu_frame = decoder.decode(packet)?;
let cpu_data = backend.copy_to_host(&gpu_frame)?;     // GPU → CPU
let gpu_result = backend.copy_to_device(&cpu_data)?;   // CPU → GPU

// ✅ RICHTIG: Alles auf GPU
let gpu_frame = decoder.decode(packet)?;
compositor.blend(&gpu_frame, &gpu_output, stream)?;     // GPU → GPU
effects.apply(&gpu_output, stream)?;                     // GPU → GPU
// Nur 1× am Ende: GPU → Staging → wgpu für Display
```

### MEDIUM: Newtype Pattern

```rust
// ✅ RICHTIG: Compiler verhindert Verwechslung
fn seek(frame: FrameNumber) { ... }
fn set_time(time: TimeCode) { ... }

// seek(TimeCode(5.0));  // COMPILE ERROR!
seek(FrameNumber(150));   // OK
```

### MEDIUM: Channel-basierte Thread-Kommunikation

```rust
// ✅ RICHTIG: Channels statt Shared State
let (tx, rx) = crossbeam::channel::bounded::<RenderCommand>(4);

thread::spawn(move || {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            RenderCommand::RenderFrame(time) => { ... }
            RenderCommand::Shutdown => break,
        }
    }
});
```

---

## 12. Build-Voraussetzungen

### Minimum (Vulkan-only, alle Plattformen)

| Tool | Version | Zweck |
|------|---------|-------|
| **Rust** | stable 1.80+ | Compiler |
| **Vulkan SDK** | 1.3+ | Vulkan Runtime + glslc |
| **GPU** | Vulkan 1.3 fähig | AMD, Intel, NVIDIA |

### CUDA Backend (optional, NVIDIA-only)

| Tool | Version | Zweck |
|------|---------|-------|
| **CUDA Toolkit** | 12.2+ | nvcc, CUDA Runtime |
| **NVIDIA Driver** | 535+ | NVDEC/NVENC Support |
| **NVIDIA Video Codec SDK** | 12.1+ | nvcuvid.h + nvEncodeAPI.h Headers |
| **NVIDIA GPU** | Turing+ (GTX 16xx/RTX 20xx+) | Hardware |

### macOS (via MoltenVK)

| Tool | Version | Zweck |
|------|---------|-------|
| **MoltenVK** | 1.2+ | Vulkan → Metal Translation |
| **macOS** | 13+ | Metal 3 Support |

### Environment Variables

```bash
# Windows (CUDA)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set NVIDIA_VIDEO_CODEC_SDK_PATH=C:\VideoCodecSDK
set VULKAN_SDK=C:\VulkanSDK\1.3.xxx

# Linux
export CUDA_PATH=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export VULKAN_SDK=/usr/share/vulkan

# macOS
export VULKAN_SDK=$HOME/VulkanSDK/1.3.xxx/macOS
```

---

## 13. Risiken

| # | Risiko | Impact | Mitigation |
|---|--------|--------|------------|
| 1 | **Eigener Demuxer** — MP4 Box Parsing + MKV EBML sind komplex, viele Edge Cases | Hoch | Gegen FFmpeg-Output testen, nur H.264/H.265/VP9/AV1 unterstützen |
| 2 | **Eigene NVDEC FFI** — CUVID API ist Callback-basiert, unsafe-lastig | Hoch | RAII + umfangreiche Tests, compute-sanitizer |
| 3 | **Vulkan Video Decode** — VK_KHR_video_decode ist relativ neu, Driver-Support variiert | Hoch | Vulkan Compute als Minimum, Video Decode als Optional |
| 4 | **Dual-Kernel Maintenance** — 30+ Effekte × 2 Sprachen (CUDA + GLSL) | Hoch | Einheitliches Params-Schema, automatisierte Vergleichstests (CUDA vs Vulkan Output) |
| 5 | **Muxide Maturity** — v0.1, möglicherweise Edge Cases bei B-Frames / VFR | Mittel | MP4-Output extensiv testen, bei Bedarf eigenen Muxer schreiben |
| 6 | **egui Timeline Komplexität** — Drag&Drop, Bezier Keyframes, Multi-Select in Immediate Mode | Mittel | egui::Painter für Custom Drawing, keine Standard-Widgets für Timeline |
| 7 | **ProRes/DNxHR** — Nicht unterstützt, professionelle Workflows betroffen | Mittel | Dokumentation: vorher konvertieren |
| 8 | **MoltenVK auf macOS** — Vulkan → Metal Translation kann Overhead haben | Niedrig | macOS ist Sekundärplattform, Performance akzeptabel |
| 9 | **cudarc API Changes** — v0.16 ist aktiv in Entwicklung | Niedrig | Version pinnen, Interop-Code isolieren |

---

## 14. LOC-Verteilung

```
Rust Engine (crates/):
  common .............. ~1.200 LOC
  gpu-hal ............. ~4.000 LOC  (CUDA + Vulkan Backends + FFI)
  demux ............... ~2.500 LOC  (Eigener MP4 + MKV Parser)
  decoder ............. ~1.500 LOC
  timeline-eval ....... ~1.500 LOC
  compositor .......... ~2.000 LOC
  effects ............. ~5.000 LOC  (30+ Effekte, Backend-agnostisch)
  encoder ............. ~2.000 LOC
  mux ................. ~800 LOC
  audio ............... ~2.500 LOC
  project ............. ~1.200 LOC
  app-state ........... ~1.500 LOC
  ─────────────────────────────
  Subtotal:           ~25.700 LOC

App (native-ui/):
  Neue Module ......... ~1.500 LOC  (engine, state, bridge)
  UI Erweiterungen .... ~2.500 LOC  (bestehende Panels mit echten Daten)
  ─────────────────────────────
  Subtotal:           ~4.000 LOC

GPU Kernels:
  CUDA (.cu) .......... ~3.200 LOC
  Vulkan (.comp) ...... ~3.200 LOC  (funktional identisch zu CUDA)
  ─────────────────────────────
  Subtotal:           ~6.400 LOC

                        ═══════════
TOTAL:                  ~36.100 LOC
```

**Vergleich zum alten Plan:** +8.400 LOC durch Vulkan Backend, eigenen Demuxer, eigene FFI Bindings, Dual-Kernels.

---

## 15. Code Quality Toolchain

```toml
# rustfmt.toml
edition = "2021"
max_width = 100
tab_spaces = 4
use_field_init_shorthand = true
```

```toml
# clippy.toml
cognitive-complexity-threshold = 30
```

```toml
# rust-toolchain.toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy", "rust-src"]
```

> **Hinweis:** Kein nightly mehr nötig. cudarc funktioniert mit stable Rust. Nur für optionale Features wie `#[bench]` wird nightly gebraucht.

### CI Pipeline

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo deny check
cargo build --release --features cuda
cargo build --release --features vulkan
```

---

## 16. Agent-Arbeitsregeln

### Für AI-Agents die an diesem Projekt arbeiten:

1. **Trait-First:** Implementiere gegen das Trait in `common/`, nicht gegen konkrete Typen
2. **Keine Cross-Crate-Änderungen:** Wenn du `common/` ändern musst, stimme dich vorher ab
3. **Tests zuerst:** Jedes Modul braucht `#[cfg(test)] mod tests` bevor es als fertig gilt
4. **Feature-gated GPU Code:** CUDA-Code hinter `#[cfg(feature = "cuda")]`, Vulkan hinter `#[cfg(feature = "vulkan")]`
5. **Dual-Kernel-Regel:** Kein Effekt ist fertig ohne `.cu` UND `.comp` Variante
6. **Kein FFmpeg:** Keine Abhängigkeit auf ffmpeg-sys, ffmpeg-next, oder ähnliche Crates. Niemals.
7. **SAFETY-Kommentare:** Jeder `unsafe` Block braucht `// SAFETY:` Erklärung
8. **Error Propagation:** `thiserror` in Crates, `anyhow` nur in `native-ui/`
9. **Commit-Regel:** `cargo fmt && cargo clippy && cargo test` vor jedem Commit
