# WebCodec ProRes Codec

A comprehensive technical specification for building a browser-native ProRes **encoder and decoder** in Rust (WASM) with **WebGPU acceleration** for near-native performance.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Legal Considerations](#legal-considerations)
3. [ProRes Technical Specification](#prores-technical-specification)
4. [Architecture Overview](#architecture-overview)
5. [Hybrid WASM + WebGPU Pipeline](#hybrid-wasm--webgpu-pipeline)
6. [Module Specifications](#module-specifications)
7. [WebGPU Compute Shaders](#webgpu-compute-shaders)
8. [WASM Integration](#wasm-integration)
9. [Performance Optimization](#performance-optimization)
10. [Testing & Conformance](#testing--conformance)
11. [Project Structure](#project-structure)
12. [Implementation Roadmap](#implementation-roadmap)
13. [References](#references)

---

## Executive Summary

### What is This?

A proposal to build the **first open-source, browser-native ProRes codec (encoder + decoder)** using a hybrid **Rust/WASM + WebGPU** architecture. This enables:
- **Professional video export** directly from web applications
- **Real-time timeline scrubbing** of ProRes files in the browser
- **GPU-accelerated performance** approaching native speeds

### Why Hybrid WASM + WebGPU?

The WebCodecs API doesn't support custom codecs—you can't extend `VideoDecoder` with ProRes. But you **can** build a custom codec that:
1. Uses the same `VideoFrame` interface as WebCodecs
2. Leverages WebGPU for GPU-accelerated parallel operations
3. Integrates seamlessly with existing WebCodecs pipelines

```
┌─────────────────────────────────────────────────────────────┐
│  PERFORMANCE COMPARISON                                     │
├─────────────────────────────────────────────────────────────┤
│  Native WebCodecs (H.264)     │  ~0.5ms/frame  │ GPU HW    │
│  FFmpeg WASM                  │  ~50ms/frame   │ CPU only  │
│  Our Hybrid WASM+WebGPU       │  ~3-5ms/frame  │ GPU ✓     │
└─────────────────────────────────────────────────────────────┘
```

### Why ProRes?

| Reason | Impact |
|--------|--------|
| Industry standard | Used by Final Cut Pro, DaVinci Resolve, Premiere Pro |
| No browser support | Zero web-based tools can import/export ProRes today |
| Professional workflows | Editors need ProRes for color grading, compositing |
| Quality preservation | Visually lossless at high bitrates |
| Timeline scrubbing | Fast random-access for editing workflows |

### Current Alternatives

| Method | Problem |
|--------|---------|
| Export H.264 → Convert locally | Extra step, quality loss from re-encode |
| Server-side FFmpeg | Requires upload, server costs, latency |
| WebCodecs H.264/VP9 | Not accepted in professional workflows |
| FFmpeg WASM | Too slow for timeline scrubbing (~5-20x slower) |

### Project Scope

- **Target**: ProRes 422 HQ (profile 3) - most common professional format
- **Codec**: Both encoder AND decoder (for editing workflows)
- **Architecture**: Hybrid Rust/WASM + WebGPU compute shaders
- **Language**: Rust → WebAssembly (serial ops) + WGSL (parallel ops)
- **Output**: MOV container with ProRes video track
- **Estimated size**: ~5000-6000 lines of Rust + ~500 lines WGSL
- **Performance target**: 200+ fps decode at 4K (real-time scrubbing)

---

## Legal Considerations

### Is This Legal?

**Yes.** Building a clean-room ProRes implementation is legal.

### Supporting Evidence

1. **Apple Published the Specification**
   - [Apple ProRes White Paper](https://www.apple.com/final-cut-pro/docs/Apple_ProRes.pdf) (public, free)
   - [SMPTE RDD 36:2022](https://pub.smpte.org/doc/rdd36/20220909-pub/rdd36-2022.pdf) - Official bitstream syntax

2. **Precedent Exists**
   - FFmpeg's `prores_ks` encoder: Open source since ~2012, no legal action
   - Blackmagic DaVinci Resolve: Has ProRes encoding
   - Adobe Premiere: Licensed ProRes support
   - Many hardware encoders (Atomos, etc.)

3. **Apple's Actual Position**
   - Apple *wants* ProRes adoption (why they published specs)
   - They care about quality certification, not blocking implementations
   - Their warning in the White Paper is about quality, not legality

### What to Avoid

| Do | Don't |
|----|-------|
| Clean-room implementation from spec | Copy Apple's code |
| Call it "ProRes-compatible" | Claim Apple certification |
| Open source (builds trust) | Use Apple's trademarks inappropriately |

### Apple's White Paper Warning (Context)

> "Using any unauthorized implementation (like the FFmpeg and derivative implementations) may lead to decoding errors, performance degradation, incompatibility, and instability."

This is a **quality warning**, not a legal threat. Apple wants users to know third-party encoders aren't tested by Apple. FFmpeg ProRes files work fine in practice.

---

## ProRes Technical Specification

### Codec Family Overview

| Profile | FourCC | Chroma | Bit Depth | Target Bitrate (1080p30) | Use Case |
|---------|--------|--------|-----------|--------------------------|----------|
| Proxy | `apco` | 4:2:2 | 10-bit | ~45 Mbps | Offline editing |
| LT | `apcs` | 4:2:2 | 10-bit | ~102 Mbps | Light workflows |
| Standard | `apcn` | 4:2:2 | 10-bit | ~147 Mbps | Standard quality |
| **HQ** | `apch` | 4:2:2 | 10-bit | **~220 Mbps** | **Recommended** |
| 4444 | `ap4h` | 4:4:4:4 | 10-bit | ~330 Mbps | With alpha |
| 4444 XQ | `ap4x` | 4:4:4:4 | 12-bit | ~500 Mbps | Highest quality |

**Initial Target**: ProRes 422 HQ (`apch`) - best balance of quality and compatibility.

### Bitstream Structure

```
ProRes Frame
├── Frame Container Atom (4 bytes)
│   └── 'icpf' magic (0x69637066)
├── Frame Header
│   ├── Header size (2 bytes)
│   ├── Reserved (2 bytes)
│   ├── Bitstream version (1 byte) = 0
│   ├── Encoder identifier (4 bytes)
│   ├── Frame width (2 bytes)
│   ├── Frame height (2 bytes)
│   ├── Chroma format (2 bits)
│   │   └── 2 = 4:2:2, 3 = 4:4:4
│   ├── Interlace mode (2 bits)
│   │   └── 0 = progressive, 1 = interlaced TFF, 2 = interlaced BFF
│   ├── Aspect ratio (4 bits)
│   ├── Frame rate code (4 bits)
│   ├── Color primaries (8 bits)
│   │   └── 1 = BT.709, 6 = BT.601
│   ├── Transfer function (8 bits)
│   ├── Matrix coefficients (8 bits)
│   ├── Alpha info (4 bits)
│   └── [Optional] Custom quantization matrices
└── Picture Data
    ├── Picture Header
    │   ├── Picture data size (4 bytes)
    │   ├── Slice count (2 bytes)
    │   └── Slice widths log2 (various)
    ├── Slice Index Table
    │   └── [slice_count × 2 bytes] offsets
    └── Slice Data Array
        └── [Encoded macroblocks]
```

### Macroblock Structure (4:2:2)

```
16×16 Macroblock
├── Y (Luma): 4 × 8×8 blocks
│   ┌───┬───┐
│   │ 0 │ 1 │
│   ├───┼───┤
│   │ 2 │ 3 │
│   └───┴───┘
├── Cb (Chroma Blue): 2 × 8×8 blocks (subsampled horizontally)
│   ┌───┬───┐
│   │ 0 │ 1 │
│   └───┴───┘
└── Cr (Chroma Red): 2 × 8×8 blocks (subsampled horizontally)
    ┌───┬───┐
    │ 0 │ 1 │
    └───┴───┘

Total: 8 blocks per macroblock (4Y + 2Cb + 2Cr)
```

### Slice Structure

Slices enable parallel encoding/decoding. Each slice:
- Height: Always 1 macroblock row (16 pixels)
- Width: 1, 2, 4, or 8 macroblocks (configurable)
- Can be encoded/decoded independently

```
Slice Header
├── Slice size (variable)
├── Scale factor (1 byte) - per-slice quantization
└── [Encoded blocks...]

Block Data Order within Slice:
1. All Y DC coefficients
2. All Y AC coefficients
3. All Cb DC coefficients
4. All Cb AC coefficients
5. All Cr DC coefficients
6. All Cr AC coefficients
```

### DCT (Discrete Cosine Transform)

ProRes uses standard 8×8 Type-II DCT:

```
F(u,v) = (C(u) × C(v) / 4) × Σx Σy f(x,y) × cos((2x+1)uπ/16) × cos((2y+1)vπ/16)

where:
  C(0) = 1/√2
  C(n) = 1 for n > 0
  f(x,y) = input sample at position (x,y)
  F(u,v) = DCT coefficient at frequency (u,v)
```

**Scan Order** (Zig-zag):
```
 0  1  5  6 14 15 27 28
 2  4  7 13 16 26 29 42
 3  8 12 17 25 30 41 43
 9 11 18 24 31 40 44 53
10 19 23 32 39 45 52 54
20 22 33 38 46 51 55 60
21 34 37 47 50 56 59 61
35 36 48 49 57 58 62 63
```

### Quantization

#### Default Quantization Matrices

**Luma (Y) Matrix:**
```
 4  7  9 11 13 14 15 63
 7  7 11 12 14 15 63 63
 9 11 13 14 15 63 63 63
11 11 13 14 63 63 63 63
11 13 14 63 63 63 63 63
13 14 63 63 63 63 63 63
13 63 63 63 63 63 63 63
63 63 63 63 63 63 63 63
```

**Chroma (Cb/Cr) Matrix:**
```
 4  7  9 11 13 14 63 63
 7  7 11 12 14 63 63 63
 9 11 13 14 63 63 63 63
11 11 13 14 63 63 63 63
11 13 14 63 63 63 63 63
13 14 63 63 63 63 63 63
63 63 63 63 63 63 63 63
63 63 63 63 63 63 63 63
```

#### Quantization Formula

```
quantized_coeff = round(dct_coeff / (matrix[i] × scale_factor))

where:
  scale_factor = per-slice value (1-224)
  Lower scale_factor = higher quality = larger file
```

#### Scale Factor by Profile (approximate)

| Profile | Typical Scale Factor Range |
|---------|---------------------------|
| Proxy | 64-96 |
| LT | 32-64 |
| Standard | 16-32 |
| HQ | 4-16 |
| 4444 | 2-8 |

### Entropy Coding

ProRes uses a **hybrid Rice/Exponential-Golomb** coding scheme (NOT Huffman).

#### Rice Code Parameters

Each codeword has three parameters packed in control bytes:
- **MP**: Maximum prefix length for Rice codes (typically 9-13)
- **R**: Rice code parameter (suffix bits, typically 0-6)
- **G**: Exp-Golomb parameter for overflow (typically 4-8)

#### Encoding Algorithm

```
function encode_value(value, MP, R, G):
    prefix = value >> R           // High bits as unary
    suffix = value & ((1 << R) - 1)  // Low bits as binary

    if prefix < MP:
        // Standard Rice code
        write_unary(prefix)       // prefix 1-bits + 0 terminator
        write_bits(suffix, R)     // R bits
    else:
        // Exp-Golomb fallback for large values
        write_unary(MP)           // MP 1-bits + 0 terminator
        write_exp_golomb(value - (MP << R), G)
```

#### DC Coefficient Coding

DC coefficients are delta-coded (difference from previous block):
```
dc_delta = current_dc - previous_dc
encode_signed(dc_delta)  // Sign bit appended if non-zero
```

#### AC Coefficient Coding

AC coefficients use run-length encoding:
```
For each non-zero AC coefficient:
    1. Encode run of preceding zeros (Rice coded)
    2. Encode |value| - 1 (Rice coded)
    3. Encode sign bit (1 bit)
```

### Color Space

#### Supported Color Matrices

| Code | Standard | Usage |
|------|----------|-------|
| 1 | BT.709 | HD video (most common) |
| 6 | BT.601 | SD video |
| 9 | BT.2020 | UHD/HDR |

#### RGB to Y'CbCr Conversion (BT.709)

```
Y'  =  0.2126 × R + 0.7152 × G + 0.0722 × B
Cb  = -0.1146 × R - 0.3854 × G + 0.5000 × B + 512
Cr  =  0.5000 × R - 0.4542 × G - 0.0458 × B + 512

Input: RGB [0-255] (8-bit)
Output: Y'CbCr [0-1023] (10-bit)
```

#### 4:2:2 Chroma Subsampling

Horizontal chroma subsampling (Cb and Cr at half horizontal resolution):
```
For every 2 horizontal pixels:
  Y:  [Y0] [Y1]     → 2 samples
  Cb: [Cb0]         → 1 sample (average or co-sited)
  Cr: [Cr0]         → 1 sample (average or co-sited)
```

---

## Architecture Overview

### The Key Insight: CPU/GPU Split

ProRes encoding/decoding has both **serial** and **parallel** operations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PRORES PIPELINE - CPU vs GPU SPLIT                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SERIAL (must be CPU/WASM):              PARALLEL (can be GPU/WebGPU): │
│  ─────────────────────────               ───────────────────────────── │
│  • Bitstream parsing                     • Dequantization              │
│  • Entropy decode (Rice/VLC)             • Inverse DCT (8x8 blocks)    │
│  • Run-length decode                     • Forward DCT                 │
│  • Entropy encode                        • Color space conversion      │
│  • Container muxing                      • Quantization                │
│                                                                         │
│  ~10-20% of compute time                 ~80-90% of compute time       │
│  Unavoidable CPU overhead                MASSIVE parallelization ✓     │
└─────────────────────────────────────────────────────────────────────────┘
```

The heavy compute (DCT, color conversion) is **embarrassingly parallel** and runs perfectly on GPU.

### High-Level Hybrid Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   DECODE PIPELINE (for timeline scrubbing)              │
└─────────────────────────────────────────────────────────────────────────┘

  ProRes          ┌──────────────────┐        ┌──────────────────┐
  Bitstream ─────▶│   WASM (CPU)     │───────▶│   WebGPU (GPU)   │
                  │                  │        │                  │
                  │ • Parse header   │        │ • Dequantize     │
                  │ • Entropy decode │  DCT   │ • Inverse DCT    │
                  │ • RLE decode     │ coeffs │ • YCbCr → RGB    │
                  │                  │───────▶│                  │
                  └──────────────────┘        └────────┬─────────┘
                                                       │
                                              GPU Texture / VideoFrame
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  Canvas / WebGL  │
                                              │  (zero-copy)     │
                                              └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   ENCODE PIPELINE (for export)                          │
└─────────────────────────────────────────────────────────────────────────┘

  VideoFrame      ┌──────────────────┐        ┌──────────────────┐
  / Canvas ──────▶│   WebGPU (GPU)   │───────▶│   WASM (CPU)     │
                  │                  │        │                  │
                  │ • RGB → YCbCr    │  DCT   │ • Entropy encode │
                  │ • Forward DCT    │ coeffs │ • RLE encode     │
                  │ • Quantize       │───────▶│ • Pack bitstream │
                  │                  │        │ • MOV muxing     │
                  └──────────────────┘        └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  .mov File Blob  │
                                              └──────────────────┘
```

### Data Flow Detail (Decode)

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│  ProRes   │     │   WASM    │     │   GPU     │     │   GPU     │
│ Bitstream │────▶│  Entropy  │────▶│  Buffer   │────▶│  Compute  │
│           │     │  Decode   │     │           │     │  Shader   │
└───────────┘     └───────────┘     └───────────┘     └─────┬─────┘
                                                            │
   Quantized DCT coefficients (i16)                         │
   transferred to GPU via writeBuffer()                     │
                                                            ▼
                                                   ┌───────────────┐
                       Zero-copy                   │  GPU Texture  │
                       texture binding             │   (RGBA8)     │
                                                   └───────┬───────┘
                                                           │
                                                           ▼
                                                   ┌───────────────┐
                                                   │  VideoFrame   │
                                                   │  (WebCodecs)  │
                                                   └───────────────┘
```

### Module Dependency Graph

```
lib.rs (WASM exports)
    │
    ├── decoder.rs (decode facade) ─────────────────┐
    │       │                                       │
    │       ├── bitreader.rs (bit-level input)      │
    │       ├── entropy_decode.rs (Rice/Golomb)     │  JavaScript
    │       └── slice_decode.rs (coefficient out)   │  WebGPU
    │                                               │  Bridge
    ├── encoder.rs (encode facade)                  │
    │       │                                       │
    │       ├── frame.rs (frame encoding)           │
    │       │       └── slice.rs (slice encoding)   │
    │       │               └── entropy.rs          │
    │       │                       └── bitwriter.rs│
    │       └── color.rs (fallback CPU color)       │
    │                                               │
    ├── mux.rs (MOV container)                      │
    │       └── atoms.rs (QuickTime atoms)          │
    │                                               │
    └── demux.rs (MOV parser) ──────────────────────┘
            └── atoms.rs

webgpu/ (WGSL shaders)
    │
    ├── idct.wgsl          (Inverse DCT - 8x8 blocks)
    ├── dct.wgsl           (Forward DCT - 8x8 blocks)
    ├── dequantize.wgsl    (Apply inverse quantization)
    ├── quantize.wgsl      (Apply quantization)
    └── colorspace.wgsl    (YCbCr ↔ RGB conversion)
```

---

## Hybrid WASM + WebGPU Pipeline

### Why This Approach Works

Native hardware video decoders use the same split we're implementing:

| Component | Native HW Decode | Our Hybrid |
|-----------|------------------|------------|
| Entropy decode | Fixed-function or CPU | WASM+SIMD |
| IDCT | GPU shader | WebGPU compute shader |
| Color convert | GPU shader | WebGPU compute shader |
| Output | GPU texture | GPU texture (same!) |

The **only** overhead vs native is entropy decoding in WASM instead of fixed-function hardware. But that's typically 10-20% of total decode time.

### JavaScript Bridge Layer

The bridge coordinates WASM and WebGPU:

```typescript
// prores-codec.ts - The WebCodecs-compatible interface

export class ProResVideoDecoder {
  private wasmDecoder: WasmProResDecoder;
  private gpuPipeline: WebGPUProResPipeline;
  private outputCallback: (frame: VideoFrame) => void;
  private errorCallback: (error: Error) => void;

  constructor(init: VideoDecoderInit) {
    this.outputCallback = init.output;
    this.errorCallback = init.error;
  }

  async configure(config: VideoDecoderConfig): Promise<void> {
    // Initialize WASM module
    this.wasmDecoder = await initWasmDecoder();

    // Initialize WebGPU pipeline
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    this.gpuPipeline = new WebGPUProResPipeline(device, config);
  }

  decode(chunk: EncodedVideoChunk): void {
    // 1. WASM: Entropy decode → quantized DCT coefficients
    const coefficients = this.wasmDecoder.entropyDecode(
      new Uint8Array(chunk.data)
    );

    // 2. WebGPU: Dequantize + IDCT + Color convert
    const texture = this.gpuPipeline.processFrame(coefficients);

    // 3. Create VideoFrame from GPU texture (zero-copy)
    const frame = new VideoFrame(texture, {
      timestamp: chunk.timestamp,
      duration: chunk.duration,
    });

    this.outputCallback(frame);
  }

  static async isConfigSupported(
    config: VideoDecoderConfig
  ): Promise<VideoDecoderSupport> {
    const supported = config.codec.startsWith('prores') &&
                      'gpu' in navigator;
    return { supported, config };
  }

  close(): void {
    this.wasmDecoder?.free();
    this.gpuPipeline?.destroy();
  }
}
```

### WebGPU Pipeline Class

```typescript
// webgpu-pipeline.ts

export class WebGPUProResPipeline {
  private device: GPUDevice;
  private dequantizePipeline: GPUComputePipeline;
  private idctPipeline: GPUComputePipeline;
  private colorConvertPipeline: GPUComputePipeline;

  private coefficientBuffer: GPUBuffer;
  private intermediateBuffer: GPUBuffer;
  private outputTexture: GPUTexture;

  private width: number;
  private height: number;

  constructor(device: GPUDevice, config: VideoDecoderConfig) {
    this.device = device;
    this.width = config.codedWidth!;
    this.height = config.codedHeight!;

    this.initBuffers();
    this.initPipelines();
  }

  private initBuffers(): void {
    const numBlocks = (this.width / 8) * (this.height / 8);

    // Buffer for quantized DCT coefficients from WASM
    this.coefficientBuffer = this.device.createBuffer({
      size: numBlocks * 64 * 2 * 8, // 8 blocks per macroblock, i16
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Intermediate buffer for dequantized coefficients
    this.intermediateBuffer = this.device.createBuffer({
      size: numBlocks * 64 * 4 * 8, // f32 after dequantize
      usage: GPUBufferUsage.STORAGE,
    });

    // Output texture
    this.outputTexture = this.device.createTexture({
      size: [this.width, this.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING |
             GPUTextureUsage.RENDER_ATTACHMENT |
             GPUTextureUsage.COPY_SRC,
    });
  }

  private initPipelines(): void {
    // Load and compile WGSL shaders
    this.dequantizePipeline = this.createComputePipeline(DEQUANTIZE_WGSL);
    this.idctPipeline = this.createComputePipeline(IDCT_WGSL);
    this.colorConvertPipeline = this.createComputePipeline(COLORSPACE_WGSL);
  }

  processFrame(coefficients: Int16Array): GPUTexture {
    // Upload coefficients from WASM to GPU
    this.device.queue.writeBuffer(
      this.coefficientBuffer,
      0,
      coefficients.buffer
    );

    const commandEncoder = this.device.createCommandEncoder();

    // Pass 1: Dequantize
    this.dispatchCompute(commandEncoder, this.dequantizePipeline);

    // Pass 2: Inverse DCT (8x8 blocks in parallel)
    this.dispatchCompute(commandEncoder, this.idctPipeline);

    // Pass 3: YCbCr → RGB
    this.dispatchCompute(commandEncoder, this.colorConvertPipeline);

    this.device.queue.submit([commandEncoder.finish()]);

    return this.outputTexture;
  }

  private dispatchCompute(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline
  ): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.createBindGroup(pipeline));

    // Dispatch one workgroup per 8x8 block
    const workgroupsX = Math.ceil(this.width / 8);
    const workgroupsY = Math.ceil(this.height / 8);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY);

    pass.end();
  }
}
```

### Memory Transfer Optimization

The critical path is WASM → GPU data transfer:

```typescript
// Efficient coefficient transfer strategies

// Option 1: Direct writeBuffer (simplest)
device.queue.writeBuffer(gpuBuffer, 0, wasmCoefficients.buffer);

// Option 2: Staging buffer with double-buffering
class DoubleBufferedTransfer {
  private buffers: [GPUBuffer, GPUBuffer];
  private currentBuffer = 0;

  upload(data: ArrayBuffer): GPUBuffer {
    const staging = this.buffers[this.currentBuffer];

    // Map, copy, unmap
    new Int16Array(staging.getMappedRange()).set(new Int16Array(data));
    staging.unmap();

    // Swap buffers
    this.currentBuffer = 1 - this.currentBuffer;

    return staging;
  }
}

// Option 3: Shared memory (if SharedArrayBuffer available)
// WASM writes directly to mapped GPU buffer
```

### Integration with Existing WebCodecs Code

Your existing masterselects code can use this transparently:

```typescript
// Unified decoder factory
function createDecoder(codec: string, init: VideoDecoderInit) {
  if (codec.startsWith('prores')) {
    return new ProResVideoDecoder(init);  // Our hybrid implementation
  }
  return new VideoDecoder(init);  // Native WebCodecs
}

// Usage - identical API
const decoder = createDecoder('prores', {
  output: (frame) => {
    // Same VideoFrame interface!
    ctx.drawImage(frame, 0, 0);
    frame.close();
  },
  error: (e) => console.error(e),
});

await decoder.configure({
  codec: 'prores',
  codedWidth: 1920,
  codedHeight: 1080,
});

// Decode frames
decoder.decode(new EncodedVideoChunk({
  type: 'key',
  timestamp: 0,
  data: proresFrameData,
}));
```

### Performance Expectations

| Resolution | Target Decode Time | FPS | Notes |
|------------|-------------------|-----|-------|
| 1080p | ~3ms | 300+ | Smooth scrubbing |
| 4K | ~5ms | 200+ | Smooth scrubbing |
| 8K | ~15ms | 60+ | Acceptable |

**Bottleneck analysis:**
- WASM entropy decode: ~1-2ms (10-20% of time)
- GPU transfer: ~0.5ms
- GPU compute: ~1-2ms
- VideoFrame creation: ~0.1ms

---

## Module Specifications

### 1. Color Converter (`color.rs`)

**Purpose**: Convert RGBA 8-bit frames to Y'CbCr 10-bit with 4:2:2 subsampling.

**Estimated Lines**: ~300

```rust
/// Color conversion configuration
pub struct ColorConfig {
    /// Color matrix (BT.709, BT.601, BT.2020)
    pub matrix: ColorMatrix,
    /// Input range (full 0-255 or limited 16-235)
    pub input_range: Range,
    /// Output range (always video range for ProRes)
    pub output_range: Range,
}

/// Color matrix coefficients
pub enum ColorMatrix {
    BT709,   // HD video (default)
    BT601,   // SD video
    BT2020,  // UHD/HDR
}

/// Y'CbCr frame with 4:2:2 subsampling
pub struct YCbCrFrame {
    /// Luma plane (full resolution, 10-bit samples)
    pub y: Vec<u16>,
    /// Chroma blue plane (half width, 10-bit samples)
    pub cb: Vec<u16>,
    /// Chroma red plane (half width, 10-bit samples)
    pub cr: Vec<u16>,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
}

impl ColorConverter {
    /// Create converter with specified color matrix
    pub fn new(config: ColorConfig) -> Self;

    /// Convert RGBA 8-bit to Y'CbCr 10-bit 4:2:2
    ///
    /// Input: RGBA interleaved [R,G,B,A,R,G,B,A,...]
    /// Output: Planar Y'CbCr with Cb/Cr at half width
    pub fn convert(&self, rgba: &[u8], width: u32, height: u32) -> YCbCrFrame;
}
```

**Implementation Notes**:
- Use fixed-point arithmetic for WASM performance
- Process 4 pixels at a time for SIMD optimization
- Cb/Cr subsampling: average of 2 horizontal neighbors

### 2. DCT Engine (`dct.rs`)

**Purpose**: Forward 8×8 Discrete Cosine Transform.

**Estimated Lines**: ~500

```rust
/// DCT computation engine with pre-computed tables
pub struct DctEngine {
    /// Cosine lookup table [8][8]
    cos_table: [[i32; 8]; 8],
    /// Normalization factors
    norm_factors: [i32; 8],
}

impl DctEngine {
    /// Create engine with pre-computed cosine tables
    pub fn new() -> Self;

    /// Forward 8×8 DCT (reference implementation)
    ///
    /// Input: 64 samples in row-major order (level-shifted: subtract 512 for 10-bit)
    /// Output: 64 DCT coefficients in row-major order
    pub fn forward_dct(&self, block: &[i16; 64]) -> [i32; 64];

    /// Optimized separable DCT (row-column decomposition)
    ///
    /// 2D DCT = 1D DCT on rows, then 1D DCT on columns
    /// Reduces operations from O(N⁴) to O(N³)
    pub fn forward_dct_fast(&self, block: &[i16; 64]) -> [i32; 64];

    /// SIMD-optimized DCT using wasm32 intrinsics
    #[cfg(target_arch = "wasm32")]
    pub fn forward_dct_simd(&self, block: &[i16; 64]) -> [i32; 64];
}
```

**DCT Implementation**:

```rust
/// 1D DCT-II on 8 samples (building block for 2D DCT)
fn dct_1d(input: &[i16; 8]) -> [i32; 8] {
    let mut output = [0i32; 8];

    for u in 0..8 {
        let mut sum = 0i64;
        for x in 0..8 {
            // cos((2x + 1) * u * π / 16)
            sum += input[x] as i64 * COS_TABLE[x][u] as i64;
        }
        // Apply normalization: C(u) = 1/√2 for u=0, 1 otherwise
        output[u] = ((sum * NORM[u] as i64) >> PRECISION) as i32;
    }

    output
}

/// 2D DCT via separable 1D transforms
fn dct_2d(block: &[i16; 64]) -> [i32; 64] {
    let mut temp = [[0i32; 8]; 8];
    let mut output = [0i32; 64];

    // Row transforms
    for row in 0..8 {
        let row_input: [i16; 8] = block[row*8..row*8+8].try_into().unwrap();
        let row_output = dct_1d(&row_input);
        temp[row] = row_output;
    }

    // Column transforms (on transposed data)
    for col in 0..8 {
        let col_input: [i16; 8] = [
            temp[0][col] as i16, temp[1][col] as i16,
            temp[2][col] as i16, temp[3][col] as i16,
            temp[4][col] as i16, temp[5][col] as i16,
            temp[6][col] as i16, temp[7][col] as i16,
        ];
        let col_output = dct_1d(&col_input);
        for row in 0..8 {
            output[row * 8 + col] = col_output[row];
        }
    }

    output
}
```

### 3. Quantizer (`quantize.rs`)

**Purpose**: Apply quantization matrices to DCT coefficients.

**Estimated Lines**: ~200

```rust
/// ProRes profile with associated quantization settings
#[derive(Clone, Copy)]
pub enum Profile {
    Proxy = 0,
    LT = 1,
    Standard = 2,
    HQ = 3,
    P4444 = 4,
    P4444XQ = 5,
}

/// Quantization tables and settings
pub struct Quantizer {
    /// Luma quantization matrix (zig-zag order)
    luma_matrix: [u16; 64],
    /// Chroma quantization matrix (zig-zag order)
    chroma_matrix: [u16; 64],
    /// Target quality (affects scale factor selection)
    quality: u8,
}

impl Quantizer {
    /// Create quantizer for specified profile
    pub fn new(profile: Profile) -> Self;

    /// Create quantizer with custom quality (0-100, 100=best)
    pub fn with_quality(profile: Profile, quality: u8) -> Self;

    /// Quantize DCT coefficients for one 8×8 block
    ///
    /// Returns quantized coefficients in zig-zag order
    pub fn quantize(&self, dct: &[i32; 64], is_luma: bool, scale: u8) -> [i16; 64];

    /// Calculate optimal scale factor for target bitrate
    pub fn calculate_scale_factor(&self, block_variance: u32) -> u8;
}

/// Default quantization matrices (from Apple spec)
pub const DEFAULT_LUMA_MATRIX: [u8; 64] = [
     4,  7,  9, 11, 13, 14, 15, 63,
     7,  7, 11, 12, 14, 15, 63, 63,
     9, 11, 13, 14, 15, 63, 63, 63,
    11, 11, 13, 14, 63, 63, 63, 63,
    11, 13, 14, 63, 63, 63, 63, 63,
    13, 14, 63, 63, 63, 63, 63, 63,
    13, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63,
];

pub const DEFAULT_CHROMA_MATRIX: [u8; 64] = [
     4,  7,  9, 11, 13, 14, 63, 63,
     7,  7, 11, 12, 14, 63, 63, 63,
     9, 11, 13, 14, 63, 63, 63, 63,
    11, 11, 13, 14, 63, 63, 63, 63,
    11, 13, 14, 63, 63, 63, 63, 63,
    13, 14, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63,
];
```

### 4. Entropy Coder (`entropy.rs`)

**Purpose**: Rice/Exponential-Golomb encoding of quantized coefficients.

**Estimated Lines**: ~600 (most complex module)

```rust
/// Rice coding parameters
#[derive(Clone, Copy)]
pub struct RiceParams {
    /// Maximum prefix length before switching to Exp-Golomb
    pub max_prefix: u8,
    /// Rice parameter (number of suffix bits)
    pub rice_param: u8,
    /// Exp-Golomb parameter for overflow values
    pub exp_golomb_param: u8,
}

/// Adaptive parameter sets for different coefficient types
pub struct EntropyParams {
    /// Parameters for DC coefficients
    pub dc_params: RiceParams,
    /// Parameters for AC coefficients (may vary by position)
    pub ac_params: [RiceParams; 64],
}

/// Entropy encoder with bit-level output
pub struct EntropyCoder {
    /// Bit-level writer
    writer: BitWriter,
    /// Current DC prediction values (per component)
    dc_pred: [i16; 3],
    /// Coding parameters
    params: EntropyParams,
}

impl EntropyCoder {
    /// Create entropy coder with default parameters
    pub fn new() -> Self;

    /// Reset DC predictors (call at slice boundary)
    pub fn reset_predictors(&mut self);

    /// Encode one 8×8 block of quantized coefficients
    pub fn encode_block(&mut self, coeffs: &[i16; 64], component: Component);

    /// Encode DC coefficient (delta from previous)
    fn encode_dc(&mut self, dc: i16, component: Component);

    /// Encode AC coefficients with run-length coding
    fn encode_ac(&mut self, ac: &[i16; 63]);

    /// Get encoded data and reset writer
    pub fn finish(&mut self) -> Vec<u8>;
}

/// Component type for DC prediction tracking
pub enum Component {
    Y = 0,   // Luma
    Cb = 1,  // Chroma blue
    Cr = 2,  // Chroma red
}
```

**Rice/Exp-Golomb Implementation**:

```rust
impl EntropyCoder {
    /// Encode unsigned value using Rice code with Exp-Golomb fallback
    fn encode_rice(&mut self, value: u16, params: &RiceParams) {
        let prefix = value >> params.rice_param;
        let suffix = value & ((1 << params.rice_param) - 1);

        if prefix < params.max_prefix as u16 {
            // Rice code: unary prefix + binary suffix
            self.write_unary(prefix as u8);
            self.writer.write_bits(suffix as u32, params.rice_param);
        } else {
            // Exp-Golomb for large values
            self.write_unary(params.max_prefix);
            let remainder = value - ((params.max_prefix as u16) << params.rice_param);
            self.encode_exp_golomb(remainder, params.exp_golomb_param);
        }
    }

    /// Write unary code: n ones followed by zero
    fn write_unary(&mut self, n: u8) {
        for _ in 0..n {
            self.writer.write_bit(true);
        }
        self.writer.write_bit(false);
    }

    /// Encode using Exponential-Golomb code
    fn encode_exp_golomb(&mut self, value: u16, k: u8) {
        let adjusted = value + (1 << k);
        let len = (16 - adjusted.leading_zeros()) as u8;

        // Write (len - k - 1) zeros
        for _ in 0..(len - k - 1) {
            self.writer.write_bit(false);
        }

        // Write len bits of adjusted value
        self.writer.write_bits(adjusted as u32, len);
    }

    /// Encode signed value (magnitude + sign bit)
    fn encode_signed(&mut self, value: i16, params: &RiceParams) {
        let magnitude = value.unsigned_abs();
        if magnitude == 0 {
            self.encode_rice(0, params);
        } else {
            self.encode_rice(magnitude - 1, params);
            self.writer.write_bit(value < 0);
        }
    }
}
```

### 5. Bit Writer (`bitwriter.rs`)

**Purpose**: Efficient bit-level output buffer.

**Estimated Lines**: ~150

```rust
/// Bit-level writer for building encoded bitstream
pub struct BitWriter {
    /// Output buffer
    buffer: Vec<u8>,
    /// Current byte being built
    current_byte: u8,
    /// Bits written to current byte (0-7)
    bit_position: u8,
}

impl BitWriter {
    /// Create new bit writer with optional capacity hint
    pub fn new() -> Self;
    pub fn with_capacity(bytes: usize) -> Self;

    /// Write single bit
    pub fn write_bit(&mut self, bit: bool);

    /// Write n bits from value (LSB first or MSB first based on spec)
    pub fn write_bits(&mut self, value: u32, n: u8);

    /// Write complete byte (flushes partial byte first)
    pub fn write_byte(&mut self, byte: u8);

    /// Write multiple bytes
    pub fn write_bytes(&mut self, bytes: &[u8]);

    /// Pad to byte boundary with zeros
    pub fn byte_align(&mut self);

    /// Get current bit position in stream
    pub fn bit_position(&self) -> usize;

    /// Finish and return buffer (pads to byte boundary)
    pub fn finish(self) -> Vec<u8>;

    /// Get buffer reference without consuming
    pub fn as_bytes(&self) -> &[u8];
}
```

### 6. Slice Encoder (`slice.rs`)

**Purpose**: Encode one horizontal slice of macroblocks.

**Estimated Lines**: ~400

```rust
/// Slice encoding context
pub struct SliceEncoder {
    /// DCT transform engine
    dct: DctEngine,
    /// Quantization tables
    quantizer: Quantizer,
    /// Entropy coder
    entropy: EntropyCoder,
    /// Slice width in macroblocks
    slice_width_mbs: u8,
}

/// Encoded slice data
pub struct EncodedSlice {
    /// Slice header (scale factor, etc.)
    pub header: Vec<u8>,
    /// Encoded coefficient data
    pub data: Vec<u8>,
    /// Total size in bytes
    pub size: usize,
}

impl SliceEncoder {
    /// Create slice encoder with settings
    pub fn new(profile: Profile, slice_width_mbs: u8) -> Self;

    /// Encode one slice from Y'CbCr frame
    ///
    /// slice_x, slice_y: position in macroblocks
    pub fn encode_slice(
        &mut self,
        frame: &YCbCrFrame,
        slice_x: u32,
        slice_y: u32,
    ) -> EncodedSlice;

    /// Extract 8×8 block from frame at specified position
    fn extract_block(
        &self,
        plane: &[u16],
        plane_width: u32,
        block_x: u32,
        block_y: u32,
    ) -> [i16; 64];

    /// Encode single 8×8 block (DCT → quantize → entropy)
    fn encode_block(&mut self, block: &[i16; 64], is_luma: bool);
}
```

### 7. Frame Encoder (`frame.rs`)

**Purpose**: Encode complete frame with header and slices.

**Estimated Lines**: ~300

```rust
/// Frame encoding configuration
pub struct FrameConfig {
    /// ProRes profile
    pub profile: Profile,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Interlace mode
    pub interlace: InterlaceMode,
    /// Color primaries
    pub color_primaries: u8,
    /// Transfer function
    pub transfer_func: u8,
    /// Matrix coefficients
    pub matrix_coeffs: u8,
}

/// Interlace mode
pub enum InterlaceMode {
    Progressive = 0,
    InterlacedTFF = 1,  // Top field first
    InterlacedBFF = 2,  // Bottom field first
}

/// Frame encoder
pub struct FrameEncoder {
    config: FrameConfig,
    slice_encoder: SliceEncoder,
}

impl FrameEncoder {
    /// Create frame encoder with configuration
    pub fn new(config: FrameConfig) -> Self;

    /// Encode complete frame
    ///
    /// Returns ProRes bitstream (without container)
    pub fn encode_frame(&mut self, frame: &YCbCrFrame) -> Vec<u8>;

    /// Write frame container atom ('icpf')
    fn write_frame_atom(&self, out: &mut Vec<u8>);

    /// Write frame header
    fn write_frame_header(&self, out: &mut Vec<u8>);

    /// Write picture header and slice index
    fn write_picture_header(&self, out: &mut Vec<u8>, num_slices: u32);
}
```

**Frame Header Format**:

```rust
fn write_frame_header(&self, out: &mut Vec<u8>) {
    let header_start = out.len();

    // Placeholder for header size (2 bytes, filled later)
    out.extend_from_slice(&[0, 0]);

    // Reserved (2 bytes)
    out.extend_from_slice(&[0, 0]);

    // Bitstream version (1 byte)
    out.push(0);

    // Encoder identifier (4 bytes) - "rust" or custom
    out.extend_from_slice(b"wasm");

    // Frame dimensions (2 + 2 bytes)
    out.extend_from_slice(&(self.config.width as u16).to_be_bytes());
    out.extend_from_slice(&(self.config.height as u16).to_be_bytes());

    // Flags byte 1: chroma_format (2 bits) | reserved (2 bits) |
    //               interlace_mode (2 bits) | reserved (2 bits)
    let flags1 = (0b10 << 6) |  // 4:2:2 chroma
                 ((self.config.interlace as u8) << 2);
    out.push(flags1);

    // Flags byte 2: aspect_ratio (4 bits) | frame_rate (4 bits)
    let flags2 = (0 << 4) | 0;  // Square pixels, unknown frame rate
    out.push(flags2);

    // Color info
    out.push(self.config.color_primaries);   // 1 = BT.709
    out.push(self.config.transfer_func);      // 1 = BT.709
    out.push(self.config.matrix_coeffs);      // 1 = BT.709

    // Alpha info (4 bits) | reserved (4 bits)
    out.push(0);  // No alpha

    // Reserved (1 byte)
    out.push(0);

    // No custom quantization matrices for now

    // Fill in header size
    let header_size = (out.len() - header_start) as u16;
    out[header_start..header_start+2].copy_from_slice(&header_size.to_be_bytes());
}
```

### 8. MOV Muxer (`mux.rs`)

**Purpose**: Wrap ProRes frames in QuickTime MOV container.

**Estimated Lines**: ~400

```rust
/// MOV container muxer
pub struct MovMuxer {
    /// Frame timing info
    timescale: u32,
    /// Frame duration in timescale units
    frame_duration: u32,
    /// Video dimensions
    width: u32,
    height: u32,
    /// ProRes profile FourCC
    codec_fourcc: [u8; 4],
}

/// Frame metadata for muxing
pub struct FrameInfo {
    /// Encoded frame data
    pub data: Vec<u8>,
    /// Byte offset in mdat atom
    pub offset: u64,
    /// Frame size in bytes
    pub size: u32,
}

impl MovMuxer {
    /// Create muxer for ProRes video
    pub fn new(width: u32, height: u32, fps: f64, profile: Profile) -> Self;

    /// Mux encoded frames into MOV container
    pub fn mux(&self, frames: &[Vec<u8>]) -> Vec<u8>;

    // Atom writers
    fn write_ftyp(&self, out: &mut Vec<u8>);
    fn write_mdat(&self, out: &mut Vec<u8>, frames: &[Vec<u8>]) -> Vec<FrameInfo>;
    fn write_moov(&self, out: &mut Vec<u8>, frame_infos: &[FrameInfo]);

    // Movie atom components
    fn write_mvhd(&self, out: &mut Vec<u8>, duration: u64);
    fn write_trak(&self, out: &mut Vec<u8>, frame_infos: &[FrameInfo]);
    fn write_tkhd(&self, out: &mut Vec<u8>, duration: u64);
    fn write_mdia(&self, out: &mut Vec<u8>, frame_infos: &[FrameInfo]);
    fn write_minf(&self, out: &mut Vec<u8>, frame_infos: &[FrameInfo]);
    fn write_stbl(&self, out: &mut Vec<u8>, frame_infos: &[FrameInfo]);
}
```

**MOV Structure**:

```
mov file
├── ftyp (file type)
│   ├── brand: 'qt  '
│   └── compatible: ['qt  ']
├── mdat (media data)
│   └── [frame 0][frame 1][frame 2]...
└── moov (movie metadata)
    ├── mvhd (movie header)
    │   ├── timescale
    │   └── duration
    └── trak (video track)
        ├── tkhd (track header)
        │   ├── width, height
        │   └── duration
        └── mdia (media)
            ├── mdhd (media header)
            ├── hdlr (handler: 'vide')
            └── minf (media info)
                ├── vmhd (video media header)
                ├── dinf (data info)
                │   └── dref (data reference)
                └── stbl (sample table)
                    ├── stsd (sample description)
                    │   └── 'apch' entry (ProRes HQ)
                    ├── stts (time to sample)
                    ├── stsc (sample to chunk)
                    ├── stsz (sample sizes)
                    └── stco/co64 (chunk offsets)
```

---

## WebGPU Compute Shaders

The GPU-accelerated portions run as WebGPU compute shaders written in WGSL.

### Inverse DCT Shader (`idct.wgsl`)

The most performance-critical shader - runs on every 8x8 block:

```wgsl
// idct.wgsl - Inverse Discrete Cosine Transform

struct Uniforms {
    width: u32,
    height: u32,
    blocks_per_row: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> coefficients: array<f32>;
@group(0) @binding(2) var<storage, read_write> pixels: array<f32>;

// Precomputed cosine table for DCT-II
const COS_TABLE: array<array<f32, 8>, 8> = array<array<f32, 8>, 8>(
    array<f32, 8>(0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536),
    array<f32, 8>(0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904),
    array<f32, 8>(0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619),
    array<f32, 8>(0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157),
    array<f32, 8>(0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536),
    array<f32, 8>(0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778),
    array<f32, 8>(0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913),
    array<f32, 8>(0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975),
);

// 1D IDCT on 8 values
fn idct_1d(input: array<f32, 8>) -> array<f32, 8> {
    var output: array<f32, 8>;
    for (var x = 0u; x < 8u; x++) {
        var sum = 0.0;
        for (var u = 0u; u < 8u; u++) {
            sum += input[u] * COS_TABLE[u][x];
        }
        output[x] = sum;
    }
    return output;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_x = global_id.x;
    let block_y = global_id.y;

    if (block_x >= uniforms.blocks_per_row || block_y >= uniforms.height / 8u) {
        return;
    }

    let block_idx = block_y * uniforms.blocks_per_row + block_x;
    let coeff_offset = block_idx * 64u;

    // Load 8x8 coefficient block
    var block: array<array<f32, 8>, 8>;
    for (var y = 0u; y < 8u; y++) {
        for (var x = 0u; x < 8u; x++) {
            block[y][x] = coefficients[coeff_offset + y * 8u + x];
        }
    }

    // 2D IDCT: rows first, then columns
    var temp: array<array<f32, 8>, 8>;

    // IDCT on rows
    for (var y = 0u; y < 8u; y++) {
        temp[y] = idct_1d(block[y]);
    }

    // IDCT on columns
    for (var x = 0u; x < 8u; x++) {
        var col: array<f32, 8>;
        for (var y = 0u; y < 8u; y++) {
            col[y] = temp[y][x];
        }
        let result = idct_1d(col);
        for (var y = 0u; y < 8u; y++) {
            temp[y][x] = result[y];
        }
    }

    // Write output pixels (add 512 for 10-bit level shift)
    let pixel_x = block_x * 8u;
    let pixel_y = block_y * 8u;
    for (var y = 0u; y < 8u; y++) {
        for (var x = 0u; x < 8u; x++) {
            let px = pixel_x + x;
            let py = pixel_y + y;
            if (px < uniforms.width && py < uniforms.height) {
                let idx = py * uniforms.width + px;
                pixels[idx] = temp[y][x] + 512.0;
            }
        }
    }
}
```

### Dequantization Shader (`dequantize.wgsl`)

```wgsl
// dequantize.wgsl - Apply inverse quantization

struct Uniforms {
    num_blocks: u32,
    scale_factor: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> quantized: array<i32>;
@group(0) @binding(2) var<storage, read> quant_matrix: array<f32, 64>;
@group(0) @binding(3) var<storage, read_write> dequantized: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_idx = global_id.x / 64u;
    let coeff_idx = global_id.x % 64u;

    if (block_idx >= uniforms.num_blocks) {
        return;
    }

    let idx = block_idx * 64u + coeff_idx;
    let qval = f32(quantized[idx]);
    let matrix_val = quant_matrix[coeff_idx];

    // Dequantize: multiply by quantization matrix and scale factor
    dequantized[idx] = qval * matrix_val * uniforms.scale_factor;
}
```

### Color Space Conversion (`colorspace.wgsl`)

```wgsl
// colorspace.wgsl - YCbCr to RGB conversion (BT.709)

struct Uniforms {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> y_plane: array<f32>;
@group(0) @binding(2) var<storage, read> cb_plane: array<f32>;
@group(0) @binding(3) var<storage, read> cr_plane: array<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba8unorm, write>;

// BT.709 YCbCr to RGB matrix
const KR: f32 = 0.2126;
const KG: f32 = 0.7152;
const KB: f32 = 0.0722;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let y_idx = y * uniforms.width + x;
    let c_idx = y * (uniforms.width / 2u) + (x / 2u);  // 4:2:2 subsampling

    // Load YCbCr values (10-bit, 0-1023 range)
    let y_val = y_plane[y_idx];
    let cb_val = cb_plane[c_idx] - 512.0;  // Center chroma
    let cr_val = cr_plane[c_idx] - 512.0;

    // Scale to 0-1 range
    let y_norm = (y_val - 64.0) / 876.0;   // Video range scaling
    let cb_norm = cb_val / 896.0;
    let cr_norm = cr_val / 896.0;

    // BT.709 conversion
    let r = y_norm + 1.5748 * cr_norm;
    let g = y_norm - 0.1873 * cb_norm - 0.4681 * cr_norm;
    let b = y_norm + 1.8556 * cb_norm;

    // Clamp and write
    let rgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(rgb, 1.0));
}
```

### Forward DCT Shader (for encoding)

```wgsl
// dct.wgsl - Forward Discrete Cosine Transform

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> pixels: array<f32>;
@group(0) @binding(2) var<storage, read_write> coefficients: array<f32>;

// Same COS_TABLE as IDCT but transposed usage

fn dct_1d(input: array<f32, 8>) -> array<f32, 8> {
    var output: array<f32, 8>;
    for (var u = 0u; u < 8u; u++) {
        var sum = 0.0;
        for (var x = 0u; x < 8u; x++) {
            sum += input[x] * COS_TABLE[u][x];
        }
        output[u] = sum;
    }
    return output;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_x = global_id.x;
    let block_y = global_id.y;

    // Load 8x8 pixel block with level shift (-512)
    var block: array<array<f32, 8>, 8>;
    let pixel_x = block_x * 8u;
    let pixel_y = block_y * 8u;

    for (var y = 0u; y < 8u; y++) {
        for (var x = 0u; x < 8u; x++) {
            let idx = (pixel_y + y) * uniforms.width + (pixel_x + x);
            block[y][x] = pixels[idx] - 512.0;  // Level shift
        }
    }

    // 2D DCT: rows first, then columns
    var temp: array<array<f32, 8>, 8>;

    // DCT on rows
    for (var y = 0u; y < 8u; y++) {
        temp[y] = dct_1d(block[y]);
    }

    // DCT on columns
    for (var x = 0u; x < 8u; x++) {
        var col: array<f32, 8>;
        for (var y = 0u; y < 8u; y++) {
            col[y] = temp[y][x];
        }
        let result = dct_1d(col);
        for (var y = 0u; y < 8u; y++) {
            temp[y][x] = result[y];
        }
    }

    // Write coefficients in zig-zag order
    let block_idx = block_y * uniforms.blocks_per_row + block_x;
    let coeff_offset = block_idx * 64u;

    for (var y = 0u; y < 8u; y++) {
        for (var x = 0u; x < 8u; x++) {
            let zigzag_idx = ZIGZAG_ORDER[y * 8u + x];
            coefficients[coeff_offset + zigzag_idx] = temp[y][x];
        }
    }
}
```

### Shader Loading and Compilation

```typescript
// shader-loader.ts

const SHADER_SOURCES = {
  idct: IDCT_WGSL,
  dct: DCT_WGSL,
  dequantize: DEQUANTIZE_WGSL,
  quantize: QUANTIZE_WGSL,
  colorspace: COLORSPACE_WGSL,
};

export async function createProResShaderModules(
  device: GPUDevice
): Promise<Map<string, GPUShaderModule>> {
  const modules = new Map<string, GPUShaderModule>();

  for (const [name, source] of Object.entries(SHADER_SOURCES)) {
    const module = device.createShaderModule({
      label: `prores-${name}`,
      code: source,
    });

    // Check for compilation errors
    const info = await module.getCompilationInfo();
    if (info.messages.some(m => m.type === 'error')) {
      throw new Error(`Shader ${name} compilation failed: ${
        info.messages.map(m => m.message).join('\n')
      }`);
    }

    modules.set(name, module);
  }

  return modules;
}
```

---

## WASM Integration

### Main Entry Point (`lib.rs`)

The WASM module exports both encoder and decoder, with the decoder optimized to output quantized coefficients for WebGPU processing.

```rust
use wasm_bindgen::prelude::*;

// ============================================================================
// DECODER (for timeline scrubbing - outputs coefficients for WebGPU)
// ============================================================================

/// ProRes decoder for WebAssembly
///
/// This decoder performs entropy decoding (CPU-bound) and outputs
/// quantized DCT coefficients for GPU processing via WebGPU.
#[wasm_bindgen]
pub struct ProResDecoder {
    config: DecoderConfig,
    demuxer: MovDemuxer,
    entropy_decoder: EntropyDecoder,
    /// Pre-allocated coefficient buffer (reused across frames)
    coeff_buffer: Vec<i16>,
}

#[wasm_bindgen]
impl ProResDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ProResDecoder {
        ProResDecoder {
            config: DecoderConfig::default(),
            demuxer: MovDemuxer::new(),
            entropy_decoder: EntropyDecoder::new(),
            coeff_buffer: Vec::new(),
        }
    }

    /// Parse MOV container and extract ProRes metadata
    ///
    /// Returns JSON with: width, height, frame_count, profile, duration
    #[wasm_bindgen]
    pub fn parse_container(&mut self, data: &[u8]) -> Result<String, JsValue> {
        let info = self.demuxer.parse(data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.config = DecoderConfig {
            width: info.width,
            height: info.height,
            profile: info.profile,
        };

        // Pre-allocate coefficient buffer
        let num_blocks = (info.width / 8) * (info.height / 8) * 8; // 8 blocks per MB
        self.coeff_buffer = vec![0i16; num_blocks as usize * 64];

        Ok(serde_json::to_string(&info).unwrap())
    }

    /// Decode a single frame - returns quantized DCT coefficients
    ///
    /// This performs ONLY the serial operations (entropy decode).
    /// The returned coefficients should be passed to WebGPU for:
    /// - Dequantization
    /// - Inverse DCT
    /// - Color conversion
    ///
    /// Returns: Int16Array of quantized coefficients
    #[wasm_bindgen]
    pub fn decode_frame(&mut self, frame_index: u32) -> Result<Vec<i16>, JsValue> {
        let frame_data = self.demuxer.get_frame(frame_index)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Parse frame header
        let header = self.parse_frame_header(&frame_data)?;

        // Entropy decode all slices → quantized coefficients
        self.entropy_decoder.reset();

        for slice in header.slices.iter() {
            self.entropy_decoder.decode_slice(
                &frame_data[slice.offset..slice.offset + slice.size],
                &mut self.coeff_buffer[slice.coeff_offset..],
            )?;
        }

        Ok(self.coeff_buffer.clone())
    }

    /// Get scale factors for each slice (needed for dequantization on GPU)
    #[wasm_bindgen]
    pub fn get_scale_factors(&self, frame_index: u32) -> Result<Vec<u8>, JsValue> {
        let frame_data = self.demuxer.get_frame(frame_index)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let header = self.parse_frame_header(&frame_data)?;
        Ok(header.slices.iter().map(|s| s.scale_factor).collect())
    }

    /// Get quantization matrices (typically same for whole file)
    #[wasm_bindgen]
    pub fn get_quant_matrices(&self) -> QuantMatrices {
        QuantMatrices {
            luma: self.config.luma_matrix.to_vec(),
            chroma: self.config.chroma_matrix.to_vec(),
        }
    }
}

#[wasm_bindgen]
pub struct QuantMatrices {
    #[wasm_bindgen(getter_with_clone)]
    pub luma: Vec<u8>,
    #[wasm_bindgen(getter_with_clone)]
    pub chroma: Vec<u8>,
}

// ============================================================================
// ENCODER (for export - receives coefficients from WebGPU)
// ============================================================================

/// ProRes encoder for WebAssembly
#[wasm_bindgen]
pub struct ProResEncoder {
    color_converter: ColorConverter,
    frame_encoder: FrameEncoder,
    muxer: MovMuxer,
    frames: Vec<Vec<u8>>,
    config: EncoderConfig,
}

#[wasm_bindgen]
impl ProResEncoder {
    /// Create new encoder
    ///
    /// # Arguments
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `fps` - Frames per second
    /// * `profile` - ProRes profile (0=Proxy, 1=LT, 2=Standard, 3=HQ)
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, fps: f64, profile: u8) -> Result<ProResEncoder, JsValue> {
        // Validate inputs
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Err(JsValue::from_str("Invalid dimensions"));
        }

        let profile = match profile {
            0 => Profile::Proxy,
            1 => Profile::LT,
            2 => Profile::Standard,
            3 => Profile::HQ,
            _ => return Err(JsValue::from_str("Invalid profile (use 0-3)")),
        };

        Ok(Self {
            color_converter: ColorConverter::new(ColorConfig::bt709()),
            frame_encoder: FrameEncoder::new(FrameConfig {
                profile,
                width,
                height,
                interlace: InterlaceMode::Progressive,
                color_primaries: 1,
                transfer_func: 1,
                matrix_coeffs: 1,
            }),
            muxer: MovMuxer::new(width, height, fps, profile),
            frames: Vec::new(),
            config: EncoderConfig { width, height, fps, profile },
        })
    }

    /// Add frame from RGBA data
    ///
    /// # Arguments
    /// * `rgba` - RGBA pixel data (width × height × 4 bytes)
    #[wasm_bindgen]
    pub fn add_frame(&mut self, rgba: &[u8]) -> Result<(), JsValue> {
        let expected_size = (self.config.width * self.config.height * 4) as usize;
        if rgba.len() != expected_size {
            return Err(JsValue::from_str(&format!(
                "Expected {} bytes, got {}", expected_size, rgba.len()
            )));
        }

        // Convert color space
        let ycbcr = self.color_converter.convert(
            rgba,
            self.config.width,
            self.config.height
        );

        // Encode frame
        let encoded = self.frame_encoder.encode_frame(&ycbcr);
        self.frames.push(encoded);

        Ok(())
    }

    /// Get number of encoded frames
    #[wasm_bindgen]
    pub fn frame_count(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Finalize and return MOV file
    ///
    /// Returns complete MOV file as byte array
    #[wasm_bindgen]
    pub fn finalize(&mut self) -> Vec<u8> {
        let mov = self.muxer.mux(&self.frames);
        self.frames.clear();  // Free memory
        mov
    }

    /// Encode single frame and return ProRes bitstream (without container)
    ///
    /// Useful for streaming or custom container formats
    #[wasm_bindgen]
    pub fn encode_frame_only(&mut self, rgba: &[u8]) -> Result<Vec<u8>, JsValue> {
        let expected_size = (self.config.width * self.config.height * 4) as usize;
        if rgba.len() != expected_size {
            return Err(JsValue::from_str("Invalid frame size"));
        }

        let ycbcr = self.color_converter.convert(
            rgba,
            self.config.width,
            self.config.height
        );

        Ok(self.frame_encoder.encode_frame(&ycbcr))
    }
}
```

### TypeScript Bindings

```typescript
// prores.d.ts (auto-generated by wasm-bindgen)

export class ProResEncoder {
  constructor(width: number, height: number, fps: number, profile: number);

  /**
   * Add frame from RGBA data
   * @param rgba - Uint8Array of RGBA pixels (width × height × 4 bytes)
   */
  add_frame(rgba: Uint8Array): void;

  /**
   * Get number of encoded frames
   */
  frame_count(): number;

  /**
   * Finalize and return MOV file
   * @returns Uint8Array containing complete MOV file
   */
  finalize(): Uint8Array;

  /**
   * Encode single frame without container
   */
  encode_frame_only(rgba: Uint8Array): Uint8Array;

  /**
   * Free resources
   */
  free(): void;
}

// Profile constants
export const PRORES_PROXY = 0;
export const PRORES_LT = 1;
export const PRORES_STANDARD = 2;
export const PRORES_HQ = 3;
```

### Usage Example (JavaScript)

```javascript
import init, { ProResEncoder, PRORES_HQ } from './prores_wasm.js';

async function exportProRes(canvas, fps, duration) {
  // Initialize WASM module
  await init();

  // Create encoder (1920×1080, 24fps, HQ profile)
  const encoder = new ProResEncoder(
    canvas.width,
    canvas.height,
    fps,
    PRORES_HQ
  );

  const ctx = canvas.getContext('2d');
  const totalFrames = Math.round(fps * duration);

  // Encode each frame
  for (let i = 0; i < totalFrames; i++) {
    // Render frame to canvas (your rendering logic here)
    renderFrame(ctx, i / fps);

    // Get RGBA data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Add frame to encoder
    encoder.add_frame(imageData.data);

    // Progress callback
    onProgress(i / totalFrames);
  }

  // Finalize and get MOV file
  const movData = encoder.finalize();

  // Clean up
  encoder.free();

  // Create download blob
  const blob = new Blob([movData], { type: 'video/quicktime' });
  return URL.createObjectURL(blob);
}
```

### Integration with WebGPU

```javascript
// Reading frames from WebGPU for ProRes encoding
async function encodeWebGPUFrames(device, texture, encoder) {
  // Create staging buffer for readback
  const bytesPerRow = Math.ceil(texture.width * 4 / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * texture.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Copy texture to buffer
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow },
    { width: texture.width, height: texture.height }
  );
  device.queue.submit([commandEncoder.finish()]);

  // Map and read data
  await buffer.mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(buffer.getMappedRange());

  // Handle row padding if bytesPerRow != width * 4
  const rgba = new Uint8Array(texture.width * texture.height * 4);
  for (let y = 0; y < texture.height; y++) {
    const srcOffset = y * bytesPerRow;
    const dstOffset = y * texture.width * 4;
    rgba.set(data.subarray(srcOffset, srcOffset + texture.width * 4), dstOffset);
  }

  buffer.unmap();

  // Add to ProRes encoder
  encoder.add_frame(rgba);
}
```

---

## Performance Optimization

### SIMD Optimization

WASM SIMD (128-bit vectors) can significantly speed up DCT and color conversion.

```rust
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// SIMD-optimized 8×8 DCT row transform
#[cfg(target_arch = "wasm32")]
pub fn dct_row_simd(input: &[i16; 8]) -> [i32; 8] {
    // Load 8 × i16 into v128
    let src = v128_load(input.as_ptr() as *const v128);

    // Extend to i32 (two v128s)
    let low = i32x4_extend_low_i16x8(src);
    let high = i32x4_extend_high_i16x8(src);

    // Apply DCT butterfly operations...
    // (Implementation depends on specific DCT algorithm)

    todo!()
}
```

### Memory Management

```rust
/// Pre-allocated buffers to avoid allocations per frame
pub struct EncoderBuffers {
    /// Y'CbCr frame buffer
    ycbcr: YCbCrFrame,
    /// DCT coefficient buffer
    dct_buffer: Vec<i32>,
    /// Quantized coefficient buffer
    quant_buffer: Vec<i16>,
    /// Slice output buffer
    slice_buffer: Vec<u8>,
}

impl EncoderBuffers {
    pub fn new(width: u32, height: u32) -> Self {
        let y_size = (width * height) as usize;
        let c_size = (width / 2 * height) as usize;

        Self {
            ycbcr: YCbCrFrame {
                y: vec![0u16; y_size],
                cb: vec![0u16; c_size],
                cr: vec![0u16; c_size],
                width,
                height,
            },
            dct_buffer: vec![0i32; 64],
            quant_buffer: vec![0i16; 64],
            slice_buffer: Vec::with_capacity(1024 * 1024),  // 1MB
        }
    }
}
```

### Parallelization with Web Workers

```javascript
// Main thread
const workers = [];
const NUM_WORKERS = navigator.hardwareConcurrency || 4;

// Initialize worker pool
for (let i = 0; i < NUM_WORKERS; i++) {
  workers.push(new Worker('prores-worker.js'));
}

// Distribute slices across workers
async function encodeFrameParallel(rgba, width, height) {
  const slicesPerWorker = Math.ceil(height / 16 / NUM_WORKERS);

  const promises = workers.map((worker, i) => {
    const startRow = i * slicesPerWorker * 16;
    const endRow = Math.min((i + 1) * slicesPerWorker * 16, height);

    return new Promise(resolve => {
      worker.onmessage = (e) => resolve(e.data);
      worker.postMessage({
        rgba,
        width,
        height,
        startRow,
        endRow
      });
    });
  });

  const sliceData = await Promise.all(promises);
  return assembleFrame(sliceData);
}
```

### Performance Targets

| Resolution | Target FPS | Notes |
|------------|------------|-------|
| 1080p | 2-5 fps | Acceptable for export |
| 720p | 5-10 fps | Good for previews |
| 480p | 15-20 fps | Near real-time |

**Comparison** (estimated):
- FFmpeg native: ~30 fps @ 1080p
- FFmpeg WASM: ~5 fps @ 1080p
- Pure Rust WASM (optimized): ~3-5 fps @ 1080p

---

## Testing & Conformance

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_only() {
        let engine = DctEngine::new();

        // Flat block should have DC only
        let flat_block = [512i16; 64];
        let dct = engine.forward_dct(&flat_block);

        assert!(dct[0] != 0);  // DC coefficient
        for i in 1..64 {
            assert_eq!(dct[i], 0);  // AC coefficients should be zero
        }
    }

    #[test]
    fn test_color_conversion_white() {
        let converter = ColorConverter::new(ColorConfig::bt709());

        // White pixel (255, 255, 255) → Y=940, Cb=512, Cr=512 (10-bit)
        let rgba = [255, 255, 255, 255];
        let ycbcr = converter.convert(&rgba, 1, 1);

        assert!((ycbcr.y[0] as i32 - 940).abs() < 2);
        assert!((ycbcr.cb[0] as i32 - 512).abs() < 2);
        assert!((ycbcr.cr[0] as i32 - 512).abs() < 2);
    }

    #[test]
    fn test_rice_coding_roundtrip() {
        let params = RiceParams {
            max_prefix: 9,
            rice_param: 3,
            exp_golomb_param: 4,
        };

        for value in [0, 1, 7, 8, 100, 1000] {
            let mut encoder = EntropyCoder::new();
            encoder.encode_rice(value, &params);
            let encoded = encoder.finish();

            let mut decoder = EntropyDecoder::new(&encoded);
            let decoded = decoder.decode_rice(&params);

            assert_eq!(decoded, value);
        }
    }
}
```

### Conformance Testing

```rust
/// Compare output against FFmpeg reference
#[test]
fn test_conformance_with_ffmpeg() {
    // Generate test pattern
    let test_frame = generate_color_bars(1920, 1080);

    // Encode with our encoder
    let our_output = encode_frame(&test_frame, Profile::HQ);

    // Decode with FFmpeg and compare
    // (Would use ffmpeg bindings or external process)
    let reference = decode_with_ffmpeg(&our_output);

    // PSNR should be > 40dB (visually lossless)
    let psnr = calculate_psnr(&test_frame, &reference);
    assert!(psnr > 40.0);
}
```

### Test Files

```
tests/
├── fixtures/
│   ├── color_bars_1080p.raw      # Raw RGB test pattern
│   ├── gradient_1080p.raw        # Gradient test pattern
│   └── reference_prores_hq.mov   # FFmpeg-generated reference
├── conformance_test.rs
├── unit_tests.rs
└── integration_tests.rs
```

### Validation with Professional Software

Files should be tested in:
- Final Cut Pro (native ProRes support)
- DaVinci Resolve (native ProRes support)
- Adobe Premiere Pro (requires codec)
- FFmpeg/ffprobe (analysis)

```bash
# Validate with FFprobe
ffprobe -v error -show_format -show_streams output.mov

# Expected output includes:
# codec_name=prores
# profile=3  (for HQ)
# pix_fmt=yuv422p10le
```

---

## Project Structure

```
prores-webcodec/
├── Cargo.toml
├── README.md
├── LICENSE                         # MIT or Apache-2.0
│
├── rust/                           # Rust WASM module
│   ├── src/
│   │   ├── lib.rs                  # WASM entry point (decoder + encoder)
│   │   ├── decoder.rs              # Decoder facade
│   │   ├── encoder.rs              # Encoder facade
│   │   │
│   │   ├── decode/                 # Decoding modules
│   │   │   ├── mod.rs
│   │   │   ├── bitreader.rs        # Bit-level input
│   │   │   ├── entropy.rs          # Rice/Exp-Golomb decoding
│   │   │   ├── slice.rs            # Slice → coefficients
│   │   │   └── header.rs           # Frame/slice header parsing
│   │   │
│   │   ├── encode/                 # Encoding modules
│   │   │   ├── mod.rs
│   │   │   ├── bitwriter.rs        # Bit-level output
│   │   │   ├── entropy.rs          # Rice/Exp-Golomb encoding
│   │   │   ├── slice.rs            # Coefficients → bitstream
│   │   │   └── header.rs           # Frame/slice header writing
│   │   │
│   │   ├── container/              # MOV container
│   │   │   ├── mod.rs
│   │   │   ├── demux.rs            # MOV parser (read)
│   │   │   ├── mux.rs              # MOV muxer (write)
│   │   │   └── atoms.rs            # QuickTime atom helpers
│   │   │
│   │   └── common/                 # Shared utilities
│   │       ├── mod.rs
│   │       ├── tables.rs           # Quantization matrices, zig-zag
│   │       └── profiles.rs         # ProRes profile definitions
│   │
│   ├── benches/
│   │   └── entropy_bench.rs        # Entropy coding performance
│   │
│   └── tests/
│       ├── decode_test.rs          # Decoder tests
│       ├── encode_test.rs          # Encoder tests
│       └── fixtures/               # Test ProRes files
│
├── webgpu/                         # WebGPU shaders (WGSL)
│   ├── dequantize.wgsl             # Inverse quantization
│   ├── quantize.wgsl               # Forward quantization
│   ├── idct.wgsl                   # Inverse DCT (8x8)
│   ├── dct.wgsl                    # Forward DCT (8x8)
│   ├── colorspace.wgsl             # YCbCr ↔ RGB conversion
│   └── common.wgsl                 # Shared constants and utilities
│
├── ts/                             # TypeScript bridge layer
│   ├── src/
│   │   ├── index.ts                # Package entry point
│   │   ├── prores-decoder.ts       # WebCodecs-like decoder API
│   │   ├── prores-encoder.ts       # WebCodecs-like encoder API
│   │   ├── webgpu-pipeline.ts      # WebGPU compute pipeline
│   │   ├── shader-loader.ts        # WGSL shader management
│   │   ├── buffer-pool.ts          # Memory pooling for perf
│   │   └── types.ts                # TypeScript type definitions
│   │
│   ├── package.json
│   ├── tsconfig.json
│   └── rollup.config.js            # Bundle for distribution
│
├── demo/                           # Demo application
│   ├── index.html
│   ├── demo.ts                     # Demo app code
│   ├── timeline.ts                 # Timeline scrubbing demo
│   └── benchmark.ts                # Performance benchmark
│
└── docs/
    ├── ARCHITECTURE.md             # Detailed architecture docs
    ├── PERFORMANCE.md              # Benchmark results
    └── API.md                      # API documentation
```

### Cargo.toml (rust/)

```toml
[package]
name = "prores-webcodec"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your@email.com>"]
description = "ProRes codec for WebAssembly with WebGPU acceleration"
license = "MIT OR Apache-2.0"
repository = "https://github.com/yourname/prores-webcodec"
keywords = ["prores", "video", "codec", "wasm", "webgpu"]
categories = ["multimedia::video", "wasm"]

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
criterion = "0.5"
wasm-bindgen-test = "0.3"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1

[features]
default = ["simd"]
simd = []  # Enable SIMD optimizations for entropy coding

[[bench]]
name = "entropy_bench"
harness = false
```

### package.json (ts/)

```json
{
  "name": "@prores/webcodec",
  "version": "0.1.0",
  "description": "ProRes codec for the web with WebGPU acceleration",
  "main": "dist/index.js",
  "module": "dist/index.esm.js",
  "types": "dist/index.d.ts",
  "files": [
    "dist",
    "wasm"
  ],
  "scripts": {
    "build:wasm": "cd ../rust && wasm-pack build --target web --release",
    "build:ts": "rollup -c",
    "build": "npm run build:wasm && npm run build:ts",
    "test": "vitest",
    "benchmark": "vite demo/benchmark.html"
  },
  "dependencies": {},
  "devDependencies": {
    "@rollup/plugin-typescript": "^11.0.0",
    "rollup": "^4.0.0",
    "typescript": "^5.0.0",
    "vitest": "^1.0.0",
    "vite": "^5.0.0"
  },
  "peerDependencies": {},
  "keywords": [
    "prores",
    "webcodecs",
    "webgpu",
    "video",
    "codec"
  ],
  "license": "MIT"
}
```

### Build Commands

```bash
# Install dependencies
cd ts && npm install

# Build WASM module with SIMD
cd rust
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --release

# Build TypeScript
cd ts && npm run build

# Run tests
npm test

# Run benchmarks
npm run benchmark

# Full build
npm run build
```

---

## Implementation Roadmap

### Phase 1: Foundation

- [ ] Project setup (Cargo, wasm-bindgen, WebGPU TypeScript)
- [ ] BitWriter and BitReader implementation
- [ ] MOV container parser (demuxer)
- [ ] Reference DCT implementation (CPU, for testing)
- [ ] Unit tests for each component

### Phase 2: Decoder - WASM Entropy Decode

- [ ] ProRes frame header parsing
- [ ] Rice/Exp-Golomb entropy decoding
- [ ] Slice decoder → quantized coefficient output
- [ ] Coefficient buffer management
- [ ] Test against FFmpeg-decoded reference

### Phase 3: Decoder - WebGPU Pipeline

- [ ] WebGPU device initialization and shader compilation
- [ ] Dequantization compute shader
- [ ] Inverse DCT compute shader (8x8 blocks)
- [ ] YCbCr → RGB color conversion shader
- [ ] GPU buffer ↔ WASM memory transfer
- [ ] VideoFrame output integration

### Phase 4: Decoder Integration & Benchmark

- [ ] JavaScript bridge layer (WebCodecs-like API)
- [ ] Integration with masterselects
- [ ] Timeline scrubbing performance testing
- [ ] Target: 200+ fps decode at 4K
- [ ] Memory optimization and pooling

### Phase 5: Encoder - WebGPU Pipeline

- [ ] RGB → YCbCr color conversion shader
- [ ] Forward DCT compute shader
- [ ] Quantization compute shader
- [ ] GPU → WASM coefficient readback

### Phase 6: Encoder - WASM Entropy Encode

- [ ] Rice/Exp-Golomb entropy encoding
- [ ] Slice encoder (coefficient → bitstream)
- [ ] Frame encoder with headers
- [ ] MOV muxer (single video track)

### Phase 7: Polish & Release

- [ ] Multiple profile support (Proxy, LT, Standard, HQ)
- [ ] WASM SIMD optimization for entropy coding
- [ ] Error handling and edge cases
- [ ] TypeScript types and documentation
- [ ] npm package publishing
- [ ] Demo page and benchmarks

### Phase 8: Advanced Features (Future)

- [ ] Alpha channel support (4444 profiles)
- [ ] Interlaced video support
- [ ] Custom quantization matrices
- [ ] Web Worker parallelization for slices
- [ ] Streaming decode (for large files)
- [ ] Safari native ProRes detection (use if available)

---

## References

### Official Specifications

- [Apple ProRes White Paper (2022)](https://www.apple.com/final-cut-pro/docs/Apple_ProRes.pdf) - Official Apple documentation
- [SMPTE RDD 36:2022](https://pub.smpte.org/doc/rdd36/20220909-pub/rdd36-2022.pdf) - Bitstream syntax specification
- [SMPTE RDD 44](https://ieeexplore.ieee.org/document/8187792) - MXF mapping for ProRes

### Technical Resources

- [MultimediaWiki - Apple ProRes](https://wiki.multimedia.cx/index.php/Apple_ProRes) - Community reverse-engineering documentation
- [Academy Software Foundation - ProRes Encoding](https://academysoftwarefoundation.github.io/EncodingGuidelines/EncodeProres.html) - Best practices
- [Library of Congress - ProRes Format](https://www.loc.gov/preservation/digital/formats/fdd/fdd000389.shtml) - Format description

### Reference Implementations

- [FFmpeg prores_ks](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/proresenc_kostya.c) - Open source encoder
- [rav1e](https://github.com/xiph/rav1e) - Rust video encoder (architecture reference)
- [rust-ffmpeg-wasi](https://github.com/jedisct1/rust-ffmpeg-wasi) - FFmpeg WASM bindings

### WASM Resources

- [Rust and WebAssembly Book](https://rustwasm.github.io/book/) - Official guide
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) - Rust/JS interop

### WebGPU Resources

- [WebGPU Specification](https://www.w3.org/TR/webgpu/) - W3C specification
- [WGSL Specification](https://www.w3.org/TR/WGSL/) - WebGPU Shading Language
- [WebGPU Fundamentals](https://webgpufundamentals.org/) - Practical tutorials
- [WebGPU Samples](https://webgpu.github.io/webgpu-samples/) - Official samples
- [wgpu-rs](https://wgpu.rs/) - Rust WebGPU implementation

### Video Codec Resources

- [DCT Implementation Guide](https://unix4lyfe.org/dct/) - DCT math explained
- [WebCodecs API](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API) - MDN documentation
- [VideoFrame API](https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame) - For output integration

---

## Community Impact

### Who Benefits?

| User Group | Benefit |
|------------|---------|
| Browser-based video editors | Professional export without conversion |
| Web developers | Native video encoding capability |
| Open source community | First pure-Rust ProRes encoder |
| Education | Reference implementation for learning |

### Potential Adoption

- Kapwing, Canva Video, Clipchamp (browser editors)
- Custom video tools built with WebCodecs
- Electron-based video applications
- Educational projects studying video codecs

### Contributing

This would be a significant open source contribution:
- Fill a real gap in the web video ecosystem
- Likely featured on Hacker News, Reddit
- Potential for industry adoption
- Foundation for additional codecs (DNxHD, etc.)

---

## Conclusion

Building a browser-native ProRes codec with GPU acceleration is:
- **Legal**: Apple published the spec, precedent exists (FFmpeg, etc.)
- **Achievable**: Hybrid WASM + WebGPU architecture is well-understood
- **Valuable**: No current solution exists for ProRes in the browser
- **Performant**: GPU acceleration enables real-time timeline scrubbing
- **Impactful**: Enables professional video workflows in web apps like masterselects

### Why This Approach Works

```
┌────────────────────────────────────────────────────────────────┐
│  The Key Insight                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Native hardware codecs use the SAME architecture:            │
│  - CPU/fixed-function for serial entropy coding               │
│  - GPU shaders for parallel DCT and color conversion          │
│                                                                │
│  We replicate this split with WASM + WebGPU.                  │
│  The only overhead is WASM entropy decode vs fixed-function.  │
│  That's ~10-20% of total decode time.                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Main Challenges

1. **Entropy coding precision** - Must match ProRes spec exactly for conformance
2. **WASM ↔ GPU transfer** - Efficient coefficient handoff is critical
3. **WebGPU compute shaders** - IDCT implementation must be numerically accurate
4. **Conformance testing** - Verify output with Final Cut Pro, DaVinci Resolve

### Performance Targets

| Resolution | Target Decode | Target Encode | Use Case |
|------------|---------------|---------------|----------|
| 1080p | ~3ms (300+ fps) | ~10ms (100 fps) | Real-time scrubbing |
| 4K | ~5ms (200+ fps) | ~30ms (30 fps) | Smooth scrubbing |
| 8K | ~15ms (60+ fps) | ~100ms (10 fps) | Acceptable |

With the hybrid WASM + WebGPU approach, a working MVP with real-time 4K scrubbing is achievable.
