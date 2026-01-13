# MasterSelects Native Helper

A lightweight, standalone Rust application that provides hardware-accelerated video codec support for the MasterSelects web application.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Protocol Design](#protocol-design)
4. [Frame Transfer Optimization](#frame-transfer-optimization)
5. [Caching Strategy](#caching-strategy)
6. [FFmpeg Integration](#ffmpeg-integration)
7. [Linux Build & Distribution](#linux-build--distribution)
8. [Security](#security)
9. [Implementation](#implementation)
10. [Project Structure](#project-structure)
11. [Build & Release](#build--release)
12. [Future: macOS & Windows](#future-macos--windows)

---

## Overview

### What is This?

A single-file executable (~15-20MB) that runs locally and provides:
- **ProRes decode/encode** at native speed
- **Hardware acceleration** where available
- **Direct file system access** for large video files
- **WebSocket API** for browser communication

### Why Native?

| Aspect | Browser-Only (WASM+WebGPU) | Native Helper |
|--------|---------------------------|---------------|
| ProRes 4K decode | ~5ms/frame | ~0.5ms/frame |
| Build time | 2-4 months | 1-2 weeks |
| All codecs (DNxHD, etc.) | Build each | FFmpeg has all |
| Hardware accel | WebGPU only | Full (VAAPI, NVDEC) |
| File access | Upload required | Direct |
| Memory limit | ~4GB | System RAM |

### User Experience

```
1. User clicks "Enable Turbo Mode" in masterselects.app
2. Downloads: masterselects-helper (single file, ~15MB)
3. Makes executable: chmod +x masterselects-helper
4. Runs it: ./masterselects-helper
5. Browser detects helper → 10x faster scrubbing
```

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BROWSER                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  masterselects.app                                               │   │
│  │                                                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │   │
│  │  │   Timeline   │  │   Preview    │  │   Export Manager     │  │   │
│  │  │   Component  │  │   Canvas     │  │                      │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │   │
│  │         │                 │                     │              │   │
│  │         └────────────┬────┴─────────────────────┘              │   │
│  │                      │                                          │   │
│  │              ┌───────▼────────┐                                 │   │
│  │              │  HelperClient  │                                 │   │
│  │              │  (TypeScript)  │                                 │   │
│  │              └───────┬────────┘                                 │   │
│  └──────────────────────┼──────────────────────────────────────────┘   │
│                         │                                               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          │ WebSocket (ws://127.0.0.1:9876)
                          │ Binary frames + JSON commands
                          │
┌─────────────────────────┼───────────────────────────────────────────────┐
│                         │           NATIVE HELPER                       │
│                         │                                               │
│              ┌──────────▼───────────┐                                   │
│              │   WebSocket Server   │                                   │
│              │   (tokio-tungstenite)│                                   │
│              └──────────┬───────────┘                                   │
│                         │                                               │
│    ┌────────────────────┼────────────────────┐                         │
│    │                    │                    │                         │
│    ▼                    ▼                    ▼                         │
│ ┌──────────┐     ┌─────────────┐     ┌─────────────┐                  │
│ │ Decoder  │     │   Encoder   │     │   File      │                  │
│ │ Pool     │     │   Queue     │     │   Manager   │                  │
│ └────┬─────┘     └──────┬──────┘     └──────┬──────┘                  │
│      │                  │                   │                          │
│      │                  │                   │                          │
│      ▼                  ▼                   ▼                          │
│ ┌──────────────────────────────────────────────────────────────────┐  │
│ │                         FFmpeg                                    │  │
│ │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │  │
│ │  │ ProRes  │  │ DNxHD   │  │ H.264   │  │ Hardware Accel      │ │  │
│ │  │         │  │         │  │ H.265   │  │ VAAPI/NVDEC/QSV     │ │  │
│ │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │  │
│ └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│ ┌──────────────────────────────────────────────────────────────────┐  │
│ │                      Frame Cache (LRU)                            │  │
│ │  [frame 42] [frame 43] [frame 44] [frame 45] [frame 46] ...      │  │
│ └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Local File System   │
              │                       │
              │  /home/user/Videos/   │
              │  ├── project.mov      │
              │  ├── footage/         │
              │  └── exports/         │
              └───────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| WebSocket Server | Accept connections, route messages, manage sessions |
| Decoder Pool | Maintain open decoder contexts, parallel decode |
| Encoder Queue | Queue encode jobs, progress reporting |
| File Manager | File access, validation, metadata extraction |
| Frame Cache | LRU cache of decoded frames for scrubbing |
| FFmpeg | All codec operations |

---

## Protocol Design

### Message Format

All messages use a binary header + payload format for efficiency:

```
┌─────────────────────────────────────────────────────────────────┐
│  MESSAGE STRUCTURE                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┬──────────┬──────────┬─────────────────────────┐  │
│  │  Magic   │  Type    │  Length  │  Payload                │  │
│  │  2 bytes │  1 byte  │  4 bytes │  variable               │  │
│  │  "MH"    │          │  u32 LE  │                         │  │
│  └──────────┴──────────┴──────────┴─────────────────────────┘  │
│                                                                 │
│  Type values:                                                   │
│  0x01 = Command (JSON payload)                                  │
│  0x02 = Frame (binary RGBA payload)                            │
│  0x03 = Response (JSON payload)                                 │
│  0x04 = Error (JSON payload)                                    │
│  0x05 = Progress (JSON payload)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Commands

#### Open File
```json
{
  "cmd": "open",
  "id": "req_001",
  "path": "/home/user/Videos/project.mov"
}
```

Response:
```json
{
  "id": "req_001",
  "ok": true,
  "file_id": "file_abc123",
  "metadata": {
    "width": 3840,
    "height": 2160,
    "fps": 24.0,
    "duration_ms": 120000,
    "frame_count": 2880,
    "codec": "prores",
    "profile": "422 HQ",
    "color_space": "bt709",
    "audio_tracks": 2
  }
}
```

#### Decode Frame
```json
{
  "cmd": "decode",
  "id": "req_002",
  "file_id": "file_abc123",
  "frame": 1337,
  "format": "rgba8",
  "scale": 1.0
}
```

Response: Binary frame message (Type 0x02)
```
┌──────────┬──────────┬──────────┬──────────────────────────────┐
│  "MH"    │  0x02    │  length  │  RGBA pixels                 │
│          │          │          │  (width × height × 4 bytes)  │
└──────────┴──────────┴──────────┴──────────────────────────────┘
```

#### Decode Range (for smooth scrubbing)
```json
{
  "cmd": "decode_range",
  "id": "req_003",
  "file_id": "file_abc123",
  "start_frame": 100,
  "end_frame": 110,
  "priority": "high"
}
```

Streams multiple frame responses.

#### Prefetch (background loading)
```json
{
  "cmd": "prefetch",
  "id": "req_004",
  "file_id": "file_abc123",
  "around_frame": 500,
  "radius": 50
}
```

No response - frames cached silently for later.

#### Encode
```json
{
  "cmd": "encode",
  "id": "req_005",
  "input": {
    "type": "frames",
    "frame_ids": ["frame_001", "frame_002", "..."]
  },
  "output": {
    "path": "/home/user/Videos/export.mov",
    "codec": "prores",
    "profile": "422_hq",
    "width": 1920,
    "height": 1080,
    "fps": 24.0
  }
}
```

Progress updates:
```json
{
  "id": "req_005",
  "progress": 0.45,
  "frames_done": 1080,
  "frames_total": 2400,
  "eta_ms": 15000
}
```

#### Close File
```json
{
  "cmd": "close",
  "id": "req_006",
  "file_id": "file_abc123"
}
```

### Error Handling

```json
{
  "id": "req_002",
  "ok": false,
  "error": {
    "code": "FILE_NOT_FOUND",
    "message": "File does not exist: /path/to/file.mov"
  }
}
```

Error codes:
- `FILE_NOT_FOUND` - Path doesn't exist
- `PERMISSION_DENIED` - Can't read/write file
- `UNSUPPORTED_CODEC` - Codec not available
- `DECODE_ERROR` - FFmpeg decode failure
- `ENCODE_ERROR` - FFmpeg encode failure
- `OUT_OF_MEMORY` - Cache/decode memory exhausted
- `INVALID_FRAME` - Frame number out of range

---

## Frame Transfer Optimization

### The Bottleneck

For 4K RGBA frames: `3840 × 2160 × 4 = 33.2 MB per frame`

At 30 fps scrubbing: `~1 GB/second` of data transfer

Even on localhost, this matters.

### Optimization Strategies

#### 1. Scaled Previews for Scrubbing

During fast scrubbing, send half or quarter resolution:

```json
{
  "cmd": "decode",
  "file_id": "file_abc123",
  "frame": 1337,
  "scale": 0.5
}
```

| Scale | 4K Frame Size | Throughput |
|-------|---------------|------------|
| 1.0 | 33.2 MB | ~30 fps |
| 0.5 | 8.3 MB | ~120 fps |
| 0.25 | 2.1 MB | ~400 fps |

Switch to full resolution when scrubbing stops.

#### 2. Compression for Frames

Optional LZ4 compression (very fast):

```json
{
  "cmd": "decode",
  "compression": "lz4"
}
```

Typical compression ratio for video frames: 2-4x

| Format | 4K Size | Compress Time | Decompress Time |
|--------|---------|---------------|-----------------|
| Raw RGBA | 33.2 MB | - | - |
| LZ4 | ~10 MB | ~2ms | ~1ms |

#### 3. Delta Frames

For sequential playback, send only changed pixels:

```
Frame N:     [full frame]
Frame N+1:   [delta: 5% changed]
Frame N+2:   [delta: 3% changed]
...
Frame N+30:  [keyframe: full]
```

#### 4. GPU Texture Format (Future)

If browser supports WebGPU `importExternalTexture`:
- Send compressed GPU texture directly
- Zero CPU-side pixel copying
- Requires shared memory or GPU buffer export

### Frame Message Format

```
┌────────────────────────────────────────────────────────────────────────┐
│  FRAME MESSAGE                                                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Header (16 bytes):                                                    │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │  Magic   │  Type    │  Flags   │  Width   │  Height  │  Frame#  │  │
│  │  2B      │  1B      │  1B      │  2B      │  2B      │  4B      │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘  │
│                                                                        │
│  Flags:                                                                │
│  bit 0: compressed (0=raw, 1=lz4)                                      │
│  bit 1: scaled (0=full, 1=scaled)                                      │
│  bit 2: delta (0=full, 1=delta from previous)                          │
│  bit 3-7: reserved                                                     │
│                                                                        │
│  Payload:                                                              │
│  [width × height × 4 bytes] (or compressed)                           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Caching Strategy

### LRU Frame Cache

```rust
struct FrameCache {
    /// Max memory usage (e.g., 2GB)
    max_bytes: usize,

    /// Current memory usage
    current_bytes: usize,

    /// Cached frames: (file_id, frame_num) -> Frame
    frames: LinkedHashMap<(String, u32), CachedFrame>,
}

struct CachedFrame {
    data: Vec<u8>,
    width: u32,
    height: u32,
    timestamp: Instant,
}
```

### Cache Behavior

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CACHE STRATEGY                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Playhead at frame 100:                                                 │
│                                                                         │
│  Priority 1 (keep in cache):                                           │
│  [95] [96] [97] [98] [99] [100] [101] [102] [103] [104] [105]         │
│                            ▲                                            │
│                         playhead                                        │
│                                                                         │
│  Priority 2 (prefetch):                                                 │
│  [85-94] and [106-115]                                                 │
│                                                                         │
│  Priority 3 (keep if space):                                           │
│  Recently accessed frames elsewhere in timeline                        │
│                                                                         │
│  Evict first:                                                          │
│  Oldest accessed frames far from playhead                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Prefetch Logic

```rust
async fn prefetch_around(file_id: &str, frame: u32, radius: u32) {
    let start = frame.saturating_sub(radius);
    let end = frame + radius;

    // Decode frames in order of distance from playhead
    let mut frames: Vec<u32> = (start..=end).collect();
    frames.sort_by_key(|f| ((*f as i64) - (frame as i64)).abs());

    for f in frames {
        if !cache.contains(file_id, f) {
            let decoded = decode_frame(file_id, f).await;
            cache.insert(file_id, f, decoded);
        }
    }
}
```

### Memory Management

```rust
impl FrameCache {
    fn insert(&mut self, file_id: &str, frame: u32, data: CachedFrame) {
        let frame_size = data.data.len();

        // Evict until we have space
        while self.current_bytes + frame_size > self.max_bytes {
            if let Some((key, evicted)) = self.frames.pop_front() {
                self.current_bytes -= evicted.data.len();
            } else {
                break; // Cache empty
            }
        }

        self.frames.insert((file_id.to_string(), frame), data);
        self.current_bytes += frame_size;
    }
}
```

---

## FFmpeg Integration

### Rust FFmpeg Bindings

Using `ffmpeg-next` crate (safe Rust wrapper):

```toml
[dependencies]
ffmpeg-next = "7"
```

### Decoder Context Pool

Maintain open decoder contexts to avoid re-initialization:

```rust
struct DecoderPool {
    decoders: HashMap<String, OpenDecoder>,
    max_decoders: usize,
}

struct OpenDecoder {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    stream_index: usize,
    last_used: Instant,

    // Seek state
    last_frame: u32,
}

impl DecoderPool {
    fn get_or_open(&mut self, path: &str) -> Result<&mut OpenDecoder> {
        if !self.decoders.contains_key(path) {
            // Evict oldest if at capacity
            if self.decoders.len() >= self.max_decoders {
                self.evict_oldest();
            }

            let decoder = Self::open_decoder(path)?;
            self.decoders.insert(path.to_string(), decoder);
        }

        let decoder = self.decoders.get_mut(path).unwrap();
        decoder.last_used = Instant::now();
        Ok(decoder)
    }

    fn open_decoder(path: &str) -> Result<OpenDecoder> {
        let input = ffmpeg::format::input(path)?;
        let stream = input.streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("No video stream")?;

        let stream_index = stream.index();
        let context = ffmpeg::codec::context::Context::from_parameters(
            stream.parameters()
        )?;

        let decoder = context.decoder().video()?;

        Ok(OpenDecoder {
            input_ctx: input,
            decoder,
            stream_index,
            last_used: Instant::now(),
            last_frame: 0,
        })
    }
}
```

### Efficient Seeking

```rust
impl OpenDecoder {
    fn decode_frame(&mut self, target_frame: u32) -> Result<Frame> {
        let stream = self.input_ctx.stream(self.stream_index).unwrap();
        let time_base = stream.time_base();
        let fps = stream.avg_frame_rate();

        // Calculate timestamp
        let target_ts = (target_frame as i64 * time_base.1 as i64)
                      / (fps.0 as i64 * time_base.0 as i64);

        // Only seek if jumping more than a few frames
        let distance = (target_frame as i64 - self.last_frame as i64).abs();
        if distance > 10 {
            self.input_ctx.seek(target_ts, ..target_ts)?;
            self.decoder.flush();
        }

        // Decode until we hit target frame
        let mut current_frame = 0;
        for (stream, packet) in self.input_ctx.packets() {
            if stream.index() != self.stream_index {
                continue;
            }

            self.decoder.send_packet(&packet)?;

            let mut frame = ffmpeg::frame::Video::empty();
            while self.decoder.receive_frame(&mut frame).is_ok() {
                if current_frame == target_frame {
                    self.last_frame = target_frame;
                    return Ok(self.frame_to_rgba(&frame));
                }
                current_frame += 1;
            }
        }

        Err("Frame not found".into())
    }

    fn frame_to_rgba(&self, frame: &ffmpeg::frame::Video) -> Frame {
        let mut scaler = ffmpeg::software::scaling::Context::get(
            frame.format(),
            frame.width(),
            frame.height(),
            ffmpeg::format::Pixel::RGBA,
            frame.width(),
            frame.height(),
            ffmpeg::software::scaling::Flags::BILINEAR,
        ).unwrap();

        let mut rgb_frame = ffmpeg::frame::Video::empty();
        scaler.run(frame, &mut rgb_frame).unwrap();

        Frame {
            width: rgb_frame.width(),
            height: rgb_frame.height(),
            data: rgb_frame.data(0).to_vec(),
        }
    }
}
```

### Hardware Acceleration (Linux)

```rust
fn create_hw_decoder(codec: &ffmpeg::codec::codec::Codec) -> Option<ffmpeg::decoder::Video> {
    // Try VAAPI first (Intel/AMD)
    if let Ok(hw_config) = codec.hw_config(ffmpeg::codec::HWConfigMethod::HW_DEVICE_CTX) {
        if hw_config.device_type() == ffmpeg::hwaccel::DeviceType::VAAPI {
            // Create VAAPI context
            if let Ok(hw_ctx) = ffmpeg::hwaccel::Context::new(
                ffmpeg::hwaccel::DeviceType::VAAPI,
                "/dev/dri/renderD128",
            ) {
                // Configure decoder with HW context
                // ...
                return Some(decoder);
            }
        }
    }

    // Try NVDEC (NVIDIA)
    // ...

    // Fall back to software
    None
}
```

---

## Linux Build & Distribution

### Static Binary

Goal: Single executable that runs on any Linux (glibc 2.17+, ~2014)

```toml
# .cargo/config.toml

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+crt-static"]
```

### FFmpeg Static Linking

Build FFmpeg with static libs:

```bash
# Build static FFmpeg
./configure \
  --enable-static \
  --disable-shared \
  --enable-gpl \
  --enable-libx264 \
  --enable-libx265 \
  --disable-programs \
  --disable-doc

make -j$(nproc)
```

Link in Cargo:

```toml
[build]
rustflags = ["-L", "/path/to/ffmpeg/lib"]

[dependencies]
ffmpeg-sys-next = { version = "7", features = ["static"] }
```

### Distribution Formats

#### 1. Raw Binary (Simplest)

```
masterselects-helper  (single file, ~20MB)
```

Usage:
```bash
chmod +x masterselects-helper
./masterselects-helper
```

#### 2. AppImage (Better UX)

Self-contained, double-click to run:

```bash
# Create AppImage structure
mkdir -p AppDir/usr/bin
cp target/release/masterselects-helper AppDir/usr/bin/

# Create .desktop file
cat > AppDir/masterselects-helper.desktop << EOF
[Desktop Entry]
Name=MasterSelects Helper
Exec=masterselects-helper
Icon=masterselects
Type=Application
Categories=AudioVideo;
EOF

# Build AppImage
appimagetool AppDir masterselects-helper.AppImage
```

#### 3. Flatpak (For App Stores)

```yaml
# com.masterselects.Helper.yml
app-id: com.masterselects.Helper
runtime: org.freedesktop.Platform
runtime-version: '23.08'
sdk: org.freedesktop.Sdk
command: masterselects-helper

modules:
  - name: masterselects-helper
    buildsystem: simple
    build-commands:
      - install -D masterselects-helper /app/bin/masterselects-helper
    sources:
      - type: file
        path: masterselects-helper
```

### System Integration

#### Auto-start (User Service)

```ini
# ~/.config/systemd/user/masterselects-helper.service

[Unit]
Description=MasterSelects Helper
After=network.target

[Service]
ExecStart=/home/%u/.local/bin/masterselects-helper --background
Restart=on-failure

[Install]
WantedBy=default.target
```

Enable:
```bash
systemctl --user enable masterselects-helper
systemctl --user start masterselects-helper
```

#### Desktop Integration

Create `.desktop` file for app menu:

```ini
# ~/.local/share/applications/masterselects-helper.desktop

[Desktop Entry]
Name=MasterSelects Helper
Comment=Video codec helper for masterselects.app
Exec=/home/user/.local/bin/masterselects-helper
Icon=masterselects-helper
Terminal=false
Type=Application
Categories=AudioVideo;Video;
StartupNotify=true
```

---

## Security

### Localhost Only

```rust
// CRITICAL: Only bind to localhost
let listener = TcpListener::bind("127.0.0.1:9876").await?;

// Never bind to 0.0.0.0!
```

### Origin Validation

```rust
async fn handle_websocket_upgrade(req: Request) -> Result<Response> {
    // Check Origin header
    let origin = req.headers().get("Origin")
        .and_then(|h| h.to_str().ok());

    match origin {
        Some("https://masterselects.app") |
        Some("https://app.masterselects.com") |
        Some("http://localhost:3000") |
        Some("http://localhost:5173") => {
            // Allowed
        }
        _ => {
            return Err(Error::Forbidden("Invalid origin"));
        }
    }

    // Proceed with upgrade
}
```

### Token Authentication

On first launch, generate a random token:

```rust
fn generate_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
        .collect()
}

fn save_token(token: &str) -> Result<()> {
    let config_dir = dirs::config_dir()
        .unwrap()
        .join("masterselects-helper");
    fs::create_dir_all(&config_dir)?;
    fs::write(config_dir.join("token"), token)?;
    Ok(())
}
```

Browser must send token:
```json
{
  "cmd": "auth",
  "token": "abc123..."
}
```

### Path Validation

```rust
fn validate_path(path: &str) -> Result<PathBuf> {
    let path = PathBuf::from(path);

    // Must be absolute
    if !path.is_absolute() {
        return Err(Error::InvalidPath("Path must be absolute"));
    }

    // Canonicalize to resolve symlinks and ..
    let canonical = path.canonicalize()?;

    // Check against allowed directories (optional)
    // ...

    // Check file exists and is readable
    if !canonical.exists() {
        return Err(Error::NotFound);
    }

    Ok(canonical)
}
```

### No Arbitrary Code Execution

- Never execute shell commands from browser input
- Never `eval()` or similar
- Only predefined operations (decode, encode, etc.)

---

## Implementation

### Main Entry Point

```rust
// src/main.rs

use clap::Parser;
use tokio::net::TcpListener;
use tracing::{info, error};

#[derive(Parser)]
#[command(name = "masterselects-helper")]
#[command(about = "Video codec helper for MasterSelects")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "9876")]
    port: u16,

    /// Run in background (no terminal output)
    #[arg(long)]
    background: bool,

    /// Maximum cache size in MB
    #[arg(long, default_value = "2048")]
    cache_mb: usize,

    /// Open browser on start
    #[arg(long)]
    open_browser: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize logging
    if !args.background {
        tracing_subscriber::fmt::init();
    }

    // Initialize FFmpeg
    ffmpeg_next::init()?;

    // Create shared state
    let state = AppState::new(args.cache_mb);

    // Start server
    let addr = format!("127.0.0.1:{}", args.port);
    let listener = TcpListener::bind(&addr).await?;

    info!("MasterSelects Helper running on ws://{}", addr);

    // Open browser if requested
    if args.open_browser {
        open::that("https://masterselects.app")?;
    }

    // Accept connections
    while let Ok((stream, addr)) = listener.accept().await {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, state).await {
                error!("Connection error from {}: {}", addr, e);
            }
        });
    }

    Ok(())
}
```

### Connection Handler

```rust
// src/connection.rs

use tokio_tungstenite::accept_async;
use futures::{StreamExt, SinkExt};

pub async fn handle_connection(
    stream: TcpStream,
    state: AppState,
) -> Result<()> {
    let ws = accept_async(stream).await?;
    let (mut write, mut read) = ws.split();

    // Session state
    let mut session = Session::new(state);
    let mut authenticated = false;

    while let Some(msg) = read.next().await {
        let msg = msg?;

        match msg {
            Message::Binary(data) => {
                let response = session.handle_binary(&data).await?;
                write.send(Message::Binary(response)).await?;
            }
            Message::Text(text) => {
                let cmd: Command = serde_json::from_str(&text)?;

                // Auth required for most commands
                if !authenticated && !matches!(cmd, Command::Auth { .. }) {
                    write.send(error_response("AUTH_REQUIRED")).await?;
                    continue;
                }

                match cmd {
                    Command::Auth { token } => {
                        if session.validate_token(&token) {
                            authenticated = true;
                            write.send(ok_response("auth", json!({}))).await?;
                        } else {
                            write.send(error_response("INVALID_TOKEN")).await?;
                        }
                    }
                    Command::Open { id, path } => {
                        let result = session.open_file(&path).await;
                        write.send(response(&id, result)).await?;
                    }
                    Command::Decode { id, file_id, frame, scale, compression } => {
                        let result = session.decode_frame(&file_id, frame, scale).await;
                        match result {
                            Ok(frame_data) => {
                                let msg = encode_frame_message(&frame_data, compression);
                                write.send(Message::Binary(msg)).await?;
                            }
                            Err(e) => {
                                write.send(error_response(&e.to_string())).await?;
                            }
                        }
                    }
                    Command::Prefetch { file_id, around_frame, radius } => {
                        // Fire and forget - no response
                        session.prefetch(&file_id, around_frame, radius);
                    }
                    Command::Encode { id, input, output } => {
                        // Start encode job, stream progress
                        let progress_tx = write.clone();
                        session.start_encode(id, input, output, progress_tx).await;
                    }
                    Command::Close { id, file_id } => {
                        session.close_file(&file_id);
                        write.send(ok_response(&id, json!({}))).await?;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    Ok(())
}
```

### Frame Encoding

```rust
// src/frame.rs

pub fn encode_frame_message(
    frame: &DecodedFrame,
    compression: Option<Compression>,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(16 + frame.data.len());

    // Magic
    msg.extend_from_slice(b"MH");

    // Type: Frame
    msg.push(0x02);

    // Flags
    let mut flags = 0u8;
    let payload = match compression {
        Some(Compression::Lz4) => {
            flags |= 0x01;
            lz4_flex::compress_prepend_size(&frame.data)
        }
        None => frame.data.clone(),
    };
    msg.push(flags);

    // Dimensions
    msg.extend_from_slice(&(frame.width as u16).to_le_bytes());
    msg.extend_from_slice(&(frame.height as u16).to_le_bytes());

    // Frame number
    msg.extend_from_slice(&frame.frame_num.to_le_bytes());

    // Reserved (pad to 16 bytes header)
    msg.extend_from_slice(&[0u8; 4]);

    // Payload
    msg.extend_from_slice(&payload);

    msg
}
```

---

## Project Structure

```
masterselects-helper/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── LICENSE
│
├── src/
│   ├── main.rs                 # Entry point, CLI args
│   ├── server.rs               # WebSocket server
│   ├── connection.rs           # Connection handler
│   ├── session.rs              # Per-connection state
│   │
│   ├── protocol/
│   │   ├── mod.rs
│   │   ├── commands.rs         # Command types
│   │   ├── responses.rs        # Response types
│   │   └── frame.rs            # Frame message encoding
│   │
│   ├── decoder/
│   │   ├── mod.rs
│   │   ├── pool.rs             # Decoder context pool
│   │   ├── ffmpeg.rs           # FFmpeg integration
│   │   └── hwaccel.rs          # Hardware acceleration
│   │
│   ├── encoder/
│   │   ├── mod.rs
│   │   ├── job.rs              # Encode job management
│   │   └── prores.rs           # ProRes-specific settings
│   │
│   ├── cache/
│   │   ├── mod.rs
│   │   ├── lru.rs              # LRU cache implementation
│   │   └── prefetch.rs         # Prefetch logic
│   │
│   └── util/
│       ├── mod.rs
│       ├── path.rs             # Path validation
│       └── token.rs            # Auth token management
│
├── build.rs                    # FFmpeg linking
│
├── scripts/
│   ├── build-linux.sh          # Linux build script
│   ├── build-appimage.sh       # AppImage packaging
│   └── build-static-ffmpeg.sh  # Static FFmpeg build
│
└── .github/
    └── workflows/
        └── release.yml         # CI/CD for releases
```

### Cargo.toml

```toml
[package]
name = "masterselects-helper"
version = "0.1.0"
edition = "2021"
authors = ["MasterSelects Team"]
description = "Video codec helper for MasterSelects web application"
license = "MIT"
repository = "https://github.com/masterselects/helper"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# WebSocket
tokio-tungstenite = "0.21"
futures = "0.3"

# FFmpeg
ffmpeg-next = "7"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# CLI
clap = { version = "4", features = ["derive"] }

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Utilities
anyhow = "1"
thiserror = "1"
lz4_flex = "0.11"
linked-hash-map = "0.5"
dirs = "5"
open = "5"
rand = "0.8"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true

[profile.release.package."*"]
opt-level = 3
```

---

## Build & Release

### Build Script

```bash
#!/bin/bash
# scripts/build-linux.sh

set -e

echo "Building MasterSelects Helper for Linux..."

# Ensure static FFmpeg is available
if [ ! -f "/opt/ffmpeg-static/lib/libavcodec.a" ]; then
    echo "Static FFmpeg not found. Building..."
    ./scripts/build-static-ffmpeg.sh
fi

# Set environment for static linking
export FFMPEG_DIR="/opt/ffmpeg-static"
export PKG_CONFIG_PATH="$FFMPEG_DIR/lib/pkgconfig"

# Build
cargo build --release

# Strip binary
strip target/release/masterselects-helper

# Show size
ls -lh target/release/masterselects-helper

echo "Done! Binary: target/release/masterselects-helper"
```

### Static FFmpeg Build

```bash
#!/bin/bash
# scripts/build-static-ffmpeg.sh

set -e

FFMPEG_VERSION="6.1"
PREFIX="/opt/ffmpeg-static"

# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    nasm \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libopus-dev

# Download FFmpeg
wget https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz
tar xf ffmpeg-${FFMPEG_VERSION}.tar.xz
cd ffmpeg-${FFMPEG_VERSION}

# Configure for static build
./configure \
    --prefix=$PREFIX \
    --enable-static \
    --disable-shared \
    --enable-gpl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libopus \
    --disable-programs \
    --disable-doc \
    --disable-debug

# Build and install
make -j$(nproc)
sudo make install

echo "Static FFmpeg installed to $PREFIX"
```

### GitHub Actions Release

```yaml
# .github/workflows/release.yml

name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-linux:
    runs-on: ubuntu-22.04

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            build-essential nasm \
            libx264-dev libx265-dev

      - name: Cache FFmpeg
        uses: actions/cache@v4
        with:
          path: /opt/ffmpeg-static
          key: ffmpeg-static-6.1

      - name: Build static FFmpeg
        run: ./scripts/build-static-ffmpeg.sh

      - name: Setup Rust
        uses: dtolnay/rust-action@stable

      - name: Build
        run: ./scripts/build-linux.sh

      - name: Create AppImage
        run: ./scripts/build-appimage.sh

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: linux-build
          path: |
            target/release/masterselects-helper
            MasterSelects-Helper.AppImage

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            target/release/masterselects-helper
            MasterSelects-Helper.AppImage
```

---

## Render Pipeline Integration

### The Key Split

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WHERE THINGS HAPPEN                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NATIVE HELPER (Rust + FFmpeg):                                        │
│  └── Decode ProRes/DNxHD → RGBA frames                                 │
│  └── Encode final output → ProRes/H.264                                │
│  └── Fast, hardware accelerated                                        │
│                                                                         │
│  BROWSER (WebGL/WebGPU):                                               │
│  └── Composite multiple video layers                                   │
│  └── Apply effects (color grade, blur, etc.)                          │
│  └── Blend modes                                                       │
│  └── Text, graphics overlays                                           │
│  └── Real-time preview                                                 │
│                                                                         │
│  This is the same split you already have with WebCodecs!               │
│  Native helper just replaces WebCodecs VideoDecoder.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Multiple Layers + Effects

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TIMELINE WITH 3 VIDEO LAYERS + EFFECTS                                │
└─────────────────────────────────────────────────────────────────────────┘

   NATIVE HELPER                           BROWSER
   ─────────────                           ───────

   Layer 1 (ProRes A-roll)                     │
   ┌─────────────────┐                         │
   │  decode(frame)  │ ──── RGBA ─────────────►│
   └─────────────────┘                         │
                                               │
   Layer 2 (ProRes B-roll)                     │    ┌───────────────────┐
   ┌─────────────────┐                         │    │                   │
   │  decode(frame)  │ ──── RGBA ─────────────►├───►│  WebGL Compositor │
   └─────────────────┘                         │    │                   │
                                               │    │  - Blend layers   │
   Layer 3 (DNxHD graphics)                    │    │  - Apply effects  │
   ┌─────────────────┐                         │    │  - Color grade    │
   │  decode(frame)  │ ──── RGBA ─────────────►│    │  - Transitions    │
   └─────────────────┘                         │    │                   │
                                               │    └─────────┬─────────┘
                                               │              │
                                               │              ▼
                                               │    ┌───────────────────┐
                                               │    │  Preview Canvas   │
                                               │    │  (what user sees) │
                                               │    └───────────────────┘
```

### Displaying Frames in Browser

#### Step 1: Receive Frame from Native Helper

```typescript
class NativeDecoder {
  private ws: WebSocket;
  private frameCallbacks: Map<number, (frame: VideoFrame) => void>;

  async decodeFrame(fileId: string, frameNum: number): Promise<VideoFrame> {
    return new Promise((resolve) => {
      // Send decode request
      this.ws.send(JSON.stringify({
        cmd: 'decode',
        id: `frame_${frameNum}`,
        file_id: fileId,
        frame: frameNum,
      }));

      // Wait for binary frame response
      this.frameCallbacks.set(frameNum, resolve);
    });
  }

  private handleMessage(event: MessageEvent) {
    if (event.data instanceof ArrayBuffer) {
      const frame = this.parseFrameMessage(event.data);

      // Create VideoFrame from RGBA data
      const videoFrame = new VideoFrame(frame.rgbaData, {
        format: 'RGBA',
        codedWidth: frame.width,
        codedHeight: frame.height,
        timestamp: frame.frameNum * (1000000 / this.fps),
      });

      // Or create ImageBitmap for WebGL
      const bitmap = await createImageBitmap(
        new ImageData(
          new Uint8ClampedArray(frame.rgbaData),
          frame.width,
          frame.height
        )
      );

      this.frameCallbacks.get(frame.frameNum)?.(videoFrame);
    }
  }
}
```

#### Step 2: Upload to WebGL Texture

```typescript
class VideoLayerRenderer {
  private gl: WebGL2RenderingContext;
  private textures: Map<string, WebGLTexture>;

  uploadFrame(layerId: string, frame: VideoFrame | ImageBitmap) {
    let texture = this.textures.get(layerId);

    if (!texture) {
      texture = this.gl.createTexture()!;
      this.textures.set(layerId, texture);
    }

    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

    // Upload frame to GPU - this is fast!
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      frame  // VideoFrame or ImageBitmap works directly
    );

    frame.close?.();  // Release VideoFrame if applicable
  }
}
```

#### Step 3: Composite with Effects

```typescript
class Compositor {
  private gl: WebGL2RenderingContext;
  private effectShaders: Map<string, WebGLProgram>;

  render(layers: Layer[], effects: Effect[], outputCanvas: HTMLCanvasElement) {
    // For each layer (bottom to top)
    for (const layer of layers) {
      // Bind layer texture
      this.gl.bindTexture(this.gl.TEXTURE_2D, layer.texture);

      // Apply layer transform (position, scale, rotation)
      this.setTransform(layer.transform);

      // Apply blend mode
      this.setBlendMode(layer.blendMode);

      // Apply layer effects
      for (const effect of layer.effects) {
        this.applyEffect(effect);
      }

      // Draw layer quad
      this.drawQuad();
    }

    // Apply global effects (adjustment layers)
    for (const effect of effects) {
      this.applyGlobalEffect(effect);
    }
  }

  private setBlendMode(mode: BlendMode) {
    switch (mode) {
      case 'normal':
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
        break;
      case 'multiply':
        this.gl.blendFunc(this.gl.DST_COLOR, this.gl.ZERO);
        break;
      case 'screen':
        this.gl.blendFunc(this.gl.ONE, this.gl.ONE_MINUS_SRC_COLOR);
        break;
      case 'overlay':
        // Use custom shader for overlay
        this.useProgram(this.effectShaders.get('overlay')!);
        break;
      // ... more blend modes
    }
  }
}
```

### Multiple Previews at Same Time

```typescript
class PreviewManager {
  private decoder: NativeDecoder;
  private previews: Map<string, Preview>;

  // Each preview requests frames independently
  async updatePreviews(currentTime: number) {
    const frameRequests: Promise<void>[] = [];

    for (const [id, preview] of this.previews) {
      // Calculate which frame this preview needs
      const frameNum = Math.floor(currentTime * preview.timeline.fps);

      // Request all layers for this preview
      for (const layer of preview.timeline.layers) {
        if (layer.isVisible && layer.containsTime(currentTime)) {
          const localTime = currentTime - layer.startTime;
          const localFrame = Math.floor(localTime * layer.source.fps);

          frameRequests.push(
            this.decoder.decodeFrame(layer.source.fileId, localFrame)
              .then(frame => preview.uploadLayer(layer.id, frame))
          );
        }
      }
    }

    // Decode all needed frames in parallel
    await Promise.all(frameRequests);

    // Composite each preview
    for (const preview of this.previews.values()) {
      preview.composite();
    }
  }
}
```

### Optimizing Multi-Layer Decode

The native helper can decode multiple frames in parallel:

```json
{
  "cmd": "decode_batch",
  "id": "batch_001",
  "requests": [
    { "file_id": "file_a", "frame": 100 },
    { "file_id": "file_b", "frame": 50 },
    { "file_id": "file_c", "frame": 75 }
  ]
}
```

Native helper processes in parallel:

```rust
async fn handle_batch_decode(requests: Vec<DecodeRequest>) -> Vec<Frame> {
    let futures = requests.into_iter().map(|req| {
        tokio::spawn(async move {
            decode_frame(&req.file_id, req.frame).await
        })
    });

    // Decode all frames concurrently
    futures::future::join_all(futures).await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect()
}
```

### Export Pipeline

When user exports, composited frames go back to native helper:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EXPORT FLOW                                                            │
└─────────────────────────────────────────────────────────────────────────┘

   For each frame:

   1. Browser composites all layers + effects → final RGBA
      ┌─────────────────────────────────────────────────────────┐
      │  WebGL renders to offscreen canvas                      │
      │  Layer 1 + Layer 2 + Effects → composited frame        │
      └─────────────────────────────────────────────────────────┘
                                │
                                ▼
   2. Read pixels from WebGL
      ┌─────────────────────────────────────────────────────────┐
      │  gl.readPixels() → Uint8Array (RGBA)                    │
      │  Or use VideoFrame from canvas                          │
      └─────────────────────────────────────────────────────────┘
                                │
                                ▼
   3. Send to native helper for encoding
      ┌─────────────────────────────────────────────────────────┐
      │  ws.send(rgbaPixels)                                    │
      │  Native helper encodes → ProRes/H.264                   │
      └─────────────────────────────────────────────────────────┘
```

#### Export Command

```json
{
  "cmd": "start_encode",
  "id": "export_001",
  "output": {
    "path": "/home/user/Videos/export.mov",
    "codec": "prores",
    "profile": "422_hq",
    "width": 1920,
    "height": 1080,
    "fps": 24
  },
  "frame_count": 2400
}
```

Then stream frames:

```typescript
async function exportVideo(timeline: Timeline, outputPath: string) {
  const fps = timeline.fps;
  const frameCount = Math.ceil(timeline.duration * fps);

  // Start encode session
  await nativeHelper.send({
    cmd: 'start_encode',
    output: { path: outputPath, codec: 'prores', ... },
    frame_count: frameCount,
  });

  // Render and send each frame
  for (let i = 0; i < frameCount; i++) {
    const time = i / fps;

    // Composite frame in WebGL
    const compositedFrame = await compositor.renderFrame(timeline, time);

    // Read pixels
    const pixels = compositor.readPixels();

    // Send to native helper (binary message)
    nativeHelper.sendFrame(i, pixels);

    // Report progress
    onProgress(i / frameCount);
  }

  // Finalize
  await nativeHelper.send({ cmd: 'finish_encode' });
}
```

### Frame Reading from WebGL

```typescript
class Compositor {
  private gl: WebGL2RenderingContext;
  private readbackBuffer: Uint8Array;
  private pbo: WebGLBuffer;  // Pixel Buffer Object for async read

  constructor() {
    // Pre-allocate buffer for readback
    this.readbackBuffer = new Uint8Array(1920 * 1080 * 4);

    // Create PBO for async readback (faster)
    this.pbo = this.gl.createBuffer()!;
    this.gl.bindBuffer(this.gl.PIXEL_PACK_BUFFER, this.pbo);
    this.gl.bufferData(
      this.gl.PIXEL_PACK_BUFFER,
      1920 * 1080 * 4,
      this.gl.STREAM_READ
    );
  }

  // Fast async pixel readback using PBO
  async readPixelsAsync(): Promise<Uint8Array> {
    const gl = this.gl;

    // Start async read to PBO
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.readPixels(0, 0, 1920, 1080, gl.RGBA, gl.UNSIGNED_BYTE, 0);

    // Sync and read from PBO
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;

    return new Promise((resolve) => {
      const checkSync = () => {
        const status = gl.clientWaitSync(sync, 0, 0);
        if (status === gl.WAIT_FAILED) {
          throw new Error('GPU sync failed');
        }
        if (status === gl.TIMEOUT_EXPIRED) {
          requestAnimationFrame(checkSync);
          return;
        }

        // Read from PBO
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.readbackBuffer);
        gl.deleteSync(sync);

        resolve(this.readbackBuffer);
      };

      requestAnimationFrame(checkSync);
    });
  }
}
```

### Transfer Overhead Analysis

**The concern**: Are we losing time transferring frames between native helper and browser?

```
┌─────────────────────────────────────────────────────────────────────────┐
│  4K FRAME TRANSFER BREAKDOWN                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Frame size: 3840 × 2160 × 4 = 33.2 MB                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  ProRes HW Decode         │  ~0.5ms   │  ████                    │  │
│  │  Memory copy (native)     │  ~1ms     │  ████████                │  │
│  │  WebSocket send           │  ~3ms     │  ████████████████        │  │
│  │  Browser receive          │  ~1ms     │  ████████                │  │
│  │  Upload to WebGL texture  │  ~2ms     │  ████████████            │  │
│  │  ────────────────────────────────────────────────────────────    │  │
│  │  TOTAL                    │  ~7.5ms   │  130+ fps at 4K          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Compare to WebGPU-only approach: ~5ms                                 │
│  Overhead: ~2.5ms (acceptable!)                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why it's not as bad as it sounds:**

1. **Localhost is FAST** - No network stack, essentially shared memory IPC
   - Localhost throughput: 10-50 Gbps
   - 33 MB transfer: ~3-5ms

2. **Still faster than WASM decode** - Even with transfer overhead:
   - Native decode + transfer: ~7.5ms
   - WASM+WebGPU decode: ~5ms
   - Pure WASM decode: ~50ms

### Optimization: Reduce Transfer Size

#### 1. LZ4 Compression (Recommended)

Video frames compress very well with LZ4:

```
┌─────────────────────────────────────────────────────────────────┐
│  WITH LZ4 COMPRESSION                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw frame:        33.2 MB                                      │
│  LZ4 compressed:   ~8-12 MB (3-4x smaller)                      │
│                                                                 │
│  Compress time:    ~2ms                                         │
│  Transfer time:    ~1ms (vs 3ms raw)                           │
│  Decompress time:  ~1ms                                         │
│  ──────────────────────────────────                            │
│  Total:            ~4ms (vs 3ms raw)                           │
│                                                                 │
│  BUT: Works better when network isn't localhost                │
│       (future: remote rendering, cloud, etc.)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Scaled Previews During Scrubbing

During fast scrubbing, send lower resolution:

```typescript
// Browser requests appropriate scale based on interaction
const scale = isDraggingScrubber ? 0.25 : 1.0;

nativeHelper.decodeFrame(fileId, frame, { scale });

// 4K at 0.25 scale = 960x540 = 2 MB instead of 33 MB
```

| Interaction | Scale | Frame Size | Transfer |
|-------------|-------|------------|----------|
| Fast scrub | 0.25 | 2 MB | ~0.2ms |
| Slow scrub | 0.5 | 8 MB | ~0.8ms |
| Paused | 1.0 | 33 MB | ~3ms |

#### 3. Shared Memory (Advanced, Linux)

For maximum performance, use shared memory instead of WebSocket:

```rust
// Native helper creates shared memory
let shm = SharedMemory::create("masterselects-frames", FRAME_SIZE)?;

// Browser maps same memory via WebAssembly
const sharedBuffer = new SharedArrayBuffer(FRAME_SIZE);
```

**Benefits:**
- Near zero-copy transfer
- Sub-millisecond latency

**Drawbacks:**
- More complex
- Requires SharedArrayBuffer (COOP/COEP headers)
- Platform-specific APIs

#### 4. GPU Texture Sharing (Future, Complex)

Ultimate zero-copy: Share GPU textures between processes.

```
Native helper → GPU texture → Browser WebGL

No CPU-side pixel copy at all!
```

**Possible on Linux via:**
- DMA-BUF export/import
- Vulkan external memory
- EGL image sharing

**Very complex**, but achievable. Save for v2.

### Realistic Performance Summary

| Scenario | Decode | Transfer | Total | FPS |
|----------|--------|----------|-------|-----|
| 4K raw, localhost | 0.5ms | 3ms | 7.5ms | 130 |
| 4K + LZ4 | 0.5ms | 4ms | 8.5ms | 115 |
| 4K + scale 0.5 (scrub) | 0.5ms | 1ms | 3.5ms | 285 |
| 4K + scale 0.25 (fast scrub) | 0.3ms | 0.3ms | 1.5ms | 666 |
| 1080p raw | 0.3ms | 0.8ms | 2.5ms | 400 |

**Bottom line**: Even with transfer overhead, native helper is plenty fast for real-time scrubbing. The optimizations (scaling, compression) make it even better.

### Summary: What Happens Where

| Operation | Where | Why |
|-----------|-------|-----|
| Decode ProRes | Native Helper | HW accel, fast |
| Decode DNxHD | Native Helper | FFmpeg has it |
| Decode H.264 | Browser or Native | Both work |
| Compositing | Browser (WebGL) | GPU, real-time |
| Effects | Browser (WebGL) | Shaders, interactive |
| Blend modes | Browser (WebGL) | GPU, real-time |
| Text/graphics | Browser (Canvas/WebGL) | Interactive editing |
| Preview display | Browser (Canvas) | Must be in browser |
| Encode ProRes | Native Helper | FFmpeg, HW accel |
| Encode H.264 | Native Helper | HW accel |

**Key insight**: The native helper is a "codec service" - it just decodes and encodes. All the creative compositing stays in the browser where you already have it working!

---

## Future: macOS & Windows

### macOS Additions

```rust
// Hardware acceleration via VideoToolbox
#[cfg(target_os = "macos")]
fn create_videotoolbox_decoder() -> Option<Decoder> {
    // VideoToolbox provides native ProRes HW decode/encode
    // This is essentially "free" performance on Mac
}
```

Build:
```bash
# Universal binary (Intel + Apple Silicon)
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
lipo -create -output masterselects-helper \
    target/x86_64-apple-darwin/release/masterselects-helper \
    target/aarch64-apple-darwin/release/masterselects-helper
```

Distribution:
- `.app` bundle (drag to Applications)
- DMG with background image
- Notarization for Gatekeeper

### Windows Additions

```rust
// Hardware acceleration via DXVA2/D3D11VA
#[cfg(target_os = "windows")]
fn create_dxva_decoder() -> Option<Decoder> {
    // ...
}
```

Build:
```powershell
# Static build with MSVC
cargo build --release
```

Distribution:
- Single `.exe` (portable)
- Optional MSI installer
- Code signing for SmartScreen

---

## Summary

### What We're Building

A **single executable** that:
- Runs on double-click, no install
- Provides WebSocket API on localhost
- Decodes ProRes (and other codecs) via FFmpeg
- Caches frames for smooth scrubbing
- Hardware accelerates where possible

### Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Basic WebSocket server + FFmpeg decode | 2-3 days |
| 2 | Frame caching and prefetch | 2 days |
| 3 | Encode support | 2 days |
| 4 | Linux packaging (binary + AppImage) | 1 day |
| 5 | Browser client integration | 2 days |
| **Total** | | **~2 weeks** |

### vs WebGPU Approach

| Aspect | Native Helper | WebGPU |
|--------|---------------|--------|
| Build time | ~2 weeks | ~3 months |
| ProRes decode | ~0.5ms (HW) | ~5ms |
| All codecs | FFmpeg has them | Build each |
| User friction | Download .exe | None |
| Offline | Full | Limited |

**Recommendation**: Build the native helper first. It's faster to build and faster to run. The WebGPU version can be a fallback for users who don't want to download anything.
