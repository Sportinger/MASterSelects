# Native Helper (Turbo Mode)

The Native Helper is an optional companion application that provides hardware-accelerated video decoding and encoding for professional codecs like ProRes and DNxHD.

## Overview

While the web app uses browser-native decoders (WebCodecs, HTMLVideoElement) and FFmpeg WASM for export, the Native Helper provides **10x faster** performance by using system FFmpeg with hardware acceleration.

## Features

- **ProRes decoding** - All profiles (Proxy to 4444 XQ)
- **DNxHD/DNxHR decoding** - All profiles
- **Hardware acceleration** - VAAPI (Intel/AMD), NVDEC (NVIDIA)
- **LRU frame cache** - Smooth scrubbing with up to 2GB cache
- **Background prefetch** - Frames loaded ahead of playhead
- **YouTube downloads** - Fast downloads via yt-dlp integration
- **Multi-platform downloads** - TikTok, Instagram, Twitter/X, and other platforms via yt-dlp

## Unified Cross-Platform Build

The Native Helper is a single unified Rust binary that builds and runs on all platforms:

| Platform | FFmpeg | Downloads (yt-dlp) | Notes |
|----------|--------|-------------------|-------|
| **Windows** | Full decode + encode | YouTube, TikTok, Instagram, Twitter/X, etc. | Requires FFMPEG_DIR + LIBCLANG_PATH env vars |
| **Linux** | Full decode + encode | YouTube, TikTok, Instagram, Twitter/X, etc. | System FFmpeg (pkg-config) |
| **macOS** | Full decode + encode | YouTube, TikTok, Instagram, Twitter/X, etc. | Homebrew FFmpeg recommended |

### Building

```bash
cd tools/native-helper
cargo run --release
```

### Windows Build Setup

On Windows, two environment variables are required for building:
- **`FFMPEG_DIR`** - Path to FFmpeg development libraries (headers + DLLs)
- **`LIBCLANG_PATH`** - Path to libclang (for FFmpeg bindings generation)

At runtime, FFmpeg DLLs must be available. The binary auto-detects them in this order:
1. DLLs next to the executable (e.g., `avcodec-61.dll`)
2. `ffmpeg/bin/` subdirectory next to the executable
3. `FFMPEG_DIR` environment variable (looks in `%FFMPEG_DIR%/bin/`)

Download FFmpeg DLLs from: https://github.com/BtbN/FFmpeg-Builds/releases

See `tools/native-helper/README.md` for detailed Windows setup instructions.

### Multi-Platform Download

The app detects the user's platform (Windows, Linux, macOS) and provides the appropriate download link from GitHub Releases. Click the Turbo indicator in the toolbar to access the download dialog.

## Architecture

```
Browser (MasterSelects App)
    │
    │ WebSocket (ws://127.0.0.1:9876)
    │
    ▼
Native Helper (Rust)
    │
    │ FFmpeg libraries
    │
    ▼
System video hardware
```

## Installation

### Linux

1. Download the helper from the toolbar (click the Turbo indicator) or from [GitHub Releases](https://github.com/Sportinger/MASterSelects/releases/latest)
2. Make it executable: `chmod +x masterselects-helper`
3. Run it: `./masterselects-helper`

The helper will automatically be detected by the app.

### Windows

1. Download from the toolbar or [GitHub Releases](https://github.com/Sportinger/MASterSelects/releases/latest)
2. Ensure FFmpeg DLLs are available (see [Windows DLL Setup](#windows-dll-setup))
3. Run `masterselects-helper.exe`

### macOS

1. Download from the toolbar or [GitHub Releases](https://github.com/Sportinger/MASterSelects/releases/latest)
2. Install FFmpeg via Homebrew: `brew install ffmpeg`
3. Make executable and run: `chmod +x masterselects-helper && ./masterselects-helper`

### Options

```bash
masterselects-helper [OPTIONS]

Options:
  -p, --port <PORT>          Port to listen on [default: 9876]
      --cache-mb <MB>        Maximum cache size in MB [default: 2048]
      --max-decoders <N>     Maximum open decoder contexts [default: 8]
      --log-level <LEVEL>    Log level (trace/debug/info/warn/error)
  -h, --help                 Print help
  -V, --version              Print version
```

## Usage

### Enabling Turbo Mode

1. Run the Native Helper
2. The toolbar will show "⚡ Turbo" when connected
3. Import ProRes/DNxHD files - they will automatically use native decoding

### Status Indicator

The toolbar shows the helper status:
- **○** - Not connected (click for download)
- **⚡ Turbo** - Connected and active

Click the indicator for details:
- Helper version
- Cache usage
- Hardware acceleration status
- Open files count

## Supported Codecs

### Decoding
- ProRes (all profiles)
- DNxHD / DNxHR (all profiles)
- H.264 / AVC
- H.265 / HEVC
- VP9
- FFV1
- UTVideo
- MJPEG

### Encoding (via native helper)
- ProRes (prores_ks encoder)
- DNxHD/DNxHR
- H.264 (libx264)
- H.265 (libx265)
- VP9 (libvpx-vp9)
- FFV1
- UTVideo
- MJPEG

## Performance

| Operation | Browser Only | With Native Helper |
|-----------|-------------|-------------------|
| ProRes decode | ~15 fps | ~60+ fps |
| DNxHD decode | ~20 fps | ~60+ fps |
| Scrubbing | Laggy | Smooth |
| Export (ProRes) | WASM (slow) | Native (10x faster) |

## Technical Details

### Protocol

The helper communicates via WebSocket with JSON commands and binary frame data.

**Commands:**
- `open` - Open a video file
- `decode` - Decode a single frame
- `prefetch` - Background cache warming
- `encode` - Start/feed/finish encode jobs
- `close` - Close a file

**Frame Format:**
Binary messages with 16-byte header containing width, height, frame number, and optional LZ4 compression.

### Security

- **Localhost only** - Binds to 127.0.0.1
- **Origin validation** - Only accepts connections from allowed origins
- **No network access** - Only local file system

### Source Code

The helper is a unified Rust binary:
```
tools/native-helper/
└── src/
    ├── main.rs          # Entry point, CLI args, platform setup
    ├── server.rs        # WebSocket + HTTP server
    ├── session.rs       # Auth token management
    ├── decoder/         # FFmpeg video decoding
    │   ├── mod.rs
    │   ├── ffmpeg.rs
    │   ├── hwaccel.rs
    │   └── pool.rs
    ├── encoder/         # FFmpeg video encoding
    │   ├── mod.rs
    │   └── job.rs
    ├── cache/           # LRU frame cache
    │   ├── mod.rs
    │   └── lru.rs
    ├── download/        # yt-dlp integration
    │   ├── mod.rs
    │   └── ytdlp.rs
    ├── protocol/        # WebSocket protocol
    │   ├── mod.rs
    │   ├── commands.rs
    │   └── frame.rs
    └── utils.rs         # Shared utilities
```

Browser client code:
```
src/services/nativeHelper/
├── NativeHelperClient.ts  # WebSocket client
├── NativeDecoder.ts       # Decoder wrapper
├── protocol.ts            # Message types
└── index.ts
```

Build with:
```bash
cd tools/native-helper
cargo build --release
```

## Troubleshooting

### Helper not detected

1. Check if running: `ps aux | grep masterselects-helper`
2. Check port: `ss -tlnp | grep 9876`
3. Try restart: Kill and run again

### Slow performance

1. Check cache size: Increase with `--cache-mb 4096`
2. Check hardware accel: Look for "vaapi" or "nvdec" in status
3. Ensure FFmpeg has hardware support

### Connection errors

1. Check firewall allows localhost:9876
2. Ensure only one instance running
3. Check browser console for WebSocket errors
