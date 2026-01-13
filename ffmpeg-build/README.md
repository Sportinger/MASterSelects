# FFmpeg WASM Build

Custom FFmpeg WASM build with ASYNCIFY support for streaming encode.

## Features

- **ASYNCIFY**: Enables streaming encode with constant memory usage (~50MB)
- **Professional Video Codecs**: ProRes, DNxHR, FFV1, UTVideo
- **Standard Video Codecs**: MJPEG, PNG, TIFF, DPX, RawVideo
- **Audio Codecs**: AAC, FLAC, ALAC, PCM (16/24-bit, float), AC3

## Build

```bash
cd ffmpeg-build
./build.sh
```

Build time: ~10-15 minutes (uses Docker cache for subsequent builds)

## Architecture

### Current (Batch Mode - without ASYNCIFY)
```
Render ALL frames → Buffer 500MB → Write to FS → FFmpeg encodes
Memory: O(frames) - grows with duration
```

### ASYNCIFY (Streaming Mode)
```
Render frame 1 → Write → FFmpeg processes → Render frame 2 → ...
Memory: O(1) - constant regardless of duration
```

## ASYNCIFY vs Multi-threading

**Important**: ASYNCIFY and pthreads are incompatible in Emscripten.

| Feature | ASYNCIFY | Multi-threaded |
|---------|----------|----------------|
| Memory | ~50MB constant | Grows with duration |
| Speed | Single-threaded | Multi-core |
| Cancel | Yes (graceful) | Hard to cancel |
| Progress | Native support | Polling required |
| Requirements | None | SharedArrayBuffer |

This build uses ASYNCIFY - ideal for long exports where memory is critical.

## Output Files

After building, files are placed in `public/ffmpeg/`:
- `ffmpeg-core.js` (~190KB)
- `ffmpeg-core.wasm` (~41MB)

## Included Encoders

### Video
- `prores_ks` - Apple ProRes (MOV)
- `dnxhd` - DNxHR (MXF, MOV)
- `ffv1` - FFV1 lossless (MKV)
- `utvideo` - UTVideo lossless (AVI, MOV)
- `mjpeg` - Motion JPEG (AVI, MOV)
- `png` - PNG sequence
- `tiff` - TIFF sequence
- `dpx` - DPX sequence
- `rawvideo` - Uncompressed

### Audio
- `aac` - AAC (MP4, MOV)
- `flac` - FLAC lossless
- `alac` - Apple Lossless
- `pcm_s16le/s24le/f32le` - PCM audio
- `ac3` - Dolby AC3

### Decoders (input support)
- H.264, HEVC, VP9, AV1
- ProRes, PNG, MJPEG, RawVideo
- AAC, PCM

### Not Included (yet)
- H.264 encoder (libx264 - pkg-config issues)
- VP9 encoder (libvpx - pkg-config issues)
- HAP (requires snappy)
- WebP (requires pkg-config)

## Container Formats

### Muxers (output)
- MOV, MP4, MKV, AVI
- MXF, MXF OP-Atom
- MPEG-TS, OGG, WAV
- Image sequences (PNG, TIFF, DPX)

### Demuxers (input)
- MOV, MKV, AVI
- Image sequences, RawVideo

## Troubleshooting

### Build fails with memory error
Increase Docker memory limit in Docker Desktop settings.

### WASM file too large
~41MB includes all codecs + debug info. Production build would be smaller.

### "SharedArrayBuffer is not defined"
Expected with ASYNCIFY build. Multi-threading requires different build.

## Technical Details

### ASYNCIFY Emscripten Flags
```
-sASYNCIFY                    # Enable async yielding
-sASYNCIFY_STACK_SIZE=65536   # Stack for async state
-sALLOW_MEMORY_GROWTH=1       # Dynamic memory
-sINITIAL_MEMORY=134217728    # 128MB initial
-sMAXIMUM_MEMORY=4294967296   # 4GB max
-sMODULARIZE=1                # ES module export
-sEXPORT_NAME=createFFmpegCore
```

### External Libraries
- zlib - PNG compression
