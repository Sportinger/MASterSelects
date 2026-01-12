# FFmpeg WASM Custom Build Plan

## Executive Summary

Build a custom FFmpeg WebAssembly module with comprehensive professional codec support for browser-based video export. This will enable HAP, ProRes, DNxHR, and other professional codecs directly in MASterSelects.

**Estimated WASM Size:** 20-30 MB (gzipped: ~8-12 MB)
**Build Time:** ~15-30 minutes on modern hardware

---

## 1. Codec Selection

### 1.1 Video Codecs (Encoding)

| Codec | Library | Priority | Use Case | License |
|-------|---------|----------|----------|---------|
| **ProRes** | native (prores_ks) | HIGH | Apple workflows, intermediate | LGPL |
| **HAP / HAP Q / HAP Alpha** | libsnappy | HIGH | VJ, real-time playback | BSD |
| **DNxHR** | native (dnxhd) | HIGH | Avid workflows, broadcast | LGPL |
| **NotchLC** | - | SKIP | Proprietary, no FFmpeg support | - |
| **CineForm** | libcfhd | MEDIUM | GoPro intermediate | Apache 2.0 |
| **FFV1** | native | MEDIUM | Lossless archival | LGPL |
| **Ut Video** | native (utvideo) | LOW | Fast lossless | LGPL |
| **MJPEG** | native | LOW | Simple, compatible | LGPL |
| **x264 (H.264)** | libx264 | HIGH | Better than WebCodecs | GPL |
| **x265 (HEVC)** | libx265 | MEDIUM | HDR, 10-bit | GPL |
| **VP9** | libvpx | MEDIUM | WebM, open | BSD |
| **SVT-AV1** | libsvtav1 | HIGH | Faster than libaom | BSD |
| **MPEG-2** | native (mpeg2video) | LOW | DVD/broadcast legacy | LGPL |

### 1.2 Emerging Codecs (Future-Proof)

| Codec | Status | Notes |
|-------|--------|-------|
| **VVC / H.266** | EXPERIMENTAL | 50% better than HEVC, hardware arriving 2027 |
| **AV2** | NOT READY | Spec finalizing late 2025, silicon 2026-2028 |
| **LCEVC** | SKIP | Enhancement layer, not standalone |

> **Recommendation:** Include VVC encoder (vvenc) as experimental feature. Skip AV2 until encoder matures.

### 1.3 Audio Codecs

| Codec | Library | Priority | Use Case |
|-------|---------|----------|----------|
| **AAC** | native (aac) | HIGH | Standard |
| **MP3** | libmp3lame | MEDIUM | Legacy compatibility |
| **Opus** | libopus | HIGH | Modern, efficient |
| **FLAC** | native | MEDIUM | Lossless |
| **ALAC** | native | LOW | Apple lossless |
| **PCM** | native | HIGH | Uncompressed WAV |
| **AC3** | native | LOW | Dolby Digital |
| **Vorbis** | libvorbis | LOW | OGG container |

### 1.4 Containers/Muxers

| Format | Priority | Codecs Supported |
|--------|----------|------------------|
| **MOV** | HIGH | ProRes, HAP, DNxHR, H.264, HEVC |
| **MP4** | HIGH | H.264, HEVC, AV1 |
| **MKV** | HIGH | Everything |
| **WebM** | MEDIUM | VP9, AV1, Opus |
| **AVI** | MEDIUM | HAP, MJPEG, legacy |
| **MXF** | MEDIUM | DNxHR, broadcast |
| **TS/M2TS** | LOW | Broadcast, AVCHD |
| **OGG** | LOW | Vorbis, Opus |

### 1.5 Image Formats (Sequences)

| Format | Library | Priority | Use Case |
|--------|---------|----------|----------|
| **PNG** | native | HIGH | Lossless + alpha |
| **JPEG** | native | HIGH | Smaller files |
| **TIFF** | native | MEDIUM | Print/archival |
| **EXR** | OpenEXR | HIGH | VFX, HDR, 32-bit |
| **DPX** | native | MEDIUM | Film industry |
| **WebP** | libwebp | MEDIUM | Modern web |
| **TGA** | native | LOW | Legacy alpha |

### 1.6 Filters

| Filter | Priority | Purpose |
|--------|----------|---------|
| scale | HIGH | Resize with quality algorithms |
| fps | HIGH | Frame rate conversion |
| colorspace | HIGH | Rec.709 ↔ Rec.2020 ↔ sRGB |
| lut3d | MEDIUM | Apply .cube LUTs |
| loudnorm | HIGH | EBU R128 audio normalization |
| aresample | HIGH | Audio resampling |
| drawtext | LOW | Burn-in timecode |
| alphamerge | MEDIUM | Alpha channel handling |
| hdr10_metadata | LOW | HDR passthrough |

---

## 2. Build Architecture

### 2.1 Directory Structure

```
ffmpeg-wasm-custom/
├── build/
│   ├── Dockerfile              # Build environment
│   ├── build.sh                # Main build script
│   ├── configure-ffmpeg.sh     # FFmpeg configure options
│   └── patches/                # Any source patches
├── libs/
│   ├── build-x264.sh
│   ├── build-x265.sh
│   ├── build-svtav1.sh
│   ├── build-snappy.sh
│   ├── build-openexr.sh
│   └── build-all.sh
├── dist/
│   ├── ffmpeg-core.wasm        # Main WASM binary
│   ├── ffmpeg-core.js          # JS loader
│   └── ffmpeg-core.worker.js   # Web Worker wrapper
├── src/
│   └── ffmpeg-bridge.ts        # TypeScript integration
└── README.md
```

### 2.2 Build Dependencies

```dockerfile
FROM emscripten/emsdk:3.1.50

# System dependencies
RUN apt-get update && apt-get install -y \
    autoconf automake libtool pkg-config \
    cmake ninja-build nasm yasm \
    git wget

# Directory structure
WORKDIR /build
```

### 2.3 External Libraries Build Order

1. **zlib** (compression, needed by many)
2. **libsnappy** (HAP codec)
3. **libx264** (H.264 encoder)
4. **libx265** (HEVC encoder)
5. **libvpx** (VP9 encoder)
6. **libsvtav1** (AV1 encoder)
7. **libmp3lame** (MP3 encoder)
8. **libopus** (Opus encoder)
9. **libvorbis** (Vorbis encoder)
10. **OpenEXR** (EXR image format)
11. **libwebp** (WebP encoder)
12. **libcfhd** (CineForm - optional)

---

## 3. Build Scripts

### 3.1 Snappy (for HAP)

```bash
#!/bin/bash
# build-snappy.sh

git clone --depth 1 https://github.com/google/snappy.git
cd snappy

mkdir build && cd build
emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/snappy \
    -DSNAPPY_BUILD_TESTS=OFF \
    -DSNAPPY_BUILD_BENCHMARKS=OFF

emmake make -j$(nproc)
emmake make install
```

### 3.2 x264

```bash
#!/bin/bash
# build-x264.sh

git clone --depth 1 https://code.videolan.org/videolan/x264.git
cd x264

emconfigure ./configure \
    --prefix=/opt/x264 \
    --host=i686-gnu \
    --enable-static \
    --disable-cli \
    --disable-asm \
    --extra-cflags="-O3"

emmake make -j$(nproc)
emmake make install
```

### 3.3 SVT-AV1 (faster than libaom)

```bash
#!/bin/bash
# build-svtav1.sh

git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1

mkdir build && cd build
emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/svtav1 \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_APPS=OFF

emmake make -j$(nproc)
emmake make install
```

### 3.4 OpenEXR

```bash
#!/bin/bash
# build-openexr.sh

git clone --depth 1 https://github.com/AcademySoftwareFoundation/openexr.git
cd openexr

mkdir build && cd build
emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/openexr \
    -DBUILD_SHARED_LIBS=OFF \
    -DOPENEXR_BUILD_TOOLS=OFF \
    -DOPENEXR_INSTALL_EXAMPLES=OFF

emmake make -j$(nproc)
emmake make install
```

### 3.5 FFmpeg Configure

```bash
#!/bin/bash
# configure-ffmpeg.sh

emconfigure ./configure \
    --prefix=/opt/ffmpeg \
    --target-os=none \
    --arch=x86_32 \
    --enable-cross-compile \
    --disable-x86asm \
    --disable-inline-asm \
    --disable-stripping \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-runtime-cpudetect \
    --disable-autodetect \
    \
    --enable-gpl \
    --enable-version3 \
    \
    # External libraries
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libsvtav1 \
    --enable-libsnappy \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libwebp \
    \
    # Video encoders
    --enable-encoder=libx264 \
    --enable-encoder=libx265 \
    --enable-encoder=libvpx_vp9 \
    --enable-encoder=libsvtav1 \
    --enable-encoder=prores_ks \
    --enable-encoder=hap \
    --enable-encoder=dnxhd \
    --enable-encoder=cfhd \
    --enable-encoder=ffv1 \
    --enable-encoder=utvideo \
    --enable-encoder=mjpeg \
    --enable-encoder=mpeg2video \
    --enable-encoder=png \
    --enable-encoder=tiff \
    --enable-encoder=dpx \
    --enable-encoder=exr \
    --enable-encoder=libwebp \
    \
    # Audio encoders
    --enable-encoder=aac \
    --enable-encoder=libmp3lame \
    --enable-encoder=libopus \
    --enable-encoder=flac \
    --enable-encoder=alac \
    --enable-encoder=pcm_s16le \
    --enable-encoder=pcm_s24le \
    --enable-encoder=ac3 \
    --enable-encoder=libvorbis \
    \
    # Muxers
    --enable-muxer=mov \
    --enable-muxer=mp4 \
    --enable-muxer=matroska \
    --enable-muxer=webm \
    --enable-muxer=avi \
    --enable-muxer=mxf \
    --enable-muxer=mxf_opatom \
    --enable-muxer=mpegts \
    --enable-muxer=ogg \
    --enable-muxer=wav \
    --enable-muxer=image2 \
    \
    # Demuxers (for input support)
    --enable-demuxer=mov \
    --enable-demuxer=matroska \
    --enable-demuxer=avi \
    --enable-demuxer=image2 \
    \
    # Decoders (for re-encoding support)
    --enable-decoder=h264 \
    --enable-decoder=hevc \
    --enable-decoder=vp9 \
    --enable-decoder=av1 \
    --enable-decoder=prores \
    --enable-decoder=png \
    --enable-decoder=mjpeg \
    \
    # Filters
    --enable-filter=scale \
    --enable-filter=fps \
    --enable-filter=colorspace \
    --enable-filter=lut3d \
    --enable-filter=loudnorm \
    --enable-filter=aresample \
    --enable-filter=format \
    --enable-filter=aformat \
    --enable-filter=null \
    --enable-filter=anull \
    \
    # Protocols
    --enable-protocol=file \
    \
    # Emscripten-specific
    --extra-cflags="-O3 -I/opt/x264/include -I/opt/svtav1/include -I/opt/snappy/include" \
    --extra-ldflags="-L/opt/x264/lib -L/opt/svtav1/lib -L/opt/snappy/lib" \
    --nm="llvm-nm" \
    --ar="emar" \
    --ranlib="emranlib" \
    --cc="emcc" \
    --cxx="em++" \
    --objcc="emcc" \
    --dep-cc="emcc"
```

### 3.6 Final WASM Build

```bash
#!/bin/bash
# build-wasm.sh

emmake make -j$(nproc)

# Create WASM module
emcc \
    -O3 \
    -I. -I./fftools \
    -Llibavcodec -Llibavformat -Llibavutil -Llibswscale -Llibavfilter -Llibswresample \
    -Wl,--whole-archive \
    -lavcodec -lavformat -lavutil -lswscale -lavfilter -lswresample \
    -Wl,--no-whole-archive \
    -L/opt/x264/lib -lx264 \
    -L/opt/svtav1/lib -lSvtAv1Enc \
    -L/opt/snappy/lib -lsnappy \
    -lmp3lame -lopus -lvorbis -lvorbisenc -lwebp \
    -o dist/ffmpeg-core.js \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createFFmpegCore" \
    -s EXPORTED_FUNCTIONS="['_main', '_malloc', '_free']" \
    -s EXPORTED_RUNTIME_METHODS="['FS', 'callMain', 'cwrap']" \
    -s INITIAL_MEMORY=268435456 \
    -s MAXIMUM_MEMORY=2147483648 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INVOKE_RUN=0 \
    -s EXIT_RUNTIME=0 \
    -s FILESYSTEM=1 \
    -s FORCE_FILESYSTEM=1 \
    -s SINGLE_FILE=0
```

---

## 4. TypeScript Integration

### 4.1 FFmpeg Bridge

```typescript
// src/engine/ffmpeg/FFmpegBridge.ts

export type FFmpegCodec =
  | 'prores' | 'prores_hq' | 'prores_4444' | 'prores_xq'
  | 'hap' | 'hap_alpha' | 'hap_q'
  | 'dnxhd' | 'dnxhr_lb' | 'dnxhr_sq' | 'dnxhr_hq' | 'dnxhr_hqx' | 'dnxhr_444'
  | 'h264' | 'h265' | 'vp9' | 'av1'
  | 'ffv1' | 'utvideo' | 'mjpeg' | 'cineform';

export type FFmpegContainer =
  | 'mov' | 'mp4' | 'mkv' | 'webm' | 'avi' | 'mxf';

export type FFmpegImageFormat =
  | 'png' | 'jpeg' | 'tiff' | 'exr' | 'dpx' | 'webp' | 'tga';

export interface FFmpegExportSettings {
  codec: FFmpegCodec;
  container: FFmpegContainer;
  width: number;
  height: number;
  fps: number;
  bitrate?: number;           // For lossy codecs
  quality?: number;           // CRF mode (0-51)
  profile?: string;           // Codec-specific profile
  pixelFormat?: string;       // yuv420p, yuv422p, yuv444p, etc.
  colorSpace?: 'bt709' | 'bt2020' | 'srgb';
  // HAP-specific
  hapChunks?: number;         // 1-64, default 4
  hapCompressor?: 'snappy' | 'none';
  // ProRes-specific
  proresProfile?: 'proxy' | 'lt' | 'standard' | 'hq' | '4444' | '4444xq';
  // Audio
  audioCodec?: 'aac' | 'mp3' | 'opus' | 'flac' | 'pcm' | 'none';
  audioSampleRate?: number;
  audioBitrate?: number;
}

export interface FFmpegProgress {
  frame: number;
  fps: number;
  time: number;
  bitrate: string;
  speed: string;
  percent: number;
}

export class FFmpegBridge {
  private ffmpeg: any = null;
  private loaded = false;

  async load(): Promise<void> {
    if (this.loaded) return;

    // Dynamic import to avoid loading 20MB unless needed
    const { createFFmpegCore } = await import('./ffmpeg-core.js');
    this.ffmpeg = await createFFmpegCore();
    this.loaded = true;
    console.log('[FFmpegBridge] Loaded WASM module');
  }

  async encode(
    frames: Uint8Array[],
    settings: FFmpegExportSettings,
    onProgress?: (progress: FFmpegProgress) => void
  ): Promise<Blob> {
    if (!this.loaded) await this.load();

    // Build FFmpeg command
    const args = this.buildArgs(settings, frames.length);

    // Write frames to virtual filesystem
    for (let i = 0; i < frames.length; i++) {
      const filename = `frame_${String(i).padStart(6, '0')}.raw`;
      this.ffmpeg.FS.writeFile(filename, frames[i]);
    }

    // Run FFmpeg
    await this.ffmpeg.callMain(args);

    // Read output file
    const outputPath = `output.${settings.container}`;
    const data = this.ffmpeg.FS.readFile(outputPath);

    // Cleanup virtual filesystem
    this.cleanup();

    return new Blob([data], { type: this.getMimeType(settings.container) });
  }

  private buildArgs(settings: FFmpegExportSettings, frameCount: number): string[] {
    const args: string[] = [
      '-y',                           // Overwrite output
      '-f', 'rawvideo',               // Input format
      '-pix_fmt', 'rgba',             // Input pixel format
      '-s', `${settings.width}x${settings.height}`,
      '-r', String(settings.fps),
      '-i', 'frame_%06d.raw',         // Input pattern
    ];

    // Video codec settings
    args.push(...this.getCodecArgs(settings));

    // Audio (if any)
    if (settings.audioCodec && settings.audioCodec !== 'none') {
      args.push(...this.getAudioArgs(settings));
    } else {
      args.push('-an');  // No audio
    }

    // Output
    args.push(`output.${settings.container}`);

    return args;
  }

  private getCodecArgs(settings: FFmpegExportSettings): string[] {
    const args: string[] = [];

    switch (settings.codec) {
      case 'prores':
      case 'prores_hq':
      case 'prores_4444':
      case 'prores_xq':
        args.push('-c:v', 'prores_ks');
        args.push('-profile:v', this.getProResProfile(settings.codec));
        args.push('-pix_fmt', settings.codec.includes('4444') ? 'yuva444p10le' : 'yuv422p10le');
        break;

      case 'hap':
      case 'hap_alpha':
      case 'hap_q':
        args.push('-c:v', 'hap');
        args.push('-format', settings.codec);
        args.push('-compressor', settings.hapCompressor || 'snappy');
        args.push('-chunks', String(settings.hapChunks || 4));
        break;

      case 'dnxhr_lb':
      case 'dnxhr_sq':
      case 'dnxhr_hq':
      case 'dnxhr_hqx':
      case 'dnxhr_444':
        args.push('-c:v', 'dnxhd');
        args.push('-profile:v', settings.codec);
        args.push('-pix_fmt', settings.codec === 'dnxhr_444' ? 'yuv444p10le' : 'yuv422p');
        break;

      case 'h264':
        args.push('-c:v', 'libx264');
        args.push('-preset', 'medium');
        if (settings.quality !== undefined) {
          args.push('-crf', String(settings.quality));
        } else if (settings.bitrate) {
          args.push('-b:v', String(settings.bitrate));
        }
        break;

      case 'av1':
        args.push('-c:v', 'libsvtav1');
        args.push('-preset', '6');  // Balance speed/quality
        if (settings.quality !== undefined) {
          args.push('-crf', String(settings.quality));
        }
        break;

      case 'ffv1':
        args.push('-c:v', 'ffv1');
        args.push('-level', '3');
        args.push('-coder', '1');
        args.push('-context', '1');
        args.push('-slicecrc', '1');
        break;

      default:
        args.push('-c:v', settings.codec);
    }

    return args;
  }

  private getProResProfile(codec: string): string {
    const profiles: Record<string, string> = {
      'prores': '2',        // Standard
      'prores_hq': '3',     // HQ
      'prores_4444': '4',   // 4444
      'prores_xq': '5',     // 4444 XQ
    };
    return profiles[codec] || '2';
  }

  // ... additional helper methods
}
```

### 4.2 Export Panel Integration

```typescript
// Add to ExportPanel.tsx

const FFMPEG_CODECS = [
  // Professional
  { id: 'prores_hq', label: 'Apple ProRes HQ', container: 'mov', category: 'Professional' },
  { id: 'prores_4444', label: 'Apple ProRes 4444', container: 'mov', category: 'Professional' },
  { id: 'dnxhr_hq', label: 'Avid DNxHR HQ', container: 'mxf', category: 'Professional' },
  { id: 'dnxhr_444', label: 'Avid DNxHR 444', container: 'mxf', category: 'Professional' },

  // Real-time / VJ
  { id: 'hap', label: 'HAP', container: 'mov', category: 'Real-time' },
  { id: 'hap_q', label: 'HAP Q', container: 'mov', category: 'Real-time' },
  { id: 'hap_alpha', label: 'HAP Alpha', container: 'mov', category: 'Real-time' },

  // Lossless
  { id: 'ffv1', label: 'FFV1 (Archival)', container: 'mkv', category: 'Lossless' },
  { id: 'utvideo', label: 'Ut Video', container: 'avi', category: 'Lossless' },

  // Delivery
  { id: 'h264', label: 'H.264 (x264)', container: 'mp4', category: 'Delivery' },
  { id: 'h265', label: 'H.265 (x265)', container: 'mp4', category: 'Delivery' },
  { id: 'av1', label: 'AV1 (SVT)', container: 'mp4', category: 'Delivery' },
];
```

---

## 5. Image Sequence Export

```typescript
// src/engine/ffmpeg/ImageSequenceExporter.ts

export interface ImageSequenceSettings {
  format: 'png' | 'jpeg' | 'tiff' | 'exr' | 'dpx' | 'webp';
  width: number;
  height: number;
  fps: number;
  startFrame: number;
  padding: number;           // Frame number padding (default 6)
  quality?: number;          // JPEG quality 1-100
  compression?: number;      // PNG compression 0-9
  bitDepth?: 8 | 16 | 32;    // EXR/TIFF bit depth
  colorSpace?: string;
  outputDir: string;
  filenamePattern: string;   // e.g., "frame_######.png"
}

export class ImageSequenceExporter {
  async export(
    frames: Uint8Array[],
    settings: ImageSequenceSettings
  ): Promise<File[]> {
    const args = this.buildImageArgs(settings);
    // ... implementation
  }

  private buildImageArgs(settings: ImageSequenceSettings): string[] {
    const args: string[] = ['-y'];

    switch (settings.format) {
      case 'exr':
        args.push('-c:v', 'exr');
        args.push('-pix_fmt', settings.bitDepth === 32 ? 'gbrpf32le' : 'rgb48le');
        args.push('-compression', '1');  // PIZ compression
        break;

      case 'dpx':
        args.push('-c:v', 'dpx');
        args.push('-pix_fmt', 'rgb48le');
        break;

      case 'tiff':
        args.push('-c:v', 'tiff');
        args.push('-compression_algo', 'lzw');
        break;

      case 'png':
        args.push('-c:v', 'png');
        args.push('-compression_level', String(settings.compression || 6));
        break;

      case 'jpeg':
        args.push('-c:v', 'mjpeg');
        args.push('-q:v', String(Math.round((100 - (settings.quality || 90)) / 3)));
        break;
    }

    return args;
  }
}
```

---

## 6. Dockerfile (Complete Build)

```dockerfile
# Dockerfile
FROM emscripten/emsdk:3.1.50

# Install dependencies
RUN apt-get update && apt-get install -y \
    autoconf automake libtool pkg-config \
    cmake ninja-build nasm yasm \
    git wget texinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# ============ BUILD EXTERNAL LIBS ============

# Snappy (for HAP)
RUN git clone --depth 1 https://github.com/google/snappy.git && \
    cd snappy && mkdir build && cd build && \
    emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/snappy \
        -DSNAPPY_BUILD_TESTS=OFF -DSNAPPY_BUILD_BENCHMARKS=OFF && \
    emmake make -j$(nproc) && emmake make install

# x264
RUN git clone --depth 1 https://code.videolan.org/videolan/x264.git && \
    cd x264 && \
    emconfigure ./configure --prefix=/opt/x264 --host=i686-gnu \
        --enable-static --disable-cli --disable-asm && \
    emmake make -j$(nproc) && emmake make install

# x265 (simplified single-lib build)
RUN git clone --depth 1 -b Release_3.5 https://bitbucket.org/multicoreware/x265_git.git && \
    cd x265_git/build/linux && \
    emcmake cmake ../../source -DCMAKE_INSTALL_PREFIX=/opt/x265 \
        -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DENABLE_ASSEMBLY=OFF && \
    emmake make -j$(nproc) && emmake make install

# libvpx
RUN git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git && \
    cd libvpx && \
    emconfigure ./configure --prefix=/opt/vpx --target=generic-gnu \
        --enable-static --disable-shared --disable-examples --disable-tools \
        --disable-docs --disable-unit-tests && \
    emmake make -j$(nproc) && emmake make install

# SVT-AV1
RUN git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    cd SVT-AV1 && mkdir build && cd build && \
    emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/svtav1 \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_APPS=OFF && \
    emmake make -j$(nproc) && emmake make install

# LAME (MP3)
RUN wget https://sourceforge.net/projects/lame/files/lame/3.100/lame-3.100.tar.gz && \
    tar xf lame-3.100.tar.gz && cd lame-3.100 && \
    emconfigure ./configure --prefix=/opt/lame --host=i686-gnu \
        --enable-static --disable-shared --disable-frontend && \
    emmake make -j$(nproc) && emmake make install

# Opus
RUN git clone --depth 1 https://github.com/xiph/opus.git && \
    cd opus && ./autogen.sh && \
    emconfigure ./configure --prefix=/opt/opus --host=i686-gnu \
        --enable-static --disable-shared --disable-doc --disable-extra-programs && \
    emmake make -j$(nproc) && emmake make install

# WebP
RUN git clone --depth 1 https://chromium.googlesource.com/webm/libwebp && \
    cd libwebp && ./autogen.sh && \
    emconfigure ./configure --prefix=/opt/webp --host=i686-gnu \
        --enable-static --disable-shared && \
    emmake make -j$(nproc) && emmake make install

# ============ BUILD FFMPEG ============

RUN git clone --depth 1 -b n6.1 https://github.com/FFmpeg/FFmpeg.git ffmpeg

WORKDIR /src/ffmpeg

# Configure FFmpeg with all codecs
RUN emconfigure ./configure \
    --prefix=/opt/ffmpeg \
    --target-os=none \
    --arch=x86_32 \
    --enable-cross-compile \
    --disable-x86asm \
    --disable-inline-asm \
    --disable-stripping \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-runtime-cpudetect \
    --disable-autodetect \
    --enable-gpl \
    --enable-version3 \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libsvtav1 \
    --enable-libsnappy \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libwebp \
    --enable-encoder=libx264 \
    --enable-encoder=libx265 \
    --enable-encoder=libvpx_vp9 \
    --enable-encoder=libsvtav1 \
    --enable-encoder=prores_ks \
    --enable-encoder=hap \
    --enable-encoder=dnxhd \
    --enable-encoder=ffv1 \
    --enable-encoder=utvideo \
    --enable-encoder=mjpeg \
    --enable-encoder=png \
    --enable-encoder=tiff \
    --enable-encoder=dpx \
    --enable-encoder=libwebp \
    --enable-encoder=aac \
    --enable-encoder=libmp3lame \
    --enable-encoder=libopus \
    --enable-encoder=flac \
    --enable-encoder=pcm_s16le \
    --enable-encoder=pcm_s24le \
    --enable-muxer=mov \
    --enable-muxer=mp4 \
    --enable-muxer=matroska \
    --enable-muxer=webm \
    --enable-muxer=avi \
    --enable-muxer=mxf \
    --enable-muxer=mxf_opatom \
    --enable-muxer=wav \
    --enable-muxer=image2 \
    --enable-demuxer=mov \
    --enable-demuxer=matroska \
    --enable-demuxer=image2 \
    --enable-decoder=h264 \
    --enable-decoder=hevc \
    --enable-decoder=vp9 \
    --enable-decoder=prores \
    --enable-decoder=png \
    --enable-filter=scale \
    --enable-filter=fps \
    --enable-filter=colorspace \
    --enable-filter=loudnorm \
    --enable-filter=aresample \
    --enable-filter=format \
    --enable-protocol=file \
    --extra-cflags="-O3 -I/opt/x264/include -I/opt/x265/include -I/opt/vpx/include -I/opt/svtav1/include -I/opt/snappy/include -I/opt/lame/include -I/opt/opus/include -I/opt/webp/include" \
    --extra-ldflags="-L/opt/x264/lib -L/opt/x265/lib -L/opt/vpx/lib -L/opt/svtav1/lib -L/opt/snappy/lib -L/opt/lame/lib -L/opt/opus/lib -L/opt/webp/lib" \
    --nm="llvm-nm" \
    --ar="emar" \
    --ranlib="emranlib" \
    --cc="emcc" \
    --cxx="em++"

RUN emmake make -j$(nproc)

# ============ CREATE WASM MODULE ============

WORKDIR /src
COPY build-wasm.sh .
RUN chmod +x build-wasm.sh && ./build-wasm.sh

# Output is in /src/dist/
```

---

## 7. Implementation Phases

### Phase 1: Core Build (Week 1)
- [ ] Set up Docker build environment
- [ ] Build essential libs: x264, snappy, opus
- [ ] Configure FFmpeg with minimal codecs
- [ ] Create basic WASM output
- [ ] Test H.264 + HAP encoding

### Phase 2: Professional Codecs (Week 2)
- [ ] Add ProRes encoder
- [ ] Add DNxHR encoder
- [ ] Add FFV1 (lossless)
- [ ] Test container compatibility (MOV, MXF)

### Phase 3: Image Sequences (Week 3)
- [ ] Add OpenEXR library
- [ ] Add DPX support
- [ ] Add PNG/TIFF/WebP sequences
- [ ] Implement batch frame export

### Phase 4: Integration (Week 4)
- [ ] Create TypeScript FFmpegBridge class
- [ ] Update ExportPanel with new codecs
- [ ] Add progress reporting
- [ ] Memory optimization (streaming)
- [ ] Web Worker integration

### Phase 5: Testing & Optimization (Week 5)
- [ ] Test all codec combinations
- [ ] Profile memory usage
- [ ] Optimize WASM size (tree-shaking)
- [ ] Create presets (YouTube, Vimeo, etc.)
- [ ] Documentation

---

## 8. Size Optimization

### Minimal Build (~8 MB)
- H.264, HAP, ProRes only
- MOV container only
- AAC audio only

### Standard Build (~20 MB)
- All professional video codecs
- All containers
- AAC + Opus audio

### Full Build (~30 MB)
- Everything including SVT-AV1
- Image sequence support (EXR, DPX)
- All filters

### Loading Strategy
```typescript
// Lazy load FFmpeg only when needed
const loadFFmpeg = async () => {
  const { FFmpegBridge } = await import('./engine/ffmpeg/FFmpegBridge');
  return new FFmpegBridge();
};

// In ExportPanel
const handleProExport = async () => {
  setLoadingFFmpeg(true);
  const ffmpeg = await loadFFmpeg();
  setLoadingFFmpeg(false);
  // ... proceed with export
};
```

---

## 9. References

- [FFmpeg WASM Official](https://github.com/ffmpegwasm/ffmpeg.wasm)
- [Build Guide](https://thamizhelango.medium.com/compiling-ffmpeg-to-webassembly-a-complete-guide-faa6a10f9cd8)
- [Jerome Wu's Tutorial](https://jeromewu.github.io/build-ffmpeg-webassembly-version-part-1-preparation/)
- [HAP Codec](https://hap.video/using-hap)
- [DNxHR Encoding Guide](https://academysoftwarefoundation.github.io/EncodingGuidelines/EncodeDNXHD.html)
- [VVC/H.266 Status](https://www.streamingmedia.com/Articles/Editorial/Featured-Articles/The-State-of-the-Video-Codec-Market-2025-168628.aspx)
- [AV2 Development](https://en.wikipedia.org/wiki/AV2)
- [NotchLC Codec](https://notchlc.notch.one/)
