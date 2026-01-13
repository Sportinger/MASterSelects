# FFmpeg WASM Build with ASYNCIFY support for streaming encode
# Build with: docker build -t ffmpeg-wasm-async .
# Extract with: docker run --rm -v $(pwd)/output:/output ffmpeg-wasm-async

FROM emscripten/emsdk:3.1.50

# Install build dependencies
RUN apt-get update && apt-get install -y \
    autoconf automake libtool pkg-config \
    git make nasm yasm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ============================================
# Build zlib (for PNG compression)
# ============================================
RUN git clone --depth 1 https://github.com/madler/zlib.git && \
    cd zlib && \
    emconfigure ./configure --prefix=/opt/zlib --static && \
    emmake make -j$(nproc) && \
    emmake make install

# ============================================
# Create stub for __stack_chk_fail
# ============================================
RUN echo 'void __stack_chk_fail(void) {}' > /opt/stack_chk_stub.c && \
    emcc -c /opt/stack_chk_stub.c -o /opt/stack_chk_stub.o

# ============================================
# Build FFmpeg with ASYNCIFY
# ============================================
RUN git clone --depth 1 --branch n5.1.4 https://github.com/FFmpeg/FFmpeg.git ffmpeg

WORKDIR /build/ffmpeg

# Configure FFmpeg with ASYNCIFY for streaming support
# Using ONLY native FFmpeg encoders to avoid pkg-config issues
# Key flags:
# - ASYNCIFY: Allows yielding back to JS during encode
# - ASYNCIFY_STACK_SIZE: Stack space for async operations
# - ALLOW_MEMORY_GROWTH: Dynamic memory allocation
# - MODULARIZE: Creates createFFmpegCore() function
#
# Included encoders (all native - no external libs):
# - Video: ProRes, DNxHR, FFV1, UTVideo, MJPEG, PNG
# - Audio: AAC, FLAC, ALAC, PCM variants, AC3
#
# Note: H.264 (libx264) and VP9 (libvpx) skipped to avoid pkg-config issues
#       Can be added back later with proper build setup
RUN emconfigure ./configure \
    --prefix=/opt/ffmpeg \
    --target-os=none \
    --arch=x86_32 \
    --enable-cross-compile \
    --cross-prefix= \
    --disable-x86asm \
    --disable-inline-asm \
    --disable-stripping \
    --enable-ffmpeg \
    --disable-ffplay \
    --disable-ffprobe \
    --disable-doc \
    --disable-debug \
    --disable-runtime-cpudetect \
    --disable-autodetect \
    --disable-pthreads \
    --enable-gpl \
    --enable-version3 \
    \
    --enable-zlib \
    \
    --enable-encoder=prores_ks \
    --enable-encoder=dnxhd \
    --enable-encoder=ffv1 \
    --enable-encoder=utvideo \
    --enable-encoder=mjpeg \
    --enable-encoder=png \
    --enable-encoder=tiff \
    --enable-encoder=dpx \
    --enable-encoder=rawvideo \
    --enable-encoder=aac \
    --enable-encoder=flac \
    --enable-encoder=alac \
    --enable-encoder=pcm_s16le \
    --enable-encoder=pcm_s24le \
    --enable-encoder=pcm_f32le \
    --enable-encoder=ac3 \
    \
    --enable-muxer=mov \
    --enable-muxer=mp4 \
    --enable-muxer=matroska \
    --enable-muxer=avi \
    --enable-muxer=mxf \
    --enable-muxer=mxf_opatom \
    --enable-muxer=mpegts \
    --enable-muxer=ogg \
    --enable-muxer=wav \
    --enable-muxer=image2 \
    \
    --enable-demuxer=mov \
    --enable-demuxer=matroska \
    --enable-demuxer=avi \
    --enable-demuxer=image2 \
    --enable-demuxer=rawvideo \
    \
    --enable-decoder=h264 \
    --enable-decoder=hevc \
    --enable-decoder=vp9 \
    --enable-decoder=av1 \
    --enable-decoder=prores \
    --enable-decoder=png \
    --enable-decoder=mjpeg \
    --enable-decoder=rawvideo \
    --enable-decoder=pcm_s16le \
    --enable-decoder=pcm_f32le \
    --enable-decoder=aac \
    \
    --enable-filter=scale \
    --enable-filter=fps \
    --enable-filter=colorspace \
    --enable-filter=format \
    --enable-filter=aformat \
    --enable-filter=aresample \
    --enable-filter=null \
    --enable-filter=anull \
    --enable-filter=volume \
    --enable-filter=amix \
    \
    --enable-parser=h264 \
    --enable-parser=hevc \
    --enable-parser=vp9 \
    --enable-parser=av1 \
    --enable-parser=aac \
    \
    --enable-protocol=file \
    --enable-protocol=pipe \
    \
    --extra-cflags='-O3 -fno-stack-protector \
        -I/opt/zlib/include' \
    --extra-ldflags='-L/opt/zlib/lib \
        -sASYNCIFY \
        -sASYNCIFY_STACK_SIZE=65536 \
        -sALLOW_MEMORY_GROWTH=1 \
        -sSTACK_SIZE=5242880 \
        -sINITIAL_MEMORY=134217728 \
        -sMAXIMUM_MEMORY=4294967296 \
        -sEXPORTED_FUNCTIONS=_main \
        -sEXPORTED_RUNTIME_METHODS=FS,callMain,cwrap,ccall \
        -sMODULARIZE=1 \
        -sEXPORT_NAME=createFFmpegCore \
        -sENVIRONMENT=web,worker' \
    --extra-libs='/opt/stack_chk_stub.o -lz -lm' \
    --nm=/emsdk/upstream/bin/llvm-nm \
    --ar=/emsdk/upstream/emscripten/emar \
    --ranlib=/emsdk/upstream/emscripten/emranlib \
    --cc=emcc \
    --cxx=em++

# Build FFmpeg
RUN emmake make -j$(nproc)

# Create output directory and copy files
# Note: The WASM file might be embedded or named differently
RUN mkdir -p /output && \
    ls -la ffmpeg* && \
    cp ffmpeg /output/ffmpeg-core.js && \
    (cp ffmpeg.wasm /output/ffmpeg-core.wasm 2>/dev/null || \
     cp ffmpeg_g.wasm /output/ffmpeg-core.wasm 2>/dev/null || \
     echo "WASM file may be embedded in JS")

# Default command: copy files to mounted volume
CMD cp /output/* /output_mount/ 2>/dev/null || echo "Mount /output_mount to extract files"
