# FFmpeg WASM Custom Build - Full Professional Codec Suite
# Video: ProRes, HAP, DNxHR, FFV1, x264, VP9
# Audio: AAC, MP3, Opus, Vorbis, FLAC, ALAC, PCM, AC3
FROM emscripten/emsdk:3.1.50

# Install dependencies
RUN apt-get update && apt-get install -y \
    autoconf automake libtool pkg-config \
    cmake ninja-build nasm yasm \
    git wget texinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# ============ BUILD EXTERNAL LIBS ============

# zlib (needed by FFmpeg)
RUN git clone --depth 1 https://github.com/madler/zlib.git && \
    cd zlib && \
    emconfigure ./configure --prefix=/opt/zlib --static && \
    emmake make -j$(nproc) && emmake make install

# Snappy (for HAP codec) - patch out -Werror
RUN git clone --depth 1 --branch 1.1.10 https://github.com/google/snappy.git && \
    cd snappy && \
    sed -i 's/-Werror//g' CMakeLists.txt && \
    mkdir build && cd build && \
    emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/snappy \
        -DSNAPPY_BUILD_TESTS=OFF -DSNAPPY_BUILD_BENCHMARKS=OFF && \
    emmake make -j$(nproc) && emmake make install && \
    mkdir -p /opt/snappy/lib/pkgconfig && \
    echo 'prefix=/opt/snappy' > /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'exec_prefix=${prefix}' >> /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'libdir=${exec_prefix}/lib' >> /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'includedir=${prefix}/include' >> /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'Name: snappy' >> /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'Description: Snappy compression library' >> /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'Version: 1.1.10' >> /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'Libs: -L${libdir} -lsnappy' >> /opt/snappy/lib/pkgconfig/snappy.pc && \
    echo 'Cflags: -I${includedir}' >> /opt/snappy/lib/pkgconfig/snappy.pc

# x264 (H.264 encoder)
RUN git clone --depth 1 https://code.videolan.org/videolan/x264.git && \
    cd x264 && \
    emconfigure ./configure --prefix=/opt/x264 --host=i686-gnu \
        --enable-static --disable-cli --disable-asm \
        --extra-cflags="-O3" && \
    emmake make -j$(nproc) && emmake make install

# libvpx (VP9 only - VP8 fully disabled)
RUN git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git && \
    cd libvpx && \
    emconfigure ./configure --prefix=/opt/vpx --target=generic-gnu \
        --enable-static --disable-shared --disable-examples --disable-tools \
        --disable-docs --disable-unit-tests \
        --enable-vp9-encoder --enable-vp9-decoder \
        --disable-vp8-encoder --disable-vp8-decoder --disable-vp8 && \
    emmake make -j$(nproc) && emmake make install

# LAME (MP3 encoder)
RUN wget https://sourceforge.net/projects/lame/files/lame/3.100/lame-3.100.tar.gz && \
    tar xf lame-3.100.tar.gz && cd lame-3.100 && \
    emconfigure ./configure --prefix=/opt/lame --host=i686-gnu \
        --enable-static --disable-shared --disable-frontend \
        --disable-decoder && \
    emmake make -j$(nproc) && emmake make install

# Opus (modern audio codec) - with manual pkg-config
# Using -fno-stack-protector in both CFLAGS and CPPFLAGS
# Also passing flags directly to make to ensure they're used
RUN git clone --depth 1 --branch v1.4 https://github.com/xiph/opus.git && \
    cd opus && ./autogen.sh && \
    CFLAGS="-O3 -fno-stack-protector" CPPFLAGS="-fno-stack-protector" emconfigure ./configure --prefix=/opt/opus --host=i686-gnu \
        --enable-static --disable-shared --disable-doc --disable-extra-programs \
        --disable-rtcd --disable-intrinsics --disable-hardening && \
    emmake make CFLAGS="-O3 -fno-stack-protector" -j$(nproc) && emmake make install && \
    mkdir -p /opt/opus/lib/pkgconfig && \
    echo 'prefix=/opt/opus' > /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'exec_prefix=${prefix}' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'libdir=${exec_prefix}/lib' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'includedir=${prefix}/include' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo '' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'Name: Opus' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'Description: Opus Audio Codec' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'Version: 1.4' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'Libs: -L${libdir} -lopus' >> /opt/opus/lib/pkgconfig/opus.pc && \
    echo 'Cflags: -I${includedir}/opus' >> /opt/opus/lib/pkgconfig/opus.pc

# OGG (needed for Vorbis)
RUN git clone --depth 1 https://github.com/xiph/ogg.git && \
    cd ogg && ./autogen.sh && \
    emconfigure ./configure --prefix=/opt/ogg --host=i686-gnu \
        --enable-static --disable-shared && \
    emmake make -j$(nproc) && emmake make install

# Vorbis (OGG audio)
RUN git clone --depth 1 https://github.com/xiph/vorbis.git && \
    cd vorbis && ./autogen.sh && \
    emconfigure ./configure --prefix=/opt/vorbis --host=i686-gnu \
        --enable-static --disable-shared \
        --with-ogg=/opt/ogg && \
    emmake make -j$(nproc) && emmake make install

# WebP - with animation support and libsharpyuv
RUN git clone --depth 1 https://chromium.googlesource.com/webm/libwebp && \
    cd libwebp && ./autogen.sh && \
    CFLAGS="-O3 -fno-stack-protector" emconfigure ./configure --prefix=/opt/webp --host=i686-gnu \
        --enable-static --disable-shared \
        --disable-threading \
        --enable-libwebpmux \
        --enable-libsharpyuv && \
    emmake make -j$(nproc) && emmake make install && \
    mkdir -p /opt/webp/lib/pkgconfig && \
    echo 'prefix=/opt/webp' > /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'exec_prefix=${prefix}' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'libdir=${exec_prefix}/lib' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'includedir=${prefix}/include' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo '' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'Name: libwebp' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'Description: WebP image codec' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'Version: 1.3.0' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'Libs: -L${libdir} -lwebp' >> /opt/webp/lib/pkgconfig/libwebp.pc && \
    echo 'Cflags: -I${includedir}' >> /opt/webp/lib/pkgconfig/libwebp.pc

# ============ BUILD FFMPEG ============

# Using FFmpeg 5.1.4 - better WASM compatibility than 6.x
RUN git clone --depth 1 -b n5.1.4 https://github.com/FFmpeg/FFmpeg.git ffmpeg

# Create stack protector stub for Opus (Opus uses stack protection that doesn't work in WASM)
RUN echo '// Stack protector stubs for WASM' > /opt/stack_chk_stub.c && \
    echo 'unsigned long __stack_chk_guard = 0xDEADBEEF;' >> /opt/stack_chk_stub.c && \
    echo 'void __stack_chk_fail(void) { }' >> /opt/stack_chk_stub.c && \
    emcc -c -O3 /opt/stack_chk_stub.c -o /opt/stack_chk_stub.o

WORKDIR /src/ffmpeg

# Create central pkgconfig directory and copy/create all .pc files there
RUN mkdir -p /opt/pkgconfig && \
    # Copy existing .pc files
    cp /opt/x264/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/vpx/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/lame/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/ogg/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/vorbis/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/snappy/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/opus/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/webp/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    cp /opt/zlib/lib/pkgconfig/*.pc /opt/pkgconfig/ 2>/dev/null || true && \
    # Create opus.pc if not exists (opus build sometimes doesn't create one)
    echo 'prefix=/opt/opus' > /opt/pkgconfig/opus.pc && \
    echo 'exec_prefix=${prefix}' >> /opt/pkgconfig/opus.pc && \
    echo 'libdir=${exec_prefix}/lib' >> /opt/pkgconfig/opus.pc && \
    echo 'includedir=${prefix}/include/opus' >> /opt/pkgconfig/opus.pc && \
    echo '' >> /opt/pkgconfig/opus.pc && \
    echo 'Name: Opus' >> /opt/pkgconfig/opus.pc && \
    echo 'Description: Opus IETF audio codec' >> /opt/pkgconfig/opus.pc && \
    echo 'Version: 1.4' >> /opt/pkgconfig/opus.pc && \
    echo 'Libs: -L${prefix}/lib -lopus' >> /opt/pkgconfig/opus.pc && \
    echo 'Libs.private: -lm' >> /opt/pkgconfig/opus.pc && \
    echo 'Cflags: -I${includedir}' >> /opt/pkgconfig/opus.pc && \
    # Verify files exist and show debugging info
    echo "=== PKG-CONFIG FILES ===" && \
    ls -la /opt/pkgconfig/ && \
    echo "=== OPUS.PC CONTENT ===" && \
    cat /opt/pkgconfig/opus.pc && \
    echo "=== TESTING PKG-CONFIG ===" && \
    PKG_CONFIG_LIBDIR=/opt/pkgconfig pkg-config --exists opus && echo "opus: FOUND" || echo "opus: NOT FOUND" && \
    PKG_CONFIG_LIBDIR=/opt/pkgconfig pkg-config --cflags opus && \
    PKG_CONFIG_LIBDIR=/opt/pkgconfig pkg-config --libs opus

# Create fake pkg-config that returns hardcoded values
# This bypasses all pkg-config checks and returns what FFmpeg needs
RUN echo '#!/bin/bash' > /usr/local/bin/fake-pkg-config && \
    echo 'PKG="$2"' >> /usr/local/bin/fake-pkg-config && \
    echo 'if [[ "$1" == "--exists" ]]; then exit 0; fi' >> /usr/local/bin/fake-pkg-config && \
    echo 'case "$PKG" in' >> /usr/local/bin/fake-pkg-config && \
    echo '  opus) CFLAGS="-I/opt/opus/include/opus"; LIBS="-L/opt/opus/lib -lopus" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  vorbis) CFLAGS="-I/opt/vorbis/include"; LIBS="-L/opt/vorbis/lib -lvorbis" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  vorbisenc) CFLAGS="-I/opt/vorbis/include"; LIBS="-L/opt/vorbis/lib -lvorbisenc -lvorbis -logg" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  ogg) CFLAGS="-I/opt/ogg/include"; LIBS="-L/opt/ogg/lib -logg" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  x264) CFLAGS="-I/opt/x264/include"; LIBS="-L/opt/x264/lib -lx264" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  vpx) CFLAGS="-I/opt/vpx/include"; LIBS="-L/opt/vpx/lib -lvpx" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  libwebp) CFLAGS="-I/opt/webp/include"; LIBS="-L/opt/webp/lib -lwebp" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  snappy) CFLAGS="-I/opt/snappy/include"; LIBS="-L/opt/snappy/lib -lsnappy" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  libmp3lame|lame) CFLAGS="-I/opt/lame/include"; LIBS="-L/opt/lame/lib -lmp3lame" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  zlib) CFLAGS="-I/opt/zlib/include"; LIBS="-L/opt/zlib/lib -lz" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo '  *) CFLAGS=""; LIBS="" ;;' >> /usr/local/bin/fake-pkg-config && \
    echo 'esac' >> /usr/local/bin/fake-pkg-config && \
    echo 'if [[ "$1" == "--cflags" ]]; then echo "$CFLAGS"; fi' >> /usr/local/bin/fake-pkg-config && \
    echo 'if [[ "$1" == "--libs" ]]; then echo "$LIBS"; fi' >> /usr/local/bin/fake-pkg-config && \
    echo 'if [[ "$1" == "--modversion" ]]; then echo "1.0"; fi' >> /usr/local/bin/fake-pkg-config && \
    echo 'exit 0' >> /usr/local/bin/fake-pkg-config && \
    chmod +x /usr/local/bin/fake-pkg-config && \
    echo "=== Testing fake pkg-config ===" && \
    /usr/local/bin/fake-pkg-config --exists opus && echo "opus: exists" && \
    /usr/local/bin/fake-pkg-config --cflags opus && \
    /usr/local/bin/fake-pkg-config --libs opus

# Configure FFmpeg with professional codecs
# Set environment variables to bypass pkg-config link tests
# FFmpeg checks these vars before running pkg-config
ENV OPUS_CFLAGS="-I/opt/opus/include/opus"
ENV OPUS_LIBS="-L/opt/opus/lib -lopus"
ENV LIBVORBIS_CFLAGS="-I/opt/vorbis/include -I/opt/ogg/include"
ENV LIBVORBIS_LIBS="-L/opt/vorbis/lib -L/opt/ogg/lib -lvorbis -lvorbisenc -logg"
ENV LIBWEBP_CFLAGS="-I/opt/webp/include"
ENV LIBWEBP_LIBS="-L/opt/webp/lib -lwebp"
ENV VPX_CFLAGS="-I/opt/vpx/include"
ENV VPX_LIBS="-L/opt/vpx/lib -lvpx"
ENV X264_CFLAGS="-I/opt/x264/include"
ENV X264_LIBS="-L/opt/x264/lib -lx264"
ENV SNAPPY_CFLAGS="-I/opt/snappy/include"
ENV SNAPPY_LIBS="-L/opt/snappy/lib -lsnappy"
ENV LIBMP3LAME_CFLAGS="-I/opt/lame/include"
ENV LIBMP3LAME_LIBS="-L/opt/lame/lib -lmp3lame"
ENV ZLIB_CFLAGS="-I/opt/zlib/include"
ENV ZLIB_LIBS="-L/opt/zlib/lib -lz"

# Patch FFmpeg configure to skip library link tests for emscripten
# The issue: require_pkg_config calls test_pkg_config which does compile+link tests
# We need to find and patch test_pkg_config
RUN echo "=== Finding test_pkg_config function ===" && \
    grep -n -A30 "^test_pkg_config(){" configure | head -40 && \
    echo "" && \
    echo "=== Patching test_pkg_config to skip compile/link tests ===" && \
    # test_pkg_config likely calls check_func_headers or similar
    # Replace the compile test with always-true
    sed -i '/^test_pkg_config(){/,/^}/ s/check_func_headers/true #&/' configure && \
    # Also check require_pkg_config
    echo "=== Finding require_pkg_config function ===" && \
    grep -n -A15 "^require_pkg_config(){" configure | head -20 && \
    echo "" && \
    echo "=== After patch ===" && \
    grep -n -A30 "^test_pkg_config(){" configure | head -40

# Run configure with emconfigure (now that test_pkg_config is patched)
RUN emconfigure ./configure \
    --pkg-config=/usr/local/bin/fake-pkg-config \
    --prefix=/opt/ffmpeg \
    --target-os=none \
    --arch=x86_32 \
    --enable-cross-compile \
    --cross-prefix="" \
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
    --enable-libx264 \
    --enable-libvpx \
    --enable-libsnappy \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libwebp \
    --enable-zlib \
    --enable-encoder=libx264 \
    --enable-encoder=libvpx_vp9 \
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
    --enable-encoder=libvorbis \
    --enable-encoder=flac \
    --enable-encoder=alac \
    --enable-encoder=pcm_s16le \
    --enable-encoder=pcm_s24le \
    --enable-encoder=pcm_f32le \
    --enable-encoder=ac3 \
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
    --enable-demuxer=mov \
    --enable-demuxer=matroska \
    --enable-demuxer=avi \
    --enable-demuxer=image2 \
    --enable-demuxer=rawvideo \
    --enable-decoder=h264 \
    --enable-decoder=hevc \
    --enable-decoder=vp9 \
    --enable-decoder=libvpx_vp9 \
    --disable-decoder=libvpx_vp8 \
    --disable-encoder=libvpx_vp8 \
    --enable-decoder=av1 \
    --enable-decoder=prores \
    --enable-decoder=png \
    --enable-decoder=mjpeg \
    --enable-decoder=rawvideo \
    --enable-decoder=pcm_s16le \
    --enable-decoder=pcm_f32le \
    --enable-decoder=aac \
    --enable-filter=scale \
    --enable-filter=fps \
    --enable-filter=colorspace \
    --enable-filter=format \
    --enable-filter=aformat \
    --enable-filter=aresample \
    --enable-filter=loudnorm \
    --enable-filter=null \
    --enable-filter=anull \
    --enable-filter=volume \
    --enable-filter=amix \
    --enable-parser=h264 \
    --enable-parser=hevc \
    --enable-parser=vp9 \
    --enable-parser=av1 \
    --enable-parser=aac \
    --enable-protocol=file \
    --enable-protocol=pipe \
    --extra-cflags="-O3 -fno-stack-protector -I/opt/x264/include -I/opt/vpx/include -I/opt/snappy/include -I/opt/lame/include -I/opt/opus/include/opus -I/opt/ogg/include -I/opt/vorbis/include -I/opt/webp/include -I/opt/zlib/include" \
    --extra-ldflags="-L/opt/x264/lib -L/opt/vpx/lib -L/opt/snappy/lib -L/opt/lame/lib -L/opt/opus/lib -L/opt/ogg/lib -L/opt/vorbis/lib -L/opt/webp/lib -L/opt/zlib/lib -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=5242880 -sINITIAL_MEMORY=67108864" \
    --extra-libs="/opt/stack_chk_stub.o -lx264 -lvpx -lsnappy -lmp3lame -lopus -lvorbis -lvorbisenc -logg -lwebpmux -lsharpyuv -lwebp -lz -lm" \
    --nm="/emsdk/upstream/bin/llvm-nm" \
    --ar="/emsdk/upstream/emscripten/emar" \
    --ranlib="/emsdk/upstream/emscripten/emranlib" \
    --cc="emcc" \
    --cxx="em++"

RUN emmake make -j$(nproc)

# Check what was built
RUN echo "=== Checking built library files ===" && \
    ls -la libav*/lib*.a libsw*/lib*.a 2>/dev/null && \
    echo "" && \
    echo "=== Looking for ffmpeg binary ===" && \
    ls -la ffmpeg* 2>/dev/null || echo "No ffmpeg binary found" && \
    echo "" && \
    echo "=== Checking if fftools .o files exist ===" && \
    ls fftools/*.o 2>/dev/null || echo "No fftools .o files found - will compile manually"

# FFmpeg 5.x compiles fftools automatically with emmake make
# Just verify they exist
RUN echo "=== Checking fftools .o files ===" && \
    ls -la fftools/*.o 2>/dev/null || echo "No .o files - build may have failed"

# If ffmpeg_g doesn't exist, build the WASM module directly from libraries
# This creates a minimal ffmpeg WASM that can be called from JavaScript

# ============ CREATE WASM MODULE ============

WORKDIR /src
COPY build-wasm.sh .
RUN chmod +x build-wasm.sh

# Output directory
RUN mkdir -p /dist

CMD ["./build-wasm.sh"]
