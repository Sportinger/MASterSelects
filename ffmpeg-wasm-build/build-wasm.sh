#!/bin/bash
set -e

echo "=========================================="
echo "Building FFmpeg WASM Module"
echo "=========================================="

cd /src/ffmpeg

# Check if fftools .o files exist
echo "Checking for fftools object files..."
ls -la fftools/*.o 2>/dev/null || true

if [ ! -f "fftools/ffmpeg.o" ]; then
    echo "ERROR: fftools/ffmpeg.o not found."
    echo "Listing fftools directory:"
    ls -la fftools/
    exit 1
fi

echo "Found fftools object files, creating WASM module..."

# External library paths for linking
EXT_LIB_PATHS="-L/opt/x264/lib -L/opt/vpx/lib -L/opt/snappy/lib -L/opt/lame/lib -L/opt/opus/lib -L/opt/ogg/lib -L/opt/vorbis/lib -L/opt/webp/lib -L/opt/zlib/lib"

# External libraries (order matters for linking!)
EXT_LIBS="/opt/stack_chk_stub.o -lx264 -lvpx -lsnappy -lmp3lame -lopus -lvorbis -lvorbisenc -logg -lwebpmux -lsharpyuv -lwebp -lz"

echo "Re-linking with Emscripten flags for browser..."

# Re-link with Emscripten-specific settings
# Single-threaded build (no pthreads)
# Include all FFmpeg libraries including libavdevice and libpostproc
emcc -O3 \
    fftools/*.o \
    libavdevice/libavdevice.a \
    libavfilter/libavfilter.a \
    libavformat/libavformat.a \
    libavcodec/libavcodec.a \
    libpostproc/libpostproc.a \
    libswresample/libswresample.a \
    libswscale/libswscale.a \
    libavutil/libavutil.a \
    $EXT_LIB_PATHS \
    $EXT_LIBS \
    -o /dist/ffmpeg-core.js \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createFFmpegCore" \
    -s EXPORTED_FUNCTIONS="['_main', '_malloc', '_free']" \
    -s EXPORTED_RUNTIME_METHODS="['FS', 'callMain', 'cwrap', 'ccall', 'setValue', 'getValue', 'UTF8ToString', 'stringToUTF8', 'lengthBytesUTF8']" \
    -s INITIAL_MEMORY=67108864 \
    -s MAXIMUM_MEMORY=2147483648 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INVOKE_RUN=0 \
    -s EXIT_RUNTIME=0 \
    -s FILESYSTEM=1 \
    -s FORCE_FILESYSTEM=1 \
    -s SINGLE_FILE=0 \
    -s ASSERTIONS=0 \
    -s STACK_SIZE=5242880 \
    -lworkerfs.js

echo ""
echo "=========================================="
echo "Build complete!"
echo "Output files in /dist:"
ls -lh /dist/
echo ""
echo "WASM file size:"
du -h /dist/ffmpeg-core.wasm
echo "=========================================="
