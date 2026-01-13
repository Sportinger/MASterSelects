#!/bin/bash
# FFmpeg WASM Build Script
# Creates ffmpeg-core.js and ffmpeg-core.wasm with ASYNCIFY support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../public/ffmpeg"

echo "============================================"
echo "FFmpeg WASM Builder with ASYNCIFY"
echo "============================================"
echo ""
echo "This will build FFmpeg WASM with:"
echo "  - ProRes, DNxHR, HAP, FFV1 encoders"
echo "  - H.264, VP9, MJPEG encoders"
echo "  - ASYNCIFY for streaming encode (low memory)"
echo "  - Single-threaded (ASYNCIFY incompatible with pthreads)"
echo ""
echo "Build time: ~15-30 minutes"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Install with: sudo apt install docker.io"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build -t ffmpeg-wasm-async "$SCRIPT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract built files
echo ""
echo "Extracting built files..."
docker run --rm -v "$OUTPUT_DIR:/output_mount" ffmpeg-wasm-async \
    sh -c "cp /output/* /output_mount/"

# Check output
if [ -f "$OUTPUT_DIR/ffmpeg-core.js" ] && [ -f "$OUTPUT_DIR/ffmpeg-core.wasm" ]; then
    echo ""
    echo "============================================"
    echo "Build successful!"
    echo "============================================"
    echo ""
    echo "Files created:"
    ls -lh "$OUTPUT_DIR"/ffmpeg-core.*
    echo ""
    echo "WASM size: $(du -h "$OUTPUT_DIR/ffmpeg-core.wasm" | cut -f1)"
    echo ""
    echo "To use: restart your dev server"
else
    echo ""
    echo "ERROR: Build failed - output files not found"
    exit 1
fi
