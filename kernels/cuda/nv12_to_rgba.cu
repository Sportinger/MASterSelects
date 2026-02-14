/**
 * nv12_to_rgba.cu -- NV12 to RGBA Color Space Conversion (CUDA)
 *
 * Converts hardware decoder output in NV12 format to linear RGBA suitable
 * for compositing and display. NV12 stores luma (Y) at full resolution and
 * chroma (UV / CbCr) at half resolution in both dimensions, with U and V
 * bytes interleaved.
 *
 * Color matrix: ITU-R BT.709 (standard for HD / UHD content)
 *
 *   R = 1.164 * (Y - 16) + 1.793 * (V - 128)
 *   G = 1.164 * (Y - 16) - 0.213 * (U - 128) - 0.533 * (V - 128)
 *   B = 1.164 * (Y - 16) + 2.112 * (U - 128)
 *   A = 255
 *
 * Each CUDA thread processes exactly one output pixel.
 *
 * Parameters
 * ----------
 *   y_plane   : Pointer to the Y (luma) plane.
 *   uv_plane  : Pointer to the UV (chroma) plane (interleaved Cb,Cr).
 *   rgba_out  : Pointer to the output RGBA buffer (4 bytes per pixel).
 *   width     : Frame width in pixels.
 *   height    : Frame height in pixels.
 *   y_pitch   : Byte stride of one row in the Y plane.
 *   uv_pitch  : Byte stride of one row in the UV plane.
 *   out_pitch : Byte stride of one row in the output RGBA buffer.
 */

#include <cstdint>

// ---------------------------------------------------------------------------
// Device helper: clamp a float to [0, 255] and return as uint8_t
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t clamp_u8(float v)
{
    // Use fminf/fmaxf for branchless clamping
    return static_cast<uint8_t>(fminf(fmaxf(v + 0.5f, 0.0f), 255.0f));
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void nv12_to_rgba(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    uint8_t*       __restrict__ rgba_out,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    int out_pitch)
{
    // Each thread handles one pixel at (x, y)
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // -----------------------------------------------------------------------
    // 1. Fetch luma sample (full resolution)
    // -----------------------------------------------------------------------
    const float Y = static_cast<float>(y_plane[y * y_pitch + x]);

    // -----------------------------------------------------------------------
    // 2. Fetch chroma samples (half resolution, shared by 2x2 pixel block)
    //    UV plane layout: row = y/2, each pair of bytes = (U, V) for a 2-pixel
    //    wide column group.
    // -----------------------------------------------------------------------
    const int uv_row = y >> 1;          // y / 2
    const int uv_col = (x & ~1);       // align x to even (x / 2 * 2)
    const int uv_offset = uv_row * uv_pitch + uv_col;

    const float U = static_cast<float>(uv_plane[uv_offset]);
    const float V = static_cast<float>(uv_plane[uv_offset + 1]);

    // -----------------------------------------------------------------------
    // 3. BT.709 YUV -> RGB conversion
    //    - Y is in [16..235] (studio range), offset by 16
    //    - U, V are in [16..240], centered at 128
    // -----------------------------------------------------------------------
    const float C = 1.164f * (Y - 16.0f);
    const float D = U - 128.0f;
    const float E = V - 128.0f;

    const float R = C + 1.793f * E;
    const float G = C - 0.213f * D - 0.533f * E;
    const float B = C + 2.112f * D;

    // -----------------------------------------------------------------------
    // 4. Write RGBA output (alpha = 255, fully opaque)
    // -----------------------------------------------------------------------
    const int out_offset = y * out_pitch + x * 4;

    rgba_out[out_offset + 0] = clamp_u8(R);
    rgba_out[out_offset + 1] = clamp_u8(G);
    rgba_out[out_offset + 2] = clamp_u8(B);
    rgba_out[out_offset + 3] = 255;
}
