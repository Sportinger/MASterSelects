/**
 * composite.cu -- Alpha Blend Compositing Kernel (CUDA)
 *
 * Performs source-over alpha blending of a foreground RGBA layer onto a
 * background RGBA layer, with an additional global opacity multiplier.
 *
 * Source-over formula (premultiplied-style, per channel):
 *
 *   fg_a_eff = (fg.a / 255.0) * opacity
 *   out.rgb  = fg.rgb * fg_a_eff + bg.rgb * (1.0 - fg_a_eff)
 *   out.a    = fg_a_eff + bg.a * (1.0 - fg_a_eff)    (clamped to 255)
 *
 * Both layers must have the same dimensions and RGBA byte layout.
 *
 * Parameters
 * ----------
 *   fg       : Foreground RGBA buffer (4 bytes per pixel).
 *   bg       : Background RGBA buffer (4 bytes per pixel).
 *   output   : Output RGBA buffer (4 bytes per pixel). May alias bg.
 *   width    : Frame width in pixels.
 *   height   : Frame height in pixels.
 *   fg_pitch : Byte stride of one row in the foreground buffer.
 *   bg_pitch : Byte stride of one row in the background buffer.
 *   out_pitch: Byte stride of one row in the output buffer.
 *   opacity  : Global foreground opacity in [0.0, 1.0].
 */

#include <cstdint>

// ---------------------------------------------------------------------------
// Device helper: clamp float to [0, 255] and convert to uint8_t
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t clamp_u8(float v)
{
    return static_cast<uint8_t>(fminf(fmaxf(v + 0.5f, 0.0f), 255.0f));
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void alpha_blend(
    const uint8_t* __restrict__ fg,
    const uint8_t* __restrict__ bg,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    int fg_pitch,
    int bg_pitch,
    int out_pitch,
    float opacity)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // -----------------------------------------------------------------------
    // 1. Read foreground pixel
    // -----------------------------------------------------------------------
    const int fg_offset = y * fg_pitch + x * 4;
    const float fg_r = static_cast<float>(fg[fg_offset + 0]);
    const float fg_g = static_cast<float>(fg[fg_offset + 1]);
    const float fg_b = static_cast<float>(fg[fg_offset + 2]);
    const float fg_a = static_cast<float>(fg[fg_offset + 3]);

    // -----------------------------------------------------------------------
    // 2. Read background pixel
    // -----------------------------------------------------------------------
    const int bg_offset = y * bg_pitch + x * 4;
    const float bg_r = static_cast<float>(bg[bg_offset + 0]);
    const float bg_g = static_cast<float>(bg[bg_offset + 1]);
    const float bg_b = static_cast<float>(bg[bg_offset + 2]);
    const float bg_a = static_cast<float>(bg[bg_offset + 3]);

    // -----------------------------------------------------------------------
    // 3. Compute effective foreground alpha (normalized to [0, 1])
    //    Combines per-pixel alpha with global opacity
    // -----------------------------------------------------------------------
    const float fg_a_eff = (fg_a / 255.0f) * opacity;
    const float inv_a    = 1.0f - fg_a_eff;

    // -----------------------------------------------------------------------
    // 4. Source-over blend
    // -----------------------------------------------------------------------
    const float out_r = fg_r * fg_a_eff + bg_r * inv_a;
    const float out_g = fg_g * fg_a_eff + bg_g * inv_a;
    const float out_b = fg_b * fg_a_eff + bg_b * inv_a;
    const float out_a = (fg_a_eff + (bg_a / 255.0f) * inv_a) * 255.0f;

    // -----------------------------------------------------------------------
    // 5. Write output
    // -----------------------------------------------------------------------
    const int out_offset = y * out_pitch + x * 4;

    output[out_offset + 0] = clamp_u8(out_r);
    output[out_offset + 1] = clamp_u8(out_g);
    output[out_offset + 2] = clamp_u8(out_b);
    output[out_offset + 3] = clamp_u8(out_a);
}
