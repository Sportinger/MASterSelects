/**
 * curves.cu -- Per-Channel Curves / Color Multiply + Gamma (CUDA)
 *
 * Applies per-channel color multiplication and a shared gamma correction
 * to an RGBA image. This provides basic curves-style color grading.
 *
 * Formula (per channel):
 *   out.r = pow(clamp(in.r/255 * red_mult,   0, 1), 1/gamma) * 255
 *   out.g = pow(clamp(in.g/255 * green_mult, 0, 1), 1/gamma) * 255
 *   out.b = pow(clamp(in.b/255 * blue_mult,  0, 1), 1/gamma) * 255
 *
 * Alpha channel is preserved unchanged.
 *
 * Parameters
 * ----------
 *   input      : Source RGBA buffer (4 bytes per pixel).
 *   output     : Destination RGBA buffer (4 bytes per pixel).
 *   width      : Image width in pixels.
 *   height     : Image height in pixels.
 *   red_mult   : Red channel multiplier (0 to 3). 1.0 = no change.
 *   green_mult : Green channel multiplier (0 to 3). 1.0 = no change.
 *   blue_mult  : Blue channel multiplier (0 to 3). 1.0 = no change.
 *   gamma      : Gamma exponent (0.1 to 5). 1.0 = linear / no change.
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
extern "C" __global__ void curves_adjust(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float red_mult,
    float green_mult,
    float blue_mult,
    float gamma_val)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Read RGBA and normalize RGB to [0, 1]
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]) / 255.0f;
    const float g = static_cast<float>(input[idx + 1]) / 255.0f;
    const float b = static_cast<float>(input[idx + 2]) / 255.0f;
    const float a = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Apply per-channel multiply, clamp to [0, 1]
    // -----------------------------------------------------------------------
    const float r_scaled = fminf(fmaxf(r * red_mult,   0.0f), 1.0f);
    const float g_scaled = fminf(fmaxf(g * green_mult, 0.0f), 1.0f);
    const float b_scaled = fminf(fmaxf(b * blue_mult,  0.0f), 1.0f);

    // -----------------------------------------------------------------------
    // 3. Apply gamma correction: pow(value, 1/gamma)
    //    Guard against gamma <= 0 by treating as 1.0 (no correction)
    // -----------------------------------------------------------------------
    const float inv_gamma = (gamma_val > 0.0f) ? (1.0f / gamma_val) : 1.0f;

    const float out_r = powf(r_scaled, inv_gamma);
    const float out_g = powf(g_scaled, inv_gamma);
    const float out_b = powf(b_scaled, inv_gamma);

    // -----------------------------------------------------------------------
    // 4. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(out_r * 255.0f);
    output[idx + 1] = clamp_u8(out_g * 255.0f);
    output[idx + 2] = clamp_u8(out_b * 255.0f);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
