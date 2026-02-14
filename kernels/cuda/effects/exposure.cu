/**
 * exposure.cu -- Exposure & Gamma Adjustment (CUDA)
 *
 * Applies photographic exposure compensation (in EV stops) and gamma
 * correction to an RGBA image. Exposure is applied as a linear multiplier
 * of pow(2, exposure), then gamma is applied as a power curve.
 *
 * Formula:
 *   linear   = (pixel / 255.0) * pow(2.0, exposure)
 *   corrected = pow(clamp(linear, 0, 1), 1 / gamma)
 *   output   = corrected * 255
 *
 * Alpha channel is preserved unchanged.
 *
 * Parameters
 * ----------
 *   input    : Source RGBA buffer (4 bytes per pixel).
 *   output   : Destination RGBA buffer (4 bytes per pixel).
 *   width    : Image width in pixels.
 *   height   : Image height in pixels.
 *   exposure : Exposure in EV stops [-5, 5]. 0 = no change.
 *   gamma    : Gamma exponent (0.1 to 5). 1.0 = linear / no change.
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
extern "C" __global__ void exposure_adjust(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float exposure,
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
    // 2. Apply exposure: multiply by 2^exposure (EV stops)
    // -----------------------------------------------------------------------
    const float ev_mult = exp2f(exposure);
    const float r_exp = fminf(fmaxf(r * ev_mult, 0.0f), 1.0f);
    const float g_exp = fminf(fmaxf(g * ev_mult, 0.0f), 1.0f);
    const float b_exp = fminf(fmaxf(b * ev_mult, 0.0f), 1.0f);

    // -----------------------------------------------------------------------
    // 3. Apply gamma correction: pow(value, 1/gamma)
    // -----------------------------------------------------------------------
    const float inv_gamma = (gamma_val > 0.0f) ? (1.0f / gamma_val) : 1.0f;

    const float out_r = powf(r_exp, inv_gamma);
    const float out_g = powf(g_exp, inv_gamma);
    const float out_b = powf(b_exp, inv_gamma);

    // -----------------------------------------------------------------------
    // 4. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(out_r * 255.0f);
    output[idx + 1] = clamp_u8(out_g * 255.0f);
    output[idx + 2] = clamp_u8(out_b * 255.0f);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
