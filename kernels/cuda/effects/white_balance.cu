/**
 * white_balance.cu -- White Balance (Color Temperature + Tint) Adjustment (CUDA)
 *
 * Adjusts the white balance of an RGBA image by computing RGB multipliers
 * from a target color temperature (in Kelvin) and a green-magenta tint
 * offset. The approach uses an approximation of the Planckian locus to
 * derive per-channel scale factors.
 *
 * Temperature model (simplified Tanner Helland approximation):
 *   For T < 6600K (warm):
 *     R = 1.0
 *     G = 0.390 * ln(T/100) - 0.632
 *     B = 0.543 * ln(T/100 - 10) - 1.186   (if T > 2000K)
 *   For T >= 6600K (cool):
 *     R = 1.293 * ((T/100 - 60)^-0.1332)
 *     G = 1.130 * ((T/100 - 60)^-0.0755)
 *     B = 1.0
 *
 * The multipliers are normalized so that 6500K produces (1, 1, 1).
 * Tint is applied as a green-magenta shift on the green channel.
 *
 * Alpha channel is preserved unchanged.
 *
 * Parameters
 * ----------
 *   input       : Source RGBA buffer (4 bytes per pixel).
 *   output      : Destination RGBA buffer (4 bytes per pixel).
 *   width       : Image width in pixels.
 *   height      : Image height in pixels.
 *   temperature : Color temperature in Kelvin [2000, 12000]. 6500 = daylight.
 *   tint        : Green-magenta shift [-1, 1]. 0 = no tint.
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
// Device helper: compute RGB multipliers from color temperature (Kelvin)
// Returns multipliers normalized relative to D65 (6500K)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void temp_to_rgb(float temp_k, float& r_mult, float& g_mult, float& b_mult)
{
    // Clamp temperature to valid range
    const float t = fminf(fmaxf(temp_k, 1000.0f), 15000.0f);
    const float t100 = t / 100.0f;

    float r, g, b;

    // Red channel
    if (t100 <= 66.0f)
    {
        r = 1.0f;
    }
    else
    {
        r = 1.2929f * powf(t100 - 60.0f, -0.1332f);
    }

    // Green channel
    if (t100 <= 66.0f)
    {
        g = 0.3900f * logf(t100) - 0.6318f;
    }
    else
    {
        g = 1.1298f * powf(t100 - 60.0f, -0.0755f);
    }

    // Blue channel
    if (t100 >= 66.0f)
    {
        b = 1.0f;
    }
    else if (t100 <= 19.0f)
    {
        b = 0.0f;
    }
    else
    {
        b = 0.5432f * logf(t100 - 10.0f) - 1.1862f;
    }

    // Clamp to [0, 1]
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    // Normalize so that 6500K produces (1, 1, 1)
    // Pre-computed reference at 6500K:
    //   r_ref = 1.0, g_ref = 0.3900 * ln(65) - 0.6318 ~ 0.9967, b_ref = 0.5432 * ln(55) - 1.1862 ~ 0.9908
    const float r_ref = 1.0f;
    const float g_ref = 0.3900f * logf(65.0f) - 0.6318f;
    const float b_ref = 0.5432f * logf(55.0f) - 1.1862f;

    r_mult = (r_ref > 0.0f) ? (r / r_ref) : 1.0f;
    g_mult = (g_ref > 0.0f) ? (g / g_ref) : 1.0f;
    b_mult = (b_ref > 0.0f) ? (b / b_ref) : 1.0f;
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void white_balance(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float temperature,
    float tint)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Compute white balance RGB multipliers from temperature
    // -----------------------------------------------------------------------
    float r_mult, g_mult, b_mult;
    temp_to_rgb(temperature, r_mult, g_mult, b_mult);

    // Apply tint: shift green channel (positive = more green, negative = magenta)
    g_mult *= (1.0f + tint);

    // -----------------------------------------------------------------------
    // 2. Read RGBA and normalize RGB to [0, 1]
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]) / 255.0f;
    const float g = static_cast<float>(input[idx + 1]) / 255.0f;
    const float b = static_cast<float>(input[idx + 2]) / 255.0f;
    const float a = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 3. Apply white balance multipliers
    // -----------------------------------------------------------------------
    const float out_r = fminf(fmaxf(r * r_mult, 0.0f), 1.0f);
    const float out_g = fminf(fmaxf(g * g_mult, 0.0f), 1.0f);
    const float out_b = fminf(fmaxf(b * b_mult, 0.0f), 1.0f);

    // -----------------------------------------------------------------------
    // 4. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(out_r * 255.0f);
    output[idx + 1] = clamp_u8(out_g * 255.0f);
    output[idx + 2] = clamp_u8(out_b * 255.0f);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
