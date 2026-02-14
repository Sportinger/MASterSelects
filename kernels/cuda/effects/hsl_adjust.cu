/**
 * hsl_adjust.cu -- HSL (Hue / Saturation / Lightness) Adjustment (CUDA)
 *
 * Converts each pixel from RGB to HSL, applies hue shift, saturation
 * adjustment, and lightness adjustment, then converts back to RGB.
 * Alpha channel is preserved unchanged.
 *
 * HSL conversion follows the standard hexagonal model.
 *
 * Parameters
 * ----------
 *   input      : Source RGBA buffer (4 bytes per pixel).
 *   output     : Destination RGBA buffer (4 bytes per pixel).
 *   width      : Image width in pixels.
 *   height     : Image height in pixels.
 *   hue_shift  : Hue rotation in degrees [-180, 180].
 *   saturation : Relative saturation adjustment [-1, 1]. 0 = no change.
 *   lightness  : Relative lightness adjustment [-1, 1]. 0 = no change.
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
// Device helper: convert RGB [0,1] to HSL
// H in [0, 360), S in [0, 1], L in [0, 1]
// ---------------------------------------------------------------------------
__device__ __forceinline__ void rgb_to_hsl(
    float r, float g, float b,
    float& h, float& s, float& l)
{
    const float cmax = fmaxf(r, fmaxf(g, b));
    const float cmin = fminf(r, fminf(g, b));
    const float delta = cmax - cmin;

    l = (cmax + cmin) * 0.5f;

    if (delta < 1e-6f)
    {
        h = 0.0f;
        s = 0.0f;
        return;
    }

    // Saturation
    if (l < 0.5f)
        s = delta / (cmax + cmin);
    else
        s = delta / (2.0f - cmax - cmin);

    // Hue
    if (cmax == r)
        h = fmodf((g - b) / delta + 6.0f, 6.0f) * 60.0f;
    else if (cmax == g)
        h = ((b - r) / delta + 2.0f) * 60.0f;
    else
        h = ((r - g) / delta + 4.0f) * 60.0f;
}

// ---------------------------------------------------------------------------
// Device helper: convert HSL to RGB [0,1]
// ---------------------------------------------------------------------------
__device__ __forceinline__ float hue_to_rgb(float p, float q, float t)
{
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

__device__ __forceinline__ void hsl_to_rgb(
    float h, float s, float l,
    float& r, float& g, float& b)
{
    if (s < 1e-6f)
    {
        r = g = b = l;
        return;
    }

    const float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
    const float p = 2.0f * l - q;
    const float h_norm = h / 360.0f;

    r = hue_to_rgb(p, q, h_norm + 1.0f / 3.0f);
    g = hue_to_rgb(p, q, h_norm);
    b = hue_to_rgb(p, q, h_norm - 1.0f / 3.0f);
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void hsl_adjust(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float hue_shift,
    float saturation,
    float lightness)
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
    // 2. Convert RGB to HSL
    // -----------------------------------------------------------------------
    float h, s, l;
    rgb_to_hsl(r, g, b, h, s, l);

    // -----------------------------------------------------------------------
    // 3. Apply adjustments
    // -----------------------------------------------------------------------
    h = fmodf(h + hue_shift + 360.0f, 360.0f);
    s = fminf(fmaxf(s + saturation, 0.0f), 1.0f);
    l = fminf(fmaxf(l + lightness, 0.0f), 1.0f);

    // -----------------------------------------------------------------------
    // 4. Convert back to RGB
    // -----------------------------------------------------------------------
    float out_r, out_g, out_b;
    hsl_to_rgb(h, s, l, out_r, out_g, out_b);

    // -----------------------------------------------------------------------
    // 5. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(out_r * 255.0f);
    output[idx + 1] = clamp_u8(out_g * 255.0f);
    output[idx + 2] = clamp_u8(out_b * 255.0f);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
