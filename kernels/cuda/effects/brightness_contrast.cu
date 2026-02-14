/**
 * brightness_contrast.cu -- Brightness & Contrast Adjustment (CUDA)
 *
 * Adjusts the brightness and contrast of an RGBA image. The adjustment is
 * performed in normalized [0, 1] space per channel (RGB). Alpha is preserved.
 *
 * Formula:
 *   normalized = pixel / 255.0
 *   adjusted   = (normalized - 0.5) * contrast + 0.5 + brightness
 *   output     = clamp(adjusted, 0, 1) * 255
 *
 * Parameters
 * ----------
 *   input      : Source RGBA buffer (4 bytes per pixel).
 *   output     : Destination RGBA buffer (4 bytes per pixel).
 *   width      : Image width in pixels.
 *   height     : Image height in pixels.
 *   brightness : Brightness offset in [-1, 1]. 0 = no change.
 *   contrast   : Contrast multiplier in [0, 3]. 1 = no change.
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
extern "C" __global__ void brightness_contrast(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float brightness,
    float contrast)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Read RGBA and normalize to [0, 1]
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]) / 255.0f;
    const float g = static_cast<float>(input[idx + 1]) / 255.0f;
    const float b = static_cast<float>(input[idx + 2]) / 255.0f;
    const float a = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Apply contrast (scale around midpoint 0.5) then brightness offset
    // -----------------------------------------------------------------------
    const float out_r = (r - 0.5f) * contrast + 0.5f + brightness;
    const float out_g = (g - 0.5f) * contrast + 0.5f + brightness;
    const float out_b = (b - 0.5f) * contrast + 0.5f + brightness;

    // -----------------------------------------------------------------------
    // 3. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(fminf(fmaxf(out_r, 0.0f), 1.0f) * 255.0f);
    output[idx + 1] = clamp_u8(fminf(fmaxf(out_g, 0.0f), 1.0f) * 255.0f);
    output[idx + 2] = clamp_u8(fminf(fmaxf(out_b, 0.0f), 1.0f) * 255.0f);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
