/**
 * luma_key.cu -- Luminance Key Effect (CUDA)
 *
 * Sets the alpha channel of each pixel based on its luminance value.
 * Pixels brighter or darker than the threshold become transparent,
 * allowing background layers to show through. Useful for removing
 * black or white backgrounds.
 *
 * Luminance is computed using BT.709 coefficients:
 *   luma = 0.2126 * R + 0.7152 * G + 0.0722 * B
 *
 * The softness parameter creates a gradual transition around the
 * threshold rather than a hard edge.
 *
 * Parameters
 * ----------
 *   input     : Source RGBA buffer (4 bytes per pixel).
 *   output    : Destination RGBA buffer (4 bytes per pixel).
 *   width     : Image width in pixels.
 *   height    : Image height in pixels.
 *   threshold : Luminance cutoff point, normalized [0, 1].
 *   softness  : Edge softness / transition width [0, 1].
 *   invert    : If 0, darks are keyed (removed). If 1, brights are keyed.
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
extern "C" __global__ void luma_key(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float threshold,
    float softness,
    int invert)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Read pixel and compute luminance (BT.709)
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]) / 255.0f;
    const float g = static_cast<float>(input[idx + 1]) / 255.0f;
    const float b = static_cast<float>(input[idx + 2]) / 255.0f;
    const float a = static_cast<float>(input[idx + 3]);

    const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    // -----------------------------------------------------------------------
    // 2. Compute alpha mask based on luminance vs threshold
    //    Without invert: pixels below threshold are transparent
    //    With invert: pixels above threshold are transparent
    // -----------------------------------------------------------------------
    float mask;
    const float half_soft = softness * 0.5f;

    if (half_soft > 0.0001f)
    {
        // Smooth transition using smoothstep-like ramp
        const float lower = threshold - half_soft;
        const float upper = threshold + half_soft;

        if (luma <= lower)
            mask = 0.0f;
        else if (luma >= upper)
            mask = 1.0f;
        else
        {
            // Linear ramp between lower and upper
            mask = (luma - lower) / (upper - lower);
        }
    }
    else
    {
        // Hard edge
        mask = (luma >= threshold) ? 1.0f : 0.0f;
    }

    // Invert if requested
    if (invert != 0)
        mask = 1.0f - mask;

    // -----------------------------------------------------------------------
    // 3. Apply mask to alpha (multiply with existing alpha)
    // -----------------------------------------------------------------------
    const float out_a = a * mask;

    // -----------------------------------------------------------------------
    // 4. Write output (RGB preserved, alpha modified)
    // -----------------------------------------------------------------------
    output[idx + 0] = input[idx + 0];
    output[idx + 1] = input[idx + 1];
    output[idx + 2] = input[idx + 2];
    output[idx + 3] = clamp_u8(out_a);
}
