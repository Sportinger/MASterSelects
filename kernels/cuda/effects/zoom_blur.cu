/**
 * zoom_blur.cu -- Zoom (Radial Motion) Blur (CUDA)
 *
 * Applies a zoom blur effect that simulates rapid camera zoom (dolly zoom).
 * For each pixel, samples are taken along the line from the pixel to the
 * center point, producing a radial streaking effect. Pixels farther from
 * the center are blurred more.
 *
 * Parameters
 * ----------
 *   input    : Source RGBA buffer (4 bytes per pixel).
 *   output   : Destination RGBA buffer (4 bytes per pixel).
 *   width    : Image width in pixels.
 *   height   : Image height in pixels.
 *   center_x : Zoom center X, normalized [0, 1]. 0.5 = image center.
 *   center_y : Zoom center Y, normalized [0, 1]. 0.5 = image center.
 *   strength : Zoom blur strength [0, 1]. Controls how far along the
 *              radial line to sample. 0 = no blur.
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
extern "C" __global__ void zoom_blur(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float center_x,
    float center_y,
    float strength)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // -----------------------------------------------------------------------
    // 1. Compute center in pixel coordinates and direction to center
    // -----------------------------------------------------------------------
    const float cx = center_x * static_cast<float>(width);
    const float cy = center_y * static_cast<float>(height);

    // Vector from pixel to center
    const float dx = cx - static_cast<float>(x);
    const float dy = cy - static_cast<float>(y);

    // Distance from pixel to center (used to scale blur amount)
    const float dist = sqrtf(dx * dx + dy * dy);

    // -----------------------------------------------------------------------
    // 2. Determine number of samples (more samples for farther pixels)
    // -----------------------------------------------------------------------
    const int num_samples = max(static_cast<int>(fminf(dist * strength * 0.05f, 32.0f)), 1);

    // -----------------------------------------------------------------------
    // 3. Sample along the line from pixel toward center
    // -----------------------------------------------------------------------
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;

    for (int i = 0; i < num_samples; ++i)
    {
        // Parametric position: 0 = original pixel, 1 = shifted by strength * direction
        const float t = strength * static_cast<float>(i) / static_cast<float>(num_samples);

        const int sx = min(max(static_cast<int>(static_cast<float>(x) + dx * t + 0.5f), 0), width - 1);
        const int sy = min(max(static_cast<int>(static_cast<float>(y) + dy * t + 0.5f), 0), height - 1);

        const int sample_idx = (sy * width + sx) * 4;

        sum_r += static_cast<float>(input[sample_idx + 0]);
        sum_g += static_cast<float>(input[sample_idx + 1]);
        sum_b += static_cast<float>(input[sample_idx + 2]);
        sum_a += static_cast<float>(input[sample_idx + 3]);
    }

    // -----------------------------------------------------------------------
    // 4. Average and write output
    // -----------------------------------------------------------------------
    const float inv_n = 1.0f / static_cast<float>(num_samples);
    const int out_idx = (y * width + x) * 4;

    output[out_idx + 0] = clamp_u8(sum_r * inv_n);
    output[out_idx + 1] = clamp_u8(sum_g * inv_n);
    output[out_idx + 2] = clamp_u8(sum_b * inv_n);
    output[out_idx + 3] = clamp_u8(sum_a * inv_n);
}
