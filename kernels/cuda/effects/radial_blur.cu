/**
 * radial_blur.cu -- Radial (Spin) Blur (CUDA)
 *
 * Applies a radial blur that spins pixels around a center point. Each pixel
 * is blurred along a circular arc centered at (center_x, center_y). Pixels
 * farther from the center experience more blur (longer arc), simulating a
 * spinning/rotational motion effect.
 *
 * The kernel samples along the tangential direction (perpendicular to the
 * radial line from center to pixel) and averages the samples.
 *
 * Parameters
 * ----------
 *   input    : Source RGBA buffer (4 bytes per pixel).
 *   output   : Destination RGBA buffer (4 bytes per pixel).
 *   width    : Image width in pixels.
 *   height   : Image height in pixels.
 *   center_x : Blur center X, normalized [0, 1]. 0.5 = image center.
 *   center_y : Blur center Y, normalized [0, 1]. 0.5 = image center.
 *   amount   : Blur strength / angle in radians. Controls how far along
 *              the arc to sample. Higher = more blur.
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
extern "C" __global__ void radial_blur(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float center_x,
    float center_y,
    float amount)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // -----------------------------------------------------------------------
    // 1. Compute center in pixel coordinates
    // -----------------------------------------------------------------------
    const float cx = center_x * static_cast<float>(width);
    const float cy = center_y * static_cast<float>(height);

    // Vector from center to current pixel
    const float dx = static_cast<float>(x) - cx;
    const float dy = static_cast<float>(y) - cy;

    // -----------------------------------------------------------------------
    // 2. Determine number of samples based on distance and blur amount
    //    Pixels closer to center need fewer samples
    // -----------------------------------------------------------------------
    const float dist = sqrtf(dx * dx + dy * dy);
    const int num_samples = max(static_cast<int>(fminf(dist * fabsf(amount) * 0.1f, 32.0f)), 1);

    // -----------------------------------------------------------------------
    // 3. Sample along circular arc around center
    // -----------------------------------------------------------------------
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;

    // Base angle of this pixel relative to center
    const float base_angle = atan2f(dy, dx);

    for (int i = 0; i < num_samples; ++i)
    {
        // Spread samples from -amount/2 to +amount/2 around the base angle
        const float t = (num_samples > 1)
            ? (-0.5f + static_cast<float>(i) / static_cast<float>(num_samples - 1))
            : 0.0f;
        const float angle = base_angle + amount * t;

        // Compute sample position on the arc (same distance from center)
        const int sx = min(max(static_cast<int>(cx + cosf(angle) * dist + 0.5f), 0), width - 1);
        const int sy = min(max(static_cast<int>(cy + sinf(angle) * dist + 0.5f), 0), height - 1);

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
