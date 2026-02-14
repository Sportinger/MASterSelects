/**
 * directional_blur.cu -- Directional (Motion) Blur (CUDA)
 *
 * Applies a linear motion blur along a specified direction. For each pixel,
 * samples are taken along a line defined by angle_rad and averaged over the
 * blur distance. This simulates motion blur in a single direction.
 *
 * The kernel samples evenly along the direction vector from -distance/2 to
 * +distance/2, producing a symmetric blur centered on each pixel.
 *
 * Parameters
 * ----------
 *   input     : Source RGBA buffer (4 bytes per pixel).
 *   output    : Destination RGBA buffer (4 bytes per pixel).
 *   width     : Image width in pixels.
 *   height    : Image height in pixels.
 *   angle_rad : Blur direction in radians. 0 = horizontal right.
 *   distance  : Blur length in pixels (number of samples along direction).
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
extern "C" __global__ void directional_blur(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float angle_rad,
    float distance)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // -----------------------------------------------------------------------
    // 1. Compute direction vector from angle
    // -----------------------------------------------------------------------
    const float dir_x = cosf(angle_rad);
    const float dir_y = sinf(angle_rad);

    // -----------------------------------------------------------------------
    // 2. Determine number of samples (minimum 1)
    // -----------------------------------------------------------------------
    const int num_samples = max(static_cast<int>(distance + 0.5f), 1);
    const float half_dist = distance * 0.5f;

    // -----------------------------------------------------------------------
    // 3. Sample along direction and accumulate
    // -----------------------------------------------------------------------
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;

    for (int i = 0; i < num_samples; ++i)
    {
        // Parametric position along the blur line: -half_dist to +half_dist
        const float t = (num_samples > 1)
            ? (-half_dist + distance * static_cast<float>(i) / static_cast<float>(num_samples - 1))
            : 0.0f;

        const int sx = min(max(static_cast<int>(static_cast<float>(x) + dir_x * t + 0.5f), 0), width - 1);
        const int sy = min(max(static_cast<int>(static_cast<float>(y) + dir_y * t + 0.5f), 0), height - 1);

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
