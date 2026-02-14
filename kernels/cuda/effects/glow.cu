/**
 * glow.cu -- Multi-Pass Glow / Bloom Effect (CUDA)
 *
 * Implements a glow/bloom effect via four kernel passes:
 *   1. glow_threshold  — Extract bright pixels above a luminance threshold
 *   2. glow_blur_h     — Horizontal Gaussian blur on the thresholded image
 *   3. glow_blur_v     — Vertical Gaussian blur on the horizontally blurred image
 *   4. glow_composite  — Additive blend of the blurred glow onto the original
 *
 * Host-side dispatch order:
 *   glow_threshold(input, temp1, ...)
 *   glow_blur_h(temp1, temp2, ...)
 *   glow_blur_v(temp2, temp1, ...)     // can reuse temp1
 *   glow_composite(input, temp1, output, ...)
 *
 * All buffers are RGBA8, same width/height dimensions.
 *
 * Parameters
 * ----------
 *   glow_threshold:
 *     input     : Source RGBA buffer.
 *     output    : Destination RGBA buffer (bright pixels only).
 *     width     : Image width in pixels.
 *     height    : Image height in pixels.
 *     threshold : Luminance threshold [0, 1]. Only pixels above this glow.
 *
 *   glow_blur_h / glow_blur_v:
 *     input     : Source RGBA buffer.
 *     output    : Destination RGBA buffer.
 *     width     : Image width in pixels.
 *     height    : Image height in pixels.
 *     radius    : Integer blur radius in pixels.
 *
 *   glow_composite:
 *     original  : Original (unmodified) RGBA buffer.
 *     glowed    : Blurred glow RGBA buffer.
 *     output    : Final composited RGBA buffer.
 *     width     : Image width in pixels.
 *     height    : Image height in pixels.
 *     intensity : Glow intensity / blend strength [0, 3].
 */

#include <cstdint>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define BLOCK_DIM 16
#define MAX_RADIUS 64

// ---------------------------------------------------------------------------
// Device helper: clamp float to [0, 255] and convert to uint8_t
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t clamp_u8(float v)
{
    return static_cast<uint8_t>(fminf(fmaxf(v + 0.5f, 0.0f), 255.0f));
}

// ---------------------------------------------------------------------------
// Device helper: Gaussian weight
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gaussian_weight(int dist, float sigma)
{
    const float d = static_cast<float>(dist);
    return expf(-(d * d) / (2.0f * sigma * sigma));
}

// ===========================================================================
// Pass 1: Threshold — extract bright pixels
// ===========================================================================
extern "C" __global__ void glow_threshold(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float threshold)
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
    // 2. Output bright pixels only; dark pixels become black transparent
    // -----------------------------------------------------------------------
    if (luma > threshold)
    {
        // Scale brightness by how much it exceeds threshold (soft glow)
        const float factor = (luma - threshold) / (1.0f - threshold + 1e-6f);

        output[idx + 0] = clamp_u8(r * factor * 255.0f);
        output[idx + 1] = clamp_u8(g * factor * 255.0f);
        output[idx + 2] = clamp_u8(b * factor * 255.0f);
        output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
    }
    else
    {
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
        output[idx + 3] = 0;
    }
}

// ===========================================================================
// Pass 2: Horizontal Gaussian blur
// ===========================================================================
extern "C" __global__ void glow_blur_h(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int clamped_radius = min(radius, MAX_RADIUS);
    const float sigma = fmaxf(static_cast<float>(clamped_radius) / 3.0f, 1.0f);

    // -----------------------------------------------------------------------
    // Sample horizontally and accumulate weighted sum
    // -----------------------------------------------------------------------
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -clamped_radius; k <= clamped_radius; ++k)
    {
        const int sx = min(max(x + k, 0), width - 1);
        const int sample_idx = (y * width + sx) * 4;
        const float w = gaussian_weight(k, sigma);

        sum_r += static_cast<float>(input[sample_idx + 0]) * w;
        sum_g += static_cast<float>(input[sample_idx + 1]) * w;
        sum_b += static_cast<float>(input[sample_idx + 2]) * w;
        sum_a += static_cast<float>(input[sample_idx + 3]) * w;
        weight_sum += w;
    }

    const float inv_w = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;
    const int out_idx = (y * width + x) * 4;

    output[out_idx + 0] = clamp_u8(sum_r * inv_w);
    output[out_idx + 1] = clamp_u8(sum_g * inv_w);
    output[out_idx + 2] = clamp_u8(sum_b * inv_w);
    output[out_idx + 3] = clamp_u8(sum_a * inv_w);
}

// ===========================================================================
// Pass 3: Vertical Gaussian blur
// ===========================================================================
extern "C" __global__ void glow_blur_v(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int clamped_radius = min(radius, MAX_RADIUS);
    const float sigma = fmaxf(static_cast<float>(clamped_radius) / 3.0f, 1.0f);

    // -----------------------------------------------------------------------
    // Sample vertically and accumulate weighted sum
    // -----------------------------------------------------------------------
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -clamped_radius; k <= clamped_radius; ++k)
    {
        const int sy = min(max(y + k, 0), height - 1);
        const int sample_idx = (sy * width + x) * 4;
        const float w = gaussian_weight(k, sigma);

        sum_r += static_cast<float>(input[sample_idx + 0]) * w;
        sum_g += static_cast<float>(input[sample_idx + 1]) * w;
        sum_b += static_cast<float>(input[sample_idx + 2]) * w;
        sum_a += static_cast<float>(input[sample_idx + 3]) * w;
        weight_sum += w;
    }

    const float inv_w = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;
    const int out_idx = (y * width + x) * 4;

    output[out_idx + 0] = clamp_u8(sum_r * inv_w);
    output[out_idx + 1] = clamp_u8(sum_g * inv_w);
    output[out_idx + 2] = clamp_u8(sum_b * inv_w);
    output[out_idx + 3] = clamp_u8(sum_a * inv_w);
}

// ===========================================================================
// Pass 4: Composite — additive blend of glow onto original
// ===========================================================================
extern "C" __global__ void glow_composite(
    const uint8_t* __restrict__ original,
    const uint8_t* __restrict__ glowed,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float intensity)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // Additive blend: original + glow * intensity
    // -----------------------------------------------------------------------
    const float orig_r = static_cast<float>(original[idx + 0]);
    const float orig_g = static_cast<float>(original[idx + 1]);
    const float orig_b = static_cast<float>(original[idx + 2]);
    const float orig_a = static_cast<float>(original[idx + 3]);

    const float glow_r = static_cast<float>(glowed[idx + 0]);
    const float glow_g = static_cast<float>(glowed[idx + 1]);
    const float glow_b = static_cast<float>(glowed[idx + 2]);

    output[idx + 0] = clamp_u8(orig_r + glow_r * intensity);
    output[idx + 1] = clamp_u8(orig_g + glow_g * intensity);
    output[idx + 2] = clamp_u8(orig_b + glow_b * intensity);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(orig_a + 0.5f, 0.0f), 255.0f));
}
