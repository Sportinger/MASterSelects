/**
 * gaussian_blur.cu -- Separable Gaussian Blur (CUDA)
 *
 * Implements a two-pass separable Gaussian blur on RGBA images.
 * Pass 1 (horizontal) writes to an intermediate buffer, then
 * Pass 2 (vertical) reads from the intermediate buffer to produce
 * the final blurred output.
 *
 * Both passes use shared memory to cache the tile plus halo region,
 * reducing global memory reads. The Gaussian weights are computed
 * on-the-fly from sigma (no LUT needed for typical radii).
 *
 * Launch configuration: 16x16 thread blocks per pass.
 *
 * Parameters (same for both passes)
 * ----------
 *   input  : Source RGBA buffer (4 bytes per pixel).
 *   output : Destination RGBA buffer (4 bytes per pixel).
 *   width  : Image width in pixels.
 *   height : Image height in pixels.
 *   radius : Integer blur radius in pixels (kernel size = 2*radius + 1).
 *   sigma  : Gaussian standard deviation. Controls blur spread.
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
// Device helper: compute Gaussian weight for a given distance and sigma
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gaussian_weight(int dist, float sigma)
{
    const float d = static_cast<float>(dist);
    return expf(-(d * d) / (2.0f * sigma * sigma));
}

// ---------------------------------------------------------------------------
// Kernel: Horizontal Gaussian blur pass
//
// Each thread reads from a shared memory tile that includes the halo region
// (radius pixels on each side). Threads at the edges of the tile load the
// halo pixels from global memory.
// ---------------------------------------------------------------------------
extern "C" __global__ void gaussian_blur_h(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    int radius,
    float sigma)
{
    // Clamp radius to maximum supported
    radius = min(radius, MAX_RADIUS);

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height)
        return;

    // Shared memory tile: BLOCK_DIM + 2 * radius pixels wide, 4 channels
    extern __shared__ float tile[];
    const int tile_width = blockDim.x + 2 * radius;

    // Local thread position within the tile (offset by radius for halo)
    const int tile_x = threadIdx.x + radius;

    // -----------------------------------------------------------------------
    // 1. Load center pixel into shared memory
    // -----------------------------------------------------------------------
    const int cx = min(x, width - 1);
    const int center_idx = (y * width + cx) * 4;
    tile[tile_x * 4 + 0] = static_cast<float>(input[center_idx + 0]);
    tile[tile_x * 4 + 1] = static_cast<float>(input[center_idx + 1]);
    tile[tile_x * 4 + 2] = static_cast<float>(input[center_idx + 2]);
    tile[tile_x * 4 + 3] = static_cast<float>(input[center_idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Load left halo
    // -----------------------------------------------------------------------
    if (threadIdx.x < radius)
    {
        const int halo_x = max(static_cast<int>(blockIdx.x * blockDim.x) - radius + static_cast<int>(threadIdx.x), 0);
        const int halo_idx = (y * width + halo_x) * 4;
        const int tile_pos = threadIdx.x;
        tile[tile_pos * 4 + 0] = static_cast<float>(input[halo_idx + 0]);
        tile[tile_pos * 4 + 1] = static_cast<float>(input[halo_idx + 1]);
        tile[tile_pos * 4 + 2] = static_cast<float>(input[halo_idx + 2]);
        tile[tile_pos * 4 + 3] = static_cast<float>(input[halo_idx + 3]);
    }

    // -----------------------------------------------------------------------
    // 3. Load right halo
    // -----------------------------------------------------------------------
    if (threadIdx.x >= blockDim.x - radius)
    {
        const int halo_x = min(static_cast<int>(blockIdx.x * blockDim.x) + static_cast<int>(threadIdx.x) + radius, width - 1);
        const int halo_idx = (y * width + halo_x) * 4;
        const int tile_pos = threadIdx.x + 2 * radius;
        tile[tile_pos * 4 + 0] = static_cast<float>(input[halo_idx + 0]);
        tile[tile_pos * 4 + 1] = static_cast<float>(input[halo_idx + 1]);
        tile[tile_pos * 4 + 2] = static_cast<float>(input[halo_idx + 2]);
        tile[tile_pos * 4 + 3] = static_cast<float>(input[halo_idx + 3]);
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // 4. Apply horizontal Gaussian filter
    // -----------------------------------------------------------------------
    if (x >= width)
        return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -radius; k <= radius; ++k)
    {
        const float w = gaussian_weight(k, sigma);
        const int sample_pos = (tile_x + k) * 4;

        sum_r += tile[sample_pos + 0] * w;
        sum_g += tile[sample_pos + 1] * w;
        sum_b += tile[sample_pos + 2] * w;
        sum_a += tile[sample_pos + 3] * w;
        weight_sum += w;
    }

    // Normalize by total weight
    const float inv_weight = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;

    // -----------------------------------------------------------------------
    // 5. Write output
    // -----------------------------------------------------------------------
    const int out_idx = (y * width + x) * 4;
    output[out_idx + 0] = clamp_u8(sum_r * inv_weight);
    output[out_idx + 1] = clamp_u8(sum_g * inv_weight);
    output[out_idx + 2] = clamp_u8(sum_b * inv_weight);
    output[out_idx + 3] = clamp_u8(sum_a * inv_weight);
}

// ---------------------------------------------------------------------------
// Kernel: Vertical Gaussian blur pass
//
// Same approach as horizontal but operates along columns. Uses shared memory
// with BLOCK_DIM + 2*radius pixels tall.
// ---------------------------------------------------------------------------
extern "C" __global__ void gaussian_blur_v(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    int radius,
    float sigma)
{
    // Clamp radius to maximum supported
    radius = min(radius, MAX_RADIUS);

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width)
        return;

    // Shared memory tile: BLOCK_DIM + 2 * radius pixels tall, 4 channels
    extern __shared__ float tile[];
    const int tile_height = blockDim.y + 2 * radius;

    // Local thread position within the tile (offset by radius for halo)
    const int tile_y = threadIdx.y + radius;

    // -----------------------------------------------------------------------
    // 1. Load center pixel into shared memory
    // -----------------------------------------------------------------------
    const int cy = min(y, height - 1);
    const int center_idx = (cy * width + x) * 4;
    tile[tile_y * 4 + 0] = static_cast<float>(input[center_idx + 0]);
    tile[tile_y * 4 + 1] = static_cast<float>(input[center_idx + 1]);
    tile[tile_y * 4 + 2] = static_cast<float>(input[center_idx + 2]);
    tile[tile_y * 4 + 3] = static_cast<float>(input[center_idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Load top halo
    // -----------------------------------------------------------------------
    if (threadIdx.y < radius)
    {
        const int halo_y = max(static_cast<int>(blockIdx.y * blockDim.y) - radius + static_cast<int>(threadIdx.y), 0);
        const int halo_idx = (halo_y * width + x) * 4;
        const int tile_pos = threadIdx.y;
        tile[tile_pos * 4 + 0] = static_cast<float>(input[halo_idx + 0]);
        tile[tile_pos * 4 + 1] = static_cast<float>(input[halo_idx + 1]);
        tile[tile_pos * 4 + 2] = static_cast<float>(input[halo_idx + 2]);
        tile[tile_pos * 4 + 3] = static_cast<float>(input[halo_idx + 3]);
    }

    // -----------------------------------------------------------------------
    // 3. Load bottom halo
    // -----------------------------------------------------------------------
    if (threadIdx.y >= blockDim.y - radius)
    {
        const int halo_y = min(static_cast<int>(blockIdx.y * blockDim.y) + static_cast<int>(threadIdx.y) + radius, height - 1);
        const int halo_idx = (halo_y * width + x) * 4;
        const int tile_pos = threadIdx.y + 2 * radius;
        tile[tile_pos * 4 + 0] = static_cast<float>(input[halo_idx + 0]);
        tile[tile_pos * 4 + 1] = static_cast<float>(input[halo_idx + 1]);
        tile[tile_pos * 4 + 2] = static_cast<float>(input[halo_idx + 2]);
        tile[tile_pos * 4 + 3] = static_cast<float>(input[halo_idx + 3]);
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // 4. Apply vertical Gaussian filter
    // -----------------------------------------------------------------------
    if (y >= height)
        return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -radius; k <= radius; ++k)
    {
        const float w = gaussian_weight(k, sigma);
        const int sample_pos = (tile_y + k) * 4;

        sum_r += tile[sample_pos + 0] * w;
        sum_g += tile[sample_pos + 1] * w;
        sum_b += tile[sample_pos + 2] * w;
        sum_a += tile[sample_pos + 3] * w;
        weight_sum += w;
    }

    // Normalize by total weight
    const float inv_weight = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;

    // -----------------------------------------------------------------------
    // 5. Write output
    // -----------------------------------------------------------------------
    const int out_idx = (y * width + x) * 4;
    output[out_idx + 0] = clamp_u8(sum_r * inv_weight);
    output[out_idx + 1] = clamp_u8(sum_g * inv_weight);
    output[out_idx + 2] = clamp_u8(sum_b * inv_weight);
    output[out_idx + 3] = clamp_u8(sum_a * inv_weight);
}
