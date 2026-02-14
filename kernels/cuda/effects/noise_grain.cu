/**
 * noise_grain.cu -- Film Grain / Noise Generator (CUDA)
 *
 * Adds pseudo-random film grain noise to an RGBA image. Uses a fast
 * GPU-friendly hash function (Wang hash) for deterministic pseudo-random
 * number generation per pixel. The seed parameter allows variation across
 * frames for animated grain.
 *
 * When monochrome is enabled, the same noise value is applied to all three
 * RGB channels. When disabled, independent noise is generated per channel,
 * producing colored grain.
 *
 * Alpha channel is preserved unchanged.
 *
 * Parameters
 * ----------
 *   input      : Source RGBA buffer (4 bytes per pixel).
 *   output     : Destination RGBA buffer (4 bytes per pixel).
 *   width      : Image width in pixels.
 *   height     : Image height in pixels.
 *   amount     : Noise intensity [0, 1]. 0 = no noise, 1 = full grain.
 *   monochrome : If non-zero, same noise for all RGB channels (grayscale grain).
 *                If zero, independent noise per channel (color grain).
 *   seed       : Random seed for variation across frames.
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
// Device helper: Wang hash — fast integer hash for GPU pseudo-random numbers
// Returns a 32-bit hash from a 32-bit input
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int wang_hash(unsigned int seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

// ---------------------------------------------------------------------------
// Device helper: convert hash to float in [-1, 1]
// ---------------------------------------------------------------------------
__device__ __forceinline__ float hash_to_float(unsigned int h)
{
    return (static_cast<float>(h) / 2147483647.0f) - 1.0f;
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void noise_grain(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float amount,
    int monochrome,
    unsigned int seed)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Read original pixel
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]);
    const float g = static_cast<float>(input[idx + 1]);
    const float b = static_cast<float>(input[idx + 2]);
    const float a = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Generate pseudo-random noise using Wang hash
    //    Each pixel gets a unique hash input based on position + seed
    // -----------------------------------------------------------------------
    const unsigned int pixel_id = static_cast<unsigned int>(y * width + x);
    const unsigned int base_hash = wang_hash(pixel_id ^ seed);

    // Noise amount scaled to pixel range [0, 255]
    const float noise_scale = amount * 255.0f;

    float noise_r, noise_g, noise_b;

    if (monochrome != 0)
    {
        // Monochrome: same noise for all channels
        const float n = hash_to_float(base_hash) * noise_scale;
        noise_r = n;
        noise_g = n;
        noise_b = n;
    }
    else
    {
        // Color grain: independent noise per channel
        noise_r = hash_to_float(base_hash) * noise_scale;
        noise_g = hash_to_float(wang_hash(base_hash + 1u)) * noise_scale;
        noise_b = hash_to_float(wang_hash(base_hash + 2u)) * noise_scale;
    }

    // -----------------------------------------------------------------------
    // 3. Add noise and clamp
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(r + noise_r);
    output[idx + 1] = clamp_u8(g + noise_g);
    output[idx + 2] = clamp_u8(b + noise_b);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
