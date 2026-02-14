/**
 * sharpen.cu -- Unsharp Mask Sharpening (CUDA)
 *
 * Applies unsharp mask sharpening to an RGBA image. The algorithm computes
 * a blurred version of each pixel using a small convolution kernel, then
 * enhances edges by adding the difference between the original and blurred
 * pixel, scaled by the `amount` parameter.
 *
 * Formula:
 *   blurred = convolution(input, kernel)
 *   output  = input + amount * (input - blurred)
 *
 * For radius 1, a 3x3 Gaussian-like kernel is used.
 * For radius 2, a 5x5 Gaussian-like kernel is used.
 * Larger radii fall back to radius 2.
 *
 * Alpha channel is preserved unchanged.
 *
 * Parameters
 * ----------
 *   input  : Source RGBA buffer (4 bytes per pixel).
 *   output : Destination RGBA buffer (4 bytes per pixel).
 *   width  : Image width in pixels.
 *   height : Image height in pixels.
 *   amount : Sharpening strength [0, 5]. 0 = no sharpening.
 *   radius : Kernel radius (1 = 3x3, 2 = 5x5). Clamped to [1, 2].
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
// Device helper: sample a pixel with bounds clamping
// ---------------------------------------------------------------------------
__device__ __forceinline__ void sample_pixel(
    const uint8_t* __restrict__ input,
    int width, int height,
    int px, int py,
    float& r, float& g, float& b)
{
    const int cx = min(max(px, 0), width - 1);
    const int cy = min(max(py, 0), height - 1);
    const int idx = (cy * width + cx) * 4;

    r = static_cast<float>(input[idx + 0]);
    g = static_cast<float>(input[idx + 1]);
    b = static_cast<float>(input[idx + 2]);
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void sharpen(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float amount,
    int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Read original pixel
    // -----------------------------------------------------------------------
    const float orig_r = static_cast<float>(input[idx + 0]);
    const float orig_g = static_cast<float>(input[idx + 1]);
    const float orig_b = static_cast<float>(input[idx + 2]);
    const float a      = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Compute blurred pixel using convolution
    // -----------------------------------------------------------------------
    float blur_r = 0.0f, blur_g = 0.0f, blur_b = 0.0f;
    float weight_sum = 0.0f;

    const int r_clamped = min(max(radius, 1), 2);

    for (int ky = -r_clamped; ky <= r_clamped; ++ky)
    {
        for (int kx = -r_clamped; kx <= r_clamped; ++kx)
        {
            // Gaussian-like weight: 1/(1 + dist^2)
            const float dist_sq = static_cast<float>(kx * kx + ky * ky);
            const float sigma = static_cast<float>(r_clamped);
            const float w = expf(-dist_sq / (2.0f * sigma * sigma));

            float sr, sg, sb;
            sample_pixel(input, width, height, x + kx, y + ky, sr, sg, sb);

            blur_r += sr * w;
            blur_g += sg * w;
            blur_b += sb * w;
            weight_sum += w;
        }
    }

    const float inv_w = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;
    blur_r *= inv_w;
    blur_g *= inv_w;
    blur_b *= inv_w;

    // -----------------------------------------------------------------------
    // 3. Apply unsharp mask: output = original + amount * (original - blurred)
    // -----------------------------------------------------------------------
    const float out_r = orig_r + amount * (orig_r - blur_r);
    const float out_g = orig_g + amount * (orig_g - blur_g);
    const float out_b = orig_b + amount * (orig_b - blur_b);

    // -----------------------------------------------------------------------
    // 4. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(out_r);
    output[idx + 1] = clamp_u8(out_g);
    output[idx + 2] = clamp_u8(out_b);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
