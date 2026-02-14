/**
 * chroma_key.cu -- Chroma Key (Green/Blue Screen) Effect (CUDA)
 *
 * Removes a specified key color from an RGBA image by comparing each pixel
 * to the key color in YCbCr color space. The distance in the CbCr (chroma)
 * plane determines how transparent the pixel becomes. This is the standard
 * approach for green-screen / blue-screen keying.
 *
 * The tolerance parameter defines the hard cutoff radius in CbCr space,
 * and softness extends a gradual falloff beyond that radius.
 *
 * Parameters
 * ----------
 *   input     : Source RGBA buffer (4 bytes per pixel).
 *   output    : Destination RGBA buffer (4 bytes per pixel).
 *   width     : Image width in pixels.
 *   height    : Image height in pixels.
 *   key_r     : Key color red component [0, 255].
 *   key_g     : Key color green component [0, 255].
 *   key_b     : Key color blue component [0, 255].
 *   key_a     : Key color alpha (unused, reserved for API consistency).
 *   tolerance : Hard edge threshold — CbCr distance below this is fully keyed.
 *   softness  : Soft edge falloff width beyond tolerance.
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
// Device helper: convert RGB [0,255] to YCbCr
// Uses BT.601 coefficients (standard for chroma keying)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void rgb_to_ycbcr(
    float r, float g, float b,
    float& y_out, float& cb_out, float& cr_out)
{
    y_out  =  0.299f * r + 0.587f * g + 0.114f * b;
    cb_out = -0.169f * r - 0.331f * g + 0.500f * b + 128.0f;
    cr_out =  0.500f * r - 0.419f * g - 0.081f * b + 128.0f;
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void chroma_key(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float key_r,
    float key_g,
    float key_b,
    float key_a,
    float tolerance,
    float softness)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Read pixel RGB values
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]);
    const float g = static_cast<float>(input[idx + 1]);
    const float b = static_cast<float>(input[idx + 2]);
    const float a = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Convert pixel and key color to YCbCr
    // -----------------------------------------------------------------------
    float pix_y, pix_cb, pix_cr;
    rgb_to_ycbcr(r, g, b, pix_y, pix_cb, pix_cr);

    float key_y_unused, key_cb, key_cr;
    rgb_to_ycbcr(key_r, key_g, key_b, key_y_unused, key_cb, key_cr);

    // -----------------------------------------------------------------------
    // 3. Compute chroma distance (CbCr plane only — luma independent)
    // -----------------------------------------------------------------------
    const float d_cb = pix_cb - key_cb;
    const float d_cr = pix_cr - key_cr;
    const float chroma_dist = sqrtf(d_cb * d_cb + d_cr * d_cr);

    // -----------------------------------------------------------------------
    // 4. Compute alpha mask based on distance, tolerance, and softness
    //    - Below tolerance: fully transparent (keyed out)
    //    - Between tolerance and tolerance+softness: gradual falloff
    //    - Above tolerance+softness: fully opaque (kept)
    // -----------------------------------------------------------------------
    float mask;
    if (chroma_dist < tolerance)
    {
        mask = 0.0f;
    }
    else if (softness > 0.0f && chroma_dist < tolerance + softness)
    {
        mask = (chroma_dist - tolerance) / softness;
    }
    else
    {
        mask = 1.0f;
    }

    // -----------------------------------------------------------------------
    // 5. Apply mask to alpha channel (multiply with existing alpha)
    // -----------------------------------------------------------------------
    const float out_a = a * mask;

    // -----------------------------------------------------------------------
    // 6. Write output (RGB preserved, alpha modified by key mask)
    // -----------------------------------------------------------------------
    output[idx + 0] = input[idx + 0];
    output[idx + 1] = input[idx + 1];
    output[idx + 2] = input[idx + 2];
    output[idx + 3] = clamp_u8(out_a);
}
