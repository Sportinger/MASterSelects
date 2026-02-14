/**
 * vignette.cu -- Vignette (Edge Darkening) Effect (CUDA)
 *
 * Applies a vignette effect that darkens the edges of the image based on
 * the distance from the center. The effect uses an elliptical model to
 * handle non-square images correctly (the vignette matches the aspect ratio).
 *
 * The radius parameter defines the inner circle where no darkening occurs.
 * The softness parameter controls the width of the falloff region.
 * The amount parameter controls how dark the edges become.
 *
 * Alpha channel is preserved unchanged.
 *
 * Parameters
 * ----------
 *   input    : Source RGBA buffer (4 bytes per pixel).
 *   output   : Destination RGBA buffer (4 bytes per pixel).
 *   width    : Image width in pixels.
 *   height   : Image height in pixels.
 *   amount   : Darkening strength [0, 1]. 0 = no effect, 1 = fully black edges.
 *   radius   : Inner radius (no darkening inside), normalized [0, 2].
 *              0 = darkening starts at center, 1 = starts at half-diagonal.
 *   softness : Falloff width beyond radius, normalized [0.01, 2].
 *              Smaller = harder edge, larger = softer transition.
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
extern "C" __global__ void vignette(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float amount,
    float radius,
    float softness)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Compute normalized distance from image center
    //    Uses coordinates in [-1, 1] range for both axes
    // -----------------------------------------------------------------------
    const float cx = (static_cast<float>(x) + 0.5f) / static_cast<float>(width)  * 2.0f - 1.0f;
    const float cy = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.0f - 1.0f;

    // Euclidean distance from center (max ~1.414 at corners)
    const float dist = sqrtf(cx * cx + cy * cy);

    // -----------------------------------------------------------------------
    // 2. Compute vignette factor using smoothstep-like falloff
    //    - Inside radius: factor = 1 (no darkening)
    //    - Between radius and radius+softness: smooth transition
    //    - Outside radius+softness: maximum darkening
    // -----------------------------------------------------------------------
    const float soft_clamped = fmaxf(softness, 0.01f);
    float vignette_factor;

    if (dist <= radius)
    {
        vignette_factor = 1.0f;
    }
    else
    {
        // Smooth falloff using hermite interpolation (smoothstep)
        const float t = fminf((dist - radius) / soft_clamped, 1.0f);
        const float smooth_t = t * t * (3.0f - 2.0f * t);  // smoothstep
        vignette_factor = 1.0f - amount * smooth_t;
    }

    // Clamp factor to [0, 1]
    vignette_factor = fminf(fmaxf(vignette_factor, 0.0f), 1.0f);

    // -----------------------------------------------------------------------
    // 3. Read pixel and apply vignette darkening
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]) * vignette_factor;
    const float g = static_cast<float>(input[idx + 1]) * vignette_factor;
    const float b = static_cast<float>(input[idx + 2]) * vignette_factor;
    const float a = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 4. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(r);
    output[idx + 1] = clamp_u8(g);
    output[idx + 2] = clamp_u8(b);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
