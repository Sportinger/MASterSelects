/**
 * color_wheels.cu -- Three-Way Color Wheels (Lift / Gamma / Gain) (CUDA)
 *
 * Applies a three-way color correction with independent Lift (shadows),
 * Gamma (midtones), and Gain (highlights) adjustments per RGB channel.
 * This is the standard DaVinci Resolve / Premiere Pro "Color Wheels" model.
 *
 * Formula (per channel):
 *   out = gain * pow(in + lift * (1 - in), 1 / (1 + gamma))
 *
 * Where:
 *   - lift  adds color to shadows (dark values affected most, bright unaffected)
 *   - gamma applies a power curve to midtones (per channel)
 *   - gain  multiplies the highlights (bright values affected most)
 *
 * All parameters are per-channel (R, G, B). Neutral values:
 *   lift = 0, gamma = 0, gain = 1
 *
 * Alpha channel is preserved unchanged.
 *
 * Parameters
 * ----------
 *   input   : Source RGBA buffer (4 bytes per pixel).
 *   output  : Destination RGBA buffer (4 bytes per pixel).
 *   width   : Image width in pixels.
 *   height  : Image height in pixels.
 *   lift_r, lift_g, lift_b     : Shadow lift per channel [-1, 1].
 *   gamma_r, gamma_g, gamma_b : Midtone gamma per channel [-1, 1].
 *   gain_r, gain_g, gain_b    : Highlight gain per channel [0, 3].
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
// Device helper: apply lift/gamma/gain to a single channel value
//   in   : normalized pixel value [0, 1]
//   lift : shadow adjustment [-1, 1]
//   gamma: midtone adjustment [-1, 1] (0 = neutral)
//   gain : highlight multiplier [0, 3] (1 = neutral)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float apply_lgg(float in_val, float lift, float gamma, float gain)
{
    // Lift: adds to shadows, diminishes toward highlights
    const float lifted = in_val + lift * (1.0f - in_val);

    // Clamp lifted to [0, 1] before power operation
    const float lifted_clamped = fminf(fmaxf(lifted, 0.0f), 1.0f);

    // Gamma: power curve (1 / (1 + gamma)), clamped to avoid div-by-zero
    const float gamma_exp = 1.0f / fmaxf(1.0f + gamma, 0.01f);
    const float gamma_corrected = powf(lifted_clamped, gamma_exp);

    // Gain: multiply (primarily affects highlights)
    return fminf(fmaxf(gain * gamma_corrected, 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void color_wheels(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    float lift_r,
    float lift_g,
    float lift_b,
    float gamma_r,
    float gamma_g,
    float gamma_b,
    float gain_r,
    float gain_g,
    float gain_b)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // -----------------------------------------------------------------------
    // 1. Read RGBA and normalize RGB to [0, 1]
    // -----------------------------------------------------------------------
    const float r = static_cast<float>(input[idx + 0]) / 255.0f;
    const float g = static_cast<float>(input[idx + 1]) / 255.0f;
    const float b = static_cast<float>(input[idx + 2]) / 255.0f;
    const float a = static_cast<float>(input[idx + 3]);

    // -----------------------------------------------------------------------
    // 2. Apply lift/gamma/gain per channel
    // -----------------------------------------------------------------------
    const float out_r = apply_lgg(r, lift_r, gamma_r, gain_r);
    const float out_g = apply_lgg(g, lift_g, gamma_g, gain_g);
    const float out_b = apply_lgg(b, lift_b, gamma_b, gain_b);

    // -----------------------------------------------------------------------
    // 3. Write output (alpha preserved)
    // -----------------------------------------------------------------------
    output[idx + 0] = clamp_u8(out_r * 255.0f);
    output[idx + 1] = clamp_u8(out_g * 255.0f);
    output[idx + 2] = clamp_u8(out_b * 255.0f);
    output[idx + 3] = static_cast<uint8_t>(fminf(fmaxf(a + 0.5f, 0.0f), 255.0f));
}
