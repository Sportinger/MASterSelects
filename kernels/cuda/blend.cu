/**
 * blend.cu -- Blend Mode Compositing Kernel (CUDA)
 *
 * Applies one of several photoshop-style blend modes to composite a
 * foreground RGBA layer onto a background RGBA layer, with a global opacity
 * multiplier. The blend is performed in straight (non-premultiplied) alpha
 * space, and the final result uses source-over compositing with the blended
 * color as the foreground.
 *
 * Supported blend modes:
 *   0 = Normal     (standard source-over, same as composite.cu)
 *   1 = Multiply   (fg * bg)
 *   2 = Screen     (1 - (1-fg)*(1-bg))
 *   3 = Overlay    (Multiply if bg < 0.5, Screen if bg >= 0.5)
 *   4 = Add        (fg + bg, clamped)
 *   5 = Subtract   (bg - fg, clamped)
 *
 * All blend math is done in normalized [0, 1] float space (BT.709
 * assumption: straight sRGB values). The output is clamped to [0, 255].
 *
 * Formula:
 *   blended_rgb = blend_func(fg.rgb, bg.rgb)     // per blend mode
 *   fg_a_eff    = (fg.a / 255) * opacity
 *   out.rgb     = lerp(bg.rgb, blended_rgb, fg_a_eff) * 255
 *   out.a       = (fg_a_eff + (bg.a/255) * (1 - fg_a_eff)) * 255
 *
 * Parameters
 * ----------
 *   fg         : Foreground RGBA buffer (4 bytes per pixel).
 *   bg         : Background RGBA buffer (4 bytes per pixel).
 *   output     : Output RGBA buffer (4 bytes per pixel). May alias bg.
 *   width      : Frame width in pixels.
 *   height     : Frame height in pixels.
 *   fg_pitch   : Byte stride of one row in the foreground buffer.
 *   bg_pitch   : Byte stride of one row in the background buffer.
 *   out_pitch  : Byte stride of one row in the output buffer.
 *   blend_mode : Integer blend mode selector (see list above).
 *   opacity    : Global foreground opacity in [0.0, 1.0].
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
// Blend mode functions — operate on normalized [0, 1] values per channel
// ---------------------------------------------------------------------------

// Normal: foreground replaces background (identity blend)
__device__ __forceinline__ float blend_normal(float fg, float bg)
{
    return fg;
}

// Multiply: darkening blend
__device__ __forceinline__ float blend_multiply(float fg, float bg)
{
    return fg * bg;
}

// Screen: lightening blend
__device__ __forceinline__ float blend_screen(float fg, float bg)
{
    return 1.0f - (1.0f - fg) * (1.0f - bg);
}

// Overlay: combination of Multiply and Screen based on background luminance
__device__ __forceinline__ float blend_overlay(float fg, float bg)
{
    if (bg < 0.5f)
        return 2.0f * fg * bg;
    else
        return 1.0f - 2.0f * (1.0f - fg) * (1.0f - bg);
}

// Add (Linear Dodge): additive blend, clamped to 1.0
__device__ __forceinline__ float blend_add(float fg, float bg)
{
    return fminf(fg + bg, 1.0f);
}

// Subtract: subtractive blend, clamped to 0.0
__device__ __forceinline__ float blend_subtract(float fg, float bg)
{
    return fmaxf(bg - fg, 0.0f);
}

// ---------------------------------------------------------------------------
// Device helper: dispatch blend by mode index
// ---------------------------------------------------------------------------
__device__ __forceinline__ float apply_blend(float fg, float bg, int mode)
{
    switch (mode)
    {
        case 0:  return blend_normal(fg, bg);
        case 1:  return blend_multiply(fg, bg);
        case 2:  return blend_screen(fg, bg);
        case 3:  return blend_overlay(fg, bg);
        case 4:  return blend_add(fg, bg);
        case 5:  return blend_subtract(fg, bg);
        default: return blend_normal(fg, bg);
    }
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void blend_rgba(
    const uint8_t* __restrict__ fg,
    const uint8_t* __restrict__ bg,
    uint8_t*       __restrict__ output,
    int width,
    int height,
    int fg_pitch,
    int bg_pitch,
    int out_pitch,
    int blend_mode,
    float opacity)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // -----------------------------------------------------------------------
    // 1. Read foreground and background pixels
    // -----------------------------------------------------------------------
    const int fg_offset = y * fg_pitch + x * 4;
    const float fg_r = static_cast<float>(fg[fg_offset + 0]) / 255.0f;
    const float fg_g = static_cast<float>(fg[fg_offset + 1]) / 255.0f;
    const float fg_b = static_cast<float>(fg[fg_offset + 2]) / 255.0f;
    const float fg_a = static_cast<float>(fg[fg_offset + 3]) / 255.0f;

    const int bg_offset = y * bg_pitch + x * 4;
    const float bg_r = static_cast<float>(bg[bg_offset + 0]) / 255.0f;
    const float bg_g = static_cast<float>(bg[bg_offset + 1]) / 255.0f;
    const float bg_b = static_cast<float>(bg[bg_offset + 2]) / 255.0f;
    const float bg_a = static_cast<float>(bg[bg_offset + 3]) / 255.0f;

    // -----------------------------------------------------------------------
    // 2. Apply blend mode (in normalized [0, 1] space)
    // -----------------------------------------------------------------------
    const float blended_r = apply_blend(fg_r, bg_r, blend_mode);
    const float blended_g = apply_blend(fg_g, bg_g, blend_mode);
    const float blended_b = apply_blend(fg_b, bg_b, blend_mode);

    // -----------------------------------------------------------------------
    // 3. Composite with alpha: lerp between background and blended result
    //    using effective foreground alpha (per-pixel alpha * global opacity)
    // -----------------------------------------------------------------------
    const float fg_a_eff = fg_a * opacity;
    const float inv_a    = 1.0f - fg_a_eff;

    const float out_r = (blended_r * fg_a_eff + bg_r * inv_a) * 255.0f;
    const float out_g = (blended_g * fg_a_eff + bg_g * inv_a) * 255.0f;
    const float out_b = (blended_b * fg_a_eff + bg_b * inv_a) * 255.0f;
    const float out_a = (fg_a_eff + bg_a * inv_a) * 255.0f;

    // -----------------------------------------------------------------------
    // 4. Write output
    // -----------------------------------------------------------------------
    const int out_offset = y * out_pitch + x * 4;

    output[out_offset + 0] = clamp_u8(out_r);
    output[out_offset + 1] = clamp_u8(out_g);
    output[out_offset + 2] = clamp_u8(out_b);
    output[out_offset + 3] = clamp_u8(out_a);
}
