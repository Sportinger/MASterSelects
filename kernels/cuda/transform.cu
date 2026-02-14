/**
 * transform.cu -- 2D Affine Transform Kernel (CUDA)
 *
 * Applies a 2D affine transformation (translation, scale, rotation) to an
 * RGBA image. The transform is computed relative to a configurable anchor
 * point (pivot), which defaults to the image center.
 *
 * The kernel works in *destination-to-source* mapping: for each output pixel
 * it computes the corresponding source coordinate via the inverse transform,
 * then samples the source image with bilinear interpolation. Pixels that map
 * outside the source bounds are set to transparent black (0, 0, 0, 0).
 *
 * Transform order (applied to source coordinates):
 *   1. Translate so anchor is at origin
 *   2. Scale by (sx, sy)
 *   3. Rotate by `rotation` radians (counter-clockwise)
 *   4. Translate back by anchor
 *   5. Translate by (tx, ty)
 *
 * The inverse of that chain is applied per output pixel to find the source
 * sample position.
 *
 * Parameters
 * ----------
 *   src       : Source RGBA buffer (4 bytes per pixel).
 *   dst       : Destination RGBA buffer (4 bytes per pixel).
 *   src_width : Source image width in pixels.
 *   src_height: Source image height in pixels.
 *   dst_width : Destination image width in pixels.
 *   dst_height: Destination image height in pixels.
 *   src_pitch : Byte stride of one row in the source buffer.
 *   dst_pitch : Byte stride of one row in the destination buffer.
 *   tx        : Translation X in pixels.
 *   ty        : Translation Y in pixels.
 *   sx        : Scale factor X (1.0 = no scale).
 *   sy        : Scale factor Y (1.0 = no scale).
 *   rotation  : Rotation angle in radians (counter-clockwise positive).
 *   anchor_x  : Pivot point X in source image coordinates.
 *   anchor_y  : Pivot point Y in source image coordinates.
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
// Device helper: bilinear sample from an RGBA buffer
// Returns (r, g, b, a) as floats in [0, 255].
// Out-of-bounds coordinates return transparent black.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void bilinear_sample(
    const uint8_t* __restrict__ src,
    int src_width,
    int src_height,
    int src_pitch,
    float sx,
    float sy,
    float& out_r,
    float& out_g,
    float& out_b,
    float& out_a)
{
    // Integer and fractional parts
    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float fx = sx - static_cast<float>(x0);
    const float fy = sy - static_cast<float>(y0);

    // Helper lambda-like macro: fetch pixel or transparent black if OOB
    #define FETCH_PIXEL(px, py, r, g, b, a) \
        if ((px) >= 0 && (px) < src_width && (py) >= 0 && (py) < src_height) { \
            int off = (py) * src_pitch + (px) * 4; \
            r = static_cast<float>(src[off + 0]); \
            g = static_cast<float>(src[off + 1]); \
            b = static_cast<float>(src[off + 2]); \
            a = static_cast<float>(src[off + 3]); \
        } else { \
            r = 0.0f; g = 0.0f; b = 0.0f; a = 0.0f; \
        }

    float r00, g00, b00, a00;
    float r10, g10, b10, a10;
    float r01, g01, b01, a01;
    float r11, g11, b11, a11;

    FETCH_PIXEL(x0, y0, r00, g00, b00, a00);
    FETCH_PIXEL(x1, y0, r10, g10, b10, a10);
    FETCH_PIXEL(x0, y1, r01, g01, b01, a01);
    FETCH_PIXEL(x1, y1, r11, g11, b11, a11);

    #undef FETCH_PIXEL

    // Bilinear weights
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    out_r = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
    out_g = g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11;
    out_b = b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11;
    out_a = a00 * w00 + a10 * w10 + a01 * w01 + a11 * w11;
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
extern "C" __global__ void transform_rgba(
    const uint8_t* __restrict__ src,
    uint8_t*       __restrict__ dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int src_pitch,
    int dst_pitch,
    float tx,
    float ty,
    float sx,
    float sy,
    float rotation,
    float anchor_x,
    float anchor_y)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_width || dy >= dst_height)
        return;

    // -----------------------------------------------------------------------
    // 1. Build inverse transform
    //
    //    Forward transform: T(tx,ty) * T(anchor) * R(rot) * S(sx,sy) * T(-anchor)
    //    Inverse:           T(anchor) * S(1/sx,1/sy) * R(-rot) * T(-anchor) * T(-tx,-ty)
    //
    //    We compute the inverse affine matrix coefficients directly.
    // -----------------------------------------------------------------------
    const float cos_r = cosf(rotation);
    const float sin_r = sinf(rotation);

    // Inverse scale
    const float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
    const float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;

    // Inverse rotation + scale matrix:  S^-1 * R^-1
    //   [inv_sx * cos_r,   inv_sx * sin_r]
    //   [-inv_sy * sin_r,  inv_sy * cos_r]
    const float m00 =  inv_sx * cos_r;
    const float m01 =  inv_sx * sin_r;
    const float m10 = -inv_sy * sin_r;
    const float m11 =  inv_sy * cos_r;

    // -----------------------------------------------------------------------
    // 2. Apply inverse transform to destination pixel (dx, dy)
    //
    //    src_pos = anchor + M * (dst_pos - anchor - translation)
    // -----------------------------------------------------------------------
    const float px = static_cast<float>(dx) - anchor_x - tx;
    const float py = static_cast<float>(dy) - anchor_y - ty;

    const float src_x = m00 * px + m01 * py + anchor_x;
    const float src_y = m10 * px + m11 * py + anchor_y;

    // -----------------------------------------------------------------------
    // 3. Bilinear sample from source
    // -----------------------------------------------------------------------
    float r, g, b, a;
    bilinear_sample(src, src_width, src_height, src_pitch, src_x, src_y, r, g, b, a);

    // -----------------------------------------------------------------------
    // 4. Write output
    // -----------------------------------------------------------------------
    const int out_offset = dy * dst_pitch + dx * 4;

    dst[out_offset + 0] = clamp_u8(r);
    dst[out_offset + 1] = clamp_u8(g);
    dst[out_offset + 2] = clamp_u8(b);
    dst[out_offset + 3] = clamp_u8(a);
}
