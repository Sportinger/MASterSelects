//! CPU-based NV12-to-RGBA color space conversion (software fallback).
//!
//! This module provides a software fallback for converting NV12 video frames
//! (the standard output format of hardware video decoders like NVDEC) to RGBA8
//! pixel data suitable for display.
//!
//! # When to use this
//!
//! Use this module when:
//! - No GPU is available for the NV12-to-RGBA conversion kernel
//! - You need to convert a decoded frame on the CPU (e.g., for thumbnail generation)
//! - As a reference implementation to validate GPU kernel output
//!
//! # Color space
//!
//! Uses the **BT.709** color matrix, which is the standard for HD content (>=720p).
//! The conversion formulas are:
//!
//! ```text
//! R = 1.164 * (Y - 16) + 1.793 * (V - 128)
//! G = 1.164 * (Y - 16) - 0.213 * (U - 128) - 0.533 * (V - 128)
//! B = 1.164 * (Y - 16) + 2.112 * (U - 128)
//! ```
//!
//! # NV12 format
//!
//! NV12 is a semi-planar YUV 4:2:0 format:
//! - **Y plane**: one luma byte per pixel (`width * height` bytes)
//! - **UV plane**: interleaved chroma at half resolution (`width * height/2` bytes)
//!   Each pair of bytes (U, V) covers a 2x2 block of pixels.
//!
//! # `SoftwareDecoder` struct
//!
//! The [`SoftwareDecoder`] implements the [`HwDecoder`](ms_common::HwDecoder) trait
//! but does **NOT** perform actual H.264/H.265 bitstream decoding. It is a
//! conversion helper that sits between a hardware decoder NV12 output and the
//! display pipeline. For actual video decoding, use [`NvDecoder`](crate::nvdec::NvDecoder)
//! or a future Vulkan Video decoder.

use ms_common::color::PixelFormat;
use ms_common::packet::{GpuFrame, VideoPacket};
use ms_common::types::Resolution;
use ms_common::{DecodeError, HwDecoder, VideoCodec};

// ---------------------------------------------------------------------------
// BT.709 fixed-point conversion constants
// ---------------------------------------------------------------------------

// We use fixed-point arithmetic with 10 bits of fractional precision (multiply
// by 1024) to avoid floating-point operations in the inner loop.
//
//   R = 1.164 * (Y - 16) + 1.793 * (V - 128)
//   G = 1.164 * (Y - 16) - 0.213 * (U - 128) - 0.533 * (V - 128)
//   B = 1.164 * (Y - 16) + 2.112 * (U - 128)
const Y_SCALE: i32 = 1192; // 1.164 * 1024
const V_TO_R: i32 = 1836;  // 1.793 * 1024
const U_TO_G: i32 = 218;   // 0.213 * 1024
const V_TO_G: i32 = 546;   // 0.533 * 1024
const U_TO_B: i32 = 2163;  // 2.112 * 1024

/// Clamp an i32 value to the [0, 255] range and return as u8.
#[inline(always)]
fn clamp_u8(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from software NV12-to-RGBA conversion.
#[derive(Debug, thiserror::Error)]
pub enum SoftwareDecodeError {
    /// Width or height is zero, or height is odd (NV12 requires even height).
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// The Y plane buffer is too small for the given dimensions and pitch.
    #[error("Y plane too small: need {needed}, got {got}")]
    YPlaneTooSmall { needed: usize, got: usize },

    /// The UV plane buffer is too small for the given dimensions and pitch.
    #[error("UV plane too small: need {needed}, got {got}")]
    UvPlaneTooSmall { needed: usize, got: usize },

    /// The output buffer is too small to hold width*height*4 RGBA bytes.
    #[error("Output buffer too small: need {needed}, got {got}")]
    OutputTooSmall { needed: usize, got: usize },
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate dimensions and buffer sizes for NV12-to-RGBA conversion.
///
/// Returns the required output buffer size in bytes (`width * height * 4`).
fn validate_inputs(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: u32,
    height: u32,
    y_pitch: u32,
    uv_pitch: u32,
) -> Result<usize, SoftwareDecodeError> {
    if width == 0 || height == 0 || !height.is_multiple_of(2) {
        return Err(SoftwareDecodeError::InvalidDimensions { width, height });
    }

    let w = width as usize;
    let h = height as usize;
    let yp = y_pitch as usize;
    let uvp = uv_pitch as usize;

    let y_needed = yp * h;
    if y_plane.len() < y_needed {
        return Err(SoftwareDecodeError::YPlaneTooSmall {
            needed: y_needed,
            got: y_plane.len(),
        });
    }

    let uv_needed = uvp * (h / 2);
    if uv_plane.len() < uv_needed {
        return Err(SoftwareDecodeError::UvPlaneTooSmall {
            needed: uv_needed,
            got: uv_plane.len(),
        });
    }

    if uvp < w {
        return Err(SoftwareDecodeError::UvPlaneTooSmall {
            needed: w,
            got: uvp,
        });
    }

    let rgba_size = w * h * 4;
    Ok(rgba_size)
}

// ---------------------------------------------------------------------------
// Public conversion functions
// ---------------------------------------------------------------------------

/// Convert an NV12 frame to RGBA8 on the CPU using BT.709 color matrix.
///
/// Returns a newly allocated `Vec<u8>` of size `width * height * 4`.
///
/// # Arguments
/// - `y_plane` -- Luma plane data (at least `y_pitch * height` bytes).
/// - `uv_plane` -- Interleaved chroma plane (at least `uv_pitch * height/2` bytes).
/// - `width` -- Frame width in pixels.
/// - `height` -- Frame height in pixels (must be even).
/// - `y_pitch` -- Byte stride of the Y plane (may be > width for alignment).
/// - `uv_pitch` -- Byte stride of the UV plane.
///
/// # Errors
/// Returns [`SoftwareDecodeError`] if dimensions are invalid or buffers too small.
///
/// # Example
/// ```
/// use ms_decoder::software::nv12_to_rgba;
///
/// let width = 4u32;
/// let height = 2u32;
/// let y_plane = vec![16u8; (width * height) as usize];
/// let uv_plane = vec![128u8; (width * height / 2) as usize];
/// let rgba = nv12_to_rgba(&y_plane, &uv_plane, width, height, width, width).unwrap();
/// assert_eq!(rgba.len(), (width * height * 4) as usize);
/// ```
pub fn nv12_to_rgba(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: u32,
    height: u32,
    y_pitch: u32,
    uv_pitch: u32,
) -> Result<Vec<u8>, SoftwareDecodeError> {
    let rgba_size = validate_inputs(y_plane, uv_plane, width, height, y_pitch, uv_pitch)?;
    let mut output = vec![0u8; rgba_size];
    convert_nv12_to_rgba(y_plane, uv_plane, width, height, y_pitch, uv_pitch, &mut output);
    Ok(output)
}

/// Convert an NV12 frame to RGBA8 into a pre-allocated output buffer.
///
/// This avoids allocation when the caller already has a buffer (e.g., a
/// reusable staging buffer for repeated conversions).
///
/// # Arguments
/// Same as [`nv12_to_rgba`], plus:
/// - `output` -- Mutable slice of at least `width * height * 4` bytes.
///
/// # Errors
/// Returns [`SoftwareDecodeError`] if dimensions are invalid, planes too small,
/// or the output buffer is too small.
pub fn nv12_to_rgba_inplace(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: u32,
    height: u32,
    y_pitch: u32,
    uv_pitch: u32,
    output: &mut [u8],
) -> Result<(), SoftwareDecodeError> {
    let rgba_size = validate_inputs(y_plane, uv_plane, width, height, y_pitch, uv_pitch)?;

    if output.len() < rgba_size {
        return Err(SoftwareDecodeError::OutputTooSmall {
            needed: rgba_size,
            got: output.len(),
        });
    }

    convert_nv12_to_rgba(y_plane, uv_plane, width, height, y_pitch, uv_pitch, output);
    Ok(())
}

/// Core conversion loop -- processes 2 pixels at a time horizontally.
///
/// NV12 chroma is 4:2:0, meaning each UV pair covers a 2x2 block of luma
/// pixels. We process two horizontal pixels per iteration, sharing the same
/// U and V values, which halves the UV lookups.
///
/// Uses fixed-point integer arithmetic (10-bit fractional precision).
#[inline]
fn convert_nv12_to_rgba(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: u32,
    height: u32,
    y_pitch: u32,
    uv_pitch: u32,
    output: &mut [u8],
) {
    let w = width as usize;
    let h = height as usize;
    let yp = y_pitch as usize;
    let uvp = uv_pitch as usize;

    for row in 0..h {
        let y_row_offset = row * yp;
        let uv_row_offset = (row / 2) * uvp;
        let out_row_offset = row * w * 4;

        let mut col = 0usize;
        while col < w {
            let uv_col = (col / 2) * 2;
            let u = uv_plane[uv_row_offset + uv_col] as i32;
            let v = uv_plane[uv_row_offset + uv_col + 1] as i32;

            let v_r = V_TO_R * (v - 128);
            let u_g = U_TO_G * (u - 128);
            let v_g = V_TO_G * (v - 128);
            let u_b = U_TO_B * (u - 128);

            // --- Pixel 0 ---
            let y0 = y_plane[y_row_offset + col] as i32;
            let y0_scaled = Y_SCALE * (y0 - 16);

            let r0 = (y0_scaled + v_r + 512) >> 10;
            let g0 = (y0_scaled - u_g - v_g + 512) >> 10;
            let b0 = (y0_scaled + u_b + 512) >> 10;

            let out_idx0 = out_row_offset + col * 4;
            output[out_idx0] = clamp_u8(r0);
            output[out_idx0 + 1] = clamp_u8(g0);
            output[out_idx0 + 2] = clamp_u8(b0);
            output[out_idx0 + 3] = 255;

            // --- Pixel 1 (if within bounds for odd-width frames) ---
            if col + 1 < w {
                let y1 = y_plane[y_row_offset + col + 1] as i32;
                let y1_scaled = Y_SCALE * (y1 - 16);

                let r1 = (y1_scaled + v_r + 512) >> 10;
                let g1 = (y1_scaled - u_g - v_g + 512) >> 10;
                let b1 = (y1_scaled + u_b + 512) >> 10;

                let out_idx1 = out_row_offset + (col + 1) * 4;
                output[out_idx1] = clamp_u8(r1);
                output[out_idx1 + 1] = clamp_u8(g1);
                output[out_idx1 + 2] = clamp_u8(b1);
                output[out_idx1 + 3] = 255;
            }

            col += 2;
        }
    }
}

// ---------------------------------------------------------------------------
// SoftwareDecoder -- HwDecoder trait implementation
// ---------------------------------------------------------------------------

/// A software NV12-to-RGBA conversion helper implementing [`HwDecoder`].
///
/// **Important:** This is **NOT** a full software video decoder. It does not
/// perform H.264/H.265/VP9/AV1 bitstream decoding. It only provides the
/// NV12-to-RGBA color space conversion step on the CPU.
///
/// ## Intended use
///
/// `SoftwareDecoder` accepts `VideoPacket`s but interprets their data as raw
/// NV12 frame bytes (not compressed video). This is useful for:
///
/// - Testing the display pipeline without a GPU
/// - Converting pre-decoded NV12 data that was read back from GPU memory
/// - Providing a CPU fallback path in the render pipeline
///
/// For actual compressed video decoding, use
/// [`NvDecoder`](crate::nvdec::NvDecoder) (NVIDIA) or a future Vulkan Video
/// decoder.
///
/// ## Frame data format
///
/// The `VideoPacket.data` field must contain a raw NV12 frame laid out as:
/// - Y plane: `width * height` bytes
/// - UV plane: `width * (height / 2)` bytes (interleaved U, V)
///
/// Total: `width * height * 3 / 2` bytes.
pub struct SoftwareDecoder {
    /// Expected frame resolution.
    resolution: Resolution,
    /// Output pixel format (always RGBA8 after conversion).
    output_format: PixelFormat,
    /// Codec label (for trait compliance; not actually decoded).
    codec: VideoCodec,
    /// Number of frames converted.
    frames_converted: u64,
    /// Reusable RGBA output buffer (avoids repeated allocation).
    rgba_buffer: Vec<u8>,
}

impl std::fmt::Debug for SoftwareDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoftwareDecoder")
            .field("resolution", &self.resolution)
            .field("output_format", &self.output_format)
            .field("codec", &self.codec)
            .field("frames_converted", &self.frames_converted)
            .finish()
    }
}

impl SoftwareDecoder {
    /// Create a new `SoftwareDecoder` for the given resolution and codec.
    ///
    /// # Arguments
    /// - `width` -- Expected frame width in pixels.
    /// - `height` -- Expected frame height in pixels (must be even).
    /// - `codec` -- The video codec label (not actually used for decoding).
    ///
    /// # Errors
    /// Returns [`DecodeError`] if dimensions are invalid.
    pub fn new(width: u32, height: u32, codec: VideoCodec) -> Result<Self, DecodeError> {
        if width == 0 || height == 0 || !height.is_multiple_of(2) {
            return Err(DecodeError::HwDecoderInit {
                codec,
                reason: format!(
                    "Invalid dimensions for SoftwareDecoder: {width}x{height} \
                     (width and height must be non-zero, height must be even)"
                ),
            });
        }

        let rgba_size = width as usize * height as usize * 4;

        Ok(Self {
            resolution: Resolution::new(width, height),
            output_format: PixelFormat::Rgba8,
            codec,
            frames_converted: 0,
            rgba_buffer: vec![0u8; rgba_size],
        })
    }

    /// Get the number of frames converted so far.
    pub fn frames_converted(&self) -> u64 {
        self.frames_converted
    }
}

impl HwDecoder for SoftwareDecoder {
    /// Convert a raw NV12 frame (in `packet.data`) to RGBA8.
    ///
    /// The `VideoPacket.data` must contain `width * height * 3 / 2`
    /// bytes of raw NV12 data (Y plane followed by interleaved UV plane).
    ///
    /// Returns a `GpuFrame` with `device_ptr` set to 0 (CPU-only).
    fn decode(&mut self, packet: &VideoPacket) -> Result<Option<GpuFrame>, DecodeError> {
        let w = self.resolution.width;
        let h = self.resolution.height;
        let y_size = w as usize * h as usize;
        let uv_size = w as usize * (h as usize / 2);
        let nv12_size = y_size + uv_size;

        if packet.data.len() < nv12_size {
            return Err(DecodeError::DecodeFailed {
                frame: self.frames_converted,
                reason: format!(
                    "Packet data too small for NV12 frame: need {nv12_size}, got {}",
                    packet.data.len()
                ),
            });
        }

        let y_plane = &packet.data[..y_size];
        let uv_plane = &packet.data[y_size..y_size + uv_size];

        convert_nv12_to_rgba(y_plane, uv_plane, w, h, w, w, &mut self.rgba_buffer);

        self.frames_converted += 1;

        Ok(Some(GpuFrame {
            device_ptr: 0,
            device_ptr_uv: None,
            resolution: self.resolution,
            format: self.output_format,
            pitch: w * 4,
            pts: packet.pts,
        }))
    }

    /// Flush -- no-op for the software decoder (no buffered frames).
    fn flush(&mut self) -> Result<Vec<GpuFrame>, DecodeError> {
        Ok(Vec::new())
    }

    /// Output format is always RGBA8 (post-conversion).
    fn output_format(&self) -> PixelFormat {
        self.output_format
    }

    /// Output resolution as configured at construction time.
    fn output_resolution(&self) -> Resolution {
        self.resolution
    }

    /// The codec label this decoder was created for.
    fn codec(&self) -> VideoCodec {
        self.codec
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_nv12(
        width: u32, height: u32, y_val: u8, u_val: u8, v_val: u8,
    ) -> (Vec<u8>, Vec<u8>) {
        let y_plane = vec![y_val; (width * height) as usize];
        let mut uv_plane = vec![0u8; (width * (height / 2)) as usize];
        for i in 0..uv_plane.len() / 2 {
            uv_plane[i * 2] = u_val;
            uv_plane[i * 2 + 1] = v_val;
        }
        (y_plane, uv_plane)
    }

    fn reference_bt709(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
        let yf = y as f64;
        let uf = u as f64;
        let vf = v as f64;
        let r = 1.164 * (yf - 16.0) + 1.793 * (vf - 128.0);
        let g = 1.164 * (yf - 16.0) - 0.213 * (uf - 128.0) - 0.533 * (vf - 128.0);
        let b = 1.164 * (yf - 16.0) + 2.112 * (uf - 128.0);
        (
            r.round().clamp(0.0, 255.0) as u8,
            g.round().clamp(0.0, 255.0) as u8,
            b.round().clamp(0.0, 255.0) as u8,
        )
    }

    #[test]
    fn test_black_frame() {
        let (yp, uvp) = make_uniform_nv12(4, 2, 16, 128, 128);
        let rgba = nv12_to_rgba(&yp, &uvp, 4, 2, 4, 4).unwrap();
        assert_eq!(rgba.len(), 32);
        for px in rgba.chunks_exact(4) {
            assert!(px[0] <= 2); assert!(px[1] <= 2); assert!(px[2] <= 2);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn test_white_frame() {
        let (yp, uvp) = make_uniform_nv12(4, 2, 235, 128, 128);
        let rgba = nv12_to_rgba(&yp, &uvp, 4, 2, 4, 4).unwrap();
        for px in rgba.chunks_exact(4) {
            assert!(px[0] >= 250, "R should be near 255, got {}", px[0]);
            assert!(px[1] >= 250, "G should be near 255, got {}", px[1]);
            assert!(px[2] >= 250, "B should be near 255, got {}", px[2]);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn test_reference_bt709_accuracy() {
        // Check a specific YUV value against the reference floating-point formula
        let (ref_r, ref_g, ref_b) = reference_bt709(180, 100, 200);
        let (yp, uvp) = make_uniform_nv12(4, 2, 180, 100, 200);
        let rgba = nv12_to_rgba(&yp, &uvp, 4, 2, 4, 4).unwrap();
        // Allow +-2 for fixed-point rounding
        assert!((rgba[0] as i32 - ref_r as i32).abs() <= 2);
        assert!((rgba[1] as i32 - ref_g as i32).abs() <= 2);
        assert!((rgba[2] as i32 - ref_b as i32).abs() <= 2);
    }

    #[test]
    fn test_invalid_dimensions() {
        let result = nv12_to_rgba(&[], &[], 0, 0, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_odd_height_rejected() {
        let result = nv12_to_rgba(&[0; 12], &[0; 4], 4, 3, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_inplace_matches_allocating() {
        let (yp, uvp) = make_uniform_nv12(8, 4, 128, 64, 200);
        let alloc_result = nv12_to_rgba(&yp, &uvp, 8, 4, 8, 8).unwrap();
        let mut inplace_result = vec![0u8; 8 * 4 * 4];
        nv12_to_rgba_inplace(&yp, &uvp, 8, 4, 8, 8, &mut inplace_result).unwrap();
        assert_eq!(alloc_result, inplace_result);
    }

    #[test]
    fn test_software_decoder_trait() {
        let mut dec = SoftwareDecoder::new(4, 2, VideoCodec::H264).unwrap();
        assert_eq!(dec.output_format(), PixelFormat::Rgba8);
        assert_eq!(dec.output_resolution(), Resolution::new(4, 2));
        assert_eq!(dec.codec(), VideoCodec::H264);
        assert_eq!(dec.frames_converted(), 0);

        // Create a valid NV12 packet (4x2 = 8 Y bytes + 4 UV bytes = 12 bytes)
        let mut nv12_data = vec![128u8; 12];
        // Set UV to neutral
        for byte in &mut nv12_data[8..12] {
            *byte = 128;
        }
        let pkt = ms_common::packet::VideoPacket {
            data: nv12_data,
            pts: ms_common::types::TimeCode::ZERO,
            dts: ms_common::types::TimeCode::ZERO,
            is_keyframe: true,
            codec: VideoCodec::H264,
        };

        let result = dec.decode(&pkt).unwrap();
        assert!(result.is_some());
        assert_eq!(dec.frames_converted(), 1);
    }
}
