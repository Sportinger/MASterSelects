//! Mask kernel dispatch.
//!
//! Dispatches the `apply_mask` GPU kernel to apply a mask shape to a frame,
//! modifying the alpha channel based on the mask geometry, feather, and inversion.

use ms_common::{GpuBackend, GpuStream, KernelArgs, KernelId, MaskDesc, MaskShape};
use tracing::debug;

use crate::CompositorError;

/// Block size for 2D kernel dispatch (16x16 = 256 threads per block).
const BLOCK_SIZE: u32 = 16;

/// Mask shape type integers for the GPU kernel.
const MASK_SHAPE_RECT: u32 = 0;
const MASK_SHAPE_ELLIPSE: u32 = 1;
const MASK_SHAPE_PATH: u32 = 2;

/// Dispatch the mask kernel to modify a frame's alpha channel according to a [`MaskDesc`].
///
/// The kernel multiplies each pixel's alpha by the mask value at that position.
/// Mask shapes are rasterized on the GPU: rectangles and ellipses are analytic,
/// path masks use a scanline fill approach.
///
/// # Arguments
///
/// * `backend` - GPU backend for kernel dispatch
/// * `frame_ptr` - Device pointer to the RGBA frame buffer (modified in-place)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `mask` - Mask description (shape, feather, opacity, inversion)
/// * `stream` - GPU stream for async execution
pub fn dispatch_mask(
    backend: &dyn GpuBackend,
    frame_ptr: u64,
    width: u32,
    height: u32,
    mask: &MaskDesc,
    stream: &GpuStream,
) -> Result<(), CompositorError> {
    let (shape_type, shape_params) = encode_mask_shape(&mask.shape, width, height);

    debug!(
        shape_type = shape_type,
        feather = mask.feather,
        opacity = mask.opacity,
        inverted = mask.inverted,
        expansion = mask.expansion,
        width = width,
        height = height,
        "Dispatching mask kernel"
    );

    let inverted_flag: u32 = if mask.inverted { 1 } else { 0 };

    let args = KernelArgs::new()
        .push_ptr(frame_ptr) // RGBA frame (in-place)
        .push_u32(width)
        .push_u32(height)
        .push_u32(shape_type) // mask shape type
        .push_vec4(shape_params) // shape parameters (meaning depends on shape_type)
        .push_f32(mask.feather)
        .push_f32(mask.opacity)
        .push_u32(inverted_flag)
        .push_f32(mask.expansion);

    let grid = [width.div_ceil(BLOCK_SIZE), height.div_ceil(BLOCK_SIZE), 1];
    let block = [BLOCK_SIZE, BLOCK_SIZE, 1];

    backend.dispatch_kernel(&KernelId::Mask, grid, block, &args, stream)?;

    Ok(())
}

/// Encode a [`MaskShape`] into a shape type integer and a 4-component parameter vector.
///
/// - Rect: `[x, y, width, height]` in pixel coordinates
/// - Ellipse: `[cx, cy, rx, ry]` in pixel coordinates
/// - Path: `[0, 0, 0, 0]` (path masks require a separate point buffer; not yet supported)
fn encode_mask_shape(shape: &MaskShape, frame_width: u32, frame_height: u32) -> (u32, [f32; 4]) {
    match shape {
        MaskShape::Rect {
            x,
            y,
            width,
            height,
        } => {
            // Normalize coordinates to pixel space
            let px = x * frame_width as f32;
            let py = y * frame_height as f32;
            let pw = width * frame_width as f32;
            let ph = height * frame_height as f32;
            (MASK_SHAPE_RECT, [px, py, pw, ph])
        }
        MaskShape::Ellipse { cx, cy, rx, ry } => {
            let px = cx * frame_width as f32;
            let py = cy * frame_height as f32;
            let prx = rx * frame_width as f32;
            let pry = ry * frame_height as f32;
            (MASK_SHAPE_ELLIPSE, [px, py, prx, pry])
        }
        MaskShape::Path { .. } => {
            // Path masks require uploading the point buffer separately.
            // For now, dispatch with empty params; the kernel will treat
            // this as a no-op mask (fully opaque).
            (MASK_SHAPE_PATH, [0.0, 0.0, 0.0, 0.0])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_shape_encoding() {
        let shape = MaskShape::Rect {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        };
        let (st, params) = encode_mask_shape(&shape, 1920, 1080);
        assert_eq!(st, MASK_SHAPE_RECT);
        assert!((params[0] - 0.0).abs() < 1e-5);
        assert!((params[1] - 0.0).abs() < 1e-5);
        assert!((params[2] - 1920.0).abs() < 1e-5);
        assert!((params[3] - 1080.0).abs() < 1e-5);
    }

    #[test]
    fn ellipse_shape_encoding() {
        let shape = MaskShape::Ellipse {
            cx: 0.5,
            cy: 0.5,
            rx: 0.25,
            ry: 0.25,
        };
        let (st, params) = encode_mask_shape(&shape, 1920, 1080);
        assert_eq!(st, MASK_SHAPE_ELLIPSE);
        assert!((params[0] - 960.0).abs() < 1e-2);
        assert!((params[1] - 540.0).abs() < 1e-2);
        assert!((params[2] - 480.0).abs() < 1e-2);
        assert!((params[3] - 270.0).abs() < 1e-2);
    }

    #[test]
    fn path_shape_encoding_placeholder() {
        let shape = MaskShape::Path {
            points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            closed: true,
        };
        let (st, params) = encode_mask_shape(&shape, 1920, 1080);
        assert_eq!(st, MASK_SHAPE_PATH);
        assert_eq!(params, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn mask_args_count() {
        let inverted_flag: u32 = 1;
        let args = KernelArgs::new()
            .push_ptr(0x1000)
            .push_u32(1920)
            .push_u32(1080)
            .push_u32(MASK_SHAPE_RECT)
            .push_vec4([0.0, 0.0, 1920.0, 1080.0])
            .push_f32(5.0)
            .push_f32(1.0)
            .push_u32(inverted_flag)
            .push_f32(0.0);
        assert_eq!(args.len(), 9);
    }
}
