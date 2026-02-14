//! Transform kernel dispatch.
//!
//! Dispatches the `transform_2d` GPU kernel to apply position, scale,
//! rotation, and anchor transforms to a source frame.

use ms_common::{GpuBackend, GpuFrame, GpuStream, KernelArgs, KernelId, Transform2D};
use tracing::debug;

use crate::CompositorError;

/// Block size for 2D kernel dispatch (16x16 = 256 threads per block).
const BLOCK_SIZE: u32 = 16;

/// Dispatch the 2D transform kernel to map a source frame into the output coordinate space.
///
/// The kernel reads from `src` and writes RGBA pixels to `dst_ptr`, applying the
/// full affine transform defined by [`Transform2D`] (position, scale, rotation, anchor).
///
/// Out-of-bounds source pixels are written as transparent (RGBA = 0).
///
/// # Arguments
///
/// * `backend` - GPU backend for kernel dispatch
/// * `src` - Source GPU frame
/// * `dst_ptr` - Device pointer to the destination RGBA buffer (output resolution)
/// * `output_width` - Width of the destination buffer in pixels
/// * `output_height` - Height of the destination buffer in pixels
/// * `transform` - The 2D transform to apply
/// * `stream` - GPU stream for async execution
pub fn dispatch_transform(
    backend: &dyn GpuBackend,
    src: &GpuFrame,
    dst_ptr: u64,
    output_width: u32,
    output_height: u32,
    transform: &Transform2D,
    stream: &GpuStream,
) -> Result<(), CompositorError> {
    debug!(
        src_w = src.resolution.width,
        src_h = src.resolution.height,
        output_w = output_width,
        output_h = output_height,
        position = ?transform.position,
        scale = ?transform.scale,
        rotation = transform.rotation,
        anchor = ?transform.anchor,
        "Dispatching transform kernel"
    );

    let args = KernelArgs::new()
        .push_ptr(src.device_ptr) // src RGBA buffer
        .push_u32(src.resolution.width) // src width
        .push_u32(src.resolution.height) // src height
        .push_u32(src.pitch) // src pitch in bytes
        .push_ptr(dst_ptr) // dst RGBA buffer
        .push_u32(output_width) // dst width
        .push_u32(output_height) // dst height
        .push_vec2(transform.position) // translation (px)
        .push_vec2(transform.scale) // scale factors
        .push_f32(transform.rotation) // rotation (degrees)
        .push_vec2(transform.anchor); // anchor point (normalized)

    let grid = [
        output_width.div_ceil(BLOCK_SIZE),
        output_height.div_ceil(BLOCK_SIZE),
        1,
    ];
    let block = [BLOCK_SIZE, BLOCK_SIZE, 1];

    backend.dispatch_kernel(&KernelId::Transform, grid, block, &args, stream)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_args_count() {
        let transform = Transform2D::default();
        let args = KernelArgs::new()
            .push_ptr(0x1000)
            .push_u32(1920)
            .push_u32(1080)
            .push_u32(1920 * 4)
            .push_ptr(0x2000)
            .push_u32(1920)
            .push_u32(1080)
            .push_vec2(transform.position)
            .push_vec2(transform.scale)
            .push_f32(transform.rotation)
            .push_vec2(transform.anchor);
        assert_eq!(args.len(), 11);
    }

    #[test]
    fn grid_covers_full_output() {
        let w = 3840u32;
        let h = 2160u32;
        let grid_x = w.div_ceil(BLOCK_SIZE);
        let grid_y = h.div_ceil(BLOCK_SIZE);
        // Grid must cover at least every pixel
        assert!(grid_x * BLOCK_SIZE >= w);
        assert!(grid_y * BLOCK_SIZE >= h);
    }
}
