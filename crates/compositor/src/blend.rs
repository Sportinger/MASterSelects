//! Blend mode kernel dispatch.
//!
//! Dispatches the `alpha_blend` GPU kernel with the appropriate blend mode
//! integer, opacity, and source/destination pointers.

use ms_common::{BlendMode, GpuBackend, GpuFrame, GpuStream, KernelArgs, KernelId};
use tracing::debug;

use crate::color::blend_mode_to_int;
use crate::CompositorError;

/// Block size for 2D kernel dispatch (16x16 = 256 threads per block).
const BLOCK_SIZE: u32 = 16;

/// Parameters for an alpha-blend kernel dispatch.
pub struct BlendParams<'a> {
    /// Source frame on GPU (already transformed / effect-processed).
    pub src: &'a GpuFrame,
    /// Device pointer to the destination RGBA buffer.
    pub dst_ptr: u64,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Layer opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// The blend mode to apply.
    pub blend_mode: &'a BlendMode,
}

/// Dispatch the alpha-blend kernel to composite a source frame onto a destination buffer.
pub fn dispatch_blend(
    backend: &dyn GpuBackend,
    params: &BlendParams<'_>,
    stream: &GpuStream,
) -> Result<(), CompositorError> {
    let mode_int = blend_mode_to_int(params.blend_mode);

    debug!(
        blend_mode = ?params.blend_mode,
        mode_int = mode_int,
        opacity = params.opacity,
        width = params.width,
        height = params.height,
        "Dispatching alpha blend kernel"
    );

    let args = KernelArgs::new()
        .push_ptr(params.src.device_ptr) // src RGBA buffer
        .push_ptr(params.dst_ptr) // dst RGBA buffer (read+write)
        .push_u32(params.width)
        .push_u32(params.height)
        .push_f32(params.opacity)
        .push_u32(mode_int);

    let grid = [
        params.width.div_ceil(BLOCK_SIZE),
        params.height.div_ceil(BLOCK_SIZE),
        1,
    ];
    let block = [BLOCK_SIZE, BLOCK_SIZE, 1];

    backend.dispatch_kernel(&KernelId::AlphaBlend, grid, block, &args, stream)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_calculation_exact_multiple() {
        let width = 1920u32;
        let height = 1080u32;
        let grid_x = width.div_ceil(BLOCK_SIZE);
        let grid_y = height.div_ceil(BLOCK_SIZE);
        assert_eq!(grid_x, 120);
        assert_eq!(grid_y, 68); // ceil(1080/16) = 68
    }

    #[test]
    fn grid_calculation_non_multiple() {
        let width = 1921u32;
        let height = 1081u32;
        let grid_x = width.div_ceil(BLOCK_SIZE);
        let grid_y = height.div_ceil(BLOCK_SIZE);
        assert_eq!(grid_x, 121);
        assert_eq!(grid_y, 68); // ceil(1081/16) = 68
    }

    #[test]
    fn blend_args_are_constructed_correctly() {
        let mode = BlendMode::Screen;
        let mode_int = blend_mode_to_int(&mode);
        let args = KernelArgs::new()
            .push_ptr(0x1000)
            .push_ptr(0x2000)
            .push_u32(1920)
            .push_u32(1080)
            .push_f32(0.75)
            .push_u32(mode_int);
        assert_eq!(args.len(), 6);
    }
}
