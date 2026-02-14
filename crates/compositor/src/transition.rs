//! Transition kernel dispatch.
//!
//! Dispatches the `transition` GPU kernel for blending between two frames
//! during a timeline transition (e.g., cross-dissolve, wipe, push).

use ms_common::{GpuBackend, GpuFrame, GpuStream, KernelArgs, KernelId};
use tracing::debug;

use crate::CompositorError;

/// Block size for 2D kernel dispatch (16x16 = 256 threads per block).
const BLOCK_SIZE: u32 = 16;

/// Transition type integers for the GPU kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TransitionType {
    /// Simple cross-dissolve (linear interpolation).
    CrossDissolve = 0,
    /// Horizontal wipe (left to right).
    WipeLeft = 1,
    /// Vertical wipe (top to bottom).
    WipeDown = 2,
    /// Push transition (incoming pushes outgoing off screen).
    PushLeft = 3,
    /// Dip to black.
    DipToBlack = 4,
}

/// Parameters for a transition kernel dispatch.
pub struct TransitionParams<'a> {
    /// The outgoing frame (source A).
    pub from: &'a GpuFrame,
    /// The incoming frame (source B).
    pub to: &'a GpuFrame,
    /// Device pointer to the destination RGBA buffer.
    pub dst_ptr: u64,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Transition progress (0.0 = fully A, 1.0 = fully B).
    pub progress: f32,
    /// The type of transition to apply.
    pub transition_type: TransitionType,
}

/// Dispatch the transition kernel to blend between two frames.
pub fn dispatch_transition(
    backend: &dyn GpuBackend,
    params: &TransitionParams<'_>,
    stream: &GpuStream,
) -> Result<(), CompositorError> {
    let progress = params.progress.clamp(0.0, 1.0);

    debug!(
        transition = ?params.transition_type,
        progress = progress,
        width = params.width,
        height = params.height,
        "Dispatching transition kernel"
    );

    let args = KernelArgs::new()
        .push_ptr(params.from.device_ptr) // outgoing frame
        .push_ptr(params.to.device_ptr) // incoming frame
        .push_ptr(params.dst_ptr) // output buffer
        .push_u32(params.width)
        .push_u32(params.height)
        .push_f32(progress)
        .push_u32(params.transition_type as u32);

    let grid = [
        params.width.div_ceil(BLOCK_SIZE),
        params.height.div_ceil(BLOCK_SIZE),
        1,
    ];
    let block = [BLOCK_SIZE, BLOCK_SIZE, 1];

    backend.dispatch_kernel(&KernelId::Transition, grid, block, &args, stream)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_type_values() {
        assert_eq!(TransitionType::CrossDissolve as u32, 0);
        assert_eq!(TransitionType::WipeLeft as u32, 1);
        assert_eq!(TransitionType::WipeDown as u32, 2);
        assert_eq!(TransitionType::PushLeft as u32, 3);
        assert_eq!(TransitionType::DipToBlack as u32, 4);
    }

    #[test]
    fn transition_args_count() {
        let args = KernelArgs::new()
            .push_ptr(0x1000)
            .push_ptr(0x2000)
            .push_ptr(0x3000)
            .push_u32(1920)
            .push_u32(1080)
            .push_f32(0.5)
            .push_u32(TransitionType::CrossDissolve as u32);
        assert_eq!(args.len(), 7);
    }

    #[test]
    fn grid_for_uhd() {
        let width = 3840u32;
        let height = 2160u32;
        let grid_x = width.div_ceil(BLOCK_SIZE);
        let grid_y = height.div_ceil(BLOCK_SIZE);
        assert_eq!(grid_x, 240);
        assert_eq!(grid_y, 135);
    }
}
