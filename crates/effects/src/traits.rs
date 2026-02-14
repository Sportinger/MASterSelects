//! Effect trait definition.
//!
//! All GPU effects implement [`Effect`], which describes their parameters,
//! kernel dispatch info, and argument layout. The actual GPU dispatch is
//! performed by the compositor using [`ms_common::GpuBackend`].

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamValue};

/// Trait for all GPU effects.
///
/// Effects define their parameters and kernel dispatch info.
/// The actual GPU dispatch is done by the compositor using `GpuBackend`.
pub trait Effect: Send + Sync {
    /// Unique effect name (matches [`KernelId::Effect`] name and kernel file).
    fn name(&self) -> &str;

    /// Display name for UI.
    fn display_name(&self) -> &str;

    /// Category for UI grouping.
    fn category(&self) -> EffectCategory;

    /// Parameter definitions (for UI generation and validation).
    fn param_defs(&self) -> &[ParamDef];

    /// The kernel ID to dispatch for this effect.
    fn kernel_id(&self) -> KernelId;

    /// Build kernel arguments from effect parameters.
    ///
    /// This converts [`ParamValue`]s into the correct [`KernelArgs`] layout
    /// that the GPU kernel expects.
    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs;

    /// Compute grid dimensions `(grid, block)` for kernel dispatch.
    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]);

    /// Number of passes needed (most effects are single-pass).
    fn num_passes(&self) -> u32 {
        1
    }
}

/// Standard 16x16 compute grid used by most single-pass image effects.
pub fn standard_compute_grid(width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
    let block = [16, 16, 1];
    let grid = [width.div_ceil(block[0]), height.div_ceil(block[1]), 1];
    (grid, block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_grid_exact_multiple() {
        let (grid, block) = standard_compute_grid(1920, 1080);
        assert_eq!(block, [16, 16, 1]);
        assert_eq!(grid, [120, 68, 1]); // 1920/16 = 120, ceil(1080/16) = 68 (67.5 -> 68)
    }

    #[test]
    fn standard_grid_non_multiple() {
        let (grid, block) = standard_compute_grid(100, 100);
        assert_eq!(block, [16, 16, 1]);
        assert_eq!(grid, [7, 7, 1]); // ceil(100/16) = 7
    }

    #[test]
    fn standard_grid_small() {
        let (grid, _) = standard_compute_grid(1, 1);
        assert_eq!(grid, [1, 1, 1]);
    }
}
