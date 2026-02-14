//! `ms-gpu-hal` — GPU Hardware Abstraction Layer.
//!
//! Provides CUDA and Vulkan Compute backends behind the `GpuBackend` trait
//! defined in `ms-common`.

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "vulkan")]
pub mod vulkan;

pub mod kernels;
pub mod select;
