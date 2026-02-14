//! `ms-compositor` — GPU-accelerated layer compositing for the MasterSelects engine.
//!
//! This crate composites multiple [`LayerDesc`] layers into a final output frame
//! using GPU kernel dispatch through the backend-agnostic [`GpuBackend`] trait.
//!
//! The compositing pipeline per layer:
//! 1. **Transform** — position, scale, rotation via [`dispatch_transform`]
//! 2. **Effects** — per-layer GPU effects via effect kernel IDs
//! 3. **Mask** — optional mask application via [`dispatch_mask`]
//! 4. **Blend** — blend onto the output buffer via [`dispatch_blend`]
//!
//! [`LayerDesc`]: ms_common::LayerDesc
//! [`GpuBackend`]: ms_common::GpuBackend
//! [`dispatch_transform`]: transform::dispatch_transform
//! [`dispatch_mask`]: mask::dispatch_mask
//! [`dispatch_blend`]: blend::dispatch_blend

pub mod blend;
pub mod color;
pub mod compositor;
pub mod mask;
pub mod pipeline;
pub mod transform;
pub mod transition;

mod error;

// Re-export primary API
pub use compositor::Compositor;
pub use error::CompositorError;
pub use pipeline::RenderPipeline;
