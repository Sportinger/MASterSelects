//! Vulkan Video hardware encoder stub.
//!
//! This module provides the Vulkan Video encode backend for the MasterSelects
//! engine. Vulkan Video encoding is a set of Vulkan extensions that expose
//! hardware video encode capabilities on AMD, Intel, and NVIDIA GPUs through
//! a vendor-neutral API.
//!
//! # Required Vulkan Extensions
//!
//! Vulkan Video encode requires the following extensions, which must be
//! supported by both the driver and the physical device:
//!
//! - **`VK_KHR_video_queue`** — Core video queue infrastructure (command
//!   buffers, video sessions, rate control).
//! - **`VK_KHR_video_encode_queue`** — Encode-specific command buffer
//!   operations, rate control parameters, and bitstream output management.
//! - **`VK_KHR_video_encode_h264`** — H.264/AVC encode profile, SPS/PPS
//!   generation, slice-level encode parameters, rate control.
//! - **`VK_KHR_video_encode_h265`** — H.265/HEVC encode profile, VPS/SPS/PPS
//!   generation, CTU-level encode parameters.
//!
//! Additionally, the following extensions are typically needed:
//!
//! - **`VK_KHR_sampler_ycbcr_conversion`** — YCbCr input format support
//!   (encode typically accepts NV12/P010 input).
//! - **`VK_KHR_synchronization2`** — Fine-grained pipeline barriers for
//!   compute queue to video encode queue transitions.
//!
//! # Current Status
//!
//! This is a **stub implementation** that always returns errors. Vulkan
//! Video encode support is newer than decode and driver support is more
//! limited:
//!
//! - **NVIDIA**: Supported since driver 535+ (beta)
//! - **AMD**: Supported in RADV (Mesa 24.0+) for RDNA2+
//! - **Intel**: Supported in ANV (Mesa 24.0+) for Arc GPUs
//!
//! # Encode Pipeline (Future)
//!
//! ```text
//! GpuFrame (NV12/RGBA on device)
//!   --> VkVideoSessionKHR (encode session)
//!     --> vkCmdEncodeVideoKHR (per-frame encode command)
//!       --> VkVideoEncodeInfoKHR (encode parameters)
//!         --> Rate control (CBR/VBR/CQP)
//!           --> Bitstream output buffer
//!             --> EncodedPacket
//!               --> Muxer
//! ```
//!
//! The full implementation will:
//! 1. Create a `VkVideoSessionKHR` with the appropriate encode profile.
//! 2. Configure rate control via `VkVideoEncodeRateControlInfoKHR`.
//! 3. Allocate DPB images for reference frame management.
//! 4. Record `vkCmdEncodeVideoKHR` operations per frame.
//! 5. Read back the bitstream from output buffers into `EncodedPacket`.

pub mod encoder;

pub use encoder::VulkanVideoEncoder;
