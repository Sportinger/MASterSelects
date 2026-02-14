//! Vulkan Video hardware decoder stub.
//!
//! This module provides the Vulkan Video decode backend for the MasterSelects
//! engine. Vulkan Video is a set of Vulkan extensions that expose hardware
//! video decode capabilities on AMD, Intel, and NVIDIA GPUs through a
//! vendor-neutral API.
//!
//! # Required Vulkan Extensions
//!
//! Vulkan Video decode requires the following extensions, which must be
//! supported by both the driver and the physical device:
//!
//! - **`VK_KHR_video_queue`** — Core video queue infrastructure (command
//!   buffers, video sessions, DPB management).
//! - **`VK_KHR_video_decode_queue`** — Decode-specific command buffer
//!   operations and decode output picture management.
//! - **`VK_KHR_video_decode_h264`** — H.264/AVC decode profile, SPS/PPS
//!   parameter sets, slice-level decode parameters.
//! - **`VK_KHR_video_decode_h265`** — H.265/HEVC decode profile, VPS/SPS/PPS
//!   parameter sets, slice segment header decode parameters.
//! - **`VK_KHR_video_decode_av1`** (provisional) — AV1 decode profile,
//!   sequence header, frame header, tile-level decode parameters.
//!
//! Additionally, the following extensions are typically required for
//! integration with the compute/graphics pipeline:
//!
//! - **`VK_KHR_sampler_ycbcr_conversion`** — YCbCr (NV12/P010) image
//!   format support and automatic color space conversion.
//! - **`VK_KHR_synchronization2`** — Fine-grained pipeline barriers for
//!   video decode queue to compute queue transitions.
//!
//! # Current Status
//!
//! This is a **stub implementation** that always returns errors. Vulkan
//! Video decode requires specific driver support that is still maturing:
//!
//! - **NVIDIA**: Supported since driver 525+ (beta), stable in 535+
//! - **AMD**: Supported in RADV (Mesa 23.1+) and AMDVLK
//! - **Intel**: Supported in ANV (Mesa 23.1+) for integrated GPUs
//!
//! The full implementation will:
//! 1. Query `VkPhysicalDeviceVideoCapabilitiesKHR` for supported codecs
//! 2. Create a `VkVideoSessionKHR` with the appropriate decode profile
//! 3. Allocate DPB (Decoded Picture Buffer) images for reference frames
//! 4. Record decode operations into video-capable command buffers
//! 5. Synchronize with compute queue for post-processing (NV12→RGBA, effects)
//!
//! # Architecture Notes
//!
//! Unlike NVDEC which uses a callback-based parser (CUVID), Vulkan Video
//! requires the application to perform its own bitstream parsing and provide
//! structured parameter sets (SPS, PPS, slice headers) to the decode API.
//! This means we need an H.264/H.265 NAL unit parser in addition to the
//! container demuxer.

pub mod decoder;

pub use decoder::VulkanVideoDecoder;
