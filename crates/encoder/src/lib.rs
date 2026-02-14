//! `ms-encoder` -- Hardware video encoding management.
//!
//! Provides hardware-accelerated video encoding through NVIDIA's NVENC API
//! and Vulkan Video encode extensions. Libraries are loaded dynamically at
//! runtime, allowing graceful fallback on systems without supported hardware.
//!
//! # Architecture
//!
//! The encoder crate mirrors the decoder crate's architecture:
//!
//! - [`nvenc`] -- NVENC hardware encoder (NVIDIA GPUs)
//!   - [`nvenc::ffi`] -- Raw FFI bindings for nvEncodeAPI
//!   - [`nvenc::params`] -- Parameter builders mapping common config to NVENC structs
//!   - [`nvenc::buffer`] -- Input/output buffer pool management
//!   - [`nvenc::NvEncoder`] -- High-level `HwEncoder` implementation
//! - [`vulkan_video`] -- Vulkan Video hardware encoder (AMD/Intel/NVIDIA)
//!   - [`vulkan_video::encoder`] -- `HwEncoder` implementation (stub)
//! - [`session`] -- Encoder session with frame counting and PTS tracking
//! - [`export`] -- Export pipeline (timeline eval -> compositor -> encoder -> muxer)
//!
//! # Encode Pipeline
//!
//! ```text
//! GpuFrame (NV12/RGBA on device)
//!   --> Register as NVENC input (external CUDA device ptr)
//!     --> NV_ENC_PIC_PARAMS (per-frame encode params)
//!       --> NVENC hardware encode
//!         --> Lock bitstream output buffer
//!           --> Copy to EncodedPacket
//!             --> Muxer
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ms_encoder::session::EncoderSession;
//! use ms_common::config::{EncoderConfig, EncoderBitrate, EncoderPreset, EncoderProfile};
//! use ms_common::{VideoCodec, Resolution, Rational};
//!
//! let config = EncoderConfig {
//!     codec: VideoCodec::H264,
//!     resolution: Resolution::HD,
//!     fps: Rational::FPS_30,
//!     bitrate: EncoderBitrate::Vbr { target: 20_000_000, max: 30_000_000 },
//!     preset: EncoderPreset::Medium,
//!     profile: EncoderProfile::High,
//! };
//!
//! let mut session = EncoderSession::new(&config)?;
//!
//! // Encode frames from the compositor
//! for frame in compositor_output {
//!     let packet = session.encode(&frame)?;
//!     muxer.write_packet(&packet)?;
//! }
//!
//! // Flush remaining packets
//! let remaining = session.flush()?;
//! for packet in &remaining {
//!     muxer.write_packet(packet)?;
//! }
//! ```

pub mod error;
pub mod export;
#[cfg(feature = "nvenc")]
pub mod nvenc;
pub mod session;
pub mod vulkan_video;

pub use vulkan_video::VulkanVideoEncoder;
