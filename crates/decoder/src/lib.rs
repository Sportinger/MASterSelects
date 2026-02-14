//! `ms-decoder` — Hardware decode management.
//!
//! Manages NVDEC/Vulkan Video decoder sessions, decode pools,
//! prefetch ring buffers, and thumbnail generation.
//!
//! # Architecture
//!
//! The decoder crate provides hardware-accelerated video decoding through
//! NVIDIA's NVDEC (via the CUVID API). The nvcuvid library is loaded
//! dynamically at runtime, allowing graceful fallback on systems without
//! NVIDIA hardware.
//!
//! ## Module Overview
//!
//! - [`nvdec`] — NVDEC hardware decoder (NVIDIA GPUs)
//!   - [`nvdec::ffi`] — Raw FFI bindings for nvcuvid
//!   - [`nvdec::session`] — Safe decoder session with RAII
//!   - [`nvdec::decoder`] — High-level `HwDecoder` implementation
//! - [`manager`] — Decoder pool manager
//!
//! ## Usage
//!
//! ```ignore
//! use ms_decoder::manager::{DecoderManager, DecoderCreateConfig, DecoderId};
//! use ms_common::VideoCodec;
//!
//! let mut manager = DecoderManager::new();
//!
//! if manager.is_nvdec_available() {
//!     let config = DecoderCreateConfig {
//!         codec: VideoCodec::H264,
//!         ..Default::default()
//!     };
//!     let decoder = manager.create_decoder(
//!         DecoderId::new("my_video.mp4"),
//!         &config,
//!     )?;
//!     // Feed packets from the demuxer...
//! }
//! ```

pub mod manager;
pub mod nvdec;
