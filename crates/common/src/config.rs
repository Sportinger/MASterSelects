//! Configuration structs for engine, decoder, encoder, and render settings.

use serde::{Deserialize, Serialize};

use crate::codec::VideoCodec;
use crate::color::PixelFormat;
use crate::types::{Rational, Resolution};

/// GPU backend preference.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuPreference {
    /// Auto-detect: CUDA if available, then Vulkan.
    #[default]
    Auto,
    /// Force CUDA backend (NVIDIA only).
    ForceCuda,
    /// Force Vulkan Compute backend.
    ForceVulkan,
}

/// Top-level engine configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    pub gpu_preference: GpuPreference,
    pub preview_resolution: Resolution,
    pub preview_fps: Rational,
    /// Maximum decode threads (0 = auto).
    pub max_decode_threads: u32,
    /// Prefetch buffer size in frames.
    pub prefetch_frames: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            gpu_preference: GpuPreference::Auto,
            preview_resolution: Resolution::HD,
            preview_fps: Rational::FPS_30,
            max_decode_threads: 0,
            prefetch_frames: 8,
        }
    }
}

/// Hardware decoder configuration.
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    pub codec: VideoCodec,
    pub resolution: Resolution,
    pub output_format: PixelFormat,
    /// Number of decode surfaces (ring buffer depth).
    pub num_surfaces: u32,
    /// Whether to enable deinterlacing.
    pub deinterlace: bool,
}

impl DecoderConfig {
    pub fn new(codec: VideoCodec, resolution: Resolution) -> Self {
        Self {
            codec,
            resolution,
            output_format: PixelFormat::Nv12,
            num_surfaces: 8,
            deinterlace: false,
        }
    }
}

/// Hardware encoder configuration.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub codec: VideoCodec,
    pub resolution: Resolution,
    pub fps: Rational,
    pub bitrate: EncoderBitrate,
    pub preset: EncoderPreset,
    pub profile: EncoderProfile,
}

/// Bitrate control mode for encoding.
#[derive(Clone, Debug)]
pub enum EncoderBitrate {
    /// Constant Bitrate (bits/sec).
    Cbr(u64),
    /// Variable Bitrate (target bits/sec, max bits/sec).
    Vbr { target: u64, max: u64 },
    /// Constant QP (quality parameter, 0-51 for H.264).
    Cqp(u32),
}

/// Encoder speed/quality preset.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EncoderPreset {
    Fastest,
    Fast,
    Medium,
    Slow,
    Lossless,
}

/// Encoder profile.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EncoderProfile {
    Baseline,
    Main,
    High,
    High10,
}

/// Render/compositing configuration.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub output_resolution: Resolution,
    pub output_format: PixelFormat,
    /// Use linear color space for compositing (recommended).
    pub linear_compositing: bool,
    /// Number of temporary GPU buffers for ping-pong rendering.
    pub temp_buffer_count: u32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            output_resolution: Resolution::HD,
            output_format: PixelFormat::Rgba8,
            linear_compositing: true,
            temp_buffer_count: 2,
        }
    }
}
