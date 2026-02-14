//! Vulkan Video encoder — implements the `HwEncoder` trait.
//!
//! `VulkanVideoEncoder` is the Vulkan Video counterpart of the NVENC encoder.
//! It will eventually provide hardware-accelerated encoding on AMD, Intel, and
//! NVIDIA GPUs through the `VK_KHR_video_encode_queue` family of extensions.
//!
//! # Current Status
//!
//! This is a **stub**. All encode operations return
//! [`EncodeError::HwEncoderInit`] and [`is_available()`] returns `false`.
//! The struct exists so that:
//!
//! 1. The `VulkanBackend::create_encoder()` method can return a concrete
//!    encoder type instead of an opaque error.
//! 2. The encoder crate's module structure is ready for the real
//!    implementation in Phase 3.
//! 3. Downstream code can conditionally check `VulkanVideoEncoder::is_available()`
//!    to determine if Vulkan Video encode is supported.
//!
//! # Future Implementation Plan
//!
//! The full implementation will require:
//!
//! - A reference to the `VulkanBackend` (or at least the `ash::Device` and
//!   video-capable queue family index with `VIDEO_ENCODE_BIT_KHR`).
//! - A `VkVideoSessionKHR` configured for the target encode profile.
//! - Rate control configuration via `VkVideoEncodeRateControlInfoKHR`:
//!   - CBR (constant bitrate) via `VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR`
//!   - VBR (variable bitrate) via `VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR`
//!   - CQP (constant QP) via `VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR`
//! - DPB management for P/B reference frames.
//! - Bitstream output buffer allocation and readback.
//! - GOP (Group of Pictures) structure management (I/P/B frame scheduling).

use ms_common::config::EncoderConfig;
use ms_common::gpu_traits::{EncodedPacket, HwEncoder};
use ms_common::packet::GpuFrame;
use ms_common::{EncodeError, VideoCodec};

use tracing::warn;

// ---------------------------------------------------------------------------
// Supported codecs
// ---------------------------------------------------------------------------

/// Vulkan Video encode extensions and the codecs they support.
///
/// This list reflects the Vulkan extensions that *exist* in the specification.
/// Actual runtime availability depends on the GPU driver. Use
/// [`VulkanVideoEncoder::is_available()`] to check at runtime.
const VULKAN_VIDEO_ENCODE_CODECS: &[(VideoCodec, &str)] = &[
    (VideoCodec::H264, "VK_KHR_video_encode_h264"),
    (VideoCodec::H265, "VK_KHR_video_encode_h265"),
    // AV1 encode is not yet part of the Vulkan Video specification.
    // VP9 encode is not part of the Vulkan Video specification.
];

/// Check whether a codec has a corresponding Vulkan Video encode extension.
fn codec_has_vulkan_encode_extension(codec: VideoCodec) -> bool {
    VULKAN_VIDEO_ENCODE_CODECS
        .iter()
        .any(|(c, _)| *c == codec)
}

/// Get the Vulkan extension name for an encode codec (if it exists).
fn vulkan_encode_extension_for_codec(codec: VideoCodec) -> Option<&'static str> {
    VULKAN_VIDEO_ENCODE_CODECS
        .iter()
        .find(|(c, _)| *c == codec)
        .map(|(_, ext)| *ext)
}

// ---------------------------------------------------------------------------
// VulkanVideoEncoder
// ---------------------------------------------------------------------------

/// Vulkan Video hardware encoder.
///
/// Implements the [`HwEncoder`] trait for Vulkan Video encode. Currently a
/// stub that returns errors for all operations. See the module-level
/// documentation for the required Vulkan extensions and implementation plan.
///
/// # Example
///
/// ```ignore
/// use ms_encoder::vulkan_video::VulkanVideoEncoder;
/// use ms_common::config::{
///     EncoderConfig, EncoderBitrate, EncoderPreset, EncoderProfile,
/// };
/// use ms_common::{VideoCodec, Resolution, Rational};
///
/// // Check availability first
/// if VulkanVideoEncoder::is_available() {
///     let config = EncoderConfig {
///         codec: VideoCodec::H264,
///         resolution: Resolution::HD,
///         fps: Rational::FPS_30,
///         bitrate: EncoderBitrate::Vbr { target: 20_000_000, max: 30_000_000 },
///         preset: EncoderPreset::Medium,
///         profile: EncoderProfile::High,
///     };
///     let mut encoder = VulkanVideoEncoder::new(&config)?;
///     // Encode frames from compositor...
/// } else {
///     // Fall back to NVENC or software encode
/// }
/// ```
pub struct VulkanVideoEncoder {
    /// The video codec this encoder is configured for.
    codec: VideoCodec,
    /// The encoder configuration.
    config: EncoderConfig,
    /// Total frames submitted to the encoder.
    frames_encoded: u64,
}

impl std::fmt::Debug for VulkanVideoEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanVideoEncoder")
            .field("codec", &self.codec)
            .field("resolution", &self.config.resolution)
            .field("fps", &self.config.fps)
            .field("frames_encoded", &self.frames_encoded)
            .field("available", &false)
            .finish()
    }
}

impl VulkanVideoEncoder {
    /// Create a new Vulkan Video encoder for the given configuration.
    ///
    /// # Current Behavior
    ///
    /// Validates that the requested codec has a corresponding Vulkan Video
    /// encode extension in the specification. The struct is created but all
    /// subsequent `encode()` calls will return errors because the Vulkan
    /// Video encode implementation is not yet complete.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::UnsupportedCodec`] if the codec does not have
    /// a corresponding Vulkan Video encode extension (currently only H.264
    /// and H.265 are specified for Vulkan Video encode).
    pub fn new(config: &EncoderConfig) -> Result<Self, EncodeError> {
        if !codec_has_vulkan_encode_extension(config.codec) {
            return Err(EncodeError::UnsupportedCodec(config.codec));
        }

        let ext_name = vulkan_encode_extension_for_codec(config.codec).unwrap_or("unknown");

        warn!(
            codec = config.codec.display_name(),
            extension = ext_name,
            width = config.resolution.width,
            height = config.resolution.height,
            "VulkanVideoEncoder created (stub — encode operations will fail)"
        );

        Ok(Self {
            codec: config.codec,
            config: config.clone(),
            frames_encoded: 0,
        })
    }

    /// Check if Vulkan Video encode is available on the current system.
    ///
    /// # Current Behavior
    ///
    /// Always returns `false`. The full implementation will:
    ///
    /// 1. Enumerate physical devices via `vkEnumeratePhysicalDevices`.
    /// 2. Query `VkPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR`.
    /// 3. Check that a video encode queue family exists
    ///    (`VK_QUEUE_VIDEO_ENCODE_BIT_KHR`).
    /// 4. Verify that at least one encode codec profile is supported.
    ///
    /// # Performance Note
    ///
    /// This function should be called once at startup and cached, not on
    /// every frame. Vulkan device enumeration is expensive.
    pub fn is_available() -> bool {
        // TODO(phase-3): Implement actual Vulkan Video encode capability detection.
        // This requires:
        // 1. A VkInstance (possibly shared with the VulkanBackend)
        // 2. Querying VK_KHR_video_queue + VK_KHR_video_encode_queue support
        // 3. Checking for a queue family with VIDEO_ENCODE_BIT_KHR
        // 4. Verifying VK_KHR_video_encode_h264/h265 device extension support
        false
    }

    /// Check if a specific codec is supported for Vulkan Video encode.
    ///
    /// # Current Behavior
    ///
    /// Always returns `false`. The full implementation will query the device's
    /// encode profile capabilities for the given codec.
    pub fn supports_codec(_codec: VideoCodec) -> bool {
        // TODO(phase-3): Query VkVideoProfileListInfoKHR encode capabilities.
        false
    }

    /// Get the Vulkan extension name required for the configured codec.
    pub fn required_extension(&self) -> Option<&'static str> {
        vulkan_encode_extension_for_codec(self.codec)
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Get the number of frames encoded so far.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }
}

impl HwEncoder for VulkanVideoEncoder {
    /// Encode a GPU frame.
    ///
    /// # Current Behavior
    ///
    /// Always returns [`EncodeError::HwEncoderInit`] because the Vulkan
    /// Video encode implementation is not yet complete.
    ///
    /// # Future Behavior
    ///
    /// Will record an encode operation into a video command buffer:
    /// 1. Transition the input image layout for video encode.
    /// 2. Configure per-frame encode parameters (QP, frame type, references).
    /// 3. Record `vkCmdEncodeVideoKHR` with codec-specific encode info.
    /// 4. Submit and wait for encode completion.
    /// 5. Lock the bitstream output buffer and copy to `EncodedPacket`.
    fn encode(&mut self, _frame: &GpuFrame) -> Result<EncodedPacket, EncodeError> {
        Err(EncodeError::HwEncoderInit(format!(
            "Vulkan Video encode not yet implemented (requires {})",
            self.required_extension().unwrap_or("unknown extension")
        )))
    }

    /// Flush remaining packets from the encode pipeline.
    ///
    /// # Current Behavior
    ///
    /// Always returns [`EncodeError::HwEncoderInit`] because the Vulkan
    /// Video encode implementation is not yet complete.
    ///
    /// # Future Behavior
    ///
    /// Will signal end-of-stream to the encoder, flush any remaining
    /// reference frames, and return all pending encoded packets.
    fn flush(&mut self) -> Result<Vec<EncodedPacket>, EncodeError> {
        Err(EncodeError::HwEncoderInit(
            "Vulkan Video encode not yet implemented — nothing to flush".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_common::config::{EncoderBitrate, EncoderPreset, EncoderProfile};
    use ms_common::types::{Rational, Resolution};

    fn make_config(codec: VideoCodec) -> EncoderConfig {
        EncoderConfig {
            codec,
            resolution: Resolution::HD,
            fps: Rational::FPS_30,
            bitrate: EncoderBitrate::Vbr {
                target: 20_000_000,
                max: 30_000_000,
            },
            preset: EncoderPreset::Medium,
            profile: EncoderProfile::High,
        }
    }

    fn make_frame() -> GpuFrame {
        GpuFrame {
            device_ptr: 0x1000_0000,
            device_ptr_uv: Some(0x1000_0000 + 1920 * 1080),
            resolution: Resolution::HD,
            format: ms_common::PixelFormat::Nv12,
            pitch: 1920,
            pts: ms_common::types::TimeCode::ZERO,
        }
    }

    #[test]
    fn is_not_available() {
        assert!(!VulkanVideoEncoder::is_available());
    }

    #[test]
    fn does_not_support_any_codec() {
        assert!(!VulkanVideoEncoder::supports_codec(VideoCodec::H264));
        assert!(!VulkanVideoEncoder::supports_codec(VideoCodec::H265));
        assert!(!VulkanVideoEncoder::supports_codec(VideoCodec::Av1));
        assert!(!VulkanVideoEncoder::supports_codec(VideoCodec::Vp9));
    }

    #[test]
    fn creation_succeeds_for_h264() {
        let config = make_config(VideoCodec::H264);
        let encoder = VulkanVideoEncoder::new(&config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn creation_succeeds_for_h265() {
        let config = make_config(VideoCodec::H265);
        let encoder = VulkanVideoEncoder::new(&config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn creation_fails_for_av1() {
        // AV1 has no Vulkan Video encode extension yet.
        let config = make_config(VideoCodec::Av1);
        let encoder = VulkanVideoEncoder::new(&config);
        assert!(encoder.is_err());
    }

    #[test]
    fn creation_fails_for_vp9() {
        // VP9 has no Vulkan Video encode extension.
        let config = make_config(VideoCodec::Vp9);
        let encoder = VulkanVideoEncoder::new(&config);
        assert!(encoder.is_err());
    }

    #[test]
    fn encode_returns_error() {
        let config = make_config(VideoCodec::H264);
        let mut encoder = VulkanVideoEncoder::new(&config).unwrap();

        let frame = make_frame();
        let result = encoder.encode(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn flush_returns_error() {
        let config = make_config(VideoCodec::H264);
        let mut encoder = VulkanVideoEncoder::new(&config).unwrap();
        let result = encoder.flush();
        assert!(result.is_err());
    }

    #[test]
    fn required_extension_for_h264() {
        let config = make_config(VideoCodec::H264);
        let encoder = VulkanVideoEncoder::new(&config).unwrap();
        assert_eq!(
            encoder.required_extension(),
            Some("VK_KHR_video_encode_h264")
        );
    }

    #[test]
    fn required_extension_for_h265() {
        let config = make_config(VideoCodec::H265);
        let encoder = VulkanVideoEncoder::new(&config).unwrap();
        assert_eq!(
            encoder.required_extension(),
            Some("VK_KHR_video_encode_h265")
        );
    }

    #[test]
    fn codec_encode_extension_mapping() {
        assert!(codec_has_vulkan_encode_extension(VideoCodec::H264));
        assert!(codec_has_vulkan_encode_extension(VideoCodec::H265));
        assert!(!codec_has_vulkan_encode_extension(VideoCodec::Av1));
        assert!(!codec_has_vulkan_encode_extension(VideoCodec::Vp9));
    }

    #[test]
    fn debug_format() {
        let config = make_config(VideoCodec::H264);
        let encoder = VulkanVideoEncoder::new(&config).unwrap();
        let debug_str = format!("{encoder:?}");
        assert!(debug_str.contains("VulkanVideoEncoder"));
        assert!(debug_str.contains("H264"));
        assert!(debug_str.contains("available: false"));
    }

    #[test]
    fn frames_encoded_starts_at_zero() {
        let config = make_config(VideoCodec::H264);
        let encoder = VulkanVideoEncoder::new(&config).unwrap();
        assert_eq!(encoder.frames_encoded(), 0);
    }

    #[test]
    fn config_accessible() {
        let config = make_config(VideoCodec::H264);
        let encoder = VulkanVideoEncoder::new(&config).unwrap();
        assert_eq!(encoder.config().codec, VideoCodec::H264);
        assert_eq!(encoder.config().resolution, Resolution::HD);
    }
}
