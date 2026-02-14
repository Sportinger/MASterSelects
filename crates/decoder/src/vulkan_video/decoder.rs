//! Vulkan Video decoder — implements the `HwDecoder` trait.
//!
//! `VulkanVideoDecoder` is the Vulkan Video counterpart of
//! [`NvDecoder`](crate::nvdec::NvDecoder). It will eventually provide
//! hardware-accelerated decode on AMD, Intel, and NVIDIA GPUs through the
//! `VK_KHR_video_decode_queue` family of extensions.
//!
//! # Current Status
//!
//! This is a **stub**. All decode operations return
//! [`DecodeError::UnsupportedCodec`] and [`is_available()`] returns `false`.
//! The struct exists so that:
//!
//! 1. The `VulkanBackend::create_decoder()` method can return a concrete
//!    decoder type instead of an opaque error.
//! 2. The decoder crate's module structure is ready for the real
//!    implementation in Phase 1.
//! 3. Downstream code can conditionally check `VulkanVideoDecoder::is_available()`
//!    to determine if Vulkan Video decode is supported.
//!
//! # Future Implementation Plan
//!
//! The full implementation will require:
//!
//! - A reference to the `VulkanBackend` (or at least the `ash::Device` and
//!   video-capable queue family index).
//! - A `VkVideoSessionKHR` configured for the target codec profile.
//! - DPB (Decoded Picture Buffer) management with `VkImage` allocations.
//! - An H.264/H.265 NAL parser to extract structured slice parameters
//!   (Vulkan Video does not include a built-in bitstream parser like CUVID).
//! - Pipeline barriers for video decode → compute queue transitions.

use ms_common::color::PixelFormat;
use ms_common::config::DecoderConfig;
use ms_common::packet::{GpuFrame, VideoPacket};
use ms_common::types::Resolution;
use ms_common::{DecodeError, HwDecoder, VideoCodec};

use tracing::warn;

// ---------------------------------------------------------------------------
// Supported codecs
// ---------------------------------------------------------------------------

/// Vulkan Video decode extensions and the codecs they support.
///
/// This list reflects the Vulkan extensions that *exist* in the specification.
/// Actual runtime availability depends on the GPU driver. Use
/// [`VulkanVideoDecoder::is_available()`] to check at runtime.
const VULKAN_VIDEO_DECODE_CODECS: &[(VideoCodec, &str)] = &[
    (VideoCodec::H264, "VK_KHR_video_decode_h264"),
    (VideoCodec::H265, "VK_KHR_video_decode_h265"),
    (VideoCodec::Av1, "VK_KHR_video_decode_av1"),
    // VP9 decode is not yet part of the Vulkan Video specification.
];

/// Check whether a codec has a corresponding Vulkan Video decode extension.
fn codec_has_vulkan_extension(codec: VideoCodec) -> bool {
    VULKAN_VIDEO_DECODE_CODECS
        .iter()
        .any(|(c, _)| *c == codec)
}

/// Get the Vulkan extension name for a codec (if it exists).
fn vulkan_extension_for_codec(codec: VideoCodec) -> Option<&'static str> {
    VULKAN_VIDEO_DECODE_CODECS
        .iter()
        .find(|(c, _)| *c == codec)
        .map(|(_, ext)| *ext)
}

// ---------------------------------------------------------------------------
// VulkanVideoDecoder
// ---------------------------------------------------------------------------

/// Vulkan Video hardware decoder.
///
/// Implements the [`HwDecoder`] trait for Vulkan Video decode. Currently a
/// stub that returns errors for all operations. See the module-level
/// documentation for the required Vulkan extensions and implementation plan.
///
/// # Example
///
/// ```ignore
/// use ms_decoder::vulkan_video::VulkanVideoDecoder;
/// use ms_common::config::DecoderConfig;
/// use ms_common::VideoCodec;
/// use ms_common::types::Resolution;
///
/// // Check availability first
/// if VulkanVideoDecoder::is_available() {
///     let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
///     let mut decoder = VulkanVideoDecoder::new(&config)?;
///     // Feed packets from demuxer...
/// } else {
///     // Fall back to NVDEC or software decode
/// }
/// ```
pub struct VulkanVideoDecoder {
    /// The video codec this decoder is configured for.
    codec: VideoCodec,
    /// The expected output resolution (from config).
    resolution: Resolution,
    /// Output pixel format. Vulkan Video typically outputs NV12 (8-bit)
    /// or P010 (10-bit), matching NVDEC behavior.
    output_format: PixelFormat,
    /// Total frames decoded (for statistics / error reporting).
    frames_decoded: u64,
}

impl std::fmt::Debug for VulkanVideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanVideoDecoder")
            .field("codec", &self.codec)
            .field("resolution", &self.resolution)
            .field("output_format", &self.output_format)
            .field("frames_decoded", &self.frames_decoded)
            .field("available", &false)
            .finish()
    }
}

impl VulkanVideoDecoder {
    /// Create a new Vulkan Video decoder for the given configuration.
    ///
    /// # Current Behavior
    ///
    /// Always succeeds (the struct is created) but all subsequent `decode()`
    /// calls will return errors because the Vulkan Video implementation is
    /// not yet complete.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::UnsupportedCodec`] if the codec does not have
    /// a corresponding Vulkan Video decode extension in the specification
    /// (currently VP9 is not supported by Vulkan Video).
    pub fn new(config: &DecoderConfig) -> Result<Self, DecodeError> {
        if !codec_has_vulkan_extension(config.codec) {
            return Err(DecodeError::UnsupportedCodec(config.codec));
        }

        let ext_name = vulkan_extension_for_codec(config.codec).unwrap_or("unknown");

        warn!(
            codec = config.codec.display_name(),
            extension = ext_name,
            "VulkanVideoDecoder created (stub — decode operations will fail)"
        );

        Ok(Self {
            codec: config.codec,
            resolution: config.resolution,
            output_format: config.output_format,
            frames_decoded: 0,
        })
    }

    /// Check if Vulkan Video decode is available on the current system.
    ///
    /// # Current Behavior
    ///
    /// Always returns `false`. The full implementation will:
    ///
    /// 1. Enumerate physical devices via `vkEnumeratePhysicalDevices`.
    /// 2. Query `VkPhysicalDeviceVideoCapabilitiesKHR` for each device.
    /// 3. Check that a video decode queue family exists.
    /// 4. Verify that at least one codec profile is supported.
    ///
    /// # Performance Note
    ///
    /// This function should be called once at startup and cached, not on
    /// every frame. Vulkan device enumeration is expensive.
    pub fn is_available() -> bool {
        // TODO(phase-1): Implement actual Vulkan Video capability detection.
        // This requires:
        // 1. A VkInstance (possibly shared with the VulkanBackend)
        // 2. Querying VK_KHR_video_queue support on the physical device
        // 3. Checking for a queue family with VIDEO_DECODE_BIT_KHR
        // 4. Verifying VK_KHR_video_decode_h264/h265 device extension support
        false
    }

    /// Check if a specific codec is supported by Vulkan Video decode.
    ///
    /// # Current Behavior
    ///
    /// Always returns `false`. The full implementation will query the device's
    /// `VkVideoProfileListInfoKHR` capabilities for the given codec.
    pub fn supports_codec(_codec: VideoCodec) -> bool {
        // TODO(phase-1): Query VkPhysicalDeviceVideoFormatInfoKHR for the codec.
        false
    }

    /// Get the Vulkan extension name required for the configured codec.
    pub fn required_extension(&self) -> Option<&'static str> {
        vulkan_extension_for_codec(self.codec)
    }

    /// Get the number of frames decoded so far.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }
}

impl HwDecoder for VulkanVideoDecoder {
    /// Decode a single video packet.
    ///
    /// # Current Behavior
    ///
    /// Always returns [`DecodeError::HwDecoderInit`] because the Vulkan
    /// Video implementation is not yet complete.
    ///
    /// # Future Behavior
    ///
    /// Will record a decode operation into a video command buffer:
    /// 1. Parse the NAL units from the packet to extract slice parameters.
    /// 2. Select a DPB slot for the decoded picture.
    /// 3. Record `vkCmdDecodeVideoKHR` with the appropriate codec-specific
    ///    decode info structure.
    /// 4. Submit and wait (or use timeline semaphores for async decode).
    /// 5. Return the decoded frame as a `GpuFrame` pointing to the DPB image.
    fn decode(&mut self, _packet: &VideoPacket) -> Result<Option<GpuFrame>, DecodeError> {
        Err(DecodeError::HwDecoderInit {
            codec: self.codec,
            reason: format!(
                "Vulkan Video decode not yet implemented (requires {})",
                self.required_extension().unwrap_or("unknown extension")
            ),
        })
    }

    /// Flush remaining frames from the decode pipeline.
    ///
    /// # Current Behavior
    ///
    /// Always returns [`DecodeError::HwDecoderInit`] because the Vulkan
    /// Video implementation is not yet complete.
    fn flush(&mut self) -> Result<Vec<GpuFrame>, DecodeError> {
        Err(DecodeError::HwDecoderInit {
            codec: self.codec,
            reason: "Vulkan Video decode not yet implemented — nothing to flush".into(),
        })
    }

    /// Get the output pixel format (NV12 for 8-bit, P010 for 10-bit).
    fn output_format(&self) -> PixelFormat {
        self.output_format
    }

    /// Get the output resolution as configured.
    fn output_resolution(&self) -> Resolution {
        self.resolution
    }

    /// Get the codec this decoder is configured for.
    fn codec(&self) -> VideoCodec {
        self.codec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_not_available() {
        // Stub always reports unavailable.
        assert!(!VulkanVideoDecoder::is_available());
    }

    #[test]
    fn does_not_support_any_codec() {
        // Stub always reports unsupported.
        assert!(!VulkanVideoDecoder::supports_codec(VideoCodec::H264));
        assert!(!VulkanVideoDecoder::supports_codec(VideoCodec::H265));
        assert!(!VulkanVideoDecoder::supports_codec(VideoCodec::Av1));
        assert!(!VulkanVideoDecoder::supports_codec(VideoCodec::Vp9));
    }

    #[test]
    fn creation_succeeds_for_h264() {
        let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn creation_succeeds_for_h265() {
        let config = DecoderConfig::new(VideoCodec::H265, Resolution::UHD);
        let decoder = VulkanVideoDecoder::new(&config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn creation_succeeds_for_av1() {
        let config = DecoderConfig::new(VideoCodec::Av1, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn creation_fails_for_vp9() {
        // VP9 has no Vulkan Video decode extension.
        let config = DecoderConfig::new(VideoCodec::Vp9, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config);
        assert!(decoder.is_err());
    }

    #[test]
    fn decode_returns_error() {
        let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
        let mut decoder = VulkanVideoDecoder::new(&config).unwrap();

        let packet = VideoPacket {
            data: vec![0u8; 1024],
            pts: ms_common::types::TimeCode::ZERO,
            dts: ms_common::types::TimeCode::ZERO,
            is_keyframe: true,
            codec: VideoCodec::H264,
        };

        let result = decoder.decode(&packet);
        assert!(result.is_err());
    }

    #[test]
    fn flush_returns_error() {
        let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
        let mut decoder = VulkanVideoDecoder::new(&config).unwrap();
        let result = decoder.flush();
        assert!(result.is_err());
    }

    #[test]
    fn output_format_matches_config() {
        let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config).unwrap();
        assert_eq!(decoder.output_format(), PixelFormat::Nv12);
        assert_eq!(decoder.output_resolution(), Resolution::HD);
        assert_eq!(decoder.codec(), VideoCodec::H264);
    }

    #[test]
    fn required_extension_for_h264() {
        let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config).unwrap();
        assert_eq!(
            decoder.required_extension(),
            Some("VK_KHR_video_decode_h264")
        );
    }

    #[test]
    fn required_extension_for_h265() {
        let config = DecoderConfig::new(VideoCodec::H265, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config).unwrap();
        assert_eq!(
            decoder.required_extension(),
            Some("VK_KHR_video_decode_h265")
        );
    }

    #[test]
    fn codec_extension_mapping() {
        assert!(codec_has_vulkan_extension(VideoCodec::H264));
        assert!(codec_has_vulkan_extension(VideoCodec::H265));
        assert!(codec_has_vulkan_extension(VideoCodec::Av1));
        assert!(!codec_has_vulkan_extension(VideoCodec::Vp9));
    }

    #[test]
    fn debug_format() {
        let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config).unwrap();
        let debug_str = format!("{decoder:?}");
        assert!(debug_str.contains("VulkanVideoDecoder"));
        assert!(debug_str.contains("H264"));
        assert!(debug_str.contains("available: false"));
    }

    #[test]
    fn frames_decoded_starts_at_zero() {
        let config = DecoderConfig::new(VideoCodec::H264, Resolution::HD);
        let decoder = VulkanVideoDecoder::new(&config).unwrap();
        assert_eq!(decoder.frames_decoded(), 0);
    }
}
