//! NVENC parameter builders.
//!
//! Maps the common `EncoderConfig` types from `ms-common` to NVENC-specific
//! GUIDs, rate control modes, and initialization parameters.

use ms_common::config::{EncoderBitrate, EncoderConfig, EncoderPreset, EncoderProfile};
use ms_common::codec::VideoCodec;
use ms_common::EncodeError;

use super::ffi::{
    NvEncBufferFormat, NvEncConfig, NvEncInitializeParams, NvEncRcMode, NvEncRcParams, NvGuid,
    NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_H264_PROFILE_BASELINE_GUID,
    NV_ENC_H264_PROFILE_HIGH_GUID, NV_ENC_H264_PROFILE_MAIN_GUID,
    NV_ENC_HEVC_PROFILE_MAIN10_GUID, NV_ENC_HEVC_PROFILE_MAIN_GUID, NV_ENC_PRESET_P1_GUID,
    NV_ENC_PRESET_P2_GUID, NV_ENC_PRESET_P4_GUID, NV_ENC_PRESET_P6_GUID, NV_ENC_PRESET_P7_GUID,
    NV_ENC_TUNING_INFO_HIGH_QUALITY, NV_ENC_TUNING_INFO_LOSSLESS, NV_ENC_TUNING_INFO_LOW_LATENCY,
};

// ---------------------------------------------------------------------------
// Codec mapping
// ---------------------------------------------------------------------------

/// Map a `VideoCodec` to the corresponding NVENC codec GUID.
///
/// Only H.264 and H.265 are supported for encoding. VP9 and AV1 encoding
/// are not available through NVENC (VP9 is decode-only, AV1 encode requires
/// newer SDK versions and is not yet implemented).
pub fn codec_to_guid(codec: VideoCodec) -> Result<NvGuid, EncodeError> {
    match codec {
        VideoCodec::H264 => Ok(NV_ENC_CODEC_H264_GUID),
        VideoCodec::H265 => Ok(NV_ENC_CODEC_HEVC_GUID),
        _ => Err(EncodeError::UnsupportedCodec(codec)),
    }
}

// ---------------------------------------------------------------------------
// Preset mapping
// ---------------------------------------------------------------------------

/// Map an `EncoderPreset` to the corresponding NVENC preset GUID.
///
/// NVENC SDK 12.x uses presets P1-P7 with tuning info:
/// - P1 = fastest (lowest quality)
/// - P4 = medium (balanced)
/// - P7 = slowest (highest quality)
pub fn preset_to_guid(preset: EncoderPreset) -> NvGuid {
    match preset {
        EncoderPreset::Fastest => NV_ENC_PRESET_P1_GUID,
        EncoderPreset::Fast => NV_ENC_PRESET_P2_GUID,
        EncoderPreset::Medium => NV_ENC_PRESET_P4_GUID,
        EncoderPreset::Slow => NV_ENC_PRESET_P6_GUID,
        EncoderPreset::Lossless => NV_ENC_PRESET_P7_GUID,
    }
}

/// Map an `EncoderPreset` to the corresponding NVENC tuning info.
pub fn preset_to_tuning(preset: EncoderPreset) -> u32 {
    match preset {
        EncoderPreset::Fastest => NV_ENC_TUNING_INFO_LOW_LATENCY,
        EncoderPreset::Fast => NV_ENC_TUNING_INFO_LOW_LATENCY,
        EncoderPreset::Medium => NV_ENC_TUNING_INFO_HIGH_QUALITY,
        EncoderPreset::Slow => NV_ENC_TUNING_INFO_HIGH_QUALITY,
        EncoderPreset::Lossless => NV_ENC_TUNING_INFO_LOSSLESS,
    }
}

// ---------------------------------------------------------------------------
// Profile mapping
// ---------------------------------------------------------------------------

/// Map an `EncoderProfile` to the corresponding NVENC profile GUID.
///
/// Returns the GUID appropriate for the given codec and profile combination.
pub fn profile_to_guid(profile: EncoderProfile, codec: VideoCodec) -> NvGuid {
    match codec {
        VideoCodec::H264 => match profile {
            EncoderProfile::Baseline => NV_ENC_H264_PROFILE_BASELINE_GUID,
            EncoderProfile::Main => NV_ENC_H264_PROFILE_MAIN_GUID,
            EncoderProfile::High => NV_ENC_H264_PROFILE_HIGH_GUID,
            // H.264 doesn't have High10 in NVENC, use High instead
            EncoderProfile::High10 => NV_ENC_H264_PROFILE_HIGH_GUID,
        },
        VideoCodec::H265 => match profile {
            // HEVC doesn't have Baseline, use Main
            EncoderProfile::Baseline => NV_ENC_HEVC_PROFILE_MAIN_GUID,
            EncoderProfile::Main => NV_ENC_HEVC_PROFILE_MAIN_GUID,
            // HEVC High maps to Main
            EncoderProfile::High => NV_ENC_HEVC_PROFILE_MAIN_GUID,
            EncoderProfile::High10 => NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        },
        // VP9/AV1 don't have NVENC encode profiles
        _ => NV_ENC_H264_PROFILE_HIGH_GUID,
    }
}

// ---------------------------------------------------------------------------
// Bitrate / rate control mapping
// ---------------------------------------------------------------------------

/// Map an `EncoderBitrate` to NVENC rate control parameters.
pub fn bitrate_to_rc_params(bitrate: &EncoderBitrate) -> NvEncRcParams {
    let mut rc = NvEncRcParams::default();

    match bitrate {
        EncoderBitrate::Cbr(bps) => {
            rc.rate_control_mode = NvEncRcMode::Cbr;
            // NVENC uses u32 for bitrate fields; clamp large values
            rc.average_bitrate = clamp_bitrate(*bps);
            rc.max_bitrate = clamp_bitrate(*bps);
            // VBV buffer = 1 second of data
            rc.vbv_buffer_size = clamp_bitrate(*bps);
            rc.vbv_initial_delay = clamp_bitrate(*bps) / 2;
        }
        EncoderBitrate::Vbr { target, max } => {
            rc.rate_control_mode = NvEncRcMode::Vbr;
            rc.average_bitrate = clamp_bitrate(*target);
            rc.max_bitrate = clamp_bitrate(*max);
            // VBV buffer = based on max bitrate
            rc.vbv_buffer_size = clamp_bitrate(*max);
            rc.vbv_initial_delay = clamp_bitrate(*max) / 2;
        }
        EncoderBitrate::Cqp(qp) => {
            rc.rate_control_mode = NvEncRcMode::ConstQp;
            let qp_val = (*qp).min(51);
            rc.const_qp_i = qp_val;
            rc.const_qp_p = qp_val;
            rc.const_qp_b = qp_val;
        }
    }

    rc
}

/// Clamp a u64 bitrate to u32 range (NVENC API limitation).
fn clamp_bitrate(bps: u64) -> u32 {
    bps.min(u32::MAX as u64) as u32
}

// ---------------------------------------------------------------------------
// Buffer format mapping
// ---------------------------------------------------------------------------

/// Map a `PixelFormat` from ms-common to an NVENC buffer format.
pub fn pixel_format_to_nvenc(format: ms_common::PixelFormat) -> NvEncBufferFormat {
    match format {
        ms_common::PixelFormat::Nv12 => NvEncBufferFormat::Nv12,
        ms_common::PixelFormat::P010 => NvEncBufferFormat::Yuv420_10bit,
        ms_common::PixelFormat::Rgba8 => NvEncBufferFormat::Abgr,
        ms_common::PixelFormat::Bgra8 => NvEncBufferFormat::Argb,
        // For float formats, we would need conversion first; default to NV12
        ms_common::PixelFormat::Rgba16F | ms_common::PixelFormat::Rgba32F => {
            NvEncBufferFormat::Nv12
        }
    }
}

// ---------------------------------------------------------------------------
// Full parameter builder
// ---------------------------------------------------------------------------

/// Build NVENC initialization parameters from an `EncoderConfig`.
///
/// Returns an `NvEncInitializeParams` struct ready to be passed to
/// `nvEncInitializeEncoder`, along with the `NvEncConfig` it references.
///
/// # Important
///
/// The returned `NvEncConfig` must outlive the `NvEncInitializeParams`
/// because the params struct holds a raw pointer to the config.
pub fn build_init_params(
    config: &EncoderConfig,
) -> Result<(NvEncInitializeParams, NvEncConfig), EncodeError> {
    let codec_guid = codec_to_guid(config.codec)?;
    let preset_guid = preset_to_guid(config.preset);
    let profile_guid = profile_to_guid(config.profile, config.codec);
    let tuning_info = preset_to_tuning(config.preset);
    let rc_params = bitrate_to_rc_params(&config.bitrate);

    // Build the encoder config struct
    let enc_config = NvEncConfig {
        profile_guid,
        gop_length: config.fps.num, // 1 keyframe per second of video
        frame_interval_p: 1,
        rc_params,
        ..NvEncConfig::default()
    };

    // Build the initialization params struct
    let init_params = NvEncInitializeParams {
        encode_guid: codec_guid,
        preset_guid,
        encode_width: config.resolution.width,
        encode_height: config.resolution.height,
        dar_width: config.resolution.width,
        dar_height: config.resolution.height,
        frame_rate_num: config.fps.num,
        frame_rate_den: config.fps.den,
        tuning_info,
        ..NvEncInitializeParams::default()
    };

    // NOTE: init_params.encode_config will be set to a pointer to enc_config
    // by the caller, after both structs are placed at stable addresses.

    Ok((init_params, enc_config))
}

/// Validate an `EncoderConfig` before attempting to create an encoder.
///
/// Returns `Ok(())` if the config is valid, or an appropriate error.
pub fn validate_config(config: &EncoderConfig) -> Result<(), EncodeError> {
    // Check codec support
    codec_to_guid(config.codec)?;

    // Check resolution
    if config.resolution.width == 0 || config.resolution.height == 0 {
        return Err(EncodeError::HwEncoderInit(
            "Resolution width and height must be > 0".to_string(),
        ));
    }

    // NVENC requires width/height to be multiples of 2
    if !config.resolution.width.is_multiple_of(2) || !config.resolution.height.is_multiple_of(2) {
        return Err(EncodeError::HwEncoderInit(
            "Resolution width and height must be even numbers".to_string(),
        ));
    }

    // Maximum supported resolution (8K)
    if config.resolution.width > 8192 || config.resolution.height > 8192 {
        return Err(EncodeError::HwEncoderInit(format!(
            "Resolution {}x{} exceeds maximum 8192x8192",
            config.resolution.width, config.resolution.height
        )));
    }

    // Check frame rate
    if config.fps.den == 0 {
        return Err(EncodeError::HwEncoderInit(
            "Frame rate denominator must be > 0".to_string(),
        ));
    }

    // Check bitrate
    match &config.bitrate {
        EncoderBitrate::Cbr(bps) => {
            if *bps == 0 {
                return Err(EncodeError::HwEncoderInit(
                    "CBR bitrate must be > 0".to_string(),
                ));
            }
        }
        EncoderBitrate::Vbr { target, max } => {
            if *target == 0 || *max == 0 {
                return Err(EncodeError::HwEncoderInit(
                    "VBR target and max bitrate must be > 0".to_string(),
                ));
            }
            if *target > *max {
                return Err(EncodeError::HwEncoderInit(format!(
                    "VBR target bitrate ({target}) exceeds max bitrate ({max})"
                )));
            }
        }
        EncoderBitrate::Cqp(qp) => {
            if *qp > 51 {
                return Err(EncodeError::HwEncoderInit(format!(
                    "CQP value {qp} exceeds maximum of 51"
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_common::types::{Rational, Resolution};

    fn make_config() -> EncoderConfig {
        EncoderConfig {
            codec: VideoCodec::H264,
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

    #[test]
    fn codec_guid_h264() {
        let guid = codec_to_guid(VideoCodec::H264).unwrap();
        assert_eq!(guid, NV_ENC_CODEC_H264_GUID);
    }

    #[test]
    fn codec_guid_hevc() {
        let guid = codec_to_guid(VideoCodec::H265).unwrap();
        assert_eq!(guid, NV_ENC_CODEC_HEVC_GUID);
    }

    #[test]
    fn codec_guid_unsupported() {
        assert!(codec_to_guid(VideoCodec::Vp9).is_err());
        assert!(codec_to_guid(VideoCodec::Av1).is_err());
    }

    #[test]
    fn preset_mapping_fastest() {
        let guid = preset_to_guid(EncoderPreset::Fastest);
        assert_eq!(guid, NV_ENC_PRESET_P1_GUID);
    }

    #[test]
    fn preset_mapping_medium() {
        let guid = preset_to_guid(EncoderPreset::Medium);
        assert_eq!(guid, NV_ENC_PRESET_P4_GUID);
    }

    #[test]
    fn preset_mapping_slow() {
        let guid = preset_to_guid(EncoderPreset::Slow);
        assert_eq!(guid, NV_ENC_PRESET_P6_GUID);
    }

    #[test]
    fn preset_mapping_lossless() {
        let guid = preset_to_guid(EncoderPreset::Lossless);
        assert_eq!(guid, NV_ENC_PRESET_P7_GUID);
    }

    #[test]
    fn tuning_info_quality() {
        assert_eq!(
            preset_to_tuning(EncoderPreset::Medium),
            NV_ENC_TUNING_INFO_HIGH_QUALITY
        );
        assert_eq!(
            preset_to_tuning(EncoderPreset::Slow),
            NV_ENC_TUNING_INFO_HIGH_QUALITY
        );
    }

    #[test]
    fn tuning_info_low_latency() {
        assert_eq!(
            preset_to_tuning(EncoderPreset::Fastest),
            NV_ENC_TUNING_INFO_LOW_LATENCY
        );
    }

    #[test]
    fn tuning_info_lossless() {
        assert_eq!(
            preset_to_tuning(EncoderPreset::Lossless),
            NV_ENC_TUNING_INFO_LOSSLESS
        );
    }

    #[test]
    fn profile_h264_high() {
        let guid = profile_to_guid(EncoderProfile::High, VideoCodec::H264);
        assert_eq!(guid, NV_ENC_H264_PROFILE_HIGH_GUID);
    }

    #[test]
    fn profile_h264_main() {
        let guid = profile_to_guid(EncoderProfile::Main, VideoCodec::H264);
        assert_eq!(guid, NV_ENC_H264_PROFILE_MAIN_GUID);
    }

    #[test]
    fn profile_h264_baseline() {
        let guid = profile_to_guid(EncoderProfile::Baseline, VideoCodec::H264);
        assert_eq!(guid, NV_ENC_H264_PROFILE_BASELINE_GUID);
    }

    #[test]
    fn profile_hevc_main() {
        let guid = profile_to_guid(EncoderProfile::Main, VideoCodec::H265);
        assert_eq!(guid, NV_ENC_HEVC_PROFILE_MAIN_GUID);
    }

    #[test]
    fn profile_hevc_main10() {
        let guid = profile_to_guid(EncoderProfile::High10, VideoCodec::H265);
        assert_eq!(guid, NV_ENC_HEVC_PROFILE_MAIN10_GUID);
    }

    #[test]
    fn bitrate_cbr() {
        let rc = bitrate_to_rc_params(&EncoderBitrate::Cbr(20_000_000));
        assert_eq!(rc.rate_control_mode, NvEncRcMode::Cbr);
        assert_eq!(rc.average_bitrate, 20_000_000);
        assert_eq!(rc.max_bitrate, 20_000_000);
    }

    #[test]
    fn bitrate_vbr() {
        let rc = bitrate_to_rc_params(&EncoderBitrate::Vbr {
            target: 15_000_000,
            max: 25_000_000,
        });
        assert_eq!(rc.rate_control_mode, NvEncRcMode::Vbr);
        assert_eq!(rc.average_bitrate, 15_000_000);
        assert_eq!(rc.max_bitrate, 25_000_000);
    }

    #[test]
    fn bitrate_cqp() {
        let rc = bitrate_to_rc_params(&EncoderBitrate::Cqp(23));
        assert_eq!(rc.rate_control_mode, NvEncRcMode::ConstQp);
        assert_eq!(rc.const_qp_i, 23);
        assert_eq!(rc.const_qp_p, 23);
        assert_eq!(rc.const_qp_b, 23);
    }

    #[test]
    fn bitrate_cqp_clamp() {
        // QP > 51 should be clamped to 51
        let rc = bitrate_to_rc_params(&EncoderBitrate::Cqp(100));
        assert_eq!(rc.const_qp_i, 51);
    }

    #[test]
    fn bitrate_clamp_large_value() {
        let large_bitrate = u64::MAX;
        let clamped = clamp_bitrate(large_bitrate);
        assert_eq!(clamped, u32::MAX);
    }

    #[test]
    fn pixel_format_nv12() {
        assert_eq!(
            pixel_format_to_nvenc(ms_common::PixelFormat::Nv12),
            NvEncBufferFormat::Nv12
        );
    }

    #[test]
    fn pixel_format_rgba8() {
        assert_eq!(
            pixel_format_to_nvenc(ms_common::PixelFormat::Rgba8),
            NvEncBufferFormat::Abgr
        );
    }

    #[test]
    fn build_params_h264() {
        let config = make_config();
        let (init, enc) = build_init_params(&config).unwrap();

        assert_eq!(init.encode_guid, NV_ENC_CODEC_H264_GUID);
        assert_eq!(init.preset_guid, NV_ENC_PRESET_P4_GUID);
        assert_eq!(init.encode_width, 1920);
        assert_eq!(init.encode_height, 1080);
        assert_eq!(init.frame_rate_num, 30);
        assert_eq!(init.frame_rate_den, 1);
        assert_eq!(init.enable_ptd, 1);
        assert_eq!(enc.profile_guid, NV_ENC_H264_PROFILE_HIGH_GUID);
        assert_eq!(enc.rc_params.rate_control_mode, NvEncRcMode::Vbr);
    }

    #[test]
    fn build_params_hevc() {
        let config = EncoderConfig {
            codec: VideoCodec::H265,
            resolution: Resolution::UHD,
            fps: Rational::FPS_60,
            bitrate: EncoderBitrate::Cbr(50_000_000),
            preset: EncoderPreset::Slow,
            profile: EncoderProfile::Main,
        };

        let (init, enc) = build_init_params(&config).unwrap();
        assert_eq!(init.encode_guid, NV_ENC_CODEC_HEVC_GUID);
        assert_eq!(init.encode_width, 3840);
        assert_eq!(init.encode_height, 2160);
        assert_eq!(enc.profile_guid, NV_ENC_HEVC_PROFILE_MAIN_GUID);
        assert_eq!(enc.rc_params.rate_control_mode, NvEncRcMode::Cbr);
    }

    #[test]
    fn validate_valid_config() {
        let config = make_config();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn validate_zero_resolution() {
        let mut config = make_config();
        config.resolution = Resolution::new(0, 1080);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_odd_resolution() {
        let mut config = make_config();
        config.resolution = Resolution::new(1921, 1080);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_too_large_resolution() {
        let mut config = make_config();
        config.resolution = Resolution::new(16384, 8192);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_zero_cbr() {
        let mut config = make_config();
        config.bitrate = EncoderBitrate::Cbr(0);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_vbr_target_exceeds_max() {
        let mut config = make_config();
        config.bitrate = EncoderBitrate::Vbr {
            target: 30_000_000,
            max: 20_000_000,
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_cqp_too_high() {
        let mut config = make_config();
        config.bitrate = EncoderBitrate::Cqp(52);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_unsupported_codec() {
        let mut config = make_config();
        config.codec = VideoCodec::Vp9;
        assert!(validate_config(&config).is_err());
    }
}
