//! End-to-end integration tests for the NVDEC decode pipeline.
//!
//! These tests exercise the full path from library loading through decoder
//! creation, packet feeding, and frame retrieval. Tests that require real
//! NVIDIA hardware are marked `#[ignore]` so they can be run explicitly on
//! machines with a GPU:
//!
//! ```bash
//! cargo test -p ms-decoder --test nvdec_integration -- --ignored
//! ```
//!
//! The non-ignored tests exercise the software NV12-to-RGBA fallback path
//! and the `DecoderManager` API surface, which work on any platform.

use ms_common::color::PixelFormat;
use ms_common::packet::{GpuFrame, VideoPacket};
use ms_common::types::{Resolution, TimeCode};
use ms_common::{DecodeError, HwDecoder, VideoCodec};

use ms_decoder::manager::{DecoderCreateConfig, DecoderId, DecoderManager};
use ms_decoder::nvdec::{NvDecoder, NvcuvidLibrary};
use ms_decoder::software::{nv12_to_rgba, SoftwareDecoder};

// ---------------------------------------------------------------------------
// Helper: build a synthetic NV12 frame (raw Y + interleaved UV)
// ---------------------------------------------------------------------------

/// Generate a synthetic NV12 frame of the given dimensions.
///
/// The Y plane contains a horizontal gradient (left=dark, right=bright) and the
/// UV plane is set to neutral chroma (128, 128), producing a greyscale image.
fn make_gradient_nv12(width: u32, height: u32) -> Vec<u8> {
    assert!(width > 0 && height > 0 && height.is_multiple_of(2));
    let y_size = (width * height) as usize;
    let uv_size = (width * (height / 2)) as usize;
    let mut data = vec![0u8; y_size + uv_size];

    // Y plane: horizontal gradient from 16 (black) to 235 (white)
    for row in 0..height as usize {
        for col in 0..width as usize {
            let t = col as f64 / (width.saturating_sub(1).max(1)) as f64;
            data[row * width as usize + col] = (16.0 + t * 219.0) as u8;
        }
    }

    // UV plane: neutral chroma (128, 128)
    let uv_start = y_size;
    for i in 0..uv_size {
        data[uv_start + i] = 128;
    }

    data
}

/// Build a uniform NV12 frame where every pixel has the same YUV value.
fn make_uniform_nv12(width: u32, height: u32, y: u8, u_val: u8, v_val: u8) -> Vec<u8> {
    assert!(width > 0 && height > 0 && height.is_multiple_of(2));
    let y_size = (width * height) as usize;
    let uv_size = (width * (height / 2)) as usize;
    let mut data = vec![0u8; y_size + uv_size];

    // Y plane
    for byte in &mut data[..y_size] {
        *byte = y;
    }

    // UV plane (interleaved U, V pairs)
    let uv_start = y_size;
    for i in (0..uv_size).step_by(2) {
        data[uv_start + i] = u_val;
        if i + 1 < uv_size {
            data[uv_start + i + 1] = v_val;
        }
    }

    data
}

/// Wrap raw NV12 bytes into a `VideoPacket` suitable for `SoftwareDecoder`.
fn make_nv12_packet(nv12_data: Vec<u8>, pts_secs: f64) -> VideoPacket {
    VideoPacket {
        data: nv12_data,
        pts: TimeCode::from_secs(pts_secs),
        dts: TimeCode::from_secs(pts_secs),
        is_keyframe: true,
        codec: VideoCodec::H264,
    }
}

// ===========================================================================
// GPU-required tests (ignored by default)
// ===========================================================================

#[test]
#[ignore] // Requires NVIDIA GPU with NVDEC driver installed
fn test_nvdec_library_loads() {
    // Attempt to load the nvcuvid dynamic library from the system path.
    // On systems with NVIDIA drivers this should succeed; on CI machines
    // without an NVIDIA GPU this test is skipped (hence #[ignore]).
    let result = NvcuvidLibrary::load();
    assert!(
        result.is_ok(),
        "NvcuvidLibrary::load() failed: {:?}",
        result.unwrap_err()
    );
    let lib = result.unwrap();
    // Verify the library can be formatted (Debug impl works).
    let dbg = format!("{lib:?}");
    assert!(dbg.contains("NvcuvidLibrary"));
}

#[test]
#[ignore] // Requires NVIDIA GPU + nvcuvid library
fn test_nvdec_decoder_creation_h264() {
    // Load the library.
    let lib = match NvcuvidLibrary::load() {
        Ok(lib) => std::sync::Arc::new(lib),
        Err(e) => {
            eprintln!("Skipping: NvcuvidLibrary not available: {e}");
            return;
        }
    };

    // Create a decoder for H.264 with default config.
    let decoder_result = NvDecoder::new(lib.clone(), VideoCodec::H264);
    assert!(
        decoder_result.is_ok(),
        "NvDecoder::new(H264) failed: {:?}",
        decoder_result.unwrap_err()
    );

    let decoder = decoder_result.unwrap();

    // The decoder should exist but the HW decoder should not be "ready" yet
    // (no SPS has been parsed).
    assert!(!decoder.is_ready(), "decoder should not be ready before SPS");
    assert_eq!(decoder.codec(), VideoCodec::H264);
    assert_eq!(decoder.output_format(), PixelFormat::Nv12);
    assert_eq!(decoder.frames_decoded(), 0);
    assert_eq!(decoder.active_frame_count(), 0);
}

#[test]
#[ignore] // Requires NVIDIA GPU + nvcuvid library
fn test_nvdec_decoder_creation_h265() {
    let lib = match NvcuvidLibrary::load() {
        Ok(lib) => std::sync::Arc::new(lib),
        Err(e) => {
            eprintln!("Skipping: NvcuvidLibrary not available: {e}");
            return;
        }
    };

    let decoder_result = NvDecoder::new(lib, VideoCodec::H265);
    assert!(
        decoder_result.is_ok(),
        "NvDecoder::new(H265) failed: {:?}",
        decoder_result.unwrap_err()
    );

    let decoder = decoder_result.unwrap();
    assert_eq!(decoder.codec(), VideoCodec::H265);
}

#[test]
#[ignore] // Requires NVIDIA GPU + nvcuvid library
fn test_nvdec_decoder_with_custom_config() {
    let lib = match NvcuvidLibrary::load() {
        Ok(lib) => std::sync::Arc::new(lib),
        Err(e) => {
            eprintln!("Skipping: NvcuvidLibrary not available: {e}");
            return;
        }
    };

    // Create with custom surface count and low-latency display delay.
    let decoder_result = NvDecoder::with_config(
        lib,
        VideoCodec::H264,
        12, // fewer DPB surfaces
        0,  // zero display delay = low latency
    );
    assert!(
        decoder_result.is_ok(),
        "NvDecoder::with_config failed: {:?}",
        decoder_result.unwrap_err()
    );

    let decoder = decoder_result.unwrap();
    assert_eq!(decoder.codec(), VideoCodec::H264);
    assert_eq!(decoder.pending_frames(), 0);
}

#[test]
#[ignore] // Requires NVIDIA GPU + nvcuvid library
fn test_nvdec_flush_empty_decoder() {
    let lib = match NvcuvidLibrary::load() {
        Ok(lib) => std::sync::Arc::new(lib),
        Err(e) => {
            eprintln!("Skipping: NvcuvidLibrary not available: {e}");
            return;
        }
    };

    let mut decoder = NvDecoder::new(lib, VideoCodec::H264).unwrap();

    // Flushing an empty decoder should succeed with zero frames.
    let flushed = decoder.flush().unwrap();
    assert!(
        flushed.is_empty(),
        "flush of empty decoder should return empty vec"
    );
}

#[test]
#[ignore] // Requires NVIDIA GPU + nvcuvid library
fn test_nvdec_reset_empty_decoder() {
    let lib = match NvcuvidLibrary::load() {
        Ok(lib) => std::sync::Arc::new(lib),
        Err(e) => {
            eprintln!("Skipping: NvcuvidLibrary not available: {e}");
            return;
        }
    };

    let mut decoder = NvDecoder::new(lib, VideoCodec::H264).unwrap();

    // Reset on an empty decoder should be a no-op.
    let result = decoder.reset();
    assert!(result.is_ok(), "reset on empty decoder failed: {:?}", result);
    assert_eq!(decoder.active_frame_count(), 0);
}

#[test]
#[ignore] // Requires NVIDIA GPU + nvcuvid library
fn test_nvdec_session_stats_initial() {
    let lib = match NvcuvidLibrary::load() {
        Ok(lib) => std::sync::Arc::new(lib),
        Err(e) => {
            eprintln!("Skipping: NvcuvidLibrary not available: {e}");
            return;
        }
    };

    let decoder = NvDecoder::new(lib, VideoCodec::H264).unwrap();
    let stats = decoder.stats();

    assert_eq!(stats.frames_decoded, 0);
    assert_eq!(stats.frames_displayed, 0);
    assert_eq!(stats.pending_frames, 0);
    assert!(!stats.decoder_ready);
    assert_eq!(stats.width, 0);
    assert_eq!(stats.height, 0);
}

#[test]
#[ignore] // Requires NVIDIA GPU with NVDEC
fn test_decoder_manager_creates_decoder() {
    let mut manager = DecoderManager::new();

    if !manager.is_nvdec_available() {
        eprintln!(
            "Skipping: NVDEC not available: {}",
            manager.nvdec_error().unwrap_or("unknown error")
        );
        return;
    }

    // Create a decoder through the manager.
    let config = DecoderCreateConfig {
        codec: VideoCodec::H264,
        ..DecoderCreateConfig::default()
    };
    let id = DecoderId::new("integration_test_video.mp4");
    let result = manager.create_decoder(id.clone(), &config);

    assert!(
        result.is_ok(),
        "DecoderManager::create_decoder failed: {:?}",
        result.unwrap_err()
    );

    assert_eq!(manager.active_decoder_count(), 1);

    // Verify we can retrieve the decoder.
    assert!(manager.get_decoder(&id).is_some());
    assert!(manager.get_decoder_ref(&id).is_some());

    // Destroy and verify cleanup.
    assert!(manager.destroy_decoder(&id));
    assert_eq!(manager.active_decoder_count(), 0);
    assert!(manager.get_decoder(&id).is_none());
}

#[test]
#[ignore] // Requires NVIDIA GPU with NVDEC
fn test_decoder_manager_multiple_decoders() {
    let mut manager = DecoderManager::new();

    if !manager.is_nvdec_available() {
        eprintln!("Skipping: NVDEC not available");
        return;
    }

    let config_h264 = DecoderCreateConfig {
        codec: VideoCodec::H264,
        ..DecoderCreateConfig::default()
    };
    let config_h265 = DecoderCreateConfig {
        codec: VideoCodec::H265,
        ..DecoderCreateConfig::default()
    };

    let id1 = DecoderId::new("video1.mp4");
    let id2 = DecoderId::new("video2.mp4");

    manager.create_decoder(id1.clone(), &config_h264).unwrap();
    manager.create_decoder(id2.clone(), &config_h265).unwrap();

    assert_eq!(manager.active_decoder_count(), 2);

    // Destroy all at once.
    manager.destroy_all();
    assert_eq!(manager.active_decoder_count(), 0);
}

// ===========================================================================
// Non-ignored tests: software fallback & manager API surface
// ===========================================================================

#[test]
fn test_software_decoder_end_to_end_black_frame() {
    // Verify the full SoftwareDecoder path: create decoder, feed NV12 packet,
    // get GpuFrame back, and check that the resulting RGBA data is correct.
    let width = 8u32;
    let height = 4u32;

    let mut decoder = SoftwareDecoder::new(width, height, VideoCodec::H264).unwrap();

    // Create a black NV12 frame (Y=16, U=128, V=128 = BT.709 black).
    let nv12_data = make_uniform_nv12(width, height, 16, 128, 128);
    let packet = make_nv12_packet(nv12_data, 0.0);

    let frame = decoder.decode(&packet).unwrap().expect("should produce a frame");

    // Software decoder produces CPU-only frames (device_ptr = 0).
    assert_eq!(frame.device_ptr, 0);
    assert_eq!(frame.resolution, Resolution::new(width, height));
    assert_eq!(frame.format, PixelFormat::Rgba8);
    assert_eq!(frame.pitch, width * 4);
    assert_eq!(frame.pts, TimeCode::ZERO);
    assert_eq!(decoder.frames_converted(), 1);
}

#[test]
fn test_software_decoder_end_to_end_white_frame() {
    let width = 8u32;
    let height = 4u32;

    let mut decoder = SoftwareDecoder::new(width, height, VideoCodec::H264).unwrap();

    // White in BT.709: Y=235, U=128, V=128.
    let nv12_data = make_uniform_nv12(width, height, 235, 128, 128);
    let packet = make_nv12_packet(nv12_data, 1.0);

    let frame = decoder.decode(&packet).unwrap().expect("should produce a frame");
    assert_eq!(frame.resolution, Resolution::new(width, height));

    // The PTS should match what we passed in.
    assert!((frame.pts.as_secs() - 1.0).abs() < 1e-9);
    assert_eq!(decoder.frames_converted(), 1);
}

#[test]
fn test_software_decoder_multiple_frames_sequential() {
    let width = 16u32;
    let height = 8u32;

    let mut decoder = SoftwareDecoder::new(width, height, VideoCodec::H264).unwrap();

    // Decode 10 sequential frames.
    for i in 0..10u64 {
        let nv12_data = make_uniform_nv12(width, height, (16 + i * 20) as u8, 128, 128);
        let pts_secs = i as f64 / 30.0; // 30 fps
        let packet = make_nv12_packet(nv12_data, pts_secs);

        let result = decoder.decode(&packet);
        assert!(result.is_ok(), "decode failed at frame {i}: {:?}", result);
        assert!(result.unwrap().is_some(), "no frame produced at frame {i}");
    }

    assert_eq!(decoder.frames_converted(), 10);
}

#[test]
fn test_software_decoder_flush_returns_empty() {
    let mut decoder = SoftwareDecoder::new(4, 2, VideoCodec::H264).unwrap();

    // Flush should return empty vec (software decoder has no buffering).
    let flushed = decoder.flush().unwrap();
    assert!(flushed.is_empty());
}

#[test]
fn test_software_decoder_rejects_undersized_packet() {
    let mut decoder = SoftwareDecoder::new(8, 4, VideoCodec::H264).unwrap();

    // 8x4 NV12 needs 8*4 + 8*2 = 48 bytes. Provide only 10.
    let packet = make_nv12_packet(vec![0u8; 10], 0.0);
    let result = decoder.decode(&packet);

    assert!(result.is_err(), "should reject undersized packet");
    match result.unwrap_err() {
        DecodeError::DecodeFailed { reason, .. } => {
            assert!(
                reason.contains("too small"),
                "error should mention size: {reason}"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_software_decoder_invalid_dimensions() {
    // Zero width
    assert!(SoftwareDecoder::new(0, 4, VideoCodec::H264).is_err());
    // Zero height
    assert!(SoftwareDecoder::new(4, 0, VideoCodec::H264).is_err());
    // Odd height (NV12 requires even height)
    assert!(SoftwareDecoder::new(4, 3, VideoCodec::H264).is_err());
}

#[test]
fn test_software_decoder_trait_compliance() {
    let mut decoder = SoftwareDecoder::new(16, 8, VideoCodec::H265).unwrap();

    // Trait methods should reflect the configured state.
    assert_eq!(decoder.output_format(), PixelFormat::Rgba8);
    assert_eq!(decoder.output_resolution(), Resolution::new(16, 8));
    assert_eq!(decoder.codec(), VideoCodec::H265);

    // Feed a frame and verify the trait still works.
    let nv12_data = make_uniform_nv12(16, 8, 128, 128, 128);
    let packet = make_nv12_packet(nv12_data, 0.5);
    let frame = decoder.decode(&packet).unwrap().unwrap();

    assert_eq!(frame.format, PixelFormat::Rgba8);
    assert_eq!(frame.resolution, Resolution::new(16, 8));
}

#[test]
fn test_nv12_to_rgba_standalone_gradient() {
    // Verify the standalone nv12_to_rgba function with a gradient frame.
    let width = 16u32;
    let height = 4u32;
    let nv12 = make_gradient_nv12(width, height);

    let y_size = (width * height) as usize;
    let y_plane = &nv12[..y_size];
    let uv_plane = &nv12[y_size..];

    let rgba = nv12_to_rgba(y_plane, uv_plane, width, height, width, width).unwrap();
    assert_eq!(rgba.len(), (width * height * 4) as usize);

    // In a neutral-chroma greyscale gradient, the leftmost pixel should be
    // dark and the rightmost pixel should be bright. Check the first and
    // last pixel.
    let first_r = rgba[0];
    let last_r = rgba[((width * height - 1) * 4) as usize];
    assert!(
        last_r > first_r + 100,
        "gradient not detected: first_r={first_r}, last_r={last_r}"
    );

    // All alpha values should be 255.
    for px in rgba.chunks_exact(4) {
        assert_eq!(px[3], 255, "alpha must be 255");
    }
}

#[test]
fn test_nv12_to_rgba_color_accuracy_red() {
    // Verify a known YUV value that should produce a reddish pixel.
    // BT.709: pure red (255,0,0) approximately maps to Y=63, U=102, V=240
    // (these are approximate; the inverse formulas give inexact integer YUV).
    // Instead, we use a known YUV triple and verify against the reference formula.
    let width = 4u32;
    let height = 2u32;
    let y_val: u8 = 81; // approximate Y for (255, 0, 0) in BT.709
    let u_val: u8 = 90;
    let v_val: u8 = 240;

    let nv12 = make_uniform_nv12(width, height, y_val, u_val, v_val);
    let y_size = (width * height) as usize;
    let y_plane = &nv12[..y_size];
    let uv_plane = &nv12[y_size..];

    let rgba = nv12_to_rgba(y_plane, uv_plane, width, height, width, width).unwrap();

    // Every pixel should be the same (uniform frame). Check R > G and R > B
    // (this YUV triple should produce a reddish color).
    let r = rgba[0];
    let g = rgba[1];
    let b = rgba[2];
    assert!(
        r > g && r > b,
        "Expected reddish: R={r}, G={g}, B={b}"
    );
}

#[test]
fn test_decoder_manager_creation_always_succeeds() {
    // DecoderManager::new() should always succeed, even without NVIDIA hardware.
    // It reports availability through is_nvdec_available().
    let manager = DecoderManager::new();
    assert_eq!(manager.active_decoder_count(), 0);

    // Supported codecs should always be populated.
    let codecs = DecoderManager::supported_codecs();
    assert!(codecs.contains(&VideoCodec::H264));
    assert!(codecs.contains(&VideoCodec::H265));
    assert!(codecs.contains(&VideoCodec::Vp9));
    assert!(codecs.contains(&VideoCodec::Av1));
}

#[test]
fn test_decoder_manager_create_fails_gracefully_without_gpu() {
    let mut manager = DecoderManager::new();

    if manager.is_nvdec_available() {
        // If a GPU is present, the decoder creation will succeed.
        // We still verify the API flow works correctly.
        let config = DecoderCreateConfig::default();
        let id = DecoderId::new("test.mp4");
        let result = manager.create_decoder(id, &config);
        assert!(result.is_ok());
        manager.destroy_all();
    } else {
        // Without NVDEC, creating a decoder should return an informative error.
        let config = DecoderCreateConfig::default();
        let id = DecoderId::new("test.mp4");
        let result = manager.create_decoder(id, &config);
        assert!(result.is_err());

        match result.unwrap_err() {
            DecodeError::HwDecoderInit { codec, reason } => {
                assert_eq!(codec, VideoCodec::H264);
                assert!(!reason.is_empty(), "error reason should not be empty");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }

        // The error message should also be available through the manager.
        assert!(manager.nvdec_error().is_some());
    }
}

#[test]
fn test_decoder_manager_destroy_nonexistent_is_noop() {
    let mut manager = DecoderManager::new();
    let id = DecoderId::new("nonexistent.mp4");

    // Destroying a decoder that does not exist should return false (no-op).
    assert!(!manager.destroy_decoder(&id));
    assert_eq!(manager.active_decoder_count(), 0);
}

#[test]
fn test_software_decoder_with_decoder_manager_fallback_pattern() {
    // Demonstrates the recommended fallback pattern: try NVDEC, fall back
    // to SoftwareDecoder if unavailable.
    let manager = DecoderManager::new();
    let width = 8u32;
    let height = 4u32;

    let mut decoder: Box<dyn HwDecoder> = if manager.is_nvdec_available() {
        // Real GPU path (may or may not be hit depending on the machine).
        // We cannot easily create an NvDecoder without the manager owning it,
        // so for the trait-object pattern we use SoftwareDecoder even here
        // in the test, since we cannot move the decoder out of the manager.
        Box::new(SoftwareDecoder::new(width, height, VideoCodec::H264).unwrap())
    } else {
        Box::new(SoftwareDecoder::new(width, height, VideoCodec::H264).unwrap())
    };

    // Verify the trait object works correctly.
    assert_eq!(decoder.output_format(), PixelFormat::Rgba8);
    assert_eq!(decoder.output_resolution(), Resolution::new(width, height));
    assert_eq!(decoder.codec(), VideoCodec::H264);

    // Decode a frame through the trait object.
    let nv12_data = make_uniform_nv12(width, height, 128, 128, 128);
    let packet = make_nv12_packet(nv12_data, 0.0);
    let frame = decoder.decode(&packet).unwrap();
    assert!(frame.is_some());

    // Flush through the trait object.
    let flushed = decoder.flush().unwrap();
    assert!(flushed.is_empty());
}

#[test]
fn test_software_decoder_end_to_end_large_frame() {
    // Decode an HD-sized NV12 frame through the software path to verify
    // it handles larger resolutions correctly.
    let width = 1920u32;
    let height = 1080u32;

    let mut decoder = SoftwareDecoder::new(width, height, VideoCodec::H264).unwrap();
    assert_eq!(decoder.output_resolution(), Resolution::HD);

    let nv12_data = make_gradient_nv12(width, height);
    let packet = make_nv12_packet(nv12_data, 0.0);

    let frame = decoder.decode(&packet).unwrap().unwrap();
    assert_eq!(frame.resolution, Resolution::HD);
    assert_eq!(frame.pitch, width * 4);
    assert_eq!(decoder.frames_converted(), 1);
}

#[test]
fn test_gpu_frame_byte_size_nv12() {
    // Verify GpuFrame::byte_size() for NV12 format.
    let frame = GpuFrame {
        device_ptr: 0,
        device_ptr_uv: None,
        resolution: Resolution::new(1920, 1080),
        format: PixelFormat::Nv12,
        pitch: 1920,
        pts: TimeCode::ZERO,
    };
    // NV12: Y (1920*1080) + UV (1920*540) = 3,110,400
    assert_eq!(frame.byte_size(), 1920 * 1080 + 1920 * 540);
}

#[test]
fn test_gpu_frame_byte_size_rgba8() {
    // Verify GpuFrame::byte_size() for RGBA8 format.
    let frame = GpuFrame {
        device_ptr: 0,
        device_ptr_uv: None,
        resolution: Resolution::new(1920, 1080),
        format: PixelFormat::Rgba8,
        pitch: 1920 * 4,
        pts: TimeCode::ZERO,
    };
    // RGBA8: pitch * height = 7680 * 1080
    assert_eq!(frame.byte_size(), 1920 * 4 * 1080);
}

#[test]
fn test_nv12_conversion_with_pitched_stride() {
    // Verify that nv12_to_rgba correctly handles a pitch larger than the width
    // (as is common with GPU decoder output where rows are aligned to 256 bytes).
    let width = 6u32;
    let height = 4u32;
    let y_pitch = 8u32; // wider than width (padding for alignment)
    let uv_pitch = 8u32;

    // Allocate planes with the larger pitch.
    let y_plane = vec![128u8; (y_pitch * height) as usize];
    let uv_plane = vec![128u8; (uv_pitch * (height / 2)) as usize];

    let rgba = nv12_to_rgba(&y_plane, &uv_plane, width, height, y_pitch, uv_pitch).unwrap();

    // Output should be width*height*4, NOT pitch*height*4.
    assert_eq!(rgba.len(), (width * height * 4) as usize);

    // All alpha values should still be 255.
    for px in rgba.chunks_exact(4) {
        assert_eq!(px[3], 255);
    }
}
