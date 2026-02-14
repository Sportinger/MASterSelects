//! Higher-level encoder session management.
//!
//! `EncoderSession` wraps a backend-specific encoder (currently NVENC) and
//! provides frame counting, PTS/DTS tracking, keyframe interval management,
//! and session statistics.
//!
//! This module is the primary entry point for encoding -- it hides the
//! backend-specific details and works through the `HwEncoder` trait.

use ms_common::config::EncoderConfig;
use ms_common::gpu_traits::{EncodedPacket, HwEncoder};
use ms_common::packet::GpuFrame;
use ms_common::types::{Rational, TimeCode};
use ms_common::EncodeError;

use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Encoder statistics
// ---------------------------------------------------------------------------

/// Statistics from an encoder session.
#[derive(Clone, Debug, Default)]
pub struct EncoderStats {
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Total keyframes generated.
    pub keyframes: u64,
    /// Total bytes of encoded output.
    pub bytes_written: u64,
    /// Average bitrate (bits/sec) so far.
    pub avg_bitrate_bps: f64,
    /// Encoding duration in seconds.
    pub encode_duration_secs: f64,
}

// ---------------------------------------------------------------------------
// EncoderSession
// ---------------------------------------------------------------------------

/// Higher-level encoder session wrapping a backend encoder.
///
/// Provides:
/// - Frame counting and PTS tracking
/// - Keyframe interval management
/// - Session statistics (bitrate, frame count, etc.)
/// - Backend-agnostic interface via `HwEncoder` trait
///
/// # Usage
///
/// ```ignore
/// let config = EncoderConfig { /* ... */ };
/// let mut session = EncoderSession::new(&config)?;
///
/// for frame in frames {
///     let packet = session.encode(&frame)?;
///     muxer.write_packet(&packet)?;
/// }
///
/// let remaining = session.flush()?;
/// ```
pub struct EncoderSession {
    /// The underlying hardware encoder.
    /// Currently always `None` when NVENC is not available -- in production
    /// this will hold a `Box<dyn HwEncoder>`.
    backend: Option<Box<dyn HwEncoder>>,
    /// Encoder configuration.
    config: EncoderConfig,
    /// Total frames submitted to the encoder.
    frames_submitted: u64,
    /// Total keyframes produced.
    keyframes: u64,
    /// Total bytes of encoded output.
    total_bytes: u64,
    /// GOP length (frames between keyframes).
    gop_length: u32,
    /// Frame rate for PTS calculations.
    fps: Rational,
    /// Start time (for bitrate calculation).
    start_time: Option<std::time::Instant>,
}

impl std::fmt::Debug for EncoderSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderSession")
            .field("codec", &self.config.codec)
            .field("resolution", &self.config.resolution)
            .field("fps", &self.fps)
            .field("frames_submitted", &self.frames_submitted)
            .field("keyframes", &self.keyframes)
            .field("total_bytes", &self.total_bytes)
            .field("has_backend", &self.backend.is_some())
            .finish()
    }
}

impl EncoderSession {
    /// Create a new encoder session.
    ///
    /// This validates the config and prepares the session. The actual NVENC
    /// encoder is created when the NVENC feature is enabled and hardware
    /// is available.
    ///
    /// # Errors
    /// Returns `EncodeError::HwEncoderInit` if the config is invalid.
    pub fn new(config: &EncoderConfig) -> Result<Self, EncodeError> {
        // Validate config
        if config.resolution.width == 0 || config.resolution.height == 0 {
            return Err(EncodeError::HwEncoderInit(
                "Resolution must be > 0".to_string(),
            ));
        }
        if config.fps.den == 0 {
            return Err(EncodeError::HwEncoderInit(
                "FPS denominator must be > 0".to_string(),
            ));
        }

        // GOP length: default to 1 second of video (fps.num / fps.den frames)
        let gop_length = config.fps.num.max(1);

        info!(
            codec = config.codec.display_name(),
            width = config.resolution.width,
            height = config.resolution.height,
            fps = %config.fps,
            gop = gop_length,
            "Encoder session created"
        );

        Ok(Self {
            backend: None,
            config: config.clone(),
            frames_submitted: 0,
            keyframes: 0,
            total_bytes: 0,
            gop_length,
            fps: config.fps,
            start_time: None,
        })
    }

    /// Create a session with an explicit backend encoder.
    ///
    /// Used when the caller has already created a hardware encoder
    /// (e.g., via `GpuBackend::create_encoder()`).
    pub fn with_backend(
        config: &EncoderConfig,
        backend: Box<dyn HwEncoder>,
    ) -> Result<Self, EncodeError> {
        let mut session = Self::new(config)?;
        session.backend = Some(backend);
        Ok(session)
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Get the number of frames submitted so far.
    pub fn frames_submitted(&self) -> u64 {
        self.frames_submitted
    }

    /// Check if a backend encoder is available.
    pub fn has_backend(&self) -> bool {
        self.backend.is_some()
    }

    /// Calculate the expected PTS for a given frame number.
    pub fn frame_pts(&self, frame_num: u64) -> TimeCode {
        TimeCode::from_secs(frame_num as f64 / self.fps.as_f64())
    }

    /// Check if the given frame number should be a keyframe.
    pub fn is_keyframe_position(&self, frame_num: u64) -> bool {
        if self.gop_length == 0 {
            return frame_num == 0;
        }
        frame_num.is_multiple_of(self.gop_length as u64)
    }

    /// Get encoding statistics.
    pub fn stats(&self) -> EncoderStats {
        let duration = self
            .start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let avg_bitrate = if duration > 0.0 {
            (self.total_bytes as f64 * 8.0) / duration
        } else {
            0.0
        };

        EncoderStats {
            frames_encoded: self.frames_submitted,
            keyframes: self.keyframes,
            bytes_written: self.total_bytes,
            avg_bitrate_bps: avg_bitrate,
            encode_duration_secs: duration,
        }
    }
}

impl HwEncoder for EncoderSession {
    /// Encode a GPU frame through the session.
    ///
    /// If no backend is available, returns a placeholder packet (for testing
    /// the pipeline without GPU hardware).
    fn encode(&mut self, frame: &GpuFrame) -> Result<EncodedPacket, EncodeError> {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        let packet = if let Some(ref mut backend) = self.backend {
            backend.encode(frame)?
        } else {
            // No backend available -- produce a placeholder packet for pipeline testing.
            // In production, we would return an error here.
            let is_keyframe = self.is_keyframe_position(self.frames_submitted);
            EncodedPacket {
                data: Vec::new(),
                pts: frame.pts,
                dts: frame.pts,
                is_keyframe,
            }
        };

        self.frames_submitted += 1;
        self.total_bytes += packet.data.len() as u64;
        if packet.is_keyframe {
            self.keyframes += 1;
        }

        debug!(
            frame = self.frames_submitted,
            pts = packet.pts.as_secs(),
            size = packet.data.len(),
            keyframe = packet.is_keyframe,
            "Session encoded frame"
        );

        Ok(packet)
    }

    /// Flush remaining packets from the encoder.
    fn flush(&mut self) -> Result<Vec<EncodedPacket>, EncodeError> {
        let packets = if let Some(ref mut backend) = self.backend {
            backend.flush()?
        } else {
            Vec::new()
        };

        for p in &packets {
            self.total_bytes += p.data.len() as u64;
            if p.is_keyframe {
                self.keyframes += 1;
            }
        }

        info!(
            flushed = packets.len(),
            total_frames = self.frames_submitted,
            total_bytes = self.total_bytes,
            "Encoder session flushed"
        );

        Ok(packets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_common::color::PixelFormat;
    use ms_common::config::{EncoderBitrate, EncoderPreset, EncoderProfile};
    use ms_common::types::Resolution;
    use ms_common::VideoCodec;

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

    fn make_frame(pts_secs: f64) -> GpuFrame {
        GpuFrame {
            device_ptr: 0x1000_0000,
            device_ptr_uv: Some(0x1000_0000 + 1920 * 1080),
            resolution: Resolution::HD,
            format: PixelFormat::Nv12,
            pitch: 1920,
            pts: TimeCode::from_secs(pts_secs),
        }
    }

    #[test]
    fn session_creation() {
        let config = make_config();
        let session = EncoderSession::new(&config).unwrap();
        assert_eq!(session.frames_submitted(), 0);
        assert!(!session.has_backend());
    }

    #[test]
    fn session_invalid_resolution() {
        let mut config = make_config();
        config.resolution = Resolution::new(0, 0);
        assert!(EncoderSession::new(&config).is_err());
    }

    #[test]
    #[should_panic(expected = "Rational denominator must be > 0")]
    fn session_invalid_fps_panics() {
        // Rational::new panics when den == 0 (enforced by the type)
        let _r = Rational::new(30, 0);
    }

    #[test]
    fn frame_pts_calculation() {
        let config = make_config();
        let session = EncoderSession::new(&config).unwrap();

        let pts0 = session.frame_pts(0);
        assert!((pts0.as_secs() - 0.0).abs() < 1e-9);

        let pts30 = session.frame_pts(30);
        assert!((pts30.as_secs() - 1.0).abs() < 1e-9);

        let pts15 = session.frame_pts(15);
        assert!((pts15.as_secs() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn keyframe_positions_30fps() {
        let config = make_config();
        let session = EncoderSession::new(&config).unwrap();

        // GOP length = fps.num = 30 for 30fps
        assert!(session.is_keyframe_position(0));
        assert!(!session.is_keyframe_position(1));
        assert!(!session.is_keyframe_position(29));
        assert!(session.is_keyframe_position(30));
        assert!(session.is_keyframe_position(60));
    }

    #[test]
    fn encode_without_backend() {
        let config = make_config();
        let mut session = EncoderSession::new(&config).unwrap();

        let frame = make_frame(0.0);
        let packet = session.encode(&frame).unwrap();

        assert!(packet.is_keyframe); // First frame is always a keyframe
        assert_eq!(session.frames_submitted(), 1);
    }

    #[test]
    fn encode_multiple_frames() {
        let config = make_config();
        let mut session = EncoderSession::new(&config).unwrap();

        for i in 0..10 {
            let frame = make_frame(i as f64 / 30.0);
            let _ = session.encode(&frame).unwrap();
        }

        assert_eq!(session.frames_submitted(), 10);
    }

    #[test]
    fn flush_without_backend() {
        let config = make_config();
        let mut session = EncoderSession::new(&config).unwrap();

        // Encode a few frames first
        for i in 0..5 {
            let frame = make_frame(i as f64 / 30.0);
            let _ = session.encode(&frame).unwrap();
        }

        let packets = session.flush().unwrap();
        assert!(packets.is_empty()); // No backend = no flushed packets
    }

    #[test]
    fn stats_default() {
        let config = make_config();
        let session = EncoderSession::new(&config).unwrap();
        let stats = session.stats();
        assert_eq!(stats.frames_encoded, 0);
        assert_eq!(stats.keyframes, 0);
        assert_eq!(stats.bytes_written, 0);
    }

    #[test]
    fn stats_after_encoding() {
        let config = make_config();
        let mut session = EncoderSession::new(&config).unwrap();

        // Without a backend, data is empty but frame counting works
        for i in 0..60 {
            let frame = make_frame(i as f64 / 30.0);
            let _ = session.encode(&frame).unwrap();
        }

        let stats = session.stats();
        assert_eq!(stats.frames_encoded, 60);
        // At 30fps with GOP=30, we expect keyframes at 0 and 30 = 2 keyframes
        assert_eq!(stats.keyframes, 2);
    }

    #[test]
    fn encoder_stats_debug() {
        let stats = EncoderStats::default();
        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("frames_encoded"));
    }

    #[test]
    fn session_debug() {
        let config = make_config();
        let session = EncoderSession::new(&config).unwrap();
        let debug_str = format!("{session:?}");
        assert!(debug_str.contains("EncoderSession"));
        assert!(debug_str.contains("H264"));
    }
}
