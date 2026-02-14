//! Export pipeline -- ties timeline evaluation, compositing, encoding, and muxing.
//!
//! The `ExportPipeline` orchestrates the full export process:
//! 1. Evaluate the timeline at each frame time
//! 2. Composite the layers into a single GPU frame
//! 3. Encode the frame via NVENC (or future Vulkan Video Encode)
//! 4. Mux the encoded packets into a container file (MP4/MKV)
//!
//! Progress is reported via a crossbeam channel so the UI can display
//! a progress bar and allow cancellation.
//!
//! # Architecture
//!
//! ```text
//! ExportPipeline::start()
//!   |
//!   +-- Spawn export thread
//!   |     |
//!   |     +-- for frame in 0..total_frames:
//!   |     |     1. timeline_eval.evaluate(frame_time)
//!   |     |     2. compositor.render(layers) -> GpuFrame
//!   |     |     3. encoder.encode(gpu_frame) -> EncodedPacket
//!   |     |     4. muxer.write_packet(packet)
//!   |     |     5. Send progress update via channel
//!   |     |
//!   |     +-- encoder.flush() -> remaining packets
//!   |     +-- muxer.finalize()
//!   |
//!   +-- Returns ExportHandle (for progress/cancel)
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam::channel::{self, Receiver, Sender};
use tracing::{debug, info};

use ms_common::codec::VideoCodec;
use ms_common::config::{EncoderBitrate, EncoderConfig, EncoderPreset, EncoderProfile};
use ms_common::types::{Rational, Resolution};

use crate::error::ExportError;

// ---------------------------------------------------------------------------
// Export configuration
// ---------------------------------------------------------------------------

/// Configuration for an export operation.
#[derive(Clone, Debug)]
pub struct ExportConfig {
    /// Output file path.
    pub output_path: PathBuf,
    /// Output resolution.
    pub resolution: Resolution,
    /// Output frame rate.
    pub fps: Rational,
    /// Video codec.
    pub codec: VideoCodec,
    /// Bitrate settings.
    pub bitrate: EncoderBitrate,
    /// Encoder speed/quality preset.
    pub preset: EncoderPreset,
    /// Encoder profile.
    pub profile: EncoderProfile,
    /// Timeline duration in seconds.
    pub duration_secs: f64,
    /// Whether to include audio.
    pub include_audio: bool,
}

impl ExportConfig {
    /// Calculate the total number of frames to encode.
    pub fn total_frames(&self) -> u64 {
        (self.duration_secs * self.fps.as_f64()).ceil() as u64
    }

    /// Build an `EncoderConfig` from this export config.
    pub fn to_encoder_config(&self) -> EncoderConfig {
        EncoderConfig {
            codec: self.codec,
            resolution: self.resolution,
            fps: self.fps,
            bitrate: self.bitrate.clone(),
            preset: self.preset,
            profile: self.profile,
        }
    }

    /// Validate the export config.
    pub fn validate(&self) -> Result<(), ExportError> {
        if self.resolution.width == 0 || self.resolution.height == 0 {
            return Err(ExportError::InvalidConfig(
                "Resolution must be > 0".to_string(),
            ));
        }
        if !self.resolution.width.is_multiple_of(2) || !self.resolution.height.is_multiple_of(2) {
            return Err(ExportError::InvalidConfig(
                "Resolution width and height must be even".to_string(),
            ));
        }
        if self.fps.den == 0 {
            return Err(ExportError::InvalidConfig(
                "FPS denominator must be > 0".to_string(),
            ));
        }
        if self.duration_secs <= 0.0 {
            return Err(ExportError::InvalidConfig(
                "Duration must be > 0".to_string(),
            ));
        }
        if self.output_path.as_os_str().is_empty() {
            return Err(ExportError::InvalidConfig(
                "Output path must not be empty".to_string(),
            ));
        }

        // Validate codec support for encoding
        match self.codec {
            VideoCodec::H264 | VideoCodec::H265 => Ok(()),
            _ => Err(ExportError::InvalidConfig(format!(
                "Codec {:?} is not supported for encoding",
                self.codec
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Export progress
// ---------------------------------------------------------------------------

/// Progress update from the export pipeline.
#[derive(Clone, Debug)]
pub enum ExportProgress {
    /// Export has started.
    Started {
        /// Total frames to encode.
        total_frames: u64,
    },
    /// A frame has been encoded.
    FrameEncoded {
        /// Current frame number (0-based).
        frame: u64,
        /// Total frames.
        total: u64,
        /// Encoded bytes for this frame.
        bytes: usize,
    },
    /// Export completed successfully.
    Completed {
        /// Total bytes written.
        total_bytes: u64,
        /// Total encoding time in seconds.
        duration_secs: f64,
    },
    /// Export failed.
    Failed {
        /// Error description.
        error: String,
    },
    /// Export was cancelled.
    Cancelled,
}

impl ExportProgress {
    /// Get the progress as a fraction (0.0 to 1.0).
    pub fn progress_fraction(&self) -> f64 {
        match self {
            Self::Started { .. } => 0.0,
            Self::FrameEncoded { frame, total, .. } => {
                if *total > 0 {
                    *frame as f64 / *total as f64
                } else {
                    0.0
                }
            }
            Self::Completed { .. } => 1.0,
            Self::Failed { .. } | Self::Cancelled => 0.0,
        }
    }

    /// Check if the export is still in progress.
    pub fn is_in_progress(&self) -> bool {
        matches!(self, Self::Started { .. } | Self::FrameEncoded { .. })
    }

    /// Check if the export has finished (success, failure, or cancellation).
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }
}

// ---------------------------------------------------------------------------
// Export handle (returned to the UI)
// ---------------------------------------------------------------------------

/// Handle for monitoring and controlling an active export.
///
/// The UI holds this handle to receive progress updates and request
/// cancellation. The export runs on a separate thread.
#[derive(Debug)]
pub struct ExportHandle {
    /// Receiver for progress updates.
    progress_rx: Receiver<ExportProgress>,
    /// Shared cancellation flag.
    cancel_flag: Arc<AtomicBool>,
    /// Export config (for display purposes).
    config: ExportConfig,
}

impl ExportHandle {
    /// Try to receive the latest progress update (non-blocking).
    pub fn try_recv_progress(&self) -> Option<ExportProgress> {
        self.progress_rx.try_recv().ok()
    }

    /// Wait for the next progress update (blocking).
    pub fn recv_progress(&self) -> Option<ExportProgress> {
        self.progress_rx.recv().ok()
    }

    /// Drain all pending progress updates.
    pub fn drain_progress(&self) -> Vec<ExportProgress> {
        let mut updates = Vec::new();
        while let Ok(p) = self.progress_rx.try_recv() {
            updates.push(p);
        }
        updates
    }

    /// Request cancellation of the export.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        info!("Export cancellation requested");
    }

    /// Check if cancellation has been requested.
    pub fn is_cancel_requested(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }

    /// Get the export configuration.
    pub fn config(&self) -> &ExportConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Export pipeline
// ---------------------------------------------------------------------------

/// The export pipeline orchestrator.
///
/// Creates and manages the export thread. Currently a skeleton that will
/// be connected to the timeline evaluator, compositor, and muxer in
/// later phases.
pub struct ExportPipeline;

impl ExportPipeline {
    /// Start an export operation.
    ///
    /// Validates the config, creates the progress channel and cancellation
    /// flag, and spawns the export thread.
    ///
    /// Returns an `ExportHandle` that the caller can use to monitor progress
    /// and request cancellation.
    pub fn start(config: ExportConfig) -> Result<ExportHandle, ExportError> {
        config.validate()?;

        let (progress_tx, progress_rx) = channel::unbounded::<ExportProgress>();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel_clone = cancel_flag.clone();

        let total_frames = config.total_frames();

        info!(
            output = %config.output_path.display(),
            codec = ?config.codec,
            resolution = %config.resolution,
            fps = %config.fps,
            frames = total_frames,
            "Starting export"
        );

        // Spawn the export thread
        let config_clone = config.clone();
        std::thread::Builder::new()
            .name("export-pipeline".to_string())
            .spawn(move || {
                Self::run_export(config_clone, progress_tx, cancel_clone);
            })
            .map_err(|e| ExportError::InitFailed(format!("Failed to spawn export thread: {e}")))?;

        Ok(ExportHandle {
            progress_rx,
            cancel_flag,
            config,
        })
    }

    /// The main export loop (runs on the export thread).
    ///
    /// Currently a skeleton that simulates frame-by-frame encoding.
    /// Will be connected to real components in later phases.
    fn run_export(
        config: ExportConfig,
        progress_tx: Sender<ExportProgress>,
        cancel_flag: Arc<AtomicBool>,
    ) {
        let total_frames = config.total_frames();
        let start_time = std::time::Instant::now();

        // Notify: export started
        let _ = progress_tx.send(ExportProgress::Started { total_frames });

        // Frame-by-frame encode loop (skeleton)
        let total_bytes: u64 = 0;

        for frame_num in 0..total_frames {
            // Check for cancellation
            if cancel_flag.load(Ordering::SeqCst) {
                let _ = progress_tx.send(ExportProgress::Cancelled);
                info!("Export cancelled at frame {frame_num}/{total_frames}");
                return;
            }

            // TODO: In later phases, this will:
            // 1. Evaluate timeline at frame_time
            // 2. Composite layers into a GPU frame
            // 3. Encode the frame
            // 4. Write to muxer

            // For now, just report progress
            let _ = progress_tx.send(ExportProgress::FrameEncoded {
                frame: frame_num,
                total: total_frames,
                bytes: 0,
            });

            debug!(
                frame = frame_num,
                total = total_frames,
                "Export frame processed (skeleton)"
            );
        }

        let duration = start_time.elapsed().as_secs_f64();

        let _ = progress_tx.send(ExportProgress::Completed {
            total_bytes,
            duration_secs: duration,
        });

        info!(
            frames = total_frames,
            bytes = total_bytes,
            duration_secs = duration,
            "Export completed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_export_config() -> ExportConfig {
        ExportConfig {
            output_path: PathBuf::from("test_output.mp4"),
            resolution: Resolution::HD,
            fps: Rational::FPS_30,
            codec: VideoCodec::H264,
            bitrate: EncoderBitrate::Vbr {
                target: 20_000_000,
                max: 30_000_000,
            },
            preset: EncoderPreset::Medium,
            profile: EncoderProfile::High,
            duration_secs: 10.0,
            include_audio: true,
        }
    }

    #[test]
    fn export_config_total_frames() {
        let config = make_export_config();
        assert_eq!(config.total_frames(), 300); // 10 sec * 30 fps
    }

    #[test]
    fn export_config_total_frames_fractional() {
        let mut config = make_export_config();
        config.duration_secs = 1.5;
        assert_eq!(config.total_frames(), 45); // 1.5 * 30 = 45
    }

    #[test]
    fn export_config_to_encoder_config() {
        let config = make_export_config();
        let enc = config.to_encoder_config();
        assert_eq!(enc.codec, VideoCodec::H264);
        assert_eq!(enc.resolution, Resolution::HD);
        assert_eq!(enc.fps, Rational::FPS_30);
    }

    #[test]
    fn export_config_validate_valid() {
        let config = make_export_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn export_config_validate_zero_resolution() {
        let mut config = make_export_config();
        config.resolution = Resolution::new(0, 1080);
        assert!(config.validate().is_err());
    }

    #[test]
    fn export_config_validate_odd_resolution() {
        let mut config = make_export_config();
        config.resolution = Resolution::new(1921, 1080);
        assert!(config.validate().is_err());
    }

    #[test]
    fn export_config_validate_zero_duration() {
        let mut config = make_export_config();
        config.duration_secs = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn export_config_validate_negative_duration() {
        let mut config = make_export_config();
        config.duration_secs = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn export_config_validate_empty_path() {
        let mut config = make_export_config();
        config.output_path = PathBuf::from("");
        assert!(config.validate().is_err());
    }

    #[test]
    fn export_config_validate_unsupported_codec() {
        let mut config = make_export_config();
        config.codec = VideoCodec::Vp9;
        assert!(config.validate().is_err());
    }

    #[test]
    fn export_config_validate_hevc() {
        let mut config = make_export_config();
        config.codec = VideoCodec::H265;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn export_progress_fraction() {
        let started = ExportProgress::Started { total_frames: 100 };
        assert!((started.progress_fraction() - 0.0).abs() < 1e-9);

        let mid = ExportProgress::FrameEncoded {
            frame: 50,
            total: 100,
            bytes: 1024,
        };
        assert!((mid.progress_fraction() - 0.5).abs() < 1e-9);

        let done = ExportProgress::Completed {
            total_bytes: 1000,
            duration_secs: 5.0,
        };
        assert!((done.progress_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn export_progress_is_in_progress() {
        assert!(ExportProgress::Started { total_frames: 100 }.is_in_progress());
        assert!(ExportProgress::FrameEncoded {
            frame: 50,
            total: 100,
            bytes: 0
        }
        .is_in_progress());
        assert!(!ExportProgress::Completed {
            total_bytes: 0,
            duration_secs: 0.0
        }
        .is_in_progress());
        assert!(!ExportProgress::Failed {
            error: "test".to_string()
        }
        .is_in_progress());
        assert!(!ExportProgress::Cancelled.is_in_progress());
    }

    #[test]
    fn export_progress_is_finished() {
        assert!(!ExportProgress::Started { total_frames: 100 }.is_finished());
        assert!(ExportProgress::Completed {
            total_bytes: 0,
            duration_secs: 0.0
        }
        .is_finished());
        assert!(ExportProgress::Failed {
            error: "test".to_string()
        }
        .is_finished());
        assert!(ExportProgress::Cancelled.is_finished());
    }

    #[test]
    fn export_pipeline_start_and_cancel() {
        let config = make_export_config();
        let handle = ExportPipeline::start(config).unwrap();

        // Cancel immediately
        handle.cancel();
        assert!(handle.is_cancel_requested());

        // Wait for the export thread to finish
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Drain progress -- should have Started and then Cancelled
        let updates = handle.drain_progress();
        assert!(!updates.is_empty());

        // The last update should be Cancelled (or Completed if it was fast enough)
        let last = updates.last().unwrap();
        assert!(last.is_finished());
    }

    #[test]
    fn export_pipeline_completes() {
        let mut config = make_export_config();
        config.duration_secs = 0.1; // Very short for fast test
        config.fps = Rational::FPS_30;

        let handle = ExportPipeline::start(config).unwrap();

        // Wait for completion
        let mut last_progress = None;
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let updates = handle.drain_progress();
            if let Some(last) = updates.last() {
                if last.is_finished() {
                    last_progress = Some(last.clone());
                    break;
                }
            }
        }

        assert!(last_progress.is_some());
        assert!(last_progress.unwrap().is_finished());
    }

    #[test]
    fn export_pipeline_invalid_config() {
        let mut config = make_export_config();
        config.resolution = Resolution::new(0, 0);
        assert!(ExportPipeline::start(config).is_err());
    }

    #[test]
    fn export_handle_config_access() {
        let config = make_export_config();
        let handle = ExportPipeline::start(config.clone()).unwrap();
        assert_eq!(handle.config().codec, config.codec);
        assert_eq!(handle.config().resolution, config.resolution);
        handle.cancel(); // Clean up
    }
}
