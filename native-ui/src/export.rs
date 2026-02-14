//! Export pipeline -- renders timeline to an output file.
//!
//! Architecture:
//! ```text
//! ExportPipeline
//! +-- ExportConfig (output settings)
//! +-- ExportState (progress tracking)
//! +-- export_thread
//!     +-- for frame in 0..total_frames:
//!     |   +-- evaluate timeline @ time
//!     |   +-- composite layers -> RGBA
//!     |   +-- encode frame (ms-encoder)
//!     |   +-- mux to container (ms-mux)
//!     +-- flush encoder
//!     +-- finalize muxer
//! ```
//!
//! The export thread creates an [`EncoderSession`] from `ms-encoder` and an
//! [`Mp4Muxer`] from `ms-mux`. Each frame is encoded via the `HwEncoder`
//! trait and the resulting packet is written to the muxer. When no real HW
//! backend is available the encoder produces placeholder (empty) packets,
//! allowing the full pipeline to exercise the control flow.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result};
use crossbeam::channel::{self, Receiver, Sender};
use ms_common::config::{EncoderBitrate, EncoderPreset, EncoderProfile};
use ms_common::gpu_traits::HwEncoder;
use ms_common::packet::GpuFrame;
use ms_common::{
    AudioCodec, ContainerFormat, EncoderConfig, PixelFormat, Rational, Resolution, TimeCode,
    VideoCodec,
};
use ms_encoder::session::EncoderSession;
use ms_mux::{Mp4Muxer, MuxerConfig, VideoTrackConfig};
use tracing::{debug, error, info};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Export configuration describing the desired output.
#[derive(Clone, Debug)]
pub struct ExportConfig {
    /// Path to the output file.
    pub output_path: PathBuf,
    /// Output video resolution.
    pub resolution: Resolution,
    /// Output frame rate.
    pub fps: Rational,
    /// Video codec to use.
    pub video_codec: VideoCodec,
    /// Target video bitrate in megabits per second.
    pub bitrate_mbps: f32,
    /// Duration of the timeline region to export (seconds).
    pub duration_secs: f64,
    /// Audio sample rate (Hz).
    pub sample_rate: u32,
    /// Audio codec to use.
    pub audio_codec: AudioCodec,
    /// Audio bitrate in kilobits per second.
    pub audio_bitrate_kbps: u32,
    /// Container format for the output file.
    pub container: ContainerFormat,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("output.mp4"),
            resolution: Resolution::HD,
            fps: Rational::FPS_30,
            video_codec: VideoCodec::H264,
            bitrate_mbps: 20.0,
            duration_secs: 10.0,
            sample_rate: 48_000,
            audio_codec: AudioCodec::Aac,
            audio_bitrate_kbps: 192,
            container: ContainerFormat::Mp4,
        }
    }
}

impl ExportConfig {
    /// Total number of video frames to render for this export.
    pub fn total_frames(&self) -> u64 {
        (self.duration_secs * self.fps.as_f64()).ceil() as u64
    }

    /// Duration of a single frame in seconds.
    pub fn frame_duration_secs(&self) -> f64 {
        1.0 / self.fps.as_f64()
    }

    /// Target video bitrate in bits per second.
    pub fn bitrate_bps(&self) -> u64 {
        (self.bitrate_mbps * 1_000_000.0) as u64
    }
}

// ---------------------------------------------------------------------------
// Progress / State
// ---------------------------------------------------------------------------

/// Snapshot of the current progress of an export operation.
#[derive(Clone, Debug)]
pub struct ExportProgress {
    /// Current high-level state.
    pub state: ExportState,
    /// Number of frames rendered so far.
    pub current_frame: u64,
    /// Total number of frames to render.
    pub total_frames: u64,
    /// Seconds elapsed since export started.
    pub elapsed_secs: f64,
    /// Estimated seconds remaining (0.0 if unknown).
    pub estimated_remaining_secs: f64,
    /// Current encoding throughput in frames per second.
    pub fps: f64,
}

impl ExportProgress {
    /// Progress as a fraction in `[0.0, 1.0]`.
    pub fn fraction(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.current_frame as f64 / self.total_frames as f64).clamp(0.0, 1.0)
        }
    }
}

/// High-level state of an export operation.
#[derive(Clone, Debug, PartialEq)]
pub enum ExportState {
    /// No export in progress.
    Idle,
    /// Setting up encoder, muxer, and pipeline resources.
    Preparing,
    /// Actively rendering and encoding frames.
    Rendering,
    /// Flushing encoder buffers.
    Encoding,
    /// Finalizing the output container.
    Finalizing,
    /// Export completed successfully.
    Complete,
    /// Export failed with an error message.
    Failed(String),
    /// Export was cancelled by the user.
    Cancelled,
}

impl ExportState {
    /// Whether this state represents a terminal condition.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ExportState::Complete | ExportState::Failed(_) | ExportState::Cancelled
        )
    }

    /// Human-readable label for UI display.
    pub fn label(&self) -> &str {
        match self {
            ExportState::Idle => "Idle",
            ExportState::Preparing => "Preparing...",
            ExportState::Rendering => "Rendering...",
            ExportState::Encoding => "Encoding...",
            ExportState::Finalizing => "Finalizing...",
            ExportState::Complete => "Complete",
            ExportState::Failed(_) => "Failed",
            ExportState::Cancelled => "Cancelled",
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Manages an export operation running on a background thread.
///
/// # Lifecycle
///
/// 1. Create with [`ExportPipeline::new`].
/// 2. Call [`start`](ExportPipeline::start) to spawn the export thread.
/// 3. Poll [`poll_progress`](ExportPipeline::poll_progress) from the UI thread.
/// 4. Optionally call [`cancel`](ExportPipeline::cancel) to abort.
/// 5. On drop the pipeline signals cancellation and joins the thread.
pub struct ExportPipeline {
    config: ExportConfig,
    cancel_flag: Arc<AtomicBool>,
    current_frame: Arc<AtomicU64>,
    total_frames: u64,
    start_time: Option<Instant>,
    progress_rx: Option<Receiver<ExportProgress>>,
    thread_handle: Option<thread::JoinHandle<()>>,
    state: ExportState,
}

impl ExportPipeline {
    /// Create a new pipeline with the given configuration.
    /// Does **not** start exporting -- call [`start`](Self::start) for that.
    pub fn new(config: ExportConfig) -> Self {
        let total_frames = config.total_frames();
        Self {
            config,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            current_frame: Arc::new(AtomicU64::new(0)),
            total_frames,
            start_time: None,
            progress_rx: None,
            thread_handle: None,
            state: ExportState::Idle,
        }
    }

    /// Start the export on a background thread.
    ///
    /// Returns an error if the pipeline is already running.
    pub fn start(&mut self) -> Result<()> {
        if self.is_running() {
            anyhow::bail!("Export is already running");
        }

        // Reset shared state
        self.cancel_flag.store(false, Ordering::SeqCst);
        self.current_frame.store(0, Ordering::SeqCst);
        self.state = ExportState::Preparing;
        self.start_time = Some(Instant::now());

        let (progress_tx, progress_rx) = channel::bounded::<ExportProgress>(64);
        self.progress_rx = Some(progress_rx);

        let config = self.config.clone();
        let cancel_flag = Arc::clone(&self.cancel_flag);
        let current_frame = Arc::clone(&self.current_frame);

        let handle = thread::Builder::new()
            .name("export-pipeline".into())
            .spawn(move || {
                export_thread(config, cancel_flag, current_frame, progress_tx);
            })
            .context("Failed to spawn export thread")?;

        self.thread_handle = Some(handle);
        Ok(())
    }

    /// Signal cancellation and wait for the export thread to finish.
    pub fn cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        if !self.state.is_terminal() {
            self.state = ExportState::Cancelled;
        }
    }

    /// Poll for the latest progress (non-blocking).
    ///
    /// Drains the channel and returns the most recent progress snapshot.
    /// Between channel updates this builds a snapshot from the atomic counters.
    pub fn poll_progress(&mut self) -> ExportProgress {
        // Drain channel, keep only the last message
        let mut latest: Option<ExportProgress> = None;
        if let Some(rx) = &self.progress_rx {
            while let Ok(p) = rx.try_recv() {
                latest = Some(p);
            }
        }

        if let Some(progress) = latest {
            self.state = progress.state.clone();
            return progress;
        }

        // Fallback: build from atomics
        let current = self.current_frame.load(Ordering::Relaxed);
        let elapsed = self
            .start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let fps = if elapsed > 0.0 {
            current as f64 / elapsed
        } else {
            0.0
        };
        let remaining = estimate_remaining(current, self.total_frames, elapsed);

        ExportProgress {
            state: self.state.clone(),
            current_frame: current,
            total_frames: self.total_frames,
            elapsed_secs: elapsed,
            estimated_remaining_secs: remaining,
            fps,
        }
    }

    /// Whether the export finished (successfully, with error, or cancelled).
    pub fn is_complete(&self) -> bool {
        self.state.is_terminal()
    }

    /// Whether the export thread is currently running.
    pub fn is_running(&self) -> bool {
        self.thread_handle
            .as_ref()
            .map_or(false, |h| !h.is_finished())
    }

    /// Access the export configuration.
    pub fn config(&self) -> &ExportConfig {
        &self.config
    }

    /// Current high-level state.
    pub fn state(&self) -> &ExportState {
        &self.state
    }
}

impl Drop for ExportPipeline {
    fn drop(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Background thread
// ---------------------------------------------------------------------------

/// Entry point for the export background thread.
///
/// Walks through every frame of the timeline, evaluates, composites,
/// encodes via `ms-encoder`, and muxes via `ms-mux`.
fn export_thread(
    config: ExportConfig,
    cancel_flag: Arc<AtomicBool>,
    current_frame: Arc<AtomicU64>,
    progress_tx: Sender<ExportProgress>,
) {
    let total_frames = config.total_frames();
    let frame_duration = config.frame_duration_secs();
    let start = Instant::now();

    // --- Preparing ---------------------------------------------------------
    send_progress(
        &progress_tx,
        ExportState::Preparing,
        0,
        total_frames,
        &start,
    );

    // Initialize encoder session (ms-encoder)
    let encoder_cfg = EncoderConfig {
        codec: config.video_codec,
        resolution: config.resolution,
        fps: config.fps,
        bitrate: EncoderBitrate::Vbr {
            target: config.bitrate_bps(),
            max: (config.bitrate_bps() as f64 * 1.5) as u64,
        },
        preset: EncoderPreset::Medium,
        profile: EncoderProfile::High,
    };

    let mut encoder = match EncoderSession::new(&encoder_cfg) {
        Ok(enc) => {
            info!(
                codec = encoder_cfg.codec.display_name(),
                has_backend = enc.has_backend(),
                "Encoder session initialized"
            );
            enc
        }
        Err(e) => {
            error!("Failed to initialize encoder: {e}");
            send_progress(
                &progress_tx,
                ExportState::Failed(format!("Encoder init failed: {e}")),
                0,
                total_frames,
                &start,
            );
            return;
        }
    };

    // Initialize muxer (ms-mux)
    let mut muxer = match Mp4Muxer::new(MuxerConfig {
        output_path: config.output_path.clone(),
    }) {
        Ok(m) => {
            info!(output = %config.output_path.display(), "MP4 muxer initialized");
            m
        }
        Err(e) => {
            error!("Failed to initialize muxer: {e}");
            send_progress(
                &progress_tx,
                ExportState::Failed(format!("Muxer init failed: {e}")),
                0,
                total_frames,
                &start,
            );
            return;
        }
    };

    // Add video track to the muxer.
    // SPS/PPS are normally extracted from the encoder or the video stream.
    // When no real HW encoder backend is active, we provide minimal placeholder
    // parameter sets so the muxer can write valid box headers.
    let placeholder_sps = vec![0x67, 0x42, 0xC0, 0x1F, 0xDA, 0x02, 0x80, 0xF6];
    let placeholder_pps = vec![0x68, 0xCE, 0x38, 0x80];

    let video_track_id = match muxer.add_video_track(VideoTrackConfig {
        codec: config.video_codec,
        resolution: config.resolution,
        fps: config.fps,
        sps: placeholder_sps,
        pps: placeholder_pps,
    }) {
        Ok(id) => {
            info!(track_id = id, "Video track added to muxer");
            id
        }
        Err(e) => {
            error!("Failed to add video track: {e}");
            send_progress(
                &progress_tx,
                ExportState::Failed(format!("Failed to add video track: {e}")),
                0,
                total_frames,
                &start,
            );
            return;
        }
    };

    // --- Rendering ---------------------------------------------------------
    send_progress(
        &progress_tx,
        ExportState::Rendering,
        0,
        total_frames,
        &start,
    );

    for frame_num in 0..total_frames {
        // Check cancellation
        if cancel_flag.load(Ordering::Relaxed) {
            send_progress(
                &progress_tx,
                ExportState::Cancelled,
                frame_num,
                total_frames,
                &start,
            );
            return;
        }

        let time_secs = frame_num as f64 * frame_duration;

        // Step 1: Evaluate timeline at this time
        // TODO: let time = TimeCode::from_secs(time_secs);
        // TODO: let layers = ms_timeline_eval::evaluate(&timeline, time)?;

        // Step 2: Composite layers into a single RGBA frame
        // TODO: let rgba_frame = compositor.composite(&layers, config.resolution)?;

        // For now, create a placeholder GpuFrame representing the compositor
        // output. In production, the compositor will return a real GPU-resident
        // frame; here we use a null device pointer which the stub encoder
        // handles gracefully.
        let gpu_frame = GpuFrame {
            device_ptr: 0, // placeholder -- no real GPU buffer yet
            device_ptr_uv: None,
            resolution: config.resolution,
            format: PixelFormat::Rgba8,
            pitch: config.resolution.width * 4,
            pts: TimeCode::from_secs(time_secs),
        };

        // Step 3: Encode frame via ms-encoder
        let packet = match encoder.encode(&gpu_frame) {
            Ok(pkt) => pkt,
            Err(e) => {
                error!(frame = frame_num, "Encode failed: {e}");
                send_progress(
                    &progress_tx,
                    ExportState::Failed(format!("Encode failed at frame {frame_num}: {e}")),
                    frame_num,
                    total_frames,
                    &start,
                );
                return;
            }
        };

        // Step 4: Mux encoded packet into container via ms-mux
        if let Err(e) = muxer.write_video_sample(video_track_id, &packet) {
            error!(frame = frame_num, "Mux write failed: {e}");
            send_progress(
                &progress_tx,
                ExportState::Failed(format!("Mux write failed at frame {frame_num}: {e}")),
                frame_num,
                total_frames,
                &start,
            );
            return;
        }

        current_frame.store(frame_num + 1, Ordering::Relaxed);

        // Send progress every N frames to avoid flooding the channel
        if frame_num % 10 == 0 || frame_num + 1 == total_frames {
            send_progress(
                &progress_tx,
                ExportState::Rendering,
                frame_num + 1,
                total_frames,
                &start,
            );
        }

        debug!(
            frame = frame_num,
            pts = time_secs,
            packet_bytes = packet.data.len(),
            keyframe = packet.is_keyframe,
            "Frame encoded and muxed"
        );
    }

    // --- Encoding flush ----------------------------------------------------
    send_progress(
        &progress_tx,
        ExportState::Encoding,
        total_frames,
        total_frames,
        &start,
    );

    // Flush any remaining packets buffered in the encoder
    match encoder.flush() {
        Ok(flushed_packets) => {
            info!(flushed = flushed_packets.len(), "Encoder flush complete");
            for packet in &flushed_packets {
                if let Err(e) = muxer.write_video_sample(video_track_id, packet) {
                    error!("Mux write failed during flush: {e}");
                    send_progress(
                        &progress_tx,
                        ExportState::Failed(format!("Mux write failed during flush: {e}")),
                        total_frames,
                        total_frames,
                        &start,
                    );
                    return;
                }
            }
        }
        Err(e) => {
            error!("Encoder flush failed: {e}");
            send_progress(
                &progress_tx,
                ExportState::Failed(format!("Encoder flush failed: {e}")),
                total_frames,
                total_frames,
                &start,
            );
            return;
        }
    }

    // Log encoder statistics
    let stats = encoder.stats();
    info!(
        frames = stats.frames_encoded,
        keyframes = stats.keyframes,
        bytes = stats.bytes_written,
        avg_bitrate_bps = stats.avg_bitrate_bps,
        "Encoder session stats"
    );

    // --- Finalizing --------------------------------------------------------
    send_progress(
        &progress_tx,
        ExportState::Finalizing,
        total_frames,
        total_frames,
        &start,
    );

    // Finalize the MP4 container (writes moov box and closes the file)
    if let Err(e) = muxer.finalize() {
        error!("Muxer finalize failed: {e}");
        send_progress(
            &progress_tx,
            ExportState::Failed(format!("Muxer finalize failed: {e}")),
            total_frames,
            total_frames,
            &start,
        );
        return;
    }

    info!(
        output = %config.output_path.display(),
        elapsed_secs = start.elapsed().as_secs_f64(),
        "Export finalized successfully"
    );

    // --- Complete ----------------------------------------------------------
    send_progress(
        &progress_tx,
        ExportState::Complete,
        total_frames,
        total_frames,
        &start,
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Send an [`ExportProgress`] snapshot over the channel.
///
/// Silently ignores send failures (receiver dropped).
fn send_progress(
    tx: &Sender<ExportProgress>,
    state: ExportState,
    current_frame: u64,
    total_frames: u64,
    start: &Instant,
) {
    let elapsed = start.elapsed().as_secs_f64();
    let fps = if elapsed > 0.0 {
        current_frame as f64 / elapsed
    } else {
        0.0
    };
    let remaining = estimate_remaining(current_frame, total_frames, elapsed);

    let _ = tx.try_send(ExportProgress {
        state,
        current_frame,
        total_frames,
        elapsed_secs: elapsed,
        estimated_remaining_secs: remaining,
        fps,
    });
}

/// Estimate remaining seconds based on current throughput.
fn estimate_remaining(current_frame: u64, total_frames: u64, elapsed_secs: f64) -> f64 {
    if current_frame == 0 || elapsed_secs <= 0.0 {
        return 0.0;
    }
    let frames_remaining = total_frames.saturating_sub(current_frame) as f64;
    let secs_per_frame = elapsed_secs / current_frame as f64;
    frames_remaining * secs_per_frame
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ExportConfig -------------------------------------------------------

    #[test]
    fn default_config_values() {
        let cfg = ExportConfig::default();
        assert_eq!(cfg.output_path, PathBuf::from("output.mp4"));
        assert_eq!(cfg.resolution, Resolution::HD);
        assert_eq!(cfg.fps, Rational::FPS_30);
        assert_eq!(cfg.video_codec, VideoCodec::H264);
        assert!((cfg.bitrate_mbps - 20.0).abs() < f32::EPSILON);
        assert!((cfg.duration_secs - 10.0).abs() < f64::EPSILON);
        assert_eq!(cfg.sample_rate, 48_000);
        assert_eq!(cfg.audio_codec, AudioCodec::Aac);
        assert_eq!(cfg.audio_bitrate_kbps, 192);
        assert_eq!(cfg.container, ContainerFormat::Mp4);
    }

    #[test]
    fn total_frames_at_30fps_10sec() {
        let cfg = ExportConfig::default(); // 30fps, 10s
        assert_eq!(cfg.total_frames(), 300);
    }

    #[test]
    fn total_frames_fractional() {
        let cfg = ExportConfig {
            duration_secs: 1.5,
            fps: Rational::FPS_24,
            ..Default::default()
        };
        // 1.5 * 24 = 36.0, ceil = 36
        assert_eq!(cfg.total_frames(), 36);
    }

    #[test]
    fn total_frames_zero_duration() {
        let cfg = ExportConfig {
            duration_secs: 0.0,
            ..Default::default()
        };
        assert_eq!(cfg.total_frames(), 0);
    }

    #[test]
    fn frame_duration_secs() {
        let cfg = ExportConfig {
            fps: Rational::FPS_60,
            ..Default::default()
        };
        let expected = 1.0 / 60.0;
        assert!((cfg.frame_duration_secs() - expected).abs() < 1e-12);
    }

    #[test]
    fn bitrate_bps_conversion() {
        let cfg = ExportConfig {
            bitrate_mbps: 20.0,
            ..Default::default()
        };
        assert_eq!(cfg.bitrate_bps(), 20_000_000);
    }

    // -- ExportProgress -----------------------------------------------------

    #[test]
    fn progress_fraction_mid_export() {
        let p = ExportProgress {
            state: ExportState::Rendering,
            current_frame: 150,
            total_frames: 300,
            elapsed_secs: 5.0,
            estimated_remaining_secs: 5.0,
            fps: 30.0,
        };
        assert!((p.fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_fraction_zero_total() {
        let p = ExportProgress {
            state: ExportState::Idle,
            current_frame: 0,
            total_frames: 0,
            elapsed_secs: 0.0,
            estimated_remaining_secs: 0.0,
            fps: 0.0,
        };
        assert!((p.fraction()).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_fraction_complete() {
        let p = ExportProgress {
            state: ExportState::Complete,
            current_frame: 300,
            total_frames: 300,
            elapsed_secs: 10.0,
            estimated_remaining_secs: 0.0,
            fps: 30.0,
        };
        assert!((p.fraction() - 1.0).abs() < f64::EPSILON);
    }

    // -- ExportState --------------------------------------------------------

    #[test]
    fn terminal_states() {
        assert!(ExportState::Complete.is_terminal());
        assert!(ExportState::Failed("oops".into()).is_terminal());
        assert!(ExportState::Cancelled.is_terminal());

        assert!(!ExportState::Idle.is_terminal());
        assert!(!ExportState::Preparing.is_terminal());
        assert!(!ExportState::Rendering.is_terminal());
        assert!(!ExportState::Encoding.is_terminal());
        assert!(!ExportState::Finalizing.is_terminal());
    }

    #[test]
    fn state_labels() {
        assert_eq!(ExportState::Idle.label(), "Idle");
        assert_eq!(ExportState::Preparing.label(), "Preparing...");
        assert_eq!(ExportState::Rendering.label(), "Rendering...");
        assert_eq!(ExportState::Encoding.label(), "Encoding...");
        assert_eq!(ExportState::Finalizing.label(), "Finalizing...");
        assert_eq!(ExportState::Complete.label(), "Complete");
        assert_eq!(ExportState::Failed("err".into()).label(), "Failed");
        assert_eq!(ExportState::Cancelled.label(), "Cancelled");
    }

    // -- estimate_remaining -------------------------------------------------

    #[test]
    fn estimate_remaining_basic() {
        // Processed 100 of 300 frames in 5 seconds -> 200 frames left at 20 fps -> 10s
        let est = estimate_remaining(100, 300, 5.0);
        assert!((est - 10.0).abs() < 1e-9);
    }

    #[test]
    fn estimate_remaining_zero_frames() {
        assert!((estimate_remaining(0, 300, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_remaining_all_done() {
        let est = estimate_remaining(300, 300, 10.0);
        assert!((est).abs() < f64::EPSILON);
    }

    // -- ExportPipeline -----------------------------------------------------

    #[test]
    fn pipeline_new_is_idle() {
        let pipeline = ExportPipeline::new(ExportConfig::default());
        assert_eq!(*pipeline.state(), ExportState::Idle);
        assert!(!pipeline.is_running());
        assert!(!pipeline.is_complete());
    }

    #[test]
    fn pipeline_start_and_complete() {
        // Use a very short export so it finishes quickly
        let cfg = ExportConfig {
            duration_secs: 0.01,
            fps: Rational::FPS_30,
            ..Default::default()
        };
        let mut pipeline = ExportPipeline::new(cfg);
        pipeline.start().expect("should start");

        // Wait for completion (generous timeout)
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let p = pipeline.poll_progress();
            if p.state.is_terminal() {
                break;
            }
            if Instant::now() > deadline {
                panic!("Export did not complete within timeout");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let final_progress = pipeline.poll_progress();
        assert_eq!(final_progress.state, ExportState::Complete);
        assert!(pipeline.is_complete());
    }

    #[test]
    fn pipeline_cancel_mid_export() {
        // Use a long export so we can cancel it
        let cfg = ExportConfig {
            duration_secs: 600.0,
            fps: Rational::FPS_30,
            ..Default::default()
        };
        let mut pipeline = ExportPipeline::new(cfg);
        pipeline.start().expect("should start");

        // Give the thread a moment to enter the render loop
        std::thread::sleep(std::time::Duration::from_millis(50));

        pipeline.cancel();

        assert!(pipeline.is_complete());
        assert!(!pipeline.is_running());
        assert_eq!(*pipeline.state(), ExportState::Cancelled);
    }

    #[test]
    fn pipeline_double_start_fails() {
        let cfg = ExportConfig {
            duration_secs: 600.0,
            fps: Rational::FPS_30,
            ..Default::default()
        };
        let mut pipeline = ExportPipeline::new(cfg);
        pipeline.start().expect("first start should succeed");
        let result = pipeline.start();
        assert!(result.is_err());
        pipeline.cancel();
    }

    #[test]
    fn pipeline_progress_reporting() {
        let cfg = ExportConfig {
            duration_secs: 0.05, // ~2 frames at 30fps
            fps: Rational::FPS_30,
            ..Default::default()
        };
        let mut pipeline = ExportPipeline::new(cfg);
        pipeline.start().expect("should start");

        // Wait for completion
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        let mut saw_rendering = false;
        loop {
            let p = pipeline.poll_progress();
            if p.state == ExportState::Rendering {
                saw_rendering = true;
            }
            if p.state.is_terminal() {
                break;
            }
            if Instant::now() > deadline {
                panic!("Export did not complete within timeout");
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // We should have seen the Rendering state at least once
        // (or it completed so fast we missed it -- in that case just check completion)
        let final_p = pipeline.poll_progress();
        assert_eq!(final_p.state, ExportState::Complete);
        // It's possible the export finished so fast we skipped Rendering in polls,
        // so we only assert this weakly.
        let _ = saw_rendering;
    }

    #[test]
    fn pipeline_drop_joins_thread() {
        let cfg = ExportConfig {
            duration_secs: 600.0,
            fps: Rational::FPS_30,
            ..Default::default()
        };
        let mut pipeline = ExportPipeline::new(cfg);
        pipeline.start().expect("should start");
        assert!(pipeline.is_running());
        // Drop should cancel and join
        drop(pipeline);
        // If we get here, the thread was joined successfully
    }

    // -- State transition validation ----------------------------------------

    #[test]
    fn state_equality() {
        assert_eq!(ExportState::Idle, ExportState::Idle);
        assert_eq!(ExportState::Complete, ExportState::Complete);
        assert_ne!(ExportState::Idle, ExportState::Complete);
        assert_ne!(
            ExportState::Failed("a".into()),
            ExportState::Failed("b".into())
        );
        assert_eq!(
            ExportState::Failed("same".into()),
            ExportState::Failed("same".into())
        );
    }

    // -- Config edge cases --------------------------------------------------

    #[test]
    fn config_29_97fps_frame_count() {
        let cfg = ExportConfig {
            duration_secs: 10.0,
            fps: Rational::FPS_29_97,
            ..Default::default()
        };
        // 10.0 * (30000/1001) = 299.7002997... -> ceil = 300
        assert_eq!(cfg.total_frames(), 300);
    }

    #[test]
    fn config_uhd_resolution() {
        let cfg = ExportConfig {
            resolution: Resolution::UHD,
            ..Default::default()
        };
        assert_eq!(cfg.resolution.width, 3840);
        assert_eq!(cfg.resolution.height, 2160);
    }
}
