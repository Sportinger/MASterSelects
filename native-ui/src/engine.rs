//! Engine Orchestrator — coordinates GPU backends, decoders, and preview output.
//!
//! Architecture:
//!
//! ```text
//! Main Thread (egui)          Decode Thread
//! ┌─────────────┐            ┌──────────────┐
//! │ update()    │◄── frame ──│ decode loop  │
//! │  - poll rx  │   channel  │  - demux     │
//! │  - upload   │            │  - decode    │
//! │  - display  │            │  - convert   │
//! └─────────────┘            └──────────────┘
//! ```
//!
//! The decode thread feeds decoded RGBA frames through a bounded crossbeam
//! channel. The main thread polls for new frames without blocking. When no
//! real video pipeline is available (Phase 0), a test pattern generator
//! serves as fallback.

use crate::bridge::PreviewBridge;
use anyhow::{Context, Result};
use crossbeam::channel::{self, Receiver, Sender, TryRecvError};
use ms_common::kernel::{KernelArgs, KernelId};
use ms_common::{HwDecoder, Rational, Resolution, VideoCodec};
use ms_decoder::nvdec::{NvDecoder, NvcuvidLibrary};
use ms_decoder::software::nv12_to_rgba;
use ms_demux::mkv::MkvDemuxer;
use ms_demux::mp4::Mp4Demuxer;
use ms_demux::probe::detect_format;
use ms_demux::traits::Demuxer;
use ms_gpu_hal::cuda::kernel::KernelManager;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------

/// Current state of the engine's playback pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineState {
    /// No file loaded, showing test pattern or black.
    Idle,
    /// A file is being opened and the decode pipeline is initializing.
    Loading,
    /// Playing back frames in real time.
    Playing,
    /// Paused on a specific frame (still displays the last decoded frame).
    Paused,
    /// An error occurred; holds a human-readable description.
    Error(String),
}

impl EngineState {
    /// Returns a short label for display in the UI.
    pub fn label(&self) -> &str {
        match self {
            Self::Idle => "Idle",
            Self::Loading => "Loading...",
            Self::Playing => "Playing",
            Self::Paused => "Paused",
            Self::Error(_) => "Error",
        }
    }
}

// ---------------------------------------------------------------------------
// File info (metadata extracted from the opened file)
// ---------------------------------------------------------------------------

/// Metadata about the currently loaded media file.
#[derive(Clone, Debug)]
pub struct FileInfo {
    pub path: PathBuf,
    pub file_name: String,
    pub resolution: Resolution,
    pub fps: Rational,
    pub duration_secs: f64,
    pub codec: VideoCodec,
}

// ---------------------------------------------------------------------------
// Decoded frame message — sent from decode thread to main thread
// ---------------------------------------------------------------------------

/// A decoded RGBA frame ready for display.
struct DecodedFrame {
    /// RGBA8 pixel data (width * height * 4 bytes).
    rgba_data: Vec<u8>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Presentation timestamp in seconds.
    pts_secs: f64,
}

/// Commands sent from the main thread to the decode thread.
enum DecodeCommand {
    /// Start or resume playback.
    Play,
    /// Pause playback.
    Pause,
    /// Seek to a specific time (seconds).
    Seek(f64),
    /// Stop and shut down the decode thread.
    Stop,
}

/// GPU info message sent once from the decode thread to the main thread.
struct GpuInfoMsg {
    /// GPU device name (e.g. "NVIDIA GeForce RTX 5080").
    gpu_name: String,
    /// Whether NVDEC hardware decode is active.
    nvdec_active: bool,
}

// ---------------------------------------------------------------------------
// Engine Orchestrator
// ---------------------------------------------------------------------------

/// Engine orchestrator that drives the render pipeline.
///
/// In Phase 0 this generates animated test patterns when no file is loaded.
/// When a file is opened, a decode thread is spawned that feeds RGBA frames
/// through a crossbeam channel. The `update()` method on the main thread
/// polls for new frames without blocking.
pub struct EngineOrchestrator {
    /// Current engine state.
    state: EngineState,

    /// Time at which the engine was created, used for test pattern animation.
    start_time: Instant,

    /// Default preview width (used when no file is loaded).
    preview_width: u32,
    /// Default preview height (used when no file is loaded).
    preview_height: u32,

    /// Metadata about the currently loaded file (None when idle).
    file_info: Option<FileInfo>,

    // -- Playback timing --
    /// Current playback position in seconds.
    current_time_secs: f64,
    /// Instant when playback started (used for wall-clock sync).
    playback_start_instant: Option<Instant>,
    /// Playback time at the moment play was pressed (for wall-clock offset).
    playback_start_time_secs: f64,

    // -- Decode thread communication --
    /// Receiver for decoded frames from the decode thread.
    frame_rx: Option<Receiver<DecodedFrame>>,
    /// Sender for commands to the decode thread.
    cmd_tx: Option<Sender<DecodeCommand>>,
    /// Handle to the decode thread (for join on stop).
    decode_thread: Option<thread::JoinHandle<()>>,

    // -- Last frame cache (for pause / repeat display) --
    /// The most recently displayed RGBA frame data.
    last_frame: Option<Vec<u8>>,
    /// Width of the last frame.
    last_frame_width: u32,
    /// Height of the last frame.
    last_frame_height: u32,

    // -- GPU pipeline info --
    /// GPU device name (populated when decode pipeline initializes).
    gpu_info: Option<String>,
    /// Whether GPU hardware decode is active in the decode thread.
    gpu_decode_active: bool,
    /// One-shot receiver for GPU info from the decode thread.
    gpu_info_rx: Option<Receiver<GpuInfoMsg>>,
}

impl EngineOrchestrator {
    /// Create a new engine orchestrator in the idle state.
    pub fn new() -> Self {
        Self {
            state: EngineState::Idle,
            start_time: Instant::now(),
            preview_width: 1920,
            preview_height: 1080,
            file_info: None,
            current_time_secs: 0.0,
            playback_start_instant: None,
            playback_start_time_secs: 0.0,
            frame_rx: None,
            cmd_tx: None,
            decode_thread: None,
            last_frame: None,
            last_frame_width: 0,
            last_frame_height: 0,
            gpu_info: None,
            gpu_decode_active: false,
            gpu_info_rx: None,
        }
    }

    // -----------------------------------------------------------------------
    // Public accessors
    // -----------------------------------------------------------------------

    /// Current engine state.
    pub fn state(&self) -> &EngineState {
        &self.state
    }

    /// Metadata about the loaded file, if any.
    pub fn file_info(&self) -> Option<&FileInfo> {
        self.file_info.as_ref()
    }

    /// Current playback time in seconds.
    pub fn current_time_secs(&self) -> f64 {
        self.current_time_secs
    }

    /// Duration of the loaded file in seconds (0.0 if no file is loaded).
    pub fn duration_secs(&self) -> f64 {
        self.file_info
            .as_ref()
            .map_or(0.0, |info| info.duration_secs)
    }

    /// Human-readable GPU device name.
    ///
    /// Returns "GPU: detecting..." before the first file is opened,
    /// or the detected device name / "None" afterward.
    pub fn gpu_name(&self) -> &str {
        match &self.gpu_info {
            Some(name) => name,
            None => "GPU: detecting...",
        }
    }

    /// Whether the decode thread is using GPU hardware decode.
    pub fn gpu_decode_active(&self) -> bool {
        self.gpu_decode_active
    }

    /// Update GPU info from the decode thread.
    pub fn set_gpu_info(&mut self, name: String, decode_active: bool) {
        self.gpu_info = Some(name);
        self.gpu_decode_active = decode_active;
    }

    // -----------------------------------------------------------------------
    // File open
    // -----------------------------------------------------------------------

    /// Open a media file and start the decode pipeline.
    ///
    /// This transitions the engine to `Loading`, spawns a decode thread,
    /// and begins feeding frames through the channel. When the first frame
    /// arrives, the engine transitions to `Paused` (waiting for the user
    /// to press play).
    ///
    /// The method probes the file using the `ms-demux` crate to extract
    /// real metadata (resolution, fps, duration, codec). If probing fails
    /// (e.g. the file doesn't exist or isn't a valid MP4), it falls back
    /// to sensible defaults so the pipeline can still be exercised with
    /// synthetic frames.
    pub fn open_file(&mut self, path: PathBuf) -> Result<()> {
        // Stop any existing pipeline first
        self.stop_pipeline();

        self.state = EngineState::Loading;
        self.current_time_secs = 0.0;
        self.playback_start_instant = None;
        self.last_frame = None;

        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Try to probe the file with the real demuxer for metadata.
        // If it fails (file missing, not a valid MP4, etc.), fall back to
        // defaults so the pipeline can still be exercised with synthetic frames.
        let info = match probe_file_info(&path, &file_name) {
            Ok(fi) => {
                tracing::info!(
                    "Engine: probed '{}' -> {}x{} @ {} fps, {:.2}s, {:?}",
                    fi.file_name,
                    fi.resolution.width,
                    fi.resolution.height,
                    fi.fps,
                    fi.duration_secs,
                    fi.codec,
                );
                fi
            }
            Err(e) => {
                tracing::warn!(
                    "Engine: could not probe '{}': {}. Using defaults.",
                    file_name,
                    e,
                );
                FileInfo {
                    path: path.clone(),
                    file_name,
                    resolution: Resolution::HD,
                    fps: Rational::FPS_30,
                    duration_secs: 10.0,
                    codec: VideoCodec::H264,
                }
            }
        };
        self.file_info = Some(info.clone());

        // Create bounded channels:
        // - frame channel: small ring buffer (4 frames max to limit memory)
        // - command channel: unbounded (commands are tiny)
        // - gpu info channel: one-shot (decode thread sends GPU info once)
        let (frame_tx, frame_rx) = channel::bounded::<DecodedFrame>(4);
        let (cmd_tx, cmd_rx) = channel::unbounded::<DecodeCommand>();
        let (gpu_info_tx, gpu_info_rx) = channel::bounded::<GpuInfoMsg>(1);

        self.frame_rx = Some(frame_rx);
        self.cmd_tx = Some(cmd_tx);
        self.gpu_info_rx = Some(gpu_info_rx);

        // Spawn decode thread
        let thread_info = info;
        let handle = thread::Builder::new()
            .name("decode-worker".to_string())
            .spawn(move || {
                decode_thread_main(thread_info, frame_tx, cmd_rx, gpu_info_tx);
            })
            .context("Failed to spawn decode thread")?;

        self.decode_thread = Some(handle);

        // Transition to Paused (will show the first frame when it arrives)
        self.state = EngineState::Paused;
        tracing::info!("Engine: opened file {:?}", path);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Playback controls
    // -----------------------------------------------------------------------

    /// Start or resume playback.
    pub fn play(&mut self) {
        if self.state == EngineState::Paused || self.state == EngineState::Idle {
            self.state = EngineState::Playing;
            self.playback_start_instant = Some(Instant::now());
            self.playback_start_time_secs = self.current_time_secs;

            if let Some(tx) = &self.cmd_tx {
                let _ = tx.send(DecodeCommand::Play);
            }

            tracing::debug!("Engine: play (from {:.2}s)", self.current_time_secs);
        }
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == EngineState::Playing {
            self.state = EngineState::Paused;
            self.playback_start_instant = None;

            if let Some(tx) = &self.cmd_tx {
                let _ = tx.send(DecodeCommand::Pause);
            }

            tracing::debug!("Engine: pause at {:.2}s", self.current_time_secs);
        }
    }

    /// Toggle play/pause.
    pub fn toggle_play_pause(&mut self) {
        match self.state {
            EngineState::Playing => self.pause(),
            EngineState::Paused | EngineState::Idle => self.play(),
            _ => {}
        }
    }

    /// Seek to a specific time in seconds.
    pub fn seek(&mut self, time_secs: f64) {
        let duration = self.duration_secs();
        self.current_time_secs = time_secs.clamp(0.0, duration);

        if self.state == EngineState::Playing {
            self.playback_start_instant = Some(Instant::now());
            self.playback_start_time_secs = self.current_time_secs;
        }

        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(DecodeCommand::Seek(self.current_time_secs));
        }

        tracing::debug!("Engine: seek to {:.2}s", self.current_time_secs);
    }

    /// Stop playback and close the current file.
    pub fn stop(&mut self) {
        tracing::info!("Engine: stop");
        self.stop_pipeline();
        self.state = EngineState::Idle;
        self.file_info = None;
        self.current_time_secs = 0.0;
        self.playback_start_instant = None;
        self.last_frame = None;
    }

    // -----------------------------------------------------------------------
    // Main update — called every frame from the egui event loop
    // -----------------------------------------------------------------------

    /// Pump one frame to the preview bridge.
    ///
    /// Called once per egui frame. This method never blocks. It polls the
    /// decode channel for new frames and uploads the result to the bridge.
    pub fn update(&mut self, ctx: &egui::Context, bridge: &mut PreviewBridge) {
        // Poll for GPU info from the decode thread (arrives once after init)
        if let Some(rx) = &self.gpu_info_rx {
            if let Ok(info) = rx.try_recv() {
                tracing::info!(
                    gpu = %info.gpu_name,
                    nvdec = info.nvdec_active,
                    "GPU info received from decode thread"
                );
                self.gpu_info = Some(info.gpu_name);
                self.gpu_decode_active = info.nvdec_active;
                self.gpu_info_rx = None; // One-shot: drop after receiving
            }
        }

        match &self.state {
            EngineState::Playing => {
                self.update_playback_time();
                self.poll_and_display(ctx, bridge);
                // Request continuous repaint for smooth playback
                ctx.request_repaint();
            }
            EngineState::Paused | EngineState::Loading => {
                // In Paused/Loading, still poll for frames (the first frame
                // after open, or frames that were in-flight before pause)
                self.poll_and_display(ctx, bridge);
            }
            EngineState::Idle => {
                // No file loaded: show the test pattern
                let frame = self.generate_test_frame(self.preview_width, self.preview_height);
                bridge.update_from_rgba_bytes(ctx, &frame, self.preview_width, self.preview_height);
                ctx.request_repaint();
            }
            EngineState::Error(msg) => {
                // Show an error pattern (red-tinted test pattern)
                let frame = self.generate_error_frame(self.preview_width, self.preview_height, msg);
                bridge.update_from_rgba_bytes(ctx, &frame, self.preview_width, self.preview_height);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Update playback time based on wall clock.
    fn update_playback_time(&mut self) {
        if let Some(start_instant) = self.playback_start_instant {
            let elapsed = start_instant.elapsed().as_secs_f64();
            self.current_time_secs = self.playback_start_time_secs + elapsed;

            // Check if we've reached the end of the file
            let duration = self.duration_secs();
            if duration > 0.0 && self.current_time_secs >= duration {
                self.current_time_secs = duration;
                self.pause();
            }
        }
    }

    /// Poll the frame channel and display the most recent frame.
    fn poll_and_display(&mut self, ctx: &egui::Context, bridge: &mut PreviewBridge) {
        let mut newest_frame: Option<DecodedFrame> = None;

        // Drain all available frames, keeping only the newest.
        // This ensures we don't accumulate a backlog when the display
        // is slower than the decode rate.
        if let Some(rx) = &self.frame_rx {
            loop {
                match rx.try_recv() {
                    Ok(frame) => {
                        newest_frame = Some(frame);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        tracing::warn!("Engine: decode channel disconnected");
                        self.frame_rx = None;
                        break;
                    }
                }
            }
        }

        if let Some(frame) = newest_frame {
            // Cache this frame for paused redisplay
            self.last_frame = Some(frame.rgba_data.clone());
            self.last_frame_width = frame.width;
            self.last_frame_height = frame.height;

            bridge.update_from_rgba_bytes(ctx, &frame.rgba_data, frame.width, frame.height);
        } else if let Some(ref last) = self.last_frame {
            // No new frame available; redisplay the cached frame
            bridge.update_from_rgba_bytes(ctx, last, self.last_frame_width, self.last_frame_height);
        } else {
            // No frames at all yet; show black
            let w = self.preview_width;
            let h = self.preview_height;
            let black = vec![0u8; w as usize * h as usize * 4];
            bridge.update_from_rgba_bytes(ctx, &black, w, h);
        }
    }

    /// Shut down the decode pipeline (stop thread, drop channels).
    fn stop_pipeline(&mut self) {
        // Send stop command
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(DecodeCommand::Stop);
        }

        // Drop the frame receiver so the decode thread's send will fail
        // if it's blocked on a full channel.
        self.frame_rx = None;

        // Join the decode thread (with a timeout to avoid hanging)
        if let Some(handle) = self.decode_thread.take() {
            // Give the thread a moment to shut down cleanly
            let _ = handle.join();
        }
    }

    // -----------------------------------------------------------------------
    // Test pattern generators (fallback when no file is loaded)
    // -----------------------------------------------------------------------

    /// Generate a colorful RGBA test pattern to verify the preview pipeline.
    ///
    /// The pattern includes:
    /// - Horizontal color gradient (red -> green -> blue)
    /// - Vertical brightness gradient
    /// - Animated diagonal stripe overlay (moves over time)
    /// - A centered crosshair to verify alignment
    /// - SMPTE-inspired color bars at the bottom
    pub fn generate_test_frame(&self, width: u32, height: u32) -> Vec<u8> {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let w = width as usize;
        let h = height as usize;
        let mut pixels = vec![0u8; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let offset = (y * w + x) * 4;

                // Normalized coordinates [0, 1]
                let nx = x as f32 / w as f32;
                let ny = y as f32 / h as f32;

                // Base color: horizontal hue gradient
                let hue = nx * 360.0;
                let (r_base, g_base, b_base) = hsv_to_rgb(hue, 0.7, 0.8);

                // Vertical brightness modulation
                let brightness = 0.3 + 0.7 * (1.0 - ny);

                // Animated diagonal stripe overlay
                let stripe_phase = (nx + ny) * 20.0 - elapsed * 2.0;
                let stripe = (stripe_phase.sin() * 0.5 + 0.5) * 0.3 + 0.7;

                // Checkerboard pattern in the center region
                let checker = if nx > 0.35 && nx < 0.65 && ny > 0.35 && ny < 0.65 {
                    let cx = (x as f32 / 32.0).floor() as i32;
                    let cy = (y as f32 / 32.0).floor() as i32;
                    if (cx + cy) % 2 == 0 {
                        0.9_f32
                    } else {
                        0.6
                    }
                } else {
                    1.0
                };

                let mut r = r_base * brightness * stripe * checker;
                let mut g = g_base * brightness * stripe * checker;
                let mut b = b_base * brightness * stripe * checker;

                // Crosshair lines (centered)
                let center_x = w / 2;
                let center_y = h / 2;
                let is_h_line = y == center_y || y == center_y + 1;
                let is_v_line = x == center_x || x == center_x + 1;
                if is_h_line || is_v_line {
                    r = 1.0;
                    g = 1.0;
                    b = 1.0;
                }

                // Color bars at the bottom (SMPTE-inspired, bottom 10%)
                if ny > 0.9 {
                    let bar_idx = (nx * 8.0).floor() as usize;
                    let (br, bg, bb) = match bar_idx {
                        0 => (1.0, 1.0, 1.0), // White
                        1 => (1.0, 1.0, 0.0), // Yellow
                        2 => (0.0, 1.0, 1.0), // Cyan
                        3 => (0.0, 1.0, 0.0), // Green
                        4 => (1.0, 0.0, 1.0), // Magenta
                        5 => (1.0, 0.0, 0.0), // Red
                        6 => (0.0, 0.0, 1.0), // Blue
                        _ => (0.0, 0.0, 0.0), // Black
                    };
                    r = br;
                    g = bg;
                    b = bb;
                }

                pixels[offset] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[offset + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[offset + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[offset + 3] = 255; // Fully opaque
            }
        }

        pixels
    }

    /// Generate a red-tinted error frame with a message baked in.
    ///
    /// This is a simple visual indicator that something went wrong.
    fn generate_error_frame(&self, width: u32, height: u32, _msg: &str) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut pixels = vec![0u8; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let offset = (y * w + x) * 4;
                let nx = x as f32 / w as f32;
                let ny = y as f32 / h as f32;

                // Red-tinted gradient with diagonal warning stripes
                let stripe = ((nx + ny) * 30.0).sin();
                let base_r = if stripe > 0.0 { 0.6_f32 } else { 0.3 };
                let base_g = if stripe > 0.0 { 0.1_f32 } else { 0.05 };
                let base_b = if stripe > 0.0 { 0.1_f32 } else { 0.05 };

                pixels[offset] = (base_r * 255.0) as u8;
                pixels[offset + 1] = (base_g * 255.0) as u8;
                pixels[offset + 2] = (base_b * 255.0) as u8;
                pixels[offset + 3] = 255;
            }
        }

        pixels
    }
}

impl Drop for EngineOrchestrator {
    fn drop(&mut self) {
        self.stop_pipeline();
    }
}

// ---------------------------------------------------------------------------
// File probing — extract metadata via the real demuxer
// ---------------------------------------------------------------------------

/// Probe a media file and return metadata using the real `ms-demux` crate.
///
/// Supports MP4/MOV/M4V and MKV/WebM containers.
fn probe_file_info(path: &Path, file_name: &str) -> Result<FileInfo> {
    // Detect container format from extension
    let format = detect_format(path).map_err(|e| anyhow::anyhow!("Unsupported format: {}", e))?;

    match format {
        ms_common::ContainerFormat::Mp4 => {
            let demuxer =
                Mp4Demuxer::open(path).map_err(|e| anyhow::anyhow!("Failed to open MP4: {}", e))?;

            let container = demuxer.probe();
            let video = container
                .video_streams
                .first()
                .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

            Ok(FileInfo {
                path: path.to_path_buf(),
                file_name: file_name.to_string(),
                resolution: video.resolution,
                fps: video.fps,
                duration_secs: video.duration.0,
                codec: video.codec,
            })
        }
        ms_common::ContainerFormat::Mkv | ms_common::ContainerFormat::WebM => {
            let demuxer = MkvDemuxer::open(path)
                .map_err(|e| anyhow::anyhow!("Failed to open MKV: {}", e))?;

            let container = demuxer.probe();
            let video = container
                .video_streams
                .first()
                .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

            Ok(FileInfo {
                path: path.to_path_buf(),
                file_name: file_name.to_string(),
                resolution: video.resolution,
                fps: video.fps,
                duration_secs: video.duration.0,
                codec: video.codec,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Decode thread — runs on a separate OS thread
// ---------------------------------------------------------------------------

/// Main loop for the decode thread.
///
/// Attempts to initialize NVDEC hardware decode and open a real demuxer.
/// Falls back gracefully through several levels:
/// 1. NVDEC + real demuxer (full hardware decode pipeline)
/// 2. Real demuxer + synthetic frames (timing from container, no decode)
/// 3. Fully synthetic frames (fallback for non-video files)
fn decode_thread_main(
    info: FileInfo,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecodeCommand>,
    gpu_info_tx: Sender<GpuInfoMsg>,
) {
    tracing::info!(
        "Decode thread started for '{}' ({}x{} @ {} fps)",
        info.file_name,
        info.resolution.width,
        info.resolution.height,
        info.fps,
    );

    // Try to initialize CUDA context + NVDEC on this thread
    let nvdec = try_init_nvdec(info.codec);

    match nvdec {
        Ok(init) => {
            // Report GPU info back to main thread
            let _ = gpu_info_tx.send(GpuInfoMsg {
                gpu_name: init.gpu_name.clone(),
                nvdec_active: true,
            });

            // Try to open real demuxer
            match try_open_demuxer(&info.path) {
                Ok(mut demuxer) => {
                    tracing::info!(
                        "NVDEC decode active on {} for '{}' (GPU kernel: {})",
                        init.gpu_name,
                        info.file_name,
                        if init.kernel_mgr.is_some() {
                            "enabled"
                        } else {
                            "disabled, using CPU fallback"
                        },
                    );
                    nvdec_decode_loop(
                        &info,
                        &mut demuxer,
                        init.decoder,
                        &init.cuda_ctx,
                        init.kernel_mgr.as_ref(),
                        &frame_tx,
                        &cmd_rx,
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Could not open demuxer for '{}': {}. Using synthetic frames.",
                        info.file_name,
                        e,
                    );
                    synthetic_decode_loop(&info, &frame_tx, &cmd_rx);
                }
            }
        }
        Err(e) => {
            tracing::warn!("NVDEC not available: {}. Using software path.", e);

            // Report no GPU decode
            let _ = gpu_info_tx.send(GpuInfoMsg {
                gpu_name: "None (software)".to_string(),
                nvdec_active: false,
            });

            // Fall back to demuxer + synthetic frames
            match try_open_demuxer(&info.path) {
                Ok(mut demuxer) => {
                    tracing::info!(
                        "Decode thread: real demuxer opened for '{}' (software path)",
                        info.file_name,
                    );
                    real_decode_loop(&info, &mut demuxer, &frame_tx, &cmd_rx);
                }
                Err(e) => {
                    tracing::warn!(
                        "Could not open demuxer for '{}': {}. Using synthetic frames.",
                        info.file_name,
                        e,
                    );
                    synthetic_decode_loop(&info, &frame_tx, &cmd_rx);
                }
            }
        }
    }
}

/// Try to open the appropriate demuxer for a given file path.
///
/// Supports MP4 and MKV/WebM. Returns the demuxer as a boxed trait
/// object so the decode loop can work generically.
fn try_open_demuxer(path: &Path) -> Result<Box<dyn Demuxer>> {
    let format = detect_format(path).map_err(|e| anyhow::anyhow!("Unsupported format: {}", e))?;

    match format {
        ms_common::ContainerFormat::Mp4 => {
            let demuxer =
                Mp4Demuxer::open(path).map_err(|e| anyhow::anyhow!("Failed to open MP4: {}", e))?;
            Ok(Box::new(demuxer))
        }
        ms_common::ContainerFormat::Mkv | ms_common::ContainerFormat::WebM => {
            let demuxer = MkvDemuxer::open(path)
                .map_err(|e| anyhow::anyhow!("Failed to open MKV: {}", e))?;
            Ok(Box::new(demuxer))
        }
    }
}

// ---------------------------------------------------------------------------
// NVDEC + CUDA kernel initialization
// ---------------------------------------------------------------------------

/// Result of NVDEC initialization.
///
/// Contains everything the decode thread needs to run the GPU decode pipeline.
struct NvdecInitResult {
    cuda_ctx: Arc<cudarc::driver::safe::CudaContext>,
    decoder: NvDecoder,
    gpu_name: String,
    /// Kernel manager with the NV12->RGBA PTX loaded (None if kernel loading failed).
    kernel_mgr: Option<KernelManager>,
}

/// Try to initialize a CUDA context and NVDEC decoder on the current thread.
///
/// Returns the CUDA context, an NvDecoder, the GPU device name, and optionally
/// a KernelManager with the NV12->RGBA PTX kernel loaded. If the kernel fails
/// to load, the decode loop will fall back to the CPU conversion path.
fn try_init_nvdec(codec: VideoCodec) -> Result<NvdecInitResult> {
    // 1. Create CUDA context on device 0
    let cuda_ctx = cudarc::driver::safe::CudaContext::new(0)
        .map_err(|e| anyhow::anyhow!("CUDA context init failed: {e}"))?;

    // Bind to this thread so NVDEC can use it
    cuda_ctx
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("CUDA bind_to_thread failed: {e}"))?;

    let gpu_name = cuda_ctx
        .name()
        .unwrap_or_else(|_| "Unknown NVIDIA GPU".to_string());

    tracing::info!("CUDA context initialized on '{gpu_name}'");

    // 2. Load NVDEC library (nvcuvid.dll)
    let nvdec_lib = Arc::new(
        NvcuvidLibrary::load()
            .map_err(|e| anyhow::anyhow!("Failed to load nvcuvid: {e}"))?,
    );

    tracing::info!("NVDEC library loaded successfully");

    // 3. Create decoder for the detected codec
    let decoder = NvDecoder::new(nvdec_lib, codec)
        .map_err(|e| anyhow::anyhow!("NvDecoder creation failed: {e}"))?;

    tracing::info!(codec = codec.display_name(), "NvDecoder created");

    // 4. Try to load the NV12->RGBA CUDA kernel for GPU color conversion.
    //    If this fails we fall back to CPU conversion (the Phase 0 path).
    let kernel_mgr = try_load_nv12_kernel(&cuda_ctx);

    Ok(NvdecInitResult {
        cuda_ctx,
        decoder,
        gpu_name,
        kernel_mgr,
    })
}

/// Try to create a KernelManager and load the NV12->RGBA PTX kernel.
///
/// Uses the embedded PTX bytecode from ms-gpu-hal (compiled at build time
/// by gpu-hal's build.rs from `kernels/cuda/nv12_to_rgba.cu`).
///
/// Returns `Some(KernelManager)` on success, `None` if the PTX is not
/// available or fails to load (in which case the CPU fallback will be used).
fn try_load_nv12_kernel(
    cuda_ctx: &Arc<cudarc::driver::safe::CudaContext>,
) -> Option<KernelManager> {
    // Get the embedded PTX bytecode compiled from nv12_to_rgba.cu
    let ptx_bytes = match ms_gpu_hal::kernels::get_ptx("nv12_to_rgba") {
        Some(bytes) => bytes,
        None => {
            tracing::warn!(
                "NV12->RGBA PTX not available (nvcc was not found at build time). \
                 Falling back to CPU color conversion."
            );
            return None;
        }
    };

    let km = KernelManager::new(cuda_ctx.clone());

    // Load the PTX module. The module name must match KernelId::Nv12ToRgba.cuda_module_name()
    // which returns "nv12_to_rgba.ptx".
    let module_name = KernelId::Nv12ToRgba.cuda_module_name();
    if let Err(e) = km.load_ptx_bytes(&module_name, ptx_bytes) {
        tracing::warn!(
            error = %e,
            "Failed to load NV12->RGBA PTX module. Falling back to CPU color conversion."
        );
        return None;
    }

    // Verify we can resolve the kernel function entry point
    match km.get_kernel_function(&KernelId::Nv12ToRgba) {
        Ok(_func) => {
            tracing::info!(
                "NV12->RGBA CUDA kernel loaded successfully -- GPU color conversion enabled"
            );
            Some(km)
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "Failed to resolve nv12_to_rgba kernel function. Falling back to CPU."
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// GPU RGBA output buffer (persistent, reused across frames)
// ---------------------------------------------------------------------------

/// Persistent GPU RGBA output buffer, allocated once and reused across frames.
///
/// This avoids re-allocating GPU memory for every frame. The buffer is freed
/// via `cuMemFree` on drop.
struct GpuRgbaBuffer {
    /// Device pointer to the RGBA buffer.
    device_ptr: u64,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Total byte size (width * height * 4).
    byte_size: usize,
}

impl GpuRgbaBuffer {
    /// Allocate a GPU RGBA buffer for the given frame dimensions.
    fn alloc(width: u32, height: u32) -> Result<Self> {
        let byte_size = width as usize * height as usize * 4;
        let device_ptr = unsafe {
            let mut ptr = MaybeUninit::uninit();
            // SAFETY: We are allocating device memory of the requested size.
            // The CUDA context must be bound to the current thread (guaranteed
            // by the caller in nvdec_decode_loop).
            let result = cudarc::driver::sys::cuMemAlloc_v2(ptr.as_mut_ptr(), byte_size);
            result
                .result()
                .map_err(|e| anyhow::anyhow!("cuMemAlloc for RGBA buffer failed: {e:?}"))?;
            ptr.assume_init()
        };

        tracing::info!(
            width,
            height,
            byte_size,
            device_ptr = format_args!("0x{:x}", device_ptr),
            "Allocated GPU RGBA output buffer"
        );

        Ok(Self {
            device_ptr,
            width,
            height,
            byte_size,
        })
    }

    /// Check if this buffer matches the given dimensions.
    fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }
}

impl Drop for GpuRgbaBuffer {
    fn drop(&mut self) {
        if self.device_ptr != 0 {
            // SAFETY: device_ptr was obtained from cuMemAlloc_v2 and has not
            // been freed yet. Best-effort free; ignore errors during drop.
            unsafe {
                let _ = cudarc::driver::sys::cuMemFree_v2(self.device_ptr);
            }
            tracing::debug!(
                device_ptr = format_args!("0x{:x}", self.device_ptr),
                "Freed GPU RGBA output buffer"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// NVDEC decode loop
// ---------------------------------------------------------------------------

/// NVDEC hardware decode loop -- real packets decoded on GPU.
///
/// For each video packet from the demuxer:
/// 1. Feed packet data to NVDEC (via NvDecoder)
/// 2. If a decoded NV12 frame is ready:
///    a. If GPU kernel is available: dispatch NV12->RGBA kernel on GPU,
///       then copy RGBA from GPU to CPU.
///    b. If GPU kernel is not available (fallback): copy NV12 from GPU to CPU,
///       then convert NV12->RGBA on CPU.
/// 3. Send the RGBA frame through the channel for display
fn nvdec_decode_loop(
    info: &FileInfo,
    demuxer: &mut Box<dyn Demuxer>,
    mut decoder: NvDecoder,
    cuda_ctx: &Arc<cudarc::driver::safe::CudaContext>,
    kernel_mgr: Option<&KernelManager>,
    frame_tx: &Sender<DecodedFrame>,
    cmd_rx: &Receiver<DecodeCommand>,
) {
    let fps = info.fps.as_f64();
    let frame_duration = Duration::from_secs_f64(1.0 / fps);

    let mut playing = false;
    let mut frame_num: u64 = 0;

    // Re-bind CUDA context to this thread (safety: we're on the decode thread)
    if let Err(e) = cuda_ctx.bind_to_thread() {
        tracing::error!("Failed to bind CUDA context: {e}");
        return;
    }

    // Persistent GPU RGBA output buffer (allocated on first frame, reused).
    // Only used when kernel_mgr is available.
    let mut rgba_gpu_buf: Option<GpuRgbaBuffer> = None;

    loop {
        // Check for commands (non-blocking)
        match cmd_rx.try_recv() {
            Ok(DecodeCommand::Play) => {
                playing = true;
                tracing::debug!("Decode thread (NVDEC): play");
            }
            Ok(DecodeCommand::Pause) => {
                playing = false;
                tracing::debug!("Decode thread (NVDEC): pause");
            }
            Ok(DecodeCommand::Seek(time_secs)) => {
                // Reset decoder state for seeking
                if let Err(e) = decoder.reset() {
                    tracing::warn!("NVDEC reset on seek failed: {e}");
                }
                if let Err(e) = demuxer.seek(time_secs) {
                    tracing::warn!("Demuxer seek failed: {e}");
                }
                frame_num = (time_secs * fps).round() as u64;
                tracing::debug!(
                    "Decode thread (NVDEC): seek to {:.2}s (frame ~{})",
                    time_secs,
                    frame_num,
                );
            }
            Ok(DecodeCommand::Stop) => {
                tracing::info!("Decode thread (NVDEC): stop command received");
                return;
            }
            Err(crossbeam::channel::TryRecvError::Empty) => {}
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                tracing::info!("Decode thread (NVDEC): command channel disconnected, exiting");
                return;
            }
        }

        if !playing {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Read the next real video packet from the demuxer
        match demuxer.next_video_packet() {
            Some(packet) => {
                let pts_secs = packet.pts.0;

                // Feed packet to NVDEC
                match decoder.decode(&packet) {
                    Ok(Some(gpu_frame)) => {
                        // Decoded frame available on GPU (NV12 format)
                        let width = gpu_frame.resolution.width;
                        let height = gpu_frame.resolution.height;
                        let pitch = gpu_frame.pitch;

                        let convert_result = if let Some(km) = kernel_mgr {
                            // GPU path: NV12->RGBA kernel on GPU, then copy RGBA to CPU
                            gpu_nv12_to_rgba_and_readback(km, &gpu_frame, &mut rgba_gpu_buf)
                        } else {
                            // CPU fallback: copy NV12 to CPU, then convert on CPU
                            cpu_nv12_copy_and_convert(gpu_frame.device_ptr, width, height, pitch)
                        };

                        match convert_result {
                            Ok(rgba_data) => {
                                // Release the NVDEC surface now that we have the data
                                decoder.release_frame(&gpu_frame);

                                let decoded = DecodedFrame {
                                    rgba_data,
                                    width,
                                    height,
                                    pts_secs,
                                };

                                if frame_tx.send(decoded).is_err() {
                                    tracing::info!(
                                        "Decode thread (NVDEC): frame channel closed, exiting"
                                    );
                                    return;
                                }

                                frame_num += 1;

                                if frame_num % 100 == 0 {
                                    tracing::debug!(
                                        "NVDEC: decoded {} frames, pts={:.2}s ({})",
                                        frame_num,
                                        pts_secs,
                                        if kernel_mgr.is_some() {
                                            "GPU kernel"
                                        } else {
                                            "CPU fallback"
                                        },
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::error!("NV12->RGBA conversion failed: {e}");
                                // Release surface even on error
                                decoder.release_frame(&gpu_frame);
                            }
                        }
                    }
                    Ok(None) => {
                        // No decoded frame yet (normal for B-frame reordering)
                        tracing::trace!("NVDEC: no output for packet {frame_num}");
                    }
                    Err(e) => {
                        tracing::error!("NVDEC decode error: {e}");
                    }
                }

                // Pace to target FPS
                thread::sleep(frame_duration);
            }
            None => {
                // End of stream -- flush remaining frames from decoder
                tracing::info!("Decode thread (NVDEC): end of stream, flushing decoder");
                match decoder.flush() {
                    Ok(frames) => {
                        for gpu_frame in &frames {
                            let width = gpu_frame.resolution.width;
                            let height = gpu_frame.resolution.height;
                            let pitch = gpu_frame.pitch;

                            let convert_result = if let Some(km) = kernel_mgr {
                                gpu_nv12_to_rgba_and_readback(km, gpu_frame, &mut rgba_gpu_buf)
                            } else {
                                cpu_nv12_copy_and_convert(
                                    gpu_frame.device_ptr,
                                    width,
                                    height,
                                    pitch,
                                )
                            };

                            if let Ok(rgba_data) = convert_result {
                                decoder.release_frame(gpu_frame);

                                let decoded = DecodedFrame {
                                    rgba_data,
                                    width,
                                    height,
                                    pts_secs: gpu_frame.pts.as_secs(),
                                };

                                if frame_tx.send(decoded).is_err() {
                                    return;
                                }
                                frame_num += 1;
                            } else {
                                decoder.release_frame(gpu_frame);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("NVDEC flush failed: {e}");
                    }
                }
                tracing::info!(
                    "Decode thread (NVDEC): finished after {} frames",
                    frame_num,
                );
                playing = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NV12 -> RGBA conversion (GPU kernel path + CPU fallback)
// ---------------------------------------------------------------------------

/// Convert NV12 to RGBA on GPU using the CUDA kernel, then readback RGBA to CPU.
///
/// This is the fast path: the NV12->RGBA conversion runs entirely on GPU,
/// and we only copy the final RGBA data to CPU. The NV12 data never leaves
/// the GPU.
///
/// Steps:
/// 1. Ensure the GPU RGBA output buffer is allocated (and matches frame size)
/// 2. Dispatch the nv12_to_rgba CUDA kernel (NV12 device ptrs -> RGBA device buffer)
/// 3. Synchronize (kernel runs on default stream / stream 0)
/// 4. cuMemcpyDtoH the RGBA result to a host Vec<u8>
fn gpu_nv12_to_rgba_and_readback(
    km: &KernelManager,
    gpu_frame: &ms_common::packet::GpuFrame,
    rgba_buf: &mut Option<GpuRgbaBuffer>,
) -> Result<Vec<u8>> {
    let width = gpu_frame.resolution.width;
    let height = gpu_frame.resolution.height;
    let pitch = gpu_frame.pitch;

    // 1. Ensure RGBA output buffer is allocated and matches the frame size.
    //    If the resolution changed (rare), reallocate.
    if rgba_buf.as_ref().map_or(true, |b| !b.matches(width, height)) {
        *rgba_buf = Some(
            GpuRgbaBuffer::alloc(width, height)
                .context("Failed to allocate GPU RGBA output buffer")?,
        );
    }
    let buf = rgba_buf.as_ref().expect("RGBA buffer just allocated");

    // 2. Build kernel arguments.
    //    The NV12 frame has Y plane at device_ptr and UV plane offset by height * pitch.
    let y_plane_ptr = gpu_frame.device_ptr;
    let uv_plane_ptr = gpu_frame
        .device_ptr_uv
        .unwrap_or_else(|| y_plane_ptr + height as u64 * pitch as u64);
    let out_pitch = width as i32 * 4; // RGBA: 4 bytes per pixel, tightly packed

    let args = KernelArgs::new()
        .push_ptr(y_plane_ptr)
        .push_ptr(uv_plane_ptr)
        .push_ptr(buf.device_ptr)
        .push_i32(width as i32)
        .push_i32(height as i32)
        .push_i32(pitch as i32)
        .push_i32(pitch as i32) // uv_pitch == y_pitch for NVDEC NV12 output
        .push_i32(out_pitch);

    // 3. Compute launch grid (16x16 blocks).
    let block_x: u32 = 16;
    let block_y: u32 = 16;
    let grid_x = width.div_ceil(block_x);
    let grid_y = height.div_ceil(block_y);

    // 4. Launch the kernel on the default CUDA stream (stream 0 / null).
    //
    // SAFETY:
    // - We pass null as the stream handle, which means the default stream.
    // - y_plane_ptr and uv_plane_ptr are valid device pointers from NVDEC's
    //   cuvidMapVideoFrame64, valid as long as the MappedFrame guard is alive
    //   (guaranteed by the caller who holds the GpuFrame).
    // - buf.device_ptr is a valid device pointer from cuMemAlloc_v2.
    // - The argument types and count match the nv12_to_rgba kernel signature.
    unsafe {
        km.launch(
            &KernelId::Nv12ToRgba,
            [grid_x, grid_y, 1],
            [block_x, block_y, 1],
            &args,
            std::ptr::null_mut(), // default stream
        )
        .map_err(|e| anyhow::anyhow!("NV12->RGBA kernel launch failed: {e}"))?;
    }

    // 5. Synchronize default stream to ensure the kernel has completed.
    unsafe {
        let result = cudarc::driver::sys::cuStreamSynchronize(std::ptr::null_mut());
        result
            .result()
            .map_err(|e| anyhow::anyhow!("cuStreamSynchronize failed: {e:?}"))?;
    }

    // 6. Copy the RGBA result from GPU to CPU.
    let mut rgba_host = vec![0u8; buf.byte_size];

    // SAFETY: buf.device_ptr is valid GPU memory of buf.byte_size bytes containing
    // RGBA data written by the kernel. rgba_host is a freshly allocated buffer of
    // exactly buf.byte_size bytes. The synchronous copy ensures data is available.
    unsafe {
        let result = cudarc::driver::sys::cuMemcpyDtoH_v2(
            rgba_host.as_mut_ptr() as *mut c_void,
            buf.device_ptr,
            buf.byte_size,
        );
        result
            .result()
            .map_err(|e| anyhow::anyhow!("cuMemcpyDtoH (RGBA readback) failed: {e:?}"))?;
    }

    Ok(rgba_host)
}

/// CPU fallback: Copy NV12 frame data from GPU to CPU and convert to RGBA on CPU.
///
/// This is the Phase 0 approach used when the CUDA NV12->RGBA kernel is not
/// available (e.g., nvcc was not found at build time).
fn cpu_nv12_copy_and_convert(
    device_ptr: u64,
    width: u32,
    height: u32,
    pitch: u32,
) -> Result<Vec<u8>> {
    let y_size = pitch as usize * height as usize;
    let uv_size = pitch as usize * (height as usize / 2);
    let total_nv12_size = y_size + uv_size;

    // Allocate host buffer for the NV12 data
    let mut nv12_host = vec![0u8; total_nv12_size];

    // Copy NV12 (Y + UV planes) from GPU to CPU
    // SAFETY: device_ptr is a valid CUdeviceptr from NVDEC's cuvidMapVideoFrame64.
    // nv12_host is a newly allocated buffer of exactly total_nv12_size bytes.
    // We perform a synchronous copy so the data is available immediately.
    unsafe {
        let result = cudarc::driver::sys::cuMemcpyDtoH_v2(
            nv12_host.as_mut_ptr() as *mut c_void,
            device_ptr,
            total_nv12_size,
        );
        result
            .result()
            .map_err(|e| anyhow::anyhow!("cuMemcpyDtoH failed: {e:?}"))?;
    }

    // Split into Y and UV planes
    let y_plane = &nv12_host[..y_size];
    let uv_plane = &nv12_host[y_size..];

    // Convert NV12 to RGBA using BT.709 CPU conversion
    let rgba = nv12_to_rgba(y_plane, uv_plane, width, height, pitch, pitch)
        .map_err(|e| anyhow::anyhow!("NV12->RGBA conversion failed: {e}"))?;

    Ok(rgba)
}

// ---------------------------------------------------------------------------
// Software decode loops (fallbacks)
// ---------------------------------------------------------------------------

/// Decode loop that reads real packets from the demuxer (software path).
///
/// Uses real PTS timing from the container for frame pacing. Since the HW
/// decoder is not available, we generate synthetic RGBA pixels but use the
/// real PTS for timing.
fn real_decode_loop(
    info: &FileInfo,
    demuxer: &mut Box<dyn Demuxer>,
    frame_tx: &Sender<DecodedFrame>,
    cmd_rx: &Receiver<DecodeCommand>,
) {
    let width = info.resolution.width;
    let height = info.resolution.height;
    let fps = info.fps.as_f64();
    let frame_duration = Duration::from_secs_f64(1.0 / fps);

    let mut playing = false;
    let mut frame_num: u64 = 0;

    loop {
        // Check for commands (non-blocking)
        match cmd_rx.try_recv() {
            Ok(DecodeCommand::Play) => {
                playing = true;
                tracing::debug!("Decode thread (real): play");
            }
            Ok(DecodeCommand::Pause) => {
                playing = false;
                tracing::debug!("Decode thread (real): pause");
            }
            Ok(DecodeCommand::Seek(time_secs)) => {
                if let Err(e) = demuxer.seek(time_secs) {
                    tracing::warn!("Decode thread (real): seek failed: {}", e);
                }
                frame_num = (time_secs * fps).round() as u64;
                tracing::debug!(
                    "Decode thread (real): seek to {:.2}s (frame ~{})",
                    time_secs,
                    frame_num,
                );
            }
            Ok(DecodeCommand::Stop) => {
                tracing::info!("Decode thread (real): stop command received");
                return;
            }
            Err(crossbeam::channel::TryRecvError::Empty) => {}
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                tracing::info!("Decode thread (real): command channel disconnected, exiting");
                return;
            }
        }

        if !playing {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Read the next real video packet from the demuxer
        match demuxer.next_video_packet() {
            Some(packet) => {
                let pts_secs = packet.pts.0;

                tracing::debug!(
                    "Decode thread (real): packet {} -> size={} bytes, pts={:.3}s, keyframe={}",
                    frame_num,
                    packet.data.len(),
                    pts_secs,
                    packet.is_keyframe,
                );

                // Software fallback: generate synthetic RGBA pixels but use the
                // real PTS for timing. (NVDEC path is in nvdec_decode_loop.)
                let rgba_data = generate_synthetic_frame(width, height, frame_num, pts_secs);

                let decoded = DecodedFrame {
                    rgba_data,
                    width,
                    height,
                    pts_secs,
                };

                match frame_tx.send(decoded) {
                    Ok(()) => {}
                    Err(_) => {
                        tracing::info!("Decode thread (real): frame channel closed, exiting");
                        return;
                    }
                }

                frame_num += 1;

                // Pace to target FPS
                thread::sleep(frame_duration);
            }
            None => {
                // End of stream
                tracing::info!(
                    "Decode thread (real): end of stream after {} frames",
                    frame_num,
                );
                playing = false;
            }
        }
    }
}

/// Fully synthetic decode loop (fallback when demuxer is unavailable).
///
/// Generates animated test frames at the target FPS. This is the original
/// Phase 0 behavior, preserved as a fallback.
fn synthetic_decode_loop(
    info: &FileInfo,
    frame_tx: &Sender<DecodedFrame>,
    cmd_rx: &Receiver<DecodeCommand>,
) {
    let width = info.resolution.width;
    let height = info.resolution.height;
    let fps = info.fps.as_f64();
    let frame_duration = Duration::from_secs_f64(1.0 / fps);
    let total_frames = (info.duration_secs * fps).ceil() as u64;

    let mut playing = false;
    let mut current_frame: u64 = 0;

    loop {
        // Check for commands (non-blocking)
        match cmd_rx.try_recv() {
            Ok(DecodeCommand::Play) => {
                playing = true;
                tracing::debug!("Decode thread (synthetic): play");
            }
            Ok(DecodeCommand::Pause) => {
                playing = false;
                tracing::debug!("Decode thread (synthetic): pause");
            }
            Ok(DecodeCommand::Seek(time_secs)) => {
                current_frame = (time_secs * fps).round() as u64;
                tracing::debug!(
                    "Decode thread (synthetic): seek to {:.2}s (frame {})",
                    time_secs,
                    current_frame,
                );
            }
            Ok(DecodeCommand::Stop) => {
                tracing::info!("Decode thread (synthetic): stop command received");
                return;
            }
            Err(crossbeam::channel::TryRecvError::Empty) => {}
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                tracing::info!("Decode thread (synthetic): command channel disconnected, exiting");
                return;
            }
        }

        if !playing {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Check if we've reached the end
        if current_frame >= total_frames {
            playing = false;
            continue;
        }

        let pts_secs = current_frame as f64 / fps;
        let rgba_data = generate_synthetic_frame(width, height, current_frame, pts_secs);

        let decoded = DecodedFrame {
            rgba_data,
            width,
            height,
            pts_secs,
        };

        match frame_tx.send(decoded) {
            Ok(()) => {}
            Err(_) => {
                tracing::info!("Decode thread (synthetic): frame channel closed, exiting");
                return;
            }
        }

        current_frame += 1;
        thread::sleep(frame_duration);
    }
}

/// Generate a synthetic decoded frame (Phase 0 placeholder).
///
/// Creates a visually distinct frame for each frame number so we can verify
/// correct sequencing and timing. Uses a sweeping gradient with the frame
/// number embedded in the pattern.
fn generate_synthetic_frame(width: u32, height: u32, frame_num: u64, pts_secs: f64) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut pixels = vec![0u8; w * h * 4];

    let phase = pts_secs as f32;

    for y in 0..h {
        for x in 0..w {
            let offset = (y * w + x) * 4;
            let nx = x as f32 / w as f32;
            let ny = y as f32 / h as f32;

            // Sweeping color based on time
            let hue = ((nx * 180.0 + phase * 60.0) % 360.0 + 360.0) % 360.0;
            let (r, g, b) = hsv_to_rgb(hue, 0.6, 0.7 + 0.3 * ny);

            // Add a moving vertical bar to show progression
            let bar_pos = (phase * 0.2) % 1.0;
            let bar_dist = (nx - bar_pos).abs();
            let bar_intensity = (1.0 - bar_dist * 10.0).clamp(0.0, 0.3);

            // Subtle grid overlay
            let grid = if (x % 64 < 2) || (y % 64 < 2) {
                0.1_f32
            } else {
                0.0
            };

            let final_r = (r + bar_intensity + grid).clamp(0.0, 1.0);
            let final_g = (g + bar_intensity + grid).clamp(0.0, 1.0);
            let final_b = (b + bar_intensity + grid).clamp(0.0, 1.0);

            pixels[offset] = (final_r * 255.0) as u8;
            pixels[offset + 1] = (final_g * 255.0) as u8;
            pixels[offset + 2] = (final_b * 255.0) as u8;
            pixels[offset + 3] = 255;
        }
    }

    // Embed frame number as a simple visual indicator:
    // A row of blocks at the top whose pattern encodes the frame number (binary).
    let block_size = 16;
    let block_y_start = 8;
    let block_y_end = block_y_start + block_size;
    for bit in 0..16 {
        let is_set = (frame_num >> bit) & 1 == 1;
        let block_x_start = 8 + bit as usize * (block_size + 4);
        let block_x_end = block_x_start + block_size;

        if block_x_end >= w {
            break;
        }

        for y in block_y_start..block_y_end.min(h) {
            for x in block_x_start..block_x_end {
                let offset = (y * w + x) * 4;
                if is_set {
                    pixels[offset] = 255; // White
                    pixels[offset + 1] = 255;
                    pixels[offset + 2] = 255;
                } else {
                    pixels[offset] = 40; // Dark gray
                    pixels[offset + 1] = 40;
                    pixels[offset + 2] = 40;
                }
                pixels[offset + 3] = 255;
            }
        }
    }

    pixels
}

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------

/// Convert HSV to RGB. H in [0, 360], S and V in [0, 1]. Returns (r, g, b) in [0, 1].
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r1 + m, g1 + m, b1 + m)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_starts_idle() {
        let engine = EngineOrchestrator::new();
        assert_eq!(*engine.state(), EngineState::Idle);
        assert!(engine.file_info().is_none());
        assert_eq!(engine.current_time_secs(), 0.0);
    }

    #[test]
    fn generate_test_frame_correct_size() {
        let engine = EngineOrchestrator::new();
        let frame = engine.generate_test_frame(640, 480);
        assert_eq!(frame.len(), 640 * 480 * 4);
    }

    #[test]
    fn generate_test_frame_all_opaque() {
        let engine = EngineOrchestrator::new();
        let frame = engine.generate_test_frame(64, 64);
        // Every 4th byte (alpha) should be 255
        for i in (3..frame.len()).step_by(4) {
            assert_eq!(frame[i], 255, "Alpha at byte {} should be 255", i);
        }
    }

    #[test]
    fn generate_error_frame_correct_size() {
        let engine = EngineOrchestrator::new();
        let frame = engine.generate_error_frame(320, 240, "test error");
        assert_eq!(frame.len(), 320 * 240 * 4);
    }

    #[test]
    fn generate_error_frame_all_opaque() {
        let engine = EngineOrchestrator::new();
        let frame = engine.generate_error_frame(64, 64, "test");
        for i in (3..frame.len()).step_by(4) {
            assert_eq!(frame[i], 255, "Alpha at byte {} should be 255", i);
        }
    }

    #[test]
    fn synthetic_frame_correct_size() {
        let frame = generate_synthetic_frame(1920, 1080, 0, 0.0);
        assert_eq!(frame.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn synthetic_frame_all_opaque() {
        let frame = generate_synthetic_frame(64, 64, 42, 1.4);
        for i in (3..frame.len()).step_by(4) {
            assert_eq!(frame[i], 255, "Alpha at byte {} should be 255", i);
        }
    }

    #[test]
    fn synthetic_frame_varies_by_frame_number() {
        let frame_a = generate_synthetic_frame(64, 64, 0, 0.0);
        let frame_b = generate_synthetic_frame(64, 64, 1, 1.0 / 30.0);
        // Frames should be different
        assert_ne!(frame_a, frame_b);
    }

    #[test]
    fn hsv_to_rgb_red() {
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!(g.abs() < 0.01);
        assert!(b.abs() < 0.01);
    }

    #[test]
    fn hsv_to_rgb_green() {
        let (r, g, b) = hsv_to_rgb(120.0, 1.0, 1.0);
        assert!(r.abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!(b.abs() < 0.01);
    }

    #[test]
    fn hsv_to_rgb_blue() {
        let (r, g, b) = hsv_to_rgb(240.0, 1.0, 1.0);
        assert!(r.abs() < 0.01);
        assert!(g.abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn engine_state_labels() {
        assert_eq!(EngineState::Idle.label(), "Idle");
        assert_eq!(EngineState::Loading.label(), "Loading...");
        assert_eq!(EngineState::Playing.label(), "Playing");
        assert_eq!(EngineState::Paused.label(), "Paused");
        assert_eq!(EngineState::Error("oops".into()).label(), "Error");
    }

    #[test]
    fn open_file_transitions_to_paused() {
        let mut engine = EngineOrchestrator::new();
        let result = engine.open_file(PathBuf::from("test.mp4"));
        assert!(result.is_ok());
        assert_eq!(*engine.state(), EngineState::Paused);
        assert!(engine.file_info().is_some());

        // Clean up
        engine.stop();
        assert_eq!(*engine.state(), EngineState::Idle);
    }

    #[test]
    fn play_pause_transitions() {
        let mut engine = EngineOrchestrator::new();
        engine.open_file(PathBuf::from("test.mp4")).unwrap();

        engine.play();
        assert_eq!(*engine.state(), EngineState::Playing);

        engine.pause();
        assert_eq!(*engine.state(), EngineState::Paused);

        engine.toggle_play_pause();
        assert_eq!(*engine.state(), EngineState::Playing);

        engine.toggle_play_pause();
        assert_eq!(*engine.state(), EngineState::Paused);

        engine.stop();
    }

    #[test]
    fn seek_clamps_to_duration() {
        let mut engine = EngineOrchestrator::new();
        engine.open_file(PathBuf::from("test.mp4")).unwrap();

        engine.seek(5.0);
        assert!((engine.current_time_secs() - 5.0).abs() < 0.01);

        // Seek past end should clamp
        engine.seek(999.0);
        assert!(engine.current_time_secs() <= engine.duration_secs());

        // Seek before start should clamp
        engine.seek(-5.0);
        assert!(engine.current_time_secs() >= 0.0);

        engine.stop();
    }

    #[test]
    fn probe_recognizes_mkv_format() {
        let result = probe_file_info(Path::new("nonexistent.mkv"), "nonexistent.mkv");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("MKV"),
            "Error should mention MKV, got: {err_msg}",
        );
    }

    #[test]
    fn probe_recognizes_webm_format() {
        let result = probe_file_info(Path::new("nonexistent.webm"), "nonexistent.webm");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("MKV"),
            "Error should mention MKV (WebM uses MkvDemuxer), got: {err_msg}",
        );
    }

    #[test]
    fn try_open_demuxer_mkv_errors_on_missing_file() {
        let result = try_open_demuxer(Path::new("nonexistent.mkv"));
        assert!(result.is_err());
        let err_msg = match result {
            Ok(_) => panic!("Expected error for nonexistent MKV file"),
            Err(e) => e.to_string(),
        };
        assert!(
            err_msg.contains("MKV"),
            "Error should mention MKV, got: {err_msg}",
        );
    }

    #[test]
    fn decode_thread_produces_frames() {
        let info = FileInfo {
            path: PathBuf::from("test.mp4"),
            file_name: "test.mp4".to_string(),
            resolution: Resolution::new(64, 64),
            fps: Rational::FPS_30,
            duration_secs: 1.0,
            codec: VideoCodec::H264,
        };

        let (frame_tx, frame_rx) = channel::bounded::<DecodedFrame>(4);
        let (cmd_tx, cmd_rx) = channel::unbounded::<DecodeCommand>();
        let (gpu_info_tx, _gpu_info_rx) = channel::bounded::<GpuInfoMsg>(1);

        let handle = thread::spawn(move || {
            decode_thread_main(info, frame_tx, cmd_rx, gpu_info_tx);
        });

        // Tell it to play
        cmd_tx.send(DecodeCommand::Play).unwrap();

        // Wait for a frame (with timeout)
        let frame = frame_rx.recv_timeout(Duration::from_secs(2));
        assert!(frame.is_ok(), "Should receive a frame within 2 seconds");

        let frame = frame.unwrap();
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.rgba_data.len(), 64 * 64 * 4);

        // Stop the thread
        cmd_tx.send(DecodeCommand::Stop).unwrap();
        handle.join().unwrap();
    }
}
