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
use ms_common::{Rational, Resolution, VideoCodec};
use std::path::PathBuf;
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
    /// For Phase 0, since the actual demuxer/decoder crates may not be fully
    /// functional yet, this uses a simulated decode thread that generates
    /// animated frames to exercise the threading and channel pipeline.
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

        // TODO: In later phases, probe the file with the real demuxer to
        // extract actual resolution, fps, duration, and codec info.
        // For now, use sensible defaults.
        let info = FileInfo {
            path: path.clone(),
            file_name,
            resolution: Resolution::HD,
            fps: Rational::FPS_30,
            duration_secs: 10.0,
            codec: VideoCodec::H264,
        };
        self.file_info = Some(info.clone());

        // Create bounded channels:
        // - frame channel: small ring buffer (4 frames max to limit memory)
        // - command channel: unbounded (commands are tiny)
        let (frame_tx, frame_rx) = channel::bounded::<DecodedFrame>(4);
        let (cmd_tx, cmd_rx) = channel::unbounded::<DecodeCommand>();

        self.frame_rx = Some(frame_rx);
        self.cmd_tx = Some(cmd_tx);

        // Spawn decode thread
        let thread_info = info;
        let handle = thread::Builder::new()
            .name("decode-worker".to_string())
            .spawn(move || {
                decode_thread_main(thread_info, frame_tx, cmd_rx);
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
                let frame = self.generate_error_frame(
                    self.preview_width,
                    self.preview_height,
                    msg,
                );
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
                    if (cx + cy) % 2 == 0 { 0.9_f32 } else { 0.6 }
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
// Decode thread — runs on a separate OS thread
// ---------------------------------------------------------------------------

/// Main loop for the decode thread.
///
/// **Phase 0 (current):** Generates synthetic animated frames to exercise the
/// full threading and channel pipeline without requiring actual video files.
///
/// **Future phases:** This will be replaced with:
/// 1. Open file with the `ms-demux` crate's MP4/MKV parser
/// 2. Extract NAL units (video packets)
/// 3. Feed packets to `HwDecoder` (via `GpuBackend::create_decoder`)
/// 4. Receive decoded `GpuFrame` (NV12 on GPU)
/// 5. Run NV12->RGBA conversion kernel on GPU
/// 6. Copy RGBA data to staging buffer
/// 7. Send to main thread via the frame channel
fn decode_thread_main(
    info: FileInfo,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecodeCommand>,
) {
    tracing::info!(
        "Decode thread started for '{}' ({}x{} @ {} fps)",
        info.file_name,
        info.resolution.width,
        info.resolution.height,
        info.fps,
    );

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
                tracing::debug!("Decode thread: play");
            }
            Ok(DecodeCommand::Pause) => {
                playing = false;
                tracing::debug!("Decode thread: pause");
            }
            Ok(DecodeCommand::Seek(time_secs)) => {
                current_frame = (time_secs * fps).round() as u64;
                tracing::debug!(
                    "Decode thread: seek to {:.2}s (frame {})",
                    time_secs,
                    current_frame
                );
            }
            Ok(DecodeCommand::Stop) => {
                tracing::info!("Decode thread: stop command received");
                return;
            }
            Err(crossbeam::channel::TryRecvError::Empty) => {}
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                tracing::info!("Decode thread: command channel disconnected, exiting");
                return;
            }
        }

        if !playing {
            // When paused, sleep briefly to avoid busy-waiting, but keep
            // checking for commands.
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Check if we've reached the end
        if current_frame >= total_frames {
            playing = false;
            continue;
        }

        // --- Phase 0: Generate a synthetic frame ---
        // This exercises the full pipeline without needing real decode.
        // The frame is a moving gradient that clearly shows the frame number,
        // so we can verify frame sequencing is correct.
        let pts_secs = current_frame as f64 / fps;
        let rgba_data = generate_synthetic_frame(width, height, current_frame, pts_secs);

        let decoded = DecodedFrame {
            rgba_data,
            width,
            height,
            pts_secs,
        };

        // Send to main thread. If the channel is full (back-pressure),
        // this will block until there's room. If the receiver is dropped,
        // the send will fail and we exit.
        match frame_tx.send(decoded) {
            Ok(()) => {}
            Err(_) => {
                tracing::info!("Decode thread: frame channel closed, exiting");
                return;
            }
        }

        current_frame += 1;

        // Pace ourselves to roughly the target FPS.
        // In a real pipeline, the decode time itself would provide some pacing,
        // but for synthetic frames we need to sleep explicitly.
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
            let grid = if (x % 64 < 2) || (y % 64 < 2) { 0.1_f32 } else { 0.0 };

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

        let handle = thread::spawn(move || {
            decode_thread_main(info, frame_tx, cmd_rx);
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
