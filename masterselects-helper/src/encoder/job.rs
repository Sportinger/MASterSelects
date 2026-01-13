//! Encode job management

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};

use crate::protocol::{AudioCodec, EncodeOutput, VideoCodec};

/// Encode job status
#[derive(Debug, Clone)]
pub enum EncodeStatus {
    Starting,
    Encoding { frames_done: u32, frames_total: u32 },
    Finishing,
    Completed { output_path: PathBuf },
    Failed { error: String },
    Cancelled,
}

/// Encode job handle for status tracking
pub struct EncodeJobHandle {
    status: Arc<Mutex<EncodeStatus>>,
    start_time: Instant,
}

impl EncodeJobHandle {
    pub fn status(&self) -> EncodeStatus {
        self.status.lock().clone()
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    pub fn progress(&self) -> f32 {
        match &*self.status.lock() {
            EncodeStatus::Encoding { frames_done, frames_total } => {
                *frames_done as f32 / *frames_total as f32
            }
            EncodeStatus::Completed { .. } => 1.0,
            _ => 0.0,
        }
    }

    pub fn frames_done(&self) -> u32 {
        match &*self.status.lock() {
            EncodeStatus::Encoding { frames_done, .. } => *frames_done,
            _ => 0,
        }
    }

    pub fn frames_total(&self) -> u32 {
        match &*self.status.lock() {
            EncodeStatus::Encoding { frames_total, .. } => *frames_total,
            _ => 0,
        }
    }

    pub fn eta_ms(&self) -> Option<u64> {
        let elapsed = self.elapsed_ms();
        let progress = self.progress();

        if progress > 0.01 {
            let total_estimated = (elapsed as f32 / progress) as u64;
            Some(total_estimated.saturating_sub(elapsed))
        } else {
            None
        }
    }
}

/// Video encode job using FFmpeg CLI
pub struct EncodeJob {
    output: EncodeOutput,
    frame_count: u32,
    frames_received: u32,

    // FFmpeg process
    ffmpeg: Option<Child>,
    stdin: Option<std::process::ChildStdin>,

    // Status
    status: Arc<Mutex<EncodeStatus>>,
    start_time: Instant,
    cancelled: bool,
}

impl EncodeJob {
    /// Create a new encode job
    pub fn new(output: EncodeOutput, frame_count: u32) -> Self {
        Self {
            output,
            frame_count,
            frames_received: 0,
            ffmpeg: None,
            stdin: None,
            status: Arc::new(Mutex::new(EncodeStatus::Starting)),
            start_time: Instant::now(),
            cancelled: false,
        }
    }

    /// Start the encode job
    pub fn start(&mut self) -> Result<EncodeJobHandle> {
        info!(
            "Starting encode job: {}x{} @ {} fps, {} frames, codec: {:?}",
            self.output.width, self.output.height, self.output.fps, self.frame_count, self.output.codec
        );

        let args = self.build_ffmpeg_args();
        debug!("FFmpeg args: {:?}", args);

        let mut child = Command::new("ffmpeg")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to start FFmpeg: {}", e))?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("Failed to get stdin"))?;

        self.ffmpeg = Some(child);
        self.stdin = Some(stdin);
        *self.status.lock() = EncodeStatus::Encoding {
            frames_done: 0,
            frames_total: self.frame_count,
        };

        Ok(EncodeJobHandle {
            status: self.status.clone(),
            start_time: self.start_time,
        })
    }

    /// Add a frame to the encode
    pub fn add_frame(&mut self, frame_data: &[u8]) -> Result<()> {
        if self.cancelled {
            return Err(anyhow!("Encode cancelled"));
        }

        let stdin = self.stdin.as_mut().ok_or_else(|| anyhow!("Encode not started"))?;

        // Write raw frame data
        stdin.write_all(frame_data)?;

        self.frames_received += 1;
        *self.status.lock() = EncodeStatus::Encoding {
            frames_done: self.frames_received,
            frames_total: self.frame_count,
        };

        if self.frames_received % 100 == 0 {
            debug!("Encoded {}/{} frames", self.frames_received, self.frame_count);
        }

        Ok(())
    }

    /// Finish the encode and write output
    pub fn finish(&mut self) -> Result<PathBuf> {
        *self.status.lock() = EncodeStatus::Finishing;

        // Close stdin to signal end of input
        self.stdin.take();

        // Wait for FFmpeg to finish
        if let Some(mut child) = self.ffmpeg.take() {
            let output = child.wait_with_output()?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                error!("FFmpeg failed: {}", stderr);
                *self.status.lock() = EncodeStatus::Failed {
                    error: stderr.to_string(),
                };
                return Err(anyhow!("FFmpeg encoding failed: {}", stderr));
            }
        }

        let output_path = PathBuf::from(&self.output.path);
        info!("Encode completed: {}", output_path.display());

        *self.status.lock() = EncodeStatus::Completed {
            output_path: output_path.clone(),
        };

        Ok(output_path)
    }

    /// Cancel the encode
    pub fn cancel(&mut self) {
        self.cancelled = true;
        *self.status.lock() = EncodeStatus::Cancelled;

        // Kill FFmpeg process
        if let Some(mut child) = self.ffmpeg.take() {
            let _ = child.kill();
        }

        self.stdin.take();

        info!("Encode cancelled");
    }

    /// Build FFmpeg command line arguments
    fn build_ffmpeg_args(&self) -> Vec<String> {
        let mut args = vec![
            "-y".to_string(),           // Overwrite output
            "-f".to_string(),           // Input format
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),     // Input pixel format
            "rgba".to_string(),
            "-s".to_string(),           // Input size
            format!("{}x{}", self.output.width, self.output.height),
            "-r".to_string(),           // Input frame rate
            format!("{}", self.output.fps),
            "-i".to_string(),           // Input from stdin
            "-".to_string(),
        ];

        // Video codec settings
        args.extend(self.build_video_args());

        // Audio settings (if provided)
        if let Some(audio) = &self.output.audio {
            args.extend(self.build_audio_args(audio));
        } else {
            args.push("-an".to_string()); // No audio
        }

        // Output path
        args.push(self.output.path.clone());

        args
    }

    /// Build video codec arguments
    fn build_video_args(&self) -> Vec<String> {
        match self.output.codec {
            VideoCodec::Prores => {
                let profile = self.output.profile.as_deref().unwrap_or("3"); // HQ default
                let profile_num = match profile {
                    "proxy" | "0" => "0",
                    "lt" | "1" => "1",
                    "standard" | "2" => "2",
                    "hq" | "3" => "3",
                    "4444" | "4" => "4",
                    "4444xq" | "5" => "5",
                    _ => "3",
                };

                vec![
                    "-c:v".to_string(),
                    "prores_ks".to_string(),
                    "-profile:v".to_string(),
                    profile_num.to_string(),
                    "-vendor".to_string(),
                    "apl0".to_string(),
                    "-pix_fmt".to_string(),
                    if profile_num == "4" || profile_num == "5" {
                        "yuva444p10le".to_string()
                    } else {
                        "yuv422p10le".to_string()
                    },
                ]
            }

            VideoCodec::Dnxhd => {
                let profile = self.output.profile.as_deref().unwrap_or("dnxhr_hq");
                vec![
                    "-c:v".to_string(),
                    "dnxhd".to_string(),
                    "-profile:v".to_string(),
                    profile.to_string(),
                    "-pix_fmt".to_string(),
                    if profile.contains("444") {
                        "yuv444p10le".to_string()
                    } else if profile.contains("hqx") {
                        "yuv422p10le".to_string()
                    } else {
                        "yuv422p".to_string()
                    },
                ]
            }

            VideoCodec::H264 => {
                vec![
                    "-c:v".to_string(),
                    "libx264".to_string(),
                    "-preset".to_string(),
                    "medium".to_string(),
                    "-crf".to_string(),
                    "18".to_string(),
                    "-pix_fmt".to_string(),
                    "yuv420p".to_string(),
                ]
            }

            VideoCodec::H265 => {
                vec![
                    "-c:v".to_string(),
                    "libx265".to_string(),
                    "-preset".to_string(),
                    "medium".to_string(),
                    "-crf".to_string(),
                    "20".to_string(),
                    "-pix_fmt".to_string(),
                    "yuv420p".to_string(),
                ]
            }

            VideoCodec::Vp9 => {
                vec![
                    "-c:v".to_string(),
                    "libvpx-vp9".to_string(),
                    "-crf".to_string(),
                    "30".to_string(),
                    "-b:v".to_string(),
                    "0".to_string(),
                    "-pix_fmt".to_string(),
                    "yuv420p".to_string(),
                ]
            }

            VideoCodec::Ffv1 => {
                vec![
                    "-c:v".to_string(),
                    "ffv1".to_string(),
                    "-level".to_string(),
                    "3".to_string(),
                    "-coder".to_string(),
                    "1".to_string(),
                    "-context".to_string(),
                    "1".to_string(),
                    "-slicecrc".to_string(),
                    "1".to_string(),
                ]
            }

            VideoCodec::Utvideo => {
                vec![
                    "-c:v".to_string(),
                    "utvideo".to_string(),
                    "-pix_fmt".to_string(),
                    "rgba".to_string(),
                ]
            }

            VideoCodec::Mjpeg => {
                vec![
                    "-c:v".to_string(),
                    "mjpeg".to_string(),
                    "-q:v".to_string(),
                    "2".to_string(),
                    "-pix_fmt".to_string(),
                    "yuvj422p".to_string(),
                ]
            }
        }
    }

    /// Build audio codec arguments
    fn build_audio_args(&self, audio: &crate::protocol::AudioSettings) -> Vec<String> {
        match audio.codec {
            AudioCodec::Aac => {
                let bitrate = audio.bitrate.unwrap_or(256000);
                vec![
                    "-c:a".to_string(),
                    "aac".to_string(),
                    "-b:a".to_string(),
                    format!("{}k", bitrate / 1000),
                    "-ar".to_string(),
                    format!("{}", audio.sample_rate),
                    "-ac".to_string(),
                    format!("{}", audio.channels),
                ]
            }

            AudioCodec::Flac => {
                vec![
                    "-c:a".to_string(),
                    "flac".to_string(),
                    "-ar".to_string(),
                    format!("{}", audio.sample_rate),
                    "-ac".to_string(),
                    format!("{}", audio.channels),
                ]
            }

            AudioCodec::Pcm => {
                vec![
                    "-c:a".to_string(),
                    "pcm_s24le".to_string(),
                    "-ar".to_string(),
                    format!("{}", audio.sample_rate),
                    "-ac".to_string(),
                    format!("{}", audio.channels),
                ]
            }

            AudioCodec::Alac => {
                vec![
                    "-c:a".to_string(),
                    "alac".to_string(),
                    "-ar".to_string(),
                    format!("{}", audio.sample_rate),
                    "-ac".to_string(),
                    format!("{}", audio.channels),
                ]
            }
        }
    }
}

impl Drop for EncodeJob {
    fn drop(&mut self) {
        // Ensure FFmpeg is killed if still running
        if let Some(mut child) = self.ffmpeg.take() {
            let _ = child.kill();
        }
    }
}
