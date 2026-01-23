//! Command types for the WebSocket protocol

use serde::{Deserialize, Serialize};

/// Incoming commands from browser
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    /// Authenticate with token
    Auth {
        id: String,
        token: String,
    },

    /// Open a video file
    Open {
        id: String,
        path: String,
    },

    /// Decode a single frame
    Decode {
        id: String,
        file_id: String,
        frame: u32,
        #[serde(default)]
        format: PixelFormat,
        #[serde(default = "default_scale")]
        scale: f32,
        #[serde(default)]
        compression: Option<Compression>,
    },

    /// Decode a range of frames
    DecodeRange {
        id: String,
        file_id: String,
        start_frame: u32,
        end_frame: u32,
        #[serde(default)]
        priority: Priority,
    },

    /// Prefetch frames around a position (no response)
    Prefetch {
        file_id: String,
        around_frame: u32,
        #[serde(default = "default_radius")]
        radius: u32,
    },

    /// Start an encode job
    StartEncode {
        id: String,
        output: EncodeOutput,
        frame_count: u32,
    },

    /// Send a frame for encoding (binary follows)
    EncodeFrame {
        id: String,
        frame_num: u32,
    },

    /// Finish encoding
    FinishEncode {
        id: String,
    },

    /// Cancel an encode job
    CancelEncode {
        id: String,
    },

    /// Close a file
    Close {
        id: String,
        file_id: String,
    },

    /// Get system info
    Info {
        id: String,
    },

    /// Ping for connection keepalive
    Ping {
        id: String,
    },

    /// Download a YouTube video using yt-dlp
    DownloadYoutube {
        id: String,
        url: String,
        #[serde(default)]
        format_id: Option<String>,
        #[serde(default)]
        output_dir: Option<String>,
    },

    /// List available formats for a YouTube video
    ListFormats {
        id: String,
        url: String,
    },

    /// Get a file from local filesystem (for serving downloads)
    GetFile {
        id: String,
        path: String,
    },
}

fn default_scale() -> f32 {
    1.0
}

fn default_radius() -> u32 {
    50
}

/// Pixel format for decoded frames
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PixelFormat {
    #[default]
    Rgba8,
    Rgb8,
    Yuv420,
}

/// Compression for frame transfer
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    Lz4,
}

/// Priority for decode requests
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
}

/// Encode output settings
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EncodeOutput {
    pub path: String,
    pub codec: VideoCodec,
    #[serde(default)]
    pub profile: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    #[serde(default)]
    pub audio: Option<AudioSettings>,
}

/// Video codec for encoding
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VideoCodec {
    Prores,
    Dnxhd,
    H264,
    H265,
    Vp9,
    Ffv1,
    Utvideo,
    Mjpeg,
}

/// Audio settings for encoding
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioSettings {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u8,
    #[serde(default)]
    pub bitrate: Option<u32>,
}

/// Audio codec for encoding
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AudioCodec {
    Aac,
    Flac,
    Pcm,
    Alac,
}

/// Response types
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Response {
    Ok(OkResponse),
    Error(ErrorResponse),
    Progress(ProgressResponse),
}

#[derive(Debug, Clone, Serialize)]
pub struct OkResponse {
    pub id: String,
    pub ok: bool,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub id: String,
    pub ok: bool,
    pub error: ErrorInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorInfo {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProgressResponse {
    pub id: String,
    pub progress: f32,
    pub frames_done: u32,
    pub frames_total: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_ms: Option<u64>,
}

/// File metadata returned on open
#[derive(Debug, Clone, Serialize)]
pub struct FileMetadata {
    pub file_id: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_ms: u64,
    pub frame_count: u64,
    pub codec: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_space: Option<String>,
    pub audio_tracks: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hw_accel: Option<String>,
}

/// System info response
#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub version: String,
    pub ffmpeg_version: String,
    pub hw_accel: Vec<String>,
    pub cache_used_mb: usize,
    pub cache_max_mb: usize,
    pub open_files: usize,
}

// Helper functions for creating responses
impl Response {
    pub fn ok(id: impl Into<String>, data: serde_json::Value) -> Self {
        Response::Ok(OkResponse {
            id: id.into(),
            ok: true,
            data,
        })
    }

    pub fn error(id: impl Into<String>, code: impl Into<String>, message: impl Into<String>) -> Self {
        Response::Error(ErrorResponse {
            id: id.into(),
            ok: false,
            error: ErrorInfo {
                code: code.into(),
                message: message.into(),
            },
        })
    }

    pub fn progress(id: impl Into<String>, progress: f32, done: u32, total: u32, eta: Option<u64>) -> Self {
        Response::Progress(ProgressResponse {
            id: id.into(),
            progress,
            frames_done: done,
            frames_total: total,
            eta_ms: eta,
        })
    }
}

/// Error codes
pub mod error_codes {
    pub const AUTH_REQUIRED: &str = "AUTH_REQUIRED";
    pub const INVALID_TOKEN: &str = "INVALID_TOKEN";
    pub const FILE_NOT_FOUND: &str = "FILE_NOT_FOUND";
    pub const PERMISSION_DENIED: &str = "PERMISSION_DENIED";
    pub const UNSUPPORTED_CODEC: &str = "UNSUPPORTED_CODEC";
    pub const DECODE_ERROR: &str = "DECODE_ERROR";
    pub const ENCODE_ERROR: &str = "ENCODE_ERROR";
    pub const OUT_OF_MEMORY: &str = "OUT_OF_MEMORY";
    pub const INVALID_FRAME: &str = "INVALID_FRAME";
    pub const INVALID_PATH: &str = "INVALID_PATH";
    pub const FILE_NOT_OPEN: &str = "FILE_NOT_OPEN";
    pub const ENCODE_NOT_STARTED: &str = "ENCODE_NOT_STARTED";
    pub const INTERNAL_ERROR: &str = "INTERNAL_ERROR";
    pub const YTDLP_NOT_FOUND: &str = "YTDLP_NOT_FOUND";
    pub const DOWNLOAD_FAILED: &str = "DOWNLOAD_FAILED";
    pub const INVALID_URL: &str = "INVALID_URL";
}
