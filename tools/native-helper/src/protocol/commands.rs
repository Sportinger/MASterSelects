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

    /// Get system info
    Info {
        id: String,
    },

    /// Ping for connection keepalive
    Ping {
        id: String,
    },

    /// Download a YouTube video using yt-dlp (legacy command name)
    DownloadYoutube {
        id: String,
        url: String,
        #[serde(default)]
        format_id: Option<String>,
        #[serde(default)]
        output_dir: Option<String>,
    },

    /// Generic download using yt-dlp (supports all platforms: YouTube, TikTok, Instagram, etc.)
    Download {
        id: String,
        url: String,
        #[serde(default)]
        format_id: Option<String>,
        #[serde(default)]
        output_dir: Option<String>,
    },

    /// List available formats for a video URL
    ListFormats {
        id: String,
        url: String,
    },

    /// Get a file from local filesystem (for serving downloads)
    GetFile {
        id: String,
        path: String,
    },

    /// Locate a file by name in common directories
    Locate {
        id: String,
        filename: String,
        /// Optional additional directories to search
        #[serde(default)]
        search_dirs: Vec<String>,
    },
}

/// Response types
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Response {
    Ok(OkResponse),
    Error(ErrorResponse),
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

/// System info response
#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub version: String,
    pub ytdlp_available: bool,
    pub download_dir: String,
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

    pub fn error(
        id: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Response::Error(ErrorResponse {
            id: id.into(),
            ok: false,
            error: ErrorInfo {
                code: code.into(),
                message: message.into(),
            },
        })
    }

    /// Progress response for download percent with speed and eta
    pub fn download_progress(id: impl Into<String>, percent: u8, speed: Option<&str>, eta: Option<&str>) -> Self {
        let mut data = serde_json::json!({ "type": "progress", "percent": percent });
        if let Some(s) = speed {
            data["speed"] = serde_json::json!(s);
        }
        if let Some(e) = eta {
            data["eta"] = serde_json::json!(e);
        }
        Response::Ok(OkResponse {
            id: id.into(),
            ok: true,
            data,
        })
    }
}

/// Error codes
pub mod error_codes {
    pub const AUTH_REQUIRED: &str = "AUTH_REQUIRED";
    pub const INVALID_TOKEN: &str = "INVALID_TOKEN";
    pub const FILE_NOT_FOUND: &str = "FILE_NOT_FOUND";
    pub const PERMISSION_DENIED: &str = "PERMISSION_DENIED";
    pub const INVALID_PATH: &str = "INVALID_PATH";
    pub const INTERNAL_ERROR: &str = "INTERNAL_ERROR";
    pub const YTDLP_NOT_FOUND: &str = "YTDLP_NOT_FOUND";
    pub const DOWNLOAD_FAILED: &str = "DOWNLOAD_FAILED";
    pub const INVALID_URL: &str = "INVALID_URL";
}
