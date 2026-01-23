//! Per-connection session management

use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::cache::FrameCache;
use crate::decoder::{DecoderPool, SharedDecoderPool, detect_hw_accel};
use crate::encoder::{EncodeJob, EncodeJobHandle};
use crate::protocol::{
    error_codes, Command, Compression, PixelFormat, Response, SystemInfo,
    DecodedFrame, encode_frame_message,
};

/// Generate a random auth token
pub fn generate_auth_token() -> String {
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
        .collect()
}

/// Shared application state
pub struct AppState {
    pub decoder_pool: SharedDecoderPool,
    pub frame_cache: Arc<FrameCache>,
    pub auth_token: Option<String>,
}

impl AppState {
    pub fn new(cache_mb: usize, max_decoders: usize, auth_token: Option<String>) -> Self {
        Self {
            decoder_pool: Arc::new(DecoderPool::new(max_decoders)),
            frame_cache: Arc::new(FrameCache::new(cache_mb)),
            auth_token,
        }
    }
}

/// Per-connection session
pub struct Session {
    state: Arc<AppState>,
    authenticated: bool,
    encode_jobs: HashMap<String, EncodeJob>,
    encode_handles: HashMap<String, EncodeJobHandle>,
    request_counter: u32,
}

impl Session {
    pub fn new(state: Arc<AppState>) -> Self {
        // If no token configured, auto-authenticate
        let authenticated = state.auth_token.is_none();

        Self {
            state,
            authenticated,
            encode_jobs: HashMap::new(),
            encode_handles: HashMap::new(),
            request_counter: 0,
        }
    }

    /// Get next request ID
    fn next_request_id(&mut self) -> u32 {
        self.request_counter += 1;
        self.request_counter
    }

    /// Check if authenticated
    pub fn is_authenticated(&self) -> bool {
        self.authenticated
    }

    /// Handle a command, return response and optional binary data
    pub async fn handle_command(&mut self, cmd: Command) -> (Option<Response>, Option<Vec<u8>>) {
        // Auth required for most commands
        if !self.authenticated {
            if let Command::Auth { .. } = cmd {
                // Allow auth command
            } else {
                return (
                    Some(Response::error("", error_codes::AUTH_REQUIRED, "Authentication required")),
                    None,
                );
            }
        }

        match cmd {
            Command::Auth { id, token } => {
                let response = self.handle_auth(&id, &token);
                (Some(response), None)
            }

            Command::Open { id, path } => {
                let response = self.handle_open(&id, &path);
                (Some(response), None)
            }

            Command::Decode {
                id,
                file_id,
                frame,
                format,
                scale,
                compression,
            } => {
                let result = self.handle_decode(&id, &file_id, frame, format, scale, compression);
                match result {
                    Ok(binary) => (None, Some(binary)),
                    Err(response) => (Some(response), None),
                }
            }

            Command::DecodeRange {
                id,
                file_id,
                start_frame,
                end_frame,
                priority: _,
            } => {
                // For now, decode range returns multiple binary messages
                // This would need streaming support in the WebSocket handler
                let response = Response::error(&id, error_codes::INTERNAL_ERROR, "DecodeRange not yet implemented - use multiple Decode calls");
                (Some(response), None)
            }

            Command::Prefetch {
                file_id,
                around_frame,
                radius,
            } => {
                self.handle_prefetch(&file_id, around_frame, radius);
                (None, None) // No response for prefetch
            }

            Command::StartEncode {
                id,
                output,
                frame_count,
            } => {
                let response = self.handle_start_encode(&id, output, frame_count);
                (Some(response), None)
            }

            Command::EncodeFrame { id, frame_num } => {
                // Frame data comes separately as binary
                // This is handled by handle_encode_frame
                let response = Response::error(&id, error_codes::INTERNAL_ERROR, "EncodeFrame should include binary data");
                (Some(response), None)
            }

            Command::FinishEncode { id } => {
                let response = self.handle_finish_encode(&id);
                (Some(response), None)
            }

            Command::CancelEncode { id } => {
                let response = self.handle_cancel_encode(&id);
                (Some(response), None)
            }

            Command::Close { id, file_id } => {
                let response = self.handle_close(&id, &file_id);
                (Some(response), None)
            }

            Command::Info { id } => {
                let response = self.handle_info(&id);
                (Some(response), None)
            }

            Command::Ping { id } => {
                (Some(Response::ok(&id, serde_json::json!({"pong": true}))), None)
            }

            Command::DownloadYoutube { id, url, output_dir } => {
                let response = self.handle_download_youtube(&id, &url, output_dir.as_deref()).await;
                (Some(response), None)
            }

            Command::GetFile { id, path } => {
                let response = self.handle_get_file(&id, &path);
                (Some(response), None)
            }
        }
    }

    /// Handle binary frame data for encoding
    pub fn handle_encode_frame(&mut self, encode_id: &str, frame_data: &[u8]) -> Option<Response> {
        let job = match self.encode_jobs.get_mut(encode_id) {
            Some(job) => job,
            None => {
                return Some(Response::error(
                    encode_id,
                    error_codes::ENCODE_NOT_STARTED,
                    "No active encode job",
                ));
            }
        };

        if let Err(e) = job.add_frame(frame_data) {
            return Some(Response::error(encode_id, error_codes::ENCODE_ERROR, e.to_string()));
        }

        // Return progress
        if let Some(handle) = self.encode_handles.get(encode_id) {
            Some(Response::progress(
                encode_id,
                handle.progress(),
                handle.frames_done(),
                handle.frames_total(),
                handle.eta_ms(),
            ))
        } else {
            None
        }
    }

    fn handle_auth(&mut self, id: &str, token: &str) -> Response {
        match &self.state.auth_token {
            Some(expected) if expected == token => {
                self.authenticated = true;
                info!("Client authenticated");
                Response::ok(id, serde_json::json!({"authenticated": true}))
            }
            Some(_) => {
                warn!("Invalid auth token");
                Response::error(id, error_codes::INVALID_TOKEN, "Invalid token")
            }
            None => {
                // No token required
                self.authenticated = true;
                Response::ok(id, serde_json::json!({"authenticated": true}))
            }
        }
    }

    fn handle_open(&self, id: &str, path: &str) -> Response {
        // Validate path
        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !path.exists() {
            return Response::error(
                id,
                error_codes::FILE_NOT_FOUND,
                format!("File not found: {}", path.display()),
            );
        }

        // Open file
        match self.state.decoder_pool.open(path) {
            Ok(metadata) => {
                Response::ok(id, serde_json::to_value(metadata).unwrap())
            }
            Err(e) => {
                Response::error(id, error_codes::DECODE_ERROR, e.to_string())
            }
        }
    }

    fn handle_decode(
        &mut self,
        id: &str,
        file_id: &str,
        frame: u32,
        format: PixelFormat,
        scale: f32,
        compression: Option<Compression>,
    ) -> Result<Vec<u8>, Response> {
        // Check cache first
        if let Some(cached) = self.state.frame_cache.get(file_id, frame) {
            debug!("Cache hit for {}:{}", file_id, frame);
            let decoded = DecodedFrame {
                width: cached.width,
                height: cached.height,
                frame_num: frame,
                data: cached.data.as_ref().clone(),
            };
            let request_id = self.next_request_id();
            return Ok(encode_frame_message(&decoded, compression, request_id, scale < 1.0));
        }

        // Decode frame
        let result = self.state.decoder_pool.decode_frame(file_id, frame, format, scale);

        match result {
            Ok(frame_data) => {
                // Cache the frame
                self.state.frame_cache.insert(
                    file_id,
                    frame,
                    frame_data.data.clone(),
                    frame_data.width,
                    frame_data.height,
                );

                let decoded = DecodedFrame {
                    width: frame_data.width,
                    height: frame_data.height,
                    frame_num: frame,
                    data: frame_data.data,
                };

                let request_id = self.next_request_id();
                Ok(encode_frame_message(&decoded, compression, request_id, scale < 1.0))
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("not open") {
                    Err(Response::error(id, error_codes::FILE_NOT_OPEN, msg))
                } else if msg.contains("out of range") {
                    Err(Response::error(id, error_codes::INVALID_FRAME, msg))
                } else {
                    Err(Response::error(id, error_codes::DECODE_ERROR, msg))
                }
            }
        }
    }

    fn handle_prefetch(&self, file_id: &str, around_frame: u32, radius: u32) {
        // Get frames that need to be cached
        let missing = self.state.frame_cache.get_missing_in_range(file_id, around_frame, radius);

        if missing.is_empty() {
            return;
        }

        debug!(
            "Prefetching {} frames around {} for {}",
            missing.len(),
            around_frame,
            file_id
        );

        // Sort by distance from center
        let mut frames: Vec<u32> = missing;
        frames.sort_by_key(|f| ((*f as i64) - (around_frame as i64)).abs());

        // Decode in background (simplified - real impl would use tokio::spawn)
        let decoder_pool = self.state.decoder_pool.clone();
        let frame_cache = self.state.frame_cache.clone();
        let file_id = file_id.to_string();

        // For now, just touch the file to keep it alive
        decoder_pool.touch(&file_id);

        // TODO: Spawn background task for actual prefetch
        // This would decode frames in order of distance from playhead
    }

    fn handle_start_encode(
        &mut self,
        id: &str,
        output: crate::protocol::EncodeOutput,
        frame_count: u32,
    ) -> Response {
        let mut job = EncodeJob::new(output, frame_count);

        match job.start() {
            Ok(handle) => {
                self.encode_jobs.insert(id.to_string(), job);
                self.encode_handles.insert(id.to_string(), handle);
                Response::ok(id, serde_json::json!({"started": true}))
            }
            Err(e) => Response::error(id, error_codes::ENCODE_ERROR, e.to_string()),
        }
    }

    fn handle_finish_encode(&mut self, id: &str) -> Response {
        let job = match self.encode_jobs.remove(id) {
            Some(job) => job,
            None => {
                return Response::error(id, error_codes::ENCODE_NOT_STARTED, "No active encode job");
            }
        };

        let mut job = job;
        match job.finish() {
            Ok(path) => {
                self.encode_handles.remove(id);
                Response::ok(
                    id,
                    serde_json::json!({
                        "completed": true,
                        "output_path": path.display().to_string()
                    }),
                )
            }
            Err(e) => {
                self.encode_handles.remove(id);
                Response::error(id, error_codes::ENCODE_ERROR, e.to_string())
            }
        }
    }

    fn handle_cancel_encode(&mut self, id: &str) -> Response {
        if let Some(mut job) = self.encode_jobs.remove(id) {
            job.cancel();
            self.encode_handles.remove(id);
            Response::ok(id, serde_json::json!({"cancelled": true}))
        } else {
            Response::error(id, error_codes::ENCODE_NOT_STARTED, "No active encode job")
        }
    }

    fn handle_close(&self, id: &str, file_id: &str) -> Response {
        // Remove from cache
        self.state.frame_cache.remove_file(file_id);

        // Close decoder
        if self.state.decoder_pool.close(file_id) {
            info!("Closed file: {}", file_id);
            Response::ok(id, serde_json::json!({"closed": true}))
        } else {
            Response::error(id, error_codes::FILE_NOT_OPEN, "File not open")
        }
    }

    fn handle_info(&self, id: &str) -> Response {
        let cache_stats = self.state.frame_cache.stats();
        let hw_accel = detect_hw_accel();

        let info = SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            ffmpeg_version: "7.x".to_string(), // TODO: Get actual version
            hw_accel,
            cache_used_mb: cache_stats.used_mb(),
            cache_max_mb: cache_stats.max_mb(),
            open_files: self.state.decoder_pool.open_count(),
        };

        Response::ok(id, serde_json::to_value(info).unwrap())
    }

    async fn handle_download_youtube(&self, id: &str, url: &str, output_dir: Option<&str>) -> Response {
        use std::process::Stdio;
        use tokio::process::Command as TokioCommand;

        // Validate URL (basic check for YouTube)
        if !url.contains("youtube.com") && !url.contains("youtu.be") {
            return Response::error(id, error_codes::INVALID_URL, "Not a valid YouTube URL");
        }

        // Determine output directory
        let download_dir = if let Some(dir) = output_dir {
            std::path::PathBuf::from(dir)
        } else {
            // Default to ~/Downloads/MasterSelects
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            std::path::PathBuf::from(home).join("Downloads").join("MasterSelects")
        };

        // Create directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(&download_dir) {
            return Response::error(id, error_codes::PERMISSION_DENIED, format!("Cannot create download directory: {}", e));
        }

        info!("Downloading YouTube video: {} to {:?}", url, download_dir);

        // Run yt-dlp
        let output_template = download_dir.join("%(title)s.%(ext)s").to_string_lossy().to_string();

        // Prefer H.264 (avc1) over AV1 for better export compatibility
        // AV1 is very slow with HTMLVideoElement seeking
        let result = TokioCommand::new("yt-dlp")
            .args([
                "-f", "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", &output_template,
                "--print", "after_move:filepath",
                "--no-playlist",
                url,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        match result {
            Ok(output) => {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let output_path = stdout.lines().last().unwrap_or("").trim();

                    if output_path.is_empty() {
                        return Response::error(id, error_codes::DOWNLOAD_FAILED, "yt-dlp did not return output path");
                    }

                    info!("Download complete: {}", output_path);
                    Response::ok(id, serde_json::json!({
                        "path": output_path
                    }))
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    warn!("yt-dlp failed: {}", stderr);
                    Response::error(id, error_codes::DOWNLOAD_FAILED, format!("yt-dlp error: {}", stderr.lines().last().unwrap_or("Unknown error")))
                }
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Response::error(id, error_codes::YTDLP_NOT_FOUND, "yt-dlp not found. Please install it: pip install yt-dlp")
                } else {
                    Response::error(id, error_codes::DOWNLOAD_FAILED, format!("Failed to run yt-dlp: {}", e))
                }
            }
        }
    }

    fn handle_get_file(&self, id: &str, path: &str) -> Response {
        use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

        let path = std::path::Path::new(path);

        // Security: Only allow absolute paths
        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        // Security: Only allow files in allowed directories
        let allowed_prefixes = [
            std::env::var("HOME").unwrap_or_default() + "/Downloads",
            "/tmp".to_string(),
        ];

        let path_str = path.to_string_lossy();
        let is_allowed = allowed_prefixes.iter().any(|prefix| path_str.starts_with(prefix));

        if !is_allowed {
            return Response::error(id, error_codes::PERMISSION_DENIED, "File path not in allowed directory");
        }

        if !path.exists() {
            return Response::error(id, error_codes::FILE_NOT_FOUND, format!("File not found: {}", path.display()));
        }

        // Read file
        match std::fs::read(path) {
            Ok(data) => {
                info!("Serving file: {} ({} bytes)", path.display(), data.len());
                let data_base64 = BASE64.encode(&data);
                Response::ok(id, serde_json::json!({
                    "size": data.len(),
                    "path": path.display().to_string(),
                    "data": data_base64
                }))
            }
            Err(e) => Response::error(id, error_codes::FILE_NOT_FOUND, format!("Cannot read file: {}", e)),
        }
    }
}
