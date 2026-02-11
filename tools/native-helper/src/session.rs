//! Per-connection session management

use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::cache::FrameCache;
use crate::decoder::{DecoderPool, SharedDecoderPool, detect_hw_accel};
use crate::download;
use crate::encoder::{EncodeJob, EncodeJobHandle};
use crate::protocol::{
    error_codes, Command, Compression, PixelFormat, Response, SystemInfo,
    DecodedFrame, encode_frame_message,
};
use crate::utils;

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
        let authenticated = state.auth_token.is_none();

        Self {
            state,
            authenticated,
            encode_jobs: HashMap::new(),
            encode_handles: HashMap::new(),
            request_counter: 0,
        }
    }

    fn next_request_id(&mut self) -> u32 {
        self.request_counter += 1;
        self.request_counter
    }

    /// Handle a command, return response and optional binary data
    /// Note: Download/ListFormats commands are handled directly in server.rs for WsSender access
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
                info!("Decode request: file={} frame={}", file_id, frame);
                let result = self.handle_decode(&id, &file_id, frame, format, scale, compression);
                match result {
                    Ok(binary) => {
                        info!("Decode OK: frame={} size={} bytes", frame, binary.len());
                        (None, Some(binary))
                    },
                    Err(response) => {
                        warn!("Decode FAILED: frame={} err={:?}", frame, response);
                        (Some(response), None)
                    },
                }
            }

            Command::DecodeRange {
                id,
                file_id: _,
                start_frame: _,
                end_frame: _,
                priority: _,
            } => {
                let response = Response::error(&id, error_codes::INTERNAL_ERROR, "DecodeRange not yet implemented - use multiple Decode calls");
                (Some(response), None)
            }

            Command::Prefetch {
                file_id,
                around_frame,
                radius,
            } => {
                self.handle_prefetch(&file_id, around_frame, radius);
                (None, None)
            }

            Command::StartEncode {
                id,
                output,
                frame_count,
            } => {
                let response = self.handle_start_encode(&id, output, frame_count);
                (Some(response), None)
            }

            Command::EncodeFrame { id, frame_num: _ } => {
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

            Command::GetFile { id, path } => {
                let response = self.handle_get_file(&id, &path);
                (Some(response), None)
            }

            Command::Locate { id, filename, search_dirs } => {
                let response = self.handle_locate(&id, &filename, &search_dirs);
                (Some(response), None)
            }

            // Download commands are handled in server.rs with WsSender
            Command::DownloadYoutube { id, .. }
            | Command::Download { id, .. }
            | Command::ListFormats { id, .. } => {
                let response = Response::error(&id, error_codes::INTERNAL_ERROR, "Download commands should be handled by server");
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
                self.authenticated = true;
                Response::ok(id, serde_json::json!({"authenticated": true}))
            }
        }
    }

    fn handle_open(&self, id: &str, path: &str) -> Response {
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

        let decoder_pool = self.state.decoder_pool.clone();
        decoder_pool.touch(file_id);
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
        self.state.frame_cache.remove_file(file_id);

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
        let ytdlp_available = download::find_ytdlp().is_some();

        let info = SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            ffmpeg_version: "7.x".to_string(),
            hw_accel,
            cache_used_mb: cache_stats.used_mb(),
            cache_max_mb: cache_stats.max_mb(),
            open_files: self.state.decoder_pool.open_count(),
            ytdlp_available: Some(ytdlp_available),
            download_dir: Some(utils::get_download_dir().to_string_lossy().to_string()),
        };

        Response::ok(id, serde_json::to_value(info).unwrap())
    }

    fn handle_locate(&self, id: &str, filename: &str, extra_dirs: &[String]) -> Response {
        // Sanitize filename: reject path traversal attempts
        if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
            return Response::error(id, error_codes::INVALID_PATH, "Filename must not contain path separators");
        }

        // Build list of directories to search
        let mut search_dirs: Vec<PathBuf> = Vec::new();

        // Add extra dirs first (highest priority)
        for dir in extra_dirs {
            let p = PathBuf::from(dir);
            if p.is_absolute() && p.is_dir() {
                search_dirs.push(p);
            }
        }

        // Common user directories
        if let Some(d) = dirs::desktop_dir() { search_dirs.push(d); }
        if let Some(d) = dirs::download_dir() { search_dirs.push(d); }
        if let Some(d) = dirs::video_dir() { search_dirs.push(d); }
        if let Some(d) = dirs::document_dir() { search_dirs.push(d); }
        if let Some(d) = dirs::home_dir() { search_dirs.push(d); }

        // Search each directory recursively (max depth 4 to avoid long scans)
        for dir in &search_dirs {
            if let Some(path) = Self::find_file_recursive(dir, filename, 0, 4) {
                info!("Located file '{}' at {}", filename, path.display());
                return Response::ok(id, serde_json::json!({
                    "found": true,
                    "path": path.to_string_lossy()
                }));
            }
        }

        debug!("File '{}' not found in {} directories", filename, search_dirs.len());
        Response::ok(id, serde_json::json!({
            "found": false,
            "searched": search_dirs.iter().map(|d| d.to_string_lossy().to_string()).collect::<Vec<_>>()
        }))
    }

    /// Recursively search for a file by name, up to max_depth levels deep.
    fn find_file_recursive(dir: &std::path::Path, filename: &str, depth: u32, max_depth: u32) -> Option<PathBuf> {
        // Check direct child first
        let candidate = dir.join(filename);
        if candidate.is_file() {
            return Some(candidate);
        }

        // Recurse into subdirectories
        if depth >= max_depth {
            return None;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return None,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip hidden directories and system directories
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "node_modules" || name == "$RECYCLE.BIN" || name == "System Volume Information" {
                        continue;
                    }
                }
                if let Some(found) = Self::find_file_recursive(&path, filename, depth + 1, max_depth) {
                    return Some(found);
                }
            }
        }

        None
    }

    fn handle_get_file(&self, id: &str, path: &str) -> Response {
        use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !utils::is_path_allowed(path) {
            return Response::error(id, error_codes::PERMISSION_DENIED, "File path not in allowed directory");
        }

        if !path.exists() {
            return Response::error(id, error_codes::FILE_NOT_FOUND, format!("File not found: {}", path.display()));
        }

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
