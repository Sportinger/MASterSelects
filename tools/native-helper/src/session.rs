//! Per-connection session management

use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::download;
use crate::protocol::{error_codes, Command, Response, SystemInfo};
use crate::utils;

/// Generate a random auth token
pub fn generate_auth_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
        .collect()
}

/// Shared application state
pub struct AppState {
    pub auth_token: Option<String>,
}

impl AppState {
    pub fn new(auth_token: Option<String>) -> Self {
        Self { auth_token }
    }
}

/// Per-connection session
pub struct Session {
    state: Arc<AppState>,
    authenticated: bool,
}

impl Session {
    pub fn new(state: Arc<AppState>) -> Self {
        let authenticated = state.auth_token.is_none();

        Self {
            state,
            authenticated,
        }
    }

    /// Handle a command, return response
    /// Note: Download/ListFormats commands are handled directly in server.rs for WsSender access
    pub async fn handle_command(&mut self, cmd: Command) -> Option<Response> {
        // Auth required for most commands
        if !self.authenticated {
            if let Command::Auth { .. } = cmd {
                // Allow auth command
            } else {
                return Some(Response::error(
                    "",
                    error_codes::AUTH_REQUIRED,
                    "Authentication required",
                ));
            }
        }

        match cmd {
            Command::Auth { id, token } => Some(self.handle_auth(&id, &token)),

            Command::Info { id } => Some(self.handle_info(&id)),

            Command::Ping { id } => {
                Some(Response::ok(&id, serde_json::json!({"pong": true})))
            }

            Command::GetFile { id, path } => Some(self.handle_get_file(&id, &path)),

            Command::Locate {
                id,
                filename,
                search_dirs,
            } => Some(self.handle_locate(&id, &filename, &search_dirs)),

            // Download commands are handled in server.rs with WsSender
            Command::DownloadYoutube { id, .. }
            | Command::Download { id, .. }
            | Command::ListFormats { id, .. } => Some(Response::error(
                &id,
                error_codes::INTERNAL_ERROR,
                "Download commands should be handled by server",
            )),
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

    fn handle_info(&self, id: &str) -> Response {
        let ytdlp_available = download::find_ytdlp().is_some();

        let info = SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            ytdlp_available,
            download_dir: utils::get_download_dir().to_string_lossy().to_string(),
        };

        Response::ok(id, serde_json::to_value(info).unwrap())
    }

    fn handle_locate(&self, id: &str, filename: &str, extra_dirs: &[String]) -> Response {
        // Sanitize filename: reject path traversal attempts
        if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
            return Response::error(
                id,
                error_codes::INVALID_PATH,
                "Filename must not contain path separators",
            );
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
        if let Some(d) = dirs::desktop_dir() {
            search_dirs.push(d);
        }
        if let Some(d) = dirs::download_dir() {
            search_dirs.push(d);
        }
        if let Some(d) = dirs::video_dir() {
            search_dirs.push(d);
        }
        if let Some(d) = dirs::document_dir() {
            search_dirs.push(d);
        }
        if let Some(d) = dirs::home_dir() {
            search_dirs.push(d);
        }

        // Search each directory recursively (max depth 4 to avoid long scans)
        for dir in &search_dirs {
            if let Some(path) = Self::find_file_recursive(dir, filename, 0, 4) {
                info!("Located file '{}' at {}", filename, path.display());
                return Response::ok(
                    id,
                    serde_json::json!({
                        "found": true,
                        "path": path.to_string_lossy()
                    }),
                );
            }
        }

        debug!(
            "File '{}' not found in {} directories",
            filename,
            search_dirs.len()
        );
        Response::ok(
            id,
            serde_json::json!({
                "found": false,
                "searched": search_dirs.iter().map(|d| d.to_string_lossy().to_string()).collect::<Vec<_>>()
            }),
        )
    }

    /// Recursively search for a file by name, up to max_depth levels deep.
    fn find_file_recursive(
        dir: &std::path::Path,
        filename: &str,
        depth: u32,
        max_depth: u32,
    ) -> Option<PathBuf> {
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
                    if name.starts_with('.')
                        || name == "node_modules"
                        || name == "$RECYCLE.BIN"
                        || name == "System Volume Information"
                    {
                        continue;
                    }
                }
                if let Some(found) =
                    Self::find_file_recursive(&path, filename, depth + 1, max_depth)
                {
                    return Some(found);
                }
            }
        }

        None
    }

    fn handle_get_file(&self, id: &str, path: &str) -> Response {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !utils::is_path_allowed(path) {
            return Response::error(
                id,
                error_codes::PERMISSION_DENIED,
                "File path not in allowed directory",
            );
        }

        if !path.exists() {
            return Response::error(
                id,
                error_codes::FILE_NOT_FOUND,
                format!("File not found: {}", path.display()),
            );
        }

        match std::fs::read(path) {
            Ok(data) => {
                info!("Serving file: {} ({} bytes)", path.display(), data.len());
                let data_base64 = BASE64.encode(&data);
                Response::ok(
                    id,
                    serde_json::json!({
                        "size": data.len(),
                        "path": path.display().to_string(),
                        "data": data_base64
                    }),
                )
            }
            Err(e) => Response::error(
                id,
                error_codes::FILE_NOT_FOUND,
                format!("Cannot read file: {}", e),
            ),
        }
    }
}
