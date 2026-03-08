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

            // File system commands
            Command::WriteFile { id, path, data, encoding } => {
                Some(self.handle_write_file(&id, &path, &data, encoding.as_deref()))
            }
            Command::CreateDir { id, path, recursive } => {
                Some(self.handle_create_dir(&id, &path, recursive.unwrap_or(true)))
            }
            Command::ListDir { id, path } => Some(self.handle_list_dir(&id, &path)),
            Command::Delete { id, path, recursive } => {
                Some(self.handle_delete(&id, &path, recursive.unwrap_or(false)))
            }
            Command::Exists { id, path } => Some(self.handle_exists(&id, &path)),
            Command::Rename { id, old_path, new_path } => {
                Some(self.handle_rename(&id, &old_path, &new_path))
            }

            Command::PickFolder { id, title, default_path } => {
                let title = title.unwrap_or_else(|| "Select folder".to_string());
                let default_path = default_path.clone();

                // Must use spawn_blocking since rfd blocks the thread
                match tokio::task::spawn_blocking(move || {
                    let mut dialog = rfd::FileDialog::new().set_title(&title);
                    if let Some(ref dp) = default_path {
                        dialog = dialog.set_directory(dp);
                    }
                    dialog.pick_folder()
                })
                .await
                {
                    Ok(Some(path)) => Some(Response::ok(
                        &id,
                        serde_json::json!({ "path": path.to_string_lossy() }),
                    )),
                    Ok(None) => Some(Response::ok(
                        &id,
                        serde_json::json!({ "path": serde_json::Value::Null, "cancelled": true }),
                    )),
                    Err(e) => Some(Response::error(
                        &id,
                        error_codes::INTERNAL_ERROR,
                        format!("Folder picker failed: {}", e),
                    )),
                }
            }

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
            project_root: utils::get_project_root().to_string_lossy().to_string(),
            fs_commands: true,
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

    // ── File System Command Handlers ──

    fn handle_write_file(&self, id: &str, path: &str, data: &str, encoding: Option<&str>) -> Response {
        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !utils::is_path_allowed(path) {
            return Response::error(id, error_codes::PERMISSION_DENIED, "Path not in allowed directory");
        }

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return Response::error(id, error_codes::WRITE_FAILED, format!("Cannot create parent dirs: {}", e));
                }
            }
        }

        // Decode data
        let bytes = match encoding.unwrap_or("utf8") {
            "base64" => {
                use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
                match BASE64.decode(data) {
                    Ok(b) => b,
                    Err(e) => return Response::error(id, error_codes::INVALID_PATH, format!("Invalid base64: {}", e)),
                }
            }
            _ => data.as_bytes().to_vec(),
        };

        let size = bytes.len();

        // Atomic write: write to .tmp then rename
        let tmp_path = path.with_extension(format!(
            "{}.tmp",
            path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default()
        ));

        if let Err(e) = std::fs::write(&tmp_path, &bytes) {
            return Response::error(id, error_codes::WRITE_FAILED, format!("Write failed: {}", e));
        }

        if let Err(e) = std::fs::rename(&tmp_path, path) {
            // Rename failed — try direct write as fallback
            let _ = std::fs::remove_file(&tmp_path);
            if let Err(e2) = std::fs::write(path, &bytes) {
                return Response::error(id, error_codes::WRITE_FAILED, format!("Write failed: {} / {}", e, e2));
            }
        }

        info!("Wrote file: {} ({} bytes)", path.display(), size);
        Response::ok(id, serde_json::json!({ "written": true, "size": size }))
    }

    fn handle_create_dir(&self, id: &str, path: &str, recursive: bool) -> Response {
        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !utils::is_path_allowed(path) {
            return Response::error(id, error_codes::PERMISSION_DENIED, "Path not in allowed directory");
        }

        if path.exists() {
            if path.is_dir() {
                return Response::ok(id, serde_json::json!({ "created": true, "existed": true }));
            }
            return Response::error(id, error_codes::ALREADY_EXISTS, "A file exists at this path");
        }

        let result = if recursive {
            std::fs::create_dir_all(path)
        } else {
            std::fs::create_dir(path)
        };

        match result {
            Ok(()) => {
                info!("Created directory: {}", path.display());
                Response::ok(id, serde_json::json!({ "created": true, "existed": false }))
            }
            Err(e) => Response::error(id, error_codes::WRITE_FAILED, format!("Cannot create directory: {}", e)),
        }
    }

    fn handle_list_dir(&self, id: &str, path: &str) -> Response {
        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !utils::is_path_allowed(path) {
            return Response::error(id, error_codes::PERMISSION_DENIED, "Path not in allowed directory");
        }

        if !path.exists() || !path.is_dir() {
            return Response::error(id, error_codes::FILE_NOT_FOUND, "Directory not found");
        }

        let entries = match std::fs::read_dir(path) {
            Ok(e) => e,
            Err(e) => return Response::error(id, error_codes::INTERNAL_ERROR, format!("Cannot read directory: {}", e)),
        };

        let mut items = Vec::new();
        for entry in entries.flatten() {
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            let modified = metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            items.push(serde_json::json!({
                "name": entry.file_name().to_string_lossy(),
                "kind": if metadata.is_dir() { "directory" } else { "file" },
                "size": metadata.len(),
                "modified": modified,
            }));
        }

        Response::ok(id, serde_json::json!({ "entries": items, "count": items.len() }))
    }

    fn handle_delete(&self, id: &str, path: &str, recursive: bool) -> Response {
        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !utils::is_path_allowed(path) {
            return Response::error(id, error_codes::PERMISSION_DENIED, "Path not in allowed directory");
        }

        if !path.exists() {
            return Response::error(id, error_codes::FILE_NOT_FOUND, "Path not found");
        }

        let result = if path.is_dir() {
            if recursive {
                std::fs::remove_dir_all(path)
            } else {
                std::fs::remove_dir(path)
            }
        } else {
            std::fs::remove_file(path)
        };

        match result {
            Ok(()) => {
                info!("Deleted: {}", path.display());
                Response::ok(id, serde_json::json!({ "deleted": true }))
            }
            Err(e) => {
                let code = if e.kind() == std::io::ErrorKind::Other || e.to_string().contains("not empty") {
                    error_codes::DIR_NOT_EMPTY
                } else {
                    error_codes::INTERNAL_ERROR
                };
                Response::error(id, code, format!("Delete failed: {}", e))
            }
        }
    }

    fn handle_exists(&self, id: &str, path: &str) -> Response {
        let path = std::path::Path::new(path);

        if !path.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Path must be absolute");
        }

        if !utils::is_path_allowed(path) {
            return Response::error(id, error_codes::PERMISSION_DENIED, "Path not in allowed directory");
        }

        let kind = if !path.exists() {
            "none"
        } else if path.is_dir() {
            "directory"
        } else {
            "file"
        };

        Response::ok(id, serde_json::json!({ "exists": path.exists(), "kind": kind }))
    }

    fn handle_rename(&self, id: &str, old_path: &str, new_path: &str) -> Response {
        let old = std::path::Path::new(old_path);
        let new = std::path::Path::new(new_path);

        if !old.is_absolute() || !new.is_absolute() {
            return Response::error(id, error_codes::INVALID_PATH, "Paths must be absolute");
        }

        if !utils::is_path_allowed(old) || !utils::is_path_allowed(new) {
            return Response::error(id, error_codes::PERMISSION_DENIED, "Path not in allowed directory");
        }

        if !old.exists() {
            return Response::error(id, error_codes::FILE_NOT_FOUND, "Source path not found");
        }

        if new.exists() {
            return Response::error(id, error_codes::ALREADY_EXISTS, "Destination already exists");
        }

        // Ensure parent of destination exists
        if let Some(parent) = new.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return Response::error(id, error_codes::WRITE_FAILED, format!("Cannot create parent dirs: {}", e));
                }
            }
        }

        match std::fs::rename(old, new) {
            Ok(()) => {
                info!("Renamed: {} -> {}", old.display(), new.display());
                Response::ok(id, serde_json::json!({ "renamed": true }))
            }
            Err(e) => Response::error(id, error_codes::INTERNAL_ERROR, format!("Rename failed: {}", e)),
        }
    }
}
