//! Cross-platform utility functions

use std::path::PathBuf;

/// Get the default download directory for videos
pub fn get_download_dir() -> PathBuf {
    let base = std::env::temp_dir();
    base.join("masterselects-downloads")
}

/// Get the default project root directory
/// Can be overridden via MASTERSELECTS_PROJECT_ROOT env var
pub fn get_project_root() -> PathBuf {
    if let Ok(custom) = std::env::var("MASTERSELECTS_PROJECT_ROOT") {
        let p = PathBuf::from(custom);
        if p.is_absolute() {
            return p;
        }
    }

    if let Some(docs) = dirs::document_dir() {
        return docs.join("MasterSelects");
    }

    // Fallback: home directory
    if let Some(home) = dirs::home_dir() {
        return home.join("MasterSelects");
    }

    PathBuf::from("MasterSelects")
}

/// Get allowed file serving prefixes (for security)
pub fn get_allowed_prefixes() -> Vec<PathBuf> {
    let mut prefixes = Vec::new();

    // User's Downloads folder (cross-platform)
    if let Some(downloads) = dirs::download_dir() {
        prefixes.push(downloads);
    }

    // Temp directory
    prefixes.push(std::env::temp_dir());

    // On Unix, also allow /tmp explicitly
    #[cfg(unix)]
    {
        prefixes.push(PathBuf::from("/tmp"));
    }

    // Documents directory (for project persistence)
    if let Some(docs) = dirs::document_dir() {
        prefixes.push(docs);
    }

    // MasterSelects project root (may be custom via env var)
    let project_root = get_project_root();
    if !prefixes.iter().any(|p| project_root.starts_with(p)) {
        prefixes.push(project_root);
    }

    // User's Videos folder (for media file serving)
    if let Some(videos) = dirs::video_dir() {
        prefixes.push(videos);
    }

    // User's Desktop (drag & drop sources)
    if let Some(desktop) = dirs::desktop_dir() {
        prefixes.push(desktop);
    }

    // Home directory (broad fallback for media anywhere under ~/)
    if let Some(home) = dirs::home_dir() {
        prefixes.push(home);
    }

    prefixes
}

/// Check if a path is within allowed directories
pub fn is_path_allowed(path: &std::path::Path) -> bool {
    let allowed = get_allowed_prefixes();

    // Normalize for case-insensitive comparison on Windows
    // Also normalize path separators (frontend may send forward slashes)
    #[cfg(windows)]
    {
        let path_str = path.to_string_lossy().to_lowercase().replace('/', "\\");
        return allowed.iter().any(|prefix| {
            let prefix_str = prefix.to_string_lossy().to_lowercase().replace('/', "\\");
            path_str.starts_with(&*prefix_str)
        });
    }

    #[cfg(not(windows))]
    {
        let path_str = path.to_string_lossy();
        return allowed.iter().any(|prefix| {
            let prefix_str = prefix.to_string_lossy();
            path_str.starts_with(prefix_str.as_ref())
        });
    }
}
