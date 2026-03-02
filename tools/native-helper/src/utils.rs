//! Cross-platform utility functions

use std::path::PathBuf;

/// Get the default download directory for videos
pub fn get_download_dir() -> PathBuf {
    let base = std::env::temp_dir();
    base.join("masterselects-downloads")
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

    prefixes
}

/// Check if a path is within allowed directories
pub fn is_path_allowed(path: &std::path::Path) -> bool {
    let allowed = get_allowed_prefixes();

    // Normalize for case-insensitive comparison on Windows
    #[cfg(windows)]
    {
        let path_str = path.to_string_lossy().to_lowercase();
        return allowed.iter().any(|prefix| {
            let prefix_str = prefix.to_string_lossy().to_lowercase();
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
