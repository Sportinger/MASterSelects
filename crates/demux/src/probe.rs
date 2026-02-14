//! File probing — detect container format and extract stream info.

use ms_common::{ContainerFormat, DemuxError};
use std::path::Path;

/// Detect container format from file extension.
pub fn detect_format(path: &Path) -> Result<ContainerFormat, DemuxError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "mp4" | "m4v" | "mov" => Ok(ContainerFormat::Mp4),
        "mkv" => Ok(ContainerFormat::Mkv),
        "webm" => Ok(ContainerFormat::WebM),
        _ => Err(DemuxError::UnsupportedContainer),
    }
}
