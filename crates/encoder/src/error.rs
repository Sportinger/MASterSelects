//! Encoder-specific error types.
//!
//! These errors cover the encoder crate's internal operations beyond the
//! common `EncodeError` defined in `ms-common`. They provide more granular
//! error information for NVENC operations and export pipeline failures.

use thiserror::Error;

/// Errors that can occur when loading the NVENC library.
#[derive(Debug, Error)]
pub enum NvencLoadError {
    #[error("NVENC library not found: {0}")]
    LibraryNotFound(String),

    #[error("Required symbol not found: {0}")]
    SymbolNotFound(String),

    #[error("NVENC API version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
}

/// Errors specific to NVENC buffer operations.
#[derive(Debug, Error)]
pub enum BufferError {
    #[error("Failed to create input buffer: {0}")]
    InputCreationFailed(String),

    #[error("Failed to create output bitstream buffer: {0}")]
    OutputCreationFailed(String),

    #[error("Failed to register external resource: {0}")]
    RegisterFailed(String),

    #[error("Failed to lock bitstream: {0}")]
    LockFailed(String),

    #[error("Failed to unlock bitstream: {0}")]
    UnlockFailed(String),

    #[error("Buffer pool exhausted: all {count} buffers are in use")]
    PoolExhausted { count: usize },

    #[error("Failed to map input buffer: {0}")]
    MapFailed(String),

    #[error("Failed to unmap input buffer: {0}")]
    UnmapFailed(String),
}

/// Errors specific to the export pipeline.
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Export pipeline initialization failed: {0}")]
    InitFailed(String),

    #[error("Export cancelled by user")]
    Cancelled,

    #[error("Frame render failed at frame {frame}: {reason}")]
    RenderFailed { frame: u64, reason: String },

    #[error("Encode failed at frame {frame}: {reason}")]
    EncodeFailed { frame: u64, reason: String },

    #[error("Mux failed: {0}")]
    MuxFailed(String),

    #[error("Invalid export config: {0}")]
    InvalidConfig(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvenc_load_error_display() {
        let err = NvencLoadError::LibraryNotFound("nvEncodeAPI64.dll".to_string());
        assert!(err.to_string().contains("nvEncodeAPI64.dll"));
    }

    #[test]
    fn nvenc_version_mismatch_display() {
        let err = NvencLoadError::VersionMismatch {
            expected: 12,
            actual: 11,
        };
        let msg = err.to_string();
        assert!(msg.contains("12"));
        assert!(msg.contains("11"));
    }

    #[test]
    fn buffer_error_display() {
        let err = BufferError::PoolExhausted { count: 4 };
        assert!(err.to_string().contains("4"));
    }

    #[test]
    fn export_error_display() {
        let err = ExportError::Cancelled;
        assert_eq!(err.to_string(), "Export cancelled by user");
    }

    #[test]
    fn export_error_render_failed() {
        let err = ExportError::RenderFailed {
            frame: 42,
            reason: "GPU timeout".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("GPU timeout"));
    }

    #[test]
    fn export_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = ExportError::from(io_err);
        assert!(err.to_string().contains("file missing"));
    }
}
