//! Muxer error types.

use thiserror::Error;

/// Errors that can occur during MP4 muxing.
#[derive(Error, Debug)]
pub enum MuxError {
    /// I/O error during file write.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid muxer configuration.
    #[error("Invalid muxer config: {0}")]
    InvalidConfig(String),

    /// Track-related error (e.g. unknown track ID, duplicate track).
    #[error("Track error: {0}")]
    TrackError(String),

    /// Internal buffer exceeded capacity.
    #[error("Buffer full: {0}")]
    BufferFull(String),
}

/// Convenience Result type for mux operations.
pub type MuxResult<T> = Result<T, MuxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mux_error_display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let mux_err = MuxError::from(io_err);
        assert!(mux_err.to_string().contains("IO error"));
        assert!(mux_err.to_string().contains("file not found"));
    }

    #[test]
    fn mux_error_display_invalid_config() {
        let err = MuxError::InvalidConfig("missing codec".into());
        assert_eq!(err.to_string(), "Invalid muxer config: missing codec");
    }

    #[test]
    fn mux_error_display_track_error() {
        let err = MuxError::TrackError("track 5 not found".into());
        assert_eq!(err.to_string(), "Track error: track 5 not found");
    }

    #[test]
    fn mux_error_display_buffer_full() {
        let err = MuxError::BufferFull("sample table overflow".into());
        assert_eq!(err.to_string(), "Buffer full: sample table overflow");
    }

    #[test]
    fn mux_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let mux_err: MuxError = io_err.into();
        matches!(mux_err, MuxError::IoError(_));
    }
}
