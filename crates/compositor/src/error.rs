//! Compositor error types.

use thiserror::Error;

/// Errors that can occur during compositing.
#[derive(Debug, Error)]
pub enum CompositorError {
    /// A GPU backend operation failed.
    #[error("GPU error: {0}")]
    Gpu(#[from] ms_common::GpuError),

    /// A required source frame was not found in the frame map.
    #[error("Missing source frame: {0}")]
    MissingSource(String),

    /// An error in the render pipeline (buffer management, etc.).
    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_source_error_displays_id() {
        let err = CompositorError::MissingSource("clip_42".into());
        let msg = err.to_string();
        assert!(msg.contains("clip_42"));
    }

    #[test]
    fn pipeline_error_displays_message() {
        let err = CompositorError::Pipeline("buffer swap failed".into());
        let msg = err.to_string();
        assert!(msg.contains("buffer swap failed"));
    }

    #[test]
    fn gpu_error_converts() {
        let gpu_err = ms_common::GpuError::AllocFailed { size: 1024 };
        let err: CompositorError = gpu_err.into();
        assert!(matches!(err, CompositorError::Gpu(_)));
    }
}
