//! Error types for the project crate (thiserror-based).

use thiserror::Error;

/// Errors that can occur during project file operations.
#[derive(Error, Debug)]
pub enum ProjectError {
    /// File I/O error (read, write, path resolution).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Project version is not supported or is from a newer format.
    #[error("Unsupported project version: {version}")]
    UnsupportedVersion { version: String },

    /// Project file is missing required fields.
    #[error("Invalid project file: {reason}")]
    InvalidProject { reason: String },

    /// Migration from an older format failed.
    #[error("Migration failed from version {from} to {to}: {reason}")]
    MigrationFailed {
        from: String,
        to: String,
        reason: String,
    },

    /// The project file path does not exist or is not a file.
    #[error("Project file not found: {path}")]
    NotFound { path: String },
}

/// Convenience Result type for project operations.
pub type ProjectResult<T> = Result<T, ProjectError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let err = ProjectError::UnsupportedVersion {
            version: "99.0".into(),
        };
        assert!(err.to_string().contains("99.0"));

        let err = ProjectError::InvalidProject {
            reason: "missing name".into(),
        };
        assert!(err.to_string().contains("missing name"));

        let err = ProjectError::MigrationFailed {
            from: "0".into(),
            to: "1".into(),
            reason: "unknown field".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("0") && msg.contains("1") && msg.contains("unknown field"));

        let err = ProjectError::NotFound {
            path: "/tmp/missing.msp".into(),
        };
        assert!(err.to_string().contains("missing.msp"));
    }

    #[test]
    fn io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let proj_err: ProjectError = io_err.into();
        assert!(matches!(proj_err, ProjectError::Io(_)));
    }

    #[test]
    fn json_error_conversion() {
        let result: Result<crate::types::ProjectFile, _> = serde_json::from_str("not json");
        let json_err = result.unwrap_err();
        let proj_err: ProjectError = json_err.into();
        assert!(matches!(proj_err, ProjectError::Json(_)));
    }
}
