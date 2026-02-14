//! Project serialization — writing `ProjectFile` to JSON files.

use std::path::Path;

use tracing::{debug, info};

use crate::error::{ProjectError, ProjectResult};
use crate::types::ProjectFile;

/// Serialize a project to a pretty-printed JSON string.
pub fn to_json_string(project: &ProjectFile) -> ProjectResult<String> {
    let json = serde_json::to_string_pretty(project)?;
    debug!(
        project_name = %project.name,
        json_len = json.len(),
        "Serialized project to JSON"
    );
    Ok(json)
}

/// Serialize a project to a compact (non-pretty) JSON string.
pub fn to_json_string_compact(project: &ProjectFile) -> ProjectResult<String> {
    let json = serde_json::to_string(project)?;
    debug!(
        project_name = %project.name,
        json_len = json.len(),
        "Serialized project to compact JSON"
    );
    Ok(json)
}

/// Save a project to a file at the given path.
///
/// The file will be written atomically: data is first written to a temporary
/// file in the same directory, then renamed to the target path. This prevents
/// data loss if the process crashes or is interrupted during write.
pub fn save_project(project: &ProjectFile, path: &Path) -> ProjectResult<()> {
    let json = to_json_string(project)?;

    // Write to a temporary file first, then rename for atomic write.
    let temp_path = path.with_extension("msp.tmp");

    std::fs::write(&temp_path, json.as_bytes()).map_err(|e| {
        tracing::error!(path = %temp_path.display(), error = %e, "Failed to write temp file");
        ProjectError::Io(e)
    })?;

    std::fs::rename(&temp_path, path).map_err(|e| {
        // If rename fails, try to clean up the temp file (best effort).
        let _ = std::fs::remove_file(&temp_path);
        tracing::error!(
            from = %temp_path.display(),
            to = %path.display(),
            error = %e,
            "Failed to rename temp file to target"
        );
        ProjectError::Io(e)
    })?;

    info!(
        project_name = %project.name,
        path = %path.display(),
        "Project saved successfully"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ProjectFile, ProjectSettings};

    fn sample_project() -> ProjectFile {
        ProjectFile::new("Save Test", ProjectSettings::default())
    }

    #[test]
    fn to_json_string_produces_valid_json() {
        let project = sample_project();
        let json = to_json_string(&project).expect("serialize");

        // Should be valid JSON that deserializes back
        let _: serde_json::Value = serde_json::from_str(&json).expect("parse as Value");
        assert!(json.contains("Save Test"));
        assert!(json.contains("\"version\": 1"));
    }

    #[test]
    fn to_json_string_compact_is_smaller() {
        let project = sample_project();
        let pretty = to_json_string(&project).expect("pretty");
        let compact = to_json_string_compact(&project).expect("compact");
        assert!(compact.len() < pretty.len());
    }

    #[test]
    fn save_project_creates_file() {
        let dir = std::env::temp_dir().join("ms_project_save_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_project.msp");

        let project = sample_project();
        save_project(&project, &path).expect("save");

        assert!(path.exists());
        let contents = std::fs::read_to_string(&path).expect("read");
        assert!(contents.contains("Save Test"));

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn save_project_atomic_no_temp_residue() {
        let dir = std::env::temp_dir().join("ms_project_atomic_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("atomic.msp");
        let temp_path = path.with_extension("msp.tmp");

        let project = sample_project();
        save_project(&project, &path).expect("save");

        // Temp file should not remain after successful save
        assert!(!temp_path.exists());
        assert!(path.exists());

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn save_project_roundtrip() {
        let dir = std::env::temp_dir().join("ms_project_roundtrip_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("roundtrip.msp");

        let project = sample_project();
        save_project(&project, &path).expect("save");

        let contents = std::fs::read_to_string(&path).expect("read");
        let loaded: ProjectFile = serde_json::from_str(&contents).expect("deserialize");
        assert_eq!(loaded.name, "Save Test");
        assert_eq!(loaded.version, 1);

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
