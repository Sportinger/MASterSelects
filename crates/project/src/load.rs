//! Project deserialization — loading `ProjectFile` from JSON files.

use std::path::Path;

use tracing::{debug, info, warn};

use crate::error::{ProjectError, ProjectResult};
use crate::migrate::migrate_project;
use crate::types::ProjectFile;

/// Deserialize a project from a JSON string.
///
/// Runs version migration if the project uses an older format.
pub fn from_json_string(json: &str) -> ProjectResult<ProjectFile> {
    // First parse as generic Value to check/migrate version
    let mut value: serde_json::Value = serde_json::from_str(json)?;

    let migrated_version = migrate_project(&mut value)?;
    debug!(version = %migrated_version, "Project version after migration");

    // Now deserialize the (possibly migrated) value into our typed struct
    let project: ProjectFile = serde_json::from_value(value)?;

    debug!(
        project_name = %project.name,
        version = project.version,
        media_count = project.media.len(),
        composition_count = project.compositions.len(),
        "Deserialized project from JSON"
    );

    validate_project(&project)?;

    Ok(project)
}

/// Load a project from a file at the given path.
///
/// Reads the file contents and parses as JSON, running version migration
/// if necessary.
pub fn load_project(path: &Path) -> ProjectResult<ProjectFile> {
    if !path.exists() {
        return Err(ProjectError::NotFound {
            path: path.display().to_string(),
        });
    }

    let json = std::fs::read_to_string(path).map_err(|e| {
        tracing::error!(path = %path.display(), error = %e, "Failed to read project file");
        ProjectError::Io(e)
    })?;

    let project = from_json_string(&json)?;

    info!(
        project_name = %project.name,
        path = %path.display(),
        compositions = project.compositions.len(),
        media_files = project.media.len(),
        "Project loaded successfully"
    );

    Ok(project)
}

/// Validate basic structural requirements of a loaded project.
fn validate_project(project: &ProjectFile) -> ProjectResult<()> {
    if project.name.is_empty() {
        warn!("Project has empty name");
        return Err(ProjectError::InvalidProject {
            reason: "project name is empty".into(),
        });
    }

    if project.settings.width == 0 || project.settings.height == 0 {
        return Err(ProjectError::InvalidProject {
            reason: format!(
                "invalid resolution: {}x{}",
                project.settings.width, project.settings.height
            ),
        });
    }

    if project.settings.frame_rate <= 0.0 {
        return Err(ProjectError::InvalidProject {
            reason: format!("invalid frame rate: {}", project.settings.frame_rate),
        });
    }

    // Validate that clip track IDs reference existing tracks within their composition
    for comp in &project.compositions {
        let track_ids: std::collections::HashSet<&str> =
            comp.tracks.iter().map(|t| t.id.as_str()).collect();

        for clip in &comp.clips {
            if !track_ids.contains(clip.track_id.as_str()) {
                warn!(
                    clip_id = %clip.id,
                    track_id = %clip.track_id,
                    composition = %comp.name,
                    "Clip references non-existent track"
                );
                // This is a warning, not an error, since we want to be lenient
                // with files from other versions
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::save::{save_project, to_json_string};
    use crate::types::{
        ClipData, CompositionData, ProjectFile, ProjectSettings, TrackData, TrackType,
        TransformData,
    };

    fn sample_project() -> ProjectFile {
        let mut project = ProjectFile::new("Load Test", ProjectSettings::default());
        let mut comp = CompositionData::new("comp-1", "Main", 1920, 1080, 30.0);
        comp.tracks.push(TrackData {
            id: "v1".into(),
            name: "Video 1".into(),
            track_type: TrackType::Video,
            height: 60,
            locked: false,
            visible: true,
            muted: false,
            solo: false,
        });
        comp.clips.push(ClipData {
            id: "clip-1".into(),
            track_id: "v1".into(),
            name: None,
            media_id: "media-1".into(),
            start_time: 0.0,
            duration: 5.0,
            in_point: 0.0,
            out_point: 5.0,
            transform: TransformData::default(),
            effects: vec![],
            masks: vec![],
            keyframes: vec![],
            volume: 1.0,
            audio_enabled: true,
            reversed: false,
            disabled: false,
            is_composition: None,
            composition_id: None,
            source_type: Some("video".into()),
            natural_duration: Some(10.0),
            linked_clip_id: None,
            linked_group_id: None,
            text_properties: None,
            solid_color: None,
        });
        project.compositions.push(comp);
        project
    }

    #[test]
    fn from_json_string_basic() {
        let project = sample_project();
        let json = to_json_string(&project).expect("serialize");
        let loaded = from_json_string(&json).expect("deserialize");

        assert_eq!(loaded.name, "Load Test");
        assert_eq!(loaded.compositions.len(), 1);
        assert_eq!(loaded.compositions[0].clips.len(), 1);
    }

    #[test]
    fn load_project_file_roundtrip() {
        let dir = std::env::temp_dir().join("ms_project_load_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("load_test.msp");

        let project = sample_project();
        save_project(&project, &path).expect("save");

        let loaded = load_project(&path).expect("load");
        assert_eq!(loaded.name, "Load Test");
        assert_eq!(loaded.compositions.len(), 1);
        assert_eq!(loaded.compositions[0].tracks.len(), 1);

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn load_project_nonexistent_file() {
        let path = std::path::PathBuf::from("/nonexistent/path/project.msp");
        let err = load_project(&path).unwrap_err();
        assert!(matches!(err, ProjectError::NotFound { .. }));
    }

    #[test]
    fn from_json_string_invalid_json() {
        let result = from_json_string("this is not json");
        assert!(result.is_err());
    }

    #[test]
    fn validate_rejects_empty_name() {
        let json = serde_json::json!({
            "version": 1,
            "name": "",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "settings": {
                "width": 1920,
                "height": 1080,
                "frameRate": 30.0,
                "sampleRate": 48000
            },
            "media": [],
            "compositions": [],
            "folders": [],
            "activeCompositionId": null,
            "openCompositionIds": [],
            "expandedFolderIds": []
        });
        let result = from_json_string(&json.to_string());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn validate_rejects_zero_resolution() {
        let json = serde_json::json!({
            "version": 1,
            "name": "Bad",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "settings": {
                "width": 0,
                "height": 1080,
                "frameRate": 30.0,
                "sampleRate": 48000
            },
            "media": [],
            "compositions": [],
            "folders": [],
            "activeCompositionId": null,
            "openCompositionIds": [],
            "expandedFolderIds": []
        });
        let result = from_json_string(&json.to_string());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("resolution"));
    }

    #[test]
    fn validate_rejects_negative_framerate() {
        let json = serde_json::json!({
            "version": 1,
            "name": "Bad FPS",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "settings": {
                "width": 1920,
                "height": 1080,
                "frameRate": -1.0,
                "sampleRate": 48000
            },
            "media": [],
            "compositions": [],
            "folders": [],
            "activeCompositionId": null,
            "openCompositionIds": [],
            "expandedFolderIds": []
        });
        let result = from_json_string(&json.to_string());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("frame rate"));
    }

    #[test]
    fn from_json_string_preserves_ui_state() {
        let json = serde_json::json!({
            "version": 1,
            "name": "UI State Test",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "settings": {
                "width": 1920,
                "height": 1080,
                "frameRate": 30.0,
                "sampleRate": 48000
            },
            "media": [],
            "compositions": [],
            "folders": [],
            "activeCompositionId": null,
            "openCompositionIds": [],
            "expandedFolderIds": [],
            "uiState": {
                "dockLayout": { "panels": [] },
                "thumbnailsEnabled": true
            }
        });
        let loaded = from_json_string(&json.to_string()).expect("load");
        assert!(loaded.ui_state.is_some());
        let ui = loaded.ui_state.as_ref().unwrap();
        assert!(ui.get("thumbnailsEnabled").is_some());
    }
}
