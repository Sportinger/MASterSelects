//! Version migration — transforms older project JSON to the current format.
//!
//! The web app uses `version: 1` as its current format. This module provides
//! forward-compatible migration so that projects from older or newer formats
//! can be handled gracefully.

use tracing::{debug, info, warn};

use crate::error::{ProjectError, ProjectResult};

/// Current project format version.
pub const CURRENT_VERSION: u32 = 1;

/// Migrate a project JSON value to the current version in-place.
///
/// Returns the version string after migration. If the project is already at
/// the current version, no changes are made.
pub fn migrate_project(value: &mut serde_json::Value) -> ProjectResult<String> {
    let obj = value
        .as_object_mut()
        .ok_or_else(|| ProjectError::InvalidProject {
            reason: "project root must be a JSON object".into(),
        })?;

    // Determine current version
    let version = extract_version(obj)?;

    if version > CURRENT_VERSION {
        return Err(ProjectError::UnsupportedVersion {
            version: version.to_string(),
        });
    }

    if version == CURRENT_VERSION {
        debug!(
            version,
            "Project is at current version, no migration needed"
        );
        return Ok(version.to_string());
    }

    // Apply sequential migrations
    let mut current = version;

    while current < CURRENT_VERSION {
        let next = current + 1;
        info!(from = current, to = next, "Migrating project");

        match current {
            0 => migrate_v0_to_v1(obj)?,
            other => {
                return Err(ProjectError::MigrationFailed {
                    from: other.to_string(),
                    to: (other + 1).to_string(),
                    reason: format!("no migration path from version {other}"),
                });
            }
        }

        current = next;
    }

    // Update version field
    obj.insert(
        "version".to_string(),
        serde_json::Value::Number(CURRENT_VERSION.into()),
    );

    info!(
        from = version,
        to = CURRENT_VERSION,
        "Project migration complete"
    );

    Ok(CURRENT_VERSION.to_string())
}

/// Extract the version number from a project JSON object.
fn extract_version(obj: &serde_json::Map<String, serde_json::Value>) -> ProjectResult<u32> {
    match obj.get("version") {
        Some(serde_json::Value::Number(n)) => {
            n.as_u64()
                .map(|v| v as u32)
                .ok_or_else(|| ProjectError::InvalidProject {
                    reason: "version must be a positive integer".into(),
                })
        }
        Some(serde_json::Value::String(s)) => {
            s.parse::<u32>().map_err(|_| ProjectError::InvalidProject {
                reason: format!("cannot parse version string: {s}"),
            })
        }
        Some(_) => Err(ProjectError::InvalidProject {
            reason: "version field has unexpected type".into(),
        }),
        None => {
            warn!("Project has no version field, assuming version 0");
            Ok(0)
        }
    }
}

/// Migrate from version 0 (no version field or version: 0) to version 1.
///
/// Version 0 represents hypothetical legacy projects that predate the
/// versioned format. This migration ensures required fields exist with defaults.
fn migrate_v0_to_v1(obj: &mut serde_json::Map<String, serde_json::Value>) -> ProjectResult<()> {
    // Ensure "settings" exists
    if !obj.contains_key("settings") {
        obj.insert(
            "settings".to_string(),
            serde_json::json!({
                "width": 1920,
                "height": 1080,
                "frameRate": 30.0,
                "sampleRate": 48000
            }),
        );
    }

    // Ensure "compositions" exists (might be called "timelines" in v0)
    if !obj.contains_key("compositions") {
        if let Some(timelines) = obj.remove("timelines") {
            obj.insert("compositions".to_string(), timelines);
        } else {
            obj.insert(
                "compositions".to_string(),
                serde_json::Value::Array(Vec::new()),
            );
        }
    }

    // Ensure "media" exists (might be called "files" in v0)
    if !obj.contains_key("media") {
        if let Some(files) = obj.remove("files") {
            obj.insert("media".to_string(), files);
        } else {
            obj.insert("media".to_string(), serde_json::Value::Array(Vec::new()));
        }
    }

    // Ensure "folders" exists
    if !obj.contains_key("folders") {
        obj.insert("folders".to_string(), serde_json::Value::Array(Vec::new()));
    }

    // Ensure required string fields exist
    ensure_string_field(obj, "name", "Untitled Project");
    ensure_string_field(obj, "createdAt", "1970-01-01T00:00:00Z");
    ensure_string_field(obj, "updatedAt", "1970-01-01T00:00:00Z");

    // Ensure required array fields exist
    if !obj.contains_key("activeCompositionId") {
        obj.insert("activeCompositionId".to_string(), serde_json::Value::Null);
    }
    if !obj.contains_key("openCompositionIds") {
        obj.insert(
            "openCompositionIds".to_string(),
            serde_json::Value::Array(Vec::new()),
        );
    }
    if !obj.contains_key("expandedFolderIds") {
        obj.insert(
            "expandedFolderIds".to_string(),
            serde_json::Value::Array(Vec::new()),
        );
    }

    Ok(())
}

/// Ensure a string field exists with a default value.
fn ensure_string_field(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: &str,
) {
    if !obj.contains_key(key) {
        obj.insert(
            key.to_string(),
            serde_json::Value::String(default.to_string()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_version_no_migration() {
        let mut value = serde_json::json!({
            "version": 1,
            "name": "Test",
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

        let version = migrate_project(&mut value).expect("migrate");
        assert_eq!(version, "1");
    }

    #[test]
    fn future_version_rejected() {
        let mut value = serde_json::json!({
            "version": 999,
            "name": "Future"
        });

        let err = migrate_project(&mut value).unwrap_err();
        assert!(matches!(err, ProjectError::UnsupportedVersion { .. }));
    }

    #[test]
    fn v0_to_v1_adds_missing_fields() {
        let mut value = serde_json::json!({
            "name": "Legacy Project"
        });

        let version = migrate_project(&mut value).expect("migrate");
        assert_eq!(version, "1");

        let obj = value.as_object().unwrap();
        assert_eq!(obj["version"], 1);
        assert!(obj.contains_key("settings"));
        assert!(obj.contains_key("compositions"));
        assert!(obj.contains_key("media"));
        assert!(obj.contains_key("folders"));
        assert!(obj.contains_key("activeCompositionId"));
        assert!(obj.contains_key("openCompositionIds"));
        assert!(obj.contains_key("expandedFolderIds"));
    }

    #[test]
    fn v0_renames_timelines_to_compositions() {
        let mut value = serde_json::json!({
            "name": "Old Format",
            "timelines": [{ "id": "t1", "name": "Main" }]
        });

        let _ = migrate_project(&mut value).expect("migrate");
        let obj = value.as_object().unwrap();
        assert!(obj.contains_key("compositions"));
        assert!(!obj.contains_key("timelines"));

        let comps = obj["compositions"].as_array().unwrap();
        assert_eq!(comps.len(), 1);
    }

    #[test]
    fn v0_renames_files_to_media() {
        let mut value = serde_json::json!({
            "name": "Old Format",
            "files": [{ "id": "f1", "name": "clip.mp4" }]
        });

        let _ = migrate_project(&mut value).expect("migrate");
        let obj = value.as_object().unwrap();
        assert!(obj.contains_key("media"));
        assert!(!obj.contains_key("files"));

        let media = obj["media"].as_array().unwrap();
        assert_eq!(media.len(), 1);
    }

    #[test]
    fn v0_adds_default_settings() {
        let mut value = serde_json::json!({
            "name": "No Settings"
        });

        let _ = migrate_project(&mut value).expect("migrate");
        let settings = &value["settings"];
        assert_eq!(settings["width"], 1920);
        assert_eq!(settings["height"], 1080);
        assert_eq!(settings["frameRate"], 30.0);
    }

    #[test]
    fn v0_preserves_existing_name() {
        let mut value = serde_json::json!({
            "name": "My Custom Name"
        });

        let _ = migrate_project(&mut value).expect("migrate");
        assert_eq!(value["name"], "My Custom Name");
    }

    #[test]
    fn v0_adds_default_name_when_missing() {
        let mut value = serde_json::json!({});

        let _ = migrate_project(&mut value).expect("migrate");
        assert_eq!(value["name"], "Untitled Project");
    }

    #[test]
    fn version_string_parsed() {
        let mut value = serde_json::json!({
            "version": "1",
            "name": "String Version",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "settings": { "width": 1920, "height": 1080, "frameRate": 30.0, "sampleRate": 48000 },
            "media": [],
            "compositions": [],
            "folders": [],
            "activeCompositionId": null,
            "openCompositionIds": [],
            "expandedFolderIds": []
        });

        let version = migrate_project(&mut value).expect("migrate");
        assert_eq!(version, "1");
    }

    #[test]
    fn non_object_root_rejected() {
        let mut value = serde_json::json!([1, 2, 3]);
        let err = migrate_project(&mut value).unwrap_err();
        assert!(matches!(err, ProjectError::InvalidProject { .. }));
    }
}
