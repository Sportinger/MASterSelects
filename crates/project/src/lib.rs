//! `ms-project` — Project file save/load for the MasterSelects native engine.
//!
//! This crate handles loading and saving project files in a JSON format
//! compatible with the web-based MasterSelects editor. It supports:
//!
//! - **Save/Load**: Serialize/deserialize `ProjectFile` to/from JSON
//! - **Migration**: Forward-compatible version migration for older project formats
//! - **Recent Projects**: Track and persist a list of recently opened projects
//! - **Auto-Save**: Timer-based dirty-state tracking for periodic saves
//!
//! # Usage
//!
//! ```rust,no_run
//! use ms_project::{load_project, save_project, ProjectFile, ProjectSettings};
//! use std::path::Path;
//!
//! // Create a new project
//! let project = ProjectFile::new("My Project", ProjectSettings::default());
//!
//! // Save to disk
//! save_project(&project, Path::new("project.msp")).unwrap();
//!
//! // Load from disk
//! let loaded = load_project(Path::new("project.msp")).unwrap();
//! assert_eq!(loaded.name, "My Project");
//! ```

pub mod autosave;
pub mod error;
pub mod load;
pub mod migrate;
pub mod recent;
pub mod save;
pub mod types;

// Re-export primary API at crate root
pub use autosave::AutoSaver;
pub use error::{ProjectError, ProjectResult};
pub use load::{from_json_string, load_project};
pub use migrate::{migrate_project, CURRENT_VERSION};
pub use recent::RecentProjects;
pub use save::{save_project, to_json_string, to_json_string_compact};
pub use types::{
    BezierHandles, ClipData, CompositionData, EffectData, FolderRef, KeyframeData, MarkerData,
    MaskData, MaskMode, MaskPosition, MaskVertex, MediaFileRef, MediaType, ProjectFile,
    ProjectSettings, RecentEntry, TangentPoint, TrackData, TrackType, TransformData,
};
