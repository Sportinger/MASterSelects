//! Recent projects list — tracks recently opened project files.
//!
//! Stores a list of recently opened projects with their paths and names.
//! The list is persisted as a JSON file in a platform-appropriate data directory.

use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use crate::error::{ProjectError, ProjectResult};
use crate::types::RecentEntry;

/// Maximum number of recent project entries to keep.
const MAX_RECENT_ENTRIES: usize = 20;

/// File name for the persisted recent projects list.
const RECENT_PROJECTS_FILE: &str = "recent_projects.json";

/// Manages a list of recently opened projects.
#[derive(Clone, Debug)]
pub struct RecentProjects {
    entries: Vec<RecentEntry>,
    storage_path: PathBuf,
}

impl RecentProjects {
    /// Load recent projects from the default storage location.
    ///
    /// If the file does not exist or is invalid, returns an empty list.
    pub fn load() -> Self {
        let storage_path = default_storage_path();
        Self::load_from(&storage_path)
    }

    /// Load recent projects from a specific file path.
    pub fn load_from(path: &Path) -> Self {
        let entries = match std::fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str::<Vec<RecentEntry>>(&json) {
                Ok(entries) => {
                    debug!(count = entries.len(), "Loaded recent projects");
                    entries
                }
                Err(e) => {
                    warn!(error = %e, "Failed to parse recent projects file, starting fresh");
                    Vec::new()
                }
            },
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    warn!(error = %e, "Failed to read recent projects file");
                }
                Vec::new()
            }
        };

        Self {
            entries,
            storage_path: path.to_path_buf(),
        }
    }

    /// Add or update a project entry. Moves it to the front of the list.
    ///
    /// If the project (by path) already exists, its entry is updated and moved
    /// to the top. Otherwise a new entry is prepended. The list is capped at
    /// `MAX_RECENT_ENTRIES`.
    pub fn add(&mut self, path: &Path, name: &str) {
        let path_str = path.display().to_string();
        let now = crate::types::touch_iso_timestamp();

        // Remove any existing entry for this path
        self.entries.retain(|e| e.path != path_str);

        // Prepend new entry
        self.entries.insert(
            0,
            RecentEntry {
                path: path_str,
                name: name.to_string(),
                last_opened: now,
            },
        );

        // Cap the list size
        self.entries.truncate(MAX_RECENT_ENTRIES);

        debug!(
            name = name,
            count = self.entries.len(),
            "Added recent project"
        );
    }

    /// Remove an entry by its file path.
    pub fn remove(&mut self, path: &Path) {
        let path_str = path.display().to_string();
        self.entries.retain(|e| e.path != path_str);
    }

    /// Get the list of recent project entries (most recent first).
    pub fn entries(&self) -> &[RecentEntry] {
        &self.entries
    }

    /// Check if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries in the list.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Save the recent projects list to disk.
    pub fn save(&self) -> ProjectResult<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.storage_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                tracing::error!(
                    path = %parent.display(),
                    error = %e,
                    "Failed to create recent projects directory"
                );
                ProjectError::Io(e)
            })?;
        }

        let json = serde_json::to_string_pretty(&self.entries)?;
        std::fs::write(&self.storage_path, json.as_bytes())?;

        info!(
            count = self.entries.len(),
            path = %self.storage_path.display(),
            "Saved recent projects list"
        );
        Ok(())
    }

    /// Remove entries whose files no longer exist on disk.
    pub fn prune_missing(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| {
            let exists = Path::new(&e.path).exists();
            if !exists {
                debug!(path = %e.path, "Pruning missing recent project");
            }
            exists
        });
        let removed = before - self.entries.len();
        if removed > 0 {
            info!(removed, "Pruned missing recent projects");
        }
        removed
    }
}

/// Return the default file path for persisting the recent projects list.
fn default_storage_path() -> PathBuf {
    // Use a platform-appropriate app data directory
    let base = if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    } else if cfg!(target_os = "macos") {
        dirs_fallback("Library/Application Support")
    } else {
        // Linux / other: use XDG data home or ~/.local/share
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs_fallback(".local/share"))
    };

    base.join("MasterSelects").join(RECENT_PROJECTS_FILE)
}

/// Fallback for constructing home-relative paths.
fn dirs_fallback(subpath: &str) -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(subpath))
        .unwrap_or_else(|_| PathBuf::from(".").join(subpath))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_recent_projects() {
        let path = std::env::temp_dir().join("ms_recent_empty_test.json");
        let _ = std::fs::remove_file(&path); // ensure clean state
        let recent = RecentProjects::load_from(&path);
        assert!(recent.is_empty());
        assert_eq!(recent.len(), 0);
    }

    #[test]
    fn add_and_retrieve() {
        let path = std::env::temp_dir().join("ms_recent_add_test.json");
        let _ = std::fs::remove_file(&path);
        let mut recent = RecentProjects::load_from(&path);

        recent.add(Path::new("/projects/a.msp"), "Project A");
        recent.add(Path::new("/projects/b.msp"), "Project B");

        assert_eq!(recent.len(), 2);
        // Most recent should be first
        assert_eq!(recent.entries()[0].name, "Project B");
        assert_eq!(recent.entries()[1].name, "Project A");
    }

    #[test]
    fn add_duplicate_moves_to_front() {
        let path = std::env::temp_dir().join("ms_recent_dedup_test.json");
        let _ = std::fs::remove_file(&path);
        let mut recent = RecentProjects::load_from(&path);

        recent.add(Path::new("/a.msp"), "A");
        recent.add(Path::new("/b.msp"), "B");
        recent.add(Path::new("/c.msp"), "C");
        // Re-add A, should move to front
        recent.add(Path::new("/a.msp"), "A Updated");

        assert_eq!(recent.len(), 3);
        assert_eq!(recent.entries()[0].name, "A Updated");
        assert_eq!(recent.entries()[0].path, "/a.msp");
    }

    #[test]
    fn max_entries_enforced() {
        let path = std::env::temp_dir().join("ms_recent_max_test.json");
        let _ = std::fs::remove_file(&path);
        let mut recent = RecentProjects::load_from(&path);

        for i in 0..30 {
            recent.add(
                Path::new(&format!("/projects/{i}.msp")),
                &format!("Project {i}"),
            );
        }

        assert_eq!(recent.len(), MAX_RECENT_ENTRIES);
        // Most recent should be first (29)
        assert_eq!(recent.entries()[0].name, "Project 29");
    }

    #[test]
    fn save_and_reload() {
        let path = std::env::temp_dir().join("ms_recent_save_test.json");
        let _ = std::fs::remove_file(&path);

        {
            let mut recent = RecentProjects::load_from(&path);
            recent.add(Path::new("/test/project.msp"), "Test Project");
            recent.save().expect("save");
        }

        {
            let recent = RecentProjects::load_from(&path);
            assert_eq!(recent.len(), 1);
            assert_eq!(recent.entries()[0].name, "Test Project");
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn remove_entry() {
        let path = std::env::temp_dir().join("ms_recent_remove_test.json");
        let _ = std::fs::remove_file(&path);
        let mut recent = RecentProjects::load_from(&path);

        recent.add(Path::new("/a.msp"), "A");
        recent.add(Path::new("/b.msp"), "B");
        recent.remove(Path::new("/a.msp"));

        assert_eq!(recent.len(), 1);
        assert_eq!(recent.entries()[0].name, "B");
    }

    #[test]
    fn corrupted_file_loads_empty() {
        let path = std::env::temp_dir().join("ms_recent_corrupt_test.json");
        std::fs::write(&path, "not valid json!!!").expect("write");

        let recent = RecentProjects::load_from(&path);
        assert!(recent.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prune_missing_removes_nonexistent() {
        let path = std::env::temp_dir().join("ms_recent_prune_test.json");
        let _ = std::fs::remove_file(&path);
        let mut recent = RecentProjects::load_from(&path);

        // Add an entry that definitely doesn't exist
        recent.add(
            Path::new("/definitely/not/a/real/path/project.msp"),
            "Ghost",
        );
        // Add an entry that does exist (the temp dir itself won't work as a
        // project but its path does exist)
        let existing = std::env::temp_dir().join("ms_prune_marker.txt");
        std::fs::write(&existing, "marker").expect("write marker");
        recent.add(&existing, "Real");

        let removed = recent.prune_missing();
        assert_eq!(removed, 1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent.entries()[0].name, "Real");

        // Clean up
        let _ = std::fs::remove_file(&existing);
    }
}
