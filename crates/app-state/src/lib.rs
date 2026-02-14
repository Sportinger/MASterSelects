//! `ms-app-state` -- Application state management for the MasterSelects native engine.
//!
//! This crate provides:
//!
//! - **`AppState`**: Central state container holding timeline, media, selection, playback, and project data.
//! - **`HistoryManager`**: Snapshot-based undo/redo system (mirrors the web app's `historyStore.ts`).
//! - **`AppSnapshot`**: Serializable state snapshot for history and project save/load.
//! - **`SelectionState`**: Clip, track, and keyframe selection management.
//! - **`PlaybackState`**: Playback transport controls (play/pause/stop, scrub, in/out points, loop, rate).
//!
//! # Architecture
//!
//! ```text
//! AppState (central state)
//! ├── tracks: Vec<TrackState>        (timeline data)
//! ├── markers: Vec<MarkerState>      (timeline markers)
//! ├── media_files: Vec<MediaEntry>   (media library)
//! ├── selection: SelectionState      (what's selected)
//! ├── playback: PlaybackState        (transport state)
//! └── project metadata               (name, path, dirty flag)
//!
//! HistoryManager
//! ├── undo_stack: Vec<HistoryEntry>  (past snapshots)
//! ├── redo_stack: Vec<HistoryEntry>  (undone snapshots)
//! └── batch support                  (group multiple changes)
//! ```

pub mod history;
pub mod playback;
pub mod selection;
pub mod snapshot;
pub mod state;

// Re-export primary types at crate root for convenience.
pub use history::{HistoryEntry, HistoryManager};
pub use playback::{PlaybackMode, PlaybackState};
pub use selection::SelectionState;
pub use snapshot::{AppSnapshot, SelectionSnapshot};
pub use state::{AppState, ClipState, MarkerState, MediaEntry, TrackState};
