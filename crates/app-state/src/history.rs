//! Snapshot-based undo/redo history manager.
//!
//! This mirrors the web app's `historyStore.ts` pattern:
//! - Undo/redo stacks of `AppSnapshot`
//! - Batch grouping to collapse multiple related changes into one undo step
//! - Configurable maximum history depth
//!
//! # Usage
//!
//! ```ignore
//! let mut history = HistoryManager::new(50);
//!
//! // Before a user action, capture the current state
//! let snapshot = AppSnapshot::capture(&app_state);
//! history.push("Move clip", snapshot);
//!
//! // Undo
//! if let Some(prev) = history.undo() {
//!     prev.restore(&mut app_state);
//! }
//!
//! // Batch grouping (for drag operations, etc.)
//! history.start_batch("Drag clip");
//! // ... multiple small changes happen, push() calls are suppressed ...
//! history.end_batch(AppSnapshot::capture(&app_state));
//! ```

use crate::snapshot::AppSnapshot;

/// A single entry in the undo/redo history.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    /// Human-readable label describing the action (e.g., "Move clip", "Delete track").
    pub label: String,
    /// The state snapshot at this point in history.
    pub snapshot: AppSnapshot,
    /// When this entry was created.
    pub timestamp: std::time::Instant,
}

/// Manages undo/redo history using state snapshots.
///
/// The design follows the web app's historyStore pattern:
/// - Two stacks: undo (past states) and redo (future states undone)
/// - Pushing a new entry clears the redo stack (new timeline branch)
/// - Batch mode suppresses individual pushes and creates a single entry on end
/// - Maximum stack depth prevents unbounded memory growth
pub struct HistoryManager {
    undo_stack: Vec<HistoryEntry>,
    redo_stack: Vec<HistoryEntry>,
    max_entries: usize,
    /// When Some, we are in batch mode and push() calls are suppressed.
    /// The string is the batch label.
    batch_label: Option<String>,
    /// Snapshot captured at the start of a batch (the "before" state).
    batch_start_snapshot: Option<AppSnapshot>,
    /// Whether an undo/redo operation is currently in progress.
    /// Used to prevent re-entrant captures during restore.
    is_applying: bool,
}

impl HistoryManager {
    /// Create a new history manager with the given maximum number of undo entries.
    pub fn new(max_entries: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_entries,
            batch_label: None,
            batch_start_snapshot: None,
            is_applying: false,
        }
    }

    /// Push a new snapshot onto the undo stack, representing the state *before*
    /// the current action.
    ///
    /// - Clears the redo stack (forking a new timeline branch).
    /// - If in batch mode, the push is suppressed.
    /// - If `is_applying` is true (during undo/redo), the push is suppressed.
    pub fn push(&mut self, label: &str, snapshot: AppSnapshot) {
        // Don't capture during undo/redo application
        if self.is_applying {
            tracing::debug!(label, "Push suppressed: undo/redo in progress");
            return;
        }

        // If in batch mode, don't push individual snapshots
        if self.batch_label.is_some() {
            tracing::debug!(label, "Push suppressed: batch in progress");
            return;
        }

        self.redo_stack.clear();

        self.undo_stack.push(HistoryEntry {
            label: label.to_string(),
            snapshot,
            timestamp: std::time::Instant::now(),
        });

        // Enforce max history size
        while self.undo_stack.len() > self.max_entries {
            self.undo_stack.remove(0);
        }

        tracing::debug!(
            label,
            undo_depth = self.undo_stack.len(),
            "History entry pushed"
        );
    }

    /// Undo the last action. Returns a reference to the previous snapshot to restore,
    /// or `None` if nothing to undo.
    ///
    /// The current state (passed separately) should be captured before calling this
    /// and restored from the returned snapshot.
    pub fn undo(&mut self) -> Option<&AppSnapshot> {
        // End any stuck batch first (safety: lost mouseup, etc.)
        if self.batch_label.is_some() {
            tracing::warn!("Ending stuck batch before undo");
            self.batch_label = None;
            self.batch_start_snapshot = None;
        }

        if self.undo_stack.is_empty() {
            return None;
        }

        self.is_applying = true;

        let entry = self.undo_stack.pop().unwrap();
        tracing::debug!(
            label = %entry.label,
            undo_remaining = self.undo_stack.len(),
            "Undo"
        );

        self.redo_stack.push(entry);
        self.is_applying = false;

        Some(&self.redo_stack.last().unwrap().snapshot)
    }

    /// Push the current state onto the redo stack before undoing, then perform undo.
    /// This is the typical pattern: save "where we are now" so redo can get back here.
    ///
    /// Returns a reference to the snapshot to restore, or None if nothing to undo.
    pub fn undo_with_current(
        &mut self,
        current_label: &str,
        current_snapshot: AppSnapshot,
    ) -> Option<&AppSnapshot> {
        if self.undo_stack.is_empty() {
            return None;
        }

        if self.batch_label.is_some() {
            tracing::warn!("Ending stuck batch before undo");
            self.batch_label = None;
            self.batch_start_snapshot = None;
        }

        self.is_applying = true;

        // Save current state to redo stack
        self.redo_stack.push(HistoryEntry {
            label: current_label.to_string(),
            snapshot: current_snapshot,
            timestamp: std::time::Instant::now(),
        });

        let entry = self.undo_stack.pop().unwrap();
        tracing::debug!(
            label = %entry.label,
            undo_remaining = self.undo_stack.len(),
            "Undo (with current)"
        );

        // Move the undo entry to the front of redo, swap with current
        self.redo_stack.push(entry);

        self.is_applying = false;

        Some(&self.redo_stack.last().unwrap().snapshot)
    }

    /// Redo the last undone action. Returns a reference to the snapshot to restore,
    /// or `None` if nothing to redo.
    pub fn redo(&mut self) -> Option<&AppSnapshot> {
        if self.batch_label.is_some() {
            tracing::warn!("Ending stuck batch before redo");
            self.batch_label = None;
            self.batch_start_snapshot = None;
        }

        if self.redo_stack.is_empty() {
            return None;
        }

        self.is_applying = true;

        let entry = self.redo_stack.pop().unwrap();
        tracing::debug!(
            label = %entry.label,
            redo_remaining = self.redo_stack.len(),
            "Redo"
        );

        self.undo_stack.push(entry);
        self.is_applying = false;

        Some(&self.undo_stack.last().unwrap().snapshot)
    }

    /// Check if undo is available.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Start a batch operation. While batching, individual `push()` calls are suppressed.
    /// Call `end_batch()` with the final snapshot to create a single undo entry for the
    /// entire batch.
    ///
    /// The `before_snapshot` captures the state before the batch begins.
    pub fn start_batch(&mut self, label: &str, before_snapshot: AppSnapshot) {
        if self.batch_label.is_some() {
            tracing::warn!(label, "start_batch called while already batching, ignoring");
            return;
        }

        self.batch_label = Some(label.to_string());
        self.batch_start_snapshot = Some(before_snapshot);

        tracing::debug!(label, "Batch started");
    }

    /// End the current batch operation. Creates a single undo entry from the
    /// snapshot captured at `start_batch` time.
    ///
    /// If no batch is in progress, this is a no-op.
    pub fn end_batch(&mut self) {
        let label = match self.batch_label.take() {
            Some(l) => l,
            None => return,
        };

        let start_snapshot = match self.batch_start_snapshot.take() {
            Some(s) => s,
            None => return,
        };

        // Push the "before batch" snapshot onto the undo stack
        self.redo_stack.clear();
        self.undo_stack.push(HistoryEntry {
            label,
            snapshot: start_snapshot,
            timestamp: std::time::Instant::now(),
        });

        // Enforce max history size
        while self.undo_stack.len() > self.max_entries {
            self.undo_stack.remove(0);
        }

        tracing::debug!(
            undo_depth = self.undo_stack.len(),
            "Batch ended, entry pushed"
        );
    }

    /// Whether a batch operation is currently in progress.
    pub fn is_batching(&self) -> bool {
        self.batch_label.is_some()
    }

    /// Get the label of the action that would be undone next.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo_stack.last().map(|e| e.label.as_str())
    }

    /// Get the label of the action that would be redone next.
    pub fn redo_label(&self) -> Option<&str> {
        self.redo_stack.last().map(|e| e.label.as_str())
    }

    /// Number of entries on the undo stack.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of entries on the redo stack.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Whether undo/redo is currently being applied (to prevent re-entrant captures).
    pub fn is_applying(&self) -> bool {
        self.is_applying
    }

    /// Clear all history (undo and redo stacks).
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.batch_label = None;
        self.batch_start_snapshot = None;
        tracing::debug!("History cleared");
    }

    /// Get the maximum number of undo entries.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Set the maximum number of undo entries. Trims the oldest entries if needed.
    pub fn set_max_entries(&mut self, max: usize) {
        self.max_entries = max;
        while self.undo_stack.len() > self.max_entries {
            self.undo_stack.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{AppSnapshot, SelectionSnapshot};
    use crate::state::{ClipState, TrackState};

    /// Create a minimal test snapshot with an identifying tag.
    fn make_snapshot(tag: &str) -> AppSnapshot {
        AppSnapshot {
            tracks: vec![TrackState {
                id: format!("track_{tag}"),
                name: format!("Track {tag}"),
                clips: vec![ClipState {
                    id: format!("clip_{tag}"),
                    source_id: "src".to_string(),
                    timeline_start: 0.0,
                    timeline_end: 5.0,
                    source_in: 0.0,
                    source_out: 5.0,
                    opacity: 1.0,
                    blend_mode: "Normal".to_string(),
                    position: [0.0, 0.0],
                    scale: [100.0, 100.0],
                    rotation: 0.0,
                    effects: Vec::new(),
                    masks: Vec::new(),
                }],
                muted: false,
                locked: false,
            }],
            timeline_duration: 5.0,
            fps_num: 30,
            fps_den: 1,
            resolution_width: 1920,
            resolution_height: 1080,
            markers: Vec::new(),
            media_files: Vec::new(),
            selection: SelectionSnapshot {
                selected_clip_ids: Vec::new(),
                selected_track_ids: Vec::new(),
                selected_keyframes: Vec::new(),
            },
            project_name: format!("Project {tag}"),
        }
    }

    #[test]
    fn new_history_is_empty() {
        let h = HistoryManager::new(50);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.undo_count(), 0);
        assert_eq!(h.redo_count(), 0);
        assert!(h.undo_label().is_none());
        assert!(h.redo_label().is_none());
    }

    #[test]
    fn push_and_undo() {
        let mut h = HistoryManager::new(50);
        h.push("Action A", make_snapshot("a"));
        h.push("Action B", make_snapshot("b"));

        assert!(h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.undo_count(), 2);
        assert_eq!(h.undo_label(), Some("Action B"));

        let name = h.undo().unwrap().project_name.clone();
        assert_eq!(name, "Project b");
        assert!(h.can_undo()); // still one more
        assert!(h.can_redo());
        assert_eq!(h.undo_count(), 1);
        assert_eq!(h.redo_count(), 1);

        let name = h.undo().unwrap().project_name.clone();
        assert_eq!(name, "Project a");
        assert!(!h.can_undo());
        assert!(h.can_redo());
        assert_eq!(h.redo_count(), 2);
    }

    #[test]
    fn undo_empty_returns_none() {
        let mut h = HistoryManager::new(50);
        assert!(h.undo().is_none());
    }

    #[test]
    fn redo_empty_returns_none() {
        let mut h = HistoryManager::new(50);
        assert!(h.redo().is_none());
    }

    #[test]
    fn redo_after_undo() {
        let mut h = HistoryManager::new(50);
        h.push("Action A", make_snapshot("a"));
        h.push("Action B", make_snapshot("b"));

        h.undo(); // undo B
        assert!(h.can_redo());
        assert_eq!(h.redo_label(), Some("Action B"));

        let name = h.redo().unwrap().project_name.clone();
        assert_eq!(name, "Project b");
        assert!(!h.can_redo());
        assert_eq!(h.undo_count(), 2);
    }

    #[test]
    fn push_clears_redo_stack() {
        let mut h = HistoryManager::new(50);
        h.push("Action A", make_snapshot("a"));
        h.push("Action B", make_snapshot("b"));

        h.undo(); // undo B, now redo has B
        assert!(h.can_redo());

        h.push("Action C", make_snapshot("c")); // new branch, clears redo
        assert!(!h.can_redo());
        assert_eq!(h.undo_count(), 2); // A and C
    }

    #[test]
    fn max_entries_enforced() {
        let mut h = HistoryManager::new(3);
        h.push("A", make_snapshot("a"));
        h.push("B", make_snapshot("b"));
        h.push("C", make_snapshot("c"));
        h.push("D", make_snapshot("d"));

        assert_eq!(h.undo_count(), 3); // oldest (A) was evicted
                                       // The oldest entry should be B
        assert_eq!(h.undo_label(), Some("D"));
    }

    #[test]
    fn set_max_entries_trims() {
        let mut h = HistoryManager::new(10);
        for i in 0..8 {
            h.push(&format!("Action {i}"), make_snapshot(&i.to_string()));
        }
        assert_eq!(h.undo_count(), 8);

        h.set_max_entries(3);
        assert_eq!(h.undo_count(), 3);
        assert_eq!(h.max_entries(), 3);
    }

    #[test]
    fn batch_suppresses_pushes() {
        let mut h = HistoryManager::new(50);
        h.push("Before batch", make_snapshot("before"));
        assert_eq!(h.undo_count(), 1);

        h.start_batch("Drag clip", make_snapshot("batch_start"));
        assert!(h.is_batching());

        // These pushes should be suppressed
        h.push("Intermediate 1", make_snapshot("i1"));
        h.push("Intermediate 2", make_snapshot("i2"));
        h.push("Intermediate 3", make_snapshot("i3"));
        assert_eq!(h.undo_count(), 1); // No new entries during batch

        h.end_batch();
        assert!(!h.is_batching());
        assert_eq!(h.undo_count(), 2); // batch_start pushed as single entry
    }

    #[test]
    fn batch_undo_restores_to_batch_start() {
        let mut h = HistoryManager::new(50);

        h.start_batch("Drag", make_snapshot("before_drag"));
        h.end_batch();

        assert_eq!(h.undo_count(), 1);
        let name = h.undo().unwrap().project_name.clone();
        assert_eq!(name, "Project before_drag");
    }

    #[test]
    fn end_batch_clears_redo() {
        let mut h = HistoryManager::new(50);
        h.push("A", make_snapshot("a"));
        h.undo(); // redo has A
        assert!(h.can_redo());

        h.start_batch("Batch", make_snapshot("batch_start"));
        h.end_batch();

        assert!(!h.can_redo()); // redo cleared by end_batch
    }

    #[test]
    fn end_batch_without_start_is_noop() {
        let mut h = HistoryManager::new(50);
        h.end_batch(); // should not panic
        assert_eq!(h.undo_count(), 0);
    }

    #[test]
    fn double_start_batch_ignored() {
        let mut h = HistoryManager::new(50);
        h.start_batch("First", make_snapshot("first"));
        h.start_batch("Second", make_snapshot("second"));

        // Should still be first batch
        h.end_batch();
        assert_eq!(h.undo_count(), 1);

        let name = h.undo().unwrap().project_name.clone();
        assert_eq!(name, "Project first");
    }

    #[test]
    fn undo_ends_stuck_batch() {
        let mut h = HistoryManager::new(50);
        h.push("A", make_snapshot("a"));
        h.start_batch("Stuck batch", make_snapshot("stuck"));

        // Undo should end the stuck batch and return a snapshot
        assert!(h.undo().is_some());
        assert!(!h.is_batching());
    }

    #[test]
    fn redo_ends_stuck_batch() {
        let mut h = HistoryManager::new(50);
        h.push("A", make_snapshot("a"));
        h.undo(); // redo has A
        h.start_batch("Stuck batch", make_snapshot("stuck"));

        // Redo should end the stuck batch and return a snapshot
        assert!(h.redo().is_some());
        assert!(!h.is_batching());
    }

    #[test]
    fn push_suppressed_during_apply() {
        let mut h = HistoryManager::new(50);
        h.push("A", make_snapshot("a"));

        // Simulate being in applying state
        h.is_applying = true;
        h.push("Should be suppressed", make_snapshot("suppressed"));
        assert_eq!(h.undo_count(), 1); // Not 2
        h.is_applying = false;
    }

    #[test]
    fn clear_resets_everything() {
        let mut h = HistoryManager::new(50);
        h.push("A", make_snapshot("a"));
        h.push("B", make_snapshot("b"));
        h.undo();
        h.start_batch("Batch", make_snapshot("batch"));

        h.clear();

        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.undo_count(), 0);
        assert_eq!(h.redo_count(), 0);
        assert!(!h.is_batching());
    }

    #[test]
    fn labels() {
        let mut h = HistoryManager::new(50);
        h.push("Move clip", make_snapshot("a"));
        h.push("Delete track", make_snapshot("b"));

        assert_eq!(h.undo_label(), Some("Delete track"));
        h.undo();
        assert_eq!(h.undo_label(), Some("Move clip"));
        assert_eq!(h.redo_label(), Some("Delete track"));
    }

    #[test]
    fn undo_with_current_saves_current_to_redo() {
        let mut h = HistoryManager::new(50);
        h.push("Action A", make_snapshot("a"));

        let current = make_snapshot("current");
        let name = h
            .undo_with_current("Current state", current)
            .unwrap()
            .project_name
            .clone();

        // undo_with_current pushes current to redo, then pops from undo to redo
        // So redo has: [current, a] -- the last one (a) is what we get back
        assert_eq!(name, "Project a");
        assert_eq!(h.undo_count(), 0);
        assert_eq!(h.redo_count(), 2); // current + a

        // redo_stack is [current, a]. redo() pops "a", pushes to undo.
        let name = h.redo().unwrap().project_name.clone();
        assert_eq!(name, "Project a");

        // One more redo gives us "current"
        let name = h.redo().unwrap().project_name.clone();
        assert_eq!(name, "Project current");
    }

    #[test]
    fn undo_with_current_empty_returns_none() {
        let mut h = HistoryManager::new(50);
        let current = make_snapshot("current");
        assert!(h.undo_with_current("Current", current).is_none());
    }

    #[test]
    fn multiple_undo_redo_cycles() {
        let mut h = HistoryManager::new(50);
        h.push("A", make_snapshot("a"));
        h.push("B", make_snapshot("b"));
        h.push("C", make_snapshot("c"));

        // Undo 3 times
        assert_eq!(h.undo().unwrap().project_name, "Project c");
        assert_eq!(h.undo().unwrap().project_name, "Project b");
        assert_eq!(h.undo().unwrap().project_name, "Project a");
        assert!(h.undo().is_none());

        // Redo 3 times
        assert_eq!(h.redo().unwrap().project_name, "Project a");
        assert_eq!(h.redo().unwrap().project_name, "Project b");
        assert_eq!(h.redo().unwrap().project_name, "Project c");
        assert!(h.redo().is_none());
    }
}
