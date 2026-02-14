//! Clip, track, and keyframe selection state management.

use serde::{Deserialize, Serialize};

/// Tracks which clips, tracks, and keyframes are currently selected.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SelectionState {
    selected_clips: Vec<String>,
    selected_tracks: Vec<String>,
    /// Keyframe selection: (clip_id, keyframe_index) pairs.
    selected_keyframes: Vec<(String, usize)>,
    /// Whether multi-select mode is active (e.g., Shift or Ctrl held).
    multi_select: bool,
}

impl SelectionState {
    /// Create a new empty selection state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a clip. If `multi` is false, clears previous clip selection first.
    pub fn select_clip(&mut self, clip_id: &str, multi: bool) {
        if !multi {
            self.selected_clips.clear();
        }
        // Avoid duplicates
        if !self.selected_clips.iter().any(|id| id == clip_id) {
            self.selected_clips.push(clip_id.to_string());
        }
        self.multi_select = multi;
    }

    /// Deselect a specific clip by ID.
    pub fn deselect_clip(&mut self, clip_id: &str) {
        self.selected_clips.retain(|id| id != clip_id);
    }

    /// Select a track. If `multi` is false, clears previous track selection first.
    pub fn select_track(&mut self, track_id: &str, multi: bool) {
        if !multi {
            self.selected_tracks.clear();
        }
        if !self.selected_tracks.iter().any(|id| id == track_id) {
            self.selected_tracks.push(track_id.to_string());
        }
        self.multi_select = multi;
    }

    /// Deselect a specific track by ID.
    pub fn deselect_track(&mut self, track_id: &str) {
        self.selected_tracks.retain(|id| id != track_id);
    }

    /// Select a keyframe on a specific clip.
    pub fn select_keyframe(&mut self, clip_id: &str, keyframe_index: usize, multi: bool) {
        if !multi {
            self.selected_keyframes.clear();
        }
        let entry = (clip_id.to_string(), keyframe_index);
        if !self.selected_keyframes.contains(&entry) {
            self.selected_keyframes.push(entry);
        }
        self.multi_select = multi;
    }

    /// Deselect a specific keyframe.
    pub fn deselect_keyframe(&mut self, clip_id: &str, keyframe_index: usize) {
        self.selected_keyframes
            .retain(|(cid, idx)| !(cid == clip_id && *idx == keyframe_index));
    }

    /// Clear all selections (clips, tracks, and keyframes).
    pub fn clear(&mut self) {
        self.selected_clips.clear();
        self.selected_tracks.clear();
        self.selected_keyframes.clear();
        self.multi_select = false;
    }

    /// Get the list of currently selected clip IDs.
    pub fn selected_clips(&self) -> &[String] {
        &self.selected_clips
    }

    /// Get the list of currently selected track IDs.
    pub fn selected_tracks(&self) -> &[String] {
        &self.selected_tracks
    }

    /// Get the list of currently selected keyframes as (clip_id, keyframe_index).
    pub fn selected_keyframes(&self) -> &[(String, usize)] {
        &self.selected_keyframes
    }

    /// Check if a clip is currently selected.
    pub fn is_clip_selected(&self, clip_id: &str) -> bool {
        self.selected_clips.iter().any(|id| id == clip_id)
    }

    /// Check if a track is currently selected.
    pub fn is_track_selected(&self, track_id: &str) -> bool {
        self.selected_tracks.iter().any(|id| id == track_id)
    }

    /// Check if a specific keyframe is currently selected.
    pub fn is_keyframe_selected(&self, clip_id: &str, keyframe_index: usize) -> bool {
        self.selected_keyframes
            .iter()
            .any(|(cid, idx)| cid == clip_id && *idx == keyframe_index)
    }

    /// Whether multi-select mode is currently active.
    pub fn is_multi_select(&self) -> bool {
        self.multi_select
    }

    /// Returns true if nothing is selected.
    pub fn is_empty(&self) -> bool {
        self.selected_clips.is_empty()
            && self.selected_tracks.is_empty()
            && self.selected_keyframes.is_empty()
    }

    /// Returns the total number of selected items across all categories.
    pub fn count(&self) -> usize {
        self.selected_clips.len() + self.selected_tracks.len() + self.selected_keyframes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_selection_is_empty() {
        let sel = SelectionState::new();
        assert!(sel.is_empty());
        assert_eq!(sel.count(), 0);
        assert!(!sel.is_multi_select());
    }

    #[test]
    fn select_clip_single() {
        let mut sel = SelectionState::new();
        sel.select_clip("clip_1", false);
        assert!(sel.is_clip_selected("clip_1"));
        assert_eq!(sel.selected_clips().len(), 1);

        // Selecting another clip without multi clears previous
        sel.select_clip("clip_2", false);
        assert!(!sel.is_clip_selected("clip_1"));
        assert!(sel.is_clip_selected("clip_2"));
        assert_eq!(sel.selected_clips().len(), 1);
    }

    #[test]
    fn select_clip_multi() {
        let mut sel = SelectionState::new();
        sel.select_clip("clip_1", false);
        sel.select_clip("clip_2", true);
        assert!(sel.is_clip_selected("clip_1"));
        assert!(sel.is_clip_selected("clip_2"));
        assert_eq!(sel.selected_clips().len(), 2);
        assert!(sel.is_multi_select());
    }

    #[test]
    fn select_clip_no_duplicates() {
        let mut sel = SelectionState::new();
        sel.select_clip("clip_1", false);
        sel.select_clip("clip_1", true);
        assert_eq!(sel.selected_clips().len(), 1);
    }

    #[test]
    fn deselect_clip() {
        let mut sel = SelectionState::new();
        sel.select_clip("clip_1", false);
        sel.select_clip("clip_2", true);
        sel.deselect_clip("clip_1");
        assert!(!sel.is_clip_selected("clip_1"));
        assert!(sel.is_clip_selected("clip_2"));
        assert_eq!(sel.selected_clips().len(), 1);
    }

    #[test]
    fn select_track_single() {
        let mut sel = SelectionState::new();
        sel.select_track("track_1", false);
        assert!(sel.is_track_selected("track_1"));
        assert_eq!(sel.selected_tracks().len(), 1);

        sel.select_track("track_2", false);
        assert!(!sel.is_track_selected("track_1"));
        assert!(sel.is_track_selected("track_2"));
    }

    #[test]
    fn select_track_multi() {
        let mut sel = SelectionState::new();
        sel.select_track("track_1", false);
        sel.select_track("track_2", true);
        assert!(sel.is_track_selected("track_1"));
        assert!(sel.is_track_selected("track_2"));
        assert_eq!(sel.selected_tracks().len(), 2);
    }

    #[test]
    fn deselect_track() {
        let mut sel = SelectionState::new();
        sel.select_track("track_1", false);
        sel.deselect_track("track_1");
        assert!(!sel.is_track_selected("track_1"));
        assert!(sel.is_empty());
    }

    #[test]
    fn select_keyframe() {
        let mut sel = SelectionState::new();
        sel.select_keyframe("clip_1", 0, false);
        assert!(sel.is_keyframe_selected("clip_1", 0));
        assert!(!sel.is_keyframe_selected("clip_1", 1));
        assert_eq!(sel.selected_keyframes().len(), 1);
    }

    #[test]
    fn select_keyframe_multi() {
        let mut sel = SelectionState::new();
        sel.select_keyframe("clip_1", 0, false);
        sel.select_keyframe("clip_1", 2, true);
        sel.select_keyframe("clip_2", 0, true);
        assert_eq!(sel.selected_keyframes().len(), 3);
    }

    #[test]
    fn deselect_keyframe() {
        let mut sel = SelectionState::new();
        sel.select_keyframe("clip_1", 0, false);
        sel.select_keyframe("clip_1", 1, true);
        sel.deselect_keyframe("clip_1", 0);
        assert!(!sel.is_keyframe_selected("clip_1", 0));
        assert!(sel.is_keyframe_selected("clip_1", 1));
    }

    #[test]
    fn clear_all() {
        let mut sel = SelectionState::new();
        sel.select_clip("clip_1", false);
        sel.select_track("track_1", true);
        sel.select_keyframe("clip_1", 0, true);
        assert_eq!(sel.count(), 3);

        sel.clear();
        assert!(sel.is_empty());
        assert_eq!(sel.count(), 0);
        assert!(!sel.is_multi_select());
    }

    #[test]
    fn count_across_categories() {
        let mut sel = SelectionState::new();
        sel.select_clip("c1", true);
        sel.select_clip("c2", true);
        sel.select_track("t1", true);
        sel.select_keyframe("c1", 0, true);
        assert_eq!(sel.count(), 4);
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut sel = SelectionState::new();
        sel.select_clip("clip_1", false);
        sel.select_track("track_1", true);
        sel.select_keyframe("clip_1", 3, true);

        let json = serde_json::to_string(&sel).unwrap();
        let restored: SelectionState = serde_json::from_str(&json).unwrap();

        assert!(restored.is_clip_selected("clip_1"));
        assert!(restored.is_track_selected("track_1"));
        assert!(restored.is_keyframe_selected("clip_1", 3));
    }
}
