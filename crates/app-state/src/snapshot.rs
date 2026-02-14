//! Serializable state snapshot for undo/redo.
//!
//! `AppSnapshot` captures the minimum state needed to restore the application
//! to a previous point in time. It is used by the `HistoryManager` to implement
//! undo/redo. Snapshots are designed to be cheaply cloneable and serializable.

use serde::{Deserialize, Serialize};

use crate::state::{AppState, ClipState, MarkerState, MediaEntry, TrackState};

/// Snapshot of the selection state (simplified for serialization).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SelectionSnapshot {
    /// IDs of selected clips.
    pub selected_clip_ids: Vec<String>,
    /// IDs of selected tracks.
    pub selected_track_ids: Vec<String>,
    /// Selected keyframes as (clip_id, keyframe_index) pairs.
    pub selected_keyframes: Vec<(String, usize)>,
}

/// A complete snapshot of the application state for undo/redo.
///
/// This captures everything needed to fully restore the app state.
/// Uses plain types (f64, u32) instead of newtypes for robust serialization.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AppSnapshot {
    /// All tracks with their clips.
    pub tracks: Vec<TrackState>,
    /// Timeline duration in seconds.
    pub timeline_duration: f64,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Composition width.
    pub resolution_width: u32,
    /// Composition height.
    pub resolution_height: u32,
    /// Timeline markers.
    pub markers: Vec<MarkerState>,
    /// Media library entries.
    pub media_files: Vec<MediaEntry>,
    /// Selection state snapshot.
    pub selection: SelectionSnapshot,
    /// Project name.
    pub project_name: String,
}

impl AppSnapshot {
    /// Capture a snapshot from the current application state.
    pub fn capture(state: &AppState) -> Self {
        Self {
            tracks: state.tracks.clone(),
            timeline_duration: state.timeline_duration.as_secs(),
            fps_num: state.fps.num,
            fps_den: state.fps.den,
            resolution_width: state.resolution.width,
            resolution_height: state.resolution.height,
            markers: state.markers.clone(),
            media_files: state.media_files.clone(),
            selection: SelectionSnapshot {
                selected_clip_ids: state.selection.selected_clips().to_vec(),
                selected_track_ids: state.selection.selected_tracks().to_vec(),
                selected_keyframes: state.selection.selected_keyframes().to_vec(),
            },
            project_name: state.project_name.clone(),
        }
    }

    /// Restore this snapshot into the given application state.
    ///
    /// This overwrites timeline data, media, markers, selection, and project name.
    /// It does NOT change playback state, project_path, or is_dirty — those are
    /// managed separately.
    pub fn restore(&self, state: &mut AppState) {
        state.tracks = self.tracks.clone();
        state.timeline_duration = ms_common::TimeCode::from_secs(self.timeline_duration);
        state.fps = ms_common::Rational::new(self.fps_num, self.fps_den);
        state.resolution =
            ms_common::Resolution::new(self.resolution_width, self.resolution_height);
        state.markers = self.markers.clone();
        state.media_files = self.media_files.clone();
        state.project_name = self.project_name.clone();

        // Restore selection
        state.selection.clear();
        for clip_id in &self.selection.selected_clip_ids {
            state.selection.select_clip(clip_id, true);
        }
        for track_id in &self.selection.selected_track_ids {
            state.selection.select_track(track_id, true);
        }
        for (clip_id, idx) in &self.selection.selected_keyframes {
            state.selection.select_keyframe(clip_id, *idx, true);
        }

        tracing::debug!(
            tracks = state.tracks.len(),
            clips = state.total_clips(),
            "Snapshot restored"
        );
    }
}

/// Helper to compute a rough size estimate of a snapshot (for memory budgeting).
impl AppSnapshot {
    /// Estimate the memory footprint of this snapshot in bytes.
    /// This is a rough approximation, not an exact measurement.
    pub fn estimated_size(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();

        // Tracks and clips
        for track in &self.tracks {
            size += std::mem::size_of::<TrackState>();
            size += track.id.len() + track.name.len();
            for clip in &track.clips {
                size += std::mem::size_of::<ClipState>();
                size += clip.id.len() + clip.source_id.len() + clip.blend_mode.len();
            }
        }

        // Markers
        size += self.markers.len() * std::mem::size_of::<MarkerState>();
        for marker in &self.markers {
            size += marker.name.len();
        }

        // Media files
        for media in &self.media_files {
            size += std::mem::size_of::<MediaEntry>();
            size += media.id.len() + media.name.len() + media.path.len() + media.media_type.len();
        }

        // Selection
        for id in &self.selection.selected_clip_ids {
            size += id.len();
        }
        for id in &self.selection.selected_track_ids {
            size += id.len();
        }
        size += self.selection.selected_keyframes.len() * std::mem::size_of::<(String, usize)>();

        size += self.project_name.len();

        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ClipState, MarkerState, MediaEntry, TrackState};
    use ms_common::{Rational, Resolution, TimeCode};

    fn make_test_state() -> AppState {
        let mut state = AppState::new();
        state.project_name = "Test Project".to_string();
        state.fps = Rational::FPS_24;
        state.resolution = Resolution::UHD;
        state.timeline_duration = TimeCode::from_secs(60.0);

        state.tracks.push(TrackState {
            id: "t1".to_string(),
            name: "Video 1".to_string(),
            clips: vec![
                ClipState {
                    id: "c1".to_string(),
                    source_id: "m1".to_string(),
                    timeline_start: 0.0,
                    timeline_end: 10.0,
                    source_in: 0.0,
                    source_out: 10.0,
                    opacity: 1.0,
                    blend_mode: "Normal".to_string(),
                    position: [0.0, 0.0],
                    scale: [100.0, 100.0],
                    rotation: 0.0,
                    effects: Vec::new(),
                    masks: Vec::new(),
                },
                ClipState {
                    id: "c2".to_string(),
                    source_id: "m1".to_string(),
                    timeline_start: 10.0,
                    timeline_end: 20.0,
                    source_in: 15.0,
                    source_out: 25.0,
                    opacity: 0.8,
                    blend_mode: "Screen".to_string(),
                    position: [0.0, 0.0],
                    scale: [100.0, 100.0],
                    rotation: 0.0,
                    effects: Vec::new(),
                    masks: Vec::new(),
                },
            ],
            muted: false,
            locked: false,
        });

        state.markers.push(MarkerState {
            time: 5.0,
            name: "Intro End".to_string(),
            color: [1.0, 0.0, 0.0, 1.0],
        });

        state.media_files.push(MediaEntry {
            id: "m1".to_string(),
            name: "footage.mp4".to_string(),
            path: "/media/footage.mp4".to_string(),
            media_type: "video".to_string(),
            duration: 120.0,
        });

        state.selection.select_clip("c1", false);
        state.selection.select_track("t1", true);

        state
    }

    #[test]
    fn capture_and_restore_roundtrip() {
        let state = make_test_state();
        let snapshot = AppSnapshot::capture(&state);

        // Verify snapshot captured correctly
        assert_eq!(snapshot.tracks.len(), 1);
        assert_eq!(snapshot.tracks[0].clips.len(), 2);
        assert!((snapshot.timeline_duration - 60.0).abs() < f64::EPSILON);
        assert_eq!(snapshot.fps_num, 24);
        assert_eq!(snapshot.fps_den, 1);
        assert_eq!(snapshot.resolution_width, 3840);
        assert_eq!(snapshot.resolution_height, 2160);
        assert_eq!(snapshot.markers.len(), 1);
        assert_eq!(snapshot.media_files.len(), 1);
        assert_eq!(snapshot.selection.selected_clip_ids, vec!["c1"]);
        assert_eq!(snapshot.selection.selected_track_ids, vec!["t1"]);
        assert_eq!(snapshot.project_name, "Test Project");

        // Modify state
        let mut modified = AppState::new();
        modified.project_name = "Modified".to_string();
        modified.tracks.clear();

        // Restore from snapshot
        snapshot.restore(&mut modified);

        assert_eq!(modified.project_name, "Test Project");
        assert_eq!(modified.tracks.len(), 1);
        assert_eq!(modified.tracks[0].clips.len(), 2);
        assert!((modified.timeline_duration.as_secs() - 60.0).abs() < f64::EPSILON);
        assert_eq!(modified.fps, Rational::FPS_24);
        assert_eq!(modified.resolution, Resolution::UHD);
        assert_eq!(modified.markers.len(), 1);
        assert_eq!(modified.media_files.len(), 1);
        assert!(modified.selection.is_clip_selected("c1"));
        assert!(modified.selection.is_track_selected("t1"));
    }

    #[test]
    fn restore_does_not_change_playback() {
        let state = make_test_state();
        let snapshot = AppSnapshot::capture(&state);

        let mut target = AppState::new();
        target.playback.play();
        target.playback.seek(TimeCode::from_secs(15.0));

        snapshot.restore(&mut target);

        // Playback state should be unchanged
        assert_eq!(target.playback.mode, crate::playback::PlaybackMode::Playing);
        assert_eq!(target.playback.current_time.as_secs(), 15.0);
    }

    #[test]
    fn restore_does_not_change_project_path() {
        let state = make_test_state();
        let snapshot = AppSnapshot::capture(&state);

        let mut target = AppState::new();
        target.project_path = Some(std::path::PathBuf::from("/saved/project.json"));

        snapshot.restore(&mut target);
        assert_eq!(
            target.project_path,
            Some(std::path::PathBuf::from("/saved/project.json"))
        );
    }

    #[test]
    fn restore_does_not_change_is_dirty() {
        let state = make_test_state();
        let snapshot = AppSnapshot::capture(&state);

        let mut target = AppState::new();
        target.is_dirty = true;

        snapshot.restore(&mut target);
        assert!(target.is_dirty); // Should remain dirty
    }

    #[test]
    fn snapshot_serialization_roundtrip() {
        let state = make_test_state();
        let snapshot = AppSnapshot::capture(&state);

        let json = serde_json::to_string(&snapshot).unwrap();
        let restored: AppSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(snapshot, restored);
    }

    #[test]
    fn empty_state_snapshot() {
        let state = AppState::new();
        let snapshot = AppSnapshot::capture(&state);

        assert!(snapshot.tracks.is_empty());
        assert!(snapshot.markers.is_empty());
        assert!(snapshot.media_files.is_empty());
        assert!(snapshot.selection.selected_clip_ids.is_empty());
        assert!(snapshot.selection.selected_track_ids.is_empty());
        assert!(snapshot.selection.selected_keyframes.is_empty());
    }

    #[test]
    fn capture_keyframe_selection() {
        let mut state = AppState::new();
        state.selection.select_keyframe("c1", 0, false);
        state.selection.select_keyframe("c1", 3, true);
        state.selection.select_keyframe("c2", 1, true);

        let snapshot = AppSnapshot::capture(&state);
        assert_eq!(snapshot.selection.selected_keyframes.len(), 3);
        assert!(snapshot
            .selection
            .selected_keyframes
            .contains(&("c1".to_string(), 0)));
        assert!(snapshot
            .selection
            .selected_keyframes
            .contains(&("c1".to_string(), 3)));
        assert!(snapshot
            .selection
            .selected_keyframes
            .contains(&("c2".to_string(), 1)));
    }

    #[test]
    fn restore_keyframe_selection() {
        let mut state = AppState::new();
        state.selection.select_keyframe("c1", 0, false);
        state.selection.select_keyframe("c2", 5, true);

        let snapshot = AppSnapshot::capture(&state);

        let mut target = AppState::new();
        snapshot.restore(&mut target);

        assert!(target.selection.is_keyframe_selected("c1", 0));
        assert!(target.selection.is_keyframe_selected("c2", 5));
    }

    #[test]
    fn estimated_size_is_positive() {
        let state = make_test_state();
        let snapshot = AppSnapshot::capture(&state);
        assert!(snapshot.estimated_size() > 0);
    }

    #[test]
    fn estimated_size_grows_with_data() {
        let empty_state = AppState::new();
        let empty_snapshot = AppSnapshot::capture(&empty_state);
        let empty_size = empty_snapshot.estimated_size();

        let full_state = make_test_state();
        let full_snapshot = AppSnapshot::capture(&full_state);
        let full_size = full_snapshot.estimated_size();

        assert!(full_size > empty_size);
    }
}
