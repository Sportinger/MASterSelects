//! Central application state container.
//!
//! `AppState` holds the complete application state: timeline data,
//! selection, playback, media library, and project metadata.

use ms_common::{Rational, Resolution, TimeCode};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::playback::PlaybackState;
use crate::selection::SelectionState;

/// Central application state container.
///
/// This is the single source of truth for the entire application.
/// All UI reads from this state, and all modifications go through
/// controlled mutation methods.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppState {
    // --- Timeline data ---
    /// All tracks in the timeline.
    pub tracks: Vec<TrackState>,
    /// Total timeline duration.
    pub timeline_duration: TimeCode,
    /// Project frame rate.
    pub fps: Rational,
    /// Project resolution (canvas size).
    pub resolution: Resolution,
    /// Timeline markers.
    pub markers: Vec<MarkerState>,

    // --- Sub-states ---
    /// Current selection (clips, tracks, keyframes).
    pub selection: SelectionState,
    /// Playback transport state.
    pub playback: PlaybackState,

    // --- Media library ---
    /// All imported media files.
    pub media_files: Vec<MediaEntry>,

    // --- Project metadata ---
    /// Name of the current project.
    pub project_name: String,
    /// File path of the project on disk (None if unsaved).
    pub project_path: Option<PathBuf>,
    /// Whether the project has unsaved changes.
    pub is_dirty: bool,
}

/// State of a single track in the timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackState {
    /// Unique track identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Clips on this track.
    pub clips: Vec<ClipState>,
    /// Whether the track's audio is muted.
    pub muted: bool,
    /// Whether the track is locked (prevents editing).
    pub locked: bool,
}

/// An effect applied to a clip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipEffect {
    /// Effect name / type identifier.
    pub name: String,
    /// Whether the effect is enabled.
    pub enabled: bool,
    /// Parameter values as (name, value, min, max) tuples.
    pub params: Vec<(String, f32, f32, f32)>,
}

/// A mask applied to a clip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipMask {
    /// Mask display name.
    pub name: String,
    /// Whether the mask is enabled.
    pub enabled: bool,
    /// Mask opacity (0.0 - 100.0).
    pub opacity: f32,
    /// Mask feather amount.
    pub feather: f32,
    /// Whether the mask is inverted.
    pub inverted: bool,
}

/// Serde default helpers for new ClipState fields.
fn default_position() -> [f32; 2] {
    [0.0, 0.0]
}

fn default_scale() -> [f32; 2] {
    [100.0, 100.0]
}

fn default_rotation() -> f32 {
    0.0
}

fn default_effects() -> Vec<ClipEffect> {
    Vec::new()
}

fn default_masks() -> Vec<ClipMask> {
    Vec::new()
}

/// State of a single clip on a track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipState {
    /// Unique clip identifier.
    pub id: String,
    /// Source media identifier (references a MediaEntry).
    pub source_id: String,
    /// Start time on the timeline (seconds).
    pub timeline_start: f64,
    /// End time on the timeline (seconds).
    pub timeline_end: f64,
    /// Source media in-point (seconds into the source file).
    pub source_in: f64,
    /// Source media out-point (seconds into the source file).
    pub source_out: f64,
    /// Clip opacity (0.0 = transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// Blend mode name (matches BlendMode variants as strings for serialization).
    pub blend_mode: String,
    /// Clip position offset [x, y] in pixels.
    #[serde(default = "default_position")]
    pub position: [f32; 2],
    /// Clip scale [x, y] as percentages (100.0 = original size).
    #[serde(default = "default_scale")]
    pub scale: [f32; 2],
    /// Clip rotation in degrees.
    #[serde(default = "default_rotation")]
    pub rotation: f32,
    /// Effects applied to this clip.
    #[serde(default = "default_effects")]
    pub effects: Vec<ClipEffect>,
    /// Masks applied to this clip.
    #[serde(default = "default_masks")]
    pub masks: Vec<ClipMask>,
}

impl ClipState {
    /// Duration of this clip on the timeline in seconds.
    pub fn duration(&self) -> f64 {
        self.timeline_end - self.timeline_start
    }

    /// Duration of the source range used by this clip in seconds.
    pub fn source_duration(&self) -> f64 {
        self.source_out - self.source_in
    }
}

/// A named marker on the timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarkerState {
    /// Time position in seconds.
    pub time: f64,
    /// Display name / label.
    pub name: String,
    /// RGBA color.
    pub color: [f32; 4],
}

/// An entry in the media library.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MediaEntry {
    /// Unique media identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// File path on disk.
    pub path: String,
    /// Media type: "video", "audio", "image", etc.
    pub media_type: String,
    /// Duration in seconds (0 for images).
    pub duration: f64,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    /// Create a new empty application state with sensible defaults.
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            timeline_duration: TimeCode::from_secs(0.0),
            fps: Rational::FPS_30,
            resolution: Resolution::HD,
            markers: Vec::new(),
            selection: SelectionState::new(),
            playback: PlaybackState::new(),
            media_files: Vec::new(),
            project_name: "Untitled Project".to_string(),
            project_path: None,
            is_dirty: false,
        }
    }

    /// Mark the project as having unsaved changes.
    pub fn mark_dirty(&mut self) {
        if !self.is_dirty {
            self.is_dirty = true;
            tracing::debug!(project = %self.project_name, "Project marked as dirty");
        }
    }

    /// Mark the project as saved (no unsaved changes).
    pub fn mark_clean(&mut self) {
        if self.is_dirty {
            self.is_dirty = false;
            tracing::debug!(project = %self.project_name, "Project marked as clean");
        }
    }

    /// Add a track to the timeline. Marks the project dirty.
    pub fn add_track(&mut self, track: TrackState) {
        tracing::debug!(track_id = %track.id, track_name = %track.name, "Adding track");
        self.tracks.push(track);
        self.mark_dirty();
    }

    /// Remove a track by ID. Returns the removed track, or None if not found.
    pub fn remove_track(&mut self, track_id: &str) -> Option<TrackState> {
        if let Some(pos) = self.tracks.iter().position(|t| t.id == track_id) {
            let track = self.tracks.remove(pos);
            tracing::debug!(track_id = %track_id, "Removed track");
            self.mark_dirty();
            Some(track)
        } else {
            None
        }
    }

    /// Find a track by ID.
    pub fn find_track(&self, track_id: &str) -> Option<&TrackState> {
        self.tracks.iter().find(|t| t.id == track_id)
    }

    /// Find a track by ID (mutable).
    pub fn find_track_mut(&mut self, track_id: &str) -> Option<&mut TrackState> {
        self.tracks.iter_mut().find(|t| t.id == track_id)
    }

    /// Find a clip across all tracks by clip ID.
    pub fn find_clip(&self, clip_id: &str) -> Option<(&TrackState, &ClipState)> {
        for track in &self.tracks {
            if let Some(clip) = track.clips.iter().find(|c| c.id == clip_id) {
                return Some((track, clip));
            }
        }
        None
    }

    /// Find a clip across all tracks by clip ID (mutable).
    pub fn find_clip_mut(&mut self, clip_id: &str) -> Option<&mut ClipState> {
        for track in &mut self.tracks {
            if let Some(clip) = track.clips.iter_mut().find(|c| c.id == clip_id) {
                return Some(clip);
            }
        }
        None
    }

    /// Add a media entry to the library. Marks the project dirty.
    pub fn add_media(&mut self, entry: MediaEntry) {
        tracing::debug!(media_id = %entry.id, name = %entry.name, "Adding media entry");
        self.media_files.push(entry);
        self.mark_dirty();
    }

    /// Remove a media entry by ID. Returns the removed entry, or None if not found.
    pub fn remove_media(&mut self, media_id: &str) -> Option<MediaEntry> {
        if let Some(pos) = self.media_files.iter().position(|m| m.id == media_id) {
            let entry = self.media_files.remove(pos);
            tracing::debug!(media_id = %media_id, "Removed media entry");
            self.mark_dirty();
            Some(entry)
        } else {
            None
        }
    }

    /// Find a media entry by ID.
    pub fn find_media(&self, media_id: &str) -> Option<&MediaEntry> {
        self.media_files.iter().find(|m| m.id == media_id)
    }

    /// Add a marker to the timeline. Marks the project dirty.
    pub fn add_marker(&mut self, marker: MarkerState) {
        tracing::debug!(time = marker.time, name = %marker.name, "Adding marker");
        self.markers.push(marker);
        self.mark_dirty();
    }

    /// Remove a marker by index. Returns the removed marker, or None if out of bounds.
    pub fn remove_marker(&mut self, index: usize) -> Option<MarkerState> {
        if index < self.markers.len() {
            let marker = self.markers.remove(index);
            tracing::debug!(time = marker.time, name = %marker.name, "Removed marker");
            self.mark_dirty();
            Some(marker)
        } else {
            None
        }
    }

    /// Recalculate the timeline duration based on the latest clip end time across all tracks.
    pub fn recalculate_duration(&mut self) {
        let max_end = self
            .tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .map(|c| c.timeline_end)
            .fold(0.0_f64, f64::max);
        self.timeline_duration = TimeCode::from_secs(max_end);
    }

    /// Total number of clips across all tracks.
    pub fn total_clips(&self) -> usize {
        self.tracks.iter().map(|t| t.clips.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: &str, start: f64, end: f64) -> ClipState {
        ClipState {
            id: id.to_string(),
            source_id: "src_1".to_string(),
            timeline_start: start,
            timeline_end: end,
            source_in: 0.0,
            source_out: end - start,
            opacity: 1.0,
            blend_mode: "Normal".to_string(),
            position: [0.0, 0.0],
            scale: [100.0, 100.0],
            rotation: 0.0,
            effects: Vec::new(),
            masks: Vec::new(),
        }
    }

    fn make_track(id: &str, clips: Vec<ClipState>) -> TrackState {
        TrackState {
            id: id.to_string(),
            name: format!("Track {id}"),
            clips,
            muted: false,
            locked: false,
        }
    }

    #[test]
    fn new_state_defaults() {
        let state = AppState::new();
        assert_eq!(state.project_name, "Untitled Project");
        assert!(state.project_path.is_none());
        assert!(!state.is_dirty);
        assert!(state.tracks.is_empty());
        assert!(state.media_files.is_empty());
        assert!(state.markers.is_empty());
        assert_eq!(state.fps, Rational::FPS_30);
        assert_eq!(state.resolution, Resolution::HD);
    }

    #[test]
    fn mark_dirty_and_clean() {
        let mut state = AppState::new();
        assert!(!state.is_dirty);
        state.mark_dirty();
        assert!(state.is_dirty);
        state.mark_clean();
        assert!(!state.is_dirty);
    }

    #[test]
    fn add_and_remove_track() {
        let mut state = AppState::new();
        let track = make_track("t1", vec![]);
        state.add_track(track);
        assert_eq!(state.tracks.len(), 1);
        assert!(state.is_dirty);

        let removed = state.remove_track("t1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "t1");
        assert!(state.tracks.is_empty());
    }

    #[test]
    fn remove_nonexistent_track_returns_none() {
        let mut state = AppState::new();
        assert!(state.remove_track("nonexistent").is_none());
    }

    #[test]
    fn find_track() {
        let mut state = AppState::new();
        state.add_track(make_track("t1", vec![]));
        state.add_track(make_track("t2", vec![]));

        assert!(state.find_track("t1").is_some());
        assert!(state.find_track("t2").is_some());
        assert!(state.find_track("t3").is_none());
    }

    #[test]
    fn find_track_mut() {
        let mut state = AppState::new();
        state.add_track(make_track("t1", vec![]));

        let track = state.find_track_mut("t1").unwrap();
        track.name = "Renamed".to_string();

        assert_eq!(state.find_track("t1").unwrap().name, "Renamed");
    }

    #[test]
    fn find_clip_across_tracks() {
        let mut state = AppState::new();
        let clip = make_clip("c1", 0.0, 5.0);
        state.add_track(make_track("t1", vec![clip]));
        state.add_track(make_track("t2", vec![make_clip("c2", 0.0, 3.0)]));

        let (track, clip) = state.find_clip("c1").unwrap();
        assert_eq!(track.id, "t1");
        assert_eq!(clip.id, "c1");

        let (track, clip) = state.find_clip("c2").unwrap();
        assert_eq!(track.id, "t2");
        assert_eq!(clip.id, "c2");

        assert!(state.find_clip("c99").is_none());
    }

    #[test]
    fn find_clip_mut() {
        let mut state = AppState::new();
        state.add_track(make_track("t1", vec![make_clip("c1", 0.0, 5.0)]));

        let clip = state.find_clip_mut("c1").unwrap();
        clip.opacity = 0.5;

        let (_, clip) = state.find_clip("c1").unwrap();
        assert!((clip.opacity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn add_and_remove_media() {
        let mut state = AppState::new();
        state.add_media(MediaEntry {
            id: "m1".to_string(),
            name: "video.mp4".to_string(),
            path: "/path/to/video.mp4".to_string(),
            media_type: "video".to_string(),
            duration: 120.0,
        });
        assert_eq!(state.media_files.len(), 1);

        let removed = state.remove_media("m1");
        assert!(removed.is_some());
        assert!(state.media_files.is_empty());

        assert!(state.remove_media("nonexistent").is_none());
    }

    #[test]
    fn find_media() {
        let mut state = AppState::new();
        state.add_media(MediaEntry {
            id: "m1".to_string(),
            name: "video.mp4".to_string(),
            path: "/path/to/video.mp4".to_string(),
            media_type: "video".to_string(),
            duration: 120.0,
        });

        assert!(state.find_media("m1").is_some());
        assert!(state.find_media("m2").is_none());
    }

    #[test]
    fn add_and_remove_marker() {
        let mut state = AppState::new();
        state.add_marker(MarkerState {
            time: 5.0,
            name: "Intro".to_string(),
            color: [1.0, 0.0, 0.0, 1.0],
        });
        assert_eq!(state.markers.len(), 1);

        let removed = state.remove_marker(0);
        assert!(removed.is_some());
        assert!(state.markers.is_empty());

        assert!(state.remove_marker(5).is_none());
    }

    #[test]
    fn recalculate_duration() {
        let mut state = AppState::new();
        state.add_track(make_track(
            "t1",
            vec![make_clip("c1", 0.0, 5.0), make_clip("c2", 5.0, 12.0)],
        ));
        state.add_track(make_track("t2", vec![make_clip("c3", 0.0, 8.0)]));

        state.recalculate_duration();
        assert!((state.timeline_duration.as_secs() - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recalculate_duration_empty() {
        let mut state = AppState::new();
        state.recalculate_duration();
        assert!((state.timeline_duration.as_secs() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn total_clips() {
        let mut state = AppState::new();
        state.add_track(make_track(
            "t1",
            vec![make_clip("c1", 0.0, 5.0), make_clip("c2", 5.0, 10.0)],
        ));
        state.add_track(make_track("t2", vec![make_clip("c3", 0.0, 3.0)]));
        assert_eq!(state.total_clips(), 3);
    }

    #[test]
    fn clip_duration() {
        let clip = make_clip("c1", 2.0, 7.5);
        assert!((clip.duration() - 5.5).abs() < f64::EPSILON);
    }

    #[test]
    fn clip_source_duration() {
        let clip = ClipState {
            id: "c1".to_string(),
            source_id: "src".to_string(),
            timeline_start: 0.0,
            timeline_end: 5.0,
            source_in: 10.0,
            source_out: 18.0,
            opacity: 1.0,
            blend_mode: "Normal".to_string(),
            position: [0.0, 0.0],
            scale: [100.0, 100.0],
            rotation: 0.0,
            effects: Vec::new(),
            masks: Vec::new(),
        };
        assert!((clip.source_duration() - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut state = AppState::new();
        state.project_name = "Test Project".to_string();
        state.project_path = Some(PathBuf::from("/projects/test.json"));
        state.fps = Rational::FPS_24;
        state.resolution = Resolution::UHD;
        state.add_track(make_track("t1", vec![make_clip("c1", 0.0, 5.0)]));
        state.add_media(MediaEntry {
            id: "m1".to_string(),
            name: "video.mp4".to_string(),
            path: "/videos/video.mp4".to_string(),
            media_type: "video".to_string(),
            duration: 30.0,
        });
        state.add_marker(MarkerState {
            time: 2.5,
            name: "Mark".to_string(),
            color: [0.0, 1.0, 0.0, 1.0],
        });

        let json = serde_json::to_string_pretty(&state).unwrap();
        let restored: AppState = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.project_name, "Test Project");
        assert_eq!(restored.tracks.len(), 1);
        assert_eq!(restored.tracks[0].clips.len(), 1);
        assert_eq!(restored.media_files.len(), 1);
        assert_eq!(restored.markers.len(), 1);
        assert_eq!(restored.fps, Rational::FPS_24);
        assert_eq!(restored.resolution, Resolution::UHD);
    }

    #[test]
    fn deserialize_without_new_fields_uses_defaults() {
        // Simulate an old JSON that doesn't have position/scale/rotation/effects/masks
        let json = r#"{
            "id": "c1",
            "source_id": "src",
            "timeline_start": 0.0,
            "timeline_end": 5.0,
            "source_in": 0.0,
            "source_out": 5.0,
            "opacity": 1.0,
            "blend_mode": "Normal"
        }"#;
        let clip: ClipState = serde_json::from_str(json).unwrap();
        assert_eq!(clip.position, [0.0, 0.0]);
        assert_eq!(clip.scale, [100.0, 100.0]);
        assert!((clip.rotation - 0.0).abs() < f32::EPSILON);
        assert!(clip.effects.is_empty());
        assert!(clip.masks.is_empty());
    }
}
