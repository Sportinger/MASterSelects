//! Project data model types — web-app compatible JSON format.
//!
//! These types match the TypeScript `ProjectFile` interface from the web app
//! (`src/services/project/types/`), enabling cross-format project compatibility
//! between the native Rust engine and the web-based editor.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level project file, matching the web app's `ProjectFile` interface.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectFile {
    /// Project format version (integer in web app, stored as `1`).
    pub version: u32,
    /// Human-readable project name.
    pub name: String,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
    /// ISO 8601 last-modified timestamp.
    pub updated_at: String,
    /// Timeline/composition settings.
    pub settings: ProjectSettings,
    /// Media file references.
    pub media: Vec<MediaFileRef>,
    /// Compositions (timelines).
    pub compositions: Vec<CompositionData>,
    /// Folder organization.
    pub folders: Vec<FolderRef>,
    /// Currently active composition ID.
    pub active_composition_id: Option<String>,
    /// IDs of currently open compositions.
    pub open_composition_ids: Vec<String>,
    /// IDs of expanded folders in the media panel.
    pub expanded_folder_ids: Vec<String>,
    /// Slot grid assignments (composition ID -> slot index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_assignments: Option<HashMap<String, u32>>,
    /// Media source folders for relinking after cache clear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_source_folders: Option<Vec<String>>,
    /// UI state (dock layout, view positions, etc.) — preserved but opaque to native engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui_state: Option<serde_json::Value>,
}

impl ProjectFile {
    /// Create a new empty project with the given name and settings.
    pub fn new(name: impl Into<String>, settings: ProjectSettings) -> Self {
        let now = current_iso_timestamp();
        Self {
            version: 1,
            name: name.into(),
            created_at: now.clone(),
            updated_at: now,
            settings,
            media: Vec::new(),
            compositions: Vec::new(),
            folders: Vec::new(),
            active_composition_id: None,
            open_composition_ids: Vec::new(),
            expanded_folder_ids: Vec::new(),
            slot_assignments: None,
            media_source_folders: None,
            ui_state: None,
        }
    }
}

/// Project-level settings (resolution, frame rate, sample rate).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSettings {
    /// Composition width in pixels.
    pub width: u32,
    /// Composition height in pixels.
    pub height: u32,
    /// Frame rate (e.g., 30.0, 29.97, 60.0).
    pub frame_rate: f64,
    /// Audio sample rate in Hz (e.g., 48000).
    pub sample_rate: u32,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            sample_rate: 48000,
        }
    }
}

/// A composition (timeline) containing tracks, clips, and markers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionData {
    /// Unique composition ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate.
    pub frame_rate: f64,
    /// Total duration in seconds.
    pub duration: f64,
    /// Background color (CSS hex string, e.g. "#000000").
    pub background_color: String,
    /// Folder ID this composition belongs to (null if root).
    pub folder_id: Option<String>,
    /// Ordered tracks.
    pub tracks: Vec<TrackData>,
    /// Clips on the timeline.
    pub clips: Vec<ClipData>,
    /// Timeline markers.
    pub markers: Vec<MarkerData>,
}

impl CompositionData {
    /// Create a new empty composition.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        width: u32,
        height: u32,
        frame_rate: f64,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            width,
            height,
            frame_rate,
            duration: 0.0,
            background_color: "#000000".to_string(),
            folder_id: None,
            tracks: Vec::new(),
            clips: Vec::new(),
            markers: Vec::new(),
        }
    }
}

/// A track within a composition.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackData {
    /// Unique track ID.
    pub id: String,
    /// Display name (e.g. "Video 1", "Audio 1").
    pub name: String,
    /// Track type.
    #[serde(rename = "type")]
    pub track_type: TrackType,
    /// Track height in pixels (for UI).
    pub height: u32,
    /// Whether the track is locked (no edits allowed).
    pub locked: bool,
    /// Whether the track is visible in the viewport.
    pub visible: bool,
    /// Whether audio output is muted.
    pub muted: bool,
    /// Whether the track is soloed.
    pub solo: bool,
}

/// Track type: video or audio.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackType {
    Video,
    Audio,
}

/// A clip on a track, referencing a media file or nested composition.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipData {
    /// Unique clip ID.
    pub id: String,
    /// Track this clip belongs to.
    pub track_id: String,
    /// Optional display name override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Reference to `MediaFileRef.id` (empty string for composition clips).
    pub media_id: String,
    /// Start time on the timeline in seconds.
    pub start_time: f64,
    /// Duration on the timeline in seconds.
    pub duration: f64,
    /// Source in-point in seconds.
    pub in_point: f64,
    /// Source out-point in seconds.
    pub out_point: f64,
    /// 2D transform.
    pub transform: TransformData,
    /// Applied effects.
    pub effects: Vec<EffectData>,
    /// Applied masks.
    pub masks: Vec<MaskData>,
    /// Keyframe animations.
    pub keyframes: Vec<KeyframeData>,
    /// Audio volume (0.0 to 1.0+).
    pub volume: f64,
    /// Whether audio is enabled for this clip.
    pub audio_enabled: bool,
    /// Whether playback is reversed.
    pub reversed: bool,
    /// Whether the clip is disabled (hidden).
    pub disabled: bool,
    /// Whether this clip is a nested composition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_composition: Option<bool>,
    /// ID of the nested composition (if `is_composition` is true).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition_id: Option<String>,
    /// Source type of the clip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_type: Option<String>,
    /// Natural (untrimmed) duration of the source media.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub natural_duration: Option<f64>,
    /// ID of the linked clip (e.g., audio link of a video clip).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linked_clip_id: Option<String>,
    /// Linked group ID for grouped clips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linked_group_id: Option<String>,
    /// Text properties for text clips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_properties: Option<serde_json::Value>,
    /// Solid color for solid clips (CSS hex).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solid_color: Option<String>,
}

/// 2D transform data matching the web app's `ProjectTransform`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransformData {
    /// X position in pixels.
    pub x: f64,
    /// Y position in pixels.
    pub y: f64,
    /// Z position (depth).
    pub z: f64,
    /// Horizontal scale factor (1.0 = 100%).
    pub scale_x: f64,
    /// Vertical scale factor (1.0 = 100%).
    pub scale_y: f64,
    /// Rotation in degrees (Z-axis).
    pub rotation: f64,
    /// X-axis rotation in degrees.
    pub rotation_x: f64,
    /// Y-axis rotation in degrees.
    pub rotation_y: f64,
    /// Anchor X (0.0 to 1.0, relative).
    pub anchor_x: f64,
    /// Anchor Y (0.0 to 1.0, relative).
    pub anchor_y: f64,
    /// Opacity (0.0 to 1.0).
    pub opacity: f64,
    /// Blend mode name (e.g. "normal", "multiply").
    pub blend_mode: String,
}

impl Default for TransformData {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            rotation: 0.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
            anchor_x: 0.5,
            anchor_y: 0.5,
            opacity: 1.0,
            blend_mode: "normal".to_string(),
        }
    }
}

/// An effect instance applied to a clip.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectData {
    /// Unique effect instance ID.
    pub id: String,
    /// Effect type identifier (e.g. "brightness", "blur").
    #[serde(rename = "type")]
    pub effect_type: String,
    /// Display name.
    pub name: String,
    /// Whether the effect is enabled.
    pub enabled: bool,
    /// Effect parameters (name -> value).
    pub params: HashMap<String, serde_json::Value>,
}

/// A keyframe animation point.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyframeData {
    /// Unique keyframe ID.
    pub id: String,
    /// Property being animated (e.g. "transform.x", "opacity").
    pub property: String,
    /// Time position in seconds.
    pub time: f64,
    /// Value at this keyframe.
    pub value: f64,
    /// Easing/interpolation type (e.g. "linear", "hold", "bezier").
    pub easing: String,
    /// Bezier curve handles (if easing is "bezier").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bezier_handles: Option<BezierHandles>,
}

/// Bezier curve handles for keyframe interpolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BezierHandles {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// A mask applied to a clip.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskData {
    /// Unique mask ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Mask compositing mode.
    pub mode: MaskMode,
    /// Whether the mask is inverted.
    pub inverted: bool,
    /// Mask opacity (0.0 to 1.0).
    pub opacity: f64,
    /// Feather radius in pixels.
    pub feather: f64,
    /// Feather quality level.
    pub feather_quality: u32,
    /// Whether the mask is visible.
    pub visible: bool,
    /// Whether the mask path is closed.
    pub closed: bool,
    /// Mask path vertices.
    pub vertices: Vec<MaskVertex>,
    /// Mask position offset.
    pub position: MaskPosition,
}

/// Mask compositing mode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskMode {
    Add,
    Subtract,
    Intersect,
}

/// A vertex in a mask path, with tangent handles for bezier curves.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskVertex {
    pub x: f64,
    pub y: f64,
    pub in_tangent: TangentPoint,
    pub out_tangent: TangentPoint,
}

/// A tangent handle for a mask vertex.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TangentPoint {
    pub x: f64,
    pub y: f64,
}

/// Position offset for a mask.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskPosition {
    pub x: f64,
    pub y: f64,
}

/// A timeline marker.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarkerData {
    /// Unique marker ID.
    pub id: String,
    /// Time position in seconds.
    pub time: f64,
    /// Marker name/label.
    pub name: String,
    /// Marker color (CSS hex string).
    pub color: String,
    /// Marker duration in seconds.
    pub duration: f64,
}

/// Reference to a media file in the project.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaFileRef {
    /// Unique media file ID.
    pub id: String,
    /// Original file name.
    pub name: String,
    /// Media type.
    #[serde(rename = "type")]
    pub media_type: MediaType,
    /// Path to original source file (absolute or relative).
    pub source_path: String,
    /// Path within the project folder (e.g. "Raw/video.mp4").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_path: Option<String>,
    /// Duration in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// Video/image width in pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Video/image height in pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Frame rate for video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_rate: Option<f64>,
    /// Video codec (e.g. "h264").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    /// Audio codec (e.g. "aac").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_codec: Option<String>,
    /// Container format (e.g. "mp4").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    /// Bitrate in bits per second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<u64>,
    /// File size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
    /// Whether the media has an audio track.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_audio: Option<bool>,
    /// Whether a proxy file has been generated.
    pub has_proxy: bool,
    /// Folder ID this media belongs to (null if root).
    pub folder_id: Option<String>,
    /// ISO 8601 import timestamp.
    pub imported_at: String,
}

/// Media type enumeration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaType {
    Video,
    Audio,
    Image,
}

/// Folder for organizing media and compositions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderRef {
    /// Unique folder ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Parent folder ID (null if root).
    pub parent_id: Option<String>,
    /// Optional color (CSS hex string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// Entry in the recent projects list.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentEntry {
    /// File system path to the project file.
    pub path: String,
    /// Project name.
    pub name: String,
    /// ISO 8601 timestamp of last open.
    pub last_opened: String,
}

/// Generate a current ISO 8601 timestamp string.
fn current_iso_timestamp() -> String {
    // Use a simple UTC-like format without external crate dependency.
    // In production this would use chrono or time, but we keep deps minimal.
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();

    // Convert epoch seconds to date-time components.
    let (year, month, day, hour, min, sec) = epoch_to_datetime(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert Unix epoch seconds to (year, month, day, hour, minute, second).
/// Simplified algorithm — accurate for dates from 1970 to ~2099.
fn epoch_to_datetime(epoch: u64) -> (u64, u64, u64, u64, u64, u64) {
    let sec = epoch % 60;
    let min = (epoch / 60) % 60;
    let hour = (epoch / 3600) % 24;
    let mut days = epoch / 86400;

    // Calculate year
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    // Calculate month and day
    let days_in_months: [u64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 0u64;
    for (i, &dm) in days_in_months.iter().enumerate() {
        if days < dm {
            month = i as u64 + 1;
            break;
        }
        days -= dm;
    }
    let day = days + 1;

    (year, month, day, hour, min, sec)
}

fn is_leap_year(y: u64) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

/// Update the `updated_at` field to the current timestamp.
pub fn touch_modified(project: &mut ProjectFile) {
    project.updated_at = current_iso_timestamp();
}

/// Generate a current ISO 8601 timestamp (public helper for other modules).
pub fn touch_iso_timestamp() -> String {
    current_iso_timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_project_settings() {
        let s = ProjectSettings::default();
        assert_eq!(s.width, 1920);
        assert_eq!(s.height, 1080);
        assert!((s.frame_rate - 30.0).abs() < f64::EPSILON);
        assert_eq!(s.sample_rate, 48000);
    }

    #[test]
    fn new_project_has_timestamps() {
        let p = ProjectFile::new("Test", ProjectSettings::default());
        assert_eq!(p.version, 1);
        assert_eq!(p.name, "Test");
        assert!(!p.created_at.is_empty());
        assert!(!p.updated_at.is_empty());
        assert!(p.created_at.ends_with('Z'));
    }

    #[test]
    fn default_transform() {
        let t = TransformData::default();
        assert!((t.scale_x - 1.0).abs() < f64::EPSILON);
        assert!((t.scale_y - 1.0).abs() < f64::EPSILON);
        assert!((t.opacity - 1.0).abs() < f64::EPSILON);
        assert_eq!(t.blend_mode, "normal");
    }

    #[test]
    fn project_file_roundtrip_json() {
        let project = ProjectFile::new("Roundtrip Test", ProjectSettings::default());
        let json = serde_json::to_string_pretty(&project).expect("serialize");
        let deserialized: ProjectFile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.name, "Roundtrip Test");
        assert_eq!(deserialized.version, 1);
        assert_eq!(deserialized.settings.width, 1920);
    }

    #[test]
    fn composition_data_roundtrip() {
        let comp = CompositionData::new("comp-1", "Main Timeline", 1920, 1080, 30.0);
        let json = serde_json::to_string(&comp).expect("serialize");
        let back: CompositionData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, "comp-1");
        assert_eq!(back.name, "Main Timeline");
        assert_eq!(back.background_color, "#000000");
    }

    #[test]
    fn clip_data_roundtrip() {
        let clip = ClipData {
            id: "clip-1".into(),
            track_id: "track-1".into(),
            name: Some("My Clip".into()),
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
        };
        let json = serde_json::to_string(&clip).expect("serialize");
        let back: ClipData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, "clip-1");
        assert_eq!(back.name.as_deref(), Some("My Clip"));
        assert!((back.duration - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn media_file_ref_roundtrip() {
        let media = MediaFileRef {
            id: "m1".into(),
            name: "test.mp4".into(),
            media_type: MediaType::Video,
            source_path: "/videos/test.mp4".into(),
            project_path: Some("Raw/test.mp4".into()),
            duration: Some(120.5),
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(29.97),
            codec: Some("h264".into()),
            audio_codec: Some("aac".into()),
            container: Some("mp4".into()),
            bitrate: Some(8_000_000),
            file_size: Some(120_000_000),
            has_audio: Some(true),
            has_proxy: false,
            folder_id: None,
            imported_at: "2024-01-01T00:00:00Z".into(),
        };
        let json = serde_json::to_string(&media).expect("serialize");
        let back: MediaFileRef = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, "m1");
        assert_eq!(back.media_type, MediaType::Video);
        assert_eq!(back.width, Some(1920));
    }

    #[test]
    fn effect_data_roundtrip() {
        let mut params = HashMap::new();
        params.insert("amount".into(), serde_json::json!(0.5));
        params.insert("enabled".into(), serde_json::json!(true));

        let effect = EffectData {
            id: "fx-1".into(),
            effect_type: "brightness".into(),
            name: "Brightness".into(),
            enabled: true,
            params,
        };
        let json = serde_json::to_string(&effect).expect("serialize");
        let back: EffectData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.effect_type, "brightness");
        assert!(back.params.contains_key("amount"));
    }

    #[test]
    fn keyframe_data_roundtrip() {
        let kf = KeyframeData {
            id: "kf-1".into(),
            property: "transform.x".into(),
            time: 1.5,
            value: 100.0,
            easing: "bezier".into(),
            bezier_handles: Some(BezierHandles {
                x1: 0.25,
                y1: 0.1,
                x2: 0.75,
                y2: 0.9,
            }),
        };
        let json = serde_json::to_string(&kf).expect("serialize");
        let back: KeyframeData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.property, "transform.x");
        assert!(back.bezier_handles.is_some());
    }

    #[test]
    fn mask_data_roundtrip() {
        let mask = MaskData {
            id: "mask-1".into(),
            name: "Mask 1".into(),
            mode: MaskMode::Add,
            inverted: false,
            opacity: 1.0,
            feather: 5.0,
            feather_quality: 3,
            visible: true,
            closed: true,
            vertices: vec![MaskVertex {
                x: 0.0,
                y: 0.0,
                in_tangent: TangentPoint { x: 0.0, y: 0.0 },
                out_tangent: TangentPoint { x: 0.0, y: 0.0 },
            }],
            position: MaskPosition { x: 0.0, y: 0.0 },
        };
        let json = serde_json::to_string(&mask).expect("serialize");
        let back: MaskData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.mode, MaskMode::Add);
        assert_eq!(back.vertices.len(), 1);
    }

    #[test]
    fn marker_data_roundtrip() {
        let marker = MarkerData {
            id: "mrk-1".into(),
            time: 10.0,
            name: "Scene Break".into(),
            color: "#ff0000".into(),
            duration: 0.0,
        };
        let json = serde_json::to_string(&marker).expect("serialize");
        let back: MarkerData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "Scene Break");
    }

    #[test]
    fn folder_ref_roundtrip() {
        let folder = FolderRef {
            id: "f-1".into(),
            name: "B-Roll".into(),
            parent_id: None,
            color: Some("#00ff00".into()),
        };
        let json = serde_json::to_string(&folder).expect("serialize");
        let back: FolderRef = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "B-Roll");
        assert!(back.parent_id.is_none());
        assert_eq!(back.color.as_deref(), Some("#00ff00"));
    }

    #[test]
    fn track_type_serialization() {
        assert_eq!(
            serde_json::to_string(&TrackType::Video).expect("ser"),
            "\"video\""
        );
        assert_eq!(
            serde_json::to_string(&TrackType::Audio).expect("ser"),
            "\"audio\""
        );
    }

    #[test]
    fn media_type_serialization() {
        assert_eq!(
            serde_json::to_string(&MediaType::Video).expect("ser"),
            "\"video\""
        );
        assert_eq!(
            serde_json::to_string(&MediaType::Image).expect("ser"),
            "\"image\""
        );
    }

    #[test]
    fn mask_mode_serialization() {
        assert_eq!(
            serde_json::to_string(&MaskMode::Add).expect("ser"),
            "\"add\""
        );
        assert_eq!(
            serde_json::to_string(&MaskMode::Subtract).expect("ser"),
            "\"subtract\""
        );
    }

    #[test]
    fn epoch_to_datetime_known_date() {
        // 2024-01-01 00:00:00 UTC = 1704067200
        let (y, m, d, h, mi, s) = epoch_to_datetime(1_704_067_200);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 1);
        assert_eq!(h, 0);
        assert_eq!(mi, 0);
        assert_eq!(s, 0);
    }

    #[test]
    fn touch_modified_updates_timestamp() {
        let mut p = ProjectFile::new("Test", ProjectSettings::default());
        let original = p.updated_at.clone();
        // Wait a tiny bit to ensure timestamp might differ (or at least doesn't crash)
        touch_modified(&mut p);
        // Timestamp should be valid ISO format
        assert!(p.updated_at.ends_with('Z'));
        assert!(p.updated_at.len() >= original.len());
    }

    #[test]
    fn optional_fields_absent_in_json() {
        let clip = ClipData {
            id: "c1".into(),
            track_id: "t1".into(),
            name: None,
            media_id: "m1".into(),
            start_time: 0.0,
            duration: 1.0,
            in_point: 0.0,
            out_point: 1.0,
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
            source_type: None,
            natural_duration: None,
            linked_clip_id: None,
            linked_group_id: None,
            text_properties: None,
            solid_color: None,
        };
        let json = serde_json::to_string(&clip).expect("serialize");
        // Optional fields with None should not appear in JSON
        assert!(!json.contains("isComposition"));
        assert!(!json.contains("compositionId"));
        assert!(!json.contains("solidColor"));
    }
}
