//! Timeline data model types: Clip, Track, Marker, Transition, Keyframe.
//!
//! These are the Rust-native timeline types that describe the structure of a
//! video editing timeline. The evaluator consumes these to produce `LayerDesc`
//! instances for the compositor.

use ms_common::{
    BlendMode, EffectInstance, MaskDesc, Rational, Resolution, SourceId, TimeCode, Transform2D,
};
use serde::{Deserialize, Serialize};

/// A complete timeline with tracks, duration, frame rate, and markers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Timeline {
    /// Ordered list of tracks (index 0 = bottom-most in the stack).
    pub tracks: Vec<Track>,
    /// Total timeline duration.
    pub duration: TimeCode,
    /// Timeline frame rate.
    pub fps: Rational,
    /// Composition resolution.
    pub resolution: Resolution,
    /// User-placed markers.
    pub markers: Vec<Marker>,
    /// Nested composition registry: maps a `SourceId` to a sub-timeline.
    /// When a clip references a source that lives here, the evaluator
    /// recursively evaluates the nested timeline.
    #[serde(default)]
    pub compositions: Vec<(SourceId, Timeline)>,
}

impl Timeline {
    /// Create a minimal timeline with the given frame rate, resolution, and duration.
    pub fn new(fps: Rational, resolution: Resolution, duration: TimeCode) -> Self {
        Self {
            tracks: Vec::new(),
            duration,
            fps,
            resolution,
            markers: Vec::new(),
            compositions: Vec::new(),
        }
    }

    /// Add a track and return a mutable reference to it.
    pub fn add_track(&mut self, name: impl Into<String>) -> &mut Track {
        let id = format!("track_{}", self.tracks.len());
        self.tracks.push(Track {
            id,
            name: name.into(),
            clips: Vec::new(),
            muted: false,
            locked: false,
        });
        self.tracks.last_mut().expect("just pushed")
    }
}

/// A single track containing an ordered list of clips.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    /// Unique track identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Clips on this track, ordered by `timeline_start`.
    pub clips: Vec<Clip>,
    /// If true, this track produces no output.
    pub muted: bool,
    /// If true, the track is locked for editing (not relevant to evaluation,
    /// but kept for project serialization).
    pub locked: bool,
}

impl Track {
    /// Add a clip to this track. Does NOT sort; caller is responsible for ordering.
    pub fn add_clip(&mut self, clip: Clip) {
        self.clips.push(clip);
    }
}

/// A clip placed on a track. References a source media file (or nested composition)
/// and describes how it maps onto the timeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Clip {
    /// Unique clip identifier.
    pub id: String,
    /// Source media identifier (media file or nested composition).
    pub source_id: SourceId,
    /// Where this clip starts on the timeline.
    pub timeline_start: TimeCode,
    /// Where this clip ends on the timeline.
    pub timeline_end: TimeCode,
    /// Source media in-point (where to start reading from the source).
    pub source_in: TimeCode,
    /// Source media out-point (where to stop reading from the source).
    pub source_out: TimeCode,
    /// Base transform for this clip.
    pub transform: Transform2D,
    /// Base opacity (0.0..=1.0).
    pub opacity: f32,
    /// Blend mode for compositing.
    pub blend_mode: BlendMode,
    /// Effects applied to this clip, in order.
    pub effects: Vec<EffectInstance>,
    /// Optional mask.
    pub mask: Option<MaskDesc>,
    /// Keyframe tracks for animating properties over time.
    pub keyframes: Vec<KeyframeTrack>,
    /// Transition applied at the beginning of this clip.
    pub transition_in: Option<TransitionDesc>,
    /// Transition applied at the end of this clip.
    pub transition_out: Option<TransitionDesc>,
}

impl Clip {
    /// Create a clip with sensible defaults.
    pub fn new(
        id: impl Into<String>,
        source_id: SourceId,
        timeline_start: TimeCode,
        timeline_end: TimeCode,
        source_in: TimeCode,
        source_out: TimeCode,
    ) -> Self {
        Self {
            id: id.into(),
            source_id,
            timeline_start,
            timeline_end,
            source_in,
            source_out,
            transform: Transform2D::default(),
            opacity: 1.0,
            blend_mode: BlendMode::default(),
            effects: Vec::new(),
            mask: None,
            keyframes: Vec::new(),
            transition_in: None,
            transition_out: None,
        }
    }

    /// Duration of this clip on the timeline.
    pub fn duration(&self) -> TimeCode {
        self.timeline_end - self.timeline_start
    }

    /// Returns `true` if this clip is active at the given time.
    pub fn is_active_at(&self, time: TimeCode) -> bool {
        time.as_secs() >= self.timeline_start.as_secs()
            && time.as_secs() < self.timeline_end.as_secs()
    }

    /// Convert a timeline time to the corresponding source time for this clip.
    pub fn timeline_to_source_time(&self, time: TimeCode) -> TimeCode {
        let offset = time.as_secs() - self.timeline_start.as_secs();
        TimeCode::from_secs(self.source_in.as_secs() + offset)
    }
}

/// A user-placed marker on the timeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Marker {
    /// Position on the timeline.
    pub time: TimeCode,
    /// Display name.
    pub name: String,
    /// RGBA color.
    pub color: [f32; 4],
}

/// A track of keyframes for animating a single property.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyframeTrack {
    /// Which property this track controls.
    pub property: AnimatableProperty,
    /// Keyframes sorted by time (relative to the clip's timeline_start).
    pub keyframes: Vec<Keyframe>,
}

/// Properties that can be animated with keyframes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AnimatableProperty {
    PositionX,
    PositionY,
    ScaleX,
    ScaleY,
    Rotation,
    Opacity,
    EffectParam {
        effect_idx: usize,
        param_name: String,
    },
}

/// A single keyframe with a time, value, and interpolation mode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keyframe {
    /// Time relative to clip start.
    pub time: TimeCode,
    /// Value at this keyframe.
    pub value: f32,
    /// How to interpolate from this keyframe to the next.
    pub interpolation: Interpolation,
}

/// Interpolation mode between keyframes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Interpolation {
    /// Linear interpolation.
    Linear,
    /// Hold the value until the next keyframe (step function).
    Hold,
    /// Cubic bezier interpolation with control tangents.
    Bezier {
        /// Incoming tangent (dx, dy) relative to this keyframe.
        in_tangent: [f32; 2],
        /// Outgoing tangent (dx, dy) relative to this keyframe.
        out_tangent: [f32; 2],
    },
}

/// Describes a transition applied to a clip edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionDesc {
    /// Type of transition.
    pub transition_type: TransitionType,
    /// Duration of the transition.
    pub duration: TimeCode,
}

/// Types of transitions between clips.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransitionType {
    /// Cross-dissolve: blend between outgoing and incoming clips.
    CrossDissolve,
    /// Wipe at a given angle.
    Wipe { angle: f32 },
    /// Slide in a direction.
    Slide { direction: SlideDirection },
    /// Fade to/from black.
    Fade,
}

/// Direction for slide transitions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlideDirection {
    Left,
    Right,
    Up,
    Down,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_is_active_at() {
        let clip = Clip::new(
            "c1",
            SourceId::new("src1"),
            TimeCode::from_secs(1.0),
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(4.0),
        );

        assert!(!clip.is_active_at(TimeCode::from_secs(0.5)));
        assert!(clip.is_active_at(TimeCode::from_secs(1.0)));
        assert!(clip.is_active_at(TimeCode::from_secs(3.0)));
        assert!(!clip.is_active_at(TimeCode::from_secs(5.0)));
    }

    #[test]
    fn clip_duration() {
        let clip = Clip::new(
            "c1",
            SourceId::new("src1"),
            TimeCode::from_secs(2.0),
            TimeCode::from_secs(7.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        let d = clip.duration();
        assert!((d.as_secs() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn clip_timeline_to_source_time() {
        let clip = Clip::new(
            "c1",
            SourceId::new("src1"),
            TimeCode::from_secs(10.0),
            TimeCode::from_secs(20.0),
            TimeCode::from_secs(5.0),
            TimeCode::from_secs(15.0),
        );
        // At timeline 12.0s, source should be 5.0 + (12.0 - 10.0) = 7.0
        let src_time = clip.timeline_to_source_time(TimeCode::from_secs(12.0));
        assert!((src_time.as_secs() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn timeline_add_track() {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(60.0));
        let track = tl.add_track("Video 1");
        track.add_clip(Clip::new(
            "c1",
            SourceId::new("src1"),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        ));
        assert_eq!(tl.tracks.len(), 1);
        assert_eq!(tl.tracks[0].clips.len(), 1);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut tl = Timeline::new(Rational::FPS_24, Resolution::HD, TimeCode::from_secs(30.0));
        tl.markers.push(Marker {
            time: TimeCode::from_secs(5.0),
            name: "Chapter 1".to_string(),
            color: [1.0, 0.0, 0.0, 1.0],
        });
        let track = tl.add_track("V1");
        let mut clip = Clip::new(
            "c1",
            SourceId::new("media_001"),
            TimeCode::ZERO,
            TimeCode::from_secs(10.0),
            TimeCode::ZERO,
            TimeCode::from_secs(10.0),
        );
        clip.keyframes.push(KeyframeTrack {
            property: AnimatableProperty::Opacity,
            keyframes: vec![
                Keyframe {
                    time: TimeCode::ZERO,
                    value: 0.0,
                    interpolation: Interpolation::Linear,
                },
                Keyframe {
                    time: TimeCode::from_secs(1.0),
                    value: 1.0,
                    interpolation: Interpolation::Hold,
                },
            ],
        });
        track.add_clip(clip);

        let json = serde_json::to_string(&tl).expect("serialize");
        let restored: Timeline = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.tracks.len(), 1);
        assert_eq!(restored.tracks[0].clips[0].keyframes.len(), 1);
        assert_eq!(restored.tracks[0].clips[0].keyframes[0].keyframes.len(), 2);
    }
}
