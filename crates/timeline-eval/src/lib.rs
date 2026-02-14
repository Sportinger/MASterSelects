//! `ms-timeline-eval` — Timeline evaluation for the MasterSelects native engine.
//!
//! This crate evaluates a timeline at a given time T and produces a list of
//! [`LayerDesc`](ms_common::LayerDesc) for the compositor. It handles:
//!
//! - **Clip activation**: determining which clips are visible at time T
//! - **Keyframe interpolation**: linear, hold, and cubic bezier
//! - **Transitions**: cross-dissolve, fade, wipe, and slide between clips
//! - **Nested compositions**: recursive evaluation of sub-timelines
//!
//! # Usage
//!
//! ```rust
//! use ms_timeline_eval::{evaluate, Timeline};
//! use ms_common::{TimeCode, Rational, Resolution};
//!
//! let timeline = Timeline::new(
//!     Rational::FPS_30,
//!     Resolution::HD,
//!     TimeCode::from_secs(60.0),
//! );
//! let layers = evaluate(&timeline, TimeCode::from_secs(5.0)).unwrap();
//! ```

pub mod error;
pub mod evaluator;
pub mod keyframe;
pub mod nested;
pub mod transition;
pub mod types;

// Re-export primary API
pub use error::TimelineEvalError;
pub use evaluator::evaluate;
pub use keyframe::{apply_keyframes, evaluate_keyframe_track, evaluate_keyframes, KeyframeValues};
pub use nested::{evaluate_nested, is_composition};
pub use transition::{
    apply_modification, compute_progress, evaluate_transition, find_overlap, LayerModification,
    TransitionResult,
};
pub use types::{
    AnimatableProperty, Clip, Interpolation, Keyframe, KeyframeTrack, Marker, SlideDirection,
    Timeline, Track, TransitionDesc, TransitionType,
};
