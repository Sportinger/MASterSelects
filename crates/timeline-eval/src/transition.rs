//! Transition evaluation between clips.
//!
//! When two clips overlap (or a clip has an explicit transition), this module
//! computes how to modify the LayerDescs to produce the transition effect.
//! For cross-dissolve, opacity is cross-faded. For wipes and slides,
//! appropriate transform/mask adjustments are produced.

use ms_common::{LayerDesc, MaskDesc, MaskShape, TimeCode};

use crate::types::{Clip, SlideDirection, TransitionDesc, TransitionType};

/// Result of evaluating a transition between two clips.
#[derive(Clone, Debug)]
pub struct TransitionResult {
    /// Modified layer for the outgoing clip.
    pub outgoing: LayerModification,
    /// Modified layer for the incoming clip.
    pub incoming: LayerModification,
}

/// Modifications to apply to a layer during a transition.
#[derive(Clone, Debug, Default)]
pub struct LayerModification {
    /// Opacity multiplier (applied on top of existing opacity).
    pub opacity_multiplier: f32,
    /// Optional transform override for slide transitions.
    pub transform_offset: Option<[f32; 2]>,
    /// Optional mask for wipe transitions.
    pub wipe_mask: Option<MaskDesc>,
}

/// Evaluate a transition between an outgoing clip and an incoming clip at the given time.
///
/// `transition` describes the transition type and duration.
/// `overlap_start` is when the transition begins (incoming clip starts).
/// `progress` is 0.0 at the start of the transition and 1.0 at the end.
pub fn evaluate_transition(transition: &TransitionDesc, progress: f32) -> TransitionResult {
    let progress = progress.clamp(0.0, 1.0);

    match &transition.transition_type {
        TransitionType::CrossDissolve => evaluate_cross_dissolve(progress),
        TransitionType::Fade => evaluate_fade(progress),
        TransitionType::Wipe { angle } => evaluate_wipe(progress, *angle),
        TransitionType::Slide { direction } => evaluate_slide(progress, *direction),
    }
}

/// Cross-dissolve: outgoing fades out, incoming fades in.
fn evaluate_cross_dissolve(progress: f32) -> TransitionResult {
    TransitionResult {
        outgoing: LayerModification {
            opacity_multiplier: 1.0 - progress,
            ..Default::default()
        },
        incoming: LayerModification {
            opacity_multiplier: progress,
            ..Default::default()
        },
    }
}

/// Fade: outgoing fades to black, then incoming fades from black.
/// First half: outgoing goes 1.0 -> 0.0, incoming stays 0.0
/// Second half: outgoing stays 0.0, incoming goes 0.0 -> 1.0
fn evaluate_fade(progress: f32) -> TransitionResult {
    if progress < 0.5 {
        let sub_progress = progress * 2.0;
        TransitionResult {
            outgoing: LayerModification {
                opacity_multiplier: 1.0 - sub_progress,
                ..Default::default()
            },
            incoming: LayerModification {
                opacity_multiplier: 0.0,
                ..Default::default()
            },
        }
    } else {
        let sub_progress = (progress - 0.5) * 2.0;
        TransitionResult {
            outgoing: LayerModification {
                opacity_multiplier: 0.0,
                ..Default::default()
            },
            incoming: LayerModification {
                opacity_multiplier: sub_progress,
                ..Default::default()
            },
        }
    }
}

/// Wipe: a rectangular mask sweeps across the frame at the given angle.
/// For simplicity, we use axis-aligned wipes based on the angle quadrant.
fn evaluate_wipe(progress: f32, angle: f32) -> TransitionResult {
    // Normalize angle to [0, 360)
    let angle = angle.rem_euclid(360.0);

    // Compute wipe mask: incoming is revealed from the wipe direction
    let mask = compute_wipe_mask(progress, angle);

    TransitionResult {
        outgoing: LayerModification {
            opacity_multiplier: 1.0,
            ..Default::default()
        },
        incoming: LayerModification {
            opacity_multiplier: 1.0,
            wipe_mask: Some(mask),
            ..Default::default()
        },
    }
}

/// Compute the rectangular mask for a wipe transition.
/// The mask starts covering the entire frame and gradually reveals the incoming clip.
fn compute_wipe_mask(progress: f32, angle: f32) -> MaskDesc {
    // Map angle to a wipe direction
    // 0 degrees = left to right, 90 = top to bottom, etc.
    let (x, y, w, h) = if !(90.0..315.0).contains(&angle) {
        // Left to right wipe
        (0.0, 0.0, progress, 1.0)
    } else if angle < 180.0 {
        // Top to bottom wipe
        (0.0, 0.0, 1.0, progress)
    } else if angle < 270.0 {
        // Right to left wipe
        (1.0 - progress, 0.0, progress, 1.0)
    } else {
        // Bottom to top wipe
        (0.0, 1.0 - progress, 1.0, progress)
    };

    MaskDesc {
        shape: MaskShape::Rect {
            x,
            y,
            width: w,
            height: h,
        },
        feather: 0.0,
        opacity: 1.0,
        inverted: false,
        expansion: 0.0,
    }
}

/// Slide: the incoming clip slides in from a direction, pushing the outgoing clip out.
fn evaluate_slide(progress: f32, direction: SlideDirection) -> TransitionResult {
    // The outgoing clip slides out and the incoming clip slides in
    let (out_offset, in_offset) = match direction {
        SlideDirection::Left => {
            // Incoming slides from right to left
            ([-progress, 0.0], [1.0 - progress, 0.0])
        }
        SlideDirection::Right => {
            // Incoming slides from left to right
            ([progress, 0.0], [-(1.0 - progress), 0.0])
        }
        SlideDirection::Up => {
            // Incoming slides from bottom to top
            ([0.0, -progress], [0.0, 1.0 - progress])
        }
        SlideDirection::Down => {
            // Incoming slides from top to bottom
            ([0.0, progress], [0.0, -(1.0 - progress)])
        }
    };

    TransitionResult {
        outgoing: LayerModification {
            opacity_multiplier: 1.0,
            transform_offset: Some(out_offset),
            ..Default::default()
        },
        incoming: LayerModification {
            opacity_multiplier: 1.0,
            transform_offset: Some(in_offset),
            ..Default::default()
        },
    }
}

/// Apply a `LayerModification` to a `LayerDesc`, mutating it in place.
pub fn apply_modification(layer: &mut LayerDesc, modification: &LayerModification) {
    layer.opacity *= modification.opacity_multiplier;

    if let Some(offset) = modification.transform_offset {
        layer.transform.position[0] += offset[0];
        layer.transform.position[1] += offset[1];
    }

    if let Some(ref mask) = modification.wipe_mask {
        // For wipe transitions, the wipe mask replaces or combines with existing mask
        layer.mask = Some(mask.clone());
    }
}

/// Compute the transition progress given the current time and the transition region.
///
/// `transition_start` is when the transition begins.
/// `transition_duration` is the total duration of the transition.
/// Returns a value in [0.0, 1.0].
pub fn compute_progress(
    time: TimeCode,
    transition_start: TimeCode,
    transition_duration: TimeCode,
) -> f32 {
    let elapsed = time.as_secs() - transition_start.as_secs();
    let duration = transition_duration.as_secs();

    if duration <= 0.0 {
        return 1.0;
    }

    (elapsed / duration).clamp(0.0, 1.0) as f32
}

/// Check if two clips overlap and determine the transition region.
/// Returns `Some((overlap_start, overlap_end))` if they overlap, `None` otherwise.
pub fn find_overlap(clip_a: &Clip, clip_b: &Clip) -> Option<(TimeCode, TimeCode)> {
    let start = if clip_a.timeline_start.as_secs() > clip_b.timeline_start.as_secs() {
        clip_a.timeline_start
    } else {
        clip_b.timeline_start
    };

    let end = if clip_a.timeline_end.as_secs() < clip_b.timeline_end.as_secs() {
        clip_a.timeline_end
    } else {
        clip_b.timeline_end
    };

    if start.as_secs() < end.as_secs() {
        Some((start, end))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TransitionDesc, TransitionType};
    use ms_common::{SourceId, TimeCode};

    #[test]
    fn cross_dissolve_at_start() {
        let desc = TransitionDesc {
            transition_type: TransitionType::CrossDissolve,
            duration: TimeCode::from_secs(1.0),
        };
        let result = evaluate_transition(&desc, 0.0);
        assert!((result.outgoing.opacity_multiplier - 1.0).abs() < 1e-6);
        assert!((result.incoming.opacity_multiplier - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cross_dissolve_at_midpoint() {
        let desc = TransitionDesc {
            transition_type: TransitionType::CrossDissolve,
            duration: TimeCode::from_secs(1.0),
        };
        let result = evaluate_transition(&desc, 0.5);
        assert!((result.outgoing.opacity_multiplier - 0.5).abs() < 1e-6);
        assert!((result.incoming.opacity_multiplier - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cross_dissolve_at_end() {
        let desc = TransitionDesc {
            transition_type: TransitionType::CrossDissolve,
            duration: TimeCode::from_secs(1.0),
        };
        let result = evaluate_transition(&desc, 1.0);
        assert!((result.outgoing.opacity_multiplier - 0.0).abs() < 1e-6);
        assert!((result.incoming.opacity_multiplier - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fade_first_half() {
        let desc = TransitionDesc {
            transition_type: TransitionType::Fade,
            duration: TimeCode::from_secs(2.0),
        };
        // At 25% (first half): outgoing should be at 0.5
        let result = evaluate_transition(&desc, 0.25);
        assert!((result.outgoing.opacity_multiplier - 0.5).abs() < 1e-6);
        assert!((result.incoming.opacity_multiplier - 0.0).abs() < 1e-6);
    }

    #[test]
    fn fade_second_half() {
        let desc = TransitionDesc {
            transition_type: TransitionType::Fade,
            duration: TimeCode::from_secs(2.0),
        };
        // At 75% (second half): incoming should be at 0.5
        let result = evaluate_transition(&desc, 0.75);
        assert!((result.outgoing.opacity_multiplier - 0.0).abs() < 1e-6);
        assert!((result.incoming.opacity_multiplier - 0.5).abs() < 1e-6);
    }

    #[test]
    fn slide_left_at_midpoint() {
        let desc = TransitionDesc {
            transition_type: TransitionType::Slide {
                direction: SlideDirection::Left,
            },
            duration: TimeCode::from_secs(1.0),
        };
        let result = evaluate_transition(&desc, 0.5);
        let out_offset = result.outgoing.transform_offset.unwrap();
        let in_offset = result.incoming.transform_offset.unwrap();
        assert!((out_offset[0] - (-0.5)).abs() < 1e-6);
        assert!((in_offset[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn wipe_left_to_right() {
        let desc = TransitionDesc {
            transition_type: TransitionType::Wipe { angle: 0.0 },
            duration: TimeCode::from_secs(1.0),
        };
        let result = evaluate_transition(&desc, 0.5);
        let mask = result.incoming.wipe_mask.unwrap();
        match mask.shape {
            MaskShape::Rect {
                x,
                y,
                width,
                height,
            } => {
                assert!((x - 0.0).abs() < 1e-6);
                assert!((y - 0.0).abs() < 1e-6);
                assert!((width - 0.5).abs() < 1e-6);
                assert!((height - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected Rect mask"),
        }
    }

    #[test]
    fn compute_progress_basic() {
        let progress = compute_progress(
            TimeCode::from_secs(5.5),
            TimeCode::from_secs(5.0),
            TimeCode::from_secs(2.0),
        );
        assert!((progress - 0.25).abs() < 1e-6);
    }

    #[test]
    fn compute_progress_clamped() {
        let progress = compute_progress(
            TimeCode::from_secs(10.0),
            TimeCode::from_secs(5.0),
            TimeCode::from_secs(2.0),
        );
        assert!((progress - 1.0).abs() < 1e-6);
    }

    #[test]
    fn find_overlap_overlapping() {
        let clip_a = crate::types::Clip::new(
            "a",
            SourceId::new("s1"),
            TimeCode::from_secs(0.0),
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        let clip_b = crate::types::Clip::new(
            "b",
            SourceId::new("s2"),
            TimeCode::from_secs(4.0),
            TimeCode::from_secs(9.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        let overlap = find_overlap(&clip_a, &clip_b).unwrap();
        assert!((overlap.0.as_secs() - 4.0).abs() < 1e-9);
        assert!((overlap.1.as_secs() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn find_overlap_no_overlap() {
        let clip_a = crate::types::Clip::new(
            "a",
            SourceId::new("s1"),
            TimeCode::from_secs(0.0),
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        let clip_b = crate::types::Clip::new(
            "b",
            SourceId::new("s2"),
            TimeCode::from_secs(5.0),
            TimeCode::from_secs(10.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        assert!(find_overlap(&clip_a, &clip_b).is_none());
    }

    #[test]
    fn apply_modification_opacity() {
        let mut layer = LayerDesc::new(SourceId::new("test"));
        layer.opacity = 0.8;
        let modification = LayerModification {
            opacity_multiplier: 0.5,
            ..Default::default()
        };
        apply_modification(&mut layer, &modification);
        assert!((layer.opacity - 0.4).abs() < 1e-6);
    }

    #[test]
    fn apply_modification_transform_offset() {
        let mut layer = LayerDesc::new(SourceId::new("test"));
        layer.transform.position = [100.0, 200.0];
        let modification = LayerModification {
            opacity_multiplier: 1.0,
            transform_offset: Some([50.0, -30.0]),
            ..Default::default()
        };
        apply_modification(&mut layer, &modification);
        assert!((layer.transform.position[0] - 150.0).abs() < 1e-6);
        assert!((layer.transform.position[1] - 170.0).abs() < 1e-6);
    }
}
