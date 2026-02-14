//! Nested composition evaluation.
//!
//! When a clip's `source_id` references another composition (sub-timeline)
//! rather than a media file, the evaluator recursively evaluates the nested
//! timeline. This module handles the recursion, loop detection, and proper
//! source-time mapping for nested compositions.

use ms_common::{LayerDesc, SourceId, TimeCode};

use crate::error::TimelineEvalError;
use crate::evaluator::evaluate_at_time;
use crate::types::Timeline;

/// Maximum nesting depth to prevent infinite recursion.
const MAX_NESTING_DEPTH: usize = 16;

/// Evaluate a nested composition at the given source time.
///
/// `timeline` is the root timeline that contains the composition registry.
/// `comp_source_id` is the SourceId that maps to a nested timeline.
/// `source_time` is the time within the nested composition to evaluate.
/// `depth` is the current recursion depth (for loop protection).
///
/// Returns `Ok(layers)` with the evaluated layers from the nested timeline,
/// or `Err` if the composition is not found or nesting is too deep.
pub fn evaluate_nested(
    timeline: &Timeline,
    comp_source_id: &SourceId,
    source_time: TimeCode,
    depth: usize,
) -> Result<Vec<LayerDesc>, TimelineEvalError> {
    // Guard against infinite recursion
    if depth >= MAX_NESTING_DEPTH {
        return Err(TimelineEvalError::NestingTooDeep {
            max_depth: MAX_NESTING_DEPTH,
            source_id: comp_source_id.0.clone(),
        });
    }

    // Look up the nested timeline in the composition registry
    let nested_timeline = find_composition(timeline, comp_source_id).ok_or_else(|| {
        TimelineEvalError::CompositionNotFound {
            source_id: comp_source_id.0.clone(),
        }
    })?;

    // Clamp source_time to the nested timeline's duration
    let clamped_time = if source_time.as_secs() > nested_timeline.duration.as_secs() {
        nested_timeline.duration
    } else if source_time.as_secs() < 0.0 {
        TimeCode::ZERO
    } else {
        source_time
    };

    // Recursively evaluate the nested timeline
    evaluate_at_time(nested_timeline, clamped_time, depth + 1)
}

/// Look up a composition in the timeline's registry.
fn find_composition<'a>(timeline: &'a Timeline, source_id: &SourceId) -> Option<&'a Timeline> {
    timeline
        .compositions
        .iter()
        .find(|(id, _)| id == source_id)
        .map(|(_, tl)| tl)
}

/// Check whether a `SourceId` refers to a nested composition.
pub fn is_composition(timeline: &Timeline, source_id: &SourceId) -> bool {
    find_composition(timeline, source_id).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Clip, Track};
    use ms_common::{Rational, Resolution, SourceId, TimeCode};

    fn make_simple_nested_timeline() -> Timeline {
        // Inner composition: a single track with one clip
        let inner_clip = Clip::new(
            "inner_c1",
            SourceId::new("media_file_1"),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        let inner_track = Track {
            id: "inner_track_0".to_string(),
            name: "Inner V1".to_string(),
            clips: vec![inner_clip],
            muted: false,
            locked: false,
        };
        let inner_timeline = Timeline {
            tracks: vec![inner_track],
            duration: TimeCode::from_secs(5.0),
            fps: Rational::FPS_30,
            resolution: Resolution::HD,
            markers: Vec::new(),
            compositions: Vec::new(),
        };

        // Outer timeline with a clip referencing the inner composition
        let comp_id = SourceId::new("comp_001");
        let outer_clip = Clip::new(
            "outer_c1",
            comp_id.clone(),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        let outer_track = Track {
            id: "outer_track_0".to_string(),
            name: "Outer V1".to_string(),
            clips: vec![outer_clip],
            muted: false,
            locked: false,
        };

        Timeline {
            tracks: vec![outer_track],
            duration: TimeCode::from_secs(10.0),
            fps: Rational::FPS_30,
            resolution: Resolution::HD,
            markers: Vec::new(),
            compositions: vec![(comp_id, inner_timeline)],
        }
    }

    #[test]
    fn is_composition_returns_true_for_known_comp() {
        let tl = make_simple_nested_timeline();
        assert!(is_composition(&tl, &SourceId::new("comp_001")));
    }

    #[test]
    fn is_composition_returns_false_for_media() {
        let tl = make_simple_nested_timeline();
        assert!(!is_composition(&tl, &SourceId::new("media_file_1")));
    }

    #[test]
    fn evaluate_nested_basic() {
        let tl = make_simple_nested_timeline();
        let layers =
            evaluate_nested(&tl, &SourceId::new("comp_001"), TimeCode::from_secs(2.0), 0).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].source_id, SourceId::new("media_file_1"));
    }

    #[test]
    fn evaluate_nested_not_found() {
        let tl = make_simple_nested_timeline();
        let result = evaluate_nested(
            &tl,
            &SourceId::new("nonexistent"),
            TimeCode::from_secs(0.0),
            0,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            TimelineEvalError::CompositionNotFound { source_id } => {
                assert_eq!(source_id, "nonexistent");
            }
            _ => panic!("Expected CompositionNotFound error"),
        }
    }

    #[test]
    fn evaluate_nested_too_deep() {
        let tl = make_simple_nested_timeline();
        let result = evaluate_nested(
            &tl,
            &SourceId::new("comp_001"),
            TimeCode::from_secs(0.0),
            MAX_NESTING_DEPTH,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            TimelineEvalError::NestingTooDeep { .. } => {}
            _ => panic!("Expected NestingTooDeep error"),
        }
    }

    #[test]
    fn evaluate_nested_clamps_time() {
        let tl = make_simple_nested_timeline();
        // Time beyond nested timeline duration should be clamped
        let layers = evaluate_nested(
            &tl,
            &SourceId::new("comp_001"),
            TimeCode::from_secs(100.0),
            0,
        )
        .unwrap();
        // At the clamped duration (5.0s), the clip ends so nothing should be active
        // (clip is active for [0, 5.0), so at 5.0 it's not active)
        assert!(layers.is_empty());
    }
}
