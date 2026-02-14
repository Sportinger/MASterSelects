//! Core timeline evaluation: `evaluate()` takes a timeline and a time
//! and produces a list of `LayerDesc` for the compositor.
//!
//! The evaluation process:
//! 1. For each track (bottom-to-top), find active clip(s) at the given time.
//! 2. Apply keyframe interpolation to get current property values.
//! 3. Handle transitions between overlapping clips.
//! 4. Handle nested compositions by recursive evaluation.
//! 5. Produce sorted `Vec<LayerDesc>` for the compositor.

use ms_common::{EffectInstance, LayerDesc, ParamValue, TimeCode, Transform2D};

use crate::error::TimelineEvalError;
use crate::keyframe::{apply_keyframes, KeyframeValues};
use crate::nested::{evaluate_nested, is_composition};
use crate::transition::{apply_modification, compute_progress, evaluate_transition, find_overlap};
use crate::types::{Clip, Timeline, Track};

/// Evaluate the timeline at the given time, producing layer descriptions for the compositor.
///
/// This is the main entry point for timeline evaluation. It processes all tracks,
/// resolves keyframes, handles transitions and nested compositions, and returns
/// a sorted list of `LayerDesc` ready for the compositor.
///
/// # Arguments
/// * `timeline` - The timeline to evaluate.
/// * `time` - The current time to evaluate at.
///
/// # Returns
/// A `Vec<LayerDesc>` sorted by `z_order` (lower = behind), or an error
/// if nested composition evaluation fails.
pub fn evaluate(timeline: &Timeline, time: TimeCode) -> Result<Vec<LayerDesc>, TimelineEvalError> {
    evaluate_at_time(timeline, time, 0)
}

/// Internal evaluation function that tracks recursion depth for nested compositions.
pub(crate) fn evaluate_at_time(
    timeline: &Timeline,
    time: TimeCode,
    depth: usize,
) -> Result<Vec<LayerDesc>, TimelineEvalError> {
    let mut layers = Vec::new();

    // Process tracks from bottom (index 0) to top
    for (track_idx, track) in timeline.tracks.iter().enumerate() {
        // Skip muted tracks
        if track.muted {
            continue;
        }

        let track_layers = evaluate_track(timeline, track, time, track_idx as i32, depth)?;
        layers.extend(track_layers);
    }

    // Sort by z_order (lower = behind, rendered first)
    layers.sort_by_key(|l| l.z_order);

    Ok(layers)
}

/// Evaluate a single track at the given time.
///
/// Finds all active clips, handles transitions between adjacent overlapping clips,
/// and produces `LayerDesc` instances.
fn evaluate_track(
    timeline: &Timeline,
    track: &Track,
    time: TimeCode,
    track_z_base: i32,
    depth: usize,
) -> Result<Vec<LayerDesc>, TimelineEvalError> {
    let mut layers = Vec::new();

    // Find all clips active at this time
    let active_clips: Vec<&Clip> = track
        .clips
        .iter()
        .filter(|clip| clip.is_active_at(time))
        .collect();

    if active_clips.is_empty() {
        return Ok(layers);
    }

    // Handle the simple case: single active clip, no transitions
    if active_clips.len() == 1 {
        let clip = active_clips[0];
        let layer = evaluate_single_clip(timeline, clip, time, track_z_base, depth)?;

        // Handle transition_in at clip start
        if let Some(ref transition_in) = clip.transition_in {
            let trans_end = TimeCode::from_secs(
                clip.timeline_start.as_secs() + transition_in.duration.as_secs(),
            );
            if time.as_secs() < trans_end.as_secs() {
                let progress = compute_progress(time, clip.timeline_start, transition_in.duration);
                let result = evaluate_transition(transition_in, progress);
                let mut modified_layer = layer;
                apply_modification(&mut modified_layer, &result.incoming);
                layers.push(modified_layer);
                return Ok(layers);
            }
        }

        // Handle transition_out at clip end
        if let Some(ref transition_out) = clip.transition_out {
            let trans_start = TimeCode::from_secs(
                clip.timeline_end.as_secs() - transition_out.duration.as_secs(),
            );
            if time.as_secs() >= trans_start.as_secs() {
                let progress = compute_progress(time, trans_start, transition_out.duration);
                let result = evaluate_transition(transition_out, progress);
                let mut modified_layer = layer;
                apply_modification(&mut modified_layer, &result.outgoing);
                layers.push(modified_layer);
                return Ok(layers);
            }
        }

        layers.push(layer);
        return Ok(layers);
    }

    // Multiple active clips: handle transitions between adjacent overlapping clips
    for (i, clip) in active_clips.iter().enumerate() {
        let z_order = track_z_base * 1000 + i as i32;

        // Check for overlap with the next clip for transition handling
        if i + 1 < active_clips.len() {
            let next_clip = active_clips[i + 1];
            if let Some((overlap_start, overlap_end)) = find_overlap(clip, next_clip) {
                let overlap_duration = overlap_end - overlap_start;

                // Determine transition type: prefer the outgoing clip's transition_out,
                // then the incoming clip's transition_in, then default to cross-dissolve.
                let transition = clip
                    .transition_out
                    .as_ref()
                    .or(next_clip.transition_in.as_ref());

                if let Some(trans) = transition {
                    let progress = compute_progress(time, overlap_start, overlap_duration);
                    let result = evaluate_transition(trans, progress);

                    // Outgoing clip
                    let mut outgoing_layer =
                        evaluate_single_clip(timeline, clip, time, z_order, depth)?;
                    apply_modification(&mut outgoing_layer, &result.outgoing);
                    layers.push(outgoing_layer);

                    // Incoming clip
                    let mut incoming_layer =
                        evaluate_single_clip(timeline, next_clip, time, z_order + 1, depth)?;
                    apply_modification(&mut incoming_layer, &result.incoming);
                    layers.push(incoming_layer);

                    // Skip the next clip since we handled it as part of this transition
                    continue;
                }
            }
        }

        let layer = evaluate_single_clip(timeline, clip, time, z_order, depth)?;
        layers.push(layer);
    }

    Ok(layers)
}

/// Evaluate a single clip at the given time, producing a `LayerDesc`.
///
/// This applies keyframe interpolation and handles nested composition evaluation.
fn evaluate_single_clip(
    timeline: &Timeline,
    clip: &Clip,
    time: TimeCode,
    z_order: i32,
    depth: usize,
) -> Result<LayerDesc, TimelineEvalError> {
    // Compute local time (relative to clip start) for keyframe evaluation
    let local_time = TimeCode::from_secs(time.as_secs() - clip.timeline_start.as_secs());

    // Apply keyframe interpolation
    let kf_values = apply_keyframes(&clip.keyframes, local_time);

    // Build the transform, overriding with keyframe values
    let transform = build_transform(&clip.transform, &kf_values);

    // Compute effective opacity
    let opacity = kf_values.opacity.unwrap_or(clip.opacity);

    // Build effects with keyframe-animated parameters
    let effects = build_effects(&clip.effects, &kf_values);

    // Check for nested composition
    if is_composition(timeline, &clip.source_id) {
        let source_time = clip.timeline_to_source_time(time);
        let nested_layers = evaluate_nested(timeline, &clip.source_id, source_time, depth)?;

        // For nested compositions, we flatten the layers and apply the clip's
        // transform/opacity as an outer wrapper. Each nested layer gets the
        // clip's properties applied on top.
        if nested_layers.is_empty() {
            // Composition produced no output at this time
            return Ok(LayerDesc {
                source_id: clip.source_id.clone(),
                transform,
                opacity: 0.0, // invisible
                blend_mode: clip.blend_mode,
                effects,
                mask: clip.mask.clone(),
                z_order,
            });
        }

        // If there's exactly one nested layer, apply our transform on top
        if nested_layers.len() == 1 {
            let mut layer = nested_layers.into_iter().next().expect("just checked len");
            layer.transform = compose_transforms(&transform, &layer.transform);
            layer.opacity *= opacity;
            layer.z_order = z_order;
            if clip.mask.is_some() {
                layer.mask = clip.mask.clone();
            }
            return Ok(layer);
        }

        // Multiple nested layers: return the first with composition transform applied.
        // In a full implementation, these would be pre-composited. For now, we merge them
        // into a single representative layer referencing the composition source.
        return Ok(LayerDesc {
            source_id: clip.source_id.clone(),
            transform,
            opacity,
            blend_mode: clip.blend_mode,
            effects,
            mask: clip.mask.clone(),
            z_order,
        });
    }

    // Regular media clip
    Ok(LayerDesc {
        source_id: clip.source_id.clone(),
        transform,
        opacity,
        blend_mode: clip.blend_mode,
        effects,
        mask: clip.mask.clone(),
        z_order,
    })
}

/// Build a `Transform2D` from the clip's base transform, overriding with keyframe values.
fn build_transform(base: &Transform2D, kf: &KeyframeValues) -> Transform2D {
    Transform2D {
        position: [
            kf.position_x.unwrap_or(base.position[0]),
            kf.position_y.unwrap_or(base.position[1]),
        ],
        scale: [
            kf.scale_x.unwrap_or(base.scale[0]),
            kf.scale_y.unwrap_or(base.scale[1]),
        ],
        rotation: kf.rotation.unwrap_or(base.rotation),
        anchor: base.anchor,
    }
}

/// Build effects list with keyframe-animated parameters applied.
fn build_effects(base_effects: &[EffectInstance], kf: &KeyframeValues) -> Vec<EffectInstance> {
    let mut effects = base_effects.to_vec();

    for (effect_idx, param_name, value) in &kf.effect_params {
        if let Some(effect) = effects.get_mut(*effect_idx) {
            // Find and update the parameter
            let mut found = false;
            for (name, param_value) in &mut effect.params {
                if name == param_name {
                    *param_value = ParamValue::Float(*value);
                    found = true;
                    break;
                }
            }
            // If the parameter wasn't found, add it
            if !found {
                effect
                    .params
                    .push((param_name.clone(), ParamValue::Float(*value)));
            }
        }
    }

    effects
}

/// Compose two transforms: apply `outer` on top of `inner`.
/// This is a simplified composition that adds positions and multiplies scales.
fn compose_transforms(outer: &Transform2D, inner: &Transform2D) -> Transform2D {
    Transform2D {
        position: [
            outer.position[0] + inner.position[0] * outer.scale[0],
            outer.position[1] + inner.position[1] * outer.scale[1],
        ],
        scale: [
            outer.scale[0] * inner.scale[0],
            outer.scale[1] * inner.scale[1],
        ],
        rotation: outer.rotation + inner.rotation,
        anchor: inner.anchor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use ms_common::{EffectId, EffectInstance, ParamValue, Rational, Resolution, SourceId};

    /// Helper: create a simple timeline with one track and one clip.
    fn single_clip_timeline() -> Timeline {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));
        let track = tl.add_track("V1");
        track.add_clip(Clip::new(
            "c1",
            SourceId::new("media_001"),
            TimeCode::from_secs(1.0),
            TimeCode::from_secs(6.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        ));
        tl
    }

    #[test]
    fn evaluate_empty_timeline() {
        let tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));
        let layers = evaluate(&tl, TimeCode::from_secs(5.0)).unwrap();
        assert!(layers.is_empty());
    }

    #[test]
    fn evaluate_single_clip() {
        let tl = single_clip_timeline();

        // Before clip
        let layers = evaluate(&tl, TimeCode::from_secs(0.5)).unwrap();
        assert!(layers.is_empty());

        // During clip
        let layers = evaluate(&tl, TimeCode::from_secs(3.0)).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].source_id, SourceId::new("media_001"));
        assert!((layers[0].opacity - 1.0).abs() < 1e-6);

        // After clip
        let layers = evaluate(&tl, TimeCode::from_secs(7.0)).unwrap();
        assert!(layers.is_empty());
    }

    #[test]
    fn evaluate_muted_track() {
        let mut tl = single_clip_timeline();
        tl.tracks[0].muted = true;

        let layers = evaluate(&tl, TimeCode::from_secs(3.0)).unwrap();
        assert!(layers.is_empty());
    }

    #[test]
    fn evaluate_multiple_tracks() {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));

        // Bottom track
        let track0 = tl.add_track("V1");
        track0.add_clip(Clip::new(
            "c1",
            SourceId::new("bg"),
            TimeCode::ZERO,
            TimeCode::from_secs(10.0),
            TimeCode::ZERO,
            TimeCode::from_secs(10.0),
        ));

        // Top track
        let track1 = tl.add_track("V2");
        track1.add_clip(Clip::new(
            "c2",
            SourceId::new("overlay"),
            TimeCode::from_secs(2.0),
            TimeCode::from_secs(8.0),
            TimeCode::ZERO,
            TimeCode::from_secs(6.0),
        ));

        // At time 5.0, both clips should be active
        let layers = evaluate(&tl, TimeCode::from_secs(5.0)).unwrap();
        assert_eq!(layers.len(), 2);

        // At time 1.0, only bg should be active
        let layers = evaluate(&tl, TimeCode::from_secs(1.0)).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].source_id, SourceId::new("bg"));
    }

    #[test]
    fn evaluate_with_keyframes() {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));
        let track = tl.add_track("V1");
        let mut clip = Clip::new(
            "c1",
            SourceId::new("media_001"),
            TimeCode::ZERO,
            TimeCode::from_secs(4.0),
            TimeCode::ZERO,
            TimeCode::from_secs(4.0),
        );

        // Animate opacity from 0 to 1 over the first 2 seconds
        clip.keyframes.push(KeyframeTrack {
            property: AnimatableProperty::Opacity,
            keyframes: vec![
                Keyframe {
                    time: TimeCode::ZERO,
                    value: 0.0,
                    interpolation: Interpolation::Linear,
                },
                Keyframe {
                    time: TimeCode::from_secs(2.0),
                    value: 1.0,
                    interpolation: Interpolation::Linear,
                },
            ],
        });
        track.add_clip(clip);

        // At t=1.0 (midpoint of animation), opacity should be 0.5
        let layers = evaluate(&tl, TimeCode::from_secs(1.0)).unwrap();
        assert_eq!(layers.len(), 1);
        assert!((layers[0].opacity - 0.5).abs() < 1e-4);

        // At t=0.0, opacity should be 0.0
        let layers = evaluate(&tl, TimeCode::from_secs(0.0)).unwrap();
        assert_eq!(layers.len(), 1);
        assert!((layers[0].opacity - 0.0).abs() < 1e-4);

        // At t=3.0 (past keyframe range), opacity should be 1.0
        let layers = evaluate(&tl, TimeCode::from_secs(3.0)).unwrap();
        assert_eq!(layers.len(), 1);
        assert!((layers[0].opacity - 1.0).abs() < 1e-4);
    }

    #[test]
    fn evaluate_with_position_keyframes() {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));
        let track = tl.add_track("V1");
        let mut clip = Clip::new(
            "c1",
            SourceId::new("media_001"),
            TimeCode::ZERO,
            TimeCode::from_secs(4.0),
            TimeCode::ZERO,
            TimeCode::from_secs(4.0),
        );

        clip.keyframes.push(KeyframeTrack {
            property: AnimatableProperty::PositionX,
            keyframes: vec![
                Keyframe {
                    time: TimeCode::ZERO,
                    value: 0.0,
                    interpolation: Interpolation::Linear,
                },
                Keyframe {
                    time: TimeCode::from_secs(2.0),
                    value: 100.0,
                    interpolation: Interpolation::Linear,
                },
            ],
        });
        track.add_clip(clip);

        let layers = evaluate(&tl, TimeCode::from_secs(1.0)).unwrap();
        assert_eq!(layers.len(), 1);
        assert!((layers[0].transform.position[0] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn evaluate_with_effect_keyframes() {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));
        let track = tl.add_track("V1");
        let mut clip = Clip::new(
            "c1",
            SourceId::new("media_001"),
            TimeCode::ZERO,
            TimeCode::from_secs(4.0),
            TimeCode::ZERO,
            TimeCode::from_secs(4.0),
        );

        // Add an effect
        clip.effects.push(
            EffectInstance::new(EffectId::new("brightness"))
                .with_param("amount", ParamValue::Float(0.5)),
        );

        // Animate the effect parameter
        clip.keyframes.push(KeyframeTrack {
            property: AnimatableProperty::EffectParam {
                effect_idx: 0,
                param_name: "amount".to_string(),
            },
            keyframes: vec![
                Keyframe {
                    time: TimeCode::ZERO,
                    value: 0.0,
                    interpolation: Interpolation::Linear,
                },
                Keyframe {
                    time: TimeCode::from_secs(2.0),
                    value: 1.0,
                    interpolation: Interpolation::Linear,
                },
            ],
        });
        track.add_clip(clip);

        let layers = evaluate(&tl, TimeCode::from_secs(1.0)).unwrap();
        assert_eq!(layers.len(), 1);

        let brightness_param = layers[0].effects[0].get_param("amount").unwrap();
        assert!((brightness_param.as_float().unwrap() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn evaluate_with_transition_in() {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));
        let track = tl.add_track("V1");
        let mut clip = Clip::new(
            "c1",
            SourceId::new("media_001"),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        clip.transition_in = Some(TransitionDesc {
            transition_type: TransitionType::CrossDissolve,
            duration: TimeCode::from_secs(1.0),
        });
        track.add_clip(clip);

        // At t=0.5 (halfway through 1s transition), opacity should be ~0.5
        let layers = evaluate(&tl, TimeCode::from_secs(0.5)).unwrap();
        assert_eq!(layers.len(), 1);
        assert!((layers[0].opacity - 0.5).abs() < 1e-4);

        // At t=2.0 (past transition), opacity should be full
        let layers = evaluate(&tl, TimeCode::from_secs(2.0)).unwrap();
        assert_eq!(layers.len(), 1);
        assert!((layers[0].opacity - 1.0).abs() < 1e-4);
    }

    #[test]
    fn evaluate_z_order_is_sorted() {
        let mut tl = Timeline::new(Rational::FPS_30, Resolution::HD, TimeCode::from_secs(10.0));

        for i in 0..3 {
            let track = tl.add_track(format!("V{}", i + 1));
            track.add_clip(Clip::new(
                format!("c{}", i + 1),
                SourceId::new(format!("media_{}", i + 1)),
                TimeCode::ZERO,
                TimeCode::from_secs(10.0),
                TimeCode::ZERO,
                TimeCode::from_secs(10.0),
            ));
        }

        let layers = evaluate(&tl, TimeCode::from_secs(5.0)).unwrap();
        assert_eq!(layers.len(), 3);

        // Verify z_order is monotonically increasing
        for i in 1..layers.len() {
            assert!(layers[i].z_order >= layers[i - 1].z_order);
        }
    }

    #[test]
    fn evaluate_nested_composition() {
        let inner_clip = Clip::new(
            "inner_c1",
            SourceId::new("inner_media"),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
            TimeCode::ZERO,
            TimeCode::from_secs(5.0),
        );
        let inner_track = Track {
            id: "inner_t0".to_string(),
            name: "Inner V1".to_string(),
            clips: vec![inner_clip],
            muted: false,
            locked: false,
        };
        let inner_tl = Timeline {
            tracks: vec![inner_track],
            duration: TimeCode::from_secs(5.0),
            fps: Rational::FPS_30,
            resolution: Resolution::HD,
            markers: Vec::new(),
            compositions: Vec::new(),
        };

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
            id: "outer_t0".to_string(),
            name: "Outer V1".to_string(),
            clips: vec![outer_clip],
            muted: false,
            locked: false,
        };
        let tl = Timeline {
            tracks: vec![outer_track],
            duration: TimeCode::from_secs(10.0),
            fps: Rational::FPS_30,
            resolution: Resolution::HD,
            markers: Vec::new(),
            compositions: vec![(comp_id, inner_tl)],
        };

        let layers = evaluate(&tl, TimeCode::from_secs(2.0)).unwrap();
        assert_eq!(layers.len(), 1);
        // The nested composition resolves to the inner media source
        assert_eq!(layers[0].source_id, SourceId::new("inner_media"));
    }

    #[test]
    fn compose_transforms_identity() {
        let outer = Transform2D::default();
        let inner = Transform2D {
            position: [100.0, 50.0],
            scale: [0.5, 0.5],
            rotation: 45.0,
            anchor: [0.5, 0.5],
        };
        let composed = compose_transforms(&outer, &inner);
        assert!((composed.position[0] - 100.0).abs() < 1e-6);
        assert!((composed.position[1] - 50.0).abs() < 1e-6);
        assert!((composed.scale[0] - 0.5).abs() < 1e-6);
        assert!((composed.rotation - 45.0).abs() < 1e-6);
    }

    #[test]
    fn compose_transforms_scaled_outer() {
        let outer = Transform2D {
            position: [10.0, 20.0],
            scale: [2.0, 2.0],
            rotation: 0.0,
            anchor: [0.5, 0.5],
        };
        let inner = Transform2D {
            position: [5.0, 5.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
            anchor: [0.5, 0.5],
        };
        let composed = compose_transforms(&outer, &inner);
        // Position: 10 + 5*2 = 20, 20 + 5*2 = 30
        assert!((composed.position[0] - 20.0).abs() < 1e-6);
        assert!((composed.position[1] - 30.0).abs() < 1e-6);
        assert!((composed.scale[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn build_transform_with_partial_keyframes() {
        let base = Transform2D {
            position: [100.0, 200.0],
            scale: [1.0, 1.0],
            rotation: 45.0,
            anchor: [0.5, 0.5],
        };
        let kf = KeyframeValues {
            position_x: Some(150.0),
            scale_y: Some(2.0),
            ..Default::default()
        };
        let result = build_transform(&base, &kf);
        assert!((result.position[0] - 150.0).abs() < 1e-6); // overridden
        assert!((result.position[1] - 200.0).abs() < 1e-6); // kept
        assert!((result.scale[0] - 1.0).abs() < 1e-6); // kept
        assert!((result.scale[1] - 2.0).abs() < 1e-6); // overridden
        assert!((result.rotation - 45.0).abs() < 1e-6); // kept
    }
}
