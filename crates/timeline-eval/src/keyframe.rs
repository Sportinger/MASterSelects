//! Keyframe interpolation: linear, hold, and cubic bezier.
//!
//! Given a sorted list of keyframes and a time value, this module computes
//! the interpolated property value using the appropriate interpolation mode.

use crate::types::{AnimatableProperty, Interpolation, Keyframe, KeyframeTrack};
use ms_common::TimeCode;

/// Evaluate a keyframe track at a given time (relative to clip start).
///
/// Returns `None` if the track has no keyframes.
/// If time is before the first keyframe, returns the first keyframe's value.
/// If time is after the last keyframe, returns the last keyframe's value.
pub fn evaluate_keyframe_track(track: &KeyframeTrack, time: TimeCode) -> Option<f32> {
    evaluate_keyframes(&track.keyframes, time)
}

/// Evaluate a list of keyframes at a given time.
///
/// The keyframes must be sorted by time. Returns `None` if the list is empty.
pub fn evaluate_keyframes(keyframes: &[Keyframe], time: TimeCode) -> Option<f32> {
    if keyframes.is_empty() {
        return None;
    }

    let t = time.as_secs() as f32;

    // Before or at first keyframe
    if keyframes.len() == 1 || t <= keyframes[0].time.as_secs() as f32 {
        return Some(keyframes[0].value);
    }

    // After or at last keyframe
    let last = &keyframes[keyframes.len() - 1];
    if t >= last.time.as_secs() as f32 {
        return Some(last.value);
    }

    // Find the pair of keyframes surrounding the current time
    for i in 0..keyframes.len() - 1 {
        let kf_a = &keyframes[i];
        let kf_b = &keyframes[i + 1];
        let t_a = kf_a.time.as_secs() as f32;
        let t_b = kf_b.time.as_secs() as f32;

        if t >= t_a && t < t_b {
            return Some(interpolate(kf_a, kf_b, t));
        }
    }

    // Fallback (should not reach here if keyframes are sorted)
    Some(last.value)
}

/// Interpolate between two keyframes at the given time.
fn interpolate(kf_a: &Keyframe, kf_b: &Keyframe, t: f32) -> f32 {
    let t_a = kf_a.time.as_secs() as f32;
    let t_b = kf_b.time.as_secs() as f32;
    let dt = t_b - t_a;

    if dt <= 0.0 {
        return kf_a.value;
    }

    // Normalized time within the segment [0, 1]
    let frac = (t - t_a) / dt;

    match &kf_a.interpolation {
        Interpolation::Hold => kf_a.value,
        Interpolation::Linear => lerp(kf_a.value, kf_b.value, frac),
        Interpolation::Bezier { out_tangent, .. } => {
            // Use the outgoing tangent of kf_a and incoming tangent of kf_b
            let in_tangent = match &kf_b.interpolation {
                Interpolation::Bezier { in_tangent, .. } => *in_tangent,
                _ => [0.0, 0.0],
            };
            cubic_bezier(kf_a.value, kf_b.value, *out_tangent, in_tangent, dt, frac)
        }
    }
}

/// Linear interpolation.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Cubic bezier interpolation for keyframe values.
///
/// The tangents are specified as (dx, dy) offsets from the keyframe point.
/// `dt` is the time span between the two keyframes, used to normalize tangents.
/// `frac` is the normalized time [0, 1] within the segment.
///
/// We compute a 1D cubic Bezier where the four control points are:
///   P0 = (0, v_a)
///   P1 = (out_tangent.x / dt, v_a + out_tangent.y)
///   P2 = (1 + in_tangent.x / dt, v_b + in_tangent.y)
///   P3 = (1, v_b)
///
/// Since we want value as a function of normalized time, and for typical
/// keyframe beziers the x-coordinates may not be monotonic, we solve for the
/// parametric t that gives us `frac` in the x dimension, then evaluate y.
/// For simplicity and performance, we use an iterative Newton-Raphson approach.
fn cubic_bezier(
    v_a: f32,
    v_b: f32,
    out_tangent: [f32; 2],
    in_tangent: [f32; 2],
    dt: f32,
    frac: f32,
) -> f32 {
    // Control points in normalized [0, 1] x-space
    let x1 = (out_tangent[0] / dt).clamp(0.0, 1.0);
    let x2 = (1.0 + in_tangent[0] / dt).clamp(0.0, 1.0);

    // y control points (value space)
    let y0 = v_a;
    let y1 = v_a + out_tangent[1];
    let y2 = v_b + in_tangent[1];
    let y3 = v_b;

    // Find parametric t for the given x (frac) using Newton-Raphson
    let param_t = solve_bezier_x(x1, x2, frac);

    // Evaluate the y-dimension bezier at param_t
    eval_cubic(y0, y1, y2, y3, param_t)
}

/// Evaluate a cubic Bezier: B(t) = (1-t)^3*p0 + 3*(1-t)^2*t*p1 + 3*(1-t)*t^2*p2 + t^3*p3
fn eval_cubic(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let one_minus_t = 1.0 - t;
    let omt2 = one_minus_t * one_minus_t;
    let omt3 = omt2 * one_minus_t;
    let t2 = t * t;
    let t3 = t2 * t;
    omt3 * p0 + 3.0 * omt2 * t * p1 + 3.0 * one_minus_t * t2 * p2 + t3 * p3
}

/// Derivative of cubic Bezier: B'(t) = 3*(1-t)^2*(p1-p0) + 6*(1-t)*t*(p2-p1) + 3*t^2*(p3-p2)
fn eval_cubic_derivative(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let one_minus_t = 1.0 - t;
    3.0 * one_minus_t * one_minus_t * (p1 - p0)
        + 6.0 * one_minus_t * t * (p2 - p1)
        + 3.0 * t * t * (p3 - p2)
}

/// Solve for parametric t given x in a cubic Bezier where x0=0, x3=1.
/// Uses Newton-Raphson with fallback to bisection.
fn solve_bezier_x(x1: f32, x2: f32, target_x: f32) -> f32 {
    // x0 = 0, x3 = 1
    let mut t = target_x; // initial guess

    // Newton-Raphson iterations
    for _ in 0..8 {
        let x = eval_cubic(0.0, x1, x2, 1.0, t) - target_x;
        let dx = eval_cubic_derivative(0.0, x1, x2, 1.0, t);

        if dx.abs() < 1e-10 {
            break;
        }

        let new_t = t - x / dx;
        t = new_t.clamp(0.0, 1.0);

        if x.abs() < 1e-6 {
            return t;
        }
    }

    // Fallback: bisection for robustness
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;
    t = target_x;

    for _ in 0..16 {
        let x = eval_cubic(0.0, x1, x2, 1.0, t);
        if (x - target_x).abs() < 1e-6 {
            return t;
        }
        if x < target_x {
            lo = t;
        } else {
            hi = t;
        }
        t = (lo + hi) * 0.5;
    }

    t
}

/// Apply keyframe animation to clip properties, returning the interpolated
/// values as a set of property overrides.
///
/// `local_time` should be relative to the clip's `timeline_start`.
pub fn apply_keyframes(keyframe_tracks: &[KeyframeTrack], local_time: TimeCode) -> KeyframeValues {
    let mut values = KeyframeValues::default();

    for track in keyframe_tracks {
        if let Some(value) = evaluate_keyframe_track(track, local_time) {
            match &track.property {
                AnimatableProperty::PositionX => values.position_x = Some(value),
                AnimatableProperty::PositionY => values.position_y = Some(value),
                AnimatableProperty::ScaleX => values.scale_x = Some(value),
                AnimatableProperty::ScaleY => values.scale_y = Some(value),
                AnimatableProperty::Rotation => values.rotation = Some(value),
                AnimatableProperty::Opacity => values.opacity = Some(value),
                AnimatableProperty::EffectParam {
                    effect_idx,
                    param_name,
                } => {
                    values
                        .effect_params
                        .push((*effect_idx, param_name.clone(), value));
                }
            }
        }
    }

    values
}

/// Collected keyframe interpolation results for a clip at a given time.
#[derive(Clone, Debug, Default)]
pub struct KeyframeValues {
    pub position_x: Option<f32>,
    pub position_y: Option<f32>,
    pub scale_x: Option<f32>,
    pub scale_y: Option<f32>,
    pub rotation: Option<f32>,
    pub opacity: Option<f32>,
    /// (effect_index, param_name, value)
    pub effect_params: Vec<(usize, String, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AnimatableProperty, Interpolation, Keyframe, KeyframeTrack};

    fn make_linear_track(times_values: &[(f64, f32)]) -> KeyframeTrack {
        KeyframeTrack {
            property: AnimatableProperty::Opacity,
            keyframes: times_values
                .iter()
                .map(|&(t, v)| Keyframe {
                    time: TimeCode::from_secs(t),
                    value: v,
                    interpolation: Interpolation::Linear,
                })
                .collect(),
        }
    }

    #[test]
    fn empty_keyframes_returns_none() {
        let track = KeyframeTrack {
            property: AnimatableProperty::Opacity,
            keyframes: vec![],
        };
        assert!(evaluate_keyframe_track(&track, TimeCode::from_secs(1.0)).is_none());
    }

    #[test]
    fn single_keyframe_returns_value() {
        let track = make_linear_track(&[(0.0, 0.75)]);
        let val = evaluate_keyframe_track(&track, TimeCode::from_secs(5.0)).unwrap();
        assert!((val - 0.75).abs() < 1e-6);
    }

    #[test]
    fn linear_interpolation_midpoint() {
        let track = make_linear_track(&[(0.0, 0.0), (1.0, 1.0)]);
        let val = evaluate_keyframe_track(&track, TimeCode::from_secs(0.5)).unwrap();
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn linear_interpolation_quarter() {
        let track = make_linear_track(&[(0.0, 0.0), (2.0, 4.0)]);
        let val = evaluate_keyframe_track(&track, TimeCode::from_secs(0.5)).unwrap();
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hold_interpolation() {
        let track = KeyframeTrack {
            property: AnimatableProperty::Opacity,
            keyframes: vec![
                Keyframe {
                    time: TimeCode::from_secs(0.0),
                    value: 1.0,
                    interpolation: Interpolation::Hold,
                },
                Keyframe {
                    time: TimeCode::from_secs(1.0),
                    value: 0.0,
                    interpolation: Interpolation::Hold,
                },
            ],
        };
        // Hold should keep value of first keyframe until the second
        let val = evaluate_keyframe_track(&track, TimeCode::from_secs(0.5)).unwrap();
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn before_first_keyframe_returns_first_value() {
        let track = make_linear_track(&[(1.0, 5.0), (2.0, 10.0)]);
        let val = evaluate_keyframe_track(&track, TimeCode::from_secs(0.0)).unwrap();
        assert!((val - 5.0).abs() < 1e-6);
    }

    #[test]
    fn after_last_keyframe_returns_last_value() {
        let track = make_linear_track(&[(0.0, 0.0), (1.0, 10.0)]);
        let val = evaluate_keyframe_track(&track, TimeCode::from_secs(5.0)).unwrap();
        assert!((val - 10.0).abs() < 1e-6);
    }

    #[test]
    fn multi_segment_linear() {
        let track = make_linear_track(&[(0.0, 0.0), (1.0, 10.0), (2.0, 5.0)]);

        // First segment midpoint
        let v1 = evaluate_keyframe_track(&track, TimeCode::from_secs(0.5)).unwrap();
        assert!((v1 - 5.0).abs() < 1e-6);

        // Second segment midpoint
        let v2 = evaluate_keyframe_track(&track, TimeCode::from_secs(1.5)).unwrap();
        assert!((v2 - 7.5).abs() < 1e-6);
    }

    #[test]
    fn bezier_interpolation_endpoints() {
        let track = KeyframeTrack {
            property: AnimatableProperty::PositionX,
            keyframes: vec![
                Keyframe {
                    time: TimeCode::from_secs(0.0),
                    value: 0.0,
                    interpolation: Interpolation::Bezier {
                        in_tangent: [0.0, 0.0],
                        out_tangent: [0.333, 0.0],
                    },
                },
                Keyframe {
                    time: TimeCode::from_secs(1.0),
                    value: 100.0,
                    interpolation: Interpolation::Bezier {
                        in_tangent: [-0.333, 0.0],
                        out_tangent: [0.0, 0.0],
                    },
                },
            ],
        };

        // At start
        let v0 = evaluate_keyframe_track(&track, TimeCode::from_secs(0.0)).unwrap();
        assert!((v0 - 0.0).abs() < 1e-4);

        // At end
        let v1 = evaluate_keyframe_track(&track, TimeCode::from_secs(1.0)).unwrap();
        assert!((v1 - 100.0).abs() < 1e-4);
    }

    #[test]
    fn apply_keyframes_collects_values() {
        let tracks = vec![
            make_linear_track(&[(0.0, 0.0), (1.0, 1.0)]),
            KeyframeTrack {
                property: AnimatableProperty::PositionX,
                keyframes: vec![
                    Keyframe {
                        time: TimeCode::from_secs(0.0),
                        value: 100.0,
                        interpolation: Interpolation::Linear,
                    },
                    Keyframe {
                        time: TimeCode::from_secs(1.0),
                        value: 200.0,
                        interpolation: Interpolation::Linear,
                    },
                ],
            },
        ];

        let values = apply_keyframes(&tracks, TimeCode::from_secs(0.5));
        assert!((values.opacity.unwrap() - 0.5).abs() < 1e-6);
        assert!((values.position_x.unwrap() - 150.0).abs() < 1e-6);
    }
}
