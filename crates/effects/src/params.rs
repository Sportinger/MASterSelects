//! Parameter validation and conversion helpers.

use ms_common::{ParamDef, ParamType, ParamValue};

use crate::error::EffectError;

/// Validate parameter values against their definitions.
///
/// Checks that:
/// - Every supplied parameter has a matching definition
/// - Float/Int values are within their defined ranges
/// - Enum values are within the number of defined options
/// - Value types match the parameter type
pub fn validate_params(
    defs: &[ParamDef],
    params: &[(String, ParamValue)],
) -> Result<(), EffectError> {
    for (name, value) in params {
        let def =
            defs.iter()
                .find(|d| d.name == *name)
                .ok_or_else(|| EffectError::UnknownParam {
                    param: name.clone(),
                })?;

        validate_single_param(def, value)?;
    }
    Ok(())
}

/// Validate a single parameter value against its definition.
fn validate_single_param(def: &ParamDef, value: &ParamValue) -> Result<(), EffectError> {
    match (&def.param_type, value) {
        (ParamType::Float { min, max }, ParamValue::Float(v)) => {
            if *v < *min || *v > *max {
                return Err(EffectError::ParamOutOfRange {
                    param: def.name.clone(),
                    value: format!("{v}"),
                    min: format!("{min}"),
                    max: format!("{max}"),
                });
            }
        }
        (ParamType::Int { min, max }, ParamValue::Int(v)) => {
            if *v < *min || *v > *max {
                return Err(EffectError::ParamOutOfRange {
                    param: def.name.clone(),
                    value: format!("{v}"),
                    min: format!("{min}"),
                    max: format!("{max}"),
                });
            }
        }
        (ParamType::Bool, ParamValue::Bool(_)) => {}
        (ParamType::Color, ParamValue::Color(_)) => {}
        (ParamType::Enum { options }, ParamValue::Enum(idx)) => {
            if *idx as usize >= options.len() {
                return Err(EffectError::ParamOutOfRange {
                    param: def.name.clone(),
                    value: format!("{idx}"),
                    min: "0".to_string(),
                    max: format!("{}", options.len().saturating_sub(1)),
                });
            }
        }
        (ParamType::Vec2, ParamValue::Vec2(_)) => {}
        (ParamType::Angle, ParamValue::Angle(_)) => {}
        // Also allow Angle where Float is expected and vice-versa (angles are stored as floats)
        (ParamType::Angle, ParamValue::Float(_)) => {}
        (ParamType::Float { .. }, ParamValue::Angle(_)) => {}
        _ => {
            return Err(EffectError::ParamTypeMismatch {
                param: def.name.clone(),
                expected: format!("{:?}", def.param_type),
                got: format!("{value:?}"),
            });
        }
    }
    Ok(())
}

/// Get a parameter value from the supplied params, falling back to the default.
///
/// Returns `None` only if the parameter name is not found in either `params` or `defs`.
pub fn get_param_or_default<'a>(
    name: &str,
    params: &'a [(String, ParamValue)],
    defs: &'a [ParamDef],
) -> Option<&'a ParamValue> {
    // First try to find in supplied params
    if let Some(val) = params.iter().find(|(n, _)| n == name).map(|(_, v)| v) {
        return Some(val);
    }
    // Fall back to default from definitions
    defs.iter().find(|d| d.name == name).map(|d| &d.default)
}

/// Helper to extract a float parameter, with fallback to default.
pub fn get_float(name: &str, params: &[(String, ParamValue)], defs: &[ParamDef]) -> f32 {
    get_param_or_default(name, params, defs)
        .and_then(|v| match v {
            ParamValue::Float(f) => Some(*f),
            ParamValue::Angle(f) => Some(*f),
            _ => None,
        })
        .unwrap_or(0.0)
}

/// Helper to extract an int parameter, with fallback to default.
pub fn get_int(name: &str, params: &[(String, ParamValue)], defs: &[ParamDef]) -> i32 {
    get_param_or_default(name, params, defs)
        .and_then(|v| v.as_int())
        .unwrap_or(0)
}

/// Helper to extract a bool parameter, with fallback to default.
pub fn get_bool(name: &str, params: &[(String, ParamValue)], defs: &[ParamDef]) -> bool {
    get_param_or_default(name, params, defs)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Helper to extract a color parameter, with fallback to default.
pub fn get_color(name: &str, params: &[(String, ParamValue)], defs: &[ParamDef]) -> [f32; 4] {
    get_param_or_default(name, params, defs)
        .and_then(|v| match v {
            ParamValue::Color(c) => Some(*c),
            _ => None,
        })
        .unwrap_or([0.0, 0.0, 0.0, 1.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_defs() -> Vec<ParamDef> {
        vec![
            ParamDef {
                name: "brightness".to_string(),
                display_name: "Brightness".to_string(),
                param_type: ParamType::Float {
                    min: -1.0,
                    max: 1.0,
                },
                default: ParamValue::Float(0.0),
            },
            ParamDef {
                name: "contrast".to_string(),
                display_name: "Contrast".to_string(),
                param_type: ParamType::Float { min: 0.0, max: 3.0 },
                default: ParamValue::Float(1.0),
            },
            ParamDef {
                name: "invert".to_string(),
                display_name: "Invert".to_string(),
                param_type: ParamType::Bool,
                default: ParamValue::Bool(false),
            },
        ]
    }

    #[test]
    fn validate_valid_params() {
        let defs = sample_defs();
        let params = vec![
            ("brightness".to_string(), ParamValue::Float(0.5)),
            ("contrast".to_string(), ParamValue::Float(1.5)),
        ];
        assert!(validate_params(&defs, &params).is_ok());
    }

    #[test]
    fn validate_out_of_range() {
        let defs = sample_defs();
        let params = vec![("brightness".to_string(), ParamValue::Float(2.0))];
        let err = validate_params(&defs, &params).unwrap_err();
        assert!(matches!(err, EffectError::ParamOutOfRange { .. }));
    }

    #[test]
    fn validate_unknown_param() {
        let defs = sample_defs();
        let params = vec![("nonexistent".to_string(), ParamValue::Float(0.0))];
        let err = validate_params(&defs, &params).unwrap_err();
        assert!(matches!(err, EffectError::UnknownParam { .. }));
    }

    #[test]
    fn validate_type_mismatch() {
        let defs = sample_defs();
        let params = vec![("brightness".to_string(), ParamValue::Bool(true))];
        let err = validate_params(&defs, &params).unwrap_err();
        assert!(matches!(err, EffectError::ParamTypeMismatch { .. }));
    }

    #[test]
    fn get_param_returns_supplied_value() {
        let defs = sample_defs();
        let params = vec![("brightness".to_string(), ParamValue::Float(0.7))];
        let val = get_param_or_default("brightness", &params, &defs).unwrap();
        assert_eq!(val.as_float(), Some(0.7));
    }

    #[test]
    fn get_param_returns_default_when_missing() {
        let defs = sample_defs();
        let params: Vec<(String, ParamValue)> = vec![];
        let val = get_param_or_default("contrast", &params, &defs).unwrap();
        assert_eq!(val.as_float(), Some(1.0));
    }

    #[test]
    fn get_param_returns_none_for_unknown() {
        let defs = sample_defs();
        let params: Vec<(String, ParamValue)> = vec![];
        assert!(get_param_or_default("nonexistent", &params, &defs).is_none());
    }

    #[test]
    fn get_float_helper() {
        let defs = sample_defs();
        let params = vec![("brightness".to_string(), ParamValue::Float(0.3))];
        assert!((get_float("brightness", &params, &defs) - 0.3).abs() < f32::EPSILON);
        // Missing param returns default
        assert!((get_float("contrast", &params, &defs) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn get_bool_helper() {
        let defs = sample_defs();
        let params = vec![("invert".to_string(), ParamValue::Bool(true))];
        assert!(get_bool("invert", &params, &defs));
        // Missing uses default
        let empty: Vec<(String, ParamValue)> = vec![];
        assert!(!get_bool("invert", &empty, &defs));
    }
}
