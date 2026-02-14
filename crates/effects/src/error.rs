//! Effect-specific error types.

use thiserror::Error;

/// Errors from the effect system.
#[derive(Error, Debug)]
pub enum EffectError {
    /// An unknown parameter name was supplied.
    #[error("Unknown parameter: {param}")]
    UnknownParam { param: String },

    /// A parameter value is outside its valid range.
    #[error("Parameter '{param}' value {value} out of range [{min}, {max}]")]
    ParamOutOfRange {
        param: String,
        value: String,
        min: String,
        max: String,
    },

    /// A parameter value has the wrong type.
    #[error("Parameter '{param}' type mismatch: expected {expected}, got {got}")]
    ParamTypeMismatch {
        param: String,
        expected: String,
        got: String,
    },

    /// The requested effect was not found in the registry.
    #[error("Effect not found: {name}")]
    NotFound { name: String },

    /// An effect with this name is already registered.
    #[error("Effect already registered: {name}")]
    AlreadyRegistered { name: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let err = EffectError::UnknownParam {
            param: "foo".to_string(),
        };
        assert_eq!(err.to_string(), "Unknown parameter: foo");

        let err = EffectError::ParamOutOfRange {
            param: "brightness".to_string(),
            value: "2.0".to_string(),
            min: "-1.0".to_string(),
            max: "1.0".to_string(),
        };
        assert!(err.to_string().contains("brightness"));
        assert!(err.to_string().contains("2.0"));
    }
}
