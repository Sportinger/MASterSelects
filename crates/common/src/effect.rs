//! Effect definitions, parameter types, and effect instances.

use serde::{Deserialize, Serialize};

/// Unique effect identifier (matches registry name).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectId(pub String);

impl EffectId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Effect category for UI grouping.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectCategory {
    Color,
    Blur,
    Distort,
    Keying,
    Stylize,
    Transform,
    Generate,
}

impl EffectCategory {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Color => "Color",
            Self::Blur => "Blur",
            Self::Distort => "Distort",
            Self::Keying => "Keying",
            Self::Stylize => "Stylize",
            Self::Transform => "Transform",
            Self::Generate => "Generate",
        }
    }
}

/// Parameter definition for an effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamDef {
    pub name: String,
    pub display_name: String,
    pub param_type: ParamType,
    pub default: ParamValue,
}

/// Parameter type with constraints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ParamType {
    Float { min: f32, max: f32 },
    Int { min: i32, max: i32 },
    Bool,
    Color,
    Enum { options: Vec<String> },
    Vec2,
    Angle,
}

/// Concrete parameter value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Color([f32; 4]),
    Enum(u32),
    Vec2([f32; 2]),
    Angle(f32),
}

impl ParamValue {
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// An instance of an effect applied to a layer, with concrete parameter values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectInstance {
    pub effect_id: EffectId,
    pub enabled: bool,
    pub params: Vec<(String, ParamValue)>,
}

impl EffectInstance {
    pub fn new(effect_id: EffectId) -> Self {
        Self {
            effect_id,
            enabled: true,
            params: Vec::new(),
        }
    }

    pub fn with_param(mut self, name: impl Into<String>, value: ParamValue) -> Self {
        self.params.push((name.into(), value));
        self
    }

    pub fn get_param(&self, name: &str) -> Option<&ParamValue> {
        self.params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_instance_builder() {
        let fx = EffectInstance::new(EffectId::new("brightness"))
            .with_param("amount", ParamValue::Float(0.5))
            .with_param("contrast", ParamValue::Float(1.2));

        assert_eq!(fx.params.len(), 2);
        assert_eq!(fx.get_param("amount").unwrap().as_float(), Some(0.5));
    }
}
