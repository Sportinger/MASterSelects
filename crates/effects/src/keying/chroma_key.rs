//! Chroma Key (green/blue screen) effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::{get_color, get_float};
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "key_color".to_string(),
            display_name: "Key Color".to_string(),
            param_type: ParamType::Color,
            default: ParamValue::Color([0.0, 1.0, 0.0, 1.0]), // Green
        },
        ParamDef {
            name: "tolerance".to_string(),
            display_name: "Tolerance".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.3),
        },
        ParamDef {
            name: "softness".to_string(),
            display_name: "Softness".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.1),
        },
    ]
}

/// Chroma Key effect (green/blue screen removal).
pub struct ChromaKeyEffect {
    params: Vec<ParamDef>,
}

impl ChromaKeyEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for ChromaKeyEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for ChromaKeyEffect {
    fn name(&self) -> &str {
        "chroma_key"
    }

    fn display_name(&self) -> &str {
        "Chroma Key"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Keying
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("chroma_key".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let key_color = get_color("key_color", params, &self.params);
        let tolerance = get_float("tolerance", params, &self.params);
        let softness = get_float("softness", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_vec4(key_color)
            .push_f32(tolerance)
            .push_f32(softness)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_key_metadata() {
        let fx = ChromaKeyEffect::new();
        assert_eq!(fx.name(), "chroma_key");
        assert_eq!(fx.category(), EffectCategory::Keying);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn chroma_key_default_is_green() {
        let fx = ChromaKeyEffect::new();
        let color_param = fx
            .param_defs()
            .iter()
            .find(|d| d.name == "key_color")
            .unwrap();
        match &color_param.default {
            ParamValue::Color(c) => {
                assert!((c[0]).abs() < f32::EPSILON); // R = 0
                assert!((c[1] - 1.0).abs() < f32::EPSILON); // G = 1
                assert!((c[2]).abs() < f32::EPSILON); // B = 0
            }
            _ => panic!("Expected Color param"),
        }
    }

    #[test]
    fn chroma_key_kernel_args() {
        let fx = ChromaKeyEffect::new();
        let params = vec![
            (
                "key_color".to_string(),
                ParamValue::Color([0.0, 0.0, 1.0, 1.0]),
            ),
            ("tolerance".to_string(), ParamValue::Float(0.4)),
            ("softness".to_string(), ParamValue::Float(0.2)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7); // ptr, ptr, w, h, vec4, f32, f32
    }
}
