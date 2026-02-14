//! Luma Key effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::{get_bool, get_float};
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "threshold".to_string(),
            display_name: "Threshold".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.5),
        },
        ParamDef {
            name: "softness".to_string(),
            display_name: "Softness".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.1),
        },
        ParamDef {
            name: "invert".to_string(),
            display_name: "Invert".to_string(),
            param_type: ParamType::Bool,
            default: ParamValue::Bool(false),
        },
    ]
}

/// Luma Key effect (key out pixels based on luminance).
pub struct LumaKeyEffect {
    params: Vec<ParamDef>,
}

impl LumaKeyEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for LumaKeyEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for LumaKeyEffect {
    fn name(&self) -> &str {
        "luma_key"
    }

    fn display_name(&self) -> &str {
        "Luma Key"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Keying
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("luma_key".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let threshold = get_float("threshold", params, &self.params);
        let softness = get_float("softness", params, &self.params);
        let invert = get_bool("invert", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(threshold)
            .push_f32(softness)
            .push_u32(u32::from(invert))
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_key_metadata() {
        let fx = LumaKeyEffect::new();
        assert_eq!(fx.name(), "luma_key");
        assert_eq!(fx.category(), EffectCategory::Keying);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn luma_key_invert_default_false() {
        let fx = LumaKeyEffect::new();
        let inv = fx.param_defs().iter().find(|d| d.name == "invert").unwrap();
        assert_eq!(inv.default.as_bool(), Some(false));
    }

    #[test]
    fn luma_key_kernel_args_with_invert() {
        let fx = LumaKeyEffect::new();
        let params = vec![
            ("threshold".to_string(), ParamValue::Float(0.3)),
            ("softness".to_string(), ParamValue::Float(0.2)),
            ("invert".to_string(), ParamValue::Bool(true)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7); // ptr, ptr, w, h, threshold, softness, invert
    }
}
