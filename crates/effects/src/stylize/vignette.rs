//! Vignette effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "amount".to_string(),
            display_name: "Amount".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(0.5),
        },
        ParamDef {
            name: "radius".to_string(),
            display_name: "Radius".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(0.8),
        },
        ParamDef {
            name: "softness".to_string(),
            display_name: "Softness".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.3),
        },
    ]
}

/// Vignette effect (darken edges of the frame).
pub struct VignetteEffect {
    params: Vec<ParamDef>,
}

impl VignetteEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for VignetteEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for VignetteEffect {
    fn name(&self) -> &str {
        "vignette"
    }

    fn display_name(&self) -> &str {
        "Vignette"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Stylize
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("vignette".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let amount = get_float("amount", params, &self.params);
        let radius = get_float("radius", params, &self.params);
        let softness = get_float("softness", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(amount)
            .push_f32(radius)
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
    fn vignette_metadata() {
        let fx = VignetteEffect::new();
        assert_eq!(fx.name(), "vignette");
        assert_eq!(fx.category(), EffectCategory::Stylize);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn vignette_kernel_args() {
        let fx = VignetteEffect::new();
        let params = vec![
            ("amount".to_string(), ParamValue::Float(1.0)),
            ("radius".to_string(), ParamValue::Float(0.6)),
            ("softness".to_string(), ParamValue::Float(0.5)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7);
    }

    #[test]
    fn vignette_defaults() {
        let fx = VignetteEffect::new();
        let amount = fx.param_defs().iter().find(|d| d.name == "amount").unwrap();
        let radius = fx.param_defs().iter().find(|d| d.name == "radius").unwrap();
        let softness = fx
            .param_defs()
            .iter()
            .find(|d| d.name == "softness")
            .unwrap();
        assert_eq!(amount.default.as_float(), Some(0.5));
        assert_eq!(radius.default.as_float(), Some(0.8));
        assert_eq!(softness.default.as_float(), Some(0.3));
    }
}
