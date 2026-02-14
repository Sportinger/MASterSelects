//! Radial Blur effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "center_x".to_string(),
            display_name: "Center X".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.5),
        },
        ParamDef {
            name: "center_y".to_string(),
            display_name: "Center Y".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.5),
        },
        ParamDef {
            name: "amount".to_string(),
            display_name: "Amount".to_string(),
            param_type: ParamType::Float {
                min: 0.0,
                max: 100.0,
            },
            default: ParamValue::Float(10.0),
        },
    ]
}

/// Radial Blur effect (blur radiating from a center point).
pub struct RadialBlurEffect {
    params: Vec<ParamDef>,
}

impl RadialBlurEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for RadialBlurEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for RadialBlurEffect {
    fn name(&self) -> &str {
        "radial_blur"
    }

    fn display_name(&self) -> &str {
        "Radial Blur"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Blur
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("radial_blur".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let center_x = get_float("center_x", params, &self.params);
        let center_y = get_float("center_y", params, &self.params);
        let amount = get_float("amount", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(center_x)
            .push_f32(center_y)
            .push_f32(amount)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_blur_metadata() {
        let fx = RadialBlurEffect::new();
        assert_eq!(fx.name(), "radial_blur");
        assert_eq!(fx.category(), EffectCategory::Blur);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn radial_blur_center_defaults() {
        let fx = RadialBlurEffect::new();
        let cx = fx
            .param_defs()
            .iter()
            .find(|d| d.name == "center_x")
            .unwrap();
        let cy = fx
            .param_defs()
            .iter()
            .find(|d| d.name == "center_y")
            .unwrap();
        assert_eq!(cx.default.as_float(), Some(0.5));
        assert_eq!(cy.default.as_float(), Some(0.5));
    }

    #[test]
    fn radial_blur_kernel_args() {
        let fx = RadialBlurEffect::new();
        let params = vec![
            ("center_x".to_string(), ParamValue::Float(0.3)),
            ("center_y".to_string(), ParamValue::Float(0.7)),
            ("amount".to_string(), ParamValue::Float(25.0)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7);
    }
}
