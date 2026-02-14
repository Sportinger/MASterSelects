//! Directional (Motion) Blur effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "angle".to_string(),
            display_name: "Angle".to_string(),
            param_type: ParamType::Angle,
            default: ParamValue::Angle(0.0),
        },
        ParamDef {
            name: "distance".to_string(),
            display_name: "Distance".to_string(),
            param_type: ParamType::Float {
                min: 0.0,
                max: 100.0,
            },
            default: ParamValue::Float(10.0),
        },
    ]
}

/// Directional (Motion) Blur effect.
pub struct DirectionalBlurEffect {
    params: Vec<ParamDef>,
}

impl DirectionalBlurEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for DirectionalBlurEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for DirectionalBlurEffect {
    fn name(&self) -> &str {
        "directional_blur"
    }

    fn display_name(&self) -> &str {
        "Directional Blur"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Blur
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("directional_blur".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let angle = get_float("angle", params, &self.params);
        let distance = get_float("distance", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(angle)
            .push_f32(distance)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directional_blur_metadata() {
        let fx = DirectionalBlurEffect::new();
        assert_eq!(fx.name(), "directional_blur");
        assert_eq!(fx.category(), EffectCategory::Blur);
        assert_eq!(fx.param_defs().len(), 2);
        assert_eq!(fx.num_passes(), 1);
    }

    #[test]
    fn directional_blur_kernel_args() {
        let fx = DirectionalBlurEffect::new();
        let params = vec![
            ("angle".to_string(), ParamValue::Angle(90.0)),
            ("distance".to_string(), ParamValue::Float(20.0)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 6);
    }
}
