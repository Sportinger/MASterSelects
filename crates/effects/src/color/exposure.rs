//! Exposure adjustment effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "exposure".to_string(),
            display_name: "Exposure".to_string(),
            param_type: ParamType::Float {
                min: -5.0,
                max: 5.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "gamma".to_string(),
            display_name: "Gamma".to_string(),
            param_type: ParamType::Float { min: 0.1, max: 3.0 },
            default: ParamValue::Float(1.0),
        },
    ]
}

/// Exposure and gamma correction effect.
pub struct ExposureEffect {
    params: Vec<ParamDef>,
}

impl ExposureEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for ExposureEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for ExposureEffect {
    fn name(&self) -> &str {
        "exposure"
    }

    fn display_name(&self) -> &str {
        "Exposure"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Color
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("exposure".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let exposure = get_float("exposure", params, &self.params);
        let gamma = get_float("gamma", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(exposure)
            .push_f32(gamma)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_effect_metadata() {
        let fx = ExposureEffect::new();
        assert_eq!(fx.name(), "exposure");
        assert_eq!(fx.category(), EffectCategory::Color);
        assert_eq!(fx.param_defs().len(), 2);
    }

    #[test]
    fn exposure_kernel_args() {
        let fx = ExposureEffect::new();
        let params = vec![
            ("exposure".to_string(), ParamValue::Float(1.5)),
            ("gamma".to_string(), ParamValue::Float(0.8)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 6);
    }
}
