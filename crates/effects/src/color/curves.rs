//! RGB Curves (simplified channel multipliers + gamma).

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "red_mult".to_string(),
            display_name: "Red".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(1.0),
        },
        ParamDef {
            name: "green_mult".to_string(),
            display_name: "Green".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(1.0),
        },
        ParamDef {
            name: "blue_mult".to_string(),
            display_name: "Blue".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(1.0),
        },
        ParamDef {
            name: "gamma".to_string(),
            display_name: "Gamma".to_string(),
            param_type: ParamType::Float { min: 0.1, max: 3.0 },
            default: ParamValue::Float(1.0),
        },
    ]
}

/// RGB Curves effect (simplified: per-channel multiplier + gamma).
pub struct CurvesEffect {
    params: Vec<ParamDef>,
}

impl CurvesEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for CurvesEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for CurvesEffect {
    fn name(&self) -> &str {
        "curves"
    }

    fn display_name(&self) -> &str {
        "RGB Curves"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Color
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("curves".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let red = get_float("red_mult", params, &self.params);
        let green = get_float("green_mult", params, &self.params);
        let blue = get_float("blue_mult", params, &self.params);
        let gamma = get_float("gamma", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(red)
            .push_f32(green)
            .push_f32(blue)
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
    fn curves_effect_metadata() {
        let fx = CurvesEffect::new();
        assert_eq!(fx.name(), "curves");
        assert_eq!(fx.display_name(), "RGB Curves");
        assert_eq!(fx.category(), EffectCategory::Color);
        assert_eq!(fx.param_defs().len(), 4);
    }

    #[test]
    fn curves_kernel_args() {
        let fx = CurvesEffect::new();
        let params = vec![
            ("red_mult".to_string(), ParamValue::Float(1.2)),
            ("green_mult".to_string(), ParamValue::Float(0.9)),
            ("blue_mult".to_string(), ParamValue::Float(1.1)),
            ("gamma".to_string(), ParamValue::Float(1.5)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 8);
    }
}
