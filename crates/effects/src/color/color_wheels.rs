//! Color Wheels (Lift / Gamma / Gain) effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        // Lift (shadows)
        ParamDef {
            name: "lift_r".to_string(),
            display_name: "Lift Red".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "lift_g".to_string(),
            display_name: "Lift Green".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "lift_b".to_string(),
            display_name: "Lift Blue".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        // Gamma (midtones)
        ParamDef {
            name: "gamma_r".to_string(),
            display_name: "Gamma Red".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "gamma_g".to_string(),
            display_name: "Gamma Green".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "gamma_b".to_string(),
            display_name: "Gamma Blue".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        // Gain (highlights)
        ParamDef {
            name: "gain_r".to_string(),
            display_name: "Gain Red".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "gain_g".to_string(),
            display_name: "Gain Green".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "gain_b".to_string(),
            display_name: "Gain Blue".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
    ]
}

/// Color Wheels effect (Lift / Gamma / Gain per-channel adjustments).
pub struct ColorWheelsEffect {
    params: Vec<ParamDef>,
}

impl ColorWheelsEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for ColorWheelsEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for ColorWheelsEffect {
    fn name(&self) -> &str {
        "color_wheels"
    }

    fn display_name(&self) -> &str {
        "Color Wheels (Lift/Gamma/Gain)"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Color
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("color_wheels".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let lift_r = get_float("lift_r", params, &self.params);
        let lift_g = get_float("lift_g", params, &self.params);
        let lift_b = get_float("lift_b", params, &self.params);
        let gamma_r = get_float("gamma_r", params, &self.params);
        let gamma_g = get_float("gamma_g", params, &self.params);
        let gamma_b = get_float("gamma_b", params, &self.params);
        let gain_r = get_float("gain_r", params, &self.params);
        let gain_g = get_float("gain_g", params, &self.params);
        let gain_b = get_float("gain_b", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(lift_r)
            .push_f32(lift_g)
            .push_f32(lift_b)
            .push_f32(gamma_r)
            .push_f32(gamma_g)
            .push_f32(gamma_b)
            .push_f32(gain_r)
            .push_f32(gain_g)
            .push_f32(gain_b)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_wheels_metadata() {
        let fx = ColorWheelsEffect::new();
        assert_eq!(fx.name(), "color_wheels");
        assert_eq!(fx.category(), EffectCategory::Color);
        assert_eq!(fx.param_defs().len(), 9); // 3 lift + 3 gamma + 3 gain
    }

    #[test]
    fn color_wheels_kernel_args() {
        let fx = ColorWheelsEffect::new();
        let params = vec![
            ("lift_r".to_string(), ParamValue::Float(0.1)),
            ("gain_b".to_string(), ParamValue::Float(-0.2)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        // ptr, ptr, w, h, 9 floats = 13
        assert_eq!(args.len(), 13);
    }

    #[test]
    fn color_wheels_all_defaults_zero() {
        let fx = ColorWheelsEffect::new();
        for def in fx.param_defs() {
            assert_eq!(
                def.default.as_float(),
                Some(0.0),
                "param {} should default to 0.0",
                def.name
            );
        }
    }
}
