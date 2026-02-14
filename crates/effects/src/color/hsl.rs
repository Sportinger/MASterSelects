//! Hue / Saturation / Lightness adjustment effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "hue".to_string(),
            display_name: "Hue".to_string(),
            param_type: ParamType::Angle,
            default: ParamValue::Angle(0.0),
        },
        ParamDef {
            name: "saturation".to_string(),
            display_name: "Saturation".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "lightness".to_string(),
            display_name: "Lightness".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
    ]
}

/// Hue / Saturation / Lightness adjustment.
pub struct HslEffect {
    params: Vec<ParamDef>,
}

impl HslEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for HslEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for HslEffect {
    fn name(&self) -> &str {
        "hsl_adjust"
    }

    fn display_name(&self) -> &str {
        "Hue / Saturation / Lightness"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Color
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("hsl_adjust".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let hue = get_float("hue", params, &self.params);
        let saturation = get_float("saturation", params, &self.params);
        let lightness = get_float("lightness", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(hue)
            .push_f32(saturation)
            .push_f32(lightness)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsl_effect_metadata() {
        let fx = HslEffect::new();
        assert_eq!(fx.name(), "hsl_adjust");
        assert_eq!(fx.category(), EffectCategory::Color);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn hsl_kernel_args() {
        let fx = HslEffect::new();
        let params = vec![
            ("hue".to_string(), ParamValue::Angle(45.0)),
            ("saturation".to_string(), ParamValue::Float(0.3)),
            ("lightness".to_string(), ParamValue::Float(-0.1)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7);
    }
}
