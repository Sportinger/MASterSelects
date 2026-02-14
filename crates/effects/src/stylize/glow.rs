//! Glow / Bloom effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "intensity".to_string(),
            display_name: "Intensity".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(0.5),
        },
        ParamDef {
            name: "radius".to_string(),
            display_name: "Radius".to_string(),
            param_type: ParamType::Float {
                min: 0.0,
                max: 50.0,
            },
            default: ParamValue::Float(10.0),
        },
        ParamDef {
            name: "threshold".to_string(),
            display_name: "Threshold".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.5),
        },
    ]
}

/// Glow / Bloom effect (bright areas emit a soft glow).
pub struct GlowEffect {
    params: Vec<ParamDef>,
}

impl GlowEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for GlowEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for GlowEffect {
    fn name(&self) -> &str {
        "glow"
    }

    fn display_name(&self) -> &str {
        "Glow"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Stylize
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("glow".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let intensity = get_float("intensity", params, &self.params);
        let radius = get_float("radius", params, &self.params);
        let threshold = get_float("threshold", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(intensity)
            .push_f32(radius)
            .push_f32(threshold)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glow_metadata() {
        let fx = GlowEffect::new();
        assert_eq!(fx.name(), "glow");
        assert_eq!(fx.category(), EffectCategory::Stylize);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn glow_kernel_args() {
        let fx = GlowEffect::new();
        let params = vec![
            ("intensity".to_string(), ParamValue::Float(1.0)),
            ("radius".to_string(), ParamValue::Float(15.0)),
            ("threshold".to_string(), ParamValue::Float(0.3)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7);
    }
}
