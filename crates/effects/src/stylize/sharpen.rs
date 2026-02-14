//! Unsharp Mask (Sharpen) effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "amount".to_string(),
            display_name: "Amount".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 5.0 },
            default: ParamValue::Float(1.0),
        },
        ParamDef {
            name: "radius".to_string(),
            display_name: "Radius".to_string(),
            param_type: ParamType::Float {
                min: 0.5,
                max: 10.0,
            },
            default: ParamValue::Float(1.0),
        },
    ]
}

/// Unsharp Mask (sharpening) effect.
pub struct SharpenEffect {
    params: Vec<ParamDef>,
}

impl SharpenEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for SharpenEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for SharpenEffect {
    fn name(&self) -> &str {
        "sharpen"
    }

    fn display_name(&self) -> &str {
        "Sharpen"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Stylize
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("sharpen".into())
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

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(amount)
            .push_f32(radius)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharpen_metadata() {
        let fx = SharpenEffect::new();
        assert_eq!(fx.name(), "sharpen");
        assert_eq!(fx.category(), EffectCategory::Stylize);
        assert_eq!(fx.param_defs().len(), 2);
    }

    #[test]
    fn sharpen_kernel_args() {
        let fx = SharpenEffect::new();
        let params = vec![
            ("amount".to_string(), ParamValue::Float(2.0)),
            ("radius".to_string(), ParamValue::Float(1.5)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 6);
    }
}
