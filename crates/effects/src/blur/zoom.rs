//! Zoom Blur effect.

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
            name: "strength".to_string(),
            display_name: "Strength".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.1),
        },
    ]
}

/// Zoom Blur effect (blur along rays from center).
pub struct ZoomBlurEffect {
    params: Vec<ParamDef>,
}

impl ZoomBlurEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for ZoomBlurEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for ZoomBlurEffect {
    fn name(&self) -> &str {
        "zoom_blur"
    }

    fn display_name(&self) -> &str {
        "Zoom Blur"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Blur
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("zoom_blur".into())
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
        let strength = get_float("strength", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(center_x)
            .push_f32(center_y)
            .push_f32(strength)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_blur_metadata() {
        let fx = ZoomBlurEffect::new();
        assert_eq!(fx.name(), "zoom_blur");
        assert_eq!(fx.category(), EffectCategory::Blur);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn zoom_blur_kernel_args() {
        let fx = ZoomBlurEffect::new();
        let params = vec![
            ("center_x".to_string(), ParamValue::Float(0.5)),
            ("center_y".to_string(), ParamValue::Float(0.5)),
            ("strength".to_string(), ParamValue::Float(0.3)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7);
    }
}
