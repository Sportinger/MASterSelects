//! Brightness / Contrast effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

// We cannot use String::new() in const context for non-empty strings,
// so we use a function to produce the definitions.
fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "brightness".to_string(),
            display_name: "Brightness".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
        ParamDef {
            name: "contrast".to_string(),
            display_name: "Contrast".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 3.0 },
            default: ParamValue::Float(1.0),
        },
    ]
}

/// Brightness / Contrast adjustment effect.
pub struct BrightnessEffect {
    params: Vec<ParamDef>,
}

impl BrightnessEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for BrightnessEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for BrightnessEffect {
    fn name(&self) -> &str {
        "brightness_contrast"
    }

    fn display_name(&self) -> &str {
        "Brightness / Contrast"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Color
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("brightness_contrast".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let brightness = get_float("brightness", params, &self.params);
        let contrast = get_float("contrast", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(brightness)
            .push_f32(contrast)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brightness_effect_metadata() {
        let fx = BrightnessEffect::new();
        assert_eq!(fx.name(), "brightness_contrast");
        assert_eq!(fx.display_name(), "Brightness / Contrast");
        assert_eq!(fx.category(), EffectCategory::Color);
        assert_eq!(fx.param_defs().len(), 2);
        assert_eq!(fx.num_passes(), 1);
    }

    #[test]
    fn brightness_kernel_args() {
        let fx = BrightnessEffect::new();
        let params = vec![
            ("brightness".to_string(), ParamValue::Float(0.5)),
            ("contrast".to_string(), ParamValue::Float(1.5)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 6); // ptr, ptr, w, h, brightness, contrast
    }

    #[test]
    fn brightness_uses_defaults() {
        let fx = BrightnessEffect::new();
        let params: Vec<(String, ParamValue)> = vec![];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 6);
    }

    #[test]
    fn brightness_grid_dimensions() {
        let fx = BrightnessEffect::new();
        let (grid, block) = fx.compute_grid(1920, 1080);
        assert_eq!(block, [16, 16, 1]);
        assert_eq!(grid[0], 120);
        assert_eq!(grid[1], 68);
    }
}
