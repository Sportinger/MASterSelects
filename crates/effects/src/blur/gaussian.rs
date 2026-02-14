//! Gaussian Blur effect (separable).

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "radius".to_string(),
            display_name: "Radius".to_string(),
            param_type: ParamType::Float {
                min: 0.0,
                max: 100.0,
            },
            default: ParamValue::Float(5.0),
        },
        ParamDef {
            name: "sigma".to_string(),
            display_name: "Sigma".to_string(),
            param_type: ParamType::Float {
                min: 0.1,
                max: 50.0,
            },
            default: ParamValue::Float(2.0),
        },
    ]
}

/// Gaussian Blur (separable two-pass).
///
/// The compositor should dispatch this kernel twice (horizontal + vertical)
/// using the `num_passes()` return value. A pass index uniform distinguishes them.
pub struct GaussianBlurEffect {
    params: Vec<ParamDef>,
}

impl GaussianBlurEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for GaussianBlurEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for GaussianBlurEffect {
    fn name(&self) -> &str {
        "gaussian_blur"
    }

    fn display_name(&self) -> &str {
        "Gaussian Blur"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Blur
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("gaussian_blur".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let radius = get_float("radius", params, &self.params);
        let sigma = get_float("sigma", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(radius)
            .push_f32(sigma)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }

    fn num_passes(&self) -> u32 {
        2 // Horizontal pass + vertical pass (separable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_blur_metadata() {
        let fx = GaussianBlurEffect::new();
        assert_eq!(fx.name(), "gaussian_blur");
        assert_eq!(fx.category(), EffectCategory::Blur);
        assert_eq!(fx.param_defs().len(), 2);
        assert_eq!(fx.num_passes(), 2);
    }

    #[test]
    fn gaussian_blur_kernel_args() {
        let fx = GaussianBlurEffect::new();
        let params = vec![
            ("radius".to_string(), ParamValue::Float(10.0)),
            ("sigma".to_string(), ParamValue::Float(4.0)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 6);
    }
}
