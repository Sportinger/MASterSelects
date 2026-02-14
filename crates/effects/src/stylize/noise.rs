//! Film Grain / Noise effect.

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::{get_bool, get_float, get_int};
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "amount".to_string(),
            display_name: "Amount".to_string(),
            param_type: ParamType::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.1),
        },
        ParamDef {
            name: "monochrome".to_string(),
            display_name: "Monochrome".to_string(),
            param_type: ParamType::Bool,
            default: ParamValue::Bool(false),
        },
        ParamDef {
            name: "seed".to_string(),
            display_name: "Seed".to_string(),
            param_type: ParamType::Int { min: 0, max: 65535 },
            default: ParamValue::Int(0),
        },
    ]
}

/// Film Grain / Noise effect.
pub struct NoiseGrainEffect {
    params: Vec<ParamDef>,
}

impl NoiseGrainEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for NoiseGrainEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for NoiseGrainEffect {
    fn name(&self) -> &str {
        "noise_grain"
    }

    fn display_name(&self) -> &str {
        "Film Grain"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Stylize
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("noise_grain".into())
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
        let monochrome = get_bool("monochrome", params, &self.params);
        let seed = get_int("seed", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(amount)
            .push_u32(u32::from(monochrome))
            .push_i32(seed)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_grain_metadata() {
        let fx = NoiseGrainEffect::new();
        assert_eq!(fx.name(), "noise_grain");
        assert_eq!(fx.display_name(), "Film Grain");
        assert_eq!(fx.category(), EffectCategory::Stylize);
        assert_eq!(fx.param_defs().len(), 3);
    }

    #[test]
    fn noise_grain_kernel_args() {
        let fx = NoiseGrainEffect::new();
        let params = vec![
            ("amount".to_string(), ParamValue::Float(0.3)),
            ("monochrome".to_string(), ParamValue::Bool(true)),
            ("seed".to_string(), ParamValue::Int(42)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 7); // ptr, ptr, w, h, amount, monochrome, seed
    }

    #[test]
    fn noise_grain_defaults() {
        let fx = NoiseGrainEffect::new();
        let amount = fx.param_defs().iter().find(|d| d.name == "amount").unwrap();
        let mono = fx
            .param_defs()
            .iter()
            .find(|d| d.name == "monochrome")
            .unwrap();
        let seed = fx.param_defs().iter().find(|d| d.name == "seed").unwrap();
        assert_eq!(amount.default.as_float(), Some(0.1));
        assert_eq!(mono.default.as_bool(), Some(false));
        assert_eq!(seed.default.as_int(), Some(0));
    }
}
