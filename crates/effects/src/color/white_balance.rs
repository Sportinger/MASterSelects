//! White Balance effect (temperature + tint).

use ms_common::{EffectCategory, KernelArgs, KernelId, ParamDef, ParamType, ParamValue};

use crate::params::get_float;
use crate::traits::{standard_compute_grid, Effect};

fn param_defs() -> Vec<ParamDef> {
    vec![
        ParamDef {
            name: "temperature".to_string(),
            display_name: "Temperature".to_string(),
            param_type: ParamType::Float {
                min: 2000.0,
                max: 12000.0,
            },
            default: ParamValue::Float(6500.0),
        },
        ParamDef {
            name: "tint".to_string(),
            display_name: "Tint".to_string(),
            param_type: ParamType::Float {
                min: -1.0,
                max: 1.0,
            },
            default: ParamValue::Float(0.0),
        },
    ]
}

/// White Balance (color temperature + tint) adjustment.
pub struct WhiteBalanceEffect {
    params: Vec<ParamDef>,
}

impl WhiteBalanceEffect {
    pub fn new() -> Self {
        Self {
            params: param_defs(),
        }
    }
}

impl Default for WhiteBalanceEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl Effect for WhiteBalanceEffect {
    fn name(&self) -> &str {
        "white_balance"
    }

    fn display_name(&self) -> &str {
        "White Balance"
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Color
    }

    fn param_defs(&self) -> &[ParamDef] {
        &self.params
    }

    fn kernel_id(&self) -> KernelId {
        KernelId::Effect("white_balance".into())
    }

    fn build_kernel_args(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        width: u32,
        height: u32,
        params: &[(String, ParamValue)],
    ) -> KernelArgs {
        let temperature = get_float("temperature", params, &self.params);
        let tint = get_float("tint", params, &self.params);

        KernelArgs::new()
            .push_ptr(input_ptr)
            .push_ptr(output_ptr)
            .push_u32(width)
            .push_u32(height)
            .push_f32(temperature)
            .push_f32(tint)
    }

    fn compute_grid(&self, width: u32, height: u32) -> ([u32; 3], [u32; 3]) {
        standard_compute_grid(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_balance_metadata() {
        let fx = WhiteBalanceEffect::new();
        assert_eq!(fx.name(), "white_balance");
        assert_eq!(fx.category(), EffectCategory::Color);
        assert_eq!(fx.param_defs().len(), 2);
    }

    #[test]
    fn white_balance_default_temperature() {
        let fx = WhiteBalanceEffect::new();
        let temp_def = fx
            .param_defs()
            .iter()
            .find(|d| d.name == "temperature")
            .unwrap();
        assert_eq!(temp_def.default.as_float(), Some(6500.0));
    }

    #[test]
    fn white_balance_kernel_args() {
        let fx = WhiteBalanceEffect::new();
        let params = vec![
            ("temperature".to_string(), ParamValue::Float(5500.0)),
            ("tint".to_string(), ParamValue::Float(0.2)),
        ];
        let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &params);
        assert_eq!(args.len(), 6);
    }
}
