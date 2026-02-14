//! `ms-effects` — GPU effect registry, parameter definitions, and kernel dispatch logic.
//!
//! This crate provides:
//! - The [`Effect`] trait that all GPU effects implement
//! - An [`EffectRegistry`] for by-name and by-category lookup
//! - Parameter validation and conversion helpers
//! - 16 built-in effects across 4 categories:
//!
//! ## Color (6 effects)
//! - Brightness / Contrast
//! - Hue / Saturation / Lightness
//! - RGB Curves
//! - Exposure
//! - White Balance
//! - Color Wheels (Lift/Gamma/Gain)
//!
//! ## Blur (4 effects)
//! - Gaussian Blur (separable, 2-pass)
//! - Directional (Motion) Blur
//! - Radial Blur
//! - Zoom Blur
//!
//! ## Keying (2 effects)
//! - Chroma Key
//! - Luma Key
//!
//! ## Stylize (4 effects)
//! - Glow / Bloom
//! - Sharpen (Unsharp Mask)
//! - Film Grain / Noise
//! - Vignette

pub mod blur;
pub mod color;
pub mod error;
pub mod keying;
pub mod params;
pub mod registry;
pub mod stylize;
pub mod traits;

// Re-export primary types at crate root.
pub use error::EffectError;
pub use params::{get_param_or_default, validate_params};
pub use registry::EffectRegistry;
pub use traits::Effect;

// Re-export all effect structs for convenience.
pub use blur::{DirectionalBlurEffect, GaussianBlurEffect, RadialBlurEffect, ZoomBlurEffect};
pub use color::{
    BrightnessEffect, ColorWheelsEffect, CurvesEffect, ExposureEffect, HslEffect,
    WhiteBalanceEffect,
};
pub use keying::{ChromaKeyEffect, LumaKeyEffect};
pub use stylize::{GlowEffect, NoiseGrainEffect, SharpenEffect, VignetteEffect};

#[cfg(test)]
mod tests {
    use ms_common::{EffectCategory, ParamValue};

    use super::*;

    #[test]
    fn all_builtin_effects_registered() {
        let reg = EffectRegistry::with_builtins();
        assert_eq!(reg.len(), 16);
    }

    #[test]
    fn all_effects_have_unique_names() {
        let reg = EffectRegistry::with_builtins();
        let names: Vec<&str> = reg.list().iter().map(|e| e.name()).collect();
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "Duplicate effect names found");
    }

    #[test]
    fn all_effects_have_kernel_ids() {
        let reg = EffectRegistry::with_builtins();
        for fx in reg.list() {
            let kid = fx.kernel_id();
            let entry = kid.entry_point();
            assert!(
                !entry.is_empty(),
                "Effect '{}' has empty kernel entry point",
                fx.name()
            );
        }
    }

    #[test]
    fn all_effects_have_param_defs() {
        let reg = EffectRegistry::with_builtins();
        for fx in reg.list() {
            let defs = fx.param_defs();
            assert!(
                !defs.is_empty(),
                "Effect '{}' has no parameter definitions",
                fx.name()
            );
            // Every param def should have a non-empty name
            for def in defs {
                assert!(
                    !def.name.is_empty(),
                    "Effect '{}' has a param with empty name",
                    fx.name()
                );
            }
        }
    }

    #[test]
    fn all_effects_produce_valid_kernel_args() {
        let reg = EffectRegistry::with_builtins();
        for fx in reg.list() {
            // Build args with defaults (empty params)
            let args = fx.build_kernel_args(0x1000, 0x2000, 1920, 1080, &[]);
            // At minimum: input_ptr, output_ptr, width, height = 4 args
            assert!(
                args.len() >= 4,
                "Effect '{}' produces fewer than 4 kernel args",
                fx.name()
            );
        }
    }

    #[test]
    fn all_effects_compute_valid_grids() {
        let reg = EffectRegistry::with_builtins();
        for fx in reg.list() {
            let (grid, block) = fx.compute_grid(1920, 1080);
            assert!(grid[0] > 0 && grid[1] > 0 && grid[2] > 0);
            assert!(block[0] > 0 && block[1] > 0 && block[2] > 0);
        }
    }

    #[test]
    fn validate_brightness_params() {
        let fx = BrightnessEffect::new();
        let valid = vec![
            ("brightness".to_string(), ParamValue::Float(0.5)),
            ("contrast".to_string(), ParamValue::Float(1.5)),
        ];
        assert!(validate_params(fx.param_defs(), &valid).is_ok());

        let invalid = vec![("brightness".to_string(), ParamValue::Float(5.0))];
        assert!(validate_params(fx.param_defs(), &invalid).is_err());
    }

    #[test]
    fn category_counts() {
        let reg = EffectRegistry::with_builtins();
        assert_eq!(reg.list_by_category(EffectCategory::Color).len(), 6);
        assert_eq!(reg.list_by_category(EffectCategory::Blur).len(), 4);
        assert_eq!(reg.list_by_category(EffectCategory::Keying).len(), 2);
        assert_eq!(reg.list_by_category(EffectCategory::Stylize).len(), 4);
        assert_eq!(reg.list_by_category(EffectCategory::Distort).len(), 0);
        assert_eq!(reg.list_by_category(EffectCategory::Transform).len(), 0);
        assert_eq!(reg.list_by_category(EffectCategory::Generate).len(), 0);
    }
}
