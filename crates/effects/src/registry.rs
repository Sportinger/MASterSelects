//! Effect registry — by-name lookup and category filtering.

use std::collections::HashMap;

use ms_common::EffectCategory;
use tracing::info;

use crate::blur::{DirectionalBlurEffect, GaussianBlurEffect, RadialBlurEffect, ZoomBlurEffect};
use crate::color::{
    BrightnessEffect, ColorWheelsEffect, CurvesEffect, ExposureEffect, HslEffect,
    WhiteBalanceEffect,
};
use crate::error::EffectError;
use crate::keying::{ChromaKeyEffect, LumaKeyEffect};
use crate::stylize::{GlowEffect, NoiseGrainEffect, SharpenEffect, VignetteEffect};
use crate::traits::Effect;

/// Registry holding all available GPU effects for lookup by name or category.
pub struct EffectRegistry {
    effects: HashMap<String, Box<dyn Effect>>,
}

impl EffectRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            effects: HashMap::new(),
        }
    }

    /// Create a registry with all built-in effects registered.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        // Color effects
        registry.register(Box::new(BrightnessEffect::new()));
        registry.register(Box::new(HslEffect::new()));
        registry.register(Box::new(CurvesEffect::new()));
        registry.register(Box::new(ExposureEffect::new()));
        registry.register(Box::new(WhiteBalanceEffect::new()));
        registry.register(Box::new(ColorWheelsEffect::new()));

        // Blur effects
        registry.register(Box::new(GaussianBlurEffect::new()));
        registry.register(Box::new(DirectionalBlurEffect::new()));
        registry.register(Box::new(RadialBlurEffect::new()));
        registry.register(Box::new(ZoomBlurEffect::new()));

        // Keying effects
        registry.register(Box::new(ChromaKeyEffect::new()));
        registry.register(Box::new(LumaKeyEffect::new()));

        // Stylize effects
        registry.register(Box::new(GlowEffect::new()));
        registry.register(Box::new(SharpenEffect::new()));
        registry.register(Box::new(NoiseGrainEffect::new()));
        registry.register(Box::new(VignetteEffect::new()));

        info!(
            count = registry.effects.len(),
            "Registered built-in effects"
        );

        registry
    }

    /// Register a new effect. Overwrites any previous effect with the same name.
    pub fn register(&mut self, effect: Box<dyn Effect>) {
        let name = effect.name().to_string();
        self.effects.insert(name, effect);
    }

    /// Try to register an effect, returning an error if one already exists with that name.
    pub fn try_register(&mut self, effect: Box<dyn Effect>) -> Result<(), EffectError> {
        let name = effect.name().to_string();
        if self.effects.contains_key(&name) {
            return Err(EffectError::AlreadyRegistered { name });
        }
        self.effects.insert(name, effect);
        Ok(())
    }

    /// Look up an effect by name.
    pub fn get(&self, name: &str) -> Option<&dyn Effect> {
        self.effects.get(name).map(|e| e.as_ref())
    }

    /// List all registered effects, sorted by name.
    pub fn list(&self) -> Vec<&dyn Effect> {
        let mut effects: Vec<_> = self.effects.values().map(|e| e.as_ref()).collect();
        effects.sort_by_key(|e| e.name());
        effects
    }

    /// List effects filtered by category, sorted by name.
    pub fn list_by_category(&self, category: EffectCategory) -> Vec<&dyn Effect> {
        let mut effects: Vec<_> = self
            .effects
            .values()
            .filter(|e| e.category() == category)
            .map(|e| e.as_ref())
            .collect();
        effects.sort_by_key(|e| e.name());
        effects
    }

    /// Number of registered effects.
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl Default for EffectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry() {
        let reg = EffectRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.get("brightness_contrast").is_none());
        assert!(reg.list().is_empty());
    }

    #[test]
    fn with_builtins_has_all_effects() {
        let reg = EffectRegistry::with_builtins();
        assert_eq!(reg.len(), 16);

        // Spot-check a few
        assert!(reg.get("brightness_contrast").is_some());
        assert!(reg.get("gaussian_blur").is_some());
        assert!(reg.get("chroma_key").is_some());
        assert!(reg.get("vignette").is_some());
        assert!(reg.get("hsl_adjust").is_some());
        assert!(reg.get("color_wheels").is_some());
    }

    #[test]
    fn get_returns_correct_effect() {
        let reg = EffectRegistry::with_builtins();
        let fx = reg.get("exposure").unwrap();
        assert_eq!(fx.name(), "exposure");
        assert_eq!(fx.category(), EffectCategory::Color);
    }

    #[test]
    fn list_sorted_by_name() {
        let reg = EffectRegistry::with_builtins();
        let list = reg.list();
        for window in list.windows(2) {
            assert!(window[0].name() <= window[1].name());
        }
    }

    #[test]
    fn list_by_category_color() {
        let reg = EffectRegistry::with_builtins();
        let color = reg.list_by_category(EffectCategory::Color);
        assert_eq!(color.len(), 6);
        for fx in &color {
            assert_eq!(fx.category(), EffectCategory::Color);
        }
    }

    #[test]
    fn list_by_category_blur() {
        let reg = EffectRegistry::with_builtins();
        let blur = reg.list_by_category(EffectCategory::Blur);
        assert_eq!(blur.len(), 4);
        for fx in &blur {
            assert_eq!(fx.category(), EffectCategory::Blur);
        }
    }

    #[test]
    fn list_by_category_keying() {
        let reg = EffectRegistry::with_builtins();
        let keying = reg.list_by_category(EffectCategory::Keying);
        assert_eq!(keying.len(), 2);
    }

    #[test]
    fn list_by_category_stylize() {
        let reg = EffectRegistry::with_builtins();
        let stylize = reg.list_by_category(EffectCategory::Stylize);
        assert_eq!(stylize.len(), 4);
    }

    #[test]
    fn list_by_category_empty() {
        let reg = EffectRegistry::with_builtins();
        let distort = reg.list_by_category(EffectCategory::Distort);
        assert!(distort.is_empty());
    }

    #[test]
    fn try_register_duplicate_fails() {
        let mut reg = EffectRegistry::new();
        reg.register(Box::new(BrightnessEffect::new()));
        let err = reg
            .try_register(Box::new(BrightnessEffect::new()))
            .unwrap_err();
        assert!(matches!(err, EffectError::AlreadyRegistered { .. }));
    }

    #[test]
    fn register_overwrites() {
        let mut reg = EffectRegistry::new();
        reg.register(Box::new(BrightnessEffect::new()));
        reg.register(Box::new(BrightnessEffect::new()));
        assert_eq!(reg.len(), 1); // Still only one
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let reg = EffectRegistry::with_builtins();
        assert!(reg.get("nonexistent_effect").is_none());
    }
}
