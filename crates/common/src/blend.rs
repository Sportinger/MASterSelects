//! Blend mode definitions for layer compositing.

use serde::{Deserialize, Serialize};

/// Blend modes for compositing layers.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
    Add,
    Subtract,
}

impl BlendMode {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Multiply => "Multiply",
            Self::Screen => "Screen",
            Self::Overlay => "Overlay",
            Self::Darken => "Darken",
            Self::Lighten => "Lighten",
            Self::ColorDodge => "Color Dodge",
            Self::ColorBurn => "Color Burn",
            Self::HardLight => "Hard Light",
            Self::SoftLight => "Soft Light",
            Self::Difference => "Difference",
            Self::Exclusion => "Exclusion",
            Self::Hue => "Hue",
            Self::Saturation => "Saturation",
            Self::Color => "Color",
            Self::Luminosity => "Luminosity",
            Self::Add => "Add",
            Self::Subtract => "Subtract",
        }
    }

    /// All blend modes in display order.
    pub fn all() -> &'static [BlendMode] {
        &[
            Self::Normal,
            Self::Multiply,
            Self::Screen,
            Self::Overlay,
            Self::Darken,
            Self::Lighten,
            Self::ColorDodge,
            Self::ColorBurn,
            Self::HardLight,
            Self::SoftLight,
            Self::Difference,
            Self::Exclusion,
            Self::Hue,
            Self::Saturation,
            Self::Color,
            Self::Luminosity,
            Self::Add,
            Self::Subtract,
        ]
    }
}
