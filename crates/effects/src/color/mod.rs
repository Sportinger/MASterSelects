//! Color adjustment effects.
//!
//! - [`BrightnessEffect`] — Brightness and contrast
//! - [`HslEffect`] — Hue, saturation, lightness
//! - [`CurvesEffect`] — RGB channel curves (simplified)
//! - [`ExposureEffect`] — Exposure and gamma
//! - [`WhiteBalanceEffect`] — Color temperature and tint
//! - [`ColorWheelsEffect`] — Lift / Gamma / Gain

pub mod brightness;
pub mod color_wheels;
pub mod curves;
pub mod exposure;
pub mod hsl;
pub mod white_balance;

pub use brightness::BrightnessEffect;
pub use color_wheels::ColorWheelsEffect;
pub use curves::CurvesEffect;
pub use exposure::ExposureEffect;
pub use hsl::HslEffect;
pub use white_balance::WhiteBalanceEffect;
