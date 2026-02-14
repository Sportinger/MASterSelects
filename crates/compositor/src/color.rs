//! Color space conversion helpers and blend mode mapping.

use ms_common::BlendMode;

/// Map a [`BlendMode`] to the integer constant expected by GPU blend kernels.
///
/// The integer values correspond to the blend mode index used in the
/// `alpha_blend` CUDA/Vulkan kernel's switch statement.
pub fn blend_mode_to_int(mode: &BlendMode) -> u32 {
    match mode {
        BlendMode::Normal => 0,
        BlendMode::Multiply => 1,
        BlendMode::Screen => 2,
        BlendMode::Overlay => 3,
        BlendMode::Darken => 4,
        BlendMode::Lighten => 5,
        BlendMode::ColorDodge => 6,
        BlendMode::ColorBurn => 7,
        BlendMode::HardLight => 8,
        BlendMode::SoftLight => 9,
        BlendMode::Difference => 10,
        BlendMode::Exclusion => 11,
        BlendMode::Hue => 12,
        BlendMode::Saturation => 13,
        BlendMode::Color => 14,
        BlendMode::Luminosity => 15,
        BlendMode::Add => 16,
        BlendMode::Subtract => 17,
    }
}

/// Convert an sRGB component (0.0..1.0) to linear light.
///
/// Uses the exact sRGB transfer function (IEC 61966-2-1).
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a linear-light component (0.0..1.0) to sRGB.
///
/// Uses the exact sRGB inverse transfer function (IEC 61966-2-1).
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_mode_normal_is_zero() {
        assert_eq!(blend_mode_to_int(&BlendMode::Normal), 0);
    }

    #[test]
    fn blend_mode_all_unique() {
        let all = BlendMode::all();
        let ints: Vec<u32> = all.iter().map(blend_mode_to_int).collect();
        // All values must be unique
        for (i, a) in ints.iter().enumerate() {
            for (j, b) in ints.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        a, b,
                        "Duplicate blend mode int for {:?} and {:?}",
                        all[i], all[j]
                    );
                }
            }
        }
    }

    #[test]
    fn srgb_linear_roundtrip() {
        // 0 and 1 should be identity
        assert!((srgb_to_linear(0.0) - 0.0).abs() < 1e-6);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!((linear_to_srgb(0.0) - 0.0).abs() < 1e-6);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);

        // Roundtrip for mid-range values
        for i in 0..=10 {
            let v = i as f32 / 10.0;
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!(
                (back - v).abs() < 1e-5,
                "Roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn srgb_to_linear_monotonic() {
        let mut prev = srgb_to_linear(0.0);
        for i in 1..=100 {
            let v = i as f32 / 100.0;
            let lin = srgb_to_linear(v);
            assert!(lin >= prev, "srgb_to_linear not monotonic at {v}");
            prev = lin;
        }
    }

    #[test]
    fn linear_to_srgb_monotonic() {
        let mut prev = linear_to_srgb(0.0);
        for i in 1..=100 {
            let v = i as f32 / 100.0;
            let srgb = linear_to_srgb(v);
            assert!(srgb >= prev, "linear_to_srgb not monotonic at {v}");
            prev = srgb;
        }
    }
}
