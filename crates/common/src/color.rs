//! Color space, pixel format, and transfer function types.

use serde::{Deserialize, Serialize};

/// Pixel format on GPU or in memory.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelFormat {
    /// 4 channels, 8 bits each (sRGB or linear depending on context).
    Rgba8,
    /// 4 channels, 16-bit float.
    Rgba16F,
    /// 4 channels, 32-bit float (internal processing).
    Rgba32F,
    /// NV12: Y plane + interleaved UV at half resolution (HW decoder output).
    Nv12,
    /// P010: 10-bit NV12 variant (HDR content).
    P010,
    /// BGRA8 (some GPU APIs prefer this ordering).
    Bgra8,
}

impl PixelFormat {
    /// Bytes per pixel (for planar formats, returns bytes for the Y component per pixel).
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Rgba16F => 8,
            Self::Rgba32F => 16,
            Self::Nv12 => 1, // Y plane only; UV is separate
            Self::P010 => 2, // 10-bit Y in 16-bit container
        }
    }

    pub fn is_planar(self) -> bool {
        matches!(self, Self::Nv12 | Self::P010)
    }

    pub fn channel_count(self) -> u32 {
        match self {
            Self::Nv12 | Self::P010 => 3, // YUV
            _ => 4,                       // RGBA
        }
    }
}

/// Color space / color primaries.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    /// sRGB (web, most consumer content).
    Srgb,
    /// Linear sRGB (internal GPU processing).
    LinearSrgb,
    /// BT.709 (HD video standard).
    Bt709,
    /// BT.2020 (HDR / UHD content).
    Bt2020,
}

/// Transfer function (gamma curve).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferFunction {
    /// sRGB gamma (~2.2).
    Srgb,
    /// Linear (1.0).
    Linear,
    /// BT.709 transfer.
    Bt709,
    /// PQ (Perceptual Quantizer, HDR10).
    Pq,
    /// HLG (Hybrid Log-Gamma, broadcast HDR).
    Hlg,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_format_sizes() {
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgba32F.bytes_per_pixel(), 16);
        assert!(PixelFormat::Nv12.is_planar());
        assert!(!PixelFormat::Rgba8.is_planar());
    }
}
