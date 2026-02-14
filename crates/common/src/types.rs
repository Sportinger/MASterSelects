//! Core types with newtype pattern for type safety.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub};

/// Frame number (absolute position in timeline).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FrameNumber(pub u64);

impl FrameNumber {
    pub const ZERO: Self = Self(0);

    pub fn as_timecode(self, fps: Rational) -> TimeCode {
        TimeCode(self.0 as f64 / fps.as_f64())
    }
}

impl Add<u64> for FrameNumber {
    type Output = Self;
    fn add(self, rhs: u64) -> Self {
        Self(self.0 + rhs)
    }
}

impl Sub for FrameNumber {
    type Output = i64;
    fn sub(self, rhs: Self) -> i64 {
        self.0 as i64 - rhs.0 as i64
    }
}

impl fmt::Display for FrameNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F{}", self.0)
    }
}

/// Time code in seconds (f64 precision).
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TimeCode(pub f64);

impl TimeCode {
    pub const ZERO: Self = Self(0.0);

    pub fn from_secs(secs: f64) -> Self {
        Self(secs)
    }

    pub fn as_secs(self) -> f64 {
        self.0
    }

    pub fn as_frame(self, fps: Rational) -> FrameNumber {
        FrameNumber((self.0 * fps.as_f64()).round() as u64)
    }

    pub fn as_millis(self) -> f64 {
        self.0 * 1000.0
    }
}

impl Add for TimeCode {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for TimeCode {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl fmt::Display for TimeCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_secs = self.0;
        let hours = (total_secs / 3600.0) as u32;
        let mins = ((total_secs % 3600.0) / 60.0) as u32;
        let secs = (total_secs % 60.0) as u32;
        let frames = ((total_secs % 1.0) * 30.0) as u32; // assume 30fps for display
        write!(f, "{hours:02}:{mins:02}:{secs:02}:{frames:02}")
    }
}

/// Rational number for frame rates (e.g., 30000/1001 for 29.97fps).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    pub num: u32,
    pub den: u32,
}

impl Rational {
    pub const FPS_24: Self = Self { num: 24, den: 1 };
    pub const FPS_25: Self = Self { num: 25, den: 1 };
    pub const FPS_30: Self = Self { num: 30, den: 1 };
    pub const FPS_29_97: Self = Self {
        num: 30000,
        den: 1001,
    };
    pub const FPS_60: Self = Self { num: 60, den: 1 };
    pub const FPS_59_94: Self = Self {
        num: 60000,
        den: 1001,
    };

    pub fn new(num: u32, den: u32) -> Self {
        assert!(den > 0, "Rational denominator must be > 0");
        Self { num, den }
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// Video/image resolution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub const HD: Self = Self {
        width: 1920,
        height: 1080,
    };
    pub const UHD: Self = Self {
        width: 3840,
        height: 2160,
    };

    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn pixel_count(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn aspect_ratio(self) -> f64 {
        self.width as f64 / self.height as f64
    }

    /// Byte size for RGBA8 pixel data.
    pub fn rgba_byte_size(self) -> usize {
        self.width as usize * self.height as usize * 4
    }

    /// Byte size for NV12 pixel data (Y plane + interleaved UV at half res).
    pub fn nv12_byte_size(self) -> usize {
        let y_size = self.width as usize * self.height as usize;
        let uv_size = self.width as usize * (self.height as usize / 2);
        y_size + uv_size
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Byte offset into a file or buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteOffset(pub u64);

/// Source identifier for media files in the timeline.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceId(pub String);

impl SourceId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_to_timecode_roundtrip() {
        let frame = FrameNumber(150);
        let tc = frame.as_timecode(Rational::FPS_30);
        assert!((tc.as_secs() - 5.0).abs() < 1e-9);
        let back = tc.as_frame(Rational::FPS_30);
        assert_eq!(back, frame);
    }

    #[test]
    fn rational_display() {
        assert_eq!(Rational::FPS_30.to_string(), "30");
        assert_eq!(Rational::FPS_29_97.to_string(), "30000/1001");
    }

    #[test]
    fn resolution_byte_sizes() {
        let hd = Resolution::HD;
        assert_eq!(hd.rgba_byte_size(), 1920 * 1080 * 4);
        assert_eq!(hd.nv12_byte_size(), 1920 * 1080 + 1920 * 540);
    }

    #[test]
    fn timecode_display() {
        let tc = TimeCode::from_secs(3661.5);
        let s = tc.to_string();
        assert!(s.starts_with("01:01:01"));
    }
}
