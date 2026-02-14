//! Video/audio codec and container format enums.

use serde::{Deserialize, Serialize};

/// Video codec identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoCodec {
    H264,
    H265,
    Vp9,
    Av1,
}

impl VideoCodec {
    /// NVDEC codec GUID name (for display/logging).
    pub fn display_name(self) -> &'static str {
        match self {
            Self::H264 => "H.264/AVC",
            Self::H265 => "H.265/HEVC",
            Self::Vp9 => "VP9",
            Self::Av1 => "AV1",
        }
    }
}

/// Audio codec identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioCodec {
    Aac,
    Mp3,
    Flac,
    Wav,
    Opus,
    Vorbis,
}

/// Container format.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContainerFormat {
    /// ISO BMFF (MP4, MOV, M4V).
    Mp4,
    /// Matroska (MKV).
    Mkv,
    /// WebM (Matroska subset).
    WebM,
}

impl ContainerFormat {
    pub fn file_extensions(self) -> &'static [&'static str] {
        match self {
            Self::Mp4 => &["mp4", "m4v", "mov"],
            Self::Mkv => &["mkv"],
            Self::WebM => &["webm"],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_display() {
        assert_eq!(VideoCodec::H264.display_name(), "H.264/AVC");
        assert_eq!(VideoCodec::H265.display_name(), "H.265/HEVC");
    }

    #[test]
    fn container_extensions() {
        assert!(ContainerFormat::Mp4.file_extensions().contains(&"mp4"));
        assert!(ContainerFormat::Mp4.file_extensions().contains(&"mov"));
    }
}
