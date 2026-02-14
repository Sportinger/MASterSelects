//! Video and audio packets — output of demuxer, input to decoder.

use crate::codec::{AudioCodec, VideoCodec};
use crate::color::PixelFormat;
use crate::types::{Resolution, TimeCode};

/// Video packet containing NAL units, ready for HW decoder.
#[derive(Clone, Debug)]
pub struct VideoPacket {
    /// NAL unit data in Annex-B format (0x00000001 + NALU).
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: TimeCode,
    /// Decode timestamp.
    pub dts: TimeCode,
    /// Whether this is a keyframe (IDR for H.264).
    pub is_keyframe: bool,
    /// Video codec.
    pub codec: VideoCodec,
}

/// Audio packet from demuxer.
#[derive(Clone, Debug)]
pub struct AudioPacket {
    /// Compressed audio data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: TimeCode,
    /// Audio codec.
    pub codec: AudioCodec,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

/// A decoded GPU frame — lives on GPU device memory.
#[derive(Debug)]
pub struct GpuFrame {
    /// Opaque device pointer (CUDA CUdeviceptr or Vulkan buffer handle).
    pub device_ptr: u64,
    /// For planar formats (NV12): pointer to UV plane.
    pub device_ptr_uv: Option<u64>,
    /// Frame dimensions.
    pub resolution: Resolution,
    /// Pixel format.
    pub format: PixelFormat,
    /// Pitch (row stride) in bytes.
    pub pitch: u32,
    /// Presentation timestamp.
    pub pts: TimeCode,
}

impl GpuFrame {
    /// Total byte size of this frame on GPU.
    pub fn byte_size(&self) -> usize {
        match self.format {
            PixelFormat::Nv12 => self.resolution.nv12_byte_size(),
            PixelFormat::P010 => self.resolution.nv12_byte_size() * 2,
            _ => self.pitch as usize * self.resolution.height as usize,
        }
    }
}

/// Media stream info extracted during probing.
#[derive(Clone, Debug)]
pub struct VideoStreamInfo {
    pub codec: VideoCodec,
    pub resolution: Resolution,
    pub fps: crate::types::Rational,
    pub duration: TimeCode,
    pub bitrate: u64,
    pub pixel_format: PixelFormat,
    /// Codec-specific data (SPS/PPS for H.264, etc.).
    pub extra_data: Vec<u8>,
}

/// Audio stream info extracted during probing.
#[derive(Clone, Debug)]
pub struct AudioStreamInfo {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: TimeCode,
    pub bitrate: u64,
}

/// Combined container info from probing.
#[derive(Clone, Debug)]
pub struct ContainerInfo {
    pub video_streams: Vec<VideoStreamInfo>,
    pub audio_streams: Vec<AudioStreamInfo>,
    pub duration: TimeCode,
}
