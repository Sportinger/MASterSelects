//! Central error types for the engine (thiserror-based).

use thiserror::Error;

use crate::codec::VideoCodec;

/// Top-level engine error.
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    #[error("Demux error: {0}")]
    Demux(#[from] DemuxError),

    #[error("Decode error: {0}")]
    Decode(#[from] DecodeError),

    #[error("Encode error: {0}")]
    Encode(#[from] EncodeError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

/// GPU backend errors.
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No GPU backend available (need CUDA or Vulkan)")]
    NoBackend,

    #[error("GPU device initialization failed: {0}")]
    DeviceInit(String),

    #[error("GPU memory allocation failed: {size} bytes")]
    AllocFailed { size: usize },

    #[error("Kernel dispatch failed: {kernel}: {reason}")]
    KernelFailed { kernel: String, reason: String },

    #[error("GPU-to-host transfer failed: {0}")]
    TransferFailed(String),

    #[error("Out of VRAM: requested {requested} bytes, available {available} bytes")]
    OutOfVram { requested: u64, available: u64 },
}

/// Demuxer/container parsing errors.
#[derive(Error, Debug)]
pub enum DemuxError {
    #[error("Unsupported container format")]
    UnsupportedContainer,

    #[error("Invalid box/element at offset {offset}: {reason}")]
    InvalidStructure { offset: u64, reason: String },

    #[error("No video track found")]
    NoVideoTrack,

    #[error("No audio track found")]
    NoAudioTrack,

    #[error("Unsupported video codec: {0:?}")]
    UnsupportedVideoCodec(VideoCodec),

    #[error("Truncated data: expected {expected} bytes, got {got}")]
    TruncatedData { expected: usize, got: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Hardware decoder errors.
#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("HW decoder init failed for {codec:?}: {reason}")]
    HwDecoderInit { codec: VideoCodec, reason: String },

    #[error("Decode failed at frame {frame}: {reason}")]
    DecodeFailed { frame: u64, reason: String },

    #[error("Unsupported codec for HW decode: {0:?}")]
    UnsupportedCodec(VideoCodec),

    #[error("Decoder session expired or invalid")]
    InvalidSession,

    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),
}

/// Hardware encoder errors.
#[derive(Error, Debug)]
pub enum EncodeError {
    #[error("HW encoder init failed: {0}")]
    HwEncoderInit(String),

    #[error("Encode failed at frame {frame}: {reason}")]
    EncodeFailed { frame: u64, reason: String },

    #[error("Unsupported codec for HW encode: {0:?}")]
    UnsupportedCodec(VideoCodec),

    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),
}

/// Convenience Result type for engine operations.
pub type EngineResult<T> = Result<T, EngineError>;
