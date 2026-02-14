//! `ms-common` — Shared types, traits, and errors for the MasterSelects native engine.
//!
//! This crate is the foundation that all other engine crates depend on.
//! It defines the core abstractions:
//!
//! - **Types**: `FrameNumber`, `TimeCode`, `Resolution`, `Rational` (newtypes for safety)
//! - **GPU Traits**: `GpuBackend`, `HwDecoder`, `HwEncoder` (backend abstraction)
//! - **Packets**: `VideoPacket`, `AudioPacket`, `GpuFrame` (data flow types)
//! - **Layer**: `LayerDesc`, `Transform2D`, `MaskDesc` (compositor interface)
//! - **Effects**: `EffectId`, `EffectInstance`, `ParamValue` (effect system)
//! - **Errors**: `EngineError`, `GpuError`, `DemuxError`, etc. (thiserror-based)
//! - **Config**: `EngineConfig`, `DecoderConfig`, `EncoderConfig`

pub mod blend;
pub mod codec;
pub mod color;
pub mod config;
pub mod effect;
pub mod error;
pub mod gpu_traits;
pub mod kernel;
pub mod layer;
pub mod packet;
pub mod types;

// Re-export commonly used items at crate root
pub use blend::BlendMode;
pub use codec::{AudioCodec, ContainerFormat, VideoCodec};
pub use color::{ColorSpace, PixelFormat, TransferFunction};
pub use config::{DecoderConfig, EncoderConfig, EngineConfig, GpuPreference, RenderConfig};
pub use effect::{EffectCategory, EffectId, EffectInstance, ParamDef, ParamType, ParamValue};
pub use error::{DecodeError, DemuxError, EncodeError, EngineError, EngineResult, GpuError};
pub use gpu_traits::{
    EncodedPacket, GpuBackend, GpuBuffer, GpuDeviceInfo, GpuStream, GpuTexture, GpuVendor,
    HwDecoder, HwEncoder, StagingBuffer,
};
pub use kernel::{KernelArg, KernelArgs, KernelId};
pub use layer::{LayerDesc, MaskDesc, MaskShape, Transform2D};
pub use packet::{
    AudioPacket, AudioStreamInfo, ContainerInfo, GpuFrame, VideoPacket, VideoStreamInfo,
};
pub use types::{ByteOffset, FrameNumber, Rational, Resolution, SourceId, TimeCode};
