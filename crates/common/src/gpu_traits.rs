//! GPU backend abstraction traits.
//!
//! These traits define the interface between the engine and GPU backends (CUDA/Vulkan).
//! All GPU-dependent crates (compositor, effects, decoder, encoder) program against
//! these traits, not against concrete backend implementations.

use crate::codec::VideoCodec;
use crate::color::PixelFormat;
use crate::config::{DecoderConfig, EncoderConfig};
use crate::error::{DecodeError, EncodeError, GpuError};
use crate::kernel::{KernelArgs, KernelId};
use crate::packet::{GpuFrame, VideoPacket};
use crate::types::Resolution;

/// Core GPU backend abstraction — implemented by CUDA and Vulkan backends.
pub trait GpuBackend: Send + Sync {
    // -- Device info --

    /// Human-readable GPU device name.
    fn device_name(&self) -> &str;

    /// Total VRAM in bytes.
    fn vram_total(&self) -> u64;

    /// Currently used VRAM in bytes (approximate).
    fn vram_used(&self) -> u64;

    /// Available VRAM in bytes.
    fn vram_available(&self) -> u64 {
        self.vram_total().saturating_sub(self.vram_used())
    }

    // -- Memory allocation --

    /// Allocate a device buffer of `size` bytes.
    fn alloc_buffer(&self, size: usize) -> Result<GpuBuffer, GpuError>;

    /// Allocate a 2D texture on GPU.
    fn alloc_texture(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<GpuTexture, GpuError>;

    /// Allocate a staging buffer for CPU↔GPU transfers (pinned/host-visible memory).
    fn alloc_staging(&self, size: usize) -> Result<StagingBuffer, GpuError>;

    // -- Streams / command queues --

    /// Create a new GPU stream (CUDA stream / Vulkan command queue).
    fn create_stream(&self) -> Result<GpuStream, GpuError>;

    /// Wait for all operations on stream to complete.
    fn synchronize(&self, stream: &GpuStream) -> Result<(), GpuError>;

    // -- Kernel dispatch --

    /// Dispatch a compute kernel/shader.
    fn dispatch_kernel(
        &self,
        kernel: &KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        args: &KernelArgs,
        stream: &GpuStream,
    ) -> Result<(), GpuError>;

    // -- Memory transfers --

    /// Copy from GPU device buffer to host memory.
    fn copy_to_host(
        &self,
        src: &GpuBuffer,
        dst: &mut [u8],
        stream: &GpuStream,
    ) -> Result<(), GpuError>;

    /// Copy from host memory to GPU device buffer.
    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &GpuBuffer,
        stream: &GpuStream,
    ) -> Result<(), GpuError>;

    /// Copy between GPU device buffers.
    fn copy_buffer(
        &self,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        stream: &GpuStream,
    ) -> Result<(), GpuError>;

    // -- Hardware decode/encode --

    /// Create a hardware video decoder.
    fn create_decoder(&self, config: &DecoderConfig) -> Result<Box<dyn HwDecoder>, DecodeError>;

    /// Create a hardware video encoder.
    fn create_encoder(&self, config: &EncoderConfig) -> Result<Box<dyn HwEncoder>, EncodeError>;

    // -- Display bridge --

    /// Transfer a GPU texture to a staging buffer for display.
    /// The staging buffer data can then be uploaded to wgpu.
    fn copy_to_staging(
        &self,
        src: &GpuTexture,
        dst: &StagingBuffer,
        stream: &GpuStream,
    ) -> Result<(), GpuError>;
}

/// Hardware video decoder trait.
pub trait HwDecoder: Send {
    /// Decode a single video packet (NAL units).
    fn decode(&mut self, packet: &VideoPacket) -> Result<Option<GpuFrame>, DecodeError>;

    /// Flush remaining frames from decoder pipeline.
    fn flush(&mut self) -> Result<Vec<GpuFrame>, DecodeError>;

    /// Get the output pixel format.
    fn output_format(&self) -> PixelFormat;

    /// Get the output resolution.
    fn output_resolution(&self) -> Resolution;

    /// Get the codec this decoder handles.
    fn codec(&self) -> VideoCodec;
}

/// Hardware video encoder trait.
pub trait HwEncoder: Send {
    /// Encode a GPU frame.
    fn encode(&mut self, frame: &GpuFrame) -> Result<EncodedPacket, EncodeError>;

    /// Flush remaining packets from encoder pipeline.
    fn flush(&mut self) -> Result<Vec<EncodedPacket>, EncodeError>;
}

/// An encoded video packet output by the encoder.
#[derive(Clone, Debug)]
pub struct EncodedPacket {
    pub data: Vec<u8>,
    pub pts: crate::types::TimeCode,
    pub dts: crate::types::TimeCode,
    pub is_keyframe: bool,
}

/// Opaque GPU device buffer handle.
#[derive(Debug)]
pub struct GpuBuffer {
    /// Backend-specific handle (CUDA CUdeviceptr / Vulkan buffer).
    pub handle: u64,
    /// Size in bytes.
    pub size: usize,
    /// Backend identifier for dispatch.
    pub backend_id: u32,
}

/// Opaque GPU 2D texture handle.
#[derive(Debug)]
pub struct GpuTexture {
    /// Backend-specific handle.
    pub handle: u64,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub pitch: u32,
    pub backend_id: u32,
}

/// Staging buffer for CPU↔GPU transfers (pinned/host-visible memory).
#[derive(Debug)]
pub struct StagingBuffer {
    /// Host-accessible pointer.
    pub host_ptr: *mut u8,
    /// Device-accessible pointer (for pinned memory).
    pub device_ptr: Option<u64>,
    /// Size in bytes.
    pub size: usize,
    pub backend_id: u32,
}

// SAFETY: StagingBuffer's host_ptr is allocated by the GPU backend and
// is only accessed through the backend's copy methods or via direct read
// after synchronization. The backend ensures proper synchronization.
unsafe impl Send for StagingBuffer {}
unsafe impl Sync for StagingBuffer {}

/// Opaque GPU stream/command queue handle.
#[derive(Debug)]
pub struct GpuStream {
    /// Backend-specific handle.
    pub handle: u64,
    pub backend_id: u32,
}

/// Information about a GPU device.
#[derive(Clone, Debug)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub vram_total: u64,
    pub compute_capability: Option<(u32, u32)>, // CUDA only
    pub api_version: String,
}

/// GPU vendor for backend selection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}
