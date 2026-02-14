//! CUDA backend — GPU compute via cudarc (NVIDIA GPUs).
//!
//! This module provides the CUDA implementation of the GPU abstraction layer.
//! It wraps cudarc 0.16 for device management, memory allocation, kernel dispatch,
//! and stream synchronization.
//!
//! # Architecture
//!
//! - [`CudaBackend`] is the main entry point — wraps a CUDA context with
//!   kernel management and stream support.
//! - [`context::CudaContextWrapper`] manages device init and info queries.
//! - [`memory::DeviceBuffer`] / [`memory::PinnedBuffer`] provide RAII GPU memory.
//! - [`kernel::KernelManager`] handles PTX module loading and function caching.
//! - [`stream::ManagedStream`] wraps CUDA streams for async execution.
//! - [`interop::DisplayBridge`] bridges CUDA output to wgpu for display.
//!
//! # GpuBackend trait
//!
//! `CudaBackend` implements the `GpuBackend` trait from `ms-common`, bridging
//! opaque handle types (`GpuBuffer`, `GpuTexture`, `GpuStream`, `StagingBuffer`)
//! with the concrete cudarc RAII wrappers via internal resource registries.

pub mod context;
pub mod error;
pub mod interop;
pub mod kernel;
pub mod memory;
pub mod stream;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use self::context::CudaContextWrapper;
use self::error::CudaError;
use self::kernel::KernelManager;
use self::stream::ManagedStream;

use ms_common::color::PixelFormat;
use ms_common::config::{DecoderConfig, EncoderConfig};
use ms_common::error::{DecodeError, EncodeError, GpuError};
use ms_common::gpu_traits::{
    GpuBackend, GpuBuffer, GpuStream, GpuTexture, HwDecoder, HwEncoder, StagingBuffer,
};
use ms_common::kernel::{KernelArgs, KernelId};

// Re-export key types at the cuda module level
pub use self::context::{enumerate_devices, CudaDeviceInfo};
pub use self::error::CudaError as Error;
pub use self::interop::DisplayBridge;
pub use self::kernel::{
    compute_launch_config_1d, compute_launch_config_2d, dispatch_alpha_blend,
    dispatch_nv12_to_rgba, AlphaBlendParams, Nv12ToRgbaParams,
};
pub use self::memory::{DeviceBuffer, PinnedBuffer};
pub use self::stream::ManagedStream as Stream;

/// Backend identifier constant for CUDA — stored in opaque handle structs
/// to distinguish CUDA resources from Vulkan resources at runtime.
const CUDA_BACKEND_ID: u32 = 1;

/// Global monotonic counter for generating unique resource handles.
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

/// Generate a unique handle ID for resource registration.
fn next_handle() -> u64 {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

/// The CUDA GPU backend — top-level struct for CUDA operations.
///
/// Wraps a CUDA device context, kernel manager, and provides methods
/// for memory allocation, kernel dispatch, and stream management.
///
/// # Thread Safety
///
/// `CudaBackend` is `Send + Sync`. The underlying cudarc context handles
/// thread binding automatically. Multiple threads can use the backend
/// concurrently via separate streams.
///
/// # Resource Management
///
/// The backend maintains internal registries that map opaque `u64` handles
/// (used by the `GpuBackend` trait) to concrete cudarc RAII types. Resources
/// are automatically freed when the registry entries are removed.
#[derive(Debug)]
pub struct CudaBackend {
    /// CUDA context wrapper with cached device info.
    context: CudaContextWrapper,
    /// Kernel (PTX module) manager with function caching.
    kernels: KernelManager,
    /// Default stream for synchronous-style operations.
    default_stream: ManagedStream,

    // -- Resource registries for GpuBackend trait --
    /// Maps opaque handle IDs to owned DeviceBuffer instances.
    buffers: RwLock<HashMap<u64, DeviceBuffer>>,
    /// Maps opaque handle IDs to owned ManagedStream instances.
    streams: RwLock<HashMap<u64, ManagedStream>>,
    /// Maps opaque handle IDs to owned PinnedBuffer instances (staging).
    staging_buffers: RwLock<HashMap<u64, PinnedBuffer>>,
}

impl CudaBackend {
    /// Initialize the CUDA backend on the given device ordinal.
    ///
    /// - `ordinal`: GPU device index (0 for first GPU, 1 for second, etc.)
    ///
    /// Returns an error if no CUDA devices are found or the ordinal is invalid.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let context = CudaContextWrapper::new(ordinal)?;
        let ctx = context.context().clone();
        let kernels = KernelManager::new(ctx.clone());
        let default_stream = ManagedStream::default_stream(&ctx);

        info!(
            device = context.device_name(),
            vram_mb = context.vram_total() / (1024 * 1024),
            driver = context.driver_version_string(),
            "CUDA backend initialized"
        );

        Ok(Self {
            context,
            kernels,
            default_stream,
            buffers: RwLock::new(HashMap::new()),
            streams: RwLock::new(HashMap::new()),
            staging_buffers: RwLock::new(HashMap::new()),
        })
    }

    /// Initialize on the first available CUDA device (ordinal 0).
    pub fn new_default() -> Result<Self, CudaError> {
        Self::new(0)
    }

    // -- Device info --

    /// Human-readable GPU device name (e.g. "NVIDIA GeForce RTX 4090").
    pub fn device_name(&self) -> &str {
        self.context.device_name()
    }

    /// Total VRAM in bytes.
    pub fn vram_total(&self) -> u64 {
        self.context.vram_total()
    }

    /// Currently used VRAM in bytes (approximate).
    pub fn vram_used(&self) -> Result<u64, CudaError> {
        let (free, total) = self.context.vram_info()?;
        Ok(total.saturating_sub(free))
    }

    /// Available (free) VRAM in bytes.
    pub fn vram_available(&self) -> Result<u64, CudaError> {
        let (free, _total) = self.context.vram_info()?;
        Ok(free)
    }

    /// Get the device info struct.
    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.context.device_info()
    }

    /// Get the common `GpuDeviceInfo` for this device.
    pub fn gpu_device_info(&self) -> ms_common::GpuDeviceInfo {
        ms_common::GpuDeviceInfo::from(self.context.device_info())
    }

    /// Get the compute capability as (major, minor).
    pub fn compute_capability(&self) -> (u32, u32) {
        self.context.compute_capability()
    }

    // -- Context access --

    /// Get a reference to the underlying cudarc context.
    pub fn cuda_context(&self) -> &Arc<cudarc::driver::safe::CudaContext> {
        self.context.context()
    }

    /// Get the context wrapper.
    pub fn context(&self) -> &CudaContextWrapper {
        &self.context
    }

    // -- Stream management --

    /// Get the default stream.
    pub fn default_stream(&self) -> &ManagedStream {
        &self.default_stream
    }

    /// Create a new CUDA stream for concurrent operations (returns concrete type).
    pub fn create_managed_stream(&self) -> Result<ManagedStream, CudaError> {
        ManagedStream::new(self.context.context())
    }

    /// Synchronize the default stream.
    pub fn synchronize_default(&self) -> Result<(), CudaError> {
        self.default_stream.synchronize()
    }

    // -- Memory allocation (concrete types) --

    /// Allocate a zero-initialized device buffer (returns concrete type).
    pub fn alloc_device_buffer(&self, size: usize) -> Result<DeviceBuffer, CudaError> {
        DeviceBuffer::alloc_zeros(self.default_stream.inner(), size)
    }

    /// Allocate a device buffer on a specific stream.
    pub fn alloc_buffer_on_stream(
        &self,
        stream: &ManagedStream,
        size: usize,
    ) -> Result<DeviceBuffer, CudaError> {
        DeviceBuffer::alloc_zeros(stream.inner(), size)
    }

    /// Allocate pinned (page-locked) host memory for fast transfers.
    ///
    /// # Safety
    /// The contents are uninitialized. Write before reading.
    pub unsafe fn alloc_pinned(&self, size: usize) -> Result<PinnedBuffer, CudaError> {
        // SAFETY: Caller guarantees they will initialize before reading.
        PinnedBuffer::alloc(self.context.context(), size)
    }

    // -- Kernel management --

    /// Get a reference to the kernel manager.
    pub fn kernels(&self) -> &KernelManager {
        &self.kernels
    }

    /// Load a PTX module from source string.
    pub fn load_ptx(&self, name: &str, ptx_source: &str) -> Result<(), CudaError> {
        self.kernels.load_ptx_source(name, ptx_source)
    }

    /// Load a PTX module from raw bytes.
    pub fn load_ptx_bytes(&self, name: &str, ptx_bytes: &[u8]) -> Result<(), CudaError> {
        self.kernels.load_ptx_bytes(name, ptx_bytes)
    }

    // -- Display bridge --

    /// Create a display bridge for transferring frames to wgpu.
    pub fn create_display_bridge(
        &self,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<DisplayBridge, CudaError> {
        DisplayBridge::new(self.context.context(), width, height, bytes_per_pixel)
    }

    // -- Internal helpers for GpuBackend trait --

    /// Look up a registered stream by handle, or fall back to default stream.
    fn resolve_stream_inner(&self, stream: &GpuStream) -> Arc<cudarc::driver::safe::CudaStream> {
        let streams = self.streams.read();
        if let Some(managed) = streams.get(&stream.handle) {
            managed.inner().clone()
        } else {
            self.default_stream.inner().clone()
        }
    }

    /// Register a DeviceBuffer in the registry and return an opaque GpuBuffer handle.
    fn register_buffer(&self, buf: DeviceBuffer) -> GpuBuffer {
        let handle = next_handle();
        let size = buf.size();
        self.buffers.write().insert(handle, buf);
        GpuBuffer {
            handle,
            size,
            backend_id: CUDA_BACKEND_ID,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBackend trait implementation
// ---------------------------------------------------------------------------

impl GpuBackend for CudaBackend {
    // -- Device info --

    fn device_name(&self) -> &str {
        self.context.device_name()
    }

    fn vram_total(&self) -> u64 {
        self.context.vram_total()
    }

    fn vram_used(&self) -> u64 {
        // vram_info() can fail if context binding fails; return 0 in that case
        match self.context.vram_info() {
            Ok((free, total)) => total.saturating_sub(free),
            Err(e) => {
                warn!("Failed to query VRAM usage: {e}");
                0
            }
        }
    }

    // -- Memory allocation --

    fn alloc_buffer(&self, size: usize) -> Result<GpuBuffer, GpuError> {
        let buf = DeviceBuffer::alloc_zeros(self.default_stream.inner(), size)
            .map_err(GpuError::from)?;
        Ok(self.register_buffer(buf))
    }

    fn alloc_texture(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
        ) -> Result<GpuTexture, GpuError> {
        // Textures are backed by linear device buffers on CUDA.
        // Pitch = width * bytes_per_pixel, aligned to 256 bytes for coalesced access.
        let bpp = format.bytes_per_pixel();
        let raw_pitch = width * bpp;
        let pitch = (raw_pitch + 255) & !255; // align to 256
        let total_size = pitch as usize * height as usize;

        let buf = DeviceBuffer::alloc_zeros(self.default_stream.inner(), total_size)
            .map_err(GpuError::from)?;

        let handle = next_handle();
        self.buffers.write().insert(handle, buf);

        debug!(
            width,
            height,
            ?format,
            pitch,
            total_size,
            "Allocated CUDA texture"
        );

        Ok(GpuTexture {
            handle,
            width,
            height,
            format,
            pitch,
            backend_id: CUDA_BACKEND_ID,
        })
    }

    fn alloc_staging(&self, size: usize) -> Result<StagingBuffer, GpuError> {
        // SAFETY: The staging buffer will be written to by CUDA memcpy operations
        // before being read by the CPU. We never read uninitialized data because
        // all reads go through copy_to_staging which writes first.
        let pinned = unsafe {
            PinnedBuffer::alloc(self.context.context(), size).map_err(GpuError::from)?
        };

        let handle = next_handle();
        let host_ptr = pinned
            .as_ptr()
            .map_err(|e| GpuError::TransferFailed(format!("Failed to get staging pointer: {e}")))?
            as *mut u8;

        self.staging_buffers.write().insert(handle, pinned);

        debug!(size, handle, "Allocated CUDA staging buffer (pinned)");

        // Store the handle in backend_id so copy_to_staging can look up
        // the PinnedBuffer later. We repurpose backend_id's lower bits
        // for the handle since StagingBuffer lacks a dedicated handle field.
        // The handle is stored as u32 (truncated from u64) — safe because
        // handles are monotonically generated and we won't exceed u32::MAX
        // staging buffers in practice.
        Ok(StagingBuffer {
            host_ptr,
            device_ptr: None,
            size,
            backend_id: handle as u32,
        })
    }

    // -- Streams / command queues --

    fn create_stream(&self) -> Result<GpuStream, GpuError> {
        let managed = ManagedStream::new(self.context.context()).map_err(GpuError::from)?;
        let handle = managed.handle();

        self.streams.write().insert(handle, managed);

        debug!("Created CUDA stream via GpuBackend trait");

        Ok(GpuStream {
            handle,
            backend_id: CUDA_BACKEND_ID,
        })
    }

    fn synchronize(&self, stream: &GpuStream) -> Result<(), GpuError> {
        let streams = self.streams.read();
        if let Some(managed) = streams.get(&stream.handle) {
            managed.synchronize().map_err(GpuError::from)
        } else {
            // Fall back to default stream sync if handle not found
            self.default_stream.synchronize().map_err(GpuError::from)
        }
    }

    // -- Kernel dispatch --

    fn dispatch_kernel(
        &self,
        kernel_id: &KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        args: &KernelArgs,
        stream: &GpuStream,
    ) -> Result<(), GpuError> {
        // Resolve the concrete CUDA stream and get the raw CUstream handle
        let cuda_stream = self.resolve_stream_inner(stream);
        let cu_stream = cuda_stream.cu_stream();

        // Delegate to the KernelManager which handles module loading,
        // function lookup, argument marshaling, and launch via raw CUDA API.
        self.kernels
            .launch(kernel_id, grid, block, args, cu_stream)
            .map_err(GpuError::from)?;

        debug!(
            kernel = kernel_id.entry_point(),
            grid = ?grid,
            block = ?block,
            args = args.len(),
            "Dispatched CUDA kernel via GpuBackend"
        );

        Ok(())
    }

    // -- Memory transfers --

    fn copy_to_host(
        &self,
        src: &GpuBuffer,
        dst: &mut [u8],
        stream: &GpuStream,
    ) -> Result<(), GpuError> {
        let buffers = self.buffers.read();
        let device_buf = buffers.get(&src.handle).ok_or_else(|| {
            GpuError::TransferFailed(format!(
                "Buffer handle {} not found in CUDA registry",
                src.handle
            ))
        })?;

        // Use the stream associated with the buffer for the transfer.
        // The GpuStream parameter indicates intent but DeviceBuffer owns its stream reference.
        let _ = stream; // acknowledged — DeviceBuffer uses its internal stream
        device_buf.copy_to_host(dst).map_err(GpuError::from)?;

        Ok(())
    }

    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &GpuBuffer,
        stream: &GpuStream,
    ) -> Result<(), GpuError> {
        let mut buffers = self.buffers.write();
        let device_buf = buffers.get_mut(&dst.handle).ok_or_else(|| {
            GpuError::TransferFailed(format!(
                "Buffer handle {} not found in CUDA registry",
                dst.handle
            ))
        })?;

        let _ = stream; // acknowledged — DeviceBuffer uses its internal stream
        device_buf.copy_from_host(src).map_err(GpuError::from)?;

        Ok(())
    }

    fn copy_buffer(
        &self,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        stream: &GpuStream,
    ) -> Result<(), GpuError> {
        let _ = stream; // acknowledged — DeviceBuffer uses its internal stream

        // We need mutable access to dst and immutable access to src from the
        // same HashMap. To satisfy the borrow checker, temporarily remove dst,
        // perform the copy, then re-insert it.
        let mut buffers = self.buffers.write();

        let mut dst_buf = buffers.remove(&dst.handle).ok_or_else(|| {
            GpuError::TransferFailed(format!(
                "Destination buffer handle {} not found in CUDA registry",
                dst.handle
            ))
        })?;

        // Now we can immutably borrow src from the map (dst is removed)
        let copy_result = match buffers.get(&src.handle) {
            Some(src_buf) => dst_buf.copy_from_device(src_buf).map_err(GpuError::from),
            None => Err(GpuError::TransferFailed(format!(
                "Source buffer handle {} not found in CUDA registry",
                src.handle
            ))),
        };

        // Always re-insert dst, even on error
        buffers.insert(dst.handle, dst_buf);
        copy_result
    }

    // -- Hardware decode/encode --

    fn create_decoder(
        &self,
        config: &DecoderConfig,
    ) -> Result<Box<dyn HwDecoder>, DecodeError> {
        // NVDEC FFI bindings are not yet implemented. Return an error indicating
        // the decoder crate is not ready.
        Err(DecodeError::HwDecoderInit {
            codec: config.codec,
            reason: "NVDEC decoder not yet implemented — decoder crate pending".to_string(),
        })
    }

    fn create_encoder(
        &self,
        config: &EncoderConfig,
    ) -> Result<Box<dyn HwEncoder>, EncodeError> {
        // NVENC FFI bindings are not yet implemented. Return an error indicating
        // the encoder crate is not ready.
        Err(EncodeError::HwEncoderInit(format!(
            "NVENC encoder not yet implemented for {:?} — encoder crate pending",
            config.codec
        )))
    }

    // -- Display bridge --

    fn copy_to_staging(
        &self,
        src: &GpuTexture,
        dst: &StagingBuffer,
        stream: &GpuStream,
    ) -> Result<(), GpuError> {
        // Look up the device buffer backing the texture
        let buffers = self.buffers.read();
        let device_buf = buffers.get(&src.handle).ok_or_else(|| {
            GpuError::TransferFailed(format!(
                "Texture handle {} not found in CUDA registry",
                src.handle
            ))
        })?;

        // Look up the staging buffer
        let mut staging_buffers = self.staging_buffers.write();
        let pinned = staging_buffers.get_mut(&dst.backend_id.into()).or_else(|| {
            // The StagingBuffer doesn't use handle in the same way — search by pointer match
            None
        });

        // Since StagingBuffer doesn't carry its registry handle, we copy
        // directly from device to the host pointer via raw CUDA memcpy.
        let copy_size = std::cmp::min(device_buf.size(), dst.size);

        let _ = stream; // acknowledged
        let _ = pinned; // we use the raw host_ptr instead

        // Bind context to current thread for raw CUDA calls
        self.context
            .context()
            .bind_to_thread()
            .map_err(|e| GpuError::TransferFailed(format!("Failed to bind context: {e}")))?;

        // SAFETY:
        // 1. device_buf.device_ptr() returns a valid CUdeviceptr for an allocated buffer.
        // 2. dst.host_ptr was obtained from PinnedBuffer::alloc and points to valid
        //    page-locked memory of at least dst.size bytes.
        // 3. copy_size does not exceed either buffer's size.
        // 4. We perform a synchronous copy (null stream) to ensure data is available
        //    when this function returns.
        unsafe {
            let result = cudarc::driver::sys::cuMemcpyDtoH_v2(
                dst.host_ptr as *mut std::ffi::c_void,
                device_buf.device_ptr(),
                copy_size,
            );
            result.result().map_err(|e| {
                GpuError::TransferFailed(format!("cuMemcpyDtoH failed: {e:?}"))
            })?;
        }

        debug!(
            texture_handle = src.handle,
            copy_size,
            "Copied texture to staging buffer"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config_helpers() {
        let (grid, block) = compute_launch_config_1d(1920 * 1080);
        assert!(grid.0 > 0);
        assert_eq!(block.0, 256);

        let (grid, block) = compute_launch_config_2d(1920, 1080);
        assert_eq!(grid.0, 120);
        assert_eq!(block, (16, 16, 1));
    }

    #[test]
    fn test_backend_id_constant() {
        assert_eq!(CUDA_BACKEND_ID, 1);
    }

    #[test]
    fn test_next_handle_monotonic() {
        let h1 = next_handle();
        let h2 = next_handle();
        let h3 = next_handle();
        assert!(h2 > h1);
        assert!(h3 > h2);
    }
}
