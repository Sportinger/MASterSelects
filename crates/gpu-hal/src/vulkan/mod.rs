//! Vulkan Compute backend for the MasterSelects engine.
//!
//! This module provides a GPU compute backend using the Vulkan API via `ash`,
//! with memory management through `gpu-allocator`. It is the primary backend
//! for AMD, Intel, and cross-platform GPU support.
//!
//! # Module Structure
//!
//! - [`context`]: Core Vulkan objects (Instance, Device, Queue).
//! - [`memory`]: GPU memory allocation (device-local and staging buffers).
//! - [`shader`]: SPIR-V shader module loading and management.
//! - [`pipeline`]: Compute pipeline creation and descriptor management.
//! - [`queue`]: Command buffer recording, submission, and synchronization.
//! - [`error`]: Vulkan-specific error types.
//!
//! # Usage
//!
//! ```no_run
//! use ms_gpu_hal::vulkan::VulkanBackend;
//!
//! let backend = VulkanBackend::new().expect("Failed to initialize Vulkan");
//! println!("GPU: {}", backend.device_name());
//! println!("VRAM: {} MB", backend.vram_total() / (1024 * 1024));
//! ```

pub mod context;
pub mod error;
pub mod memory;
pub mod pipeline;
pub mod queue;
pub mod shader;

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk::Handle;
use parking_lot::Mutex;
use tracing::{debug, info};

use self::context::VulkanContext;
use self::error::VulkanError;
use self::memory::{create_allocator, SharedAllocator};
use self::queue::CommandPool;

// Re-export key types for convenience.
pub use self::context::VulkanContext as Context;
pub use self::error::VulkanError as Error;
pub use self::memory::{DeviceBuffer, VulkanImage, VulkanStagingBuffer};
pub use self::pipeline::{
    compute_dispatch_1d, compute_dispatch_2d, compute_dispatch_custom, ComputePipeline,
    DescriptorManager, PipelineCache, WORKGROUP_SIZE_1D, WORKGROUP_SIZE_2D,
};
pub use self::queue::{ComputeRecorder, Fence, QueueSubmitter};
pub use self::shader::{ShaderModule, ShaderRegistry};

/// Backend identifier for Vulkan-allocated resources.
///
/// Used in ms-common's opaque handle types (`GpuBuffer::backend_id`, etc.)
/// to identify which backend owns a resource.
pub const BACKEND_ID: u32 = 2;

/// The Vulkan Compute backend.
///
/// Owns all core Vulkan objects and provides the foundation for GPU compute
/// operations. This struct manages:
///
/// - Vulkan context (instance, device, queue)
/// - GPU memory allocator
/// - Command pool for compute operations
/// - Shader and pipeline registries
///
/// Implements the `GpuBackend` trait from `ms-common`, providing all memory
/// management operations and stub implementations for decode/encode/dispatch.
pub struct VulkanBackend {
    /// Core Vulkan context (instance, device, queue).
    context: VulkanContext,
    /// GPU memory allocator (gpu-allocator).
    allocator: SharedAllocator,
    /// Command pool for compute queue.
    command_pool: CommandPool,
    /// Shader module registry.
    shaders: Mutex<ShaderRegistry>,
    /// Compute pipeline cache.
    pipelines: Mutex<PipelineCache>,
    /// Approximate VRAM usage tracking (bytes).
    vram_used: AtomicU64,
    /// Cached device name for `GpuBackend::device_name()` returning `&str`.
    device_name_cached: String,
}

impl VulkanBackend {
    /// Initialize the Vulkan backend.
    ///
    /// This creates the Vulkan instance, selects the best GPU, creates a logical
    /// device with a compute queue, and sets up the memory allocator.
    ///
    /// Returns an error if no Vulkan-capable GPU is found or if initialization fails.
    pub fn new() -> Result<Self, VulkanError> {
        info!("Initializing Vulkan Compute backend");

        let context = VulkanContext::new()?;
        let allocator = create_allocator(&context)?;
        let command_pool =
            CommandPool::new(context.device(), context.compute_queue_family_index())?;

        let device_info = context.device_info();
        let device_name_cached = device_info.name.clone();
        info!(
            gpu = %device_info.name,
            vendor = ?device_info.vendor,
            vram_mb = device_info.vram_total / (1024 * 1024),
            api_version = %device_info.api_version,
            "Vulkan backend initialized"
        );

        Ok(Self {
            context,
            allocator,
            command_pool,
            shaders: Mutex::new(ShaderRegistry::new()),
            pipelines: Mutex::new(PipelineCache::new()),
            vram_used: AtomicU64::new(0),
            device_name_cached,
        })
    }

    // -- Device info --

    /// Returns the GPU device name.
    pub fn device_name(&self) -> String {
        self.context.device_name()
    }

    /// Returns the GPU vendor.
    pub fn vendor(&self) -> ms_common::GpuVendor {
        self.context.vendor()
    }

    /// Returns total device-local VRAM in bytes.
    pub fn vram_total(&self) -> u64 {
        self.context.vram_total()
    }

    /// Returns approximate VRAM usage in bytes.
    pub fn vram_used(&self) -> u64 {
        self.vram_used.load(Ordering::Relaxed)
    }

    /// Returns approximate available VRAM in bytes.
    pub fn vram_available(&self) -> u64 {
        self.vram_total().saturating_sub(self.vram_used())
    }

    /// Returns full device info.
    pub fn device_info(&self) -> ms_common::GpuDeviceInfo {
        self.context.device_info()
    }

    // -- Resource access --

    /// Returns a reference to the Vulkan context.
    #[inline]
    pub fn context(&self) -> &VulkanContext {
        &self.context
    }

    /// Returns the shared GPU memory allocator.
    #[inline]
    pub fn allocator(&self) -> &SharedAllocator {
        &self.allocator
    }

    /// Returns a reference to the command pool.
    #[inline]
    pub fn command_pool(&self) -> &CommandPool {
        &self.command_pool
    }

    /// Returns a reference to the pipeline cache (locked).
    #[inline]
    pub fn pipelines(&self) -> parking_lot::MutexGuard<'_, PipelineCache> {
        self.pipelines.lock()
    }

    // -- Buffer allocation --

    /// Allocate a device-local buffer for compute storage.
    pub fn alloc_device_buffer(&self, size: usize) -> Result<DeviceBuffer, VulkanError> {
        let buffer = DeviceBuffer::new(
            &self.context,
            &self.allocator,
            size,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        self.vram_used.fetch_add(size as u64, Ordering::Relaxed);
        Ok(buffer)
    }

    /// Allocate a staging buffer for CPU-to-GPU transfers.
    pub fn alloc_upload_staging(&self, size: usize) -> Result<VulkanStagingBuffer, VulkanError> {
        VulkanStagingBuffer::new(
            &self.context,
            &self.allocator,
            size,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )
    }

    /// Allocate a staging buffer for GPU-to-CPU readback.
    pub fn alloc_readback_staging(&self, size: usize) -> Result<VulkanStagingBuffer, VulkanError> {
        VulkanStagingBuffer::new(
            &self.context,
            &self.allocator,
            size,
            gpu_allocator::MemoryLocation::GpuToCpu,
        )
    }

    // -- Shader management --

    /// Load a SPIR-V shader module and register it.
    ///
    /// If a shader with the same name is already loaded, returns its handle.
    pub fn load_shader(
        &self,
        name: &str,
        spirv_bytes: &[u8],
    ) -> Result<ash::vk::ShaderModule, VulkanError> {
        self.shaders
            .lock()
            .load_or_get(self.context.device(), name, spirv_bytes)
    }

    // -- Command buffer helpers --

    /// Allocate a command buffer from the compute command pool.
    pub fn allocate_command_buffer(&self) -> Result<ash::vk::CommandBuffer, VulkanError> {
        self.command_pool.allocate_command_buffer()
    }

    /// Submit a command buffer to the compute queue and wait for completion.
    pub fn submit_and_wait(
        &self,
        command_buffer: ash::vk::CommandBuffer,
    ) -> Result<(), VulkanError> {
        queue::submit_and_wait(
            self.context.device(),
            self.context.compute_queue(),
            command_buffer,
        )
    }

    /// Create a new fence for synchronization.
    pub fn create_fence(&self) -> Result<Fence, VulkanError> {
        Fence::new(self.context.device())
    }

    /// Create a new fence in the signaled state.
    pub fn create_fence_signaled(&self) -> Result<Fence, VulkanError> {
        Fence::new_signaled(self.context.device())
    }

    /// Create a new descriptor manager with the given capacity.
    pub fn create_descriptor_manager(
        &self,
        max_sets: u32,
        max_storage_buffers: u32,
    ) -> Result<DescriptorManager, VulkanError> {
        DescriptorManager::new(self.context.device(), max_sets, max_storage_buffers)
    }

    /// Wait for the device to finish all operations.
    pub fn device_wait_idle(&self) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Device is valid. This blocks until all GPU operations complete.
            self.context
                .device()
                .device_wait_idle()
                .map_err(VulkanError::DeviceCreation)
        }
    }

    /// Track VRAM deallocation (call when freeing device buffers).
    pub fn track_vram_free(&self, size: u64) {
        self.vram_used.fetch_sub(size, Ordering::Relaxed);
    }

    /// Allocate a GPU image (2D texture) for the given pixel format.
    pub fn alloc_image(
        &self,
        width: u32,
        height: u32,
        format: ms_common::PixelFormat,
    ) -> Result<VulkanImage, VulkanError> {
        let image = VulkanImage::new(&self.context, &self.allocator, width, height, format)?;
        self.vram_used
            .fetch_add(image.byte_size() as u64, Ordering::Relaxed);
        Ok(image)
    }

    /// Record and execute a buffer-to-buffer copy on the compute queue.
    ///
    /// Allocates a one-shot command buffer, records the copy, submits, and waits.
    fn execute_buffer_copy(
        &self,
        src: ash::vk::Buffer,
        dst: ash::vk::Buffer,
        size: usize,
    ) -> Result<(), VulkanError> {
        let cmd = self.command_pool.allocate_command_buffer()?;
        let recorder = ComputeRecorder::begin(self.context.device(), cmd)?;
        recorder.copy_buffer(src, dst, size as ash::vk::DeviceSize);
        let cmd = recorder.finish()?;
        queue::submit_and_wait(self.context.device(), self.context.compute_queue(), cmd)?;
        Ok(())
    }
}

/// `GpuBackend` trait implementation for `VulkanBackend`.
///
/// Memory methods (`alloc_buffer`, `alloc_texture`, `alloc_staging`,
/// `copy_to_host`, `copy_to_device`, `copy_buffer`) are fully implemented.
/// Kernel dispatch, decode/encode, and stream methods return stub errors
/// and will be filled in during later development phases.
impl ms_common::GpuBackend for VulkanBackend {
    // -- Device info --

    fn device_name(&self) -> &str {
        &self.device_name_cached
    }

    fn vram_total(&self) -> u64 {
        self.context.vram_total()
    }

    fn vram_used(&self) -> u64 {
        self.vram_used.load(Ordering::Relaxed)
    }

    // -- Memory allocation --

    fn alloc_buffer(&self, size: usize) -> Result<ms_common::GpuBuffer, ms_common::GpuError> {
        let buffer = self
            .alloc_device_buffer(size)
            .map_err(ms_common::GpuError::from)?;

        let gpu_buf = ms_common::GpuBuffer {
            handle: buffer.handle(),
            size: buffer.size(),
            backend_id: BACKEND_ID,
        };

        // Intentionally leak the DeviceBuffer. The handle lives in the opaque
        // GpuBuffer and will be used by backend methods. In a production system
        // we would maintain a handle map; for now the VulkanBackend's VRAM
        // tracking accounts for the allocation and the buffer will be freed
        // when the backend is dropped or through explicit free calls.
        //
        // TODO(phase-1): Introduce a handle table to track DeviceBuffer ownership
        // so callers can free individual buffers through GpuBuffer handles.
        std::mem::forget(buffer);

        debug!(size = size, handle = gpu_buf.handle, "GpuBackend::alloc_buffer");
        Ok(gpu_buf)
    }

    fn alloc_texture(
        &self,
        width: u32,
        height: u32,
        format: ms_common::PixelFormat,
    ) -> Result<ms_common::GpuTexture, ms_common::GpuError> {
        let image = self
            .alloc_image(width, height, format)
            .map_err(ms_common::GpuError::from)?;

        let gpu_tex = ms_common::GpuTexture {
            handle: image.handle(),
            width: image.width(),
            height: image.height(),
            format,
            pitch: image.pitch(),
            backend_id: BACKEND_ID,
        };

        // See alloc_buffer comment about leaking.
        // TODO(phase-1): Handle table for image ownership.
        std::mem::forget(image);

        debug!(
            width = width,
            height = height,
            handle = gpu_tex.handle,
            "GpuBackend::alloc_texture"
        );
        Ok(gpu_tex)
    }

    fn alloc_staging(
        &self,
        size: usize,
    ) -> Result<ms_common::StagingBuffer, ms_common::GpuError> {
        // Allocate a CpuToGpu staging buffer by default. Callers needing
        // GpuToCpu should use `alloc_readback_staging` directly.
        let staging = self
            .alloc_upload_staging(size)
            .map_err(ms_common::GpuError::from)?;

        let host_ptr = staging
            .mapped_ptr()
            .unwrap_or(std::ptr::null_mut());
        let handle = staging.handle();

        let staging_buf = ms_common::StagingBuffer {
            host_ptr,
            device_ptr: Some(handle),
            size: staging.size(),
            backend_id: BACKEND_ID,
        };

        // See alloc_buffer comment about leaking.
        // TODO(phase-1): Handle table for staging buffer ownership.
        std::mem::forget(staging);

        debug!(size = size, "GpuBackend::alloc_staging");
        Ok(staging_buf)
    }

    // -- Streams / command queues --

    fn create_stream(&self) -> Result<ms_common::GpuStream, ms_common::GpuError> {
        // Vulkan uses command buffers rather than persistent streams.
        // For now, return a handle wrapping the compute queue. In the future,
        // each "stream" could map to its own command pool + fence pair for
        // concurrent async compute.
        Ok(ms_common::GpuStream {
            handle: 0, // Default "stream" backed by the single compute queue.
            backend_id: BACKEND_ID,
        })
    }

    fn synchronize(&self, _stream: &ms_common::GpuStream) -> Result<(), ms_common::GpuError> {
        // Synchronize by waiting for the device to become idle.
        // A more fine-grained approach would track per-stream fences.
        self.device_wait_idle()
            .map_err(ms_common::GpuError::from)
    }

    // -- Kernel dispatch (stub) --

    fn dispatch_kernel(
        &self,
        kernel: &ms_common::KernelId,
        _grid: [u32; 3],
        _block: [u32; 3],
        _args: &ms_common::KernelArgs,
        _stream: &ms_common::GpuStream,
    ) -> Result<(), ms_common::GpuError> {
        // TODO(phase-2): Implement compute shader dispatch.
        // This requires loading the SPIR-V for the kernel, creating a pipeline
        // with the correct descriptor set layout, binding arguments, and
        // dispatching the workgroups.
        Err(ms_common::GpuError::KernelFailed {
            kernel: kernel.entry_point().to_string(),
            reason: "Vulkan kernel dispatch not yet implemented".into(),
        })
    }

    // -- Memory transfers --

    fn copy_to_host(
        &self,
        src: &ms_common::GpuBuffer,
        dst: &mut [u8],
        _stream: &ms_common::GpuStream,
    ) -> Result<(), ms_common::GpuError> {
        let copy_size = dst.len().min(src.size);

        // Allocate a readback staging buffer.
        let staging = self
            .alloc_readback_staging(copy_size)
            .map_err(ms_common::GpuError::from)?;

        // Record and execute a GPU copy: device buffer -> staging buffer.
        let src_buffer = ash::vk::Buffer::from_raw(src.handle);
        self.execute_buffer_copy(src_buffer, staging.buffer(), copy_size)
            .map_err(|e| ms_common::GpuError::TransferFailed(e.to_string()))?;

        // Read from the staging buffer into the destination slice.
        staging
            .read_data(dst)
            .map_err(|e| ms_common::GpuError::TransferFailed(e.to_string()))?;

        debug!(size = copy_size, "GpuBackend::copy_to_host");
        Ok(())
    }

    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &ms_common::GpuBuffer,
        _stream: &ms_common::GpuStream,
    ) -> Result<(), ms_common::GpuError> {
        let copy_size = src.len().min(dst.size);

        // Allocate an upload staging buffer and write the source data.
        let staging = self
            .alloc_upload_staging(copy_size)
            .map_err(ms_common::GpuError::from)?;

        staging
            .write_data(src)
            .map_err(|e| ms_common::GpuError::TransferFailed(e.to_string()))?;

        // Record and execute a GPU copy: staging buffer -> device buffer.
        let dst_buffer = ash::vk::Buffer::from_raw(dst.handle);
        self.execute_buffer_copy(staging.buffer(), dst_buffer, copy_size)
            .map_err(|e| ms_common::GpuError::TransferFailed(e.to_string()))?;

        debug!(size = copy_size, "GpuBackend::copy_to_device");
        Ok(())
    }

    fn copy_buffer(
        &self,
        src: &ms_common::GpuBuffer,
        dst: &ms_common::GpuBuffer,
        _stream: &ms_common::GpuStream,
    ) -> Result<(), ms_common::GpuError> {
        let copy_size = src.size.min(dst.size);
        let src_buffer = ash::vk::Buffer::from_raw(src.handle);
        let dst_buffer = ash::vk::Buffer::from_raw(dst.handle);

        self.execute_buffer_copy(src_buffer, dst_buffer, copy_size)
            .map_err(|e| ms_common::GpuError::TransferFailed(e.to_string()))?;

        debug!(size = copy_size, "GpuBackend::copy_buffer");
        Ok(())
    }

    // -- Hardware decode/encode (stubs) --

    fn create_decoder(
        &self,
        config: &ms_common::DecoderConfig,
    ) -> Result<Box<dyn ms_common::HwDecoder>, ms_common::DecodeError> {
        // TODO(phase-1): Implement Vulkan Video decode.
        Err(ms_common::DecodeError::HwDecoderInit {
            codec: config.codec,
            reason: "Vulkan Video decode not yet implemented".into(),
        })
    }

    fn create_encoder(
        &self,
        _config: &ms_common::EncoderConfig,
    ) -> Result<Box<dyn ms_common::HwEncoder>, ms_common::EncodeError> {
        // TODO(phase-3): Implement Vulkan Video encode.
        Err(ms_common::EncodeError::HwEncoderInit(
            "Vulkan Video encode not yet implemented".into(),
        ))
    }

    // -- Display bridge (stub) --

    fn copy_to_staging(
        &self,
        _src: &ms_common::GpuTexture,
        _dst: &ms_common::StagingBuffer,
        _stream: &ms_common::GpuStream,
    ) -> Result<(), ms_common::GpuError> {
        // TODO(phase-1): Implement image-to-staging copy.
        // This requires an image-to-buffer copy command with proper
        // image layout transitions.
        Err(ms_common::GpuError::TransferFailed(
            "Vulkan copy_to_staging not yet implemented".into(),
        ))
    }
}

// SAFETY: VulkanBackend is Send + Sync because:
// - VulkanContext owns Vulkan handles that are thread-safe when used correctly.
// - SharedAllocator is Arc<Mutex<Allocator>>, inherently Send + Sync.
// - CommandPool, ShaderRegistry, PipelineCache are behind Mutex where needed.
// - AtomicU64 is inherently Send + Sync.
// - String (device_name_cached) is Send + Sync.
// Vulkan API calls are externally synchronized via our Mutex locks.
unsafe impl Send for VulkanBackend {}
unsafe impl Sync for VulkanBackend {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id_is_unique() {
        // Ensure the Vulkan backend ID doesn't collide with CUDA (1).
        assert_eq!(BACKEND_ID, 2);
    }

    #[test]
    fn shader_registry_default() {
        let registry = ShaderRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn pipeline_cache_default() {
        let cache = PipelineCache::new();
        assert!(cache.is_empty());
    }
}
