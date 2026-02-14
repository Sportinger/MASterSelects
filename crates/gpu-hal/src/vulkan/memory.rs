//! Vulkan memory management using `gpu-allocator`.
//!
//! Provides RAII wrappers around Vulkan buffers backed by `gpu-allocator` allocations.
//! Two main buffer types:
//!
//! - **DeviceBuffer**: GPU-local memory for compute operations (fast, not CPU-accessible).
//! - **StagingBuffer**: Host-visible memory for CPU↔GPU transfers.

use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;
use tracing::{debug, info, warn};

use super::context::VulkanContext;
use super::error::VulkanError;

/// Thread-safe handle to the gpu-allocator instance.
///
/// Wrapped in `Arc<Mutex<>>` because gpu-allocator's `Allocator` requires `&mut self`
/// for allocate/free, but we need shared access from multiple subsystems.
pub type SharedAllocator = Arc<Mutex<Allocator>>;

/// Create a gpu-allocator instance from a VulkanContext.
pub fn create_allocator(ctx: &VulkanContext) -> Result<SharedAllocator, VulkanError> {
    let allocator_desc = gpu_allocator::vulkan::AllocatorCreateDesc {
        instance: ctx.instance().clone(),
        device: ctx.device().clone(),
        physical_device: ctx.physical_device(),
        debug_settings: gpu_allocator::AllocatorDebugSettings {
            log_memory_information: true,
            log_leaks_on_shutdown: true,
            ..Default::default()
        },
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    };

    let allocator = Allocator::new(&allocator_desc)
        .map_err(|e| VulkanError::MemoryAllocation(e.to_string()))?;

    info!("GPU memory allocator created");
    Ok(Arc::new(Mutex::new(allocator)))
}

/// A GPU device-local buffer for compute operations.
///
/// This buffer resides in fast GPU memory and is not directly accessible from the CPU.
/// Data must be transferred via staging buffers.
///
/// Implements Drop to automatically free the Vulkan buffer and its allocation.
pub struct DeviceBuffer {
    /// The Vulkan buffer handle.
    buffer: vk::Buffer,
    /// The gpu-allocator allocation backing this buffer.
    /// Wrapped in Option so we can take it during Drop.
    allocation: Option<Allocation>,
    /// Buffer size in bytes.
    size: usize,
    /// Reference to the logical device (for cleanup).
    device: ash::Device,
    /// Reference to the allocator (for freeing).
    allocator: SharedAllocator,
}

impl DeviceBuffer {
    /// Allocate a new device-local buffer of `size` bytes.
    pub fn new(
        ctx: &VulkanContext,
        allocator: &SharedAllocator,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, VulkanError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            // SAFETY: Device is valid, buffer_info is properly initialized.
            ctx.device()
                .create_buffer(&buffer_info, None)
                .map_err(VulkanError::BufferCreation)?
        };

        let requirements = unsafe {
            // SAFETY: Device and buffer are valid. Buffer was just created.
            ctx.device().get_buffer_memory_requirements(buffer)
        };

        let alloc_desc = AllocationCreateDesc {
            name: "device_buffer",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true, // Buffers are always linear
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator
            .lock()
            .allocate(&alloc_desc)
            .map_err(|e| VulkanError::MemoryAllocation(e.to_string()))?;

        unsafe {
            // SAFETY: Device, buffer, and allocation are all valid.
            // The allocation's memory() returns the VkDeviceMemory backing it.
            // The allocation's offset() returns the offset within that memory.
            ctx.device()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(VulkanError::BufferCreation)?;
        }

        debug!(size = size, "Allocated device-local buffer");

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
            device: ctx.device().clone(),
            allocator: Arc::clone(allocator),
        })
    }

    /// Returns the Vulkan buffer handle.
    #[inline]
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Returns the buffer size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns a device address-style handle (buffer handle as u64).
    ///
    /// This is used for the opaque `GpuBuffer::handle` in ms-common.
    #[inline]
    pub fn handle(&self) -> u64 {
        self.buffer.as_raw()
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            if let Err(e) = self.allocator.lock().free(allocation) {
                warn!(error = %e, "Failed to free device buffer allocation");
            }
        }
        unsafe {
            // SAFETY: We own this buffer and are dropping it. The allocation
            // has already been freed above.
            self.device.destroy_buffer(self.buffer, None);
        }
        debug!(size = self.size, "Freed device-local buffer");
    }
}

/// A host-visible staging buffer for CPU↔GPU data transfers.
///
/// This buffer is allocated in host-visible memory and can be mapped for direct
/// CPU access. It is used for uploading data to the GPU and reading back results.
///
/// Implements Drop to automatically free the Vulkan buffer and its allocation.
pub struct VulkanStagingBuffer {
    /// The Vulkan buffer handle.
    buffer: vk::Buffer,
    /// The gpu-allocator allocation backing this buffer.
    /// Wrapped in Option so we can take it during Drop.
    allocation: Option<Allocation>,
    /// Buffer size in bytes.
    size: usize,
    /// Mapped host pointer (if available).
    mapped_ptr: Option<*mut u8>,
    /// Reference to the logical device (for cleanup).
    device: ash::Device,
    /// Reference to the allocator (for freeing).
    allocator: SharedAllocator,
}

impl VulkanStagingBuffer {
    /// Allocate a new host-visible staging buffer.
    ///
    /// The `direction` parameter determines the memory location:
    /// - `CpuToGpu`: For uploading data from CPU to GPU.
    /// - `GpuToCpu`: For reading data back from GPU to CPU.
    pub fn new(
        ctx: &VulkanContext,
        allocator: &SharedAllocator,
        size: usize,
        direction: MemoryLocation,
    ) -> Result<Self, VulkanError> {
        let usage = match direction {
            MemoryLocation::CpuToGpu => vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuToCpu => vk::BufferUsageFlags::TRANSFER_DST,
            _ => vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        };

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            // SAFETY: Device is valid, buffer_info is properly initialized.
            ctx.device()
                .create_buffer(&buffer_info, None)
                .map_err(VulkanError::BufferCreation)?
        };

        let requirements = unsafe {
            // SAFETY: Device and buffer are valid. Buffer was just created.
            ctx.device().get_buffer_memory_requirements(buffer)
        };

        let alloc_desc = AllocationCreateDesc {
            name: "staging_buffer",
            requirements,
            location: direction,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator
            .lock()
            .allocate(&alloc_desc)
            .map_err(|e| VulkanError::MemoryAllocation(e.to_string()))?;

        unsafe {
            // SAFETY: Device, buffer, and allocation are all valid.
            ctx.device()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(VulkanError::BufferCreation)?;
        }

        // Get the mapped pointer from the allocation (gpu-allocator maps host-visible
        // memory automatically).
        let mapped_ptr = allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8);

        debug!(
            size = size,
            has_mapped_ptr = mapped_ptr.is_some(),
            "Allocated staging buffer"
        );

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
            mapped_ptr,
            device: ctx.device().clone(),
            allocator: Arc::clone(allocator),
        })
    }

    /// Returns the Vulkan buffer handle.
    #[inline]
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Returns the buffer size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the mapped host pointer, if the buffer is host-visible.
    #[inline]
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr
    }

    /// Returns a device address-style handle.
    #[inline]
    pub fn handle(&self) -> u64 {
        self.buffer.as_raw()
    }

    /// Write data from a slice into the staging buffer.
    ///
    /// The buffer must be host-visible (have a mapped pointer).
    pub fn write_data(&self, data: &[u8]) -> Result<(), VulkanError> {
        let ptr = self
            .mapped_ptr
            .ok_or_else(|| VulkanError::MemoryAllocation("Buffer not host-visible".into()))?;

        let copy_len = data.len().min(self.size);
        unsafe {
            // SAFETY: The mapped pointer is valid for self.size bytes (guaranteed
            // by gpu-allocator). We clamp copy_len to not exceed the buffer size.
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, copy_len);
        }

        Ok(())
    }

    /// Read data from the staging buffer into a slice.
    ///
    /// The buffer must be host-visible (have a mapped pointer).
    pub fn read_data(&self, dst: &mut [u8]) -> Result<(), VulkanError> {
        let ptr = self
            .mapped_ptr
            .ok_or_else(|| VulkanError::MemoryAllocation("Buffer not host-visible".into()))?;

        let copy_len = dst.len().min(self.size);
        unsafe {
            // SAFETY: The mapped pointer is valid for self.size bytes (guaranteed
            // by gpu-allocator). We clamp copy_len to not exceed the buffer size.
            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), copy_len);
        }

        Ok(())
    }
}

impl Drop for VulkanStagingBuffer {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            if let Err(e) = self.allocator.lock().free(allocation) {
                warn!(error = %e, "Failed to free staging buffer allocation");
            }
        }
        unsafe {
            // SAFETY: We own this buffer and are dropping it. The allocation
            // has already been freed above.
            self.device.destroy_buffer(self.buffer, None);
        }
        debug!(size = self.size, "Freed staging buffer");
    }
}

// SAFETY: VulkanStagingBuffer's mapped_ptr points to persistently mapped memory
// managed by gpu-allocator. Access is synchronized by the caller (through fences
// and command buffer submission ordering). The pointer itself is stable for the
// lifetime of the allocation.
unsafe impl Send for VulkanStagingBuffer {}
unsafe impl Sync for VulkanStagingBuffer {}

/// A GPU device-local image for 2D textures (RGBA, NV12, etc.).
///
/// Backed by `VkImage` + `gpu-allocator` allocation. Used for compositor
/// and effect inputs/outputs. Supports RGBA8, RGBA16F, RGBA32F, and NV12
/// pixel formats.
///
/// Implements Drop to automatically free the Vulkan image, image view,
/// and its allocation.
pub struct VulkanImage {
    /// The Vulkan image handle.
    image: vk::Image,
    /// Image view for shader access.
    view: vk::ImageView,
    /// The gpu-allocator allocation backing this image.
    /// Wrapped in Option so we can take it during Drop.
    allocation: Option<Allocation>,
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
    /// Vulkan format of the image.
    format: vk::Format,
    /// Row pitch in bytes (for tightly packed images this is width * bpp).
    pitch: u32,
    /// Reference to the logical device (for cleanup).
    device: ash::Device,
    /// Reference to the allocator (for freeing).
    allocator: SharedAllocator,
}

impl VulkanImage {
    /// Allocate a 2D GPU image for the given pixel format.
    ///
    /// The image is created as a `STORAGE` + `TRANSFER_SRC` + `TRANSFER_DST`
    /// image, suitable for compute shader read/write and copy operations.
    pub fn new(
        ctx: &VulkanContext,
        allocator: &SharedAllocator,
        width: u32,
        height: u32,
        pixel_format: ms_common::PixelFormat,
    ) -> Result<Self, VulkanError> {
        let vk_format = pixel_format_to_vk(pixel_format);
        let bpp = format_bytes_per_pixel(vk_format);
        let pitch = width * bpp;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            // SAFETY: Device is valid, image_info is properly initialized.
            ctx.device()
                .create_image(&image_info, None)
                .map_err(VulkanError::ImageCreation)?
        };

        let requirements = unsafe {
            // SAFETY: Device and image are valid. Image was just created.
            ctx.device().get_image_memory_requirements(image)
        };

        let alloc_desc = AllocationCreateDesc {
            name: "gpu_image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false, // Images use optimal tiling
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator
            .lock()
            .allocate(&alloc_desc)
            .map_err(|e| VulkanError::MemoryAllocation(e.to_string()))?;

        unsafe {
            // SAFETY: Device, image, and allocation are all valid.
            ctx.device()
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(VulkanError::ImageCreation)?;
        }

        // Create an image view for shader access.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe {
            // SAFETY: Device and image are valid, view_info is properly initialized.
            ctx.device()
                .create_image_view(&view_info, None)
                .map_err(VulkanError::ImageCreation)?
        };

        debug!(
            width = width,
            height = height,
            format = ?vk_format,
            "Allocated GPU image"
        );

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            width,
            height,
            format: vk_format,
            pitch,
            device: ctx.device().clone(),
            allocator: Arc::clone(allocator),
        })
    }

    /// Returns the Vulkan image handle.
    #[inline]
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Returns the image view handle.
    #[inline]
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Returns the image width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the Vulkan format of this image.
    #[inline]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Returns the row pitch in bytes.
    #[inline]
    pub fn pitch(&self) -> u32 {
        self.pitch
    }

    /// Returns the image handle as a u64 for opaque handle storage.
    #[inline]
    pub fn handle(&self) -> u64 {
        self.image.as_raw()
    }

    /// Returns the total byte size of the image data (width * height * bpp).
    pub fn byte_size(&self) -> usize {
        self.pitch as usize * self.height as usize
    }
}

impl Drop for VulkanImage {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: We own the image view and it was created from our image.
            // Destroy view before image.
            self.device.destroy_image_view(self.view, None);
        }
        if let Some(allocation) = self.allocation.take() {
            if let Err(e) = self.allocator.lock().free(allocation) {
                warn!(error = %e, "Failed to free image allocation");
            }
        }
        unsafe {
            // SAFETY: We own this image and are dropping it. The allocation
            // has already been freed above.
            self.device.destroy_image(self.image, None);
        }
        debug!(width = self.width, height = self.height, "Freed GPU image");
    }
}

/// Convert an `ms_common::PixelFormat` to the corresponding Vulkan format.
pub fn pixel_format_to_vk(pf: ms_common::PixelFormat) -> vk::Format {
    match pf {
        ms_common::PixelFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        ms_common::PixelFormat::Bgra8 => vk::Format::B8G8R8A8_UNORM,
        ms_common::PixelFormat::Rgba16F => vk::Format::R16G16B16A16_SFLOAT,
        ms_common::PixelFormat::Rgba32F => vk::Format::R32G32B32A32_SFLOAT,
        // NV12 and P010 are multi-plane formats. For storage purposes,
        // we allocate the Y plane as R8 (or R16 for P010) and handle
        // the UV plane separately.
        ms_common::PixelFormat::Nv12 => vk::Format::R8_UNORM,
        ms_common::PixelFormat::P010 => vk::Format::R16_UNORM,
    }
}

/// Returns the bytes per pixel for a Vulkan format.
fn format_bytes_per_pixel(format: vk::Format) -> u32 {
    match format {
        vk::Format::R8_UNORM => 1,
        vk::Format::R16_UNORM => 2,
        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM => 4,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::R32G32B32A32_SFLOAT => 16,
        _ => 4, // Default to 4 bytes (RGBA8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_location_variants() {
        // Verify the gpu-allocator types we depend on exist
        let _gpu_only = MemoryLocation::GpuOnly;
        let _cpu_to_gpu = MemoryLocation::CpuToGpu;
        let _gpu_to_cpu = MemoryLocation::GpuToCpu;
    }

    #[test]
    fn pixel_format_conversion() {
        assert_eq!(
            pixel_format_to_vk(ms_common::PixelFormat::Rgba8),
            vk::Format::R8G8B8A8_UNORM
        );
        assert_eq!(
            pixel_format_to_vk(ms_common::PixelFormat::Bgra8),
            vk::Format::B8G8R8A8_UNORM
        );
        assert_eq!(
            pixel_format_to_vk(ms_common::PixelFormat::Rgba16F),
            vk::Format::R16G16B16A16_SFLOAT
        );
        assert_eq!(
            pixel_format_to_vk(ms_common::PixelFormat::Rgba32F),
            vk::Format::R32G32B32A32_SFLOAT
        );
        assert_eq!(
            pixel_format_to_vk(ms_common::PixelFormat::Nv12),
            vk::Format::R8_UNORM
        );
        assert_eq!(
            pixel_format_to_vk(ms_common::PixelFormat::P010),
            vk::Format::R16_UNORM
        );
    }

    #[test]
    fn format_bpp_values() {
        assert_eq!(format_bytes_per_pixel(vk::Format::R8_UNORM), 1);
        assert_eq!(format_bytes_per_pixel(vk::Format::R16_UNORM), 2);
        assert_eq!(format_bytes_per_pixel(vk::Format::R8G8B8A8_UNORM), 4);
        assert_eq!(format_bytes_per_pixel(vk::Format::R16G16B16A16_SFLOAT), 8);
        assert_eq!(format_bytes_per_pixel(vk::Format::R32G32B32A32_SFLOAT), 16);
    }
}
