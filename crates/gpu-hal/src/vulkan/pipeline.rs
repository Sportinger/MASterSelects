//! Vulkan compute pipeline management.
//!
//! Handles creation of compute pipelines from SPIR-V shader modules,
//! descriptor set layouts, descriptor pools, and pipeline layouts.
//!
//! # Architecture
//!
//! - [`ComputePipeline`]: RAII wrapper owning a `VkPipeline`, `VkPipelineLayout`,
//!   and `VkDescriptorSetLayout`. All objects are destroyed in correct order on drop.
//! - [`DescriptorManager`]: Manages a `VkDescriptorPool` and allocates descriptor
//!   sets with storage buffer bindings for compute shaders.
//! - [`PipelineCache`]: In-memory cache of named compute pipelines for reuse.
//!   Supports creation from [`KernelId`] for standardized kernel lookup.
//!
//! # Dispatch Helpers
//!
//! [`compute_dispatch_1d`] and [`compute_dispatch_2d`] calculate optimal workgroup
//! counts for 1D (data processing) and 2D (image processing) compute dispatches,
//! mirroring the CUDA backend's `compute_launch_config_*` helpers.

use std::collections::HashMap;
use std::ffi::CStr;

use ash::vk;
use tracing::{debug, info};

use ms_common::kernel::KernelId;

use super::error::VulkanError;

// ---------------------------------------------------------------------------
// ComputePipeline
// ---------------------------------------------------------------------------

/// A compute pipeline with its associated layout and descriptor set layout.
///
/// Owns all Vulkan objects and implements RAII cleanup. The destruction order
/// is: pipeline, then pipeline layout, then descriptor set layout.
pub struct ComputePipeline {
    /// The Vulkan pipeline handle.
    pipeline: vk::Pipeline,
    /// The pipeline layout.
    pipeline_layout: vk::PipelineLayout,
    /// The descriptor set layout (defines binding points for buffers/images).
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Push constant size in bytes (0 if none).
    push_constant_size: u32,
    /// Number of storage buffer bindings.
    binding_count: u32,
    /// Reference to the device for cleanup.
    device: ash::Device,
    /// Name for debugging.
    name: String,
}

impl ComputePipeline {
    /// Create a compute pipeline from a shader module.
    ///
    /// # Arguments
    /// - `device`: The Vulkan logical device.
    /// - `shader_module`: The SPIR-V shader module containing the compute shader.
    /// - `entry_point`: The entry point function name in the shader (e.g., `c"main"`).
    /// - `binding_count`: Number of storage buffer bindings (descriptors).
    /// - `push_constant_size`: Size of push constants in bytes (0 for none).
    /// - `name`: Name for debugging.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::PipelineCreation`] if any Vulkan object creation fails.
    pub fn new(
        device: &ash::Device,
        shader_module: vk::ShaderModule,
        entry_point: &CStr,
        binding_count: u32,
        push_constant_size: u32,
        name: impl Into<String>,
    ) -> Result<Self, VulkanError> {
        let name = name.into();

        // Create descriptor set layout with N storage buffer bindings.
        let descriptor_set_layout = Self::create_descriptor_set_layout(device, binding_count)?;

        // Create pipeline layout with optional push constants.
        let pipeline_layout =
            Self::create_pipeline_layout(device, descriptor_set_layout, push_constant_size)?;

        // Create the compute pipeline.
        let pipeline = Self::create_pipeline(device, shader_module, entry_point, pipeline_layout)?;

        debug!(
            name = %name,
            bindings = binding_count,
            push_constants = push_constant_size,
            "Created compute pipeline"
        );

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            push_constant_size,
            binding_count,
            device: device.clone(),
            name,
        })
    }

    /// Create a compute pipeline for a [`KernelId`].
    ///
    /// Convenience wrapper that uses the kernel's entry point name. The SPIR-V
    /// shader module must already be loaded (typically via [`ShaderRegistry`]).
    ///
    /// [`ShaderRegistry`]: super::shader::ShaderRegistry
    pub fn for_kernel(
        device: &ash::Device,
        shader_module: vk::ShaderModule,
        kernel_id: &KernelId,
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<Self, VulkanError> {
        let name = kernel_id.vulkan_module_name();
        // Compute shaders use "main" as the entry point by convention.
        Self::new(
            device,
            shader_module,
            c"main",
            binding_count,
            push_constant_size,
            name,
        )
    }

    /// Create a descriptor set layout with `count` storage buffer bindings.
    ///
    /// Each binding is a `STORAGE_BUFFER` accessible from the compute stage.
    fn create_descriptor_set_layout(
        device: &ash::Device,
        count: u32,
    ) -> Result<vk::DescriptorSetLayout, VulkanError> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..count)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            // SAFETY: Device is valid, layout_info is properly initialized
            // with valid binding descriptions.
            device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(VulkanError::PipelineCreation)
        }
    }

    /// Create a pipeline layout with optional push constants.
    fn create_pipeline_layout(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        push_constant_size: u32,
    ) -> Result<vk::PipelineLayout, VulkanError> {
        let set_layouts = [descriptor_set_layout];

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size);

        let push_constant_ranges = if push_constant_size > 0 {
            std::slice::from_ref(&push_constant_range)
        } else {
            &[]
        };

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(push_constant_ranges);

        unsafe {
            // SAFETY: Device is valid, layout_info references valid descriptor
            // set layouts and push constant ranges.
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(VulkanError::PipelineCreation)
        }
    }

    /// Create a compute pipeline with the given shader module.
    fn create_pipeline(
        device: &ash::Device,
        shader_module: vk::ShaderModule,
        entry_point: &CStr,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, VulkanError> {
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipelines = unsafe {
            // SAFETY: Device is valid, pipeline_info references a valid shader
            // module and pipeline layout.
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_pipelines, err)| VulkanError::PipelineCreation(err))?
        };

        Ok(pipelines[0])
    }

    /// Returns the Vulkan pipeline handle.
    #[inline]
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Returns the pipeline layout.
    #[inline]
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Returns the descriptor set layout.
    #[inline]
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    /// Returns the push constant size in bytes (0 if none).
    #[inline]
    pub fn push_constant_size(&self) -> u32 {
        self.push_constant_size
    }

    /// Returns the number of storage buffer bindings.
    #[inline]
    pub fn binding_count(&self) -> u32 {
        self.binding_count
    }

    /// Returns the pipeline name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: We own these Vulkan objects and the device is valid.
            // Destruction order: pipeline first, then layout, then descriptor layout.
            // This is the reverse of creation order, ensuring no dangling references.
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
        debug!(name = %self.name, "Destroyed compute pipeline");
    }
}

// ---------------------------------------------------------------------------
// DescriptorManager
// ---------------------------------------------------------------------------

/// Manages descriptor pools and descriptor set allocation for compute pipelines.
///
/// Allocates descriptor sets from a pool and handles RAII cleanup. Supports
/// storage buffer descriptors (the primary descriptor type for compute shaders)
/// and storage image descriptors (for texture-based effects).
///
/// When the pool is dropped, all allocated descriptor sets are implicitly freed.
pub struct DescriptorManager {
    /// The descriptor pool.
    pool: vk::DescriptorPool,
    /// Maximum number of sets that can be allocated.
    max_sets: u32,
    /// Currently allocated set count.
    allocated_count: u32,
    /// Reference to the device.
    device: ash::Device,
}

impl DescriptorManager {
    /// Create a new descriptor manager with a pool for storage buffer descriptors.
    ///
    /// - `max_sets`: Maximum number of descriptor sets that can be allocated.
    /// - `max_storage_buffers`: Maximum number of storage buffer descriptors across all sets.
    pub fn new(
        device: &ash::Device,
        max_sets: u32,
        max_storage_buffers: u32,
    ) -> Result<Self, VulkanError> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(max_storage_buffers)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let pool = unsafe {
            // SAFETY: Device is valid, pool_info is properly initialized.
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(VulkanError::DescriptorPoolCreation)?
        };

        debug!(
            max_sets = max_sets,
            max_storage_buffers = max_storage_buffers,
            "Created descriptor pool"
        );

        Ok(Self {
            pool,
            max_sets,
            allocated_count: 0,
            device: device.clone(),
        })
    }

    /// Create a descriptor manager with support for both storage buffers and
    /// storage images.
    ///
    /// Storage images are used for texture-based compute effects (e.g. image
    /// processing kernels that read/write textures directly).
    pub fn with_images(
        device: &ash::Device,
        max_sets: u32,
        max_storage_buffers: u32,
        max_storage_images: u32,
    ) -> Result<Self, VulkanError> {
        let mut pool_sizes = Vec::with_capacity(2);

        if max_storage_buffers > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(max_storage_buffers),
            );
        }
        if max_storage_images > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(max_storage_images),
            );
        }

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let pool = unsafe {
            // SAFETY: Device is valid, pool_info is properly initialized with
            // valid pool sizes for storage buffers and/or storage images.
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(VulkanError::DescriptorPoolCreation)?
        };

        debug!(
            max_sets = max_sets,
            max_storage_buffers = max_storage_buffers,
            max_storage_images = max_storage_images,
            "Created descriptor pool (buffers + images)"
        );

        Ok(Self {
            pool,
            max_sets,
            allocated_count: 0,
            device: device.clone(),
        })
    }

    /// Allocate a descriptor set for the given layout.
    pub fn allocate_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, VulkanError> {
        let set_layouts = [layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&set_layouts);

        let sets = unsafe {
            // SAFETY: Device and pool are valid. Layout is a valid descriptor
            // set layout compatible with this pool's type.
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(VulkanError::DescriptorAllocation)?
        };

        self.allocated_count += 1;
        Ok(sets[0])
    }

    /// Free a previously allocated descriptor set back to the pool.
    ///
    /// The pool must have been created with `FREE_DESCRIPTOR_SET` flag (which
    /// is the default for this manager).
    pub fn free_set(&mut self, descriptor_set: vk::DescriptorSet) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Device and pool are valid. descriptor_set was allocated
            // from this pool (caller must ensure this). The pool was created
            // with FREE_DESCRIPTOR_SET flag.
            self.device
                .free_descriptor_sets(self.pool, &[descriptor_set])
                .map_err(VulkanError::DescriptorAllocation)?;
        }

        self.allocated_count = self.allocated_count.saturating_sub(1);
        Ok(())
    }

    /// Reset the entire descriptor pool, freeing all allocated descriptor sets.
    ///
    /// This is more efficient than freeing sets individually when all sets
    /// should be released at once (e.g. between frames).
    pub fn reset(&mut self) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Device and pool are valid. After reset, all descriptor sets
            // previously allocated from this pool become invalid.
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(VulkanError::DescriptorPoolCreation)?;
        }

        let prev_count = self.allocated_count;
        self.allocated_count = 0;
        debug!(freed = prev_count, "Reset descriptor pool");
        Ok(())
    }

    /// Update a descriptor set to bind a storage buffer at a specific binding.
    pub fn update_buffer_binding(
        &self,
        descriptor_set: vk::DescriptorSet,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(range);

        let buffer_infos = [buffer_info];

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos);

        unsafe {
            // SAFETY: Device is valid, descriptor_set was allocated from our pool,
            // buffer is a valid Vulkan buffer.
            self.device.update_descriptor_sets(&[write], &[]);
        }
    }

    /// Update a descriptor set to bind a storage image at a specific binding.
    ///
    /// Storage images are used for compute shaders that read/write textures
    /// directly (e.g. image processing effects).
    pub fn update_image_binding(
        &self,
        descriptor_set: vk::DescriptorSet,
        binding: u32,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(image_layout);

        let image_infos = [image_info];

        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos);

        unsafe {
            // SAFETY: Device is valid, descriptor_set was allocated from our pool,
            // image_view references a valid image.
            self.device.update_descriptor_sets(&[write], &[]);
        }
    }

    /// Batch-update multiple storage buffer bindings on a descriptor set.
    ///
    /// Each entry in `bindings` is `(binding_index, buffer, offset, range)`.
    /// More efficient than calling `update_buffer_binding` in a loop because
    /// all updates are submitted in a single Vulkan call.
    pub fn update_buffer_bindings(
        &self,
        descriptor_set: vk::DescriptorSet,
        bindings: &[(u32, vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
    ) {
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .map(|&(_, buffer, offset, range)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(offset)
                    .range(range)
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = bindings
            .iter()
            .zip(buffer_infos.iter())
            .map(|(&(binding, _, _, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe {
            // SAFETY: Device is valid, all descriptor sets and buffers are valid.
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Returns the descriptor pool handle.
    #[inline]
    pub fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// Returns the maximum number of sets this pool can allocate.
    #[inline]
    pub fn max_sets(&self) -> u32 {
        self.max_sets
    }

    /// Returns how many sets have been allocated.
    #[inline]
    pub fn allocated_count(&self) -> u32 {
        self.allocated_count
    }

    /// Returns how many sets can still be allocated.
    #[inline]
    pub fn remaining_capacity(&self) -> u32 {
        self.max_sets.saturating_sub(self.allocated_count)
    }
}

impl Drop for DescriptorManager {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: We own the pool and the device is valid.
            // Destroying the pool implicitly frees all allocated descriptor sets.
            self.device.destroy_descriptor_pool(self.pool, None);
        }
        debug!(
            allocated = self.allocated_count,
            "Destroyed descriptor pool"
        );
    }
}

// ---------------------------------------------------------------------------
// PipelineCache
// ---------------------------------------------------------------------------

/// Cache for compute pipelines, keyed by name.
///
/// Provides fast lookup of previously created pipelines and supports
/// creation via [`KernelId`] for standardized kernel dispatch.
pub struct PipelineCache {
    pipelines: HashMap<String, ComputePipeline>,
}

impl PipelineCache {
    /// Create an empty pipeline cache.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// Insert a pipeline into the cache.
    ///
    /// If a pipeline with the same name already exists, the old pipeline is
    /// dropped (and its Vulkan objects are destroyed).
    pub fn insert(&mut self, pipeline: ComputePipeline) {
        let name = pipeline.name().to_string();
        if self.pipelines.contains_key(&name) {
            debug!(name = %name, "Replacing existing pipeline in cache");
        }
        self.pipelines.insert(name, pipeline);
    }

    /// Get a pipeline by name.
    pub fn get(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name)
    }

    /// Get a pipeline for a [`KernelId`].
    pub fn get_kernel(&self, kernel_id: &KernelId) -> Option<&ComputePipeline> {
        let name = kernel_id.vulkan_module_name();
        self.get(&name)
    }

    /// Get or create a pipeline for a [`KernelId`].
    ///
    /// If the pipeline is already cached, returns a reference to it. Otherwise,
    /// creates a new pipeline from the given shader module and caches it.
    ///
    /// # Arguments
    /// - `device`: Vulkan logical device.
    /// - `shader_module`: SPIR-V shader module handle (must already be loaded).
    /// - `kernel_id`: The kernel identifier.
    /// - `binding_count`: Number of storage buffer bindings.
    /// - `push_constant_size`: Size of push constants in bytes.
    pub fn get_or_create_kernel(
        &mut self,
        device: &ash::Device,
        shader_module: vk::ShaderModule,
        kernel_id: &KernelId,
        binding_count: u32,
        push_constant_size: u32,
    ) -> Result<&ComputePipeline, VulkanError> {
        let name = kernel_id.vulkan_module_name();

        if !self.pipelines.contains_key(&name) {
            let pipeline = ComputePipeline::for_kernel(
                device,
                shader_module,
                kernel_id,
                binding_count,
                push_constant_size,
            )?;
            self.pipelines.insert(name.clone(), pipeline);
        }

        Ok(self.pipelines.get(&name).expect("just inserted"))
    }

    /// Remove a pipeline by name, destroying its Vulkan objects.
    ///
    /// Returns `true` if the pipeline was found and removed.
    pub fn remove(&mut self, name: &str) -> bool {
        let removed = self.pipelines.remove(name).is_some();
        if removed {
            info!(name = name, "Removed pipeline from cache");
        }
        removed
    }

    /// Remove a pipeline for a [`KernelId`].
    ///
    /// Returns `true` if the pipeline was found and removed.
    pub fn remove_kernel(&mut self, kernel_id: &KernelId) -> bool {
        let name = kernel_id.vulkan_module_name();
        self.remove(&name)
    }

    /// Remove all pipelines, destroying all Vulkan objects.
    pub fn clear(&mut self) {
        let count = self.pipelines.len();
        self.pipelines.clear();
        if count > 0 {
            info!(count = count, "Cleared all pipelines from cache");
        }
    }

    /// Returns the number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Returns true if no pipelines are cached.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Returns an iterator over cached pipeline names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.pipelines.keys().map(|s| s.as_str())
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Dispatch dimension helpers
// ---------------------------------------------------------------------------

/// Default workgroup size for 1D compute dispatches.
pub const WORKGROUP_SIZE_1D: u32 = 256;

/// Default workgroup size for 2D compute dispatches (per dimension).
pub const WORKGROUP_SIZE_2D: u32 = 16;

/// Calculate optimal dispatch dimensions for a 1D compute kernel.
///
/// Returns `(group_count_x, group_count_y, group_count_z)` for
/// `vkCmdDispatch`. Uses a workgroup size of [`WORKGROUP_SIZE_1D`] (256).
///
/// # Example
///
/// ```
/// # use ms_gpu_hal::vulkan::pipeline::compute_dispatch_1d;
/// let (gx, gy, gz) = compute_dispatch_1d(1024);
/// assert_eq!((gx, gy, gz), (4, 1, 1));
/// ```
pub fn compute_dispatch_1d(num_elements: u32) -> (u32, u32, u32) {
    let group_x = num_elements.div_ceil(WORKGROUP_SIZE_1D);
    (group_x, 1, 1)
}

/// Calculate optimal dispatch dimensions for a 2D compute kernel
/// (e.g. image processing).
///
/// Returns `(group_count_x, group_count_y, group_count_z)` for
/// `vkCmdDispatch`. Uses a workgroup size of [`WORKGROUP_SIZE_2D`] x
/// [`WORKGROUP_SIZE_2D`] (16x16 = 256 threads).
///
/// # Example
///
/// ```
/// # use ms_gpu_hal::vulkan::pipeline::compute_dispatch_2d;
/// let (gx, gy, gz) = compute_dispatch_2d(1920, 1080);
/// assert_eq!((gx, gy, gz), (120, 68, 1));
/// ```
pub fn compute_dispatch_2d(width: u32, height: u32) -> (u32, u32, u32) {
    let group_x = width.div_ceil(WORKGROUP_SIZE_2D);
    let group_y = height.div_ceil(WORKGROUP_SIZE_2D);
    (group_x, group_y, 1)
}

/// Calculate dispatch dimensions with custom workgroup sizes.
///
/// Useful for kernels that require non-standard workgroup dimensions.
pub fn compute_dispatch_custom(
    total_x: u32,
    total_y: u32,
    total_z: u32,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
) -> (u32, u32, u32) {
    (
        total_x.div_ceil(wg_x),
        total_y.div_ceil(wg_y),
        total_z.div_ceil(wg_z),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_cache_empty() {
        let cache = PipelineCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn pipeline_cache_default_trait() {
        let cache = PipelineCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn pipeline_cache_kernel_lookup() {
        let cache = PipelineCache::new();
        assert!(cache.get_kernel(&KernelId::Nv12ToRgba).is_none());
        assert!(cache.get_kernel(&KernelId::AlphaBlend).is_none());
        assert!(
            cache
                .get_kernel(&KernelId::Effect("blur".into()))
                .is_none()
        );
    }

    #[test]
    fn dispatch_1d_exact() {
        let (gx, gy, gz) = compute_dispatch_1d(256);
        assert_eq!((gx, gy, gz), (1, 1, 1));
    }

    #[test]
    fn dispatch_1d_round_up() {
        let (gx, gy, gz) = compute_dispatch_1d(257);
        assert_eq!((gx, gy, gz), (2, 1, 1));
    }

    #[test]
    fn dispatch_1d_large() {
        let (gx, gy, gz) = compute_dispatch_1d(1920 * 1080 * 4);
        assert_eq!(gx, (1920 * 1080 * 4u32).div_ceil(256));
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }

    #[test]
    fn dispatch_2d_standard_resolutions() {
        // 1080p
        let (gx, gy, gz) = compute_dispatch_2d(1920, 1080);
        assert_eq!((gx, gy, gz), (120, 68, 1)); // ceil(1920/16)=120, ceil(1080/16)=68

        // 4K
        let (gx, gy, gz) = compute_dispatch_2d(3840, 2160);
        assert_eq!((gx, gy, gz), (240, 135, 1)); // ceil(3840/16)=240, ceil(2160/16)=135

        // 720p
        let (gx, gy, gz) = compute_dispatch_2d(1280, 720);
        assert_eq!((gx, gy, gz), (80, 45, 1)); // ceil(1280/16)=80, ceil(720/16)=45
    }

    #[test]
    fn dispatch_2d_non_aligned() {
        let (gx, gy, gz) = compute_dispatch_2d(100, 100);
        assert_eq!((gx, gy, gz), (7, 7, 1)); // ceil(100/16)=7
    }

    #[test]
    fn dispatch_custom() {
        let (gx, gy, gz) = compute_dispatch_custom(1000, 500, 10, 32, 32, 1);
        assert_eq!((gx, gy, gz), (32, 16, 10)); // ceil(1000/32)=32, ceil(500/32)=16
    }

    #[test]
    fn workgroup_sizes() {
        assert_eq!(WORKGROUP_SIZE_1D, 256);
        assert_eq!(WORKGROUP_SIZE_2D, 16);
        // 2D workgroup has 16*16 = 256 threads, matching 1D
        assert_eq!(WORKGROUP_SIZE_2D * WORKGROUP_SIZE_2D, WORKGROUP_SIZE_1D);
    }
}
