//! Vulkan queue and command buffer management.
//!
//! Provides command buffer allocation, recording, submission, and fence-based
//! synchronization for compute dispatch operations.
//!
//! # Key types
//!
//! - [`CommandPool`]: Manages a Vulkan command pool for the compute queue family.
//! - [`Fence`]: GPU-CPU synchronization primitive with RAII cleanup.
//! - [`ComputeRecorder`]: Builder-style API for recording compute dispatches.
//! - [`QueueSubmitter`]: Manages async command buffer submission with fence tracking.

use ash::vk;
use tracing::debug;

use super::error::VulkanError;

/// Manages a Vulkan command pool and allocates command buffers from it.
///
/// The command pool is created for the compute queue family. All command
/// buffers allocated from this pool must be used on the compute queue.
///
/// Implements RAII: the command pool (and all allocated command buffers)
/// are destroyed on drop.
pub struct CommandPool {
    /// The Vulkan command pool handle.
    pool: vk::CommandPool,
    /// The queue family this pool is created for.
    queue_family_index: u32,
    /// Reference to the device.
    device: ash::Device,
}

impl CommandPool {
    /// Create a new command pool for the given queue family.
    ///
    /// The pool is created with `RESET_COMMAND_BUFFER` flag, allowing individual
    /// command buffers to be reset and re-recorded.
    pub fn new(device: &ash::Device, queue_family_index: u32) -> Result<Self, VulkanError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let pool = unsafe {
            // SAFETY: Device is valid, queue_family_index references a valid
            // queue family (verified during device creation).
            device
                .create_command_pool(&pool_info, None)
                .map_err(VulkanError::CommandPoolCreation)?
        };

        debug!(queue_family = queue_family_index, "Created command pool");

        Ok(Self {
            pool,
            queue_family_index,
            device: device.clone(),
        })
    }

    /// Allocate a primary command buffer from this pool.
    pub fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer, VulkanError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe {
            // SAFETY: Device and pool are valid. We request exactly 1 primary
            // command buffer.
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(VulkanError::CommandBufferAllocation)?
        };

        Ok(buffers[0])
    }

    /// Allocate multiple primary command buffers from this pool.
    pub fn allocate_command_buffers(
        &self,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>, VulkanError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        unsafe {
            // SAFETY: Device and pool are valid.
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(VulkanError::CommandBufferAllocation)
        }
    }

    /// Reset the command pool, recycling all allocated command buffers.
    ///
    /// After this call, all command buffers allocated from this pool are in the
    /// initial state and can be re-recorded. This is more efficient than
    /// resetting individual command buffers when you want to recycle all of them.
    ///
    /// # Safety requirement
    ///
    /// The caller must ensure no command buffers from this pool are currently
    /// pending execution on the GPU.
    pub fn reset(&self) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Device and pool are valid. The caller ensures no command
            // buffers are pending on the GPU.
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(VulkanError::CommandPoolCreation)
        }
    }

    /// Returns the command pool handle.
    #[inline]
    pub fn pool(&self) -> vk::CommandPool {
        self.pool
    }

    /// Returns the queue family index.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: We own the command pool and device is valid.
            // Destroying the pool implicitly frees all command buffers
            // allocated from it.
            self.device.destroy_command_pool(self.pool, None);
        }
        debug!(
            queue_family = self.queue_family_index,
            "Destroyed command pool"
        );
    }
}

/// A fence for GPU-CPU synchronization with RAII cleanup.
pub struct Fence {
    /// The Vulkan fence handle.
    fence: vk::Fence,
    /// Reference to the device.
    device: ash::Device,
}

impl Fence {
    /// Create a new fence in the unsignaled state.
    pub fn new(device: &ash::Device) -> Result<Self, VulkanError> {
        let fence_info = vk::FenceCreateInfo::default();

        let fence = unsafe {
            // SAFETY: Device is valid, fence_info is default (unsignaled).
            device
                .create_fence(&fence_info, None)
                .map_err(VulkanError::FenceCreation)?
        };

        Ok(Self {
            fence,
            device: device.clone(),
        })
    }

    /// Create a new fence in the signaled state.
    ///
    /// Useful for fences that are waited on before first use.
    pub fn new_signaled(device: &ash::Device) -> Result<Self, VulkanError> {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let fence = unsafe {
            // SAFETY: Device is valid.
            device
                .create_fence(&fence_info, None)
                .map_err(VulkanError::FenceCreation)?
        };

        Ok(Self {
            fence,
            device: device.clone(),
        })
    }

    /// Wait for this fence to be signaled, with a timeout in nanoseconds.
    ///
    /// Use `u64::MAX` for no timeout.
    pub fn wait(&self, timeout_ns: u64) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Device and fence are valid.
            self.device
                .wait_for_fences(&[self.fence], true, timeout_ns)
                .map_err(VulkanError::FenceWait)
        }
    }

    /// Check whether this fence is currently signaled, without blocking.
    ///
    /// Returns `true` if the fence is signaled (GPU work completed),
    /// `false` if still pending.
    pub fn is_signaled(&self) -> Result<bool, VulkanError> {
        unsafe {
            // SAFETY: Device and fence are valid.
            // In ash 0.38, get_fence_status returns Result<bool, vk::Result>
            // where true = VK_SUCCESS (signaled), false = VK_NOT_READY.
            self.device
                .get_fence_status(self.fence)
                .map_err(VulkanError::FenceWait)
        }
    }

    /// Reset this fence to the unsignaled state.
    pub fn reset(&self) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Device and fence are valid. Fence must not be in use
            // by a queue submission (caller must ensure this).
            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::FenceWait)
        }
    }

    /// Returns the Vulkan fence handle.
    #[inline]
    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: We own this fence and the device is valid.
            self.device.destroy_fence(self.fence, None);
        }
    }
}

/// Helper for recording and submitting compute command buffers.
///
/// Provides a builder-style API for recording compute dispatches.
pub struct ComputeRecorder<'a> {
    /// The command buffer being recorded.
    command_buffer: vk::CommandBuffer,
    /// The device for recording commands.
    device: &'a ash::Device,
    /// Whether recording has been started.
    recording: bool,
}

impl<'a> ComputeRecorder<'a> {
    /// Begin recording commands into a command buffer.
    pub fn begin(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
    ) -> Result<Self, VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            // SAFETY: Device and command_buffer are valid. The command buffer
            // must not be in the recording state (caller ensures this).
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(VulkanError::CommandBufferRecording)?;
        }

        Ok(Self {
            command_buffer,
            device,
            recording: true,
        })
    }

    /// Bind a compute pipeline.
    pub fn bind_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            // Pipeline is a valid compute pipeline.
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        }
    }

    /// Bind descriptor sets for compute.
    pub fn bind_descriptor_sets(
        &self,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            // Pipeline layout and descriptor sets are valid.
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                descriptor_sets,
                &[],
            );
        }
    }

    /// Push constant data for compute.
    ///
    /// The data is uploaded inline with the command buffer and is available
    /// to the shader immediately (no buffer allocation needed).
    pub fn push_constants(
        &self,
        pipeline_layout: vk::PipelineLayout,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            // Pipeline layout is valid. Data slice is valid for its length.
            self.device.cmd_push_constants(
                self.command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                offset,
                data,
            );
        }
    }

    /// Record a compute dispatch.
    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            // A compute pipeline must be bound before dispatching.
            self.device.cmd_dispatch(
                self.command_buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    /// Record a memory barrier for compute-to-compute synchronization.
    ///
    /// This ensures all writes from a previous dispatch are visible to
    /// subsequent dispatches.
    pub fn compute_barrier(&self) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }

    /// Record a memory barrier for transfer-to-compute synchronization.
    ///
    /// Use this after a buffer copy (upload) before dispatching a compute
    /// shader that reads the buffer.
    pub fn transfer_to_compute_barrier(&self) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }

    /// Record a memory barrier for compute-to-transfer synchronization.
    ///
    /// Use this after a compute dispatch before reading back results
    /// via a buffer copy (download).
    pub fn compute_to_transfer_barrier(&self) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }

    /// Record a buffer copy command.
    pub fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: vk::DeviceSize) {
        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(size);

        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            // src and dst buffers are valid with sufficient size.
            self.device
                .cmd_copy_buffer(self.command_buffer, src, dst, &[region]);
        }
    }

    /// Record a buffer copy with explicit source and destination offsets.
    pub fn copy_buffer_region(
        &self,
        src: vk::Buffer,
        src_offset: vk::DeviceSize,
        dst: vk::Buffer,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);

        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            // src and dst buffers are valid with sufficient size for the offsets.
            self.device
                .cmd_copy_buffer(self.command_buffer, src, dst, &[region]);
        }
    }

    /// Fill a buffer with a constant u32 value.
    ///
    /// Useful for clearing buffers to zero or initializing with a pattern.
    /// `offset` and `size` must be multiples of 4.
    pub fn fill_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        data: u32,
    ) {
        unsafe {
            // SAFETY: Device and command buffer are valid and in recording state.
            // Buffer is valid. Offset and size must be multiples of 4 (caller ensures).
            self.device
                .cmd_fill_buffer(self.command_buffer, buffer, offset, size, data);
        }
    }

    /// End recording and return the command buffer handle.
    pub fn finish(mut self) -> Result<vk::CommandBuffer, VulkanError> {
        self.recording = false;
        unsafe {
            // SAFETY: Device and command buffer are valid. The command buffer
            // is in the recording state.
            self.device
                .end_command_buffer(self.command_buffer)
                .map_err(VulkanError::CommandBufferRecording)?;
        }
        Ok(self.command_buffer)
    }

    /// Returns the command buffer handle (for advanced use).
    #[inline]
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

/// Manages asynchronous command buffer submission with fence-based tracking.
///
/// `QueueSubmitter` provides a higher-level API over raw queue submission,
/// tracking in-flight work via fences. This is useful for double- or
/// triple-buffered rendering where you need to wait for a previous frame's
/// GPU work to complete before reusing its command buffer.
///
/// # Example flow
///
/// ```no_run
/// # use ms_gpu_hal::vulkan::queue::QueueSubmitter;
/// // Create submitter
/// // let submitter = QueueSubmitter::new(&device, queue)?;
///
/// // Submit work
/// // submitter.submit(command_buffer)?;
///
/// // Wait for completion before reusing resources
/// // submitter.wait_idle()?;
/// ```
pub struct QueueSubmitter {
    /// The compute queue.
    queue: vk::Queue,
    /// The queue family index.
    queue_family_index: u32,
    /// Fence for tracking the most recent submission.
    fence: Fence,
    /// Whether there is currently in-flight work.
    has_pending_work: bool,
    /// Reference to the device.
    device: ash::Device,
}

impl QueueSubmitter {
    /// Create a new queue submitter for the given queue.
    pub fn new(
        device: &ash::Device,
        queue: vk::Queue,
        queue_family_index: u32,
    ) -> Result<Self, VulkanError> {
        let fence = Fence::new(device)?;

        Ok(Self {
            queue,
            queue_family_index,
            fence,
            has_pending_work: false,
            device: device.clone(),
        })
    }

    /// Submit a command buffer to the queue.
    ///
    /// If there is pending work from a previous submission, this waits for it
    /// to complete first (to ensure the fence can be reused).
    pub fn submit(&mut self, command_buffer: vk::CommandBuffer) -> Result<(), VulkanError> {
        // Wait for any previous submission to complete before reusing the fence
        if self.has_pending_work {
            self.fence.wait(u64::MAX)?;
            self.fence.reset()?;
        }

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        unsafe {
            // SAFETY: Device, queue, and command buffer are valid.
            // The fence was just reset or is freshly created.
            self.device
                .queue_submit(self.queue, &[submit_info], self.fence.fence())
                .map_err(VulkanError::QueueSubmit)?;
        }

        self.has_pending_work = true;
        Ok(())
    }

    /// Wait for all submitted work to complete.
    ///
    /// This blocks until the GPU has finished all operations submitted
    /// through this submitter.
    pub fn wait_idle(&mut self) -> Result<(), VulkanError> {
        if self.has_pending_work {
            self.fence.wait(u64::MAX)?;
            self.has_pending_work = false;
        }
        Ok(())
    }

    /// Check whether the most recent submission has completed.
    ///
    /// Returns `true` if there is no pending work or if the GPU has finished.
    pub fn is_idle(&self) -> Result<bool, VulkanError> {
        if !self.has_pending_work {
            return Ok(true);
        }
        self.fence.is_signaled()
    }

    /// Returns the queue handle.
    #[inline]
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Returns the queue family index.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns whether there is currently pending GPU work.
    #[inline]
    pub fn has_pending_work(&self) -> bool {
        self.has_pending_work
    }
}

/// Submit a command buffer to the compute queue and wait for completion.
///
/// This is a synchronous convenience function. For async operation, use
/// fences directly or [`QueueSubmitter`].
pub fn submit_and_wait(
    device: &ash::Device,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) -> Result<(), VulkanError> {
    let fence = Fence::new(device)?;

    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

    unsafe {
        // SAFETY: Device, queue, and command buffer are valid.
        // The fence was just created in unsignaled state.
        device
            .queue_submit(queue, &[submit_info], fence.fence())
            .map_err(VulkanError::QueueSubmit)?;
    }

    // Wait indefinitely for completion
    fence.wait(u64::MAX)?;

    Ok(())
}

/// Submit a command buffer to the compute queue with a pre-existing fence.
///
/// The caller is responsible for waiting on the fence.
pub fn submit_with_fence(
    device: &ash::Device,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
    fence: &Fence,
) -> Result<(), VulkanError> {
    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

    unsafe {
        // SAFETY: Device, queue, and command buffer are valid.
        // Fence must be in unsignaled state (caller ensures this).
        device
            .queue_submit(queue, &[submit_info], fence.fence())
            .map_err(VulkanError::QueueSubmit)
    }
}

/// Wait for multiple fences to be signaled.
///
/// If `wait_all` is true, waits for ALL fences. If false, returns when ANY
/// fence is signaled.
pub fn wait_fences(
    device: &ash::Device,
    fences: &[&Fence],
    wait_all: bool,
    timeout_ns: u64,
) -> Result<(), VulkanError> {
    let fence_handles: Vec<vk::Fence> = fences.iter().map(|f| f.fence()).collect();

    unsafe {
        // SAFETY: Device is valid. All fences were created from this device.
        device
            .wait_for_fences(&fence_handles, wait_all, timeout_ns)
            .map_err(VulkanError::FenceWait)
    }
}

/// Reset multiple fences to the unsignaled state.
///
/// All fences must not be in use by a queue submission.
pub fn reset_fences(device: &ash::Device, fences: &[&Fence]) -> Result<(), VulkanError> {
    let fence_handles: Vec<vk::Fence> = fences.iter().map(|f| f.fence()).collect();

    unsafe {
        // SAFETY: Device is valid. All fences were created from this device
        // and are not in use (caller ensures this).
        device
            .reset_fences(&fence_handles)
            .map_err(VulkanError::FenceWait)
    }
}

#[cfg(test)]
mod tests {
    // Queue/command buffer tests require a live Vulkan device,
    // so we only test compilation and basic type construction here.

    #[test]
    fn module_compiles() {
        // Verify the module compiles cleanly.
        // Integration tests with a real GPU belong in tests/ directory.
    }
}
