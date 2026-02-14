//! Vulkan-specific error types.

use thiserror::Error;

/// Errors from the Vulkan backend.
#[derive(Error, Debug)]
pub enum VulkanError {
    #[error("Vulkan loader not available: {0}")]
    LoaderUnavailable(String),

    #[error("No Vulkan-capable physical device found")]
    NoDevice,

    #[error("No compute-capable queue family found")]
    NoComputeQueue,

    #[error("Vulkan instance creation failed: {0}")]
    InstanceCreation(ash::vk::Result),

    #[error("Vulkan device creation failed: {0}")]
    DeviceCreation(ash::vk::Result),

    #[error("Shader module creation failed: {0}")]
    ShaderCreation(ash::vk::Result),

    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(ash::vk::Result),

    #[error("Descriptor set allocation failed: {0}")]
    DescriptorAllocation(ash::vk::Result),

    #[error("Command buffer allocation failed: {0}")]
    CommandBufferAllocation(ash::vk::Result),

    #[error("Buffer creation failed: {0}")]
    BufferCreation(ash::vk::Result),

    #[error("Image creation failed: {0}")]
    ImageCreation(ash::vk::Result),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("Fence creation failed: {0}")]
    FenceCreation(ash::vk::Result),

    #[error("Queue submit failed: {0}")]
    QueueSubmit(ash::vk::Result),

    #[error("Fence wait failed: {0}")]
    FenceWait(ash::vk::Result),

    #[error("Command pool creation failed: {0}")]
    CommandPoolCreation(ash::vk::Result),

    #[error("Command buffer recording failed: {0}")]
    CommandBufferRecording(ash::vk::Result),

    #[error("Descriptor pool creation failed: {0}")]
    DescriptorPoolCreation(ash::vk::Result),
}

impl From<VulkanError> for ms_common::GpuError {
    fn from(err: VulkanError) -> Self {
        match err {
            VulkanError::MemoryAllocation(_) => ms_common::GpuError::AllocFailed {
                size: 0, // Size not available from gpu-allocator error string
            },
            VulkanError::NoDevice | VulkanError::LoaderUnavailable(_) => {
                ms_common::GpuError::NoBackend
            }
            other => ms_common::GpuError::DeviceInit(other.to_string()),
        }
    }
}
