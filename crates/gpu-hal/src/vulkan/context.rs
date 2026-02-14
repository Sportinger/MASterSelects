//! Vulkan context — Instance, PhysicalDevice, Device, and Queue management.
//!
//! This module owns all core Vulkan objects and handles RAII cleanup via Drop.
//! The `VulkanContext` is the foundation that all other Vulkan modules build upon.
//!
//! In debug builds, validation layers (`VK_LAYER_KHRONOS_validation`) and
//! `VK_EXT_debug_utils` are enabled to catch Vulkan API misuse early.

use std::ffi::CStr;

use ash::vk;
use tracing::{debug, info, warn};

use super::error::VulkanError;

/// Vulkan vendor IDs for device type identification.
const VENDOR_NVIDIA: u32 = 0x10DE;
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_INTEL: u32 = 0x8086;
const VENDOR_APPLE: u32 = 0x106B;

/// The standard Khronos validation layer name.
const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Core Vulkan context holding the entry loader, instance, device, and queues.
///
/// All Vulkan objects are destroyed in reverse creation order when this is dropped.
/// In debug builds, a debug messenger is created for validation layer output.
pub struct VulkanContext {
    /// The ash entry point (Vulkan loader).
    entry: ash::Entry,
    /// The Vulkan instance.
    instance: ash::Instance,
    /// Debug utils extension loader (debug builds only, if available).
    debug_utils_loader: Option<ash::ext::debug_utils::Instance>,
    /// Debug messenger handle (debug builds only, if available).
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// The selected physical device.
    physical_device: vk::PhysicalDevice,
    /// Physical device properties (name, limits, etc.).
    physical_device_properties: vk::PhysicalDeviceProperties,
    /// Physical device memory properties (heaps, memory types).
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// The logical device.
    device: ash::Device,
    /// Compute queue family index.
    compute_queue_family_index: u32,
    /// The compute queue.
    compute_queue: vk::Queue,
    /// Whether validation layers are active.
    validation_enabled: bool,
}

impl VulkanContext {
    /// Create a new Vulkan context.
    ///
    /// This initializes the Vulkan loader, creates an instance (with validation
    /// layers in debug builds), selects the best physical device (preferring
    /// discrete GPUs), and creates a logical device with a compute queue.
    pub fn new() -> Result<Self, VulkanError> {
        // Load the Vulkan entry points at runtime
        let entry = unsafe {
            // SAFETY: We are loading the system Vulkan loader. The returned Entry
            // must outlive all Vulkan objects created from it, which is guaranteed
            // by storing it in VulkanContext alongside all derived objects.
            ash::Entry::load()
        }
        .map_err(|e| VulkanError::LoaderUnavailable(e.to_string()))?;

        info!("Vulkan loader initialized");

        // Determine whether to enable validation layers
        let enable_validation = cfg!(debug_assertions) && Self::has_validation_layer(&entry);
        if enable_validation {
            info!("Vulkan validation layers enabled (debug build)");
        } else if cfg!(debug_assertions) {
            warn!("Validation layer not available — running without validation");
        }

        // Create Vulkan instance
        let instance = Self::create_instance(&entry, enable_validation)?;
        info!("Vulkan instance created");

        // Set up debug messenger (debug builds with validation layer)
        let (debug_utils_loader, debug_messenger) = if enable_validation {
            match Self::setup_debug_messenger(&entry, &instance) {
                Ok((loader, messenger)) => {
                    debug!("Vulkan debug messenger created");
                    (Some(loader), Some(messenger))
                }
                Err(e) => {
                    warn!(error = %e, "Failed to create debug messenger, continuing without it");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // Select physical device
        let (physical_device, properties, memory_props) = Self::select_physical_device(&instance)?;

        let device_name = decode_device_name(&properties.device_name);
        info!(
            device = %device_name,
            device_type = ?properties.device_type,
            api_version = format!(
                "{}.{}.{}",
                vk::api_version_major(properties.api_version),
                vk::api_version_minor(properties.api_version),
                vk::api_version_patch(properties.api_version),
            ),
            "Selected Vulkan physical device"
        );

        // Find compute queue family
        let compute_family = Self::find_compute_queue_family(&instance, physical_device)?;
        debug!(queue_family = compute_family, "Found compute queue family");

        // Create logical device + queue
        let device = Self::create_device(&instance, physical_device, compute_family)?;
        info!("Vulkan logical device created");

        let compute_queue = unsafe {
            // SAFETY: The device was just created with at least one queue in
            // the compute family. Queue index 0 is guaranteed to exist.
            device.get_device_queue(compute_family, 0)
        };

        Ok(Self {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            physical_device,
            physical_device_properties: properties,
            memory_properties: memory_props,
            device,
            compute_queue_family_index: compute_family,
            compute_queue,
            validation_enabled: enable_validation,
        })
    }

    /// Check whether the Khronos validation layer is available.
    fn has_validation_layer(entry: &ash::Entry) -> bool {
        let layer_props = unsafe {
            // SAFETY: Entry is valid, this enumerates available instance layers.
            entry.enumerate_instance_layer_properties()
        };

        match layer_props {
            Ok(layers) => {
                let found = layers.iter().any(|layer| {
                    let name = unsafe {
                        // SAFETY: LayerProperties.layer_name is a null-terminated
                        // C string filled by the Vulkan loader.
                        CStr::from_ptr(layer.layer_name.as_ptr())
                    };
                    name == VALIDATION_LAYER_NAME
                });

                if found {
                    debug!("Found Khronos validation layer");
                } else {
                    debug!(
                        "Khronos validation layer not found among {} layers",
                        layers.len()
                    );
                }

                found
            }
            Err(e) => {
                warn!(error = ?e, "Failed to enumerate instance layers");
                false
            }
        }
    }

    /// Check whether the debug utils extension is available.
    fn has_debug_utils_extension(entry: &ash::Entry) -> bool {
        let ext_props = unsafe {
            // SAFETY: Entry is valid, this enumerates available instance extensions.
            entry.enumerate_instance_extension_properties(None)
        };

        match ext_props {
            Ok(extensions) => extensions.iter().any(|ext| {
                let name = unsafe {
                    // SAFETY: ExtensionProperties.extension_name is a null-terminated
                    // C string filled by the Vulkan loader.
                    CStr::from_ptr(ext.extension_name.as_ptr())
                };
                name == ash::ext::debug_utils::NAME
            }),
            Err(_) => false,
        }
    }

    /// Create a Vulkan instance with minimal extensions (compute only, no display).
    ///
    /// In debug builds with validation layers available, enables:
    /// - `VK_LAYER_KHRONOS_validation` layer
    /// - `VK_EXT_debug_utils` extension
    fn create_instance(
        entry: &ash::Entry,
        enable_validation: bool,
    ) -> Result<ash::Instance, VulkanError> {
        let app_name = c"MasterSelects Engine";
        let engine_name = c"MasterSelects";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_2);

        // Collect enabled layers
        let mut enabled_layers: Vec<*const std::ffi::c_char> = Vec::new();
        if enable_validation {
            enabled_layers.push(VALIDATION_LAYER_NAME.as_ptr());
        }

        // Collect enabled extensions
        let mut enabled_extensions: Vec<*const std::ffi::c_char> = Vec::new();
        if enable_validation && Self::has_debug_utils_extension(entry) {
            enabled_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_extensions);

        unsafe {
            // SAFETY: The entry loader is valid (just loaded above), and we pass
            // a valid InstanceCreateInfo. Layer/extension names are static CStr
            // pointers that outlive this call. The returned Instance must not
            // outlive the Entry, which is ensured by our ownership model.
            entry
                .create_instance(&create_info, None)
                .map_err(VulkanError::InstanceCreation)
        }
    }

    /// Set up the debug utils messenger for validation layer output.
    ///
    /// The messenger routes validation messages to the `tracing` logger.
    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT), VulkanError> {
        let debug_utils_loader = ash::ext::debug_utils::Instance::new(entry, instance);

        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let messenger = unsafe {
            // SAFETY: The instance was just created with the debug utils extension
            // enabled. The messenger_info references a valid callback function.
            // The messenger will be destroyed before the instance in our Drop impl.
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_info, None)
                .map_err(VulkanError::InstanceCreation)?
        };

        Ok((debug_utils_loader, messenger))
    }

    /// Select the best physical device, preferring discrete GPUs.
    ///
    /// Returns (PhysicalDevice, Properties, MemoryProperties).
    fn select_physical_device(
        instance: &ash::Instance,
    ) -> Result<
        (
            vk::PhysicalDevice,
            vk::PhysicalDeviceProperties,
            vk::PhysicalDeviceMemoryProperties,
        ),
        VulkanError,
    > {
        let physical_devices = unsafe {
            // SAFETY: Instance is valid and was created by us.
            instance.enumerate_physical_devices()
        }
        .map_err(VulkanError::DeviceCreation)?;

        if physical_devices.is_empty() {
            return Err(VulkanError::NoDevice);
        }

        info!(
            count = physical_devices.len(),
            "Found Vulkan physical devices"
        );

        // Score and select the best device
        let mut best_device = None;
        let mut best_score = 0i64;

        for &pd in &physical_devices {
            let props = unsafe {
                // SAFETY: Instance is valid, pd is a device enumerated from it.
                instance.get_physical_device_properties(pd)
            };
            let mem_props = unsafe {
                // SAFETY: Instance is valid, pd is a device enumerated from it.
                instance.get_physical_device_memory_properties(pd)
            };

            let name = decode_device_name(&props.device_name);

            // Skip devices with no compute queue
            let has_compute = Self::device_has_compute_queue(instance, pd);
            if !has_compute {
                debug!(device = %name, "Skipping device: no compute queue");
                continue;
            }

            let score = Self::score_device(&props, &mem_props);

            debug!(
                device = %name,
                device_type = ?props.device_type,
                score = score,
                "Evaluated physical device"
            );

            if score > best_score {
                best_score = score;
                best_device = Some((pd, props, mem_props));
            }
        }

        best_device.ok_or(VulkanError::NoDevice)
    }

    /// Check if a physical device has at least one compute-capable queue family.
    fn device_has_compute_queue(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let queue_families = unsafe {
            // SAFETY: Instance is valid, physical_device was enumerated from it.
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        queue_families.iter().any(|family| {
            family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.queue_count > 0
        })
    }

    /// Score a physical device for selection. Higher is better.
    ///
    /// Discrete GPUs score highest, then integrated, then others.
    /// Within each category, more VRAM scores higher.
    fn score_device(
        props: &vk::PhysicalDeviceProperties,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
    ) -> i64 {
        let mut score: i64 = 0;

        // Prefer discrete GPUs strongly
        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => score += 10_000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => score += 1_000,
            vk::PhysicalDeviceType::VIRTUAL_GPU => score += 500,
            vk::PhysicalDeviceType::CPU => score += 100,
            _ => score += 50,
        }

        // Bonus for known discrete GPU vendors
        match props.vendor_id {
            VENDOR_NVIDIA | VENDOR_AMD => score += 500,
            VENDOR_INTEL => score += 100,
            _ => {}
        }

        // Add VRAM size as tiebreaker (in MB)
        let vram_bytes = Self::calculate_device_local_vram(mem_props);
        score += (vram_bytes / (1024 * 1024)) as i64;

        // Bonus for higher Vulkan API version support
        let api_minor = vk::api_version_minor(props.api_version);
        score += api_minor as i64 * 10;

        score
    }

    /// Calculate total device-local VRAM from memory properties.
    fn calculate_device_local_vram(mem_props: &vk::PhysicalDeviceMemoryProperties) -> u64 {
        let mut total = 0u64;
        for i in 0..mem_props.memory_heap_count as usize {
            let heap = mem_props.memory_heaps[i];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total += heap.size;
            }
        }
        total
    }

    /// Find a queue family that supports compute operations.
    ///
    /// Prefers a dedicated compute queue (no graphics) for async compute,
    /// but falls back to a combined graphics+compute queue.
    fn find_compute_queue_family(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<u32, VulkanError> {
        let queue_families = unsafe {
            // SAFETY: Instance is valid, physical_device was enumerated from it.
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        // First pass: look for a dedicated compute queue (compute but NOT graphics).
        // This is ideal for async compute workloads.
        for (index, family) in queue_families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && family.queue_count > 0
            {
                debug!(index = index, "Found dedicated compute queue family");
                return Ok(index as u32);
            }
        }

        // Second pass: any queue family with compute support.
        for (index, family) in queue_families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.queue_count > 0 {
                debug!(
                    index = index,
                    "Found compute-capable queue family (shared with graphics)"
                );
                return Ok(index as u32);
            }
        }

        Err(VulkanError::NoComputeQueue)
    }

    /// Create a logical device with a single compute queue.
    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        compute_queue_family: u32,
    ) -> Result<ash::Device, VulkanError> {
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priorities);

        let queue_create_infos = [queue_create_info];

        let device_create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

        unsafe {
            // SAFETY: Instance is valid, physical_device was enumerated from it,
            // and the queue_create_info references a valid compute queue family.
            // The returned Device must not outlive the Instance.
            instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(VulkanError::DeviceCreation)
        }
    }

    // -- Accessors --

    /// Returns a reference to the ash Entry loader.
    #[inline]
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// Returns a reference to the ash Instance.
    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Returns the physical device handle.
    #[inline]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns the physical device properties.
    #[inline]
    pub fn physical_device_properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.physical_device_properties
    }

    /// Returns the physical device memory properties.
    #[inline]
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    /// Returns a reference to the logical device.
    #[inline]
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Returns the compute queue family index.
    #[inline]
    pub fn compute_queue_family_index(&self) -> u32 {
        self.compute_queue_family_index
    }

    /// Returns the compute queue handle.
    #[inline]
    pub fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }

    /// Returns whether validation layers are active.
    #[inline]
    pub fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// Returns the GPU device name as a string.
    pub fn device_name(&self) -> String {
        decode_device_name(&self.physical_device_properties.device_name)
    }

    /// Returns the GPU vendor based on the vendor ID.
    pub fn vendor(&self) -> ms_common::GpuVendor {
        match self.physical_device_properties.vendor_id {
            VENDOR_NVIDIA => ms_common::GpuVendor::Nvidia,
            VENDOR_AMD => ms_common::GpuVendor::Amd,
            VENDOR_INTEL => ms_common::GpuVendor::Intel,
            VENDOR_APPLE => ms_common::GpuVendor::Apple,
            _ => ms_common::GpuVendor::Unknown,
        }
    }

    /// Returns total device-local VRAM in bytes.
    pub fn vram_total(&self) -> u64 {
        Self::calculate_device_local_vram(&self.memory_properties)
    }

    /// Returns the Vulkan API version string.
    pub fn api_version_string(&self) -> String {
        let v = self.physical_device_properties.api_version;
        format!(
            "{}.{}.{}",
            vk::api_version_major(v),
            vk::api_version_minor(v),
            vk::api_version_patch(v),
        )
    }

    /// Returns the maximum compute work group count limits [x, y, z].
    pub fn max_compute_work_group_count(&self) -> [u32; 3] {
        self.physical_device_properties
            .limits
            .max_compute_work_group_count
    }

    /// Returns the maximum compute work group size limits [x, y, z].
    pub fn max_compute_work_group_size(&self) -> [u32; 3] {
        self.physical_device_properties
            .limits
            .max_compute_work_group_size
    }

    /// Returns the maximum compute shared memory size in bytes.
    pub fn max_compute_shared_memory_size(&self) -> u32 {
        self.physical_device_properties
            .limits
            .max_compute_shared_memory_size
    }

    /// Returns the maximum compute work group invocations (threads per workgroup).
    pub fn max_compute_work_group_invocations(&self) -> u32 {
        self.physical_device_properties
            .limits
            .max_compute_work_group_invocations
    }

    /// Build a `GpuDeviceInfo` for this context.
    pub fn device_info(&self) -> ms_common::GpuDeviceInfo {
        ms_common::GpuDeviceInfo {
            name: self.device_name(),
            vendor: self.vendor(),
            vram_total: self.vram_total(),
            compute_capability: None, // CUDA only
            api_version: self.api_version_string(),
        }
    }

    /// Log device compute limits for debugging.
    pub fn log_compute_limits(&self) {
        let limits = &self.physical_device_properties.limits;
        debug!(
            max_work_group_count = ?limits.max_compute_work_group_count,
            max_work_group_size = ?limits.max_compute_work_group_size,
            max_work_group_invocations = limits.max_compute_work_group_invocations,
            max_shared_memory = limits.max_compute_shared_memory_size,
            max_push_constants = limits.max_push_constants_size,
            max_storage_buffer_range = limits.max_storage_buffer_range,
            max_bound_descriptor_sets = limits.max_bound_descriptor_sets,
            "Vulkan compute limits"
        );
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        info!("Destroying Vulkan context");

        unsafe {
            // SAFETY: We own the device and instance. Resources must be destroyed
            // in reverse creation order:
            //   1. Wait for device idle
            //   2. Destroy device
            //   3. Destroy debug messenger (before instance)
            //   4. Destroy instance
            //
            // All child resources (buffers, pipelines, etc.) must be dropped
            // before VulkanContext.
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);

            // Destroy debug messenger before the instance
            if let (Some(loader), Some(messenger)) =
                (self.debug_utils_loader.as_ref(), self.debug_messenger)
            {
                // SAFETY: The debug utils loader and messenger were created from
                // our instance, which is still alive at this point. The messenger
                // is destroyed before the instance.
                loader.destroy_debug_utils_messenger(messenger, None);
                debug!("Debug messenger destroyed");
            }

            self.instance.destroy_instance(None);
        }

        debug!("Vulkan context destroyed");
    }
}

/// Vulkan debug messenger callback.
///
/// Routes validation layer messages to the `tracing` logger at the appropriate
/// severity level. This function is called by the Vulkan driver whenever a
/// validation message is generated.
///
/// # Safety
///
/// This is an `extern "system"` callback invoked by the Vulkan driver.
/// The `p_callback_data` pointer is guaranteed to be valid by the Vulkan spec
/// when this callback is invoked through the debug messenger.
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }

    // SAFETY: The Vulkan spec guarantees p_callback_data is valid when this
    // callback is invoked. The pointer is non-null (checked above).
    let callback_data = unsafe { &*p_callback_data };

    let message = if callback_data.p_message.is_null() {
        "(no message)"
    } else {
        // SAFETY: p_message is a valid null-terminated C string per Vulkan spec.
        unsafe { CStr::from_ptr(callback_data.p_message) }
            .to_str()
            .unwrap_or("(invalid UTF-8)")
    };

    let type_str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE",
        _ => "UNKNOWN",
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            tracing::error!(target: "vulkan_validation", type_str, "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            tracing::warn!(target: "vulkan_validation", type_str, "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            tracing::info!(target: "vulkan_validation", type_str, "{}", message);
        }
        _ => {
            tracing::debug!(target: "vulkan_validation", type_str, "{}", message);
        }
    }

    // Returning VK_FALSE tells the validation layer not to abort the call
    // that triggered this message.
    vk::FALSE
}

/// Decode a Vulkan device name from a fixed-size `c_char` array.
fn decode_device_name(name: &[std::ffi::c_char; 256]) -> String {
    // SAFETY: VkPhysicalDeviceProperties.deviceName is a null-terminated
    // UTF-8 string filled by the Vulkan driver.
    let cstr = unsafe { CStr::from_ptr(name.as_ptr()) };
    cstr.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_device_name_basic() {
        let mut name = [0i8; 256];
        let test_name = b"Test GPU\0";
        for (i, &b) in test_name.iter().enumerate() {
            name[i] = b as i8;
        }
        let result = decode_device_name(&name);
        assert_eq!(result, "Test GPU");
    }

    #[test]
    fn decode_device_name_empty() {
        let name = [0i8; 256];
        let result = decode_device_name(&name);
        assert_eq!(result, "");
    }

    #[test]
    fn validation_layer_name_is_valid() {
        // Ensure our validation layer CStr is properly null-terminated.
        let name = VALIDATION_LAYER_NAME;
        assert_eq!(name.to_str().unwrap(), "VK_LAYER_KHRONOS_validation");
    }

    #[test]
    fn score_device_prefers_discrete() {
        let mut props_discrete = vk::PhysicalDeviceProperties::default();
        props_discrete.device_type = vk::PhysicalDeviceType::DISCRETE_GPU;
        props_discrete.vendor_id = VENDOR_NVIDIA;

        let mut props_integrated = vk::PhysicalDeviceProperties::default();
        props_integrated.device_type = vk::PhysicalDeviceType::INTEGRATED_GPU;
        props_integrated.vendor_id = VENDOR_INTEL;

        let mem_props = vk::PhysicalDeviceMemoryProperties::default();

        let score_discrete = VulkanContext::score_device(&props_discrete, &mem_props);
        let score_integrated = VulkanContext::score_device(&props_integrated, &mem_props);

        assert!(
            score_discrete > score_integrated,
            "Discrete GPU should score higher than integrated"
        );
    }

    #[test]
    fn score_device_vram_tiebreaker() {
        let mut props = vk::PhysicalDeviceProperties::default();
        props.device_type = vk::PhysicalDeviceType::DISCRETE_GPU;
        props.vendor_id = VENDOR_NVIDIA;

        // 4 GB VRAM
        let mut mem_props_4gb = vk::PhysicalDeviceMemoryProperties::default();
        mem_props_4gb.memory_heap_count = 1;
        mem_props_4gb.memory_heaps[0] = vk::MemoryHeap {
            size: 4 * 1024 * 1024 * 1024,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };

        // 8 GB VRAM
        let mut mem_props_8gb = vk::PhysicalDeviceMemoryProperties::default();
        mem_props_8gb.memory_heap_count = 1;
        mem_props_8gb.memory_heaps[0] = vk::MemoryHeap {
            size: 8 * 1024 * 1024 * 1024,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };

        let score_4gb = VulkanContext::score_device(&props, &mem_props_4gb);
        let score_8gb = VulkanContext::score_device(&props, &mem_props_8gb);

        assert!(score_8gb > score_4gb, "More VRAM should score higher");
    }
}
