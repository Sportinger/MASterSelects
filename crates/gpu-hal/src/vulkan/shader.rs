//! Vulkan SPIR-V shader module management.
//!
//! Handles loading SPIR-V bytecode and creating Vulkan shader modules.
//! Each shader module wraps a `VkShaderModule` with RAII cleanup.
//!
//! # Architecture
//!
//! - [`ShaderModule`]: RAII wrapper around `VkShaderModule`. Automatically
//!   destroys the Vulkan object on drop.
//! - [`ShaderRegistry`]: Cache of loaded shader modules keyed by name.
//!   Integrates with [`ms_common::KernelId`] for standardized kernel lookup.
//!
//! # Usage
//!
//! ```ignore
//! use ms_gpu_hal::vulkan::shader::*;
//!
//! fn example(device: &ash::Device) -> Result<(), Box<dyn std::error::Error>> {
//!     // Load from embedded SPIR-V bytes
//!     let spirv = include_bytes!("path/to/shader.spv");
//!     let module = ShaderModule::from_spirv(device, spirv, "my_shader")?;
//!
//!     // Or use the registry for caching
//!     let mut registry = ShaderRegistry::new();
//!     let handle = registry.load_or_get(device, "my_shader", spirv)?;
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::path::Path;

use ash::vk;
use tracing::{debug, info, warn};

use ms_common::kernel::KernelId;

use super::error::VulkanError;

/// SPIR-V magic number (first 4 bytes of any valid SPIR-V module).
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// A SPIR-V shader module wrapper with RAII cleanup.
///
/// Owns a `VkShaderModule` and destroys it on drop. Each module corresponds
/// to a single `.comp` (compute shader) compiled to SPIR-V.
pub struct ShaderModule {
    /// The Vulkan shader module handle.
    module: vk::ShaderModule,
    /// Reference to the device for cleanup.
    device: ash::Device,
    /// Name/identifier for debugging.
    name: String,
}

impl ShaderModule {
    /// Create a shader module from SPIR-V bytecode.
    ///
    /// The `spirv_bytes` must be valid SPIR-V bytecode. The length must be a
    /// multiple of 4 bytes. If the input is not 4-byte aligned, the data is
    /// copied into an aligned buffer automatically.
    ///
    /// The `name` is used for debugging and logging.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::ShaderCreation`] if:
    /// - The bytecode length is not a multiple of 4.
    /// - The bytecode is empty.
    /// - The SPIR-V magic number is invalid.
    /// - The Vulkan driver rejects the shader module.
    pub fn from_spirv(
        device: &ash::Device,
        spirv_bytes: &[u8],
        name: impl Into<String>,
    ) -> Result<Self, VulkanError> {
        let name = name.into();

        // Validate minimum size (SPIR-V header is at least 20 bytes: magic + version + generator + bound + schema).
        if spirv_bytes.len() < 20 {
            warn!(name = %name, size = spirv_bytes.len(), "SPIR-V bytecode too small");
            return Err(VulkanError::ShaderCreation(vk::Result::ERROR_UNKNOWN));
        }

        // SPIR-V must have a length that is a multiple of 4.
        if !spirv_bytes.len().is_multiple_of(4) {
            warn!(name = %name, size = spirv_bytes.len(), "SPIR-V bytecode length not a multiple of 4");
            return Err(VulkanError::ShaderCreation(vk::Result::ERROR_UNKNOWN));
        }

        // Convert &[u8] to &[u32] for Vulkan. We must handle potential alignment
        // issues: the source slice may not be 4-byte aligned.
        let spirv_words = bytes_to_words(spirv_bytes);

        // Validate SPIR-V magic number.
        if spirv_words[0] != SPIRV_MAGIC {
            warn!(
                name = %name,
                found_magic = format!("0x{:08X}", spirv_words[0]),
                "Invalid SPIR-V magic number"
            );
            return Err(VulkanError::ShaderCreation(vk::Result::ERROR_UNKNOWN));
        }

        let create_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);

        let module = unsafe {
            // SAFETY: Device is valid, create_info contains valid SPIR-V code
            // (validated magic number above). The SPIR-V data is copied by the
            // driver so the source slice doesn't need to outlive this call.
            device
                .create_shader_module(&create_info, None)
                .map_err(VulkanError::ShaderCreation)?
        };

        debug!(name = %name, size = spirv_bytes.len(), "Created shader module");

        Ok(Self {
            module,
            device: device.clone(),
            name,
        })
    }

    /// Load a shader module from a SPIR-V file on disk.
    ///
    /// Reads the file contents and delegates to [`Self::from_spirv`].
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::ShaderCreation`] if the file cannot be read or
    /// contains invalid SPIR-V bytecode.
    pub fn from_spirv_file(
        device: &ash::Device,
        path: &Path,
        name: impl Into<String>,
    ) -> Result<Self, VulkanError> {
        let spirv_bytes = std::fs::read(path).map_err(|e| {
            warn!(path = %path.display(), error = %e, "Failed to read SPIR-V file");
            VulkanError::ShaderCreation(vk::Result::ERROR_UNKNOWN)
        })?;

        Self::from_spirv(device, &spirv_bytes, name)
    }

    /// Returns the Vulkan shader module handle.
    #[inline]
    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }

    /// Returns the shader module name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: We own this shader module and the device is still valid
            // (VulkanContext outlives all shader modules by construction).
            self.device.destroy_shader_module(self.module, None);
        }
        debug!(name = %self.name, "Destroyed shader module");
    }
}

/// A registry of loaded SPIR-V shader modules, keyed by name.
///
/// Provides a cache so each shader is only loaded once. Integrates with
/// [`KernelId`] for standardized kernel lookup using the common type system.
///
/// All modules are destroyed when the registry is dropped.
pub struct ShaderRegistry {
    /// Cached shader modules indexed by name.
    modules: HashMap<String, ShaderModule>,
}

impl ShaderRegistry {
    /// Create an empty shader registry.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Load a SPIR-V shader module from bytes and register it under `name`.
    ///
    /// If a module with the same name already exists, this is a no-op
    /// and returns the existing module's handle.
    pub fn load_or_get(
        &mut self,
        device: &ash::Device,
        name: &str,
        spirv_bytes: &[u8],
    ) -> Result<vk::ShaderModule, VulkanError> {
        if let Some(existing) = self.modules.get(name) {
            return Ok(existing.module());
        }

        let module = ShaderModule::from_spirv(device, spirv_bytes, name)?;
        let handle = module.module();
        self.modules.insert(name.to_string(), module);
        Ok(handle)
    }

    /// Load a SPIR-V shader module from a file and register it under `name`.
    ///
    /// If a module with the same name already exists, this is a no-op
    /// and returns the existing module's handle.
    pub fn load_or_get_file(
        &mut self,
        device: &ash::Device,
        name: &str,
        path: &Path,
    ) -> Result<vk::ShaderModule, VulkanError> {
        if let Some(existing) = self.modules.get(name) {
            return Ok(existing.module());
        }

        let module = ShaderModule::from_spirv_file(device, path, name)?;
        let handle = module.module();
        self.modules.insert(name.to_string(), module);
        Ok(handle)
    }

    /// Load a SPIR-V shader module for a specific [`KernelId`].
    ///
    /// Uses the kernel's Vulkan module name (e.g. `"nv12_to_rgba.spv"`) as the
    /// registry key. If already loaded, returns the cached handle.
    pub fn load_kernel(
        &mut self,
        device: &ash::Device,
        kernel_id: &KernelId,
        spirv_bytes: &[u8],
    ) -> Result<vk::ShaderModule, VulkanError> {
        let name = kernel_id.vulkan_module_name();
        self.load_or_get(device, &name, spirv_bytes)
    }

    /// Get a shader module handle for a [`KernelId`], if loaded.
    pub fn get_kernel(&self, kernel_id: &KernelId) -> Option<vk::ShaderModule> {
        let name = kernel_id.vulkan_module_name();
        self.get(&name)
    }

    /// Get a shader module handle by name, if loaded.
    pub fn get(&self, name: &str) -> Option<vk::ShaderModule> {
        self.modules.get(name).map(|m| m.module())
    }

    /// Remove and destroy a shader module by name.
    ///
    /// Returns `true` if the module was found and removed. The `VkShaderModule`
    /// is destroyed via the `Drop` implementation.
    pub fn remove(&mut self, name: &str) -> bool {
        let removed = self.modules.remove(name).is_some();
        if removed {
            info!(name = name, "Removed shader module from registry");
        }
        removed
    }

    /// Remove and destroy a shader module for a [`KernelId`].
    ///
    /// Returns `true` if the module was found and removed.
    pub fn remove_kernel(&mut self, kernel_id: &KernelId) -> bool {
        let name = kernel_id.vulkan_module_name();
        self.remove(&name)
    }

    /// Remove all shader modules, destroying all `VkShaderModule` handles.
    pub fn clear(&mut self) {
        let count = self.modules.len();
        self.modules.clear();
        if count > 0 {
            info!(count = count, "Cleared all shader modules from registry");
        }
    }

    /// Returns the number of loaded shader modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Returns true if no shader modules are loaded.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Returns an iterator over loaded module names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.modules.keys().map(|s| s.as_str())
    }
}

impl Default for ShaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a `&[u8]` SPIR-V buffer to a `Vec<u32>` of SPIR-V words.
///
/// This handles alignment safely by copying bytes into properly aligned `u32`
/// values using `from_ne_bytes`, avoiding undefined behavior from unaligned
/// pointer casts.
///
/// # Panics
///
/// Panics if `spirv_bytes.len()` is not a multiple of 4. The caller must
/// validate this before calling.
fn bytes_to_words(spirv_bytes: &[u8]) -> Vec<u32> {
    assert!(
        spirv_bytes.len().is_multiple_of(4),
        "SPIR-V bytecode length must be a multiple of 4"
    );

    spirv_bytes
        .chunks_exact(4)
        .map(|chunk| {
            // Convert 4 bytes to a u32 using native endianness.
            // SPIR-V files are stored in the host's native byte order
            // (the magic number determines endianness).
            u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_registry_empty() {
        let registry = ShaderRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn shader_registry_default_trait() {
        let registry = ShaderRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn bytes_to_words_valid() {
        // SPIR-V magic in little-endian bytes: 0x07230203
        let bytes: Vec<u8> = vec![0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00];
        let words = bytes_to_words(&bytes);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], SPIRV_MAGIC);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn bytes_to_words_invalid_length() {
        let bytes = vec![0x03, 0x02, 0x23];
        bytes_to_words(&bytes);
    }

    #[test]
    fn kernel_id_module_names() {
        assert_eq!(
            KernelId::Nv12ToRgba.vulkan_module_name(),
            "nv12_to_rgba.spv"
        );
        assert_eq!(KernelId::AlphaBlend.vulkan_module_name(), "alpha_blend.spv");
        assert_eq!(
            KernelId::Effect("gaussian_blur".into()).vulkan_module_name(),
            "gaussian_blur.spv"
        );
    }

    #[test]
    fn kernel_id_registry_lookup() {
        let registry = ShaderRegistry::new();
        assert!(registry.get_kernel(&KernelId::Nv12ToRgba).is_none());
        assert!(registry.get_kernel(&KernelId::AlphaBlend).is_none());
    }

    #[test]
    fn spirv_magic_constant() {
        // Verify our constant matches the SPIR-V spec.
        assert_eq!(SPIRV_MAGIC, 0x0723_0203);
    }
}
