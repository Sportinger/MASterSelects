//! Compiled GPU kernel bytecode embedding.
//!
//! Provides access to pre-compiled PTX (CUDA) and SPIR-V (Vulkan) kernel
//! bytecode that was compiled at build time by `build.rs`.
//!
//! If the GPU toolchains (nvcc / glslc) were not available at build time,
//! the lookup functions return `None` and log a warning.

/// Returns compiled PTX bytecode for a CUDA kernel by source file stem.
///
/// The `name` should be the `.cu` filename without the extension.
/// For example, `get_ptx("nv12_to_rgba")` returns the PTX compiled from
/// `kernels/cuda/nv12_to_rgba.cu`.
///
/// Returns `None` if the kernel was not compiled (nvcc not available at
/// build time, or the source file does not exist).
///
/// # Available kernels
///
/// | Source file        | `get_ptx()` key     | Entry point      |
/// |--------------------|---------------------|------------------|
/// | `nv12_to_rgba.cu`  | `"nv12_to_rgba"`    | `nv12_to_rgba`   |
/// | `composite.cu`     | `"composite"`       | `alpha_blend`    |
/// | `blend.cu`         | `"blend"`           | `blend_rgba`     |
/// | `transform.cu`     | `"transform"`       | `transform_rgba` |
///
/// The caller (typically `KernelManager`) is responsible for mapping
/// `KernelId` to the correct source file stem.
#[cfg(feature = "cuda")]
pub fn get_ptx(name: &str) -> Option<&'static [u8]> {
    #[cfg(no_cuda_kernels)]
    {
        let _ = name;
        tracing::warn!(
            kernel = name,
            "CUDA kernels not available (nvcc was not found at build time)"
        );
        None
    }

    #[cfg(not(no_cuda_kernels))]
    {
        cuda_kernels::get(name)
    }
}

/// Returns compiled SPIR-V bytecode for a Vulkan compute shader by source
/// file stem.
///
/// The `name` should be the `.comp` filename without the extension.
/// For example, `get_spirv("nv12_to_rgba")` returns the SPIR-V compiled from
/// `kernels/vulkan/nv12_to_rgba.comp`.
///
/// Returns `None` if the shader was not compiled (glslc not available at
/// build time, or the source file does not exist).
///
/// # Available shaders
///
/// | Source file            | `get_spirv()` key   |
/// |------------------------|---------------------|
/// | `nv12_to_rgba.comp`    | `"nv12_to_rgba"`    |
/// | `composite.comp`       | `"composite"`       |
/// | `blend.comp`           | `"blend"`           |
/// | `transform.comp`       | `"transform"`       |
#[cfg(feature = "vulkan")]
pub fn get_spirv(name: &str) -> Option<&'static [u8]> {
    #[cfg(no_vulkan_kernels)]
    {
        let _ = name;
        tracing::warn!(
            kernel = name,
            "Vulkan shaders not available (glslc was not found at build time)"
        );
        None
    }

    #[cfg(not(no_vulkan_kernels))]
    {
        vulkan_kernels::get(name)
    }
}

/// Returns a list of all available compiled CUDA PTX kernel names.
#[cfg(feature = "cuda")]
pub fn available_ptx_kernels() -> &'static [&'static str] {
    #[cfg(no_cuda_kernels)]
    {
        &[]
    }

    #[cfg(not(no_cuda_kernels))]
    {
        cuda_kernels::NAMES
    }
}

/// Returns a list of all available compiled Vulkan SPIR-V shader names.
#[cfg(feature = "vulkan")]
pub fn available_spirv_shaders() -> &'static [&'static str] {
    #[cfg(no_vulkan_kernels)]
    {
        &[]
    }

    #[cfg(not(no_vulkan_kernels))]
    {
        vulkan_kernels::NAMES
    }
}

// =============================================================================
// CUDA kernel bytecode (embedded at compile time)
// =============================================================================

#[cfg(all(feature = "cuda", not(no_cuda_kernels)))]
mod cuda_kernels {
    /// Embedded PTX for nv12_to_rgba kernel (entry point: `nv12_to_rgba`).
    static NV12_TO_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ptx/nv12_to_rgba.ptx"));

    /// Embedded PTX for composite kernel (entry point: `alpha_blend`).
    static COMPOSITE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ptx/composite.ptx"));

    /// Embedded PTX for blend kernel (entry point: `blend_rgba`).
    static BLEND: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ptx/blend.ptx"));

    /// Embedded PTX for transform kernel (entry point: `transform_rgba`).
    static TRANSFORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ptx/transform.ptx"));

    /// All available kernel names.
    pub static NAMES: &[&str] = &["nv12_to_rgba", "composite", "blend", "transform"];

    /// Look up compiled PTX bytecode by kernel name (source file stem).
    pub fn get(name: &str) -> Option<&'static [u8]> {
        match name {
            "nv12_to_rgba" => Some(NV12_TO_RGBA),
            "composite" => Some(COMPOSITE),
            "blend" => Some(BLEND),
            "transform" => Some(TRANSFORM),
            _ => None,
        }
    }
}

// =============================================================================
// Vulkan shader bytecode (embedded at compile time)
// =============================================================================

#[cfg(all(feature = "vulkan", not(no_vulkan_kernels)))]
mod vulkan_kernels {
    /// Embedded SPIR-V for nv12_to_rgba shader.
    static NV12_TO_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/nv12_to_rgba.spv"));

    /// Embedded SPIR-V for composite shader.
    static COMPOSITE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/composite.spv"));

    /// Embedded SPIR-V for blend shader.
    static BLEND: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/blend.spv"));

    /// Embedded SPIR-V for transform shader.
    static TRANSFORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/transform.spv"));

    /// All available shader names.
    pub static NAMES: &[&str] = &["nv12_to_rgba", "composite", "blend", "transform"];

    /// Look up compiled SPIR-V bytecode by shader name (source file stem).
    pub fn get(name: &str) -> Option<&'static [u8]> {
        match name {
            "nv12_to_rgba" => Some(NV12_TO_RGBA),
            "composite" => Some(COMPOSITE),
            "blend" => Some(BLEND),
            "transform" => Some(TRANSFORM),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn available_kernels_list() {
        // When compiled without GPU toolchains, lists should be empty.
        // When compiled with toolchains, lists should be non-empty.
        // Either way, the function should not panic.
        #[cfg(feature = "cuda")]
        {
            let names = super::available_ptx_kernels();
            // Just verify it returns without panicking
            let _ = names;
        }

        #[cfg(feature = "vulkan")]
        {
            let names = super::available_spirv_shaders();
            let _ = names;
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn get_ptx_unknown_returns_none() {
        // An unknown kernel name should return None regardless of build state
        let result = super::get_ptx("nonexistent_kernel_xyz");
        assert!(result.is_none());
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn get_spirv_unknown_returns_none() {
        let result = super::get_spirv("nonexistent_kernel_xyz");
        assert!(result.is_none());
    }
}
