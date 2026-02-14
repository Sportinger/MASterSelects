//! Runtime GPU backend selection.

use ms_common::config::GpuPreference;

/// Select the best available GPU backend at runtime.
pub fn describe_preference(pref: GpuPreference) -> &'static str {
    match pref {
        GpuPreference::Auto => "Auto (CUDA preferred, Vulkan fallback)",
        GpuPreference::ForceCuda => "Force CUDA (NVIDIA only)",
        GpuPreference::ForceVulkan => "Force Vulkan Compute",
    }
}
